"""
exp20b_beyond_category.py
-------------------------
Experiment 20b: カテゴリを超えた BC（NSD × MindEye2、Exp20 の出力のみ）

再構成がカテゴリ程度の情報しか持っていなくても、同じ刺激の 3 試行は同じカテゴリの出力になるので BC > 1 になる
（Shirakawa et al., 2024 の指摘）。刺激の間のばらつきを「意味的なクラスタの間」と「同じクラスタの中の刺激の間」に分け、
クラスタの中だけで BC を測る。

  BC        = 1 + n(C−1)/(N−1) · σ_s²   / σ_w²    （Exp20 の閉じた式と同じ値）
  BC_beyond = 1 + n(C−1)/(N−1) · σ_s|g² / σ_w²    （クラスタの中の刺激の違いだけ）
  クラスタの割合 = (BC − BC_beyond) / (BC − 1)      （BC の 1 を超えた分のうち、カテゴリで説明できる割合）

  σ_w² = MS_W、σ_s² = (MS_S − MS_W)/n、σ_s|g² = (MS_S|G − MS_W)/n、MS_S|G = クラスタの中の平方和 / (C − G)

クラスタ：shared1000 の正解画像 1,000 枚を CLIP ViT-L/14（OpenAI、QuickGELU）で埋め込み、L2 正規化して
k-means（k = 40 が主、10 と 100 が副。random_state = 42、n_init = 10）。全 1,000 枚で一度だけ作り、K によらず固定する。
S3 ではクラスタの中のペア識別精度（比較相手を同じクラスタの正解画像だけにする）も出す。
信頼区間は Exp20 と同じ（刺激をリサンプリングする bootstrap 1,000 回、両モデルで同じリサンプリング、バイアス補正パーセンタイル法）。

計画：~/brain-control/plan_exp21_23.md の §3b（予測 R1〜R4 は解析の前に固定した）

使い方（解析用の venv で、リポジトリの直下で実行する。GPU は使わない）：
    venv/bin/python experiments/exp20b_beyond_category.py [$EXP20/out]
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp19_existing_metrics import pairwise_identification
from exp20_status import complete_prefix, MODELS
from exp20_mindeye2_bc import (EXP20, GT_PATH, SEED, N_BOOT, LABELS, COLORS, STAGES, EMBEDDERS,
                               stimulus_stats, load_stage, summarize, images_to_224)

# ── 設定 ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
KS_CLUSTER = [40, 10, 100]                             # 先頭が主（Shirakawa らの NSD の約 40 クラスタに合わせた）
GT_CLIP_ALL = f"{EXP20}/out/gt_clip_all_cpu.npy"       # クラスタ用の正解画像の埋め込み（リポジトリには入れない）


# ── クラスタ ──────────────────────────────────────────────────────────────

def gt_clip_all():
    """正解画像 1,000 枚の CLIP ViT-L/14（QuickGELU）の埋め込み。CPU で計算してキャッシュする（生成と GPU を取り合わないため）。"""
    if os.path.exists(GT_CLIP_ALL):
        return np.load(GT_CLIP_ALL)
    import torch, open_clip
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
    model = model.eval()
    mean = torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1)
    gt = torch.load(GT_PATH, map_location="cpu").numpy()
    with torch.no_grad():
        emb = np.concatenate([model.encode_image((images_to_224(gt[s:s + 32]) - mean) / std).float().numpy()
                              for s in range(0, len(gt), 32)])
    np.save(GT_CLIP_ALL, emb)
    print(f"  正解画像 {len(emb)} 枚を CPU で埋め込んだ: {GT_CLIP_ALL}")
    return emb


def make_clusters(emb, k):
    from sklearn.cluster import KMeans
    x = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(x)


# ── カテゴリを超えた BC ────────────────────────────────────────────────────

def beyond_from_stats(ssw_c, G, n, cl, w=None):
    """(BC, BC_beyond)。ssw_c・G は stimulus_stats の出力（刺激の番号の昇順）、cl は各刺激のクラスタ。w は bootstrap の重み。

    bootstrap で複製した刺激は、Exp20 と同じく別の刺激として数える（同じクラスタの中の、平均が同じ刺激になる）。
    """
    C = len(ssw_c)
    w = np.ones(C) if w is None else np.asarray(w, dtype=np.float64)
    N = n * C
    with np.errstate(all="ignore"):                      # matmul は値が正しくても警告を出すことがあるため
        ms_w = (w @ ssw_c) / (N - C)
        ss_s = n * (w @ np.diag(G) - (w @ G @ w) / C)
        ss_sg, n_g = 0.0, 0
        for g in np.unique(cl):
            idx = np.flatnonzero(cl == g)
            wg = w[idx]
            if wg.sum() == 0:
                continue
            n_g += 1
            Gg = G[np.ix_(idx, idx)]
            ss_sg += n * (wg @ np.diag(Gg) - (wg @ Gg @ wg) / wg.sum())
    s_s = (ss_s / (C - 1) - ms_w) / n
    s_sg = (ss_sg / (C - n_g) - ms_w) / n
    scale = n * (C - 1) / (N - 1)
    return 1 + scale * s_s / ms_w, 1 + scale * s_sg / ms_w


def within_cluster_identification(pred, true_feat, img_index, cl_true):
    """ペア識別精度の、比較相手を同じクラスタの正解画像だけにしたもの。同じクラスタにほかの画像がない行は除く。"""
    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    r = np.einsum("ik,jk->ij", zrows(pred.astype(np.float64)), zrows(true_feat.astype(np.float64)))
    acc = []
    for t in range(len(pred)):
        same = np.flatnonzero(cl_true == cl_true[img_index[t]])
        others = same[same != img_index[t]]
        if len(others):
            acc.append(np.mean(r[t, others] < r[t, img_index[t]]))
    return float(np.mean(acc)), len(acc)


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(res, k, K, path):
    ink, muted, grid = "#222222", "#666666", "#e6e6e6"
    show = ["S0", "S1", "S2", "S3_dreamsim", "S3_clip"]
    names = ["S0\nridge", "S1\nbackbone", "S2\nprior", "S3\nDreamSim", "S3\nCLIP"]
    j = [STAGES.index(s) for s in show]
    x = np.arange(len(show))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    for i, m in enumerate(MODELS):
        lab = LABELS[m]
        for key, dx, filled in (("bc", -0.07, True), (f"beyond{k}", 0.07, False)):
            xo = x + (i - 0.5) * 0.3 + dx
            y, ci = res[key][i, j], res[f"ci_{key}"][i, j]
            ax.vlines(xo, ci[:, 0], ci[:, 1], color=COLORS[lab], lw=1.5, zorder=2)
            ax.plot(xo, y, ls="none", marker="o" if filled else "D", ms=7, color=COLORS[lab],
                    mec="white" if filled else COLORS[lab], mew=1.5, mfc=COLORS[lab] if filled else "white",
                    zorder=3, label=f"{lab} {'BC' if filled else 'BC beyond category'}")
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("BC (95% CI)", color=ink)
    ax.set_title(f"(A) BC vs. BC within semantic clusters (k = {k}, K = {K})", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[1]
    for i, m in enumerate(MODELS):
        lab = LABELS[m]
        xo = x + (i - 0.5) * 0.25
        y, ci = res[f"share{k}"][i, j], res[f"ci_share{k}"][i, j]
        ax.vlines(xo, ci[:, 0], ci[:, 1], color=COLORS[lab], lw=1.5, zorder=2)
        ax.plot(xo, y, ls="none", marker="o", ms=7, color=COLORS[lab], mec="white", mew=1.5, label=lab, zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of BC − 1 explained by clusters", color=ink)
    ax.set_title("(B) Category share", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbbbbb")
        ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"図を保存しました: {path}")


# ── 本体 ──────────────────────────────────────────────────────────────────

def analyze_set(out_dirs, stages, n_rows, labels, clusters, W):
    shuf = np.random.default_rng(SEED).permutation(labels)
    cats = np.unique(labels)                             # stimulus_stats の刺激の並び（番号の昇順）
    shape = (len(out_dirs), len(STAGES))
    res = {"bc": np.full(shape, np.nan), "boot_bc": np.full(shape + (len(W),), np.nan)}
    for k in KS_CLUSTER:
        for key in (f"beyond{k}", f"beyond{k}_shuf", f"share{k}"):
            res[key] = np.full(shape, np.nan)
        for key in (f"boot_beyond{k}", f"boot_share{k}"):
            res[key] = np.full(shape + (len(W),), np.nan)
    for i, d in enumerate(out_dirs):
        for stage in stages:
            j = STAGES.index(stage)
            st, st_shuf = stimulus_stats(load_stage(d, stage, n_rows), [labels, shuf])
            for k in KS_CLUSTER:
                cl = clusters[k][cats]
                bc, bb = beyond_from_stats(*st, cl)
                res["bc"][i, j], res[f"beyond{k}"][i, j] = bc, bb
                res[f"share{k}"][i, j] = (bc - bb) / (bc - 1)
                res[f"beyond{k}_shuf"][i, j] = beyond_from_stats(*st_shuf, cl)[1]
                boot = np.array([beyond_from_stats(*st, cl, w) for w in W])
                res["boot_bc"][i, j] = boot[:, 0]
                res[f"boot_beyond{k}"][i, j] = boot[:, 1]
                res[f"boot_share{k}"][i, j] = (boot[:, 0] - boot[:, 1]) / (boot[:, 0] - 1)
            k0 = KS_CLUSTER[0]
            print(f"  {LABELS[os.path.basename(d)]:>6} {stage:<12} BC = {res['bc'][i, j]:.4f}  "
                  + "  ".join(f"k={k}: beyond {res[f'beyond{k}'][i, j]:.4f}（割合 {res[f'share{k}'][i, j]:.3f}）"
                              for k in KS_CLUSTER)
                  + f"  Shuffled beyond(k={k0}) = {res[f'beyond{k0}_shuf'][i, j]:.4f}")
    for key in ["bc"] + [f"{a}{k}" for k in KS_CLUSTER for a in ("beyond", "share")]:
        res[f"ci_{key}"], res[f"ci_raw_{key}"] = summarize(res[f"boot_{key}"], res[key])
        res[f"diff_ci_{key}"], res[f"diff_ci_raw_{key}"] = summarize(
            res[f"boot_{key}"][0] - res[f"boot_{key}"][1], res[key][0] - res[key][1])
    return res


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else f"{EXP20}/out"
    out_dirs = [f"{out_root}/{m}" for m in MODELS]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stim = np.load(f"{out_dirs[0]}/meta.npz")["stim"]
    da = [np.load(f"{d}/done_prior.npy", mmap_mode="r") for d in out_dirs]
    db = [np.load(f"{d}/done_unclip.npy", mmap_mode="r") for d in out_dirs]
    K_A, rows_A = complete_prefix(stim, da)
    K_B, rows_B = complete_prefix(stim, da + db)
    print(f"K_A = {K_A}、K_B = {K_B}")
    with open(f"{out_root}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), script="exp20b", K_A=K_A, K_B=K_B)) + "\n")

    print("クラスタを作る（正解画像 1,000 枚、CLIP ViT-L/14、k-means）")
    emb = gt_clip_all()
    clusters = {k: make_clusters(emb, k) for k in KS_CLUSTER}
    for k in KS_CLUSTER:
        sizes = np.bincount(clusters[k], minlength=k)
        print(f"  k = {k}: クラスタの大きさ 最小 {sizes.min()}、中央値 {int(np.median(sizes))}、最大 {sizes.max()}")

    sets = {"A": (K_A, rows_A, ["S0", "S1", "S2", "S1_tok", "S2_tok"]), "B": (K_B, rows_B, STAGES)}
    results = {}
    for name, (K, rows, stages) in sets.items():
        labels = stim[rows]
        W = np.random.default_rng(SEED).multinomial(K, np.ones(K) / K, size=N_BOOT)
        print(f"集合 {name}（K = {K}、{len(rows)} 行）")
        results[name] = analyze_set(out_dirs, stages, len(rows), labels, clusters, W)

    # ── S3：全体と、クラスタの中のペア識別精度 ──
    stims_B = np.unique(stim[rows_B])
    img_index = np.searchsorted(stims_B, stim[rows_B])
    ident = np.full((len(MODELS), len(EMBEDDERS)), np.nan)
    ident_within = {k: np.full((len(MODELS), len(EMBEDDERS)), np.nan) for k in KS_CLUSTER}
    n_within = {}
    for e, emb_name in enumerate(EMBEDDERS):
        gt_emb = np.load(f"{out_root}/gt_{emb_name}.npy")[stims_B]
        for i, d in enumerate(out_dirs):
            pred = np.load(f"{d}/s3_{emb_name}.npy")[rows_B]
            with np.errstate(all="ignore"):
                ident[i, e] = pairwise_identification(pred, gt_emb, img_index)
            for k in KS_CLUSTER:
                ident_within[k][i, e], n_within[k] = within_cluster_identification(pred, gt_emb, img_index, clusters[k][stims_B])
            print(f"  識別精度 {LABELS[MODELS[i]]:>6} {emb_name:<8} 全体 {ident[i, e]:.4f}  "
                  + "  ".join(f"クラスタの中（k={k}）{ident_within[k][i, e]:.4f}" for k in KS_CLUSTER))

    out = dict(models=np.array([LABELS[m] for m in MODELS]), stages=np.array(STAGES), embedders=np.array(EMBEDDERS),
               ks_cluster=np.array(KS_CLUSTER), K_A=K_A, K_B=K_B, n_boot=N_BOOT, seed=SEED, identification=ident)
    for k in KS_CLUSTER:
        out[f"clusters{k}"] = clusters[k]
        out[f"identification_within{k}"] = ident_within[k]
        out[f"n_rows_within{k}"] = n_within[k]
    for name, res in results.items():
        for key, v in res.items():
            out[f"{key}_{name}"] = v
    npz_path = os.path.join(OUTPUT_DIR, "exp20b_beyond_category.npz")
    np.savez(npz_path, **out)
    print(f"結果を保存しました: {npz_path}")
    plot(results["B"], KS_CLUSTER[0], K_B, os.path.join(OUTPUT_DIR, "exp20b_beyond_category.png"))

    for name, res in results.items():
        print(f"\n集合 {name}（K = {sets[name][0]}）  値 [95% CI（バイアス補正）]")
        for j, stage in enumerate(STAGES):
            if np.isnan(res["bc"][0, j]):
                continue
            for key in ["bc"] + [f"{a}{k}" for k in KS_CLUSTER for a in ("beyond", "share")]:
                cells = [f"{res[key][i, j]:.4f} [{res[f'ci_{key}'][i, j, 0]:.4f}, {res[f'ci_{key}'][i, j, 1]:.4f}]"
                         for i in range(len(MODELS))]
                print(f"  {stage:<12} {key:<10} 40sess {cells[0]}  1sess {cells[1]}  "
                      f"差 {res[key][0, j] - res[key][1, j]:+.4f} "
                      f"[{res[f'diff_ci_{key}'][j, 0]:+.4f}, {res[f'diff_ci_{key}'][j, 1]:+.4f}]")
    print("完了!")


if __name__ == "__main__":
    main()
