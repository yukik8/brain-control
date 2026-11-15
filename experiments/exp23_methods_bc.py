"""
exp23_methods_bc.py
-------------------
Experiment 23: NSD の被験者 1 で、MindEye2（40sess・1sess）、MindEye1、Brain-Diffuser の試行ごとの BC を比べる

4 つの手法で同じ解析（閉じた式の BC、Shuffled、刺激をリサンプリングする bootstrap、S3 のペア識別精度）を行う。
  - 主な集合：4 つすべてで S3 まで済んだ刺激の共通部分。手法の間の差は、この集合で、全手法に同じ bootstrap の重みを
    使って出す（paired）。MindEye1・Brain-Diffuser のテスト（839 刺激）は MindEye2 の 1,000 刺激の一部なので、
    それぞれの先頭 500 刺激は完全には重ならない
  - S3 は最終の画像を DreamSim と CLIP ViT-L/14（QuickGELU）で埋め込んだもの（Exp20 と同じ関数）
  - 識別精度の信頼区間は、刺激ごとの識別精度（3 試行の平均）を bootstrap の重みで平均して出す（相手の刺激は固定）
  - 途中の段階は手法ごとに記述的に出す（MindEye1 と Brain-Diffuser だけ。MindEye2 は Exp20・22 にある）
  - 補足：各手法の自分の 500 刺激での S3 の BC と識別精度

  - P7：主な集合の全手法・全段階で Shuffled が 0.95〜1.05
  - P8：各手法の S3 の BC の信頼区間の下限が、DreamSim・CLIP とも 1 より大きい

計画：~/brain-control/plan_exp21_23.md の §5（P7・P8 の基準と解析の細部は、生成のあと・結果を見る前に決めた）

使い方（解析用の venv で、リポジトリの直下で実行する）：
    venv/bin/python experiments/exp23_methods_bc.py
    venv/bin/python experiments/exp23_methods_bc.py --plot_only    # npz から図だけ描き直す
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp19_existing_metrics import pairwise_identification
from exp20_status import complete_prefix
from exp20_mindeye2_bc import (EXP20, GT_PATH, SEED, N_BOOT, EMBEDDERS, load_embedders, embed_rows,
                               stimulus_stats, bc_from_stats, summarize)
from exp22_multisubject_bc import place_labels

# ── 設定 ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
E23        = f"{EXP20}/exp23"
ANALYSIS   = f"{E23}/analysis"                     # 正解画像の埋め込み（Exp20 の出力には書き込まない）
BD_DATA    = f"{E23}/bd_data"
SYSTEMS    = ["MindEye2_40sess", "MindEye2_1sess", "MindEye1", "Brain-Diffuser"]
NAMES      = {"MindEye2_40sess": "MindEye2\n40sess", "MindEye2_1sess": "MindEye2\n1sess",
              "MindEye1": "MindEye1", "Brain-Diffuser": "Brain-\nDiffuser"}
COLORS     = {"MindEye2_40sess": "#2c7bb6", "MindEye2_1sess": "#7b3294", "MindEye1": "#e66101",
              "Brain-Diffuser": "#1b9e77"}
DIRS       = {"MindEye2_40sess": f"{EXP20}/out/final_subj01_pretrained_40sess_24bs",
              "MindEye2_1sess": f"{EXP20}/out/final_subj01_pretrained_1sess_24bs",
              "MindEye1": f"{E23}/me1_runs/final_subj01",
              "Brain-Diffuser": f"{E23}/bd_runs/final_subj01"}
# 段階の名前。_tok はトークン平均、_dreamsim は画像を DreamSim で埋め込んだもの
STAGES     = ["S1", "S1_tok", "S2", "S2_tok", "S2low_dreamsim",                          # MindEye1
              "S1_vdvae", "S1_clipvision", "S1_clipvision_tok", "S1_cliptext", "S1_cliptext_tok",
              "S2_vdvae_dreamsim",                                                      # Brain-Diffuser
              "S3_dreamsim", "S3_clip"]                                                 # 全手法
SYSTEM_STAGES = {"MindEye2_40sess": ["S3_dreamsim", "S3_clip"],
                 "MindEye2_1sess": ["S3_dreamsim", "S3_clip"],
                 "MindEye1": ["S1", "S1_tok", "S2", "S2_tok", "S2low_dreamsim", "S3_dreamsim", "S3_clip"],
                 "Brain-Diffuser": ["S1_vdvae", "S1_clipvision", "S1_clipvision_tok", "S1_cliptext",
                                    "S1_cliptext_tok", "S2_vdvae_dreamsim", "S3_dreamsim", "S3_clip"]}
SHUF_TOL   = 0.05                                  # P7 の「Shuffled ≈ 1」の基準（結果を見る前に決めた）


class HWCtoCHW:
    """(N, H, W, 3) の画像の配列を、行を取り出すときに (B, 3, H, W) にして返す（Brain-Diffuser の VDVAE の画像用）。"""
    def __init__(self, a):
        self.a = a

    def __len__(self):
        return len(self.a)

    def __getitem__(self, r):
        return np.asarray(self.a[r]).transpose(0, 3, 1, 2)


# ── 手法ごとの、S3 まで済んだ行 ──────────────────────────────────────────────

def complete_rows(system):
    """手法の S3 まで済んだ行（処理の順の先頭から）と、行ごとの刺激の番号（all_images.pt の行）を返す。"""
    if system.startswith("MindEye2"):                # Exp20 と同じく、2 つのモデルの両方で済んだ刺激
        dirs = [DIRS["MindEye2_40sess"], DIRS["MindEye2_1sess"]]
        dones = [np.load(f"{d}/done_{p}.npy", mmap_mode="r") for d in dirs for p in ("prior", "unclip")]
    elif system == "MindEye1":
        dirs = [DIRS[system]]
        dones = [np.load(f"{dirs[0]}/done_{p}.npy", mmap_mode="r") for p in ("prior", "vd")]
    else:
        dirs = [DIRS[system]]
        dones = [np.load(f"{dirs[0]}/done_{p}.npy", mmap_mode="r") for p in ("vdvae", "vd")]
    stim = np.load(f"{DIRS[system]}/meta.npz")["stim"]
    K, rows = complete_prefix(stim, dones)
    return stim, K, rows


def load_stage(system, stage, rows):
    """手法の段階の特徴量（rows の行）を float32 で返す。"""
    d = DIRS[system]
    if stage in ("S3_dreamsim", "S3_clip"):
        return np.load(f"{d}/s3_{stage.split('_')[1]}.npy", mmap_mode="r")[rows]
    if stage == "S2low_dreamsim":
        return np.load(f"{d}/s2low_dreamsim.npy", mmap_mode="r")[rows]
    if stage == "S2_vdvae_dreamsim":
        return np.load(f"{d}/s2vdvae_dreamsim.npy", mmap_mode="r")[rows]
    if system == "MindEye1":
        X = np.load(f"{d}/{'s1_backbone' if stage.startswith('S1') else 's2_prior'}.npy", mmap_mode="r")[rows]
    else:
        pred_row = np.load(f"{d}/meta.npz")["pred_row"]
        feature = stage.split("_")[1]
        X = np.load(f"{BD_DATA}/pred_{feature}_trial.npy", mmap_mode="r")[pred_row[rows]]
    X = X.astype(np.float32)
    return X.mean(axis=1) if stage.endswith("_tok") else X


def embed_images(system, rows, embedders):
    """手法の画像（S3 と途中の画像）を埋め込む。済んだ行は計算し直さない。MindEye2 は Exp20 で済んでいる。"""
    d = DIRS[system]
    if system.startswith("MindEye2"):
        for emb in EMBEDDERS:
            assert np.load(f"{d}/done_s3_{emb}.npy", mmap_mode="r")[rows].all(), f"{system}：S3 の埋め込みが済んでいない"
        return
    embed_rows(np.load(f"{d}/s3_images.npy", mmap_mode="r"), rows, d, "s3", embedders)
    ds = {"dreamsim": embedders["dreamsim"]}
    if system == "MindEye1":
        embed_rows(np.load(f"{d}/low_images.npy", mmap_mode="r"), rows, d, "s2low", ds)
    else:
        embed_rows(HWCtoCHW(np.load(f"{d}/s2_vdvae64.npy", mmap_mode="r")), rows, d, "s2vdvae", ds)


# ── 識別精度（刺激ごと）──────────────────────────────────────────────────────

def identification_per_stimulus(pred, gt_emb, img_index):
    """ペア識別精度を刺激ごとに返す（exp19 の pairwise_identification と同じ定義。試行ごとの値を刺激ごとに平均）。"""
    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    with np.errstate(all="ignore"):                  # matmul の見かけだけの警告（値は正しい）
        r = zrows(pred.astype(np.float64)) @ zrows(gt_emb.astype(np.float64)).T
    r_true = r[np.arange(len(pred)), img_index]
    per_trial = (r < r_true[:, None]).sum(axis=1) / (r.shape[1] - 1)
    per_stim = np.bincount(img_index, per_trial, minlength=len(gt_emb)) / np.bincount(img_index, minlength=len(gt_emb))
    with np.errstate(all="ignore"):
        assert np.isclose(per_stim.mean(), pairwise_identification(pred, gt_emb, img_index)), "識別精度が exp19 と違う"
    return per_stim


# ── 1 つの刺激の集合での解析 ───────────────────────────────────────────────

def analyze(rows_of, stims, systems, stages_of, gt):
    """stims（並べた刺激の番号）について、手法 × 段階の BC・Shuffled・bootstrap と、S3 の識別精度を出す。"""
    C = len(stims)
    W = np.random.default_rng(SEED).multinomial(C, np.ones(C) / C, size=N_BOOT)
    shape = (len(systems), len(STAGES))
    res = {"bc": np.full(shape, np.nan), "bc_shuf": np.full(shape, np.nan),
           "boot": np.full(shape + (N_BOOT,), np.nan),
           "ident": np.full((len(systems), len(EMBEDDERS)), np.nan),
           "ident_boot": np.full((len(systems), len(EMBEDDERS), N_BOOT), np.nan)}
    for i, system in enumerate(systems):
        rows = rows_of[system]
        labels = np.load(f"{DIRS[system]}/meta.npz")["stim"][rows]
        assert np.array_equal(np.unique(labels), stims), f"{system}：刺激の集合が違う"
        shuf = np.random.default_rng(SEED).permutation(labels)
        for stage in stages_of[system]:
            j = STAGES.index(stage)
            st, st_shuf = stimulus_stats(load_stage(system, stage, rows), [labels, shuf])
            res["bc"][i, j] = bc_from_stats(*st)
            res["bc_shuf"][i, j] = bc_from_stats(*st_shuf)
            res["boot"][i, j] = [bc_from_stats(*st, w=w) for w in W]
            print(f"  {system:<16} {stage:<18} BC = {res['bc'][i, j]:.4f}  Shuffled = {res['bc_shuf'][i, j]:.4f}")
        img_index = np.searchsorted(stims, labels)
        for e, emb in enumerate(EMBEDDERS):
            per_stim = identification_per_stimulus(load_stage(system, f"S3_{emb}", rows), gt[emb][stims], img_index)
            res["ident"][i, e] = per_stim.mean()
            res["ident_boot"][i, e] = W @ per_stim / C
    res["ci"], res["ci_raw"] = summarize(res["boot"], res["bc"])
    res["ident_ci"], res["ident_ci_raw"] = summarize(res["ident_boot"], res["ident"])
    return res


def pair_differences(res, systems):
    """全部の組（a, b）について、S3 の BC と識別精度の差（a − b）と信頼区間。"""
    pairs = list(combinations(range(len(systems)), 2))
    js = [STAGES.index(f"S3_{emb}") for emb in EMBEDDERS]
    out = {"pairs": np.array(pairs),
           "diff_bc": np.array([[res["bc"][a, j] - res["bc"][b, j] for j in js] for a, b in pairs]),
           "diff_ident": np.array([res["ident"][a] - res["ident"][b] for a, b in pairs])}
    boot_bc = np.array([[res["boot"][a, j] - res["boot"][b, j] for j in js] for a, b in pairs])
    boot_id = np.array([res["ident_boot"][a] - res["ident_boot"][b] for a, b in pairs])
    out["diff_bc_ci"], out["diff_bc_ci_raw"] = summarize(boot_bc, out["diff_bc"])
    out["diff_ident_ci"], out["diff_ident_ci_raw"] = summarize(boot_id, out["diff_ident"])
    return out


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(z, path):
    ink, muted, grid = "#222222", "#666666", "#e6e6e6"
    err = lambda v, lo, hi: [[max(v - lo, 0)], [max(hi - v, 0)]]   # バイアス補正した区間が点を含まないときは 0 にする
    systems = [str(s) for s in z["systems"]]
    x = np.arange(len(systems))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    for e, (emb, marker) in enumerate(zip(EMBEDDERS, ("o", "s"))):
        j = STAGES.index(f"S3_{emb}")
        for i, system in enumerate(systems):
            bc, (lo, hi) = z["bc"][i, j], z["ci"][i, j]
            ax.errorbar(x[i] + (e - 0.5) * 0.28, bc, yerr=err(bc, lo, hi), fmt=marker, ms=8, color=COLORS[system],
                        mec="white", mew=1.5, elinewidth=2, capsize=0, zorder=3)
        ax.plot([], [], marker, color=muted, ms=8, ls="none", label={"dreamsim": "DreamSim", "clip": "CLIP"}[emb])
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.set_xticks(x, [NAMES[s] for s in systems])
    ax.set_ylabel("BC at S3 (final image)", color=ink)
    ax.set_title(f"(A) S3 BC by method, subject 1 (common {int(z['C'])} stimuli, 95% CI)", loc="left", color=ink,
                 fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.margins(y=0.15)
    ax = axes[1]
    e = EMBEDDERS.index("dreamsim")
    j = STAGES.index("S3_dreamsim")
    pts = []
    for i, system in enumerate(systems):
        bc, (lo, hi) = z["bc"][i, j], z["ci"][i, j]
        idn, (ilo, ihi) = z["ident"][i, e], z["ident_ci"][i, e]
        ax.errorbar(idn, bc, xerr=err(idn, ilo, ihi), yerr=err(bc, lo, hi), fmt="o", ms=8,
                    color=COLORS[system], mec="white", mew=1.5, elinewidth=1.5, capsize=0, zorder=3)
        pts.append((NAMES[system].replace("-\n", "-").replace("\n", " "), idn, bc))
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.margins(x=0.35, y=0.2)
    lo, hi = ax.get_xlim()                               # 識別精度は 1 を超えないので、1 より大きい目盛りは出さない
    ax.set_xticks([t for t in np.arange(0.5, 1.0001, 0.05) if lo <= t <= hi])
    ax.set_xlabel("Pairwise identification, DreamSim (chance = 0.5)", color=ink)
    ax.set_ylabel("BC at S3 (DreamSim)", color=ink)
    ax.set_title("(B) S3: BC vs identification by method", loc="left", color=ink, fontsize=11)
    for ax in axes:
        ax.grid(axis="y", color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbbbbb")
        ax.tick_params(colors=muted)
    fig.tight_layout()
    place_labels(fig, axes[1], pts, ink)                # 重なりは表示の座標で調べるので、配置が決まってから
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"図を保存しました: {path}")


# ── 本体 ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUTPUT_DIR, "exp23_methods_bc.npz")
    png_path = os.path.join(OUTPUT_DIR, "exp23_methods_bc.png")
    if "--plot_only" in sys.argv:                        # 解析はせず、npz から図だけ描き直す
        plot(np.load(npz_path), png_path)
        return
    os.makedirs(ANALYSIS, exist_ok=True)
    import torch

    own = {}
    for system in SYSTEMS:
        stim, K, rows = complete_rows(system)
        own[system] = rows
        print(f"{system}：S3 まで済んだ刺激 K = {K}（{len(rows)} 行）")
    stims_own = {s: np.unique(np.load(f"{DIRS[s]}/meta.npz")["stim"][r]) for s, r in own.items()}
    common = np.array(sorted(set.intersection(*[set(v.tolist()) for v in stims_own.values()])))
    print(f"4 つで共通の刺激：{len(common)}")

    # 刺激の番号が、手法の間で同じ COCO の画像を指すことの確認
    ref = dict(zip(*[np.load(f"{DIRS['MindEye2_40sess']}/meta.npz")[k] for k in ("stim", "coco73k")]))
    for system in SYSTEMS:
        m = np.load(f"{DIRS[system]}/meta.npz")
        assert all(ref[k] == c for k, c in zip(m["stim"], m["coco73k"])), f"{system}：刺激の番号と COCO の画像の対応が違う"
    print("刺激の番号と COCO の画像の対応：4 つで同じ")

    embedders = load_embedders("mps")
    for system in SYSTEMS:
        embed_images(system, own[system], embedders)
    need = np.unique(np.concatenate(list(stims_own.values())))
    embed_rows(torch.load(GT_PATH, map_location="cpu").numpy(), need, ANALYSIS, "gt", embedders)
    gt = {emb: np.load(f"{ANALYSIS}/gt_{emb}.npy") for emb in EMBEDDERS}
    with open(f"{ANALYSIS}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), script="exp23",
                                K={s: int(len(v)) for s, v in stims_own.items()}, C=int(len(common)))) + "\n")

    print("\n主な集合（共通の刺激）")
    rows_common = {s: own[s][np.isin(np.load(f"{DIRS[s]}/meta.npz")["stim"][own[s]], common)] for s in SYSTEMS}
    main_res = analyze(rows_common, common, SYSTEMS, SYSTEM_STAGES, gt)
    diffs = pair_differences(main_res, SYSTEMS)

    print("\n補足：各手法の自分の刺激（S3 だけ）")
    own_res = {s: analyze({s: own[s]}, stims_own[s], [s], {s: ["S3_dreamsim", "S3_clip"]}, gt) for s in SYSTEMS}

    shuf = main_res["bc_shuf"][np.isfinite(main_res["bc_shuf"])]
    p7 = bool(np.all(np.abs(shuf - 1) <= SHUF_TOL))
    js = [STAGES.index(f"S3_{emb}") for emb in EMBEDDERS]
    p8 = np.array([all(main_res["ci"][i, j, 0] > 1 for j in js) for i in range(len(SYSTEMS))])

    out = dict(systems=np.array(SYSTEMS), stages=np.array(STAGES), embedders=np.array(EMBEDDERS), n_boot=N_BOOT,
               seed=SEED, shuf_tol=SHUF_TOL, common=common, C=len(common), P7=p7, P8=p8,
               K_own=np.array([len(stims_own[s]) for s in SYSTEMS]))
    for k, v in main_res.items():
        out[k] = v
    for k, v in diffs.items():
        out[k] = v
    for k in ("bc", "ci", "bc_shuf", "ident", "ident_ci"):
        out[f"own_{k}"] = np.concatenate([own_res[s][k] for s in SYSTEMS])
    np.savez(npz_path, **out)
    print(f"結果を保存しました: {npz_path}")
    plot(np.load(npz_path), png_path)

    print(f"\n主な集合（共通の {len(common)} 刺激）：BC [95% CI]")
    for i, system in enumerate(SYSTEMS):
        print(f"{system}")
        for stage in SYSTEM_STAGES[system]:
            j = STAGES.index(stage)
            print(f"  {stage:<18} {main_res['bc'][i, j]:.4f} [{main_res['ci'][i, j, 0]:.4f}, {main_res['ci'][i, j, 1]:.4f}]"
                  f"  Shuffled {main_res['bc_shuf'][i, j]:.4f}")
        for e, emb in enumerate(EMBEDDERS):
            print(f"  識別精度 {emb:<9} {main_res['ident'][i, e]:.4f} "
                  f"[{main_res['ident_ci'][i, e, 0]:.4f}, {main_res['ident_ci'][i, e, 1]:.4f}]")
    print("\n手法の間の差（a − b）[95% CI]")
    for (a, b), dbc, dbc_ci, did, did_ci in zip(diffs["pairs"], diffs["diff_bc"], diffs["diff_bc_ci"],
                                                diffs["diff_ident"], diffs["diff_ident_ci"]):
        cells = [f"BC {emb} {dbc[e]:+.4f} [{dbc_ci[e, 0]:+.4f}, {dbc_ci[e, 1]:+.4f}]  "
                 f"識別 {emb} {did[e]:+.4f} [{did_ci[e, 0]:+.4f}, {did_ci[e, 1]:+.4f}]" for e, emb in enumerate(EMBEDDERS)]
        print(f"  {SYSTEMS[a]} − {SYSTEMS[b]}\n    " + "\n    ".join(cells))
    print("\n補足：自分の刺激での S3（BC [95% CI]、識別精度）")
    for system in SYSTEMS:
        r = own_res[system]
        cells = [f"{emb} {r['bc'][0, j]:.4f} [{r['ci'][0, j, 0]:.4f}, {r['ci'][0, j, 1]:.4f}] 識別 {r['ident'][0, e]:.4f}"
                 for e, (emb, j) in enumerate(zip(EMBEDDERS, js))]
        print(f"  {system:<16} K = {len(stims_own[system])}  " + "  ".join(cells))
    print(f"\nP7（Shuffled が 1 ± {SHUF_TOL}）：{'成立' if p7 else '不成立'}（範囲 {shuf.min():.4f}〜{shuf.max():.4f}）")
    print("P8（S3 の BC の下限 > 1）：" + "  ".join(f"{s} {'成立' if v else '不成立'}" for s, v in zip(SYSTEMS, p8)))
    print("完了!")


if __name__ == "__main__":
    main()
