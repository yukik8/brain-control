"""
exp20_mindeye2_bc.py
--------------------
Experiment 20: NSD × MindEye2 での試行ごとの BC（パイプラインの段階ごと）

NSD 被験者 1 の shared1000（1,000 枚 × 3 試行）について、MindEye2 の 40sess と 1sess のモデルで
試行ごとに作った出力（experiments/exp20_mindeye2_pertrial.py）を読み、段階ごとに BC を測る。

  - S0 : ridge の出力（4096 次元）
  - S1 : backbone の出力（拡散 prior の入力。256 × 1664）
  - S2 : 拡散 prior の出力（unCLIP の入力。256 × 1664）
  - S3 : unCLIP の再構成画像を DreamSim と CLIP ViT-L/14 で埋め込んだもの

条件：
  - Real     : 正しい刺激ラベルで BC を計算する
  - Shuffled : ラベルを 1 回並べ替えて BC を計算する（≈ 1 になるはず。較正の確認）

各刺激の試行数がそろっているので、BC は閉じた式 BC = (N-C)/(N-1) · SS_T/SS_W で計算する。
S0、トークン平均の S1・S2、S3 では compute_bc（並べ替え 1,000 回）でも計算し、閉じた式と一致することを確かめる。
不確かさは刺激をリサンプリングする bootstrap（1,000 回）で出す。同じ刺激の集合には同じリサンプリングを
全段階・両モデルに使う（40sess と 1sess の差は paired bootstrap になる）。
複製した刺激を別のカテゴリとして数えるため、bootstrap の BC は約 1.3·(BC−1)/K 低めに出る。そのため 95% 信頼区間は
バイアス補正パーセンタイル法（パーセンタイルの区間を「点推定 − bootstrap の平均」だけずらす）で出す（2026-11-04 に決定）。
補正前の区間も ci_raw として保存する。

生成を途中で止めた出力でもそのまま動く。処理の順の先頭から数えて、両モデルで全試行が済んだ刺激だけを使う。
  - K_A : パス A（S0〜S2）まで済んだ刺激の数
  - K_B : パス B（S3）まで済んだ刺激の数（K_B < 200 なら予備的な結果）

期待される結果（予測）：
  - BC は S1 → S2 → S3 と進むほど下がる
  - 1sess は 40sess より BC が低い

使い方（解析用の venv で、リポジトリの直下で実行する）：
    venv/bin/python experiments/exp20_mindeye2_bc.py [$EXP20/out]
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compute_bc import compute_bc
from exp19_existing_metrics import pairwise_identification
from exp20_status import complete_prefix, MODELS

# ── 設定 ──────────────────────────────────────────────────────────────────

EXP20      = os.environ.get("EXP20", os.path.expanduser("~/exp20"))
GT_PATH    = f"{EXP20}/data/evals/all_images.pt"   # 正解画像（1000 × 3 × 224 × 224、0〜1）
OUTPUT_DIR = "outputs"
SEED       = 42
N_SHUFFLE  = 1000
N_BOOT     = 1000
MIN_KB     = 200                                   # S3 の結果を論文に使う最低ライン
EMB_BATCH  = 32
TOKMEAN_BATCH = 200

LABELS   = {"final_subj01_pretrained_40sess_24bs": "40sess",
            "final_subj01_pretrained_1sess_24bs": "1sess"}
COLORS   = {"40sess": "#2c7bb6", "1sess": "#7b3294"}
EMBEDDERS = ["dreamsim", "clip"]
# 段階の名前。S1・S2 は全次元（_tok はトークン平均の 1664 次元）
STAGES   = ["S0", "S1", "S2", "S1_tok", "S2_tok", "S3_dreamsim", "S3_clip"]
STAGE_FILES = {"S0": "s0_ridge.npy", "S1": "s1_backbone.npy", "S2": "s2_prior.npy",
               "S1_tok": "s1_backbone.npy", "S2_tok": "s2_prior.npy",
               "S3_dreamsim": "s3_dreamsim.npy", "S3_clip": "s3_clip.npy"}
PERM_STAGES = ["S0", "S1_tok", "S2_tok", "S3_dreamsim", "S3_clip"]   # compute_bc でも計算する段階


# ── S3 の埋め込み ─────────────────────────────────────────────────────────

def open_out(path, dtype, shape):
    """出力用の配列をディスク上に置く。あれば開き直し、なければ作る（exp20_mindeye2_pertrial.py と同じ）。"""
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, "r+")
        assert a.shape == shape and a.dtype == dtype, f"{path} の形が違う"
        return a
    return np.lib.format.open_memmap(path, "w+", dtype, shape)


def load_embedders(device):
    """DreamSim と CLIP ViT-L/14 を読み込み、(B, 3, 224, 224) の 0〜1 の画像 → (B, D) の関数を返す。"""
    import torch
    import open_clip
    from dreamsim import dreamsim
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

    ds_model, _ = dreamsim(pretrained=True, device=device)
    # OpenAI の重みは QuickGELU で学習されている。open_clip は "ViT-L-14" だと GELU で作るので、QuickGELU を指定する
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
    clip_model = clip_model.to(device).eval()
    mean = torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)

    def embed_dreamsim(x):
        with torch.no_grad():
            return ds_model.embed(x.to(device)).float().cpu().numpy()

    def embed_clip(x):
        with torch.no_grad():
            return clip_model.encode_image((x.to(device) - mean) / std).float().cpu().numpy()

    return {"dreamsim": embed_dreamsim, "clip": embed_clip}


def images_to_224(imgs):
    """(B, 3, H, W) の画像（uint8 か 0〜1 の float）を、PIL で 224×224 に縮めた 0〜1 の float にする（exp11 と同じ）。"""
    import torch
    from PIL import Image
    imgs = np.asarray(imgs)
    if imgs.dtype != np.uint8:
        imgs = (np.clip(imgs, 0, 1) * 255).round().astype(np.uint8)
    out = [np.array(Image.fromarray(im.transpose(1, 2, 0)).resize((224, 224))) / 255.0 for im in imgs]
    return torch.tensor(np.stack(out), dtype=torch.float32).permute(0, 3, 1, 2)


def embed_rows(images, rows, out_dir_or_root, prefix, embedders):
    """images の rows 行を埋め込み、<prefix>_<name>.npy に保存する。済んだ行は計算し直さない。"""
    import torch
    N = len(images)
    for name, fn in embedders.items():
        D = fn(torch.zeros(1, 3, 224, 224)).shape[1]
        emb = open_out(f"{out_dir_or_root}/{prefix}_{name}.npy", np.float32, (N, D))
        done = open_out(f"{out_dir_or_root}/done_{prefix}_{name}.npy", np.bool_, (N,))
        todo = rows[~np.asarray(done[rows])]
        for s in range(0, len(todo), EMB_BATCH):
            r = todo[s:s + EMB_BATCH]
            emb[r] = fn(images_to_224(images[r]))
            emb.flush()
            done[r] = True
            done.flush()
        print(f"  {prefix}_{name}: {len(todo)} 行を新しく計算（D = {D}）")


# ── BC（閉じた式と bootstrap）──────────────────────────────────────────────

def stimulus_stats(X, label_sets, chunk=8192):
    """刺激ごとの十分統計量。各刺激の試行数がそろっていることが前提。

    label_sets の各ラベルは、X の先頭 len(labels) 行に対応する（1 回の読み込みで複数の集合を計算する）。

    Returns:
        [(ssw_c, G, n), ...] : label_sets と同じ順
            ssw_c: (C,) 刺激ごとのカテゴリ内平方和（全次元の合計）
            G:     (C, C) 刺激平均どうしの内積（全体平均を引いたあと）
            n:     刺激あたりの試行数
    """
    X2 = X.reshape(X.shape[0], -1)
    n_max = max(len(l) for l in label_sets)
    jobs = []
    for labels in label_sets:
        order = np.argsort(labels, kind="stable")
        cats, counts = np.unique(labels, return_counts=True)
        assert (counts == counts[0]).all(), "刺激ごとの試行数がそろっていない"
        C, n = len(cats), int(counts[0])
        jobs.append((order, C, n, np.zeros(C), np.zeros((C, C))))
    for s in range(0, X2.shape[1], chunk):
        x_all = np.asarray(X2[:n_max, s:s + chunk], dtype=np.float64)
        for order, C, n, ssw_c, G in jobs:
            x = x_all[order]
            x -= x.mean(axis=0)                              # 桁落ちを防ぐため全体平均を引く（BC は変わらない）
            x = x.reshape(C, n, -1)
            m = x.mean(axis=1)
            ssw_c += ((x - m[:, None]) ** 2).sum(axis=(1, 2))
            with np.errstate(all="ignore"):                  # matmul は値が正しくても警告を出すことがあるため
                G += m @ m.T
    for _, _, _, ssw_c, G in jobs:
        assert np.isfinite(ssw_c).all() and np.isfinite(G).all(), "統計量に inf か nan"
    return [(ssw_c, G, n) for _, _, n, ssw_c, G in jobs]


def bc_from_stats(ssw_c, G, n, w=None):
    """閉じた式の BC。w は各刺激が選ばれた回数（bootstrap 用。None なら全刺激 1 回ずつ）。"""
    C = len(ssw_c)
    w = np.ones(C) if w is None else np.asarray(w, dtype=np.float64)
    N = n * C
    with np.errstate(all="ignore"):                          # 同上（stimulus_stats）
        ss_w = w @ ssw_c
        ss_b = n * (w @ np.diag(G) - (w @ G @ w) / C)
    return (N - C) / (N - 1) * (ss_w + ss_b) / ss_w


def token_mean(X, n_rows):
    """(N, 256, 1664) の先頭 n_rows 行をトークンで平均する（200 行ずつ float32 で）。"""
    return np.concatenate([np.asarray(X[s:min(s + TOKMEAN_BATCH, n_rows)], dtype=np.float32).mean(axis=1)
                           for s in range(0, n_rows, TOKMEAN_BATCH)])


def load_stage(out_dir, stage, n_rows):
    """段階の特徴量（先頭 n_rows 行）を返す。S1・S2 の全次元はメモリに載せず memmap のまま返す。"""
    X = np.load(f"{out_dir}/{STAGE_FILES[stage]}", mmap_mode="r")
    if stage.endswith("_tok"):
        return token_mean(X, n_rows)
    if stage in ("S1", "S2"):
        return X[:n_rows]
    return np.asarray(X[:n_rows], dtype=np.float32)


def analyze_set(out_dirs, stages, n_rows, labels, W):
    """1 つの刺激の集合（先頭 n_rows 行）について、両モデル × 段階の BC・Shuffled・bootstrap を計算する。"""
    shuf = np.random.default_rng(SEED).permutation(labels)
    res = {k: np.full((len(out_dirs), len(STAGES)), np.nan) for k in ("bc", "bc_shuf")}
    res["boot"] = np.full((len(out_dirs), len(STAGES), len(W)), np.nan)
    for i, d in enumerate(out_dirs):
        for stage in stages:
            j = STAGES.index(stage)
            X = load_stage(d, stage, n_rows)
            (st, st_shuf) = stimulus_stats(X, [labels, shuf])
            res["bc"][i, j] = bc_from_stats(*st)
            res["bc_shuf"][i, j] = bc_from_stats(*st_shuf)
            res["boot"][i, j] = [bc_from_stats(*st, w=w) for w in W]
            print(f"  {LABELS[os.path.basename(d)]:>6} {stage:<12} BC = {res['bc'][i, j]:.4f}"
                  f"  Shuffled = {res['bc_shuf'][i, j]:.4f}")
    return res


def summarize(boot, point):
    """bootstrap の 95% 信頼区間。最後の軸が bootstrap。

    Returns:
        ci    : バイアス補正パーセンタイル法。2.5 と 97.5 パーセンタイルを (点推定 − bootstrap の平均) だけずらす
        ci_raw: 補正前（2.5 と 97.5 パーセンタイル）
    """
    raw = np.percentile(boot, [2.5, 97.5], axis=-1)
    ci = raw + (point - boot.mean(axis=-1))
    to_last = lambda a: a.transpose(*range(1, a.ndim), 0)
    return to_last(ci), to_last(raw)


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(res_main, ci_main, ident, K_main, K_B, preliminary, path):
    """(A) 段階ごとの BC、(B) S3 での BC と識別精度の関係。"""
    ink, muted, grid = "#222222", "#666666", "#e6e6e6"
    show = ["S0", "S1", "S2", "S3_dreamsim", "S3_clip"]
    names = ["S0\nridge", "S1\nbackbone", "S2\nprior", "S3\nDreamSim", "S3\nCLIP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))

    ax = axes[0]
    x = np.arange(len(show))
    for i, m in enumerate(MODELS):
        lab = LABELS[m]
        j = [STAGES.index(s) for s in show]
        y, ci = res_main["bc"][i, j], ci_main[i, j]
        ok = ~np.isnan(y)
        xo = x + (i - 0.5) * 0.12
        # 信頼区間は下限から上限への線で描く（刺激が少ないと点推定が区間の外に出ることがあるため）
        ax.vlines(xo[ok], ci[ok, 0], ci[ok, 1], color=COLORS[lab], lw=1.5, zorder=2)
        line = ok.copy(); line[-1] = False                   # S3 の CLIP は同じ段階の別の埋め込みなので線でつながない
        ax.plot(xo[line], y[line], color=COLORS[lab], lw=2, marker="o", ms=8, mec="white", mew=2,
                label=lab, zorder=3)
        ax.plot(xo[-1:][ok[-1:]], y[-1:][ok[-1:]], ls="none", marker="s", ms=8, color=COLORS[lab],
                mec="white", mew=2, zorder=3)
        ax.annotate(lab, (xo[0], y[0]), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", color=ink, fontsize=10)
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.set_xticks(x, names)
    ax.set_xlim(-0.9, len(show) - 0.5)
    ax.set_ylabel("BC (95% CI)", color=ink)
    title = f"(A) BC by pipeline stage (K = {K_main} stimuli)"
    ax.set_title(title + (" — preliminary" if preliminary else ""), loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    if K_B > 0:
        markers = {"dreamsim": "o", "clip": "s"}
        for i, m in enumerate(MODELS):
            lab = LABELS[m]
            for e, emb in enumerate(EMBEDDERS):
                j = STAGES.index(f"S3_{emb}")
                ax.scatter(ident[i, e], res_main["bc"][i, j], s=64, marker=markers[emb],
                           color=COLORS[lab], edgecolor="white", linewidth=2, zorder=3)
                ax.annotate(f"{lab} {'DreamSim' if emb == 'dreamsim' else 'CLIP'}",
                            (ident[i, e], res_main["bc"][i, j]), xytext=(8, 4), textcoords="offset points",
                            color=ink, fontsize=9)
        ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
        ax.margins(x=0.35, y=0.15)
        lo, hi = ax.get_xlim()                               # 識別精度は 1 を超えないので、1 より大きい目盛りは出さない
        ax.set_xticks([t for t in np.arange(0.5, 1.0001, 0.05) if lo <= t <= hi])
        ax.set_xlabel("Pairwise identification (chance = 0.5)", color=ink)
        ax.set_ylabel("BC at S3", color=ink)
    else:
        ax.text(0.5, 0.5, "S3 not available (K_B = 0)", ha="center", va="center", color=muted,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(B) S3: BC vs identification (K_B = {K_B})", loc="left", color=ink, fontsize=11)

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


def plot_examples(out_dirs, stim, rows_B, path, n_stim=4):
    """再構成画像の例（先頭 n_stim 刺激 × 3 試行、40sess・1sess・正解画像）。コミットするかは人間に確認する。"""
    import torch
    gt = torch.load(GT_PATH, map_location="cpu")
    fig, axes = plt.subplots(n_stim, 7, figsize=(10.5, 1.6 * n_stim))
    for r in range(n_stim):
        t = rows_B[3 * r: 3 * r + 3]
        k = int(stim[t[0]])
        axes[r, 0].imshow(gt[k].permute(1, 2, 0).numpy().clip(0, 1))
        for i, d in enumerate(out_dirs):
            ims = np.load(f"{d}/s3_images.npy", mmap_mode="r")
            for c, tt in enumerate(t):
                axes[r, 1 + 3 * i + c].imshow(np.asarray(ims[tt]).transpose(1, 2, 0))
    for c, name in enumerate(["Stimulus"] + [f"{LABELS[m]} trial {q + 1}" for m in MODELS for q in range(3)]):
        axes[0, c].set_title(name, fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"例の図を保存しました（コミット前に人間に確認する）: {path}")


# ── メイン ────────────────────────────────────────────────────────────────

def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else f"{EXP20}/out"
    out_dirs = [f"{out_root}/{m}" for m in MODELS]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metas = [np.load(f"{d}/meta.npz") for d in out_dirs]
    stim = metas[0]["stim"]
    assert all(np.array_equal(m["stim"], stim) for m in metas), "モデルの間で試行の並びが違う"
    N, C_total = len(stim), len(np.unique(stim))
    da = [np.load(f"{d}/done_prior.npy", mmap_mode="r") for d in out_dirs]
    db = [np.load(f"{d}/done_unclip.npy", mmap_mode="r") if os.path.exists(f"{d}/done_unclip.npy")
          else np.zeros(N, bool) for d in out_dirs]
    K_A, rows_A = complete_prefix(stim, da)
    K_B, rows_B = complete_prefix(stim, da + db)
    preliminary_B = K_B < MIN_KB
    print(f"K_A = {K_A}、K_B = {K_B}（全 {C_total} 刺激）" + ("。S3 は予備的な結果" if 0 < K_B < MIN_KB else ""))
    assert K_A > 0, "パス A が済んだ刺激がない"

    with open(f"{out_root}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), K_A=K_A, K_B=K_B)) + "\n")

    # ── S3 の埋め込み ──
    ident = np.full((len(MODELS), len(EMBEDDERS)), np.nan)
    if K_B > 0:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embedders = load_embedders(device)
        print("S3 と正解画像を埋め込む")
        for d in out_dirs:
            embed_rows(np.load(f"{d}/s3_images.npy", mmap_mode="r"), rows_B, d, "s3", embedders)
        gt = torch.load(GT_PATH, map_location="cpu").numpy()
        stims_B = np.unique(stim[rows_B])
        embed_rows(gt, stims_B, out_root, "gt", embedders)

        # ── 識別精度 ──
        img_index = np.searchsorted(stims_B, stim[rows_B])
        for e, emb in enumerate(EMBEDDERS):
            gt_emb = np.load(f"{out_root}/gt_{emb}.npy")[stims_B]
            for i, d in enumerate(out_dirs):
                ident[i, e] = pairwise_identification(np.load(f"{d}/s3_{emb}.npy")[rows_B], gt_emb, img_index)
                print(f"  識別精度 {LABELS[MODELS[i]]:>6} {emb:<8} {ident[i, e]:.4f}")

    # ── BC（閉じた式、Shuffled、bootstrap）──
    sets = {"A": (K_A, rows_A, ["S0", "S1", "S2", "S1_tok", "S2_tok"])}
    if K_B > 0:
        sets["B"] = (K_B, rows_B, STAGES)
    results = {}
    for name, (K, rows, stages) in sets.items():
        labels = stim[rows]
        W = np.random.default_rng(SEED).multinomial(K, np.ones(K) / K, size=N_BOOT)
        print(f"集合 {name}（K = {K}、{len(rows)} 行）")
        res = analyze_set(out_dirs, stages, len(rows), labels, W)
        res["ci"], res["ci_raw"] = summarize(res["boot"], res["bc"])      # (モデル, 段階, 2)
        res["diff_ci"], res["diff_ci_raw"] = summarize(                   # 40sess − 1sess（paired）
            res["boot"][0] - res["boot"][1], res["bc"][0] - res["bc"][1])
        results[name] = res

    # ── 閉じた式の検算と並べ替え検定（compute_bc）──
    perm = {k: np.full((len(MODELS), len(STAGES)), np.nan) for k in ("bc", "p")}
    for stage in PERM_STAGES:
        name = "B" if stage.startswith("S3") else "A"
        if name not in sets:
            continue
        K, rows, _ = sets[name]
        j = STAGES.index(stage)
        for i, d in enumerate(out_dirs):
            X = load_stage(d, stage, len(rows))
            bc_mean, _, _, _, p = compute_bc(X, stim[rows], N_SHUFFLE, SEED, mode="across")
            perm["bc"][i, j], perm["p"][i, j] = bc_mean, p
            gap = abs(bc_mean - results[name]["bc"][i, j])
            print(f"  検算 {LABELS[MODELS[i]]:>6} {stage:<12} 並べ替え {bc_mean:.4f}  閉じた式との差 {gap:.5f}  p = {p:.4f}"
                  + ("" if gap < 0.001 else "  ← 差が 0.001 以上"))

    # ── 実行の記録 ──
    runs = []
    if os.path.exists(f"{out_root}/runs.jsonl"):
        with open(f"{out_root}/runs.jsonl") as f:
            runs = [json.loads(l) for l in f if l.strip()]
    last_unclip = [r for r in runs if r["stage"] == "unclip" and set(r["dirs"]) == set(MODELS)]
    stop_reason = last_unclip[-1]["stop_reason"] if last_unclip else ""
    sec = {}
    for st, fn in (("prior", "times_prior.npy"), ("unclip", "times_unclip.npy")):
        t = [np.load(f"{d}/{fn}")[np.asarray(dn, bool)] for d, dn in
             zip(out_dirs, da if st == "prior" else db) if os.path.exists(f"{d}/{fn}")]
        t = np.concatenate(t) if t else np.array([])
        sec[st] = float(np.median(t)) if len(t) else np.nan
    chip = runs[-1]["chip"] if runs else ""

    # ── 保存 ──
    main_set = "B" if K_B > 0 else "A"
    out = dict(models=np.array([LABELS[m] for m in MODELS]), stages=np.array(STAGES),
               embedders=np.array(EMBEDDERS), K_A=K_A, K_B=K_B, C_total=C_total,
               n_excluded_imgs=int(metas[0]["n_excluded_imgs"]), n_excluded_trials=int(metas[0]["n_excluded_trials"]),
               identification=ident, bc_perm=perm["bc"], p_perm=perm["p"],
               stop_reason=stop_reason, sec_per_trial_prior=sec["prior"], sec_per_trial_unclip=sec["unclip"],
               chip=chip, preliminary_S3=preliminary_B, n_boot=N_BOOT, seed=SEED)
    for name, res in results.items():
        for k, v in res.items():
            out[f"{k}_{name}"] = v
    npz_path = os.path.join(OUTPUT_DIR, "exp20_mindeye2_bc.npz")
    np.savez(npz_path, **out)
    print(f"結果を保存しました: {npz_path}")

    res = results[main_set]
    plot(res, res["ci"], ident, sets[main_set][0], K_B, preliminary_B and K_B > 0,
         os.path.join(OUTPUT_DIR, "exp20_mindeye2_bc.png"))
    if K_B >= 4:
        plot_examples(out_dirs, stim, rows_B, os.path.join(OUTPUT_DIR, "exp20_examples.png"))

    # ── 表 ──
    for name, res in results.items():
        K = sets[name][0]
        print(f"\n集合 {name}（K = {K}）  BC [95% CI（バイアス補正）]、Shuffled、40sess − 1sess [95% CI]")
        for j, stage in enumerate(STAGES):
            if np.isnan(res["bc"][0, j]):
                continue
            cells = [f"{res['bc'][i, j]:.4f} [{res['ci'][i, j, 0]:.4f}, {res['ci'][i, j, 1]:.4f}] "
                     f"(shuf {res['bc_shuf'][i, j]:.4f})" for i in range(len(MODELS))]
            print(f"  {stage:<12} 40sess {cells[0]}  1sess {cells[1]}  "
                  f"差 {res['bc'][0, j] - res['bc'][1, j]:+.4f} [{res['diff_ci'][j, 0]:+.4f}, {res['diff_ci'][j, 1]:+.4f}]")
    print("完了!")


if __name__ == "__main__":
    main()
