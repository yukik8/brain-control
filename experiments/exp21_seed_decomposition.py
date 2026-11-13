"""
exp21_seed_decomposition.py
---------------------------
Experiment 21: MindEye2 の試行ごとの出力のばらつきを、seed（拡散のサンプリング）と試行（脳活動）に分ける

Exp20 では試行ごとに seed を変えて生成したので、同じ刺激の 3 試行の違いには、脳活動の違いと
拡散 prior・unCLIP のサンプリングの違いが混ざっている。同じ試行を seed だけ変えて作り直した出力
（experiments/exp21_run.sh）と Exp20 の出力（r = 0）を合わせ、入れ子の分散分析で分ける。

  x_str = μ + a_s + b_st + e_str     （s: 刺激 C 個、t: 試行 n = 3、r: seed R = 4）
  σ_e² = MS_E、σ_b² = (MS_B − MS_E)/R、σ_a² = (MS_A − MS_B)/(nR)   （全次元の合計）
  BC(k) = 1 + n(C−1)/(N−1) · σ_a² / (σ_b² + σ_e²/k)                （seed を k 個平均したときの BC。N = nC）
  サンプリングの割合 = σ_e² / (σ_b² + σ_e²)                        （1 回の生成で、同じ刺激の中のばらつきのうち seed による分）

BC(1) は Exp20 と同じ条件（1 試行 1 回の生成）の BC、BC(∞) はサンプリングを除いた BC。σ_b² ≤ 0 なら BC(∞) は定義しない。
段階：S2（全次元・トークン平均）、S3（DreamSim・CLIP ViT-L/14）。S0・S1 は乱数を使わないので、seed によらず同じことを確かめる。

  - 集合 A：パス A まで全部の seed で済んだ刺激（ふつうは 1,000）。S2 だけ
  - 集合 B：パス B まで全部の seed で済んだ刺激（目標 100）。S2 と S3

Shuffled：刺激のラベルを試行ごとに 1 回並べ替える（同じ試行の seed はまとめたまま。≈ 1 になるはず）。
信頼区間は Exp20 と同じ（刺激をリサンプリングする bootstrap 1,000 回、両モデルで同じリサンプリング、バイアス補正パーセンタイル法）。

計画：~/brain-control/plan_exp21_23.md の §3（予測 P1〜P3 は生成の前に固定した）

使い方（解析用の venv で、リポジトリの直下で実行する）：
    venv/bin/python experiments/exp21_seed_decomposition.py
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp20_status import complete_prefix, MODELS
from exp20_mindeye2_bc import (EXP20, SEED, N_BOOT, LABELS, COLORS, load_embedders, embed_rows,
                               stimulus_stats, bc_from_stats, token_mean)

# ── 設定 ──────────────────────────────────────────────────────────────────

OUT_SEED   = f"{EXP20}/out_seed"
OUTPUT_DIR = "outputs"
SEEDS      = [0, 1, 2, 3]                              # r = 0 は Exp20 の出力
KS         = [1, 4, np.inf]                            # BC(k) の k
SB_POS_MIN = 0.975                                     # BC(∞) を出すのは、bootstrap でこの割合以上 σ_b² > 0 のとき
STAGES_A   = ["S2", "S2_tok"]
STAGES_B   = ["S2", "S2_tok", "S3_dreamsim", "S3_clip"]
STAGE_FILES = {"S2": "s2_prior.npy", "S2_tok": "s2_prior.npy",
               "S3_dreamsim": "s3_dreamsim.npy", "S3_clip": "s3_clip.npy"}


def seed_dir(r, model):
    return f"{EXP20}/out/{model}" if r == 0 else f"{OUT_SEED}/r{r}/{model}"


# ── 入れ子の分散分析 ──────────────────────────────────────────────────────

def nested_stats(Xs, labels, chunk=8192):
    """刺激ごとの十分統計量（seed × 試行 × 刺激の入れ子）。

    Xs: R 個の配列（どれも同じ行の並び。行 = 試行）。labels: 各行の刺激。各刺激の試行数がそろっていることが前提。
    Returns:
        sse_c: (C,) 刺激ごとの、seed の違いによる平方和
        ssb_c: (C,) 刺激ごとの、試行の違いによる平方和（seed で平均した試行の、刺激平均からのずれ × R）
        G:     (C, C) 刺激平均どうしの内積（全体平均を引いたあと）
        n, R:  刺激あたりの試行数、seed の数
    """
    R = len(Xs)
    order = np.argsort(labels, kind="stable")
    cats, counts = np.unique(labels, return_counts=True)
    assert (counts == counts[0]).all(), "刺激ごとの試行数がそろっていない"
    C, n = len(cats), int(counts[0])
    X2 = [X.reshape(X.shape[0], -1) for X in Xs]
    sse_c, ssb_c, G = np.zeros(C), np.zeros(C), np.zeros((C, C))
    for s in range(0, X2[0].shape[1], chunk):
        x = np.stack([np.asarray(X[:len(labels), s:s + chunk], dtype=np.float64)[order] for X in X2])  # (R, rows, d)
        x -= x.mean(axis=(0, 1))                         # 桁落ちを防ぐため全体平均を引く（結果は変わらない）
        x = x.reshape(R, C, n, -1)
        xt = x.mean(axis=0)                              # 試行の平均（seed で平均）(C, n, d)
        m = xt.mean(axis=1)                              # 刺激の平均 (C, d)
        sse_c += ((x - xt[None]) ** 2).sum(axis=(0, 2, 3))
        ssb_c += R * ((xt - m[:, None]) ** 2).sum(axis=(1, 2))
        with np.errstate(all="ignore"):                  # matmul は値が正しくても警告を出すことがあるため
            G += m @ m.T
    assert np.isfinite(sse_c).all() and np.isfinite(ssb_c).all() and np.isfinite(G).all(), "統計量に inf か nan"
    return sse_c, ssb_c, G, n, R


def components(stats, w=None):
    """分散成分 (σ_a², σ_b², σ_e²)。w は各刺激が選ばれた回数（bootstrap 用）。"""
    sse_c, ssb_c, G, n, R = stats
    C = len(sse_c)
    w = np.ones(C) if w is None else np.asarray(w, dtype=np.float64)
    with np.errstate(all="ignore"):                      # 同上
        ss_a = n * R * (w @ np.diag(G) - (w @ G @ w) / C)
    ms_e = (w @ sse_c) / (C * n * (R - 1))
    ms_b = (w @ ssb_c) / (C * (n - 1))
    ms_a = ss_a / (C - 1)
    return (ms_a - ms_b) / (n * R), (ms_b - ms_e) / R, ms_e


def bc_k(comp, n, C, k):
    """seed を k 個平均したときの BC（k = inf はサンプリングを除いた BC。σ_b² ≤ 0 なら nan）。"""
    s_a, s_b, s_e = comp
    if k == np.inf and s_b <= 0:
        return np.nan
    N = n * C
    return 1 + n * (C - 1) / (N - 1) * s_a / (s_b + (0 if k == np.inf else s_e / k))


def sampling_share(comp):
    _, s_b, s_e = comp
    return s_e / (max(s_b, 0) + s_e) if s_e > 0 else 0.0


def summarize_nan(boot, point):
    """Exp20 の summarize と同じバイアス補正パーセンタイル法。nan（BC(∞) が定義されない回）は除き、その割合も返す。"""
    raw = np.nanpercentile(boot, [2.5, 97.5], axis=-1)
    ci = raw + (point - np.nanmean(boot, axis=-1))
    to_last = lambda a: a.transpose(*range(1, a.ndim), 0)
    return to_last(ci), to_last(raw), np.isnan(boot).mean(axis=-1)


# ── 読み込み ──────────────────────────────────────────────────────────────

def load_stage(d, stage, n_rows):
    X = np.load(f"{d}/{STAGE_FILES[stage]}", mmap_mode="r")
    if stage == "S2_tok":
        return token_mean(X, n_rows)
    if stage == "S2":
        return X[:n_rows]
    return np.asarray(X[:n_rows], dtype=np.float32)


def trial_shuffled_labels(labels):
    """刺激のラベルを試行ごとに 1 回並べ替える。行 = 試行で、seed は別の配列の同じ行なので、同じ試行の seed はまとめたまま。"""
    return np.random.default_rng(SEED).permutation(labels)


def analyze_set(models, stages, rows, labels, W):
    n = int(np.unique(labels, return_counts=True)[1][0])
    C = len(np.unique(labels))
    shuf = trial_shuffled_labels(labels)
    shape = (len(models), len(STAGES_B))
    res = {f"bc{k}": np.full(shape, np.nan) for k in ("1", "4", "inf")}
    res.update({f"bc{k}_shuf": np.full(shape, np.nan) for k in ("1", "inf")})
    res.update(share=np.full(shape, np.nan), comp=np.full(shape + (3,), np.nan),
               bc_r0_closed=np.full(shape, np.nan))
    for k in ("1", "4", "inf"):
        res[f"boot_bc{k}"] = np.full(shape + (len(W),), np.nan)
    res["boot_share"] = np.full(shape + (len(W),), np.nan)
    res["sb_pos_frac"] = np.full(shape, np.nan)
    for i, m in enumerate(models):
        for stage in stages:
            j = STAGES_B.index(stage)
            Xs = [load_stage(seed_dir(r, m), stage, len(rows)) for r in SEEDS]
            st = nested_stats(Xs, labels)
            st_shuf = nested_stats(Xs, shuf)
            comp = components(st)
            comps_w = [components(st, w) for w in W]
            res["comp"][i, j] = comp
            for k, key in zip(KS, ("1", "4", "inf")):
                res[f"bc{key}"][i, j] = bc_k(comp, n, C, k)
                res[f"boot_bc{key}"][i, j] = [bc_k(c, n, C, k) for c in comps_w]
            # BC(∞) は、bootstrap の 97.5% 以上で σ_b² > 0 のときだけ出す（試行の違いが seed のばらつきと区別できないときは定義しない）
            res["sb_pos_frac"][i, j] = np.mean([c[1] > 0 for c in comps_w])
            if res["sb_pos_frac"][i, j] < SB_POS_MIN:
                res["bcinf"][i, j] = np.nan
                res["boot_bcinf"][i, j] = np.nan
            res["share"][i, j] = sampling_share(comp)
            res["boot_share"][i, j] = [sampling_share(c) for c in comps_w]
            comp_shuf = components(st_shuf)
            res["bc1_shuf"][i, j] = bc_k(comp_shuf, n, C, 1)
            res["bcinf_shuf"][i, j] = bc_k(comp_shuf, n, C, np.inf)
            # 検算：r = 0 だけで閉じた式から出した BC（Exp20 と同じ計算）は BC(1) に近いはず
            res["bc_r0_closed"][i, j] = bc_from_stats(*stimulus_stats(Xs[0], [labels])[0])
            print(f"  {LABELS[m]:>6} {stage:<12} BC(1) = {res['bc1'][i, j]:.4f}  BC(4) = {res['bc4'][i, j]:.4f}  "
                  f"BC(∞) = {res['bcinf'][i, j]:.4f}  サンプリングの割合 = {res['share'][i, j]:.3f}  "
                  f"r=0 の閉じた式 {res['bc_r0_closed'][i, j]:.4f}  Shuffled BC(1) = {res['bc1_shuf'][i, j]:.4f}")
    for key in ("bc1", "bc4", "bcinf", "share"):
        bkey = f"boot_{key}"
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 全部 nan の列（BC(∞) を定義しない段階）の警告
            res[f"ci_{key}"], res[f"ci_raw_{key}"], res[f"nanfrac_{key}"] = summarize_nan(res[bkey], res[key])
            d = res[bkey][0] - res[bkey][1]               # 40sess − 1sess（paired）
            res[f"diff_ci_{key}"], res[f"diff_ci_raw_{key}"], _ = summarize_nan(d, res[key][0] - res[key][1])
    for k in ("ci_share", "diff_ci_share"):              # 割合なので、補正でずれた区間を [0, 1]（差は [−1, 1]）に収める
        res[k] = np.clip(res[k], -1 if k.startswith("diff") else 0, 1)
    return res


def check_deterministic(models, rows_A):
    """S0・S1 は乱数を使わないので、全部の seed でビット単位で同じはず。"""
    ok = True
    for m in models:
        for name in ("s0_ridge.npy", "s1_backbone.npy"):
            a = np.load(f"{seed_dir(0, m)}/{name}", mmap_mode="r")
            for r in SEEDS[1:]:
                b = np.load(f"{seed_dir(r, m)}/{name}", mmap_mode="r")
                same = all(np.array_equal(a[s:s + 200], b[s:s + 200]) for s in range(0, len(rows_A), 200))
                ok &= same
                print(f"  {LABELS[m]:>6} {name:<16} r = 0 と r = {r} がビット単位で同じ: {same}")
    return ok


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(res, K, path):
    ink, muted, grid = "#222222", "#666666", "#e6e6e6"
    names = ["S2 prior", "S2 prior\n(token mean)", "S3\nDreamSim", "S3\nCLIP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    x = np.arange(len(STAGES_B))
    markers = {"1": "o", "inf": "D"}
    ax = axes[0]
    for i, m in enumerate(MODELS):
        lab = LABELS[m]
        for key, dx in (("1", -0.08), ("inf", 0.08)):
            xo = x + (i - 0.5) * 0.3 + dx
            y, ci = res[f"bc{key}"][i], res[f"ci_bc{key}"][i]
            ax.vlines(xo, ci[:, 0], ci[:, 1], color=COLORS[lab], lw=1.5, zorder=2)
            filled = key == "1"                          # BC(1) は塗った丸、BC(∞) は白抜きのひし形
            ax.plot(xo, y, ls="none", marker=markers[key], ms=7, color=COLORS[lab],
                    mec="white" if filled else COLORS[lab], mew=1.5, mfc=COLORS[lab] if filled else "white",
                    zorder=3, label=f"{lab} {'BC(1)' if filled else 'BC(∞)'}")
    for j in range(len(STAGES_B)):                       # BC(∞) を定義しない段階は、点がないことを段階ごとに 1 つの文字で示す
        na = [LABELS[m] for i, m in enumerate(MODELS) if np.isnan(res["bcinf"][i, j]) and not np.isnan(res["bc1"][i, j])]
        if na:
            top = np.nanmax(res["ci_bc1"][:, j, 1])
            ax.annotate("BC(∞) n/a" + ("" if len(na) == len(MODELS) else f" ({na[0]})"), (x[j], top),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=muted)
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("BC (95% CI)", color=ink)
    ax.set_title(f"(A) BC with one sample vs. sampling removed (K = {K})", loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))  # 点と重ならないよう、図の下に置く
    ax = axes[1]
    for i, m in enumerate(MODELS):
        lab = LABELS[m]
        xo = x + (i - 0.5) * 0.25
        y, ci = res["share"][i], res["ci_share"][i]
        ax.vlines(xo, ci[:, 0], ci[:, 1], color=COLORS[lab], lw=1.5, zorder=2)
        ax.plot(xo, y, ls="none", marker="o", ms=7, color=COLORS[lab], mec="white", mew=1.5, label=lab, zorder=3)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of within-stimulus variance\nfrom sampling (seed)", color=ink)
    ax.set_title("(B) Sampling share", loc="left", color=ink, fontsize=11)
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"図を保存しました: {path}")


# ── 本体 ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stim = np.load(f"{seed_dir(0, MODELS[0])}/meta.npz")["stim"]
    for r in SEEDS:
        for m in MODELS:
            meta = np.load(f"{seed_dir(r, m)}/meta.npz")
            assert np.array_equal(meta["stim"], stim), f"r = {r} の {m} の試行の並びが Exp20 と違う"
    N = len(stim)
    load = lambda r, m, f: (np.load(f"{seed_dir(r, m)}/{f}", mmap_mode="r") if os.path.exists(f"{seed_dir(r, m)}/{f}")
                            else np.zeros(N, bool))
    da = [load(r, m, "done_prior.npy") for r in SEEDS for m in MODELS]
    db = [load(r, m, "done_unclip.npy") for r in SEEDS for m in MODELS]
    K_A, rows_A = complete_prefix(stim, da)
    K_B, rows_B = complete_prefix(stim, da + db)
    print(f"全部の seed でそろった刺激：パス A まで K_A = {K_A}、パス B まで K_B = {K_B}")
    assert K_A > 0, "パス A が済んだ刺激がない"
    with open(f"{OUT_SEED}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), K_A=K_A, K_B=K_B)) + "\n")

    print("S0・S1 が seed によらないことの確認")
    deterministic = check_deterministic(MODELS, rows_A)

    if K_B > 0:
        print("S3 を埋め込む")
        embedders = load_embedders("mps")
        for r in SEEDS:
            for m in MODELS:
                d = seed_dir(r, m)
                embed_rows(np.load(f"{d}/s3_images.npy", mmap_mode="r"), rows_B, d, "s3", embedders)

    sets = {"A": (K_A, rows_A, STAGES_A)}
    if K_B > 0:
        sets["B"] = (K_B, rows_B, STAGES_B)
    results = {}
    for name, (K, rows, stages) in sets.items():
        labels = stim[rows]
        W = np.random.default_rng(SEED).multinomial(K, np.ones(K) / K, size=N_BOOT)
        print(f"集合 {name}（K = {K}、{len(rows)} 行 × seed {len(SEEDS)}）")
        results[name] = analyze_set(MODELS, stages, rows, labels, W)

    out = dict(models=np.array([LABELS[m] for m in MODELS]), stages=np.array(STAGES_B), seeds=np.array(SEEDS),
               ks=np.array(KS), K_A=K_A, K_B=K_B, deterministic_S0_S1=deterministic, n_boot=N_BOOT, seed=SEED)
    for name, res in results.items():
        for k, v in res.items():
            out[f"{k}_{name}"] = v
    npz_path = os.path.join(OUTPUT_DIR, "exp21_seed_decomposition.npz")
    np.savez(npz_path, **out)
    print(f"結果を保存しました: {npz_path}")
    main_set = "B" if K_B > 0 else "A"
    plot(results[main_set], sets[main_set][0], os.path.join(OUTPUT_DIR, "exp21_seed_decomposition.png"))

    for name, res in results.items():
        print(f"\n集合 {name}（K = {sets[name][0]}）  値 [95% CI（バイアス補正）]")
        for j, stage in enumerate(STAGES_B):
            if np.isnan(res["bc1"][0, j]):
                continue
            for key, label in (("bc1", "BC(1)"), ("bcinf", "BC(∞)"), ("share", "サンプリングの割合")):
                cells = [f"{res[key][i, j]:.4f} [{res[f'ci_{key}'][i, j, 0]:.4f}, {res[f'ci_{key}'][i, j, 1]:.4f}]"
                         for i in range(len(MODELS))]
                print(f"  {stage:<12} {label:<10} 40sess {cells[0]}  1sess {cells[1]}  "
                      f"差 {res[key][0, j] - res[key][1, j]:+.4f} "
                      f"[{res[f'diff_ci_{key}'][j, 0]:+.4f}, {res[f'diff_ci_{key}'][j, 1]:+.4f}]")
    print("完了!")


if __name__ == "__main__":
    main()
