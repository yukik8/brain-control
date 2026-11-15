"""
exp22_multisubject_bc.py
------------------------
Experiment 22: NSD × MindEye2 での試行ごとの BC を、被験者 1・2・5・7 で比べる

被験者ごとに Exp20 と同じ解析（閉じた式の BC、Shuffled、刺激をリサンプリングする bootstrap、S3 のペア識別精度）を行い、
4 人のまとめ（平均と [最小, 最大]、予測 P4〜P6 が成り立つ被験者の数）を出す。
被験者 1 は Exp20 の出力（$EXP20/out）、ほかの被験者は $EXP20/out_subj0<s>（experiments/exp22_run.sh）。
Exp20 の関数をそのまま使う（CLIP は QuickGELU）。

  - P4：S3 の BC は両モデルで 1 より大きい（信頼区間の下限 > 1）
  - P5：S3 で 40sess > 1sess（差の信頼区間が 0 を含まない）
  - P6：トークン平均で S1 > S2 > S3（S3 は DreamSim）

被験者が 4 人しかいないので、被験者の間の検定はしない。被験者ごとの信頼区間と、成り立つ人数で示す。

計画：~/brain-control/plan_exp21_23.md の §4（予測は生成の前に固定した）

使い方（解析用の venv で、リポジトリの直下で実行する）：
    venv/bin/python experiments/exp22_multisubject_bc.py
    venv/bin/python experiments/exp22_multisubject_bc.py --plot_only    # npz から図だけ描き直す
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
from exp20_status import complete_prefix
from exp20_mindeye2_bc import (EXP20, GT_PATH, SEED, N_BOOT, MIN_KB, COLORS, STAGES, EMBEDDERS,
                               load_embedders, embed_rows, stimulus_stats, bc_from_stats, load_stage, summarize)

# ── 設定 ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = "outputs"
SUBJECTS   = [1, 2, 5, 7]
VARIANTS   = ["40sess", "1sess"]


def out_root(subj):
    return f"{EXP20}/out" if subj == 1 else f"{EXP20}/out_subj0{subj}"


def models_of(subj):
    return [f"final_subj0{subj}_pretrained_{v}_24bs" for v in VARIANTS]


# ── 1 人分の解析（Exp20 と同じ）──────────────────────────────────────────────

def analyze_subject(subj, embedders):
    root = out_root(subj)
    dirs = [f"{root}/{m}" for m in models_of(subj)]
    metas = [np.load(f"{d}/meta.npz") for d in dirs]
    stim = metas[0]["stim"]
    assert all(np.array_equal(m["stim"], stim) for m in metas), f"被験者 {subj}：モデルの間で試行の並びが違う"
    N = len(stim)
    load = lambda d, f: np.load(f"{d}/{f}", mmap_mode="r") if os.path.exists(f"{d}/{f}") else np.zeros(N, bool)
    da = [load(d, "done_prior.npy") for d in dirs]
    db = [load(d, "done_unclip.npy") for d in dirs]
    K_A, rows_A = complete_prefix(stim, da)
    K_B, rows_B = complete_prefix(stim, da + db)
    print(f"被験者 {subj}：K_A = {K_A}、K_B = {K_B}" + ("（S3 は予備的な結果）" if 0 < K_B < MIN_KB else ""))
    with open(f"{root}/analysis_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now().isoformat(timespec="seconds"), script="exp22", K_A=K_A, K_B=K_B)) + "\n")

    ident = np.full((len(VARIANTS), len(EMBEDDERS)), np.nan)
    if K_B > 0:
        import torch
        for d in dirs:
            embed_rows(np.load(f"{d}/s3_images.npy", mmap_mode="r"), rows_B, d, "s3", embedders)
        gt = torch.load(GT_PATH, map_location="cpu").numpy()
        stims_B = np.unique(stim[rows_B])
        embed_rows(gt, stims_B, root, "gt", embedders)
        img_index = np.searchsorted(stims_B, stim[rows_B])
        for e, emb in enumerate(EMBEDDERS):
            gt_emb = np.load(f"{root}/gt_{emb}.npy")[stims_B]
            for i, d in enumerate(dirs):
                with np.errstate(all="ignore"):              # matmul の見かけだけの警告（値は正しい）
                    ident[i, e] = pairwise_identification(np.load(f"{d}/s3_{emb}.npy")[rows_B], gt_emb, img_index)

    shape = (len(VARIANTS), len(STAGES))
    res = {"K_A": K_A, "K_B": K_B, "identification": ident, "coco73k": metas[0]["coco73k"], "stim": stim}
    for name, (K, rows, stages) in {"A": (K_A, rows_A, ["S0", "S1", "S2", "S1_tok", "S2_tok"]),
                                    "B": (K_B, rows_B, STAGES if K_B > 0 else [])}.items():
        out = {"bc": np.full(shape, np.nan), "bc_shuf": np.full(shape, np.nan),
               "boot": np.full(shape + (N_BOOT,), np.nan)}
        if K > 0 and stages:
            labels = stim[rows]
            shuf = np.random.default_rng(SEED).permutation(labels)
            W = np.random.default_rng(SEED).multinomial(K, np.ones(K) / K, size=N_BOOT)
            for i, d in enumerate(dirs):
                for stage in stages:
                    j = STAGES.index(stage)
                    st, st_shuf = stimulus_stats(load_stage(d, stage, len(rows)), [labels, shuf])
                    out["bc"][i, j] = bc_from_stats(*st)
                    out["bc_shuf"][i, j] = bc_from_stats(*st_shuf)
                    out["boot"][i, j] = [bc_from_stats(*st, w=w) for w in W]
                    print(f"  被験者 {subj} {VARIANTS[i]:>6} 集合 {name} {stage:<12} BC = {out['bc'][i, j]:.4f}"
                          f"  Shuffled = {out['bc_shuf'][i, j]:.4f}")
        out["ci"], out["ci_raw"] = summarize(out["boot"], out["bc"])
        out["diff_ci"], out["diff_ci_raw"] = summarize(out["boot"][0] - out["boot"][1], out["bc"][0] - out["bc"][1])
        for k, v in out.items():
            res[f"{k}_{name}"] = v
    return res


# ── 予測の判定 ────────────────────────────────────────────────────────────

def predictions(res):
    """P4〜P6 が成り立つか（S3 がない被験者は nan）。"""
    if res["K_B"] == 0:
        return dict(P4=np.nan, P5=np.nan, P6=np.nan)
    j = {s: STAGES.index(s) for s in STAGES}
    ci, bc, dci = res["ci_B"], res["bc_B"], res["diff_ci_B"]
    p4 = bool(all(ci[i, j[s], 0] > 1 for i in range(len(VARIANTS)) for s in ("S3_dreamsim", "S3_clip")))
    p5 = bool(all(dci[j[s], 0] > 0 for s in ("S3_dreamsim", "S3_clip")))
    p6 = bool(all(bc[i, j["S1_tok"]] > bc[i, j["S2_tok"]] > bc[i, j["S3_dreamsim"]] for i in range(len(VARIANTS))))
    return dict(P4=p4, P5=p5, P6=p6)


# ── 図 ────────────────────────────────────────────────────────────────────

def plot(results, path):
    ink, muted, grid = "#222222", "#666666", "#e6e6e6"
    show = ["S0", "S1", "S2", "S3_dreamsim", "S3_clip"]
    names = ["S0\nridge", "S1\nbackbone", "S2\nprior", "S3\nDreamSim", "S3\nCLIP"]
    j = [STAGES.index(s) for s in show]
    x = np.arange(len(show))
    subs = [s for s in SUBJECTS if s in results and results[s]["K_B"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    for i, v in enumerate(VARIANTS):
        Y = np.array([results[s]["bc_B"][i, j] for s in subs])        # (被験者, 段階)
        xo = x + (i - 0.5) * 0.25
        for y in Y:                                                    # 被験者ごとの点（小さく、薄く）
            ax.plot(xo, y, ls="none", marker="o", ms=4, color=COLORS[v], alpha=0.45, zorder=2)
        m = Y.mean(axis=0)
        line = np.ones(len(show), bool); line[-1] = False             # S3 の CLIP は同じ段階の別の埋め込みなので線でつながない
        ax.plot(xo[line], m[line], color=COLORS[v], lw=2, marker="o", ms=8, mec="white", mew=2, label=f"{v} (mean)", zorder=3)
        ax.plot(xo[-1:], m[-1:], ls="none", marker="s", ms=8, color=COLORS[v], mec="white", mew=2, zorder=3)
        ax.annotate(v, (xo[0], m[0]), xytext=(-0.3, m[0]), textcoords="data", ha="right",   # 2 本の平均が近いので、
                    va="bottom" if i == 0 else "top", color=ink, fontsize=10)                # 上下にずらし、点の左に置く
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.set_xticks(x, names)
    ax.set_xlim(-0.9, len(show) - 0.5)
    ax.set_ylabel("BC", color=ink)
    ax.set_title(f"(A) BC by stage: subjects {', '.join(map(str, subs))} (small dots) and mean", loc="left",
                 color=ink, fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    e = EMBEDDERS.index("dreamsim")
    js = STAGES.index("S3_dreamsim")
    pts = []
    for i, v in enumerate(VARIANTS):
        for s in subs:
            r = results[s]
            ax.scatter(r["identification"][i, e], r["bc_B"][i, js], s=50, color=COLORS[v], edgecolor="white",
                       linewidth=1.5, zorder=3, label=v if s == subs[0] else None)
            pts.append((f"subj {s}", r["identification"][i, e], r["bc_B"][i, js]))
    ax.axhline(1, color="#999999", ls="--", lw=1, zorder=1)
    ax.margins(x=0.3, y=0.2)
    lo, hi = ax.get_xlim()                               # 識別精度は 1 を超えないので、1 より大きい目盛りは出さない
    ax.set_xticks([t for t in np.arange(0.5, 1.0001, 0.05) if lo <= t <= hi])
    ax.set_xlabel("Pairwise identification, DreamSim (chance = 0.5)", color=ink)
    ax.set_ylabel("BC at S3 (DreamSim)", color=ink)
    ax.set_title("(B) S3: BC vs identification by subject", loc="left", color=ink, fontsize=11)
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
    place_labels(fig, axes[1], pts, muted)              # 重なりは表示の座標で調べるので、配置が決まってから
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"図を保存しました: {path}")


def place_labels(fig, ax, pts, color):
    """点の名前を、ほかの点・置いた名前・軸の外と重ならない最初の候補の位置に置く（なければ最初の候補）。"""
    cands = [((6, 3), "left", "bottom"), ((6, -3), "left", "top"), ((-6, 3), "right", "bottom"),
             ((-6, -3), "right", "top"), ((0, 7), "center", "bottom"), ((0, -7), "center", "top")]
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    r = 5 * fig.dpi / 72                                 # 点の半径（s = 50 で約 3.5 pt）に少し足したもの、ピクセル
    dots = [ax.transData.transform((x, y)) for _, x, y in pts]
    placed = []
    for name, x, y in pts:
        for k, ((dx, dy), ha, va) in enumerate(cands):
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=8, color=color)
            bb = t.get_window_extent(renderer)
            ok = (frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
                  and not any(bb.overlaps(b) for b in placed)
                  and not any(bb.x0 - r < px < bb.x1 + r and bb.y0 - r < py < bb.y1 + r for px, py in dots))
            if ok or k == len(cands) - 1:
                if not ok:                               # どこにも置けないときは最初の候補に戻す
                    t.remove()
                    t = ax.annotate(name, (x, y), xytext=cands[0][0], textcoords="offset points",
                                    ha=cands[0][1], va=cands[0][2], fontsize=8, color=color)
                placed.append(t.get_window_extent(renderer))
                break
            t.remove()


def results_from_npz(npz_path):
    """保存した npz から、図に使う値だけを戻す（--plot_only 用）。"""
    z = np.load(npz_path, allow_pickle=False)
    return {int(s): {k: z[f"s{s}_{k}"] for k in ("K_B", "bc_B", "identification")} for s in z["subjects"]}


# ── 本体 ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if "--plot_only" in sys.argv:                        # 解析はせず、npz から図だけ描き直す
        plot(results_from_npz(os.path.join(OUTPUT_DIR, "exp22_multisubject_bc.npz")),
             os.path.join(OUTPUT_DIR, "exp22_multisubject_bc.png"))
        return
    have = [s for s in SUBJECTS if os.path.exists(f"{out_root(s)}/{models_of(s)[1]}/meta.npz")]
    print(f"出力のある被験者：{have}")
    need_s3 = any(os.path.exists(f"{out_root(s)}/{models_of(s)[1]}/done_unclip.npy") for s in have)
    embedders = load_embedders("mps") if need_s3 else None
    results = {s: analyze_subject(s, embedders) for s in have}

    # 刺激の番号（all_images.pt の行）が、被験者の間で同じ COCO の画像を指すことの確認
    ref = results[have[0]]
    same_images = {}
    for s in have:
        r = results[s]
        a = dict(zip(ref["stim"], ref["coco73k"]))
        same_images[s] = bool(all(a.get(k, c) == c for k, c in zip(r["stim"], r["coco73k"])))
        print(f"被験者 {s}：刺激の番号と COCO の画像の対応が被験者 {have[0]} と同じ {same_images[s]}")

    preds = {s: predictions(results[s]) for s in have}
    out = dict(subjects=np.array(have), variants=np.array(VARIANTS), stages=np.array(STAGES),
               embedders=np.array(EMBEDDERS), n_boot=N_BOOT, seed=SEED,
               same_images=np.array([same_images[s] for s in have]))
    for s in have:
        for k, v in results[s].items():
            out[f"s{s}_{k}"] = v
        for p, v in preds[s].items():
            out[f"s{s}_{p}"] = v
    S = [s for s in have if results[s]["K_B"] > 0]
    if S:
        allB = np.stack([results[s]["bc_B"] for s in S])             # (被験者, モデル, 段階)
        out.update(mean_bc_B=allB.mean(0), min_bc_B=allB.min(0), max_bc_B=allB.max(0), subjects_B=np.array(S))
    npz_path = os.path.join(OUTPUT_DIR, "exp22_multisubject_bc.npz")
    np.savez(npz_path, **out)
    print(f"結果を保存しました: {npz_path}")
    if S:
        plot(results, os.path.join(OUTPUT_DIR, "exp22_multisubject_bc.png"))

    print("\n被験者ごとの BC（集合 B）[95% CI]、40sess − 1sess [95% CI]")
    for s in S:
        r = results[s]
        print(f"被験者 {s}（K_B = {r['K_B']}）  予測 " + "  ".join(f"{p} {'成立' if v else '不成立'}" for p, v in preds[s].items()))
        for j, stage in enumerate(STAGES):
            cells = [f"{r['bc_B'][i, j]:.4f} [{r['ci_B'][i, j, 0]:.4f}, {r['ci_B'][i, j, 1]:.4f}]" for i in range(len(VARIANTS))]
            print(f"  {stage:<12} 40sess {cells[0]}  1sess {cells[1]}  差 {r['bc_B'][0, j] - r['bc_B'][1, j]:+.4f} "
                  f"[{r['diff_ci_B'][j, 0]:+.4f}, {r['diff_ci_B'][j, 1]:+.4f}]")
        print(f"  識別精度 DreamSim {r['identification'][:, 0].round(4).tolist()}  CLIP {r['identification'][:, 1].round(4).tolist()}")
    if S:
        print(f"\n{len(S)} 人のまとめ（集合 B）：平均 [最小, 最大]")
        for j, stage in enumerate(STAGES):
            cells = [f"{out['mean_bc_B'][i, j]:.3f} [{out['min_bc_B'][i, j]:.3f}, {out['max_bc_B'][i, j]:.3f}]"
                     for i in range(len(VARIANTS))]
            print(f"  {stage:<12} 40sess {cells[0]}  1sess {cells[1]}")
        for p in ("P4", "P5", "P6"):
            print(f"  {p}：{sum(bool(preds[s][p]) for s in S)} / {len(S)} 人で成立")
    print("完了!")


if __name__ == "__main__":
    main()
