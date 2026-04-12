"""
exp09_bc_residual.py
---------------------
Experiment 9: BC vs mean_r 残差分析

BC と mean_r の線形回帰からの残差を分析する。
「精度が同じなのに BC が高い / 低い」条件は何か？

- 全400条件（5被験者 × 10ROI × 8層）
- 残差 = BC_actual - BC_predicted（from mean_r）
- 残差を層・ROI別に可視化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compute_bc import (
    load_brain_data, load_image_features, run_one,
    DATA_DIR, FEAT_FILE, ALL_SUBJECTS, ALL_ROIS, ALL_LAYERS, N_SHUFFLE
)


def collect_all():
    results = []
    for subject in ALL_SUBJECTS:
        brain_file = f"{DATA_DIR}/{subject}.mat"
        print(f"\n{subject}...")
        brain_all = {}
        datatype = stim_id = cat_id = None
        for roi in ALL_ROIS:
            b, dt, sid, cid = load_brain_data(brain_file, roi_key=roi)
            brain_all[roi] = b
            datatype, stim_id, cat_id = dt, sid, cid

        for layer in ALL_LAYERS:
            feat, img_ids = load_image_features(FEAT_FILE, layer=layer)
            for roi in ALL_ROIS:
                r = run_one(roi, layer, brain_all, datatype, stim_id, cat_id,
                            feat, img_ids, shuffle_mode="across")
                r["subject"] = subject
                results.append(r)
                print(f"  {layer} {roi}: r={r['mean_r']:.3f} BC={r['bc_mean']:.3f}", end="\r")
    return results


def analyze_residuals(results):
    mean_r = np.array([r["mean_r"] for r in results])
    bc     = np.array([r["bc_mean"] for r in results])
    layers  = [r["layer"]   for r in results]
    rois    = [r["roi"]     for r in results]
    subjects = [r["subject"] for r in results]

    # 全体線形回帰
    slope, intercept, r_val, p_val, _ = stats.linregress(mean_r, bc)
    bc_pred = slope * mean_r + intercept
    residuals = bc - bc_pred

    print(f"\n{'='*55}")
    print(f"  全体回帰: BC = {slope:.4f} * mean_r + {intercept:.4f}")
    print(f"  r = {r_val:.4f},  p = {p_val:.2e}")
    print(f"  残差: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    layer_cmap = plt.get_cmap("viridis", len(ALL_LAYERS))
    roi_short  = {r: r.replace("ROI_", "") for r in ALL_ROIS}

    # (1) 残差の分布（層別）
    ax = axes[0, 0]
    layer_residuals = {l: [] for l in ALL_LAYERS}
    for i, res in enumerate(results):
        layer_residuals[res["layer"]].append(residuals[i])

    positions = range(len(ALL_LAYERS))
    bp = ax.boxplot([layer_residuals[l] for l in ALL_LAYERS],
                    positions=positions, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(layer_cmap(i))
    ax.set_xticks(positions)
    ax.set_xticklabels(ALL_LAYERS, fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Residual (BC - predicted)")
    ax.set_title("Residual distribution by layer")

    # (2) 残差の分布（ROI別）
    ax = axes[0, 1]
    roi_residuals = {r: [] for r in ALL_ROIS}
    for i, res in enumerate(results):
        roi_residuals[res["roi"]].append(residuals[i])

    cmap2 = plt.get_cmap("tab10", len(ALL_ROIS))
    bp2 = ax.boxplot([roi_residuals[r] for r in ALL_ROIS],
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for i, patch in enumerate(bp2["boxes"]):
        patch.set_facecolor(cmap2(i))
    ax.set_xticks(range(len(ALL_ROIS)))
    ax.set_xticklabels([roi_short[r] for r in ALL_ROIS], fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Residual (BC - predicted)")
    ax.set_title("Residual distribution by ROI")

    # (3) 層 × ROI の残差ヒートマップ
    ax = axes[1, 0]
    heatmap = np.zeros((len(ALL_LAYERS), len(ALL_ROIS)))
    for i, res in enumerate(results):
        li = ALL_LAYERS.index(res["layer"])
        ri = ALL_ROIS.index(res["roi"])
        heatmap[li, ri] += residuals[i]
    counts = len(ALL_SUBJECTS)
    heatmap /= counts  # 被験者平均

    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(heatmap).max(), vmax=np.abs(heatmap).max())
    ax.set_xticks(range(len(ALL_ROIS)))
    ax.set_xticklabels([roi_short[r] for r in ALL_ROIS], fontsize=8)
    ax.set_yticks(range(len(ALL_LAYERS)))
    ax.set_yticklabels(ALL_LAYERS, fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean residual")
    ax.set_title("Residual heatmap (layer x ROI, averaged over subjects)")

    # (4) BC vs mean_r（残差を色で表現）
    ax = axes[1, 1]
    sc = ax.scatter(mean_r, bc, c=residuals, cmap="RdBu_r", alpha=0.6, s=20,
                    vmin=-np.abs(residuals).max(), vmax=np.abs(residuals).max())
    x_line = np.linspace(mean_r.min(), mean_r.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", lw=1.5,
            label=f"r={r_val:.3f}")
    plt.colorbar(sc, ax=ax, label="Residual")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Decoding accuracy (mean_r)")
    ax.set_ylabel("BC")
    ax.set_title("BC vs mean_r (residual color-encoded)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("outputs/bc_residual.png", dpi=150)
    print("Saved: outputs/bc_residual.png")

    # 層別の残差平均（表示）
    print(f"\n  層別 平均残差（被験者・ROI平均）:")
    print(f"  {'Layer':<7} {'mean_resid':>11} {'std_resid':>10}")
    print(f"  {'─'*30}")
    for l in ALL_LAYERS:
        res_l = [residuals[i] for i, r in enumerate(results) if r["layer"] == l]
        print(f"  {l:<7} {np.mean(res_l):>11.4f} {np.std(res_l):>10.4f}")

    # ROI別の残差平均
    print(f"\n  ROI別 平均残差（被験者・層平均）:")
    print(f"  {'ROI':<10} {'mean_resid':>11} {'std_resid':>10}")
    print(f"  {'─'*33}")
    for roi in ALL_ROIS:
        res_r = [residuals[i] for i, r in enumerate(results) if r["roi"] == roi]
        print(f"  {roi_short[roi]:<10} {np.mean(res_r):>11.4f} {np.std(res_r):>10.4f}")


if __name__ == "__main__":
    print("Collecting all 400 conditions (this may take a while)...")
    results = collect_all()
    print("\nAnalyzing residuals...")
    analyze_residuals(results)
    print("\nDone!")
