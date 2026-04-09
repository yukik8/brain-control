"""
analyze_bc_vs_accuracy.py
--------------------------
Experiment 6: BC と デコーダー精度（mean_r）の関係

- BCがmean_rの単純な言い換えに過ぎないなら相関が高くなる
- 独立した指標であれば、精度が同じでもBCが異なる条件が存在するはず
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compute_bc import (
    load_brain_data, load_image_features, run_one,
    DATA_DIR, FEAT_FILE, ALL_SUBJECTS, ALL_ROIS, ALL_LAYERS, N_SHUFFLE
)


def collect_results():
    """全条件の結果を収集して返す。"""
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


def plot_bc_vs_accuracy(results):
    mean_r_all = np.array([r["mean_r"] for r in results])
    bc_all     = np.array([r["bc_mean"] for r in results])
    layers     = [r["layer"] for r in results]
    rois       = [r["roi"] for r in results]

    # 全条件の散布図
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── (1) 全条件 scatter ──
    ax = axes[0]
    ax.scatter(mean_r_all, bc_all, alpha=0.3, s=15, color="steelblue")
    slope, intercept, r, p, _ = stats.linregress(mean_r_all, bc_all)
    x_line = np.linspace(mean_r_all.min(), mean_r_all.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
            label=f"r={r:.3f}, p={p:.2e}")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Decoding accuracy (mean Pearson r)")
    ax.set_ylabel("BC")
    ax.set_title("All conditions (N=400)")
    ax.legend(fontsize=8)

    # ── (2) 層ごとに色分け ──
    ax = axes[1]
    cmap = plt.get_cmap("viridis", len(ALL_LAYERS))
    for i, layer in enumerate(ALL_LAYERS):
        idx = [j for j, r in enumerate(results) if r["layer"] == layer]
        ax.scatter(mean_r_all[idx], bc_all[idx],
                   alpha=0.5, s=20, color=cmap(i), label=layer)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Decoding accuracy (mean Pearson r)")
    ax.set_ylabel("BC")
    ax.set_title("By layer")
    ax.legend(fontsize=7, ncol=2)

    # ── (3) ROIごとに色分け ──
    ax = axes[2]
    roi_short = {r: r.replace("ROI_", "") for r in ALL_ROIS}
    cmap2 = plt.get_cmap("tab10", len(ALL_ROIS))
    for i, roi in enumerate(ALL_ROIS):
        idx = [j for j, r in enumerate(results) if r["roi"] == roi]
        ax.scatter(mean_r_all[idx], bc_all[idx],
                   alpha=0.5, s=20, color=cmap2(i), label=roi_short[roi])
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Decoding accuracy (mean Pearson r)")
    ax.set_ylabel("BC")
    ax.set_title("By ROI")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig("bc_vs_accuracy.png", dpi=150)
    print(f"\nSaved: bc_vs_accuracy.png")

    # 全体の相関
    print(f"\n{'='*50}")
    print(f"  BC vs mean_r 相関 (全400条件)")
    print(f"{'='*50}")
    print(f"  Pearson r = {r:.4f},  p = {p:.2e}")
    print(f"  → {'BCはmean_rと強く相関' if abs(r) > 0.8 else 'BCはmean_rと中程度の相関' if abs(r) > 0.5 else 'BCはmean_rと独立している（弱い相関）'}")

    # 層ごとの相関
    print(f"\n  層ごとの相関:")
    for layer in ALL_LAYERS:
        idx = [j for j, res in enumerate(results) if res["layer"] == layer]
        r_l, _, r_val, p_l, _ = stats.linregress(mean_r_all[idx], bc_all[idx])
        print(f"    {layer}: r={r_val:.3f}, p={p_l:.3f}")

    return r, p


if __name__ == "__main__":
    print("Collecting results (this may take a while)...")
    results = collect_results()
    print("\nPlotting...")
    plot_bc_vs_accuracy(results)
    print("\nDone!")
