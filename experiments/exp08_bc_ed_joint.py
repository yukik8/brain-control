"""
exp08_bc_ed_joint.py
---------------------
Experiment 8: BC × ED 合同分析

BC と ED（Effective Dimensionality）の2次元構造を可視化する。

- 軸: x=ED, y=BC
- 色: 層（cnn1〜cnn8）
- マーカー形状: ROI
- 設定: Subject1, 全ROI × 全層
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compute_bc import (
    load_brain_data, load_image_features, run_one,
    DATA_DIR, FEAT_FILE, ALL_ROIS, ALL_LAYERS, N_SHUFFLE
)


def collect_subject1():
    subject = "Subject1"
    brain_file = f"{DATA_DIR}/{subject}.mat"
    print(f"Loading brain data: {subject}")
    brain_all = {}
    datatype = stim_id = cat_id = None
    for roi in ALL_ROIS:
        b, dt, sid, cid = load_brain_data(brain_file, roi_key=roi)
        brain_all[roi] = b
        datatype, stim_id, cat_id = dt, sid, cid
        print(f"  {roi}: {b.shape[1]} voxels")

    results = []
    for layer in ALL_LAYERS:
        feat, img_ids = load_image_features(FEAT_FILE, layer=layer)
        for roi in ALL_ROIS:
            print(f"  {layer} | {roi}", end="  →  ", flush=True)
            r = run_one(roi, layer, brain_all, datatype, stim_id, cat_id,
                        feat, img_ids, shuffle_mode="across")
            r["subject"] = subject
            results.append(r)
            print(f"BC={r['bc_mean']:.4f}  ED={r['ed']:.2f}")

    return results


def plot_bc_ed(results):
    bc   = np.array([r["bc_mean"] for r in results])
    ed   = np.array([r["ed"]      for r in results])
    layers = [r["layer"] for r in results]
    rois   = [r["roi"]   for r in results]

    layer_cmap = plt.get_cmap("viridis", len(ALL_LAYERS))
    layer_color = {l: layer_cmap(i) for i, l in enumerate(ALL_LAYERS)}

    roi_short = {r: r.replace("ROI_", "") for r in ALL_ROIS}
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"]
    roi_marker = {r: markers[i] for i, r in enumerate(ALL_ROIS)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── (1) BC × ED scatter（層で色分け、ROIでマーカー）──
    ax = axes[0]
    for i, r in enumerate(results):
        ax.scatter(ed[i], bc[i],
                   color=layer_color[layers[i]],
                   marker=roi_marker[rois[i]],
                   s=80, alpha=0.8, edgecolors="none")

    # 層の凡例
    for i, l in enumerate(ALL_LAYERS):
        ax.scatter([], [], color=layer_cmap(i), s=60, label=l)
    leg1 = ax.legend(title="Layer", fontsize=8, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)

    # ROIの凡例
    for i, roi in enumerate(ALL_ROIS):
        ax.scatter([], [], color="gray", marker=markers[i],
                   s=60, label=roi_short[roi])
    ax.legend(title="ROI", fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 0.45))

    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Effective Dimensionality (ED)")
    ax.set_ylabel("BC")
    ax.set_title("BC x ED  (Subject1, all ROIs x all layers)")

    # ── (2) 層ごとの BC・ED 平均軌跡 ──
    ax = axes[1]
    layer_bc_mean = [np.mean([r["bc_mean"] for r in results if r["layer"] == l])
                     for l in ALL_LAYERS]
    layer_ed_mean = [np.mean([r["ed"]      for r in results if r["layer"] == l])
                     for l in ALL_LAYERS]

    ax.plot(layer_ed_mean, layer_bc_mean, "k-", lw=1, alpha=0.4, zorder=1)
    for i, l in enumerate(ALL_LAYERS):
        ax.scatter(layer_ed_mean[i], layer_bc_mean[i],
                   color=layer_cmap(i), s=120, zorder=2, label=l)
        ax.annotate(l, (layer_ed_mean[i], layer_bc_mean[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Mean ED (averaged over ROIs)")
    ax.set_ylabel("Mean BC (averaged over ROIs)")
    ax.set_title("Layer trajectory in BC x ED space\n(Subject1, ROI-averaged)")

    fig.tight_layout()
    fig.savefig("outputs/bc_ed_joint.png", dpi=150, bbox_inches="tight")
    print("\nSaved: outputs/bc_ed_joint.png")

    # Summary
    print(f"\n{'='*55}")
    print(f"  Layer   mean_BC   mean_ED")
    print(f"  {'─'*40}")
    for i, l in enumerate(ALL_LAYERS):
        print(f"  {l:<7} {layer_bc_mean[i]:.4f}    {layer_ed_mean[i]:.2f}")


if __name__ == "__main__":
    results = collect_subject1()
    plot_bc_ed(results)
    print("\nDone!")
