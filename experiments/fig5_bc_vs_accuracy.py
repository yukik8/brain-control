"""
fig5_bc_vs_accuracy.py
-----------------------
論文 Figure 5: BC vs 識別精度の散布図（ROI別）+ 精度マッチの強調

左パネル: 全 ROI の BC vs accuracy 散布図（r=0.967）
右パネル: V1+noise と HVC の精度マッチ箇所を拡大表示

キーメッセージ:
  - BC と精度は高相関（妥当性）
  - しかし精度が同じでも BC は有意に異なる（non-redundancy）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUTPUT_DIR  = "outputs"
NPZ15_PATH  = "outputs/exp15_roi_bc_vs_accuracy.npz"
NPZ16_PATH  = "outputs/exp16_same_acc_diff_bc.npz"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    d15 = np.load(NPZ15_PATH, allow_pickle=True)
    rois     = list(d15["rois"])
    bc_vals  = d15["bc_vals"]
    bc_stds  = d15["bc_stds"]
    acc_vals = d15["acc_vals"]
    r_val    = float(d15["r_bc_acc"])
    p_val    = float(d15["p_bc_acc"])

    d16 = np.load(NPZ16_PATH, allow_pickle=True)
    noise_levels = d16["noise_levels"]
    bc_noise     = d16["bc_v1_noise"]
    acc_noise    = d16["acc_v1_noise"]
    bc_hvc       = float(d16["bc_hvc"])
    acc_hvc      = float(d16["acc_hvc"])
    cohens_d     = float(d16["cohens_d"])
    p_match      = float(d16["p_val"])

    # σ=0.75 のインデックス
    match_idx   = np.argmin(np.abs(noise_levels - 0.75))
    bc_match    = float(bc_noise[match_idx])
    acc_match   = float(acc_noise[match_idx])

    # カラーマップ: ROI を高次視覚から低次視覚でグラデーション
    roi_order = ["V1", "V2", "V3", "V4", "LOC", "FFA", "PPA", "LVC", "HVC", "VC"]
    cmap      = plt.get_cmap("RdYlBu_r", len(roi_order))
    roi_color = {r: cmap(i) for i, r in enumerate(roi_order)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
    fig.subplots_adjust(wspace=0.40)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 左パネル: ROI散布図
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[0]

    for roi, bc, bc_s, acc in zip(rois, bc_vals, bc_stds, acc_vals):
        c = roi_color.get(roi, "gray")
        ax.scatter(acc, bc, color=c, s=80, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.errorbar(acc, bc, yerr=bc_s,
                    fmt="none", ecolor=c, elinewidth=1.0, capsize=3, zorder=2)
        # ラベルオフセット（重ならないよう調整）
        offsets = {"V1": (0.008, 0.001), "V2": (0.008, 0.001),
                   "V3": (0.008, 0.001), "HVC": (-0.012, 0.002),
                   "LOC": (0.008, 0.001), "VC": (0.008, -0.003)}
        dx, dy = offsets.get(roi, (0.008, 0.001))
        ax.text(acc + dx, bc + dy, roi,
                fontsize=8, color=c, fontweight="bold", va="center")

    # 回帰直線
    m, b_lin = np.polyfit(acc_vals, bc_vals, 1)
    x_line = np.linspace(acc_vals.min() - 0.02, acc_vals.max() + 0.02, 100)
    ax.plot(x_line, m * x_line + b_lin,
            color="#aaaaaa", linewidth=1.2, linestyle="--", zorder=1,
            label=f"$r = {r_val:.3f}$, $p < 0.001$")

    ax.set_xlabel("Identification accuracy (LOO cosine similarity)", fontsize=10)
    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_title("(A)  BC vs. identification accuracy across ROIs\n(Subject 1, AlexNet relu7)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 右パネル: 精度マッチ強調
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ax = axes[1]

    # V1+noise のカーブ全体（薄く）
    ax.plot(acc_noise, bc_noise, "o-", color="#2c7bb6",
            linewidth=1.5, markersize=4, alpha=0.4, zorder=2,
            label="V1 + noise (varying $\\sigma$)")

    # ノイズカーブの σ=0 と σ=2.5 にラベル
    ax.annotate("$\\sigma=0$", xy=(acc_noise[0], bc_noise[0]),
                xytext=(acc_noise[0] + 0.01, bc_noise[0] + 0.003),
                fontsize=7.5, color="#2c7bb6")
    ax.annotate(f"$\\sigma={noise_levels[-1]:.1f}$",
                xy=(acc_noise[-1], bc_noise[-1]),
                xytext=(acc_noise[-1] + 0.01, bc_noise[-1] - 0.004),
                fontsize=7.5, color="#2c7bb6")

    # マッチポイント（σ=0.75）を強調
    ax.scatter([acc_match], [bc_match], color="#2c7bb6",
               s=120, zorder=5, edgecolors="white", linewidths=1.5,
               label=f"V1+noise ($\\sigma$=0.75): BC={bc_match:.3f}")

    # HVC
    ax.scatter([acc_hvc], [bc_hvc], color="#555555",
               marker="s", s=120, zorder=5, edgecolors="white", linewidths=1.5,
               label=f"HVC: BC={bc_hvc:.3f}")
    ax.annotate("HVC", xy=(acc_hvc, bc_hvc),
                xytext=(acc_hvc + 0.01, bc_hvc - 0.003),
                fontsize=8.5, color="#555555", fontweight="bold")

    # 同精度ラインと BC の差を示す垂直矢印
    ax.annotate("",
                xy=(acc_match, bc_match),
                xytext=(acc_match, bc_hvc),
                arrowprops=dict(arrowstyle="<->", color="#e67e22",
                                lw=1.5, mutation_scale=10))
    ax.text(acc_match + 0.005,
            (bc_match + bc_hvc) / 2,
            f"$d$={cohens_d:.2f}\n$p$={p_match:.3f}",
            fontsize=8.5, color="#e67e22", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3e0",
                      edgecolor="#e67e22", linewidth=1.0))

    # 精度一致を示す垂直線
    ax.axvline(acc_match, color="#aaaaaa", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(acc_match, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1.032,
            f"acc ≈ {acc_match:.3f}", ha="center", va="bottom",
            fontsize=7.5, color="#aaaaaa")

    ax.set_xlabel("Identification accuracy (LOO cosine similarity)", fontsize=10)
    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_title("(B)  Same accuracy, different BC\n(V1+noise vs. HVC, matched accuracy)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "BC and identification accuracy are correlated but not equivalent",
        fontsize=11, y=1.02, color="#222222"
    )

    out_path = os.path.join(OUTPUT_DIR, "fig5_bc_vs_accuracy.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
