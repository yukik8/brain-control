"""
fig3_bc_barplot.py
--------------------
論文 Figure 3: BC バープロット（3条件）+ Feature norm（もっともらしさの確認）

2パネル構成:
  左: BC（Real=1.259, Shuffled=1.001, Random=1.000）
  右: Feature norm（Real=Shuffled=70.9, Random=118.4）

キーメッセージ:
  - 右パネル: Shuffled の feature norm は Real と同一 → 視覚品質は区別不可
  - 左パネル: BC は Real だけが > 1 → 脳の寄与を検出
  → 既存メトリクスが捉えられない先を BC は測定している
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "outputs"
NPZ_PATH   = "outputs/exp13_prior_vs_brain.npz"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── データ読み込み ──
    data = np.load(NPZ_PATH, allow_pickle=True)
    conditions = list(data["conditions"])   # ["Real", "Shuffled", "Random"]
    bc_vals    = data["bc_vals"]
    norm_vals  = data["norm_vals"]

    colors     = ["#2c7bb6", "#d7442a", "#888888"]
    bg_colors  = ["#d0e8f7", "#fde0d0", "#e8e8e8"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), facecolor="white")
    fig.subplots_adjust(wspace=0.38)

    # ── 左パネル: BC ──
    ax = axes[0]
    bars = ax.bar(conditions, bc_vals, color=colors,
                  alpha=0.88, edgecolor="white", linewidth=1.5,
                  width=0.52, zorder=3)

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2,
               label="BC = 1  (prior-only baseline)", zorder=2)

    for bar, val, c in zip(bars, bc_vals, colors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.004,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=c)

    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_title("(A)  Brain control metric", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0.96, max(bc_vals) * 1.10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # "brain-controlled" / "prior-dominated" アノテーション
    ax.annotate("brain-\ncontrolled",
                xy=(0, bc_vals[0]), xytext=(0, bc_vals[0] + 0.025),
                ha="center", fontsize=8, color=colors[0],
                arrowprops=dict(arrowstyle="-", color=colors[0], lw=0.8))

    ax.annotate("prior-\ndominated",
                xy=(1, bc_vals[1]), xytext=(1, bc_vals[1] + 0.025),
                ha="center", fontsize=8, color=colors[1],
                arrowprops=dict(arrowstyle="-", color=colors[1], lw=0.8))

    # ── 右パネル: Feature norm ──
    ax = axes[1]
    bars = ax.bar(conditions, norm_vals, color=colors,
                  alpha=0.88, edgecolor="white", linewidth=1.5,
                  width=0.52, zorder=3)

    for bar, val, c in zip(bars, norm_vals, colors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=c)

    # Real と Shuffled が同じであることを強調するブレース
    y_brace = max(norm_vals[:2]) + 6.0
    ax.annotate("",
                xy=(0, y_brace), xytext=(1, y_brace),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
    ax.text(0.5, y_brace + 1.5, "identical",
            ha="center", va="bottom", fontsize=9,
            color="#555555", style="italic")

    ax.set_ylabel("Mean L2 norm of decoded features", fontsize=10)
    ax.set_title("(B)  Feature plausibility (proxy for visual quality)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(norm_vals) * 1.22)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── 全体タイトル ──
    fig.suptitle(
        "Real fMRI produces BC > 1 while maintaining identical feature statistics to Shuffled",
        fontsize=11, y=1.02, color="#222222"
    )

    out_path = os.path.join(OUTPUT_DIR, "fig3_bc_barplot.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
