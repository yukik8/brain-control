"""
fig1_concept.py
----------------
論文 Figure 1: 概念図 + 実データによる問題提起

左パネル: パイプラインの概念図
  - Real fMRI    → Decoder → Features → Generator → Image (BC=1.26)
  - Shuffled fMRI → Decoder → Features → Generator → Image (BC=1.00)
  → 同じパイプライン、同じ見た目、でも BC だけが違う

右パネル: 実データ
  - 上段: Real 再構成画像 (3枚)
  - 下段: Shuffled 再構成画像 (3枚)
  - 右端: BC バープロット
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUTPUT_DIR = "outputs"

RECON_BASE = (
    "brain-decoding-cookbook-public/reconstruction/data/reconstruction/"
    "fg_GOD/relu7generator"
)

# 表示するカテゴリと試行（Exp14 の goldfish が最も分かりやすい）
CATEGORY = "goldfish"
REPS     = [1, 2, 3]

CONDITIONS = {
    "Real":     ("GOD_Subject1_VC",         1.259, "steelblue"),
    "Shuffled": ("GOD_Shuffled_Subject1_VC", 1.001, "coral"),
}


def load_img(path, size=(100, 100)):
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img)


def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """角丸ボックスを描画する。"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="white",
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="white", zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=12),
                zorder=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── レイアウト ──
    # 全体: 横長 (16 x 6)
    # 左半分: 概念図, 右半分: 実画像 + BC バー
    fig = plt.figure(figsize=(16, 6))

    # 左パネル（概念図）
    ax_concept = fig.add_axes([0.01, 0.05, 0.42, 0.90])
    ax_concept.set_xlim(0, 10)
    ax_concept.set_ylim(0, 10)
    ax_concept.axis("off")

    # ── 概念図を描く ──

    # タイトル
    ax_concept.text(5, 9.5, "Same pipeline, different brain input",
                    ha="center", va="center", fontsize=12, fontweight="bold")

    # --- Real 条件（上段, y=7） ---
    y_real = 7.0
    y_shuf = 3.5

    # Real ラベル
    ax_concept.text(0.3, y_real, "Real\nfMRI", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="steelblue")

    # fMRI ボックス
    draw_box(ax_concept, 1.5, y_real, 1.4, 0.8, "fMRI\nsignals", "#2c7bb6", fontsize=8)
    draw_arrow(ax_concept, 2.2, y_real, 2.9, y_real, "black")

    # Decoder ボックス（共有）
    draw_box(ax_concept, 3.5, y_real, 1.4, 0.8, "Ridge\nDecoder", "#555555", fontsize=8)
    draw_arrow(ax_concept, 4.2, y_real, 4.9, y_real, "black")

    # Features ボックス
    draw_box(ax_concept, 5.5, y_real, 1.4, 0.8, "Decoded\nFeatures", "#555555", fontsize=8)
    draw_arrow(ax_concept, 6.2, y_real, 6.9, y_real, "black")

    # Generator ボックス（共有）
    draw_box(ax_concept, 7.5, y_real, 1.4, 0.8, "Generator\n(relu7)", "#555555", fontsize=8)
    draw_arrow(ax_concept, 8.2, y_real, 8.9, y_real, "black")

    # BC ラベル
    ax_concept.text(9.5, y_real + 0.5, "BC = 1.259", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="steelblue",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#d0e8f7", edgecolor="steelblue"))
    ax_concept.text(9.5, y_real - 0.5, "Brain-\ncontrolled", ha="center", va="center",
                    fontsize=8, color="steelblue")

    # --- Shuffled 条件（下段） ---
    ax_concept.text(0.3, y_shuf, "Shuffled\nfMRI", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="coral")

    draw_box(ax_concept, 1.5, y_shuf, 1.4, 0.8, "Shuffled\nsignals", "#d7442a", fontsize=8)
    draw_arrow(ax_concept, 2.2, y_shuf, 2.9, y_shuf, "black")
    draw_box(ax_concept, 3.5, y_shuf, 1.4, 0.8, "Ridge\nDecoder", "#555555", fontsize=8)
    draw_arrow(ax_concept, 4.2, y_shuf, 4.9, y_shuf, "black")
    draw_box(ax_concept, 5.5, y_shuf, 1.4, 0.8, "Decoded\nFeatures", "#555555", fontsize=8)
    draw_arrow(ax_concept, 6.2, y_shuf, 6.9, y_shuf, "black")
    draw_box(ax_concept, 7.5, y_shuf, 1.4, 0.8, "Generator\n(relu7)", "#555555", fontsize=8)
    draw_arrow(ax_concept, 8.2, y_shuf, 8.9, y_shuf, "black")

    ax_concept.text(9.5, y_shuf + 0.5, "BC = 1.001", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="coral",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde0d0", edgecolor="coral"))
    ax_concept.text(9.5, y_shuf - 0.5, "Prior-\ndominated", ha="center", va="center",
                    fontsize=8, color="coral")

    # Decoder と Generator が共通であることを示すブレース
    ax_concept.annotate("", xy=(3.5, y_shuf + 0.55), xytext=(3.5, y_real - 0.55),
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                        lw=1.0, linestyle="dashed"))
    ax_concept.annotate("", xy=(7.5, y_shuf + 0.55), xytext=(7.5, y_real - 0.55),
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                        lw=1.0, linestyle="dashed"))
    ax_concept.text(3.5, 5.25, "same", ha="center", va="center",
                    fontsize=7, color="#888888", style="italic")
    ax_concept.text(7.5, 5.25, "same", ha="center", va="center",
                    fontsize=7, color="#888888", style="italic")

    # 中央の等号記号
    ax_concept.text(5.5, 5.25,
                    "feature norm: identical\nvisual quality: indistinguishable",
                    ha="center", va="center", fontsize=8, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#cccccc"))

    # 問いかけ
    ax_concept.text(5.0, 1.2,
                    "How do we know if the brain is actually controlling the reconstruction?",
                    ha="center", va="center", fontsize=9.5, style="italic",
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffde7",
                              edgecolor="#f9a825", linewidth=1.5))

    # ── 右パネル: 実画像 + BC バー ──
    n_reps = len(REPS)
    img_size = (90, 90)

    # 画像を配置
    for row, (cond_name, (subdir, bc_val, color)) in enumerate(CONDITIONS.items()):
        recon_dir = os.path.join(RECON_BASE, subdir, "Subject1", "VC")
        y_top = 0.88 - row * 0.44

        # 条件ラベル
        label_ax = fig.add_axes([0.44, y_top - 0.35, 0.07, 0.33])
        label_ax.set_facecolor("#f8f8f8")
        label_ax.axis("off")
        label_ax.text(0.5, 0.65, cond_name, ha="center", va="center",
                      fontsize=11, fontweight="bold", color=color,
                      transform=label_ax.transAxes)
        label_ax.text(0.5, 0.25, f"BC={bc_val:.3f}", ha="center", va="center",
                      fontsize=10, color=color, transform=label_ax.transAxes,
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor=color, linewidth=1.2))

        # 画像
        for col, rep in enumerate(REPS):
            path = os.path.join(recon_dir,
                                f"recon_image-{CATEGORY}_rep{rep:02d}.tiff")
            ax_img = fig.add_axes([0.52 + col * 0.115, y_top - 0.35, 0.10, 0.33])
            if os.path.exists(path):
                ax_img.imshow(load_img(path, size=img_size))
            ax_img.axis("off")
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)

    # カテゴリラベル
    fig.text(0.67, 0.97, f'Category: "{CATEGORY}" (3 trials)',
             ha="center", va="top", fontsize=10, fontweight="bold")
    fig.text(0.67, 0.92,
             "Same image shown 35x — each tile = one fMRI trial",
             ha="center", va="top", fontsize=8, color="#555555")

    # BC バープロット（右端）
    ax_bar = fig.add_axes([0.87, 0.12, 0.11, 0.75])
    cond_names = list(CONDITIONS.keys()) + ["Random"]
    bc_vals    = [1.259, 1.001, 1.000]
    bar_colors = ["steelblue", "coral", "gray"]
    bars = ax_bar.bar(cond_names, bc_vals, color=bar_colors,
                      alpha=0.85, edgecolor="black", linewidth=0.7)
    ax_bar.axhline(1.0, color="black", linestyle="--", linewidth=1)
    for bar, val in zip(bars, bc_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
    ax_bar.set_ylabel("BC (Degree of Brain Control)", fontsize=8)
    ax_bar.set_ylim(0.97, 1.32)
    ax_bar.set_title("BC\ncomparison", fontsize=9)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.tick_params(axis="y", labelsize=7)

    # 全体タイトル
    fig.text(0.5, 0.995,
             "Visual plausibility does not imply brain control",
             ha="center", va="top", fontsize=13, fontweight="bold")

    out_path = os.path.join(OUTPUT_DIR, "fig1_concept.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
