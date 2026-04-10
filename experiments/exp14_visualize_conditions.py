"""
exp14_visualize_conditions.py
------------------------------
Real / Shuffled / Random 3条件の再構成画像を並べて比較する。

「もっともらしく見える」≠「脳が制御している」を視覚的に示す論文図。

出力:
  outputs/exp14_comparison_{category}.png  - カテゴリ別の比較図
  outputs/exp14_comparison_summary.png     - 複数カテゴリのサマリー図
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ── 設定 ──────────────────────────────────────────────────────────────────

RECON_BASE = (
    "brain-decoding-cookbook-public/reconstruction/data/reconstruction/"
    "fg_GOD/relu7generator"
)

CONDITIONS = {
    "Real":     ("GOD_Subject1_VC",         1.259, "steelblue"),
    "Shuffled": ("GOD_Shuffled_Subject1_VC", 1.001, "coral"),
    "Random":   ("GOD_Random_Subject1_VC",   1.000, "gray"),
}

OUTPUT_DIR = "outputs"
N_SHOW_REPS = 5  # 各カテゴリで表示するリピート数

# サマリー図で使うカテゴリ（多様なものを選択）
SUMMARY_CATEGORIES = [
    "goldfish", "butterfly", "bicycle", "church", "washer"
]


def load_img(path, size=(112, 112)):
    """画像を読み込んで numpy array に変換する。"""
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img)


def make_category_figure(category, n_reps=5):
    """1カテゴリの3条件比較図を作成する（論文品質）。"""
    IMG_SIZE = 150
    PAD = 6   # 行間の色帯の太さ
    cond_items = list(CONDITIONS.items())
    n_conds = len(cond_items)

    fig = plt.figure(figsize=(n_reps * 1.8 + 2.4, n_conds * 2.0 + 1.4))

    row_colors = {"Real": "#d0e8f7", "Shuffled": "#fde0d0", "Random": "#e8e8e8"}

    # ヘッダー高さ: 15%, 各行: (85% / n_conds)
    header_h = 0.15
    row_h = (1.0 - header_h) / n_conds

    for row, (cond_name, (subdir, bc_val, color)) in enumerate(cond_items):
        recon_dir = os.path.join(RECON_BASE, subdir, "Subject1", "VC")
        y_bottom = 1.0 - header_h - (row + 1) * row_h

        for col in range(n_reps):
            ax = fig.add_axes([
                0.26 + col * (0.72 / n_reps),
                y_bottom + 0.01,
                0.72 / n_reps - 0.012,
                row_h - 0.02
            ])
            path = os.path.join(recon_dir, f"recon_image-{category}_rep{col+1:02d}.tiff")
            if os.path.exists(path):
                ax.imshow(load_img(path, size=(IMG_SIZE, IMG_SIZE)))
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5 if cond_name == "Real" else 1.0)
                spine.set_visible(True)

        # 行ラベル（左側）
        label_ax = fig.add_axes([0.01, y_bottom + 0.01, 0.23, row_h - 0.02])
        label_ax.set_facecolor(row_colors.get(cond_name, "white"))
        label_ax.axis("off")
        label_ax.text(0.5, 0.62, cond_name, ha="center", va="center",
                      fontsize=13, fontweight="bold", color=color,
                      transform=label_ax.transAxes)
        label_ax.text(0.5, 0.25, f"BC = {bc_val:.3f}",
                      ha="center", va="center", fontsize=11, color=color,
                      transform=label_ax.transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor=color, linewidth=1.5))

    fig.text(0.5, 0.98, f'Category: "{category}"',
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
             "Same generator & decoder — only brain input differs  |  "
             "Real: actual fMRI  ·  Shuffled: permuted fMRI  ·  Random: Gaussian noise",
             ha="center", va="top", fontsize=8.5, color="#444444")
    return fig


def make_summary_figure(categories, n_reps=5):
    """複数カテゴリの比較サマリー図を作成する。"""
    n_cats = len(categories)
    n_conds = len(CONDITIONS)

    # レイアウト: 行 = カテゴリ × 条件、列 = リピート
    fig, axes = plt.subplots(
        n_cats * n_conds, n_reps,
        figsize=(n_reps * 1.8, n_cats * n_conds * 1.8 + 1)
    )

    row = 0
    for cat_idx, category in enumerate(categories):
        for cond_name, (subdir, bc_val, color) in CONDITIONS.items():
            recon_dir = os.path.join(RECON_BASE, subdir, "Subject1", "VC")
            for col in range(n_reps):
                ax = axes[row, col]
                path = os.path.join(recon_dir,
                                    f"recon_image-{category}_rep{col+1:02d}.tiff")
                if os.path.exists(path):
                    ax.imshow(load_img(path, size=(84, 84)))
                ax.axis("off")

                # 左端ラベル
                if col == 0:
                    label = f"{category}\n{cond_name} BC={bc_val:.3f}"
                    ax.text(-0.05, 0.5, label,
                            transform=ax.transAxes,
                            fontsize=7, color=color, fontweight="bold",
                            ha="right", va="center")
            row += 1

        # カテゴリ間の区切り線
        if cat_idx < n_cats - 1:
            axes[row - 1, 0].axhline(y=-0.1, color="black", linewidth=0.5)

    # 凡例
    patches = [
        mpatches.Patch(color=color, label=f"{cond} (BC={bc:.3f})")
        for cond, (_, bc, color) in CONDITIONS.items()
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=9,
               title="Condition", title_fontsize=9)

    fig.suptitle(
        "Reconstruction comparison: Real vs Shuffled vs Random brain input\n"
        "Same generator (relu7generator) — visual quality ≠ brain control",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── サマリー図（論文メイン図）──
    print("サマリー図を作成中...")
    fig = make_summary_figure(SUMMARY_CATEGORIES, n_reps=N_SHOW_REPS)
    out_path = os.path.join(OUTPUT_DIR, "exp14_comparison_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}")

    # ── カテゴリ別図（補足用）──
    for category in SUMMARY_CATEGORIES:
        print(f"カテゴリ図: {category}")
        fig = make_category_figure(category, n_reps=N_SHOW_REPS)
        out_path = os.path.join(OUTPUT_DIR, f"exp14_comparison_{category}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n完了!")


if __name__ == "__main__":
    main()
