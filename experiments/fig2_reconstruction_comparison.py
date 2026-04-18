"""
fig2_reconstruction_comparison.py
-----------------------------------
論文 Figure 2: Real / Shuffled / Random の再構成画像比較（論文品質）

3条件 × 4リピート の比較図。
同じ generator・decoder で、脳入力だけを変えたときの見た目の違い（なさ）を示す。

キーメッセージ:
  - 3行の画像は視覚的にほぼ区別できない
  - しかし BC は Real=1.259, Shuffled=1.001 と鋭く異なる
  - → 視覚品質では "brain control" を検出できない
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUTPUT_DIR = "outputs"

RECON_BASE = (
    "brain-decoding-cookbook-public/reconstruction/data/reconstruction/"
    "fg_GOD/relu7generator"
)

CONDITIONS = [
    ("Real",     "GOD_Subject1_VC",         1.259, "#2c7bb6", "#d0e8f7"),
    ("Shuffled", "GOD_Shuffled_Subject1_VC", 1.001, "#d7442a", "#fde0d0"),
    ("Random",   "GOD_Random_Subject1_VC",   1.000, "#888888", "#e8e8e8"),
]

# 論文図として載せるカテゴリ（多様性を確保）
CATEGORIES  = ["goldfish", "butterfly", "bicycle"]
N_REPS      = 4
IMG_SIZE    = 128


def load_img(path, size=(IMG_SIZE, IMG_SIZE)):
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.array(img)


def make_placeholder(size=(IMG_SIZE, IMG_SIZE)):
    """画像がないときのグレープレースホルダー。"""
    arr = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
    return arr


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_conds = len(CONDITIONS)
    n_cats  = len(CATEGORIES)

    # ── レイアウト設計 ──
    # 列 = カテゴリ (N_CATS) × リピート (N_REPS) + ラベル列
    # 行 = 条件 (N_CONDS)
    # 各カテゴリブロックの左にカテゴリラベル、右端にスペーサー
    label_w  = 1.2   # 条件ラベルの列幅
    img_w    = 1.35  # 1画像の幅
    sep_w    = 0.25  # カテゴリ間スペーサー
    img_h    = 1.35  # 1画像の高さ
    header_h = 0.50  # カテゴリラベル行の高さ

    total_w = label_w + n_cats * (N_REPS * img_w + sep_w)
    total_h = header_h + n_conds * img_h + 0.6  # 下マージン

    fig = plt.figure(figsize=(total_w, total_h), facecolor="white")

    # 全体座標 → figure 座標への変換係数
    fw = total_w
    fh = total_h

    def fx(x): return x / fw
    def fy(y): return y / fh

    # ── カテゴリラベル行（上部）──
    for ci, cat in enumerate(CATEGORIES):
        x_block_start = label_w + ci * (N_REPS * img_w + sep_w)
        x_center = x_block_start + N_REPS * img_w / 2
        y_top = total_h - header_h * 0.5
        fig.text(fx(x_center), fy(y_top),
                 f'"{cat}"',
                 ha="center", va="center",
                 fontsize=11, fontweight="bold", color="#222222")

    # ── 画像グリッド ──
    for ri, (cond_name, subdir, bc_val, edge_color, bg_color) in enumerate(CONDITIONS):
        y_img_bottom = total_h - header_h - (ri + 1) * img_h + 0.05
        recon_dir = os.path.join(RECON_BASE, subdir, "Subject1", "VC")

        # 条件ラベルパネル（左端）
        ax_label = fig.add_axes([fx(0.05), fy(y_img_bottom + 0.03),
                                  fx(label_w - 0.15), fy(img_h - 0.08)])
        ax_label.set_facecolor(bg_color)
        ax_label.axis("off")
        ax_label.text(0.5, 0.65, cond_name,
                      ha="center", va="center",
                      fontsize=12, fontweight="bold", color=edge_color,
                      transform=ax_label.transAxes)
        ax_label.text(0.5, 0.28,
                      f"BC = {bc_val:.3f}",
                      ha="center", va="center",
                      fontsize=10, color=edge_color,
                      transform=ax_label.transAxes,
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                edgecolor=edge_color, linewidth=1.2))

        for ci, cat in enumerate(CATEGORIES):
            x_block_start = label_w + ci * (N_REPS * img_w + sep_w)

            for rep in range(N_REPS):
                x_img = x_block_start + rep * img_w + 0.05
                path = os.path.join(
                    recon_dir, f"recon_image-{cat}_rep{rep+1:02d}.tiff"
                )
                ax_img = fig.add_axes([fx(x_img), fy(y_img_bottom + 0.03),
                                        fx(img_w - 0.10), fy(img_h - 0.08)])
                if os.path.exists(path):
                    ax_img.imshow(load_img(path))
                else:
                    ax_img.imshow(make_placeholder())
                ax_img.axis("off")
                for spine in ax_img.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(edge_color)
                    spine.set_linewidth(2.0 if cond_name == "Real" else 1.0)

    # ── 区切り線（カテゴリ間）──
    for ci in range(1, n_cats):
        x_sep = label_w + ci * (N_REPS * img_w + sep_w) - sep_w / 2
        line = plt.Line2D([fx(x_sep), fx(x_sep)],
                          [fy(0.3), fy(total_h - 0.15)],
                          transform=fig.transFigure,
                          color="#cccccc", linewidth=0.8, linestyle="--")
        fig.add_artist(line)

    # ── 全体注釈 ──
    fig.text(0.5, fy(total_h - 0.12),
             "Same decoder and generator — only brain input differs",
             ha="center", va="top",
             fontsize=10, color="#555555", style="italic")

    fig.text(0.5, fy(0.12),
             "Real: genuine fMRI signals  ·  Shuffled: permuted fMRI (BC = 1 by construction)  ·  "
             "Random: Gaussian noise",
             ha="center", va="bottom",
             fontsize=8.5, color="#666666")

    out_path = os.path.join(OUTPUT_DIR, "fig2_reconstruction_comparison.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
