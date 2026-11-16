"""
exp18_prior_strength.py
------------------------
Experiment 18: BC as a function of the mixing weight between real and shuffled decoded features

デコードした特徴量に shuffled の特徴量を混ぜる割合を変えて、BC が 1 まで連続的に下がるかを見る。

手法:
  x_mixed(α) = (1 - α) * x_real + α * x_shuffled
  α = 0: Real decoded features (genuine brain control)
  α = 1: Shuffled decoded features (prior-dominated by construction)

  α を 0 → 1 で変化させると BC が単調に 1.259 → 1.001 に低下する（S 字形）。
  混ぜるのは shuffled の特徴量で、生成モデルの prior そのものではない。

使い方（リポジトリの直下で実行する）：
    python experiments/exp18_prior_strength.py               # GOD のデータから計算して図を描く
    python experiments/exp18_prior_strength.py --plot_only   # 保存した npz から図だけ描き直す
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compute_bc import (
    load_brain_data, load_image_features, compute_bc,
    DATA_DIR, FEAT_FILE
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── 設定 ──────────────────────────────────────────────────────────────────

SUBJECT    = "Subject1"
ROI        = "ROI_VC"
LAYER      = "cnn8"
ALPHA      = 100.0
N_SHUFFLE  = 1000
SEED       = 42
OUTPUT_DIR = "outputs"

ALPHAS = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, 0.10, ..., 1.00


def decode(brain_train_n, brain_test_n, feat_train_n, f_sc, alpha=100.0):
    decoder = Ridge(alpha=alpha)
    decoder.fit(brain_train_n, feat_train_n)
    pred_n = decoder.predict(brain_test_n)
    return f_sc.inverse_transform(pred_n).astype(np.float32)


def plot(bc_means, bc_sems):
    """BC と α の関係を描く（main と --plot_only の両方から使う）。"""
    # ── プロット ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    fig.subplots_adjust(wspace=0.38)

    color_main = "#2c7bb6"

    # 左パネル: BC vs α
    ax = axes[0]
    ax.plot(ALPHAS, bc_means, "o-", color=color_main,
            linewidth=2.0, markersize=5, zorder=3)
    ax.fill_between(ALPHAS,
                    bc_means - bc_sems,
                    bc_means + bc_sems,
                    color=color_main, alpha=0.15, zorder=2)

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2,
               label="BC = 1  (no brain information)", zorder=1)

    # α=0 と α=1 のアノテーション
    ax.annotate(f"$\\alpha$=0 (Real)\nBC={bc_means[0]:.3f}",
                xy=(0, bc_means[0]),
                xytext=(0.03, bc_means[-1] + 0.62 * (bc_means[0] - bc_means[-1])),
                fontsize=9, color=color_main,
                arrowprops=dict(arrowstyle="->", color=color_main, lw=0.9))
    ax.annotate(f"$\\alpha$=1 (Shuffled)\nBC={bc_means[-1]:.3f}",
                xy=(1.0, bc_means[-1]),
                xytext=(0.62, bc_means[-1] + 0.45 * (bc_means[0] - bc_means[-1])),
                fontsize=9, color="#d7442a",
                arrowprops=dict(arrowstyle="->", color="#d7442a", lw=0.9))

    ax.set_xlabel("Mixing weight $\\alpha$\n"
                  "($\\alpha$=0: real decoded features, $\\alpha$=1: shuffled)",
                  fontsize=10)
    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_title("(A)  BC decreases monotonically\nas the shuffled share increases",
                 fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # 右パネル: 正規化カーブ（0→1 スケール）
    ax = axes[1]
    bc_norm = (bc_means - bc_means[-1]) / (bc_means[0] - bc_means[-1])
    ax.plot(ALPHAS, bc_norm, "o-", color=color_main,
            linewidth=2.0, markersize=5, zorder=3,
            label="BC (normalized)")
    ax.plot(ALPHAS, 1 - ALPHAS, "--", color="#888888",
            linewidth=1.5, zorder=2, label="Linear reference")

    ax.set_xlabel("Mixing weight $\\alpha$", fontsize=10)
    ax.set_ylabel("Normalized BC\n(1 = real, 0 = shuffled)",
                  fontsize=10)
    ax.set_title("(B)  Normalized BC vs. mixing weight\n(dashed: linear reference)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_ylim(-0.05, 1.15)

    fig.suptitle(
        "Mixing real with shuffled decoded features (GOD, Subject 1)\n"
        "(feature mixing: $x_{\\mathrm{mixed}} = (1-\\alpha)\\, x_{\\mathrm{real}} + \\alpha\\, x_{\\mathrm{shuffled}}$)",
        fontsize=10.5, y=1.08, color="#222222"
    )

    out_path = os.path.join(OUTPUT_DIR, "exp18_prior_strength.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── データ読み込み ──
    print("データを読み込み中...")
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{SUBJECT}.mat", roi_key=ROI
    )
    feat, img_ids = load_image_features(FEAT_FILE, layer=LAYER)

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    cat_test    = cat_id[i_test].astype(int)
    n_test, n_voxels = brain_test.shape

    id_to_feat   = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train   = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    # ── Real と Shuffled のデコード ──
    print("Real 条件をデコード中...")
    pred_real = decode(brain_train_n, brain_test_n, feat_train_n, f_sc, ALPHA)

    print("Shuffled 条件をデコード中...")
    brain_test_shuffled_n = brain_test_n[rng.permutation(n_test)]
    pred_shuffled = decode(brain_train_n, brain_test_shuffled_n, feat_train_n, f_sc, ALPHA)

    # ── α を変えながら BC を計算 ──
    print(f"\nα = 0.00 → 1.00 で BC を計算中（N={len(ALPHAS)} ステップ）...")
    bc_means = []
    bc_sems  = []

    for a in ALPHAS:
        pred_mixed = (1 - a) * pred_real + a * pred_shuffled
        bc_m, bc_s, _, _, _ = compute_bc(pred_mixed, cat_test, N_SHUFFLE, SEED, mode="across")
        bc_means.append(bc_m)
        bc_sems.append(bc_s)
        print(f"  α={a:.2f}  BC={bc_m:.4f} ± {bc_s:.4f}")

    bc_means = np.array(bc_means)
    bc_sems  = np.array(bc_sems)

    plot(bc_means, bc_sems)

    np.savez(
        os.path.join(OUTPUT_DIR, "exp18_prior_strength.npz"),
        alphas=ALPHAS,
        bc_means=bc_means,
        bc_sems=bc_sems,
    )
    print("完了!")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:                       # 保存した npz から図だけ描き直す（データは読まない）
        d = np.load(os.path.join(OUTPUT_DIR, "exp18_prior_strength.npz"))
        plot(d["bc_means"], d["bc_sems"])
    else:
        main()
