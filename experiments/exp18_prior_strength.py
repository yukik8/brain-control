"""
exp18_prior_strength.py
------------------------
Experiment 18: Prior strength simulation — BC as a function of prior dominance

「prior が強くなるほど BC → 1」を feature mixing で直接実証する。

手法:
  x_mixed(α) = (1 - α) * x_real + α * x_shuffled
  α = 0: Real decoded features (genuine brain control)
  α = 1: Shuffled decoded features (prior-dominated by construction)

  α を 0 → 1 で変化させると BC が連続的に 1.259 → 1.001 に低下することを示す。

論文への寄与:
  Discussion 5.2「Diffusion の prior が強いほど BC → 1」という仮説を
  feature-mixing という制御実験で直接実証する。
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
               label="BC = 1  (prior-only baseline)", zorder=1)

    # α=0 と α=1 のアノテーション
    ax.annotate(f"$\\alpha$=0 (Real)\nBC={bc_means[0]:.3f}",
                xy=(0, bc_means[0]),
                xytext=(0.08, bc_means[0] + 0.012),
                fontsize=9, color=color_main,
                arrowprops=dict(arrowstyle="->", color=color_main, lw=0.9))
    ax.annotate(f"$\\alpha$=1 (Shuffled)\nBC={bc_means[-1]:.3f}",
                xy=(1.0, bc_means[-1]),
                xytext=(0.72, bc_means[-1] + 0.012),
                fontsize=9, color="#d7442a",
                arrowprops=dict(arrowstyle="->", color="#d7442a", lw=0.9))

    ax.set_xlabel("Prior dominance $\\alpha$\n"
                  "($\\alpha$=0: genuine brain signals, $\\alpha$=1: shuffled signals)",
                  fontsize=10)
    ax.set_ylabel("BC (Degree of Brain Control)", fontsize=10)
    ax.set_title("(A)  BC decreases monotonically\nas prior dominance increases",
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

    ax.set_xlabel("Prior dominance $\\alpha$", fontsize=10)
    ax.set_ylabel("Normalized BC\n(1 = full brain control, 0 = prior-dominated)",
                  fontsize=10)
    ax.set_title("(B)  Normalized BC vs. prior dominance\n(near-linear relationship)",
                 fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_ylim(-0.05, 1.15)

    fig.suptitle(
        "Stronger generative prior continuously suppresses BC toward 1\n"
        "(feature mixing: $x_{\\mathrm{mixed}} = (1-\\alpha)\\, x_{\\mathrm{real}} + \\alpha\\, x_{\\mathrm{shuffled}}$)",
        fontsize=10.5, y=1.02, color="#222222"
    )

    out_path = os.path.join(OUTPUT_DIR, "exp18_prior_strength.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    np.savez(
        os.path.join(OUTPUT_DIR, "exp18_prior_strength.npz"),
        alphas=ALPHAS,
        bc_means=bc_means,
        bc_sems=bc_sems,
    )
    print("完了!")


if __name__ == "__main__":
    main()
