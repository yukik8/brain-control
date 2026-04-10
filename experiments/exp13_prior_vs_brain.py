"""
exp13_prior_vs_brain.py
------------------------
Experiment 13: Prior-dominated baseline との BC 比較

「再構成画像がもっともらしく見える」≠「脳が再構成を制御している」
を定量的に示す。

3条件を比較:
  Real     : 本物の脳活動 → Ridge デコーダー → decoded features
  Shuffled : 脳活動をランダムシャッフル → 同デコーダー → decoded features
  Random   : ガウスノイズ → 同デコーダー → decoded features

期待:
  - Real: BC > 1（脳が制御）
  - Shuffled: BC ≈ 1（prior 支配）
  - Random:   BC ≈ 1（prior 支配）
  - 3条件の decoded features の統計（mean, norm）は類似
    → どれも「もっともらしい」特徴量になりうる
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compute_bc import (
    load_brain_data, load_image_features, compute_bc,
    DATA_DIR, FEAT_FILE
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── 設定 ──────────────────────────────────────────────────────────────────

SUBJECT   = "Subject1"
ROI       = "ROI_VC"
LAYER     = "cnn8"
ALPHA     = 100.0
N_SHUFFLE = 1000
SEED      = 42
OUTPUT_DIR = "outputs"


def decode(brain_train_n, brain_test_n, feat_train_n, f_sc, alpha=100.0):
    """Ridge デコーダーで brain → features を予測する。"""
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

    id_to_feat  = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train  = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    # スケーリング（条件間で共通）
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    n_test, n_voxels = brain_test.shape

    # ── 3条件でデコード ──

    # (1) Real: 本物の脳活動
    print("条件 1/3: Real（本物の脳活動）...")
    pred_real = decode(brain_train_n, brain_test_n, feat_train_n, f_sc, ALPHA)

    # (2) Shuffled: テスト脳活動をシャッフル（刺激ラベルと脳活動の対応を破壊）
    print("条件 2/3: Shuffled（脳活動をシャッフル）...")
    brain_test_shuffled_n = brain_test_n[rng.permutation(n_test)]
    pred_shuffled = decode(brain_train_n, brain_test_shuffled_n, feat_train_n, f_sc, ALPHA)

    # (3) Random: 脳活動をガウスノイズで置換
    print("条件 3/3: Random（ガウスノイズ）...")
    brain_test_random_n = rng.standard_normal((n_test, n_voxels)).astype(np.float32)
    pred_random = decode(brain_train_n, brain_test_random_n, feat_train_n, f_sc, ALPHA)

    # ── BC の計算 ──
    print(f"\nBC を計算中（N_SHUFFLE={N_SHUFFLE}）...")

    bc_real_m,     bc_real_s,     var_pres_real,     _, _ = compute_bc(pred_real,     cat_test, N_SHUFFLE, SEED, mode="across")
    bc_shuf_m,     bc_shuf_s,     var_pres_shuffled, _, _ = compute_bc(pred_shuffled, cat_test, N_SHUFFLE, SEED, mode="across")
    bc_rand_m,     bc_rand_s,     var_pres_random,   _, _ = compute_bc(pred_random,   cat_test, N_SHUFFLE, SEED, mode="across")

    # Var(broken) = Var(preserved) * BC_mean の近似値
    var_broken_real     = var_pres_real     * bc_real_m
    var_broken_shuffled = var_pres_shuffled * bc_shuf_m
    var_broken_random   = var_pres_random   * bc_rand_m

    bc_real     = {"bc_mean": bc_real_m, "bc_std": bc_real_s}
    bc_shuffled = {"bc_mean": bc_shuf_m, "bc_std": bc_shuf_s}
    bc_random   = {"bc_mean": bc_rand_m, "bc_std": bc_rand_s}

    print(f"\n=== 結果 ===")
    print(f"{'条件':<12} {'BC':>8} {'Var(pres)':>12} {'Var(broken)':>12} {'feature norm':>14}")
    for label, pred, bc, vp, vb in [
        ("Real",     pred_real,     bc_real,     var_pres_real,     var_broken_real),
        ("Shuffled", pred_shuffled, bc_shuffled, var_pres_shuffled, var_broken_shuffled),
        ("Random",   pred_random,   bc_random,   var_pres_random,   var_broken_random),
    ]:
        bc = bc  # dict
        norm = float(np.mean(np.linalg.norm(pred, axis=1)))
        print(f"{label:<12} {bc['bc_mean']:>8.4f} {vp:>12.4f} {vb:>12.4f} {norm:>14.2f}")

    # ── 特徴量の統計比較（「もっともらしさ」の代理指標）──
    print("\n=== 特徴量の統計（もっともらしさ） ===")
    for label, pred in [("Real", pred_real), ("Shuffled", pred_shuffled), ("Random", pred_random)]:
        print(f"{label:<12}  mean={pred.mean():.3f}  std={pred.std():.3f}  "
              f"min={pred.min():.2f}  max={pred.max():.2f}  "
              f"frac_positive={float((pred > 0).mean()):.3f}")

    # ── プロット ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions   = ["Real", "Shuffled", "Random"]
    bc_vals      = [bc_real["bc_mean"],     bc_shuffled["bc_mean"],     bc_random["bc_mean"]]
    vp_vals      = [var_pres_real,          var_pres_shuffled,          var_pres_random]
    vb_vals      = [var_broken_real,        var_broken_shuffled,        var_broken_random]
    norm_vals    = [float(np.mean(np.linalg.norm(p, axis=1)))
                    for p in [pred_real, pred_shuffled, pred_random]]
    colors       = ["steelblue", "coral", "gray"]

    # (1) BC 比較
    ax = axes[0]
    bars = ax.bar(conditions, bc_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="BC = 1 (prior)")
    for bar, val in zip(bars, bc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("BC (Degree of Brain Control)")
    ax.set_title("BC per condition")
    ax.set_ylim(0.9, max(bc_vals) * 1.1)
    ax.legend(fontsize=8)

    # (2) Var(preserved) vs Var(broken)
    ax = axes[1]
    x = np.arange(len(conditions))
    w = 0.35
    ax.bar(x - w/2, vp_vals, w, label="Var(preserved)", color=colors, alpha=0.7)
    ax.bar(x + w/2, vb_vals, w, label="Var(broken)",    color=colors, alpha=0.4,
           edgecolor="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Within-category variance")
    ax.set_title("Var(preserved) vs Var(broken)")
    ax.legend(fontsize=8)

    # (3) Feature norm（もっともらしさの代理）
    ax = axes[2]
    bars = ax.bar(conditions, norm_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, norm_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean L2 norm of decoded features")
    ax.set_title("Feature plausibility (norm)")

    plt.suptitle(
        "Prior-dominated baseline vs Brain-driven decoding\n"
        "Shuffled/Random ≈ prior-only: looks plausible but BC ≈ 1",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "exp13_prior_vs_brain.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nグラフ保存: {out_path}")

    # 数値保存
    np.savez(
        os.path.join(OUTPUT_DIR, "exp13_prior_vs_brain.npz"),
        conditions=np.array(conditions),
        bc_vals=np.array(bc_vals),
        vp_vals=np.array(vp_vals),
        vb_vals=np.array(vb_vals),
        norm_vals=np.array(norm_vals),
    )
    print("完了!")


if __name__ == "__main__":
    main()
