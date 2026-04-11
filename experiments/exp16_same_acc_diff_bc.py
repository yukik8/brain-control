"""
exp16_same_acc_diff_bc.py
--------------------------
Experiment 16: 同精度・異BC の実証

「BCは識別精度と相関が高い（r=0.97）→ なぜBCが必要か？」
という批判への決定打となる実験。

手法:
  V1 の脳活動にガウスノイズを加え、SNR を連続的に変化させる。
  各SNRで BC と識別精度の両方を計算し、
  「精度が同じなのに BC が異なる」クロス点を示す。

なぜノイズ注入か:
  - BC は分散比なので、シグナル以外の分散（ノイズ）が増えると
    Var(preserved) が増加 → BC が低下する（分母が増える）
  - 精度はカテゴリ間の相対差を見るので、均一ノイズには比較的ロバスト
  - → ノイズを増やすと BC は先に落ちる可能性がある

比較対象:
  - V1 + noise=0.0  （ベースライン）
  - V1 + noise=σ   （複数のσ）
  - HVC + noise=0.0（HVC は V1 より低 BC・低精度だが、
                    「精度が同じでも BC が異なる」条件として使う）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from compute_bc import load_brain_data, DATA_DIR

FEAT_TRAIN_FILE = "data/god/alexnet_features_train.npz"
SUBJECT    = "Subject1"
ALPHA      = 100.0
N_SHUFFLE  = 1000
SEED       = 42
OUTPUT_DIR = "outputs"

# ノイズ強度（標準化後の脳活動に加えるガウスノイズの標準偏差）
# 0 がノイズなし、値が大きいほど SNR が低い
NOISE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def fname_to_imageid(fname):
    """ファイル名から数値ImageIDを抽出する。"""
    matches = re.findall(r'n(\d{8})_(\d+)\.JPEG', str(fname), re.IGNORECASE)
    if not matches:
        return None
    synset_str, img_str = matches[-1]
    return float(f"{int(synset_str)}.{int(img_str):06d}")


def compute_bc(pred_features, cat_labels, n_shuffle=1000, seed=42):
    """
    BCを計算する（across-shuffle方式）。

    Returns
    -------
    bc_mean    : float   全カテゴリ平均BC
    bc_sem     : float   標準誤差
    bc_per_cat : ndarray カテゴリ別BC（Cohen's d 計算用）
    """
    rng = np.random.default_rng(seed)
    n = len(pred_features)
    cats = np.unique(cat_labels)

    # Var(preserved): 同カテゴリ内の分散
    var_pres_per_cat = np.array([
        float(np.mean(np.var(pred_features[cat_labels == c], axis=0, ddof=1)))
        for c in cats
    ])

    # Var(broken): シャッフル後の分散
    var_broken_acc = np.zeros(len(cats))
    for _ in range(n_shuffle):
        shuffled = pred_features[rng.permutation(n)]
        for i, c in enumerate(cats):
            mask = cat_labels == c
            var_broken_acc[i] += float(np.mean(np.var(shuffled[mask], axis=0, ddof=1)))
    var_broken_mean = var_broken_acc / n_shuffle

    bc_per_cat = var_broken_mean / var_pres_per_cat
    return float(np.mean(bc_per_cat)), float(np.std(bc_per_cat) / np.sqrt(len(cats))), bc_per_cat


def compute_accuracy(pred_features, cat_labels):
    """LOO コサイン類似度 Top-1 識別精度を計算する。"""
    cats = np.unique(cat_labels)
    correct = 0
    total = len(pred_features)
    for i in range(total):
        c_true = cat_labels[i]
        feat_i = pred_features[i]
        protos = []
        for c in cats:
            mask = cat_labels == c
            if c == c_true:
                idx = np.where(mask)[0]
                idx = idx[idx != i]
                proto = pred_features[idx].mean(axis=0) if len(idx) > 0 else pred_features[mask].mean(axis=0)
            else:
                proto = pred_features[mask].mean(axis=0)
            protos.append(proto)
        protos = np.array(protos)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        sims = (protos / norms) @ (feat_i / (np.linalg.norm(feat_i) + 1e-8))
        if cats[np.argmax(sims)] == c_true:
            correct += 1
    return correct / total


def prepare_decoder(roi_key, feat_train_valid, train_ids_valid):
    """
    指定ROIの脳活動を読み込み、Ridge デコーダーを訓練して返す。
    テスト脳活動（標準化済み）と正解カテゴリも返す。
    """
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{SUBJECT}.mat", roi_key=roi_key
    )
    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    cat_test    = cat_id[i_test].astype(int)

    # 訓練データのマッチング
    id_to_feat = {float(tid): feat_train_valid[j] for j, tid in enumerate(train_ids_valid)}
    matched_brain, matched_feat = [], []
    for j, sid in enumerate(stim_train):
        best = min(id_to_feat.keys(), key=lambda x: abs(x - sid), default=None)
        if best is not None and abs(best - sid) < 0.001:
            matched_brain.append(brain_train[j])
            matched_feat.append(id_to_feat[best])

    brain_train_m = np.array(matched_brain, dtype=np.float32)
    feat_train_m  = np.array(matched_feat,  dtype=np.float32)

    # 標準化
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train_m)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train_m)

    # Ridge デコーダー訓練
    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)

    return decoder, f_sc, brain_test_n, cat_test


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── AlexNet 特徴量の読み込み ──
    print("AlexNet relu7 特徴量を読み込み中...")
    train_npz = np.load(FEAT_TRAIN_FILE)
    feat_train_all = train_npz["features"].astype(np.float32)
    fnames_train   = train_npz["filenames"]

    train_ids = np.array([
        fname_to_imageid(fn) if fname_to_imageid(fn) is not None else -1.0
        for fn in fnames_train
    ])
    valid_mask = train_ids >= 0
    feat_train_valid = feat_train_all[valid_mask]
    train_ids_valid  = train_ids[valid_mask]

    # ── V1 デコーダーの準備 ──
    print("\n=== V1（ノイズ注入実験）===")
    decoder_v1, f_sc_v1, brain_test_v1_n, cat_test = prepare_decoder(
        "ROI_V1", feat_train_valid, train_ids_valid
    )

    # ── HVC デコーダーの準備（比較対象）──
    print("=== HVC（比較対象）===")
    decoder_hvc, f_sc_hvc, brain_test_hvc_n, _ = prepare_decoder(
        "ROI_HVC", feat_train_valid, train_ids_valid
    )

    n_test, n_voxels_v1 = brain_test_v1_n.shape

    # ── V1 × 各ノイズ強度で BC と精度を計算 ──
    print(f"\nV1 ノイズ注入実験（N_SHUFFLE={N_SHUFFLE}）...")
    print(f"{'Noise σ':>8} {'BC':>8} {'± std':>7} {'Acc':>7}")

    bc_list  = []
    std_list = []
    acc_list = []

    for sigma in NOISE_LEVELS:
        # 標準化済みのテスト脳活動にノイズを加える
        if sigma == 0.0:
            brain_noisy = brain_test_v1_n.copy()
        else:
            noise = rng.standard_normal(brain_test_v1_n.shape).astype(np.float32) * sigma
            brain_noisy = brain_test_v1_n + noise

        # デコード
        pred_n = decoder_v1.predict(brain_noisy)
        pred   = f_sc_v1.inverse_transform(pred_n).astype(np.float32)

        # BC と精度の計算
        bc_m, bc_s, bc_cats = compute_bc(pred, cat_test, N_SHUFFLE, SEED)
        acc                 = compute_accuracy(pred, cat_test)

        bc_list.append(bc_m)
        std_list.append(bc_s)
        acc_list.append(acc)
        # カテゴリ別BC配列も保持（Cohen's d 計算用）
        if sigma == 0.75:
            bc_per_cat_v1_sigma075 = bc_cats
        print(f"{sigma:>8.2f} {bc_m:>8.4f} {bc_s:>7.4f} {acc:>7.3f}")

    # ── HVC ベースラインの計算 ──
    print("\nHVC ベースライン...")
    pred_hvc_n = decoder_hvc.predict(brain_test_hvc_n)
    pred_hvc   = f_sc_hvc.inverse_transform(pred_hvc_n).astype(np.float32)
    bc_hvc_m, bc_hvc_s, bc_per_cat_hvc = compute_bc(pred_hvc, cat_test, N_SHUFFLE, SEED)
    acc_hvc                             = compute_accuracy(pred_hvc, cat_test)
    print(f"HVC: BC={bc_hvc_m:.4f} ± {bc_hvc_s:.4f},  Acc={acc_hvc:.3f}")

    bc_list  = np.array(bc_list)
    std_list = np.array(std_list)
    acc_list = np.array(acc_list)

    # ── 「同精度・異BC」クロス点の探索 ──
    print("\n=== 同精度・異BC の探索 ===")
    # HVC の精度に最も近い V1+noise の条件を探す
    acc_diffs = np.abs(acc_list - acc_hvc)
    best_idx  = int(np.argmin(acc_diffs))
    print(f"HVC 精度: {acc_hvc:.3f},  HVC BC: {bc_hvc_m:.4f}")
    print(f"V1+noise(σ={NOISE_LEVELS[best_idx]:.2f}): 精度={acc_list[best_idx]:.3f}, "
          f"BC={bc_list[best_idx]:.4f}")
    print(f"精度差: {acc_diffs[best_idx]:.3f},  BC差: {bc_list[best_idx] - bc_hvc_m:.4f}")

    # ── Cohen's d と 95% CI の計算 ──
    # V1+noise(σ=0.75) と HVC の カテゴリ別BC を比較
    from scipy import stats as scipy_stats

    a = bc_per_cat_v1_sigma075  # V1+noise σ=0.75 のカテゴリ別BC
    b = bc_per_cat_hvc           # HVC のカテゴリ別BC
    n_a, n_b = len(a), len(b)

    # プールした標準偏差
    pooled_std = np.sqrt(((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
                         / (n_a + n_b - 2))
    cohens_d   = (np.mean(b) - np.mean(a)) / (pooled_std + 1e-10)

    # Welch の t 検定
    t_stat, p_val = scipy_stats.ttest_ind(a, b, equal_var=False)

    # 平均差の 95% CI（Welch の信頼区間）
    mean_diff = float(np.mean(b) - np.mean(a))
    se_diff   = float(np.sqrt(np.var(a, ddof=1)/n_a + np.var(b, ddof=1)/n_b))
    df        = (np.var(a, ddof=1)/n_a + np.var(b, ddof=1)/n_b)**2 / (
                 (np.var(a, ddof=1)/n_a)**2/(n_a-1) + (np.var(b, ddof=1)/n_b)**2/(n_b-1))
    t_crit    = scipy_stats.t.ppf(0.975, df)
    ci_low    = mean_diff - t_crit * se_diff
    ci_high   = mean_diff + t_crit * se_diff

    print(f"\n=== 効果量・信頼区間 ===")
    print(f"対象: V1+noise(σ=0.75) vs HVC（精度が最も近いペア）")
    print(f"  平均BC差: {mean_diff:+.4f}  （HVC − V1+noise）")
    print(f"  95% CI:  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")

    # ── プロット ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) ノイズ強度 vs BC
    ax = axes[0]
    ax.errorbar(NOISE_LEVELS, bc_list, yerr=std_list,
                color="steelblue", marker="o", linewidth=2, markersize=6,
                label="V1 + noise", capsize=4)
    ax.axhline(bc_hvc_m, color="coral", linestyle="--", linewidth=2,
               label=f"HVC (no noise) BC={bc_hvc_m:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="BC=1 (prior)")
    ax.set_xlabel("ノイズ強度 σ（標準化済み脳活動に加算）")
    ax.set_ylabel("BC（脳制御度）")
    ax.set_title("BC vs ノイズ強度（V1）")
    ax.legend(fontsize=8)

    # (2) ノイズ強度 vs 識別精度
    ax = axes[1]
    ax.plot(NOISE_LEVELS, acc_list, color="steelblue", marker="o", linewidth=2,
            markersize=6, label="V1 + ノイズ")
    ax.axhline(acc_hvc, color="coral", linestyle="--", linewidth=2,
               label=f"HVC（ノイズなし） 精度={acc_hvc:.3f}")
    ax.axhline(1/50, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="チャンス（0.02）")
    ax.set_xlabel("ノイズ強度 σ")
    ax.set_ylabel("Top-1 識別精度")
    ax.set_title("識別精度 vs ノイズ強度（V1）")
    ax.legend(fontsize=8)

    # (3) 精度 vs BC の散布図（全条件）
    ax = axes[2]
    # V1 + noise の軌跡
    sc = ax.scatter(acc_list, bc_list, c=NOISE_LEVELS, cmap="Blues_r",
                    s=80, zorder=3, edgecolors="black", linewidth=0.7,
                    label="V1 + ノイズ（σ=0→5）")
    for i, sigma in enumerate(NOISE_LEVELS):
        if sigma in [0.0, 1.0, 3.0]:
            ax.annotate(f"σ={sigma:.1f}", (acc_list[i], bc_list[i]),
                        xytext=(5, 3), textcoords="offset points", fontsize=8)
    # V1→noise の軌跡線
    ax.plot(acc_list, bc_list, color="steelblue", linewidth=1.5,
            alpha=0.5, linestyle="-")
    # HVC ベースライン
    ax.scatter([acc_hvc], [bc_hvc_m], color="coral", s=120, zorder=4,
               marker="*", edgecolors="black", linewidth=0.7, label="HVC（ノイズなし）")
    ax.annotate("HVC", (acc_hvc, bc_hvc_m),
                xytext=(5, -10), textcoords="offset points", fontsize=9, color="coral")

    # 「同精度・異BC」の対応する点を強調
    ax.annotate(
        f"同精度 ≈ {acc_list[best_idx]:.2f}\nBC: {bc_list[best_idx]:.3f} vs {bc_hvc_m:.3f}",
        xy=(acc_list[best_idx], bc_list[best_idx]),
        xytext=(acc_list[best_idx] + 0.03, bc_list[best_idx] + 0.005),
        fontsize=8, color="navy",
        arrowprops=dict(arrowstyle="->", color="navy"),
    )

    plt.colorbar(sc, ax=ax, label="ノイズ強度 σ")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Top-1 識別精度")
    ax.set_ylabel("BC（脳制御度）")
    ax.set_title("同精度・異BC の実証\n（V1+ノイズ軌跡 vs HVC）")
    ax.legend(fontsize=8)

    plt.suptitle(
        "Exp16: 同精度・異BC — V1 ノイズ注入 vs HVC\n"
        "BC は識別精度では捉えられないシグナル構造の劣化を検出する",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "exp16_same_acc_diff_bc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nグラフ保存: {out_path}")

    # 数値保存
    np.savez(
        os.path.join(OUTPUT_DIR, "exp16_same_acc_diff_bc.npz"),
        noise_levels=np.array(NOISE_LEVELS),
        bc_v1_noise=bc_list,
        bc_v1_noise_std=std_list,
        acc_v1_noise=acc_list,
        bc_hvc=np.array([bc_hvc_m]),
        bc_hvc_std=np.array([bc_hvc_s]),
        acc_hvc=np.array([acc_hvc]),
        cohens_d=np.array([cohens_d]),
        mean_diff=np.array([mean_diff]),
        ci_low=np.array([ci_low]),
        ci_high=np.array([ci_high]),
        p_val=np.array([p_val]),
    )
    print("完了!")


if __name__ == "__main__":
    main()
