"""
exp15_roi_bc_vs_accuracy.py
----------------------------
Experiment 15: ROI別のBC（脳制御度）と識別精度の比較

「BCは精度で代替できる」という反論（r=0.85相関）への反証実験。
異なるROIでBCと識別精度を同時に計算し、両指標が解離するケースを示す。

比較するROI:
  V1, V2, V3, V4, LOC, FFA, PPA, LVC, HVC, VC（全10領域）

指標:
  - BC: Var(broken) / Var(preserved)  ← 脳が再構成をどれだけ制御しているか
  - 識別精度: デコードされた特徴量でカテゴリを何%正しく識別できるか
             （カテゴリプロトタイプへのコサイン類似度によるTop-1識別）

期待: 低次視覚野（V1）は精度が低くBCも低い、
      高次視覚野（HVC）は精度が高くBCも高い、
      かつ両者の比（BC/精度）がROIによって異なれば解離の証拠になる。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from compute_bc import load_brain_data, DATA_DIR

# AlexNet 特徴量ファイル（relu7, 4096次元）
FEAT_TRAIN_FILE = "data/god/alexnet_features_train.npz"
FEAT_TEST_FILE  = "data/god/alexnet_features_test.npz"

SUBJECT    = "Subject1"
ALPHA      = 100.0
N_SHUFFLE  = 1000
SEED       = 42
OUTPUT_DIR = "outputs"

# 比較対象のROI（小さい順から大きい順へ、機能的ヒエラルキーに沿って並べる）
ROIS = [
    "ROI_V1",
    "ROI_V2",
    "ROI_V3",
    "ROI_V4",
    "ROI_LOC",
    "ROI_FFA",
    "ROI_PPA",
    "ROI_LVC",
    "ROI_HVC",
    "ROI_VC",
]

# 表示用の短縮名
ROI_LABELS = {
    "ROI_V1":  "V1",
    "ROI_V2":  "V2",
    "ROI_V3":  "V3",
    "ROI_V4":  "V4",
    "ROI_LOC": "LOC",
    "ROI_FFA": "FFA",
    "ROI_PPA": "PPA",
    "ROI_LVC": "LVC",
    "ROI_HVC": "HVC",
    "ROI_VC":  "VC",
}


def load_alexnet_features():
    """AlexNet relu7 特徴量と対応するImageIDを読み込む。"""
    train_npz = np.load(FEAT_TRAIN_FILE)
    test_npz  = np.load(FEAT_TEST_FILE)
    return (
        train_npz["features"].astype(np.float32),
        train_npz["filenames"],
        test_npz["features"].astype(np.float32),
        test_npz["filenames"],
    )


def fname_to_imageid(fname):
    """ファイル名から数値ImageIDを抽出する（例: n01518878_10042.JPEG → 1518878.010042）。"""
    import re
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
    bc_mean : float  全カテゴリ平均BC
    bc_std  : float  BCの標準誤差
    var_pres: float  Var(preserved) の平均
    """
    rng = np.random.default_rng(seed)
    n = len(pred_features)
    cats = np.unique(cat_labels)

    # Var(preserved): 同カテゴリ内の分散
    var_pres_per_cat = np.array([
        float(np.mean(np.var(pred_features[cat_labels == c], axis=0, ddof=1)))
        for c in cats
    ])
    var_pres_mean = float(np.mean(var_pres_per_cat))

    # Var(broken): シャッフルで壊した場合の分散（N_SHUFFLE回の平均）
    var_broken_acc = np.zeros(len(cats))
    for _ in range(n_shuffle):
        shuffled = pred_features[rng.permutation(n)]
        for i, c in enumerate(cats):
            mask = cat_labels == c
            var_broken_acc[i] += float(np.mean(np.var(shuffled[mask], axis=0, ddof=1)))
    var_broken_mean_per_cat = var_broken_acc / n_shuffle

    # カテゴリ別BC → 全体平均
    bc_per_cat = var_broken_mean_per_cat / var_pres_per_cat
    bc_mean = float(np.mean(bc_per_cat))
    bc_std  = float(np.std(bc_per_cat) / np.sqrt(len(cats)))

    return bc_mean, bc_std, var_pres_mean


def compute_identification_accuracy(pred_features, cat_labels):
    """
    カテゴリプロトタイプへのコサイン類似度によるTop-1識別精度を計算する。

    各テスト試行のデコード特徴量を、各カテゴリのプロトタイプ（平均特徴量）と
    コサイン類似度で比較し、最も近いカテゴリが正解かどうかを判定する。

    Returns
    -------
    accuracy : float  Top-1識別精度（0〜1）
    """
    cats = np.unique(cat_labels)

    # カテゴリプロトタイプ（leave-one-out で過学習を避ける）
    correct = 0
    total = 0
    for i in range(len(pred_features)):
        c_true = cat_labels[i]
        feat_i = pred_features[i]

        # 各カテゴリのプロトタイプ（試行iを除く）
        protos = []
        for c in cats:
            mask = cat_labels == c
            # 同カテゴリなら試行iを除外
            if c == c_true:
                indices = np.where(mask)[0]
                indices = indices[indices != i]
                if len(indices) == 0:
                    proto = pred_features[mask].mean(axis=0)
                else:
                    proto = pred_features[indices].mean(axis=0)
            else:
                proto = pred_features[mask].mean(axis=0)
            protos.append(proto)
        protos = np.array(protos)

        # コサイン類似度
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        protos_n = protos / norms
        feat_n = feat_i / (np.linalg.norm(feat_i) + 1e-8)
        sims = protos_n @ feat_n

        pred_cat = cats[np.argmax(sims)]
        if pred_cat == c_true:
            correct += 1
        total += 1

    return correct / total


def run_roi(roi_key, feat_train, feat_test, stim_id_train, cat_test):
    """
    1つのROIについて脳活動を読み込み、Ridge デコーダーを訓練し、
    BC と識別精度を計算する。
    """
    # 脳活動の読み込み
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{SUBJECT}.mat", roi_key=roi_key
    )

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    # テストのカテゴリは ROI 非依存なので引数から受け取る

    # ImageID → 特徴量 のマッピング
    import re
    id_to_feat = {}
    for j, fname in enumerate(stim_id_train):
        # stim_id_train は数値ID（float）
        id_to_feat[float(fname)] = feat_train[j]

    # 訓練試行と特徴量の対応付け（stimulus_id で突合）
    matched_brain = []
    matched_feat  = []
    for j, sid in enumerate(stim_train):
        best = min(id_to_feat.keys(), key=lambda x: abs(x - sid), default=None)
        if best is not None and abs(best - sid) < 0.001:
            matched_brain.append(brain_train[j])
            matched_feat.append(id_to_feat[best])

    if len(matched_brain) < 100:
        print(f"  警告: {roi_key} マッチ数が少ない ({len(matched_brain)})")
        return None

    brain_train_m = np.array(matched_brain, dtype=np.float32)
    feat_train_m  = np.array(matched_feat,  dtype=np.float32)

    # 標準化
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train_m)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train_m)

    # Ridge デコーダーの訓練と予測
    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)
    pred_test_n = decoder.predict(brain_test_n)
    pred_test   = f_sc.inverse_transform(pred_test_n).astype(np.float32)

    return pred_test


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── AlexNet 特徴量の読み込み ──
    print("AlexNet relu7 特徴量を読み込み中...")
    feat_train_all, fnames_train, feat_test_all, fnames_test = load_alexnet_features()

    # ファイル名 → 数値ImageID のマッピング（訓練用）
    train_ids = np.array([
        fname_to_imageid(fn) if fname_to_imageid(fn) is not None else -1.0
        for fn in fnames_train
    ])
    valid_mask = train_ids >= 0
    feat_train_valid = feat_train_all[valid_mask]
    train_ids_valid  = train_ids[valid_mask]

    # テストカテゴリラベルの読み込み（ROI非依存なのでROI_VCで代表取得）
    brain_ref, datatype_ref, stim_id_ref, cat_id_ref = load_brain_data(
        f"{DATA_DIR}/{SUBJECT}.mat", roi_key="ROI_VC"
    )
    cat_test = cat_id_ref[datatype_ref == 2].astype(int)

    print(f"テスト試行数: {len(cat_test)}, カテゴリ数: {len(np.unique(cat_test))}")
    print()

    # ── 各ROIで実験 ──
    results = {}
    for roi in ROIS:
        label = ROI_LABELS[roi]
        print(f"=== {label} ({roi}) ===")

        pred_test = run_roi(
            roi_key=roi,
            feat_train=feat_train_valid,
            feat_test=feat_test_all,
            stim_id_train=train_ids_valid,
            cat_test=cat_test,
        )
        if pred_test is None:
            continue

        # BC の計算
        print(f"  BC を計算中（N_SHUFFLE={N_SHUFFLE}）...")
        bc_mean, bc_std, var_pres = compute_bc(pred_test, cat_test, N_SHUFFLE, SEED)

        # 識別精度の計算
        print(f"  識別精度を計算中...")
        acc = compute_identification_accuracy(pred_test, cat_test)

        results[label] = {
            "bc_mean":  bc_mean,
            "bc_std":   bc_std,
            "var_pres": var_pres,
            "accuracy": acc,
        }
        print(f"  BC={bc_mean:.4f} ± {bc_std:.4f},  Acc={acc:.3f}")
        print()

    # ── 結果表 ──
    print("\n=== 結果サマリー ===")
    print(f"{'ROI':<6} {'BC':>8} {'± std':>7} {'Acc':>7}")
    for label, r in results.items():
        print(f"{label:<6} {r['bc_mean']:>8.4f} {r['bc_std']:>7.4f} {r['accuracy']:>7.3f}")

    labels     = list(results.keys())
    bc_vals    = np.array([results[l]["bc_mean"]  for l in labels])
    bc_stds    = np.array([results[l]["bc_std"]   for l in labels])
    acc_vals   = np.array([results[l]["accuracy"] for l in labels])

    # ── 相関分析 ──
    r_val, p_val = stats.pearsonr(bc_vals, acc_vals)
    print(f"\nBC vs 識別精度 相関: r = {r_val:.3f}, p = {p_val:.3e}")

    # ── プロット ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # (1) ROI別 BC バープロット
    ax = axes[0]
    bars = ax.bar(labels, bc_vals, yerr=bc_stds, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7, capsize=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="BC=1 (prior)")
    for bar, val in zip(bars, bc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + bc_stds[list(bc_vals).index(val)] + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("BC (Degree of Brain Control)")
    ax.set_title("BC per ROI")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0.9, max(bc_vals + bc_stds) * 1.15)

    # (2) ROI別 識別精度 バープロット
    ax = axes[1]
    bars = ax.bar(labels, acc_vals, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.7)
    chance = 1.0 / len(np.unique(cat_test))
    ax.axhline(chance, color="black", linestyle="--", linewidth=1,
               label=f"Chance = {chance:.3f}")
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Top-1 Category Identification Accuracy")
    ax.set_title("Accuracy per ROI")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    # (3) BC vs 識別精度 散布図（解離の検出）
    ax = axes[2]
    scatter = ax.scatter(acc_vals, bc_vals, c=range(len(labels)),
                         cmap="tab10", s=100, alpha=0.85, edgecolors="black", linewidth=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label, (acc_vals[i], bc_vals[i]),
                    xytext=(5, 3), textcoords="offset points", fontsize=9)
    # 回帰直線
    m, b_coef = np.polyfit(acc_vals, bc_vals, 1)
    x_line = np.linspace(acc_vals.min() - 0.01, acc_vals.max() + 0.01, 100)
    ax.plot(x_line, m * x_line + b_coef, color="red", linewidth=1.5,
            linestyle="--", label=f"r={r_val:.3f}, p={p_val:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Top-1 Identification Accuracy")
    ax.set_ylabel("BC (Degree of Brain Control)")
    ax.set_title(f"BC vs Accuracy across ROIs\nr={r_val:.3f}, p={p_val:.3f}")
    ax.legend(fontsize=8)

    plt.suptitle(
        f"ROI comparison: BC vs Identification Accuracy — GOD Subject1, relu7 features\n"
        f"BC and Accuracy across 10 ROIs: r={r_val:.3f}",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "exp15_roi_bc_vs_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nグラフ保存: {out_path}")

    # 数値保存
    np.savez(
        os.path.join(OUTPUT_DIR, "exp15_roi_bc_vs_accuracy.npz"),
        rois=np.array(labels),
        bc_vals=bc_vals,
        bc_stds=bc_stds,
        acc_vals=acc_vals,
        r_bc_acc=np.array([r_val]),
        p_bc_acc=np.array([p_val]),
    )
    print("完了!")


if __name__ == "__main__":
    main()
