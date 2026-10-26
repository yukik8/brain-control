"""
exp19_existing_metrics.py
--------------------------
Experiment 19: 既存の評価指標と BC の比較

Exp17 と同じ Real / Shuffled / Random 条件（5被験者, VC, cnn8）で、
次の指標を並べて比較する。

  - Feature norm              : 参照不要（正解刺激を使わない）
  - Pairwise identification   : 正解刺激の特徴量が必要（チャンス = 0.5）
  - Calinski–Harabasz (CH)    : 試行のカテゴリラベルのみ必要（= トレースの一元配置 F 値）
  - BC                        : 試行のカテゴリラベルのみ必要

あわせて、1カテゴリあたりの試行数 n を変えたときの BC と CH の変化を調べる。
各カテゴリの試行数がそろっていれば BC = 1 + (C-1)(CH-1)/(N-1) が成り立つ。
CH は n にほぼ比例して増える（検定統計量）が、BC - 1 は n によらない（効果量）はず。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from compute_bc import (
    load_brain_data, load_image_features, compute_bc,
    DATA_DIR, FEAT_FILE
)

# ── 設定 ──────────────────────────────────────────────────────────────────

ROI        = "ROI_VC"
LAYER      = "cnn8"
ALPHA      = 100.0
N_SHUFFLE  = 1000
SEED       = 42
OUTPUT_DIR = "outputs"

SUBJECTS   = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]
CONDITIONS = ["Real", "Shuffled", "Random"]
COLORS     = {"Real": "#2c7bb6", "Shuffled": "#d7442a", "Random": "#888888"}

# 試行数の影響（Subject1, Real）
SUBSAMPLE_N       = [5, 10, 15, 20, 25, 30, 35]
SUBSAMPLE_DRAWS   = 20
SUBSAMPLE_SHUFFLE = 200


def decode(brain_train_n, brain_test_n, feat_train_n, f_sc, alpha=100.0):
    """Ridge デコーダーで brain → features を予測する。"""
    decoder = Ridge(alpha=alpha)
    decoder.fit(brain_train_n, feat_train_n)
    pred_n = decoder.predict(brain_test_n)
    return f_sc.inverse_transform(pred_n).astype(np.float32)


def pairwise_identification(pred, true_feat, img_index):
    """
    ペア識別精度: 各試行のデコード特徴量と50枚のテスト画像の正解特徴量との相関を求め、
    正解画像の相関が上回った割合（49 枚中）を平均する。チャンス = 0.5。
    """
    def zrows(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    r = zrows(pred.astype(np.float64)) @ zrows(true_feat.astype(np.float64)).T
    r_true = r[np.arange(len(pred)), img_index]
    return float(np.mean((r < r_true[:, None]).sum(axis=1) / (r.shape[1] - 1)))


def bc_closed_form(ch, n_trials, n_cats):
    """各カテゴリの試行数がそろっているときの BC の閉じた式。"""
    return 1 + (n_cats - 1) * (ch - 1) / (n_trials - 1)


def run_subject(subject, feat, img_ids, rng):
    """1被験者分の Real / Shuffled / Random をデコードし、各指標を計算して返す。"""
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{subject}.mat", roi_key=ROI
    )

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    stim_test   = stim_id[i_test]
    cat_test    = cat_id[i_test].astype(int)

    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    # テスト画像（50枚）の正解特徴量
    test_imgs = np.unique(stim_test)
    true_feat = np.array([id_to_feat[s] for s in test_imgs], dtype=np.float32)
    img_index = np.searchsorted(test_imgs, stim_test)

    # 標準化
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    n_test, n_voxels = brain_test.shape

    # Exp17 と同じ順序で乱数を使う（同じ Shuffled / Random 条件になる）
    preds = {"Real": decode(brain_train_n, brain_test_n, feat_train_n, f_sc, ALPHA)}
    brain_test_shuffled_n = brain_test_n[rng.permutation(n_test)]
    preds["Shuffled"] = decode(brain_train_n, brain_test_shuffled_n, feat_train_n, f_sc, ALPHA)
    brain_test_random_n = rng.standard_normal((n_test, n_voxels)).astype(np.float32)
    preds["Random"] = decode(brain_train_n, brain_test_random_n, feat_train_n, f_sc, ALPHA)

    n_cats = len(np.unique(cat_test))
    out = {}
    for cond, pred in preds.items():
        bc, _, _, _, _ = compute_bc(pred, cat_test, N_SHUFFLE, SEED, mode="across")
        ch = calinski_harabasz_score(pred, cat_test)
        out[cond] = {
            "norm":   float(np.mean(np.linalg.norm(pred, axis=1))),
            "pairid": pairwise_identification(pred, true_feat, img_index),
            "ch":     float(ch),
            "bc":     float(bc),
            "bc_cf":  float(bc_closed_form(ch, n_test, n_cats)),
        }
    return out, preds["Real"], cat_test


def subsample_analysis(pred, cat_test, rng):
    """1カテゴリあたりの試行数 n を変えて BC と CH を計算する（n ごとに複数回抽出して平均）。"""
    cats = np.unique(cat_test)
    bc_mean, ch_mean = [], []
    for n in SUBSAMPLE_N:
        bcs, chs = [], []
        for _ in range(SUBSAMPLE_DRAWS):
            idx = np.concatenate([
                rng.choice(np.where(cat_test == c)[0], size=n, replace=False) for c in cats
            ])
            bc, _, _, _, _ = compute_bc(pred[idx], cat_test[idx], SUBSAMPLE_SHUFFLE,
                                        SEED, mode="across")
            bcs.append(bc)
            chs.append(calinski_harabasz_score(pred[idx], cat_test[idx]))
        bc_mean.append(np.mean(bcs))
        ch_mean.append(np.mean(chs))
        print(f"  n={n:>2}: BC={bc_mean[-1]:.4f}  CH={ch_mean[-1]:.2f}")
    return np.array(bc_mean), np.array(ch_mean)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"特徴量を読み込み中（layer={LAYER}）...")
    feat, img_ids = load_image_features(FEAT_FILE, layer=LAYER)

    # ── 全被験者で各指標を計算 ──
    results = []
    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")
        r, pred_real, cat_test = run_subject(subject, feat, img_ids, rng)
        if subject == "Subject1":
            pred_real_s1, cat_test_s1 = pred_real, cat_test
        results.append(r)
        for cond in CONDITIONS:
            m = r[cond]
            print(f"  {cond:<9} norm={m['norm']:6.1f}  pairID={m['pairid']:.3f}  "
                  f"CH={m['ch']:6.2f}  BC={m['bc']:.4f} (closed form {m['bc_cf']:.4f})")

    metrics = ["norm", "pairid", "ch", "bc", "bc_cf"]
    table = {m: np.array([[r[c][m] for c in CONDITIONS] for r in results]) for m in metrics}

    print(f"\n{'='*70}")
    print("5被験者の平均 [最小, 最大]")
    for m in ["norm", "pairid", "ch", "bc"]:
        cells = [f"{table[m][:, j].mean():.3f} [{table[m][:, j].min():.3f}, {table[m][:, j].max():.3f}]"
                 for j in range(len(CONDITIONS))]
        print(f"  {m:<7} " + "  ".join(f"{c}: {s}" for c, s in zip(CONDITIONS, cells)))
    max_cf_diff = float(np.max(np.abs(table["bc"] - table["bc_cf"])))
    print(f"\nBC（1,000 回シャッフル）と閉じた式の差の最大値: {max_cf_diff:.5f}")

    # ── 試行数の影響（Subject1, Real）──
    print("\n=== 試行数 n の影響（Subject1, Real）===")
    rng_sub = np.random.default_rng(SEED)
    bc_n, ch_n = subsample_analysis(pred_real_s1, cat_test_s1, rng_sub)

    # ── プロット ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor="white")
    fig.subplots_adjust(wspace=0.42)
    x = np.arange(len(CONDITIONS))

    panels = [
        ("norm",   "(A) Feature norm\nno reference needed",           "Mean L2 norm of decoded features", None),
        ("pairid", "(B) Pairwise identification\nneeds true stimulus features", "Accuracy",                    0.5),
        ("bc",     "(C) BC\nneeds trial labels only",                  "BC",                                1.0),
    ]
    for ax, (m, title, ylabel, ref) in zip(axes[:3], panels):
        vals = table[m]
        ax.bar(x, vals.mean(axis=0), width=0.6,
               color=[COLORS[c] for c in CONDITIONS], alpha=0.85, zorder=2)
        for j in range(len(CONDITIONS)):
            ax.scatter(np.full(len(SUBJECTS), x[j]), vals[:, j], s=18, color="#333333",
                       zorder=3, linewidths=0)
        if ref is not None:
            ax.axhline(ref, color="#555555", linestyle="--", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)

    axes[1].set_ylim(0.4, 0.8)
    axes[1].text(0.5, 0.503, "chance", ha="center", va="bottom", fontsize=8, color="#555555")
    axes[2].set_ylim(0.95, table["bc"].max() * 1.03)
    axes[2].text(0.5, 1.002, "BC = 1", ha="center", va="bottom", fontsize=8, color="#555555")

    # (D) 試行数の影響: n=35 の値で割って共通の基準にそろえる
    ax = axes[3]
    bc_idx = (bc_n - 1) / (bc_n[-1] - 1)
    ch_idx = (ch_n - 1) / (ch_n[-1] - 1)
    ax.plot(SUBSAMPLE_N, ch_idx, "o-", color="#555555", linewidth=2, markersize=6, zorder=3)
    ax.plot(SUBSAMPLE_N, bc_idx, "o-", color=COLORS["Real"], linewidth=2, markersize=6, zorder=3)
    ax.text(27, 0.6, "CH − 1", ha="left", fontsize=9, color="#333333")
    ax.text(SUBSAMPLE_N[0] + 1, bc_idx[0] + 0.07, "BC − 1", ha="left", fontsize=9, color="#333333")
    ax.set_xlabel("Trials per category (n)", fontsize=9)
    ax.set_ylabel("Value relative to n = 35", fontsize=9)
    ax.set_title("(D) Dependence on trials per category\n(Subject 1, Real)", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", linewidth=0.6, zorder=0)

    out_path = os.path.join(OUTPUT_DIR, "exp19_existing_metrics.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    np.savez(
        os.path.join(OUTPUT_DIR, "exp19_existing_metrics.npz"),
        subjects=np.array(SUBJECTS),
        conditions=np.array(CONDITIONS),
        norm=table["norm"],
        pairid=table["pairid"],
        ch=table["ch"],
        bc=table["bc"],
        bc_closed_form=table["bc_cf"],
        subsample_n=np.array(SUBSAMPLE_N),
        subsample_bc=bc_n,
        subsample_ch=ch_n,
    )
    print("完了!")


if __name__ == "__main__":
    main()
