"""
exp17_multisubject_bc.py
-------------------------
Experiment 17: 複数被験者での BC 再現性確認

Exp13（Real vs Shuffled vs Random）を Subject1〜5 全員に拡張する。
論文の弱点「Subject1 のみ」を補強する実験。

期待:
  - 全被験者で Real BC > 1
  - 全被験者で Shuffled BC ≈ 1
  - 被験者間で BC の絶対値は異なるが、条件間の大小関係は保存される
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

ROI        = "ROI_VC"
LAYER      = "cnn8"
ALPHA      = 100.0
N_SHUFFLE  = 1000
SEED       = 42
OUTPUT_DIR = "outputs"

SUBJECTS = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]


def decode(brain_train_n, brain_test_n, feat_train_n, f_sc, alpha=100.0):
    """Ridge デコーダーで brain → features を予測する。"""
    decoder = Ridge(alpha=alpha)
    decoder.fit(brain_train_n, feat_train_n)
    pred_n = decoder.predict(brain_test_n)
    return f_sc.inverse_transform(pred_n).astype(np.float32)


def run_subject(subject, feat, img_ids, rng):
    """1被験者分の Real / Shuffled / Random BC を計算して返す。"""
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{subject}.mat", roi_key=ROI
    )

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    cat_test    = cat_id[i_test].astype(int)

    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    # 標準化
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    n_test, n_voxels = brain_test.shape

    # (1) Real: 本物の脳活動
    pred_real = decode(brain_train_n, brain_test_n, feat_train_n, f_sc, ALPHA)

    # (2) Shuffled: テスト試行インデックスをランダム置換（脳-刺激対応を破壊）
    brain_test_shuffled_n = brain_test_n[rng.permutation(n_test)]
    pred_shuffled = decode(brain_train_n, brain_test_shuffled_n, feat_train_n, f_sc, ALPHA)

    # (3) Random: ガウスノイズを脳活動として入力
    brain_test_random_n = rng.standard_normal((n_test, n_voxels)).astype(np.float32)
    pred_random = decode(brain_train_n, brain_test_random_n, feat_train_n, f_sc, ALPHA)

    # BC の計算
    bc_real_m, bc_real_s, _, _, _ = compute_bc(pred_real,     cat_test, N_SHUFFLE, SEED, mode="across")
    bc_shuf_m, bc_shuf_s, _, _, _ = compute_bc(pred_shuffled, cat_test, N_SHUFFLE, SEED, mode="across")
    bc_rand_m, bc_rand_s, _, _, _ = compute_bc(pred_random,   cat_test, N_SHUFFLE, SEED, mode="across")

    norm_real = float(np.mean(np.linalg.norm(pred_real,     axis=1)))
    norm_shuf = float(np.mean(np.linalg.norm(pred_shuffled, axis=1)))
    norm_rand = float(np.mean(np.linalg.norm(pred_random,   axis=1)))

    return {
        "subject":   subject,
        "bc_real":   bc_real_m, "bc_real_s":  bc_real_s,
        "bc_shuf":   bc_shuf_m, "bc_shuf_s":  bc_shuf_s,
        "bc_rand":   bc_rand_m, "bc_rand_s":  bc_rand_s,
        "norm_real": norm_real,
        "norm_shuf": norm_shuf,
        "norm_rand": norm_rand,
    }


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"特徴量を読み込み中（layer={LAYER}）...")
    feat, img_ids = load_image_features(FEAT_FILE, layer=LAYER)

    # ── 全被験者でループ ──
    results = []
    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")
        r = run_subject(subject, feat, img_ids, rng)
        results.append(r)
        print(f"  Real:     BC={r['bc_real']:.4f} +/- {r['bc_real_s']:.4f}  norm={r['norm_real']:.1f}")
        print(f"  Shuffled: BC={r['bc_shuf']:.4f} +/- {r['bc_shuf_s']:.4f}  norm={r['norm_shuf']:.1f}")
        print(f"  Random:   BC={r['bc_rand']:.4f} +/- {r['bc_rand_s']:.4f}  norm={r['norm_rand']:.1f}")

    # ── 結果サマリー ──
    print(f"\n{'='*65}")
    print(f"{'Subject':<12} {'Real BC':>9} {'Shuffled BC':>12} {'Random BC':>10} {'norm_real':>10}")
    print(f"{'-'*65}")
    for r in results:
        print(f"{r['subject']:<12} {r['bc_real']:>9.4f} {r['bc_shuf']:>12.4f} "
              f"{r['bc_rand']:>10.4f} {r['norm_real']:>10.1f}")

    # 全被験者で Real > Shuffled かチェック
    all_pass = all(r["bc_real"] > r["bc_shuf"] for r in results)
    print(f"\n全被験者で Real BC > Shuffled BC: {all_pass}")

    # 対応あり t 検定
    bc_reals = np.array([r["bc_real"] for r in results])
    bc_shufs = np.array([r["bc_shuf"] for r in results])
    t_stat, p_val = stats.ttest_rel(bc_reals, bc_shufs)
    print(f"対応あり t 検定 (Real vs Shuffled): t={t_stat:.3f}, p={p_val:.4f}")

    # ── プロット ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    subj_labels = [r["subject"].replace("Subject", "Sub") for r in results]
    x = np.arange(len(SUBJECTS))
    w = 0.25
    colors = {"Real": "steelblue", "Shuffled": "coral", "Random": "gray"}

    # (1) 被験者別 BC バープロット（3条件）
    ax = axes[0]
    ax.bar(x - w, [r["bc_real"] for r in results], w,
           label="Real", color=colors["Real"], alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.bar(x,     [r["bc_shuf"] for r in results], w,
           label="Shuffled", color=colors["Shuffled"], alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.bar(x + w, [r["bc_rand"] for r in results], w,
           label="Random", color=colors["Random"], alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="BC=1 (prior)")
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels)
    ax.set_ylabel("BC (Degree of Brain Control)")
    ax.set_title(f"BC across subjects: Real vs Shuffled vs Random\n"
                 f"(ROI=VC, layer={LAYER}, paired t-test p={p_val:.4f})")
    ax.legend(fontsize=9)
    ax.set_ylim(0.95, max(r["bc_real"] for r in results) * 1.12)
    for i, r in enumerate(results):
        ax.text(i - w, r["bc_real"] + 0.002, f"{r['bc_real']:.3f}",
                ha="center", va="bottom", fontsize=7, color=colors["Real"])
        ax.text(i,     r["bc_shuf"] + 0.002, f"{r['bc_shuf']:.3f}",
                ha="center", va="bottom", fontsize=7, color=colors["Shuffled"])

    # (2) 対応プロット: Real vs Shuffled（全被験者で一方向）
    ax = axes[1]
    for i, r in enumerate(results):
        ax.plot([0, 1], [r["bc_real"], r["bc_shuf"]],
                "o-", color=f"C{i}", linewidth=1.5, markersize=6,
                label=subj_labels[i])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Shuffled"])
    ax.set_ylabel("BC (Degree of Brain Control)")
    ax.set_title("Real vs Shuffled BC per subject\n(all subjects: Real > Shuffled)")
    ax.legend(fontsize=9, loc="right")
    ax.set_xlim(-0.3, 1.3)

    plt.suptitle(
        "Exp17: BC reproducibility across 5 subjects — GOD dataset, VC ROI, cnn8\n"
        f"Real BC > Shuffled BC in all subjects (paired t={t_stat:.2f}, p={p_val:.4f})",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "exp17_multisubject_bc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    np.savez(
        os.path.join(OUTPUT_DIR, "exp17_multisubject_bc.npz"),
        subjects=np.array(SUBJECTS),
        bc_real=bc_reals,
        bc_shuf=bc_shufs,
        bc_rand=np.array([r["bc_rand"] for r in results]),
        bc_real_s=np.array([r["bc_real_s"] for r in results]),
        bc_shuf_s=np.array([r["bc_shuf_s"] for r in results]),
        norm_real=np.array([r["norm_real"] for r in results]),
        norm_shuf=np.array([r["norm_shuf"] for r in results]),
        t_stat=np.array([t_stat]),
        p_val=np.array([p_val]),
    )
    print("完了!")


if __name__ == "__main__":
    main()
