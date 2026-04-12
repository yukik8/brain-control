"""
exp07_bc_per_category.py
---------------------------
Experiment 7: カテゴリ別 BC

BC_cat = Var(decoded_features | broken, cat) / Var(decoded_features | preserved, cat)

- 全50カテゴリについてBCを個別に計算
- 高BC / 低BC カテゴリの傾向を分析
- 設定: Subject1, ROI_VC, cnn8（最もBCが高かった条件）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compute_bc import (
    load_brain_data, load_image_features, run_one,
    DATA_DIR, FEAT_FILE, N_SHUFFLE
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# GOD テストカテゴリ（category_index 1-50 → 名前）
CATEGORY_NAMES = {
    1:  "goldfish",       2:  "eagle owl",      3:  "iguana",
    4:  "duck",           5:  "lorikeet",        6:  "conch",
    7:  "lobster",        8:  "killer whale",    9:  "leopard",
    10: "dugong",         11: "fly",             12: "butterfly",
    13: "ibex",           14: "camel",           15: "llama",
    16: "airliner",       17: "baseball",        18: "bicycle",
    19: "bow tie",        20: "bullet train",    21: "cannon",
    22: "canoe",          23: "container ship",  24: "church",
    25: "cloak",          26: "drain",           27: "electric fan",
    28: "fire engine",    29: "football helmet", 30: "grand piano",
    31: "greenhouse",     32: "hair slide",      33: "hammer",
    34: "iron",           35: "knot",            36: "mailbag",
    37: "medicine chest", 38: "mobile home",     39: "monastery",
    40: "ping-pong ball", 41: "plate",           42: "shovel",
    43: "ski",            44: "slot machine",    45: "snowplow",
    46: "tape player",    47: "umbrella",        48: "violin",
    49: "washer",         50: "whistle",
}


def compute_bc_per_category(pred_features, cat_labels, n_shuffle=1000, seed=42):
    """across-shuffle版（旧実装）: broken は全試行シャッフル"""
    rng = np.random.default_rng(seed)
    n = len(pred_features)
    cats = np.unique(cat_labels)

    var_pres = {}
    for cat in cats:
        mask = cat_labels == cat
        var_pres[cat] = float(np.mean(np.var(pred_features[mask], axis=0, ddof=1)))

    var_brok = {cat: [] for cat in cats}
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        shuffled = pred_features[perm]
        for cat in cats:
            mask = cat_labels == cat
            var_brok[cat].append(
                float(np.mean(np.var(shuffled[mask], axis=0, ddof=1)))
            )

    bc_per_cat = {
        cat: float(np.mean(var_brok[cat])) / var_pres[cat]
        for cat in cats
    }
    return bc_per_cat, var_pres


def compute_bc_per_category_other(pred_features, cat_labels, n_shuffle=1000, seed=42):
    """
    案1: other-category broken
    各カテゴリの枠に、そのカテゴリ以外の試行をランダムサンプリングして入れる。

    BC_cat = Var(broken_other, cat) / Var(preserved, cat)

    - broken_other の枠のサイズ = preserved と同じ（= そのカテゴリの試行数）
    - これにより Var(broken) がカテゴリごとに独立して計算され、比較が意味を持つ
    """
    rng = np.random.default_rng(seed)
    cats = np.unique(cat_labels)

    # preserved
    var_pres = {}
    cat_indices = {}
    for cat in cats:
        idx = np.where(cat_labels == cat)[0]
        cat_indices[cat] = idx
        var_pres[cat] = float(np.mean(np.var(pred_features[idx], axis=0, ddof=1)))

    # broken_other: 他カテゴリからサンプリング
    var_brok = {cat: [] for cat in cats}
    for _ in range(n_shuffle):
        for cat in cats:
            n_cat = len(cat_indices[cat])
            # cat 以外のインデックスをすべて集める
            other_idx = np.concatenate([
                cat_indices[c] for c in cats if c != cat
            ])
            # n_cat 個をランダムサンプリング（復元抽出）
            sampled = pred_features[rng.choice(other_idx, size=n_cat, replace=False)]
            var_brok[cat].append(
                float(np.mean(np.var(sampled, axis=0, ddof=1)))
            )

    bc_per_cat = {
        cat: float(np.mean(var_brok[cat])) / var_pres[cat]
        for cat in cats
    }
    return bc_per_cat, var_pres


def main():
    subject = "Subject1"
    roi     = "ROI_VC"
    layer   = "cnn8"

    print(f"Loading data: {subject} | {roi} | {layer}")
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{subject}.mat", roi_key=roi
    )
    feat, img_ids = load_image_features(FEAT_FILE, layer=layer)

    # Train/test split
    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train, brain_test = brain[i_train], brain[i_test]
    stim_train, stim_test   = stim_id[i_train], stim_id[i_test]
    cat_test = cat_id[i_test]

    # Feature matching
    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    # Normalize & decode
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)
    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    decoder = Ridge(alpha=100.0)
    decoder.fit(brain_train_n, feat_train_n)
    pred_test = decoder.predict(brain_test_n)

    # ── across-shuffle BC（旧） ──
    print(f"Computing BC [across-shuffle] (N_SHUFFLE={N_SHUFFLE})...")
    bc_across, var_pres = compute_bc_per_category(pred_test, cat_test, n_shuffle=N_SHUFFLE)

    # ── other-category BC（案1） ──
    print(f"Computing BC [other-category] (N_SHUFFLE={N_SHUFFLE})...")
    bc_other, _ = compute_bc_per_category_other(pred_test, cat_test, n_shuffle=N_SHUFFLE)

    cats = np.unique(cat_test)

    # 比較表
    print(f"\n{'='*70}")
    print(f"  Per-category BC 比較  |  {subject} | {roi} | {layer}")
    print(f"{'='*70}")
    print(f"  {'Rank(other)':<12} {'Cat':>4} {'Name':<20} {'BC(across)':>11} {'BC(other)':>10} {'Var(pres)':>10}")
    print(f"  {'─'*12} {'─'*4} {'─'*20} {'─'*11} {'─'*10} {'─'*10}")
    sorted_by_other = sorted(bc_other.items(), key=lambda x: x[1], reverse=True)
    for rank, (cat, bc_o) in enumerate(sorted_by_other, 1):
        name = CATEGORY_NAMES.get(int(cat), f"cat{int(cat)}")
        print(f"  {rank:<12} {int(cat):>4} {name:<20} {bc_across[cat]:>11.4f} {bc_o:>10.4f} {var_pres[cat]:>10.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    bc_across_arr = np.array([bc_across[c] for c in cats])
    bc_other_arr  = np.array([bc_other[c]  for c in cats])
    var_pres_arr  = np.array([var_pres[c]  for c in cats])
    names_all     = [CATEGORY_NAMES.get(int(c), f"cat{int(c)}") for c in cats]

    # (1) BC(other) バーチャート（案1）
    sorted_idx = np.argsort(bc_other_arr)[::-1]
    ax = axes[0]
    bc_s = bc_other_arr[sorted_idx]
    colors = ["tomato" if bc > np.mean(bc_s) + np.std(bc_s)
              else "steelblue" if bc < np.mean(bc_s) - np.std(bc_s)
              else "lightgray" for bc in bc_s]
    ax.barh(range(50), bc_s, color=colors, edgecolor="none")
    ax.set_yticks(range(50))
    ax.set_yticklabels([names_all[i] for i in sorted_idx], fontsize=7)
    ax.axvline(np.mean(bc_s), color="black", lw=1, ls="--", label=f"mean={np.mean(bc_s):.3f}")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("BC (other-category broken)")
    ax.set_title("Alt: BC per category\n(broken = other-category sampling)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # (2) BC(other) vs Var(preserved)
    ax = axes[1]
    ax.scatter(var_pres_arr, bc_other_arr, alpha=0.6, s=40, color="tomato")
    for i, name in enumerate(names_all):
        if bc_other_arr[i] > np.mean(bc_other_arr) + np.std(bc_other_arr) or \
           bc_other_arr[i] < np.mean(bc_other_arr) - np.std(bc_other_arr):
            ax.annotate(name, (var_pres_arr[i], bc_other_arr[i]), fontsize=6)
    slope, intercept, r_o, p_o, _ = stats.linregress(var_pres_arr, bc_other_arr)
    x_line = np.linspace(var_pres_arr.min(), var_pres_arr.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
            label=f"r={r_o:.3f}, p={p_o:.3f}")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Var(preserved)")
    ax.set_ylabel("BC (other-category)")
    ax.set_title("Alt: BC(other) vs Var(preserved)")
    ax.legend(fontsize=8)

    # (3) BC(across) vs BC(other) 比較
    ax = axes[2]
    ax.scatter(bc_across_arr, bc_other_arr, alpha=0.6, s=40, color="purple")
    for i, name in enumerate(names_all):
        if bc_other_arr[i] > np.mean(bc_other_arr) + np.std(bc_other_arr) or \
           bc_other_arr[i] < np.mean(bc_other_arr) - np.std(bc_other_arr):
            ax.annotate(name, (bc_across_arr[i], bc_other_arr[i]), fontsize=6)
    slope2, intercept2, r2, p2, _ = stats.linregress(bc_across_arr, bc_other_arr)
    x2 = np.linspace(bc_across_arr.min(), bc_across_arr.max(), 100)
    ax.plot(x2, slope2 * x2 + intercept2, "r-", lw=1.5, label=f"r={r2:.3f}")
    ax.set_xlabel("BC (across-shuffle, original)")
    ax.set_ylabel("BC (other-category, alt)")
    ax.set_title("across vs other-category BC")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("outputs/bc_per_category_v2.png", dpi=150)
    print(f"\nSaved: outputs/bc_per_category_v2.png")

    # Summary
    print(f"\n{'='*55}")
    print(f"  [across]  BC vs Var(pres): r = {stats.pearsonr(var_pres_arr, bc_across_arr)[0]:.3f}")
    print(f"  [other]   BC vs Var(pres): r = {r_o:.3f}  ← 改善されたか？")
    print(f"  [other]   mean={bc_other_arr.mean():.4f}  std={bc_other_arr.std():.4f}")
    print(f"  [other]   max={bc_other_arr.max():.4f} ({CATEGORY_NAMES[int(cats[np.argmax(bc_other_arr)])]})")
    print(f"  [other]   min={bc_other_arr.min():.4f} ({CATEGORY_NAMES[int(cats[np.argmin(bc_other_arr)])]})")
    print("\nDone!")


if __name__ == "__main__":
    main()
