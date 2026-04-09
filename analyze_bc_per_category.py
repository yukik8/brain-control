"""
analyze_bc_per_category.py
---------------------------
Experiment 7: カテゴリ別 BC

BC_cat = Var(decoded_features | broken, cat) / Var(decoded_features | preserved, cat)

- 全50カテゴリについてBCを個別に計算
- 高BC / 低BC カテゴリの傾向を分析
- 設定: Subject1, ROI_VC, cnn8（最もBCが高かった条件）
"""

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
    """
    各カテゴリ別に BC を計算する。
    Returns: dict {cat_id: bc_mean}
    """
    rng = np.random.default_rng(seed)
    n = len(pred_features)
    cats = np.unique(cat_labels)

    # preserved: カテゴリ内分散
    var_pres = {}
    for cat in cats:
        mask = cat_labels == cat
        var_pres[cat] = float(np.mean(np.var(pred_features[mask], axis=0, ddof=1)))

    # broken: シャッフル後のカテゴリ内分散
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

    # Per-category BC
    print(f"Computing per-category BC (N_SHUFFLE={N_SHUFFLE})...")
    bc_per_cat, var_pres = compute_bc_per_category(
        pred_test, cat_test, n_shuffle=N_SHUFFLE
    )

    # Sort by BC
    sorted_cats = sorted(bc_per_cat.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*55}")
    print(f"  Per-category BC  |  {subject} | {roi} | {layer}")
    print(f"{'='*55}")
    print(f"  {'Rank':<5} {'Cat':>4} {'Name':<20} {'BC':>8} {'Var(pres)':>10}")
    print(f"  {'─'*5} {'─'*4} {'─'*20} {'─'*8} {'─'*10}")
    for rank, (cat, bc) in enumerate(sorted_cats, 1):
        name = CATEGORY_NAMES.get(int(cat), f"cat{int(cat)}")
        print(f"  {rank:<5} {int(cat):>4} {name:<20} {bc:>8.4f} {var_pres[cat]:>10.4f}")

    # Plot
    cats_sorted  = [int(c) for c, _ in sorted_cats]
    bc_sorted    = [bc for _, bc in sorted_cats]
    names_sorted = [CATEGORY_NAMES.get(c, f"cat{c}") for c in cats_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── (1) カテゴリ別BC バーチャート ──
    ax = axes[0]
    colors = ["tomato" if bc > np.mean(bc_sorted) + np.std(bc_sorted)
              else "steelblue" if bc < np.mean(bc_sorted) - np.std(bc_sorted)
              else "lightgray"
              for bc in bc_sorted]
    bars = ax.barh(range(50), bc_sorted, color=colors, edgecolor="none")
    ax.set_yticks(range(50))
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.axvline(np.mean(bc_sorted), color="black", lw=1, ls="--",
               label=f"mean={np.mean(bc_sorted):.3f}")
    ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("BC")
    ax.set_title(f"BC per category\n({subject} | {roi} | {layer})")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # ── (2) BC vs Var(preserved) scatter ──
    ax = axes[1]
    bc_vals   = np.array([bc_per_cat[c] for c in np.unique(cat_test)])
    var_vals  = np.array([var_pres[c]   for c in np.unique(cat_test)])
    names_all = [CATEGORY_NAMES.get(int(c), f"cat{int(c)}") for c in np.unique(cat_test)]

    ax.scatter(var_vals, bc_vals, alpha=0.6, s=40, color="steelblue")
    for i, name in enumerate(names_all):
        if bc_vals[i] > np.mean(bc_vals) + np.std(bc_vals) or \
           bc_vals[i] < np.mean(bc_vals) - np.std(bc_vals):
            ax.annotate(name, (var_vals[i], bc_vals[i]),
                        fontsize=6, ha="left", va="bottom")

    slope, intercept, r, p, _ = stats.linregress(var_vals, bc_vals)
    x_line = np.linspace(var_vals.min(), var_vals.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
            label=f"r={r:.3f}, p={p:.3f}")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Var(preserved)  — within-cat variance of decoded features")
    ax.set_ylabel("BC")
    ax.set_title("BC vs within-category variability (preserved)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("bc_per_category.png", dpi=150)
    print(f"\nSaved: bc_per_category.png")

    # Summary stats
    bc_vals_all = np.array(list(bc_per_cat.values()))
    print(f"\n  mean BC = {bc_vals_all.mean():.4f}")
    print(f"  std  BC = {bc_vals_all.std():.4f}")
    print(f"  max  BC = {bc_vals_all.max():.4f}  ({CATEGORY_NAMES[int(cats_sorted[0])]})")
    print(f"  min  BC = {bc_vals_all.min():.4f}  ({CATEGORY_NAMES[int(cats_sorted[-1])]})")

    print(f"\n  BC vs Var(pres): r={r:.3f}, p={p:.3f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
