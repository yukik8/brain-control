"""
exp10_category_attributes.py
-----------------------------
Experiment 10: カテゴリ属性 × Var(preserved) / BC 分析

カテゴリ別BCが「Var(preserved)の逆数」であることを踏まえ、
「どんな属性を持つカテゴリが高/低 Var(preserved) を示すか」を分析する。

属性分類:
  animate / inanimate
  natural / artificial
  small / large（おおよその実物サイズ）

設定: Subject1, ROI_VC, cnn8
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from compute_bc import (
    load_brain_data, load_image_features,
    DATA_DIR, FEAT_FILE
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── カテゴリ属性辞書 ────────────────────────────────────────────────────────
# (animate, natural, size)  size: "small"=手に乗る程度, "large"=それ以上
CATEGORY_ATTRS = {
    1:  ("goldfish",        "animate",   "natural",    "small"),
    2:  ("eagle owl",       "animate",   "natural",    "small"),
    3:  ("iguana",          "animate",   "natural",    "small"),
    4:  ("duck",            "animate",   "natural",    "small"),
    5:  ("lorikeet",        "animate",   "natural",    "small"),
    6:  ("conch",           "animate",   "natural",    "small"),
    7:  ("lobster",         "animate",   "natural",    "small"),
    8:  ("killer whale",    "animate",   "natural",    "large"),
    9:  ("leopard",         "animate",   "natural",    "large"),
    10: ("dugong",          "animate",   "natural",    "large"),
    11: ("fly",             "animate",   "natural",    "small"),
    12: ("butterfly",       "animate",   "natural",    "small"),
    13: ("ibex",            "animate",   "natural",    "large"),
    14: ("camel",           "animate",   "natural",    "large"),
    15: ("llama",           "animate",   "natural",    "large"),
    16: ("airliner",        "inanimate", "artificial", "large"),
    17: ("baseball",        "inanimate", "artificial", "small"),
    18: ("bicycle",         "inanimate", "artificial", "large"),
    19: ("bow tie",         "inanimate", "artificial", "small"),
    20: ("bullet train",    "inanimate", "artificial", "large"),
    21: ("cannon",          "inanimate", "artificial", "large"),
    22: ("canoe",           "inanimate", "artificial", "large"),
    23: ("container ship",  "inanimate", "artificial", "large"),
    24: ("church",          "inanimate", "artificial", "large"),
    25: ("cloak",           "inanimate", "artificial", "large"),
    26: ("drain",           "inanimate", "artificial", "small"),
    27: ("electric fan",    "inanimate", "artificial", "small"),
    28: ("fire engine",     "inanimate", "artificial", "large"),
    29: ("football helmet", "inanimate", "artificial", "small"),
    30: ("grand piano",     "inanimate", "artificial", "large"),
    31: ("greenhouse",      "inanimate", "artificial", "large"),
    32: ("hair slide",      "inanimate", "artificial", "small"),
    33: ("hammer",          "inanimate", "artificial", "small"),
    34: ("iron",            "inanimate", "artificial", "small"),
    35: ("knot",            "inanimate", "natural",    "small"),
    36: ("mailbag",         "inanimate", "artificial", "large"),
    37: ("medicine chest",  "inanimate", "artificial", "large"),
    38: ("mobile home",     "inanimate", "artificial", "large"),
    39: ("monastery",       "inanimate", "artificial", "large"),
    40: ("ping-pong ball",  "inanimate", "artificial", "small"),
    41: ("plate",           "inanimate", "artificial", "small"),
    42: ("shovel",          "inanimate", "artificial", "large"),
    43: ("ski",             "inanimate", "artificial", "large"),
    44: ("slot machine",    "inanimate", "artificial", "large"),
    45: ("snowplow",        "inanimate", "artificial", "large"),
    46: ("tape player",     "inanimate", "artificial", "small"),
    47: ("umbrella",        "inanimate", "artificial", "large"),
    48: ("violin",          "inanimate", "artificial", "large"),
    49: ("washer",          "inanimate", "artificial", "large"),
    50: ("whistle",         "inanimate", "artificial", "small"),
}


def get_pred_features(subject="Subject1", roi="ROI_VC", layer="cnn8"):
    brain, datatype, stim_id, cat_id = load_brain_data(
        f"{DATA_DIR}/{subject}.mat", roi_key=roi
    )
    feat, img_ids = load_image_features(FEAT_FILE, layer=layer)

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train, brain_test = brain[i_train], brain[i_test]
    stim_train = stim_id[i_train]
    cat_test   = cat_id[i_test]

    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)
    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    decoder = Ridge(alpha=100.0)
    decoder.fit(brain_train_n, feat_train_n)
    pred_test = decoder.predict(brain_test_n)

    return pred_test, cat_test


def compute_var_per_cat(pred_features, cat_labels):
    cats = np.unique(cat_labels)
    var_pres = {}
    for cat in cats:
        mask = cat_labels == cat
        var_pres[cat] = float(np.mean(np.var(pred_features[mask], axis=0, ddof=1)))
    return var_pres


def main():
    subject, roi, layer = "Subject1", "ROI_VC", "cnn8"
    print(f"Loading: {subject} | {roi} | {layer}")
    pred_test, cat_test = get_pred_features(subject, roi, layer)

    var_pres = compute_var_per_cat(pred_test, cat_test)
    cats = np.array(sorted(var_pres.keys()), dtype=int)
    var_arr = np.array([var_pres[c] for c in cats])

    # 属性ベクトル
    animate  = np.array([CATEGORY_ATTRS[c][1] for c in cats])
    natural  = np.array([CATEGORY_ATTRS[c][2] for c in cats])
    size     = np.array([CATEGORY_ATTRS[c][3] for c in cats])
    names    = [CATEGORY_ATTRS[c][0] for c in cats]

    # ── 統計検定 ──
    def report(label, mask_a, mask_b, name_a, name_b):
        a, b = var_arr[mask_a], var_arr[mask_b]
        t, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {label}: {name_a}(n={mask_a.sum()}, mean={a.mean():.4f}) vs "
              f"{name_b}(n={mask_b.sum()}, mean={b.mean():.4f})  "
              f"Mann-Whitney p={p:.4f}")
        return a, b

    print(f"\n{'='*65}")
    print(f"  Var(preserved) の属性別比較  ({subject} | {roi} | {layer})")
    print(f"{'='*65}")
    a_anim, a_inan = report("animate/inanimate",
                             animate=="animate", animate=="inanimate",
                             "animate", "inanimate")
    a_nat, a_art   = report("natural/artificial",
                             natural=="natural", natural=="artificial",
                             "natural", "artificial")
    a_sm, a_lg     = report("small/large",
                             size=="small", size=="large",
                             "small", "large")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    def boxplot_comparison(ax, group_a, group_b, labels, title, colors):
        bp = ax.boxplot([group_a, group_b], patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        # jitter
        for i, (grp, col) in enumerate(zip([group_a, group_b], colors), 1):
            x = np.random.default_rng(i).uniform(-0.15, 0.15, len(grp)) + i
            ax.scatter(x, grp, color=col, s=25, alpha=0.6, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Var(preserved)")
        ax.set_title(title)

    boxplot_comparison(axes[0], a_anim, a_inan,
                       ["animate", "inanimate"],
                       "animate vs inanimate",
                       ["steelblue", "tomato"])

    boxplot_comparison(axes[1], a_nat, a_art,
                       ["natural", "artificial"],
                       "natural vs artificial",
                       ["mediumseagreen", "orange"])

    boxplot_comparison(axes[2], a_sm, a_lg,
                       ["small", "large"],
                       "small vs large",
                       ["mediumpurple", "gold"])

    # 外れ値ラベル
    for ax, mask_a, mask_b in [
        (axes[0], animate=="animate",  animate=="inanimate"),
        (axes[1], natural=="natural",  natural=="artificial"),
        (axes[2], size=="small",       size=="large"),
    ]:
        for grp_idx, mask in [(1, mask_a), (2, mask_b)]:
            grp_var  = var_arr[mask]
            grp_name = [names[i] for i in range(len(names)) if mask[i]]
            q1, q3 = np.percentile(grp_var, [25, 75])
            iqr = q3 - q1
            for v, n in zip(grp_var, grp_name):
                if v > q3 + 1.5 * iqr or v < q1 - 1.5 * iqr:
                    ax.annotate(n, (grp_idx, v),
                                textcoords="offset points",
                                xytext=(6, 0), fontsize=7)

    fig.tight_layout()
    fig.savefig("outputs/bc_category_attributes.png", dpi=150)
    print("\nSaved: outputs/bc_category_attributes.png")

    # 属性ごとの上位/下位カテゴリ表示
    print(f"\n  Top 5 Var(preserved) — 各属性グループ:")
    for label, mask in [("animate", animate=="animate"),
                         ("inanimate", animate=="inanimate"),
                         ("small", size=="small"),
                         ("large", size=="large")]:
        idx_sorted = np.where(mask)[0][np.argsort(var_arr[mask])[::-1]]
        top = [(names[i], var_arr[i]) for i in idx_sorted[:5]]
        print(f"  [{label}] " + "  ".join(f"{n}({v:.3f})" for n, v in top))

    print("\nDone!")


if __name__ == "__main__":
    main()
