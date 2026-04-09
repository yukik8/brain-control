"""
exp12_phase1_vs_phase2_bc.py
-----------------------------
Experiment 12: Phase 1 BC vs Phase 2 BC のカテゴリ別比較

- Phase 1 BC: 特徴量空間（decoded cnn8 features, 1000次元）
- Phase 2 BC: 画像空間（DreamSim 埋め込み分散, Exp11 結果を再利用）

カテゴリを単位として2つのBCの相関・散布図を描く。
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


# ── 設定 ──────────────────────────────────────────────────────────────────

SUBJECT  = "Subject1"
ROI      = "ROI_VC"
LAYER    = "cnn8"
ALPHA    = 100.0
N_SHUFFLE = 1000
SEED      = 42

OUTPUT_DIR = "outputs"

CATEGORY_NAMES = {
    1:  "goldfish",       2:  "eagle_owl",      3:  "iguana",
    4:  "duck",           5:  "lorikeet",        6:  "conch",
    7:  "lobster",        8:  "killer_whale",    9:  "leopard",
    10: "dugong",         11: "fly",             12: "butterfly",
    13: "ibex",           14: "camel",           15: "llama",
    16: "airliner",       17: "baseball",        18: "bicycle",
    19: "bow_tie",        20: "bullet_train",    21: "cannon",
    22: "canoe",          23: "container_ship",  24: "church",
    25: "cloak",          26: "drain",           27: "electric_fan",
    28: "fire_engine",    29: "football_helmet", 30: "grand_piano",
    31: "greenhouse",     32: "hair_slide",      33: "hammer",
    34: "iron",           35: "knot",            36: "mailbag",
    37: "medicine_chest", 38: "mobile_home",     39: "monastery",
    40: "ping_pong_ball", 41: "plate",           42: "shovel",
    43: "ski",            44: "slot_machine",    45: "snowplow",
    46: "tape_player",    47: "umbrella",        48: "violin",
    49: "washer",         50: "whistle",
}


def compute_phase1_bc_per_category(pred_features, cat_labels, n_shuffle=1000, seed=42):
    """
    Phase 1 BC をカテゴリ別に計算する。

    Returns
    -------
    bc_per_cat : dict {cat_int: float}
    var_pres   : dict {cat_int: float}
    """
    rng = np.random.default_rng(seed)
    n = len(pred_features)
    cats = np.unique(cat_labels).astype(int)

    # カテゴリ内分散（preserved）
    var_pres = {}
    for cat in cats:
        mask = cat_labels == cat
        var_pres[cat] = float(np.mean(np.var(pred_features[mask], axis=0, ddof=1)))

    # across-shuffle（全試行をシャッフル）
    var_broken_acc = np.zeros(len(cats))
    for _ in range(n_shuffle):
        shuffled = pred_features[rng.permutation(n)]
        for i, cat in enumerate(cats):
            mask = cat_labels == cat
            var_broken_acc[i] += float(np.mean(np.var(shuffled[mask], axis=0, ddof=1)))
    var_broken_mean = var_broken_acc / n_shuffle

    bc_per_cat = {
        int(cat): var_broken_mean[i] / var_pres[int(cat)]
        for i, cat in enumerate(cats)
    }
    return bc_per_cat, var_pres


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Phase 1 BC の計算 ──
    print("脳活動・特徴量データを読み込み中...")
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

    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)

    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)
    pred_test = f_sc.inverse_transform(decoder.predict(brain_test_n)).astype(np.float32)

    print(f"Phase 1 BC（カテゴリ別）を計算中... N_SHUFFLE={N_SHUFFLE}")
    bc1_per_cat, var_pres1 = compute_phase1_bc_per_category(
        pred_test, cat_test, n_shuffle=N_SHUFFLE, seed=SEED
    )

    # ── Phase 2 BC の読み込み ──
    print("Phase 2 BC を読み込み中...")
    p2 = np.load(os.path.join(OUTPUT_DIR, "phase2_bc_dreamsim.npz"), allow_pickle=True)
    p2_cats  = list(p2["categories"])   # カテゴリ名リスト
    p2_bc    = p2["bc_vals"]
    p2_wvar  = p2["within_vars"]

    # カテゴリ名 → Phase 1 のカテゴリ番号に変換して整合させる
    name_to_int = {v: k for k, v in CATEGORY_NAMES.items()}

    # 共通カテゴリだけ抽出
    common_cats = [c for c in p2_cats if c in name_to_int and name_to_int[c] in bc1_per_cat]
    print(f"共通カテゴリ数: {len(common_cats)}")

    bc1 = np.array([bc1_per_cat[name_to_int[c]] for c in common_cats])
    bc2 = np.array([p2_bc[p2_cats.index(c)] for c in common_cats])
    vp1 = np.array([var_pres1[name_to_int[c]] for c in common_cats])
    vp2 = np.array([p2_wvar[p2_cats.index(c)] for c in common_cats])

    # ── 相関分析 ──
    r_bc, p_bc     = stats.pearsonr(bc1, bc2)
    r_vp, p_vp     = stats.pearsonr(vp1, vp2)

    print(f"\n=== Phase 1 vs Phase 2 BC 相関 ===")
    print(f"BC 相関:          r = {r_bc:.3f}, p = {p_bc:.3e}")
    print(f"Var(pres) 相関:   r = {r_vp:.3f}, p = {p_vp:.3e}")

    # ── プロット ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) Phase 1 BC vs Phase 2 BC
    ax = axes[0]
    ax.scatter(bc1, bc2, alpha=0.7, s=40, color="steelblue")
    # カテゴリ名ラベル（上位・下位5件）
    for i, cat in enumerate(common_cats):
        if bc2[i] > np.percentile(bc2, 90) or bc2[i] < np.percentile(bc2, 10):
            ax.annotate(cat, (bc1[i], bc2[i]), fontsize=6, alpha=0.8,
                        xytext=(4, 2), textcoords="offset points")
    # 回帰直線
    m, b = np.polyfit(bc1, bc2, 1)
    x_line = np.linspace(bc1.min(), bc1.max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Phase 1 BC (feature space, cnn8)")
    ax.set_ylabel("Phase 2 BC (image space, DreamSim)")
    ax.set_title(f"Phase 1 vs Phase 2 BC\nr={r_bc:.3f}, p={p_bc:.2e}")

    # (2) Var(preserved) 比較: Phase 1 vs Phase 2
    ax = axes[1]
    ax.scatter(vp1, vp2, alpha=0.7, s=40, color="coral")
    for i, cat in enumerate(common_cats):
        if vp1[i] > np.percentile(vp1, 90) or vp1[i] < np.percentile(vp1, 10):
            ax.annotate(cat, (vp1[i], vp2[i]), fontsize=6, alpha=0.8,
                        xytext=(4, 2), textcoords="offset points")
    m2, b2 = np.polyfit(vp1, vp2, 1)
    x2 = np.linspace(vp1.min(), vp1.max(), 100)
    ax.plot(x2, m2 * x2 + b2, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Var(preserved) Phase 1 (feature space)")
    ax.set_ylabel("Var(preserved) Phase 2 (DreamSim)")
    ax.set_title(f"Within-category variance\nr={r_vp:.3f}, p={p_vp:.2e}")

    # (3) カテゴリ別 BC1 vs BC2 ランキング比較（dot plot）
    ax = axes[2]
    sorted_by_bc1 = np.argsort(bc1)[::-1]
    y = np.arange(len(common_cats))
    ax.scatter(bc1[sorted_by_bc1], y, label="Phase 1 BC", alpha=0.7, s=20, color="steelblue")
    ax.scatter(bc2[sorted_by_bc1], y, label="Phase 2 BC", alpha=0.7, s=20, color="coral")
    ax.set_yticks(y[::5])
    ax.set_yticklabels([common_cats[i] for i in sorted_by_bc1[::5]], fontsize=7)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("BC value")
    ax.set_title("BC ranking (sorted by Phase 1)")
    ax.legend(fontsize=8)

    plt.suptitle("Phase 1 (feature) vs Phase 2 (image) BC — GOD Subject1 VC cnn8",
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "phase1_vs_phase2_bc.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nグラフ保存: {out_path}")

    # 数値保存
    np.savez(
        os.path.join(OUTPUT_DIR, "exp12_phase1_vs_phase2.npz"),
        categories=np.array(common_cats),
        bc1=bc1, bc2=bc2,
        var_pres1=vp1, var_pres2=vp2,
        r_bc=r_bc, p_bc=p_bc,
        r_vp=r_vp, p_vp=p_vp
    )
    print("完了!")


if __name__ == "__main__":
    main()
