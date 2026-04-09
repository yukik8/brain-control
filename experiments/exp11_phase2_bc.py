"""
exp11_phase2_bc.py
-------------------
Phase 2 BC: 画像空間での脳制御度（DreamSim 埋め込みを使用）

手順:
1. 全1750枚の再構成画像の DreamSim 埋め込みを一括計算
2. カテゴリ内（35 rep）の埋め込み分散 = Var(preserved)
3. カテゴリ間シャッフル後の埋め込み分散 = Var(broken)
4. Phase 2 BC = Var(broken) / Var(preserved)

埋め込みレベルで分散を計算することで高速化（ペアワイズ計算不要）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from dreamsim import dreamsim
from tqdm import tqdm


# ── 設定 ──────────────────────────────────────────────────────────────────

RECON_DIR = (
    "brain-decoding-cookbook-public/reconstruction/data/reconstruction/"
    "fg_GOD/relu7generator/GOD_Subject1_VC/Subject1/VC"
)
OUTPUT_DIR = "outputs"
N_REPS = 35
N_SHUFFLE = 1000  # シャッフル試行数
SEED = 42

CATEGORY_NAMES = [
    "goldfish", "eagle_owl", "iguana", "duck", "lorikeet",
    "conch", "lobster", "killer_whale", "leopard", "dugong",
    "fly", "butterfly", "ibex", "camel", "llama",
    "airliner", "baseball", "bicycle", "bow_tie", "bullet_train",
    "cannon", "canoe", "container_ship", "church", "cloak",
    "drain", "electric_fan", "fire_engine", "football_helmet", "grand_piano",
    "greenhouse", "hair_slide", "hammer", "iron", "knot",
    "mailbag", "medicine_chest", "mobile_home", "monastery", "ping_pong_ball",
    "plate", "shovel", "ski", "slot_machine", "snowplow",
    "tape_player", "umbrella", "violin", "washer", "whistle",
]


def load_image_tensor(path):
    """TIFF 画像を (1, 3, H, W) テンソルに変換する。"""
    img = Image.open(path).convert("RGB").resize((224, 224))
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def extract_embeddings(model, image_paths, device, batch_size=32):
    """
    画像パスのリストから DreamSim 埋め込みを一括抽出する。
    Returns: np.ndarray of shape (N, D)
    """
    imgs = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            imgs.append(load_image_tensor(path))
            valid_indices.append(i)
        else:
            print(f"  見つからない: {os.path.basename(path)}")

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(imgs), batch_size):
            batch = torch.cat(imgs[start:start+batch_size], dim=0).to(device)
            # DreamSim の内部エンコーダーで埋め込みを取得
            emb = model.embed(batch)  # (B, D)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings), valid_indices


def mean_pairwise_l2(X):
    """X: (N, D) → 全ペアの L2 距離の平均"""
    n = X.shape[0]
    # 分散（各次元の分散の和 = 平均二乗距離 / 2 の近似）
    # 実際には ||xi - xj||^2 の平均 = 2 * tr(Cov)
    centered = X - X.mean(axis=0)
    return float(np.mean(np.sum(centered**2, axis=1)))


def compute_bc_phase2(embeddings_per_cat, n_shuffle=200, seed=42):
    """
    Phase 2 BC を計算する。

    Parameters
    ----------
    embeddings_per_cat : dict {cat_name: np.ndarray (N, D)}
    n_shuffle : シャッフル回数

    Returns
    -------
    bc_per_cat : dict {cat_name: float}
    """
    cats = list(embeddings_per_cat.keys())
    all_embs = np.vstack([embeddings_per_cat[c] for c in cats])  # (N_total, D)
    cat_labels = np.concatenate([
        np.full(len(embeddings_per_cat[c]), i) for i, c in enumerate(cats)
    ])

    rng = np.random.default_rng(seed)
    bc_per_cat = {}

    for cat in cats:
        emb_pres = embeddings_per_cat[cat]  # (N, D)
        var_pres = mean_pairwise_l2(emb_pres)

        # across-category shuffle: 他カテゴリから同数をランダムサンプリング
        n = len(emb_pres)
        cat_idx = cats.index(cat)
        other_mask = cat_labels != cat_idx

        var_broken_list = []
        for _ in range(n_shuffle):
            other_embs = all_embs[other_mask]
            shuffled = other_embs[rng.choice(len(other_embs), size=n, replace=False)]
            var_broken_list.append(mean_pairwise_l2(shuffled))

        var_broken = float(np.mean(var_broken_list))
        bc = var_broken / var_pres if var_pres > 0 else np.nan
        bc_per_cat[cat] = bc

    return bc_per_cat


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── DreamSim モデルの読み込み ──
    print("DreamSim モデルを読み込み中...")
    model, _ = dreamsim(pretrained=True, device=device)
    model.eval()

    # ── 全画像の埋め込みを一括計算 ──
    print("\n全画像の埋め込みを計算中...")
    embeddings_per_cat = {}

    for cat_name in tqdm(CATEGORY_NAMES):
        paths = [
            os.path.join(RECON_DIR, f"recon_image-{cat_name}_rep{rep:02d}.tiff")
            for rep in range(1, N_REPS + 1)
        ]
        embs, valid = extract_embeddings(model, paths, device)
        if len(embs) >= 2:
            embeddings_per_cat[cat_name] = embs
        else:
            print(f"  スキップ: {cat_name} (有効画像数={len(embs)})")

    print(f"埋め込み計算完了: {len(embeddings_per_cat)} カテゴリ")

    # ── Phase 2 BC の計算 ──
    print("\nPhase 2 BC を計算中...")
    bc_per_cat = compute_bc_phase2(embeddings_per_cat, n_shuffle=N_SHUFFLE, seed=SEED)

    cats = list(bc_per_cat.keys())
    bc_vals = np.array([bc_per_cat[c] for c in cats])

    print(f"\n=== Phase 2 BC（画像空間）結果 ===")
    print(f"BC 平均: {np.nanmean(bc_vals):.4f} ± {np.nanstd(bc_vals):.4f}")
    print(f"BC 最大: {cats[np.argmax(bc_vals)]} = {bc_vals.max():.4f}")
    print(f"BC 最小: {cats[np.argmin(bc_vals)]} = {bc_vals.min():.4f}")

    # カテゴリ別多様性
    within_vars = {c: mean_pairwise_l2(embeddings_per_cat[c]) for c in cats}
    print(f"\nカテゴリ内分散（DreamSim 埋め込み空間）:")
    for c in sorted(within_vars, key=within_vars.get, reverse=True)[:10]:
        print(f"  {c}: {within_vars[c]:.6f}, BC={bc_per_cat[c]:.4f}")

    # ── Phase 1 BC との比較のための within_var ──
    within_vars_arr = np.array([within_vars[c] for c in cats])
    print(f"\nカテゴリ内分散（画像空間）: mean={within_vars_arr.mean():.6f}")

    # ── 棒グラフ ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # (1) Phase 2 BC
    ax = axes[0]
    sorted_idx = np.argsort(bc_vals)[::-1]
    ax.bar(range(len(cats)), bc_vals[sorted_idx], color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([cats[i] for i in sorted_idx], rotation=90, fontsize=7)
    ax.axhline(np.nanmean(bc_vals), color="red", linestyle="--", label=f"mean={np.nanmean(bc_vals):.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="BC=1 (base)")
    ax.set_ylabel("Phase 2 BC")
    ax.set_title("Phase 2 BC: Image-space brain control (GOD, Subject1, VC)")
    ax.legend()

    # (2) カテゴリ内分散
    ax = axes[1]
    wv = np.array([within_vars[c] for c in cats])
    sorted_idx2 = np.argsort(wv)[::-1]
    ax.bar(range(len(cats)), wv[sorted_idx2], color="coral", alpha=0.8)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([cats[i] for i in sorted_idx2], rotation=90, fontsize=7)
    ax.axhline(wv.mean(), color="red", linestyle="--", label=f"mean={wv.mean():.5f}")
    ax.set_ylabel("Within-cat variance (DreamSim embedding)")
    ax.set_title("Within-category image diversity")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "phase2_bc_dreamsim.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nグラフ保存: {out_path}")

    # 数値保存
    np.savez(
        os.path.join(OUTPUT_DIR, "phase2_bc_dreamsim.npz"),
        categories=np.array(cats),
        bc_vals=bc_vals,
        within_vars=within_vars_arr
    )
    print("完了!")


if __name__ == "__main__":
    main()
