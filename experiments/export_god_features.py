"""
export_god_features.py
-----------------------
GOD テストデータの decoded AlexNet relu7 features を
brain-decoding-cookbook の DecodedFeatures フォーマットに変換する。

出力ディレクトリ構造:
  {output_dir}/relu7/{subject}/{roi}/{category}_{rep:02d}.mat

各 .mat ファイルには {'feat': array(1, 4096)} が格納される。

AlexNet relu7 (fc7) = pretrained torchvision AlexNet, 4096次元
→ recon_fg_image.py の relu7generator と直接互換。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
import h5py
import hdf5storage
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── 設定 ──────────────────────────────────────────────────────────────────

DATA_DIR   = "data/god"
SUBJECT    = "Subject1"
ROI        = "ROI_VC"
ALPHA      = 100.0

FEAT_TRAIN_FILE = f"{DATA_DIR}/alexnet_features_train.npz"
FEAT_TEST_FILE  = f"{DATA_DIR}/alexnet_features_test.npz"

# cookbook が期待するディレクトリ名
LAYER_DIR   = "relu7"
SUBJECT_DIR = "Subject1"
ROI_DIR     = "VC"

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "brain-decoding-cookbook-public", "reconstruction", "data",
    "decoded_features", "GOD", LAYER_DIR, SUBJECT_DIR, ROI_DIR
)

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


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────────

def fname_to_imageid(fname):
    """
    ファイル名から ImageID（浮動小数点数）を復元する。

    例:
      'n01518878_10042.JPEG'
        → synset=1518878, img=10042 → 1518878.010042
      '+Alaska.jpg,2374451.011539,n02374451_11539.JPEG'
        → カンマ区切りの最後にある標準形式部分を使用 → 2374451.011539
    """
    # nXXXXXXXX_YYYY.JPEG 形式を末尾から探す
    matches = re.findall(r'n(\d{8})_(\d+)\.JPEG', fname, re.IGNORECASE)
    if not matches:
        return None
    synset_str, img_str = matches[-1]
    synset_int = int(synset_str)
    img_int = int(img_str)
    # ImageID = シネット番号 + 小数6桁の画像番号
    return float(f"{synset_int}.{img_int:06d}")


def load_brain_data(subject, roi_key="ROI_VC"):
    """脳活動データを HDF5 ファイルから読み込む。
    Returns: brain, datatype, stimulus_id, category_index
    """
    path = f"{DATA_DIR}/{subject}.h5"
    with h5py.File(path, "r") as f:
        meta = f["metaData"]
        key_names = [
            "".join(chr(c) for c in f[meta["key"][i, 0]][:].flatten())
            for i in range(meta["key"].shape[0])
        ]
        values = meta["value"]   # (n_cols, n_meta_keys)
        ds = f["dataSet"]        # (n_cols, n_samples)

        def col(name):
            return np.where(values[:, key_names.index(name)] == 1)[0][0]

        def roi_cols(name):
            return np.where(values[:, key_names.index(name)] == 1)[0]

        brain    = ds[roi_cols(roi_key), :].T.astype(np.float32)
        datatype = ds[col("DataType"),      :].astype(np.float32)
        stim_id  = ds[col("stimulus_id"),   :].astype(np.float64)
        cat_id   = ds[col("category_index"),:].astype(np.float32)

    return brain, datatype, stim_id, cat_id


def main():
    # ── AlexNet 特徴量の読み込み ──
    print(f"AlexNet 特徴量を読み込み中...")
    train_npz = np.load(FEAT_TRAIN_FILE)
    feat_train_all = train_npz["features"].astype(np.float32)   # (N_train_imgs, 4096)
    fnames_train   = train_npz["filenames"]                      # (N_train_imgs,)

    # ImageID → 特徴量ベクトルの辞書を構築
    id_to_feat = {}
    for i, fname in enumerate(fnames_train):
        img_id = fname_to_imageid(str(fname))
        if img_id is not None:
            id_to_feat[img_id] = feat_train_all[i]

    print(f"  ImageID にマッピングされた訓練画像数: {len(id_to_feat)}")

    # ── 脳活動データの読み込み ──
    print(f"\n脳活動データを読み込み中: {SUBJECT} | {ROI}")
    brain, datatype, stim_id, cat_id = load_brain_data(SUBJECT, roi_key=ROI)

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    cat_test    = cat_id[i_test]

    print(f"  訓練試行数: {brain_train.shape[0]}, テスト試行数: {brain_test.shape[0]}")

    # ── 訓練試行と AlexNet 特徴量のマッチング ──
    matched, skipped = [], []
    for idx, sid in enumerate(stim_train):
        # 最近傍 ImageID を探す（浮動小数点の誤差に対応）
        best = min(id_to_feat.keys(), key=lambda x: abs(x - sid), default=None)
        if best is not None and abs(best - sid) < 0.001:
            matched.append((idx, best))
        else:
            skipped.append(sid)

    print(f"  マッチ成功: {len(matched)}, スキップ（画像なし）: {len(skipped)}")

    if len(matched) < 100:
        raise RuntimeError("マッチした訓練サンプルが少なすぎます — 画像ダウンロードを確認してください。")

    match_idx = [m[0] for m in matched]
    brain_train_matched = brain_train[match_idx]
    feat_train_matched  = np.array([id_to_feat[m[1]] for m in matched], dtype=np.float32)

    # ── Ridge デコーダーの訓練 ──
    print("Ridge デコーダーを訓練中 (脳活動 → AlexNet relu7)...")
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train_matched)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train_matched)

    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)
    pred_test_n = decoder.predict(brain_test_n)

    # 元のスケールに逆変換
    pred_test = f_sc.inverse_transform(pred_test_n).astype(np.float32)
    print(f"デコードされた特徴量の形状: {pred_test.shape}")  # (1750, 4096)

    # ── cookbook DecodedFeatures 形式で保存 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cats = np.unique(cat_test)
    n_saved = 0
    for cat in cats:
        cat_int = int(cat)
        cat_name = CATEGORY_NAMES.get(cat_int, f"cat{cat_int:02d}")
        idx = np.where(cat_test == cat)[0]  # 各カテゴリ 35 試行

        for rep, trial_idx in enumerate(idx):
            label = f"{cat_name}_rep{rep+1:02d}"
            filepath = os.path.join(OUTPUT_DIR, f"{label}.mat")
            feat_vec = pred_test[trial_idx][np.newaxis, :]  # (1, 4096)
            hdf5storage.savemat(filepath, {"feat": feat_vec},
                                format="7.3", oned_as="row",
                                store_python_metadata=False)
            n_saved += 1

    print(f"{n_saved} ファイルを保存しました → {OUTPUT_DIR}")
    print(f"例: {CATEGORY_NAMES[1]}_rep01.mat → shape (1, 4096)")
    print("完了!")


if __name__ == "__main__":
    main()
