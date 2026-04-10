"""
export_baseline_features.py
----------------------------
Shuffled / Random 条件の decoded features を
brain-decoding-cookbook の DecodedFeatures フォーマットで保存する。

出力:
  decoded_features/GOD_Shuffled/relu7/Subject1/VC/*.mat
  decoded_features/GOD_Random/relu7/Subject1/VC/*.mat

Real 条件（decoded_features/GOD/）と同じ generator で再構成することで、
「画像の見た目は同等だが BC ≈ 1」を視覚的に示す。
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

DATA_DIR        = "data/god"
SUBJECT         = "Subject1"
ROI             = "ROI_VC"
ALPHA           = 100.0
SEED            = 42

FEAT_TRAIN_FILE = f"{DATA_DIR}/alexnet_features_train.npz"

COOKBOOK_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "brain-decoding-cookbook-public", "reconstruction", "data", "decoded_features"
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


def fname_to_imageid(fname):
    matches = re.findall(r'n(\d{8})_(\d+)\.JPEG', fname, re.IGNORECASE)
    if not matches:
        return None
    synset_str, img_str = matches[-1]
    return float(f"{int(synset_str)}.{int(img_str):06d}")


def load_brain_data(subject, roi_key="ROI_VC"):
    path = f"{DATA_DIR}/{subject}.h5"
    with h5py.File(path, "r") as f:
        meta = f["metaData"]
        key_names = [
            "".join(chr(c) for c in f[meta["key"][i, 0]][:].flatten())
            for i in range(meta["key"].shape[0])
        ]
        values = meta["value"]
        ds = f["dataSet"]

        def col(name):
            return np.where(values[:, key_names.index(name)] == 1)[0][0]
        def roi_cols(name):
            return np.where(values[:, key_names.index(name)] == 1)[0]

        brain    = ds[roi_cols(roi_key), :].T.astype(np.float32)
        datatype = ds[col("DataType"),      :].astype(np.float32)
        stim_id  = ds[col("stimulus_id"),   :].astype(np.float64)
        cat_id   = ds[col("category_index"),:].astype(np.float32)

    return brain, datatype, stim_id, cat_id


def save_features(pred_test, cat_test, output_dir):
    """decoded features を cookbook 形式で保存する。"""
    os.makedirs(output_dir, exist_ok=True)
    cats = np.unique(cat_test)
    n_saved = 0
    for cat in cats:
        cat_int = int(cat)
        cat_name = CATEGORY_NAMES.get(cat_int, f"cat{cat_int:02d}")
        idx = np.where(cat_test == cat)[0]
        for rep, trial_idx in enumerate(idx):
            label = f"{cat_name}_rep{rep+1:02d}"
            feat_vec = pred_test[trial_idx][np.newaxis, :]  # (1, 4096)
            hdf5storage.savemat(
                os.path.join(output_dir, f"{label}.mat"),
                {"feat": feat_vec},
                format="7.3", oned_as="row", store_python_metadata=False
            )
            n_saved += 1
    return n_saved


def main():
    rng = np.random.default_rng(SEED)

    # ── AlexNet 特徴量の読み込み ──
    print("AlexNet 特徴量を読み込み中...")
    train_npz = np.load(FEAT_TRAIN_FILE)
    feat_train_all = train_npz["features"].astype(np.float32)
    fnames_train   = train_npz["filenames"]

    id_to_feat = {}
    for i, fname in enumerate(fnames_train):
        img_id = fname_to_imageid(str(fname))
        if img_id is not None:
            id_to_feat[img_id] = feat_train_all[i]

    # ── 脳活動データの読み込み ──
    print(f"脳活動データを読み込み中: {SUBJECT} | {ROI}")
    brain, datatype, stim_id, cat_id = load_brain_data(SUBJECT, roi_key=ROI)

    i_train = datatype == 1
    i_test  = datatype == 2
    brain_train = brain[i_train]
    brain_test  = brain[i_test]
    stim_train  = stim_id[i_train]
    cat_test    = cat_id[i_test].astype(int)

    n_test, n_voxels = brain_test.shape

    # 訓練データのマッチング
    matched = []
    for idx, sid in enumerate(stim_train):
        best = min(id_to_feat.keys(), key=lambda x: abs(x - sid), default=None)
        if best is not None and abs(best - sid) < 0.001:
            matched.append((idx, best))

    match_idx = [m[0] for m in matched]
    brain_train_matched = brain_train[match_idx]
    feat_train_matched  = np.array([id_to_feat[m[1]] for m in matched], dtype=np.float32)

    # スケーリング
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train_matched)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train_matched)

    # Ridge デコーダーの訓練
    print("Ridge デコーダーを訓練中...")
    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)

    # ── 3条件でデコード ──

    # Shuffled: テスト脳活動をシャッフル
    print("\n条件: Shuffled（脳活動をシャッフル）")
    brain_test_shuffled_n = brain_test_n[rng.permutation(n_test)]
    pred_shuffled = f_sc.inverse_transform(
        decoder.predict(brain_test_shuffled_n)
    ).astype(np.float32)

    # Random: ガウスノイズ
    print("条件: Random（ガウスノイズ）")
    brain_test_random_n = rng.standard_normal((n_test, n_voxels)).astype(np.float32)
    pred_random = f_sc.inverse_transform(
        decoder.predict(brain_test_random_n)
    ).astype(np.float32)

    # ── 保存 ──
    for condition, pred in [("GOD_Shuffled", pred_shuffled), ("GOD_Random", pred_random)]:
        out_dir = os.path.join(COOKBOOK_BASE, condition, "relu7", "Subject1", "VC")
        n = save_features(pred, cat_test, out_dir)
        print(f"保存完了: {n} ファイル → {out_dir}")

    print("\n完了!")


if __name__ == "__main__":
    main()
