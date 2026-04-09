"""
compute_bc.py
-------------
Phase 1: Feature-space Degree of Brain Control (BC)

BC = mean_category( Var(decoded_features | broken) / Var(decoded_features | preserved) )

- Preserved : correct brain-stimulus pairing
- Broken    : trial-wise shuffle of brain activity (stimulus labels stay fixed)
- Variability: mean within-category variance across feature dimensions
- Decoder   : Ridge regression (brain → cnn8 features, 1000 units)

Usage:
    python compute_bc.py [--roi ROI_NAME] [--layer LAYER_NAME]
    python compute_bc.py --all-rois
"""

import argparse
import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR   = "data/god"
SUBJECT    = "Subject1"
BRAIN_FILE = f"{DATA_DIR}/{SUBJECT}.mat"
FEAT_FILE  = f"{DATA_DIR}/ImageFeatures.h5"
LAYER      = "cnn8"
ALPHA      = 100.0
N_SHUFFLE  = 50

ALL_ROIS = ["ROI_V1", "ROI_V2", "ROI_V3", "ROI_V4",
            "ROI_LOC", "ROI_FFA", "ROI_PPA",
            "ROI_LVC", "ROI_HVC", "ROI_VC"]


# ── Loader: Subject*.mat (MATLAB v7.3 / HDF5) ───────────────────────────────

def load_brain_data(path, roi_key="ROI_VC"):
    """Returns brain, datatype, stimulus_id, category_index (all numpy arrays)."""
    with h5py.File(path, "r") as f:
        meta = f["metaData"]
        key_names = [
            "".join(chr(c) for c in f[meta["key"][i, 0]][:].flatten())
            for i in range(meta["key"].shape[0])
        ]
        values = meta["value"]   # shape: (n_cols, n_meta_keys)
        ds     = f["dataSet"]    # shape: (n_cols, n_samples)  [h5py Fortran-order]

        def col(name):
            return np.where(values[:, key_names.index(name)] == 1)[0][0]

        def roi_cols(name):
            return np.where(values[:, key_names.index(name)] == 1)[0]

        brain    = ds[roi_cols(roi_key), :].T.astype(np.float32)  # (n_samples, n_voxels)
        datatype = ds[col("DataType"),      :].astype(np.float32)  # (n_samples,)
        stim_id  = ds[col("stimulus_id"),   :].astype(np.float64)  # (n_samples,)
        cat_id   = ds[col("category_index"),:].astype(np.float32)  # (n_samples,)

    return brain, datatype, stim_id, cat_id


# ── Loader: ImageFeatures.h5 ────────────────────────────────────────────────

def load_image_features(path, layer="cnn8"):
    """Returns features (n_images, n_units) and image_ids (n_images,)."""
    with h5py.File(path, "r") as f:
        meta      = f["metaData"]
        key_names = [k.decode() for k in meta["key"][:]]
        values    = meta["value"]   # shape: (n_meta_keys, n_cols)
        ds        = f["dataSet"]    # shape: (n_rows, n_cols)

        feat_cols = np.where(values[key_names.index(layer), :] == 1)[0]
        img_col   = np.where(values[key_names.index("ImageID"), :] == 1)[0][0]

        image_ids = ds[:, img_col]
        valid     = ~np.isnan(image_ids)

        features  = ds[valid, :][:, feat_cols].astype(np.float32)
        image_ids = image_ids[valid]

    return features, image_ids


# ── BC Computation ───────────────────────────────────────────────────────────

def within_category_variance(features, cat_labels):
    """Mean of per-category, per-unit variance (scalar)."""
    variances = []
    for cat in np.unique(cat_labels):
        mask = cat_labels == cat
        if mask.sum() < 2:
            continue
        variances.append(np.mean(np.var(features[mask], axis=0, ddof=1)))
    return float(np.mean(variances))


def compute_bc(pred_features, cat_labels, n_shuffle=50, seed=42):
    """
    Returns:
        bc_mean   : mean BC across shuffles
        bc_std    : std  of BC across shuffles
        var_pres  : variability (preserved)
        var_brok_list : list of variabilities (broken, one per shuffle)
    """
    rng = np.random.default_rng(seed)
    var_pres = within_category_variance(pred_features, cat_labels)

    var_brok_list = []
    n = len(pred_features)
    for _ in range(n_shuffle):
        perm = rng.permutation(n)
        var_brok = within_category_variance(pred_features[perm], cat_labels)
        var_brok_list.append(var_brok)

    bc_values = np.array(var_brok_list) / var_pres
    return bc_values.mean(), bc_values.std(), var_pres, var_brok_list


def effective_dimensionality(features):
    """Participation ratio: (sum λ)^2 / sum(λ^2)"""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(features)
    lambdas = pca.explained_variance_
    return float((lambdas.sum() ** 2) / (lambdas ** 2).sum())


# ── Single-ROI Run ───────────────────────────────────────────────────────────

def run_one(roi_key, layer, brain_all, datatype, stim_id, cat_id, feat, img_ids):
    """Run BC analysis for one ROI × layer combination. Returns result dict."""

    # Split train / test (perception)
    i_train = (datatype == 1)
    i_test  = (datatype == 2)

    brain_train = brain_all[roi_key][i_train]
    brain_test  = brain_all[roi_key][i_test]
    stim_train  = stim_id[i_train]
    stim_test   = stim_id[i_test]
    cat_test    = cat_id[i_test]

    # Match image features to brain data
    id_to_feat = {img_ids[i]: feat[i] for i in range(len(img_ids))}
    feat_train = np.array([id_to_feat[s] for s in stim_train], dtype=np.float32)
    feat_test  = np.array([id_to_feat[s] for s in stim_test],  dtype=np.float32)

    # Normalize
    b_sc = StandardScaler()
    brain_train_n = b_sc.fit_transform(brain_train)
    brain_test_n  = b_sc.transform(brain_test)

    f_sc = StandardScaler()
    feat_train_n = f_sc.fit_transform(feat_train)

    # Train Ridge
    decoder = Ridge(alpha=ALPHA)
    decoder.fit(brain_train_n, feat_train_n)

    # Predict
    pred_test = decoder.predict(brain_test_n)

    # Decoding accuracy
    true_test = f_sc.transform(feat_test)
    r_vals = np.array([
        np.corrcoef(pred_test[:, u], true_test[:, u])[0, 1]
        for u in range(pred_test.shape[1])
    ])
    mean_r = float(np.nanmean(r_vals))

    # BC
    bc_mean, bc_std, var_pres, var_brok_list = compute_bc(
        pred_test, cat_test, n_shuffle=N_SHUFFLE
    )

    # Effective Dimensionality
    ed = effective_dimensionality(pred_test)

    n_voxels = brain_train.shape[1]

    return {
        "roi":       roi_key,
        "layer":     layer,
        "n_voxels":  n_voxels,
        "mean_r":    mean_r,
        "var_pres":  var_pres,
        "var_brok":  float(np.mean(var_brok_list)),
        "bc_mean":   bc_mean,
        "bc_std":    bc_std,
        "ed":        ed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi",       default=None, help="single ROI key, e.g. ROI_V1")
    parser.add_argument("--layer",     default=LAYER)
    parser.add_argument("--all-rois",  action="store_true")
    args = parser.parse_args()

    rois_to_run = ALL_ROIS if args.all_rois else [args.roi or "ROI_VC"]
    layer = args.layer

    print("=" * 60)
    print(f"  BC — Feature Space  |  {SUBJECT} | {layer}")
    print(f"  ROIs: {rois_to_run}")
    print("=" * 60)

    # Load brain data for all ROIs at once
    print("\n[1] Loading brain data...")
    brain_all = {}
    datatype = stim_id = cat_id = None
    for roi in rois_to_run:
        b, dt, sid, cid = load_brain_data(BRAIN_FILE, roi_key=roi)
        brain_all[roi] = b
        datatype, stim_id, cat_id = dt, sid, cid
        print(f"    {roi:10s}: {b.shape[1]:5d} voxels")

    print("\n[2] Loading image features...")
    feat, img_ids = load_image_features(FEAT_FILE, layer=layer)
    print(f"    {layer}: {feat.shape}")

    # Run BC for each ROI
    results = []
    for roi in rois_to_run:
        print(f"\n{'─'*50}")
        print(f"  ROI: {roi}  |  Layer: {layer}")
        r = run_one(roi, layer, brain_all, datatype, stim_id, cat_id, feat, img_ids)
        results.append(r)
        print(f"  n_voxels = {r['n_voxels']:4d}  |  mean_r = {r['mean_r']:.4f}")
        print(f"  Var(pres) = {r['var_pres']:.4f}  |  Var(brok) = {r['var_brok']:.4f}")
        print(f"  >>> BC = {r['bc_mean']:.4f} ± {r['bc_std']:.4f}   ED = {r['ed']:.2f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY  —  {layer}")
    print(f"{'='*60}")
    print(f"  {'ROI':<10} {'Voxels':>7} {'mean_r':>8} {'BC':>8} {'±':>6} {'ED':>7}")
    print(f"  {'─'*10} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*7}")
    for r in results:
        print(f"  {r['roi']:<10} {r['n_voxels']:>7} {r['mean_r']:>8.4f} "
              f"{r['bc_mean']:>8.4f} {r['bc_std']:>6.4f} {r['ed']:>7.2f}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
