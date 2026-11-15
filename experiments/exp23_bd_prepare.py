"""
exp23_bd_prepare.py
-------------------
Experiment 23b: Brain-Diffuser のデータの準備（NSD の AWS のデータから）

Brain-Diffuser（Ozcelik & VanRullen, 2023）の data/prepare_nsddata.py を元にしている。
学習データは元のコードと同じ（1〜37 セッション、nsdgeneral のマスク、同じ画像の試行を平均）。
テストデータは、元のコードと同じ平均（再現の確認 G2 用）に加えて、平均しない単一試行も保存する。

元のコードから変えた点（どれも値は変わらない）：
  - betas を get_fdata（float64）ではなく int16 のまま読んでからマスクする（slope 1、inter 0 を確かめる）
  - 刺激を float64 ではなく uint8 で持つ。hdf5 を全部メモリに読まず、使う画像だけ読む
  - fMRI を float32 で保存する（元のコードは float32 の平均を float64 の配列に入れて保存する。値は同じ）
  - 単一試行のテストデータ（test_trials.npz）を足した。試行は時間の順（MindEye1 の webdataset の 3 試行と同じ順）

出力（--out_dir。NSD のデータなのでリポジトリの外に置く）：
  train_fmri.npy（8,859 × ボクセル）、train_stim.npy（uint8、425 × 425 × 3）、train_cap.npy（5 つのキャプション）
  test_fmri_avg.npy（982 × ボクセル）、test_stim.npy、test_cap.npy（並びは元のコードと同じ）
  test_trials.npz：fmri（単一試行）、img（test_* の行）、trial（NSD の試行の番号、0 から）、rep（時間の順）、n_reps
  meta.npz：train_ids、test_ids（73k の番号、0 から）、test_stim20（Exp20 の刺激の番号。Exp20 にない画像は -1）
"""

import os
import argparse
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

parser = argparse.ArgumentParser(description="Exp23b: prepare NSD data for Brain-Diffuser")
parser.add_argument("--nsd_dir", type=str, required=True, help="NSD の AWS から落とした場所（nsddata、nsddata_betas、nsddata_stimuli がある）")
parser.add_argument("--annots", type=str, required=True, help="COCO_73k_annots_curated.npy")
parser.add_argument("--exp20_meta", type=str, required=True, help="Exp20 の meta.npz（刺激の番号と COCO の番号）")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7])
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
out = lambda name: os.path.join(args.out_dir, name)
sub = args.subj

# ── 学習とテストの画像（元のコードと同じ：37 セッション、masterordering ≤ 1000 がテスト）──
d = spio.loadmat(f"{args.nsd_dir}/nsddata/experiments/nsd/nsd_expdesign.mat", squeeze_me=True)
num_trials = 37 * 750
sig_train, sig_test = {}, {}                                        # 画像 → 試行（dict の順 = 初めて出た順。元のコードと同じ）
for idx in range(num_trials):
    nsd_id = int(d["subjectim"][sub - 1, d["masterordering"][idx] - 1] - 1)
    (sig_train if d["masterordering"][idx] > 1000 else sig_test).setdefault(nsd_id, []).append(idx)
train_ids, test_ids = list(sig_train), list(sig_test)
print(f"学習 {len(train_ids)} 枚、テスト {len(test_ids)} 枚")

# ── fMRI（int16 のまま読んでマスクする。get_fdata().astype(float32) と同じ値）──
roi_dir = f"{args.nsd_dir}/nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi"
betas_dir = f"{args.nsd_dir}/nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
mask = nib.load(f"{roi_dir}/nsdgeneral.nii.gz").get_fdata() > 0
num_voxel = int(mask.sum())
fmri = np.zeros((num_trials, num_voxel), np.float32)
for i in range(37):
    img = nib.load(f"{betas_dir}/betas_session{i + 1:02d}.nii.gz")
    assert img.dataobj.slope == 1 and img.dataobj.inter == 0, "betas に scl_slope か scl_inter がある"
    beta = np.asarray(img.dataobj)
    assert beta.dtype == np.int16 and beta.shape[-1] == 750, (beta.dtype, beta.shape)
    fmri[i * 750:(i + 1) * 750] = beta[mask].T
    del beta
    print(f"セッション {i + 1} / 37", flush=True)

# ── 平均した fMRI（元のコードと同じく、float32 の試行を平均する）──
np.save(out("train_fmri.npy"), np.stack([fmri[sorted(sig_train[k])].mean(0) for k in train_ids]))
np.save(out("test_fmri_avg.npy"), np.stack([fmri[sorted(sig_test[k])].mean(0) for k in test_ids]))

# ── 単一試行のテストデータ（画像は元のコードの順、試行は時間の順）──
rows = [(j, t, r, len(sig_test[k])) for j, k in enumerate(test_ids) for r, t in enumerate(sorted(sig_test[k]))]
img_idx, trial, rep, n_reps = (np.array(c) for c in zip(*rows))
np.savez(out("test_trials.npz"), fmri=fmri[trial], img=img_idx, trial=trial, rep=rep, n_reps=n_reps)
print(f"単一試行のテスト {len(rows)} 行（3 試行そろう画像 {int((n_reps[rep == 0] == 3).sum())} 枚）")
del fmri

# ── 刺激（uint8。hdf5 から使う画像だけ、番号の順に読んでから並べ直す）──
with h5py.File(f"{args.nsd_dir}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5", "r") as f:
    brick = f["imgBrick"]
    for name, ids in (("train_stim.npy", train_ids), ("test_stim.npy", test_ids)):
        arr = np.lib.format.open_memmap(out(name), "w+", np.uint8, (len(ids), 425, 425, 3))
        order = np.argsort(ids)
        for s in range(0, len(ids), 500):
            pos = order[s:s + 500]
            arr[pos] = brick[np.asarray(ids)[pos]]
        arr.flush()
        del arr
        print(name, "保存", flush=True)

# ── キャプション（元のコードと同じ）──
annots = np.load(args.annots)
np.save(out("train_cap.npy"), annots[train_ids])
np.save(out("test_cap.npy"), annots[test_ids])

# ── Exp20 の刺激の番号 ──
m20 = np.load(args.exp20_meta)
k_of = dict(zip(m20["coco73k"].tolist(), m20["stim"].tolist()))
test_stim20 = np.array([k_of.get(k, -1) for k in test_ids])
np.savez(out("meta.npz"), train_ids=np.array(train_ids), test_ids=np.array(test_ids), test_stim20=test_stim20,
         num_voxel=num_voxel, subj=sub)
print(f"テスト {len(test_ids)} 枚のうち Exp20 にある {int((test_stim20 >= 0).sum())} 枚")
print("終了")
