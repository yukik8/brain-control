"""
exp23_bd_regression.py
----------------------
Experiment 23b: Brain-Diffuser の ridge 回帰（fMRI → VDVAE、CLIP-Vision、CLIP-Text）と、テストの予測

元のコード（scripts/vdvae_regression.py、clipvision_regression.py、cliptext_regression.py）と同じ回帰。
bd_env の python で実行する。入力は exp23_bd_prepare.py と exp23_bd_features.py の出力。

計算のしかた：sklearn の Ridge(alpha, fit_intercept=True) は、ボクセル数（15,724）が学習の数（8,859）より多いとき、
中心化した X と y で dual = (X Xᵀ + αI)⁻¹ y を解き、coef = Xᵀ dual とする（cholesky）。ここでは同じ式を
pred = (X_test − x̄)(X − x̄)ᵀ (X Xᵀ + αI)⁻¹ (y − ȳ) + ȳ と書き、重み（CLIP-Vision で 12 GB）を作らずに予測だけを出す。
CLIP は元のコードではトークンごとに 257 回（77 回）回帰するが、X と α が同じなので、まとめて解いても同じ。
--check で、一部の出力について sklearn の Ridge と予測が一致することを確かめる。

予測の標準化（元のコードと同じ）：次元ごとに、テストの予測全体の平均・標準偏差で標準化し、学習の特徴量の
平均・標準偏差に戻す。テストの予測全体として、
  - avg：元のコードと同じく、3 試行（1〜37 セッションにある分）を平均したテスト 982 枚（再現の確認 G2 用）
  - trial：平均しない単一試行のテスト全部（982 枚の、1〜37 セッションのすべての試行）
のそれぞれで別々に行う。

出力（--data_dir）：pred_{feature}_avg.npy、pred_{feature}_trial.npy（float32）、regression_{feature}.json（R² など）
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.linalg

parser = argparse.ArgumentParser(description="Exp23b: Brain-Diffuser ridge regressions")
parser.add_argument("--feature", type=str, required=True, choices=["vdvae", "clipvision", "cliptext"])
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--chunk", type=int, default=12288, help="一度に解く出力の次元の数（メモリにだけ効く）")
parser.add_argument("--check", action="store_true", help="最初の 768 次元で sklearn の Ridge と比べる")
args = parser.parse_args()
path = lambda name: os.path.join(args.data_dir, name)
ALPHA = {"vdvae": 50000, "clipvision": 60000, "cliptext": 100000}[args.feature]   # 元のコードと同じ
if os.path.exists(path(f"regression_{args.feature}.json")):
    print(f"{args.feature} は済んでいる")
    sys.exit(0)
t0 = time.time()

# ── fMRI の前処理（元のコードと同じ：300 で割り、学習の平均・標準偏差（ddof=1）で標準化）──
train_fmri = np.load(path("train_fmri.npy")).astype(np.float64) / 300
norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
tests = {"avg": np.load(path("test_fmri_avg.npy")),
         "trial": np.load(path("test_trials.npz"))["fmri"]}
tests = {k: (v.astype(np.float64) / 300 - norm_mean_train) / norm_scale_train for k, v in tests.items()}
for k, v in tests.items():
    print(f"test {k}: {v.shape}、平均 {v.mean():.3f}、標準偏差 {v.std():.3f}")

# ── (X Xᵀ + αI)⁻¹ を使う形にしておく ──
x_mean = train_fmri.mean(0)
Xc = train_fmri - x_mean
K = Xc @ Xc.T
K[np.diag_indices_from(K)] += ALPHA
cho = scipy.linalg.cho_factor(K, lower=False, overwrite_a=True, check_finite=False)
M = {k: scipy.linalg.cho_solve(cho, Xc @ (v - x_mean).T, check_finite=False).T for k, v in tests.items()}  # (n_test, n_train)
del K
print(f"カーネルの準備 {time.time() - t0:.0f} 秒", flush=True)

# ── 特徴量ごとに予測する。標準化も R² も次元ごとの計算なので、--chunk 次元ずつ済ませて書き出す ──
Y = np.load(path(f"{args.feature}_train.npy"), mmap_mode="r")
Y_test = np.load(path(f"{args.feature}_test.npy"), mmap_mode="r")
shape = Y.shape[1:]
Y, Y_test = Y.reshape(len(Y), -1), Y_test.reshape(len(Y_test), -1)
D = Y.shape[1]
outs = {k: np.lib.format.open_memmap(path(f"pred_{args.feature}_{k}.npy.part"), "w+", np.float32, (len(m),) + shape)
        for k, m in M.items()}
info = dict(feature=args.feature, alpha=ALPHA, n_train=len(Y), n_voxel=train_fmri.shape[1], dim=D,
            n_test={k: len(m) for k, m in M.items()})
r2_dims = np.zeros(D)


def r2(y_true, y_pred):
    """sklearn の r2_score（次元ごと、force_finite=True）と同じ。"""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    out = np.where(ss_res == 0, 1.0, 0.0)
    ok = ss_tot > 0
    out[ok] = 1 - ss_res[ok] / ss_tot[ok]
    return out


for s in range(0, D, args.chunk):
    e = min(s + args.chunk, D)
    y = np.asarray(Y[:, s:e], dtype=np.float64)
    y_mean, y_std = y.mean(0), y.std(0)
    raw = {k: m @ (y - y_mean) + y_mean for k, m in M.items()}
    if args.check and s == 0:                                      # 最初の 768 次元で、sklearn と同じ予測になるか
        import sklearn.linear_model as skl
        max_iter = {"vdvae": 10000, "clipvision": 50000, "cliptext": 50000}[args.feature]
        reg = skl.Ridge(alpha=ALPHA, max_iter=max_iter, fit_intercept=True)
        reg.fit(train_fmri, y[:, :768])
        diff = {k: float(np.abs(reg.predict(tests[k]) - raw[k][:, :768]).max()) for k in raw}
        scale = float(np.abs(raw["avg"][:, :768]).max())
        print("sklearn との差（最大）：", diff, "予測の大きさ（最大）：", scale)
        assert max(diff.values()) < 1e-6 * max(scale, 1), "sklearn の Ridge と予測が違う"
        info["sklearn_check_max_abs_diff"] = diff
    r2_dims[s:e] = r2(np.asarray(Y_test[:, s:e], dtype=np.float64), raw["avg"])   # 元のコードの reg.score（平均のテスト）
    for k, p in raw.items():                                       # 予測の標準化（元のコードと同じ。np.std は ddof=0）
        p = (p - p.mean(0)) / p.std(0) * y_std + y_mean
        outs[k].reshape(len(p), D)[:, s:e] = p
    print(f"{e} / {D} 次元（{time.time() - t0:.0f} 秒）", flush=True)

for k, o in outs.items():
    o.flush()
    del o
    os.replace(path(f"pred_{args.feature}_{k}.npy.part"), path(f"pred_{args.feature}_{k}.npy"))
info["r2_avg_test"] = float(r2_dims.mean())                        # 次元ごとの R² の平均（CLIP はトークンごとにも）
if len(shape) == 2:
    info["r2_avg_test_per_token"] = r2_dims.reshape(shape).mean(1).round(4).tolist()
print(f"R²（平均のテスト）：{info['r2_avg_test']:.4f}")
info["seconds"] = round(time.time() - t0)
with open(path(f"regression_{args.feature}.json"), "w") as f:
    json.dump(info, f, ensure_ascii=False, indent=1)
print("終了", info["seconds"], "秒")
