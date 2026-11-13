"""
exp20_mindeye2_pertrial.py
--------------------------
Experiment 20: NSD × MindEye2 での試行ごとの推論（S0〜S3 の出力を保存する）

MindEye2（Scotti et al., 2024）の src/recon_inference.ipynb を元に、同じ画像の 3 試行を平均せずに、
試行ごとに再構成を作る。パイプラインの各段階の出力をディスクに保存し、あとで段階ごとに BC を測る
（experiments/exp20_mindeye2_bc.py）。

  - パス A（--stage prior） : 脳活動 → ridge（S0）→ backbone（S1）→ 拡散 prior（S2）。1 回の実行で 1 モデル
  - パス B（--stage unclip）: S2 → SDXL unCLIP → 256×256 の画像（S3）。刺激ごとに 40sess → 1sess の順に生成

刺激は seed 42 の無作為な順で処理する。どこで止めても、先頭から K 枚の刺激が 1,000 枚からの無作為な標本になる。
行ごとに乱数を固定し、済んだ行を done_*.npy に記録するので、止めて同じコマンドで再開しても、
止めずに通したときと同じ結果になる。STOP ファイル（--out_dirs の親ディレクトリ）か --stop_at の時刻で止まる。

実行のしかた：このファイルを $EXP20/MindEyeV2/src にコピーし、そこで fmri 環境の python で実行する
（models.py、utils.py、generative_models を import するため）。ふつうは experiments/exp20_run.sh から実行する。

MPS 用に元のコードから変えた点：
  - Accelerator を使わず device = mps にする。utils.device も mps にする
  - torch.cuda.amp.autocast の代わりに torch.autocast("mps", dtype=torch.float16) を使う
  - unCLIP の attention を xformers から softmax（UNet）と vanilla（VAE）にし、サンプラーの device を mps にする
  - unCLIP のサンプリングは unclip_recon_mps を使う（VAE のデコードは autocast の外で fp32）
  - チェックポイントの読み込みの try/except（失敗すると strict=False で黙って進む）を外し、strict=True にする

そのほかの変更：
  - 3 試行そろわない画像は、元のコードのように試行を複製せず、除外する
  - キャプション（GIT）、ぼかし再構成、clip_img_embedder は使わない（--blurry_recon は無視する）
"""

import os
import sys
import argparse
import numpy as np
import time
import h5py
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn
from torchvision import transforms

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# custom functions #
import utils
from models import *

device = torch.device("mps")
utils.device = device                                        # utils は読み込み時に cuda か cpu を決めるため
print("device:", device)

parser = argparse.ArgumentParser(description="Exp20: per-trial MindEye2 inference")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
    help="exp20 では使わない（元のコードと同じ引数を受け付けるためだけに残す）",
)
parser.add_argument(
    "--n_blocks", type=int, default=4,
)
parser.add_argument(
    "--hidden_dim", type=int, default=2048,
)
parser.add_argument(
    "--new_test", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--seed", type=int, default=42,
)
# ── exp20 で追加した引数 ──
parser.add_argument(
    "--stage", type=str, required=True, choices=["prior", "unclip"],
    help="prior = パス A（S0〜S2）、unclip = パス B（S3）",
)
parser.add_argument(
    "--out_dirs", type=str, nargs="+", required=True,
    help="出力先。パス A は 1 つ、パス B は 40sess、1sess の順に 2 つ（再現確認では 1 つ）",
)
parser.add_argument(
    "--limit", type=int, default=0,
    help="この実行で処理する数の上限（パス A は行、パス B は刺激）。0 なら上限なし",
)
parser.add_argument(
    "--stop_at", type=str, default=None,
    help="この時刻になったら止める（例：2026-11-10T08:00）",
)
parser.add_argument(
    "--average_reps", action="store_true",
    help="再現確認用：元のコードと同じく 3 試行を平均し、画像 1 枚につき 1 行にする",
)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

import fcntl, json, subprocess
from datetime import datetime

# ── 途中で止めても大丈夫にするための部品 ──
out_dirs = [os.path.abspath(d) for d in out_dirs]
out_root = os.path.dirname(out_dirs[0])                      # ふつうは $EXP20/out
stop_file = f"{out_root}/STOP"
stop_at = datetime.fromisoformat(args.stop_at) if args.stop_at else None


def stop_reason():
    """止める理由があれば返す（STOP ファイルか --stop_at の時刻）。"""
    if os.path.exists(stop_file):
        return "STOP ファイルがある"
    if stop_at is not None and datetime.now() >= stop_at:
        return f"--stop_at（{stop_at}）を過ぎた"
    return None


if stop_reason():                                            # 0 以外で終わるので、exp20_run.sh も次の段階に進まない
    sys.exit(f"始めない：{stop_reason()}（STOP ファイルなら、消してからもう一度実行する）")
lock = open(f"{out_root}/.lock", "w")                       # 同時に 2 つ動かさない（モデルを読む前に確かめる）
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)         # プロセスが終われば（kill されても）自動で外れる
except BlockingIOError:
    sys.exit("この実験のプロセスがもう動いている（exp20_status.py で確認する）")


def open_out(out_dir, name, dtype, shape):
    """出力用の配列をディスク上に置く。あれば開き直し（再開）、なければ作る。"""
    path = f"{out_dir}/{name}"
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, "r+")
        assert a.shape == shape and a.dtype == dtype, f"{path} の形が違う（別の設定の出力が残っている）"
        return a
    return np.lib.format.open_memmap(path, "w+", dtype, shape)


def commit(arrays, done, pending):
    """データを先にディスクへ書き、そのあとで done を立てて書く。

    この順番なので、電源が落ちても「done なのにデータがない」行はできない（その行はやり直しになるだけ）。
    """
    if not pending:
        return
    for a in arrays:
        a.flush()
    done[pending] = True
    done.flush()
    pending.clear()


def reset_empty(done, arrays):
    """念のための確認：done なのに中身が全部 0 の行があれば、done を外してやり直す。外した行の数を返す。"""
    idx = np.flatnonzero(done)
    bad = np.zeros(len(idx), bool)
    for s in range(0, len(idx), 100):
        r = idx[s:s + 100]
        for a in arrays:
            bad[s:s + 100] |= ~np.asarray(a[r]).reshape(len(r), -1).any(axis=1)
    done[idx[bad]] = False
    done.flush()
    return int(bad.sum())


def log_run(stage, t_start, times, peak, reason):
    """正常に終わった実行を $EXP20/out/runs.jsonl に 1 行足す（kill で止めた回は残らない）。"""
    info = dict(stage=stage, dirs=[os.path.basename(d) for d in out_dirs],
                start=t_start.isoformat(timespec="seconds"),
                end=datetime.now().isoformat(timespec="seconds"),
                n_trials=len(times), sec_per_trial=float(np.mean(times)) if times else None,
                peak_mps_gb=round(peak / 2**30, 1), stop_reason=reason or "全部済んだ",
                chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip(),
                torch=torch.__version__)
    with open(f"{out_root}/runs.jsonl", "a") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")


clip_seq_dim = 256
clip_emb_dim = 1664


# ═════════════════════════════════════════════════════════════════════════
# パス A（S0〜S2）
# ═════════════════════════════════════════════════════════════════════════
if stage == "prior":
    voxels = {}
    # Load hdf5 data for betas
    f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu")
    num_voxels = betas[0].shape[-1]
    voxels[f'subj0{subj}'] = betas
    print(f"num_voxels for subj0{subj}: {num_voxels}")

    if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
        if subj==3:
            num_test=2113
        elif subj==4:
            num_test=1985
        elif subj==6:
            num_test=2113
        elif subj==8:
            num_test=1985
        else:
            num_test=2770
        test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
    else: # using larger test set from after full dataset released
        if subj==3:
            num_test=2371
        elif subj==4:
            num_test=2188
        elif subj==6:
            num_test=2371
        elif subj==8:
            num_test=2188
        else:
            num_test=3000
        test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"

    print(test_url)
    def my_split_by_node(urls): return urls
    test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=False)
    print(f"Loaded test dl for subj{subj}!\n")

    # Prep test voxels and indices of test images
    test_images_idx = []
    for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
        test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
        test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
    test_images_idx = test_images_idx.astype(int)

    assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
    print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))

    # 手順書 §6 の確認 3（データの構造）
    global_trial = behav[:, 0, 5].cpu().numpy()
    n_per_img = np.unique(test_images_idx, return_counts=True)[1]
    reps, n_imgs = np.unique(n_per_img, return_counts=True)
    print(f"確認 3：num_test = {num_test}、画像 {len(n_per_img)} 枚、"
          f"画像ごとの試行数の分布 {dict(zip(reps.tolist(), n_imgs.tolist()))}、"
          f"global_trial の重複 {len(global_trial) - len(np.unique(global_trial))}")

    # ── 試行の並び ──
    uniq_imgs = np.unique(test_images_idx)                      # 1000 枚（昇順 = all_images.pt と同じ順）
    groups = []                                                  # (刺激の番号 k, その刺激の test_voxels の行)
    for k, img in enumerate(uniq_imgs):
        locs = [int(l) for l in np.where(test_images_idx == img)[0]]
        if len(locs) == 3:                                       # そろわない画像は複製せずに除外（§6 の確認 3）
            groups.append((k, locs))
    n_excluded_imgs = len(uniq_imgs) - len(groups)
    n_excluded_trials = len(test_images_idx) - 3 * len(groups)
    order = np.random.default_rng(42).permutation(len(groups))   # 刺激を処理する順（§2.6）。どこで止めても無作為な標本になる
    rows = []                                                    # (刺激の番号 k, test_voxels の行のタプル)
    for g in order:
        k, locs = groups[g]
        if average_reps:                                         # §6 の再現確認用：元のコードと同じ 3 試行平均
            rows.append((k, tuple(locs)))
        else:
            rows.extend((k, (l,)) for l in locs)                 # 同じ刺激の 3 試行は連続する
    N = len(rows)
    stim = np.array([k for k, _ in rows])
    locs_arr = np.array([l for _, l in rows])                    # (N, 1)、平均のときは (N, 3)
    print(f"除外した画像 {n_excluded_imgs} 枚（{n_excluded_trials} 試行）、C = {len(groups)}、N = {N}")

    assert len(out_dirs) == 1, "パス A の出力先は 1 つ"
    out_dir = out_dirs[0]
    os.makedirs(out_dir, exist_ok=True)
    meta_path = f"{out_dir}/meta.npz"
    if os.path.exists(meta_path):                                # 再開するときは同じ試行の並びであることを確かめる
        old = np.load(meta_path)
        assert np.array_equal(old["stim"], stim) and np.array_equal(old["locs"], locs_arr), "前回と試行の並びが違う"
    else:
        np.savez(meta_path,
                 stim=stim,                                      # 刺激の番号（0〜999、all_images.pt の行）
                 coco73k=uniq_imgs[stim].astype(int),            # 73k 内の画像番号
                 locs=locs_arr,                                  # test_voxels の行
                 behav=behav.cpu().numpy(),                      # テスト試行の behav 配列
                 n_excluded_imgs=n_excluded_imgs, n_excluded_trials=n_excluded_trials,
                 model_name=model_name, subj=subj, seed=seed, average_reps=average_reps)

    s0 = open_out(out_dir, "s0_ridge.npy",    np.float16, (N, hidden_dim))
    s1 = open_out(out_dir, "s1_backbone.npy", np.float16, (N, clip_seq_dim, clip_emb_dim))
    s2 = open_out(out_dir, "s2_prior.npy",    np.float16, (N, clip_seq_dim, clip_emb_dim))
    tm = open_out(out_dir, "times_prior.npy", np.float32, (N,))
    done = open_out(out_dir, "done_prior.npy", np.bool_, (N,))
    print(f"done なのに空の行：{reset_empty(done, [s0, s1, s2])}（ふつうは 0）")
    print(f"パス A：{int(done.sum())} / {N} 行が済んでいる")
    if done.all():                                               # モデルを読まずに、正常終了（0）で終わる
        print("パス A は全部済んでいる")
        sys.exit(0)

    # ── MindEye2 の構築と読み込み ──
    class MindEyeModule(nn.Module):
        def __init__(self):
            super(MindEyeModule, self).__init__()
        def forward(self, x):
            return x

    model = MindEyeModule()

    class RidgeRegression(torch.nn.Module):
        # make sure to add weight_decay when initializing optimizer to enable regularization
        def __init__(self, input_sizes, out_features):
            super(RidgeRegression, self).__init__()
            self.out_features = out_features
            self.linears = torch.nn.ModuleList([
                    torch.nn.Linear(input_size, out_features) for input_size in input_sizes
                ])
        def forward(self, x, subj_idx):
            out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
            return out

    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

    from diffusers.models.vae import Decoder
    from models import BrainNetwork
    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1,
                              clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim)
    utils.count_params(model.ridge)
    utils.count_params(model.backbone)
    utils.count_params(model)

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
    timesteps = 100

    prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    model.to(device)

    utils.count_params(model.diffusion_prior)
    utils.count_params(model)

    # Load pretrained model ckpt（失敗を隠さないよう、try/except を外して strict=True だけにする）
    tag='last'
    outdir = os.path.abspath(f'../train_logs/{model_name}')
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    ckpt_hidden_dim = state_dict['ridge.linears.0.weight'].shape[0]
    assert ckpt_hidden_dim == hidden_dim, f"チェックポイントの hidden_dim は {ckpt_hidden_dim}（--hidden_dim={hidden_dim}）"
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    print("ckpt loaded!")

    # ── ループ（元のコードの 3 試行平均のループを置き換える）──
    model.to(device)
    model.eval().requires_grad_(False)
    pending, times, peak, reason = [], [], 0, None
    t_start = datetime.now()
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
        for t, (k, locs) in enumerate(tqdm(rows)):
            if done[t]:
                continue
            reason = stop_reason() or ("--limit" if limit and len(times) >= limit else None)
            if reason:
                break
            t0 = time.time()
            torch.manual_seed(seed + t)                          # 行ごとに固定の乱数（止めて再開しても同じ結果）
            voxel = test_voxels[None, list(locs)].to(device)     # (1, 試行数, num_voxels)
            assert voxel.shape == (1, len(locs), num_voxels)
            vr = [model.ridge(voxel[:, [r]], 0) for r in range(len(locs))]
            bb = [model.backbone(v)[0] for v in vr]              # backbone の戻り値は (backbone, clip_voxels, blurry)
            voxel_ridge, backbone = sum(vr) / len(vr), sum(bb) / len(bb)
            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape, text_cond=dict(text_embed=backbone),
                cond_scale=1., timesteps=20)
            if not times:                                        # この実行の最初の行だけ形を表示する（§5.3）
                print("shapes:", tuple(voxel.shape), tuple(voxel_ridge.shape), tuple(backbone.shape), tuple(prior_out.shape),
                      voxel_ridge.dtype, backbone.dtype, prior_out.dtype)
            out = [x.float().cpu().numpy().astype(np.float16)
                   for x in (voxel_ridge[0, 0], backbone[0], prior_out[0])]
            assert all(np.isfinite(x).all() for x in out), f"行 {t} に inf か nan（float16 であふれた可能性。§7）"
            s0[t], s1[t], s2[t] = out
            tm[t] = time.time() - t0
            times.append(float(tm[t]))
            pending.append(t)
            peak = max(peak, torch.mps.driver_allocated_memory())
            if len(pending) >= 20:                               # 20 行ごとに保存（kill で失うのは最大 20 行）
                commit([s0, s1, s2, tm], done, pending)
    commit([s0, s1, s2, tm], done, pending)
    log_run("prior", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")


# ═════════════════════════════════════════════════════════════════════════
# パス B（S3）
# ═════════════════════════════════════════════════════════════════════════
if stage == "unclip":

    def unclip_recon_mps(x, diffusion_engine, vector_suffix, offset_noise_level=0.04):
        """utils.unclip_recon の MPS 版（num_samples=1）。

        元の関数との違いは 2 点だけ：
        - CUDA の autocast の代わりに MPS の autocast（fp16）でサンプリングする
        - VAE のデコードは autocast の外で fp32 で行う（元のコードも disable_first_stage_autocast=True で
          VAE を fp32 にしている。その仕組みは MPS の autocast を止めないため、外に出す）
        """
        with torch.no_grad(), diffusion_engine.ema_scope():
            with torch.autocast("mps", dtype=torch.float16):
                z = torch.randn(1, 4, 96, 96).to(device)                   # 初期ノイズ
                c = {"crossattn": x, "vector": vector_suffix}
                uc = {"crossattn": torch.randn_like(x), "vector": vector_suffix}  # 条件なし
                noise = torch.randn_like(z)
                sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
                sigma = sigmas[0].to(z.device)
                noise = noise + offset_noise_level * append_dims(
                    torch.randn(z.shape[0], device=z.device), z.ndim)
                noised_z = z + noise * append_dims(sigma, z.ndim)
                noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                def denoiser(x, sigma, c):
                    return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

                samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
            samples_x = diffusion_engine.decode_first_stage(samples_z.float())   # fp32
            return torch.clamp(samples_x * .8 + .2, min=0.0, max=1.0)

    # ── 刺激の並び（unCLIP を構築する前に確かめる。済んでいれば 18 GB を読まずに終わる）──
    metas = [np.load(f"{d}/meta.npz") for d in out_dirs]
    stim = metas[0]["stim"]
    for m in metas[1:]:                                          # 両モデルで同じ試行の並びであること
        assert np.array_equal(m["stim"], stim) and np.array_equal(m["locs"], metas[0]["locs"]), "モデルの間で試行の並びが違う"
    N = len(stim)
    starts = np.r_[0, np.flatnonzero(np.diff(stim)) + 1]         # 同じ刺激の行は連続している（パス A）
    groups = np.split(np.arange(N), starts[1:])                  # 刺激ごとの行（パス A と同じ無作為な順）

    done_a = [np.load(f"{d}/done_prior.npy", mmap_mode="r") for d in out_dirs]
    s2 = [np.load(f"{d}/s2_prior.npy", mmap_mode="r") for d in out_dirs]
    im = [open_out(d, "s3_images.npy", np.uint8, (N, 3, 256, 256)) for d in out_dirs]
    tm = [open_out(d, "times_unclip.npy", np.float32, (N,)) for d in out_dirs]
    done = [open_out(d, "done_unclip.npy", np.bool_, (N,)) for d in out_dirs]
    for i, d in enumerate(out_dirs):
        print(f"{d}: done なのに空の行 {reset_empty(done[i], [im[i]])}（ふつうは 0）、済んだ行 {int(done[i].sum())} / {N}")
    todo = [g for g in groups if not all(dn[g].all() for dn in done)]
    if not todo:
        print("パス B は全部済んでいる")
        sys.exit(0)
    if not all(da[todo[0]].all() for da in done_a):
        sys.exit("次の刺激のパス A が済んでいない。先にパス A を行う")

    # ── unCLIP の構築と読み込み ──
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
    offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

    network_config["params"]["spatial_transformer_attn_type"] = "softmax"     # xformers を使わない
    first_stage_config["params"]["ddconfig"]["attn_type"] = "vanilla"         # 同上（VAE）
    first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
    sampler_config['params']['num_steps'] = 38

    diffusion_engine = DiffusionEngine(network_config=network_config,
                           denoiser_config=denoiser_config,
                           first_stage_config=first_stage_config,
                           conditioner_config=conditioner_config,
                           sampler_config=sampler_config,
                           scale_factor=scale_factor,
                           disable_first_stage_autocast=disable_first_stage_autocast)
    # set to inference
    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)  # mmap：18 GB を一度にメモリへ読まない
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    del ckpt
    diffusion_engine.sampler.device = "mps"                                    # 既定の "cuda" を置き換える

    batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
          "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
          "crop_coords_top_left": torch.zeros(1, 2).to(device)}
    vector_suffix = diffusion_engine.conditioner(batch)["vector"].to(device)
    print("vector_suffix", vector_suffix.shape)
    del diffusion_engine.conditioner                                           # ViT-bigG（約 7 GB）。以後使わない
    import gc; gc.collect(); torch.mps.empty_cache()

    # ── ループ（元のコードの 3 試行平均のループを置き換える）──
    pending, times, peak, reason, n_stim = [], [], 0, None, 0
    t_start = datetime.now()
    for g in tqdm(groups):                                       # 刺激ごと
        if all(dn[g].all() for dn in done):
            continue
        if not all(da[g].all() for da in done_a):
            reason = "パス A が済んでいない刺激に来た"
            break
        if limit and n_stim >= limit:
            reason = "--limit"
            break
        for i in range(len(out_dirs)):                           # 同じ刺激を、40sess → 1sess の順に続けて生成する
            for t in g:
                if done[i][t]:
                    continue
                reason = stop_reason()
                if reason:
                    break
                t0 = time.time()
                torch.manual_seed(seed + 1_000_000 + int(t))     # パス A の行で決まる乱数（両モデルで同じ。再開しても同じ）
                x = torch.from_numpy(np.asarray(s2[i][t], dtype=np.float32))[None].to(device)
                img = unclip_recon_mps(x, diffusion_engine, vector_suffix, offset_noise_level)
                img = transforms.Resize((256, 256), antialias=False)(img.float()).clamp(0, 1)  # 元のコード（torchvision 0.16）と同じ縮め方
                assert torch.isfinite(img).all(), f"{out_dirs[i]} の行 {t} の画像に nan（VAE が fp16 で動いた可能性。§5.4）"
                im[i][t] = (img[0].cpu().numpy() * 255).round().astype(np.uint8)
                tm[i][t] = time.time() - t0
                times.append(float(tm[i][t]))
                pending.append(t)
                peak = max(peak, torch.mps.driver_allocated_memory())
            commit([im[i], tm[i]], done[i], pending)             # モデルごと・刺激ごとに保存（kill で失うのは最大 3 試行）
            if reason:
                break
        if reason:
            break
        n_stim += 1
    log_run("unclip", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")
