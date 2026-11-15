"""
exp23_bd_pertrial.py
--------------------
Experiment 23b: NSD × Brain-Diffuser での試行ごとの再構成（各段階の出力を保存する）

Brain-Diffuser（Ozcelik & VanRullen, 2023）の scripts/vdvae_reconstruct_images.py と
versatilediffusion_reconstruct_images.py を元に、単一試行の予測（exp23_bd_regression.py の pred_*_trial.npy）から
試行ごとに再構成を作る。Exp20（experiments/exp20_mindeye2_pertrial.py）、Exp23a と同じ作りにしてある。

  - S1：回帰の予測（VDVAE の潜在表現、CLIP-Vision 257 × 768、CLIP-Text 77 × 768）。決定的なので、ここでは保存せず
        bd_data の pred_*.npy を meta.npz の pred_row で引く
  - パス A（--stage vdvae）: VDVAE の予測 → VDVAE の画像（S2、64 × 64）。予測にない深い層の潜在表現と、出力の画素は
                             乱数で引く
  - パス B（--stage vd）    : S2 を 512 × 512 にして img2img の初期値にし、CLIP-Vision と CLIP-Text の予測で条件づけた
                             Versatile Diffusion で画像を作る（S3）

刺激は Exp20 と同じ順（Exp20 の meta.npz の、seed 42 の無作為な順）で、1〜37 セッションに 3 試行そろう刺激だけ
（Exp23a と同じ 839 刺激）。行ごとに乱数を固定し（パス A は seed + t、パス B は seed + 1,000,000 + t）、済んだ行を
done_*.npy に記録するので、止めて再開しても同じ結果になる。STOP ファイル（--out_dir の親ディレクトリ）か
--stop_at の時刻で止まる。

実行のしかた：Brain-Diffuser のリポジトリの一番上で、bd_env の python で実行する（元のコードと同じく、vdvae と
versatile_diffusion をそこからの相対パスで読むため）。

元のコードから変えた点：
  - device = mps（元のコードは cuda:0 と cuda:1 の 2 枚）。VD の DDIM の register_buffer は cuda を直書きしているので
    device に置き換える（MPS には float64 がないので float32 にする。使うのは eta = 0 の σ と、もともと float32 の値）
  - versatile_diffusion/lib/sync.py の get_rank・get_world_size を 1 台・1 プロセスの値にする（exp23_bd_features.py と同じ）
  - 乱数：元のコードは最初に torch.manual_seed(0) を 1 回だけ（VDVAE は固定しない）→ 行ごとに固定する
  - VDVAE の画像は PNG に保存せず、64 × 64 の uint8 のまま持つ。パス B で、元のコードと同じく PIL の bicubic で
    512 × 512 にする（PNG は可逆なので同じ）
  - VD のチェックポイントは元のコードと同じく strict=False で読み、読めなかった重みと余った重みを表示して確かめる
  - 無条件の埋め込み（空の文字列と、0 の画像）は行ごとに同じ値なので、最初に 1 回だけ計算する
"""

import os
import sys
import argparse
import time
import fcntl
import json
import subprocess
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as tvtrans

parser = argparse.ArgumentParser(description="Exp23b: per-trial Brain-Diffuser reconstruction")
parser.add_argument("--data_dir", type=str, required=True, help="exp23_bd_prepare.py、_features.py、_regression.py の出力")
parser.add_argument("--exp20_meta", type=str, required=True, help="Exp20 の meta.npz（刺激の順）")
parser.add_argument("--vd_pth", type=str, help="vd-four-flow-v1-0-fp16-deprecated.pth（パス B）")
parser.add_argument("--vdvae_dir", type=str, help="VDVAE の重みの場所（パス A）")
parser.add_argument("--stage", type=str, required=True, choices=["vdvae", "vd"],
                    help="vdvae = パス A（S2）、vd = パス B（S3）")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--diff_str", type=float, default=0.75, help="元のコードの既定値")
parser.add_argument("--mix_str", type=float, default=0.4, help="元のコードの既定値")
parser.add_argument("--limit", type=int, default=0, help="この実行で処理する数の上限（パス A は行、パス B は刺激）。0 なら上限なし")
parser.add_argument("--stop_at", type=str, default=None, help="この時刻になったら止める（例：2026-11-10T08:00）")
parser.add_argument("--average_reps", action="store_true", help="再現確認用：元のコードと同じく 3 試行を平均した予測を使い、画像 1 枚につき 1 行にする")
parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"],
                    help="cpu は動作確認用。乱数の列が変わるので、本番の出力には使わない")
args = parser.parse_args()
device = torch.device(args.device)
seed, limit = args.seed, args.limit
data = lambda name: os.path.join(args.data_dir, name)
pred_kind = "avg" if args.average_reps else "trial"
print("device:", device)

# ── 途中で止めても大丈夫にするための部品（Exp20 と同じ）──
out_dir = os.path.abspath(args.out_dir)
out_root = os.path.dirname(out_dir)
stop_file = f"{out_root}/STOP"
stop_at = datetime.fromisoformat(args.stop_at) if args.stop_at else None


def stop_reason():
    if os.path.exists(stop_file):
        return "STOP ファイルがある"
    if stop_at is not None and datetime.now() >= stop_at:
        return f"--stop_at（{stop_at}）を過ぎた"
    return None


if stop_reason():
    sys.exit(f"始めない：{stop_reason()}（STOP ファイルなら、消してからもう一度実行する）")
lock = open(f"{out_root}/.lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit("この実験のプロセスがもう動いている")


def open_out(name, dtype, shape):
    path = f"{out_dir}/{name}"
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, "r+")
        assert a.shape == shape and a.dtype == dtype, f"{path} の形が違う（別の設定の出力が残っている）"
        return a
    return np.lib.format.open_memmap(path, "w+", dtype, shape)


def commit(arrays, done, pending):
    """データを先にディスクへ書き、そのあとで done を立てて書く。"""
    if not pending:
        return
    for a in arrays:
        a.flush()
    done[pending] = True
    done.flush()
    pending.clear()


def reset_empty(done, arrays):
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
    info = dict(stage=stage, dirs=[os.path.basename(out_dir)],
                start=t_start.isoformat(timespec="seconds"), end=datetime.now().isoformat(timespec="seconds"),
                n_trials=len(times), sec_per_trial=float(np.mean(times)) if times else None,
                peak_mps_gb=round(peak / 2**30, 1), stop_reason=reason or "全部済んだ",
                chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip(),
                torch=torch.__version__)
    with open(f"{out_root}/runs.jsonl", "a") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")


os.makedirs(out_dir, exist_ok=True)

# ── 行の並び：Exp20 と同じ順で、1〜37 セッションに 3 試行そろう刺激だけ ──
bd = np.load(data("meta.npz"))
tr = np.load(data("test_trials.npz"))
m20 = np.load(args.exp20_meta)
first = np.r_[0, np.flatnonzero(np.diff(m20["stim"])) + 1]
order20 = m20["stim"][first]
img_of = {int(k): j for j, k in enumerate(bd["test_stim20"]) if k >= 0}
n_reps_img = np.zeros(len(bd["test_ids"]), int)
n_reps_img[tr["img"]] = tr["n_reps"]
row_of = {(int(j), int(r)): i for i, (j, r) in enumerate(zip(tr["img"], tr["rep"]))}
rows = []                                                           # (刺激の番号 k、テストの画像 j、試行 r（平均なら -1）、予測の行)
for k in order20:
    j = img_of.get(int(k))
    if j is None or n_reps_img[j] != 3:
        continue
    if args.average_reps:
        rows.append((int(k), j, -1, j))
    else:
        rows.extend((int(k), j, r, row_of[(j, r)]) for r in range(3))
N = len(rows)
stim = np.array([k for k, _, _, _ in rows])
src = np.array([[j, r] for _, j, r, _ in rows])
pred_row = np.array([p for _, _, _, p in rows])
print(f"Exp20 の刺激 {len(order20)} 枚のうち、Brain-Diffuser のテストにない {int(sum(int(k) not in img_of for k in order20))} 枚、"
      f"試行がそろわない {int(sum(int(k) in img_of and n_reps_img[img_of[int(k)]] != 3 for k in order20))} 枚を除き、"
      f"C = {len(np.unique(stim))}、N = {N}")
meta_path = f"{out_dir}/meta.npz"
if os.path.exists(meta_path):
    old = np.load(meta_path)
    assert np.array_equal(old["stim"], stim) and np.array_equal(old["pred_row"], pred_row), "前回と試行の並びが違う"
else:
    np.savez(meta_path, stim=stim, src=src, pred_row=pred_row, pred_kind=pred_kind,
             coco73k=bd["test_ids"][src[:, 0]], trial=np.where(src[:, 1] >= 0, tr["trial"][pred_row], -1),
             seed=seed, diff_str=args.diff_str, mix_str=args.mix_str, average_reps=args.average_reps)

# ═════════════════════════════════════════════════════════════════════════
# パス A（S2：VDVAE の画像）
# ═════════════════════════════════════════════════════════════════════════
if args.stage == "vdvae":
    sys.path.append("vdvae")
    from vae import VAE
    from train_helpers import restore_params

    H = {'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}

    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    H = dotdict(H)                                                  # 元のコードと同じ設定
    H.restore_ema_path = f"{args.vdvae_dir}/imagenet64-iter-1600000-model-ema.th"

    s2 = open_out("s2_vdvae64.npy", np.uint8, (N, 64, 64, 3))
    tm = open_out("times_vdvae.npy", np.float32, (N,))
    done = open_out("done_vdvae.npy", np.bool_, (N,))
    print(f"done なのに空の行：{reset_empty(done, [s2])}（ふつうは 0）")
    print(f"パス A：{int(done.sum())} / {N} 行が済んでいる")
    if done.all():
        print("パス A は全部済んでいる")
        sys.exit(0)

    ema_vae = VAE(H)                                                # model_utils.load_vaes と同じ（.cuda() だけ変える）
    restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
    ema_vae.requires_grad_(False)
    ema_vae = ema_vae.to(device)
    pred = np.load(data(f"pred_vdvae_{pred_kind}.npy"), mmap_mode="r")
    shapes = [tuple(s) for s in np.load(data("vdvae_latent_shapes.npy"))]   # 31 層の (c, h, w)
    sizes = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    assert sizes[-1] == pred.shape[1], (sizes[-1], pred.shape)

    pending, times, peak, reason = [], [], 0, None
    t_start = datetime.now()
    with torch.no_grad():
        for t in tqdm(range(N)):
            if done[t]:
                continue
            reason = stop_reason() or ("--limit" if limit and len(times) >= limit else None)
            if reason:
                break
            t0 = time.time()
            z = np.asarray(pred[pred_row[t]], dtype=np.float32)
            latents = [torch.tensor(z[sizes[i]:sizes[i + 1]].reshape((1,) + shapes[i])).float().to(device)
                       for i in range(len(shapes))]                 # latent_transformation と同じ
            torch.manual_seed(seed + t)                             # 行ごとに固定の乱数（予測にない層と、出力の画素）
            px_z = ema_vae.decoder.forward_manual_latents(1, latents, t=None)
            im = ema_vae.decoder.out_net.sample(px_z)               # (1, 64, 64, 3) uint8
            s2[t] = im[0]
            tm[t] = time.time() - t0
            times.append(float(tm[t]))
            pending.append(t)
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == "mps" else 0)
            if len(pending) >= 50:
                commit([s2, tm], done, pending)
    commit([s2, tm], done, pending)
    log_run("vdvae", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")

# ═════════════════════════════════════════════════════════════════════════
# パス B（S3：Versatile Diffusion の画像）
# ═════════════════════════════════════════════════════════════════════════
if args.stage == "vd":
    sys.path.append("versatile_diffusion")
    import lib.sync as sync                                         # exp23_bd_features.py と同じ（CUDA の数で割るため）
    sync.get_rank = lambda type="local": (0, 0, 0) if type == "all" else 0
    sync.get_world_size = lambda type="local": (1, 1, 1) if type == "all" else 1
    from lib.cfg_helper import model_cfg_bank
    from lib.model_zoo import get_model
    from lib.model_zoo.ddim import DDIMSampler
    from lib.model_zoo.ddim_vd import DDIMSampler_VD

    def register_buffer(self, name, attr):                          # 元は cuda を直書き
        if type(attr) == torch.Tensor:
            if attr.dtype == torch.float64 and device.type == "mps":
                attr = attr.float()
            attr = attr.to(device)
        setattr(self, name, attr)
    DDIMSampler.register_buffer = register_buffer

    def regularize_image(x):                                        # 元のコードの regularize_image（PIL の画像の場合）
        x = x.resize([512, 512], resample=Image.Resampling.BICUBIC)
        return tvtrans.ToTensor()(x)

    done_a = np.load(f"{out_dir}/done_vdvae.npy", mmap_mode="r")
    s2 = np.load(f"{out_dir}/s2_vdvae64.npy", mmap_mode="r")
    starts = np.r_[0, np.flatnonzero(np.diff(stim)) + 1]
    groups = np.split(np.arange(N), starts[1:])
    im = open_out("s3_images.npy", np.uint8, (N, 3, 512, 512))
    tm = open_out("times_vd.npy", np.float32, (N,))
    done = open_out("done_vd.npy", np.bool_, (N,))
    print(f"done なのに空の行 {reset_empty(done, [im])}（ふつうは 0）、済んだ行 {int(done.sum())} / {N}")
    todo = [g for g in groups if not done[g].all()]
    if not todo:
        print("パス B は全部済んでいる")
        sys.exit(0)
    if not done_a[todo[0]].all():
        sys.exit("次の刺激のパス A が済んでいない。先にパス A を行う")

    # ── Versatile Diffusion（元のコードと同じ設定。CLIP は fp32、autokl と拡散モデルは fp16）──
    net = get_model()(model_cfg_bank()("vd_noema"))
    sd = torch.load(args.vd_pth, map_location="cpu", mmap=True)
    r = net.load_state_dict(sd, strict=False)                      # 元のコードと同じ
    print("VD：読めなかった重み", len(r.missing_keys), r.missing_keys[:10], "余った重み", len(r.unexpected_keys), r.unexpected_keys[:10])
    del sd
    net.clip = net.clip.to(device)
    net.autokl = net.autokl.to(device).half()
    sampler = DDIMSampler_VD(net)
    sampler.model.model.diffusion_model.device = device
    sampler.model.model.diffusion_model.half().to(device)
    ddim_steps, ddim_eta, scale = 50, 0, 7.5                        # 元のコードと同じ
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    t_enc = int(args.diff_str * ddim_steps)
    pred_vision = np.load(data(f"pred_clipvision_{pred_kind}.npy"), mmap_mode="r")
    pred_text = np.load(data(f"pred_cliptext_{pred_kind}.npy"), mmap_mode="r")
    with torch.no_grad():
        utx = net.clip_encode_text("").to(device).half()            # 無条件の埋め込み（行によらない）
        uim = net.clip_encode_vision(torch.zeros((1, 3, 224, 224)).to(device)).to(device).half()

    pending, times, peak, reason, n_stim = [], [], 0, None, 0
    t_start = datetime.now()
    with torch.no_grad():
        for g in tqdm(groups):
            if done[g].all():
                continue
            if not done_a[g].all():
                reason = "パス A が済んでいない刺激に来た"
                break
            if limit and n_stim >= limit:
                reason = "--limit"
                break
            for t in g:
                if done[t]:
                    continue
                reason = stop_reason()
                if reason:
                    break
                t0 = time.time()
                torch.manual_seed(seed + 1_000_000 + int(t))        # 行ごとに固定の乱数（autokl の事後分布、初期の雑音）
                zim = regularize_image(Image.fromarray(np.asarray(s2[t])))
                zin = (zim * 2 - 1).unsqueeze(0).to(device).half()
                init_latent = net.autokl_encode(zin)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
                cim = torch.tensor(np.asarray(pred_vision[pred_row[t]])).half().to(device).unsqueeze(0)
                ctx = torch.tensor(np.asarray(pred_text[pred_row[t]])).half().to(device).unsqueeze(0)
                z = sampler.decode_dc(x_latent=z_enc, first_conditioning=[uim, cim], second_conditioning=[utx, ctx],
                                      t_start=t_enc, unconditional_guidance_scale=scale, xtype="image",
                                      first_ctype="vision", second_ctype="prompt", mixed_ratio=(1 - args.mix_str))
                x = net.autokl_decode(z.half())
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)  # color_adj = None のときと同じ
                assert torch.isfinite(x).all(), f"行 {t} の画像に nan"
                im[t] = x[0].mul(255).byte().cpu().numpy()          # ToPILImage と同じ（fp16 のまま 255 を掛けて切り捨て）
                tm[t] = time.time() - t0
                times.append(float(tm[t]))
                pending.append(int(t))
                peak = max(peak, torch.mps.driver_allocated_memory() if device.type == "mps" else 0)
            commit([im, tm], done, pending)
            if reason:
                break
            n_stim += 1
    log_run("vd", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")
