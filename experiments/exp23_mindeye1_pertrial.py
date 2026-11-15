"""
exp23_mindeye1_pertrial.py
--------------------------
Experiment 23a: NSD × MindEye1 での試行ごとの推論（各段階の出力を保存する）

MindEye1（Scotti et al., 2023）の src/Reconstructions.py を元に、同じ画像の 3 試行を平均せずに、
試行ごとに再構成を作る。Exp20（experiments/exp20_mindeye2_pertrial.py）と同じ作りにしてある。

  - パス A（--stage prior）: 脳活動 → voxel2clip（S1、257 × 768）→ 拡散 prior（S2、257 × 768）、
                             と低次の経路 voxel2sd（SD の VAE の潜在表現、4 × 64 × 64。乱数を使わない）
  - パス B（--stage vd）    : 低次の潜在表現 → ぼかし画像 → img2img の初期値にして、S2 で条件づけた
                             Versatile Diffusion で 512 × 512 の画像を作る（S3）

刺激は Exp20 と同じ順（Exp20 の meta.npz の、seed 42 の無作為な順）で、MindEye1 のテスト集合にある刺激だけを処理する。
刺激の番号は Exp20 と同じ（0〜999、MindEye2 の all_images.pt の行）。行ごとに乱数を固定し（パス A は seed + t、
パス B は seed + 1,000,000 + t）、済んだ行を done_*.npy に記録するので、止めて再開しても同じ結果になる。
STOP ファイル（--out_dir の親ディレクトリ）か --stop_at の時刻で止まる。

実行のしかた：このファイルを MindEye1 の src（fMRI-reconstruction-NSD/src）にコピーし、そこで MindEyeV2 の
fmri 環境の python で実行する（models.py、utils.py を import するため）。

元のコードから変えた点：
  - device = mps。utils.device も mps にする
  - 3 試行を平均せず、試行ごとに 1 枚（--average_reps のときだけ元のコードと同じ平均）
  - 1 試行に 1 枚だけ作る（recons_per_sample = 1。論文は 16 枚から CLIP で選ぶ。選ぶと BC が変わるので選ばない）
  - utils.reconstruction の Versatile Diffusion の部分を vd_recon に書き写した（prior を通さずに、保存した S2 を使うため）。
    計算の中身は同じ。乱数の generator は、パス A とパス B で別々にする
  - チェックポイントの読み込みは strict=True（元のコードは strict=False で、読めない重みがあっても黙って進む）。
    学習でだけ使う重み（voxel2clip.temp、maps_projector）は、名前を挙げて捨てる
  - 低次の経路（voxel2sd）の注意層の重みの名前を、diffusers 0.13（元の環境）から 0.23 の名前に直す
  - diffusers 0.23.0 では Versatile Diffusion のテキストの U-Net が作れない（get_down_block に、ない引数を渡す）→
    その引数を捨てる（作られるモデルは同じ）。U-Net 2 つと VAE の重みが、ファイルと名前・値とも全部一致するかを確かめる。
    同じく DualTransformer2DModel.forward に、ない引数 encoder_attention_mask を渡す → 捨てる（値は None、もともと使わない）
  - clip_retrieval（LAION の検索にだけ使う）は読み込まない
"""

import os
import sys
import types
import argparse
import time
import numpy as np
from tqdm import tqdm
import webdataset as wds

import torch
import torch.nn as nn

sys.modules["clip_retrieval"] = types.ModuleType("clip_retrieval")             # utils.py が先頭で import する。検索は使わない
sys.modules["clip_retrieval.clip_client"] = types.SimpleNamespace(ClipClient=None)
import utils
from models import BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork, Voxel2StableDiffusionModel

parser = argparse.ArgumentParser(description="Exp23a: per-trial MindEye1 inference")
parser.add_argument("--data_path", type=str, required=True,
                    help="pscotti/naturalscenesdataset を落とした場所（webdataset_avg_split と mindeye_models がある）")
parser.add_argument("--vd_dir", type=str, required=True, help="shi-labs/versatile-diffusion を落とした場所（diffusers 版）")
parser.add_argument("--exp20_meta", type=str, required=True, help="Exp20 の meta.npz（刺激の順と COCO の番号）")
parser.add_argument("--subj", type=int, default=1, choices=[1, 2, 5, 7])
parser.add_argument("--model_name", type=str, default="prior_257_final_subj01_bimixco_softclip_byol")
parser.add_argument("--autoencoder_name", type=str, default="autoencoder_subj01_4x_locont_no_reconst")
parser.add_argument("--img2img_strength", type=float, default=.85, help="元のコードの既定値")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stage", type=str, required=True, choices=["prior", "vd"],
                    help="prior = パス A（S1・S2・低次の潜在表現）、vd = パス B（S3）")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--limit", type=int, default=0, help="この実行で処理する数の上限（パス A は行、パス B は刺激）。0 なら上限なし")
parser.add_argument("--stop_at", type=str, default=None, help="この時刻になったら止める（例：2026-11-10T08:00）")
parser.add_argument("--average_reps", action="store_true", help="再現確認用：元のコードと同じく 3 試行を平均し、画像 1 枚につき 1 行にする")
parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"],
                    help="cpu は動作確認用（GPU を使う本番と同時に確かめるため）。乱数の列が変わるので、本番の出力には使わない")
args = parser.parse_args()
for k, v in vars(args).items():
    if k != "device":
        globals()[k] = v

device = torch.device(args.device)
utils.device = device
print("device:", device)

utils.seed_everything(seed)

import fcntl, json, subprocess
from datetime import datetime

NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}[subj]
TIMESTEPS_PRIOR = 100            # 元のコード（reconstruction の timesteps_prior）
NUM_INFERENCE_STEPS = 20         # 同上（Versatile Diffusion、UniPC）
GUIDANCE_SCALE = 3.5             # 同上（img_variations = False のとき）

# ── 途中で止めても大丈夫にするための部品（Exp20 と同じ）──
out_dir = os.path.abspath(out_dir)
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


def old_attention_names(k):
    """diffusers 0.13（元の環境）の注意層の重みの名前を、0.23 の名前に直す（diffusers の
    _convert_deprecated_attention_blocks と同じ付け替え）。元のコードは strict=False なので、0.23 で動かすと
    この重みが読まれず、乱数のままになる。"""
    if ".attentions." in k:
        for a, b in ((".query.", ".to_q."), (".key.", ".to_k."), (".value.", ".to_v."), (".proj_attn.", ".to_out.0.")):
            k = k.replace(a, b)
    return k


def load_strict(module, path, key="model_state_dict", prefix=None, drop=(), rename=None):
    """チェックポイントを strict=True で読む。prefix を渡すと、その部分の重みだけを取り出して読む。
    drop に挙げた名前で始まる重みは、チェックポイントにあることを確かめてから捨てる（学習でだけ使う重み）。
    rename を渡すと、重みの名前をそれで直してから読む。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)   # mmap：大きいファイルを一度にメモリへ読まない
    sd = ckpt[key]
    if prefix is not None:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if rename is not None:
        sd = {rename(k): v for k, v in sd.items()}
    for d in drop:
        dropped = [k for k in sd if k.startswith(d)]
        assert dropped, f"捨てる重み {d} がチェックポイントにない"
        print(f"{os.path.basename(path)}：学習でだけ使う重みを捨てる：{d}（{len(dropped)} 個）")
    sd = {k: v for k, v in sd.items() if not k.startswith(tuple(drop))}
    module.load_state_dict(sd, strict=True)
    epoch = ckpt.get("epoch")
    del ckpt, sd
    return epoch


os.makedirs(out_dir, exist_ok=True)
models_dir = f"{data_path}/mindeye_models"

# ═════════════════════════════════════════════════════════════════════════
# パス A（S1・S2・低次の潜在表現）
# ═════════════════════════════════════════════════════════════════════════
if stage == "prior":
    # ── テストデータ（元のコードと同じ webdataset。試行は 3 つ別々に入っている）──
    url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    data = wds.WebDataset(url, resampled=False).decode("torch")\
        .rename(voxels="nsdgeneral.npy", trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple("voxels", "trial", "coco", "reps")
    vox, trial, coco, reps = [], [], [], []
    for v, tr, c, r in data:
        vox.append(torch.as_tensor(v).float()); trial.append(np.asarray(tr)); coco.append(int(np.asarray(c).ravel()[0]))
        reps.append(np.asarray(r))
    vox = torch.stack(vox)                                          # (982, 3, num_voxels)
    coco = np.array(coco)
    print("テストデータ：", tuple(vox.shape), "trial", np.array(trial).shape, "num_uniques", np.unique(np.array(reps), return_counts=True))
    assert vox.shape[1:] == (3, NUM_VOXELS), vox.shape
    assert len(np.unique(coco)) == len(coco), "同じ COCO の画像が 2 回ある"
    # 3 試行がそろっていない画像は除外する。1〜37 セッションで 1 回か 2 回しか見ていない画像は、webdataset が
    # 足りない試行を埋めている（1 回なら同じ試行の複製、2 回なら 2 試行の平均に近いもの。完全に同じではない）。
    # num_uniques（本当の試行の数）で判定する
    reps = np.array([int(np.asarray(r).ravel()[0]) for r in reps])
    exact = np.array([any(torch.equal(v[a], v[b]) for a, b in ((0, 1), (0, 2), (1, 2))) for v in vox])
    assert not exact[reps == 3].any(), "num_uniques = 3 なのに同じ試行がある"
    dup = reps < 3

    # ── 刺激の順：Exp20 と同じ順で、MindEye1 のテスト集合にある刺激だけ ──
    m20 = np.load(exp20_meta)
    first = np.r_[0, np.flatnonzero(np.diff(m20["stim"])) + 1]      # Exp20 の刺激ごとの最初の行
    order20, coco20 = m20["stim"][first], m20["coco73k"][first]
    pos = {c: i for i, c in enumerate(coco)}
    groups = [(int(k), pos[c]) for k, c in zip(order20, coco20) if c in pos and not dup[pos[c]]]
    n_not_in_me1 = int(sum(c not in pos for c in coco20))
    print(f"Exp20 の刺激 {len(order20)} 枚のうち、MindEye1 のテストにない {n_not_in_me1} 枚、"
          f"試行がそろわない {int(sum(dup[pos[c]] for c in coco20 if c in pos))} 枚を除き、C = {len(groups)}")
    rows = []                                                       # (刺激の番号 k、テストデータの行、試行 r（平均なら -1））
    for k, i in groups:
        if average_reps:
            rows.append((k, i, -1))
        else:
            rows.extend((k, i, r) for r in range(3))                # 同じ刺激の 3 試行は連続する
    N = len(rows)
    stim = np.array([k for k, _, _ in rows])
    src = np.array([[i, r] for _, i, r in rows])

    meta_path = f"{out_dir}/meta.npz"
    if os.path.exists(meta_path):
        old = np.load(meta_path)
        assert np.array_equal(old["stim"], stim) and np.array_equal(old["src"], src), "前回と試行の並びが違う"
    else:
        np.savez(meta_path, stim=stim, src=src, coco73k=coco[src[:, 0]], trial=np.array(trial)[src[:, 0]],
                 n_not_in_me1=n_not_in_me1, n_dup=int(dup.sum()), model_name=model_name,
                 autoencoder_name=autoencoder_name, subj=subj, seed=seed, average_reps=average_reps)

    s1 = open_out("s1_backbone.npy", np.float16, (N, 257, 768))
    s2 = open_out("s2_prior.npy", np.float16, (N, 257, 768))
    lo = open_out("low_latent.npy", np.float16, (N, 4, 64, 64))
    tm = open_out("times_prior.npy", np.float32, (N,))
    done = open_out("done_prior.npy", np.bool_, (N,))
    print(f"done なのに空の行：{reset_empty(done, [s1, s2, lo])}（ふつうは 0）")
    print(f"パス A：{int(done.sum())} / {N} 行が済んでいる")
    if done.all():
        print("パス A は全部済んでいる")
        sys.exit(0)

    # ── モデル（元のコードと同じ設定）──
    voxel2clip = BrainNetwork(in_dim=NUM_VOXELS, out_dim=257 * 768)
    prior_network = VersatileDiffusionPriorNetwork(dim=768, depth=6, dim_head=64, heads=12, causal=False,
                                                   learned_query_mode="pos_emb")
    diffusion_prior = BrainDiffusionPrior(net=prior_network, image_embed_dim=768, condition_on_text_encodings=False,
                                          timesteps=100, cond_drop_prob=0.2, image_embed_scale=None, voxel2clip=voxel2clip)
    # voxel2clip.temp は対照学習の温度（学習でだけ使う。models.py の BrainNetwork では定義がコメントアウトされている）
    epoch = load_strict(diffusion_prior, f"{models_dir}/{model_name}/last.pth", drop=("voxel2clip.temp",))
    print("prior の epoch：", epoch)
    voxel2sd = Voxel2StableDiffusionModel(in_dim=NUM_VOXELS)
    # maps_projector は低次の経路の対照学習でだけ使う（元のコードも use_cont=False で作り、読まない）
    load_strict(voxel2sd, f"{models_dir}/{autoencoder_name}/epoch120.pth", drop=("maps_projector.",),
                rename=old_attention_names)
    diffusion_prior.eval().requires_grad_(False).to(device)
    voxel2sd.eval().requires_grad_(False).to(device)

    pending, times, peak, reason = [], [], 0, None
    t_start = datetime.now()
    with torch.no_grad():                                           # 元のコードと同じく fp32
        for t, (k, i, r) in enumerate(tqdm(rows)):
            if done[t]:
                continue
            reason = stop_reason() or ("--limit" if limit and len(times) >= limit else None)
            if reason:
                break
            t0 = time.time()
            voxel = (vox[i].mean(0) if r < 0 else vox[i, r])[None].to(device)   # (1, num_voxels)
            clip0, _ = diffusion_prior.voxel2clip(voxel)
            clip0 = clip0.view(1, -1, 768)                          # S1
            gen = torch.Generator(device=device).manual_seed(seed + t)          # 行ごとに固定の乱数
            prior_out = diffusion_prior.p_sample_loop(clip0.shape, text_cond=dict(text_embed=clip0), cond_scale=1.,
                                                      timesteps=TIMESTEPS_PRIOR, generator=gen)   # S2
            low = voxel2sd(voxel)                                   # (1, 4, 64, 64)
            if not times:
                print("shapes:", tuple(voxel.shape), tuple(clip0.shape), tuple(prior_out.shape), tuple(low.shape))
            out = [x[0].float().cpu().numpy().astype(np.float16) for x in (clip0, prior_out, low)]
            assert all(np.isfinite(x).all() for x in out), f"行 {t} に inf か nan"
            s1[t], s2[t], lo[t] = out
            tm[t] = time.time() - t0
            times.append(float(tm[t]))
            pending.append(t)
            peak = max(peak, torch.mps.driver_allocated_memory() if device.type == "mps" else 0)
            if len(pending) >= 20:
                commit([s1, s2, lo, tm], done, pending)
    commit([s1, s2, lo, tm], done, pending)
    log_run("prior", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")


# ═════════════════════════════════════════════════════════════════════════
# パス B（S3）
# ═════════════════════════════════════════════════════════════════════════
if stage == "vd":
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    from torchvision import transforms

    clip_normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                          [0.26862954, 0.26130258, 0.27577711])   # models.Clipper.normalize と同じ

    def vd_recon(emb, img_lowlevel, unet, vae, scheduler, gen):
        """utils.reconstruction の Versatile Diffusion の部分（recons_per_sample = 1、テキストなし、img2img あり）。"""
        emb = emb / (emb[:, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)      # CLS トークンのノルムで割る（元のコードと同じ）
        prompt_embeds = torch.zeros(1, 77, 768)
        input_embedding = torch.cat([torch.zeros_like(emb), emb]).to(device).to(unet.dtype)   # classifier-free guidance
        prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)
        scheduler.set_timesteps(num_inference_steps=NUM_INFERENCE_STEPS, device=device)
        init_timestep = min(int(NUM_INFERENCE_STEPS * img2img_strength), NUM_INFERENCE_STEPS)
        t_start = max(NUM_INFERENCE_STEPS - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1].repeat(1)
        init_latents = vae.encode(clip_normalize(img_lowlevel).to(device).to(vae.dtype)).latent_dist.sample(gen)
        init_latents = vae.config.scaling_factor * init_latents
        noise = torch.randn([1, 4, 64, 64], device=device, generator=gen, dtype=input_embedding.dtype)
        latents = scheduler.add_noise(init_latents, noise, latent_timestep)
        for t in timesteps:
            latent_model_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        return utils.decode_latents(latents, vae)

    meta = np.load(f"{out_dir}/meta.npz")
    stim = meta["stim"]
    N = len(stim)
    starts = np.r_[0, np.flatnonzero(np.diff(stim)) + 1]
    groups = np.split(np.arange(N), starts[1:])
    done_a = np.load(f"{out_dir}/done_prior.npy", mmap_mode="r")
    s2 = np.load(f"{out_dir}/s2_prior.npy", mmap_mode="r")
    lo = np.load(f"{out_dir}/low_latent.npy", mmap_mode="r")
    im = open_out("s3_images.npy", np.uint8, (N, 3, 512, 512))
    blur = open_out("low_images.npy", np.uint8, (N, 3, 512, 512))
    tm = open_out("times_vd.npy", np.float32, (N,))
    done = open_out("done_vd.npy", np.bool_, (N,))
    print(f"done なのに空の行 {reset_empty(done, [im])}（ふつうは 0）、済んだ行 {int(done.sum())} / {N}")
    todo = [g for g in groups if not done[g].all()]
    if not todo:
        print("パス B は全部済んでいる")
        sys.exit(0)
    if not done_a[todo[0]].all():
        sys.exit("次の刺激のパス A が済んでいない。先にパス A を行う")

    # ── Versatile Diffusion（元のコードと同じ設定。fp16）──
    vd_dtype = torch.float16 if device.type == "mps" else torch.float32           # CPU（動作確認）では fp16 の演算がないものがある
    # diffusers 0.23.0 の不具合：テキストの U-Net（UNetFlatConditionModel）を作るとき、get_down_block・get_up_block に
    # これらの関数にない引数を 4 つ渡して止まる → 受け取って捨てるようにする。どちらの関数も、もともとこの引数をブロックに
    # 渡さない（ブロックの既定値 1、"default" になる）ので、VD の設定（値は 1、"default"）では作られるモデルは同じ
    import inspect
    import diffusers.pipelines.versatile_diffusion.modeling_text_unet as text_unet_module
    from diffusers import AutoencoderKL, UNet2DConditionModel

    def drop_new_args(f):
        def g(*a, transformer_layers_per_block=1, attention_type="default", attention_head_dim=None, resolution_idx=None, **k):
            assert transformer_layers_per_block == 1 and attention_type == "default", (transformer_layers_per_block, attention_type)
            return f(*a, **k)
        return g

    if "transformer_layers_per_block" not in inspect.signature(text_unet_module.get_down_block).parameters:
        text_unet_module.get_down_block = drop_new_args(text_unet_module.get_down_block)
        text_unet_module.get_up_block = drop_new_args(text_unet_module.get_up_block)
    # 同じく、U-Net のブロックが DualTransformer2DModel.forward に encoder_attention_mask を渡して止まる → 受け取って捨てる。
    # ここでは渡すマスクがない（None）うえ、DualTransformer2DModel はもともとマスクを使わない
    if "encoder_attention_mask" not in inspect.signature(DualTransformer2DModel.forward).parameters:
        dual_forward = DualTransformer2DModel.forward

        def dual_forward_drop_mask(self, *a, encoder_attention_mask=None, **k):
            assert encoder_attention_mask is None
            return dual_forward(self, *a, **k)
        DualTransformer2DModel.forward = dual_forward_drop_mask
    parts = {}                                                       # diffusers は重みが足りないと止まるが、余るときは警告だけ
    for name, cls in (("image_unet", UNet2DConditionModel), ("text_unet", text_unet_module.UNetFlatConditionModel),
                      ("vae", AutoencoderKL)):                       # → ファイルの重みと名前・値が全部一致するかを確かめる
        model = cls.from_pretrained(vd_dir, subfolder=name)
        sd = torch.load(f"{vd_dir}/{name}/diffusion_pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
        model._convert_deprecated_attention_blocks(sd)                # VAE の注意層の古い名前を付け替える（diffusers が読むときと同じ）
        msd = model.state_dict()
        assert set(sd) == set(msd), f"{name}：余る {sorted(set(sd) - set(msd))[:5]}、足りない {sorted(set(msd) - set(sd))[:5]}"
        assert all(torch.equal(msd[k], sd[k].to(msd[k].dtype)) for k in sd), f"{name}：ファイルと値が違う重みがある"
        print(f"{name}：重み {len(sd)} 個がファイルと一致")
        parts[name] = model
        del sd, msd
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_dir, **parts).to(device).to(vd_dtype)
    vd_pipe.image_unet.eval().requires_grad_(False)
    vd_pipe.vae.eval().requires_grad_(False)
    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_dir, subfolder="scheduler")
    for name, module in vd_pipe.image_unet.named_modules():          # テキストの重みを 0 にして、画像の条件だけを使う
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = 0.0
            for j, kind in enumerate(("text", "image")):
                if kind == "text":
                    module.condition_lengths[j] = 77
                    module.transformer_index_for_condition[j] = 1
                else:
                    module.condition_lengths[j] = 257
                    module.transformer_index_for_condition[j] = 0
    unet, vae, scheduler = vd_pipe.image_unet, vd_pipe.vae, vd_pipe.scheduler
    del vd_pipe.text_unet                                            # 使わない（mix_ratio = 0）。メモリを空ける
    import gc; gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

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
                gen = torch.Generator(device=device).manual_seed(seed + 1_000_000 + int(t))
                low = torch.from_numpy(np.asarray(lo[t], dtype=np.float32))[None].to(device)
                blurry = vae.decode(low.to(vae.dtype) / 0.18215).sample / 2 + 0.5   # 元のコードと同じ（fp16、クランプしない）
                emb = torch.from_numpy(np.asarray(s2[t], dtype=np.float32))[None].to(device)
                img = vd_recon(emb, blurry, unet, vae, scheduler, gen)
                assert torch.isfinite(img).all() and torch.isfinite(blurry).all(), f"行 {t} の画像に nan"
                im[t] = (img[0].float().cpu().numpy() * 255).round().astype(np.uint8)
                blur[t] = (blurry[0].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                tm[t] = time.time() - t0
                times.append(float(tm[t]))
                pending.append(int(t))
                peak = max(peak, torch.mps.driver_allocated_memory() if device.type == "mps" else 0)
            commit([im, blur, tm], done, pending)
            if reason:
                break
            n_stim += 1
    log_run("vd", t_start, times, peak, reason)
    print("止めた理由：", reason or "全部済んだ")
