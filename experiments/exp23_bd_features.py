"""
exp23_bd_features.py
--------------------
Experiment 23b: Brain-Diffuser の特徴量（VDVAE の潜在表現、CLIP-Vision、CLIP-Text）を、学習とテストの画像から取り出す

元のコード（scripts/vdvae_extract_features.py、clipvision_extract_features.py、cliptext_extract_features.py）と
同じ計算。Brain-Diffuser のリポジトリの一番上で、bd_env の python で実行する（元のコードと同じく、vdvae と
versatile_diffusion をそこからの相対パスで読むため）。入力は exp23_bd_prepare.py の出力。

元のコードから変えた点：
  - device を引数にした（mps か cpu。元のコードは cuda）。VDVAE の前処理（model_utils.set_up_data）は .cuda() を
    直書きしているので、同じ計算をここに書いた
  - VDVAE の潜在表現は事後分布からのサンプルなので乱数を使う。元のコードは seed を固定していない → 最初に
    torch.manual_seed(0)（VDVAE の H.seed と同じ値）。元のコードと同じく、テスト → 学習の順に 30 枚ずつ
  - CLIP は Versatile Diffusion のモデル全体を作らず、CLIP の部分（clip_frozen）だけを作り、チェックポイントの
    clip.* を strict=True で読む（元のコードは全体を作って strict=False で読む。使う重みは同じ）
  - CLIP-Vision は 1 枚ずつではなく --bs 枚ずつ計算する（値は丸めの誤差の範囲で同じ）
  - versatile_diffusion/lib/sync.py の get_rank・get_world_size を、1 台・1 プロセスの値を返すものに差し替える
    （GPU の数で割るので、CUDA のない環境では 0 で割ってしまう。ログの表示にだけ使われる）
  - 出力は float32 の .npy（元のコードは float64 の配列に入れる。計算は float32）
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

parser = argparse.ArgumentParser(description="Exp23b: extract Brain-Diffuser features")
parser.add_argument("--feature", type=str, required=True, choices=["vdvae", "clipvision", "cliptext"])
parser.add_argument("--data_dir", type=str, required=True, help="exp23_bd_prepare.py の出力")
parser.add_argument("--vd_pth", type=str, help="vd-four-flow-v1-0-fp16-deprecated.pth（CLIP に使う）")
parser.add_argument("--vdvae_dir", type=str, help="VDVAE の重みの場所（imagenet64-iter-1600000-model-ema.th）")
parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
parser.add_argument("--bs", type=int, default=30, help="VDVAE は元のコードと同じ 30。CLIP-Vision は動作の速さだけに効く")
args = parser.parse_args()
device = torch.device(args.device)
path = lambda name: os.path.join(args.data_dir, name)
done_flag = path(f"features_{args.feature}.done")
if os.path.exists(done_flag):
    print(f"{args.feature} は済んでいる（{done_flag}）")
    sys.exit(0)


def images(name, size):
    """元のコードの batch_generator_external_images と同じ：PIL で size × size に縮める。"""
    im = np.load(path(name), mmap_mode="r")
    for s in range(0, len(im), args.bs):
        yield s, [T.functional.resize(Image.fromarray(np.asarray(x)), (size, size)) for x in im[s:s + args.bs]]


if args.feature == "vdvae":
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
    assert args.bs == 30, "VDVAE は元のコードと同じ 30 枚ずつ（乱数の使われ方が同じになる）"

    torch.manual_seed(H.seed)
    ema_vae = VAE(H)                                                # model_utils.load_vaes と同じ（.cuda() だけ変える）
    restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
    ema_vae.requires_grad_(False)
    ema_vae = ema_vae.to(device)
    shift = torch.tensor([-115.92961967], device=device).view(1, 1, 1, 1)   # set_up_data（imagenet64）と同じ
    scale = torch.tensor([1. / 69.37404], device=device).view(1, 1, 1, 1)
    num_latents = 31

    for split in ("test", "train"):                                 # 元のコードと同じ順
        n = len(np.load(path(f"{split}_stim.npy"), mmap_mode="r"))
        out = None
        for s, batch in images(f"{split}_stim.npy", 64):
            x = torch.stack([torch.tensor(np.array(im)).float() for im in batch])
            inp = x.to(device).float()
            inp.add_(shift).mul_(scale)
            with torch.no_grad():
                activations = ema_vae.encoder.forward(inp)
                px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
                z = np.hstack([stats[i]["z"].cpu().numpy().reshape(len(inp), -1) for i in range(num_latents)])
            if out is None:
                out = np.lib.format.open_memmap(path(f"vdvae_{split}.npy.part"), "w+", np.float32, (n, z.shape[1]))
            out[s:s + len(z)] = z
            print(split, s, flush=True)
        out.flush()
        del out
        os.replace(path(f"vdvae_{split}.npy.part"), path(f"vdvae_{split}.npy"))
    shapes = [tuple(stats[i]["z"].shape[1:]) for i in range(num_latents)]
    np.save(path("vdvae_latent_shapes.npy"), np.array(shapes))     # 再構成で潜在表現を層ごとに戻すのに使う

else:
    sys.path.append("versatile_diffusion")
    import lib.sync as sync                                         # ログの表示で GPU の数（torch.cuda.device_count()）で割る。
    sync.get_rank = lambda type="local": (0, 0, 0) if type == "all" else 0          # CUDA のない環境では 0 で割るので、
    sync.get_world_size = lambda type="local": (1, 1, 1) if type == "all" else 1    # 1 台・1 プロセスとして扱う
    from lib.cfg_helper import model_cfg_bank
    from lib.model_zoo import get_model

    clip = get_model()(model_cfg_bank()("clip_frozen"))             # VD の cfg（vd.yaml の clip_cfg）と同じ
    sd = torch.load(args.vd_pth, map_location="cpu", mmap=True)
    clip.load_state_dict({k[len("clip."):]: v for k, v in sd.items() if k.startswith("clip.")}, strict=True)
    del sd
    clip = clip.to(device)
    clip.eval()

    if args.feature == "clipvision":
        clip.encode_type = "encode_vision"                          # VD.clip_encode_vision と同じ
        for split in ("test", "train"):
            n = len(np.load(path(f"{split}_stim.npy"), mmap_mode="r"))
            out = np.lib.format.open_memmap(path(f"clipvision_{split}.npy.part"), "w+", np.float32, (n, 257, 768))
            for s, batch in images(f"{split}_stim.npy", 512):
                cin = torch.stack([T.functional.to_tensor(im).float() for im in batch]) * 2 - 1
                vision = ((cin + 1) / 2).cpu().numpy()
                vision = [vi for vi in np.transpose(vision, (0, 2, 3, 1))]
                with torch.no_grad():
                    out[s:s + len(batch)] = clip.encode(vision).float().cpu().numpy()
                print(split, s, flush=True)
            out.flush()
            del out
            os.replace(path(f"clipvision_{split}.npy.part"), path(f"clipvision_{split}.npy"))

    else:
        clip.encode_type = "encode_text"                            # VD.clip_encode_text と同じ
        for split in ("test", "train"):
            caps = np.load(path(f"{split}_cap.npy"))
            out = np.zeros((len(caps), 77, 768), np.float32)
            with torch.no_grad():
                for i, annots in enumerate(caps):
                    cin = list(annots[annots != ""])
                    out[i] = clip.encode(cin).float().cpu().numpy().mean(0)
                    if i % 500 == 0:
                        print(split, i, flush=True)
            np.save(path(f"cliptext_{split}.npy"), out)

open(done_flag, "w").close()
print("終了")
