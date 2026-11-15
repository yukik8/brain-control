"""
exp23_gate_g2.py
----------------
Experiment 23 の関門 G2（再現）：3 試行を平均した入力の再構成で、CLIP の 2 択の識別精度を出し、論文の値と比べる

識別精度は Brain-Diffuser の scripts/evaluate_reconstruction.py と eval_extract_features.py と同じ計算：
OpenAI の CLIP ViT-L/14 の画像の埋め込み（最終の出力、768 次元。fp32）、画像は 224 × 224 に縮めて（切り抜かない）
CLIP の平均・標準偏差で正規化、本物の画像と再構成の相関（np.corrcoef）で、再構成ごとに「ほかの本物の画像との
相関が、正しい組の相関より低い」割合を数えて平均する（pairwise_corr_all）。MindEye1 にも同じ計算を使う。
CLIP は transformers の CLIPModel.get_image_features（clip.load("ViT-L/14") の visual と同じ重み・同じ計算）。

bd_env の python で実行する。本物の画像は exp23_bd_prepare.py の test_stim.npy（COCO の番号で引く）。
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPModel

parser = argparse.ArgumentParser(description="Exp23 gate G2: CLIP two-way identification")
parser.add_argument("--out_dir", type=str, required=True, help="--average_reps で作った出力（meta.npz、s3_images.npy、done_vd.npy）")
parser.add_argument("--data_dir", type=str, required=True, help="exp23_bd_prepare.py の出力（test_stim.npy、meta.npz）")
parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
args = parser.parse_args()
device = torch.device(args.device)

meta = np.load(f"{args.out_dir}/meta.npz")
done = np.load(f"{args.out_dir}/done_vd.npy")
recon = np.load(f"{args.out_dir}/s3_images.npy", mmap_mode="r")
idx = np.flatnonzero(done)
assert bool(meta["average_reps"]), "G2 は 3 試行を平均した出力で行う"
bd = np.load(f"{args.data_dir}/meta.npz")
row_of = {int(c): j for j, c in enumerate(bd["test_ids"])}
gt = np.load(f"{args.data_dir}/test_stim.npy", mmap_mode="r")
print(f"済んだ刺激 {len(idx)} 枚で計算する")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).float().eval()
normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


def embed(images):
    out = []
    with torch.no_grad():
        for s in range(0, len(images), 32):
            x = torch.stack([normalize(T.functional.to_tensor(T.functional.resize(im, (224, 224))).float())
                             for im in images[s:s + 32]])
            out.append(model.get_image_features(pixel_values=x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


gt_feat = embed([Image.fromarray(np.asarray(gt[row_of[int(meta["coco73k"][t])]])) for t in idx])
rc_feat = embed([Image.fromarray(np.asarray(recon[t]).transpose(1, 2, 0)) for t in idx])


def pairwise_corr_all(ground_truth, predictions):                  # evaluate_reconstruction.py と同じ
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    return np.mean(success_cnt) / (len(ground_truth) - 1)


acc = float(pairwise_corr_all(gt_feat, rc_feat))
print(f"CLIP の 2 択の識別精度：{acc * 100:.1f}%（{len(idx)} 枚）")
with open(f"{args.out_dir}/gate_g2.json", "w") as f:
    json.dump(dict(clip_two_way=acc, n=int(len(idx))), f, ensure_ascii=False)
