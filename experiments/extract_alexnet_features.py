"""
extract_alexnet_features.py
----------------------------
AlexNet relu7 (fc7, 4096次元) 特徴量を
GOD 訓練・テスト画像から抽出して保存する。

出力:
  data/god/alexnet_features_train.npz  → {features: (N, 4096), filenames: (N,)}
  data/god/alexnet_features_test.npz   → {features: (N, 4096), filenames: (N,)}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

TRAIN_DIR = "data/god/images_train"
TEST_DIR  = "data/god/images_test"
OUT_TRAIN = "data/god/alexnet_features_train.npz"
OUT_TEST  = "data/god/alexnet_features_test.npz"


def load_alexnet_relu7():
    """AlexNet の relu7（fc7 の後）まで出力するモデルを返す。"""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    # classifier[0:6] = Dropout, fc6, ReLU, Dropout, fc7, ReLU(relu7)
    extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(),
        *list(model.classifier.children())[:6]  # up to relu7 (index 5 = ReLU after fc7)
    )
    extractor.eval()
    return extractor


def extract_features(image_dir, model, transform):
    files = sorted([f for f in os.listdir(image_dir)
                    if f.upper().endswith((".JPEG", ".JPG", ".PNG"))])
    features = []
    filenames = []
    for fname in tqdm(files, desc=image_dir):
        fpath = os.path.join(image_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = model(x).squeeze(0).numpy()
            features.append(feat)
            filenames.append(fname)
        except Exception as e:
            print(f"  skip {fname}: {e}")
    return np.array(features, dtype=np.float32), np.array(filenames)


def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Loading AlexNet (pretrained)...")
    model = load_alexnet_relu7()

    print(f"\nExtracting training features from {TRAIN_DIR}...")
    feat_train, names_train = extract_features(TRAIN_DIR, model, transform)
    np.savez(OUT_TRAIN, features=feat_train, filenames=names_train)
    print(f"Saved {OUT_TRAIN}: {feat_train.shape}")

    print(f"\nExtracting test features from {TEST_DIR}...")
    feat_test, names_test = extract_features(TEST_DIR, model, transform)
    np.savez(OUT_TEST, features=feat_test, filenames=names_test)
    print(f"Saved {OUT_TEST}: {feat_test.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
