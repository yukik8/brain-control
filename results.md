# Brain Control (BC) 実験ログ

## 概要

fMRI脳活動から画像特徴量をデコードし、Degree of Brain Control (BC) を測定する。

```
BC = Var(decoded_features | broken) / Var(decoded_features | preserved)

BC ≈ 1  → prior-dominated（脳活動が再構成を制御していない）
BC >> 1 → brain-controlled（脳活動が再構成を強く制御している）
```

- **データ**: GOD (Generic Object Decoding) dataset
- **被験者**: Subject1
- **デコーダ**: Ridge regression (alpha=100)
- **変動測定**: カテゴリ内分散（特徴次元の平均）
- **シャッフル回数**: 50

---

## Experiment 1: VC × cnn8（2026-04-09）

**設定**
- ROI: VC（視覚野全体, 4466 voxels）
- 特徴量: cnn8（AlexNet-like の最終 FC 層, 1000 units）
- Train: 1200 trials, Test: 1750 trials (50 categories × 35 reps)

**結果**
| 指標 | 値 |
|---|---|
| Decoding accuracy (mean r) | 0.2299 |
| Var (preserved) | 0.365793 |
| Var (broken) | 0.455081 ± 0.000775 |
| **BC** | **1.2441 ± 0.0021** |
| Effective Dimensionality (ED) | 11.74 |

**解釈**
- BC = 1.24：脳活動のシャッフルで変動が約24%増加 → 脳活動はある程度特徴量を制御しているが強くはない
- ED = 11.74：1000次元中、実質的に使われているのは約12次元（低次元collapse）

---

## Experiment 2: ROI比較 × cnn8（2026-04-09）

**設定**
- 特徴量: cnn8（1000 units）
- Train: 1200 trials, Test: 1750 trials (50 categories × 35 reps)
- デコーダ: Ridge (alpha=100), シャッフル: 50回

**結果**

| ROI | Voxels | mean_r | BC | ± | ED |
|---|---|---|---|---|---|
| ROI_V1  | 1004 | 0.1067 | 1.1472 | 0.0018 | 12.51 |
| ROI_V2  | 1018 | 0.1167 | 1.1389 | 0.0019 | 12.57 |
| ROI_V3  |  759 | 0.1585 | 1.1559 | 0.0022 |  9.75 |
| ROI_V4  |  740 | 0.1982 | 1.1969 | 0.0025 |  8.67 |
| ROI_LOC |  540 | 0.1583 | 1.2265 | 0.0034 |  7.23 |
| ROI_FFA |  568 | 0.1641 | 1.2292 | 0.0028 |  7.49 |
| ROI_PPA |  356 | 0.1414 | 1.1255 | 0.0027 |  6.88 |
| ROI_LVC | 2281 | 0.1637 | 1.1647 | 0.0019 | 12.98 |
| ROI_HVC | 2049 | 0.1857 | 1.1732 | 0.0024 | 11.23 |
| **ROI_VC**  | **4466** | **0.2299** | **1.2441** | 0.0021 | 11.74 |

**解釈**

- BCは全ROIで 1.13〜1.24 の範囲 → 脳活動はある程度特徴量を制御しているが穏やか
- **高次ROI（LOC, FFA）のBCが高い**（1.22〜1.23）、低次ROI（V1, V2）は低い（1.14）
  → 高次視覚野の方が cnn8 特徴量を強く制御しているという仮説と一致
- **EDは高次ROIほど低い**（V1=12.5 → LOC=7.2）
  → 高次ROIのデコードは低次元に収束している（semantic collapse の可能性）
- PPA の BC が低い（1.13）のは、PPA がシーン・場所に特化しており cnn8 の分類特徴と相性が悪い可能性
- VC（全視覚野）は voxel 数が多い分、平均的に最も高い BC（1.24）を示した
