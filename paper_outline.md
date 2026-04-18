# 論文骨格

## タイトル候補

**メイン候補（採用）**
> Plausible but Not Brain-Driven: Quantifying Prior Dominance in Neural Image Reconstruction

**サブ候補**
> Degree of Brain Control: A Metric for Evaluating Prior Dominance in Neural Image Reconstruction
> Beyond Visual Quality: Measuring How Much the Brain Controls Neural Image Reconstruction
> When Reconstructions Lie: Measuring Brain Control Beyond Visual Quality

---

## ターゲット会議・ジャーナル

| ベニュー | タイプ | 締切目安 |
|---------|--------|---------|
| NeurIPS 2026 Workshop (BCI / NeuroAI) | Workshop | 2026年秋 |
| ICLR 2027 | 本会議 | 2026年10月 |
| NeuroImage | ジャーナル | 随時 |

---

## 構成

---

### Abstract（150語以内）

**流れ**:
1. 問題提起: 脳デコーディングの再構成画像評価は視覚品質に偏っている
2. 洞察: generator prior が強いほど、脳活動とは無関係に「良い画像」が生成される
3. 提案: BC（Degree of Brain Control）= Var(broken)/Var(preserved)
4. 結果の要約: Real BC=1.259, Shuffled BC=1.001（同じ画像品質なのに区別可能）
5. 意義: BC は既存メトリクスでは検出できない prior 支配を定量化する

**草稿（v2 — polished）**:
> Brain-to-image reconstruction can produce strikingly realistic images, but it remains unclear whether these reconstructions are actually driven by brain signals. Existing metrics primarily assess perceptual similarity (e.g., feature similarity or visual fidelity), rather than whether brain activity genuinely controls the output. Strong generative priors can produce plausible images even when brain activity is randomized—a failure mode that current metrics cannot detect.
>
> We introduce the Degree of Brain Control (BC), defined as the ratio of within-category feature variance under shuffled versus preserved brain–stimulus correspondence. BC = 1 indicates prior-dominated reconstruction, whereas BC > 1 indicates that brain signals impose structure on the output.
>
> Applied to the GOD dataset, real fMRI yields BC = 1.259 while shuffled signals yield BC = 1.001, despite identical feature norms and comparable visual quality. This dissociation replicates across all five subjects (paired t = 7.44, p = 0.0017). Furthermore, BC captures structural degradation of decoded representations that is not detectable by identification accuracy alone (Cohen's d = 0.465, p = 0.023).
>
> BC provides an interpretable absolute scale for quantifying how much brain signals constrain reconstruction beyond model priors.

---

### 1. Introduction

Brain-to-image reconstruction has progressed from recovering coarse spatial patterns [Miyawaki et al., 2008; Nishimoto et al., 2011] to producing images that appear to reflect fine-grained visual content [Shen et al., 2019; Ozcelik & VanRullen, 2023; Scotti et al., 2023]. This progress has been driven in large part by increasingly powerful generative models—GANs, VAEs, and diffusion models—that serve as the image synthesis stage of reconstruction pipelines. The reconstructed images are evaluated primarily on visual quality: perceptual similarity, identification accuracy, or pixel-level fidelity. By these measures, recent methods appear highly successful.

Yet visual quality is not the same as brain control. A generative model with a strong prior can produce realistic, category-consistent images even when its inputs carry no brain-specific information. Consider a decoder that maps fMRI signals to intermediate features, followed by a generator that reconstructs an image from those features. If the generator's prior is sufficiently strong, it will produce a natural-looking image regardless of whether the decoded features reflect the actual stimulus—simply because the prior constrains the output to the manifold of natural images. Under these conditions, reconstructions may look convincing while being dominated by the model prior rather than by brain signals. Existing metrics—structural similarity (SSIM), perceptual distance (LPIPS, DreamSim), and identification accuracy—measure properties of the output image, not the extent to which that output is constrained by brain activity. None can distinguish a brain-driven reconstruction from a prior-driven one when both produce visually similar results.

We make this failure mode concrete: when the correspondence between fMRI trials and stimuli is randomly shuffled, the resulting reconstructions have identical feature norms and comparable visual quality to those produced by genuine brain signals—yet by construction contain no stimulus-specific information. Standard metrics cannot detect this shuffling. This motivates a different question: not *how realistic* is the reconstruction, but *how much is the reconstruction controlled by the brain*?

To answer this question, we introduce the **Degree of Brain Control (BC)**, a metric defined as the ratio of within-category feature variance under shuffled versus preserved brain–stimulus correspondence. When brain signals carry no category-relevant information, shuffling does not change the variance structure of decoded features, and BC ≈ 1. When brain signals impose consistent structure within categories, preserved correspondence yields lower within-category variance than shuffled correspondence, and BC > 1. BC = 1 thus provides an empirically grounded baseline for prior-dominated reconstruction, and deviations above 1 are directly interpretable as evidence of brain influence.

We make the following contributions:

1. **Identifying a fundamental failure mode**: We show that prior-dominated reconstruction—plausible-looking output with no genuine brain influence—is undetectable by existing metrics. This failure mode becomes more severe as generative priors grow stronger, making it increasingly relevant to the field.
2. **BC as the metric that uniquely detects it**: We define the Degree of Brain Control (BC) and establish its theoretical grounding, including an information-theoretic interpretation ($I(c;\,\hat{\mathbf{x}}) \approx \frac{1}{2}\log\mathrm{BC}$) and its relationship to decoder SNR. BC = 1 provides an empirically grounded, interpretable floor for prior-dominated reconstruction.
3. **Empirical validation across five subjects**: Real fMRI yields BC = 1.259; shuffled signals yield BC = 1.001—despite identical feature norms and comparable visual quality. This dissociation replicates across all five subjects (paired $t = 7.44$, $p = 0.0017$), ruling out subject-specific effects.
4. **Non-redundancy with identification accuracy**: BC captures structural degradation of decoded representations not detectable by identification accuracy alone. Matched-accuracy conditions differ significantly in BC (Cohen's $d = 0.465$, $p = 0.023$), and BC degrades faster than accuracy under noise injection—demonstrating that BC and accuracy are complementary, not equivalent.

The remainder of this paper is organized as follows. Section 2 reviews related work on brain-to-image reconstruction and evaluation metrics. Section 3 formally defines BC and discusses its properties. Section 4 presents our experiments. Section 5 discusses implications and limitations.

---

**（段落構成メモ）**
- P1: Hook — 再構成の進歩は generative model に依存
- P2: 問題提起 — prior が強いほど「良い画像」≠「脳が制御している」; 既存メトリクスは検出不可
- P3: 問題を具体化 — shuffled でも視覚品質は同じ; 問いの転換
- P4: BC の定義と解釈
- P5: 貢献リスト（箇条書き）
- P6: 論文構成

---

### 2. Related Work

#### 2.1 Brain-to-Image Reconstruction

Early work on neural image reconstruction focused on recovering low-level visual features—such as oriented gratings and spatial frequency maps—directly from fMRI responses using linear models [Miyawaki et al., 2008; Nishimoto et al., 2011]. A key advance came with the introduction of intermediate feature representations: rather than predicting pixels directly, Shen et al. [2019] proposed decoding deep CNN features (AlexNet layers) from brain signals and inverting them into images using a feature-to-image generator trained on natural images. This approach, demonstrated on the Generic Object Decoding (GOD) dataset [Huth et al., 2012; Shen et al., 2019], produced recognizable object-level reconstructions for the first time at scale.

Subsequent work shifted toward more powerful generative backbones. Brain-Diffuser [Ozcelik & VanRullen, 2023] and MindEye [Scotti et al., 2023] leverage latent diffusion models conditioned on CLIP embeddings decoded from fMRI, achieving high perceptual quality and identification accuracy. MinD-Vis [Chen et al., 2023] similarly exploits masked brain modeling as a pretraining objective before diffusion-based synthesis. While these methods represent impressive engineering achievements, they share a common structural property: a powerful generative prior whose output quality is largely determined by the prior itself, independent of the quality of the brain-decoded input. Our work addresses a measurement gap that becomes increasingly important as these priors grow stronger.

#### 2.2 Evaluation Metrics for Brain Decoding

Reconstruction quality in brain decoding is typically assessed along two axes: pixel-level fidelity and semantic accuracy.

**Pixel-level metrics** (MSE, SSIM, PSNR) compare reconstructed images to target images at the pixel level. These metrics are well-understood but poorly suited to high-level generative pipelines where the output lies on a low-dimensional natural-image manifold regardless of input quality.

**Perceptual similarity metrics** (LPIPS [Zhang et al., 2018], DreamSim [Fu et al., 2023]) measure distance in deep feature spaces and better reflect human perceptual judgments. However, they still assess how similar the reconstruction *looks* to the target, not whether the brain signal determined that appearance.

**Identification accuracy** (top-$k$ or leave-one-out ranking in feature space) tests whether the reconstruction is closer to its target image than to distractors. This is a stronger test than pixel similarity and has become a standard benchmark [Shen et al., 2019; Scotti et al., 2023]. Yet identification accuracy can be high even when the generator prior contributes most of the signal: a decoder that consistently maps brain responses to near-the-mean of the training distribution may still rank correctly if the distractor features are sufficiently different.

None of these metrics addresses a more fundamental question: *to what degree are brain signals controlling the reconstruction, rather than the model prior?* BC is designed to answer exactly this question.

#### 2.3 Role of Generative Priors

The tension between model priors and input fidelity is well-recognized in the generative modeling literature. In VAEs [Kingma & Welling, 2014], the KL regularization term explicitly pulls latent codes toward a standard Gaussian prior, reducing the influence of individual inputs. In GANs [Goodfellow et al., 2014] and diffusion models [Ho et al., 2020; Song et al., 2021], the prior is implicit but strong: sampling procedures are designed to produce outputs on the natural-image manifold. As model capacity increases, these priors become more constraining relative to the decoder input.

In the brain decoding context, this tension manifests as a trade-off between visual quality and brain control: stronger priors yield more realistic reconstructions but may reduce the degree to which individual brain states are reflected in the output. Several authors have noted informally that diffusion-based reconstructions may be "over-smoothed" toward category prototypes [Ozcelik & VanRullen, 2023], but no quantitative metric has been proposed to measure this effect. BC formalizes this intuition as a measurable, interpretable quantity.

Concurrent work on hallucination detection in neural decoding [Tang et al., 2023] shares the motivation of identifying when generative models produce outputs not supported by the input signal, but focuses on linguistic decoding rather than image reconstruction and does not provide a variance-ratio formulation comparable to BC.

---

### 3. Method: Degree of Brain Control (BC)

#### 3.1 Problem Setup

Let $\mathbf{b}_i \in \mathbb{R}^V$ denote the fMRI response on trial $i$, and let $c_i \in \{1, \ldots, C\}$ denote the category label of the presented stimulus. A reconstruction pipeline consists of two stages: a decoder $f : \mathbb{R}^V \to \mathbb{R}^d$ that maps brain signals to an intermediate feature vector, and a generator $g : \mathbb{R}^d \to \mathcal{I}$ that maps features to image space. In practice, $f$ is a linear (Ridge regression) decoder trained on held-out data, and the decoded feature $\hat{\mathbf{x}}_i = f(\mathbf{b}_i) \in \mathbb{R}^d$ is used as input to $g$.

We evaluate brain control at the feature level, i.e., on the set of decoded features $\{\hat{\mathbf{x}}_i\}$ over $N$ test trials. Let $S_c = \{i : c_i = c\}$ be the index set of trials belonging to category $c$, with $|S_c| = n$ for all $c$.

#### 3.2 Definition

We define two variance quantities over the decoded features.

**Within-category variance under preserved correspondence** ($V_\text{pres}$) measures trial-to-trial variability within the same category when brain signals are correctly paired with stimuli:

$$V_\text{pres} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{d} \sum_{k=1}^{d} \mathrm{Var}_{i \in S_c}[\hat{x}_{ik}]$$

**Within-category variance under broken correspondence** ($V_\text{brok}$) measures the same quantity after randomly permuting the trial indices, destroying the brain–stimulus pairing:

$$V_\text{brok} = \mathbb{E}_{\pi}\!\left[\frac{1}{C} \sum_{c=1}^{C} \frac{1}{d} \sum_{k=1}^{d} \mathrm{Var}_{i \in S_c}[\hat{x}_{\pi(i),k}]\right]$$

where $\pi$ is a uniformly random permutation of $\{1, \ldots, N\}$. The expectation is estimated by averaging over $N_\text{shuf} = 1000$ independent permutations.

The **Degree of Brain Control** is defined as:

$$\boxed{\mathrm{BC} = \frac{V_\text{brok}}{V_\text{pres}}}$$

#### 3.3 Interpretation

**BC = 1** (prior-dominated): Shuffling trial indices does not change the variance structure of decoded features. This means the decoder output carries no category-specific information—reconstruction is determined entirely by the model prior, not by the brain signals.

**BC > 1** (brain-controlled): Preserved correspondence yields lower within-category variance than broken correspondence. Trials within the same category produce more similar decoded features when correctly paired with their stimuli than when randomly paired. This indicates that brain signals impose consistent structure on the reconstruction.

The quantity BC − 1 serves as an effect-size measure of brain influence. Under a Gaussian approximation of the feature distribution, BC has a direct information-theoretic interpretation:

$$I(c;\, \hat{\mathbf{x}}) \approx \frac{1}{2} \log \mathrm{BC}$$

where $I(c;\,\hat{\mathbf{x}})$ is the mutual information between the category label and the decoded feature vector. BC = 1 corresponds to zero mutual information; BC > 1 corresponds to positive information transmission from stimulus category through brain signals to decoded features.

#### 3.4 Relationship to Identification Accuracy

Identification accuracy (e.g., leave-one-out cosine similarity ranking) is a commonly used metric in brain decoding. Both BC and accuracy increase with the signal-to-noise ratio of the decoder. However, identification accuracy is a nonlinear (sigmoidal) function of SNR and saturates at high SNR, whereas BC scales approximately linearly with SNR:

$$\mathrm{BC} \approx 1 + \mathrm{SNR}_\text{eff}$$

where $\mathrm{SNR}_\text{eff}$ is the effective signal-to-noise ratio in feature space. This means BC continues to grow in regimes where accuracy has already saturated, enabling detection of structural differences that accuracy cannot resolve. We demonstrate this empirically in Section 4.3.

#### 3.5 Computation

Given $N$ test trials with decoded features $\{\hat{\mathbf{x}}_i\}$ and category labels $\{c_i\}$:
1. Compute $V_\text{pres}$ directly from the decoded features.
2. Estimate $V_\text{brok}$ by averaging over $N_\text{shuf} = 1000$ random permutations.
3. Compute $\mathrm{BC} = V_\text{brok} / V_\text{pres}$.

Category-level BC values $\{\mathrm{BC}_c\}$ are computed per category and averaged; these are also used for effect-size analyses (Cohen's $d$). The shuffle-based standard error provides an uncertainty estimate for BC.

---

### 4. Experiments

#### 4.1 Setup

**Dataset.** We use the Generic Object Decoding (GOD) dataset [Shen et al., 2019], which contains fMRI responses from five subjects viewing 1,200 training images and 50 test images (one image per category, repeated 35 times each). Test trials thus consist of 50 categories × 35 repetitions = 1,750 trials per subject. All analyses use the Visual Cortex (VC) ROI unless otherwise stated.

**Decoder.** We train a Ridge regression decoder ($\alpha = 100$) with standard scaling to predict AlexNet relu7 features (fc7, $d = 4{,}096$) from brain signals. The decoder is trained on training-set trials and evaluated on held-out test trials. This follows the original GOD experimental protocol [Shen et al., 2019].

**BC computation.** For each experiment, BC is computed using $N_\text{shuf} = 1{,}000$ across-trial permutations (Section 3.5). Category-level BC values are averaged to obtain a scalar summary; their standard error across categories provides an uncertainty estimate.

---

#### 4.2 Experiment 1: Prior-Dominated Baseline

**Setup.** We compare three conditions that differ only in the brain signals fed to the decoder:

- **Real**: genuine fMRI test signals (correct brain–stimulus pairing).
- **Shuffled**: test trial indices randomly permuted before decoding, destroying brain–stimulus correspondence while preserving the marginal distribution of brain signals.
- **Random**: standard Gaussian noise substituted for brain signals, representing complete absence of brain information.

The decoder and generator are identical across all three conditions.

**Results.**

| Condition | BC | Feature norm |
|-----------|-----|-------------|
| Real | **1.259** | 70.9 |
| Shuffled | 1.001 | 70.9 |
| Random | 1.000 | 118.4 |

The key finding is the dissociation between BC and feature norm. Real and Shuffled conditions produce decoded features with identical L2 norms (70.9), meaning that any metric based on feature magnitude—and by extension, visual quality—cannot distinguish them. Yet BC separates them sharply: Real yields BC = 1.259, indicating that brain signals impose measurable structure on the reconstruction, while Shuffled yields BC = 1.001, consistent with pure prior domination (BC = 1 by construction when correspondence is destroyed). Random signals yield a higher norm (118.4) due to the different marginal statistics of Gaussian noise relative to true brain signals, reflecting a qualitatively different failure mode.

Figure 2 shows reconstructed images for three representative categories under all three conditions. Rows are visually similar across conditions, confirming that feature norm (and visual quality) is insufficient to distinguish brain-controlled from prior-dominated reconstruction. Figure 3 presents the BC and feature norm comparison as bar charts, making the dissociation explicit.

**Multi-subject replication.** To assess generalizability, we repeat Experiment 1 for all five subjects in the GOD dataset. Real BC exceeds Shuffled BC in every subject (Subject 1–5: Real BC = 1.259, 1.135, 1.250, 1.208, 1.138; Shuffled BC ≈ 1.000 throughout). A paired $t$-test across subjects yields $t = 7.44$, $p = 0.0017$, confirming that the Real–Shuffled dissociation is not specific to a single subject.

---

#### 4.3 Experiment 2: BC vs. Identification Accuracy

**Motivation.** A natural concern is that BC may be redundant with identification accuracy, which is widely used in brain decoding evaluation. We investigate the relationship between the two metrics and examine whether BC provides information beyond what accuracy already captures.

**ROI comparison.** We compute both BC and leave-one-out cosine similarity accuracy across ten visual ROIs (V1, V2, V3, V4, LOC, FFA, PPA, LVC, HVC, VC) for Subject 1. Across ROIs, BC and accuracy are highly correlated ($r = 0.967$), which is expected: both ultimately reflect the quality of the decoded representation, and both are computed from the same Ridge decoder. This high correlation confirms that BC tracks the same underlying signal as accuracy, providing a basic validity check.

However, correlation does not imply equivalence. BC and accuracy measure different properties—accuracy captures the ability to rank the correct image above distractors, while BC measures the absolute variance compression imposed by brain signals within categories.

**Same accuracy, different BC.** To demonstrate the non-redundancy of BC directly, we construct a matched-accuracy comparison. We inject Gaussian noise ($\sigma = 0.75$) into standardized V1 signals, which degrades the decoder and reduces BC. We then identify the noise level at which the V1+noise condition matches the identification accuracy of the HVC ROI—a condition with naturally lower SNR. Despite being matched in accuracy, V1+noise and HVC differ significantly in BC (Cohen's $d = 0.465$, 95% CI $= [0.003, 0.043]$, $p = 0.023$, two-sample $t$-test over per-category BC values). This demonstrates that two conditions can have equivalent identification accuracy while differing in the structural quality of their decoded representations—a difference that BC captures and accuracy cannot.

Figure 4 shows the noise sensitivity curves for BC and accuracy under progressive noise injection. BC degrades faster and more continuously than accuracy, which saturates near a floor value. This confirms the theoretical prediction that BC tracks SNR linearly while accuracy tracks it sigmoidally (Section 3.4).

---

#### 4.4 Experiment 3: Image-Space Brain Control (Phase 2 BC)

**Setup.** Experiments 1–2 measure BC in the intermediate feature space (relu7, Phase 1). To assess whether brain control extends to the final image space, we define Phase 2 BC using DreamSim embeddings [Fu et al., 2023] ($d = 1{,}792$) computed directly from the generated images. This measures the degree to which brain signals structure the final pixel-space output, after passing through the generator.

**Results.** Phase 2 BC across 50 categories has mean $1.037 \pm 0.107$ (std). The per-category distribution is wide: goldfish reaches BC = 1.295 (high brain control at image level), while washer reaches BC = 0.850 (below 1, indicating that the generator introduces category-inconsistent variation not present in the feature space). The correlation between Phase 1 and Phase 2 BC across categories is $r = 0.308$ ($p = 0.030$), indicating a weak but significant positive relationship.

The modest Phase 1–Phase 2 correlation suggests that feature-space brain control does not fully translate to image-space brain control. Categories where the generator is well-calibrated to the decoded features tend to show higher Phase 2 BC, while categories where the generator introduces additional variation (or where the feature-to-image mapping is less consistent) show lower Phase 2 BC. This motivates measuring BC at both stages of the pipeline when evaluating reconstruction methods.

---

### 5. Discussion

#### 5.1 What BC Measures — and What It Does Not

BC is a variance ratio, not a proportion of variance explained. A value of BC = 1.259 does not mean "25.9% of the reconstruction is brain-controlled." It means that within-category feature variance is 25.9% higher when brain–stimulus correspondence is destroyed than when it is preserved—that is, genuine brain signals make same-category reconstructions more similar to each other than random pairing would. BC − 1 is best interpreted as an effect-size measure of brain influence in feature space, directly analogous to a signal-to-noise ratio in the decoded representation.

Crucially, BC = 1 is not simply a null result; it is an empirically grounded floor. When brain signals carry no category-relevant information—as in the Shuffled condition—BC = 1 by construction, regardless of how realistic the reconstructions appear. This makes BC = 1 a meaningful, interpretable baseline: any reconstruction pipeline that yields BC ≈ 1 on genuine brain data should be considered prior-dominated, regardless of its visual quality or identification accuracy.

#### 5.2 Implications for Diffusion-Based Reconstruction

The most visually impressive brain reconstruction methods today rely on latent diffusion models [Ozcelik & VanRullen, 2023; Scotti et al., 2023]. These models have extremely strong priors—they are trained on billions of images and optimized to produce outputs that lie precisely on the natural-image manifold. This is precisely the regime where we expect BC to be most informative: as the prior strengthens, the generator becomes increasingly capable of producing high-quality, category-consistent images from near-arbitrary inputs, and BC should approach 1 even for genuine brain signals.

To empirically ground this prediction, we simulate increasing prior dominance via feature mixing. We linearly interpolate decoded features between the Real condition ($\alpha=0$) and the Shuffled condition ($\alpha=1$, prior-dominated by construction):
$$\hat{\mathbf{x}}_\alpha = (1-\alpha)\,\hat{\mathbf{x}}_\text{real} + \alpha\,\hat{\mathbf{x}}_\text{shuffled}$$
As $\alpha$ increases from 0 to 1, BC decreases monotonically from 1.259 to 1.001 (Figure~6A), with a near-linear trajectory (Figure~6B). This demonstrates that BC is a continuous, sensitive measure of prior dominance: any generative process that partially replaces brain-specific information with prior-driven information will produce a BC value between 1.001 and 1.259, proportional to the degree of replacement.

This directly supports the hypothesis that diffusion-based reconstruction methods—whose stronger generative priors impose more structure on the output independently of the input—will yield BC values closer to 1 than the relu7generator used here, despite achieving higher perceptual quality and identification accuracy. Applying BC to publicly available diffusion-based reconstructions remains an important direction for future work.

#### 5.3 BC Beyond the Feature Decoder

Our results show a weak but significant correlation between Phase 1 BC (feature space) and Phase 2 BC (image space, DreamSim embeddings; $r = 0.308$, $p = 0.030$). The modest correlation indicates that brain control in feature space does not fully transfer to image space—the generator introduces additional variability that can either amplify or dilute the brain signal depending on the category. This suggests that Phase 2 BC, measured directly on generated images, captures complementary information about the full reconstruction pipeline and should be reported alongside Phase 1 BC when the generator is not a simple deterministic inversion.

#### 5.4 Limitations

**Linear decoder.** BC as computed here depends on the Ridge regression decoder. A more expressive decoder (e.g., nonlinear neural network) could in principle extract more category-relevant information from brain signals, increasing BC. Our results thus represent a lower bound on the brain control achievable with the GOD dataset and AlexNet features.

**Feature space selection.** BC is measured in the AlexNet relu7 space. Different feature spaces may yield different BC values; in particular, higher-level semantic features (e.g., CLIP embeddings) may show higher BC because they are more directly predictable from high-level visual cortex activity. The choice of feature space should be reported when comparing BC values across studies.

**Absolute scale of BC.** While BC provides an absolute scale relative to the prior-dominated baseline (BC = 1), the numerical value of BC − 1 is not directly comparable across datasets, decoders, or feature spaces because it depends on the intrinsic structure of the decoded feature distribution. BC is most informative as a within-study comparison (e.g., Real vs. Shuffled within the same experimental setup).

---

### 6. Conclusion

Brain-to-image reconstruction has made remarkable progress, but visual quality alone cannot tell us whether the brain is actually driving the output. We introduced the Degree of Brain Control (BC), a metric that directly addresses this question by comparing within-category feature variance under preserved versus shuffled brain–stimulus correspondence. BC = 1 provides an empirically grounded floor corresponding to pure prior-dominated generation; BC > 1 indicates that brain signals impose measurable structure on the reconstruction.

Applied to the GOD dataset, we demonstrated that genuine fMRI signals yield BC = 1.259, while shuffled signals yield BC = 1.001—despite producing decoded features with identical norms and visually indistinguishable reconstructions. This dissociation, replicated across all five subjects ($t = 7.44$, $p = 0.0017$), shows that BC detects a property of brain decoding that no existing metric captures. Furthermore, BC reveals structural degradation of decoded representations not detectable by identification accuracy alone (Cohen's $d = 0.465$, $p = 0.023$), demonstrating that the two metrics are complementary rather than redundant.

As generative priors in brain reconstruction grow stronger, the risk of conflating visual quality with brain control will only increase. We argue that BC should be reported alongside perceptual quality metrics in all brain-to-image reconstruction studies, and that maximizing BC—not just visual fidelity—should be an explicit goal of future decoder development.

---

## Figure 一覧（論文用）

| Figure | 内容 | ソース |
|--------|------|--------|
| Fig 1 | 問題提起の概念図（BC の必要性） | 新規作成 |
| Fig 2 | Real / Shuffled / Random の再構成画像比較 | Exp14 |
| Fig 3 | BC バープロット（3条件） | Exp13 |
| Fig 4 | BC 感度カーブ（ノイズ注入） | Exp16b |
| Fig 5 | Same Acc, Different BC（散布図） | Exp16 |

---

## 次のステップ

1. **★今すぐ**: Subject2-5 で Exp13 を実行（複数被験者の再現性）
2. Fig 1（概念図）を作成
3. Abstract を polish
4. Introduction の P1-P4 を書く
