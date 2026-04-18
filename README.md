# Plausible but Not Brain-Driven
### Quantifying Prior Dominance in Neural Image Reconstruction

> **Core finding**: A reconstruction pipeline can produce realistic, category-correct images even when brain signals are completely randomized — and standard metrics cannot detect this.

---

## The Problem

Brain-to-image reconstruction has achieved striking visual quality. But *visual quality is not the same as brain control.*

Strong generative priors (GANs, diffusion models) can produce plausible images regardless of whether the input brain signals carry any stimulus-specific information. Existing evaluation metrics — SSIM, LPIPS, identification accuracy — measure how the output *looks*, not how much the brain *controls* it.

**The failure mode is concrete:**

| Condition | BC | Feature norm | Visual quality |
|-----------|-----|-------------|----------------|
| Real fMRI | **1.259** | 70.9 | plausible |
| Shuffled fMRI | 1.001 | 70.9 | plausible |
| Gaussian noise | 1.000 | 118.4 | plausible |

Shuffled signals destroy all brain–stimulus correspondence — yet feature norms are identical to real fMRI, and reconstructed images are visually indistinguishable. **Standard metrics cannot tell these apart. BC can.**

---

## The Metric: Degree of Brain Control (BC)

$$\mathrm{BC} = \frac{V_\text{brok}}{V_\text{pres}}$$

where:
- $V_\text{pres}$: within-category feature variance under **preserved** brain–stimulus correspondence
- $V_\text{brok}$: within-category feature variance under **shuffled** correspondence (expectation over 1000 permutations)

**Interpretation:**
- **BC = 1**: Shuffling doesn't change the variance structure → reconstruction is prior-dominated
- **BC > 1**: Preserved correspondence yields lower within-category variance → brain signals impose structure

Under Gaussian approximation, BC has a direct information-theoretic interpretation:
$$I(c;\, \hat{\mathbf{x}}) \approx \tfrac{1}{2} \log \mathrm{BC}$$

BC = 1 corresponds to zero mutual information between the category and the decoded features.

---

## Key Results

**Result 1: Prior dominance is undetectable by existing metrics**

Real fMRI → BC = 1.259. Shuffled fMRI → BC = 1.001. Feature norms: identical (70.9). The dissociation between BC and feature statistics directly demonstrates the failure mode.

**Result 2: Replicates across all five subjects**

| Subject | Real BC | Shuffled BC |
|---------|---------|-------------|
| Subject 1 | 1.259 | 1.001 |
| Subject 2 | 1.135 | 0.999 |
| Subject 3 | 1.250 | 0.999 |
| Subject 4 | 1.208 | 1.001 |
| Subject 5 | 1.138 | 1.002 |

Paired t-test (Real vs. Shuffled): **t = 7.44, p = 0.0017**

**Result 3: BC ≠ identification accuracy**

BC and accuracy are correlated across ROIs (r = 0.967), confirming BC tracks meaningful signal. But at matched accuracy levels, BC differs significantly (Cohen's d = 0.465, p = 0.023) — demonstrating the two metrics are complementary, not redundant.

---

## Figures

| Figure | Description |
|--------|-------------|
| `outputs/fig1_concept.png` | Pipeline schematic: same decoder/generator, different brain input |
| `outputs/fig2_reconstruction_comparison.png` | Real vs. Shuffled vs. Random reconstructions (3 categories × 4 reps) |
| `outputs/fig3_bc_barplot.png` | BC and feature norm comparison across conditions |
| `outputs/fig4_noise_sensitivity.png` | BC degrades faster than accuracy under noise injection |
| `outputs/fig5_bc_vs_accuracy.png` | BC vs. accuracy across ROIs + matched-accuracy comparison |

---

## Reproducing the Results

### Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pillow h5py
```

### Data

This project uses the [Generic Object Decoding (GOD) dataset](https://github.com/KamitaniLab/GenericObjectDecoding) and the [brain-decoding-cookbook](https://github.com/KamitaniLab/brain-decoding-cookbook-public) reconstructions.

Place data as follows:
```
brain-decoding-cookbook-public/   # reconstruction images (tiff)
data/
  Subject1.mat                    # fMRI data (GOD format)
  Subject2.mat
  ...
  ImageFeatures.h5                # AlexNet features
```

### Running Experiments

```bash
# Main result: Real vs. Shuffled vs. Random (Subject 1)
python experiments/exp13_prior_vs_brain.py

# Multi-subject replication (all 5 subjects)
python experiments/exp17_multisubject_bc.py

# BC vs. identification accuracy across ROIs
python experiments/exp15_roi_bc_vs_accuracy.py

# Same accuracy, different BC
python experiments/exp16_same_acc_diff_bc.py
```

### Generating Paper Figures

```bash
python experiments/fig1_concept.py
python experiments/fig2_reconstruction_comparison.py
python experiments/fig3_bc_barplot.py
python experiments/fig4_noise_sensitivity.py
python experiments/fig5_bc_vs_accuracy.py
```

### Computing BC for Your Own Data

```python
from compute_bc import compute_bc

# pred_features: (N_trials, N_features) array of decoded features
# cat_labels:    (N_trials,) array of integer category labels
bc_mean, bc_sem, bc_per_cat = compute_bc(
    pred_features, cat_labels,
    n_shuffle=1000, seed=42, mode="across"
)
print(f"BC = {bc_mean:.4f} ± {bc_sem:.4f}")
# BC ≈ 1.0 → prior-dominated
# BC > 1.0 → brain signals are controlling the reconstruction
```

---

## Repository Structure

```
compute_bc.py                  # Core BC computation
experiments/
  exp13_prior_vs_brain.py      # Main experiment: 3-condition comparison
  exp15_roi_bc_vs_accuracy.py  # BC vs. accuracy across ROIs
  exp16_same_acc_diff_bc.py    # Same accuracy, different BC
  exp17_multisubject_bc.py     # Multi-subject replication
  fig1_concept.py              # Paper Figure 1
  fig2_reconstruction_comparison.py
  fig3_bc_barplot.py
  fig4_noise_sensitivity.py
  fig5_bc_vs_accuracy.py
outputs/                       # Generated figures and saved results
theory_bc_definition.md        # Mathematical definition and derivations
paper_outline.md               # Full paper draft
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{bc2026,
  title   = {Plausible but Not Brain-Driven: Quantifying Prior Dominance
             in Neural Image Reconstruction},
  author  = {Kikuchi, Yuki},
  year    = {2026},
  note    = {Preprint}
}
```

*(arXiv link coming soon)*

---

## License

MIT
