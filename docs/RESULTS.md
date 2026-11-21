# Final benchmark results: 3 seeds (2026-10-28)

Source: `kaggle/results_04/` (Kaggle commit of `03_train_mil.ipynb`, seeds 0–2). 360 runs = 4 encoders × 6 MIL models × 5 folds × 3 seeds
(seed 0 copied from `results_03`). Wall time 3.3 h on 2× T4.
**Test** = locked 108-slide set. **Seed ensemble** = class probabilities of all 15 models (3 seeds × 5 folds) averaged per slide; 95% CI = bootstrap over slides.
**CV** = out-of-fold validation predictions on the other 430 slides (model selection; slightly optimistic, equally for every configuration).
Seed-0-only tables are in `kaggle/results_03/`.

## Test balanced accuracy, seed ensemble [95% CI]
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.834 [0.74–0.92] | 0.796 [0.69–0.89] | 0.811 [0.71–0.90] | 0.805 [0.70–0.89] |
| MeanPool | 0.836 [0.74–0.92] | 0.792 [0.69–0.89] | 0.799 [0.69–0.89] | 0.763 [0.66–0.86] |
| ABMIL | 0.811 [0.71–0.90] | 0.800 [0.71–0.88] | 0.834 [0.74–0.91] | 0.789 [0.70–0.88] |
| DSMIL | 0.816 [0.71–0.91] | 0.800 [0.70–0.90] | 0.786 [0.68–0.89] | 0.788 [0.68–0.88] |
| CLAM-SB | 0.814 [0.72–0.90] | 0.813 [0.72–0.90] | 0.838 [0.74–0.92] | 0.756 [0.65–0.86] |
| TransMIL | 0.816 [0.72–0.91] | 0.818 [0.72–0.90] | 0.798 [0.70–0.89] | 0.718 [0.62–0.83] |

## Test balanced accuracy, mean ± std over the 3 seeds
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.838 ± 0.012 | 0.798 ± 0.009 | 0.807 ± 0.007 | 0.796 ± 0.023 |
| MeanPool | 0.824 ± 0.031 | 0.782 ± 0.025 | 0.795 ± 0.017 | 0.755 ± 0.033 |
| ABMIL | 0.823 ± 0.002 | 0.800 ± 0.005 | 0.846 ± 0.020 | 0.772 ± 0.043 |
| DSMIL | 0.830 ± 0.035 | 0.811 ± 0.011 | 0.778 ± 0.020 | 0.771 ± 0.014 |
| CLAM-SB | 0.815 ± 0.010 | 0.805 ± 0.009 | 0.828 ± 0.020 | 0.762 ± 0.008 |
| TransMIL | 0.803 ± 0.019 | 0.819 ± 0.015 | 0.814 ± 0.025 | 0.699 ± 0.018 |

## Test macro F1, seed ensemble [95% CI]
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.842 [0.75–0.91] | 0.816 [0.71–0.89] | 0.828 [0.73–0.90] | 0.816 [0.72–0.89] |
| MeanPool | 0.832 [0.73–0.90] | 0.822 [0.72–0.90] | 0.820 [0.72–0.89] | 0.773 [0.66–0.86] |
| ABMIL | 0.823 [0.73–0.90] | 0.806 [0.71–0.88] | 0.856 [0.77–0.92] | 0.807 [0.72–0.88] |
| DSMIL | 0.845 [0.75–0.92] | 0.836 [0.74–0.91] | 0.829 [0.74–0.91] | 0.819 [0.71–0.90] |
| CLAM-SB | 0.824 [0.73–0.90] | 0.827 [0.74–0.90] | 0.857 [0.77–0.92] | 0.769 [0.66–0.85] |
| TransMIL | 0.846 [0.76–0.92] | 0.841 [0.75–0.91] | 0.833 [0.73–0.91] | 0.748 [0.64–0.84] |

## Test macro AUROC, seed ensemble [95% CI]
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.976 [0.95–0.99] | 0.967 [0.94–0.99] | 0.978 [0.96–0.99] | 0.966 [0.94–0.99] |
| MeanPool | 0.972 [0.95–0.99] | 0.958 [0.93–0.98] | 0.964 [0.94–0.99] | 0.960 [0.93–0.98] |
| ABMIL | 0.973 [0.95–0.99] | 0.961 [0.94–0.98] | 0.976 [0.96–0.99] | 0.960 [0.93–0.98] |
| DSMIL | 0.976 [0.95–0.99] | 0.968 [0.95–0.99] | 0.973 [0.95–0.99] | 0.963 [0.93–0.99] |
| CLAM-SB | 0.975 [0.95–0.99] | 0.961 [0.94–0.98] | 0.978 [0.96–0.99] | 0.958 [0.93–0.98] |
| TransMIL | 0.972 [0.95–0.99] | 0.964 [0.94–0.98] | 0.973 [0.95–0.99] | 0.947 [0.92–0.97] |

## Test accuracy, seed ensemble [95% CI]
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.852 [0.78–0.92] | 0.824 [0.75–0.89] | 0.833 [0.77–0.90] | 0.833 [0.76–0.90] |
| MeanPool | 0.852 [0.78–0.92] | 0.824 [0.75–0.90] | 0.824 [0.75–0.89] | 0.815 [0.74–0.89] |
| ABMIL | 0.833 [0.76–0.90] | 0.796 [0.72–0.87] | 0.852 [0.78–0.92] | 0.824 [0.76–0.89] |
| DSMIL | 0.852 [0.79–0.92] | 0.833 [0.76–0.90] | 0.833 [0.76–0.90] | 0.852 [0.79–0.92] |
| CLAM-SB | 0.833 [0.76–0.91] | 0.815 [0.74–0.89] | 0.852 [0.79–0.92] | 0.796 [0.72–0.86] |
| TransMIL | 0.852 [0.79–0.92] | 0.833 [0.76–0.91] | 0.843 [0.78–0.91] | 0.796 [0.72–0.87] |

## CV balanced accuracy (mean over seeds; used for model selection)
| MIL model | ViT-S/16 Lunit | CTransPath | Phikon-v2 | PLIP |
|---|---|---|---|---|
| MaxPool | 0.811 | 0.809 | 0.804 | 0.755 |
| MeanPool | 0.790 | 0.787 | 0.765 | 0.764 |
| ABMIL | 0.812 | 0.791 | 0.789 | 0.748 |
| DSMIL | 0.803 | 0.777 | 0.778 | 0.747 |
| CLAM-SB | 0.804 | 0.808 | 0.796 | 0.739 |
| TransMIL | 0.767 | 0.742 | 0.721 | 0.667 |

## Significance: locked test set (108 slides)
0 of 54 comparisons significant after Holm.

| Family | A | B | A − B | 95% CI | p (Holm) | Sig. |
|---|---|---|---|---|---|---|
| encoder_overall | ctranspath | vits16_lunit | -0.018 | [-0.053, +0.023] | 0.703 |  |
| encoder_overall | phikon_v2 | vits16_lunit | -0.010 | [-0.057, +0.027] | 0.703 |  |
| encoder_overall | plip | vits16_lunit | -0.052 | [-0.128, +0.030] | 0.621 |  |
| model_overall | meanpool | maxpool | -0.014 | [-0.057, +0.029] | 1.000 |  |
| model_overall | abmil | maxpool | -0.003 | [-0.055, +0.040] | 1.000 |  |
| model_overall | dsmil | maxpool | -0.014 | [-0.079, +0.043] | 1.000 |  |
| model_overall | clam_sb | maxpool | -0.006 | [-0.057, +0.037] | 1.000 |  |
| model_overall | transmil | maxpool | -0.024 | [-0.076, +0.022] | 1.000 |  |

## Significance: CV predictions (430 slides)
10 of 54 comparisons significant after Holm.

| Family | A | B | A − B | 95% CI | p (Holm) | Sig. |
|---|---|---|---|---|---|---|
| encoder_vs_vits16 | ctranspath+dsmil | vits16_lunit+dsmil | -0.038 | [-0.067, -0.013] | 0.029 | ✅ |
| encoder_vs_vits16 | plip+dsmil | vits16_lunit+dsmil | -0.068 | [-0.110, -0.028] | 0.011 | ✅ |
| encoder_vs_vits16 | plip+clam_sb | vits16_lunit+clam_sb | -0.076 | [-0.119, -0.033] | 0.005 | ✅ |
| encoder_vs_vits16 | plip+transmil | vits16_lunit+transmil | -0.111 | [-0.161, -0.062] | 0.002 | ✅ |
| model_vs_maxpool | phikon_v2+meanpool | phikon_v2+maxpool | -0.058 | [-0.094, -0.026] | 0.006 | ✅ |
| capstone_claims | phikon_v2+maxpool | phikon_v2+meanpool | +0.058 | [+0.026, +0.094] | 0.002 | ✅ |
| encoder_overall | ctranspath | vits16_lunit | -0.029 | [-0.049, -0.011] | 0.002 | ✅ |
| encoder_overall | phikon_v2 | vits16_lunit | -0.029 | [-0.056, -0.004] | 0.023 | ✅ |
| encoder_overall | plip | vits16_lunit | -0.069 | [-0.097, -0.040] | 0.000 | ✅ |
| model_overall | meanpool | maxpool | -0.026 | [-0.047, -0.006] | 0.056 |  |
| model_overall | abmil | maxpool | -0.013 | [-0.029, +0.004] | 0.239 |  |
| model_overall | dsmil | maxpool | -0.019 | [-0.037, -0.001] | 0.106 |  |
| model_overall | clam_sb | maxpool | -0.011 | [-0.028, +0.007] | 0.239 |  |
| model_overall | transmil | maxpool | -0.034 | [-0.057, -0.012] | 0.009 | ✅ |

## Conclusions
1. **Model selection by CV is a tie** between ViT-S/16 + ABMIL (CV 0.812) and ViT-S/16 + MaxPool (CV 0.811).
   - ViT-S/16 + MaxPool: test bacc **0.834 [0.74–0.92]**, accuracy 0.852, AUROC 0.976, seed mean 0.838 ± 0.012.
   - ViT-S/16 + ABMIL: test bacc **0.811 [0.71–0.90]**, accuracy 0.833, AUROC 0.973, seed mean 0.823 ± 0.002.
2. **(⚠️ revised after tuning — see the tuning section below)** **Encoder: the small pathology-pretrained ViT-S/16 Lunit is best with default settings.** On CV it is significantly better than CTransPath (−0.029), **Phikon-v2 (−0.029, p = 0.023)** and PLIP (−0.069). The larger foundation model does not help on UBC-OCEAN.
3. **(⚠️ default settings only: MaxPool, MeanPool, CLAM-SB and TransMIL were never tuned, so the tuning caveat below applies here too)** **Aggregator: simple MaxPool is as good as the attention-based models** (ABMIL, DSMIL, CLAM-SB: no significant difference). **TransMIL is significantly worse** (−0.034 on CV); MeanPool is borderline worse (p = 0.056).
4. **The 108-slide test set cannot separate any pair** (0 of 54 significant). All test CIs are about ±0.09 wide. Report test CIs and use CV for comparisons.
5. **One seed is not enough:** with seed 0 only, "DSMIL < MaxPool" looked significant on CV (p = 0.011); with 3 seeds it is not (p = 0.106). Seed-to-seed std of test bacc reaches 0.03–0.04 for some configurations. The old capstone compared single runs.
6. Old capstone claims revisited: "DSMIL > ABMIL" is not supported (no significant difference either way). "MaxPool > MeanPool" holds significantly only for Phikon-v2 (+0.058 on CV).

## Interpretability and feature analysis (Kaggle notebook 04, 2026-10-28)
Source: `kaggle/results_05/figures/` (copied to `figures/`). Figures: `fig_heatmaps`, `fig_tsne`.

### Does ABMIL attention point at tumour? (pathologist masks)
ViT-S/16 + ABMIL, attention = mean percentile rank over models that never trained on the slide (test slides: all 15 models; CV slides: the 3 models whose validation fold held the slide). Tile = tumour if ≥50% of its pixels are red (tumour) in the supplemental mask; non-tumour if ≥50% green/blue (stroma/necrosis) and <10% red.

| | Value |
|---|---|
| Slides with masks / usable (≥5 tumour and ≥5 non-tumour tiles) | 152 / 138 (28 test, 110 CV) |
| **AUROC of attention, tumour vs non-tumour tiles** | **mean 0.968 [95% CI 0.949–0.983]**, median 0.999, IQR 0.994–1.000 |
| Slides with AUROC > 0.9 | 93% |
| Top-10% attention tiles that are tumour | 99.8% (vs 86.8% of all annotated tiles) |
| Test slides / CV slides | 0.956 (n = 28) / 0.970 (n = 110) |
| By subtype | MC 0.987, HGSC 0.982, EC 0.975, CC 0.946, LGSC 0.915 |

Lowest: 63165 (CC, test, 0.43; only 5 non-tumour tiles), 57162 (LGSC, 0.45), 44232 (EC, 0.55).
**Reading:** without any tile-level labels, the model's attention concentrates on the regions pathologists marked as tumour. This makes the professor's "qualitatively validated heatmap" deliverable quantitative. Caveats: masks are partial (unannotated tiles excluded); most annotated tiles are tumour (base rate 87%), which AUROC accounts for.

### Feature-space separability (t-SNE, silhouette, linear probe)
| Encoder | Silhouette, slide level (mean of tiles) | Silhouette, 1,000 patches | Linear-probe balanced acc. (slide means, 5-fold) |
|---|---|---|---|
| ViT-S/16 Lunit | 0.045 | 0.016 | 0.756 |
| CTransPath | 0.037 | 0.017 | 0.778 |
| Phikon-v2 | 0.021 | 0.011 | 0.775 |
| PLIP | 0.025 | −0.003 | 0.767 |

**Reading:**
- Subtypes overlap heavily in every encoder's raw feature space (silhouettes near 0); single patches carry little subtype signal (many tiles are stroma/background), which is why slide-level MIL aggregation is needed.
- The foundation model does **not** give clearer clusters (Phikon-v2 has the lowest slide-level silhouette), contrary to the roadmap's expectation.
- On simple mean features all encoders are equal (linear probe 0.76–0.78). ViT-S/16's advantage only appears once a trained MIL head (max/attention pooling) picks out the informative tiles.
- A small separate cluster appears in the slide-level plots of ViT-S/16 and Phikon-v2, mixing HGSC and LGSC. It is probably the 25 TMAs (different magnification and centre): a WSI/TMA domain shift. **To verify.**

## Hyperparameter tuning of ABMIL and DSMIL (Kaggle notebook 05, 2026-10-29)
Source: `kaggle/results_06/`. 520 search runs (13 configs × 5 folds × 2 models × 4 encoders, seed 0) + 120 confirm runs (best config × 3 seeds). 3.5 h on 2× T4. Protocol: `METHODS_NOTES.md` (selection on CV only).

### Configuration chosen by CV (seed 0)

| Encoder | Model | Chosen | lr | wd | dropout | width | class wts | CV default → chosen |
|---|---|---|---|---|---|---|---|---|
| CTransPath | ABMIL | c03 | 0.0005 | 0.01 | 0 | default | yes | 0.794 → 0.823 |
| CTransPath | DSMIL | c11 | 0.001 | 0.01 | 0.25 | small | no | 0.779 → 0.798 |
| Phikon-v2 | ABMIL | c08 | 0.0002 | 1e-05 | 0.25 | small | no | 0.772 → 0.808 |
| Phikon-v2 | DSMIL | c11 | 0.001 | 0.01 | 0.25 | small | no | 0.770 → 0.806 |
| PLIP | ABMIL | c05 | 0.0005 | 1e-05 | 0.5 | default | no | 0.752 → 0.791 |
| PLIP | DSMIL | c06 | 0.001 | 1e-05 | 0.5 | large | yes | 0.722 → 0.769 |
| ViT-S/16 | ABMIL | c00 | 0.0002 | 0.0001 | 0.25 | default | no | 0.809 → 0.809 |
| ViT-S/16 | DSMIL | c00 | 0.0002 | 0.0001 | 0.25 | default | no | 0.804 → 0.804 |

ViT-S/16: the default (c00) was already the best of 13 for both models. CV gains for the others are optimistic (best of 13 on the same folds).

### Tuned models on the locked test set (3 seeds, seed ensemble)

| Encoder | Model | Test bacc [95% CI] | seed mean ± std | AUROC | accuracy |
|---|---|---|---|---|---|
| CTransPath | ABMIL | 0.809 [0.72–0.89] | 0.820 ± 0.009 | 0.962 | 0.815 |
| Phikon-v2 | ABMIL | 0.872 [0.79–0.94] | 0.857 ± 0.010 | 0.978 | 0.880 |
| PLIP | ABMIL | 0.792 [0.70–0.88] | 0.777 ± 0.019 | 0.970 | 0.824 |
| ViT-S/16 | ABMIL | 0.811 [0.71–0.90] | 0.823 ± 0.002 | 0.973 | 0.833 |
| CTransPath | DSMIL | 0.857 [0.78–0.93] | 0.841 ± 0.009 | 0.974 | 0.870 |
| Phikon-v2 | DSMIL | 0.864 [0.78–0.94] | 0.847 ± 0.016 | 0.979 | 0.870 |
| PLIP | DSMIL | 0.767 [0.66–0.86] | 0.772 ± 0.006 | 0.962 | 0.824 |
| ViT-S/16 | DSMIL | 0.816 [0.71–0.91] | 0.830 ± 0.035 | 0.976 | 0.852 |

### Tuned vs default (paired bootstrap, Holm over 8 pairs per split)

| Split | Encoder | Model | Default | Tuned | Δ [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|
| test | Phikon-v2 | ABMIL | 0.834 | 0.872 | +0.038 [+0.000, +0.095] | 0.557 |
| test | Phikon-v2 | DSMIL | 0.786 | 0.864 | +0.078 [+0.010, +0.163] | 0.145 |
| test | CTransPath | ABMIL | 0.800 | 0.809 | +0.009 [+0.000, +0.023] | 1.000 |
| test | CTransPath | DSMIL | 0.800 | 0.857 | +0.057 [+0.004, +0.130] | 0.263 |
| test | ViT-S/16 | ABMIL | 0.811 | 0.811 | +0.000 [+0.000, +0.000] | 1.000 |
| test | ViT-S/16 | DSMIL | 0.816 | 0.816 | +0.000 [+0.000, +0.000] | 1.000 |
| test | PLIP | ABMIL | 0.789 | 0.792 | +0.004 [-0.025, +0.032] | 1.000 |
| test | PLIP | DSMIL | 0.788 | 0.767 | -0.020 [-0.046, +0.000] | 0.557 |
| cv | Phikon-v2 | ABMIL | 0.793 | 0.789 | -0.004 [-0.030, +0.024] | 1.000 |
| cv | Phikon-v2 | DSMIL | 0.783 | 0.813 | +0.030 [-0.000, +0.062] | 0.308 |
| cv | CTransPath | ABMIL | 0.786 | 0.822 | +0.036 [+0.011, +0.063] | 0.018 ✅ |
| cv | CTransPath | DSMIL | 0.784 | 0.817 | +0.033 [+0.010, +0.059] | 0.026 ✅ |
| cv | ViT-S/16 | ABMIL | 0.823 | 0.823 | +0.000 [+0.000, +0.000] | 1.000 |
| cv | ViT-S/16 | DSMIL | 0.822 | 0.822 | +0.000 [+0.000, +0.000] | 1.000 |
| cv | PLIP | ABMIL | 0.764 | 0.783 | +0.019 [-0.005, +0.045] | 0.602 |
| cv | PLIP | DSMIL | 0.754 | 0.777 | +0.023 [-0.005, +0.055] | 0.602 |

### Encoders after tuning: each vs tuned ViT-S/16 (paired bootstrap, Holm over 6 per split)

| Split | Comparison | Δ [95% CI] | p (Holm) |
|---|---|---|---|
| test | phikon_v2 (tuned) vs vits16 (tuned), abmil | +0.061 [+0.007, +0.131] | 0.133 |
| test | ctranspath (tuned) vs vits16 (tuned), abmil | -0.002 [-0.048, +0.058] | 1.000 |
| test | plip (tuned) vs vits16 (tuned), abmil | -0.019 [-0.107, +0.065] | 1.000 |
| test | phikon_v2 (tuned) vs vits16 (tuned), dsmil | +0.048 [-0.012, +0.120] | 0.718 |
| test | ctranspath (tuned) vs vits16 (tuned), dsmil | +0.041 [-0.011, +0.108] | 0.787 |
| test | plip (tuned) vs vits16 (tuned), dsmil | -0.049 [-0.151, +0.051] | 1.000 |
| cv | phikon_v2 (tuned) vs vits16 (tuned), abmil | -0.034 [-0.072, +0.004] | 0.372 |
| cv | ctranspath (tuned) vs vits16 (tuned), abmil | -0.001 [-0.028, +0.026] | 1.000 |
| cv | plip (tuned) vs vits16 (tuned), abmil | -0.040 [-0.085, +0.005] | 0.372 |
| cv | phikon_v2 (tuned) vs vits16 (tuned), dsmil | -0.009 [-0.048, +0.029] | 1.000 |
| cv | ctranspath (tuned) vs vits16 (tuned), dsmil | -0.005 [-0.035, +0.025] | 1.000 |
| cv | plip (tuned) vs vits16 (tuned), dsmil | -0.045 [-0.090, +0.001] | 0.327 |

### Reading
1. **Tuning did not significantly improve any model on the locked test set** (0 of 8 after Holm). Largest test gains: Phikon-v2 + DSMIL +0.078 (raw p = 0.018), CTransPath + DSMIL +0.057, Phikon-v2 + ABMIL +0.038. On CV, CTransPath ABMIL/DSMIL gains are significant, but the CV comparison is biased toward the tuned config.
2. **The default settings were already optimal for ViT-S/16**, and tuning mostly helped the other encoders (smaller heads, stronger regularisation for the 1024-d Phikon-v2).
3. **⚠️ Correction to conclusion 2 above:** after per-encoder tuning, **no encoder differs significantly from ViT-S/16** for ABMIL or DSMIL (0 of 12 comparisons). The earlier "ViT-S/16 significantly beats CTransPath and Phikon-v2" (default settings) was at least partly a hyperparameter effect. Claim for the paper: *encoder choice matters less than commonly assumed; with fair tuning, a small pathology ViT-S/16 performs on par with the larger Phikon-v2; PLIP trends lowest.*
4. Highest locked-test scores overall are now **tuned Phikon-v2 + ABMIL (0.872 [0.79–0.95], AUROC 0.978)** and **Phikon-v2 + DSMIL (0.864)**, but they are not significantly different from ViT-S/16 (0.81–0.82). Report these as "best observed", not "best".
5. Seed stability of tuned models is good (seed std 0.002–0.02, except ViT-S/16 + DSMIL 0.035).

## High-resolution close-ups (Kaggle notebook 06, 2026-10-30)
Figure: `figures/fig_heatmap_closeups` (ViT-S/16 + ABMIL, 15-model average; same 6 test slides as `fig_heatmaps`). Each row: overview, 1536 px zoom at ~10x, top-3 attended tiles (at least 3 tiles apart), lowest-attention tile.
Observation (not a pathologist reading; needs expert confirmation): most-attended tiles show dense epithelial, tumour-like structures in all 6 slides, including the misclassified LGSC slide 33976 (predicted CC); lowest-attention tiles show stroma and red blood cells.

## Inter-subtype confusion, EC vs HGSC (mentor request, 2026-11-06)
Figure: `figures/fig_confusion_normalized` (row-normalised 5-class matrices). Script: `code/figures/make_confusion.py`. Table for every configuration: `figures/confusion_ec_hgsc_all_configs.csv`.

| ViT-S/16 + MaxPool | EC → HGSC (% of true EC) | HGSC → EC (% of true HGSC) | Share of EC errors that go to HGSC |
|---|---|---|---|
| CV (430 slides, 3-seed mean) | **16.2%** (16/99) [95% Wilson CI 10.2–24.7] | 5.1% (9/177) [2.7–9.4] | 62% (16/26) |
| Locked test (108 slides) | **16.0%** (4/25) [6.4–34.7] | 4.4% (2/45) [1.2–14.8] | 57% (4/7) |

Across **all 32 trained configurations** (24 default + 8 tuned):
- CV: EC → HGSC median **20.2%** (range 15.2–27.3%); HGSC → EC median 5.6% (range ≈ 4–9%).
- EC → HGSC is the **single largest off-diagonal confusion in 31 of 32** configurations on CV (29 of 32 on test).
- The confusion is **asymmetric**: EC is called HGSC about 3–4× more often than the reverse, so models fall back to the majority class (HGSC = 41% of slides) when an EC slide is ambiguous.
- Other notable confusions (ViT-S/16 + MaxPool, CV): MC → EC 16% (6/37); LGSC → HGSC 13% (5/38) and LGSC → EC 11% (4/38).
- Clinically plausible: high-grade endometrioid and HGSC overlap morphologically (glandular/papillary architecture, high-grade nuclei), which pathologists also resolve with immunohistochemistry (e.g. WT1, p53). Wording for the paper to be checked against the literature.
