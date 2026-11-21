# Methods notes (for writing the paper)

Every methods decision, with the reason and the evidence. Add to this as you go.

## Data and splits (Week 1)
- UBC-OCEAN Kaggle training set only: 538 images (513 WSI, 25 TMA), 5 subtypes (CC, EC, HGSC, LGSC, MC). Official labels from Kaggle `train.csv`.
- No patient IDs are released, so splits are per image. A stratified 20% locked test set (108 images: 103 WSI, 5 TMA); the remaining 430 go into 5 stratified CV folds. Stratified by subtype × WSI/TMA. Seed 42. `make_splits.py`.
- Model selection uses the validation fold only. Reported test metrics come from the average of the 5 fold models' probabilities on the locked test set, with 95% percentile bootstrap CIs (1,000 resamples over slides).

## Tiling (same as the capstone)
- WSIs (20×) shrunk 2× and TMAs (40×) shrunk 4× → both ~10×. Box-filter shrink with pyvips.
- 256×256 tiles; stride 256 (WSI, no overlap), 64 (TMA). Keep a tile if ≥25% of its pixels are tissue (0 < gray ≤ 220).
- Whole tile resized to 224×224 (bilinear, antialiased) for every encoder; each encoder's own mean/std.

## Stain normalization (Week 1 test + Week 2)
- Macenko, fitted **once per slide** on tissue pixels pooled from up to 200 random tiles (≤2M pixels), then applied to every tile of that slide on the GPU.
  - Why per slide: per-tile Macenko silently produced artefacts on single-stain or near-empty tiles in test run 01 (`kaggle/results_01/FINDINGS.md`).
- Target = the standard Macenko reference (HE = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], maxC = [1.9705, 1.0308]; Macenko et al. 2009, torchstain/StainTools defaults).
  - Why not a hand-picked tile: the auto-picked tile from test run 01 was pale (maxC 0.79/0.65) and made all outputs lavender.
- β (optical-density threshold) = 0.03, not the usual 0.15, and the fit uses tissue pixels only.
  - Evidence: physically simulated restaining (Beer-Lambert: shifted stain vectors × stain amounts 0.35–1.6). With β=0.15 the faded variants failed (eosin lost, error 50 grey levels, or no fit at all). With β=0.03 on tissue pixels, all variants came within ≤8 grey levels of the reference result. Figure: `kaggle/macenko_fix_check.png`.
- Background pixels (gray > 220 or black padding) are left unchanged, to avoid amplifying glass noise into fake texture.
- Our implementation matches torchstain exactly (fit identical; transfer within 1 grey level).

## Encoders (Week 2)
| Encoder | Dim | Source | Notes |
|---|---|---|---|
| ViT-S/16 Lunit-DINO | 384 | `hf-hub:1aurent/vit_small_patch16_224.lunit_dino` | capstone baseline |
| CTransPath | 768 | `hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath` (ConvStem) | capstone baseline |
| Phikon-v2 | 1024 | `owkin/phikon-v2` (CLS token) | main foundation model; ViT-L |
| PLIP | 512 | `vinid/plip` (`get_image_features`) | vision-language model |
| Prov-GigaPath (optional) | 1536 | `hf_hub:prov-gigapath/prov-gigapath` | gated click-through; separate pass |

UNI/CONCH were not used: they need institutional-email access. The research guide approved the open alternatives (2026-10-22).

## MIL aggregators (Week 3 code)
maxpool, meanpool, ABMIL (gated attention, corrected), DSMIL, CLAM-SB (k=8, 0.7/0.3 loss), TransMIL (Nyström). AdamW lr 2e-4, wd 1e-4, ≤50 epochs, early stopping on validation loss (patience 10, min 10 epochs), one bag per step, random cap of 8,000 tiles per training bag.

## Hyperparameter tuning of ABMIL and DSMIL (roadmap item; notebook `kaggle/05_tune.ipynb`, code `code/mil/tune.py`)
- Search space: learning rate {5e-5, 1e-4, 2e-4, 5e-4, 1e-3} × weight decay {1e-5, 1e-4, 1e-3, 1e-2} × dropout {0, 0.1, 0.25, 0.5} × hidden width {small, default, large} (ABMIL 128/256/512, DSMIL 64/128/256) × class-weighted loss {no, yes}.
- 13 configurations per model: the benchmark default (c00) + 12 drawn uniformly at random (numpy `default_rng(42)`), fixed in code.
- Each configuration: 5-fold CV, seed 0, same training schedule as the benchmark (≤50 epochs, early stopping on validation loss).
- **Selection uses pooled out-of-fold validation balanced accuracy only**; ties go to the default. The locked test set is predicted by every run but never used for selection.
- The selected configuration is retrained with seeds 1 and 2 (seed 0 reused) and compared with the default (also 3 seeds) by paired bootstrap on the test set and on CV, Holm-corrected over the 8 encoder × model pairs.
- Note: the selected configuration's CV score is optimistically biased (best of 13 on the same folds); the test-set comparison is the unbiased one.
