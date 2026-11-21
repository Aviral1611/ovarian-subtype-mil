# Test run 01 findings (2026-10-16)

Notebook: `../01_test_patching_macenko.ipynb`, run on Kaggle with a GPU.

## Measured on 3 slides
| Slide | Type | Label | Size after shrink | Tiles | Load | Macenko | Macenko fails |
|---|---|---|---|---|---|---|---|
| 61320 | WSI | EC | 21536×18311 | 2,816 | 44 s | 46.5 s | 1 |
| 35239 | WSI | HGSC | 22276×30783 | 5,611 | 83 s | 87.4 s | 9 |
| 29084 | TMA | LGSC | 847×847 | 93 | 0.8 s | 1.7 s | 0 |

GPU speed: ViT-S/16 Lunit **1,212 tiles/s**; ViT-L/16 (UNI-size) **151 tiles/s**.

## Whole-dataset estimate
- ~**1.64 M tiles** (almost all WSI; TMAs ≈ 2k)
- CPU: load 6.9 h + tiling 0.2 h + Macenko 7.3 h = **14.3 h**
- GPU: ViT-S 0.4 h, UNI-size 3.0 h

## Visual checks
- ✅ **Tissue detection is good.** Tiles cover the tissue and skip the black padding and white background, on both WSI and TMA.
- ✅ The colour fix visibly evens out colour across slides (hot-pink HGSC stroma toned down, darker TMA brought into line).
- ❌ **Silent failure 1:** slide 35239, tile 2 (mostly empty, pale mucoid area). The output turned into a grey-blue wash with bright cracks. It was not counted as a failure because no error was raised.
- ❌ **Silent failure 2:** slide 61320, tile 3 (pink stroma, few nuclei). Eosin pink was lost and the tile turned grey-blue.
- Cause of both: Macenko estimates the stain colours **per tile**. That breaks on tiles that contain mostly one stain or almost none.
- Reference patch: acceptable but slightly pale and lavender, so every output shifts toward lavender.

## Decision for Week 2
**Fit Macenko once per slide, not once per tile:**
1. Pool pixels from ~200 random tissue tiles of the slide → estimate that slide's stain matrix and max concentrations.
2. Apply that fixed transform to every tile of the slide, on the GPU in batches.

This fixes both silent failures (single-stain and empty tiles inherit the slide's estimate) and cuts Macenko from ~7 h to minutes (it becomes batched matrix maths).

Also for Week 2: read each slide **once** and run all encoders in the same pass, and load 2–3 slides in parallel on Kaggle's 4 CPU cores. Expected total ≈ 8–10 h for all encoders.
