# Dataset table (Week 1, Step 3) — complete

Source: Kaggle UBC-OCEAN `train.csv` (538 images, official labels) + tile counts from the Week 2 extraction run (`kaggle/results_02/manifest_core.csv`, 2026-10-25). Patient IDs are not released, so counts are per image.
Tiles: 256×256 px after shrinking to ~10× (WSI ÷2, TMA ÷4); WSI stride 256, TMA stride 64; kept if ≥25% tissue.

| Subtype | Full name | WSIs | TMAs | Images | Tiles extracted | Median tiles / image |
|---|---|---|---|---|---|---|
| HGSC | High-Grade Serous Carcinoma | 217 | 5 | 222 | 580,260 | 2,322 |
| EC | Endometrioid Carcinoma | 119 | 5 | 124 | 354,436 | 2,670 |
| CC | Clear Cell Carcinoma | 94 | 5 | 99 | 319,284 | 2,873 |
| LGSC | Low-Grade Serous Carcinoma | 42 | 5 | 47 | 99,248 | 1,968 |
| MC | Mucinous Carcinoma | 41 | 5 | 46 | 161,755 | 3,464 |
| **Total** | | **513** | **25** | **538** | **1,514,983** | 2,507 |

Notes:
- HGSC is ~5× the size of LGSC or MC (class imbalance).
- Each subtype has exactly 5 TMAs. TMAs contribute only 1,991 tiles (0.1%).
- Per-slide Macenko succeeded on all 538 images.

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{UBC-OCEAN training set by subtype. Patient IDs are not released, so counts are per image. Tiles: 256$\times$256 px at $\sim$10$\times$ containing $\geq$25\% tissue.}
\label{tab:dataset}
\begin{tabular}{llrrrr}
\hline
Subtype & Full name & WSIs & TMAs & Images & Tiles \\
\hline
HGSC & High-Grade Serous Carcinoma & 217 & 5 & 222 & 580,260 \\
EC   & Endometrioid Carcinoma      & 119 & 5 & 124 & 354,436 \\
CC   & Clear Cell Carcinoma        &  94 & 5 &  99 & 319,284 \\
LGSC & Low-Grade Serous Carcinoma  &  42 & 5 &  47 &  99,248 \\
MC   & Mucinous Carcinoma          &  41 & 5 &  46 & 161,755 \\
\hline
Total &                            & 513 & 25 & 538 & 1,514,983 \\
\hline
\end{tabular}
\end{table}
```
