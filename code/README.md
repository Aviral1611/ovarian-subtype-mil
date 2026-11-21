# MIL code for UBC-OCEAN subtype classification

Clean rewrite (October 2026) of the capstone's training code. The old repo (`E:\UBC-OCEAN-7th`) stays as reference only. Its bugs B1 (ABMIL), B4 (early stop) and M1 (no test set) are fixed here.

## Layout
```
code/
├── make_splits.py          leakage-safe splits (copied from the repo; regenerates splits_v2/ identically)
├── data/kaggle_train.csv   official Kaggle labels, 538 slides
├── splits_v2/              locked test.csv (108) + train/val folds 1-5 (430)
├── mil/
│   ├── models.py           maxpool, meanpool, abmil, dsmil, clam_sb, transmil
│   ├── data.py             BagDataset: <features_dir>/<slide_id>.npy, shape (N, D)
│   ├── train.py            train one model on fold k or all folds, predict the locked test set
│   ├── metrics.py          acc, balanced acc, macro F1, macro AUROC, per-class recall, bootstrap CI
│   └── evaluate.py         summarize runs: CV (out-of-fold) + test (5-fold ensemble, 95% CI)
└── tests/make_fake_features.py   fake bags for testing without real data
```

## Rules the code enforces
- The best epoch is chosen on the fold's **validation** set only (default: lowest val loss).
- The **test** set is predicted once per fold, with the chosen checkpoint. The reported test score is the average of the 5 fold models' probabilities, with 95% bootstrap CIs.
- Seeds are fixed (`--seed`, + fold number).

## Run
```bash
pip install torch pandas scikit-learn nystrom-attention einops

# train one model on all 5 folds
python -m mil.train --features features/uni --splits splits_v2 --model abmil --fold all --out runs/uni/abmil/seed0

# summarize one or many runs into a table
python -m mil.evaluate runs/uni/*/seed0 --out results/uni.csv
```

Many runs at once (used by the Kaggle notebook `kaggle/03_train_mil.ipynb`): each encoder's features are loaded onto the device once and shared by all its runs; finished runs are skipped; `--prev` copies finished runs from earlier outputs; `--time_budget_h` stops cleanly.
```bash
python -m mil.grid --features_root features --splits splits_v2 --out runs --encoders phikon_v2,vits16_lunit --device cuda:0
```
`splits_v2/assignment.csv` is the whole split in one table (fold 0 = locked test, fold k = validation of fold k); the notebook unpacks it.

Useful flags: `--class_weights` (imbalance ablation), `--select bacc` (pick the epoch by balanced accuracy instead of loss), `--max_patches 8000` (random cap per training bag), `--cache` (keep features in RAM).

## Models
| name | what | heatmap (`attn`) |
|---|---|---|
| `maxpool` / `meanpool` | MLP per patch → max / mean → linear. Layer names match the old capstone checkpoints. | max: share of features each patch wins · mean: none |
| `abmil` | Gated attention MIL (Ilse 2018), **fixed**: softmax over patches | attention weights |
| `dsmil` | Dual-stream MIL (Li 2021); loss = ½ bag CE + ½ max-instance CE, as in the capstone | attention to the predicted class |
| `clam_sb` | CLAM single-branch (Lu 2021); 0.7 bag + 0.3 instance-clustering loss, k = 8 | attention weights |
| `transmil` | TransMIL (Shao 2021): Nyström attention + PPEG | exact class-token attention row, last layer |

## Testing without real data
```bash
python tests/make_fake_features.py --splits splits_v2 --out fake_features
python -m mil.train --features fake_features --splits splits_v2 --model abmil --fold 1 --epochs 15 --out runs/fake/abmil/seed0
python -m mil.evaluate runs/fake/abmil/seed0
```
