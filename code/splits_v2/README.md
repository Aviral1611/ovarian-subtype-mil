# splits_v2

Made by `make_splits.py --seed 42 --test-size 0.2` from `data/kaggle_train.csv`.
Labels: {0: 'CC', 1: 'EC', 2: 'HGSC', 3: 'LGSC', 4: 'MC'}. No patient IDs exist in UBC-OCEAN, so splits are per image.
**Never use test.csv for model selection or tuning.**

| Split | CC | EC | HGSC | LGSC | MC | TMA | Total |
|---|---|---|---|---|---|---|---|
| All | 99 | 124 | 222 | 47 | 46 | 25 | 538 |
| Test (locked) | 20 | 25 | 45 | 9 | 9 | 5 | 108 |
| Fold 1 train | 63 | 79 | 141 | 31 | 30 | 16 | 344 |
| Fold 1 val | 16 | 20 | 36 | 7 | 7 | 4 | 86 |
| Fold 2 train | 63 | 80 | 141 | 31 | 29 | 16 | 344 |
| Fold 2 val | 16 | 19 | 36 | 7 | 8 | 4 | 86 |
| Fold 3 train | 63 | 79 | 142 | 30 | 30 | 16 | 344 |
| Fold 3 val | 16 | 20 | 35 | 8 | 7 | 4 | 86 |
| Fold 4 train | 64 | 79 | 142 | 30 | 29 | 16 | 344 |
| Fold 4 val | 15 | 20 | 35 | 8 | 8 | 4 | 86 |
| Fold 5 train | 63 | 79 | 142 | 30 | 30 | 16 | 344 |
| Fold 5 val | 16 | 20 | 35 | 8 | 7 | 4 | 86 |
