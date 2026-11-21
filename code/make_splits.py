"""Week 1, Step 1: leakage-safe splits (fixes ISSUES M1, M2).

UBC-OCEAN has no patient IDs, so we split by image:
  1. Lock away a stratified 20% test set. It is never used to pick models.
  2. Split the remaining 80% into 5 stratified CV folds (train/val) for model selection.
Stratification key = subtype + WSI/TMA, so every fold has the same mix of both.

Output (same CSV format as the old split/, so NpyDataset still works):
  splits_v2/test.csv
  splits_v2/train/{1..5}.csv
  splits_v2/val/{1..5}.csv

Usage:
  python make_splits.py
  python make_splits.py --labels data/kaggle_train.csv --out splits_v2 --seed 42 --test-size 0.2
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Must match label ids used by the model checkpoints (alphabetical).
CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'image_id': str})
    df = df[df['label'].isin(CLASSES)]  # drop 'Other' if a future dataset version has it
    out = pd.DataFrame({
        'file_name': df['image_id'],
        'label': df['label'].map(CLASSES.index),
        'is_tma': df['is_tma'].astype(str).str.lower().eq('true').astype(int),
    })
    return out.reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size: float, n_folds: int, seed: int):
    strata = df['label'].astype(str) + '_' + df['is_tma'].astype(str)
    dev, test = train_test_split(df, test_size=test_size, stratify=strata, random_state=seed)
    dev = dev.reset_index(drop=True)

    dev_strata = dev['label'].astype(str) + '_' + dev['is_tma'].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [(dev.iloc[tr], dev.iloc[va]) for tr, va in skf.split(dev, dev_strata)]
    return test, folds


def check(df, test, folds):
    test_ids = set(test['file_name'])
    val_ids = []
    for k, (tr, va) in enumerate(folds, 1):
        assert not set(tr['file_name']) & set(va['file_name']), f'fold {k}: train/val overlap'
        assert not set(tr['file_name']) & test_ids, f'fold {k}: train/test overlap'
        assert not set(va['file_name']) & test_ids, f'fold {k}: val/test overlap'
        val_ids += list(va['file_name'])
    assert len(val_ids) == len(set(val_ids)), 'an image is in two val folds'
    assert set(val_ids) | test_ids == set(df['file_name']), 'some images are in no split'


def summary(df, test, folds) -> str:
    def counts(d):
        return [int((d['label'] == i).sum()) for i in range(len(CLASSES))] + [int(d['is_tma'].sum()), len(d)]

    cols = CLASSES + ['TMA', 'Total']
    rows = [('All', counts(df)), ('Test (locked)', counts(test))]
    for k, (tr, va) in enumerate(folds, 1):
        rows.append((f'Fold {k} train', counts(tr)))
        rows.append((f'Fold {k} val', counts(va)))
    lines = ['| Split | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
    lines += [f'| {name} | ' + ' | '.join(map(str, c)) + ' |' for name, c in rows]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Make leakage-safe train/val/test splits')
    parser.add_argument('--labels', default='data/kaggle_train.csv')
    parser.add_argument('--out', default='splits_v2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    df = load_labels(args.labels)
    test, folds = make_splits(df, args.test_size, args.folds, args.seed)
    check(df, test, folds)

    os.makedirs(f'{args.out}/train', exist_ok=True)
    os.makedirs(f'{args.out}/val', exist_ok=True)
    test.sort_values('file_name').to_csv(f'{args.out}/test.csv', index=False)
    for k, (tr, va) in enumerate(folds, 1):
        tr.sort_values('file_name').to_csv(f'{args.out}/train/{k}.csv', index=False)
        va.sort_values('file_name').to_csv(f'{args.out}/val/{k}.csv', index=False)

    table = summary(df, test, folds)
    with open(f'{args.out}/README.md', 'w', encoding='utf-8') as f:
        f.write(f'# splits_v2\n\nMade by `make_splits.py --seed {args.seed} --test-size {args.test_size}` '
                f'from `{args.labels}`.\nLabels: {dict(enumerate(CLASSES))}. '
                'No patient IDs exist in UBC-OCEAN, so splits are per image.\n'
                '**Never use test.csv for model selection or tuning.**\n\n' + table + '\n')
    print(table)
    print(f'\nAll checks passed. Wrote {args.out}/')


if __name__ == '__main__':
    main()
