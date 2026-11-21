"""Summarize finished runs into one results table.

For each run folder (containing fold1/ ... fold5/ from mil.train):
  * CV  = pooled out-of-fold validation predictions (used for model selection only)
  * Test = the 5 fold models' probabilities averaged per slide on the locked test set,
           with 95% bootstrap confidence intervals. This is the number to report.

Example:
    python -m mil.evaluate runs/uni/*/seed0 --out results/summary.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, compute_metrics, ensemble

MAIN = ['acc', 'bacc', 'macro_f1', 'macro_auc']


def summarize_run(run_dir, n_boot=1000):
    folds = sorted(glob.glob(os.path.join(run_dir, 'fold*')))
    folds = [f for f in folds if os.path.exists(os.path.join(f, 'test_preds.csv'))]
    if not folds:
        raise FileNotFoundError(f'no finished folds in {run_dir}')

    val = pd.concat([pd.read_csv(os.path.join(f, 'val_preds.csv'), dtype={'image_id': str}) for f in folds])
    tests = [pd.read_csv(os.path.join(f, 'test_preds.csv'), dtype={'image_id': str}) for f in folds]
    test_ens = ensemble(tests)
    test_ens.to_csv(os.path.join(run_dir, 'test_preds_ensemble.csv'), index=False)

    cv = compute_metrics(val)
    te = compute_metrics(test_ens)
    ci = bootstrap_ci(test_ens, MAIN, n_boot=n_boot) if n_boot else {m: (np.nan, np.nan) for m in MAIN}
    per_fold = pd.DataFrame([compute_metrics(t) for t in tests])

    row = {'run': run_dir, 'n_folds': len(folds), 'n_test': te['n']}
    row.update({f'cv_{m}': cv[m] for m in MAIN})
    for m in MAIN:
        row[f'test_{m}'] = te[m]
        row[f'test_{m}_lo'], row[f'test_{m}_hi'] = ci[m]
        row[f'foldtest_{m}_mean'] = per_fold[m].mean()
        row[f'foldtest_{m}_std'] = per_fold[m].std()
    row.update({f'test_{k}': v for k, v in te.items() if k.startswith('recall_')})
    if 'is_tma' in test_ens:
        for name, sub in [('wsi', test_ens[test_ens.is_tma == 0]), ('tma', test_ens[test_ens.is_tma == 1])]:
            if len(sub):
                s = compute_metrics(sub)
                row[f'test_{name}_n'], row[f'test_{name}_acc'], row[f'test_{name}_bacc'] = s['n'], s['acc'], s['bacc']

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(row, f, indent=2, default=float)
    return row


def summarize_config(config_dir, n_boot=1000):
    """Aggregate all seeds of one encoder+model folder (<config_dir>/seed*/fold*).

    Returns mean ± std across seeds of each seed's test metrics, plus the SEED ENSEMBLE:
    probabilities of all seeds x folds averaged per test slide (saved as test_preds_seedensemble.csv), with 95% CI.
    """
    seed_dirs = sorted(d for d in glob.glob(os.path.join(config_dir, 'seed*'))
                       if glob.glob(os.path.join(d, 'fold*', 'test_preds.csv')))
    per_seed = pd.DataFrame([summarize_run(d, n_boot=0) for d in seed_dirs])
    tests = [pd.read_csv(f, dtype={'image_id': str}) for d in seed_dirs
             for f in sorted(glob.glob(os.path.join(d, 'fold*', 'test_preds.csv')))]
    ens = ensemble(tests)
    ens.to_csv(os.path.join(config_dir, 'test_preds_seedensemble.csv'), index=False)
    te = compute_metrics(ens)
    ci = bootstrap_ci(ens, MAIN, n_boot=n_boot) if n_boot else {m: (np.nan, np.nan) for m in MAIN}
    parts = os.path.normpath(config_dir).split(os.sep)
    row = {'encoder': parts[-2], 'model': parts[-1], 'n_seeds': len(seed_dirs), 'n_models': len(tests)}
    for m in MAIN:
        row[f'seedmean_{m}'] = per_seed[f'test_{m}'].mean()
        row[f'seedstd_{m}'] = per_seed[f'test_{m}'].std() if len(per_seed) > 1 else np.nan
        row[f'cv_{m}_mean'] = per_seed[f'cv_{m}'].mean()
        row[f'ens_{m}'] = te[m]
        row[f'ens_{m}_lo'], row[f'ens_{m}_hi'] = ci[m]
    row.update({f'ens_{k}': v for k, v in te.items() if k.startswith('recall_')})
    return row


def main():
    p = argparse.ArgumentParser(description='Summarize MIL runs')
    p.add_argument('runs', nargs='+', help='run folders (each with fold1..fold5)')
    p.add_argument('--out', default=None, help='CSV path for the combined table')
    p.add_argument('--n_boot', type=int, default=1000)
    args = p.parse_args()

    rows = [summarize_run(r, args.n_boot) for r in args.runs]
    table = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        table.to_csv(args.out, index=False)

    def fmt(r, m):
        return f"{r[f'test_{m}']:.3f} [{r[f'test_{m}_lo']:.3f}-{r[f'test_{m}_hi']:.3f}]"

    print(f"{'run':45s} {'CV bacc':>8s}  {'test bacc [95% CI]':>22s}  {'test F1 [95% CI]':>22s}  {'test AUC [95% CI]':>22s}")
    for r in rows:
        print(f"{r['run'][-45:]:45s} {r['cv_bacc']:8.3f}  {fmt(r, 'bacc'):>22s}  {fmt(r, 'macro_f1'):>22s}  {fmt(r, 'macro_auc'):>22s}")


if __name__ == '__main__':
    main()
