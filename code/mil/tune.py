"""Hyperparameter tuning for ABMIL and DSMIL, with model selection on validation data only.

Three phases per encoder x model (each phase skips work that is already done):
  1. search  : every config in CONFIGS, seed 0, 5 folds   -> <out>/search/<enc>/<model>/<cfg>/seed0/fold<k>
  2. select  : pooled out-of-fold (CV) balanced accuracy per config; best config wins (ties -> c00 = default)
  3. confirm : best config with seeds 1 and 2 (seed 0 copied from the search) -> <out>/tuned/<enc>/<model>/seed<s>/fold<k>
The locked test set is predicted by every run but is never used to choose anything.

    python -m mil.tune --features_root features --splits splits_v2 --out tune --encoders vits16_lunit,plip --device cuda:0
"""
import argparse
import glob
import json
import os
import shutil
import time
from types import SimpleNamespace

import pandas as pd
import torch

from .data import preload_features
from .grid import all_slide_ids, is_done
from .metrics import compute_metrics
from .train import train_fold

# 'size': small / default / large hidden width per model (ABMIL 128/256/512, DSMIL 64/128/256).
# c00 = the default used in the main benchmark; c01-c12 drawn at random (numpy default_rng(42)) from
# lr {5e-5,1e-4,2e-4,5e-4,1e-3} x wd {1e-5,1e-4,1e-3,1e-2} x dropout {0,0.1,0.25,0.5} x size x class_weights.
CONFIGS = {
    'c00': {'lr': 2e-4, 'wd': 1e-4, 'dropout': 0.25, 'size': 'default', 'class_weights': False},
    'c01': {'lr': 2e-4, 'wd': 1e-5, 'dropout': 0.25, 'size': 'default', 'class_weights': False},
    'c02': {'lr': 5e-5, 'wd': 1e-4, 'dropout': 0.25, 'size': 'large', 'class_weights': False},
    'c03': {'lr': 5e-4, 'wd': 1e-2, 'dropout': 0.0, 'size': 'default', 'class_weights': True},
    'c04': {'lr': 2e-4, 'wd': 1e-3, 'dropout': 0.25, 'size': 'small', 'class_weights': False},
    'c05': {'lr': 5e-4, 'wd': 1e-5, 'dropout': 0.5, 'size': 'default', 'class_weights': False},
    'c06': {'lr': 1e-3, 'wd': 1e-5, 'dropout': 0.5, 'size': 'large', 'class_weights': True},
    'c07': {'lr': 5e-5, 'wd': 1e-4, 'dropout': 0.25, 'size': 'large', 'class_weights': True},
    'c08': {'lr': 2e-4, 'wd': 1e-5, 'dropout': 0.25, 'size': 'small', 'class_weights': False},
    'c09': {'lr': 5e-5, 'wd': 1e-4, 'dropout': 0.5, 'size': 'default', 'class_weights': False},
    'c10': {'lr': 5e-4, 'wd': 1e-4, 'dropout': 0.5, 'size': 'default', 'class_weights': False},
    'c11': {'lr': 1e-3, 'wd': 1e-2, 'dropout': 0.25, 'size': 'small', 'class_weights': False},
    'c12': {'lr': 5e-5, 'wd': 1e-2, 'dropout': 0.5, 'size': 'large', 'class_weights': True},
}
HIDDEN = {'abmil': {'small': 128, 'default': 256, 'large': 512}, 'dsmil': {'small': 64, 'default': 128, 'large': 256}}
FIXED = dict(epochs=50, min_epochs=10, patience=10, accum=1, max_patches=8000, select='loss', cache=False)


def make_args(cfg, model, enc, seed, out, features_root, splits):
    c = CONFIGS[cfg]
    return SimpleNamespace(**FIXED, lr=c['lr'], wd=c['wd'], dropout=c['dropout'], hidden=HIDDEN[model][c['size']],
                           class_weights=c['class_weights'], features=os.path.join(features_root, enc),
                           splits=splits, model=model, seed=seed, out=out)


def cv_bacc(seed_dir):
    files = sorted(glob.glob(os.path.join(seed_dir, 'fold*', 'val_preds.csv')))
    if len(files) < 5:
        return None
    return compute_metrics(pd.concat([pd.read_csv(f, dtype={'image_id': str}) for f in files]))['bacc']


def select(out, enc, model):
    rows = []
    for cfg in CONFIGS:
        b = cv_bacc(os.path.join(out, 'search', enc, model, cfg, 'seed0'))
        if b is not None:
            rows.append(dict(encoder=enc, model=model, config=cfg, cv_bacc=b, **CONFIGS[cfg]))
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    best = df.sort_values(['cv_bacc', 'config'], ascending=[False, True]).iloc[0].config   # ties -> lower id (c00 first)
    return best, df


def compare(default_runs, tune_out, encoders, models=('abmil', 'dsmil'), n_boot=10000):
    """Tuned (3 seeds) vs default (3 seeds, main benchmark) per encoder x model, on the test set and on CV.

    Paired bootstrap of balanced accuracy, Holm-corrected within each split. Returns a DataFrame."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from .stats import holm, load_config_preds, paired_test
    rows = []
    for split in ('test', 'cv'):
        for e in encoders:
            for m in models:
                d = load_config_preds(os.path.join(default_runs, e, m), split)
                t = load_config_preds(os.path.join(tune_out, 'tuned', e, m), split)
                if d is None or t is None:
                    continue
                assert (d[2] == t[2]).all() and (d[0] == t[0]).all(), f'{e}/{m}: slides differ'
                obs, lo, hi, p = paired_test(d[0], t[1], d[1], 'bacc', n_boot)
                cfg = json.load(open(os.path.join(tune_out, 'tuned', e, m, 'config.json')))['config']
                rows.append(dict(split=split, encoder=e, model=m, config=cfg,
                                 default_bacc=balanced_accuracy_score(d[0], d[1]), tuned_bacc=balanced_accuracy_score(t[0], t[1]),
                                 diff=obs, diff_lo=lo, diff_hi=hi, p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df['p_holm'] = np.nan
        for s, g in df.groupby('split'):
            df.loc[g.index, 'p_holm'] = holm(g.p.to_numpy())
        df['significant'] = df.p_holm < 0.05
    return df


def main():
    p = argparse.ArgumentParser(description='Tune ABMIL/DSMIL on CV, confirm best config with 3 seeds')
    p.add_argument('--features_root', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--encoders', required=True)
    p.add_argument('--models', default='abmil,dsmil')
    p.add_argument('--configs', default=','.join(CONFIGS))
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--time_budget_h', type=float, default=1e9)
    p.add_argument('--prev', default='', help='comma-separated earlier tune output roots to copy finished runs from')
    p.add_argument('--epochs', type=int, default=None, help='override (for quick tests)')
    a = p.parse_args()
    if a.epochs:
        FIXED.update(epochs=a.epochs, min_epochs=1, patience=2)

    t0 = time.time()
    over = lambda: time.time() - t0 > a.time_budget_h * 3600
    encoders, models, cfgs = a.encoders.split(','), a.models.split(','), a.configs.split(',')
    prev = [r for r in a.prev.split(',') if r]
    ids = all_slide_ids(a.splits)
    device = torch.device(a.device)

    def run(rel, cfg, model, enc, seed, fold, cache):
        d = os.path.join(a.out, rel, f'seed{seed}', f'fold{fold}')
        if is_done(d):
            return False
        for root in prev:
            src = os.path.join(root, rel, f'seed{seed}', f'fold{fold}')
            if is_done(src):
                shutil.copytree(src, d, dirs_exist_ok=True)
                return False
        t = time.time()
        train_fold(make_args(cfg, model, enc, seed, os.path.join(a.out, rel, f'seed{seed}'), a.features_root, a.splits),
                   fold, device, cache)
        print(f'== {rel}/seed{seed}/fold{fold} done in {(time.time() - t) / 60:.1f} min | elapsed {(time.time() - t0) / 3600:.2f} h', flush=True)
        return True

    for enc in encoders:
        if over():
            break
        cache = None
        for model in models:
            # 1. search
            for cfg in cfgs:
                for fold in range(1, 6):
                    if over():
                        break
                    rel = os.path.join('search', enc, model, cfg)
                    if not is_done(os.path.join(a.out, rel, 'seed0', f'fold{fold}')) and cache is None:
                        cache = preload_features(os.path.join(a.features_root, enc), ids, device)
                        print(f'[{enc}] features loaded to {device}', flush=True)
                    run(rel, cfg, model, enc, 0, fold, cache)
            # 2. select
            best, table = select(a.out, enc, model)
            if best is None or len(table) < len(cfgs):
                print(f'[{enc}/{model}] search incomplete ({len(table)}/{len(cfgs)} configs): not selecting yet', flush=True)
                continue
            os.makedirs(os.path.join(a.out, 'selection'), exist_ok=True)
            table.to_csv(os.path.join(a.out, 'selection', f'{enc}_{model}.csv'), index=False)
            print(f'[{enc}/{model}] best config {best} (CV bacc {table.cv_bacc.max():.3f}; default c00 '
                  f'{table[table.config == "c00"].cv_bacc.iloc[0]:.3f})', flush=True)
            # 3. confirm: seed 0 from the search, seeds 1-2 trained
            tuned = os.path.join(a.out, 'tuned', enc, model)
            os.makedirs(tuned, exist_ok=True)
            with open(os.path.join(tuned, 'config.json'), 'w') as f:
                json.dump({'config': best, **CONFIGS[best], 'hidden': HIDDEN[model][CONFIGS[best]['size']]}, f, indent=2)
            src0 = os.path.join(a.out, 'search', enc, model, best, 'seed0')
            if not os.path.exists(os.path.join(tuned, 'seed0')):
                shutil.copytree(src0, os.path.join(tuned, 'seed0'))
            for seed in (1, 2):
                for fold in range(1, 6):
                    if over():
                        break
                    rel = os.path.join('tuned', enc, model)
                    if not is_done(os.path.join(a.out, rel, f'seed{seed}', f'fold{fold}')) and cache is None:
                        cache = preload_features(os.path.join(a.features_root, enc), ids, device)
                    run(rel, best, model, enc, seed, fold, cache)
        del cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    n_search = sum(len(glob.glob(os.path.join(a.out, 'search', e, m, c, 'seed0', 'fold*', 'test_preds.csv')))
                   for e in encoders for m in models for c in cfgs)
    n_tuned = sum(len(glob.glob(os.path.join(a.out, 'tuned', e, m, 'seed*', 'fold*', 'test_preds.csv')))
                  for e in encoders for m in models)
    want_s, want_t = len(encoders) * len(models) * len(cfgs) * 5, len(encoders) * len(models) * 15
    print(f'TUNE FINISHED: search {n_search}/{want_s}, tuned {n_tuned}/{want_t}'
          + ('' if (n_search, n_tuned) == (want_s, want_t) else ' -> run again with this output attached to continue'), flush=True)


if __name__ == '__main__':
    main()
