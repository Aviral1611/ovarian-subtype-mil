"""Run many MIL trainings in one process: every encoder x model x seed x fold.

Each encoder's features are loaded ONCE onto the device and shared by all its runs.
A run is "done" when <out>/<encoder>/<model>/seed<s>/fold<k>/test_preds.csv exists; done runs are
skipped, and finished runs found in --prev roots (earlier Kaggle versions) are copied instead of redone.
No new run starts after --time_budget_h.

    python -m mil.grid --features_root features --splits splits_v2 --out runs \
        --encoders vits16_lunit,plip --models maxpool,abmil --device cuda:0
"""
import argparse
import os
import shutil
import time
from types import SimpleNamespace

import pandas as pd
import torch

from .data import preload_features
from .train import train_fold

DEFAULTS = dict(epochs=50, min_epochs=10, patience=10, lr=2e-4, wd=1e-4, accum=1, max_patches=8000,
                select='loss', class_weights=False, cache=False)


def run_dir(out, enc, model, seed, fold):
    return os.path.join(out, enc, model, f'seed{seed}', f'fold{fold}')


def is_done(d):
    return os.path.exists(os.path.join(d, 'test_preds.csv'))


def all_slide_ids(splits):
    files = [os.path.join(splits, 'test.csv')] + [os.path.join(splits, s, f'{k}.csv') for s in ('train', 'val') for k in range(1, 6)]
    return sorted(set(pd.concat([pd.read_csv(f, dtype={'file_name': str}) for f in files]).file_name))


def main():
    p = argparse.ArgumentParser(description='Grid of MIL trainings')
    p.add_argument('--features_root', required=True, help='folder containing <encoder>/<slide_id>.npy')
    p.add_argument('--splits', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--encoders', required=True, help='comma-separated')
    p.add_argument('--models', default='maxpool,meanpool,abmil,dsmil,clam_sb,transmil')
    p.add_argument('--folds', default='1,2,3,4,5')
    p.add_argument('--seeds', default='0')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--time_budget_h', type=float, default=1e9)
    p.add_argument('--prev', default='', help='comma-separated earlier output roots (with <encoder>/... inside)')
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f'--{k}', action='store_true', default=v)
        else:
            p.add_argument(f'--{k}', type=type(v), default=v)
    a = p.parse_args()

    t0 = time.time()
    encoders, models = a.encoders.split(','), a.models.split(',')
    folds = [int(f) for f in a.folds.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    prev = [r for r in a.prev.split(',') if r]
    ids = all_slide_ids(a.splits)
    device = torch.device(a.device)
    total = len(encoders) * len(models) * len(seeds) * len(folds)
    n = 0

    for enc in encoders:
        todo = [(m, s, f) for m in models for s in seeds for f in folds if not is_done(run_dir(a.out, enc, m, s, f))]
        # reuse finished runs from earlier versions
        for m, s, f in list(todo):
            for root in prev:
                src = run_dir(root, enc, m, s, f)
                if is_done(src):
                    shutil.copytree(src, run_dir(a.out, enc, m, s, f), dirs_exist_ok=True)
                    todo.remove((m, s, f))
                    print(f'[{enc}] copied {m} seed{s} fold{f} from {root}', flush=True)
                    break
        n += (len(models) * len(seeds) * len(folds)) - len(todo)
        if not todo:
            print(f'[{enc}] all runs done', flush=True)
            continue
        if time.time() - t0 > a.time_budget_h * 3600:
            break

        t_load = time.time()
        cache = preload_features(os.path.join(a.features_root, enc), ids, device)
        gb = sum(x.numel() * x.element_size() for x in cache.values()) / 1e9
        print(f'[{enc}] loaded {len(cache)} bags ({gb:.2f} GB) to {device} in {time.time() - t_load:.0f}s', flush=True)

        for m, s, f in todo:
            if time.time() - t0 > a.time_budget_h * 3600:
                print(f'time budget reached ({a.time_budget_h} h): stopping before {enc}/{m}/seed{s}/fold{f}', flush=True)
                break
            t_run = time.time()
            args = SimpleNamespace(**{k: getattr(a, k) for k in DEFAULTS}, features=os.path.join(a.features_root, enc),
                                   splits=a.splits, model=m, seed=s, out=os.path.join(a.out, enc, m, f'seed{s}'))
            train_fold(args, f, device, feature_cache=cache)
            n += 1
            el = time.time() - t0
            print(f'== [{n}/{total}] {enc}/{m}/seed{s}/fold{f} done in {(time.time() - t_run) / 60:.1f} min '
                  f'| elapsed {el / 3600:.2f} h', flush=True)
        else:
            del cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        break

    done = sum(is_done(run_dir(a.out, e, m, s, f)) for e in encoders for m in models for s in seeds for f in folds)
    print(f'GRID FINISHED: {done}/{total} runs complete'
          + ('' if done == total else ' -> run again with this output attached to continue'), flush=True)


if __name__ == '__main__':
    main()
