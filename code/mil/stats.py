"""Paired bootstrap significance tests between configurations on the locked test set.

For every pair (A, B) the SAME resampled set of test slides is scored for both models
(paired bootstrap, 10,000 resamples), giving the difference A - B, its 95% CI and a two-sided p-value.
p-values are Holm-corrected within each family of pre-specified comparisons:

  encoder_vs_vits16   each encoder vs ViT-S/16 Lunit, same MIL model
  model_vs_maxpool    each MIL model vs MaxPool, same encoder
  capstone_claims     ABMIL vs DSMIL and MaxPool vs MeanPool, per encoder (claims of the old paper)
  encoder_overall     each encoder vs ViT-S/16, averaged over the 6 MIL models
  model_overall       each MIL model vs MaxPool, averaged over the 4 encoders

Predictions used per configuration (runs/<encoder>/<model>/): test_preds_seedensemble.csv if present
(all seeds x folds averaged), else seed0/test_preds_ensemble.csv.

    python -m mil.stats runs --out results/significance.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from .data import CLASSES
from .metrics import PROB_COLS

ENCODERS = ['vits16_lunit', 'ctranspath', 'phikon_v2', 'plip']
MODELS = ['maxpool', 'meanpool', 'abmil', 'dsmil', 'clam_sb', 'transmil']
K = len(CLASSES)


def load_config_preds(config_dir, split='test'):
    """Predictions of one configuration folder (<config_dir>/seed*/fold*/...), as (labels, predicted class, image ids).

    split='test': locked test set; probabilities of all seeds x folds averaged per slide
                  (uses test_preds_seedensemble.csv when present).
    split='cv'  : out-of-fold validation predictions of the 430 non-test slides, averaged over seeds.
                  (Each fold's checkpoint was chosen on that same fold, so CV is slightly optimistic, equally for all configs.)
    """
    import glob
    if split == 'cv':
        files = glob.glob(os.path.join(config_dir, 'seed*', 'fold*', 'val_preds.csv'))
    else:
        ens = os.path.join(config_dir, 'test_preds_seedensemble.csv')
        files = [ens] if os.path.exists(ens) else glob.glob(os.path.join(config_dir, 'seed*', 'fold*', 'test_preds.csv'))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, dtype={'image_id': str}) for f in files])
    d = d.groupby('image_id', as_index=False).agg({'label': 'first', **{c: 'mean' for c in PROB_COLS}})
    d = d.sort_values('image_id').reset_index(drop=True)
    return d['label'].to_numpy(), d[PROB_COLS].to_numpy().argmax(1), d['image_id'].to_numpy()


def load_preds(runs, enc, model, split='test'):
    return load_config_preds(os.path.join(runs, enc, model), split)


def paired_test(y, pred_a, pred_b, metric='bacc', n_boot=10000, seed=0):
    """Paired bootstrap of metric(A) - metric(B) on the same slides: (observed diff, ci_lo, ci_hi, two-sided p)."""
    n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    full = np.arange(n)[None]
    diff = boot_scores(y, pred_a, idx, metric) - boot_scores(y, pred_b, idx, metric)
    obs = (boot_scores(y, pred_a, full, metric) - boot_scores(y, pred_b, full, metric))[0]
    p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / n_boot)
    return obs, np.percentile(diff, 2.5), np.percentile(diff, 97.5), p


def boot_scores(y, pred, idx, metric):
    """Metric for every bootstrap resample at once. idx: (B, n) -> (B,)"""
    yi, pi = y[idx], pred[idx]
    tp = np.stack([((yi == c) & (pi == c)).sum(1) for c in range(K)], 1).astype(float)   # (B, K)
    support = np.stack([(yi == c).sum(1) for c in range(K)], 1).astype(float)
    predicted = np.stack([(pi == c).sum(1) for c in range(K)], 1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        recall = tp / support
        if metric == 'bacc':
            return np.nanmean(recall, 1)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        return np.nanmean(np.where(support > 0, f1, np.nan), 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def run(runs, n_boot=10000, seed=0, metrics=('bacc', 'macro_f1'), split='test'):
    preds, ids = {}, None
    for e in ENCODERS:
        for m in MODELS:
            r = load_preds(runs, e, m, split)
            if r is None:
                continue
            y, p, i = r
            if ids is None:
                ids, y_ref = i, y
            assert (i == ids).all() and (y == y_ref).all(), f'{e}/{m}: test slides differ'
            preds[(e, m)] = p
    n = len(ids)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))       # shared resamples (paired)
    full = np.arange(n)[None]

    def score(keys, metric, which):
        """mean metric over one or more configs, per resample (which=idx) or on the full set."""
        return np.mean([boot_scores(y_ref, preds[k], which, metric) for k in keys], 0)

    def test(family, label_a, keys_a, label_b, keys_b, metric):
        keys_a, keys_b = [k for k in keys_a if k in preds], [k for k in keys_b if k in preds]
        if not keys_a or not keys_b:
            return None
        diff = score(keys_a, metric, idx) - score(keys_b, metric, idx)
        obs = (score(keys_a, metric, full) - score(keys_b, metric, full))[0]
        p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / n_boot)
        return dict(family=family, metric=metric, a=label_a, b=label_b, diff=obs,
                    diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5), p=p)

    rows = []
    for metric in metrics:
        for m in MODELS:
            for e in ENCODERS[1:]:
                rows.append(test('encoder_vs_vits16', f'{e}+{m}', [(e, m)], f'vits16_lunit+{m}', [('vits16_lunit', m)], metric))
        for e in ENCODERS:
            for m in MODELS[1:]:
                rows.append(test('model_vs_maxpool', f'{e}+{m}', [(e, m)], f'{e}+maxpool', [(e, 'maxpool')], metric))
            rows.append(test('capstone_claims', f'{e}+abmil', [(e, 'abmil')], f'{e}+dsmil', [(e, 'dsmil')], metric))
            rows.append(test('capstone_claims', f'{e}+maxpool', [(e, 'maxpool')], f'{e}+meanpool', [(e, 'meanpool')], metric))
        for e in ENCODERS[1:]:
            rows.append(test('encoder_overall', e, [(e, m) for m in MODELS], 'vits16_lunit', [('vits16_lunit', m) for m in MODELS], metric))
        for m in MODELS[1:]:
            rows.append(test('model_overall', m, [(e, m) for e in ENCODERS], 'maxpool', [(e, 'maxpool') for e in ENCODERS], metric))

    df = pd.DataFrame([r for r in rows if r])
    df['p_holm'] = np.nan
    for (fam, metric), g in df.groupby(['family', 'metric']):
        df.loc[g.index, 'p_holm'] = holm(g.p.to_numpy())
    df['significant'] = df.p_holm < 0.05
    df.insert(0, 'split', split)
    return df, n


def main():
    ap = argparse.ArgumentParser(description='Paired bootstrap significance tests on the locked test set')
    ap.add_argument('runs', help='runs/ folder with <encoder>/<model>/...')
    ap.add_argument('--out', default=None)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--split', choices=['test', 'cv'], default='test')
    a = ap.parse_args()
    df, n = run(a.runs, a.n_boot, split=a.split)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        df.to_csv(a.out, index=False)
    pd.set_option('display.width', 200)
    show = df[df.metric == 'bacc'].copy()
    for c in ['diff', 'diff_lo', 'diff_hi']:
        show[c] = show[c].round(3)
    show['p'] = show.p.round(4); show['p_holm'] = show.p_holm.round(4)
    print(f'[{a.split}] paired bootstrap on {n} slides, {a.n_boot} resamples, Holm-corrected within each family (balanced accuracy)')
    for fam, g in show.groupby('family', sort=False):
        print(f'\n== {fam}\n' + g[['a', 'b', 'diff', 'diff_lo', 'diff_hi', 'p', 'p_holm', 'significant']].to_string(index=False))


if __name__ == '__main__':
    main()
