"""Slide-level metrics and bootstrap confidence intervals.

All functions take a predictions DataFrame with columns:
    image_id, label, pred, prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC  (+ optional is_tma)
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from .data import CLASSES

PROB_COLS = [f'prob_{c}' for c in CLASSES]
LABELS = list(range(len(CLASSES)))


def macro_auc(y, probs):
    """One-vs-rest AUROC averaged over classes that have both positives and negatives."""
    aucs = []
    for c in LABELS:
        pos = (y == c)
        if 0 < pos.sum() < len(y):
            aucs.append(roc_auc_score(pos, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float('nan')


def compute_metrics(df):
    y, p = df['label'].to_numpy(), df['pred'].to_numpy()
    probs = df[PROB_COLS].to_numpy()
    out = dict(
        n=len(df),
        acc=accuracy_score(y, p),
        bacc=balanced_accuracy_score(y, p) if len(set(y)) > 1 else float('nan'),
        macro_f1=f1_score(y, p, labels=LABELS, average='macro', zero_division=0),
        macro_auc=macro_auc(y, probs),
    )
    recalls = recall_score(y, p, labels=LABELS, average=None, zero_division=0)
    out.update({f'recall_{c}': float(r) for c, r in zip(CLASSES, recalls)})
    return out


def bootstrap_ci(df, metrics=('acc', 'bacc', 'macro_f1', 'macro_auc'), n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over slides. Returns {metric: (low, high)}."""
    rng = np.random.default_rng(seed)
    values = {m: [] for m in metrics}
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        res = compute_metrics(sample)
        for m in metrics:
            values[m].append(res[m])
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in values.items()}


def ensemble(pred_dfs):
    """Average class probabilities of several models (e.g. the 5 fold models) per slide."""
    base = pred_dfs[0][['image_id', 'label'] + (['is_tma'] if 'is_tma' in pred_dfs[0] else [])].copy()
    probs = np.mean([d.set_index('image_id').loc[base.image_id, PROB_COLS].to_numpy() for d in pred_dfs], axis=0)
    base[PROB_COLS] = probs
    base['pred'] = probs.argmax(1)
    return base
