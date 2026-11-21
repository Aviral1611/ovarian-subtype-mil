"""Paper figures from the 3-seed results (no GPU needed).

    python figures/make_figures.py --results ../kaggle/results_04 --out ../figures

Writes PDF (vector, for the paper) + PNG (300 dpi, for slides) of:
    fig_benchmark_grid      test balanced accuracy (seed ensemble) and CV balanced accuracy, encoder x MIL model
    fig_significance        CV paired-bootstrap differences vs the reference (ViT-S/16; MaxPool) with 95% CI
    fig_confusion           confusion matrices of the two CV-selected models (locked test set)
    fig_roc                 one-vs-rest ROC curves per subtype, ViT-S/16 + MaxPool (locked test set)
    fig_recall              per-subtype recall by encoder (MaxPool head, locked test set)
Colours: validated categorical palette (dataviz skill reference, light mode) + single-hue blue sequential ramp.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
PROB = [f'prob_{c}' for c in CLASSES]
E = ['vits16_lunit', 'ctranspath', 'phikon_v2', 'plip']
M = ['maxpool', 'meanpool', 'abmil', 'dsmil', 'clam_sb', 'transmil']
EN = {'vits16_lunit': 'ViT-S/16 (Lunit)', 'ctranspath': 'CTransPath', 'phikon_v2': 'Phikon-v2', 'plip': 'PLIP'}
MN = {'maxpool': 'MaxPool', 'meanpool': 'MeanPool', 'abmil': 'ABMIL', 'dsmil': 'DSMIL', 'clam_sb': 'CLAM-SB', 'transmil': 'TransMIL'}

# validated categorical order (blue, orange, aqua, yellow, magenta) + blue sequential ramp + text inks
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('blue_seq', BLUE)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#9a9893', '#e4e3df'

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    print('wrote', name)


def preds(results, enc, model):
    return pd.read_csv(os.path.join(results, 'runs', enc, model, 'test_preds_seedensemble.csv'), dtype={'image_id': str})


def text_on(value, vmin, vmax):
    """dark ink on light cells, white on dark cells"""
    return INK if (value - vmin) / (vmax - vmin + 1e-9) < 0.55 else 'white'


def fig_benchmark_grid(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.08))
    panels = [('ens_bacc', 'Locked test set (n = 108), seed ensemble', True), ('cv_bacc_mean', 'Cross-validation (n = 430), mean of 3 seeds', False)]
    vmin, vmax = 0.70, 0.85
    for ax, (col, title, ci) in zip(axes, panels):
        grid = r.pivot(index='model', columns='encoder', values=col).loc[M, E].to_numpy()
        ax.imshow(grid, cmap=SEQ, vmin=vmin, vmax=vmax, aspect='auto')
        for i, m in enumerate(M):
            for j, e in enumerate(E):
                v = grid[i, j]
                ax.text(j, i - (0.12 if ci else 0), f'{v:.3f}', ha='center', va='center', fontsize=8, color=text_on(v, vmin, vmax), fontweight='bold')
                if ci:
                    x = r[(r.encoder == e) & (r.model == m)].iloc[0]
                    ax.text(j, i + 0.22, f'[{x.ens_bacc_lo:.2f}–{x.ens_bacc_hi:.2f}]', ha='center', va='center', fontsize=6.3, color=text_on(v, vmin, vmax))
        ax.set_xticks(range(len(E)), [EN[e].replace(' (Lunit)', '\n(Lunit)') for e in E])
        ax.set_yticks(range(len(M)), [MN[m] for m in M] if ax is axes[0] else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xticks(np.arange(-.5, len(E)), minor=True); ax.set_yticks(np.arange(-.5, len(M)), minor=True)
        ax.grid(which='minor', color='white', linewidth=2); ax.tick_params(which='minor', length=0)
    fig.suptitle('Balanced accuracy by encoder and MIL aggregator', x=0.02, ha='left', y=1.02, fontsize=10, fontweight='bold')
    save(fig, out, 'fig_benchmark_grid')


def fig_significance(results, out):
    s = pd.read_csv(os.path.join(results, 'results', 'significance_cv.csv'))
    s = s[(s.metric == 'bacc') & s.family.isin(['encoder_overall', 'model_overall'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.55))
    for ax, fam, ref, names in [(axes[0], 'encoder_overall', 'ViT-S/16 (Lunit)', EN), (axes[1], 'model_overall', 'MaxPool', MN)]:
        g = s[s.family == fam].iloc[::-1].reset_index(drop=True)
        for i, x in g.iterrows():
            sig = bool(x.significant)
            col = CAT[0] if sig else MUTED
            ax.plot([x.diff_lo, x.diff_hi], [i, i], color=col, linewidth=2, solid_capstyle='round')
            ax.plot(x['diff'], i, 'o', markersize=8, markerfacecolor=col if sig else 'white', markeredgecolor=col, markeredgewidth=1.6)
            ax.text(max(x.diff_hi, 0) + 0.006, i, f'p = {x.p_holm:.3f}' if x.p_holm >= 0.001 else 'p < 0.001',
                    va='center', fontsize=7, color=INK if sig else INK2)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(g)), [names[a] for a in g.a])
        ax.set_xlabel(f'Δ balanced accuracy vs {ref}')
        ax.grid(axis='x', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.set_xlim(min(-0.11, g.diff_lo.min() - 0.01), 0.075)
        ax.set_title(('Encoders' if fam == 'encoder_overall' else 'MIL aggregators') + f' vs {ref}', loc='left', color=INK)
    fig.text(0.02, -0.24, 'Cross-validation predictions (n = 430 slides, 3 seeds), paired bootstrap (10,000 resamples), Holm-corrected p.\n'
             'Filled = significant (p < 0.05); hollow = not significant. Encoder rows average over the 6 aggregators; aggregator rows over the 4 encoders.',
             fontsize=6.8, color=INK2)
    save(fig, out, 'fig_significance')


def fig_confusion(results, out, configs):
    fig, axes = plt.subplots(1, len(configs), figsize=(3.6 * len(configs), 3.1), gridspec_kw=dict(wspace=0.35))
    for ax, (enc, model) in zip(np.atleast_1d(axes), configs):
        d = preds(results, enc, model)
        cm = confusion_matrix(d.label, d[PROB].to_numpy().argmax(1), labels=range(len(CLASSES)))
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=8.5, color=text_on(norm[i, j], 0, 1),
                        fontweight='bold' if i == j else 'normal')
        ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f'{EN[enc]} + {MN[model]}\naccuracy {acc:.3f} (n = {cm.sum()})', loc='left', color=INK)
    fig.text(0.02, -0.05, 'Locked test set, seed ensemble (15 models). Cell colour = row-normalised recall; numbers = slide counts.', fontsize=6.8, color=INK2)
    save(fig, out, 'fig_confusion')


def fig_roc(results, out, enc, model):
    d = preds(results, enc, model)
    y, p = d.label.to_numpy(), d[PROB].to_numpy()
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle='--')
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y == k, p[:, k])
        auc = roc_auc_score(y == k, p[:, k])
        ax.plot(fpr, tpr, color=CAT[k], linewidth=2, label=f'{c}  (AUC {auc:.3f}, n = {(y == k).sum()})')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, handlelength=1.6)
    ax.set_title(f'One-vs-rest ROC, {EN[enc]} + {MN[model]}\nlocked test set, seed ensemble', loc='left', color=INK)
    save(fig, out, 'fig_roc')


def fig_recall(r, out, model='maxpool'):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.19
    x = np.arange(len(CLASSES))
    for k, e in enumerate(E):
        row = r[(r.encoder == e) & (r.model == model)].iloc[0]
        vals = [row[f'ens_recall_{c}'] for c in CLASSES]
        bars = ax.bar(x + (k - 1.5) * (w + 0.012), vals, width=w, color=CAT[k], label=EN[e], edgecolor='white', linewidth=0)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=6.2, color=INK2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel('Recall'); ax.set_ylim(0, 1.08)
    ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.16))
    ax.set_title(f'Per-subtype recall by encoder ({MN[model]} head, locked test set, seed ensemble)', loc='left', color=INK, pad=22)
    save(fig, out, 'fig_recall')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='results_04 folder (with results/ and runs/)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pd.read_csv(os.path.join(a.results, 'results', 'summary.csv'))
    fig_benchmark_grid(r, a.out)
    fig_significance(a.results, a.out)
    fig_confusion(a.results, a.out, [('vits16_lunit', 'maxpool'), ('vits16_lunit', 'abmil')])
    fig_roc(a.results, a.out, 'vits16_lunit', 'maxpool')
    fig_recall(r, a.out)


if __name__ == '__main__':
    main()
