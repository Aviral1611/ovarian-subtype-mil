"""Row-normalised 5-class confusion matrices + EC/HGSC confusion quantified across every trained configuration.

    python figures/make_confusion.py --kaggle ../kaggle --out ../figures

Uses predictions already on disk (no GPU):
  results_04/runs/<enc>/<model>/       default settings, 3 seeds (24 configurations)
  results_06/tune/tuned/<enc>/<model>/ tuned ABMIL/DSMIL, 3 seeds (8 configurations)
Test = locked 108 slides (seed ensemble). CV = out-of-fold predictions of the other 430 slides (averaged over seeds).
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mil.stats import load_config_preds  # noqa: E402

CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
EC, HGSC = 1, 2
EN = {'vits16_lunit': 'ViT-S/16', 'ctranspath': 'CTransPath', 'phikon_v2': 'Phikon-v2', 'plip': 'PLIP'}
MN = {'maxpool': 'MaxPool', 'meanpool': 'MeanPool', 'abmil': 'ABMIL', 'dsmil': 'DSMIL', 'clam_sb': 'CLAM-SB', 'transmil': 'TransMIL'}
BLUE = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('blue_seq', BLUE)
INK, INK2, MUTED, GRID, ACCENT = '#0b0b0b', '#52514e', '#9a9893', '#e4e3df', '#eb6834'
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.labelcolor': INK2, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def configs(kaggle):
    out = []
    for d in sorted(glob.glob(os.path.join(kaggle, 'results_04', 'runs', '*', '*'))):
        enc, model = d.replace('\\', '/').split('/')[-2:]
        out.append((f'{EN[enc]} + {MN[model]}', 'default', d))
    for d in sorted(glob.glob(os.path.join(kaggle, 'results_06', 'tune', 'tuned', '*', '*'))):
        enc, model = d.replace('\\', '/').split('/')[-2:]
        out.append((f'{EN[enc]} + {MN[model]}', 'tuned', d))
    return out


def cm_of(cfg_dir, split):
    y, p, _ = load_config_preds(cfg_dir, split)
    return confusion_matrix(y, p, labels=range(len(CLASSES)))


def draw_cm(ax, cm, title, ylabel=True):
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = norm[i, j]
            col = INK if v < 0.55 else 'white'
            ax.text(j, i - 0.12, f'{100 * v:.0f}%', ha='center', va='center', fontsize=8.5, color=col, fontweight='bold' if i == j else 'normal')
            ax.text(j, i + 0.24, f'({cm[i, j]})', ha='center', va='center', fontsize=6.3, color=col)
    for (i, j) in [(EC, HGSC), (HGSC, EC)]:     # outline the EC <-> HGSC cells
        ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=ACCENT, lw=2))
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), [f'{c}  n={n}' for c, n in zip(CLASSES, cm.sum(1))])
    ax.set_xlabel('Predicted')
    if ylabel:
        ax.set_ylabel('True')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which='minor', color='white', linewidth=2); ax.tick_params(which='minor', length=0)
    ax.set_title(title, loc='left', color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kaggle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--main', default='vits16_lunit/maxpool', help='configuration shown in the matrices (results_04 path)')
    a = ap.parse_args()

    # ---- EC/HGSC confusion for every configuration and split
    rows = []
    for name, setting, d in configs(a.kaggle):
        for split in ('cv', 'test'):
            cm = cm_of(d, split)
            nEC, nH = cm[EC].sum(), cm[HGSC].sum()
            off = cm.copy().astype(float); np.fill_diagonal(off, -1)
            i, j = np.unravel_index(off.argmax(), off.shape)
            rows.append(dict(config=name, setting=setting, split=split, n_EC=nEC, n_HGSC=nH,
                             EC_to_HGSC=cm[EC, HGSC] / nEC, HGSC_to_EC=cm[HGSC, EC] / nH,
                             EC_recall=cm[EC, EC] / nEC, HGSC_recall=cm[HGSC, HGSC] / nH,
                             EC_errors_into_HGSC=cm[EC, HGSC] / max(1, nEC - cm[EC, EC]),
                             largest_confusion=f'{CLASSES[i]}->{CLASSES[j]}'))
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, 'confusion_ec_hgsc_all_configs.csv'), index=False)

    # ---- main configuration: CV and test matrices
    main_dir = os.path.join(a.kaggle, 'results_04', 'runs', *a.main.split('/'))
    enc, model = a.main.split('/')
    cm_cv, cm_te = cm_of(main_dir, 'cv'), cm_of(main_dir, 'test')

    fig = plt.figure(figsize=(10.6, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.5)
    draw_cm(fig.add_subplot(gs[0]), cm_cv, f'a  Cross-validation (430 slides)\n{EN[enc]} + {MN[model]}, mean of 3 seeds')
    draw_cm(fig.add_subplot(gs[1]), cm_te, f'b  Locked test set (108 slides)\n{EN[enc]} + {MN[model]}, seed ensemble', ylabel=False)

    # panel c: EC->HGSC and HGSC->EC on CV across all 32 configurations
    ax = fig.add_subplot(gs[2])
    cv = t[t.split == 'cv']
    for k, (col, label, colour) in enumerate([('EC_to_HGSC', 'EC predicted as HGSC', ACCENT), ('HGSC_to_EC', 'HGSC predicted as EC', '#2a78d6')]):
        v = 100 * cv[col].to_numpy()
        jitter = np.random.default_rng(k).uniform(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), k) + jitter, v, s=16, color=colour, alpha=0.75, linewidths=0, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
        ax.text(k + 0.3, np.median(v), f'median {np.median(v):.0f}%\nrange {v.min():.0f}–{v.max():.0f}%', va='center', fontsize=7, color=INK)
    ax.set_xticks([0, 1], ['EC → HGSC\n(of 99 EC)', 'HGSC → EC\n(of 177 HGSC)'])
    ax.set_xlim(-0.5, 1.95); ax.set_ylim(0, max(30, 100 * cv[['EC_to_HGSC', 'HGSC_to_EC']].to_numpy().max() + 3))
    ax.set_ylabel('% of true class (CV)')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f'c  Same confusion in all {len(cv)} trained\nconfigurations (4 encoders × 6 MIL, + tuned)', loc='left', color=INK)
    fig.text(0.01, -0.06, 'Cells: % of the true class (row-normalised); counts in brackets. Orange outline = EC↔HGSC cells.', fontsize=7, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'fig_confusion_normalized.{ext}'), dpi=300)
    plt.close(fig)

    # ---- printed summary
    for split, cm in (('cv', cm_cv), ('test', cm_te)):
        nEC, nH = cm[EC].sum(), cm[HGSC].sum()
        lo, hi = wilson(cm[EC, HGSC], nEC); lo2, hi2 = wilson(cm[HGSC, EC], nH)
        err = nEC - cm[EC, EC]
        print(f'[{split}] {EN[enc]}+{MN[model]}: EC->HGSC {cm[EC, HGSC]}/{nEC} = {100 * cm[EC, HGSC] / nEC:.1f}% [95% CI {100 * lo:.1f}-{100 * hi:.1f}] '
              f'| HGSC->EC {cm[HGSC, EC]}/{nH} = {100 * cm[HGSC, EC] / nH:.1f}% [{100 * lo2:.1f}-{100 * hi2:.1f}] '
              f'| share of EC errors going to HGSC {cm[EC, HGSC]}/{err} = {100 * cm[EC, HGSC] / max(1, err):.0f}%')
    for split in ('cv', 'test'):
        s = t[t.split == split]
        print(f'[{split}] across {len(s)} configs: EC->HGSC median {100 * s.EC_to_HGSC.median():.1f}% (range {100 * s.EC_to_HGSC.min():.1f}-{100 * s.EC_to_HGSC.max():.1f}) '
              f'| HGSC->EC median {100 * s.HGSC_to_EC.median():.1f}% | largest single confusion = EC->HGSC in {(s.largest_confusion == "EC->HGSC").sum()}/{len(s)} '
              f'| most common largest: {s.largest_confusion.value_counts().head(3).to_dict()}')


if __name__ == '__main__':
    main()
