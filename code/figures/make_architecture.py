"""Architecture figure: (a) end-to-end pipeline, (b) two-tier MIL aggregation (ABMIL) + aggregator variants.

    python figures/make_architecture.py --figs ../figures --out ../figures

Uses real image content cut from fig_heatmaps.png / fig_heatmap_closeups.png (slide 10143, HGSC, test set) and the
pipeline's own Macenko code for the before/after tile. Vector PDF + PNG + SVG.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'extract'))
import featx  # noqa: E402  (Macenko implementation used in the pipeline)

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#9a9893'
FROZEN_FILL, FROZEN_EDGE = '#efeeea', '#9a9893'
TRAIN_FILL, TRAIN_EDGE = '#e3eefc', '#2a78d6'
ATT = '#eb6834'
CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

YMAX = 97.0                      # top of the drawing (y units)
FIG_W, FIG_H = 9.6, 7.4 * YMAX / 100   # inches; 1 x-unit = 0.096 in, 1 y-unit = 0.074 in
XU, YU = FIG_W / 100, 7.4 / 100


def sq(w):
    """height (y units) that makes something w x-units wide look square"""
    return w * XU / YU


# ----------------------------------------------------------------------------- assets
def crop(path, y, x, h, w):
    return np.asarray(Image.open(path).convert('RGB'))[y:y + h, x:x + w]


def assets(figs):
    heat = os.path.join(figs, 'fig_heatmaps.png')
    close = os.path.join(figs, 'fig_heatmap_closeups.png')
    a = dict(
        slide=crop(heat, 78, 873, 350, 333),       # slide 10143 (HGSC), tissue
        heat=crop(heat, 506, 873, 350, 333),       # same slide, attention overlay
        tiles=[crop(close, 894, x, 324, 324) for x in (785, 1173, 1562, 1951)],   # top-1..3, lowest
        att=[1.00, 1.00, 1.00, 0.00],
    )
    raw = a['tiles'][0]
    he, maxc = featx.macenko_fit(raw.reshape(-1, 3)[(raw.reshape(-1, 3).astype(np.float32) @ featx.GRAY_W) <= 220])
    a['norm'] = featx.macenko_apply(torch.from_numpy(raw.copy())[None], he, maxc)[0].numpy().astype(np.uint8)
    return a


# ----------------------------------------------------------------------------- drawing helpers
def box(ax, x, y, w, h, text=None, kind='frozen', fs=7.6, weight='normal', r=1.2):
    fill, edge = (TRAIN_FILL, TRAIN_EDGE) if kind == 'trained' else (FROZEN_FILL, FROZEN_EDGE)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fill, ec=edge, lw=1.1))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight=weight, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.2, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def image(ax, arr, x, y, w, edge=None):
    h = w * arr.shape[0] / arr.shape[1] * XU / YU
    ax.imshow(arr, extent=(x, x + w, y, y + h), zorder=3, interpolation='lanczos', aspect='auto')
    if edge:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=edge, lw=0.8, zorder=4))
    return h


def stage(ax, x, w, y, num, title, desc):
    ax.text(x, y, f'{num}', fontsize=8, color='white', weight='bold', ha='center', va='center', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.25', fc=INK, ec='none'))
    ax.text(x + 1.8, y, title, fontsize=8.2, weight='bold', color=INK, va='center')
    ax.text(x - 1.0, y - 3.4, desc, fontsize=6.6, color=INK2, va='top', linespacing=1.3)


# ----------------------------------------------------------------------------- figure
def draw(a, out):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, YMAX); ax.axis('off')

    # ===== (a) pipeline
    ax.text(0.5, 95.6, 'a', fontsize=12, weight='bold', color=INK, va='center')
    ax.text(3.0, 95.6, 'End-to-end pipeline', fontsize=10, weight='bold', color=INK, va='center')
    top, base = 90.0, 62.5           # stage title line, image baseline

    # 1 slide
    stage(ax, 2.0, 14, top, 1, 'Slide', '538 images\n(WSI 20×, TMA 40×)')
    image(ax, a['slide'], 1.0, base, 13.0)
    arrow(ax, 15.0, base + 9, 18.0, base + 9)

    # 2 tiling: same slide with a tile grid
    stage(ax, 19.5, 14, top, 2, 'Tissue tiling', '256 px tiles at ~10×\nkept if ≥25% tissue\n(1.5 M tiles in total)')
    h = image(ax, a['slide'], 18.5, base, 13.0)
    for gx in np.linspace(18.5, 31.5, 12):
        ax.plot([gx, gx], [base, base + h], color='white', lw=0.45, alpha=0.9, zorder=4)
    for gy in np.linspace(base, base + h, 13):
        ax.plot([18.5, 31.5], [gy, gy], color='white', lw=0.45, alpha=0.9, zorder=4)
    arrow(ax, 32.5, base + 9, 35.5, base + 9)

    # 3 stain normalisation: before -> after on a real tile
    stage(ax, 37.0, 14, top, 3, 'Stain normalisation', 'Macenko, fitted once\nper slide')
    image(ax, a['tiles'][0], 36.0, base + 3.0, 6.0, edge=MUTED)
    arrow(ax, 42.4, base + 6.7, 44.1, base + 6.7, lw=1.0)
    image(ax, a['norm'], 44.5, base + 3.0, 6.0, edge=MUTED)
    ax.text(39.0, base + 1.6, 'raw', fontsize=6.3, color=INK2, ha='center', va='top')
    ax.text(47.5, base + 1.6, 'normalised', fontsize=6.3, color=INK2, ha='center', va='top')
    arrow(ax, 51.2, base + 9, 54.0, base + 9)

    # 4 frozen encoder -> feature matrix
    stage(ax, 55.5, 14, top, 4, 'Tile encoder', 'frozen, pre-trained\nN tiles × D features\n(D = 384–1,024)')
    box(ax, 54.5, base + 5.5, 13.5, 11.0, 'ViT-S/16 (Lunit)\nCTransPath\nPhikon-v2 · PLIP', kind='frozen', fs=6.8)
    rng = np.random.default_rng(3)
    for k in range(7):                                   # feature matrix: N rows of D values
        for j in range(10):
            ax.add_patch(Rectangle((55.3 + j * 1.2, base + 0.4 + k * 0.62), 1.15, 0.55,
                                   fc=plt.cm.Blues(0.25 + 0.6 * rng.random()), ec='none', zorder=3))
    ax.text(68.6, base + 2.6, 'N×D', fontsize=6.3, color=INK2, va='center')
    arrow(ax, 69.2, base + 9, 72.0, base + 9)

    # 5 MIL aggregator (trained)
    stage(ax, 73.5, 12, top, 5, 'MIL aggregator', 'trained; slide label\nonly (weak supervision)\ndetail in b')
    box(ax, 72.5, base + 4.0, 11.0, 10.0, 'tier 1: tiles\n↓\ntier 2: slide', kind='trained', fs=7.0)
    arrow(ax, 84.2, base + 9, 87.0, base + 9)

    # 6 outputs
    stage(ax, 88.5, 11, top, 6, 'Output', 'subtype probabilities\n+ attention heatmap')
    probs = [0.01, 0.06, 0.88, 0.04, 0.01]
    for k, (c, p) in enumerate(zip(CLASSES, probs)):
        yk = base + 16.0 - k * 2.1
        ax.text(91.6, yk, c, fontsize=6.0, color=INK2, ha='right', va='center')
        ax.add_patch(Rectangle((92.0, yk - 0.75), 7.0 * p, 1.5, fc=TRAIN_EDGE if c == 'HGSC' else '#9ec5f4', ec='none'))
    image(ax, a['heat'], 88.0, base - 5.8, 9.0)

    # separator
    ax.plot([0.5, 99.5], [53.0, 53.0], color='#d6d5d0', lw=0.8)

    # ===== (b) two-tier MIL
    ax.text(0.5, 49.0, 'b', fontsize=12, weight='bold', color=INK, va='center')
    ax.text(3.0, 49.0, 'Two-tier MIL aggregation (ABMIL shown; same slot for every aggregator)', fontsize=10, weight='bold', color=INK, va='center')

    ys = [35.0, 26.6, 18.2, 6.0]                           # 3 top tiles, gap, lowest tile
    labels = ['x₁', 'x₂', 'x₃', 'xₙ']
    t = 5.6                                                # tile width
    ax.text(0.8, 44.2, 'bag of tiles', fontsize=6.8, color=INK2, ha='left', va='center')
    ax.text(2.4 + t / 2, 15.8, '⋮', fontsize=12, color=INK2, ha='center', va='center', zorder=6)
    for yk, tile, lab in zip(ys, a['tiles'], labels):
        image(ax, tile, 2.4, yk, t, edge=MUTED)
        ax.text(1.4, yk + sq(t) / 2, lab, fontsize=7.5, color=INK, ha='right', va='center')

    # tier 1 band
    ax.add_patch(FancyBboxPatch((11.0, 4.5), 38.0, 41.5, boxstyle='round,pad=0,rounding_size=1.5', fc='#fafaf8', ec='#d6d5d0', lw=0.8))
    ax.text(12.0, 44.2, 'Tier 1 · instance level (shared weights for every tile)', fontsize=7.4, weight='bold', color=INK, va='center')
    for yk, att, rank in zip(ys, a['att'], ['top 1', 'top 2', 'top 3', 'lowest']):
        yc = yk + sq(t) / 2
        arrow(ax, 8.6, yc, 12.5, yc, lw=0.9)
        box(ax, 12.6, yc - 2.0, 8.0, 4.0, 'frozen\nencoder', kind='frozen', fs=5.9, r=0.8)
        arrow(ax, 20.8, yc, 23.2, yc, lw=0.9)
        box(ax, 23.3, yc - 2.0, 8.5, 4.0, 'MLP\nhᵢ = f(xᵢ)', kind='trained', fs=6.0, r=0.8)
        arrow(ax, 32.0, yc, 34.4, yc, lw=0.9)
        box(ax, 34.5, yc - 2.0, 8.0, 4.0, 'gated\nattention', kind='trained', fs=6.0, r=0.8)
        # attention bar
        ax.add_patch(Rectangle((43.3, yc - 1.2), 4.8, 2.4, fc='#f6f5f2', ec='#d6d5d0', lw=0.6))
        ax.add_patch(Rectangle((43.3, yc - 1.2), max(0.15, 4.8 * att), 2.4, fc=ATT, ec='none'))
        ax.text(45.7, yc + 1.6, rank, fontsize=5.9, color=INK2, ha='center', va='bottom')
    ax.text(30.0, 15.8, 'aᵢ = softmaxᵢ( wᵀ [ tanh(V hᵢ) ⊙ σ(U hᵢ) ] )', fontsize=7.0, color=INK, ha='center', va='center', style='italic')
    ax.text(45.7, ys[-1] + sq(t) / 2 - 1.8, 'aᵢ → heatmap', fontsize=6.3, color=ATT, ha='center', va='top')

    # tier 2 band
    ax.add_patch(FancyBboxPatch((51.0, 4.5), 22.0, 41.5, boxstyle='round,pad=0,rounding_size=1.5', fc='#fafaf8', ec='#d6d5d0', lw=0.8))
    ax.text(52.0, 44.2, 'Tier 2 · bag (slide) level', fontsize=7.4, weight='bold', color=INK, va='center')
    pool_y = 24.0
    for yk in ys:
        arrow(ax, 48.5, yk + sq(t) / 2, 54.6, pool_y, lw=0.8, color=MUTED)
    ax.add_patch(plt.Circle((56.4, pool_y), 1.8, fc=TRAIN_FILL, ec=TRAIN_EDGE, lw=1.1, transform=ax.transData))
    ax.text(56.4, pool_y, 'Σ', fontsize=10, ha='center', va='center', color=INK)
    ax.text(56.4, pool_y - 4.2, 'z = Σ aᵢ hᵢ\nslide embedding', fontsize=6.3, color=INK2, ha='center', va='top')
    arrow(ax, 58.4, pool_y, 61.0, pool_y)
    box(ax, 61.1, pool_y - 3.0, 8.0, 6.0, 'linear\nclassifier\nsoftmax', kind='trained', fs=6.2, r=0.8)
    arrow(ax, 65.1, pool_y + 3.2, 65.1, 33.2)
    ax.text(65.1, 34.0, 'p(subtype | slide)', fontsize=6.6, color=INK, ha='center', va='bottom')

    # variants
    vx = 75.0
    ax.add_patch(FancyBboxPatch((vx, 4.5), 24.5, 41.5, boxstyle='round,pad=0,rounding_size=1.5', fc='white', ec='#d6d5d0', lw=0.8))
    ax.text(vx + 1.0, 44.2, 'Aggregators compared', fontsize=7.4, weight='bold', color=INK, va='center')
    rows = [('MaxPool / MeanPool', 'max / mean over hᵢ (no attention)'),
            ('ABMIL', 'gated attention pooling (shown)'),
            ('CLAM-SB', 'attention + clustering loss\non top/bottom-k tiles'),
            ('DSMIL', 'critical (max-scoring) tile +\nattention to it; two streams'),
            ('TransMIL', 'self-attention between tiles\n(Nyström) + class token')]
    for k, (name, desc) in enumerate(rows):
        yk = 39.6 - k * 6.6
        ax.text(vx + 1.0, yk, name, fontsize=7.0, weight='bold', color=TRAIN_EDGE, va='top')
        ax.text(vx + 1.0, yk - 2.1, desc, fontsize=6.2, color=INK2, va='top', linespacing=1.25)
    ax.text(vx + 1.0, 5.6, 'all: tile features in → slide logits out', fontsize=6.0, color=INK2, style='italic')

    # legend + training note
    box(ax, 1.0, 0.6, 3.0, 2.0, kind='frozen', r=0.5); ax.text(4.6, 1.6, 'frozen / preprocessing', fontsize=6.6, color=INK2, va='center')
    box(ax, 22.0, 0.6, 3.0, 2.0, kind='trained', r=0.5); ax.text(25.6, 1.6, 'trained', fontsize=6.6, color=INK2, va='center')
    ax.add_patch(Rectangle((33.5, 0.7), 3.0, 1.8, fc=ATT, ec='none')); ax.text(37.1, 1.6, 'attention', fontsize=6.6, color=INK2, va='center')
    ax.text(99.5, 1.6, 'Training: 5-fold CV × 3 seeds · selection on validation only · locked 108-slide test set',
            fontsize=6.4, color=INK2, va='center', ha='right')

    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(out, f'fig_architecture.{ext}'), dpi=300)
    plt.close(fig)
    print('wrote fig_architecture (pdf, png, svg)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', required=True, help='folder with fig_heatmaps.png and fig_heatmap_closeups.png')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    draw(assets(a.figs), a.out)
