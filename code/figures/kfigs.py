"""Kaggle-side figures: t-SNE of encoder features, ABMIL attention heatmaps, attention vs tumour-mask agreement.

Used by kaggle/04_figures.ipynb (embedded there). All paths are passed in, so it also runs locally on fake data.
"""
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']          # validated categorical order (dataviz skill)
HEAT = LinearSegmentedColormap.from_list('heat', ['#fde3d6', '#f4a27f', '#eb6834', '#b8461b', '#6e2405'])
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
EN = {'vits16_lunit': 'ViT-S/16 (Lunit)', 'ctranspath': 'CTransPath', 'phikon_v2': 'Phikon-v2', 'plip': 'PLIP'}
PATCH, SHRINK = 256, {False: 2, True: 4}

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#9a9893', 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=250)
    plt.close(fig)
    print('wrote', name, flush=True)


# ----------------------------------------------------------------------------- t-SNE
def tsne_figure(features_root, labels, out, encoders, n_patches=1000, seed=0):
    """labels: DataFrame image_id, label (int). Slide-level (mean of tiles) and patch-level (n_patches sampled) t-SNE."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    ids, y = labels.image_id.to_numpy(), labels.label.to_numpy()
    # same 1000 (slide, tile) picks for every encoder: stratified by subtype, 200 per class
    per_class = n_patches // len(CLASSES)
    picks = []
    for c in range(len(CLASSES)):
        cls_ids = ids[y == c]
        for s in rng.choice(cls_ids, per_class, replace=True):
            picks.append(s)
    picks = np.array(picks)
    rows, scores = [], []
    fig, axes = plt.subplots(2, len(encoders), figsize=(2.05 * len(encoders), 4.4))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, enc in enumerate(encoders):
        means, patch_feats, patch_y = [], [], []
        tile_choice = {}
        for s in ids:
            f = np.load(os.path.join(features_root, enc, f'{s}.npy')).astype(np.float32)
            means.append(f.mean(0))
        for k, s in enumerate(picks):
            f = np.load(os.path.join(features_root, enc, f'{s}.npy'), mmap_mode='r')
            t = tile_choice.setdefault((k, s), int(np.random.default_rng(seed + k).integers(len(f))))
            patch_feats.append(np.asarray(f[t], dtype=np.float32))
            patch_y.append(y[ids == s][0])
        for row, (X, yy, name) in enumerate([(np.stack(means), y, 'slide'), (np.stack(patch_feats), np.array(patch_y), 'patch')]):
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=min(50, Xs.shape[1], len(Xs) - 1), random_state=seed).fit_transform(Xs)
            sil = silhouette_score(Xp, yy)
            probe = ''
            if name == 'slide':   # linear probe: 5-fold CV logistic regression on slide-mean features
                pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced'), Xs, yy,
                                         cv=StratifiedKFold(5, shuffle=True, random_state=seed))
                probe = balanced_accuracy_score(yy, pred)
            emb = TSNE(n_components=2, perplexity=30 if len(Xp) > 100 else 10, random_state=seed, init='pca').fit_transform(Xp)
            ax = axes[row, j]
            for c in range(len(CLASSES)):
                m = yy == c
                ax.scatter(emb[m, 0], emb[m, 1], s=6 if name == 'slide' else 3, color=CAT[c], linewidths=0, alpha=0.85, label=CLASSES[c])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            sub = f'silhouette {sil:.3f}' + (f'\nlinear probe bacc {probe:.3f}' if probe != '' else '')
            ax.set_title((f'{EN.get(enc, enc)}\n' if row == 0 else '') + sub, fontsize=7.5, color=INK, loc='left')
            scores.append(dict(encoder=enc, level=name, n=len(yy), silhouette=sil, linear_probe_bacc=probe if probe != '' else np.nan))
        print(f'[tsne] {enc} done', flush=True)
    axes[0, 0].set_ylabel('slide level\n(mean of tiles, n = %d)' % len(ids), fontsize=7.5, color=INK2)
    axes[1, 0].set_ylabel('patch level\n(%d tiles)' % len(picks), fontsize=7.5, color=INK2)
    handles = [Patch(color=CAT[c], label=CLASSES[c]) for c in range(len(CLASSES))]
    fig.legend(handles=handles, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
    save(fig, out, 'fig_tsne')
    pd.DataFrame(scores).to_csv(os.path.join(out, 'tsne_scores.csv'), index=False)
    return pd.DataFrame(scores)


# ----------------------------------------------------------------------------- attention
def load_models(runs_dir, build_model, in_dim, device):
    """{(seed, fold): model} for every best.pt under runs_dir/seed*/fold*/"""
    models = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, 'seed*', 'fold*', 'best.pt'))):
        seed = int(re.search(r'seed(\d+)', p).group(1)); fold = int(re.search(r'fold(\d+)', p).group(1))
        ck = torch.load(p, map_location=device, weights_only=False)
        m = build_model(ck.get('model_name', 'abmil'), in_dim, len(CLASSES), **ck.get('model_kwargs', {})).to(device).eval()
        m.load_state_dict(ck['model'])
        models[(seed, fold)] = m
    return models


@torch.no_grad()
def slide_attention(models, keys, x):
    """Average over the given models: probabilities (C,) and attention as per-model percentile ranks (N,)."""
    probs, ranks = [], []
    for k in keys:
        out = models[k](x)
        probs.append(out['logits'].float().softmax(0).cpu().numpy())
        a = out['attn'].float().cpu().numpy()
        ranks.append(a.argsort().argsort() / max(1, len(a) - 1))     # rank-normalise so every model counts equally
    return np.mean(probs, 0), np.mean(ranks, 0)


def tile_boxes(coords, is_tma, scale):
    """tile rectangles (x0, y0, size) in a canvas that is `scale` x the original slide resolution"""
    sh = SHRINK[bool(is_tma)]
    return coords[:, 1] * sh * scale, coords[:, 0] * sh * scale, PATCH * sh * scale


def paint(canvas_hw, coords, is_tma, scale, values):
    canvas = np.full(canvas_hw, np.nan, dtype=np.float32)
    xs, ys, size = tile_boxes(coords, is_tma, scale)
    for x0, y0, v in zip(xs, ys, values):
        canvas[int(y0):int(np.ceil(y0 + size)), int(x0):int(np.ceil(x0 + size))] = v
    return canvas


def load_image_scaled(path, target_w):
    """RGB uint8 array resized to target_w (keeps aspect) via pyvips (streams big PNGs)."""
    import pyvips
    img = pyvips.Image.thumbnail(path, target_w, height=10 ** 7)
    if img.bands > 3:
        img = img.extract_band(0, n=3)
    return img.numpy()


def mask_classes(mask_rgb):
    """per pixel: 1 tumour (red), 2 other annotated tissue (green stroma / blue necrosis), 0 unannotated"""
    r, g, b = [mask_rgb[..., i].astype(int) for i in range(3)]
    out = np.zeros(mask_rgb.shape[:2], np.uint8)
    out[(g > 128) & (r < 128) | (b > 128) & (r < 128)] = 2
    out[(r > 128) & (g < 128) & (b < 128)] = 1
    return out


def tile_mask_labels(cls_map, coords, is_tma, scale):
    """per tile: 1 tumour (>=50% tumour px), 0 non-tumour (>=50% annotated non-tumour, <10% tumour), -1 otherwise"""
    xs, ys, size = tile_boxes(coords, is_tma, scale)
    lab = np.full(len(coords), -1, np.int8)
    for i, (x0, y0) in enumerate(zip(xs, ys)):
        patch = cls_map[int(y0):int(np.ceil(y0 + size)), int(x0):int(np.ceil(x0 + size))]
        if patch.size == 0:
            continue
        tumour, other = (patch == 1).mean(), (patch == 2).mean()
        if tumour >= 0.5:
            lab[i] = 1
        elif other >= 0.5 and tumour < 0.1:
            lab[i] = 0
    return lab


def mask_agreement(models, slides, features_dir, coords_dir, mask_files, device, out, canvas_w=1024):
    """For every slide with a mask: AUROC of attention for tumour vs non-tumour tiles, using only models that never
    trained on that slide (test slides: all models; CV slides: the models whose validation fold contained it)."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for _, s in slides[slides.image_id.isin(mask_files)].iterrows():
        keys = [k for k in models if s.fold == 0 or k[1] == s.fold]
        if not keys:
            continue
        x = torch.from_numpy(np.load(os.path.join(features_dir, f'{s.image_id}.npy'))).float().to(device)
        coords = np.load(os.path.join(coords_dir, f'{s.image_id}.npy'))
        _, att = slide_attention(models, keys, x)
        mask = load_image_scaled(mask_files[s.image_id], canvas_w)
        scale = mask.shape[1] / s.image_width
        lab = tile_mask_labels(mask_classes(mask), coords, s.is_tma, scale)
        t, n = att[lab == 1], att[lab == 0]
        row = dict(image_id=s.image_id, label=CLASSES[s.label], split='test' if s.fold == 0 else f'cv{s.fold}',
                   n_models=len(keys), n_tumour_tiles=len(t), n_nontumour_tiles=len(n))
        if len(t) >= 5 and len(n) >= 5:
            row['auroc'] = roc_auc_score(np.r_[np.ones(len(t)), np.zeros(len(n))], np.r_[t, n])
            top = att >= np.quantile(att, 0.9)
            known = lab >= 0
            row['top10_tumour_frac'] = (lab[top & known] == 1).mean() if (top & known).any() else np.nan
            row['base_tumour_frac'] = (lab[known] == 1).mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'heatmap_mask_agreement.csv'), index=False)
    return df


def load_slide_10x(path, is_tma):
    """Slide at the resolution the tiles were cut from (WSI 20x/2, TMA 40x/4 = ~10x), RGB uint8."""
    import pyvips
    img = pyvips.Image.new_from_file(path, access='sequential')
    if img.bands > 3:
        img = img.extract_band(0, n=3)
    sh = SHRINK[bool(is_tma)]
    return img.shrink(sh, sh).numpy()


def spread_top(att, coords, k, min_dist=3 * PATCH):
    """indices of the k highest-attention tiles, at least min_dist apart (10x pixels) so they show different spots"""
    chosen = []
    for i in np.argsort(-att):
        if all(np.abs(coords[i] - coords[j]).max() >= min_dist for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return chosen


def closeup_figure(models, picks, features_dir, coords_dir, thumb_files, full_files, device, out, n_top=3, zoom_tiles=6):
    """Per slide: overview heatmap | high-res zoom with attention | top-n attended tiles (10x) | lowest-attention tile."""
    ncol = 3 + n_top
    fig, axes = plt.subplots(len(picks), ncol, figsize=(1.95 * ncol, 2.05 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (_, s) in enumerate(picks.iterrows()):
        keys = [k for k in models if s.fold == 0 or k[1] == s.fold]
        x = torch.from_numpy(np.load(os.path.join(features_dir, f'{s.image_id}.npy'))).float().to(device)
        coords = np.load(os.path.join(coords_dir, f'{s.image_id}.npy'))
        prob, att = slide_attention(models, keys, x)
        arr = load_slide_10x(full_files[s.image_id], s.is_tma)
        top = spread_top(att, coords, n_top)
        low = int(np.argmin(att))
        # zoom window (10x pixels) centred on the top tile
        half = zoom_tiles * PATCH // 2
        cy, cx = coords[top[0]] + PATCH // 2
        y0 = int(np.clip(cy - half, 0, max(0, arr.shape[0] - 2 * half))); x0 = int(np.clip(cx - half, 0, max(0, arr.shape[1] - 2 * half)))
        crop = arr[y0:y0 + 2 * half, x0:x0 + 2 * half]
        zheat = np.full(crop.shape[:2], np.nan, np.float32)
        for (ty, tx), v in zip(coords, att):
            ys, xs = ty - y0, tx - x0
            if -PATCH < ys < crop.shape[0] and -PATCH < xs < crop.shape[1]:
                zheat[max(0, ys):max(0, ys + PATCH), max(0, xs):max(0, xs + PATCH)] = v
        # overview
        ov = load_image_scaled(thumb_files[s.image_id], 600) if s.image_id in thumb_files else arr[::max(1, arr.shape[1] // 600), ::max(1, arr.shape[1] // 600)]
        scale = ov.shape[1] / s.image_width
        ax = axes[r, 0]
        ax.imshow(ov)
        ax.imshow(np.ma.masked_invalid(paint(ov.shape[:2], coords, s.is_tma, scale, att)), cmap=HEAT, vmin=0, vmax=1, alpha=0.6, interpolation='nearest')
        sh = SHRINK[bool(s.is_tma)]
        ax.add_patch(plt.Rectangle((x0 * sh * scale, y0 * sh * scale), 2 * half * sh * scale, 2 * half * sh * scale, fill=False, ec=INK, lw=1.2))
        pred = int(prob.argmax())
        ax.set_title(f'{s.image_id} · true {CLASSES[s.label]}\npred {CLASSES[pred]} ({prob[pred]:.2f}) {"✓" if pred == s.label else "✗"}',
                     fontsize=7, loc='left', color=INK)
        # zoom with attention + numbered top tiles
        ax = axes[r, 1]
        ax.imshow(crop)
        ax.imshow(np.ma.masked_invalid(zheat), cmap=HEAT, vmin=0, vmax=1, alpha=0.35, interpolation='nearest')
        for n, i in enumerate(top, 1):
            ty, tx = coords[i][0] - y0, coords[i][1] - x0
            if 0 <= ty < crop.shape[0] and 0 <= tx < crop.shape[1]:
                ax.add_patch(plt.Rectangle((tx, ty), PATCH, PATCH, fill=False, ec=INK, lw=1.2))
                ax.text(tx + 8, ty + 40, str(n), color='white', fontsize=7, fontweight='bold')
        ax.set_title(f'zoom ({2 * half} px at ~10x)', fontsize=7, loc='left', color=INK2)
        # tiles
        panels = [(2 + n, i, f'#{n + 1} attention {att[i]:.2f}', INK) for n, i in enumerate(top)]
        panels.append((ncol - 1, low, f'lowest {att[low]:.2f}', INK2))      # always the last column
        for c, i, label, color in panels:
            ty, tx = coords[i]
            axes[r, c].imshow(arr[ty:ty + PATCH, tx:tx + PATCH])
            axes[r, c].set_title(label, fontsize=7, loc='left', color=color)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        del arr
        print(f'[closeup] {s.image_id} done', flush=True)
    save(fig, out, 'fig_heatmap_closeups')


def heatmap_figure(models, picks, features_dir, coords_dir, image_files, mask_files, device, out, canvas_w=900):
    """One column per picked slide: tissue | attention overlay | (mask if available)."""
    has_mask = any(p.image_id in mask_files for _, p in picks.iterrows())
    nrow = 3 if has_mask else 2
    fig, axes = plt.subplots(nrow, len(picks), figsize=(2.0 * len(picks), 2.1 * nrow))
    axes = np.atleast_2d(axes).reshape(nrow, -1)
    for j, (_, s) in enumerate(picks.iterrows()):
        keys = [k for k in models if s.fold == 0 or k[1] == s.fold]
        x = torch.from_numpy(np.load(os.path.join(features_dir, f'{s.image_id}.npy'))).float().to(device)
        coords = np.load(os.path.join(coords_dir, f'{s.image_id}.npy'))
        prob, att = slide_attention(models, keys, x)
        if s.image_id in image_files:
            img = load_image_scaled(image_files[s.image_id], canvas_w)
        else:   # no slide image attached: plain canvas
            h = int(round(canvas_w * s.image_height / s.image_width))
            img = np.full((h, canvas_w, 3), 245, np.uint8)
        scale = img.shape[1] / s.image_width
        heat = paint(img.shape[:2], coords, s.is_tma, scale, att)
        pred = int(prob.argmax())
        axes[0, j].imshow(img)
        axes[1, j].imshow(img)
        axes[1, j].imshow(np.ma.masked_invalid(heat), cmap=HEAT, vmin=0, vmax=1, alpha=0.65, interpolation='nearest')
        ok = '✓' if pred == s.label else '✗'
        axes[0, j].set_title(f'{s.image_id} · true {CLASSES[s.label]}\npred {CLASSES[pred]} ({prob[pred]:.2f}) {ok}', fontsize=7.2,
                             color=INK, loc='left')
        if has_mask:
            if s.image_id in mask_files:
                m = load_image_scaled(mask_files[s.image_id], img.shape[1])
                axes[2, j].imshow(img)
                axes[2, j].imshow(np.ma.masked_where(mask_classes(m) == 0, mask_classes(m)),
                                  cmap=LinearSegmentedColormap.from_list('m', ['#e34948', '#1baf7a']), vmin=1, vmax=2, alpha=0.5)
            else:
                axes[2, j].text(0.5, 0.5, 'no mask', ha='center', va='center', color=INK2, transform=axes[2, j].transAxes)
        for i in range(nrow):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel('tissue', color=INK2)
    axes[1, 0].set_ylabel('attention\n(dark = high)', color=INK2)
    if has_mask:
        axes[2, 0].set_ylabel('mask (red tumour,\ngreen other)', color=INK2)
    save(fig, out, 'fig_heatmaps')
