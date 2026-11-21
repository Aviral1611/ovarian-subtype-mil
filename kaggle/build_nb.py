"""Builds kaggle/01_test_patching_macenko.ipynb in the Research_Work workspace."""
import json, os, sys

OUT = sys.argv[1]

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 01 · Test run: patching + Macenko colour fix (3 slides)

**Week 1, Step 2.** Goal: prove the pipeline works on Kaggle and measure how long the full dataset will take.

Nothing is saved except small check images and a timing report. No patches are written to disk.

### Settings (right-hand panel) before you press *Run All*
1. **Input → Add Input → Competitions → UBC-OCEAN**
2. **Session options → Accelerator → GPU T4 x2** (or P100)
3. **Session options → Internet → On** (needed for `pip install` and the model download)

Expected run time: about 15–30 minutes.
""")

code(r"""
# Cell 1 — install libraries (needs Internet = On)
!pip install -q pyvips-binary pyvips torchstain
""")

code(r"""
# Cell 2 — imports, find the data, check the GPU
import os, gc, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, timm, pyvips
from torchstain.numpy.normalizers import NumpyMacenkoNormalizer

hits = [p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)
        if os.path.isdir(os.path.join(os.path.dirname(p), 'train_images'))]
assert hits, 'UBC-OCEAN not attached. Right panel: Add Input -> Competitions -> UBC-OCEAN'
DATA = os.path.dirname(hits[0])
OUT = '/kaggle/working'

df = pd.read_csv(f'{DATA}/train.csv')
df['image_id'] = df['image_id'].astype(str)
df['is_tma'] = df['is_tma'].astype(str).str.lower().eq('true')
df['pixels'] = df['image_width'].astype(np.int64) * df['image_height'].astype(np.int64)

print('Data folder :', DATA)
print('Slides      :', len(df), '| WSI', (~df.is_tma).sum(), '| TMA', df.is_tma.sum())
print('GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE -> turn on GPU in Session options')
""")

code(r"""
# Cell 3 — settings (same as the old capstone pipeline, so results stay comparable)
PATCH      = 256      # tile size in pixels, after downsampling
WSI_SHRINK = 2        # WSI 20x -> 10x
TMA_SHRINK = 4        # TMA 40x -> 10x
WSI_STEP   = 256      # no overlap on WSIs
TMA_STEP   = 64       # TMAs are small, so tiles overlap
MIN_TISSUE = 0.25     # keep a tile if >= 25% of its pixels are tissue
SEED = 42

# The 3 test slides: a median-size WSI, a large WSI (90th percentile), and a TMA
wsi = df[~df.is_tma].sort_values('pixels').reset_index(drop=True)
tma = df[df.is_tma].sort_values('pixels').reset_index(drop=True)
picks = pd.DataFrame([wsi.iloc[len(wsi) // 2], wsi.iloc[int(len(wsi) * 0.9)], tma.iloc[len(tma) // 2]])
picks[['image_id', 'label', 'is_tma', 'image_width', 'image_height']]
""")

code(r"""
# Cell 4 — helper functions
GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

def load_slide(path, shrink):
    # pyvips streams the giant PNG instead of loading it all at once
    img = pyvips.Image.new_from_file(path, access='sequential')
    if img.bands > 3:
        img = img.extract_band(0, n=3)
    img = img.shrink(shrink, shrink)
    return img.numpy()

def tissue_tiles(arr, patch, step, min_tissue):
    # Returns (N, 2) array of (y, x) for tiles with enough tissue.
    # Tissue = not black padding (gray > 0) and not white background (gray <= 220).
    coords, fracs = [], []
    H, W = arr.shape[:2]
    for y in range(0, H - patch + 1, step):
        for x in range(0, W - patch + 1, step):
            g = arr[y:y + patch, x:x + patch].astype(np.float32) @ GRAY_W
            frac = float(np.mean((g > 0) & (g <= 220)))
            if frac >= min_tissue:
                coords.append((y, x))
                fracs.append(frac)
    return np.array(coords, dtype=np.int32).reshape(-1, 2), np.array(fracs, dtype=np.float32)

def pick_reference(arr, coords, fracs, patch):
    # Tissue-rich tile that is neither too dark (ink, folds) nor too pale
    best, best_score = None, -1.0
    for (y, x), f in zip(coords, fracs):
        t = arr[y:y + patch, x:x + patch]
        mean_gray = float((t.astype(np.float32) @ GRAY_W).mean())
        if 130 <= mean_gray <= 200 and f > best_score:
            best, best_score = t.copy(), f
    return best

def macenko(normalizer, tile):
    # Returns (normalized_tile, ok). Falls back to the original tile if Macenko fails
    # (it does on tiles with very little stained tissue).
    try:
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            out = normalizer.normalize(I=tile, stains=False)[0]
        if not np.isfinite(out).all():
            return tile, False
        return out, True
    except Exception:
        return tile, False
""")

code(r"""
# Cell 5 — run the 3 slides: load -> tiles -> Macenko (timed)
rng = np.random.default_rng(SEED)
normalizer = NumpyMacenkoNormalizer()
reference = None
report, samples = [], {}

for _, row in picks.iterrows():
    sid, is_tma = row.image_id, bool(row.is_tma)
    shrink = TMA_SHRINK if is_tma else WSI_SHRINK
    step = TMA_STEP if is_tma else WSI_STEP

    t0 = time.time()
    arr = load_slide(f'{DATA}/train_images/{sid}.png', shrink)
    t_load = time.time() - t0

    t0 = time.time()
    coords, fracs = tissue_tiles(arr, PATCH, step, MIN_TISSUE)
    t_tile = time.time() - t0

    if reference is None:  # fit Macenko once, on the first WSI
        reference = pick_reference(arr, coords, fracs, PATCH)
        assert reference is not None, 'No good reference tile found; try another slide'
        normalizer.fit(reference)
        plt.imsave(f'{OUT}/reference_patch.png', reference)

    t0 = time.time()
    fails = 0
    for (y, x) in coords:
        _, ok = macenko(normalizer, arr[y:y + PATCH, x:x + PATCH])
        fails += not ok
    t_norm = time.time() - t0

    pick = rng.choice(len(coords), size=min(6, len(coords)), replace=False)
    samples[sid] = dict(
        tiles=[arr[coords[i][0]:coords[i][0] + PATCH, coords[i][1]:coords[i][1] + PATCH].copy() for i in pick],
        thumb=arr[::max(1, arr.shape[0] // 800), ::max(1, arr.shape[0] // 800)].copy(),
        coords=coords, shape=arr.shape)

    report.append(dict(image_id=sid, label=row.label, is_tma=is_tma, pixels=row.pixels,
                       shape_after_shrink=arr.shape[:2], n_tiles=len(coords), macenko_fail=fails,
                       t_load_s=round(t_load, 1), t_tile_s=round(t_tile, 1), t_macenko_s=round(t_norm, 1)))
    print(report[-1])
    del arr; gc.collect()

report = pd.DataFrame(report)
report
""")

code(r"""
# Cell 6 — check 1: are the tiles on tissue? (red dots = kept tiles)
fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 6))
for ax, (sid, s) in zip(np.atleast_1d(axes), samples.items()):
    k = s['shape'][0] / s['thumb'].shape[0]
    ax.imshow(s['thumb'])
    c = s['coords']
    if len(c):
        ax.scatter((c[:, 1] + PATCH / 2) / k, (c[:, 0] + PATCH / 2) / k, s=2, c='red')
    ax.set_title(f'{sid}  ({len(c)} tiles)'); ax.axis('off')
plt.tight_layout(); plt.savefig(f'{OUT}/check_tissue_tiles.png', dpi=100); plt.show()
""")

code(r"""
# Cell 7 — check 2: Macenko before (top) vs after (bottom), plus the reference tile
for sid, s in samples.items():
    tiles = s['tiles']
    fig, axes = plt.subplots(2, len(tiles) + 1, figsize=(2.4 * (len(tiles) + 1), 5))
    axes[0, 0].imshow(reference); axes[0, 0].set_title('reference'); axes[1, 0].axis('off')
    for j, t in enumerate(tiles, start=1):
        n, ok = macenko(normalizer, t)
        axes[0, j].imshow(t); axes[1, j].imshow(n)
        axes[1, j].set_title('ok' if ok else 'FAILED -> original', fontsize=8)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'slide {sid}: top = original, bottom = Macenko')
    plt.tight_layout(); plt.savefig(f'{OUT}/check_macenko_{sid}.png', dpi=100); plt.show()
""")

code(r"""
# Cell 8 — GPU speed: real ViT-S/16 (Lunit) and a ViT-L/16 stand-in for UNI (same size, random weights)
dev = 'cuda'
tiles = np.stack([t for s in samples.values() for t in s['tiles']] * 30)[:512]   # 512 real tiles
x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div(255)
x = torch.nn.functional.interpolate(x, size=224, mode='bilinear', antialias=True)

def throughput(model, batch=128, reps=3):
    model = model.eval().to(dev).half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.inference_mode():
        model(((x[:batch].to(dev) - mean) / std).half())            # warm-up
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            for i in range(0, len(x), batch):
                model(((x[i:i + batch].to(dev) - mean) / std).half())
        torch.cuda.synchronize()
    return reps * len(x) / (time.time() - t0)

speed = {}
speed['vits16_lunit'] = throughput(timm.create_model('hf-hub:1aurent/vit_small_patch16_224.lunit_dino', pretrained=True, num_classes=0))
speed['vitl16_uni_size'] = throughput(timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0))
for k, v in speed.items(): print(f'{k:18s} {v:8.0f} tiles/sec')
""")

code(r"""
# Cell 9 — estimate for ALL 538 slides (scaled by pixel count, separately for WSI and TMA)
est = {}
for kind, sub in [('WSI', df[~df.is_tma]), ('TMA', df[df.is_tma])]:
    r = report[report.is_tma == (kind == 'TMA')]
    scale = sub.pixels.sum() / r.pixels.sum()
    est[kind] = dict(load_h=r.t_load_s.sum() * scale / 3600,
                     tile_h=r.t_tile_s.sum() * scale / 3600,
                     macenko_h=r.t_macenko_s.sum() * scale / 3600,
                     tiles=r.n_tiles.sum() * scale)
est = pd.DataFrame(est).T
total_tiles = est.tiles.sum()
cpu_h = est[['load_h', 'tile_h', 'macenko_h']].sum().sum()
print(est.round(2))
print(f'\nEstimated tiles in whole dataset : {total_tiles:,.0f}')
print(f'CPU time (load + tiles + Macenko): {cpu_h:.1f} h')
for k, v in speed.items():
    print(f'GPU time for {k:18s}: {total_tiles / v / 3600:.1f} h')
print(f'Macenko failure rate in test     : {report.macenko_fail.sum() / report.n_tiles.sum():.1%}')

report.to_csv(f'{OUT}/test_report.csv', index=False)
pd.Series({**{f'speed_{k}': v for k, v in speed.items()}, 'total_tiles_est': total_tiles, 'cpu_hours_est': cpu_h}).to_csv(f'{OUT}/estimate.csv')
print('\nSaved: test_report.csv, estimate.csv, reference_patch.png, check_*.png  (Output tab, right panel)')
""")

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}},
    'cells': [],
}
for i, (kind, src) in enumerate(cells):
    c = {'cell_type': kind, 'id': f'c{i}', 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), indent=1)
print('wrote', OUT, len(cells), 'cells')
