"""Builds 02_extract_features.ipynb with code/extract/featx.py embedded (single file to import into Kaggle).

    python kaggle/build_02.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FEATX = open(os.path.join(HERE, '..', 'code', 'extract', 'featx.py'), encoding='utf-8').read()
OUT = os.path.join(HERE, '02_extract_features.ipynb')

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 02 · Feature extraction (Week 2)

Reads every UBC-OCEAN slide **once**, fixes colour **per slide** (Macenko, on GPU), and saves features from 4 encoders:
**ViT-S/16 Lunit**, **CTransPath**, **Phikon-v2**, **PLIP**. Patches are never saved, only features (~9 GB total).

### Settings (right panel), every time
1. **Add Input → Competitions → UBC-OCEAN**
2. **Session options → Accelerator → GPU T4 x2**
3. **Session options → Internet → On**

### Step A · smoke test (~15 min, interactive)
`SMOKE_TEST = True` (default) → **Run All**. Check the last two cells: colour check image + time estimate.

### Step B · full run (~6–10 h, runs in the background, laptop can be closed)
Set `SMOKE_TEST = False` → **Save Version → Save & Run All (Commit)**.

### Step C · only if Step B stopped before all 538 slides
Add Input → **Your Work → Notebooks → this notebook** (its output), then commit again. It copies the finished slides and continues.
""")

code(r"""
# 1 · libraries (Kaggle already has torch, timm, transformers)
!pip install -q pyvips-binary pyvips
import torch, timm, transformers
print('torch', torch.__version__, '| timm', timm.__version__, '| transformers', transformers.__version__,
      '| GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

code('%%writefile /tmp/featx.py\n' + FEATX)

code(r"""
# 3 · settings
import os, sys, glob
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/tmp'); import featx

SMOKE_TEST    = True      # True = 3 slides only. Set False for the full run (Step B).
ENCODERS      = ['vits16_lunit', 'ctranspath', 'phikon_v2', 'plip']
TIME_BUDGET_H = 9.5       # stop starting new slides after this; leaves margin for Kaggle's data copy + saving (12 h kill)
N_THREADS     = 2         # slides loaded in parallel while the GPU works
BATCH         = 128       # tiles per GPU batch

hits = [p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)
        if os.path.isdir(os.path.join(os.path.dirname(p), 'train_images'))]
assert hits, 'UBC-OCEAN not attached: Add Input -> Competitions -> UBC-OCEAN'
DATA = os.path.dirname(hits[0])
OUT = '/kaggle/working'

df = pd.read_csv(f'{DATA}/train.csv', dtype={'image_id': str})
df['is_tma'] = df['is_tma'].astype(str).str.lower().eq('true')
df['pixels'] = df.image_width.astype(np.int64) * df.image_height
slides = df.sort_values('pixels').reset_index(drop=True)          # small slides first
if SMOKE_TEST:                                                     # same 3 slides as test run 01
    slides = df[df.image_id.isin(['61320', '35239', '29084'])].reset_index(drop=True)

n_gpu = torch.cuda.device_count()
assert n_gpu > 0, 'No GPU: Session options -> Accelerator -> GPU T4 x2'
heavy = {'phikon_v2', 'gigapath'}                                  # big models go on the 2nd GPU if there is one
DEVICES = {e: f'cuda:{1 if (n_gpu > 1 and e in heavy) else 0}' for e in ENCODERS}

PREV = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/manifest.csv', recursive=True)]
print('data:', DATA, '| slides to process:', len(slides), '| devices:', DEVICES, '| previous runs:', PREV)
""")

code(r"""
# 4 · run
featx.run(slides, DATA, OUT, ENCODERS, DEVICES, time_budget_h=TIME_BUDGET_H, n_threads=N_THREADS,
          batch=BATCH, prev_roots=PREV)
""")

code(r"""
# 5 · checks: completeness, sizes, NaNs, time estimate for the full dataset
man = pd.read_csv(f'{OUT}/manifest.csv', dtype={'image_id': str}).drop_duplicates('image_id', keep='last')
print(f'slides done: {len(man)} / 538 | tiles: {man.n_tiles.sum():,} | Macenko skipped: {(man.macenko != "ok").sum()}')
for e in ENCODERS:
    files = glob.glob(f'{OUT}/features/{e}/*.npy')
    size = sum(os.path.getsize(f) for f in files) / 1e9
    sample = np.load(files[0]) if files else None
    bad = sample is not None and not np.isfinite(sample.astype(np.float32)).all()
    print(f'{e:13s} {len(files):4d} files  {size:6.2f} GB  dim {sample.shape[1] if sample is not None else "-"}  {"NaN/inf FOUND!" if bad else "ok"}')

m = man.merge(df[['image_id', 'pixels']], on='image_id')
for kind, sub_all in [('WSI', df[~df.is_tma]), ('TMA', df[df.is_tma])]:
    r = m[m.is_tma == int(kind == 'TMA')]
    if len(r):
        sec = (r[['t_load_s', 't_tile_s']].sum().sum() / N_THREADS + r.t_gpu_s.sum()) * sub_all.pixels.sum() / r.pixels.sum()
        print(f'estimated full-dataset time for {kind}s: {sec / 3600:.1f} h')
""")

code(r"""
# 6 · colour check: original (top) vs per-slide Macenko (bottom) for the smallest finished slide
import matplotlib.pyplot as plt
row = df[df.image_id.isin(man.image_id)].sort_values('pixels').iloc[0]
arr = featx.load_slide(f'{DATA}/train_images/{row.image_id}.png', featx.TMA_SHRINK if row.is_tma else featx.WSI_SHRINK)
coords = np.load(f'{OUT}/coords/{row.image_id}.npy')
params = featx.fit_slide(arr, coords, np.random.default_rng(0))
pick = np.random.default_rng(1).choice(len(coords), min(6, len(coords)), replace=False)
tiles = np.stack([arr[y:y + 256, x:x + 256] for y, x in coords[pick]])
norm = featx.macenko_apply(torch.from_numpy(tiles), *params).byte().numpy() if params else tiles
fig, ax = plt.subplots(2, len(tiles), figsize=(2.6 * len(tiles), 5.4))
for j in range(len(tiles)):
    ax[0, j].imshow(tiles[j]); ax[1, j].imshow(norm[j]); ax[0, j].axis('off'); ax[1, j].axis('off')
fig.suptitle(f'slide {row.image_id} ({row.label}): top = original, bottom = per-slide Macenko')
plt.tight_layout(); plt.savefig(f'{OUT}/check_colour.png', dpi=90); plt.show()
""")

nb = {'nbformat': 4, 'nbformat_minor': 5,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'cells': []}
for i, (kind, src) in enumerate(cells):
    c = {'cell_type': kind, 'id': f'c{i}', 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)
json.dump(nb, open(OUT, 'w', encoding='utf-8'), indent=1)
print('wrote', OUT)
