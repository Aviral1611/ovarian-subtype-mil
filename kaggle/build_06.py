"""Builds 06_closeups.ipynb: high-resolution attention close-ups (top-attended tiles at ~10x) for the heatmap slides.

    python kaggle/build_06.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, '..', 'code')
OUT = os.path.join(HERE, '06_closeups.ipynb')
read = lambda p: open(p, encoding='utf-8').read()

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 06 · High-resolution attention close-ups (ViT-S/16 + ABMIL)

Same 6 test slides as `fig_heatmaps` (one per subtype + one error). Per slide: overview heatmap · high-res zoom around
the most-attended spot · the 3 most-attended tiles at ~10x (spread apart) · the least-attended tissue tile.

### Inputs (Add Input)
1. `notebook4297f7f927` (Your Work → Notebooks) — features + tile positions
2. `notebookb9c62d68fa` (Your Work → Notebooks) — trained models
3. **UBC-OCEAN** (Competitions) — full-resolution slides

**Settings:** Accelerator **None** (CPU) · Internet **On** · **Run All** (~15–30 min, wait for the data banner first) or commit.
Download `closeups_bundle.zip`.
""")

code(r"""
# 1 · libraries
import time; T0 = time.time()
!pip install -q pyvips-binary pyvips
import os, sys, glob, re
os.makedirs('/tmp/code/mil', exist_ok=True)
""")
code('%%writefile /tmp/code/mil/__init__.py\n' + read(os.path.join(CODE, 'mil', '__init__.py')))
code('%%writefile /tmp/code/mil/models.py\n' + read(os.path.join(CODE, 'mil', 'models.py')))
code('%%writefile /tmp/code/kfigs.py\n' + read(os.path.join(CODE, 'figures', 'kfigs.py')))
code('%%writefile /tmp/code/assignment.csv\n' + read(os.path.join(CODE, 'splits_v2', 'assignment.csv')))
code('%%writefile /tmp/code/kaggle_train.csv\n' + read(os.path.join(CODE, 'data', 'kaggle_train.csv')))

code(r"""
# 2 · inputs
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/tmp/code')
import kfigs
from mil.models import build_model
OUT = '/kaggle/working/figures'; os.makedirs(OUT, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PICK_IDS = ['1952', '22221', '10143', '11557', '2227', '33976']   # same slides as fig_heatmaps

metas = [os.path.dirname(m) for m in glob.glob('/kaggle/input/**/meta.json', recursive=True)]
roots = [r for r in metas if os.path.isdir(os.path.join(r, 'features', 'vits16_lunit')) and os.path.isdir(os.path.join(r, 'coords'))]
assert roots, 'features not attached: Add Input -> Your Work -> Notebooks -> notebook4297f7f927'
FEATS, COORDS = os.path.join(roots[0], 'features', 'vits16_lunit'), os.path.join(roots[0], 'coords')
cands = sorted(glob.glob('/kaggle/input/**/runs/vits16_lunit/abmil', recursive=True),
               key=lambda d: len(glob.glob(f'{d}/seed*/fold*/best.pt')), reverse=True)
assert cands and glob.glob(f'{cands[0]}/seed*/fold*/best.pt'), 'models not attached: Add Input -> notebookb9c62d68fa'
pngs = glob.glob('/kaggle/input/**/*.png', recursive=True)
num = lambda p: (re.match(r'(\d+)', os.path.basename(p)) or [None, None])[1]
FULL = {num(p): p for p in pngs if 'train_images' in p}
THUMBS = {num(p): p for p in pngs if 'train_thumbnails' in p}
assert FULL, 'slides not found: Add Input -> Competitions -> UBC-OCEAN (wait until the data banner finishes)'

a = pd.read_csv('/tmp/code/assignment.csv', dtype={'file_name': str}).rename(columns={'file_name': 'image_id'})
dims = pd.read_csv('/tmp/code/kaggle_train.csv', dtype={'image_id': str})[['image_id', 'image_width', 'image_height']]
SLIDES = a.merge(dims, on='image_id')
dim = np.load(glob.glob(os.path.join(FEATS, '*.npy'))[0], mmap_mode='r').shape[1]
MODELS = kfigs.load_models(cands[0], build_model, dim, device)
PICKS = SLIDES.set_index('image_id').loc[[i for i in PICK_IDS if i in FULL]].reset_index()
print(f'{len(MODELS)} models | {len(FULL)} slides, {len(THUMBS)} thumbnails | close-ups for {list(PICKS.image_id)}')
""")

code(r"""
# 3 · close-up figure
kfigs.closeup_figure(MODELS, PICKS, FEATS, COORDS, THUMBS, FULL, device, OUT)
""")

code(r"""
# 4 · download bundle
import zipfile
with zipfile.ZipFile('/kaggle/working/closeups_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{OUT}/*'):
        z.write(f, os.path.relpath(f, '/kaggle/working'))
print(f'closeups_bundle.zip: {os.path.getsize("/kaggle/working/closeups_bundle.zip") / 1e6:.1f} MB | {(time.time() - T0) / 60:.1f} min')
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
print('wrote', OUT, f'({len(cells)} cells)')
