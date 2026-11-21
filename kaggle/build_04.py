"""Builds 04_figures.ipynb (t-SNE, attention heatmaps, attention-vs-tumour-mask agreement). One file to import.

    python kaggle/build_04.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, '..', 'code')
OUT = os.path.join(HERE, '04_figures.ipynb')
read = lambda p: open(p, encoding='utf-8').read()

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 04 · Figures: t-SNE, attention heatmaps, heatmap vs tumour masks

1. **t-SNE** of all 4 encoders (slide level + 1,000 patches), with silhouette score and a linear-probe accuracy.
2. **Attention heatmaps** of ViT-S/16 + ABMIL (average of its 15 models), one test slide per subtype + one error.
3. **Do heatmaps point at tumour?** For every slide with a pathologist mask: AUROC of attention for tumour vs non-tumour
   tiles, using only models that never trained on that slide.

### Inputs (right panel → Add Input)
| Input | Where | Needed for |
|---|---|---|
| `notebook4297f7f927` | Your Work → Notebooks | features + tile positions (**required**) |
| the 3-seed training notebook | Your Work → Notebooks | trained ABMIL models (**required**) |
| UBC-OCEAN | Competitions | tissue images under the heatmaps (recommended) |
| *UBC Ovarian Cancer Competition Supplemental Masks* | Datasets (search the name) | heatmap vs tumour check (recommended) |

**Settings:** Accelerator **None** (CPU is enough; saves GPU hours) · Internet **On**.
Run interactively (**Run All**) or as a commit. Download `figures_bundle.zip` at the end.
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
# 2 · find the inputs
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/tmp/code')
import kfigs
from mil.models import build_model
OUT = '/kaggle/working/figures'
os.makedirs(OUT, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

metas = [os.path.dirname(m) for m in glob.glob('/kaggle/input/**/meta.json', recursive=True)]
roots = [r for r in metas if os.path.isdir(os.path.join(r, 'features', 'vits16_lunit')) and os.path.isdir(os.path.join(r, 'coords'))]
assert roots, 'features not attached: Add Input -> Your Work -> Notebooks -> notebook4297f7f927'
FEATURES_ROOT, COORDS = os.path.join(roots[0], 'features'), os.path.join(roots[0], 'coords')
ENCODERS = [e for e in ['vits16_lunit', 'ctranspath', 'phikon_v2', 'plip'] if os.path.isdir(os.path.join(FEATURES_ROOT, e))]

cands = glob.glob('/kaggle/input/**/runs/vits16_lunit/abmil', recursive=True)
cands = sorted(cands, key=lambda d: len(glob.glob(f'{d}/seed*/fold*/best.pt')), reverse=True)
assert cands and glob.glob(f'{cands[0]}/seed*/fold*/best.pt'), 'trained models not attached: Add Input -> the 3-seed training notebook'
RUNS = cands[0]

pngs = glob.glob('/kaggle/input/**/*.png', recursive=True)
num = lambda p: (re.match(r'(\d+)', os.path.basename(p)) or [None, None])[1]
IMAGES = {num(p): p for p in pngs if 'train_images' in p}                        # full slides (fallback)
IMAGES.update({num(p): p for p in pngs if 'train_thumbnails' in p})             # thumbnails win (faster)
MASKS = {num(p): p for p in pngs if 'mask' in p.lower() and 'train_images' not in p and 'thumbnail' not in p and num(p)}

a = pd.read_csv('/tmp/code/assignment.csv', dtype={'file_name': str}).rename(columns={'file_name': 'image_id'})
dims = pd.read_csv('/tmp/code/kaggle_train.csv', dtype={'image_id': str})[['image_id', 'image_width', 'image_height']]
SLIDES = a.merge(dims, on='image_id')
print(f'features: {FEATURES_ROOT} ({ENCODERS})\nmodels: {RUNS} ({len(glob.glob(RUNS + "/seed*/fold*/best.pt"))} checkpoints)')
print(f'slide images: {len(IMAGES)} | masks: {len(MASKS)} | slides: {len(SLIDES)} | device: {device}')
""")

code(r"""
# 3 · t-SNE + silhouette + linear probe for every encoder
scores = kfigs.tsne_figure(FEATURES_ROOT, SLIDES[['image_id', 'label']], OUT, ENCODERS)
print(scores.round(3).to_string(index=False))
""")

code(r"""
# 4 · load the ViT-S/16 + ABMIL models, predict the test set, pick slides to show
dim = np.load(glob.glob(os.path.join(FEATURES_ROOT, 'vits16_lunit', '*.npy'))[0], mmap_mode='r').shape[1]
MODELS = kfigs.load_models(RUNS, build_model, dim, device)
test = SLIDES[SLIDES.fold == 0].copy()
pp = []
for _, s in test.iterrows():
    x = torch.from_numpy(np.load(os.path.join(FEATURES_ROOT, 'vits16_lunit', f'{s.image_id}.npy'))).float().to(device)
    prob, _ = kfigs.slide_attention(MODELS, list(MODELS), x)
    pp.append(prob)
test['pred'] = np.array(pp).argmax(1); test['conf'] = np.array(pp).max(1)
test['has_mask'] = test.image_id.isin(MASKS); test['has_img'] = test.image_id.isin(IMAGES)
print(f'{len(MODELS)} models | test accuracy {(test.pred == test.label).mean():.3f}')

picks = []
for c in range(5):   # one correctly classified test WSI per subtype; prefer one with a mask and an image
    cand = test[(test.label == c) & (test.pred == c) & (test.is_tma == 0)]
    cand = cand.sort_values(['has_mask', 'has_img', 'conf'], ascending=False)
    if len(cand): picks.append(cand.iloc[0])
wrong = test[(test.pred != test.label) & (test.is_tma == 0)].sort_values(['has_mask', 'has_img', 'conf'], ascending=False)
if len(wrong): picks.append(wrong.iloc[0])
PICKS = pd.DataFrame(picks)
print(PICKS[['image_id', 'label', 'pred', 'conf', 'has_mask', 'has_img']].to_string(index=False))
""")

code(r"""
# 5 · heatmap figure
kfigs.heatmap_figure(MODELS, PICKS, os.path.join(FEATURES_ROOT, 'vits16_lunit'), COORDS, IMAGES, MASKS, device, OUT)
""")

code(r"""
# 6 · does attention point at tumour? (every slide with a mask, only models that never trained on it)
if MASKS:
    agree = kfigs.mask_agreement(MODELS, SLIDES, os.path.join(FEATURES_ROOT, 'vits16_lunit'), COORDS, MASKS, device, OUT)
    ok = agree.dropna(subset=['auroc'])
    print(f'slides with a mask: {len(agree)} | usable (>=5 tumour and >=5 non-tumour tiles): {len(ok)}')
    if len(ok):
        print(f'attention AUROC tumour vs non-tumour: mean {ok.auroc.mean():.3f} ± {ok.auroc.std():.3f}, median {ok.auroc.median():.3f}, '
              f'> 0.5 in {(ok.auroc > 0.5).mean():.0%} of slides')
        print(f'top-10% attention tiles that are tumour: {ok.top10_tumour_frac.mean():.3f} vs {ok.base_tumour_frac.mean():.3f} for all annotated tiles')
        print(ok.groupby('label').auroc.agg(['count', 'mean']).round(3).to_string())
else:
    print('no masks attached: skipped (Add Input -> Datasets -> "UBC Ovarian Cancer Competition Supplemental Masks")')
""")

code(r"""
# 7 · download bundle
import zipfile
with zipfile.ZipFile('/kaggle/working/figures_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{OUT}/*'):
        z.write(f, os.path.relpath(f, '/kaggle/working'))
print(f'figures_bundle.zip: {os.path.getsize("/kaggle/working/figures_bundle.zip") / 1e6:.1f} MB | total {(time.time() - T0) / 60:.1f} min')
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
