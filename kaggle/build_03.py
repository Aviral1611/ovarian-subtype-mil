"""Builds 03_train_mil.ipynb with the whole mil/ package and the splits embedded (one file to import).

    python kaggle/build_03.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, '..', 'code')
OUT = os.path.join(HERE, '03_train_mil.ipynb')
MODULES = ['__init__', 'models', 'data', 'metrics', 'train', 'evaluate', 'grid', 'stats']
read = lambda p: open(p, encoding='utf-8').read()

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 03 · Train MIL models (Week 4)

Trains **6 MIL models** (maxpool, meanpool, ABMIL, DSMIL, CLAM-SB, TransMIL) on **4 encoders' features**
(ViT-S/16 Lunit, CTransPath, Phikon-v2, PLIP) × **5 folds** = 120 runs, then scores the **locked 108-slide test set**
with 95% confidence intervals. Features are loaded onto the GPUs once; both GPUs work in parallel.

### Settings (right panel)
1. **Add Input → Your Work → Notebooks → `notebook4297f7f927`** (the Week 2 features). The competition data is NOT needed.
2. **Session options → Accelerator → GPU T4 x2**
3. **Session options → Internet → On**

### Step A · smoke test (~15 min, interactive)
`SMOKE_TEST = True` → **Run All**. It trains 3 models on 1 fold of ViT-S/16 for 5 epochs and prints a small results table.

### Step B · full run (background)
`SMOKE_TEST = False` → **Save Version → Save & Run All (Commit)**.

### Step C · only if Step B stopped early
Add Input → Your Work → Notebooks → **this notebook** (its output), then commit again. Finished runs are copied, not redone.

### Seeds run (3 seeds) · `SEEDS = '0,1,2'` (default)
Attach the output of the earlier seed-0 training run (Add Input → Your Work → Notebooks → that notebook). Its 120 seed-0 runs
are copied; only seeds 1 and 2 are trained (240 runs, ~3.5 h). Results then use all 15 models (3 seeds × 5 folds) per
combination, and cell 5b runs the paired significance tests on the test set (108 slides) and on the CV predictions (430 slides).
""")

code(r"""
# 1 · libraries + start clock
import time; T0 = time.time()
!pip install -q nystrom-attention einops
import torch; print('torch', torch.__version__, '| GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

code("import os; os.makedirs('/tmp/code/mil', exist_ok=True); os.makedirs('/tmp/code/splits_v2', exist_ok=True)")
for m in MODULES:
    code(f'%%writefile /tmp/code/mil/{m}.py\n' + read(os.path.join(CODE, 'mil', f'{m}.py')))

code('%%writefile /tmp/code/splits_v2/assignment.csv\n' + read(os.path.join(CODE, 'splits_v2', 'assignment.csv')))

code(r"""
# 2 · unpack splits: fold 0 = locked test set, fold k = validation set of CV fold k
import pandas as pd
S = '/tmp/code/splits_v2'
a = pd.read_csv(f'{S}/assignment.csv', dtype={'file_name': str})
os.makedirs(f'{S}/train', exist_ok=True); os.makedirs(f'{S}/val', exist_ok=True)
cols = ['file_name', 'label', 'is_tma']
a[a.fold == 0][cols].to_csv(f'{S}/test.csv', index=False)
for k in range(1, 6):
    a[(a.fold != 0) & (a.fold != k)][cols].to_csv(f'{S}/train/{k}.csv', index=False)
    a[a.fold == k][cols].to_csv(f'{S}/val/{k}.csv', index=False)
assert len(a) == 538 and (a.fold == 0).sum() == 108
print('splits ready:', a.fold.value_counts().sort_index().to_dict())
""")

code(r"""
# 3 · settings
import glob
SMOKE_TEST    = True       # True = quick check. Set False for the full run (Step B).
ENCODERS      = ['phikon_v2', 'ctranspath', 'vits16_lunit', 'plip']   # 2 GPUs -> [phikon_v2, vits16_lunit] + [ctranspath, plip]
MODELS        = ['maxpool', 'meanpool', 'abmil', 'dsmil', 'clam_sb', 'transmil']
SEEDS         = '0,1,2'    # seed 0 is copied from an attached earlier run if present
TIME_BUDGET_H = 10.0      # no new run starts after this (Kaggle stops sessions at 12 h)
OUT           = '/kaggle/working'

metas = glob.glob('/kaggle/input/**/meta.json', recursive=True)
roots = [os.path.dirname(m) for m in metas if os.path.isdir(os.path.join(os.path.dirname(m), 'features', 'plip'))]
assert roots, 'Week 2 features not attached: Add Input -> Your Work -> Notebooks -> notebook4297f7f927'
FEATURES_ROOT = os.path.join(roots[0], 'features')
# earlier versions of THIS notebook attached as input (Step C): their runs/ folders
from pathlib import Path
PREV = sorted({str(Path(p).parent.parent.parent) for p in glob.glob('/kaggle/input/**/runs/*/*/seed*', recursive=True)})

extra = []
if SMOKE_TEST:
    ENCODERS, MODELS, SEEDS = ['vits16_lunit'], ['maxpool', 'abmil', 'transmil'], '0'
    extra = ['--folds', '1', '--epochs', '5', '--min_epochs', '1', '--patience', '5']

n_gpu = torch.cuda.device_count()
assert n_gpu > 0, 'No GPU: Session options -> Accelerator -> GPU T4 x2'
# split encoders over the GPUs so both get similar work
groups = [ENCODERS[i::n_gpu] for i in range(n_gpu)] if len(ENCODERS) > 1 else [ENCODERS]
print('features:', FEATURES_ROOT, '\nprevious runs:', PREV, '\nGPU plan:', {f'cuda:{i}': g for i, g in enumerate(groups)})
""")

code(r"""
# 4 · run: one background process per GPU, progress printed every 5 minutes
import subprocess
os.makedirs(f'{OUT}/logs', exist_ok=True)
budget = TIME_BUDGET_H - (time.time() - T0) / 3600
procs = []
for i, encs in enumerate(groups):
    cmd = ['python', '-m', 'mil.grid', '--features_root', FEATURES_ROOT, '--splits', '/tmp/code/splits_v2',
           '--out', f'{OUT}/runs', '--encoders', ','.join(encs), '--models', ','.join(MODELS), '--seeds', SEEDS,
           '--device', f'cuda:{i}', '--time_budget_h', f'{budget:.2f}', '--prev', ','.join(PREV)] + extra
    log = open(f'{OUT}/logs/gpu{i}.log', 'w')
    procs.append((subprocess.Popen(cmd, cwd='/tmp/code', stdout=log, stderr=subprocess.STDOUT), f'{OUT}/logs/gpu{i}.log'))

def tail(path, keys=('==', 'loaded', 'copied', 'GRID', 'budget', 'Error', 'Traceback')):
    lines = [l.rstrip() for l in open(path) if l.startswith(keys) or 'Error' in l]
    return lines[-3:]

while any(p.poll() is None for p, _ in procs):
    time.sleep(10 if SMOKE_TEST else 300)
    print(f'--- {(time.time() - T0) / 3600:.2f} h')
    for p, path in procs:
        for l in tail(path): print(f'  {os.path.basename(path)}: {l}')
for p, path in procs:
    print(f'{os.path.basename(path)} exit code {p.returncode}')
    if p.returncode != 0:
        print(''.join(open(path).readlines()[-30:]))
""")

code(r"""
# 4b · safety copy: bundle the raw predictions right after training (cell 6 rebuilds it with the results tables)
import zipfile
with zipfile.ZipFile(f'{OUT}/results_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{OUT}/logs/*') + glob.glob(f'{OUT}/runs/**/*.csv', recursive=True) + glob.glob(f'{OUT}/runs/**/*.json', recursive=True):
        z.write(f, os.path.relpath(f, OUT))
print(f'safety bundle: {os.path.getsize(f"{OUT}/results_bundle.zip") / 1e6:.1f} MB')
""")

code(r"""
# 5 · results on the LOCKED TEST SET
#   per seed: 5-fold ensemble (summary_per_seed.csv)
#   per encoder+model: mean ± std over seeds, and the SEED ENSEMBLE (all seeds x folds averaged) with 95% CI (summary.csv)
import sys; sys.path.insert(0, '/tmp/code')
from pathlib import Path
from mil.evaluate import summarize_run, summarize_config
seed_runs = sorted(d for d in glob.glob(f'{OUT}/runs/*/*/seed*') if glob.glob(f'{d}/fold*/test_preds.csv'))
per_seed = pd.DataFrame([summarize_run(d, n_boot=1000) for d in seed_runs])
per_seed['encoder'] = [Path(r).parts[-3] for r in per_seed.run]; per_seed['model'] = [Path(r).parts[-2] for r in per_seed.run]
per_seed['seed'] = [Path(r).parts[-1] for r in per_seed.run]
configs = sorted({str(Path(d).parent) for d in seed_runs})
res = pd.DataFrame([summarize_config(c, n_boot=1000) for c in configs])
os.makedirs(f'{OUT}/results', exist_ok=True)
per_seed.to_csv(f'{OUT}/results/summary_per_seed.csv', index=False)
res.to_csv(f'{OUT}/results/summary.csv', index=False)

piv = lambda col: res.pivot(index='model', columns='encoder', values=col)
ens = lambda m: res.assign(v=[f"{r[f'ens_{m}']:.3f} [{r[f'ens_{m}_lo']:.2f}-{r[f'ens_{m}_hi']:.2f}]" for _, r in res.iterrows()])
for m, name in [('bacc', 'balanced accuracy'), ('macro_f1', 'macro F1'), ('macro_auc', 'macro AUC')]:
    print(f'\nTEST {name}, seed ensemble [95% CI]  (rows = MIL model, columns = encoder)\n'
          + ens(m).pivot(index='model', columns='encoder', values='v').to_string())
ms = res.assign(v=[f"{r.seedmean_bacc:.3f} ± {r.seedstd_bacc:.3f}" for _, r in res.iterrows()])
print('\nTEST balanced accuracy, mean ± std over seeds\n' + ms.pivot(index='model', columns='encoder', values='v').to_string())
print('\nCV balanced accuracy (mean over seeds; for model selection)\n' + piv('cv_bacc_mean').round(3).to_string())
print(f'\nseeds per config: {res.n_seeds.min()}-{res.n_seeds.max()} | models per config: {res.n_models.min()}-{res.n_models.max()}')
""")

code(r"""
# 5b · paired bootstrap significance tests (Holm-corrected), on the test set and on the CV predictions
from mil.stats import run as significance
for split in ['test', 'cv']:
    sig, n = significance(f'{OUT}/runs', n_boot=10000, split=split)
    sig.to_csv(f'{OUT}/results/significance_{split}.csv', index=False)
    b = sig[sig.metric == 'bacc']
    print(f'\n[{split}: {n} slides] significant after Holm: {int(b.significant.sum())} of {len(b)} comparisons (balanced accuracy)')
    show = b[b.significant | b.family.str.endswith('overall')][['family', 'a', 'b', 'diff', 'diff_lo', 'diff_hi', 'p_holm', 'significant']]
    print(show.round(3).to_string(index=False))
""")

code(r"""
# 6 · small download bundle: results/ + every prediction file (no model weights)
import shutil, zipfile
with zipfile.ZipFile(f'{OUT}/results_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{OUT}/results/*') + glob.glob(f'{OUT}/logs/*'):
        z.write(f, os.path.relpath(f, OUT))
    for f in glob.glob(f'{OUT}/runs/**/*.csv', recursive=True) + glob.glob(f'{OUT}/runs/**/*.json', recursive=True):
        z.write(f, os.path.relpath(f, OUT))
print(f'results_bundle.zip: {os.path.getsize(f"{OUT}/results_bundle.zip") / 1e6:.1f} MB (download this one file)')
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
