"""Builds 05_tune.ipynb: ABMIL/DSMIL hyperparameter tuning (selection on CV only) + tuned-vs-default test comparison.

    python kaggle/build_05.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, '..', 'code')
OUT = os.path.join(HERE, '05_tune.ipynb')
MODULES = ['__init__', 'models', 'data', 'metrics', 'train', 'evaluate', 'grid', 'stats', 'tune']
read = lambda p: open(p, encoding='utf-8').read()

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# 05 · Hyperparameter tuning for ABMIL and DSMIL (roadmap item)

For each encoder × {ABMIL, DSMIL}: **13 settings** (the default + 12 random: learning rate, weight decay, dropout,
hidden size, class weights) × 5 folds, seed 0 → pick the best by **cross-validation only** → retrain the winner with
seeds 1–2 → compare **tuned vs default** on the locked test set and on CV (paired bootstrap, Holm). ~600 runs, ~3–4 h.

### Inputs (Add Input → Your Work → Notebooks)
1. `notebook4297f7f927` — features (**required**)
2. `notebookb9c62d68fa` — the 3-seed benchmark (the **default** models to compare against; recommended)

**Settings:** Accelerator **GPU T4 x2** · Internet **On** · then **Save Version → Save & Run All (Commit)**.
If it stops early: attach **this notebook's** output too and commit again (finished runs are copied).
""")

code(r"""
# 1 · start clock
import time; T0 = time.time()
import os, sys, glob
import torch; print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
os.makedirs('/tmp/code/mil', exist_ok=True); os.makedirs('/tmp/code/splits_v2', exist_ok=True)
""")
for m in MODULES:
    code(f'%%writefile /tmp/code/mil/{m}.py\n' + read(os.path.join(CODE, 'mil', f'{m}.py')))
code('%%writefile /tmp/code/splits_v2/assignment.csv\n' + read(os.path.join(CODE, 'splits_v2', 'assignment.csv')))

code(r"""
# 2 · unpack splits (fold 0 = locked test set)
import pandas as pd
S = '/tmp/code/splits_v2'
a = pd.read_csv(f'{S}/assignment.csv', dtype={'file_name': str})
os.makedirs(f'{S}/train', exist_ok=True); os.makedirs(f'{S}/val', exist_ok=True)
cols = ['file_name', 'label', 'is_tma']
a[a.fold == 0][cols].to_csv(f'{S}/test.csv', index=False)
for k in range(1, 6):
    a[(a.fold != 0) & (a.fold != k)][cols].to_csv(f'{S}/train/{k}.csv', index=False)
    a[a.fold == k][cols].to_csv(f'{S}/val/{k}.csv', index=False)
print('splits ready:', a.fold.value_counts().sort_index().to_dict())
""")

code(r"""
# 3 · settings + inputs
from pathlib import Path
SMOKE_TEST    = False     # True = 1 encoder, 2 configs, 3 epochs (quick check)
ENCODERS      = ['phikon_v2', 'ctranspath', 'vits16_lunit', 'plip']   # 2 GPUs -> [phikon_v2, vits16_lunit] + [ctranspath, plip]
MODELS        = ['abmil', 'dsmil']
TIME_BUDGET_H = 10.0
OUT           = '/kaggle/working'

metas = [os.path.dirname(m) for m in glob.glob('/kaggle/input/**/meta.json', recursive=True)]
roots = [r for r in metas if os.path.isdir(os.path.join(r, 'features', 'plip'))]
assert roots, 'features not attached: Add Input -> Your Work -> Notebooks -> notebook4297f7f927'
FEATURES_ROOT = os.path.join(roots[0], 'features')
DEFAULT_RUNS = sorted({str(Path(p).parent.parent.parent) for p in glob.glob('/kaggle/input/**/runs/*/abmil/seed2', recursive=True)})
DEFAULT_RUNS = DEFAULT_RUNS[0] if DEFAULT_RUNS else None
PREV = sorted({str(Path(p).parent) for p in glob.glob('/kaggle/input/**/tune/search', recursive=True)})

extra, configs = [], None
if SMOKE_TEST:
    ENCODERS, configs, extra = ['vits16_lunit'], 'c00,c03', ['--epochs', '3']
n_gpu = torch.cuda.device_count()
assert n_gpu > 0, 'No GPU: Session options -> Accelerator -> GPU T4 x2'
groups = [ENCODERS[i::n_gpu] for i in range(n_gpu)] if len(ENCODERS) > 1 else [ENCODERS]
print('features:', FEATURES_ROOT, '\ndefault runs to compare against:', DEFAULT_RUNS, '\nprevious tune outputs:', PREV,
      '\nGPU plan:', {f'cuda:{i}': g for i, g in enumerate(groups)})
""")

code(r"""
# 4 · run: one process per GPU, progress every 5 minutes
import subprocess
os.makedirs(f'{OUT}/logs', exist_ok=True)
budget = TIME_BUDGET_H - (time.time() - T0) / 3600
procs = []
for i, encs in enumerate(groups):
    cmd = ['python', '-m', 'mil.tune', '--features_root', FEATURES_ROOT, '--splits', '/tmp/code/splits_v2',
           '--out', f'{OUT}/tune', '--encoders', ','.join(encs), '--models', ','.join(MODELS), '--device', f'cuda:{i}',
           '--time_budget_h', f'{budget:.2f}', '--prev', ','.join(PREV)] + (['--configs', configs] if configs else []) + extra
    log = open(f'{OUT}/logs/tune_gpu{i}.log', 'w')
    procs.append((subprocess.Popen(cmd, cwd='/tmp/code', stdout=log, stderr=subprocess.STDOUT), f'{OUT}/logs/tune_gpu{i}.log'))

def tail(path):
    return [l.rstrip() for l in open(path) if l.startswith(('==', '[', 'TUNE')) or 'Error' in l][-3:]

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
# 4b · safety bundle right after training (raw predictions + selection tables)
import zipfile
def bundle():
    with zipfile.ZipFile(f'{OUT}/tune_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in (glob.glob(f'{OUT}/logs/*') + glob.glob(f'{OUT}/results/*')
                  + glob.glob(f'{OUT}/tune/**/*.csv', recursive=True) + glob.glob(f'{OUT}/tune/**/*.json', recursive=True)):
            z.write(f, os.path.relpath(f, OUT))
    print(f'tune_bundle.zip: {os.path.getsize(f"{OUT}/tune_bundle.zip") / 1e6:.1f} MB')
bundle()
""")

code(r"""
# 5 · what tuning chose (CV, seed 0) and the tuned models on the LOCKED TEST SET (3 seeds, seed ensemble, 95% CI)
sys.path.insert(0, '/tmp/code')
from mil.evaluate import summarize_config
from mil.tune import compare
os.makedirs(f'{OUT}/results', exist_ok=True)
sel = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/tune/selection/*.csv'))], ignore_index=True)
sel.to_csv(f'{OUT}/results/search_all_configs.csv', index=False)
best = sel.loc[sel.groupby(['encoder', 'model']).cv_bacc.idxmax()]
dflt = sel[sel.config == 'c00'][['encoder', 'model', 'cv_bacc']].rename(columns={'cv_bacc': 'default_cv_bacc'})
choice = best.merge(dflt, on=['encoder', 'model'])
print('Chosen by CV (seed 0):\n' + choice[['encoder', 'model', 'config', 'default_cv_bacc', 'cv_bacc', 'lr', 'wd', 'dropout', 'size', 'class_weights']].round(4).to_string(index=False))

tuned_dirs = sorted(d for d in glob.glob(f'{OUT}/tune/tuned/*/*') if glob.glob(f'{d}/seed*/fold*/test_preds.csv'))
tuned = pd.DataFrame([summarize_config(d, n_boot=1000) for d in tuned_dirs])
tuned.to_csv(f'{OUT}/results/tuned_summary.csv', index=False)
fmt = lambda r: f"{r.ens_bacc:.3f} [{r.ens_bacc_lo:.2f}-{r.ens_bacc_hi:.2f}]"
print('\nTuned, TEST balanced accuracy, seed ensemble [95% CI]\n'
      + tuned.assign(v=[fmt(r) for _, r in tuned.iterrows()]).pivot(index='model', columns='encoder', values='v').to_string())
""")

code(r"""
# 6 · tuned vs default (paired bootstrap, Holm-corrected within each split)
if DEFAULT_RUNS:
    cmp = compare(DEFAULT_RUNS, f'{OUT}/tune', ENCODERS, MODELS, n_boot=10000)
    cmp.to_csv(f'{OUT}/results/tuned_vs_default.csv', index=False)
    print(cmp.round(3).to_string(index=False))
else:
    print('default runs not attached: skipped (Add Input -> Your Work -> Notebooks -> notebookb9c62d68fa)')
bundle()
print(f'total {(time.time() - T0) / 3600:.2f} h')
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
