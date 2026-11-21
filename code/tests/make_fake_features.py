"""Make fake patch features for every slide in the splits, to test the pipeline without real data.

Each slide gets 30-300 patches. 10-30% of them are "tumour" patches near a class-specific
prototype; the rest are background noise shared by all classes. So a model that finds the
few informative patches (max-pool, attention) should beat one that averages them away.

    python tests/make_fake_features.py --splits splits_v2 --out fake_features --dim 64
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', default='splits_v2')
    p.add_argument('--out', default='fake_features')
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--signal', type=float, default=1.2, help='distance of class prototypes from background')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    ids = pd.concat([pd.read_csv(f, dtype={'file_name': str}) for f in
                     [os.path.join(args.splits, 'test.csv')] +
                     [os.path.join(args.splits, s, f'{k}.csv') for s in ('train', 'val') for k in range(1, 6)]])
    ids = ids.drop_duplicates('file_name')

    protos = rng.normal(0, 1, (5, args.dim))
    protos = args.signal * protos / np.linalg.norm(protos, axis=1, keepdims=True) * np.sqrt(args.dim) / 4
    os.makedirs(args.out, exist_ok=True)
    for row in ids.itertuples():
        n = int(rng.integers(30, 301))
        x = rng.normal(0, 1, (n, args.dim))
        k = max(1, int(n * rng.uniform(0.1, 0.3)))
        x[rng.choice(n, k, replace=False)] += protos[row.label]
        np.save(os.path.join(args.out, f'{row.file_name}.npy'), x.astype(np.float16))
    print(f'wrote {len(ids)} fake bags to {args.out}/ (dim {args.dim})')


if __name__ == '__main__':
    main()
