"""Bags of patch features, one per slide.

Expected layout (written by the Week 2 extraction notebook):
    <features_dir>/<slide_id>.npy    (N, D) float16, one row per tissue tile
Split CSVs (from make_splits.py): file_name,label,is_tma
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASSES = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']


def read_split(path):
    return pd.read_csv(path, dtype={'file_name': str})


class BagDataset(Dataset):
    """Returns (features (N, D) float32, label (), slide_id str).

    max_patches: if set, a random subset of at most this many patches is drawn each time
    (training only — saves memory on huge slides and acts as augmentation).
    cache: False = read from disk every time; True = keep bags in RAM after first load;
           a dict {slide_id: array or torch tensor} = use these preloaded bags (e.g. already on the GPU,
           see preload_features), shared between datasets.
    """

    def __init__(self, df, features_dir, max_patches=None, cache=False, seed=0):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.max_patches = max_patches
        self.cache = cache if isinstance(cache, dict) else ({} if cache else None)
        self.rng = np.random.default_rng(seed)
        missing = [s for s in self.df.file_name
                   if not (self.cache is not None and s in self.cache) and not os.path.exists(self._path(s))]
        if missing:
            raise FileNotFoundError(f'{len(missing)} slides have no features in {features_dir}, e.g. {missing[:3]}')

    def _path(self, slide_id):
        return os.path.join(self.features_dir, f'{slide_id}.npy')

    def _load(self, slide_id):
        if self.cache is not None and slide_id in self.cache:
            return self.cache[slide_id]
        x = np.load(self._path(slide_id))
        if self.cache is not None:
            self.cache[slide_id] = x
        return x

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self._load(row.file_name)
        if self.max_patches and len(x) > self.max_patches:
            idx = np.sort(self.rng.choice(len(x), self.max_patches, replace=False))
            x = x[torch.from_numpy(idx).to(x.device)] if torch.is_tensor(x) else x[idx]
        x = x.float() if torch.is_tensor(x) else torch.from_numpy(np.asarray(x, dtype=np.float32))
        return x, torch.tensor(int(row.label)), row.file_name

    def feature_dim(self):
        return self._load(self.df.file_name.iloc[0]).shape[1]


def preload_features(features_dir, slide_ids, device='cpu'):
    """Load every bag once as a float16 tensor on `device` (e.g. 'cuda:0'). Returns {slide_id: tensor}."""
    return {s: torch.from_numpy(np.load(os.path.join(features_dir, f'{s}.npy'))).to(device) for s in slide_ids}
