"""UBC-OCEAN feature extraction: read each slide once, colour-normalize per slide, run every encoder.

Pipeline per slide:
    PNG --pyvips shrink--> 10x array --tissue tiles 256px--> per-slide Macenko (GPU)
        --> resize 224 --> each encoder (own mean/std) --> <out>/features/<encoder>/<slide_id>.npy

Outputs (all under OUT):
    features/<encoder>/<slide_id>.npy   (N, D) float16, one row per tile
    coords/<slide_id>.npy               (N, 2) int32, (y, x) of each tile's top-left corner, at 10x
    manifest.csv                        one row per finished slide (tiles, timings, Macenko status)
    meta.json                           settings used

Resumable: slides whose files already exist are skipped, files are written atomically, and the run
stops starting new slides once TIME_BUDGET_H is used up, so Kaggle can save the output.
"""
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- settings
PATCH = 256          # tile size after shrinking
WSI_SHRINK = 2       # WSI 20x -> 10x
TMA_SHRINK = 4       # TMA 40x -> 10x
WSI_STEP = 256       # WSI tiles do not overlap
TMA_STEP = 64        # TMA tiles overlap (TMAs are small)
MIN_TISSUE = 0.25    # keep a tile if >= 25% of its pixels are tissue
MODEL_INPUT = 224    # every encoder gets the whole 256 tile resized to 224

GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# ----------------------------------------------------------------------------- Macenko
# BETA: pixels with optical density below this in any channel are ignored when estimating stain colours.
# The usual 0.15 fails on faded slides (eosin pink vanishes, or the fit fails outright). In a test with
# physically simulated restaining (Beer-Lambert), BETA 0.03 + fitting on tissue pixels only brought every
# variant within 8 grey levels of the reference result, vs up to 53 (or a failed fit) with 0.15.
IO, ALPHA, BETA = 240.0, 1.0, 0.03
# Target colours = the standard Macenko reference (Macenko et al. 2009; the defaults in torchstain
# and StainTools), so every slide is mapped to classic H&E colours.
# Not used: the tile auto-picked in test run 01 (kaggle/results_01/reference_patch.png) is pale
# (maxC 0.79/0.65 vs 1.97/1.03) and made everything lavender. Its fit, for the record:
#   HE [[0.65754, 0.40933], [0.69118, 0.85353], [0.29985, 0.32239]], maxC [0.79426, 0.65119]
HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])
FIT_TILES = 200      # tiles sampled per slide to estimate that slide's stain colours
FIT_PIXELS = 2_000_000


def macenko_fit(pixels):
    """Standard Macenko (same maths as torchstain) on an (P, 3) uint8 pixel array.

    Returns (HE (3, 2), maxC (2,)) or None if there is too little stained tissue.
    """
    od = -np.log((pixels.astype(np.float64) + 1) / IO)
    od_hat = od[~np.any(od < BETA, axis=1)]
    if len(od_hat) < 1000:
        return None
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    t_hat = od_hat @ eigvecs[:, 1:3]
    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])
    min_phi, max_phi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v_min = eigvecs[:, 1:3] @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v_max = eigvecs[:, 1:3] @ np.array([np.cos(max_phi), np.sin(max_phi)])
    he = np.stack([v_min, v_max], 1) if v_min[0] > v_max[0] else np.stack([v_max, v_min], 1)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.array([np.percentile(conc[0], 99), np.percentile(conc[1], 99)])
    if not np.all(np.isfinite(he)) or not np.all(max_c > 1e-6):
        return None
    return he, max_c


def macenko_apply(tiles, he, max_c):
    """tiles: (B, H, W, 3) uint8 tensor on GPU -> (B, H, W, 3) float 0..255, normalized to HE_REF."""
    he_ref = torch.as_tensor(HE_REF, dtype=torch.float32, device=tiles.device)
    he_pinv = torch.linalg.pinv(torch.as_tensor(he, dtype=torch.float32, device=tiles.device))   # (2, 3)
    scale = torch.as_tensor(MAXC_REF / max_c, dtype=torch.float32, device=tiles.device)
    rgb = tiles.float()
    od = -torch.log((rgb + 1) / IO)                                          # (B, H, W, 3)
    conc = od @ he_pinv.T * scale                                            # (B, H, W, 2)
    out = (IO * torch.exp(-conc @ he_ref.T)).clamp_(0, 255)
    # leave background (white glass, black padding) untouched: rescaling its faint colour only adds grain
    gray = rgb @ torch.as_tensor(GRAY_W, device=tiles.device)
    background = ((gray > 220) | (gray == 0)).unsqueeze(-1)
    return torch.where(background, rgb, out)


# ----------------------------------------------------------------------------- slides and tiles
def load_slide(path, shrink):
    import pyvips
    img = pyvips.Image.new_from_file(path, access='sequential')
    if img.bands > 3:
        img = img.extract_band(0, n=3)
    return img.shrink(shrink, shrink).numpy()


def tissue_tiles(arr, patch=PATCH, step=WSI_STEP, min_tissue=MIN_TISSUE):
    """(y, x) of tiles where >= min_tissue of pixels are tissue (not black padding, not white)."""
    coords = []
    h, w = arr.shape[:2]
    for y in range(0, h - patch + 1, step):
        for x in range(0, w - patch + 1, step):
            g = arr[y:y + patch, x:x + patch].astype(np.float32) @ GRAY_W
            if np.mean((g > 0) & (g <= 220)) >= min_tissue:
                coords.append((y, x))
    return np.array(coords, dtype=np.int32).reshape(-1, 2)


def prepare_slide(data_dir, row):
    """Runs in a background thread: load + tile one slide."""
    t0 = time.time()
    shrink, step = (TMA_SHRINK, TMA_STEP) if row.is_tma else (WSI_SHRINK, WSI_STEP)
    arr = load_slide(os.path.join(data_dir, 'train_images', f'{row.image_id}.png'), shrink)
    t_load = time.time() - t0
    coords = tissue_tiles(arr, PATCH, step, MIN_TISSUE)
    return row, arr, coords, t_load, time.time() - t0 - t_load


def fit_slide(arr, coords, rng):
    pick = rng.choice(len(coords), size=min(FIT_TILES, len(coords)), replace=False)
    pixels = np.concatenate([arr[y:y + PATCH, x:x + PATCH].reshape(-1, 3) for y, x in coords[pick]])
    gray = pixels.astype(np.float32) @ GRAY_W
    pixels = pixels[(gray > 0) & (gray <= 220)]          # tissue only: same rule as tissue_tiles()
    if len(pixels) > FIT_PIXELS:
        pixels = pixels[rng.choice(len(pixels), FIT_PIXELS, replace=False)]
    return macenko_fit(pixels)


# ----------------------------------------------------------------------------- encoders
class Encoder:
    def __init__(self, name, model, mean, std, dim, device, forward):
        self.name, self.model, self.dim, self.device, self._forward = name, model, dim, device, forward
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, x01):
        """x01: (B, 3, 224, 224) float in [0, 1] -> (B, dim) float16 (still on device)."""
        x = (x01.to(self.device, non_blocking=True) - self.mean) / self.std
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            return self._forward(x).half()


def _conv_stem():
    from torch import nn

    class ConvStem(nn.Module):
        """CTransPath patch embedding (from 1aurent/swin_tiny_patch4_window7_224.CTransPath)."""

        def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
            super().__init__()
            stem, d_in, d_out = [], 3, embed_dim // 8
            for _ in range(2):
                stem += [nn.Conv2d(d_in, d_out, 3, 2, 1, bias=False), nn.BatchNorm2d(d_out), nn.ReLU(inplace=True)]
                d_in, d_out = d_out, d_out * 2
            stem.append(nn.Conv2d(d_in, embed_dim, 1))
            self.proj = nn.Sequential(*stem)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
            pair = lambda v: tuple(v) if isinstance(v, (list, tuple)) else (v, v)   # timm may pass a list
            self.img_size, self.patch_size = pair(img_size), pair(patch_size)
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]

        def forward(self, x):
            return self.norm(self.proj(x).permute(0, 2, 3, 1))

    return ConvStem


def load_encoder(name, device):
    device = torch.device(device)
    if name in ('vits16_lunit', 'ctranspath', 'gigapath'):
        import timm
        if name == 'vits16_lunit':
            m = timm.create_model('hf-hub:1aurent/vit_small_patch16_224.lunit_dino', pretrained=True, num_classes=0)
        elif name == 'ctranspath':
            m = timm.create_model('hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath',
                                  embed_layer=_conv_stem(), pretrained=True, num_classes=0)
        else:
            m = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
        cfg = timm.data.resolve_model_data_config(m)
        mean, std, fwd = cfg['mean'], cfg['std'], m
    elif name == 'phikon_v2':
        from transformers import AutoImageProcessor, AutoModel
        m = AutoModel.from_pretrained('owkin/phikon-v2')
        proc = AutoImageProcessor.from_pretrained('owkin/phikon-v2')
        mean, std = proc.image_mean, proc.image_std
        fwd = lambda x: m(pixel_values=x).last_hidden_state[:, 0]
    elif name == 'plip':
        from transformers import CLIPImageProcessor, CLIPModel
        m = CLIPModel.from_pretrained('vinid/plip')
        proc = CLIPImageProcessor.from_pretrained('vinid/plip')
        mean, std = proc.image_mean, proc.image_std

        def fwd(x):
            out = m.get_image_features(pixel_values=x)
            return out if torch.is_tensor(out) else out.pooler_output
    else:
        raise ValueError(f'unknown encoder {name!r}')
    m = m.eval().to(device)
    with torch.inference_mode():
        dim = fwd(torch.zeros(1, 3, MODEL_INPUT, MODEL_INPUT, device=device)).shape[-1]
    return Encoder(name, m, mean, std, dim, device, fwd)


# ----------------------------------------------------------------------------- run
def _save_atomic(path, array):
    tmp = path + '.tmp.npy'
    np.save(tmp, array)
    os.replace(tmp, path)


def _done(out_dir, sid, encoder_names):
    return os.path.exists(os.path.join(out_dir, 'coords', f'{sid}.npy')) and all(
        os.path.exists(os.path.join(out_dir, 'features', e, f'{sid}.npy')) for e in encoder_names)


def copy_previous(prev_roots, out_dir, encoder_names):
    """Copy finished slides from earlier runs (attached as Kaggle inputs) into out_dir."""
    rows = []
    for root in prev_roots:
        man = os.path.join(root, 'manifest.csv')
        if not os.path.exists(man):
            continue
        for r in pd.read_csv(man, dtype={'image_id': str}).to_dict('records'):
            sid = r['image_id']
            if _done(out_dir, sid, encoder_names) or not _done(root, sid, encoder_names):
                continue
            for sub in ['coords'] + [os.path.join('features', e) for e in encoder_names]:
                os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
                shutil.copy(os.path.join(root, sub, f'{sid}.npy'), os.path.join(out_dir, sub, f'{sid}.npy'))
            rows.append(r)
    if rows:
        _append_manifest(out_dir, rows)
    return len(rows)


def _append_manifest(out_dir, rows):
    path = os.path.join(out_dir, 'manifest.csv')
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def run(slides, data_dir, out_dir, encoder_names, devices, time_budget_h=11.0, n_threads=2, batch=128,
        prev_roots=(), seed=0, log=print):
    """Extract features for every slide in `slides` (DataFrame: image_id, is_tma, pixels)."""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    for e in encoder_names:
        os.makedirs(os.path.join(out_dir, 'features', e), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'coords'), exist_ok=True)

    n_copied = copy_previous(prev_roots, out_dir, encoder_names)
    if n_copied:
        log(f'copied {n_copied} finished slides from previous runs')

    encoders = [load_encoder(e, devices[e]) for e in encoder_names]
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(dict(patch=PATCH, wsi_shrink=WSI_SHRINK, tma_shrink=TMA_SHRINK, wsi_step=WSI_STEP,
                       tma_step=TMA_STEP, min_tissue=MIN_TISSUE, model_input=MODEL_INPUT,
                       he_ref=np.asarray(HE_REF).tolist(), maxc_ref=np.asarray(MAXC_REF).tolist(),
                       encoders={e.name: dict(dim=e.dim, mean=e.mean.flatten().tolist(),
                                              std=e.std.flatten().tolist()) for e in encoders}), f, indent=2)
    log('encoders: ' + ', '.join(f'{e.name} ({e.dim}-d on {e.device})' for e in encoders))

    todo = slides[[not _done(out_dir, s, encoder_names) for s in slides.image_id]]
    log(f'{len(slides) - len(todo)} slides already done, {len(todo)} to go')
    rng = np.random.default_rng(seed)
    main_dev = encoders[0].device
    budget_s = time_budget_h * 3600

    rows_iter = iter(todo.itertuples())
    pending, n_done, tiles_done = [], 0, 0
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        def submit_next():
            if time.time() - t_start > budget_s:
                return
            row = next(rows_iter, None)
            if row is not None:
                pending.append(pool.submit(prepare_slide, data_dir, row))

        for _ in range(n_threads):
            submit_next()

        while pending:
            row, arr, coords, t_load, t_tile = pending.pop(0).result()
            submit_next()                                  # start loading the next slide right away
            t0 = time.time()
            sid = row.image_id
            params = fit_slide(arr, coords, rng) if len(coords) else None
            feats = {e.name: [] for e in encoders}
            for i in range(0, len(coords), batch):
                tiles = np.stack([arr[y:y + PATCH, x:x + PATCH] for y, x in coords[i:i + batch]])
                x = torch.from_numpy(tiles).to(main_dev)
                x = macenko_apply(x, *params) if params is not None else x.float()
                x = F.interpolate(x.permute(0, 3, 1, 2) / 255.0, size=MODEL_INPUT, mode='bilinear',
                                  antialias=True, align_corners=False)
                outs = [e(x) for e in encoders]            # launches on all GPUs before syncing
                for e, o in zip(encoders, outs):
                    feats[e.name].append(o.float().cpu().numpy().astype(np.float16))
            for e in encoders:
                f = np.concatenate(feats[e.name]) if feats[e.name] else np.zeros((0, e.dim), np.float16)
                _save_atomic(os.path.join(out_dir, 'features', e.name, f'{sid}.npy'), f)
            _save_atomic(os.path.join(out_dir, 'coords', f'{sid}.npy'), coords)

            t_gpu = time.time() - t0
            _append_manifest(out_dir, [dict(image_id=sid, label=row.label, is_tma=int(row.is_tma), n_tiles=len(coords),
                                            macenko='ok' if params is not None else 'skipped',
                                            t_load_s=round(t_load, 1), t_tile_s=round(t_tile, 1), t_gpu_s=round(t_gpu, 1),
                                            he=json.dumps(np.round(params[0], 5).tolist()) if params else '',
                                            maxc=json.dumps(np.round(params[1], 5).tolist()) if params else '')])
            n_done += 1
            tiles_done += len(coords)
            el = time.time() - t_start
            left = len(todo) - n_done
            log(f'[{n_done}/{len(todo)}] {sid} {"TMA" if row.is_tma else "WSI"} {len(coords):6d} tiles | '
                f'load {t_load:5.1f}s gpu {t_gpu:5.1f}s | elapsed {el / 3600:.2f} h, ~{el / n_done * left / 3600:.1f} h left')
            del arr, feats

    finished = sum(_done(out_dir, s, encoder_names) for s in slides.image_id)
    log(f'run finished: {finished}/{len(slides)} slides complete in {out_dir}'
        + ('' if finished == len(slides) else ' -> run again with this output attached to continue'))
    return finished
