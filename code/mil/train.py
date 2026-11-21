"""Train one MIL model on one (or all) CV folds, then predict the locked test set.

Model selection uses the fold's VALIDATION set only. The test set is predicted once,
with the selected checkpoint, and never influences training (fixes old issue M1).

Example:
    python -m mil.train --features features/uni --splits splits_v2 --model abmil --fold all --out runs/uni/abmil/seed0

Outputs per fold in <out>/fold<k>/:
    config.json  history.csv  best.pt  val_preds.csv  test_preds.csv
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CLASSES, BagDataset, read_split
from .metrics import compute_metrics
from .models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bag_loader(ds, shuffle, seed):
    g = torch.Generator().manual_seed(seed)
    # batch_size=1 bag; collate keeps (x, y, id) without stacking
    return DataLoader(ds, batch_size=1, shuffle=shuffle, generator=g, num_workers=0,
                      collate_fn=lambda b: b[0])


def loss_fn(model, out, y, class_weights):
    main = F.cross_entropy(out.get('loss_logits', out['logits']).unsqueeze(0), y.view(1), weight=class_weights)
    if out['aux_loss'] is None:
        return main
    w = getattr(model, 'aux_weight', 0.0)
    return (1 - w) * main + w * out['aux_loss']


@torch.no_grad()
def predict(model, loader, device, df, class_weights=None):
    """Returns (preds DataFrame, mean loss)."""
    model.eval()
    rows, total = [], 0.0
    for x, y, sid in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, label=y)
        total += loss_fn(model, out, y, class_weights).item()
        probs = out['logits'].float().softmax(0).cpu().numpy()
        rows.append(dict(image_id=sid, label=int(y), pred=int(probs.argmax()),
                         **{f'prob_{c}': float(p) for c, p in zip(CLASSES, probs)}))
    preds = pd.DataFrame(rows).merge(df[['file_name', 'is_tma']].rename(columns={'file_name': 'image_id'}),
                                     on='image_id', how='left')
    return preds, total / max(1, len(loader))


def train_fold(args, fold, device, feature_cache=None):
    """feature_cache: optional {slide_id: tensor} from data.preload_features, shared across runs."""
    set_seed(args.seed + fold)
    out_dir = os.path.join(args.out, f'fold{fold}')
    os.makedirs(out_dir, exist_ok=True)

    train_df = read_split(os.path.join(args.splits, 'train', f'{fold}.csv'))
    val_df = read_split(os.path.join(args.splits, 'val', f'{fold}.csv'))
    test_df = read_split(os.path.join(args.splits, 'test.csv'))

    cache = feature_cache if feature_cache is not None else args.cache
    train_ds = BagDataset(train_df, args.features, max_patches=args.max_patches, cache=cache, seed=args.seed + fold)
    val_ds = BagDataset(val_df, args.features, cache=cache)
    test_ds = BagDataset(test_df, args.features, cache=feature_cache if feature_cache is not None else False)
    in_dim = train_ds.feature_dim()

    model_kwargs = {k: getattr(args, k, None) for k in ('hidden', 'dropout') if getattr(args, k, None) is not None}
    model = build_model(args.model, in_dim, len(CLASSES), **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    class_weights = None
    if args.class_weights:
        counts = np.bincount(train_df.label, minlength=len(CLASSES))
        class_weights = torch.tensor(len(train_df) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'fold': fold, 'in_dim': in_dim, 'device': str(device)}, f, indent=2)

    train_loader = bag_loader(train_ds, shuffle=True, seed=args.seed + fold)
    val_loader = bag_loader(val_ds, shuffle=False, seed=0)

    best, best_epoch, bad_epochs, history = None, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        train_loss = 0.0
        for i, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model, model(x, label=y), y, class_weights) / args.accum
            loss.backward()
            train_loss += loss.item() * args.accum
            if i % args.accum == 0 or i == len(train_loader):
                opt.step()
                opt.zero_grad()

        val_preds, val_loss = predict(model, val_loader, device, val_df, class_weights)
        m = compute_metrics(val_preds)
        history.append(dict(epoch=epoch, train_loss=train_loss / len(train_loader), val_loss=val_loss,
                            val_bacc=m['bacc'], val_f1=m['macro_f1'], val_auc=m['macro_auc'], sec=time.time() - t0))
        score = -val_loss if args.select == 'loss' else m['bacc']
        improved = best is None or score > best
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'model_name': args.model, 'model_kwargs': model_kwargs,
                        'in_dim': in_dim, 'classes': CLASSES}, os.path.join(out_dir, 'best.pt'))
        elif epoch >= args.min_epochs:
            bad_epochs += 1

        h = history[-1]
        print(f'fold {fold} ep {epoch:3d} | train {h["train_loss"]:.3f} | val loss {val_loss:.3f} '
              f'bacc {m["bacc"]:.3f} f1 {m["macro_f1"]:.3f} | {h["sec"]:.0f}s{" *" if improved else ""}', flush=True)
        if bad_epochs >= args.patience:
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)

    # reload the selected checkpoint, then predict val (for OOF metrics) and the locked test set
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pt'), map_location=device)['model'])
    val_preds, _ = predict(model, val_loader, device, val_df)
    test_preds, _ = predict(model, bag_loader(test_ds, shuffle=False, seed=0), device, test_df)
    val_preds.to_csv(os.path.join(out_dir, 'val_preds.csv'), index=False)
    test_preds.to_csv(os.path.join(out_dir, 'test_preds.csv'), index=False)
    print(f'fold {fold} done: best epoch {best_epoch}, val bacc {compute_metrics(val_preds)["bacc"]:.3f}')


def main():
    p = argparse.ArgumentParser(description='Train a MIL aggregator on precomputed patch features')
    p.add_argument('--features', required=True, help='folder with <slide_id>.npy feature files')
    p.add_argument('--splits', required=True, help='folder from make_splits.py (train/, val/, test.csv)')
    p.add_argument('--model', required=True, help='maxpool, meanpool, abmil, dsmil, clam_sb, transmil')
    p.add_argument('--fold', default='all', help="fold number (1-5) or 'all'")
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--min_epochs', type=int, default=10, help='early stopping cannot trigger before this')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--accum', type=int, default=1, help='gradient accumulation over this many bags')
    p.add_argument('--max_patches', type=int, default=8000, help='random patch cap per training bag (0 = no cap)')
    p.add_argument('--select', choices=['loss', 'bacc'], default='loss', help='validation criterion for the best epoch')
    p.add_argument('--class_weights', action='store_true', help='inverse-frequency class weights in the loss')
    p.add_argument('--hidden', type=int, default=None, help='model hidden size (default: model default)')
    p.add_argument('--dropout', type=float, default=None, help='model dropout (default: model default)')
    p.add_argument('--cache', action='store_true', help='keep features in RAM')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    args.max_patches = args.max_patches or None

    folds = range(1, 6) if args.fold == 'all' else [int(args.fold)]
    for k in folds:
        train_fold(args, k, torch.device(args.device))


if __name__ == '__main__':
    main()
