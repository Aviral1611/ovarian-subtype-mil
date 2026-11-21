"""Slide-level MIL aggregators.

Every model takes one bag of patch features ``x`` with shape (N, D) and returns a dict:
    logits    (C,)      class scores for the slide
    attn      (N,)      per-patch importance (for heatmaps), or None
    aux_loss  scalar    extra training loss (DSMIL instance loss, CLAM clustering loss), or None

Optional extra key:
    loss_logits (C,)    logits to use for the main training loss instead of ``logits`` (DSMIL)

Models with an auxiliary loss define ``aux_weight``: total = (1 - w) * main + w * aux.
Models that need the slide label for their auxiliary loss (DSMIL, CLAM) accept ``label=``;
the rest ignore it. Build models with ``build_model(name, in_dim, num_classes)``.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class PoolMIL(nn.Module):
    """Per-patch MLP, then max or mean pooling, then a linear classifier.

    Layer names match the old capstone checkpoints (encoder.0, encoder.3, classifier),
    so those weights can still be loaded.
    """

    def __init__(self, in_dim, num_classes, pool='max', dropout=0.25):
        super().__init__()
        assert pool in ('max', 'mean')
        self.pool = pool
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, label=None):
        h = self.encoder(x)                                   # (N, 128)
        if self.pool == 'max':
            pooled, idx = h.max(dim=0)
            # importance = how many of the 128 features each patch "won"
            attn = torch.bincount(idx, minlength=len(x)).float() / h.shape[1]
        else:
            pooled, attn = h.mean(dim=0), None
        return dict(logits=self.classifier(pooled), attn=attn, aux_loss=None)


class GatedAttention(nn.Module):
    """Gated attention scores (Ilse et al. 2018). Returns raw scores (N, n_out)."""

    def __init__(self, dim, hidden, n_out=1, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.c = nn.Linear(hidden, n_out)

    def forward(self, h):
        return self.c(self.a(h) * self.b(h))


class ABMIL(nn.Module):
    """Gated attention MIL (Ilse et al. 2018).

    Fixes old bug B1: attention is normalized over PATCHES (dim 0 of an (N, 1) tensor),
    not over a size-1 batch axis.
    """

    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(hidden, hidden // 2, 1, dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, label=None):
        h = self.fc(x)                                        # (N, H)
        a = torch.softmax(self.attention(h).squeeze(1), 0)    # (N,)
        m = a @ h                                             # (H,)
        return dict(logits=self.classifier(m), attn=a, aux_loss=None)


class DSMIL(nn.Module):
    """Dual-stream MIL (Li, Li & Eliceiri, CVPR 2021). Cleaned port of the capstone version.

    Stream 1 scores each patch; the top patch per class is the "critical instance".
    Stream 2 attends to all patches by similarity to the critical instances.
    Prediction = mean of both streams; aux_loss = max-instance cross-entropy.
    """

    aux_weight = 0.5

    def __init__(self, in_dim, num_classes, hidden=128, dropout=0.25):
        super().__init__()
        self.i_classifier = nn.Linear(in_dim, num_classes)
        self.q = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.v = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, in_dim), nn.ReLU())
        self.b_classifier = nn.Conv1d(num_classes, num_classes, kernel_size=in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, label=None):
        inst_logits = self.i_classifier(x)                    # (N, C)
        q, v = self.q(x), self.v(x)                           # (N, h), (N, D)
        crit = inst_logits.argmax(dim=0)                      # (C,) top patch per class
        attn = torch.softmax(q @ q[crit].T / math.sqrt(q.shape[1]), dim=0)  # (N, C)
        bag_feat = self.dropout(attn.T @ v)                   # (C, D)
        bag_logits = self.b_classifier(bag_feat.unsqueeze(0)).squeeze()      # (C,)
        max_inst = inst_logits.max(dim=0).values              # (C,)

        aux = None
        if label is not None:
            aux = F.cross_entropy(max_inst.unsqueeze(0), label.view(1))
        # Prediction = mean of both streams' probabilities (as log-probs, so softmax(logits) == probs).
        # Training loss = 0.5 * CE(bag_logits) + 0.5 * CE(max_inst), as in the original DSMIL.
        probs = 0.5 * (bag_logits.softmax(0) + max_inst.softmax(0))
        pred = probs.argmax()
        return dict(logits=probs.clamp_min(1e-8).log(), attn=attn[:, pred], aux_loss=aux,
                    loss_logits=bag_logits)


class CLAM_SB(nn.Module):
    """CLAM single-branch (Lu et al., Nat. Biomed. Eng. 2021).

    Gated attention pooling plus an instance-level clustering loss: the k most-attended
    patches are pushed to "positive" for the true class and "negative" for other classes;
    the k least-attended are "negative" for the true class.
    """

    aux_weight = 0.3   # official CLAM: 0.7 * bag loss + 0.3 * instance loss

    def __init__(self, in_dim, num_classes, hidden=512, attn_hidden=256, dropout=0.25, k_sample=8):
        super().__init__()
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(hidden, attn_hidden, 1, dropout)
        self.classifier = nn.Linear(hidden, num_classes)
        self.instance_classifiers = nn.ModuleList(nn.Linear(hidden, 2) for _ in range(num_classes))

    def _inst_loss(self, a, h, c, in_class):
        k = min(self.k_sample, len(h) // 2 if in_class else len(h))
        if k < 1:
            return h.new_zeros(())
        top = h[torch.topk(a, k).indices]
        if in_class:
            bottom = h[torch.topk(-a, k).indices]
            feats = torch.cat([top, bottom])
            targets = torch.cat([torch.ones(k), torch.zeros(k)]).long().to(h.device)
        else:
            feats, targets = top, torch.zeros(k, dtype=torch.long, device=h.device)
        return F.cross_entropy(self.instance_classifiers[c](feats), targets)

    def forward(self, x, label=None):
        h = self.fc(x)                                        # (N, H)
        a = torch.softmax(self.attention(h).squeeze(1), 0)    # (N,)
        logits = self.classifier(a @ h)

        aux = None
        if label is not None:
            y = int(label)
            losses = [self._inst_loss(a, h, c, in_class=(c == y)) for c in range(self.num_classes)]
            aux = torch.stack(losses).mean()
        return dict(logits=logits, attn=a, aux_loss=aux)


class PPEG(nn.Module):
    """Pyramid position encoding generator from TransMIL."""

    def __init__(self, dim):
        super().__init__()
        self.c7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.c5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.c3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, side):
        cls, feat = x[:, :1], x[:, 1:]
        grid = feat.transpose(1, 2).reshape(x.shape[0], -1, side, side)
        grid = grid + self.c7(grid) + self.c5(grid) + self.c3(grid)
        return torch.cat([cls, grid.flatten(2).transpose(1, 2)], dim=1)


class TransLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        from nystrom_attention import NystromAttention
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim=dim, dim_head=dim // 8, heads=8, num_landmarks=dim // 2,
                                     pinv_iterations=6, residual=True, dropout=dropout)

    def forward(self, x):
        return x + self.attn(self.norm(x))

    @torch.no_grad()
    def cls_attention(self, x):
        """Exact softmax attention from token 0 (class token) to every token, mean over heads.

        O(N) memory, unlike asking NystromAttention for its full N×N matrix.
        """
        q, k, _ = self.attn.to_qkv(self.norm(x)).chunk(3, dim=-1)       # (1, L, heads*dh)
        heads = self.attn.heads
        q = q[0, :1].view(1, heads, -1).transpose(0, 1)                  # (heads, 1, dh)
        k = k[0].view(x.shape[1], heads, -1).transpose(0, 1)             # (heads, L, dh)
        scores = (q * self.attn.scale) @ k.transpose(1, 2)               # (heads, 1, L)
        return scores.softmax(-1).mean(0)[0]                             # (L,)


class TransMIL(nn.Module):
    """TransMIL (Shao et al., NeurIPS 2021): Nystrom self-attention between patches + PPEG.

    attn = attention from the class token to each patch in the last layer (mean over heads).
    """

    def __init__(self, in_dim, num_classes, hidden=512, dropout=0.25):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.layer1 = TransLayer(hidden)
        self.ppeg = PPEG(hidden)
        self.layer2 = TransLayer(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, label=None):
        n = len(x)
        h = self.fc(x).unsqueeze(0)                           # (1, N, H)
        side = math.ceil(math.sqrt(n))
        pad = side * side - n
        if pad:                                               # fill the square grid by repeating patches
            h = torch.cat([h, h[:, :pad]], dim=1)
        h = torch.cat([self.cls_token, h], dim=1)             # (1, 1 + side², H)
        h = self.layer1(h)
        h = self.ppeg(h, side)
        attn = None if self.training else self.layer2.cls_attention(h)[1:n + 1]   # (N,) real patches only
        h = self.layer2(h)
        logits = self.classifier(self.norm(h)[0, 0])
        return dict(logits=logits, attn=attn, aux_loss=None)


MODELS = {
    'maxpool': lambda d, c, **kw: PoolMIL(d, c, pool='max', **kw),
    'meanpool': lambda d, c, **kw: PoolMIL(d, c, pool='mean', **kw),
    'abmil': ABMIL,
    'dsmil': DSMIL,
    'clam_sb': CLAM_SB,
    'transmil': TransMIL,
}


def build_model(name, in_dim, num_classes, **kwargs):
    """kwargs: model hyperparameters, e.g. hidden=256, dropout=0.25 (None values are ignored)."""
    if name not in MODELS:
        raise ValueError(f'unknown model {name!r}; choose from {sorted(MODELS)}')
    return MODELS[name](in_dim, num_classes, **{k: v for k, v in kwargs.items() if v is not None})
