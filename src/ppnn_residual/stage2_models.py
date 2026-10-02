"""Stage 2 networks: different ways of obtaining a station representation for new stations.

emb_mode:
  'unk'    free embedding table; index 0 = UNK (trained by station dropout)
  'none'   no embedding
  'attr'   embedding = MLP(station attributes), used for every station
  'hybrid' embedding = MLP(attributes) + free offset delta_s; delta_s is zeroed with prob
           `delta_dropout` per training sample; new stations use delta = 0
Head: Rasp & Lerch style (mu, |sigma|) or residual mean + constrained spread.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

SOFTPLUS_INV_1 = math.log(math.e - 1.0)


class StationNet(nn.Module):
    def __init__(self, n_features, n_stations_plus_unk, n_attr, emb_mode="unk", emb_dim=2, hidden=512,
                 attr_hidden=32, residual=False, constrained_spread=False, delta_dropout=0.5):
        super().__init__()
        self.emb_mode, self.residual, self.constrained = emb_mode, residual, constrained_spread
        self.delta_dropout = delta_dropout
        self.emb_dim = emb_dim
        if emb_mode in ("unk", "hybrid"):
            self.table = nn.Embedding(n_stations_plus_unk, emb_dim)
            if emb_mode == "hybrid":
                nn.init.zeros_(self.table.weight)
        if emb_mode in ("attr", "hybrid"):
            self.attr_mlp = nn.Sequential(nn.Linear(n_attr, attr_hidden), nn.ReLU(), nn.Linear(attr_hidden, emb_dim))
        d_in = n_features + (0 if emb_mode == "none" else emb_dim)
        self.hidden = nn.Linear(d_in, hidden)
        self.out = nn.Linear(hidden, 3 if constrained_spread else 2)
        with torch.no_grad():
            if residual:
                self.out.weight[0].zero_(); self.out.bias[0].zero_()
            if constrained_spread:
                self.out.weight[1:].mul_(0.1); self.out.bias[1:].fill_(SOFTPLUS_INV_1)

    def station_vector(self, idx, attr, training=False, generator=None):
        if self.emb_mode == "none":
            return None
        if self.emb_mode == "unk":
            return self.table(idx)
        a = self.attr_mlp(attr)
        if self.emb_mode == "attr":
            return a
        delta = self.table(idx)  # UNK row stays 0 (never receives gradient because it is masked below)
        keep = (idx != 0).float()
        if training and self.delta_dropout > 0:
            # masks are drawn on the (CPU) generator's device and moved: identical stream on CPU and GPU
            r = torch.rand(len(idx), generator=generator).to(idx.device)
            keep = keep * (r >= self.delta_dropout).float()
        return a + delta * keep[:, None]

    def forward(self, x, idx, attr, ens_mean, ens_sd, emb_vec=None, training=False, generator=None):
        if self.emb_mode == "none":
            h = x
        else:
            e = emb_vec if emb_vec is not None else self.station_vector(idx, attr, training, generator)
            h = torch.cat([x, e], 1)
        o = self.out(F.relu(self.hidden(h)))
        mu = o[:, 0] + (ens_mean if self.residual else 0.0)
        if self.constrained:
            c, d = F.softplus(o[:, 1]), F.softplus(o[:, 2])
            sigma = torch.sqrt(c * c + d * d * ens_sd * ens_sd + 1e-6)
        else:
            sigma = o[:, 1].abs() + 1e-6
        return mu, sigma


def haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_idw(query_lat, query_lon, query_alt, ref_lat, ref_lon, ref_alt, ref_vec, k=5, alt_km_per_m=0.1):
    """Inverse-distance-weighted mean of ref_vec over the k nearest reference stations,
    distance = sqrt(d_km^2 + (alt_km_per_m * d_alt_m)^2)."""
    import numpy as np
    d = haversine_km(query_lat, query_lon, ref_lat, ref_lon)
    d = np.sqrt(d ** 2 + (alt_km_per_m * (np.asarray(query_alt)[:, None] - np.asarray(ref_alt)[None, :])) ** 2)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    dd = np.take_along_axis(d, nn_idx, 1)
    w = 1.0 / np.maximum(dd, 1e-3)
    w /= w.sum(1, keepdims=True)
    return (w[:, :, None] * ref_vec[nn_idx]).sum(1)
