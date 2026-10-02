"""Distributional-regression networks with Gaussian output.

Variants (selected by flags):
  residual:  mu = ens_mean + f_mu(x, e)      else  mu = f_mu(x, e)         (Rasp & Lerch style)
  spread:
    'free_abs'      sigma = |f_s(x, e)|                (Rasp & Lerch: linear output, sqrt(sigma^2))
    'free_softplus' sigma = softplus(f_s(x, e))
    'constrained'   sigma = sqrt(softplus(f_c)^2 + softplus(f_d)^2 * ens_sd^2)
                    i.e. an EMOS-type variance model whose intercept c(x,e) and spread
                    coefficient d(x,e) are positive and predicted by the network, so that
                    sigma is always >= c and scales with the ensemble spread.
  embedding: station embedding (index 0 reserved for unknown stations) concatenated to x.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

SOFTPLUS_INV_1 = math.log(math.e - 1.0)  # softplus^{-1}(1)
EPS = 1e-6


class DistNet(nn.Module):
    def __init__(self, n_features, n_emb, emb_dim=2, hidden=512, residual=True,
                 spread="constrained", embedding=True):
        super().__init__()
        assert spread in ("free_abs", "free_softplus", "constrained")
        self.residual, self.spread, self.use_emb = residual, spread, embedding
        self.emb = nn.Embedding(n_emb, emb_dim) if embedding else None
        d_in = n_features + (emb_dim if embedding else 0)
        self.hidden = nn.Linear(d_in, hidden)
        n_out = 3 if spread == "constrained" else 2
        self.out = nn.Linear(hidden, n_out)
        with torch.no_grad():
            if residual:  # start exactly at the raw ensemble mean
                self.out.weight[0].zero_()
                self.out.bias[0].zero_()
            if spread == "constrained":  # start at sigma = sqrt(1 + ens_sd^2)
                self.out.weight[1:].mul_(0.1)
                self.out.bias[1:].fill_(SOFTPLUS_INV_1)

    def forward(self, x, emb_idx, ens_mean, ens_sd):
        h = x if self.emb is None else torch.cat([x, self.emb(emb_idx)], dim=1)
        o = self.out(F.relu(self.hidden(h)))
        mu = o[:, 0] + (ens_mean if self.residual else 0.0)
        if self.spread == "free_abs":
            sigma = o[:, 1].abs() + EPS
        elif self.spread == "free_softplus":
            sigma = F.softplus(o[:, 1]) + EPS
        else:
            c = F.softplus(o[:, 1])
            d = F.softplus(o[:, 2])
            sigma = torch.sqrt(c * c + d * d * ens_sd * ens_sd + EPS)
        return mu, sigma
