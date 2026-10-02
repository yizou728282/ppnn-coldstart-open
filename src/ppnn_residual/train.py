"""Training loop: early stopping on the validation year, then refit on train+val."""
from __future__ import annotations

import time

import numpy as np
import torch

from .data import Split, UNK
from .metrics import crps_gaussian_torch
from .models import DistNet


def _tensors(s: Split):
    return (torch.from_numpy(s.x), torch.from_numpy(s.emb_idx), torch.from_numpy(s.ens_mean),
            torch.from_numpy(s.ens_sd), torch.from_numpy(s.y))


@torch.no_grad()
def predict(model, s: Split, batch=65536):
    model.eval()
    x, e, m, sd, _ = _tensors(s)
    mus, sigs = [], []
    for i in range(0, len(x), batch):
        mu, sig = model(x[i:i + batch], e[i:i + batch], m[i:i + batch], sd[i:i + batch])
        mus.append(mu)
        sigs.append(sig)
    return torch.cat(mus).numpy().astype(np.float64), torch.cat(sigs).numpy().astype(np.float64)


def _val_crps(model, s):
    mu, sig = predict(model, s)
    return float(crps_gaussian_torch(torch.from_numpy(mu), torch.from_numpy(sig),
                                     torch.from_numpy(s.y.astype(np.float64))).mean())


def train_network(fit: Split, n_emb: int, arch: dict, hp: dict, seed: int, epochs: int,
                  val: Split | None = None, patience: int | None = None, log=print):
    """Train for up to `epochs`. If val is given, early-stop with `patience` and return the best
    epoch (1-based) with the model restored to the best state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = DistNet(fit.x.shape[1], n_emb, emb_dim=hp["emb_dim"], hidden=hp["hidden"], **arch)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    x, e, m, sd, y = _tensors(fit)
    n, bs, p_drop = len(x), hp["batch_size"], hp["station_dropout"]
    history, best, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            eb = e[b]
            if model.use_emb and p_drop > 0:  # teach the reserved UNK embedding a "generic station"
                mask = torch.rand(len(b), generator=gen) < p_drop
                eb = torch.where(mask, torch.full_like(eb, UNK), eb)
            mu, sig = model(x[b], eb, m[b], sd[b])
            loss = crps_gaussian_torch(mu, sig, y[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        rec = {"epoch": ep, "train_crps": tot / n, "sec": round(time.time() - t0, 1)}
        if val is not None:
            rec["val_crps"] = _val_crps(model, val)
            if rec["val_crps"] < best - 1e-6:
                best, best_epoch, bad = rec["val_crps"], ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        log(rec)
        if val is not None and bad >= patience:
            break
    if val is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch
