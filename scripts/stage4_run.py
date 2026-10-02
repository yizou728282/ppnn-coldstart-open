"""Stage 4 runner: Lakatos-style baselines (protocols/stage4_baselines.json) and the pilot (protocols/stage4_pilot.json).

  python scripts/stage4_run.py --dataset german --fold-type random --methods drn_lak gnn_geo --threads 4
Resumable: each finished (method, fold, seed) writes preds + log; existing logs are skipped.
Predictions: held-out test rows of fold k, in Stage-2/3 test_meta order (like Stage-3 files).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.metrics import crps_gaussian_torch  # noqa: E402
from ppnn_residual.stage2_models import haversine_km  # noqa: E402
from ppnn_residual.stage4_data import load_cube, results_dir  # noqa: E402

PB = json.loads((ROOT / "protocols/stage4_baselines.json").read_text())
PP_PATH = ROOT / "protocols/stage4_pilot.json"


def savez_atomic(path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz"); np.savez(tmp, **arrs); os.replace(tmp, path)


def json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)


# ------------------------------------------------------------------------------------------ fold setup
class Fold:
    """Normalised tensors for one fold; held-out stations never enter any statistic."""

    def __init__(self, c, folds_df, k):
        self.c, self.k = c, k
        fs = folds_df.fold.reindex(c.stations).to_numpy()
        self.tr = fs != k; self.held = fs == k
        self.tr_idx = np.nonzero(self.tr)[0]; self.held_idx = np.nonzero(self.held)[0]
        self.avail = np.isfinite(c.y) & np.isfinite(c.X[:, :, c.i_mean])
        fit_t = c.split == 0
        rows = self.avail[fit_t][:, self.tr]
        Xf = c.X[fit_t][:, self.tr][rows]
        m, s = Xf.mean(0), Xf.std(0); s[s == 0] = 1
        del Xf
        Xn = (c.X - m) / s
        Xn[~np.isfinite(Xn)] = 0.0
        self.Xn = torch.from_numpy(Xn.astype(np.float32)); del Xn
        A = c.st[["lat", "lon", "alt", "orog"]].to_numpy(np.float64)
        am, asd = A[self.tr].mean(0), A[self.tr].std(0)
        self.A = torch.from_numpy(((A - am) / asd).astype(np.float32))
        tf = np.c_[np.sin(2 * np.pi * c.doy / 365.25), c.doy, c.month].astype(np.float64)
        tm, tsd = tf[fit_t].mean(0), tf[fit_t].std(0)
        tf = (tf - tm) / tsd
        if len(c.leads) > 1:
            tf = np.c_[tf, (c.lead[:, None] == np.array(c.leads)[None]).astype(np.float64)]
        self.TF = torch.from_numpy(tf.astype(np.float32))
        self.y = torch.from_numpy(np.nan_to_num(c.y).astype(np.float32))
        self.av = torch.from_numpy(self.avail)
        self.n_feat = self.Xn.shape[2] + self.A.shape[1] + self.TF.shape[1]
        # test rows of held-out stations (test_meta order)
        self.test_rows = np.nonzero(self.held[c.test_n])[0]
        # distances
        self.D = haversine_km(c.st.lat.values, c.st.lon.values, c.st.lat.values, c.st.lon.values)

    def rows(self, split, stations_mask):
        t_ok = self.c.split == split
        m = self.avail & t_ok[:, None] & stations_mask[None, :]
        t, n = np.nonzero(m)
        return torch.from_numpy(t), torch.from_numpy(n)

    def feats(self, t, n):
        return torch.cat([self.Xn[t, n], self.A[n], self.TF[t]], 1)

    def node_feats(self, tb, nodes):
        B, Nn = len(tb), len(nodes)
        x = self.Xn[tb][:, nodes]
        return torch.cat([x, self.A[nodes][None].expand(B, Nn, -1), self.TF[tb][:, None].expand(B, Nn, -1)], 2)


# ------------------------------------------------------------------------------------------ drn_lak
class DRN(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.h = nn.Linear(n_in, hidden); self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        o = self.out(self.drop(F.relu(self.h(x))))
        return o[:, 0], o[:, 1].abs() + 1e-6


def train_drn(fd, seed, hp):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = DRN(fd.n_feat, hp["hidden"], hp["dropout"]); opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    tf_, nf_ = fd.rows(0, fd.tr); tv, nv = fd.rows(1, fd.tr)
    Xv, yv = fd.feats(tv, nv), fd.y[tv, nv]
    best, state, best_ep, bad, hist = np.inf, None, 0, 0, []
    for ep in range(1, hp["max_epochs"] + 1):
        model.train(); t0 = time.time(); perm = torch.randperm(len(tf_), generator=gen); tot = 0.0
        for i in range(0, len(perm), hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]; t, n = tf_[b], nf_[b]
            mu, sg = model(fd.feats(t, n)); loss = crps_gaussian_torch(mu, sg, fd.y[t, n]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            mu, sg = model(Xv); v = float(crps_gaussian_torch(mu.double(), sg.double(), yv.double()).mean())
        hist.append({"epoch": ep, "train_crps": tot / len(perm), "val_crps": v, "sec": round(time.time() - t0, 2)})
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0; state = {a: b.clone() for a, b in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(state); model.eval()
    c = fd.c; r = fd.test_rows
    t, n = torch.from_numpy(c.test_t[r]), torch.from_numpy(c.test_n[r])
    with torch.no_grad():
        mu, sg = model(fd.feats(t, n))
    return mu.numpy(), sg.numpy(), {"best_epoch": best_ep, "best_val_crps": best, "history": hist}


# ------------------------------------------------------------------------------------------ gnn_geo
class SAGE(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.root = nn.Linear(n_in, hidden); self.nb = nn.Linear(n_in, hidden, bias=False)
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(hidden, 2)

    def forward(self, x, av, adj):
        """x [B,N,F], av [B,N] (float 0/1), adj [N,N] binary without self loops."""
        s = torch.einsum("ij,bjf->bif", adj, x * av[..., None])
        cnt = av @ adj.T  # [B,N] number of available neighbours
        agg = s / cnt.clamp(min=1.0)[..., None]
        h = self.drop(F.relu(self.root(x) + self.nb(agg)))
        o = self.out(h)
        return o[..., 0], o[..., 1].abs() + 1e-6


def train_gnn(fd, seed, hp):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    c = fd.c
    adj_all = torch.from_numpy(((fd.D <= hp["radius_km"]) & ~np.eye(len(c.stations), dtype=bool)).astype(np.float32))
    tri = torch.from_numpy(fd.tr_idx); adj_tr = adj_all[tri][:, tri]
    model = SAGE(fd.n_feat, hp["hidden"], hp["dropout"]); opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    fit_t = torch.from_numpy(np.nonzero(c.split == 0)[0]); val_t = torch.from_numpy(np.nonzero(c.split == 1)[0])

    def run(tb, nodes, adj, train):
        x = fd.node_feats(tb, nodes); av = fd.av[tb][:, nodes].float()
        mu, sg = model(x, av, adj)
        cr = crps_gaussian_torch(mu, sg, fd.y[tb][:, nodes])
        return mu, sg, cr, av

    best, state, best_ep, bad, hist = np.inf, None, 0, 0, []
    B = hp["batch_slices"]
    for ep in range(1, hp["max_epochs"] + 1):
        model.train(); t0 = time.time(); perm = fit_t[torch.randperm(len(fit_t), generator=gen)]; tot = 0.0; nn_ = 0.0
        for i in range(0, len(perm), B):
            _, _, cr, av = run(perm[i:i + B], tri, adj_tr, True)
            loss = (cr * av).sum() / av.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float((cr * av).sum().detach()); nn_ += float(av.sum())
        model.eval(); vs, vn = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(val_t), B):
                _, _, cr, av = run(val_t[i:i + B], tri, adj_tr, False)
                vs += float((cr.double() * av).sum()); vn += float(av.sum())
        v = vs / vn
        hist.append({"epoch": ep, "train_crps": tot / nn_, "val_crps": v, "sec": round(time.time() - t0, 2)})
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0; state = {a: b.clone() for a, b in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(state); model.eval()
    # test: all stations are nodes; read off held-out rows
    test_t = np.nonzero(c.split == 2)[0]; alln = torch.arange(len(c.stations))
    MU = np.full((len(c.split), len(c.stations)), np.nan, np.float32); SG = MU.copy()
    with torch.no_grad():
        for i in range(0, len(test_t), B):
            tb = torch.from_numpy(test_t[i:i + B]); mu, sg, _, _ = run(tb, alln, adj_all, False)
            MU[tb.numpy()] = mu.numpy(); SG[tb.numpy()] = sg.numpy()
    r = fd.test_rows
    n_iso = int((adj_all[torch.from_numpy(fd.held_idx)].sum(1) == 0).sum())
    return MU[c.test_t[r], c.test_n[r]], SG[c.test_t[r], c.test_n[r]], {
        "best_epoch": best_ep, "best_val_crps": best, "history": hist, "n_heldout_isolated_at_test": n_iso,
        "n_edges_train_graph": int(adj_tr.sum()), "n_edges_test_graph": int(adj_all.sum())}


# ------------------------------------------------------------------------------------------ main
def method_table(requested=()):
    M = {"drn_lak": (train_drn, PB["methods"]["drn_lak"]),
         "gnn_geo": (train_gnn, {**PB["methods"]["gnn_geo"], "radius_km": 50.0, "batch_slices": 256})}
    if PP_PATH.exists() and any(m not in M for m in requested):
        from ppnn_residual.stage4_pilot import train_samos  # noqa: E402
        PP = json.loads(PP_PATH.read_text())
        for name, hp in PP["methods"].items():
            M[name] = (train_samos, hp)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["german", "euppbench"], required=True)
    ap.add_argument("--fold-type", choices=["random", "spatial"], required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="SMOKE TEST ONLY")
    ap.add_argument("--subsample-slices", type=int, default=None, help="SMOKE TEST ONLY")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else ROOT / "results/stage4" / a.dataset
    c = load_cube(a.dataset)
    if a.subsample_slices:
        rng = np.random.default_rng(0)
        keep = np.zeros(len(c.split), bool)
        for sp in (0, 1):
            ids = np.nonzero(c.split == sp)[0]; keep[rng.choice(ids, min(a.subsample_slices, len(ids)), replace=False)] = True
        c.split = np.where(keep | (c.split == 2), c.split, -1)
    folds_df = pd.read_csv(results_dir(a.dataset) / f"folds_{a.fold_type}.csv", index_col=0)
    K = int(folds_df.fold.max()) + 1
    M = method_table(a.methods)
    for k in (a.folds if a.folds is not None else range(K)):
        todo = [(m, s) for m in a.methods for s in a.seeds if not (out / "logs" / a.fold_type / f"{m}_f{k}_s{s}.json").exists()]
        if not todo:
            continue
        fd = Fold(c, folds_df, k)
        for m, s in todo:
            fn, hp = M[m]; hp = dict(hp)
            if a.max_epochs:
                hp["max_epochs"] = a.max_epochs
            t0 = time.time()
            mu, sg, info = fn(fd, s, hp)
            savez_atomic(out / "preds" / a.fold_type / f"{m}_f{k}_s{s}.npz", mu=mu.astype(np.float32), sigma=sg.astype(np.float32))
            info.update({"method": m, "fold": k, "seed": s, "dataset": a.dataset, "sec": round(time.time() - t0, 1),
                         "threads": a.threads, "torch": torch.__version__, "n_heldout_test_rows": int(len(mu)),
                         "smoke": bool(a.max_epochs or a.subsample_slices)})
            json_atomic(out / "logs" / a.fold_type / f"{m}_f{k}_s{s}.json", info)
            print(a.dataset, a.fold_type, m, "fold", k, "seed", s, "best_ep", info["best_epoch"],
                  "val", round(info["best_val_crps"], 4), "sec", info["sec"], flush=True)
        del fd


if __name__ == "__main__":
    main()
