"""Stage 4 pilot (protocols/stage4_pilot.json): SAMOS-style climatological standardisation + attention over the
K nearest TRAINING stations, trained with random station masking (pseudo-new stations).

Pieces
  1. climatology: ridge regression of the climatological mean and log-variance of obs and of the t2m ensemble mean on
     [1, lat, lon, alt, alt-orog] x [1, sin/cos(2 pi doy/365.25), sin/cos(4 pi doy/365.25)] (+ lead one-hot x [1, sin1, cos1]),
     fitted on training stations / fit period only -> m_y, s_y, m_f, s_f for every (slice, station)
  2. standardised inputs z_f = (f - m_f)/s_f, z_s = s/s_f; target in z-space; output mu = m_y + s_y * (z_f + g),
     sigma = s_y * sqrt(softplus(c)^2 + softplus(d)^2 z_s^2)
  3. single-head attention over the K nearest available training stations (self excluded) + a learned null slot.
     neighbour token = [z_f_j, z_s_j, z_f_j - z_f_i, d_ij/100 km, (alt_j-alt_i)/500 m, (aom_j-aom_i)/500 m, e_j]
     (e_j = learned embedding of training station j)
  4. training-time masking: per batch a random fraction U(0, p_drop_max) of training stations is removed from the
     neighbour pool, and per sample with prob p_disc all training stations within r ~ U(0, r_max) km of the target
     are removed (mimics random and spatially blocked new stations).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .metrics import crps_gaussian_torch

SP_INV_1 = math.log(math.e - 1.0)
LOGCHI2_MEAN = -1.2703628454614782  # E[log chi^2_1]


# ------------------------------------------------------------------------------------------ climatology
class Climatology:
    def __init__(self, c, tr, ridge):
        st = c.st
        a = np.c_[st.lat, st.lon, st.alt, st.alt - st.orog].astype(np.float64)
        self.am, self.asd = a[tr].mean(0), a[tr].std(0)
        self.Ast = np.c_[np.ones(len(a)), (a - self.am) / self.asd]                     # [N, 5]
        w = 2 * np.pi * c.doy / 365.25
        self.H = np.c_[np.ones(len(w)), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]  # [T, 5]
        if len(c.leads) > 1:
            L = (c.lead[:, None] == np.array(c.leads)[None, 1:]).astype(np.float64)
            self.G = np.einsum("tl,tj->tlj", L, self.H[:, :3]).reshape(len(w), -1)       # [T, 12]
        else:
            self.G = np.zeros((len(w), 0))
        self.ridge = ridge

    def design(self, t, n):
        return np.c_[np.einsum("ri,rj->rij", self.Ast[n], self.H[t]).reshape(len(t), -1), self.G[t]]

    def fit_field(self, t, n, v):
        Z = self.design(t, n)
        beta = np.linalg.solve(Z.T @ Z + self.ridge * np.eye(Z.shape[1]), Z.T @ v)
        return beta

    def field(self, beta):
        nb = self.Ast.shape[1] * self.H.shape[1]
        B = beta[:nb].reshape(self.Ast.shape[1], self.H.shape[1])
        return self.H @ B.T @ self.Ast.T + (self.G @ beta[nb:])[:, None]              # [T, N]

    def fit_mean_sd(self, t, n, v):
        bm = self.fit_field(t, n, v)
        m = self.field(bm)
        r2 = (v - m[t, n]) ** 2
        bs = self.fit_field(t, n, np.log(r2 + 1e-6))
        s = np.exp(0.5 * (self.field(bs) - LOGCHI2_MEAN))
        return m.astype(np.float32), np.maximum(s, 0.05).astype(np.float32)


# ------------------------------------------------------------------------------------------ model
class SamosAttn(nn.Module):
    def __init__(self, n_own, n_tr, hp):
        super().__init__()
        H, d, e = hp["hidden"], hp["att_dim"], hp["emb_dim"]
        self.use_nb = hp["use_neighbors"]
        self.own = nn.Sequential(nn.Linear(n_own, H), nn.ReLU())
        if self.use_nb:
            self.emb = nn.Embedding(n_tr, e)
            nn.init.normal_(self.emb.weight, std=0.1)
            self.tok = nn.Sequential(nn.Linear(6 + e, H), nn.ReLU())
            self.q, self.k, self.v = nn.Linear(H, d), nn.Linear(H, d), nn.Linear(H, d)
            self.null_k = nn.Parameter(torch.zeros(d)); self.null_v = nn.Parameter(torch.zeros(d))
            self.d = d
        self.head = nn.Sequential(nn.Linear(H + (d if self.use_nb else 0), H), nn.ReLU())
        self.out = nn.Linear(H, 3)
        with torch.no_grad():
            self.out.weight.mul_(0.1); self.out.bias.zero_(); self.out.bias[1:].fill_(SP_INV_1)

    def forward(self, u, tok=None, tok_ok=None, nb_idx=None):
        h = self.own(u)
        if self.use_nb:
            z = self.tok(torch.cat([tok, self.emb(nb_idx.clamp(min=0))], -1))            # [B,K,H]
            q = self.q(h)[:, None]                                                         # [B,1,d]
            k = torch.cat([self.null_k.expand(len(h), 1, -1), self.k(z)], 1)              # [B,K+1,d]
            v = torch.cat([self.null_v.expand(len(h), 1, -1), self.v(z)], 1)
            sc = (q * k).sum(-1) / math.sqrt(self.d)
            ok = torch.cat([torch.ones(len(h), 1, dtype=torch.bool), tok_ok], 1)
            w = torch.softmax(sc.masked_fill(~ok, -1e9), 1)
            h = torch.cat([h, (w[..., None] * v).sum(1)], 1)
        return self.out(self.head(h))


# ------------------------------------------------------------------------------------------ training
class PilotData:
    def __init__(self, fd, hp):
        c = fd.c; self.fd = fd; self.hp = hp
        tr = fd.tr; av = fd.avail
        clim = Climatology(c, tr, hp["clim_ridge"])
        fit = av & (c.split == 0)[:, None] & tr[None]
        t, n = np.nonzero(fit)
        rng = np.random.default_rng(0)
        if len(t) > hp["clim_max_rows"]:
            sel = rng.choice(len(t), hp["clim_max_rows"], replace=False); t, n = t[sel], n[sel]
        f = c.X[:, :, c.i_mean].astype(np.float64); s = c.ens_sd().astype(np.float64)
        self.m_y, self.s_y = clim.fit_mean_sd(t, n, c.y[t, n].astype(np.float64))
        self.m_f, self.s_f = clim.fit_mean_sd(t, n, f[t, n])
        zf = np.nan_to_num((f - self.m_f) / self.s_f).astype(np.float32)
        zs = np.nan_to_num(s / self.s_f).astype(np.float32)
        self.zf, self.zs = torch.from_numpy(zf), torch.from_numpy(zs)
        self.my, self.sy = torch.from_numpy(self.m_y), torch.from_numpy(self.s_y)
        A5 = clim.Ast[:, 1:].astype(np.float32)
        self.A5 = torch.from_numpy(A5)
        self.n_own = fd.Xn.shape[2] + 2 + A5.shape[1] + fd.TF.shape[1]
        # neighbour candidates: all training stations sorted by distance (self distance -> inf)
        D = fd.D.copy(); np.fill_diagonal(D, np.inf)
        Dtr = D[:, fd.tr_idx]                                                               # [N, Ntr]
        order = np.argsort(Dtr, 1)
        self.cand = torch.from_numpy(order)                                                 # position in tr_idx
        self.cand_d = torch.from_numpy(np.take_along_axis(Dtr, order, 1).astype(np.float32))
        self.tr_idx = torch.from_numpy(fd.tr_idx)
        alt = c.st.alt.to_numpy(np.float32); aom = (c.st.alt - c.st.orog).to_numpy(np.float32)
        self.alt, self.aom = torch.from_numpy(alt), torch.from_numpy(aom)
        self.clim_info = {"clim_rows": int(len(t))}

    def own(self, t, n):
        fd = self.fd
        return torch.cat([fd.Xn[t, n], self.zf[t, n, None], self.zs[t, n, None], self.A5[n], fd.TF[t]], 1)

    def neighbours(self, t, n, gen=None, mask=False):
        hp = self.hp; K = hp["K"]
        cand = self.cand[n]; cd = self.cand_d[n]                                            # [B, Ntr]
        st_j = self.tr_idx[cand]                                                            # station index
        ok = self.fd.av[t[:, None], st_j] & torch.isfinite(cd)
        if mask:
            p = float(torch.rand(1, generator=gen)) * hp["p_drop_max"]
            keep_st = torch.rand(len(self.tr_idx), generator=gen) >= p
            ok &= keep_st[cand]
            use_disc = torch.rand(len(t), generator=gen) < hp["p_disc"]
            r = torch.rand(len(t), generator=gen) * hp["r_max_km"] * use_disc
            ok &= cd > r[:, None]
        rank = torch.arange(cand.shape[1], dtype=torch.float32)[None]
        score = torch.where(ok, -rank, torch.full_like(rank, -1e9).expand_as(cd))
        top = score.topk(K, 1).indices                                                      # first K available
        sel_ok = torch.gather(ok, 1, top)
        j_pos = torch.gather(cand, 1, top); j = self.tr_idx[j_pos]; d = torch.gather(cd, 1, top)
        tt = t[:, None].expand_as(j)
        zfi = self.zf[t, n][:, None]
        tok = torch.stack([self.zf[tt, j], self.zs[tt, j], self.zf[tt, j] - zfi, torch.nan_to_num(d / 100.0, posinf=0.0),
                           (self.alt[j] - self.alt[n][:, None]) / 500.0, (self.aom[j] - self.aom[n][:, None]) / 500.0], -1)
        tok = tok * sel_ok[..., None]
        return tok, sel_ok, torch.where(sel_ok, j_pos, torch.full_like(j_pos, -1))

    def predict(self, model, t, n, gen=None, mask=False):
        u = self.own(t, n)
        if model.use_nb:
            tok, ok, jp = self.neighbours(t, n, gen, mask)
            o = model(u, tok, ok, jp)
        else:
            o = model(u)
        zmu = self.zf[t, n] + o[:, 0]
        zsg = torch.sqrt(F.softplus(o[:, 1]) ** 2 + F.softplus(o[:, 2]) ** 2 * self.zs[t, n] ** 2 + 1e-6)
        return self.my[t, n] + self.sy[t, n] * zmu, self.sy[t, n] * zsg


def train_samos(fd, seed, hp):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    pdx = PilotData(fd, hp)
    model = SamosAttn(pdx.n_own, len(fd.tr_idx), hp)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    tf_, nf_ = fd.rows(0, fd.tr); tv, nv = fd.rows(1, fd.tr)
    best, state, best_ep, bad, hist = np.inf, None, 0, 0, []
    bs = hp["batch_size"]
    for ep in range(1, hp["max_epochs"] + 1):
        model.train(); t0 = time.time(); perm = torch.randperm(len(tf_), generator=gen); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; t, n = tf_[b], nf_[b]
            mu, sg = pdx.predict(model, t, n, gen, mask=hp["masking"])
            loss = crps_gaussian_torch(mu, sg, fd.y[t, n]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        model.eval(); vs = 0.0
        with torch.no_grad():
            for i in range(0, len(tv), 65536):
                t, n = tv[i:i + 65536], nv[i:i + 65536]
                mu, sg = pdx.predict(model, t, n)
                vs += float(crps_gaussian_torch(mu.double(), sg.double(), fd.y[t, n].double()).sum())
        v = vs / len(tv)
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
    mus, sgs = [], []
    with torch.no_grad():
        for i in range(0, len(t), 65536):
            mu, sg = pdx.predict(model, t[i:i + 65536], n[i:i + 65536]); mus.append(mu); sgs.append(sg)
    # climatology-only diagnostic on held-out test rows (standardised-anomaly persistence: mu = m_y + s_y z_f)
    zf = pdx.zf[t, n]; clim_mu = (pdx.my[t, n] + pdx.sy[t, n] * zf).numpy()
    return torch.cat(mus).numpy(), torch.cat(sgs).numpy(), {
        "best_epoch": best_ep, "best_val_crps": best, "history": hist, **pdx.clim_info,
        "heldout_test_mae_samos_anchor": float(np.mean(np.abs(clim_mu - fd.c.y[c.test_t[r], c.test_n[r]])))}
