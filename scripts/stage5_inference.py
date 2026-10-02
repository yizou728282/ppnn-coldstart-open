"""Stage 5 evaluation and inference (protocols/stage5_supplementary.json, sections D, E, F, C4).
Everything is computed from stored prediction files; nothing is trained.

  python scripts/stage5_inference.py --tag existing --seed-cap-spatial 3     # existing 3-seed spatial predictions
  python scripts/stage5_inference.py --tag final                             # everything available
Outputs: results/stage5/eval/<tag>/
"""
import argparse
import itertools
import json
import math
import time
import zlib

import numpy as np
import pandas as pd
from scipy import special, stats

from stage5_common import OLD_DIR, PARTITIONS, P5, R5, ROOT, folds_df, json_atomic

BASE = ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]
OLD_NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_knn", "nn_hybrid_res"]
S4 = ["drn_lak", "gnn_geo", "samos_mlp", "samos_attn"]
STAT = ["emos_nn1", "emos_idw", "emos_reg", "emos_regidw", "samos_lin"]
M1A = ["gnn_geo_fcavail", "samos_attn_fcavail"]
ALL = BASE + STAT + OLD_NETS + S4
D5 = P5["D_inference"]
H_DAYS = D5["dm"]["hln"]["h_days"]
_ISQPI = 1 / math.sqrt(math.pi)


def crps(mu, sg, y):
    z = (y - mu) / sg
    return sg * (z * (2 * special.ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) - _ISQPI)


# ------------------------------------------------------------------------------------------ loading
def dirs_for(ds, design, m):
    ftype = design if design in ("random", "spatial") else "spatial"
    if m.endswith("_fcavail"):
        return [R5 / ds / "m1a" / "preds" / ftype]
    if design.startswith("part_"):
        return [R5 / ds / design / "preds" / "spatial"]
    d5 = R5 / ds / design / "preds" / ftype
    if m in BASE or m in OLD_NETS:
        return [OLD_DIR[ds] / "preds" / ftype, d5]
    if m in S4:
        return [ROOT / "results/stage4" / ds / "preds" / ftype, d5]
    return [d5]


def load(ds, design, m, row_fold, K, seed_cap=None):
    """-> {seed or None: (mu, sigma)} full-length float32 arrays, only seeds with all K folds."""
    base, _, cap = m.partition("@")
    cap = int(cap) if cap else seed_cap
    dirs = dirs_for(ds, design, base)
    n = len(row_fold)

    def read(fn_of_k):
        mu = np.full(n, np.nan, np.float32); sg = mu.copy()
        for k in range(K):
            f = next((d / fn_of_k(k) for d in dirs if (d / fn_of_k(k)).exists()), None)
            if f is None:
                return None
            z = np.load(f); r = row_fold == k; a, b = z["mu"], z["sigma"]
            if len(a) == n:
                a, b = a[r], b[r]
            if len(a) != r.sum():
                raise ValueError(f"{f}: {len(a)} rows, expected {r.sum()}")
            mu[r], sg[r] = a, b
        return mu, sg
    if base in BASE or base in STAT:
        r = read(lambda k: f"{base}_f{k}.npz")
        return {} if r is None else {None: r}
    out = {}
    for s in range(10 if cap is None else cap):
        r = read(lambda k: f"{base}_f{k}_s{s}.npz")
        if r is not None:
            out[s] = r
    return out


# ------------------------------------------------------------------------------------------ HAC / DM
def autocov(d, j):
    d = d - d.mean(); n = len(d)
    return float(d[j:] @ d[:n - j]) / n


def nw_var(d, lag):
    v = autocov(d, 0)
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * autocov(d, k)
    return v / len(d)


def bw_nw94(d):
    n = len(d); nT = int(4 * (n / 100) ** (2 / 9))
    g = [autocov(d, j) for j in range(nT + 1)]
    s0 = g[0] + 2 * sum(g[1:]); s1 = 2 * sum(j * g[j] for j in range(1, nT + 1))
    if s0 <= 0:
        return 0
    return int(min(n - 1, math.floor(1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * n ** (1 / 3))))


def bw_andrews(d):
    x = d - d.mean(); rho = float(x[1:] @ x[:-1] / (x[:-1] @ x[:-1]))
    rho = min(rho, 0.97)
    alpha = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(min(len(d) - 1, math.floor(1.1447 * (alpha * len(d)) ** (1 / 3))))


def dm_all(d, h):
    n = len(d); res = {"n_days": n, "mean": float(d.mean())}
    for tag, lag in (("nw94", bw_nw94(d)), ("andrews", bw_andrews(d)), ("lag5", 5), ("lag20", 20), ("lag40", 40)):
        v = nw_var(d, lag); stat = d.mean() / math.sqrt(v) if v > 0 else float("nan")
        res[f"lag_{tag}"] = lag; res[f"dm_{tag}"] = stat
        res[f"p_{tag}_normal"] = float(2 * stats.norm.sf(abs(stat)))
        hln = stat * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        res[f"p_{tag}_hln"] = float(2 * stats.t.sf(abs(hln), n - 1))
        if tag in ("nw94", "andrews"):
            b = (lag + 1) / n
            cv = 1.96 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3
            res[f"fixedb_cv_{tag}"] = cv; res[f"fixedb_reject05_{tag}"] = bool(abs(stat) > cv)
    rng = np.random.default_rng(12345)
    for L in D5["dm"]["block_bootstrap_days"]:
        nb = int(math.ceil(n / L)); reps = D5["dm"]["block_bootstrap_reps"]
        starts = rng.integers(0, n - L + 1, size=(reps, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None]).reshape(reps, -1)[:, :n]
        mm = d[idx].mean(1)
        res[f"mbb{L}_lo"], res[f"mbb{L}_hi"] = (float(q) for q in np.quantile(mm, [0.025, 0.975]))
    return res


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    passed = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool); rej[o[:k]] = True
    return rej


# ------------------------------------------------------------------------------------------ one design
class Design:
    def __init__(self, ds, design, methods, seed_cap, reps, W_st, W_fold_seed):
        t0 = time.time()
        self.ds, self.design = ds, design
        meta = np.load(OLD_DIR[ds] / "test_meta.npz", allow_pickle=True)
        self.y = meta["y"].astype(np.float64); st = meta["station"].astype(np.int64)
        self.day = meta["init"] if "init" in meta else meta["date"]
        fdf = folds_df(ds, design); self.K = int(fdf.fold.max()) + 1
        self.row_fold = fdf.fold.reindex(st).to_numpy()
        self.stations = np.sort(np.unique(st)); self.st_idx = np.searchsorted(self.stations, st)
        self.st_fold = fdf.fold.reindex(self.stations).to_numpy()
        self.cnt = np.bincount(self.st_idx, minlength=len(self.stations)).astype(float)
        cap = seed_cap if design == "spatial" else None
        self.P = {}
        for m in methods:
            r = load(ds, design, m, self.row_fold, self.K, cap)
            if r:
                self.P[m] = r
        self.methods = list(self.P)
        if not self.methods:
            return
        n = len(self.y)
        # full-ensemble CRPS rows, per-seed pooled CRPS
        self.cr, self.single, self.ens = {}, {}, {}
        for m, r in self.P.items():
            mu = np.mean([v[0] for v in r.values()], 0, dtype=np.float64); sg = np.mean([v[1] for v in r.values()], 0, dtype=np.float64)
            self.ens[m] = (mu, sg); self.cr[m] = crps(mu, sg, self.y)
            self.single[m] = {s: float(crps(v[0].astype(np.float64), v[1].astype(np.float64), self.y).mean()) for s, v in r.items() if s is not None}
        # nested bootstrap: seed-resampled station sums S[m] [reps, Ns]
        rng = np.random.default_rng(W_fold_seed + zlib.crc32(design.encode()) % 1000)
        self.S = {}
        for m, r in self.P.items():
            keys = list(r)
            if keys == [None]:
                self.S[m] = np.broadcast_to(np.bincount(self.st_idx, self.cr[m], len(self.stations)), (reps, len(self.stations)))
                continue
            MU = np.stack([r[s][0] for s in keys]).astype(np.float64); SG = np.stack([r[s][1] for s in keys]).astype(np.float64)
            out = np.empty((reps, len(self.stations)))
            for i in range(reps):
                c = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)) / len(keys)
                out[i] = np.bincount(self.st_idx, crps(c @ MU, c @ SG, self.y), len(self.stations))
            self.S[m] = out
        self.W = W_st
        self._summary = self._make_summary()
        del self.P, self.ens
        print(f"  {ds}/{design}: {len(self.methods)} methods, {n} rows, nested bootstrap {time.time() - t0:.0f}s", flush=True)

    def boot_stat(self, m, W=None):
        W = self.W if W is None else W
        return (W * self.S[m]).sum(1) / (W @ self.cnt)

    def fold_boot(self, a, b, reps, seed):
        rng = np.random.default_rng(seed)
        WF = rng.multinomial(self.K, np.ones(self.K) / self.K, size=reps).astype(float)
        agg = lambda X: np.stack([X[:, self.st_fold == k].sum(1) for k in range(self.K)], 1)
        cF = np.array([self.cnt[self.st_fold == k].sum() for k in range(self.K)])
        den = WF @ cF
        return ((WF * agg(self.S[a])).sum(1) - (WF * agg(self.S[b])).sum(1)) / den

    def compare(self, a, b, reps):
        d = self.cr[a] - self.cr[b]
        res = {"a": a, "b": b, "crps_a": self.cr[a].mean(), "crps_b": self.cr[b].mean(), "diff": d.mean(),
               "rel_diff_pct": 100 * (self.cr[a].mean() / self.cr[b].mean() - 1),
               "n_seeds_a": len(self.single[a]) or 0, "n_seeds_b": len(self.single[b]) or 0}
        # nested station bootstrap
        bs = self.boot_stat(a) - self.boot_stat(b)
        res.update(st_nested_lo=np.quantile(bs, .025), st_nested_hi=np.quantile(bs, .975), p_a_better_nested=float((bs < 0).mean()))
        # un-nested station bootstrap (fixed ensembles), 10000 reps
        ds_ = np.bincount(self.st_idx, d, len(self.stations))
        rng = np.random.default_rng(777)
        W = rng.multinomial(len(self.stations), np.ones(len(self.stations)) / len(self.stations), size=D5["also_unnested_station_bootstrap_reps"]).astype(float)
        bu = (W @ ds_) / (W @ self.cnt)
        res.update(st_fixed_lo=np.quantile(bu, .025), st_fixed_hi=np.quantile(bu, .975))
        fb = self.fold_boot(a, b, reps, 4242)
        res.update(fold_nested_lo=np.quantile(fb, .025), fold_nested_hi=np.quantile(fb, .975), p_a_better_fold=float((fb < 0).mean()))
        # rank tests
        fd = np.array([d[self.row_fold == k].mean() for k in range(self.K)])
        sd = ds_ / self.cnt
        for tag, v in (("fold", fd), ("station", sd)):
            nz = v[v != 0]
            res[f"{tag}_n_a_better"] = int((v < 0).sum()); res[f"{tag}_n"] = int(len(v))
            res[f"{tag}_sign_p"] = float(stats.binomtest(int((nz < 0).sum()), len(nz), 0.5).pvalue) if len(nz) else float("nan")
            res[f"{tag}_wilcoxon_p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) > 1 else float("nan")
        res["fold_diffs"] = json.dumps([round(float(x), 4) for x in fd])
        # DM on daily series
        dd = pd.Series(d).groupby(self.day).mean().sort_index().to_numpy()
        res.update(dm_all(dd, H_DAYS[self.ds]))
        # seed-only
        sa, sb = list(self.single[a].values()) or [self.cr[a].mean()], list(self.single[b].values()) or [self.cr[b].mean()]
        res["p_single_seed_a_better"] = float(np.mean([x < z for x in sa for z in sb]))
        return res

    def summary(self):
        return self._summary

    def _make_summary(self):
        rows = []
        for m in self.methods:
            mu, sg = self.ens[m]; e = self.y - mu; pit = special.ndtr(e / sg)
            r = {"method": m, "n_seeds": len(self.single[m]), "crps": self.cr[m].mean(), "mae": np.abs(e).mean(),
                 "rmse": np.sqrt((e ** 2).mean()), "spread_skill": float(np.sqrt((sg ** 2).mean()) / np.sqrt((e ** 2).mean())),
                 "cov80": float(((pit >= .1) & (pit <= .9)).mean()), "bias": float(-e.mean())}
            if len(self.single[m]) > 1:
                v = np.array(list(self.single[m].values()))
                r.update(single_seed_mean=v.mean(), single_seed_sd=v.std(ddof=1), single_seed_min=v.min(), single_seed_max=v.max(),
                         per_seed=json.dumps({int(s): round(x, 5) for s, x in self.single[m].items()}))
            rows.append(r)
        return pd.DataFrame(rows).sort_values("crps")

    def rank_probs(self, methods):
        ms = [m for m in methods if m in self.S]
        B = np.stack([self.boot_stat(m) for m in ms], 1)
        R = B.argsort(1).argsort(1) + 1
        P = pd.DataFrame({"method": ms, "p_rank1": (R == 1).mean(0), "mean_rank": R.mean(0),
                          "boot_crps_lo": np.quantile(B, .025, 0), "boot_crps_hi": np.quantile(B, .975, 0)})
        pw = pd.DataFrame([[float((B[:, i] < B[:, j]).mean()) if i != j else np.nan for j in range(len(ms))] for i in range(len(ms))],
                          index=ms, columns=ms)
        return P.sort_values("mean_rank"), pw

    def per_fold(self):
        return pd.DataFrame({m: [self.cr[m][self.row_fold == k].mean() for k in range(self.K)] for m in self.methods}).assign(
            n_stations=[int((self.st_fold == k).sum()) for k in range(self.K)])


def pair_list(methods):
    E = "emos_bst_gl"; pr = []
    for a in D5["pairs_vs_emos_bst_gl"]:
        pr.append((a, E))
    for a in D5["pairs_vs_nn_hybrid_res"]:
        pr.append((a, "nn_hybrid_res"))
    pr += [tuple(p) for p in D5["extra_pairs"]]
    return [(a, b) for a, b in pr if a in methods and b in methods]


def did(Dsp, Drd, F, E="emos_bst_gl"):
    """Per-station DiD (F-E)_sp - (F-E)_rd with the SAME station weights (coupled) and independent seed resampling."""
    assert np.array_equal(Dsp.stations, Drd.stations) and np.array_equal(Dsp.cnt, Drd.cnt)
    b = (Dsp.boot_stat(F) - Dsp.boot_stat(E)) - (Drd.boot_stat(F) - Drd.boot_stat(E))
    pt = (Dsp.cr[F].mean() - Dsp.cr[E].mean()) - (Drd.cr[F].mean() - Drd.cr[E].mean())
    return {"F": F, "did": float(pt), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975)),
            "p_did_gt0": float((b > 0).mean()), "spatial_F_minus_E": float(Dsp.cr[F].mean() - Dsp.cr[E].mean()),
            "random_F_minus_E": float(Drd.cr[F].mean() - Drd.cr[E].mean())}


def bst_sensitivity(ds_obj, ftype, out):
    d = R5 / "euppbench" / ftype / "bstsens"
    fs = sorted(d.glob("emos_bst_gl_path_f*.npz")) if d.exists() else []
    if len(fs) != ds_obj.K:
        return None
    y = ds_obj.y; rows_all, preds = [], {}
    for k in range(ds_obj.K):
        z = np.load(d / f"emos_bst_gl_path_f{k}.npz"); rows = z["rows"]
        for key in z.files:
            if key.startswith("mu_m"):
                tag = key[4:]
                preds.setdefault(tag, [np.full(len(y), np.nan), np.full(len(y), np.nan)])
                preds[tag][0][rows] = z[key]; preds[tag][1][rows] = z["sigma_m" + tag]
    res = []
    for tag, (mu, sg) in preds.items():
        e = y - mu; pit = special.ndtr(e / sg)
        res.append({"m": tag, "crps": crps(mu, sg, y).mean(), "spread_skill": np.sqrt((sg ** 2).mean()) / np.sqrt((e ** 2).mean()),
                    "cov80": ((pit >= .1) & (pit <= .9)).mean()})
    info = [json.loads((d / f"bstsens_f{k}.json").read_text()) for k in range(ds_obj.K)]
    ms = pd.DataFrame([{"fold": i["fold"], "lead": L, **{kk: v for kk, v in i[L].items() if kk != "val_crps_grid"}}
                       for i in info for L in i if L not in ("fold", "sec")])
    ms.to_csv(out / f"bstsens_mstop_euppbench_{ftype}.csv", index=False)
    t = pd.DataFrame(res); t.to_csv(out / f"bstsens_euppbench_{ftype}.csv", index=False, float_format="%.5f")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("--datasets", nargs="*", default=["euppbench", "german"])
    ap.add_argument("--seed-cap-spatial", type=int, default=None)
    ap.add_argument("--reps", type=int, default=D5["primary"]["reps"])
    ap.add_argument("--no-partitions", action="store_true")
    a = ap.parse_args()
    out = R5 / "eval" / a.tag; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); decisions = {"tag": a.tag, "seed_cap_spatial": a.seed_cap_spatial, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for ds in a.datasets:
        rng = np.random.default_rng(2026)
        Ns = len(np.unique(np.load(OLD_DIR[ds] / "test_meta.npz", allow_pickle=True)["station"]))
        W = rng.multinomial(Ns, np.ones(Ns) / Ns, size=a.reps).astype(float)  # shared across designs (coupled DiD)
        designs = ["random", "spatial"]
        if ds == "euppbench" and not a.no_partitions:
            designs += [f"part_{p}" for p in PARTITIONS]
        Dd = {}
        for design in designs:
            meths = ALL + (M1A + ["gnn_geo@3", "samos_attn@3"] if ds == "euppbench" and design in ("random", "spatial") else [])
            D = Design(ds, design, meths, a.seed_cap_spatial, a.reps, W, 99)
            if design.startswith("part_"):
                need = P5["B_partitions_euppbench"]["methods"]
                if not all(m in D.methods for m in need) or any(len(D.single[m]) < 3 for m in need if m not in BASE + STAT):
                    print(f"  {design}: incomplete -> excluded ({[m for m in need if m not in D.methods]})", flush=True)
                    decisions.setdefault("partitions_excluded_incomplete", []).append(design)
                    continue
            Dd[design] = D
            tag = f"{ds}_{design}"
            D.summary().to_csv(out / f"summary_{tag}.csv", index=False, float_format="%.5f")
            D.per_fold().to_csv(out / f"per_fold_{tag}.csv", float_format="%.5f")
            pairs = pair_list(D.methods)
            if design.startswith("part_"):
                pairs = [(x, "emos_bst_gl") for x in P5["B_partitions_euppbench"]["methods"] if x != "emos_bst_gl" and x in D.methods]
            if "gnn_geo_fcavail" in D.methods and "gnn_geo@3" in D.methods:
                pairs.append(("gnn_geo_fcavail", "gnn_geo@3"))
            if "samos_attn_fcavail" in D.methods and "samos_attn@3" in D.methods:
                pairs.append(("samos_attn_fcavail", "samos_attn@3"))
            rows = [D.compare(x, z, a.reps) for x, z in pairs]
            sig = pd.DataFrame(rows)
            if len(sig):
                prim = ~sig.a.str.contains("fcavail")
                sig["family_primary"] = prim
                sig["bh_reject_q05_nw94_hln"] = False
                sig.loc[prim, "bh_reject_q05_nw94_hln"] = benjamini_hochberg(sig.loc[prim, "p_nw94_hln"].values)
                sig["claim_C4"] = [
                    ("a_better" if r.diff < 0 else "a_worse") if (r.bh_reject_q05_nw94_hln and (r.st_nested_hi < 0 or r.st_nested_lo > 0))
                    else "tied" for r in sig.itertuples()]
                sig["C4_regional_caveat"] = [(c != "tied") and (r.fold_nested_lo <= 0 <= r.fold_nested_hi) for c, r in zip(sig.claim_C4, sig.itertuples())]
                sig["C5_no_ordering"] = sig.p_a_better_nested.between(0.1, 0.9)
            sig.to_csv(out / f"pairs_{tag}.csv", index=False, float_format="%.6g")
            core = [m for m in ["emos_bst_gl", "emos_regidw", "samos_lin", "nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid",
                                "nn_hybrid_res", "drn_lak", "gnn_geo", "samos_mlp", "samos_attn"] if m in D.methods]
            rp, pw = D.rank_probs(core)
            rp.to_csv(out / f"rankprob_{tag}.csv", index=False, float_format="%.4f")
            pw.to_csv(out / f"pairwise_p_row_better_{tag}.csv", float_format="%.4f")
            print(f"  {tag}: done ({time.time() - t0:.0f}s)", flush=True)
        # decision rules (section F)
        dec = {}
        if "random" in Dd and "spatial" in Dd:
            for F in ["drn_lak", "gnn_geo", "nn_unk", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res", "samos_mlp", "samos_attn"]:
                if all(F in Dd[x].methods for x in ("random", "spatial")):
                    dec[F] = did(Dd["spatial"], Dd["random"], F)
                    for x in ("random", "spatial"):
                        bs = Dd[x].boot_stat(F) - Dd[x].boot_stat("emos_bst_gl")
                        dec[F][f"{x}_F_minus_E_lo"], dec[F][f"{x}_F_minus_E_hi"] = (float(q) for q in np.quantile(bs, [.025, .975]))
                    dec[F]["n_seeds_spatial"] = len(Dd["spatial"].single[F]); dec[F]["n_seeds_random"] = len(Dd["random"].single[F])
                    parts = {}
                    for p in [d for d in Dd if d.startswith("part_")]:
                        if F in Dd[p].methods:
                            parts[p] = did(Dd[p], Dd["random"], F)
                    dec[F]["partitions"] = parts
        decisions[ds] = dec
        if ds == "euppbench":
            for ft in ("spatial", "random"):
                if ft in Dd:
                    bst_sensitivity(Dd[ft], ft, out)
    # C1-C3 verdicts
    v = {}
    e = decisions.get("euppbench", {}); g = decisions.get("german", {})
    if all(F in e for F in ("drn_lak", "gnn_geo")):
        ci_pos = all(e[F]["lo"] > 0 for F in ("drn_lak", "gnn_geo"))
        ci_neg = any(e[F]["hi"] < 0 for F in ("drn_lak", "gnn_geo"))
        np_ = {F: len(e[F]["partitions"]) for F in ("drn_lak", "gnn_geo")}
        pos = {F: sum(p["did"] > 0 for p in e[F]["partitions"].values()) for F in ("drn_lak", "gnn_geo")}
        nonpos = {F: np_[F] - pos[F] for F in pos}
        if ci_pos and all(pos[F] >= 4 for F in pos):
            c1 = "supported"
        elif ci_neg or any(nonpos[F] >= 4 for F in pos):
            c1 = "contradicted"
        else:
            c1 = "inconclusive"
        v["C1"] = {"verdict": c1, "partitions_available": np_, "partitions_did_positive": pos,
                   "note": "needs all 6 partitions for a full 'supported' check" if min(np_.values()) < 6 else ""}
        v["C2"] = {F: bool(e[F]["random_F_minus_E_hi"] < 0 and e[F]["spatial_F_minus_E_lo"] > 0) for F in ("drn_lak", "gnn_geo")}
        if all(F in g for F in ("drn_lak", "gnn_geo")):
            v["C3_sparse_specific"] = bool(c1 == "supported" and all(g[F]["lo"] <= 0 for F in ("drn_lak", "gnn_geo")))
    decisions["verdicts"] = v
    json_atomic(out / "decisions.json", decisions)
    print("verdicts:", json.dumps(v, indent=1), f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
