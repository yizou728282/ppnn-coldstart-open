"""Stage 3: EUPPBench Zarr subset -> compact training/test tables (npz).

  python scripts/stage3_preprocess.py --zarr /path/euppbench/zarr --out data/stage3_euppbench.npz

Output arrays (see protocols/stage3_euppbench.json):
  train_*: reforecast rows with valid date <= 2016-12-31 (fit = valid 1997-2014, early stopping = 2015-2016)
  test_*:  forecast rows (730 daily 00 UTC inits 2017-2018), features from 51 members (test_X) and from
           members 0-10 (test_X11, sensitivity)
  stations: per-station metadata table (all stations; 'included' flag from the inclusion rule)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["austria", "belgium", "france", "germany", "netherlands"]
LEADS = [24, 48, 72, 96, 120]
SURFACE = ["t2m", "u10", "v10", "tcc", "sd", "stl1", "swvl1", "tcwv", "cape"]
POSTPROC = ["mx2t6", "mn2t6", "sshf6", "slhf6", "ssr6", "str6"]
P850 = ["t850"]
VARS = SURFACE + POSTPROC + P850
KELVIN = {"t2m", "stl1", "mx2t6", "mn2t6", "t850"}
FEATURES = [f"{v}_{s}" for v in VARS for s in ("mean", "sd")]


def open_store(root, name):
    import zarr  # lazy: tests and the Colab/CPU runner do not need zarr
    return zarr.open_consolidated(str(root / name), mode="r")


def step_index(g, leads):
    st = np.asarray(g["step"][:], float)
    return [int(np.nonzero(np.isclose(st, L))[0][0]) for L in leads]


def ens_stats(arr, member_axis, members=None):
    a = np.asarray(arr, np.float64)
    if members is not None:
        a = np.take(a, members, axis=member_axis)
    return a.mean(member_axis), a.std(member_axis, ddof=1)


def load_var(root, kind, country, var, member_subsets):
    """Return dict subset -> (mean, sd) with shape [S, T, (Y,) L] and lead order LEADS."""
    if var in SURFACE:
        g = open_store(root, f"stations_ensemble_{kind}_surface_{country}.zarr"); key = var
    elif var in POSTPROC:
        g = open_store(root, f"stations_ensemble_{kind}_surface_postprocessed_{country}.zarr"); key = var
    else:
        g = open_store(root, f"stations_ensemble_{kind}_pressure_850_{country}.zarr"); key = "t"
    si = step_index(g, LEADS)
    a = g[key]
    # reforecasts: [S, T, M, Y, step, 1]; forecasts: [S, M, T, step, 1]
    out = {}
    for sub, members in member_subsets.items():
        ms, ss = [], []
        for s in range(a.shape[0]):  # one station chunk at a time (memory)
            x = a[s][..., si, 0]
            m, sd = ens_stats(x, 1 if kind == "reforecasts" else 0, members)
            ms.append(m); ss.append(sd)
        m, sd = np.stack(ms), np.stack(ss)
        if var in KELVIN:
            m = m - 273.15
        out[sub] = (m.astype(np.float32), sd.astype(np.float32))
    return out


def landuse_group(code):
    c = int(code)
    if 1 <= c <= 11: return 0
    if 12 <= c <= 22: return 1
    if 23 <= c <= 34: return 2
    if 35 <= c <= 39: return 3
    return 4


def build(root: Path):
    train_parts, test_parts, st_rows = [], [], []
    for ci, c in enumerate(COUNTRIES):
        g = open_store(root, f"stations_ensemble_reforecasts_surface_{c}.zarr")
        gf = open_store(root, f"stations_ensemble_forecasts_surface_{c}.zarr")
        ro = open_store(root, f"stations_reforecasts_observations_surface_{c}.zarr")
        fo = open_store(root, f"stations_forecasts_observations_surface_{c}.zarr")
        sid = g["station_id"][:]
        for other in (gf, ro, fo):
            assert np.array_equal(other["station_id"][:], sid), c
        S = len(sid)
        # --- times
        vt_h = g["valid_time"][:]  # [T, Y, step] hours since 1997-01-02
        si = step_index(g, LEADS)
        vt = (np.datetime64("1997-01-02T00", "h") + vt_h[..., si].astype("timedelta64[h]"))  # [T,Y,L]
        init_r = np.datetime64("1997-01-02T00", "h") + vt_h[..., 0].astype("timedelta64[h]")  # [T,Y]
        ft = np.datetime64("1970-01-01T00:00:00", "s") + gf["time"][:].astype("timedelta64[s]")  # [T2]
        # --- obs
        ro_si = step_index(ro, LEADS); fo_si = step_index(fo, LEADS)
        y_r = np.transpose(ro["t2m"][:][:, :, ro_si, :], (3, 0, 1, 2)) - 273.15  # [S,T,Y,L]
        y_f = np.transpose(fo["t2m"][:][:, fo_si, :], (2, 0, 1)) - 273.15  # [S,T2,L]
        # --- features
        Fr = np.full((S,) + vt.shape + (len(FEATURES),), np.nan, np.float32)
        Ff = np.full((S, len(ft), len(LEADS), len(FEATURES)), np.nan, np.float32)
        Ff11 = np.full_like(Ff, np.nan)
        for vi, v in enumerate(VARS):
            r = load_var(root, "reforecasts", c, v, {"all": None})["all"]
            Fr[..., 2 * vi], Fr[..., 2 * vi + 1] = r
            f = load_var(root, "forecasts", c, v, {"all": None, "m11": list(range(11))})
            Ff[..., 2 * vi], Ff[..., 2 * vi + 1] = f["all"]
            Ff11[..., 2 * vi], Ff11[..., 2 * vi + 1] = f["m11"]
            print(c, v, "done", flush=True)
        # --- station metadata
        for s in range(S):
            st_rows.append({"station": ci * 100000 + int(sid[s]), "country": c, "station_id_raw": int(sid[s]),
                            "name": str(g["station_name"][s]), "lat": float(g["station_latitude"][s]),
                            "lon": float(g["station_longitude"][s]), "alt": float(g["station_altitude"][s]),
                            "orog": float(g["model_orography"][s]), "model_altitude": float(g["model_altitude"][s]),
                            "land_use": int(g["station_land_usage"][s]), "lu_group": landuse_group(g["station_land_usage"][s])})
        gid = ci * 100000 + sid.astype(np.int64)
        # --- long tables
        T, Y, L = vt.shape
        sidx, tidx, yidx, lidx = np.meshgrid(np.arange(S), np.arange(T), np.arange(Y), np.arange(L), indexing="ij")
        train_parts.append(dict(station=gid[sidx].ravel(), lead=np.array(LEADS)[lidx].ravel(),
                                init=np.broadcast_to(init_r[None, :, :, None], sidx.shape).ravel().astype("datetime64[D]"),
                                valid=np.broadcast_to(vt[None], sidx.shape).ravel().astype("datetime64[D]"),
                                y=y_r.ravel().astype(np.float32), X=Fr.reshape(-1, len(FEATURES))))
        T2 = len(ft)
        sidx, tidx, lidx = np.meshgrid(np.arange(S), np.arange(T2), np.arange(L), indexing="ij")
        init_f = ft.astype("datetime64[D]")
        test_parts.append(dict(station=gid[sidx].ravel(), lead=np.array(LEADS)[lidx].ravel(),
                               init=init_f[tidx].ravel(),
                               valid=(init_f[tidx] + (np.array(LEADS) // 24)[lidx].astype("timedelta64[D]")).ravel(),
                               y=y_f.ravel().astype(np.float32), X=Ff.reshape(-1, len(FEATURES)),
                               X11=Ff11.reshape(-1, len(FEATURES))))
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])
    tr = {k: cat(train_parts, k) for k in train_parts[0]}
    te = {k: cat(test_parts, k) for k in test_parts[0]}
    return tr, te, pd.DataFrame(st_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr, te, st = build(Path(a.zarr))
    keep_tr = tr["valid"] <= np.datetime64("2016-12-31")
    n_excl_2017 = int((~keep_tr).sum())
    tr = {k: v[keep_tr] for k, v in tr.items()}
    it = FEATURES.index("t2m_mean"), FEATURES.index("t2m_sd")
    ok_tr = np.isfinite(tr["y"]) & np.isfinite(tr["X"][:, it[0]]) & np.isfinite(tr["X"][:, it[1]])
    ok_te = np.isfinite(te["y"]) & np.isfinite(te["X"][:, it[0]]) & np.isfinite(te["X"][:, it[1]])
    # station inclusion rule: >= 50% non-missing obs (with t2m forecast) in reforecast (<=2016) and test periods
    fr_tr = pd.Series(ok_tr).groupby(tr["station"]).mean(); fr_te = pd.Series(ok_te).groupby(te["station"]).mean()
    st = st.set_index("station")
    st["frac_obs_reforecast"] = fr_tr.reindex(st.index).values
    st["frac_obs_test"] = fr_te.reindex(st.index).values
    st["included"] = (st.frac_obs_reforecast >= 0.5) & (st.frac_obs_test >= 0.5)
    inc = set(st.index[st.included])
    m_tr = ok_tr & np.isin(tr["station"], list(inc)); m_te = ok_te & np.isin(te["station"], list(inc))
    tr = {k: v[m_tr] for k, v in tr.items()}; te = {k: v[m_te] for k, v in te.items()}
    info = {"n_train_rows": int(len(tr["y"])), "n_test_rows": int(len(te["y"])),
            "n_reforecast_rows_dropped_valid_ge_2017": n_excl_2017,
            "n_train_rows_dropped_missing_obs_or_t2m": int((~ok_tr).sum()),
            "n_test_rows_dropped_missing_obs_or_t2m": int((~ok_te).sum()),
            "n_stations_total": int(len(st)), "n_stations_included": int(st.included.sum()),
            "excluded_stations": st.index[~st.included].tolist(),
            "missing_other_predictors_train_frac": np.isnan(tr["X"]).mean(0).round(6).tolist(),
            "missing_other_predictors_test_frac": np.isnan(te["X"]).mean(0).round(6).tolist(),
            "features": FEATURES, "leads": LEADS,
            "train_valid_range": [str(tr["valid"].min()), str(tr["valid"].max())],
            "test_valid_range": [str(te["valid"].min()), str(te["valid"].max())],
            "test_init_range": [str(te["init"].min()), str(te["init"].max())]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=np.array(FEATURES),
                        **{f"train_{k}": (v.astype("datetime64[D]").astype(np.int32) if v.dtype.kind == "M" else v) for k, v in tr.items()},
                        **{f"test_{k}": (v.astype("datetime64[D]").astype(np.int32) if v.dtype.kind == "M" else v) for k, v in te.items()},
                        st_station=st.index.to_numpy(), st_table=st.reset_index().to_json())
    st.to_csv(out.with_name("stage3_stations.csv"))
    out.with_name("stage3_preprocess_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if "frac" not in k}, indent=1))


if __name__ == "__main__":
    main()
