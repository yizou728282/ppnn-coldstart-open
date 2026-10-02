"""Fit baselines and networks under the year-isolated protocol and save 2016 test predictions.

Protocol (configs/default.json):
  stage 1: fit on train_years (2007-2014), early-stop on val_year (2015) -> best epoch
  stage 2: refit from the same seed on 2007-2015 for exactly `best epoch` epochs
  test:    2016, never used for fitting, normalization, or model selection.
Stations absent from the fitting data get the reserved UNK embedding (networks) or global
EMOS parameters (local EMOS); they are flagged in results/preds/test_meta.npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppnn_residual.data import Preprocessor, feature_columns, load_table, year_split, UNK  # noqa: E402
from ppnn_residual.emos import fit_emos, predict_emos  # noqa: E402
from ppnn_residual.train import predict, train_network  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument("--baselines", action="store_true", help="fit raw/EMOS baselines")
    ap.add_argument("--networks", nargs="*", default=[], help="network names from config (or 'all')")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["nn"]["num_threads"])

    df = load_table(str(ROOT / cfg["data_path"]), cfg["drop_columns"])
    cols = feature_columns(df)
    tr1 = year_split(df, cfg["train_years"])
    va = year_split(df, [cfg["val_year"]])
    tr2 = year_split(df, cfg["train_years"] + [cfg["val_year"]])
    te = year_split(df, [cfg["test_year"]])
    print(f"rows: train {len(tr1)}  val {len(va)}  train+val {len(tr2)}  test {len(te)}; features {len(cols)}")

    pp2 = Preprocessor().fit(tr2, cols)
    te2 = pp2.transform(te)
    unseen = te2.emb_idx == UNK
    np.savez(out / "preds/test_meta.npz", station=te2.station, date=te2.date.astype("datetime64[D]").astype(str),
             y=te2.y, unseen=unseen, lat=te["lat"].to_numpy(), lon=te["lon"].to_numpy(),
             ens_mean=te2.ens_mean, ens_sd=te2.ens_sd)
    print(f"test unseen-station rows: {unseen.sum()} stations {sorted(set(te2.station[unseen].tolist()))}")

    if args.baselines:
        np.savez(out / "preds/raw_ensemble_gauss.npz", mu=te2.ens_mean.astype(np.float64),
                 sigma=te2.ens_sd.astype(np.float64))
        t0 = time.time()
        pg, kg, info_g = fit_emos(tr2.t2m_mean, tr2.t2m_var, tr2.obs)
        mu, sg, _ = predict_emos(pg, kg, te.t2m_mean, te.t2m_var)
        np.savez(out / "preds/emos_gl.npz", mu=mu, sigma=sg)
        pl, kl, info_l = fit_emos(tr2.t2m_mean, tr2.t2m_var, tr2.obs, groups=tr2.station.to_numpy())
        mu, sl, uns = predict_emos(pl, kl, te.t2m_mean, te.t2m_var, groups=te.station.to_numpy(), fallback=pg[0])
        assert (uns == unseen).all()
        np.savez(out / "preds/emos_loc.npz", mu=mu, sigma=sl)
        json.dump({"emos_gl": {"params_a_b_c_d": pg[0].tolist(), **info_g},
                   "emos_loc": {"n_stations": int(len(kl)), "fallback": "emos_gl params", **info_l},
                   "sec": round(time.time() - t0, 1)},
                  open(out / "logs/emos_fit.json", "w"), indent=2)
        np.savez(out / "preds/emos_loc_params.npz", params=pl, stations=kl)
        print("EMOS done", info_g, info_l)

    nets = list(cfg["networks"]) if args.networks == ["all"] else args.networks
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if nets:
        pp1 = Preprocessor().fit(tr1, cols)
        tr1s, vas = pp1.transform(tr1), pp1.transform(va)
        tr2s = pp2.transform(tr2)
    hp = cfg["nn"]
    for name in nets:
        arch = cfg["networks"][name]
        for seed in seeds:
            tag = f"{name}_seed{seed}"
            if (out / f"preds/{tag}.npz").exists():
                print("skip existing", tag)
                continue
            t0 = time.time()
            logf = open(out / f"logs/{tag}.log", "w")

            def log(r, _f=logf):
                print(tag, r, flush=True)
                _f.write(json.dumps(r) + "\n")
                _f.flush()
            log({"stage": 1, "fit_years": cfg["train_years"], "val_year": cfg["val_year"]})
            _, hist1, best_ep = train_network(tr1s, pp1.n_emb, arch, hp, seed, hp["max_epochs"],
                                              val=vas, patience=hp["patience"], log=log)
            log({"stage": 2, "refit_epochs": best_ep})
            model, hist2, _ = train_network(tr2s, pp2.n_emb, arch, hp, seed, best_ep, log=log)
            mu, sig = predict(model, te2)
            np.savez(out / f"preds/{tag}.npz", mu=mu, sigma=sig)
            json.dump({"name": name, "seed": seed, "arch": arch, "hp": hp, "best_epoch": best_ep,
                       "best_val_crps": min(h["val_crps"] for h in hist1), "stage1": hist1, "stage2": hist2,
                       "n_params": sum(p.numel() for p in model.parameters()),
                       "sec": round(time.time() - t0, 1)}, open(out / f"logs/{tag}.json", "w"), indent=2)
            torch.save(model.state_dict(), out / f"logs/{tag}.pt")
            logf.close()


if __name__ == "__main__":
    main()
