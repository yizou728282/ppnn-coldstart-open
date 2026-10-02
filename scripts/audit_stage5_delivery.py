"""Audit the delivered Stage-5 logs and final reports without rerunning inference.

The original Stage-2/3/4 predictions are not bundled in the Stage-5 release;
therefore this audit checks completeness and report consistency, not a fresh
recalculation of every statistic from all original prediction arrays.
"""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results/stage5"
FINAL = R5 / "eval/final"
PARTS = {"km7_s2027": 7, "km7_s2028": 7, "km7_s2029": 7,
         "km5_s2026": 5, "km10_s2026": 10, "loco": 5}
TRAINED = ["unk", "noemb", "attr", "hybrid", "hybrid_res", "drn_lak", "gnn_geo", "samos_mlp", "samos_attn"]
NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_knn", "nn_hybrid_res",
        "drn_lak", "gnn_geo", "samos_mlp", "samos_attn"]


def rows(name):
    with (FINAL / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    log = (R5 / "logs/launcher.log").read_text(encoding="utf-8")
    starts = re.findall(r" START (\S+) ", log)
    ends = re.findall(r" END (\S+) exit (-?\d+)", log)
    assert len(starts) == len(ends) == 47
    assert set(starts) == {name for name, _ in ends}
    assert all(rc == "0" for _, rc in ends)
    assert "SCOPE-CUT" not in log
    missing = []
    count = 0
    for ds, stage in (("german", "stage2"), ("euppbench", "stage3")):
        for m in TRAINED:
            old = ROOT / "results" / (stage if m in TRAINED[:5] else f"stage4/{ds}") / "logs/spatial"
            extra = R5 / ds / "spatial/logs/spatial"
            for fold in range(7):
                for seed in range(10):
                    name = f"{m}_f{fold}_s{seed}.json"
                    count += 1
                    if not any((d / name).is_file() for d in (old, extra)):
                        missing.append(f"{ds}/{name}")
        summary = {row["method"]: row for row in rows(f"summary_{ds}_spatial.csv")}
        assert all(int(summary[m]["n_seeds"]) == 10 for m in NETS)
    for part, folds in PARTS.items():
        base = R5 / "euppbench" / f"part_{part}" / "logs"
        for fold in range(folds):
            for rel in (f"baselines_spatial_f{fold}.json", f"spatial/statbase_f{fold}.json"):
                count += 1
                if not (base / rel).is_file():
                    missing.append(f"part_{part}/{rel}")
            for method in ("unk", "hybrid_res", "drn_lak", "gnn_geo", "samos_mlp"):
                for seed in range(3):
                    rel = f"spatial/{method}_f{fold}_s{seed}.json"
                    count += 1
                    if not (base / rel).is_file():
                        missing.append(f"part_{part}/{rel}")
        summary = rows(f"summary_euppbench_part_{part}.csv")
        assert all(int(r["n_seeds"]) == 3 for r in summary if r["method"] in NETS)
    assert not missing, missing
    decisions = json.loads((FINAL / "decisions.json").read_text())
    eu, de = decisions["euppbench"], decisions["german"]
    c1 = all(eu[m]["n_seeds_spatial"] == 10 and eu[m]["lo"] > 0
             and len(eu[m]["partitions"]) == 6
             and sum(x["did"] > 0 for x in eu[m]["partitions"].values()) >= 4
             for m in ("drn_lak", "gnn_geo"))
    c2 = {m: eu[m]["random_F_minus_E_hi"] < 0 and eu[m]["spatial_F_minus_E_lo"] > 0
          for m in ("drn_lak", "gnn_geo")}
    c3 = c1 and all(de[m]["lo"] <= 0 for m in ("drn_lak", "gnn_geo"))
    assert decisions["verdicts"]["C1"]["verdict"] == ("supported" if c1 else "inconclusive")
    assert decisions["verdicts"]["C2"] == c2
    assert decisions["verdicts"]["C3_sparse_specific"] == c3
    report = {
        "source_commit": "8a7c56d", "completed_jobs": 47, "nonzero_exits": 0, "scope_cut": False,
        "checked_training_logs": count, "missing_training_logs": missing,
        "spatial_seeds_each_network_each_dataset": 10, "additional_partitions": len(PARTS),
        "C1_supported": c1, "C2_reversal": c2, "C3_condition_met": c3,
        "limitation": "Consistency/completeness audit of committed logs and final reports; not a fresh full inference run."
    }
    out = ROOT / "handoff/STAGE5_DELIVERY_AUDIT.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
