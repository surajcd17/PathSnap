"""Build the candidate/feature table for the train and student_test partitions.

Outputs (results/ and artifacts/dataset/):
  artifacts/dataset/<partition>_cands.pkl   one row per (observation, candidate)
  artifacts/dataset/<partition>_labels.pkl  one row per observation (reference label)
  results/dataset_build_<partition>.csv     per-trip summary (miss rate, reliability...)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from pathsnap.features import prepare_trip
from pathsnap.io import data_root, list_trips

OUT = ROOT / "artifacts" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

for part in ("train", "student_test"):
    cands, labels, summ = [], [], []
    for tid in list_trips(part):
        t0 = time.time()
        td = prepare_trip(data_root() / part / tid)
        lab = td.labels.copy()
        lab.insert(0, "obs", range(len(lab)))
        lab.insert(0, "trip", tid)
        lab["t"] = td.track.t.to_numpy()
        cands.append(td.cands)
        labels.append(lab)
        summ.append(dict(trip=tid, n_obs=len(td.track), n_cand_rows=len(td.cands),
                         cands_per_obs=len(td.cands) / len(td.track),
                         build_s=time.time() - t0, **td.info))
        print(part, tid, f"{time.time() - t0:.1f}s", f"miss={td.info['cand_miss_rate']:.4f}",
              f"reliable={td.info['frac_reliable']:.3f}")
    pd.concat(cands, ignore_index=True).to_pickle(OUT / f"{part}_cands.pkl")
    pd.concat(labels, ignore_index=True).to_pickle(OUT / f"{part}_labels.pkl")
    pd.DataFrame(summ).to_csv(ROOT / "results" / f"dataset_build_{part}.csv", index=False)
