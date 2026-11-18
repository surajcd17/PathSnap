"""Export human-readable (CSV) views of the preprocessed data, for inspection/demo.

The full preprocessed tables are large pickles (artifacts/dataset/*.pkl, rebuilt by
scripts/02_build_dataset.py). This script writes small CSVs that open in Excel/any editor:

  results/preprocessed_samples/
    00_overview.csv                    partition-level counts (raw vs preprocessed)
    01_raw_track_<trip>.csv            the raw .track file with column names
    02_clean_track_<trip>.csv          after cleaning + projection: x/y metres, speed, heading,
                                       time gaps, reference label, alignment distance, reliable
    03_candidates_<trip>.csv           the model-ready table for that trip: one row per
                                       (GPS point, candidate road) with all 23 features and y
    04_candidates_first_20_points.csv  same, first 20 GPS points only (easy to read on screen)

Run:  python scripts/11_export_preprocessed_samples.py [trip_id]   (default train 00000003)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from pathsnap.features import FEATURES, prepare_trip
from pathsnap.io import data_root, load_trip

OUT = ROOT / "results" / "preprocessed_samples"
OUT.mkdir(parents=True, exist_ok=True)
TRIP = sys.argv[1] if len(sys.argv) > 1 else "00000003"

# ---- overview (raw files vs preprocessed tables)
rows = []
for part in ("train", "student_test"):
    c = pd.read_pickle(ROOT / f"artifacts/dataset/{part}_cands.pkl")
    lab = pd.read_pickle(ROOT / f"artifacts/dataset/{part}_labels.pkl")
    raw_pts = sum(sum(1 for _ in open(p)) for p in (data_root() / part).glob("*/*.track"))
    rows.append(dict(partition=part, trips=lab.trip.nunique(), raw_gps_rows=raw_pts, clean_gps_rows=len(lab),
                     reliable_gps_rows=int(lab.reliable.sum()), candidate_rows=len(c),
                     candidate_rows_reliable=int(c.reliable.sum()), features_per_row=len(FEATURES),
                     positive_rows_y1=int(c.y.sum()), candidate_miss_rate=1 - lab.in_cands[lab.reliable].mean()))
pd.DataFrame(rows).to_csv(OUT / "00_overview.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ---- one trip, step by step
part = "train" if (data_root() / "train" / TRIP).exists() else "student_test"
raw = load_trip(data_root() / part / TRIP)
raw.track.to_csv(OUT / f"01_raw_track_{TRIP}.csv", index=False)

td = prepare_trip(raw)
trk = td.track.copy()
trk["obs"] = range(len(trk))
trk = pd.concat([trk, td.labels], axis=1)
cols = ["obs", "orig_idx", "lon", "lat", "t", "x", "y", "dt_prev", "speed", "heading", "heading_long",
        "label_arc", "route_pos", "align_dist", "reliable", "in_cands"]
trk[cols].round({"lon": 6, "lat": 6, "x": 2, "y": 2, "dt_prev": 1, "speed": 2, "heading": 1,
                  "heading_long": 1, "align_dist": 2}).to_csv(OUT / f"02_clean_track_{TRIP}.csv", index=False)

cand_cols = ["trip", "obs", "arc_id", "y", "reliable"] + FEATURES
cands = td.cands[cand_cols].round(4)
cands.to_csv(OUT / f"03_candidates_{TRIP}.csv", index=False)
cands[cands.obs < 20].to_csv(OUT / "04_candidates_first_20_points.csv", index=False)
print(f"\ntrip {TRIP} ({part}): {len(raw.track)} raw rows -> {len(trk)} clean rows -> {len(cands)} candidate rows")
print("written to", OUT)
