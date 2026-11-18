"""Quick checks of the raw-data conventions (arc-id base, direction, connectivity).

Run:  python scripts/00_sanity_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial import cKDTree

from pathsnap.geo import Projector, point_segment
from pathsnap.io import data_root, list_trips, load_trip


def check(trip_dir):
    tr = load_trip(trip_dir)
    lon0, lat0 = tr.track[["lon", "lat"]].mean()
    pj = Projector(lon0, lat0)
    P = pj.to_xy(tr.track.lon, tr.track.lat)
    N = pj.to_xy(tr.nodes[:, 0], tr.nodes[:, 1])
    out = {"trip": tr.trip_id, "n_pts": len(P), "n_arcs": len(tr.arcs), "route_len": len(tr.route),
           "route_max_lt_A": bool(tr.route.max() < len(tr.arcs)), "route_min": int(tr.route.min())}
    for base in (0, 1):
        r = tr.route - base
        r = r[(r >= 0) & (r < len(tr.arcs))]
        a, b = N[tr.arcs[r, 0]], N[tr.arcs[r, 1]]
        # nearest route segment per GPS point (brute force over route arcs)
        best = np.full(len(P), np.inf)
        for k in range(len(r)):
            d, _ = point_segment(P, np.repeat(a[k:k + 1], len(P), 0), np.repeat(b[k:k + 1], len(P), 0))
            best = np.minimum(best, d)
        out[f"median_dist_base{base}"] = round(float(np.median(best)), 2)
        out[f"p95_dist_base{base}"] = round(float(np.percentile(best, 95)), 2)
    r = tr.route
    fwd = np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0])
    rev = np.mean(tr.arcs[r[:-1], 0] == tr.arcs[r[1:], 1])
    same = np.mean(r[:-1] == r[1:])
    out.update(route_fwd_connected=round(float(fwd), 3), route_rev_connected=round(float(rev), 3),
               route_repeats=round(float(same), 3))
    pairs = set(map(tuple, tr.arcs.tolist()))
    twins = np.mean([(v, u) in pairs for u, v in tr.arcs.tolist()])
    out["frac_arcs_with_reverse_twin"] = round(float(twins), 3)
    out["dup_arcs"] = int(len(tr.arcs) - len(pairs))
    seglen = np.hypot(*(N[tr.arcs[:, 1]] - N[tr.arcs[:, 0]]).T)
    out["arc_len_median_m"] = round(float(np.median(seglen)), 1)
    out["arc_len_max_m"] = round(float(seglen.max()), 1)
    out["zero_len_arcs"] = int((seglen < 0.01).sum())
    t = tr.track.t.to_numpy()
    out["nan_rows"] = int(tr.track.isna().any(axis=1).sum())
    out["non_monotone_t"] = int((np.diff(t) <= 0).sum())
    out["max_gap_s"] = float(np.diff(t).max())
    return out


if __name__ == "__main__":
    root = data_root()
    for tid in list_trips("train")[:4] + list_trips("student_test")[:2]:
        part = "train" if (root / "train" / tid).exists() else "student_test"
        print(check(root / part / tid))
