"""Exploratory data analysis of the raw dataset (train + student_test partitions).

Outputs in results/eda/:
  inventory_<partition>.csv     per-trip file sizes, row counts, gaps, extent, network stats
  data_quality_notes.md         short record of issues found
  *.png                         distributions, trajectory plots, road-network plot
  map_train_<id>.html           Folium map (GPS + road network + reference route)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geo import Projector
from pathsnap.io import data_root, list_trips, load_trip

OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def trip_stats(part, tid):
    d = data_root() / part / tid
    tr = load_trip(d)
    trk = tr.track
    pj = Projector(trk.lon.mean(), trk.lat.mean())
    xy = pj.to_xy(trk.lon, trk.lat)
    step = np.hypot(*np.diff(xy, axis=0).T)
    dt = np.diff(trk.t.to_numpy())
    N, A = len(tr.nodes), len(tr.arcs)
    G = coo_matrix((np.ones(A), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(N, N)).tocsr()
    n_weak, _ = connected_components(G, directed=True, connection="weak")
    n_strong, lab_s = connected_components(G, directed=True, connection="strong")
    deg = np.bincount(tr.arcs.ravel(), minlength=N)
    route_ok = np.mean(tr.arcs[tr.route[:-1], 1] == tr.arcs[tr.route[1:], 0]) if tr.route is not None else np.nan
    sizes = {ext: (d / f"{tid}.{ext}").stat().st_size / 1024 for ext in ("track", "nodes", "arcs", "route")
             if (d / f"{tid}.{ext}").exists()}
    return dict(
        partition=part, trip=tid, utm_epsg=pj.epsg,
        files=len(list(d.iterdir())), **{f"kb_{k}": round(v, 1) for k, v in sizes.items()},
        n_gps=len(trk), n_nodes=N, n_arcs=A, route_len=len(tr.route) if tr.route is not None else 0,
        missing_values=int(trk.isna().sum().sum()), non_monotone_t=int((dt <= 0).sum()),
        duration_s=float(trk.t.iloc[-1] - trk.t.iloc[0]), length_km=float(step.sum() / 1000),
        frac_dt_1s=float(np.mean(dt == 1)), n_gaps_gt_5s=int((dt > 5).sum()), max_gap_s=float(dt.max()),
        median_speed_ms=float(np.median(step / np.maximum(dt, 1e-6))),
        max_step_m=float(step.max()), lon_min=trk.lon.min(), lon_max=trk.lon.max(),
        lat_min=trk.lat.min(), lat_max=trk.lat.max(),
        extent_km_x=float(np.ptp(xy[:, 0]) / 1000), extent_km_y=float(np.ptp(xy[:, 1]) / 1000),
        isolated_nodes=int((deg == 0).sum()), weak_components=int(n_weak), strong_components=int(n_strong),
        largest_scc_frac=float(np.bincount(lab_s).max() / N), route_consecutive_connected=float(route_ok),
    ), (tr, xy, pj, dt, step)


def main():
    inv, cache = {}, {}
    for part in ("train", "student_test"):
        rows = []
        for tid in list_trips(part):
            s, c = trip_stats(part, tid)
            rows.append(s); cache[(part, tid)] = c
        inv[part] = pd.DataFrame(rows)
        inv[part].to_csv(OUT / f"inventory_{part}.csv", index=False)
    allinv = pd.concat(inv.values())
    summary = allinv.groupby("partition")[["n_gps", "n_nodes", "n_arcs", "route_len", "duration_s", "length_km",
                                           "max_gap_s", "missing_values"]].describe().T
    summary.to_csv(OUT / "summary_stats.csv")
    print(allinv.groupby("partition").agg(trips=("trip", "count"), gps=("n_gps", "sum"), missing=("missing_values", "sum"),
                                          nonmono=("non_monotone_t", "sum"), gaps5=("n_gaps_gt_5s", "sum")))

    # --- distributions (train partition)
    tri = inv["train"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ax[0, 0].hist(tri.n_gps, bins=30); ax[0, 0].set_title("GPS observations per trip (train)")
    dts = np.concatenate([cache[("train", t)][3] for t in tri.trip])
    ax[0, 1].hist(np.clip(dts, 0, 10), bins=np.arange(0, 11.5, .5)); ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Sampling interval (s, clipped at 10) - train")
    sp = np.concatenate([cache[("train", t)][4] / np.maximum(cache[("train", t)][3], 1e-6) for t in tri.trip])
    ax[0, 2].hist(np.clip(sp, 0, 50), bins=50); ax[0, 2].set_title("Speed between fixes (m/s) - train")
    ax[1, 0].hist(tri.length_km, bins=30); ax[1, 0].set_title("Trip length (km) - train")
    ax[1, 1].scatter(tri.n_arcs, tri.n_nodes, s=12); ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("arcs"); ax[1, 1].set_ylabel("nodes"); ax[1, 1].set_title("Road-network size per trip - train")
    ax[1, 2].hist(tri.max_gap_s, bins=30); ax[1, 2].set_yscale("log"); ax[1, 2].set_title("Largest GPS gap per trip (s) - train")
    plt.tight_layout(); plt.savefig(OUT / "distributions_train.png", dpi=120); plt.close()

    # GPS-to-reference-route distance (from the label alignment)
    lab = pd.read_pickle(ROOT / "artifacts/dataset/train_labels.pkl") if (ROOT / "artifacts/dataset/train_labels.pkl").exists() else None
    if lab is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(np.clip(lab.align_dist, 0, 100), bins=100)
        plt.axvline(50, color="r", ls="--", label="reliability threshold 50 m"); plt.legend(); plt.yscale("log")
        plt.xlabel("distance GPS point -> aligned reference arc (m, clipped 100)")
        plt.title("GPS noise relative to reference route - train"); plt.tight_layout()
        plt.savefig(OUT / "gps_to_route_distance_train.png", dpi=120); plt.close()

    # --- trajectory plots (first 12 train trips, metric coordinates)
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for a, t in zip(ax.ravel(), tri.trip[:12]):
        xy = cache[("train", t)][1]
        a.plot(xy[:, 0] - xy[0, 0], xy[:, 1] - xy[0, 1], ".", ms=1)
        a.set_title(f"train {t}"); a.set_aspect("equal"); a.tick_params(labelsize=6)
    plt.suptitle("GPS trajectories (UTM metres, relative to first fix) - train"); plt.tight_layout()
    plt.savefig(OUT / "trajectories_train.png", dpi=120); plt.close()

    # --- road network + track + route for one trip
    tid = "00000003"
    tr, xy, pj, _, _ = cache[("train", tid)]
    N = pj.to_xy(tr.nodes[:, 0], tr.nodes[:, 1])
    lo, hi = xy.min(0) - 300, xy.max(0) + 300
    a, b = N[tr.arcs[:, 0]], N[tr.arcs[:, 1]]
    m = ((a >= lo) & (a <= hi)).all(1)
    plt.figure(figsize=(8, 8))
    for (x0, y0), (x1, y1) in zip(a[m], b[m]):
        plt.plot([x0, x1], [y0, y1], color="0.75", lw=.6)
    for k in tr.route:
        plt.plot(*np.c_[N[tr.arcs[k, 0]], N[tr.arcs[k, 1]]], color="g", lw=2.5)
    plt.plot(xy[:, 0], xy[:, 1], ".", color="b", ms=2)
    plt.gca().set_aspect("equal"); plt.title(f"Road network (grey), reference route (green), GPS (blue) - train {tid}")
    plt.tight_layout(); plt.savefig(OUT / f"road_network_train_{tid}.png", dpi=130); plt.close()

    # --- Folium map for the same trip
    fm = folium.Map(location=[tr.track.lat.mean(), tr.track.lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    net = folium.FeatureGroup("Road network (arcs near track)", show=True)
    for i in np.flatnonzero(m)[:6000]:
        u, v = tr.arcs[i]
        folium.PolyLine([[tr.nodes[u, 1], tr.nodes[u, 0]], [tr.nodes[v, 1], tr.nodes[v, 0]]],
                        color="#888", weight=1, opacity=.6).add_to(net)
    route = folium.FeatureGroup("Reference route", show=True)
    for k in tr.route:
        u, v = tr.arcs[k]
        folium.PolyLine([[tr.nodes[u, 1], tr.nodes[u, 0]], [tr.nodes[v, 1], tr.nodes[v, 0]]],
                        color="green", weight=5, opacity=.8).add_to(route)
    gps = folium.FeatureGroup("Raw GPS", show=True)
    for lat, lon in tr.track[["lat", "lon"]].to_numpy():
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, weight=1).add_to(gps)
    for g in (net, route, gps):
        g.add_to(fm)
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.save(str(OUT / f"map_train_{tid}.html"))

    # --- data quality notes
    notes = [
        "# Data-quality notes (generated by scripts/01_eda.py)", "",
        f"* Trips: train={len(inv['train'])}, student_test={len(inv['student_test'])}; every trip has .track/.nodes/.arcs/.route.",
        f"* Missing values in .track: {int(allinv.missing_values.sum())}; non-monotone timestamps: {int(allinv.non_monotone_t.sum())}.",
        f"* Fraction of 1 s sampling steps (train mean): {tri.frac_dt_1s.mean():.3f}; gaps > 5 s: {int(tri.n_gaps_gt_5s.sum())}; largest gap: {tri.max_gap_s.max():.0f} s (trip {tri.loc[tri.max_gap_s.idxmax(), 'trip']}).",
        f"* Trips cover several regions (UTM zones used: {sorted(allinv.utm_epsg.unique().tolist())}); each trip is projected into its own UTM zone.",
        "* Arc ids in .route are 0-based row indices of .arcs; consecutive route arcs are connected head-to-tail (directed graph).",
        f"* Road networks are given per trip and are much larger than the corridor around the track (median {int(tri.n_arcs.median())} arcs); we crop to a 400 m corridor.",
        f"* Isolated nodes (train total): {int(tri.isolated_nodes.sum())}; trips whose network has >1 weak component: {int((tri.weak_components > 1).sum())}.",
        "* Some reference routes cover only part of the GPS track (e.g. train 00000054, 00000033): points farther than 50 m from the aligned reference arc are marked unreliable and excluded from training/scoring.",
    ]
    (OUT / "data_quality_notes.md").write_text("\n".join(notes))
    print("\n".join(notes))


if __name__ == "__main__":
    main()
