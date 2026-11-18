"""Folium maps of predicted routes (student-test trips by default).

Layers: road network near the track, reference route, raw GPS, classifier-only route,
HMM route, and observations where the HMM disagrees with the reference label.

Run:  python scripts/06_maps.py [trip_id ...]
Output: results/maps/student_test_<trip>.html
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import folium
import numpy as np
from branca.element import Element

from pathsnap.hmm import argmax_pred, viterbi
from pathsnap.inference import PathSnapModel, route_geometry
from pathsnap.io import data_root, list_trips, load_trip
from pathsnap.features import prepare_trip
from pathsnap.metrics import evaluate_trip

OUT = ROOT / "results" / "maps"
OUT.mkdir(parents=True, exist_ok=True)

LEGEND = """
<div style="position:fixed; bottom:30px; left:30px; z-index:9999; background:white; padding:8px 10px;
            border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.3); font:13px sans-serif; line-height:1.6">
<b>{title}</b><br>
<span style="display:inline-block;width:18px;height:3px;background:#999"></span> Road network<br>
<span style="display:inline-block;width:18px;height:5px;background:#1a9641"></span> Reference route<br>
<span style="display:inline-block;width:10px;height:10px;border-radius:5px;background:#2463eb"></span> Raw GPS<br>
<span style="display:inline-block;width:18px;height:3px;background:#e07b00"></span> Classifier only<br>
<span style="display:inline-block;width:18px;height:4px;background:#c51b7d"></span> HMM / Viterbi route<br>
<span style="display:inline-block;width:10px;height:10px;background:black"></span> HMM error (vs reference)
</div>"""


def make_map(model, tid, part="student_test"):
    raw = load_trip(data_root() / part / tid)
    td = prepare_trip(raw)
    p = model.predict_proba(td.cands)
    clf = argmax_pred(td, p)
    hmm, _ = viterbi(td, p, model.hmm_cfg)
    mh, mc = evaluate_trip(td, hmm, raw.route), evaluate_trip(td, clf, raw.route)
    net = td.net
    trk = td.track
    fm = folium.Map(location=[trk.lat.mean(), trk.lon.mean()], zoom_start=15, tiles="OpenStreetMap")

    g_net = folium.FeatureGroup("Road network (within 60 m of GPS)", show=True)
    near = np.unique(np.concatenate(net.arcs_near(trk[["x", "y"]].to_numpy(), 60.0)))
    for a in near:
        u, v = int(net.u[a]), int(net.v[a])
        folium.PolyLine([[net.nodes_lonlat[u, 1], net.nodes_lonlat[u, 0]],
                         [net.nodes_lonlat[v, 1], net.nodes_lonlat[v, 0]]],
                        color="#999", weight=1.5, opacity=.7).add_to(g_net)
    g_ref = folium.FeatureGroup("Reference route", show=True)
    for k in raw.route:
        u, v = raw.arcs[k]
        folium.PolyLine([[raw.nodes[u, 1], raw.nodes[u, 0]], [raw.nodes[v, 1], raw.nodes[v, 0]]],
                        color="#1a9641", weight=7, opacity=.6).add_to(g_ref)
    g_gps = folium.FeatureGroup("Raw GPS", show=True)
    for lat, lon, t in trk[["lat", "lon", "t"]].to_numpy():
        folium.CircleMarker([lat, lon], radius=2, color="#2463eb", fill=True, weight=1,
                            tooltip=f"t={t:.0f}s").add_to(g_gps)
    g_clf = folium.FeatureGroup("Classifier-only route", show=False)
    folium.PolyLine(route_geometry(td, clf), color="#e07b00", weight=3, dash_array="6,6").add_to(g_clf)
    g_hmm = folium.FeatureGroup("HMM / Viterbi route", show=True)
    folium.PolyLine(route_geometry(td, hmm), color="#c51b7d", weight=3.5).add_to(g_hmm)
    g_err = folium.FeatureGroup("HMM errors", show=False)
    pred_arc = np.where(hmm >= 0, net.arc_ids[np.maximum(hmm, 0)], -1)
    wrong = (pred_arc != td.labels.label_arc.to_numpy()) & td.labels.reliable.to_numpy()
    for lat, lon in trk.loc[wrong, ["lat", "lon"]].to_numpy():
        folium.RegularPolygonMarker([lat, lon], number_of_sides=4, radius=4, color="black", fill=True).add_to(g_err)
    for g in (g_net, g_ref, g_gps, g_clf, g_hmm, g_err):
        g.add_to(fm)
    folium.Marker(trk[["lat", "lon"]].iloc[0].tolist(), tooltip="start", icon=folium.Icon(color="green")).add_to(fm)
    folium.Marker(trk[["lat", "lon"]].iloc[-1].tolist(), tooltip="end", icon=folium.Icon(color="red")).add_to(fm)
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.fit_bounds([[trk.lat.min(), trk.lon.min()], [trk.lat.max(), trk.lon.max()]])
    title = (f"{part} trip {tid}: Top-1 classifier {100 * mc['top1_acc']:.1f}% -> HMM {100 * mh['top1_acc']:.1f}%")
    fm.get_root().html.add_child(Element(LEGEND.format(title=title)))
    path = OUT / f"{part}_{tid}.html"
    fm.save(str(path))
    print(path, title)


if __name__ == "__main__":
    model = PathSnapModel()
    for t in (sys.argv[1:] or list_trips("student_test")):
        make_map(model, t)
