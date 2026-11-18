"""Inference pipeline used by the API (and reusable from scripts).

predict_trip() never sees the reference route. Scoring against the reference route is
a separate, server-side step (score_prediction) that runs AFTER prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, attach_labels, prepare_trip
from .hmm import HMMConfig, argmax_pred, viterbi
from .io import PROJECT_ROOT, load_trip
from .metrics import evaluate_trip

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


class PathSnapModel:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        model_dir = Path(model_dir)
        self.clf = joblib.load(model_dir / "model.joblib")
        cfg = json.loads((model_dir / "config.json").read_text())
        self.features = cfg.get("features", FEATURES)
        self.hmm_cfg = HMMConfig(**cfg.get("hmm", {}))
        self.meta = cfg

    def predict_proba(self, cands: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(cands[self.features])[:, 1]

    def predict_trip(self, trip_dir: Path | str):
        t0 = time.perf_counter()
        raw = load_trip(trip_dir, with_route=False)          # no ground truth here
        td = prepare_trip(raw, with_labels=False)
        t1 = time.perf_counter()
        proba = self.predict_proba(td.cands)
        t2 = time.perf_counter()
        clf_pred = argmax_pred(td, proba)
        hmm_pred, info = viterbi(td, proba, self.hmm_cfg)
        t3 = time.perf_counter()
        timing = {"features_s": t1 - t0, "classifier_s": t2 - t1, "hmm_s": t3 - t2, "total_s": t3 - t0}
        return td, raw, proba, clf_pred, hmm_pred, dict(info, **timing)


def score_prediction(td, trip_dir: Path | str, clf_pred, hmm_pred) -> dict:
    """Server-side evaluation against the reference route (never sent to the client)."""
    trip_dir = Path(trip_dir)
    rpath = trip_dir / f"{trip_dir.name}.route"
    if not rpath.exists():
        return {}
    route = np.loadtxt(rpath, dtype=np.int64, ndmin=1)
    arcs = np.loadtxt(trip_dir / f"{trip_dir.name}.arcs", dtype=np.int64, ndmin=2)
    attach_labels(td, route, arcs)
    return {"classifier_only": evaluate_trip(td, clf_pred, route),
            "hmm": evaluate_trip(td, hmm_pred, route),
            "candidate_miss_rate": td.info["cand_miss_rate"],
            "frac_reliable_labels": td.info["frac_reliable"]}


def route_geometry(td, pred_loc) -> list[list[float]]:
    """Polyline [lat, lon] following the predicted arcs in order (dedup consecutive)."""
    net = td.net
    seq = [int(a) for a in pred_loc if a >= 0]
    seq = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    pts = []
    for a in seq:
        for node in (int(net.u[a]), int(net.v[a])):
            lon, lat = net.nodes_lonlat[node]
            if not pts or pts[-1] != [lat, lon]:
                pts.append([float(lat), float(lon)])
    return pts
