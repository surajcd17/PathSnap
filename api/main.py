"""PathSnap FastAPI backend (Laptop 1).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_EVAL_DIR   folder that contains the evaluation trip folders
                        (default: data/Student folder/student_test)
    PATHSNAP_MODEL_DIR  folder with model.joblib + config.json (default: artifacts/model)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.inference import MODEL_DIR, PathSnapModel, route_geometry, score_prediction

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "Student folder" / "student_test"))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL_DIR", MODEL_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = PathSnapModel(MODEL_PATH)
log.info("Loaded model '%s' from %s; serving trips from %s", model.meta.get("model_name"), MODEL_PATH, EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dirs():
    if not EVAL_DIR.exists():
        return {}
    return {p.name: p for p in sorted(EVAL_DIR.iterdir()) if p.is_dir() and (p / f"{p.name}.track").exists()}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path, client,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": model.meta.get("model_name"), "n_trips": len(_trip_dirs())}


@app.get("/trips")
def trips():
    out = []
    for tid, p in _trip_dirs().items():
        with open(p / f"{tid}.track") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": tid, "n_points": n})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    dirs = _trip_dirs()
    if req.trip_id not in dirs:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    trip_dir = dirs[req.trip_id]
    log.info("[PREDICT] trip=%s received, running pipeline", req.trip_id)
    td, raw, proba, clf_pred, hmm_pred, info = model.predict_trip(trip_dir)
    # ---- server-side scoring (reference route stays on this machine)
    metrics = score_prediction(td, trip_dir, clf_pred, hmm_pred)
    h = metrics.get("hmm", {})
    c = metrics.get("classifier_only", {})
    log.info("[RESULT] trip=%s obs=%d | classifier top1=%.4f | HMM top1=%.4f macroF1=%.4f "
             "continuity=%.4f routeF1=%.4f | %.2fs",
             req.trip_id, len(td.track), c.get("top1_acc", np.nan), h.get("top1_acc", np.nan),
             h.get("macro_f1", np.nan), h.get("continuity", np.nan), h.get("route_f1", np.nan),
             info["total_s"])

    trk = td.track
    pred_arc = np.where(hmm_pred >= 0, td.net.arc_ids[np.maximum(hmm_pred, 0)], -1)
    return {
        "trip_id": req.trip_id,
        "model": model.meta.get("model_name"),
        "n_observations": int(len(trk)),
        "gps": trk[["lat", "lon"]].round(7).values.tolist(),
        "predicted_route": route_geometry(td, hmm_pred),
        "classifier_route": route_geometry(td, clf_pred),
        "predicted_arc_per_observation": pred_arc.tolist(),
        "metrics": metrics,
        "timing": {k: round(v, 3) for k, v in info.items() if k.endswith("_s")},
        "decoder": {"n_steps": info["n_steps"], "n_breaks": info["n_breaks"]},
    }
