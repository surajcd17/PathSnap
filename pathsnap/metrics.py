"""Evaluation metrics shared by experiments and the API.

All observation-level metrics use only 'reliable' observations (aligned within
RELIABLE_DIST_M of the reference route), see labels.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

CONT_SLACK_M = 100.0


def argmax_pred(td, proba: np.ndarray) -> np.ndarray:
    """Classifier-only decoding: highest-scoring candidate per observation
    (local arc index, -1 if the observation has no candidates)."""
    c = td.cands.assign(p=proba)
    best = c.loc[c.groupby("obs").p.idxmax()]
    pred = np.full(len(td.track), -1, dtype=np.int64)
    pred[best.obs.to_numpy()] = best["loc"].to_numpy()
    return pred


def candidate_metrics(y, proba, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    out = {
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "accuracy": float((pred == y).mean()),
    }
    p = np.clip(np.asarray(proba, float), 1e-7, 1 - 1e-7)
    out["log_loss"] = log_loss(y, p, labels=[0, 1])
    return out


def top1_from_scores(cands, scores, labels):
    """Observation-level Top-1 over reliable observations of possibly many trips.

    cands needs columns trip, obs, y; labels needs trip, obs, reliable.
    Observations with no correct candidate (candidate miss) count as errors."""
    c = cands[["trip", "obs", "y"]].assign(p=np.asarray(scores))
    best = c.loc[c.groupby(["trip", "obs"], sort=False).p.idxmax(), ["trip", "obs", "y"]]
    rel = labels.loc[labels.reliable, ["trip", "obs"]]
    m = rel.merge(best, on=["trip", "obs"], how="left").fillna({"y": 0})
    return float(m.y.mean())


def continuity(td, pred_loc):
    """Fraction of consecutive predicted segments that are network-connected:
    same arc, or end of a reaches start of b within d_gc + CONT_SLACK_M along arcs."""
    net = td.net
    xy = td.track[["x", "y"]].to_numpy()
    idx = np.flatnonzero(pred_loc >= 0)
    ok = 0
    for i0, i1 in zip(idx[:-1], idx[1:]):
        a, b = pred_loc[i0], pred_loc[i1]
        if a == b:
            ok += 1
            continue
        gc = float(np.hypot(*(xy[i1] - xy[i0])))
        if int(net.u[b]) in net.node_dist(int(net.v[a]), gc + CONT_SLACK_M):
            ok += 1
    return ok / max(len(idx) - 1, 1)


def evaluate_trip(td, pred_loc, route=None):
    """Metrics for one trip given predicted local arc index per observation."""
    lab = td.labels
    net = td.net
    pred_arc = np.where(pred_loc >= 0, net.arc_ids[np.maximum(pred_loc, 0)], -1)
    rel = lab.reliable.to_numpy()
    correct = pred_arc == lab.label_arc.to_numpy()
    out = {
        "n_obs": int(len(lab)),
        "n_reliable": int(rel.sum()),
        "top1_acc": float(correct[rel].mean()) if rel.any() else float("nan"),
        "continuity": continuity(td, pred_loc),
    }
    if route is not None:
        P = set(pred_arc[pred_arc >= 0].tolist())
        R = set(np.asarray(route).tolist())
        tp = len(P & R)
        pr = tp / max(len(P), 1)
        rc = tp / max(len(R), 1)
        out.update(route_precision=pr, route_recall=rc,
                   route_f1=2 * pr * rc / max(pr + rc, 1e-9))
    # candidate-level F1 when the decision is "this candidate is chosen"
    c = td.cands
    chosen = (c["loc"].to_numpy() == pred_loc[c.obs.to_numpy()]).astype(int)
    r = c.reliable.to_numpy()
    out["macro_f1"] = float(f1_score(c.y.to_numpy()[r], chosen[r], average="macro"))
    return out
