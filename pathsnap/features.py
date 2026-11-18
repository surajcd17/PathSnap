"""Candidate generation + feature engineering for one trip.

For every GPS observation we generate a set of candidate road segments (arcs) and
describe each (observation, candidate) pair with GPS-only features. A classifier then
scores how likely each candidate is the true segment; the scores feed the HMM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import Projector, angle_diff, bearing, point_segment
from .io import RawTrip, clean_track, load_trip
from .labels import RELIABLE_DIST_M, align_track_to_route
from .network import RoadNetwork

CAND_RADIUS_M = 50.0      # candidate search radius
CAND_FALLBACK_M = 200.0   # widen search if nothing is within CAND_RADIUS_M
CAND_K = 10               # keep at most K nearest candidates per observation
CTX = (1, 2, 3, 5)        # neighbour offsets used for context-distance features

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "head_diff", "head_diff_minus_min", "head_diff_long", "speed", "speed_x_cos",
    "arc_len", "t_proj", "dist_to_arc_end", "has_twin", "out_degree",
    "ctx_dist_1", "ctx_dist_2", "ctx_dist_3", "ctx_dist_5", "ctx_mean",
    "ctx_mean_minus_min", "dt_prev", "dt_next",
]


@dataclass
class TripData:
    trip_id: str
    track: pd.DataFrame            # cleaned track with x, y, speed, heading
    net: RoadNetwork
    cands: pd.DataFrame            # one row per (obs, candidate)
    labels: pd.DataFrame | None    # per-observation reference label (server-side only)
    info: dict = field(default_factory=dict)


def _kinematics(xy: np.ndarray, t: np.ndarray):
    n = len(xy)
    i = np.arange(n)

    def head(k):
        a, b = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
        d = xy[b] - xy[a]
        return bearing(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1]), np.maximum(t[b] - t[a], 1e-6)

    h1, dd, dt = head(1)
    h5, _, _ = head(5)
    speed = dd / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    return h1, h5, speed, dt_prev, dt_next


def generate_candidates(net: RoadNetwork, P: np.ndarray):
    near = net.arcs_near(P, CAND_RADIUS_M)
    rows_i, rows_c = [], []
    need = [k for k, h in enumerate(near) if len(h) == 0]
    if need:
        far = net.arcs_near(P[need], CAND_FALLBACK_M)
        for k, h in zip(need, far):
            near[k] = h
    for i, h in enumerate(near):
        rows_i.append(np.full(len(h), i))
        rows_c.append(h)
    oi = np.concatenate(rows_i) if rows_i else np.empty(0, int)
    ci = np.concatenate(rows_c) if rows_c else np.empty(0, int)
    d, t = point_segment(P[oi], net.A[ci], net.B[ci])
    df = pd.DataFrame({"obs": oi, "loc": ci, "dist": d, "t_proj": t})
    df = df.sort_values(["obs", "dist"], kind="stable")
    df["dist_rank"] = df.groupby("obs").cumcount()
    df = df[df.dist_rank < CAND_K]
    radius_ok = df.dist <= CAND_RADIUS_M
    first = df.dist_rank == 0
    df = df[radius_ok | first].reset_index(drop=True)  # fallback keeps only the nearest
    return df


def build_features(net: RoadNetwork, trk: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    P = trk[["x", "y"]].to_numpy()
    n = len(P)
    oi, ci = df.obs.to_numpy(), df["loc"].to_numpy()
    g = df.groupby("obs")
    df["n_cands"] = g.dist.transform("size")
    dmin = g.dist.transform("min")
    df["dist_minus_min"] = df.dist - dmin
    df["dist_ratio"] = df.dist / (dmin + 1.0)
    hd = angle_diff(trk.heading.to_numpy()[oi], net.bearing[ci])
    df["head_diff"] = hd
    df["head_diff_minus_min"] = hd - df.assign(h=hd).groupby("obs").h.transform("min")
    df["head_diff_long"] = angle_diff(trk.heading_long.to_numpy()[oi], net.bearing[ci])
    sp = trk.speed.to_numpy()[oi]
    df["speed"] = sp
    df["speed_x_cos"] = sp * np.cos(np.radians(hd))  # signed along-arc speed
    df["arc_len"] = net.length[ci]
    df["dist_to_arc_end"] = np.minimum(df.t_proj, 1 - df.t_proj) * net.length[ci]
    df["has_twin"] = net.has_twin[ci].astype(int)
    df["out_degree"] = net.out_degree[ci]
    ctx = []
    for k in CTX:
        dd = []
        for s in (-k, k):
            j = np.clip(oi + s, 0, n - 1)
            d, _ = point_segment(P[j], net.A[ci], net.B[ci])
            dd.append(d)
        v = (dd[0] + dd[1]) / 2
        df[f"ctx_dist_{k}"] = v
        ctx.append(v)
    df["ctx_mean"] = np.mean(ctx, axis=0)
    df["ctx_mean_minus_min"] = df.ctx_mean - df.groupby("obs").ctx_mean.transform("min")
    df["dt_prev"] = trk.dt_prev.to_numpy()[oi]
    df["dt_next"] = trk.dt_next.to_numpy()[oi]
    df["arc_id"] = net.arc_ids[ci]
    return df


def prepare_trip(trip: RawTrip | str | Path, with_labels: bool = True) -> TripData:
    if not isinstance(trip, RawTrip):
        trip = load_trip(trip, with_route=with_labels)
    trk, clean_info = clean_track(trip.track)
    pj = Projector(trk.lon.mean(), trk.lat.mean())
    xy = pj.to_xy(trk.lon, trk.lat)
    trk["x"], trk["y"] = xy[:, 0], xy[:, 1]
    t = trk.t.to_numpy()
    h1, h5, speed, dtp, dtn = _kinematics(xy, t)
    trk["heading"], trk["heading_long"], trk["speed"] = h1, h5, speed
    trk["dt_prev"], trk["dt_next"] = dtp, dtn

    net = RoadNetwork(trip, xy, pj)
    cands = build_features(net, trk, generate_candidates(net, xy))
    info = dict(clean_info, epsg=pj.epsg, n_arcs_cropped=len(net.arc_ids),
                n_arcs_total=net.n_arcs_total)

    cands.insert(0, "trip", trip.trip_id)
    td = TripData(trip.trip_id, trk, net, cands, None, info)
    if with_labels and trip.route is not None:
        attach_labels(td, trip.route, trip.arcs)
    return td


def attach_labels(td: TripData, route: np.ndarray, arcs: np.ndarray) -> None:
    """Attach reference labels (training / server-side scoring only)."""
    xy = td.track[["x", "y"]].to_numpy()
    lab, pos, adist = align_track_to_route(xy, td.net.nodes_xy, arcs, route)
    labels = pd.DataFrame({"label_arc": lab, "route_pos": pos, "align_dist": adist,
                           "reliable": adist <= RELIABLE_DIST_M})
    c = td.cands
    c["y"] = (c.arc_id.to_numpy() == lab[c.obs.to_numpy()]).astype(int)
    c["reliable"] = labels.reliable.to_numpy()[c.obs.to_numpy()]
    hit = c.groupby("obs").y.max().reindex(range(len(td.track)), fill_value=0).to_numpy()
    labels["in_cands"] = hit.astype(bool)
    td.labels = labels
    td.info["frac_reliable"] = float(labels.reliable.mean())
    td.info["cand_miss_rate"] = float(1 - labels.in_cands[labels.reliable].mean())
