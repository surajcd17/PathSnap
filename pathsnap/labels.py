"""Observation-level reference labels.

The .route file is an ordered list of arc ids WITHOUT timestamps, so it does not say
which arc each GPS point belongs to. We recover that with a monotone alignment
(dynamic programming, DTW-like): GPS point i is assigned route position j(i) with
j(0) <= j(1) <= ... minimising the sum of point-to-arc distances. Route arcs may be
skipped (short arcs traversed between two samples) and the route may start before /
end after the track.

These labels are only used for training and for SERVER-SIDE scoring; they are never
used as model inputs.
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment

COST_CAP_M = 200.0        # robustness: cap per-point cost so outliers do not dominate
RELIABLE_DIST_M = 50.0    # points aligned farther than this from their arc are 'unreliable'


def align_track_to_route(P: np.ndarray, nodes_xy: np.ndarray, arcs: np.ndarray,
                         route: np.ndarray):
    """Return (label_arc_id per point, route position per point, alignment distance)."""
    A, B = nodes_xy[arcs[route, 0]], nodes_xy[arcs[route, 1]]
    T, R = len(P), len(route)
    cost = np.empty((T, R), dtype=np.float64)
    for j0 in range(0, R, 128):
        j1 = min(R, j0 + 128)
        k = j1 - j0
        p = np.repeat(P, k, axis=0)
        a = np.tile(A[j0:j1], (T, 1))
        b = np.tile(B[j0:j1], (T, 1))
        d, _ = point_segment(p, a, b)
        cost[:, j0:j1] = d.reshape(T, k)
    raw = cost.copy()
    np.minimum(cost, COST_CAP_M, out=cost)

    back = np.empty((T, R), dtype=np.int32)
    idx = np.arange(R, dtype=np.int32)
    prev = cost[0].copy()
    for i in range(1, T):
        pm = np.minimum.accumulate(prev)
        is_new = prev <= pm
        back[i] = np.maximum.accumulate(np.where(is_new, idx, 0))
        prev = cost[i] + pm
    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(T - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    dist = raw[np.arange(T), j]
    return route[j], j, dist
