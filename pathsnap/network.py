"""Road network for one trip: projection, corridor cropping, spatial index and
bounded shortest paths on the directed arc graph."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import Projector, bearing
from .io import RawTrip

SAMPLE_STEP = 10.0  # metres between densified sample points along an arc


class RoadNetwork:
    def __init__(self, trip: RawTrip, track_xy: np.ndarray, projector: Projector,
                 corridor_m: float = 400.0):
        self.projector = projector
        nodes_xy = projector.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
        arcs = trip.arcs
        a_all, b_all = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]

        # --- crop: keep arcs whose geometry passes within `corridor_m` of the track
        lo, hi = track_xy.min(0) - corridor_m, track_xy.max(0) + corridor_m
        bbox_ok = ((np.minimum(a_all[:, 0], b_all[:, 0]) <= hi[0]) & (np.maximum(a_all[:, 0], b_all[:, 0]) >= lo[0]) &
                   (np.minimum(a_all[:, 1], b_all[:, 1]) <= hi[1]) & (np.maximum(a_all[:, 1], b_all[:, 1]) >= lo[1]))
        cand = np.flatnonzero(bbox_ok)
        s_arc, s_xy = self._densify(a_all[cand], b_all[cand], np.arange(len(cand)))
        d, _ = cKDTree(track_xy).query(s_xy, distance_upper_bound=corridor_m + SAMPLE_STEP)
        keep_local = np.unique(s_arc[np.isfinite(d)])
        self.arc_ids = cand[keep_local]                 # original arc ids kept
        self.n_arcs_total = len(arcs)
        self.n_nodes_total = len(trip.nodes)

        self.u = arcs[self.arc_ids, 0]
        self.v = arcs[self.arc_ids, 1]
        self.A = a_all[self.arc_ids]
        self.B = b_all[self.arc_ids]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing(d[:, 0], d[:, 1])
        self.nodes_xy = nodes_xy
        self.nodes_lonlat = trip.nodes
        self.local_of = {int(a): i for i, a in enumerate(self.arc_ids)}

        # spatial index over densified samples
        self.s_arc, s_xy = self._densify(self.A, self.B, np.arange(len(self.arc_ids)))
        self.tree = cKDTree(s_xy)

        # adjacency (directed): node -> list of (local arc idx) leaving it
        self.out_arcs: dict[int, list[int]] = {}
        for i, uu in enumerate(self.u.tolist()):
            self.out_arcs.setdefault(uu, []).append(i)
        self.out_degree = np.array([len(self.out_arcs.get(int(x), ())) for x in self.v])  # at arc end
        cnt = {}
        for vv in self.v.tolist():
            cnt[vv] = cnt.get(vv, 0) + 1
        self.in_degree_start = np.array([cnt.get(int(x), 0) for x in self.u])
        pairs = set(zip(self.u.tolist(), self.v.tolist()))
        self.has_twin = np.array([(vv, uu) in pairs for uu, vv in zip(self.u.tolist(), self.v.tolist())])
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    @staticmethod
    def _densify(A, B, ids):
        L = np.hypot(*(B - A).T)
        n = np.maximum(np.ceil(L / SAMPLE_STEP).astype(int), 1) + 1
        rep = np.repeat(np.arange(len(A)), n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        t = offs / np.repeat(n - 1, n)
        xy = A[rep] + (B[rep] - A[rep]) * t[:, None]
        return ids[rep], xy

    # ------------------------------------------------------------------ queries
    def arcs_near(self, xy: np.ndarray, radius: float) -> list[np.ndarray]:
        """Local arc indices whose densified samples lie within radius (+ step/2)."""
        hits = self.tree.query_ball_point(xy, r=radius + SAMPLE_STEP / 2)
        return [np.unique(self.s_arc[h]) if len(h) else np.empty(0, int) for h in hits]

    def node_dist(self, src: int, limit: float) -> dict:
        """Bounded Dijkstra from node `src` over directed arcs. Cached per source."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for ai in self.out_arcs.get(n, ()):
                nd = d + self.length[ai]
                m = int(self.v[ai])
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(pq, (nd, m))
        self._sp_cache[src] = (limit, dist)
        return dist
