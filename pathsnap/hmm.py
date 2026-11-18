"""HMM + Viterbi route reconstruction.

Hidden states   : at observation i, the candidate road segments (arcs) of i.
Emission        : classifier probability of the candidate, normalised over the
                  candidates of i, raised to a weight w_e (log-space: w_e * log p).
Transition      : Newson & Krumm style. For candidate a at i and b at i+1 we compute the
                  route distance d_route (along directed arcs: rest of a + shortest path
                  v_a -> u_b + part of b) and the GPS displacement d_gc.
                  log P(b|a) = -|d_route - d_gc| / (beta * sqrt(max(dt, 1))).
                  Unreachable pairs get a large finite penalty (LOG_FLOOR) rather than
                  -inf, so a disconnected step does not kill the whole sequence: if every
                  pair is unreachable the decoder effectively restarts from emissions.
Gaps            : beta grows with sqrt(dt); search limit grows with d_gc.
Missing cands   : observations without candidates are skipped and linked across.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import argmax_pred  # noqa: F401  (classifier-only decoding, re-exported)

LOG_FLOOR = -200.0


@dataclass
class HMMConfig:
    beta: float = 10.0            # transition scale (m)
    w_emission: float = 1.0       # weight of log emission
    back_tol: float = 15.0        # metres of backward movement on same arc treated as noise
    limit_extra: float = 300.0    # shortest-path search limit beyond d_gc (m)
    eps: float = 1e-6


def _route_dist(net, a, ta, b, tb, gc, cfg: HMMConfig):
    la, lb = net.length[a], net.length[b]
    pa, pb = ta * la, tb * lb
    if a == b and pb >= pa - cfg.back_tol:
        return abs(pb - pa)
    dist = net.node_dist(int(net.v[a]), gc + cfg.limit_extra)
    sp = dist.get(int(net.u[b]))
    if sp is None:
        return None
    return (la - pa) + sp + pb


def viterbi(td, proba: np.ndarray, cfg: HMMConfig = HMMConfig()):
    """Decode the most likely candidate sequence.

    proba: classifier probability per row of td.cands (same order).
    Returns (pred_loc per observation (-1 if none), info dict).
    """
    c = td.cands
    obs = c.obs.to_numpy()
    loc = c["loc"].to_numpy()
    tp = c.t_proj.to_numpy()
    n = len(td.track)
    starts = np.searchsorted(obs, np.arange(n + 1))  # cands are sorted by obs
    xy = td.track[["x", "y"]].to_numpy()
    tt = td.track.t.to_numpy()
    net = td.net

    steps = [i for i in range(n) if starts[i + 1] > starts[i]]
    pred = np.full(n, -1, dtype=np.int64)
    if not steps:
        return pred, {"n_breaks": 0, "n_steps": 0}

    def emis(i):
        s, e = starts[i], starts[i + 1]
        p = proba[s:e] + cfg.eps
        return cfg.w_emission * np.log(p / p.sum())

    score = emis(steps[0])
    backs = []
    n_breaks = 0
    for k in range(1, len(steps)):
        i0, i1 = steps[k - 1], steps[k]
        s0, e0, s1, e1 = starts[i0], starts[i0 + 1], starts[i1], starts[i1 + 1]
        gc = float(np.hypot(*(xy[i1] - xy[i0])))
        dt = max(tt[i1] - tt[i0], 1.0)
        beta = cfg.beta * np.sqrt(dt)
        T = np.full((e0 - s0, e1 - s1), LOG_FLOOR)
        for ia in range(e0 - s0):
            a, ta = loc[s0 + ia], tp[s0 + ia]
            for ib in range(e1 - s1):
                d = _route_dist(net, a, ta, loc[s1 + ib], tp[s1 + ib], gc, cfg)
                if d is not None:
                    T[ia, ib] = -abs(d - gc) / beta
        if T.max() <= LOG_FLOOR:
            n_breaks += 1
        tot = score[:, None] + T
        bp = tot.argmax(0)
        backs.append(bp)
        score = tot[bp, np.arange(e1 - s1)] + emis(i1)
    # backtrack
    j = int(score.argmax())
    pred[steps[-1]] = loc[starts[steps[-1]] + j]
    for k in range(len(steps) - 1, 0, -1):
        j = int(backs[k - 1][j])
        pred[steps[k - 1]] = loc[starts[steps[k - 1]] + j]
    return pred, {"n_breaks": n_breaks, "n_steps": len(steps)}
