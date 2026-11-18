"""Feature-engineering evidence (stage 3, model-free).

* Univariate separability: ROC AUC of each single feature for true vs wrong candidates
  (direction chosen so AUC >= 0.5), plus true/wrong medians. No model is trained.
* Candidate statistics: candidates per fix, positive rate, miss rate, nearest-is-true.
* Distributions: true vs wrong candidates for the key features.
* Illustrations: candidates of one GPS observation; GPS-to-route label alignment.
Outputs: results/feature_univariate.csv, results/candidate_stats.csv, results/figures/feat_*.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathsnap.features import FEATURES, prepare_trip
from pathsnap.io import data_root

RES, FIG = ROOT / "results", ROOT / "results" / "figures"

GROUPS = {
    "distance": ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands"],
    "heading/speed": ["head_diff", "head_diff_minus_min", "head_diff_long", "speed", "speed_x_cos"],
    "arc geometry/topology": ["arc_len", "t_proj", "dist_to_arc_end", "has_twin", "out_degree"],
    "temporal context": ["ctx_dist_1", "ctx_dist_2", "ctx_dist_3", "ctx_dist_5", "ctx_mean",
                         "ctx_mean_minus_min", "dt_prev", "dt_next"],
}


def univariate(tr):
    rows = []
    for g, feats in GROUPS.items():
        for f in feats:
            auc = roc_auc_score(tr.y, tr[f])
            rows.append(dict(group=g, feature=f, auc=max(auc, 1 - auc),
                             higher_means_true=auc >= 0.5,
                             median_true=tr.loc[tr.y == 1, f].median(), median_wrong=tr.loc[tr.y == 0, f].median()))
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df.to_csv(RES / "feature_univariate.csv", index=False)
    print(df.round(3).to_string(index=False))
    plt.figure(figsize=(7, 6))
    d = df.sort_values("auc")
    cols = {g: f"C{i}" for i, g in enumerate(GROUPS)}
    plt.barh(d.feature, d.auc - 0.5, left=0.5, color=[cols[g] for g in d.group])
    from matplotlib.patches import Patch
    plt.axvline(0.5, color="k", lw=.8); plt.xlim(0.45, 1)
    plt.legend(handles=[Patch(color=c, label=g) for g, c in cols.items()], fontsize=8, loc="lower right")
    plt.xlabel("single-feature ROC AUC (0.5 = no information)")
    plt.title("Univariate separability of each feature (train, reliable fixes)")
    plt.tight_layout(); plt.savefig(FIG / "feat_univariate_auc.png", dpi=130); plt.close()


def candidate_stats():
    rows = []
    for part in ("train", "student_test"):
        c = pd.read_pickle(ROOT / f"artifacts/dataset/{part}_cands.pkl")
        lab = pd.read_pickle(ROOT / f"artifacts/dataset/{part}_labels.pkl")
        rel = c[c.reliable]
        per = rel.groupby(["trip", "obs"]).size()
        nearest = rel[rel.dist_rank == 0].groupby(["trip", "obs"]).y.max()
        rows.append(dict(partition=part, n_obs=len(lab), n_rows=len(c), n_rows_reliable=len(rel),
                         frac_reliable_obs=lab.reliable.mean(), cands_per_obs_mean=per.mean(),
                         cands_per_obs_median=per.median(), positive_rate=rel.y.mean(),
                         candidate_miss_rate=1 - lab.in_cands[lab.reliable].mean(),
                         nearest_is_true=nearest.mean()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "candidate_stats.csv", index=False)
    print(df.round(4).to_string(index=False))


def distributions(tr):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (k, lab, hi) in zip(ax, [("dist", "distance GPS -> segment (m)", 50),
                                    ("head_diff_long", "heading difference (deg)", 180),
                                    ("ctx_mean", "mean distance of neighbouring fixes (m)", 80)]):
        bins = np.linspace(0, hi, 50)
        a.hist(tr.loc[tr.y == 0, k].clip(0, hi), bins=bins, density=True, alpha=.6, label="wrong candidate")
        a.hist(tr.loc[tr.y == 1, k].clip(0, hi), bins=bins, density=True, alpha=.6, label="true segment")
        a.set_xlabel(lab); a.legend()
    plt.suptitle("Feature distributions: true vs wrong candidates (train, reliable obs)")
    plt.tight_layout(); plt.savefig(FIG / "feat_distributions.png", dpi=130); plt.close()


def illustrations():
    td = prepare_trip(data_root() / "train" / "00000003")
    net, trk, c = td.net, td.track, td.cands
    xy = trk[["x", "y"]].to_numpy()
    # pick an observation with many candidates incl. a reverse twin
    cnt = c.groupby("obs").size()
    i = int(cnt[cnt >= 8].index[len(cnt[cnt >= 8]) // 2])
    ci = c[c.obs == i]
    x0, y0 = xy[i]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    a = ax[0]
    near = np.unique(np.concatenate(net.arcs_near(xy[i:i + 1], 120)))
    for k in near:
        a.plot([net.A[k, 0] - x0, net.B[k, 0] - x0], [net.A[k, 1] - y0, net.B[k, 1] - y0], color="0.8", lw=1)
    for _, r in ci.iterrows():
        k = int(r["loc"])
        col = "g" if r.y == 1 else "tab:orange"
        a.annotate("", xy=(net.B[k, 0] - x0, net.B[k, 1] - y0), xytext=(net.A[k, 0] - x0, net.A[k, 1] - y0),
                   arrowprops=dict(arrowstyle="->", color=col, lw=2.5 if r.y == 1 else 1.4))
    a.plot(xy[max(i - 8, 0):i + 9, 0] - x0, xy[max(i - 8, 0):i + 9, 1] - y0, ".", color="b", ms=5, label="neighbouring fixes")
    a.plot(0, 0, "o", color="b", ms=10, label=f"GPS fix #{i}")
    a.add_patch(plt.Circle((0, 0), 50, fill=False, ls="--", color="b"))
    a.plot([], [], color="g", lw=2.5, label="true segment (label)"); a.plot([], [], color="tab:orange", label="other candidates")
    a.set_xlim(-120, 120); a.set_ylim(-120, 120); a.set_aspect("equal"); a.legend(fontsize=8, loc="lower left")
    a.set_title(f"Candidates of one observation ({len(ci)} arcs within 50 m, arrows = direction)")
    a.set_xlabel("m"); a.set_ylabel("m")

    a = ax[1]
    lab = td.labels
    s = slice(200, 260)
    route_arcs = np.unique(lab.label_arc.to_numpy()[s])
    nodes = net.nodes_xy
    raw_arcs = np.loadtxt(data_root() / "train/00000003/00000003.arcs", dtype=int)
    for k in route_arcs:
        u, v = raw_arcs[k]
        a.plot(*np.c_[nodes[u], nodes[v]] - np.array([[x0], [y0]]) * 0 - xy[s][0][:, None], color="g", lw=3)
    seg = xy[s] - xy[s][0]
    a.plot(seg[:, 0], seg[:, 1], ".", color="b", ms=6, label="GPS fixes 200-259")
    from pathsnap.geo import point_segment
    la = lab.label_arc.to_numpy()[s]
    A, B = nodes[raw_arcs[la, 0]] - xy[s][0], nodes[raw_arcs[la, 1]] - xy[s][0]
    _, t = point_segment(seg, A, B)
    proj = A + (B - A) * t[:, None]
    for p, q in zip(seg, proj):
        a.plot([p[0], q[0]], [p[1], q[1]], color="r", lw=.8)
    a.plot([], [], color="g", lw=3, label="aligned reference arcs"); a.plot([], [], color="r", label="point -> aligned arc")
    a.set_aspect("equal"); a.legend(fontsize=8); a.set_title("Label alignment: each fix assigned to a route arc (monotone DP)")
    a.set_xlabel("m"); a.set_ylabel("m")
    plt.tight_layout(); plt.savefig(FIG / "feat_candidates_and_alignment.png", dpi=130); plt.close()


if __name__ == "__main__":
    tr = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    tr = tr[tr.reliable].reset_index(drop=True)
    candidate_stats()
    univariate(tr)
    distributions(tr)
    illustrations()
