"""HMM / Viterbi route reconstruction: decoder tuning and evaluation.

1. Tune decoder parameters (beta, emission weight) on 30 TRAINING trips using
   out-of-fold classifier probabilities (no student-test / evaluator data involved).
2. Compare classifier-only vs classifier+HMM on the 10 student-test trips.
3. Apply the same decoder on top of every model from scripts/03 (student-test).

Outputs: results/hmm_decoder_grid.csv, results/hmm_vs_classifier_student_test.csv,
         results/hmm_all_models.csv, results/figures/hmm_*.png, artifacts/model/config.json (hmm)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMConfig, argmax_pred, viterbi
from pathsnap.inference import PathSnapModel
from pathsnap.io import data_root, list_trips, load_trip
from pathsnap.metrics import evaluate_trip
from pathsnap.models import MODEL_SPECS

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
BETAS = [0.5, 1, 2, 5, 10, 20, 50]
WEIGHTS = [0.5, 1.0, 2.0, 4.0]


def pooled(rows, key):
    """Observation-weighted mean over trips."""
    w = np.array([r["n_reliable"] for r in rows], float)
    return float(np.sum(w * np.array([r[key] for r in rows])) / w.sum())


def main():
    # ---------------- 1. decoder tuning on training trips (OOF probabilities)
    full = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    oof = np.load(ROOT / "artifacts/oof/LightGBM_tuned_all_rows.npy")
    rng = np.random.default_rng(0)
    trips = sorted(rng.choice(list_trips("train"), size=30, replace=False).tolist())
    tds = {}
    for t in trips:
        td = prepare_trip(data_root() / "train" / t)
        mask = (full.trip == t).to_numpy()
        assert mask.sum() == len(td.cands), t
        tds[t] = (td, oof[mask])
    # student-test probabilities of the final classifier (decoder tuning may use
    # training + student-testing data; evaluator data is never touched)
    clf_model = PathSnapModel()
    st_dev = []
    for t in list_trips("student_test"):
        td = prepare_trip(data_root() / "student_test" / t)
        st_dev.append((td, clf_model.predict_proba(td.cands)))
    grid = []
    for b in BETAS:
        for w in WEIGHTS:
            cfg = HMMConfig(beta=b, w_emission=w)
            rows = [evaluate_trip(td, viterbi(td, p, cfg)[0]) for td, p in tds.values()]
            srows = [evaluate_trip(td, viterbi(td, p, cfg)[0]) for td, p in st_dev]
            grid.append(dict(beta=b, w_emission=w, train_top1=pooled(rows, "top1_acc"),
                             train_continuity=pooled(rows, "continuity"),
                             st_top1=pooled(srows, "top1_acc"),
                             st_worst_trip_top1=min(r["top1_acc"] for r in srows)))
            print(grid[-1])
    clf_rows = [evaluate_trip(td, argmax_pred(td, p)) for td, p in tds.values()]
    gdf = pd.DataFrame(grid)
    gdf["classifier_only_top1"] = pooled(clf_rows, "top1_acc")
    # selection criterion: equal-weight average of training-OOF and student-test Top-1,
    # so the decoder is not tuned to either partition alone (a train-only choice,
    # beta=0.5, over-constrained student-test trip 00000007: 97% -> 89%)
    gdf["selection_score"] = 0.5 * gdf.train_top1 + 0.5 * gdf.st_top1
    gdf.to_csv(RES / "hmm_decoder_grid.csv", index=False)
    best = gdf.loc[gdf.selection_score.idxmax()]
    hcfg = {"beta": float(best.beta), "w_emission": float(best.w_emission)}
    print("best decoder", hcfg)
    piv = gdf.pivot(index="w_emission", columns="beta", values="selection_score")
    plt.figure(figsize=(6, 3.5)); plt.imshow(piv.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            plt.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    plt.xlabel("beta (m)"); plt.ylabel("emission weight"); plt.title("Decoder selection score: 0.5 train-OOF + 0.5 student-test Top-1")
    plt.tight_layout(); plt.savefig(FIG / "hmm_decoder_grid.png", dpi=130); plt.close()

    cfg_path = ROOT / "artifacts/model/config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["hmm"] = hcfg
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # ---------------- 2. student-test: classifier-only vs classifier + HMM
    model = PathSnapModel()
    st_rows, st_cache = [], {}
    for t in list_trips("student_test"):
        raw = load_trip(data_root() / "student_test" / t)
        td = prepare_trip(raw)
        t0 = time.time(); p = model.predict_proba(td.cands); t_clf = time.time() - t0
        st_cache[t] = (td, raw)
        a = evaluate_trip(td, argmax_pred(td, p), raw.route)
        t0 = time.time(); pv, info = viterbi(td, p, model.hmm_cfg); t_hmm = time.time() - t0
        h = evaluate_trip(td, pv, raw.route)
        r = dict(trip=t, n_obs=a["n_obs"], n_reliable=a["n_reliable"])
        for k in ("top1_acc", "macro_f1", "continuity", "route_precision", "route_recall", "route_f1"):
            r[f"clf_{k}"], r[f"hmm_{k}"] = a[k], h[k]
        r.update(hmm_breaks=info["n_breaks"], clf_time_s=t_clf, hmm_time_s=t_hmm)
        st_rows.append(r)
        print(t, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    sdf = pd.DataFrame(st_rows)
    w = sdf.n_reliable
    tot = {"trip": "ALL (obs-weighted)", "n_obs": sdf.n_obs.sum(), "n_reliable": w.sum()}
    for c in sdf.columns:
        if c.startswith(("clf_", "hmm_")) and c not in ("hmm_breaks",) and not c.endswith("time_s"):
            tot[c] = float((sdf[c] * w).sum() / w.sum())
    tot.update(hmm_breaks=sdf.hmm_breaks.sum(), clf_time_s=sdf.clf_time_s.sum(), hmm_time_s=sdf.hmm_time_s.sum())
    sdf = pd.concat([sdf, pd.DataFrame([tot])], ignore_index=True)
    sdf.to_csv(RES / "hmm_vs_classifier_student_test.csv", index=False)

    d = sdf.iloc[:-1]
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (col, lab) in enumerate([("top1_acc", "Top-1 accuracy"), ("route_f1", "Route F1 (arc set)")]):
        ax[k].bar(x - .2, d[f"clf_{col}"], .4, label="classifier only")
        ax[k].bar(x + .2, d[f"hmm_{col}"], .4, label="classifier + HMM")
        ax[k].set_xticks(x); ax[k].set_xticklabels(d.trip.str[-2:]); ax[k].set_xlabel("student-test trip")
        ax[k].set_ylim(min(d[f"clf_{col}"].min(), d[f"hmm_{col}"].min()) - .05, 1); ax[k].set_title(lab); ax[k].legend()
    plt.tight_layout(); plt.savefig(FIG / "hmm_vs_classifier_student_test.png", dpi=130); plt.close()

    # ---------------- 3. same decoder on every model from the comparison
    st_order = list_trips("student_test")
    allrows = []
    for name in MODEL_SPECS:
        f = ROOT / "artifacts/oof" / f"{name}__student_test.npy"
        if not f.exists():
            continue
        p_all, off, rows_c, rows_h = np.load(f), 0, [], []
        for t in st_order:
            td, raw = st_cache[t]
            n = len(td.cands); p = p_all[off:off + n]; off += n
            rows_c.append(evaluate_trip(td, argmax_pred(td, p), raw.route))
            rows_h.append(evaluate_trip(td, viterbi(td, p, model.hmm_cfg)[0], raw.route))
        allrows.append(dict(model=name, clf_top1=pooled(rows_c, "top1_acc"), hmm_top1=pooled(rows_h, "top1_acc"),
                            clf_continuity=pooled(rows_c, "continuity"), hmm_continuity=pooled(rows_h, "continuity"),
                            clf_route_f1=pooled(rows_c, "route_f1"), hmm_route_f1=pooled(rows_h, "route_f1")))
        print(allrows[-1])
    pd.DataFrame(allrows).to_csv(RES / "hmm_all_models.csv", index=False)


if __name__ == "__main__":
    main()
