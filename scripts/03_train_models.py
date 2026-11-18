"""Train and compare the candidate classifiers.

Protocol
  * 5-fold GroupKFold over the 80 training trips (a trip never spans folds).
  * Only observations with reliable reference labels are used for fitting/scoring.
  * Student-test (10 trips) is used only for development-stage comparison.
  * Classifier-only route continuity is measured on student-test (argmax candidate).

Outputs: results/model_comparison.csv, results/cv_folds.csv, results/figures/*.png,
         artifacts/oof/*.npy, artifacts/models_untuned/*.joblib
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES, prepare_trip

from pathsnap.io import data_root, list_trips, load_trip
from pathsnap.metrics import argmax_pred, candidate_metrics, continuity, top1_from_scores
from pathsnap.models import MODEL_SPECS, SEED

RES = ROOT / "results"
FIG = RES / "figures"
OOF = ROOT / "artifacts" / "oof"
MOD = ROOT / "artifacts" / "models_untuned"
for d in (FIG, OOF, MOD):
    d.mkdir(parents=True, exist_ok=True)

ONLY = sys.argv[1:]  # optionally restrict to some model names


def subsample(df, n):
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=SEED)


def main():
    tr = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    tr_lab = pd.read_pickle(ROOT / "artifacts/dataset/train_labels.pkl")
    tr = tr[tr.reliable].reset_index(drop=True)
    groups = tr.trip.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(tr, tr.y, groups))

    # student-test trip objects (for continuity) + table
    st_tds = {t: prepare_trip(data_root() / "student_test" / t) for t in list_trips("student_test")}
    st = pd.concat([td.cands for td in st_tds.values()], ignore_index=True)
    st_lab = pd.concat([td.labels.assign(trip=t, obs=np.arange(len(td.labels))) for t, td in st_tds.items()],
                       ignore_index=True)
    st_rel = st.reliable.to_numpy()

    rows, fold_rows = [], []
    # ---------------- distance baseline (no learning) for reference
    base_scores = -st.dist.to_numpy()
    rows.append(dict(model="NearestSegment (baseline)", family="Heuristic", config="pick nearest candidate",
                     cv_macro_f1=np.nan, cv_top1=float(np.mean([
                         top1_from_scores(tr.iloc[va], -tr.dist.to_numpy()[va], tr_lab[tr_lab.trip.isin(tr.trip.iloc[va].unique())])
                         for _, va in folds])),
                     st_top1=top1_from_scores(st, base_scores, st_lab),
                     st_continuity=np.mean([continuity(td, argmax_pred(td, -td.cands.dist.to_numpy())) for td in st_tds.values()])))

    for name, spec in MODEL_SPECS.items():
        if ONLY and name not in ONLY:
            continue
        print(f"=== {name}")
        oof = np.zeros(len(tr))
        fts = []
        for k, (fi, va) in enumerate(folds):
            fit_df = subsample(tr.iloc[fi], spec["fit_rows"])
            m = spec["make"]()
            t0 = time.time()
            m.fit(fit_df[FEATURES], fit_df.y)
            fts.append(time.time() - t0)
            p = m.predict_proba(tr.iloc[va][FEATURES])[:, 1]
            oof[va] = p
            cm = candidate_metrics(tr.y.iloc[va], p)
            va_trips = tr.trip.iloc[va].unique()
            t1 = top1_from_scores(tr.iloc[va], p, tr_lab[tr_lab.trip.isin(va_trips)])
            fold_rows.append(dict(model=name, fold=k, macro_f1=cm["macro_f1"], top1=t1, log_loss=cm["log_loss"],
                                  n_val_trips=len(va_trips)))
            print(f"  fold {k}: F1={cm['macro_f1']:.4f} top1={t1:.4f} ll={cm['log_loss']:.4f} fit={fts[-1]:.1f}s")
        np.save(OOF / f"{name}.npy", oof)

        # ---- refit on all training trips, evaluate on student-test
        fit_df = subsample(tr, spec["fit_rows"])
        m = spec["make"]()
        t0 = time.time()
        m.fit(fit_df[FEATURES], fit_df.y)
        train_s = time.time() - t0
        joblib.dump(m, MOD / f"{name}.joblib")
        t0 = time.time()
        p_st = m.predict_proba(st[FEATURES])[:, 1]
        inf_s = time.time() - t0
        np.save(OOF / f"{name}__student_test.npy", p_st)
        p_tr = m.predict_proba(fit_df[FEATURES])[:, 1]
        cm_tr = candidate_metrics(fit_df.y, p_tr)
        cm_st = candidate_metrics(st.y.to_numpy()[st_rel], p_st[st_rel])
        top1 = top1_from_scores(st, p_st, st_lab)
        cont, off = [], 0
        for t, td in st_tds.items():
            n = len(td.cands)
            cont.append(continuity(td, argmax_pred(td, p_st[off:off + n])))
            off += n
        f = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        rows.append(dict(
            model=name, family=spec["family"], config=spec["config"],
            cv_macro_f1=f.macro_f1.mean(), cv_macro_f1_std=f.macro_f1.std(),
            cv_top1=f.top1.mean(), cv_top1_std=f.top1.std(), cv_log_loss=f.log_loss.mean(),
            train_macro_f1=cm_tr["macro_f1"], train_log_loss=cm_tr["log_loss"],
            st_top1=top1, st_macro_f1=cm_st["macro_f1"], st_precision=cm_st["precision_pos"],
            st_recall=cm_st["recall_pos"], st_accuracy=cm_st["accuracy"], st_log_loss=cm_st["log_loss"],
            st_continuity=float(np.mean(cont)),
            train_time_s=train_s, mean_fold_fit_s=float(np.mean(fts)),
            inference_ms_per_1k_cands=1000 * inf_s / len(st) * 1000,
            inference_s_student_test=inf_s, fit_rows=len(fit_df)))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})

        y_pred = (p_st[st_rel] >= 0.5).astype(int)
        disp = ConfusionMatrixDisplay(confusion_matrix(st.y.to_numpy()[st_rel], y_pred),
                                      display_labels=["wrong seg (0)", "true seg (1)"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} - student-test candidate confusion matrix")
        plt.tight_layout()
        plt.savefig(FIG / f"confusion_{name}.png", dpi=130)
        plt.close("all")

    out = RES / ("model_comparison.csv" if not ONLY else f"model_comparison_{'_'.join(ONLY)}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(fold_rows).to_csv(RES / ("cv_folds.csv" if not ONLY else f"cv_folds_{'_'.join(ONLY)}.csv"), index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
