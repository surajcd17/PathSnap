"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (LightGBM).

Objective : mean Macro-F1 over a 3-fold GroupKFold of the 80 training trips (grouped by
            trip, reliable observations only). 3 folds inside the search keeps each trial
            cheap; the final tuned-vs-untuned comparison uses the same 5-fold split as
            scripts/03_train_models.py.
Outputs   : results/optuna_trials.csv, results/tuned_vs_untuned.csv,
            results/figures/optuna_*.png, artifacts/model/{model.joblib,config.json},
            artifacts/oof/LightGBM_tuned_all_rows.npy (out-of-fold, used for decoder tuning)
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
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES, prepare_trip

from pathsnap.io import data_root, list_trips
from pathsnap.metrics import argmax_pred, candidate_metrics, continuity, top1_from_scores
from pathsnap.models import MODEL_SPECS, SEED

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
MODEL_DIR = ROOT / "artifacts" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_SPACE = {
    "num_leaves": "int log [15, 255]",
    "learning_rate": "float log [0.01, 0.2]",
    "n_estimators": "int [100, 800] step 50",
    "min_child_samples": "int log [10, 300]",
    "subsample": "float [0.5, 1.0] (bagging, freq=1)",
    "colsample_bytree": "float [0.4, 1.0]",
    "reg_lambda": "float log [1e-3, 10]",
    "reg_alpha": "float log [1e-3, 10]",
}


def suggest(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    )


def make(params):
    return LGBMClassifier(**params, random_state=SEED, verbose=-1)


def cv_eval(make_model, tr, tr_lab, folds, X_all=None):
    scores, oof = [], np.zeros(len(tr))
    for fi, va in folds:
        m = make_model()
        m.fit(tr.iloc[fi][FEATURES], tr.y.iloc[fi])
        p = m.predict_proba(tr.iloc[va][FEATURES])[:, 1]
        oof[va] = p
        cm = candidate_metrics(tr.y.iloc[va], p)
        t1 = top1_from_scores(tr.iloc[va], p, tr_lab[tr_lab.trip.isin(tr.trip.iloc[va].unique())])
        scores.append(dict(macro_f1=cm["macro_f1"], top1=t1, log_loss=cm["log_loss"]))
    return pd.DataFrame(scores), oof


def main():
    full = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    tr_lab = pd.read_pickle(ROOT / "artifacts/dataset/train_labels.pkl")
    tr = full[full.reliable].reset_index(drop=True)
    g = tr.trip.to_numpy()
    folds3 = list(GroupKFold(n_splits=3).split(tr, tr.y, g))
    folds5 = list(GroupKFold(n_splits=5).split(tr, tr.y, g))

    def objective(trial):
        params = suggest(trial)
        vals = []
        for k, (fi, va) in enumerate(folds3):
            m = make(params).fit(tr.iloc[fi][FEATURES], tr.y.iloc[fi])
            p = m.predict_proba(tr.iloc[va][FEATURES])[:, 1]
            cm = candidate_metrics(tr.y.iloc[va], p)
            vals.append(cm["macro_f1"])
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
            trial.set_user_attr(f"fold{k}_log_loss", cm["log_loss"])
        return float(np.mean(vals))

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    # trial 0 = the untuned configuration, so the search starts from the baseline
    study.enqueue_trial(dict(num_leaves=63, learning_rate=0.05, n_estimators=400, min_child_samples=20,
                             subsample=1.0, colsample_bytree=1.0, reg_lambda=1e-3, reg_alpha=1e-3))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_s = time.time() - t0
    df = study.trials_dataframe()
    df.to_csv(RES / "optuna_trials.csv", index=False)
    best = dict(study.best_params, subsample_freq=1)
    print("best", study.best_value, best, f"search {search_s:.0f}s")

    # convergence plot
    comp = df[df.state == "COMPLETE"]
    plt.figure(figsize=(7, 4))
    plt.plot(comp.number, comp.value, "o", alpha=.6, label="trial")
    plt.plot(comp.number, comp.value.cummax(), "-", color="C3", label="best so far")
    plt.xlabel("trial"); plt.ylabel("3-fold CV macro F1"); plt.title("Optuna TPE - LightGBM")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "optuna_convergence.png", dpi=130); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(6, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1])
        plt.title("Hyperparameter importance (fANOVA)"); plt.tight_layout()
        plt.savefig(FIG / "optuna_param_importance.png", dpi=130); plt.close()
    except Exception as e:  # noqa
        print("importance failed", e)

    # ---------------- tuned vs untuned on identical 5-fold split + student-test
    st_tds = {t: prepare_trip(data_root() / "student_test" / t) for t in list_trips("student_test")}
    st = pd.concat([td.cands for td in st_tds.values()], ignore_index=True)
    st_lab = pd.concat([td.labels.assign(trip=t, obs=np.arange(len(td.labels))) for t, td in st_tds.items()],
                       ignore_index=True)
    rel = st.reliable.to_numpy()
    rows = []
    variants = {"LightGBM (untuned)": MODEL_SPECS["LightGBM"]["make"], "LightGBM (Bayes-tuned)": lambda: make(best)}
    for name, mk in variants.items():
        cvdf, _ = cv_eval(mk, tr, tr_lab, folds5)
        m = mk()
        t0 = time.time(); m.fit(tr[FEATURES], tr.y); fit_s = time.time() - t0
        t0 = time.time(); p = m.predict_proba(st[FEATURES])[:, 1]; inf_s = time.time() - t0
        cm = candidate_metrics(st.y.to_numpy()[rel], p[rel])
        cont, off = [], 0
        for td in st_tds.values():
            n = len(td.cands); cont.append(continuity(td, argmax_pred(td, p[off:off + n]))); off += n
        rows.append(dict(model=name, cv_macro_f1=cvdf.macro_f1.mean(), cv_macro_f1_std=cvdf.macro_f1.std(),
                         cv_top1=cvdf.top1.mean(), cv_log_loss=cvdf.log_loss.mean(),
                         st_top1=top1_from_scores(st, p, st_lab), st_macro_f1=cm["macro_f1"],
                         st_log_loss=cm["log_loss"], st_continuity=float(np.mean(cont)),
                         train_time_s=fit_s, inference_ms_per_1k_cands=1e6 * inf_s / len(st)))
        print(rows[-1])
        if "tuned" in name and "untuned" not in name:
            joblib.dump(m, MODEL_DIR / "model.joblib")
    pd.DataFrame(rows).to_csv(RES / "tuned_vs_untuned.csv", index=False)

    # out-of-fold predictions on ALL training rows (incl. unreliable obs) for decoder tuning
    oof_all = np.zeros(len(full))
    gfull = full.trip.to_numpy()
    for fi, va in GroupKFold(n_splits=5).split(full, full.y, gfull):
        fit = full.iloc[fi]
        fit = fit[fit.reliable]
        m = make(best).fit(fit[FEATURES], fit.y)
        oof_all[va] = m.predict_proba(full.iloc[va][FEATURES])[:, 1]
    np.save(ROOT / "artifacts/oof/LightGBM_tuned_all_rows.npy", oof_all)

    cfg_path = MODEL_DIR / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.update(model_name="LightGBM (Bayes-tuned)", features=FEATURES, params=best,
               search_space=SEARCH_SPACE, n_trials=N_TRIALS, best_cv3_macro_f1=study.best_value,
               search_time_s=search_s, seed=SEED)
    cfg.setdefault("hmm", {"beta": 10.0, "w_emission": 1.0})
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print("saved model + config")


if __name__ == "__main__":
    main()
