"""Comparison plots from the experiment CSVs + epoch-wise training curves.

Training curves are produced ONLY for models with a genuine iterative history:
  * LightGBM: per-boosting-round log loss on train vs a held-out group of trips.
  * MLP: per-epoch log loss on train vs held-out trips (partial_fit loop).
Outputs: results/figures/*.png, results/training_curves.csv
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
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURES
from pathsnap.models import SEED

RES, FIG = ROOT / "results", ROOT / "results" / "figures"


def comparison_plots():
    df = pd.read_csv(RES / "model_comparison.csv")
    m = df[df.model != "NearestSegment (baseline)"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(m.model, m.cv_macro_f1, yerr=m.cv_macro_f1_std, capsize=4, color="C0")
    ax.set_ylim(m.cv_macro_f1.min() - .05, 1); ax.set_ylabel("macro F1 (mean ± sd, 5 folds)")
    ax.set_title("Cross-validation macro F1 (GroupKFold by trip, train)")
    for i, v in enumerate(m.cv_macro_f1):
        ax.text(i, v + .005, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIG / "cv_macro_f1.png", dpi=130); plt.close()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - .2, df.st_top1, .4, label="Top-1 accuracy (obs-level)")
    ax.bar(x + .2, df.st_macro_f1.fillna(0), .4, label="Macro F1 (candidate-level)")
    ax.set_xticks(x); ax.set_xticklabels(df.model, rotation=20)
    ax.set_ylim(.5, 1); ax.legend(); ax.set_title("Student-test performance (classifier only)")
    plt.tight_layout(); plt.savefig(FIG / "student_test_top1_f1.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .2, df.cv_log_loss.fillna(0), .4, label="CV")
    ax.bar(x + .2, df.st_log_loss.fillna(0), .4, label="student-test")
    ax.set_xticks(x); ax.set_xticklabels(df.model, rotation=20); ax.set_yscale("log")
    ax.set_ylabel("log loss (lower is better)"); ax.legend(); ax.set_title("Log loss (baseline: N/A)")
    plt.tight_layout(); plt.savefig(FIG / "log_loss.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(m.inference_ms_per_1k_cands, m.st_top1)
    for _, r in m.iterrows():
        ax.annotate(r.model, (r.inference_ms_per_1k_cands, r.st_top1), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("inference ms per 1k candidates"); ax.set_ylabel("student-test Top-1")
    ax.set_title("Accuracy vs inference cost"); plt.tight_layout(); plt.savefig(FIG / "accuracy_vs_speed.png", dpi=130); plt.close()

    folds = pd.read_csv(RES / "cv_folds.csv")
    plt.figure(figsize=(8, 4))
    for name, g in folds.groupby("model"):
        plt.plot(g.fold, g.macro_f1, "o-", label=name)
    plt.xlabel("fold"); plt.ylabel("macro F1"); plt.title("Per-fold macro F1 (stability)"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "cv_folds_stability.png", dpi=130); plt.close()

    tv = RES / "tuned_vs_untuned.csv"
    if tv.exists():
        t = pd.read_csv(tv)
        cols = ["cv_macro_f1", "st_top1", "st_macro_f1"]
        x = np.arange(len(cols))
        plt.figure(figsize=(7, 4))
        for k, (_, r) in enumerate(t.iterrows()):
            plt.bar(x + (k - .5) * .4, r[cols], .4, label=r.model)
        plt.xticks(x, ["CV macro F1", "student-test Top-1", "student-test macro F1"])
        plt.ylim(t[cols].values.min() - .03, 1); plt.legend(); plt.title("Untuned vs Bayes-tuned LightGBM")
        plt.tight_layout(); plt.savefig(FIG / "tuned_vs_untuned.png", dpi=130); plt.close()


def training_curves():
    tr = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    tr = tr[tr.reliable].reset_index(drop=True)
    fi, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, tr.y, tr.trip))
    Xtr, ytr, Xva, yva = tr.iloc[fi][FEATURES], tr.y.iloc[fi], tr.iloc[va][FEATURES], tr.y.iloc[va]
    rows = []
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "valid"], eval_metric="binary_logloss")
    ev = m.evals_result_
    for i, (a, b) in enumerate(zip(ev["train"]["binary_logloss"], ev["valid"]["binary_logloss"])):
        rows.append(dict(model="LightGBM", step=i + 1, train_loss=a, val_loss=b))

    sc = StandardScaler().fit(Xtr)
    sub = np.random.default_rng(SEED).choice(len(Xtr), size=min(300_000, len(Xtr)), replace=False)
    Xs, ys = sc.transform(Xtr.iloc[sub]), ytr.iloc[sub].to_numpy()
    Xv = sc.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                        random_state=SEED)
    for ep in range(40):
        mlp.partial_fit(Xs, ys, classes=[0, 1])
        rows.append(dict(model="MLP", step=ep + 1,
                         train_loss=log_loss(ys, mlp.predict_proba(Xs)[:, 1], labels=[0, 1]),
                         val_loss=log_loss(yva, mlp.predict_proba(Xv)[:, 1], labels=[0, 1])))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "training_curves.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (name, g) in zip(ax, df.groupby("model", sort=False)):
        a.plot(g.step, g.train_loss, label="train"); a.plot(g.step, g.val_loss, label="validation (held-out trips)")
        a.set_xlabel("boosting round" if name == "LightGBM" else "epoch"); a.set_ylabel("log loss")
        a.set_title(f"{name} training history"); a.legend()
    plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=130); plt.close()


def per_model_curves():
    """One training-history figure per model (from results/training_curves.csv)."""
    df = pd.read_csv(RES / "training_curves.csv")
    for name, g in df.groupby("model", sort=False):
        best = g.loc[g.val_loss.idxmin()]
        plt.figure(figsize=(6.5, 4))
        plt.plot(g.step, g.train_loss, label="train")
        plt.plot(g.step, g.val_loss, label="validation (held-out trips)")
        plt.axvline(best.step, color="grey", ls="--", lw=.8)
        plt.annotate(f"min val loss {best.val_loss:.4f} @ {int(best.step)}", (best.step, best.val_loss),
                     xytext=(10, 25), textcoords="offset points", fontsize=8)
        plt.xlabel("boosting round" if name == "LightGBM" else "epoch"); plt.ylabel("log loss")
        plt.title(f"{name} training history"); plt.legend()
        plt.tight_layout(); plt.savefig(FIG / f"training_curve_{name}.png", dpi=130); plt.close()


if __name__ == "__main__":
    comparison_plots()
    training_curves()
    per_model_curves()
