"""Feature ablation and importance (stage 4: uses trained models).

* Ablation: LightGBM with the untuned stage-4 configuration, trained on cumulative feature
  groups, scored with 5-fold GroupKFold on the 80 training trips (Top-1, macro F1, log loss).
* Importance: gain importance of the untuned LightGBM trained in scripts/03_train_models.py.
Outputs: results/feature_ablation.csv, results/feature_importance_untuned.csv,
         results/figures/feat_ablation.png, results/figures/feat_importance_untuned.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES
from pathsnap.metrics import candidate_metrics, top1_from_scores
from pathsnap.models import MODEL_SPECS

RES, FIG = ROOT / "results", ROOT / "results" / "figures"

GROUPS = {
    "G1 distance": ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands"],
    "G2 + heading/speed": ["head_diff", "head_diff_minus_min", "head_diff_long", "speed", "speed_x_cos"],
    "G3 + arc geometry/topology": ["arc_len", "t_proj", "dist_to_arc_end", "has_twin", "out_degree"],
    "G4 + temporal context": ["ctx_dist_1", "ctx_dist_2", "ctx_dist_3", "ctx_dist_5", "ctx_mean",
                              "ctx_mean_minus_min", "dt_prev", "dt_next"],
}


def ablation(tr, lab):
    folds = list(GroupKFold(n_splits=5).split(tr, tr.y, tr.trip))
    rows, feats = [], []
    for gname, g in GROUPS.items():
        feats = feats + g
        res = []
        for fi, va in folds:
            m = MODEL_SPECS["LightGBM"]["make"]().fit(tr.iloc[fi][feats], tr.y.iloc[fi])
            p = m.predict_proba(tr.iloc[va][feats])[:, 1]
            cm = candidate_metrics(tr.y.iloc[va], p)
            res.append(dict(top1=top1_from_scores(tr.iloc[va], p, lab[lab.trip.isin(tr.trip.iloc[va].unique())]),
                            macro_f1=cm["macro_f1"], log_loss=cm["log_loss"]))
        r = pd.DataFrame(res)
        rows.append(dict(feature_set=gname, n_features=len(feats), cv_top1=r.top1.mean(), cv_top1_std=r.top1.std(),
                         cv_macro_f1=r.macro_f1.mean(), cv_log_loss=r.log_loss.mean()))
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(RES / "feature_ablation.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(df)), df.cv_top1, yerr=df.cv_top1_std, capsize=4)
    for i, v in enumerate(df.cv_top1):
        plt.text(i, v + .005, f"{v:.3f}", ha="center")
    plt.xticks(range(len(df)), df.feature_set, rotation=12); plt.ylim(df.cv_top1.min() - .08, 1)
    plt.ylabel("5-fold grouped CV Top-1"); plt.title("Feature ablation (LightGBM, cumulative feature groups)")
    plt.tight_layout(); plt.savefig(FIG / "feat_ablation.png", dpi=130); plt.close()


def importance():
    m = joblib.load(ROOT / "artifacts/models_untuned/LightGBM.joblib")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values()
    imp.rename("gain_share").to_csv(RES / "feature_importance_untuned.csv")
    print(imp.sort_values(ascending=False).round(4).head(8).to_string())
    plt.figure(figsize=(7, 6)); plt.barh(imp.index, imp.values)
    plt.xlabel("share of total gain"); plt.title("Feature importance - LightGBM (stage-4 config, untuned)")
    plt.tight_layout(); plt.savefig(FIG / "feat_importance_untuned.png", dpi=130); plt.close()


if __name__ == "__main__":
    tr = pd.read_pickle(ROOT / "artifacts/dataset/train_cands.pkl")
    lab = pd.read_pickle(ROOT / "artifacts/dataset/train_labels.pkl")
    tr = tr[tr.reliable].reset_index(drop=True)
    importance()
    ablation(tr, lab)
