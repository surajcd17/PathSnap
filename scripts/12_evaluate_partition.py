"""Evaluate the FROZEN final pipeline on a folder of trips (e.g. the evaluator partition).

Runs exactly what the API runs: PathSnapModel.predict_trip (never sees the reference
route) followed by score_prediction (server-side scoring). Nothing is trained or tuned here;
the evaluator partition is used for final evaluation only.

Run:  python scripts/12_evaluate_partition.py <trips_folder> <name>
e.g.  python scripts/12_evaluate_partition.py data/evaluator/evaluator_test evaluator_test
Outputs: results/<name>_results.csv (per trip + observation-weighted total)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from pathsnap.inference import PathSnapModel, score_prediction

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/evaluator/evaluator_test"
name = sys.argv[2] if len(sys.argv) > 2 else folder.name
if not folder.is_absolute():
    folder = ROOT / folder

model = PathSnapModel()
print(f"model: {model.meta.get('model_name')} | HMM beta={model.hmm_cfg.beta} w_e={model.hmm_cfg.w_emission} | trips: {folder}")
rows = []
for d in sorted(p for p in folder.iterdir() if p.is_dir()):
    t0 = time.perf_counter()
    td, raw, proba, clf, hmm, info = model.predict_trip(d)
    m = score_prediction(td, d, clf, hmm)
    c, h = m["classifier_only"], m["hmm"]
    r = dict(trip=d.name, n_obs=h["n_obs"], n_reliable=h["n_reliable"],
             frac_reliable=m["frac_reliable_labels"], candidate_miss_rate=m["candidate_miss_rate"])
    for k in ("top1_acc", "macro_f1", "continuity", "route_precision", "route_recall", "route_f1"):
        r[f"clf_{k}"], r[f"hmm_{k}"] = c[k], h[k]
    r.update(hmm_breaks=info["n_breaks"], pipeline_s=info["total_s"], total_s=time.perf_counter() - t0)
    rows.append(r)
    print(f"{d.name}: obs={r['n_obs']:5d} reliable={r['frac_reliable']:.3f} miss={r['candidate_miss_rate']:.4f} | "
          f"Top-1 clf {r['clf_top1_acc']:.4f} -> HMM {r['hmm_top1_acc']:.4f} | continuity {r['hmm_continuity']:.4f} | "
          f"route F1 {r['hmm_route_f1']:.4f} | {r['total_s']:.2f}s")
df = pd.DataFrame(rows)
w = df.n_reliable
tot = {"trip": "ALL (obs-weighted)", "n_obs": int(df.n_obs.sum()), "n_reliable": int(w.sum()),
       "frac_reliable": float(w.sum() / df.n_obs.sum()),
       "candidate_miss_rate": float((df.candidate_miss_rate * w).sum() / w.sum())}
for c in df.columns:
    if c.startswith(("clf_", "hmm_")) and c != "hmm_breaks":
        tot[c] = float((df[c] * w).sum() / w.sum())
tot.update(hmm_breaks=int(df.hmm_breaks.sum()), pipeline_s=float(df.pipeline_s.sum()), total_s=float(df.total_s.sum()))
df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
out = ROOT / "results" / f"{name}_results.csv"
df.to_csv(out, index=False)
t = df.iloc[-1]
print(f"\nALL: Top-1 classifier {t.clf_top1_acc:.4f} -> HMM {t.hmm_top1_acc:.4f} | macro F1 {t.clf_macro_f1:.4f} -> "
      f"{t.hmm_macro_f1:.4f} | continuity {t.clf_continuity:.4f} -> {t.hmm_continuity:.4f} | route F1 "
      f"{t.clf_route_f1:.4f} -> {t.hmm_route_f1:.4f} | {t.total_s:.1f}s total")
print("saved", out)
