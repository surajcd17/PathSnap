# PathSnap — Map-Matching & Route Reconstruction

PathSnap matches noisy 1 Hz GPS tracks to a road network. For each GPS observation it:

1. generates candidate road segments (arcs) within 50 m,
2. scores each candidate with a trained classifier (Bayes-tuned LightGBM), and
3. decodes a connected route with an HMM and the Viterbi algorithm.

The system runs on two laptops. A **FastAPI** backend on Laptop 1 runs the model, the HMM and the server-side scoring. A **web UI** on Laptop 2 lists the evaluation trips and draws the predicted route.

## Repository layout

| Folder / file | Contents |
|---|---|
| `pathsnap/io.py` | Raw-file loading and track cleaning (invalid rows, ordering, duplicate timestamps) |
| `pathsnap/geo.py` | Projection of each trip into its own UTM zone, plus point–segment geometry |
| `pathsnap/network.py` | Corridor-cropped road network, KD-tree spatial index and bounded Dijkstra on directed arcs |
| `pathsnap/labels.py` | Monotone DP alignment of GPS points to the `.route` arc sequence, which gives each observation a label |
| `pathsnap/features.py` | Candidate generation and feature engineering |
| `pathsnap/models.py` | The model zoo: 6 algorithm families |
| `pathsnap/hmm.py` | HMM and Viterbi decoder |
| `pathsnap/metrics.py` | Top-1, macro F1, log loss, route continuity and route precision / recall / F1 |
| `pathsnap/inference.py` | Inference pipeline used by the API, with server-side scoring kept separate |
| `scripts/00_sanity_checks.py` | Checks of arc-ID base, direction and route connectivity |
| `scripts/01_eda.py` | EDA: inventory, missing values, gaps, connectivity, plots and a Folium map → `results/eda/` |
| `scripts/02_build_dataset.py` | Candidate/feature tables for train and student_test |
| `scripts/03_train_models.py` | Trains 6 models + a baseline with 5-fold GroupKFold CV and student-test evaluation |
| `scripts/04_bayes_opt.py` | Optuna TPE search for LightGBM; writes the final model to `artifacts/model/` |
| `scripts/05_hmm_eval.py` | Decoder tuning, then classifier-only vs classifier + HMM |
| `scripts/06_maps.py` | Folium maps of predicted routes → `results/maps/` |
| `scripts/07_plots.py` | Comparison plots and training curves (LightGBM and MLP only) |
| `scripts/08_api_smoke_test.py` | Calls the running API for every trip |
| `scripts/09_feature_analysis.py` | Model-free feature evidence: candidate statistics, single-feature AUC, distributions |
| `scripts/10_feature_ablation.py` | Feature-group ablation and LightGBM feature importance |
| `scripts/11_export_preprocessed_samples.py` | Readable CSV views of the preprocessed data → `results/preprocessed_samples/` |
| `scripts/12_evaluate_partition.py` | Runs the frozen pipeline on any folder of trips (final evaluation) |
| `scripts/models/train_<model>.py` | Trains and evaluates one model at a time (same protocol as `03_train_models.py`) |
| `results/logs/models/` | Training log and per-fold results of each model |
| `api/main.py` | FastAPI backend |
| `ui/index.html`, `ui/serve.py` | Evaluator UI (Leaflet) |
| `artifacts/model/` | Final model (`model.joblib`) and `config.json` (features, hyperparameters, HMM parameters) |
| `results/` | CSVs, logs and figures that back every number in the report |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the organisers' data in `data/Student folder/{train,student_test}/<trip_id>/`. To use another location, set `PATHSNAP_DATA`.

## Reproducing the experiments

```bash
python scripts/00_sanity_checks.py
python scripts/02_build_dataset.py
python scripts/01_eda.py
python scripts/03_train_models.py
python scripts/04_bayes_opt.py 30
python scripts/05_hmm_eval.py
python scripts/07_plots.py
python scripts/06_maps.py
python scripts/09_feature_analysis.py
python scripts/10_feature_ablation.py
python scripts/11_export_preprocessed_samples.py
```

All random seeds are fixed to 42.

## Deployment (two laptops)

**Laptop 1: API**

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluation_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `PATHSNAP_EVAL_DIR` is the folder holding the 10 evaluation trip folders. If it is not set, the API serves the student_test trips.
- Allow port 8000 through the Windows firewall when prompted.
- Find the laptop's IP address with `ipconfig`.
- Every request is logged in this terminal. Each prediction logs a `[PREDICT]` line and a `[RESULT]` line with the trip ID and its metrics.

**Laptop 2: UI**

```bash
python ui/serve.py 5500
```

Open `http://localhost:5500`, enter `http://<laptop1-ip>:8000` and click **Connect**. Then pick any of the listed trips.

**Endpoints**

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /trips` | Lists the available trips |
| `POST /predict {"trip_id": "..."}` | Returns the GPS points, the predicted route (HMM and classifier-only) and the server-computed metrics |

The reference route is read only on the server, after prediction, to compute accuracy. It is never sent to the model or the UI.

## Results (student-test partition, 10 trips, 19,160 observations)

All numbers are produced by the scripts above. The CSV each one comes from is named in brackets.

**Classifier comparison** (`results/model_comparison.csv`; CV = 5-fold GroupKFold over the 80 training trips)

| Model | Family | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Student-test log loss | Continuity |
|---|---|---|---|---|---|---|
| Nearest segment | heuristic baseline | N/A | 0.540 | N/A | N/A | 0.917 |
| Logistic Regression | linear | 0.914 | 0.876 | 0.912 | 0.114 | 0.981 |
| Gaussian NB | probabilistic | 0.829 | 0.850 | 0.828 | 0.768 | 0.979 |
| KNN (k=25) | instance-based | 0.919 | 0.877 | 0.923 | 0.132 | 0.977 |
| Random Forest | bagging | 0.941 | 0.898 | 0.938 | 0.077 | 0.984 |
| **LightGBM** | boosting | **0.941** | **0.911** | **0.944** | **0.075** | 0.984 |
| MLP (64, 32) | neural network | 0.934 | 0.899 | 0.935 | 0.081 | 0.986 |

**Bayesian optimisation** (`results/tuned_vs_untuned.csv`)

- Settings: Optuna TPE, 30 trials, 8 hyperparameters, objective = 3-fold grouped CV macro F1.
- The tuned model improves CV macro F1 from 0.9403 to 0.9417 and CV log loss from 0.083 to 0.078.
- It is also about 1.8× faster to train and to run.
- On student-test, Top-1 drops slightly (0.911 → 0.908).

**HMM / Viterbi** (`results/hmm_vs_classifier_student_test.csv`; decoder β=1, emission weight 4)

| | Top-1 | Macro F1 | Continuity | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|
| Classifier only (tuned LightGBM) | 0.908 | 0.947 | 0.983 | 0.912 | 0.918 | 0.913 |
| Classifier + HMM | **0.936** | **0.963** | **1.000** | **0.960** | **0.923** | **0.940** |

The same decoder improves every model (`results/hmm_all_models.csv`). End-to-end API latency is about 2–3 s per trip, most of it feature building and server-side scoring.

## Final evaluation (evaluator partition, 10 trips, 19,674 observations)

The frozen pipeline (same model, features and decoder, with nothing changed after seeing these trips) was run once on the evaluator partition. It was run both directly with `scripts/12_evaluate_partition.py` and through the API with `PATHSNAP_EVAL_DIR`; the two give identical results (`results/evaluator_test_results.csv`, `results/logs/api_smoke_test_evaluator_test.log`).

| | Top-1 | Macro F1 | Continuity | Route F1 |
|---|---|---|---|---|
| Classifier only | 0.918 | 0.952 | 0.995 | 0.840 |
| Classifier + HMM | **0.951** | **0.972** | **1.000** | **0.896** |

Per trip, HMM Top-1 ranges from 0.855 (trip 00000030) to 1.000 (trip 00000049), and the HMM improves every trip.

```bash
python scripts/12_evaluate_partition.py data/evaluator/evaluator_test evaluator_test
```

## Method summary

- **Projection:** each trip is projected into the UTM zone of its own centroid. The trips span 25 zones.
- **Labels:** `.route` has no timestamps, so a monotone DP alignment assigns each GPS point to a route position.
  - Points more than 50 m from their aligned arc are marked *unreliable* and excluded from training and scoring.
  - This mainly affects train trips 54 and 33, whose routes cover only part of the track.
- **Candidates:** up to 10 arcs within 50 m. If none are found, the nearest arc within 200 m is used.
- **Features:** distance and its rank, heading difference (short and long window), speed, arc length, projection position, twin arc, out-degree, and context distances of neighbouring fixes to the arc.
- **HMM:**
  - Hidden states are the candidate arcs.
  - Emission is the classifier probability normalised per observation, raised to the power `w_e`.
  - Transition is `exp(-|d_route − d_gc| / (β·√Δt))`, where `d_route` is the bounded-Dijkstra distance along directed arcs.
  - Unreachable transitions get a floor penalty, so a disconnected step restarts the decoding instead of breaking it.
  - Observations without candidates are skipped.
