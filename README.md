# PathSnap: Map-Matching & Route Reconstruction

Work in progress. PathSnap matches noisy GPS tracks to a road network using candidate
classification and HMM/Viterbi decoding, served by a FastAPI backend and an evaluator UI.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Place the organisers' data in `data/Student folder/{train,student_test}/<trip_id>/`.

The full README (results, deployment, reproduction) is added in the final commit.
