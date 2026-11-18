"""Train and evaluate ONE model: RandomForest.

Same protocol as scripts/03_train_models.py (which trains all models in one run):
5-fold GroupKFold over the 80 training trips (reliable observations only), refit on all
training trips, evaluation on the 10 student-test trips. The model configuration lives in
pathsnap/models.py (MODEL_SPECS["RandomForest"]).

Run from the project root:  python scripts/models/train_random_forest.py
Outputs: results/model_comparison_RandomForest.csv, results/cv_folds_RandomForest.csv,
         results/figures/confusion_RandomForest.png, artifacts/models_untuned/RandomForest.joblib
"""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "RandomForest"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "03_train_models.py"), run_name="__main__")
