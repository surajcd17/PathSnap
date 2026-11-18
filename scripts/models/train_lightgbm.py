"""Train and evaluate ONE model: LightGBM.

Same protocol as scripts/03_train_models.py (which trains all models in one run):
5-fold GroupKFold over the 80 training trips (reliable observations only), refit on all
training trips, evaluation on the 10 student-test trips. The model configuration lives in
pathsnap/models.py (MODEL_SPECS["LightGBM"]).

Run from the project root:  python scripts/models/train_lightgbm.py
Outputs: results/model_comparison_LightGBM.csv, results/cv_folds_LightGBM.csv,
         results/figures/confusion_LightGBM.png, artifacts/models_untuned/LightGBM.joblib
"""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "LightGBM"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "03_train_models.py"), run_name="__main__")
