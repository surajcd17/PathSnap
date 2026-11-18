"""Train and evaluate ONE model: LogisticRegression.

Same protocol as scripts/03_train_models.py (which trains all models in one run):
5-fold GroupKFold over the 80 training trips (reliable observations only), refit on all
training trips, evaluation on the 10 student-test trips. The model configuration lives in
pathsnap/models.py (MODEL_SPECS["LogisticRegression"]).

Run from the project root:  python scripts/models/train_logistic_regression.py
Outputs: results/model_comparison_LogisticRegression.csv, results/cv_folds_LogisticRegression.csv,
         results/figures/confusion_LogisticRegression.png, artifacts/models_untuned/LogisticRegression.joblib
"""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "LogisticRegression"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "03_train_models.py"), run_name="__main__")
