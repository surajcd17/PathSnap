"""Model zoo for the candidate classifier (one model per algorithm family).

`fit_rows` = max training rows used when fitting (random subsample, fixed seed) for the
models whose training/prediction cost grows super-linearly; None = all rows.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

MODEL_SPECS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        config="StandardScaler + L2 logistic regression, C=1.0, lbfgs, max_iter=1000",
        make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        fit_rows=None),
    "GaussianNB": dict(
        family="Probabilistic (generative)",
        config="StandardScaler + Gaussian Naive Bayes, var_smoothing=1e-9",
        make=lambda: make_pipeline(StandardScaler(), GaussianNB()),
        fit_rows=None),
    "KNN": dict(
        family="Instance-based",
        config="StandardScaler + KNN k=25, distance-weighted, euclidean; fit on 60k-row subsample",
        make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        fit_rows=60_000),
    "RandomForest": dict(
        family="Tree ensemble (bagging)",
        config="200 trees, max_features=sqrt, min_samples_leaf=5; fit on 300k-row subsample",
        make=lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        fit_rows=300_000),
    "LightGBM": dict(
        family="Gradient boosting",
        config="400 trees, lr=0.05, num_leaves=63, default regularisation",
        make=lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                    random_state=SEED, verbose=-1),
        fit_rows=None),
    "MLP": dict(
        family="Neural network",
        config="StandardScaler + MLP (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, batch 512, early stopping; 300k-row subsample",
        make=lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
            max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=SEED)),
        fit_rows=300_000),
}
