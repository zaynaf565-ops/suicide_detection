"""
Configuration for the final evaluation stage.

This stage reports test set performance for the base models and the ensembles,
with bootstrap confidence intervals and a McNemar comparison. It reads saved
test probabilities and the tuned thresholds. Nothing is retrained.
"""
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent

CM_PRED_DIR  = REPO_ROOT / "Classical_Models" / "results" / "predictions"
CM_RESULTS   = REPO_ROOT / "Classical_Models" / "results"
ENS_RESULTS  = REPO_ROOT / "Ensembles" / "results"
ENS_MODELS   = REPO_ROOT / "Ensembles" / "models"

RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
for _d in (RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED   = 42
FBETA         = 2.0
N_BOOTSTRAP   = 1000
CI_LEVEL      = 0.95

BASE_MODELS = [
    ("logistic_regression",     "Logistic Regression"),
    ("multinomial_naive_bayes", "Multinomial Naive Bayes"),
    ("linear_svm",              "Linear SVM"),
    ("random_forest",           "Random Forest"),
    ("xgboost",                 "XGBoost"),
]
