"""
Configuration for the ensemble stage.

This stage combines the five trained classical models. It reads their saved
validation and calibration probabilities rather than the models themselves, so
no model is retrained here.
"""
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent          # Ensembles/
REPO_ROOT = ROOT_DIR.parent

# Where the classical models stage saved its per model probabilities.
# Change this single line if your folder names differ.
PRED_IN_DIR = REPO_ROOT / "Classical_Models" / "results" / "predictions"

MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PRED_OUT_DIR = RESULTS_DIR / "predictions"
for _d in (MODELS_DIR, RESULTS_DIR, PRED_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
FBETA       = 2.0

# The five base models, in a fixed order. The safe names match the files saved
# by the classical stage, for example "xgboost_val_probs.csv".
BASE_MODELS = [
    "logistic_regression",
    "multinomial_naive_bayes",
    "linear_svm",
    "random_forest",
    "xgboost",
]
MODEL_LABELS = {
    "logistic_regression":     "Logistic Regression",
    "multinomial_naive_bayes": "Multinomial Naive Bayes",
    "linear_svm":              "Linear SVM",
    "random_forest":           "Random Forest",
    "xgboost":                 "XGBoost",
}
