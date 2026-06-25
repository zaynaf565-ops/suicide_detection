"""
Configuration for the trustworthiness stage (calibration and conformal prediction).

This stage reads saved probabilities and saved models. It does not retrain.
"""
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent

# Inputs from earlier stages. Change these lines if your folder names differ.
PRE_OUTPUTS   = REPO_ROOT / "Pre_Processing" / "outputs"
TFIDF_DIR     = PRE_OUTPUTS / "tfidf"
FEAT_DIR      = PRE_OUTPUTS / "features"
CM_MODELS_DIR = REPO_ROOT / "Classical_Models" / "models"
CM_PRED_DIR   = REPO_ROOT / "Classical_Models" / "results" / "predictions"

# Outputs of this stage.
RESULTS_DIR = ROOT_DIR / "results"
PRED_DIR    = RESULTS_DIR / "predictions"
PLOTS_DIR   = RESULTS_DIR / "plots"
for _d in (RESULTS_DIR, PRED_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ECE_N_BINS  = 10
ALPHA       = 0.10          # target miscoverage, so 90 percent coverage
RAPS_LAMBDA_CANDIDATES = [0.0, 0.001, 0.01, 0.05, 0.1]

# The five base models. The flag marks which one needs the Naive Bayes safe
# feature matrix when predicting on the test set.
BASE_MODELS = [
    ("logistic_regression",     "Logistic Regression",     "full"),
    ("multinomial_naive_bayes", "Multinomial Naive Bayes", "mnb"),
    ("linear_svm",              "Linear SVM",              "full"),
    ("random_forest",           "Random Forest",           "full"),
    ("xgboost",                 "XGBoost",                 "full"),
]
