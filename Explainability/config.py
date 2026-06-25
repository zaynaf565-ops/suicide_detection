"""
Configuration for the explainability stage.

This stage explains what drives the models. It computes global feature
importance for every model, produces local LIME explanations for a few example
posts, and measures how far SHAP and LIME agree on the same posts.

It reads saved models, the saved vectoriser, the saved features and the cleaned
text. It also imports the preprocessing functions so that LIME can run the full
path from raw text to a prediction.
"""
import sys
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent

PRE_DIR      = REPO_ROOT / "Pre_Processing"
PRE_OUT      = PRE_DIR / "outputs"
VECTORIZER   = PRE_OUT / "tfidf_vectorizer.joblib"
TFIDF_DIR    = PRE_OUT / "tfidf"
FEAT_DIR     = PRE_OUT / "features"
CLEAN_DIR    = PRE_OUT / "clean"
CM_MODELS    = REPO_ROOT / "Classical_Models" / "models"
SPLITS_DIR   = PRE_DIR / "data" / "splits"

RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
for _d in (RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Make the preprocessing package importable for the LIME pipeline.
# The module folder is "Preprocessing" inside "Pre_Processing".
for p in (PRE_DIR, PRE_DIR / "Preprocessing", PRE_DIR / "preprocessing"):
    if p.exists():
        sys.path.insert(0, str(PRE_DIR))

RANDOM_SEED = 42
TOP_FEATURES = 20          # top features to show per model
SHAP_SAMPLE  = 300         # rows sampled for SHAP on tree models
LIME_EXAMPLES = 4          # number of posts to explain locally with LIME
LIME_NUM_SAMPLES = 500     # perturbations per LIME explanation
AGREEMENT_SAMPLE = 25      # posts used for the SHAP vs LIME agreement
AGREEMENT_TOPK = 10        # top tokens compared per post

HANDCRAFTED_NAMES = [
    "word_count", "char_count", "avg_word_length",
    "exclamation_count", "question_count", "exclamation_density", "question_density",
    "ellipsis_count", "uppercase_ratio",
    "first_person_count", "first_person_density",
    "negation_count", "negation_density",
    "hopelessness_count", "hopelessness_density",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]

BASE_MODELS = [
    ("logistic_regression",     "Logistic Regression",     "linear", "full"),
    ("multinomial_naive_bayes", "Multinomial Naive Bayes", "nb",     "mnb"),
    ("linear_svm",              "Linear SVM",              "linear", "full"),
    ("random_forest",           "Random Forest",           "tree",   "full"),
    ("xgboost",                 "XGBoost",                 "tree",   "full"),
]
