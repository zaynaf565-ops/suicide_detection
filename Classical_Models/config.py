"""
Configuration for the classical models phase.

This phase reads the artifacts produced by the preprocessing phase rather than
recomputing them, so the exact same cleaning, TF-IDF and feature set is used.
All paths and constants live here.
"""
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent          # Classical_Models/
REPO_ROOT = ROOT_DIR.parent                           # repository root

# ---------------------------------------------------------------------------
# Where the preprocessing phase wrote its outputs.
# If your preprocessing folder has a different name, change PRE_OUTPUTS only.
# ---------------------------------------------------------------------------
PRE_OUTPUTS = REPO_ROOT / "Pre_Processing" / "outputs"
TFIDF_DIR   = PRE_OUTPUTS / "tfidf"
FEAT_DIR    = PRE_OUTPUTS / "features"
PRE_STATS   = PRE_OUTPUTS / "preprocessing_stats.json"

# ---------------------------------------------------------------------------
# Where this phase writes its outputs.
# ---------------------------------------------------------------------------
MODELS_DIR  = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PRED_DIR    = RESULTS_DIR / "predictions"
for _d in (MODELS_DIR, RESULTS_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training constants (kept the same as the original pipeline)
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
FBETA          = 2.0     # recall weighted twice as heavily as precision
N_ITER_SEARCH  = 15      # RandomizedSearchCV iterations per model
CV_FOLDS       = 3       # inner cross validation folds during the search

SPLITS = ["train", "val", "calibration"]
