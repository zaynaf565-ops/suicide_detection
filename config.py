"""
Central configuration for the project.

Every path and constant lives here so that no other file hardcodes a location.
If you move the project, you only change things in this one file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
SPLITS_DIR    = DATA_DIR / "splits"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

RAW_CSV       = RAW_DIR / "Suicide_Detection.csv"

for _d in (SPLITS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column names and labels
# ---------------------------------------------------------------------------
TEXT_COL  = "text"
LABEL_COL = "label"          # we map the raw "class" column to this 0/1 column
RAW_CLASS_COL = "class"
POSITIVE_LABEL = "suicide"   # mapped to 1

# ---------------------------------------------------------------------------
# Data split sizes (four-way stratified split)
# Test is carved out first, then calibration, then train/val from the rest.
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TEST_SIZE     = 0.20   # of the whole dataset
CAL_SIZE      = 0.10   # of the whole dataset
VAL_FRACTION  = 0.20   # of the remaining train+val pool

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
MIN_WORDS = 5          # drop posts shorter than this (too little signal)

# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 3

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
FBETA = 2.0            # recall weighted twice as heavily as precision

# ---------------------------------------------------------------------------
# Improvement switch
# Set to False to reproduce the original client pipeline exactly (no slang
# normalisation). Set to True to enable the algospeak / slang normalisation
# layer. Keeping this as a switch lets us run a clean before and after
# comparison and report the difference.
# ---------------------------------------------------------------------------
ENABLE_SLANG_NORMALISATION = True
