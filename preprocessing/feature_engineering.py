"""
Handcrafted psycholinguistic features.

These nineteen features are taken from the original pipeline, which followed
the clinical and psycholinguistic literature: post length and surface
statistics, first person singular density, negation, a hopelessness lexicon and
VADER sentiment. They are computed from the raw text rather than the cleaned
text, because case and punctuation carry signal that cleaning would remove.

The only change is an optional slang normalisation pass, kept consistent with
the rest of the pipeline through the switch in config.py, so that terms like
"wanna" are read correctly by the sentiment and negation features.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import ENABLE_SLANG_NORMALISATION
from preprocessing.slang_normaliser import normalise_slang

_VADER = SentimentIntensityAnalyzer()

_FIRST_PERSON_SINGULAR = re.compile(
    r"\b(i|me|my|myself|mine|i'm|i've|i'll|i'd)\b", re.IGNORECASE
)
_NEGATION = re.compile(
    r"\b(not|no|never|nothing|nobody|neither|nor|cannot|can't|won't|don't|"
    r"didn't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_HOPELESSNESS = re.compile(
    r"\b(hopeless|worthless|pointless|meaningless|useless|helpless|"
    r"nothing|nobody|empty|numb|tired|exhausted|done|over|end)\b",
    re.IGNORECASE,
)


def _feature_names() -> list:
    return [
        "word_count", "char_count", "avg_word_length",
        "exclamation_count", "question_count",
        "exclamation_density", "question_density",
        "ellipsis_count", "uppercase_ratio",
        "first_person_count", "first_person_density",
        "negation_count", "negation_density",
        "hopelessness_count", "hopelessness_density",
        "vader_neg", "vader_neu", "vader_pos", "vader_compound",
    ]


ALL_FEATURE_NAMES = _feature_names()
MNB_INCOMPATIBLE_FEATURES = {"vader_neg", "vader_compound", "vader_pos"}
MNB_SAFE_FEATURES = [f for f in ALL_FEATURE_NAMES if f not in MNB_INCOMPATIBLE_FEATURES]


def extract_features(text: str, use_slang: bool = ENABLE_SLANG_NORMALISATION) -> dict:
    """Extract the nineteen handcrafted features from one post."""
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in ALL_FEATURE_NAMES}

    if use_slang:
        text = normalise_slang(text)

    words = text.split()
    n_words = max(len(words), 1)
    n_chars = len(text)
    vader = _VADER.polarity_scores(text)

    return {
        "word_count":           n_words,
        "char_count":           n_chars,
        "avg_word_length":      np.mean([len(w) for w in words]),
        "exclamation_count":    text.count("!"),
        "question_count":       text.count("?"),
        "exclamation_density":  text.count("!") / n_words,
        "question_density":     text.count("?") / n_words,
        "ellipsis_count":       text.count("..."),
        "uppercase_ratio":      sum(1 for c in text if c.isupper()) / max(n_chars, 1),
        "first_person_count":   len(_FIRST_PERSON_SINGULAR.findall(text)),
        "first_person_density": len(_FIRST_PERSON_SINGULAR.findall(text)) / n_words,
        "negation_count":       len(_NEGATION.findall(text)),
        "negation_density":     len(_NEGATION.findall(text)) / n_words,
        "hopelessness_count":   len(_HOPELESSNESS.findall(text)),
        "hopelessness_density": len(_HOPELESSNESS.findall(text)) / n_words,
        "vader_neg":            vader["neg"],
        "vader_neu":            vader["neu"],
        "vader_pos":            vader["pos"],
        "vader_compound":       vader["compound"],
    }


def build_feature_matrix(texts: pd.Series, for_mnb: bool = False) -> np.ndarray:
    """Build a feature matrix from a Series of posts.

    If for_mnb is True, the three VADER features that can be negative are
    dropped, because Multinomial Naive Bayes needs non negative inputs.
    """
    records = texts.apply(extract_features).tolist()
    df_feat = pd.DataFrame(records)
    cols = MNB_SAFE_FEATURES if for_mnb else ALL_FEATURE_NAMES
    df_feat = df_feat[cols]
    mode = "MNB" if for_mnb else "full"
    print(f"[features] {mode} mode: {df_feat.shape[1]} features over {len(df_feat):,} rows")
    return df_feat.values.astype(np.float32)
