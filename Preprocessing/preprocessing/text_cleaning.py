"""
Classical text cleaning and TF-IDF construction.

The cleaning steps follow the original pipeline: lowercase, strip HTML, replace
URLs, drop mentions and non-alphabetic characters, tokenise, remove stopwords
and lemmatise. The one addition is an optional slang normalisation step that
runs first, controlled by the switch in config.py.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    OUTPUTS_DIR, ENABLE_SLANG_NORMALISATION,
)
from preprocessing.slang_normaliser import normalise_slang

# NLTK is needed for tokenising, stopwords and lemmatising.
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _STOPWORDS  = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except (ImportError, LookupError):
    _NLTK_AVAILABLE = False
    print(
        "[text_cleaning] NLTK resources not found. Run:\n"
        "  python -c \"import nltk; nltk.download('stopwords'); "
        "nltk.download('wordnet'); nltk.download('punkt_tab')\""
    )

TFIDF_PATH = OUTPUTS_DIR / "tfidf_vectorizer.joblib"


def clean_text(text: str, use_slang: bool = ENABLE_SLANG_NORMALISATION) -> str:
    """Clean a single post for the bag of words models.

    If use_slang is True, the slang normalisation runs first so that coded
    crisis terms survive into the cleaned text.
    """
    if not isinstance(text, str):
        return ""

    if use_slang:
        text = normalise_slang(text)

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)            # HTML
    text = re.sub(r"http\S+|www\S+", " url ", text)  # URLs -> placeholder
    text = re.sub(r"@\w+", " ", text)               # mentions
    text = re.sub(r"[^a-z\s]", " ", text)           # non-alpha
    text = re.sub(r"\s+", " ", text).strip()

    if not _NLTK_AVAILABLE:
        return text

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_series(series: pd.Series, use_slang: bool = ENABLE_SLANG_NORMALISATION) -> pd.Series:
    """Apply clean_text across a Series of posts."""
    return series.apply(lambda t: clean_text(t, use_slang=use_slang))


def build_tfidf(train_texts, other_splits: dict = None, save: bool = True):
    """Fit TF-IDF on the training text only, then transform the other splits.

    Fitting on anything other than train would leak information, so the
    vectoriser only ever learns its vocabulary from the training set.
    """
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=TFIDF_MIN_DF,
        strip_accents="unicode",
    )
    out = {"train": vectorizer.fit_transform(train_texts), "vectorizer": vectorizer}
    print(f"[tfidf] vocabulary size {len(vectorizer.vocabulary_):,}")
    print(f"[tfidf] train matrix {out['train'].shape}")

    for name, texts in (other_splits or {}).items():
        out[name] = vectorizer.transform(texts)
        print(f"[tfidf] {name} matrix {out[name].shape}")

    if save:
        joblib.dump(vectorizer, TFIDF_PATH)
        print(f"[tfidf] vectoriser saved to {TFIDF_PATH}")
    return out


if __name__ == "__main__":
    for t in [
        "I want to kill myself, nobody cares about me anymore",
        "been struggling lately, just wanna kms honestly",
        "Just had the best pizza of my life, fantastic day!",
    ]:
        print("IN :", t)
        print("OUT:", clean_text(t))
        print()
