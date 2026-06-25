"""
Self contained cleaning and feature extraction for the LIME pipeline.

This mirrors the preprocessing stage so that LIME can run the full path from raw
text to a prediction without importing the preprocessing package, which avoids a
name clash between the two config files. Slang normalisation is on, matching the
settings used to train the models.
"""
import re
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _STOP = set(stopwords.words("english"))
    _LEM = WordNetLemmatizer()
    _NLTK = True
except Exception:
    _NLTK = False

_VADER = SentimentIntensityAnalyzer()

_CRISIS = {
    r"\bunalive\s+myself\b": "kill myself", r"\bunalive\b": "suicide",
    r"\bun-?alive\b": "suicide", r"\bkms\b": "kill myself", r"\bkys\b": "kill yourself",
    r"\bsewer\s*slide\b": "suicide", r"\bsewerslide\b": "suicide",
    r"\bself\s*delete\b": "kill myself", r"\bend\s+it\s+all\b": "suicide",
}
_INFORMAL = {r"\bwanna\b": "want to", r"\bgonna\b": "going to", r"\bgotta\b": "got to",
    r"\bidk\b": "i do not know", r"\bcant\b": "cannot", r"\bdont\b": "do not",
    r"\bim\b": "i am", r"\bive\b": "i have"}
_COMPILED = [(re.compile(p, re.I), r) for p, r in {**_CRISIS, **_INFORMAL}.items()]

_FP = re.compile(r"\b(i|me|my|myself|mine|i'm|i've|i'll|i'd)\b", re.I)
_NEG = re.compile(r"\b(not|no|never|nothing|nobody|neither|nor|cannot|can't|won't|don't|didn't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't)\b", re.I)
_HOPE = re.compile(r"\b(hopeless|worthless|pointless|meaningless|useless|helpless|nothing|nobody|empty|numb|tired|exhausted|done|over|end)\b", re.I)


def normalise_slang(t):
    if not isinstance(t, str):
        return ""
    for pat, rep in _COMPILED:
        t = pat.sub(rep, t)
    return t


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = normalise_slang(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not _NLTK:
        return text
    toks = [t for t in word_tokenize(text) if t not in _STOP and len(t) > 1]
    return " ".join(_LEM.lemmatize(t) for t in toks)


HANDCRAFTED_NAMES = [
    "word_count", "char_count", "avg_word_length",
    "exclamation_count", "question_count", "exclamation_density", "question_density",
    "ellipsis_count", "uppercase_ratio",
    "first_person_count", "first_person_density",
    "negation_count", "negation_density",
    "hopelessness_count", "hopelessness_density",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]


def extract_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in HANDCRAFTED_NAMES}
    text = normalise_slang(text)
    words = text.split(); n = max(len(words), 1); nc = len(text)
    v = _VADER.polarity_scores(text)
    return {
        "word_count": n, "char_count": nc, "avg_word_length": np.mean([len(w) for w in words]),
        "exclamation_count": text.count("!"), "question_count": text.count("?"),
        "exclamation_density": text.count("!")/n, "question_density": text.count("?")/n,
        "ellipsis_count": text.count("..."), "uppercase_ratio": sum(c.isupper() for c in text)/max(nc,1),
        "first_person_count": len(_FP.findall(text)), "first_person_density": len(_FP.findall(text))/n,
        "negation_count": len(_NEG.findall(text)), "negation_density": len(_NEG.findall(text))/n,
        "hopelessness_count": len(_HOPE.findall(text)), "hopelessness_density": len(_HOPE.findall(text))/n,
        "vader_neg": v["neg"], "vader_neu": v["neu"], "vader_pos": v["pos"], "vader_compound": v["compound"],
    }


def features_matrix(texts, for_mnb=False):
    df = pd.DataFrame([extract_features(t) for t in texts])
    cols = [c for c in HANDCRAFTED_NAMES if not (for_mnb and c in {"vader_neg","vader_pos","vader_compound"})]
    return df[cols].values.astype(np.float32)
