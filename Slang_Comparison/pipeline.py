"""
Self contained preprocessing for the slang comparison.

This rebuilds the cleaning, feature extraction and the exact four way split, with
slang normalisation as a switch. It is kept independent so the slang off branch
can be produced without touching the slang on artifacts that are already saved.

The split depends only on the raw data and the seed, so train and validation are
identical to the main pipeline. Only the cleaning and the features change when
the slang switch is flipped.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _STOP = set(stopwords.words("english")); _LEM = WordNetLemmatizer(); _NLTK = True
except Exception:
    _NLTK = False

_VADER = SentimentIntensityAnalyzer()
RANDOM_SEED = 42

_CRISIS = {
    r"\bunalive\s+myself\b": "kill myself", r"\bunalive\b": "suicide", r"\bun-?alive\b": "suicide",
    r"\bkms\b": "kill myself", r"\bkys\b": "kill yourself", r"\bsewer\s*slide\b": "suicide",
    r"\bsewerslide\b": "suicide", r"\bself\s*delete\b": "kill myself", r"\bend\s+it\s+all\b": "suicide",
}
_INFORMAL = {r"\bwanna\b": "want to", r"\bgonna\b": "going to", r"\bgotta\b": "got to",
    r"\bidk\b": "i do not know", r"\bcant\b": "cannot", r"\bdont\b": "do not",
    r"\bim\b": "i am", r"\bive\b": "i have"}
_COMPILED = [(re.compile(p, re.I), r) for p, r in {**_CRISIS, **_INFORMAL}.items()]

_FP = re.compile(r"\b(i|me|my|myself|mine|i'm|i've|i'll|i'd)\b", re.I)
_NEG = re.compile(r"\b(not|no|never|nothing|nobody|neither|nor|cannot|can't|won't|don't|didn't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't)\b", re.I)
_HOPE = re.compile(r"\b(hopeless|worthless|pointless|meaningless|useless|helpless|nothing|nobody|empty|numb|tired|exhausted|done|over|end)\b", re.I)

HANDCRAFTED_NAMES = [
    "word_count", "char_count", "avg_word_length", "exclamation_count", "question_count",
    "exclamation_density", "question_density", "ellipsis_count", "uppercase_ratio",
    "first_person_count", "first_person_density", "negation_count", "negation_density",
    "hopelessness_count", "hopelessness_density", "vader_neg", "vader_neu", "vader_pos", "vader_compound",
]
_MNB_DROP = {"vader_neg", "vader_pos", "vader_compound"}


def normalise_slang(t):
    if not isinstance(t, str):
        return ""
    for pat, rep in _COMPILED:
        t = pat.sub(rep, t)
    return t


def clean_text(text, use_slang):
    if not isinstance(text, str):
        return ""
    if use_slang:
        text = normalise_slang(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not _NLTK:
        return text
    toks = [t for t in word_tokenize(text) if t not in _STOP and len(t) > 1]
    return " ".join(_LEM.lemmatize(t) for t in toks)


def _features_one(text, use_slang):
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in HANDCRAFTED_NAMES}
    if use_slang:
        text = normalise_slang(text)
    w = text.split(); n = max(len(w), 1); nc = len(text); v = _VADER.polarity_scores(text)
    return {
        "word_count": n, "char_count": nc, "avg_word_length": np.mean([len(x) for x in w]),
        "exclamation_count": text.count("!"), "question_count": text.count("?"),
        "exclamation_density": text.count("!")/n, "question_density": text.count("?")/n,
        "ellipsis_count": text.count("..."), "uppercase_ratio": sum(c.isupper() for c in text)/max(nc,1),
        "first_person_count": len(_FP.findall(text)), "first_person_density": len(_FP.findall(text))/n,
        "negation_count": len(_NEG.findall(text)), "negation_density": len(_NEG.findall(text))/n,
        "hopelessness_count": len(_HOPE.findall(text)), "hopelessness_density": len(_HOPE.findall(text))/n,
        "vader_neg": v["neg"], "vader_neu": v["neu"], "vader_pos": v["pos"], "vader_compound": v["compound"],
    }


def features_matrix(texts, use_slang, for_mnb=False, desc="features"):
    try:
        from tqdm import tqdm as _tq
        rows = [_features_one(t, use_slang) for t in _tq(texts, desc=f"  {desc}")]
    except Exception:
        rows = [_features_one(t, use_slang) for t in texts]
    df = pd.DataFrame(rows)
    cols = [c for c in HANDCRAFTED_NAMES if not (for_mnb and c in _MNB_DROP)]
    return df[cols].values.astype(np.float32)


def load_filter_split(raw_csv):
    """Reproduce the exact four way split, return train and val frames."""
    df = pd.read_csv(raw_csv, encoding="latin-1")
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    df = df[df["text"].notna()]
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    df = df[df["text"].astype(str).str.split().str.len() >= 5]
    df = df.drop_duplicates(subset=["text"])
    df["label"] = (df["class"].astype(str).str.strip() == "suicide").astype(int)
    df = df[["text", "label"]].reset_index(drop=True)

    dev, _test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=RANDOM_SEED)
    dev2, _cal = train_test_split(dev, test_size=0.10 / 0.80, stratify=dev["label"], random_state=RANDOM_SEED)
    train, val = train_test_split(dev2, test_size=0.20, stratify=dev2["label"], random_state=RANDOM_SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True)