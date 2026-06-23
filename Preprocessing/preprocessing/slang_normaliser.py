"""
Slang and algospeak normalisation layer.

Why this exists
---------------
Social media users routinely replace crisis words with coded substitutes so
that automated moderation does not catch them. This behaviour is called
"algospeak". The clearest example is "unalive", which has become a widely used
stand in for "suicide" or "kill", and there is a whole family of related forms
such as "kms", "kys", "sewer slide" and leetspeak spellings like "5u1c1d3".
This is documented in peer reviewed work on TikTok content moderation
(Steen, Yurechko and Klug, 2023) and in wider reporting on algospeak.

A plain bag of words model never sees these signals, because the exact tokens
are missing from its vocabulary. A post whose only crisis marker is "wanna kms"
produces almost no useful features and slips through. The original pipeline
flagged this as a known weakness but only handled it inside the demo app, not
during training. This module moves that handling into the training pipeline so
that the model actually learns from these signals.

What it does
------------
Before any other cleaning, it rewrites a known set of coded and informal terms
into their plain equivalents. It also expands common informal contractions so
that the sentiment and negation features read them correctly. The mapping is
deliberately conservative and only covers well established terms, so it does
not invent meaning that was not there.
"""
import re

# ---------------------------------------------------------------------------
# Crisis algospeak and coded substitutes -> plain wording
# These are the terms most likely to carry suicide signal while being invisible
# to a vocabulary built from cleaned text.
# ---------------------------------------------------------------------------
CRISIS_SLANG = {
    r"\bunalive\s+myself\b": "kill myself",
    r"\bunalive\b": "suicide",
    r"\bun-?alive\b": "suicide",
    r"\bkms\b": "kill myself",
    r"\bkys\b": "kill yourself",
    r"\bsewer\s*slide\b": "suicide",
    r"\bsuwerslide\b": "suicide",
    r"\bself\s*delete\b": "kill myself",
    r"\bself-?deletion\b": "suicide",
    r"\bend\s+it\s+all\b": "suicide",
    r"\bketch\b": "suicide",          # documented TikTok substitute
    # Leetspeak spellings of suicide
    r"\b5u1c1d3\b": "suicide",
    r"\bsu1c1de\b": "suicide",
    r"\bsewerslide\b": "suicide",
}

# ---------------------------------------------------------------------------
# Common informal contractions and chat shorthand -> plain English.
# These help the negation, first person and sentiment features fire correctly.
# ---------------------------------------------------------------------------
INFORMAL = {
    r"\bwanna\b": "want to",
    r"\bgonna\b": "going to",
    r"\bgotta\b": "got to",
    r"\bgimme\b": "give me",
    r"\blemme\b": "let me",
    r"\btryna\b": "trying to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\bdunno\b": "do not know",
    r"\bidk\b": "i do not know",
    r"\bcant\b": "cannot",
    r"\bdont\b": "do not",
    r"\bwont\b": "will not",
    r"\bisnt\b": "is not",
    r"\bdidnt\b": "did not",
    r"\bwouldnt\b": "would not",
    r"\bcouldnt\b": "could not",
    r"\bshouldnt\b": "should not",
    r"\bim\b": "i am",
    r"\bive\b": "i have",
    r"\bid\b": "i would",
    r"\byoure\b": "you are",
    r"\btheyre\b": "they are",
    r"\bthats\b": "that is",
    r"\bdoesnt\b": "does not",
    r"\bhavent\b": "have not",
    r"\bhasnt\b": "has not",
    r"\baint\b": "is not",
}

# Compile once for speed. Order matters: crisis terms first, then informal.
_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in CRISIS_SLANG.items()]
_COMPILED += [(re.compile(p, re.IGNORECASE), r) for p, r in INFORMAL.items()]


def normalise_slang(text: str) -> str:
    """Rewrite known algospeak and informal terms into plain wording.

    Case and punctuation outside the matched terms are left untouched, so the
    handcrafted features that rely on raw text still work as before.
    """
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text


if __name__ == "__main__":
    samples = [
        "been struggling a lot lately, honestly just wanna kms, nothing feels worth it anymore",
        "thinking about sewer slide, idk what to do",
        "lol just unalive me already im so done",
        "had the best pizza of my life today",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", normalise_slang(s))
        print()
