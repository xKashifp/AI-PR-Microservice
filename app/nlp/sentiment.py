import re
from functools import lru_cache

# Pure Python lexicon-based sentiment analysis to avoid loading heavy transformer weights
POSITIVE_WORDS = {
    "secure", "secured", "raise", "raised", "raising", "partner", "partnership",
    "collaboration", "success", "successful", "announces", "announcement", "launch",
    "launches", "launched", "innovative", "growth", "expansion", "expand",
    "funding", "investment", "invest", "support", "gain", "profit", "bullish"
}

NEGATIVE_WORDS = {
    "hack", "hacked", "breach", "leak", "leaked", "backlash", "investigation",
    "investigate", "probe", "crisis", "fail", "failure", "failed", "drop", "dropped",
    "loss", "lost", "regulation", "regulatory", "lawsuit", "sue", "sued", "exploit",
    "exploited", "vulnerability", "risk", "risky", "bearish"
}

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    # Return a mock to maintain compatibility with startup preloader
    return None

def analyze(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    if pos_count > neg_count:
        label = "positive"
        score = min(1.0, 0.5 + 0.1 * (pos_count - neg_count))
    elif neg_count > pos_count:
        label = "negative"
        score = min(1.0, 0.5 + 0.1 * (neg_count - pos_count))
    else:
        label = "neutral"
        score = 0.5
        
    return {
        "label": label,
        "score": round(score, 4)
    }
