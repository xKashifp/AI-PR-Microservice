from transformers import pipeline
from app.config import settings
from functools import lru_cache

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    return pipeline(
        "text-classification",
        model=settings.SENTIMENT_MODEL,
        truncation=True,
        max_length=512
    )

def analyze(text: str) -> dict:
    pipe = get_sentiment_pipeline()
    result = pipe(text[:512])[0]
    return {
        "label": result["label"].lower(),   # "positive" / "negative"
        "score": round(result["score"], 4)
    }
