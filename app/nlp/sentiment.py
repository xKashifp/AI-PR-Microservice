from transformers import pipeline
import torch
import gc
from functools import lru_cache
from app.config import settings

@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    pipe = pipeline(
        "text-classification",
        model=settings.SENTIMENT_MODEL,
        device=-1
    )
    try:
        pipe.model = torch.quantization.quantize_dynamic(
            pipe.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception:
        pass
    gc.collect()
    return pipe

def unload_sentiment_pipeline():
    import os
    if os.environ.get("TESTING") == "True":
        return
    get_sentiment_pipeline.cache_clear()
    import gc
    gc.collect()

def analyze(text: str) -> dict:
    pipe = get_sentiment_pipeline()
    res = pipe(text[:512])[0]
    label = res["label"].lower()
    return {
        "label": label,
        "score": round(res["score"], 4)
    }
