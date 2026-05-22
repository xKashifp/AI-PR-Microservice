from fastapi import APIRouter
from app.search.faiss_store import get_store
from app.nlp.classifier import load_classifier
from app.utils.metrics import get_metrics
import os

router = APIRouter()

@router.get("/health", tags=["Health"])
def health():
    store = get_store()

    try:
        load_classifier()
        classifier_ready = True
    except Exception:
        classifier_ready = False

    metrics = get_metrics()

    return {
        "status": "ok",
        "index_total_docs": store.index.ntotal,
        "classifier_ready": classifier_ready,
        "embedding_model": "all-MiniLM-L6-v2",
        "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "vector_db": "FAISS",
        "metrics": metrics
    }
