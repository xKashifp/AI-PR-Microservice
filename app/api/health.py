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
        "embedding_model": "gemini-text-embedding-004",
        "sentiment_model": "lexicon-v1",
        "vector_db": "FAISS",
        "metrics": metrics
    }
