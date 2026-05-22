import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.config import settings

TOPICS = ["product", "funding", "partnership", "thought-leadership", "crisis"]
MODEL_PATH = os.path.join(settings.MODEL_DIR, "topic_classifier.joblib")

_classifier = None

def load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    if os.path.exists(MODEL_PATH):
        _classifier = joblib.load(MODEL_PATH)
        return _classifier
    raise FileNotFoundError("Topic classifier not trained. Run: python -m ml.train_classifier")

def predict(texts: list) -> list:
    clf = load_classifier()
    proba = clf.predict_proba(texts)
    results = []
    for p in proba:
        labels = [TOPICS[i] for i, score in enumerate(p) if score > 0.30]
        results.append(labels if labels else [TOPICS[list(p).index(max(p))]])
    return results
