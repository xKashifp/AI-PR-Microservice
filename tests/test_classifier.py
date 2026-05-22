import pytest
import os
from app.nlp.classifier import load_classifier, predict, TOPICS
from ml.train_classifier import train


def test_classifier_loads():
    """Classifier must exist or train successfully."""
    from app.config import settings
    model_path = os.path.join(settings.MODEL_DIR, "topic_classifier.joblib")
    if not os.path.exists(model_path):
        train()
    clf = load_classifier()
    assert clf is not None


def test_classifier_predict_shape():
    texts = [
        "Acme Corp raises $50M in Series B led by Sequoia",
        "Company partners with TechCo to deliver analytics",
        "Users report outage affecting the platform",
    ]
    results = predict(texts)
    assert len(results) == 3
    for labels in results:
        assert isinstance(labels, list)
        assert len(labels) >= 1


def test_classifier_predicts_correct_topic():
    texts = [
        "Company raises $100M in Series A funding round led by Sequoia",
    ]
    results = predict(texts)
    assert "funding" in results[0]


def test_classifier_f1_threshold():
    """Verify 5-fold CV macro-F1 >= 0.70."""
    from ml.generate_synthetic import generate_dataset
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts, labels = generate_dataset(n=500)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                   solver="lbfgs"))
    ])
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    assert scores.mean() >= 0.70, f"F1 {scores.mean():.3f} below required 0.70"


def test_topics_list():
    assert len(TOPICS) == 5
    assert "product" in TOPICS
    assert "funding" in TOPICS
    assert "crisis" in TOPICS
