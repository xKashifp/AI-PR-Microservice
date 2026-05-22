"""
Run: python -m ml.train_classifier
Generates synthetic weak labels if no labeled data exists.
Outputs: ml/models/topic_classifier.joblib
Reports: 5-fold CV macro-F1
"""
import joblib
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from ml.generate_synthetic import generate_dataset

TOPICS = ["product", "funding", "partnership", "thought-leadership", "crisis"]
MODEL_PATH = "ml/models/topic_classifier.joblib"

def train():
    os.makedirs("ml/models", exist_ok=True)

    print("Generating synthetic dataset...")
    texts, labels = generate_dataset(n=1500)
    print(f"Dataset size: {len(texts)} examples across {len(set(labels))} topics")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    print("Running 5-fold cross-validation...")
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    print(f"5-fold CV macro-F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    assert scores.mean() >= 0.70, f"F1 {scores.mean():.3f} below required 0.70"

    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipeline

if __name__ == "__main__":
    train()
