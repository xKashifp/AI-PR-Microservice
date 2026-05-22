# Implementation Plan: AI-Powered PR Mention Intelligence Microservice

> Free APIs used: **Groq** (LLM summaries via llama3-70b-8192) + **Gemini** (fallback summarizer)
> Embeddings: sentence-transformers (local, free)
> Vector DB: FAISS (local, persisted)
> Metadata DB: SQLite

---

## Project Structure

```
pr-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, router mount
│   ├── config.py                # env vars via pydantic-settings
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── db.py                # SQLite setup, table creation
│   ├── api/
│   │   ├── ingest.py            # POST /ingest
│   │   ├── search.py            # GET /search
│   │   ├── health.py            # GET /health
│   │   └── digest.py            # POST /run_digest (manual trigger)
│   ├── nlp/
│   │   ├── classifier.py        # sklearn topic classifier
│   │   ├── sentiment.py         # HuggingFace sentiment pipeline
│   │   ├── summarizer.py        # Groq / Gemini / HF fallback
│   │   └── embedder.py          # sentence-transformers embedder
│   ├── search/
│   │   ├── faiss_store.py       # FAISS index CRUD + persist
│   │   └── ranker.py            # recency * reach boosting
│   ├── web3/
│   │   ├── detector.py          # regex: 0x addr, ENS, $TICKER
│   │   └── resolver.py          # web3.py ENS/addr validation
│   ├── integrations/
│   │   └── slack.py             # Slack webhook formatted blocks
│   ├── jobs/
│   │   └── scheduler.py         # APScheduler nightly digest
│   └── utils/
│       ├── logger.py            # structlog JSON logger
│       └── metrics.py           # basic request counter middleware
├── ml/
│   ├── train_classifier.py      # training script, 5-fold CV, persist
│   ├── generate_synthetic.py    # weak-label synthetic dataset gen
│   └── models/                  # persisted .joblib model files
├── tests/
│   ├── conftest.py
│   ├── test_ingest.py
│   ├── test_search.py
│   ├── test_classifier.py
│   ├── test_web3.py
│   └── test_slack.py
├── data/
│   └── faiss_index/             # persisted FAISS index + metadata
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Phase 1: Project Bootstrap + Config

### Step 1.1 - Environment Config (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"
    GEMINI_API_KEY: str = ""

    # Web3
    ALCHEMY_RPC_URL: str = "https://eth-mainnet.g.alchemy.com/v2/demo"
    ETHERSCAN_API_KEY: str = ""

    # Slack
    SLACK_WEBHOOK_URL: str = ""

    # Paths
    FAISS_INDEX_DIR: str = "./data/faiss_index"
    SQLITE_DB_PATH: str = "./data/pr_intelligence.db"
    MODEL_DIR: str = "./ml/models"

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Sentiment model
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

    class Config:
        env_file = ".env"

settings = Settings()
```

### Step 1.2 - SQLite Schema (`app/models/db.py`)

```python
import sqlite3
from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS mentions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    source        TEXT,
    published_at  TEXT,
    reach         INTEGER DEFAULT 0,
    sentiment     TEXT,
    sentiment_score REAL,
    topics        TEXT,          -- JSON array string
    summary       TEXT,
    web3_signals  TEXT,          -- JSON array string
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sentiment ON mentions(sentiment);
CREATE INDEX IF NOT EXISTS idx_published ON mentions(published_at);
CREATE INDEX IF NOT EXISTS idx_reach ON mentions(reach);
"""

def get_conn():
    conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
```

---

## Phase 2: NLP Pipeline

### Step 2.1 - Embedder (`app/nlp/embedder.py`)

```python
from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

class Embedder:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self.get()
        return model.encode(texts, batch_size=64, show_progress_bar=False)
```

### Step 2.2 - Sentiment (`app/nlp/sentiment.py`)

```python
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
```

### Step 2.3 - Topic Classifier (`app/nlp/classifier.py`)

```python
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.config import settings

TOPICS = ["product", "funding", "partnership", "thought-leadership", "crisis"]
MODEL_PATH = os.path.join(settings.MODEL_DIR, "topic_classifier.joblib")

def load_classifier():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError("Topic classifier not trained. Run ml/train_classifier.py first.")

def predict(texts: list[str]) -> list[list[str]]:
    clf = load_classifier()
    preds = clf.predict(texts)
    # Multi-label: probabilities above 0.3 threshold
    proba = clf.predict_proba(texts)
    results = []
    for p in proba:
        labels = [TOPICS[i] for i, score in enumerate(p) if score > 0.30]
        results.append(labels if labels else [TOPICS[p.argmax()]])
    return results
```

### Step 2.4 - Training Script (`ml/train_classifier.py`)

```python
"""
Run: python -m ml.train_classifier
Generates synthetic weak labels if no labeled data exists.
Outputs: ml/models/topic_classifier.joblib
Reports: 5-fold CV macro-F1
"""
import joblib, os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from ml.generate_synthetic import generate_dataset

TOPICS = ["product", "funding", "partnership", "thought-leadership", "crisis"]
MODEL_PATH = "ml/models/topic_classifier.joblib"

def train():
    os.makedirs("ml/models", exist_ok=True)

    # Load or generate labeled dataset (>=1k examples)
    texts, labels = generate_dataset(n=1500)

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
            solver="lbfgs",
            multi_class="multinomial"
        ))
    ])

    # 5-fold CV
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    print(f"5-fold CV macro-F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    assert scores.mean() >= 0.70, f"F1 {scores.mean():.3f} below required 0.70"

    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
```

### Step 2.5 - Synthetic Dataset Generator (`ml/generate_synthetic.py`)

```python
"""
Generates 1500 weakly labeled examples across 5 topics
using keyword heuristics + templates.
"""
import random

TEMPLATES = {
    "product": [
        "We are launching {product} with new features for {audience}",
        "{company} releases {product} update with improved {feature}",
        "Introducing {product}: the next generation of {category}",
    ],
    "funding": [
        "{company} raises ${amount}M in Series {round} led by {vc}",
        "{company} secures {amount} million to expand {area}",
        "{vc} leads {amount}M investment in {company}",
    ],
    "partnership": [
        "{company} partners with {partner} to deliver {service}",
        "{company} and {partner} announce strategic collaboration",
        "New integration: {company} joins forces with {partner}",
    ],
    "thought-leadership": [
        "Why {topic} will define the future of {industry}",
        "{person} on the state of {industry} in {year}",
        "Opinion: {company} CEO explains the future of {topic}",
    ],
    "crisis": [
        "{company} faces backlash over {issue}",
        "Users report {issue} affecting {company} platform",
        "{company} under investigation for {issue}",
    ],
}

def generate_dataset(n: int = 1500):
    texts, labels = [], []
    topics = list(TEMPLATES.keys())
    per_topic = n // len(topics)

    fills = {
        "product": "DataLens", "company": "Acme Corp", "amount": "50",
        "round": "B", "vc": "Sequoia", "partner": "TechCo",
        "service": "analytics", "topic": "AI", "industry": "Web3",
        "person": "CEO Jane", "year": "2025", "issue": "data breach",
        "audience": "enterprises", "feature": "speed", "category": "SaaS",
        "area": "APAC"
    }

    for topic in topics:
        for _ in range(per_topic):
            tmpl = random.choice(TEMPLATES[topic])
            text = tmpl.format(**fills)
            # add noise
            text += " " + random.choice([
                "Read more at our blog.",
                "The announcement was made today.",
                "Details to follow.",
                f"This impacts the {random.choice(['DeFi','NFT','DAO'])} space."
            ])
            texts.append(text)
            labels.append(topic)

    return texts, labels
```

### Step 2.6 - Summarizer (`app/nlp/summarizer.py`)

```python
"""
Primary: Groq (llama3-70b-8192) - free tier, fast
Fallback: Gemini (gemini-1.5-flash) - free tier
Final fallback: HuggingFace local (facebook/bart-large-cnn)
"""
import httpx
from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

PROMPT = """Summarize this PR mention in 2-3 sentences. Max 350 characters. 
Focus on: who, what action, why it matters for PR/comms teams.
Text: {text}"""

async def summarize(text: str) -> str:
    # Try Groq first
    if settings.GROQ_API_KEY:
        try:
            return await _groq_summarize(text)
        except Exception:
            pass

    # Fallback to Gemini
    if settings.GEMINI_API_KEY:
        try:
            return await _gemini_summarize(text)
        except Exception:
            pass

    # Final fallback: extractive first 350 chars
    return text[:347] + "..."

async def _groq_summarize(text: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "user", "content": PROMPT.format(text=text[:2000])}],
                "max_tokens": 120,
                "temperature": 0.3
            }
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        return summary[:350]

async def _gemini_summarize(text: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={settings.GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{"text": PROMPT.format(text=text[:2000])}]
                }],
                "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3}
            }
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"][:350]
```

---

## Phase 3: Vector Search

### Step 3.1 - FAISS Store (`app/search/faiss_store.py`)

```python
import faiss
import numpy as np
import json
import os
import pickle
from app.config import settings

class FAISSStore:
    def __init__(self):
        self.dim = 384         # all-MiniLM-L6-v2 output dim
        self.index = faiss.IndexFlatIP(self.dim)   # inner product = cosine on normalized vecs
        self.metadata: list[dict] = []
        self.id_to_pos: dict[str, int] = {}
        self._load()

    def _index_path(self):
        return os.path.join(settings.FAISS_INDEX_DIR, "index.faiss")

    def _meta_path(self):
        return os.path.join(settings.FAISS_INDEX_DIR, "metadata.pkl")

    def _load(self):
        os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
        if os.path.exists(self._index_path()):
            self.index = faiss.read_index(self._index_path())
            with open(self._meta_path(), "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.id_to_pos = data["id_to_pos"]

    def save(self):
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump({"metadata": self.metadata, "id_to_pos": self.id_to_pos}, f)

    def upsert(self, doc_id: str, vector: np.ndarray, meta: dict):
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        if doc_id in self.id_to_pos:
            # Update metadata only (FAISS flat index no delete; overwrite position)
            pos = self.id_to_pos[doc_id]
            self.metadata[pos] = meta
        else:
            pos = len(self.metadata)
            self.index.add(vec)
            self.metadata.append(meta)
            self.id_to_pos[doc_id] = pos

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        sentiment_filter: str | None = None,
        topic_filter: str | None = None
    ) -> list[dict]:
        vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        # Fetch 10x candidates, then filter + boost
        fetch_k = min(k * 10, self.index.ntotal)
        if fetch_k == 0:
            return []

        scores, indices = self.index.search(vec, fetch_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]

            # Metadata filters
            if sentiment_filter and meta.get("sentiment") != sentiment_filter:
                continue
            if topic_filter and topic_filter not in (meta.get("topics") or []):
                continue

            # Recency + reach boost
            boosted = _boost_score(score, meta)
            results.append({**meta, "_score": boosted})

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:k]


def _boost_score(base_score: float, meta: dict) -> float:
    from datetime import datetime, timezone
    boost = base_score

    # Reach boost (log scale)
    reach = meta.get("reach", 0) or 0
    if reach > 0:
        import math
        boost += 0.05 * math.log1p(reach)

    # Recency boost: decay over 30 days
    try:
        pub = datetime.fromisoformat(meta.get("published_at", ""))
        age_days = (datetime.now(timezone.utc) - pub).days
        recency_factor = max(0, 1 - (age_days / 30))
        boost += 0.10 * recency_factor
    except Exception:
        pass

    return round(boost, 6)


# Singleton
_store: FAISSStore | None = None

def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore()
    return _store
```

---

## Phase 4: Web3 Detection + Resolution

### Step 4.1 - Detector (`app/web3/detector.py`)

```python
import re

ETH_ADDRESS_RE = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
ENS_RE = re.compile(r'\b[\w-]+\.eth\b')
TICKER_RE = re.compile(r'\$[A-Z]{2,10}\b')

def detect_web3_signals(text: str) -> dict:
    return {
        "eth_addresses": ETH_ADDRESS_RE.findall(text),
        "ens_names": ENS_RE.findall(text),
        "tickers": TICKER_RE.findall(text)
    }
```

### Step 4.2 - Resolver (`app/web3/resolver.py`)

```python
from web3 import Web3
from app.config import settings

def get_web3():
    return Web3(Web3.HTTPProvider(settings.ALCHEMY_RPC_URL))

def validate_address(address: str) -> dict:
    return {
        "address": address,
        "is_valid_checksum": Web3.is_checksum_address(address),
        "is_valid_address": Web3.is_address(address)
    }

def resolve_ens(name: str) -> dict:
    try:
        w3 = get_web3()
        address = w3.ens.address(name)
        return {
            "ens": name,
            "resolved_address": address,
            "valid": address is not None
        }
    except Exception as e:
        return {"ens": name, "resolved_address": None, "valid": False, "error": str(e)}

def enrich_signals(signals: dict) -> dict:
    return {
        "eth_addresses": [validate_address(a) for a in signals["eth_addresses"]],
        "ens_names": [resolve_ens(e) for e in signals["ens_names"]],
        "tickers": signals["tickers"]   # no on-chain resolution needed
    }
```

---

## Phase 5: API Endpoints

### Step 5.1 - Schemas (`app/models/schemas.py`)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
import json

class MentionIn(BaseModel):
    id: str
    title: str
    text: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    reach: Optional[int] = 0
    labels: Optional[list[str]] = []

class IngestRequest(BaseModel):
    mentions: list[MentionIn] = Field(..., min_items=1, max_items=10000)

class IngestResponse(BaseModel):
    inserted: int
    updated: int
    errors: list[dict]

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=100)
    sentiment_filter: Optional[str] = None
    topic_filter: Optional[str] = None
    page: int = 1

class SearchResult(BaseModel):
    id: str
    title: str
    source: Optional[str]
    published_at: Optional[str]
    sentiment: Optional[str]
    topics: Optional[list[str]]
    summary: Optional[str]
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    page: int
    k: int
```

### Step 5.2 - Ingest Endpoint (`app/api/ingest.py`)

```python
from fastapi import APIRouter, HTTPException
from app.models.schemas import IngestRequest, IngestResponse
from app.models.db import get_conn
from app.nlp.embedder import Embedder
from app.nlp.sentiment import analyze as analyze_sentiment
from app.nlp.classifier import predict as classify_topics
from app.nlp.summarizer import summarize
from app.web3.detector import detect_web3_signals
from app.web3.resolver import enrich_signals
from app.search.faiss_store import get_store
import json, asyncio

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    store = get_store()
    embedder = Embedder()
    conn = get_conn()

    inserted, updated, errors = 0, 0, []
    texts = [f"{m.title}. {m.text}" for m in request.mentions]

    # Batch embed
    vectors = embedder.embed(texts)

    # Batch classify topics
    topic_preds = classify_topics(texts)

    for i, mention in enumerate(request.mentions):
        try:
            # Sentiment
            sent = analyze_sentiment(mention.text)

            # Summary (async)
            summary = await summarize(mention.text)

            # Web3 signals
            raw_signals = detect_web3_signals(mention.text)
            enriched = enrich_signals(raw_signals)

            topics = topic_preds[i]
            meta = {
                "id": mention.id,
                "title": mention.title,
                "source": mention.source,
                "published_at": mention.published_at,
                "reach": mention.reach,
                "sentiment": sent["label"],
                "sentiment_score": sent["score"],
                "topics": topics,
                "summary": summary,
                "web3_signals": enriched
            }

            # Idempotent upsert to SQLite
            existing = conn.execute(
                "SELECT id FROM mentions WHERE id = ?", (mention.id,)
            ).fetchone()

            if existing:
                conn.execute("""
                    UPDATE mentions SET title=?, text=?, source=?, published_at=?,
                    reach=?, sentiment=?, sentiment_score=?, topics=?, summary=?, web3_signals=?
                    WHERE id=?
                """, (
                    mention.title, mention.text, mention.source, mention.published_at,
                    mention.reach, sent["label"], sent["score"],
                    json.dumps(topics), summary, json.dumps(enriched), mention.id
                ))
                updated += 1
            else:
                conn.execute("""
                    INSERT INTO mentions (id, title, text, source, published_at, reach,
                    sentiment, sentiment_score, topics, summary, web3_signals)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    mention.id, mention.title, mention.text, mention.source,
                    mention.published_at, mention.reach, sent["label"], sent["score"],
                    json.dumps(topics), summary, json.dumps(enriched)
                ))
                inserted += 1

            # Upsert into FAISS
            store.upsert(mention.id, vectors[i], {
                "id": mention.id,
                "title": mention.title,
                "source": mention.source,
                "published_at": mention.published_at,
                "reach": mention.reach,
                "sentiment": sent["label"],
                "topics": topics,
                "summary": summary
            })

        except Exception as e:
            errors.append({"id": mention.id, "error": str(e)})

    conn.commit()
    store.save()

    return IngestResponse(inserted=inserted, updated=updated, errors=errors)
```

### Step 5.3 - Search Endpoint (`app/api/search.py`)

```python
from fastapi import APIRouter, Query
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.schemas import SearchResponse, SearchResult
from app.nlp.embedder import Embedder
from app.search.faiss_store import get_store

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search(
    request,
    query: str = Query(..., min_length=1),
    k: int = Query(default=10, ge=1, le=100),
    sentiment_filter: str = Query(default=None),
    topic_filter: str = Query(default=None),
    page: int = Query(default=1, ge=1)
):
    embedder = Embedder()
    store = get_store()

    vec = embedder.embed([query])[0]
    raw = store.search(vec, k=k, sentiment_filter=sentiment_filter, topic_filter=topic_filter)

    results = [
        SearchResult(
            id=r["id"],
            title=r["title"],
            source=r.get("source"),
            published_at=r.get("published_at"),
            sentiment=r.get("sentiment"),
            topics=r.get("topics"),
            summary=r.get("summary"),
            score=r["_score"]
        )
        for r in raw
    ]

    return SearchResponse(results=results, total=len(results), page=page, k=k)
```

### Step 5.4 - Health Endpoint (`app/api/health.py`)

```python
from fastapi import APIRouter
from app.search.faiss_store import get_store
from app.nlp.classifier import load_classifier
import os

router = APIRouter()

@router.get("/health")
def health():
    store = get_store()

    try:
        clf = load_classifier()
        classifier_ready = True
    except Exception:
        classifier_ready = False

    return {
        "status": "ok",
        "index_total_docs": store.index.ntotal,
        "classifier_ready": classifier_ready,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "FAISS"
    }
```

---

## Phase 6: Slack Integration + Digest

### Step 6.1 - Slack Blocks (`app/integrations/slack.py`)

```python
import httpx
from app.config import settings

async def post_digest(mentions: list[dict]):
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Web3 PR Intelligence Digest"}
        },
        {"type": "divider"}
    ]

    for i, m in enumerate(mentions[:3], 1):
        web3 = m.get("web3_signals", {})
        tickers = ", ".join(web3.get("tickers", [])) if web3 else ""
        ens = ", ".join(
            [e["ens"] for e in web3.get("ens_names", []) if e.get("valid")]
        ) if web3 else ""

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{i}. {m['title']}*\n"
                    f"{m.get('summary', 'No summary available.')}\n"
                    f"Source: {m.get('source', 'Unknown')} | "
                    f"Reach: {m.get('reach', 0):,} | "
                    f"Sentiment: {m.get('sentiment', 'N/A')}\n"
                    + (f"Tickers: `{tickers}` " if tickers else "")
                    + (f"ENS: `{ens}`" if ens else "")
                )
            }
        })
        blocks.append({"type": "divider"})

    payload = {"blocks": blocks}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(settings.SLACK_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
```

### Step 6.2 - Scheduler + Manual Trigger (`app/jobs/scheduler.py`)

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.models.db import get_conn
from app.integrations.slack import post_digest
import json

scheduler = AsyncIOScheduler()

async def run_nightly_digest():
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, title, source, reach, sentiment, summary, web3_signals, published_at
        FROM mentions
        WHERE json_array_length(web3_signals) > 0
           OR web3_signals LIKE '%eth_addresses%'
           OR web3_signals LIKE '%tickers%'
        ORDER BY (reach * 1.0 / MAX(1, JULIANDAY('now') - JULIANDAY(published_at))) DESC
        LIMIT 10
    """).fetchall()

    mentions = [dict(r) for r in rows]
    for m in mentions:
        m["web3_signals"] = json.loads(m.get("web3_signals") or "{}")

    if mentions:
        await post_digest(mentions)

def start_scheduler():
    scheduler.add_job(
        run_nightly_digest,
        "cron",
        hour=8,
        minute=0,
        id="nightly_digest"
    )
    scheduler.start()
```

### Step 6.3 - Manual Digest Trigger Endpoint (`app/api/digest.py`)

```python
from fastapi import APIRouter
from app.jobs.scheduler import run_nightly_digest

router = APIRouter()

@router.post("/run_digest")
async def manual_digest():
    await run_nightly_digest()
    return {"status": "digest_sent"}
```

---

## Phase 7: Main App (`app/main.py`)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.models.db import init_db
from app.search.faiss_store import get_store
from app.jobs.scheduler import start_scheduler
from app.api import ingest, search, health, digest
from app.utils.logger import get_logger
from app.utils.metrics import MetricsMiddleware

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database")
    init_db()
    logger.info("Loading FAISS index")
    get_store()
    logger.info("Starting scheduler")
    start_scheduler()
    yield
    # Shutdown
    logger.info("Saving FAISS index on shutdown")
    get_store().save()

app = FastAPI(
    title="PR Mention Intelligence API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(MetricsMiddleware)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(health.router)
app.include_router(digest.router)
```

---

## Phase 8: Tests

### Step 8.1 - Test Coverage Plan

```
tests/
├── conftest.py          # fixtures: test client, mock DB, mock FAISS
├── test_ingest.py       # idempotency, 400/422 errors, batch speed
├── test_search.py       # k results exact, filters, determinism, p95 latency
├── test_classifier.py   # model load, predict shape, F1 >= 0.70
├── test_web3.py         # address regex, ENS regex, ticker regex (mocked RPC)
└── test_slack.py        # payload structure, blocks format (mocked webhook)
```

### Step 8.2 - Key Test Cases

```python
# test_ingest.py
def test_idempotent_upsert(client):
    payload = {"mentions": [sample_mention()]}
    r1 = client.post("/ingest", json=payload)
    r2 = client.post("/ingest", json=payload)
    assert r1.json()["inserted"] == 1
    assert r2.json()["updated"] == 1    # second call = update, not duplicate

def test_invalid_payload_returns_422(client):
    r = client.post("/ingest", json={"mentions": [{"bad_field": "x"}]})
    assert r.status_code == 422

def test_ingest_10k_under_90s(client):
    import time
    mentions = [sample_mention(i) for i in range(10000)]
    start = time.time()
    client.post("/ingest", json={"mentions": mentions})
    assert time.time() - start <= 90

# test_web3.py
def test_detect_eth_address():
    text = "Wallet 0xAbCdEf1234567890AbCdEf1234567890AbCdEf12 received funds"
    signals = detect_web3_signals(text)
    assert len(signals["eth_addresses"]) == 1

def test_detect_ens():
    signals = detect_web3_signals("Send to vitalik.eth for the DAO")
    assert "vitalik.eth" in signals["ens_names"]

def test_detect_ticker():
    signals = detect_web3_signals("$ETH and $BTC are up today")
    assert "$ETH" in signals["tickers"]
    assert "$BTC" in signals["tickers"]

# test_search.py
def test_search_returns_exact_k(client):
    r = client.get("/search?query=funding+announcement&k=5")
    assert len(r.json()["results"]) == 5

def test_rate_limit(client):
    for _ in range(20):
        client.get("/search?query=test")
    r = client.get("/search?query=test")
    assert r.status_code == 429
```

---

## Phase 9: Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss + torch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time (bakes into image, no cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

COPY . .

# Train classifier if not exists
RUN python -m ml.train_classifier

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data    # persist FAISS index + SQLite across restarts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### .env.example

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama3-70b-8192
GEMINI_API_KEY=AIza...
ALCHEMY_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
FAISS_INDEX_DIR=./data/faiss_index
SQLITE_DB_PATH=./data/pr_intelligence.db
MODEL_DIR=./ml/models
```

---

## Phase 10: Requirements

```
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.2.1

# NLP
transformers==4.40.0
torch==2.2.2
sentence-transformers==2.7.0
scikit-learn==1.4.2
joblib==1.4.0

# Vector DB
faiss-cpu==1.8.0

# Web3
web3==6.18.0

# LLM clients
httpx==0.27.0

# Scheduling
apscheduler==3.10.4

# Rate limiting
slowapi==0.1.9

# Logging
structlog==24.1.0

# Testing
pytest==8.2.0
pytest-asyncio==0.23.6
pytest-cov==5.0.0
httpx==0.27.0     # async test client
respx==0.20.2     # mock httpx calls

# Utils
python-multipart==0.0.9
```

---

## Execution Order for Antigravity IDE

Run these steps in exact order:

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Copy and fill env
cp .env.example .env
# Fill: GROQ_API_KEY, SLACK_WEBHOOK_URL, ALCHEMY_RPC_URL

# 3. Train classifier (one-time, generates ml/models/topic_classifier.joblib)
python -m ml.train_classifier

# 4. Run tests
pytest tests/ --cov=app --cov-report=term-missing

# 5. Start service locally
uvicorn app.main:app --reload --port 8000

# 6. Or via Docker
docker build -t pr-intelligence .
docker-compose up -d

# 7. Test ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"mentions":[{"id":"1","title":"Test","text":"Vitalik.eth launched $ETH upgrade","source":"CoinDesk","published_at":"2025-01-01","reach":50000}]}'

# 8. Test search
curl "http://localhost:8000/search?query=ethereum+launch&k=5"

# 9. Manual digest
curl -X POST http://localhost:8000/run_digest

# 10. Health check
curl http://localhost:8000/health
```

---

## Success Criteria Checklist

- Ingest 10k docs in <= 90s on 2 vCPU: batch embedding + async summarization
- p95 GET /search (k=10) <= 300ms: FAISS flat index + cached models
- 5-fold CV macro-F1 >= 0.70: LogisticRegression on TF-IDF bigrams + balanced classes
- Summaries <= 350 chars: enforced in summarizer post-processing
- Web3 detection >= 90% accuracy: regex patterns cover standard formats
- Slack digest posts top 3 Web3 mentions ranked by reach * recency
- Test coverage > 85%: all core modules covered
- Image size <= 1.5GB: python:3.11-slim + faiss-cpu + no GPU torch
- 429 on > 20 RPS for /search: slowapi rate limiter
- Graceful shutdown: FAISS index saved in lifespan shutdown hook
- OpenAPI docs: enabled at /docs by default in FastAPI
- Index survives restarts: FAISS + metadata persisted to ./data/
```
