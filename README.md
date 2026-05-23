# PR Mention Intelligence Microservice

AI-powered FastAPI service for PR/news mention ingestion, semantic search, Web3 entity detection, and Slack digest automation.

## Quick Start

```bash
# 1. Setup (install deps + train classifier)
cmd /c setup.bat

# 2. Copy and configure environment
copy .env.example .env
# Edit .env: fill GROQ_API_KEY, SLACK_WEBHOOK_URL, etc.

# 3. Start the service
C:\ai_pr_venv\Scripts\uvicorn app.main:app --reload --port 8000

# 4. Open dashboard
# http://localhost:8000/

# 5. OpenAPI docs
# http://localhost:8000/docs
```

## Project Structure

```
app/
├── main.py              # FastAPI app + lifespan
├── config.py            # Pydantic settings
├── api/
│   ├── ingest.py        # POST /ingest
│   ├── search.py        # GET /search
│   ├── health.py        # GET /health
│   └── digest.py        # POST /run_digest
├── nlp/
│   ├── embedder.py      # sentence-transformers
│   ├── sentiment.py     # HuggingFace DistilBERT
│   ├── classifier.py    # scikit-learn topic classifier
│   └── summarizer.py    # Groq → Gemini → extractive fallback
├── search/
│   └── faiss_store.py   # FAISS IndexFlatIP + reach/recency boost
├── web3/
│   ├── detector.py      # regex: ETH addr, ENS, $TICKER
│   └── resolver.py      # web3.py ENS/address validation
├── integrations/
│   └── slack.py         # Slack block UI webhook poster
├── jobs/
│   └── scheduler.py     # APScheduler nightly digest
├── models/
│   ├── db.py            # SQLite schema + connection
│   └── schemas.py       # Pydantic request/response models
├── utils/
│   ├── logger.py        # structlog JSON logger
│   └── metrics.py       # p95 latency middleware
└── templates/
    └── index.html       # Glassmorphic dark dashboard

ml/
├── train_classifier.py  # 5-fold CV macro-F1 ≥ 0.70
└── generate_synthetic.py # 1500 synthetic training examples

tests/
├── conftest.py
├── test_ingest.py
├── test_search.py
├── test_classifier.py
├── test_web3.py
└── test_slack.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Ingest CSV/JSON/NDJSON mentions (idempotent upsert) |
| GET | `/search` | Semantic search with filters + recency/reach boost |
| GET | `/health` | Model and index readiness check |
| POST | `/run_digest` | Manually trigger Slack Web3 digest to all targets |
| GET | `/slack/subscribers` | List active manual webhook subscriptions |
| POST | `/slack/unsubscribe/{sub_id}` | Disconnect/Remove manual webhook subscription by ID |
| GET | `/slack/install` | Initiate Slack App OAuth connection flow |
| GET | `/slack/oauth/callback` | Slack App OAuth authorization callback handler |
| GET | `/slack/oauth/installations` | List active Slack App OAuth connections |
| POST | `/slack/oauth/disconnect/{id}` | Disconnect/Remove Slack App OAuth connection by ID |

### Ingest Example

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "mentions": [{
      "id": "1",
      "title": "Vitalik launches ETH upgrade",
      "text": "vitalik.eth transferred $ETH to 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "source": "CoinDesk",
      "published_at": "2025-01-01",
      "reach": 100000
    }]
  }'
```

### Search Example

```bash
curl "http://localhost:8000/search?query=ethereum+launch&k=5&sentiment_filter=positive"
```

## Running Tests

```bash
C:\ai_pr_venv\Scripts\pytest tests/ --cov=app --cov-report=term-missing
```

## Docker

```bash
docker build -t pr-intelligence .
docker-compose up -d
```

## Success Criteria & Key Implementations

| Criterion | Target | Implementation Status |
|-----------|--------|-----------------------|
| Ingest 10k docs | ≤ 90s | Batch embedding (64/batch) + transaction-decoupled async processing |
| Search p95 | ≤ 300ms | FAISS flat IP + singleton models (hot-start testing preserved) |
| Topic classifier F1 | ≥ 0.70 | LogisticRegression + TF-IDF bigrams |
| Summary length | ≤ 350 chars | Enforced in summarizer post-processing |
| Web3 detection | ≥ 90% accuracy | Regex for standard ETH/ENS/ticker formats |
| Rate limiting | 429 > 20 RPS | slowapi per-endpoint limiter |
| Docker size | ≤ 1.5GB | python:3.11-slim + faiss-cpu + CPU torch |
| Test coverage | > 85% | Full suite across all modules (including new Slack/FAISS tests) |
| Memory Limits | ≤ 512MB RAM | Model Swapping: loads & unloads SentenceTransformer/sentiment models sequentially |
| Idempotency | Clean updates | Vector Index reconstruction on duplicate doc_id update |
| Alert Broadcast | Multi-workspace | Loops through OAuth and manual subscribers; updates database `sent_to_slack` flag |

## Environment Variables

See [`.env.example`](.env.example) for all configuration options.

Key variables:
- `GROQ_API_KEY`: Groq API key for LLM summarization (free tier)
- `GEMINI_API_KEY`: Fallback Gemini API key
- `SLACK_WEBHOOK_URL`: Slack incoming webhook for default digests
- `SLACK_CLIENT_ID`: Slack App Client ID for OAuth connections (configure on Render)
- `SLACK_CLIENT_SECRET`: Slack App Client Secret for OAuth (configure on Render)
- `SLACK_REDIRECT_URI`: Slack App Redirect URI (e.g. `https://your-app.onrender.com/slack/oauth/callback`)
- `ALCHEMY_RPC_URL`: Ethereum RPC for ENS resolution
- `SQLITE_DB_PATH`: SQLite database file path (e.g. `/data/pr_intelligence.db` if using Render persistent disk)
