import pytest
import sys
import os

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test env vars before any app imports
os.environ.setdefault("SQLITE_DB_PATH", "./data/test_pr_intelligence.db")
os.environ.setdefault("FAISS_INDEX_DIR", "./data/test_faiss_index")
os.environ.setdefault("MODEL_DIR", "./ml/models")
os.environ["TESTING"] = "True"

from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_mention():
    return {
        "id": "test-001",
        "title": "Acme Corp raises $50M in Series B",
        "text": "Acme Corp has secured $50 million in a Series B round led by Sequoia Capital to expand its analytics platform.",
        "source": "TechCrunch",
        "published_at": "2025-01-15",
        "reach": 50000
    }

@pytest.fixture
def web3_mention():
    return {
        "id": "web3-001",
        "title": "Vitalik transfers ETH",
        "text": "Wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 received $ETH from vitalik.eth in the DAO treasury.",
        "source": "CoinDesk",
        "published_at": "2025-01-20",
        "reach": 100000
    }
