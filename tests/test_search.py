import pytest
import time


def test_search_requires_query(client):
    r = client.get("/search")
    assert r.status_code == 422


def test_search_returns_results(client, sample_mention):
    # Ensure some data is ingested
    client.post("/ingest", json={"mentions": [sample_mention]})
    r = client.get("/search?query=funding+announcement&k=5")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "total" in data
    assert "k" in data
    assert data["k"] == 5


def test_search_result_schema(client, sample_mention):
    client.post("/ingest", json={"mentions": [sample_mention]})
    r = client.get("/search?query=Series+B+funding&k=1")
    assert r.status_code == 200
    results = r.json()["results"]
    if results:
        res = results[0]
        assert "id" in res
        assert "title" in res
        assert "score" in res


def test_search_sentiment_filter(client, sample_mention):
    client.post("/ingest", json={"mentions": [sample_mention]})
    r = client.get("/search?query=funding&k=10&sentiment_filter=positive")
    assert r.status_code == 200
    data = r.json()
    for result in data["results"]:
        assert result.get("sentiment") == "positive"


def test_search_empty_index_returns_empty(client):
    r = client.get("/search?query=zzz_nonexistent_query_xyz&k=3")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_search_p95_latency(client, sample_mention):
    """p95 search latency < 300ms"""
    client.post("/ingest", json={"mentions": [sample_mention]})
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        client.get("/search?query=funding+announcement&k=5")
        latencies.append(time.perf_counter() - start)
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < 0.300, f"p95 latency {p95*1000:.0f}ms exceeds 300ms"


def test_search_vector_updates_on_upsert(client):
    """Verify that when a document is updated with new text, its FAISS vector is updated too."""
    doc_id = "test-update-vec"
    
    # 1. Ingest document with first text
    doc_1 = {
        "id": doc_id,
        "title": "Red Fruits",
        "text": "Apples are delicious red round fruits that grow on apple trees.",
        "source": "FruitNews",
        "published_at": "2025-01-01",
        "reach": 1000
    }
    r1 = client.post("/ingest", json={"mentions": [doc_1]})
    assert r1.status_code == 200
    
    # Search for "apples" and verify we get high score
    s1 = client.get("/search?query=apples&k=5")
    assert s1.status_code == 200
    res_1 = s1.json()["results"]
    assert len(res_1) > 0
    assert res_1[0]["id"] == doc_id
    score_for_apples_init = res_1[0]["score"]
    
    # 2. Ingest document with updated text (now about bananas)
    doc_2 = {
        "id": doc_id,
        "title": "Yellow Fruits",
        "text": "Bananas are sweet yellow long tropical fruits that grow in bunches.",
        "source": "FruitNews",
        "published_at": "2025-01-01",
        "reach": 1000
    }
    r2 = client.post("/ingest", json={"mentions": [doc_2]})
    assert r2.status_code == 200
    
    # Search for "bananas" and verify we get the document
    s2 = client.get("/search?query=bananas&k=5")
    assert s2.status_code == 200
    res_2 = s2.json()["results"]
    assert len(res_2) > 0
    assert res_2[0]["id"] == doc_id
    
    # Search for "apples" again. The score should be lower now that the document describes bananas.
    s3 = client.get("/search?query=apples&k=5")
    assert s3.status_code == 200
    res_3 = s3.json()["results"]
    
    # If the document is found, its score for "apples" should be lower than initial
    if res_3 and res_3[0]["id"] == doc_id:
        assert res_3[0]["score"] < score_for_apples_init
