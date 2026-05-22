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
