import pytest


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "index_total_docs" in data
    assert "classifier_ready" in data
    assert data["vector_db"] == "FAISS"


def test_invalid_payload_returns_422(client):
    r = client.post("/ingest", json={"mentions": [{"bad_field": "x"}]})
    assert r.status_code == 422


def test_empty_mentions_returns_422(client):
    r = client.post("/ingest", json={"mentions": []})
    assert r.status_code == 422


def test_ingest_single(client, sample_mention):
    r = client.post("/ingest", json={"mentions": [sample_mention]})
    assert r.status_code == 200
    data = r.json()
    assert data["inserted"] + data["updated"] == 1
    assert isinstance(data["errors"], list)


def test_idempotent_upsert(client, sample_mention):
    # First call
    r1 = client.post("/ingest", json={"mentions": [sample_mention]})
    assert r1.status_code == 200
    # Second call with same id
    r2 = client.post("/ingest", json={"mentions": [sample_mention]})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["updated"] >= 1


def test_ingest_batch_10(client):
    mentions = [
        {
            "id": f"batch-{i:03d}",
            "title": f"Company {i} announces funding",
            "text": f"Company {i} raises $10M in Series A to expand its product.",
            "source": "TechCrunch",
            "published_at": "2025-01-01",
            "reach": i * 1000
        }
        for i in range(10)
    ]
    r = client.post("/ingest", json={"mentions": mentions})
    assert r.status_code == 200
    data = r.json()
    assert data["inserted"] + data["updated"] == 10


def test_ingest_csv(client):
    csv_data = "id,title,text,source,published_at,reach,labels\ncsv-001,CSV Title,CSV Text content here.,TechCrunch,2025-01-01,5000,product"
    r = client.post("/ingest", content=csv_data, headers={"Content-Type": "text/csv"})
    assert r.status_code == 200
    data = r.json()
    assert data["inserted"] + data["updated"] == 1


def test_ingest_ndjson(client):
    ndjson_data = '{"id": "ndjson-001", "title": "NDJSON Title", "text": "NDJSON Text content here.", "source": "TechCrunch", "published_at": "2025-01-01", "reach": 5000, "labels": ["product"]}\n'
    r = client.post("/ingest", content=ndjson_data, headers={"Content-Type": "application/x-ndjson"})
    assert r.status_code == 200
    data = r.json()
    assert data["inserted"] + data["updated"] == 1

