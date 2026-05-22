from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import IngestRequest, IngestResponse, MentionIn
from app.models.db import get_conn, db_conn
from app.nlp.embedder import Embedder
from app.nlp.sentiment import analyze as analyze_sentiment
from app.nlp.classifier import predict as classify_topics
from app.nlp.summarizer import summarize
from app.web3.detector import detect_web3_signals
from app.web3.resolver import enrich_signals
from app.search.faiss_store import get_store
import json

router = APIRouter()

import csv
import io
from typing import Optional

@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": IngestRequest.model_json_schema()
                },
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "description": "CSV content with columns: id, title, text, source, published_at, reach, labels"
                    }
                },
                "application/x-ndjson": {
                    "schema": {
                        "type": "string",
                        "description": "NDJSON content, one JSON mention per line"
                    }
                }
            }
        }
    }
)
async def ingest(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "")
    mentions_data = []

    body = await raw_request.body()

    if "text/csv" in content_type:
        csv_text = body.decode("utf-8")
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id") or not row.get("title") or not row.get("text"):
                raise HTTPException(status_code=422, detail="Missing required fields: id, title, or text")
            
            labels_str = row.get("labels", "")
            labels = []
            if labels_str:
                clean_str = labels_str.strip("[]\"' ")
                labels = [l.strip() for l in clean_str.split(",") if l.strip()]

            try:
                reach = int(row.get("reach", 0))
            except Exception:
                reach = 0

            mentions_data.append(MentionIn(
                id=row.get("id"),
                title=row.get("title"),
                text=row.get("text"),
                source=row.get("source"),
                published_at=row.get("published_at"),
                reach=reach,
                labels=labels
            ))
    elif "ndjson" in content_type or "x-ndjson" in content_type:
        ndjson_text = body.decode("utf-8")
        for line_num, line in enumerate(ndjson_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                mentions_data.append(MentionIn(**item))
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid NDJSON line {line_num}: {str(e)}")
    else:
        try:
            body_json = json.loads(body.decode("utf-8"))
            if isinstance(body_json, dict) and "mentions" in body_json:
                mentions_data = [MentionIn(**m) for m in body_json["mentions"]]
            elif isinstance(body_json, list):
                mentions_data = [MentionIn(**m) for m in body_json]
            else:
                raise HTTPException(status_code=422, detail="JSON must contain a 'mentions' list")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

async def process_and_ingest_mentions(mentions_data: list[MentionIn]) -> dict:
    if not mentions_data:
        return {"inserted": 0, "updated": 0, "errors": []}

    store = get_store()
    embedder = Embedder()

    inserted, updated, errors = 0, 0, []
    texts = [f"{m.title}. {m.text}" for m in mentions_data]

    # Chunk embedding generation to keep memory flat
    vectors = []
    chunk_size = 10
    for idx in range(0, len(texts), chunk_size):
        chunk = texts[idx:idx + chunk_size]
        vectors.extend(embedder.embed(chunk))

    # Batch classify topics
    topic_preds = classify_topics(texts)

    with db_conn() as conn:
        for i, mention in enumerate(mentions_data):
            try:
                # Sentiment analysis
                sent = analyze_sentiment(mention.text)

                # Summarization (async, with fallback)
                summary = await summarize(mention.text)

                # Web3 detection + enrichment
                raw_signals = detect_web3_signals(mention.text)
                enriched = enrich_signals(raw_signals)

                topics = topic_preds[i]

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

    store.save()
    return {"inserted": inserted, "updated": updated, "errors": errors}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": IngestRequest.model_json_schema()
                },
                "text/csv": {
                    "schema": {
                        "type": "string",
                        "description": "CSV content with columns: id, title, text, source, published_at, reach, labels"
                    }
                },
                "application/x-ndjson": {
                    "schema": {
                        "type": "string",
                        "description": "NDJSON content, one JSON mention per line"
                    }
                }
            }
        }
    }
)
async def ingest(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "")
    mentions_data = []

    body = await raw_request.body()

    if "text/csv" in content_type:
        csv_text = body.decode("utf-8")
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id") or not row.get("title") or not row.get("text"):
                raise HTTPException(status_code=422, detail="Missing required fields: id, title, or text")
            
            labels_str = row.get("labels", "")
            labels = []
            if labels_str:
                clean_str = labels_str.strip("[]\"' ")
                labels = [l.strip() for l in clean_str.split(",") if l.strip()]

            try:
                reach = int(row.get("reach", 0))
            except Exception:
                reach = 0

            mentions_data.append(MentionIn(
                id=row.get("id"),
                title=row.get("title"),
                text=row.get("text"),
                source=row.get("source"),
                published_at=row.get("published_at"),
                reach=reach,
                labels=labels
            ))
    elif "ndjson" in content_type or "x-ndjson" in content_type:
        ndjson_text = body.decode("utf-8")
        for line_num, line in enumerate(ndjson_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                mentions_data.append(MentionIn(**item))
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid NDJSON line {line_num}: {str(e)}")
    else:
        try:
            body_json = json.loads(body.decode("utf-8"))
            if isinstance(body_json, dict) and "mentions" in body_json:
                mentions_data = [MentionIn(**m) for m in body_json["mentions"]]
            elif isinstance(body_json, list):
                mentions_data = [MentionIn(**m) for m in body_json]
            else:
                raise HTTPException(status_code=422, detail="JSON must contain a 'mentions' list")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    if not mentions_data:
        raise HTTPException(status_code=422, detail="No mentions to ingest")

    res = await process_and_ingest_mentions(mentions_data)
    return IngestResponse(**res)
