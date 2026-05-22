from fastapi import APIRouter, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.schemas import SearchResponse, SearchResult
from app.nlp.embedder import Embedder
from app.search.faiss_store import get_store
from typing import Optional

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/search", response_model=SearchResponse, tags=["Search"])
@limiter.limit("20/second")
async def search(
    request: Request,
    query: str = Query(..., min_length=1, description="Semantic search query"),
    k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    sentiment_filter: Optional[str] = Query(default=None, description="Filter: positive/negative"),
    topic_filter: Optional[str] = Query(default=None, description="Filter by topic"),
    page: int = Query(default=1, ge=1, description="Page number")
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
