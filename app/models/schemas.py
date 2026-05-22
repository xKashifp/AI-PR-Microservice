from pydantic import BaseModel, Field
from typing import Optional, List
import json

class MentionIn(BaseModel):
    id: str
    title: str
    text: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    reach: Optional[int] = 0
    labels: Optional[List[str]] = []

class IngestRequest(BaseModel):
    mentions: List[MentionIn] = Field(..., min_length=1, max_length=10000)

class IngestResponse(BaseModel):
    inserted: int
    updated: int
    errors: List[dict]

class SearchResult(BaseModel):
    id: str
    title: str
    source: Optional[str] = None
    published_at: Optional[str] = None
    sentiment: Optional[str] = None
    topics: Optional[List[str]] = None
    summary: Optional[str] = None
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    page: int
    k: int
