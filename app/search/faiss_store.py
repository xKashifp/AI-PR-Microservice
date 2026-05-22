import faiss
import numpy as np
import os
import pickle
from app.config import settings

class FAISSStore:
    def __init__(self):
        self.dim = 384         # all-MiniLM-L6-v2 output dim
        self.index = faiss.IndexFlatIP(self.dim)   # inner product = cosine on normalized vecs
        self.metadata: list = []
        self.id_to_pos: dict = {}
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
        os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, self._index_path())
        with open(self._meta_path(), "wb") as f:
            pickle.dump({"metadata": self.metadata, "id_to_pos": self.id_to_pos}, f)

    def upsert(self, doc_id: str, vector: np.ndarray, meta: dict):
        vec = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        if doc_id in self.id_to_pos:
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
        sentiment_filter: str = None,
        topic_filter: str = None
    ) -> list:
        vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        fetch_k = min(k * 10, self.index.ntotal)
        if fetch_k == 0:
            return []

        scores, indices = self.index.search(vec, fetch_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]

            if sentiment_filter and meta.get("sentiment") != sentiment_filter:
                continue
            if topic_filter and topic_filter not in (meta.get("topics") or []):
                continue

            boosted = _boost_score(score, meta)
            results.append({**meta, "_score": boosted})

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:k]


def _boost_score(base_score: float, meta: dict) -> float:
    from datetime import datetime, timezone
    boost = float(base_score)

    # Reach boost (log scale)
    reach = meta.get("reach", 0) or 0
    if reach > 0:
        import math
        boost += 0.05 * math.log1p(reach)

    # Recency boost: decay over 30 days
    try:
        pub_str = meta.get("published_at", "")
        if pub_str:
            pub = datetime.fromisoformat(pub_str)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - pub).days
            recency_factor = max(0, 1 - (age_days / 30))
            boost += 0.10 * recency_factor
    except Exception:
        pass

    return round(boost, 6)


# Singleton
_store: FAISSStore = None

def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore()
    return _store
