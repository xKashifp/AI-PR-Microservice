import numpy as np
import hashlib
import httpx
import os
from app.config import settings

class Embedder:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _hash_embed_single(self, text: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        words = text.lower().split()
        if not words:
            words = ["empty"]
        for w in words:
            h = hashlib.sha256(w.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], byteorder="big") % 384
            val = 1.0 if (h[4] % 2 == 0) else -1.0
            vec[idx] += val
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        
        if settings.GEMINI_API_KEY and os.environ.get("TESTING") != "True":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={settings.GEMINI_API_KEY}"
                payload = {
                    "requests": [
                        {
                            "model": "models/text-embedding-004",
                            "content": {"parts": [{"text": t}]},
                            "outputDimensionality": 384
                        }
                        for t in texts
                    ]
                }
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    
                embeddings = []
                for emb_data in data.get("embeddings", []):
                    embeddings.append(emb_data["values"])
                
                if len(embeddings) == len(texts):
                    return np.array(embeddings, dtype=np.float32)
            except Exception:
                pass
                
        res = [self._hash_embed_single(t) for t in texts]
        return np.array(res, dtype=np.float32)
