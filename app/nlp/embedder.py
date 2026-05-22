from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

class Embedder:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer(settings.EMBEDDING_MODEL)
        return cls._instance

    def embed(self, texts: list) -> np.ndarray:
        model = self.get()
        return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
