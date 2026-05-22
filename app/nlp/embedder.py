from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

class Embedder:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            import torch
            import gc
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
            cls._instance = model
            gc.collect()
        return cls._instance

    def embed(self, texts: list) -> np.ndarray:
        model = self.get()
        return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
