from __future__ import annotations
from functools import lru_cache
import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
