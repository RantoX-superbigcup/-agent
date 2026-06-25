from __future__ import annotations
from functools import lru_cache
import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer
        _device = _resolve_device(device)
        self._model = SentenceTransformer(model_name, device=_device)
        self._actual_device = str(self._model.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _resolve_device(requested: str) -> str:
    """将 'auto' 解析为实际可用设备，或直接透传 'cuda'/'cpu'。"""
    if requested == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
