from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class VectorIndex:
    """FAISS-backed per-KB vector index."""

    def __init__(self, kb_id: str, index_dir: Path) -> None:
        self.kb_id = kb_id
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._entity_ids: list[str] = []

    def _index_path(self) -> Path:
        return self.index_dir / f"{self.kb_id}.faiss"

    def _meta_path(self) -> Path:
        return self.index_dir / f"{self.kb_id}.ids.json"

    def build(self, entity_ids: list[str], vectors: np.ndarray) -> None:
        import faiss
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        faiss.write_index(index, str(self._index_path()))
        self._meta_path().write_text(json.dumps(entity_ids), encoding="utf-8")
        self._index = index
        self._entity_ids = entity_ids

    def load(self) -> bool:
        if not self._index_path().exists():
            return False
        import faiss
        self._index = faiss.read_index(str(self._index_path()))
        self._entity_ids = json.loads(self._meta_path().read_text(encoding="utf-8"))
        return True

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index is None:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        return [(self._entity_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]

    def exists(self) -> bool:
        return self._index_path().exists()
