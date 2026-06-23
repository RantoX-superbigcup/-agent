"""Alias prior model learned from labeled entity-linking data."""

from __future__ import annotations

import json
from pathlib import Path

from entity_linking_agent.utils.text import normalize_text


class AliasPrior:
    """Stores mention-to-entity priors learned from the training split."""

    def __init__(self, mapping: dict[str, dict[str, float]]) -> None:
        self.mapping = mapping

    @classmethod
    def load(cls, path: Path) -> "AliasPrior":
        if not path.exists():
            return cls({})
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("mapping", {}))

    def score(self, mention: str, entity_id: str) -> float:
        candidates = self.mapping.get(normalize_text(mention), {})
        return float(candidates.get(entity_id, 0.0))
