from __future__ import annotations
import re
from app.models.entity import Entity

_NORMALIZE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    return _NORMALIZE.sub("", text.lower())


class NameIndex:
    """normalized name -> list[Entity] fast lookup."""

    def __init__(self, entities: list[Entity]) -> None:
        self._index: dict[str, list[Entity]] = {}
        for entity in entities:
            names = [entity.canonical_name] + entity.aliases + entity.former_names
            seen: set[str] = set()
            for name in names:
                key = normalize(name)
                if not key or key in seen:
                    continue
                seen.add(key)
                self._index.setdefault(key, []).append(entity)

    def lookup(self, text: str) -> list[Entity]:
        return self._index.get(normalize(text), [])

    def all_entities(self) -> list[Entity]:
        seen_ids: set[str] = set()
        result: list[Entity] = []
        for entities in self._index.values():
            for e in entities:
                if e.entity_id not in seen_ids:
                    seen_ids.add(e.entity_id)
                    result.append(e)
        return result
