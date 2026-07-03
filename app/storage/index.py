from __future__ import annotations
import re
from app.models.entity import Entity

_NORMALIZE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    return _NORMALIZE.sub("", text.lower())


class NameIndex:
    """Separate exact-name indexes for canonical names, aliases, and short names."""

    def __init__(self, entities: list[Entity]) -> None:
        self._canonical_index: dict[str, list[Entity]] = {}
        self._alias_index: dict[str, list[Entity]] = {}
        self._short_name_index: dict[str, list[Entity]] = {}
        for entity in entities:
            self._add(self._canonical_index, entity.canonical_name, entity)
            for alias in entity.aliases:
                self._add(self._alias_index, alias, entity)
            for short_name in entity.short_names:
                self._add(self._short_name_index, short_name, entity)

    @staticmethod
    def _add(index: dict[str, list[Entity]], name: str, entity: Entity) -> None:
        key = normalize(name)
        if not key:
            return
        bucket = index.setdefault(key, [])
        if all(item.entity_id != entity.entity_id for item in bucket):
            bucket.append(entity)

    def lookup_canonical(self, text: str) -> list[Entity]:
        return self._canonical_index.get(normalize(text), [])

    def lookup_alias(self, text: str) -> list[Entity]:
        return self._alias_index.get(normalize(text), [])

    def lookup_short_name(self, text: str) -> list[Entity]:
        return self._short_name_index.get(normalize(text), [])

    def lookup_exact(self, text: str) -> dict[str, list[Entity]]:
        return {
            "canonical_match": self.lookup_canonical(text),
            "alias_match": self.lookup_alias(text),
            "short_name_match": self.lookup_short_name(text),
        }

    def lookup(self, text: str) -> list[Entity]:
        seen_ids: set[str] = set()
        result: list[Entity] = []
        for entities in self.lookup_exact(text).values():
            for entity in entities:
                if entity.entity_id not in seen_ids:
                    seen_ids.add(entity.entity_id)
                    result.append(entity)
        return result

    def all_entities(self) -> list[Entity]:
        seen_ids: set[str] = set()
        result: list[Entity] = []
        for index in (self._canonical_index, self._alias_index, self._short_name_index):
            for entities in index.values():
                for entity in entities:
                    if entity.entity_id not in seen_ids:
                        seen_ids.add(entity.entity_id)
                        result.append(entity)
        return result
