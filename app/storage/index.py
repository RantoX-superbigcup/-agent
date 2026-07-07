from __future__ import annotations

from dataclasses import dataclass
from math import log
import re

from app.models.entity import Entity

_NORMALIZE = re.compile(r"[\s\W_]+", re.UNICODE)
_NAME_SOURCE_PRIORITY = {
    "canonical": 3,
    "alias": 2,
    "former_name": 1,
}


def normalize(text: str) -> str:
    return _NORMALIZE.sub("", text.lower())


@dataclass(frozen=True)
class ExactNameHit:
    entity: Entity
    matched_name: str
    match_source: str


@dataclass(frozen=True)
class EntityNameRecord:
    entity: Entity
    name_text: str
    name_source: str


class NameIndex:
    """Raw exact-name indices plus per-entity name records for fuzzy retrieval."""

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = list(entities)
        self._canonical_index: dict[str, list[ExactNameHit]] = {}
        self._alias_index: dict[str, list[ExactNameHit]] = {}
        self._former_name_index: dict[str, list[ExactNameHit]] = {}
        self._names_by_entity: dict[str, tuple[EntityNameRecord, ...]] = {}
        self._char_idf: dict[str, float] = {}
        self._name_count = 0

        all_name_texts: list[str] = []
        for entity in self._entities:
            records = self._build_entity_name_records(entity)
            self._names_by_entity[entity.entity_id] = tuple(records)
            for record in records:
                all_name_texts.append(record.name_text)
                self._register_exact_hit(record)

        self._name_count = len(all_name_texts)
        self._char_idf = self._build_char_idf(all_name_texts)

    def exact_lookup(self, text: str) -> list[ExactNameHit]:
        merged: dict[str, ExactNameHit] = {}
        for hit in (
            self._canonical_index.get(text, [])
            + self._alias_index.get(text, [])
            + self._former_name_index.get(text, [])
        ):
            existing = merged.get(hit.entity.entity_id)
            if existing is None or _exact_priority(hit.match_source) > _exact_priority(existing.match_source):
                merged[hit.entity.entity_id] = hit
        return list(merged.values())

    def lookup(self, text: str) -> list[Entity]:
        return [hit.entity for hit in self.exact_lookup(text)]

    def names_for_entity(self, entity_id: str) -> tuple[EntityNameRecord, ...]:
        return self._names_by_entity.get(entity_id, ())

    def iter_entity_name_records(self) -> list[tuple[Entity, tuple[EntityNameRecord, ...]]]:
        return [(entity, self._names_by_entity.get(entity.entity_id, ())) for entity in self._entities]

    def all_entities(self) -> list[Entity]:
        return list(self._entities)

    def char_idf(self, char: str) -> float:
        return self._char_idf.get(char, 1.0)

    @property
    def name_count(self) -> int:
        return self._name_count

    def _build_entity_name_records(self, entity: Entity) -> list[EntityNameRecord]:
        records: list[EntityNameRecord] = []
        seen_names: set[str] = set()

        def add_name(name_text: str, name_source: str) -> None:
            raw = str(name_text or "")
            if not raw or raw in seen_names:
                return
            seen_names.add(raw)
            records.append(
                EntityNameRecord(
                    entity=entity,
                    name_text=raw,
                    name_source=name_source,
                )
            )

        add_name(entity.canonical_name, "canonical")
        for alias in entity.aliases:
            add_name(alias, "alias")
        for former_name in entity.former_names:
            add_name(former_name, "former_name")
        return records

    def _register_exact_hit(self, record: EntityNameRecord) -> None:
        match_source = {
            "canonical": "canonical_match",
            "alias": "alias_match",
            "former_name": "former_name_match",
        }[record.name_source]
        hit = ExactNameHit(
            entity=record.entity,
            matched_name=record.name_text,
            match_source=match_source,
        )
        if record.name_source == "canonical":
            self._canonical_index.setdefault(record.name_text, []).append(hit)
        elif record.name_source == "alias":
            self._alias_index.setdefault(record.name_text, []).append(hit)
        else:
            self._former_name_index.setdefault(record.name_text, []).append(hit)

    @staticmethod
    def _build_char_idf(name_texts: list[str]) -> dict[str, float]:
        if not name_texts:
            return {}

        doc_freq: dict[str, int] = {}
        for name_text in name_texts:
            for char in set(name_text):
                doc_freq[char] = doc_freq.get(char, 0) + 1

        total = len(name_texts)
        return {
            char: log((total + 1) / (freq + 1)) + 1.0
            for char, freq in doc_freq.items()
        }


def _exact_priority(match_source: str) -> int:
    if match_source == "canonical_match":
        return _NAME_SOURCE_PRIORITY["canonical"]
    if match_source == "alias_match":
        return _NAME_SOURCE_PRIORITY["alias"]
    if match_source == "former_name_match":
        return _NAME_SOURCE_PRIORITY["former_name"]
    return 0
