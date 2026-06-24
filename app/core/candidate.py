from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

from app.models.entity import Entity
from app.models.request import MentionInput
from app.storage.index import NameIndex, normalize

_MAX_FUZZY_SCAN = 5000
_MIN_SIMILARITY = 0.35


class CandidateResult:
    __slots__ = ("entity", "score", "matched_name", "match_source")

    def __init__(self, entity: Entity, score: float, matched_name: str, match_source: str) -> None:
        self.entity = entity
        self.score = score
        self.matched_name = matched_name
        self.match_source = match_source  # canonical_match | alias_match | former_name_match | similarity_match


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def retrieve(mention: MentionInput, index: NameIndex, entities: list[Entity], top_k: int) -> list[CandidateResult]:
    query = normalize(mention.surface_form)

    # exact / alias lookup via index
    exact_hits = index.lookup(mention.surface_form)
    if exact_hits:
        results = []
        for entity in exact_hits:
            source = _match_source(mention.surface_form, entity)
            results.append(CandidateResult(entity=entity, score=1.0, matched_name=mention.surface_form, match_source=source))
        return results[:max(top_k * 20, 50)]

    if len(entities) > _MAX_FUZZY_SCAN:
        return []

    candidates: list[CandidateResult] = []
    for entity in entities:
        best_score = 0.0
        best_name = ""
        best_source = "similarity_match"
        for name in [entity.canonical_name] + entity.aliases + entity.former_names:
            norm_name = normalize(name)
            if not norm_name:
                continue
            if query == norm_name or query in norm_name or norm_name in query:
                score = 0.9 if query != norm_name else 1.0
            else:
                score = _similarity(query, norm_name)
            if score > best_score:
                best_score = score
                best_name = name
                best_source = _match_source(name, entity)
        if best_score >= _MIN_SIMILARITY:
            candidates.append(CandidateResult(entity=entity, score=round(best_score, 3), matched_name=best_name, match_source=best_source))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _match_source(name: str, entity: Entity) -> str:
    norm = normalize(name)
    if norm == normalize(entity.canonical_name):
        return "canonical_match"
    if any(normalize(a) == norm for a in entity.aliases):
        return "alias_match"
    if any(normalize(f) == norm for f in entity.former_names):
        return "former_name_match"
    return "similarity_match"
