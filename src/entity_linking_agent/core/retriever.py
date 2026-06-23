"""Candidate generation for the Topic 10 entity linker."""

from __future__ import annotations

from typing import Optional

from entity_linking_agent.core.contracts import CandidateScore, KnowledgeBaseEntity, MentionRecord
from entity_linking_agent.utils.text import clamp, normalize_text, sequence_similarity


class CandidateRetriever:
    """Alias-driven candidate retriever for a service baseline."""

    minimum_similarity = 0.35
    max_fuzzy_scan = 5000

    def __init__(self) -> None:
        self._alias_index_cache: dict[int, dict[str, list[KnowledgeBaseEntity]]] = {}

    def retrieve(
        self,
        mention: MentionRecord,
        entities: list[KnowledgeBaseEntity],
        top_k: int,
    ) -> list[CandidateScore]:
        normalized_mention = normalize_text(mention.text)
        alias_index = self._get_alias_index(entities)
        exact_matches = alias_index.get(normalized_mention, [])
        if exact_matches:
            return [
                self._build_candidate(
                    mention=mention,
                    entity=entity,
                    matched_alias=mention.text,
                    similarity=1.0,
                )
                for entity in exact_matches[: max(top_k * 2, top_k)]
            ]

        if len(entities) > self.max_fuzzy_scan:
            return []

        candidates: list[CandidateScore] = []
        for entity in entities:
            matched_alias: Optional[str] = None
            best_similarity = 0.0

            for alias in [entity.canonical_name, *entity.aliases]:
                normalized_alias = normalize_text(alias)
                if not normalized_alias:
                    continue

                similarity = sequence_similarity(normalized_mention, normalized_alias)
                exact_match = normalized_mention == normalized_alias
                contains_match = (
                    normalized_mention in normalized_alias or normalized_alias in normalized_mention
                )

                if exact_match:
                    similarity = 1.0
                elif contains_match and len(normalized_mention) >= 2:
                    similarity = max(similarity, 0.90)

                if similarity > best_similarity:
                    best_similarity = similarity
                    matched_alias = alias

            if best_similarity < self.minimum_similarity:
                continue

            candidates.append(
                self._build_candidate(
                    mention=mention,
                    entity=entity,
                    matched_alias=matched_alias,
                    similarity=best_similarity,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(top_k * 2, top_k)]

    def _get_alias_index(self, entities: list[KnowledgeBaseEntity]) -> dict[str, list[KnowledgeBaseEntity]]:
        cache_key = id(entities)
        cached = self._alias_index_cache.get(cache_key)
        if cached is not None:
            return cached

        alias_index: dict[str, list[KnowledgeBaseEntity]] = {}
        for entity in entities:
            entity_seen_aliases: set[str] = set()
            for alias in [entity.canonical_name, *entity.aliases]:
                normalized_alias = normalize_text(alias)
                if not normalized_alias or normalized_alias in entity_seen_aliases:
                    continue
                entity_seen_aliases.add(normalized_alias)
                alias_index.setdefault(normalized_alias, []).append(entity)

        self._alias_index_cache[cache_key] = alias_index
        return alias_index

    def _build_candidate(
        self,
        mention: MentionRecord,
        entity: KnowledgeBaseEntity,
        matched_alias: Optional[str],
        similarity: float,
    ) -> CandidateScore:
        reasons: list[str] = []
        if matched_alias is not None:
            if normalize_text(matched_alias) == normalize_text(mention.text):
                reasons.append("exact_or_alias_match")
            else:
                reasons.append("fuzzy_alias_match")

        if mention.entity_type and mention.entity_type == entity.entity_type:
            reasons.append("entity_type_aligned")

        return CandidateScore(
            entity_id=entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type,
            score=round(clamp(similarity), 3),
            alias_similarity=round(clamp(similarity), 3),
            matched_alias=matched_alias,
            reasons=reasons,
        )
