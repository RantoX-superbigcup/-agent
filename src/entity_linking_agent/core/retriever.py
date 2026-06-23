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
        query_texts = self._query_texts(mention)
        normalized_queries = [normalize_text(query) for query in query_texts]
        alias_index = self._get_alias_index(entities)
        exact_matches: list[CandidateScore] = []
        exact_seen_entity_ids: set[str] = set()
        for query_text, normalized_query in zip(query_texts, normalized_queries):
            for entity in alias_index.get(normalized_query, []):
                if entity.entity_id in exact_seen_entity_ids:
                    continue
                exact_seen_entity_ids.add(entity.entity_id)
                exact_matches.append(
                    self._build_candidate(
                        mention=mention,
                        entity=entity,
                        matched_alias=query_text,
                        similarity=1.0,
                    )
                )
        if exact_matches:
            return exact_matches[: max(top_k * 20, 50)]

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

                similarity = max(
                    sequence_similarity(normalized_query, normalized_alias)
                    for normalized_query in normalized_queries
                )
                exact_match = any(normalized_query == normalized_alias for normalized_query in normalized_queries)
                contains_match = (
                    any(
                        normalized_query
                        and len(normalized_query) >= 2
                        and normalized_query in normalized_alias
                        for normalized_query in normalized_queries
                    )
                    or any(
                        normalized_query
                        and len(normalized_query) >= 2
                        and normalized_alias in normalized_query
                        for normalized_query in normalized_queries
                    )
                )

                if exact_match:
                    similarity = 1.0
                elif contains_match:
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

    @staticmethod
    def _query_texts(mention: MentionRecord) -> list[str]:
        query_texts = [mention.text]
        for alias in mention.metadata.get("candidate_aliases", []):
            alias_text = str(alias).strip()
            if alias_text and alias_text not in query_texts:
                query_texts.append(alias_text)
        return query_texts

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
            normalized_matched_alias = normalize_text(matched_alias)
            normalized_mention = normalize_text(mention.text)
            normalized_expansions = {
                normalize_text(str(alias))
                for alias in mention.metadata.get("candidate_aliases", [])
                if str(alias).strip()
            }
            if normalized_matched_alias in normalized_expansions and normalized_matched_alias != normalized_mention:
                reasons.append("llm_alias_expansion")
            elif normalized_matched_alias == normalized_mention:
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
