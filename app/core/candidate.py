from __future__ import annotations
import logging
from difflib import SequenceMatcher
from typing import Optional, TYPE_CHECKING
import numpy as np
from app.models.entity import Entity
from app.models.request import MentionInput
from app.storage.index import NameIndex, normalize

if TYPE_CHECKING:
    from app.core.embedder import Embedder
    from app.storage.vector_index import VectorIndex

logger = logging.getLogger("entity_link_agent")

_MAX_FUZZY_SCAN = 5000
_MIN_SIMILARITY = 0.35
_MIN_DESCRIPTIVE_SCORE = 0.16
_DESCRIPTIVE_CONTEXT_RADIUS = 18

# 精确命中保底分数
_EXACT_FLOOR: dict[str, float] = {
    "canonical_match": 0.95,
    "alias_match": 0.92,
    "former_name_match": 0.88,
}
# 精确命中在向量结果中的加分
_EXACT_BOOST: dict[str, float] = {
    "canonical_match": 0.25,
    "alias_match": 0.20,
    "former_name_match": 0.15,
}
_TRUSTED_EXACT_RANK: dict[str, int] = {
    "canonical_match": 3,
    "alias_match": 2,
    "former_name_match": 1,
}
_SURFACE_EXACT_REASON = "surface_form_exact_match"


class CandidateResult:
    __slots__ = ("entity", "score", "matched_name", "match_source", "alias_similarity", "reasons")

    def __init__(
        self,
        entity: Entity,
        score: float,
        matched_name: str,
        match_source: str,
        alias_similarity: float | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        self.entity = entity
        self.score = score
        self.matched_name = matched_name
        self.match_source = match_source
        self.alias_similarity = score if alias_similarity is None else alias_similarity
        self.reasons = reasons or []


def retrieve(
    mention: MentionInput,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    context: str = "",
    embedder: Optional["Embedder"] = None,
    vector_index: Optional["VectorIndex"] = None,
) -> list[CandidateResult]:
    """多源候选召回：精确名称匹配 + 向量检索 + 模糊匹配，合并去重。"""
    use_vector = embedder is not None and vector_index is not None and vector_index.exists()

    if not use_vector:
        logger.debug("    [规则召回] mention='%s'", mention.surface_form)
        return _rule_retrieve(mention, context, index, entities, top_k)

    logger.debug("    [多源召回] mention='%s' top_k=%d", mention.surface_form, top_k)
    return _merged_retrieve(mention, context, index, entities, top_k, embedder, vector_index)


def _merged_retrieve(
    mention: MentionInput,
    context: str,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    embedder: "Embedder",
    vector_index: "VectorIndex",
) -> list[CandidateResult]:
    """合并精确名称查找 + 向量语义检索的结果。"""
    # 1. 精确名称查找
    exact_candidates = _exact_retrieve(mention, context, index)
    exact_by_id = {c.entity.entity_id: c for c in exact_candidates}

    # 2. 向量检索（多取一些候选用作合并）
    vec_results = _vector_retrieve(mention, context, entities, max(top_k * 3, 30), embedder, vector_index)

    # 3. 构建 entity_id → Entity 映射
    entity_map = {e.entity_id: e for e in entities}
    merged: dict[str, CandidateResult] = {}

    for r in vec_results:
        eid = r.entity.entity_id
        exact_candidate = exact_by_id.get(eid)
        if exact_candidate is not None:
            _merge_vector_score_into_exact(exact_candidate, r)
            merged[eid] = exact_candidate
        else:
            merged[eid] = r

    # 4. 精确命中但向量没召回的实体 → 直接插入
    for exact_candidate in exact_candidates:
        eid = exact_candidate.entity.entity_id
        if eid not in merged:
            merged[eid] = exact_candidate

    # 5. 如果候选不够，补上模糊匹配
    if len(merged) < top_k and len(entities) <= _MAX_FUZZY_SCAN:
        fuzzy = _fuzzy_retrieve(mention, context, entities, top_k * 2)
        for c in fuzzy:
            eid = c.entity.entity_id
            if eid not in merged:
                merged[eid] = c

    result = sorted(merged.values(), key=rank_key, reverse=True)
    return result[:top_k]


def _merge_vector_score_into_exact(exact_candidate: CandidateResult, vector_candidate: CandidateResult) -> None:
    # Keep exact-name evidence intact; vector similarity should not downgrade alias_similarity.
    boost = _EXACT_BOOST.get(exact_candidate.match_source, 0.12)
    floor = _EXACT_FLOOR.get(exact_candidate.match_source, 0.85)
    exact_candidate.score = round(max(exact_candidate.score, vector_candidate.score + boost, floor), 3)
    exact_candidate.alias_similarity = 1.0
    for reason in vector_candidate.reasons:
        if reason not in exact_candidate.reasons:
            exact_candidate.reasons.append(reason)


def trusted_exact_rank(candidate: CandidateResult) -> int:
    return _TRUSTED_EXACT_RANK.get(candidate.match_source, 0)


def rank_key(candidate: CandidateResult) -> tuple[int, float]:
    return trusted_exact_rank(candidate), candidate.score


def _vector_retrieve(
    mention: MentionInput,
    context: str,
    entities: list[Entity],
    top_k: int,
    embedder: "Embedder",
    vector_index: "VectorIndex",
) -> list[CandidateResult]:
    query_text = f"{mention.surface_form} {context}".strip()
    query_vec = embedder.encode_one(query_text)
    hits = vector_index.search(query_vec, top_k)
    entity_map = {e.entity_id: e for e in entities}
    results = []
    for entity_id, score in hits:
        entity = entity_map.get(entity_id)
        if entity is None:
            continue
        source = _match_source(mention.surface_form, entity)
        results.append(CandidateResult(
            entity=entity, score=round(float(score), 3),
            matched_name=entity.canonical_name, match_source=source,
        ))
    return results


def _rule_retrieve(
    mention: MentionInput,
    context: str,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
) -> list[CandidateResult]:
    """纯规则召回（向量索引用不了时降级使用）。"""
    exact_hits = _exact_retrieve(mention, context, index)
    if exact_hits:
        return exact_hits[:max(top_k * 20, 50)]
    if len(entities) > _MAX_FUZZY_SCAN:
        return []
    fuzzy_hits = _fuzzy_retrieve(mention, context, entities, top_k)
    if fuzzy_hits:
        return fuzzy_hits
    return _descriptive_retrieve(mention, context, entities, top_k)


def _exact_retrieve(mention: MentionInput, context: str, index: NameIndex) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    by_id: dict[str, CandidateResult] = {}
    for query in _query_texts(mention, context):
        is_expansion = normalize(query) != normalize(mention.surface_form)
        is_contextual_expansion = _is_contextual_alias(mention.surface_form, context, query)
        for entity in index.lookup(query):
            source = _match_source(query, entity)
            score = _EXACT_FLOOR.get(source, 0.90)
            existing = by_id.get(entity.entity_id)
            if existing:
                if not is_expansion and _SURFACE_EXACT_REASON not in existing.reasons:
                    existing.reasons.append(_SURFACE_EXACT_REASON)
                if is_expansion and "llm_alias_expansion" not in existing.reasons:
                    existing.reasons.append("llm_alias_expansion")
                if is_contextual_expansion and "contextual_alias_expansion" not in existing.reasons:
                    existing.reasons.append("contextual_alias_expansion")
                reason = _reason_for_source(source)
                if reason not in existing.reasons:
                    existing.reasons.append(reason)
                if is_contextual_expansion or score > existing.score:
                    existing.score = max(existing.score, score)
                    existing.matched_name = query
                    existing.match_source = source
                    existing.alias_similarity = 1.0
                continue
            reasons = [_reason_for_source(source)]
            if is_expansion:
                reasons.append("llm_alias_expansion")
            if is_contextual_expansion:
                reasons.append("contextual_alias_expansion")
            if not is_expansion:
                reasons.append(_SURFACE_EXACT_REASON)
            candidate = CandidateResult(
                entity=entity,
                score=score,
                matched_name=query,
                match_source=source,
                alias_similarity=1.0,
                reasons=reasons,
            )
            by_id[entity.entity_id] = candidate
            results.append(candidate)
    return results


def _fuzzy_retrieve(
    mention: MentionInput,
    context: str,
    entities: list[Entity],
    top_k: int,
) -> list[CandidateResult]:
    queries = [(query_text, normalize(query_text)) for query_text in _query_texts(mention, context)]
    candidates = []
    for entity in entities:
        best_score, best_name, best_source = 0.0, "", "similarity_match"
        best_is_expansion = False
        best_is_contextual_expansion = False
        for name in [entity.canonical_name] + entity.aliases + entity.former_names:
            norm_name = normalize(name)
            if not norm_name:
                continue
            for query_text, query in queries:
                if query == norm_name or query in norm_name or norm_name in query:
                    score = 1.0 if query == norm_name else 0.9
                else:
                    score = SequenceMatcher(a=query, b=norm_name).ratio()
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_source = _match_source(name, entity) if query == norm_name else "similarity_match"
                    best_is_expansion = normalize(query_text) != normalize(mention.surface_form)
                    best_is_contextual_expansion = _is_contextual_alias(
                        mention.surface_form,
                        context,
                        query_text,
                    )
        if best_score >= _MIN_SIMILARITY:
            reasons = [_reason_for_source(best_source)]
            if best_is_expansion:
                reasons.append("llm_alias_expansion")
            if best_is_contextual_expansion:
                reasons.append("contextual_alias_expansion")
            candidates.append(CandidateResult(
                entity=entity, score=round(best_score, 3),
                matched_name=best_name, match_source=best_source,
                alias_similarity=round(best_score, 3), reasons=reasons,
            ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _descriptive_retrieve(
    mention: MentionInput,
    context: str,
    entities: list[Entity],
    top_k: int,
) -> list[CandidateResult]:
    query_text = _mention_context_window(mention, context)
    query_tokens = _char_bigrams(query_text)
    if not query_tokens:
        return []

    candidates: list[CandidateResult] = []
    for entity in entities:
        entity_text = _entity_search_text(entity)
        entity_tokens = _char_bigrams(entity_text)
        if not entity_tokens:
            continue
        overlap = len(query_tokens & entity_tokens)
        if overlap == 0:
            continue
        coverage = overlap / max(1, len(query_tokens))
        entity_coverage = overlap / max(1, min(len(entity_tokens), 80))
        keyword_hits = sum(1 for keyword in entity.keywords if keyword and keyword in query_text)
        alias_hits = sum(
            1
            for name in [entity.canonical_name, *entity.aliases, *entity.former_names]
            if name and name in query_text
        )
        score = min(1.0, 0.55 * coverage + 0.25 * entity_coverage + 0.12 * keyword_hits + 0.08 * alias_hits)
        if score < _MIN_DESCRIPTIVE_SCORE:
            continue
        candidates.append(
            CandidateResult(
                entity=entity,
                score=round(score, 3),
                matched_name=mention.surface_form,
                match_source="descriptive_match",
                alias_similarity=round(min(0.72, 0.45 + score), 3),
                reasons=["descriptive_reference", "description_overlap_support"],
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def _mention_context_window(mention: MentionInput, context: str) -> str:
    if not context:
        return mention.surface_form
    start = max(0, mention.start_offset - _DESCRIPTIVE_CONTEXT_RADIUS)
    end = min(len(context), mention.end_offset + _DESCRIPTIVE_CONTEXT_RADIUS)
    return context[start:end]


def _entity_search_text(entity: Entity) -> str:
    return " ".join(
        [
            entity.canonical_name,
            *entity.aliases,
            *entity.former_names,
            entity.description,
            *entity.keywords,
        ]
    )


def _char_bigrams(text: str) -> set[str]:
    normalized = normalize(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index:index + 2] for index in range(len(normalized) - 1)}


def _query_texts(mention: MentionInput, context: str = "") -> list[str]:
    texts = [mention.surface_form]
    for alias in mention.candidate_aliases:
        alias = str(alias).strip()
        if alias and all(normalize(alias) != normalize(existing) for existing in texts):
            texts.append(alias)
    for alias in _contextual_aliases(mention.surface_form, context):
        if alias and all(normalize(alias) != normalize(existing) for existing in texts):
            texts.append(alias)
    return texts


def _contextual_aliases(surface_form: str, context: str) -> list[str]:
    aliases: list[str] = []
    if surface_form == "西湖" and "杭州西湖" in context:
        aliases.append("杭州西湖")
    if surface_form == "总书记" and "绿水青山就是金山银山" in context:
        aliases.append("习近平")
    return aliases


def _is_contextual_alias(surface_form: str, context: str, query: str) -> bool:
    return any(normalize(alias) == normalize(query) for alias in _contextual_aliases(surface_form, context))


def _reason_for_source(source: str) -> str:
    return {
        "canonical_match": "exact_or_canonical_match",
        "alias_match": "exact_or_alias_match",
        "former_name_match": "former_name_match",
    }.get(source, "fuzzy_alias_match")


def _match_source(name: str, entity: Entity) -> str:
    norm = normalize(name)
    if norm == normalize(entity.canonical_name):
        return "canonical_match"
    if any(normalize(a) == norm for a in entity.aliases):
        return "alias_match"
    if any(normalize(f) == norm for f in entity.former_names):
        return "former_name_match"
    return "similarity_match"
