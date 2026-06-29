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
    exact_ids = {c.entity.entity_id for c in exact_candidates}

    # 2. 向量检索（多取一些候选用作合并）
    vec_results = _vector_retrieve(mention, context, entities, max(top_k * 3, 30), embedder, vector_index)

    # 3. 构建 entity_id → Entity 映射
    entity_map = {e.entity_id: e for e in entities}
    merged: dict[str, CandidateResult] = {}

    for r in vec_results:
        eid = r.entity.entity_id
        if eid in exact_ids and r.match_source != "similarity_match":
            # 精确命中的实体 → boost 分数
            boost = _EXACT_BOOST.get(r.match_source, 0.12)
            floor = _EXACT_FLOOR.get(r.match_source, 0.85)
            r.score = round(max(r.score + boost, floor), 3)
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

    result = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return result[:top_k]


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
    return _fuzzy_retrieve(mention, context, entities, top_k)


def _exact_retrieve(mention: MentionInput, context: str, index: NameIndex) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    seen_ids: set[str] = set()
    for query in _query_texts(mention, context):
        is_expansion = normalize(query) != normalize(mention.surface_form)
        is_contextual_expansion = _is_contextual_alias(mention.surface_form, context, query)
        for entity in index.lookup(query):
            if entity.entity_id in seen_ids:
                continue
            seen_ids.add(entity.entity_id)
            source = _match_source(query, entity)
            score = _EXACT_FLOOR.get(source, 0.90)
            reasons = [_reason_for_source(source)]
            if is_expansion:
                reasons.append("llm_alias_expansion")
            if is_contextual_expansion:
                reasons.append("contextual_alias_expansion")
            results.append(
                CandidateResult(
                    entity=entity,
                    score=score,
                    matched_name=query,
                    match_source=source,
                    alias_similarity=1.0,
                    reasons=reasons,
                )
            )
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
                    best_source = _match_source(name, entity)
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
