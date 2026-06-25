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
    __slots__ = ("entity", "score", "matched_name", "match_source")

    def __init__(self, entity: Entity, score: float, matched_name: str, match_source: str) -> None:
        self.entity = entity
        self.score = score
        self.matched_name = matched_name
        self.match_source = match_source


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
        return _rule_retrieve(mention, index, entities, top_k)

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
    exact_entities = index.lookup(mention.surface_form)
    exact_ids = {e.entity_id for e in exact_entities}

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
    for entity in exact_entities:
        eid = entity.entity_id
        if eid not in merged:
            source = _match_source(mention.surface_form, entity)
            score = _EXACT_FLOOR.get(source, 0.90)
            merged[eid] = CandidateResult(
                entity=entity, score=score,
                matched_name=mention.surface_form, match_source=source,
            )

    # 5. 如果候选不够，补上模糊匹配
    if len(merged) < top_k and len(entities) <= _MAX_FUZZY_SCAN:
        fuzzy = _fuzzy_retrieve(mention, entities, top_k * 2)
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
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
) -> list[CandidateResult]:
    """纯规则召回（向量索引用不了时降级使用）。"""
    exact_hits = index.lookup(mention.surface_form)
    if exact_hits:
        return [
            CandidateResult(
                entity=e, score=1.0, matched_name=mention.surface_form,
                match_source=_match_source(mention.surface_form, e),
            )
            for e in exact_hits
        ][:max(top_k * 20, 50)]
    if len(entities) > _MAX_FUZZY_SCAN:
        return []
    return _fuzzy_retrieve(mention, entities, top_k)


def _fuzzy_retrieve(
    mention: MentionInput,
    entities: list[Entity],
    top_k: int,
) -> list[CandidateResult]:
    query = normalize(mention.surface_form)
    candidates = []
    for entity in entities:
        best_score, best_name, best_source = 0.0, "", "similarity_match"
        for name in [entity.canonical_name] + entity.aliases + entity.former_names:
            norm_name = normalize(name)
            if not norm_name:
                continue
            if query == norm_name or query in norm_name or norm_name in query:
                score = 1.0 if query == norm_name else 0.9
            else:
                score = SequenceMatcher(a=query, b=norm_name).ratio()
            if score > best_score:
                best_score, best_name, best_source = score, name, _match_source(name, entity)
        if best_score >= _MIN_SIMILARITY:
            candidates.append(CandidateResult(
                entity=entity, score=round(best_score, 3),
                matched_name=best_name, match_source=best_source,
            ))
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
