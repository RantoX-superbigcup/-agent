from __future__ import annotations
import logging
import re
from enum import Enum
from typing import Optional, TYPE_CHECKING
from app.models.entity import Entity
from app.models.request import MentionInput
from app.storage.index import NameIndex

if TYPE_CHECKING:
    from app.core.embedder import Embedder
    from app.storage.vector_index import VectorIndex

logger = logging.getLogger("entity_link_agent")

_SENTENCE_END = re.compile(r"[。！？!?；;]\s*")

# 精确命中保底分数
_EXACT_FLOOR: dict[str, float] = {
    "canonical_match": 0.95,
    "alias_match": 0.92,
    "short_name_match": 0.88,
}


class CandidateDecisionReason(str, Enum):
    unique_exact_match = "unique_exact_match"
    no_candidates = "no_candidates"
    multiple_candidates_need_disambiguation = "multiple_candidates_need_disambiguation"


class CandidateResult:
    __slots__ = ("entity", "rank", "raw_score", "match_source", "matched_text")

    def __init__(
        self,
        entity: Entity,
        rank: int,
        raw_score: float,
        match_source: str,
        matched_text: str | None,
    ) -> None:
        self.entity = entity
        self.rank = rank
        self.raw_score = raw_score
        self.match_source = match_source
        self.matched_text = matched_text


class CandidateListResult:
    __slots__ = ("candidates", "disambiguation_required", "reason")

    def __init__(
        self,
        candidates: list[CandidateResult],
        disambiguation_required: bool,
        reason: CandidateDecisionReason,
    ) -> None:
        self.candidates = candidates
        self.disambiguation_required = disambiguation_required
        self.reason = reason


def retrieve(
    mention: MentionInput,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    context: str = "",
    embedder: Optional["Embedder"] = None,
    vector_index: Optional["VectorIndex"] = None,
    vector_top_k: int = 20,
    semantic_min_score: float = 0.55,
) -> CandidateListResult:
    """候选召回：精确名称唯一命中则返回，否则补充 description/keywords 向量召回。"""
    exact_results = _exact_retrieve(mention, index)
    if len(exact_results) == 1:
        logger.debug("    [精确唯一召回] mention='%s'", mention.surface_form)
        return CandidateListResult(
            candidates=_assign_rank(exact_results[:top_k]),
            disambiguation_required=False,
            reason=CandidateDecisionReason.unique_exact_match,
        )

    use_vector = embedder is not None and vector_index is not None and vector_index.exists()
    if not use_vector:
        logger.debug("    [精确召回] mention='%s' candidates=%d，无向量召回", mention.surface_form, len(exact_results))
        return _wrap_non_unique(_assign_rank(_sort_and_limit(exact_results, top_k)))

    logger.debug("    [精确+语义召回] mention='%s' exact=%d vector_top_k=%d", mention.surface_form, len(exact_results), vector_top_k)
    candidates = _merge_candidates(
        exact_results,
        _vector_retrieve(mention, context, entities, vector_top_k, embedder, vector_index, semantic_min_score),
        top_k,
    )
    return _wrap_non_unique(candidates)


def _merge_candidates(
    exact_results: list[CandidateResult],
    vector_results: list[CandidateResult],
    top_k: int,
) -> list[CandidateResult]:
    merged: dict[str, CandidateResult] = {}
    for r in exact_results:
        merged[r.entity.entity_id] = r
    for r in vector_results:
        eid = r.entity.entity_id
        if eid not in merged:
            merged[eid] = r
    return _assign_rank(_sort_and_limit(list(merged.values()), top_k))


def _sort_and_limit(candidates: list[CandidateResult], top_k: int) -> list[CandidateResult]:
    return sorted(candidates, key=lambda r: r.raw_score, reverse=True)[:top_k]


def _assign_rank(candidates: list[CandidateResult]) -> list[CandidateResult]:
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


def _wrap_non_unique(candidates: list[CandidateResult]) -> CandidateListResult:
    if not candidates:
        return CandidateListResult(
            candidates=[],
            disambiguation_required=False,
            reason=CandidateDecisionReason.no_candidates,
        )
    return CandidateListResult(
        candidates=candidates,
        disambiguation_required=True,
        reason=CandidateDecisionReason.multiple_candidates_need_disambiguation,
    )


def _exact_retrieve(mention: MentionInput, index: NameIndex) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    seen_ids: set[str] = set()
    for source, entities in index.lookup_exact(mention.surface_form).items():
        for entity in entities:
            if entity.entity_id in seen_ids:
                continue
            seen_ids.add(entity.entity_id)
            results.append(CandidateResult(
                entity=entity,
                rank=0,
                raw_score=_EXACT_FLOOR[source],
                match_source=source,
                matched_text=mention.surface_form,
            ))
    return results


def _vector_retrieve(
    mention: MentionInput,
    context: str,
    entities: list[Entity],
    top_k: int,
    embedder: "Embedder",
    vector_index: "VectorIndex",
    semantic_min_score: float,
) -> list[CandidateResult]:
    query_text = f"{mention.surface_form} {_related_sentence_group(mention, context)}".strip()
    query_vec = embedder.encode_one(query_text)
    hits = vector_index.search(query_vec, top_k)
    entity_map = {e.entity_id: e for e in entities}
    results = []
    for entity_id, score in hits:
        entity = entity_map.get(entity_id)
        if entity is None:
            continue
        if float(score) < semantic_min_score:
            continue
        results.append(CandidateResult(
            entity=entity,
            rank=0,
            raw_score=round(float(score), 3),
            match_source="semantic_match",
            matched_text=None,
        ))
    return results


def _related_sentence_group(mention: MentionInput, text: str) -> str:
    if not text:
        return ""
    spans = _sentence_spans(text)
    index = _sentence_index_for_mention(spans, mention)
    if index is None:
        return text

    start = index
    while start > 0 and mention.surface_form in spans[start - 1][2]:
        start -= 1
    end = index
    while end + 1 < len(spans) and mention.surface_form in spans[end + 1][2]:
        end += 1
    return "".join(sentence for _, _, sentence in spans[start:end + 1])


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            spans.append((start, end, sentence))
        start = end
    if start < len(text):
        sentence = text[start:].strip()
        if sentence:
            spans.append((start, len(text), sentence))
    return spans or [(0, len(text), text)]


def _sentence_index_for_mention(spans: list[tuple[int, int, str]], mention: MentionInput) -> int | None:
    for i, (start, end, _) in enumerate(spans):
        if start <= mention.start_offset < end:
            return i
    for i, (_, _, sentence) in enumerate(spans):
        if mention.surface_form in sentence:
            return i
    return None
