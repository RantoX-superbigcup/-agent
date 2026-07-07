from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app.models.entity import Entity
from app.models.request import WorkflowMentionInput as MentionInput
from app.storage.index import EntityNameRecord, NameIndex, normalize

if TYPE_CHECKING:
    from app.core.embedder import Embedder
    from app.storage.vector_index import VectorIndex


RECALL_STATUS_RETRIEVED = "retrieved"
RECALL_STATUS_EMPTY = "empty"
RECALL_STATUS_SKIPPED_COREFERENCE = "skipped_coreference"

RECALL_SOURCE_EXACT = "exact"
RECALL_SOURCE_FUZZY = "fuzzy"
RECALL_SOURCE_VECTOR = "vector"

MATCH_SLOT_CANONICAL = "canonical"
MATCH_SLOT_ALIAS = "alias"
MATCH_SLOT_FORMER_NAME = "former_name"
MATCH_SLOT_SEMANTIC = "semantic"

ROUTE_DIRECT_LINK = "direct_link"
ROUTE_NEED_DISAMBIGUATION = "need_disambiguation"
ROUTE_NIL_PENDING = "nil_pending"
ROUTE_COREFERENCE_PENDING = "coreference_pending"

_EXACT_FLOOR: dict[str, float] = {
    "canonical_match": 0.95,
    "alias_match": 0.92,
    "former_name_match": 0.88,
}
_TRUSTED_EXACT_RANK: dict[str, int] = {
    "canonical_match": 3,
    "alias_match": 2,
    "former_name_match": 1,
}
_FUZZY_SOURCE_BY_NAME_SOURCE = {
    "canonical": "canonical_fuzzy_match",
    "alias": "alias_fuzzy_match",
    "former_name": "former_name_fuzzy_match",
}
_NAME_SOURCE_PRIORITY = {
    "canonical": 3,
    "alias": 2,
    "former_name": 1,
}
_SEMANTIC_REASON = "semantic_vector_retrieval"

_FUZZY_EDIT_WEIGHT = 0.25
_FUZZY_LCS_WEIGHT = 0.25
_FUZZY_BIGRAM_WEIGHT = 0.10
_FUZZY_IDF_WEIGHT = 0.40

_MATCH_SOURCE_TO_RECALL_SOURCE = {
    "canonical_match": RECALL_SOURCE_EXACT,
    "alias_match": RECALL_SOURCE_EXACT,
    "former_name_match": RECALL_SOURCE_EXACT,
    "canonical_fuzzy_match": RECALL_SOURCE_FUZZY,
    "alias_fuzzy_match": RECALL_SOURCE_FUZZY,
    "former_name_fuzzy_match": RECALL_SOURCE_FUZZY,
    "semantic_match": RECALL_SOURCE_VECTOR,
}
_MATCH_SOURCE_TO_MATCH_SLOT = {
    "canonical_match": MATCH_SLOT_CANONICAL,
    "alias_match": MATCH_SLOT_ALIAS,
    "former_name_match": MATCH_SLOT_FORMER_NAME,
    "canonical_fuzzy_match": MATCH_SLOT_CANONICAL,
    "alias_fuzzy_match": MATCH_SLOT_ALIAS,
    "former_name_fuzzy_match": MATCH_SLOT_FORMER_NAME,
    "semantic_match": MATCH_SLOT_SEMANTIC,
}


@dataclass(frozen=True)
class RecallCandidateRef:
    entity_id: str
    recall_source: str
    match_slot: str


@dataclass(frozen=True)
class MentionRecallResult:
    mention_id: str
    surface_form: str
    recall_status: str
    candidates: tuple[RecallCandidateRef, ...]


class CandidateResult:
    __slots__ = (
        "entity",
        "score",
        "matched_name",
        "match_source",
        "alias_similarity",
        "reasons",
        "score_components",
        "retrieval_stage",
    )

    def __init__(
        self,
        entity: Entity,
        score: float,
        matched_name: str,
        match_source: str,
        alias_similarity: float | None = None,
        reasons: list[str] | None = None,
        score_components: dict[str, float] | None = None,
        retrieval_stage: str = "exact",
    ) -> None:
        self.entity = entity
        self.score = score
        self.matched_name = matched_name
        self.match_source = match_source
        self.alias_similarity = score if alias_similarity is None else alias_similarity
        self.reasons = reasons or []
        self.score_components = score_components or {}
        self.retrieval_stage = retrieval_stage


def recall(
    mention: MentionInput,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    context: str = "",
    embedder: Optional["Embedder"] = None,
    vector_index: Optional["VectorIndex"] = None,
    candidate_pool_limit: int | None = None,
) -> MentionRecallResult:
    candidates = retrieve(
        mention,
        index,
        entities,
        top_k,
        context=context,
        embedder=embedder,
        vector_index=vector_index,
        candidate_pool_limit=candidate_pool_limit,
    )
    if not candidates:
        return MentionRecallResult(
            mention_id=mention.mention_id,
            surface_form=mention.surface_form,
            recall_status=RECALL_STATUS_EMPTY,
            candidates=(),
        )
    return MentionRecallResult(
        mention_id=mention.mention_id,
        surface_form=mention.surface_form,
        recall_status=RECALL_STATUS_RETRIEVED,
        candidates=tuple(candidate_to_ref(candidate) for candidate in candidates),
    )


def skipped_recall(
    mention: MentionInput,
    *,
    recall_status: str = RECALL_STATUS_SKIPPED_COREFERENCE,
) -> MentionRecallResult:
    return MentionRecallResult(
        mention_id=mention.mention_id,
        surface_form=mention.surface_form,
        recall_status=recall_status,
        candidates=(),
    )


def materialize_recall(
    mention: MentionInput,
    recall_result: MentionRecallResult,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    context: str = "",
    embedder: Optional["Embedder"] = None,
    vector_index: Optional["VectorIndex"] = None,
    candidate_pool_limit: int | None = None,
) -> list[CandidateResult]:
    if recall_result.recall_status != RECALL_STATUS_RETRIEVED or not recall_result.candidates:
        return []

    candidates = retrieve(
        mention,
        index,
        entities,
        top_k,
        context=context,
        embedder=embedder,
        vector_index=vector_index,
        candidate_pool_limit=candidate_pool_limit,
    )
    allowed = {_recall_ref_key(ref) for ref in recall_result.candidates}
    return [
        candidate
        for candidate in candidates
        if _candidate_recall_key(candidate) in allowed
    ]


def candidate_to_ref(candidate: CandidateResult) -> RecallCandidateRef:
    return RecallCandidateRef(
        entity_id=candidate.entity.entity_id,
        recall_source=recall_source_for_match_source(candidate.match_source),
        match_slot=match_slot_for_match_source(candidate.match_source),
    )


def recall_source_for_match_source(match_source: str) -> str:
    return _MATCH_SOURCE_TO_RECALL_SOURCE.get(match_source, RECALL_SOURCE_FUZZY)


def match_slot_for_match_source(match_source: str) -> str:
    return _MATCH_SOURCE_TO_MATCH_SLOT.get(match_source, MATCH_SLOT_SEMANTIC)


def retrieve(
    mention: MentionInput,
    index: NameIndex,
    entities: list[Entity],
    top_k: int,
    context: str = "",
    embedder: Optional["Embedder"] = None,
    vector_index: Optional["VectorIndex"] = None,
    candidate_pool_limit: int | None = None,
) -> list[CandidateResult]:
    del entities

    candidate_pool_limit = candidate_pool_limit or max(1, top_k + 5)

    exact_candidates = _exact_retrieve(mention, index)
    if exact_candidates:
        return sorted(exact_candidates, key=rank_key, reverse=True)

    fuzzy_candidates = _fuzzy_retrieve(mention, index, candidate_pool_limit)
    use_vector = embedder is not None and vector_index is not None and vector_index.exists()
    if not use_vector or len(fuzzy_candidates) >= candidate_pool_limit:
        return fuzzy_candidates[:candidate_pool_limit]

    vector_candidates = _vector_retrieve(
        mention,
        context,
        index.all_entities(),
        candidate_pool_limit - len(fuzzy_candidates),
        embedder,
        vector_index,
    )
    return _merge_non_exact_candidates(fuzzy_candidates, vector_candidates, candidate_pool_limit)


def trusted_exact_rank(candidate: CandidateResult) -> int:
    return _TRUSTED_EXACT_RANK.get(candidate.match_source, 0)


def rank_key(candidate: CandidateResult) -> tuple[int, float]:
    return candidate.score, trusted_exact_rank(candidate)


def is_exact_match_source(match_source: str) -> bool:
    return match_source in _TRUSTED_EXACT_RANK


def _exact_retrieve(mention: MentionInput, index: NameIndex) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for hit in index.exact_lookup(mention.surface_form):
        source = hit.match_source
        results.append(
            CandidateResult(
                entity=hit.entity,
                score=_EXACT_FLOOR.get(source, 0.90),
                matched_name=hit.matched_name,
                match_source=source,
                alias_similarity=1.0,
                reasons=[_reason_for_source(source), "surface_form_exact_match"],
                retrieval_stage="exact",
            )
        )
    return results


def _fuzzy_retrieve(
    mention: MentionInput,
    index: NameIndex,
    candidate_pool_limit: int,
) -> list[CandidateResult]:
    surface_form = str(mention.surface_form or "")
    threshold = _fuzzy_threshold(len(surface_form))
    if threshold > 1.0:
        return []

    results: list[CandidateResult] = []
    for entity, name_records in index.iter_entity_name_records():
        best_candidate: CandidateResult | None = None
        best_source_priority = -1
        for record in name_records:
            score, components = _score_name(surface_form, record, index)
            if score < threshold:
                continue

            rounded_score = round(score, 3)
            source_priority = _NAME_SOURCE_PRIORITY.get(record.name_source, 0)
            should_replace = best_candidate is None or rounded_score > best_candidate.score
            if not should_replace and best_candidate is not None and rounded_score == best_candidate.score:
                should_replace = source_priority > best_source_priority

            if should_replace:
                best_candidate = CandidateResult(
                    entity=entity,
                    score=rounded_score,
                    matched_name=record.name_text,
                    match_source=_FUZZY_SOURCE_BY_NAME_SOURCE[record.name_source],
                    alias_similarity=rounded_score,
                    reasons=["fuzzy_name_recall"],
                    score_components=components,
                    retrieval_stage="fuzzy",
                )
                best_source_priority = source_priority

        if best_candidate is not None:
            results.append(best_candidate)

    return sorted(results, key=rank_key, reverse=True)[:candidate_pool_limit]


def _vector_retrieve(
    mention: MentionInput,
    context: str,
    entities: list[Entity],
    top_k: int,
    embedder: "Embedder",
    vector_index: "VectorIndex",
) -> list[CandidateResult]:
    if top_k <= 0:
        return []

    query_text = _build_vector_query(mention, context)
    query_vec = embedder.encode_one(query_text)
    hits = vector_index.search(query_vec, top_k)
    entity_map = {entity.entity_id: entity for entity in entities}

    results: list[CandidateResult] = []
    for entity_id, score in hits:
        entity = entity_map.get(entity_id)
        if entity is None:
            continue
        rounded_score = round(float(score), 3)
        results.append(
            CandidateResult(
                entity=entity,
                score=rounded_score,
                matched_name=entity.canonical_name,
                match_source="semantic_match",
                alias_similarity=rounded_score,
                reasons=[_SEMANTIC_REASON],
                retrieval_stage="vector",
            )
        )
    return results


def _merge_non_exact_candidates(
    fuzzy_candidates: list[CandidateResult],
    vector_candidates: list[CandidateResult],
    candidate_pool_limit: int,
) -> list[CandidateResult]:
    merged: dict[str, CandidateResult] = {
        candidate.entity.entity_id: candidate for candidate in fuzzy_candidates
    }
    for vector_candidate in vector_candidates:
        entity_id = vector_candidate.entity.entity_id
        existing = merged.get(entity_id)
        if existing is None:
            merged[entity_id] = vector_candidate
            continue
        if _SEMANTIC_REASON not in existing.reasons:
            existing.reasons.append(_SEMANTIC_REASON)

    return sorted(merged.values(), key=rank_key, reverse=True)[:candidate_pool_limit]


def _score_name(surface_form: str, record: EntityNameRecord, index: NameIndex) -> tuple[float, dict[str, float]]:
    candidate_name = record.name_text
    if not surface_form or not candidate_name:
        return 0.0, {}

    edit_similarity = _edit_similarity(surface_form, candidate_name)
    lcs_similarity = _lcs_similarity(surface_form, candidate_name)
    bigram_similarity = _bigram_similarity(surface_form, candidate_name)
    idf_overlap = _idf_overlap_similarity(surface_form, candidate_name, index)

    total = (
        _FUZZY_EDIT_WEIGHT * edit_similarity
        + _FUZZY_LCS_WEIGHT * lcs_similarity
        + _FUZZY_BIGRAM_WEIGHT * bigram_similarity
        + _FUZZY_IDF_WEIGHT * idf_overlap
    )
    return total, {
        "edit_similarity": round(edit_similarity, 3),
        "lcs_similarity": round(lcs_similarity, 3),
        "bigram_similarity": round(bigram_similarity, 3),
        "idf_overlap": round(idf_overlap, 3),
    }


def _edit_similarity(left: str, right: str) -> float:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 0.0
    distance = _levenshtein_distance(left, right)
    return max(0.0, 1.0 - distance / max_len)


def _lcs_similarity(left: str, right: str) -> float:
    total_len = len(left) + len(right)
    if total_len == 0:
        return 0.0
    return (2.0 * _lcs_length(left, right)) / total_len


def _bigram_similarity(left: str, right: str) -> float:
    left_bigrams = _char_bigrams(left)
    right_bigrams = _char_bigrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0
    overlap = len(left_bigrams & right_bigrams)
    return (2.0 * overlap) / (len(left_bigrams) + len(right_bigrams))


def _idf_overlap_similarity(left: str, right: str, index: NameIndex) -> float:
    left_chars = set(left)
    right_chars = set(right)
    if not left_chars or not right_chars:
        return 0.0

    overlap_chars = left_chars & right_chars
    if not overlap_chars:
        return 0.0

    overlap_weight = sum(index.char_idf(char) for char in overlap_chars)
    left_weight = sum(index.char_idf(char) for char in left_chars)
    if left_weight <= 0:
        return 0.0
    return overlap_weight / left_weight


def _fuzzy_threshold(length: int) -> float:
    if length <= 1:
        return 1.01
    if length == 2:
        return 0.52
    if length == 3:
        return 0.55
    if length <= 6:
        return 0.58
    return 0.55


def _build_vector_query(mention: MentionInput, context: str) -> str:
    sentence_window = _sentence_window(mention, context)
    return " ".join(part for part in (mention.surface_form, sentence_window) if part).strip()


def _sentence_window(mention: MentionInput, context: str) -> str:
    text = str(context or "")
    if not text:
        return ""

    spans = _sentence_spans(text)
    if not spans:
        return text.strip()

    target_index = None
    for index, (start, end, _) in enumerate(spans):
        if start <= mention.start_offset < end or start < mention.end_offset <= end:
            target_index = index
            break
    if target_index is None:
        return text.strip()

    current = spans[target_index][2].strip()
    if len(normalize(current)) >= 12:
        return current

    parts: list[str] = []
    if target_index > 0:
        parts.append(spans[target_index - 1][2].strip())
    parts.append(current)
    if target_index + 1 < len(spans):
        parts.append(spans[target_index + 1][2].strip())
    return " ".join(part for part in parts if part)


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    sentence_endings = {".", "!", "?", ";", "\u3002", "\uff01", "\uff1f", "\uff1b"}
    for index, char in enumerate(text):
        if char not in sentence_endings:
            continue
        end = index + 1
        spans.append((start, end, text[start:end]))
        start = end
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    return [span for span in spans if span[2].strip()]


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            substitution = previous[col_index - 1] + (0 if left_char == right_char else 1)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for col_index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[col_index - 1] + 1)
            else:
                current.append(max(current[-1], previous[col_index]))
        previous = current
    return previous[-1]


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return set()
    return {text[index:index + 2] for index in range(len(text) - 1)}


def _reason_for_source(source: str) -> str:
    return {
        "canonical_match": "exact_or_canonical_match",
        "alias_match": "exact_or_alias_match",
        "former_name_match": "former_name_match",
    }.get(source, "fuzzy_name_recall")


def _recall_ref_key(ref: RecallCandidateRef) -> tuple[str, str, str]:
    return (ref.entity_id, ref.recall_source, ref.match_slot)


def _candidate_recall_key(candidate: CandidateResult) -> tuple[str, str, str]:
    return (
        candidate.entity.entity_id,
        recall_source_for_match_source(candidate.match_source),
        match_slot_for_match_source(candidate.match_source),
    )
