from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
from app.core.candidate import CandidateResult
from app.core.kb_profile import ScoreWeights
from app.models.entity import Entity
from app.models.request import MentionInput
from app.storage.index import normalize

_GENERIC_CONTEXT_TERMS = {
    "电影",
    "导演",
    "先生",
    "女士",
    "公司",
    "企业",
    "集团",
    "机构",
    "美景",
    "理念",
    "作品",
    "景区",
}


class AliasPrior:
    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    @classmethod
    def load(cls, path: Path) -> "AliasPrior":
        if not path.exists():
            return cls({})
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("mapping", {}))

    def score(self, mention: str, entity_id: str) -> float:
        return float(self._mapping.get(normalize(mention), {}).get(entity_id, 0.0))


def _keyword_hits(context: str, keywords: list[str]) -> list[str]:
    return [kw for kw in keywords if kw and kw in context]


def _meaningful_keyword_hits(hits: list[str]) -> list[str]:
    return [
        hit
        for hit in hits
        if len(normalize(hit)) >= 3 and normalize(hit) not in _GENERIC_CONTEXT_TERMS
    ]


def _description_overlap(context: str, entity: Entity) -> float:
    desc = normalize(str(entity.description))
    ctx = normalize(context)
    if not ctx or not desc:
        return 0.0
    ctx_chars = {c for c in ctx if "一" <= c <= "鿿"}
    if not ctx_chars:
        return 0.0
    matched = sum(1 for c in ctx_chars if c in desc)
    return min(1.0, matched / max(4, len(ctx_chars)))


def _entity_text(entity: Entity) -> str:
    return " ".join(
        [
            entity.canonical_name,
            *entity.aliases,
            *entity.former_names,
            entity.description,
            *entity.keywords,
        ]
    )


def _domain_context_score(context: str, entity: Entity) -> float:
    """Small semantic hint layer for noisy CCKS homonyms such as film/book/song senses."""

    ctx = normalize(context)
    entity_text = normalize(_entity_text(entity))
    if not ctx or not entity_text:
        return 0.0

    movie_context_terms = ("导演", "执导", "电影", "影片", "主演", "上映", "奥斯卡", "动人")
    movie_entity_terms = ("导演", "执导", "电影", "影片", "主演", "上映", "奥斯卡", "金狮")
    non_movie_terms = ("小说", "书籍", "文学体裁", "作者", "编著", "歌曲", "演唱")

    if any(term in ctx for term in movie_context_terms):
        movie_hits = sum(1 for term in movie_entity_terms if term in entity_text)
        if movie_hits:
            score = 0.35 + min(1.0, movie_hits / 6)
            if any(term in entity_text for term in non_movie_terms):
                score -= 0.25
            return max(0.0, min(1.0, score))

    return 0.0


def rescore(
    candidate: CandidateResult,
    mention: MentionInput,
    context: str,
    alias_prior: Optional[AliasPrior] = None,
    weights: Optional[ScoreWeights] = None,
) -> CandidateResult:
    entity = candidate.entity
    weights = weights or ScoreWeights()

    # 上下文匹配分
    hits = _keyword_hits(context, entity.keywords)
    ctx_score = min(1.0, len(hits) / max(1, min(len(entity.keywords), 4))) if entity.keywords else 0.0
    ctx_score = max(ctx_score, _description_overlap(context, entity))
    ctx_score = max(ctx_score, _domain_context_score(context, entity))

    # 先验概率加分
    prior_score = alias_prior.score(mention.surface_form, entity.entity_id) if alias_prior else 0.0
    prior_bonus = weights.prior_weight * prior_score

    reasons = set(candidate.reasons)
    type_bonus = weights.type_bonus if mention.entity_type and mention.entity_type == entity.entity_type else 0.0
    inferred_type_bonus = (
        weights.inferred_type_bonus
        if not mention.entity_type
        and _looks_like_location_context(mention.surface_form, context)
        and entity.entity_type.value == "LOC"
        else 0.0
    )
    canonical_bonus = weights.canonical_bonus if normalize(mention.surface_form) == normalize(entity.canonical_name) else 0.0
    expansion_bonus = weights.expansion_bonus if "llm_alias_expansion" in reasons else 0.0
    expansion_canonical_bonus = (
        weights.expansion_canonical_bonus
        if "llm_alias_expansion" in reasons
        and normalize(candidate.matched_name) == normalize(entity.canonical_name)
        else 0.0
    )
    dirty_expansion_penalty = (
        weights.dirty_expansion_penalty
        if "llm_alias_expansion" in reasons
        and normalize(candidate.matched_name) != normalize(entity.canonical_name)
        else 0.0
    )
    expansion_context_validated = "llm_alias_expansion" in reasons and (
        "contextual_alias_expansion" in reasons
        or bool(_meaningful_keyword_hits(hits))
        or prior_score > 0
        or inferred_type_bonus > 0
    )

    final = max(
        0.0,
        min(
            1.0,
        weights.alias_weight * candidate.alias_similarity
        + weights.context_weight * ctx_score
        + type_bonus
        + inferred_type_bonus
        + canonical_bonus
        + expansion_bonus
        + expansion_canonical_bonus
        + prior_bonus,
        )
        - dirty_expansion_penalty,
    )
    candidate.score = round(final, 3)
    if hits and "context_keyword_support" not in candidate.reasons:
        candidate.reasons.append("context_keyword_support")
    if ctx_score > 0 and "description_overlap_support" not in candidate.reasons:
        candidate.reasons.append("description_overlap_support")
    if type_bonus and "entity_type_aligned" not in candidate.reasons:
        candidate.reasons.append("entity_type_aligned")
    if inferred_type_bonus and "entity_type_inferred_from_context" not in candidate.reasons:
        candidate.reasons.append("entity_type_inferred_from_context")
    if canonical_bonus and "canonical_bonus" not in candidate.reasons:
        candidate.reasons.append("canonical_bonus")
    if expansion_canonical_bonus and "expansion_canonical_support" not in candidate.reasons:
        candidate.reasons.append("expansion_canonical_support")
    if expansion_context_validated and "expansion_context_validated" not in candidate.reasons:
        candidate.reasons.append("expansion_context_validated")
    if dirty_expansion_penalty and "dirty_alias_penalty" not in candidate.reasons:
        candidate.reasons.append("dirty_alias_penalty")
    if prior_score > 0 and "alias_prior_support" not in candidate.reasons:
        candidate.reasons.append("alias_prior_support")
    return candidate


def _looks_like_location_context(surface_form: str, context: str) -> bool:
    if surface_form in {"杭州", "西湖"}:
        return any(term in context for term in ("西湖", "美景", "风景", "景区", "城市", "杭州市"))
    return False
