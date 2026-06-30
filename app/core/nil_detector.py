from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.request import LinkOptions
from app.storage.index import normalize

# 精确命中豁免 NIL 检测
_NIL_EXEMPT = {"canonical_match", "alias_match", "former_name_match"}


def decide(
    candidates: list[CandidateResult],
    options: LinkOptions,
) -> tuple[str, CandidateResult | None]:
    """返回 (status, top_candidate)。status: linked | nil | ambiguous"""
    if not candidates:
        return "nil", None

    top = candidates[0]

    # 精确命中 → 跳过 NIL 阈值检测
    if options.enable_nil and top.match_source not in _NIL_EXEMPT:
        if top.score < options.nil_threshold:
            return "nil", top

    # CCKS 里常见同一标准名对应多条实体记录。若近邻候选都是同名实体，
    # 这里优先当作知识库脏数据处理，而不是让 LLM 别名扩展直接进入复核。
    if _has_duplicate_canonical_auto_accept(top, candidates):
        return "linked", top

    if _requires_review_for_weak_expansion(top):
        return "ambiguous", top

    if _is_strong_descriptive_reference(top, candidates):
        return "linked", top

    # 歧义检测
    if len(candidates) > 1 and (top.score - candidates[1].score) < options.ambiguity_margin:
        if _has_strong_auto_accept_evidence(top, top.score - candidates[1].score):
            return "linked", top
        return "ambiguous", top

    return "linked", top


def _has_strong_auto_accept_evidence(candidate: CandidateResult, margin: float) -> bool:
    reasons = set(candidate.reasons)
    if (
        "contextual_alias_expansion" in reasons
        and "expansion_context_validated" in reasons
        and candidate.score >= 0.88
        and margin >= 0.02
    ):
        return True
    if (
        "llm_alias_expansion" in reasons
        and "expansion_context_validated" in reasons
        and candidate.score >= 0.90
        and margin >= 0.03
    ):
        return True
    has_context = bool(reasons & {"context_keyword_support", "description_overlap_support"})
    if candidate.match_source == "canonical_match" and has_context and candidate.score >= 0.80 and margin >= 0.045:
        return True
    return False


def _requires_review_for_weak_expansion(candidate: CandidateResult) -> bool:
    reasons = set(candidate.reasons)
    return "llm_alias_expansion" in reasons and "expansion_context_validated" not in reasons


def _is_strong_descriptive_reference(top: CandidateResult, candidates: list[CandidateResult]) -> bool:
    reasons = set(top.reasons)
    if "descriptive_reference" not in reasons or top.score < 0.68:
        return False
    if len(candidates) > 1 and top.score - candidates[1].score < 0.06:
        return False
    if "descriptive_reference_auto_accept" not in top.reasons:
        top.reasons.append("descriptive_reference_auto_accept")
    return True


def _has_duplicate_canonical_auto_accept(top: CandidateResult, candidates: list[CandidateResult]) -> bool:
    reasons = set(top.reasons)
    has_context = bool(reasons & {"context_keyword_support", "description_overlap_support"})
    if top.match_source != "canonical_match" or not has_context or top.score < 0.74:
        return False

    top_name = normalize(top.entity.canonical_name)
    close_candidates = [candidate for candidate in candidates[:5] if top.score - candidate.score < 0.08]
    same_name_count = sum(1 for candidate in close_candidates if normalize(candidate.entity.canonical_name) == top_name)
    has_close_different_name = any(normalize(candidate.entity.canonical_name) != top_name for candidate in close_candidates)
    if same_name_count >= 2 and not has_close_different_name:
        if "duplicate_canonical_auto_accept" not in top.reasons:
            top.reasons.append("duplicate_canonical_auto_accept")
        return True
    return False
