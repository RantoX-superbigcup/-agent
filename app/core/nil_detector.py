from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.request import LinkOptions

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

    if _requires_review_for_weak_expansion(top):
        return "ambiguous", top

    # 歧义检测
    if len(candidates) > 1 and (top.score - candidates[1].score) < options.ambiguity_margin:
        if _has_strong_auto_accept_evidence(top, top.score - candidates[1].score):
            return "linked", top
        return "ambiguous", top

    return "linked", top


def _has_strong_auto_accept_evidence(candidate: CandidateResult, margin: float) -> bool:
    reasons = set(candidate.reasons)
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
