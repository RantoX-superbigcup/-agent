from __future__ import annotations

from app.core.candidate import CandidateResult, trusted_exact_rank
from app.models.request import LinkOptions
from app.storage.index import normalize

_NIL_EXEMPT = {"canonical_match", "alias_match", "former_name_match"}
_FUZZY_SOURCES = {"canonical_fuzzy_match", "alias_fuzzy_match", "former_name_fuzzy_match"}


def decide(
    candidates: list[CandidateResult],
    options: LinkOptions,
) -> tuple[str, CandidateResult | None]:
    if not candidates:
        return "nil", None

    top = candidates[0]

    if options.enable_nil and top.match_source not in _NIL_EXEMPT:
        if top.score < options.nil_threshold:
            return "nil", top

    if _has_duplicate_canonical_auto_accept(top, candidates):
        return "linked", top

    if _is_strong_descriptive_reference(top, candidates):
        return "linked", top

    if _has_trusted_exact_priority(top, candidates):
        if "trusted_exact_priority_auto_accept" not in top.reasons:
            top.reasons.append("trusted_exact_priority_auto_accept")
        return "linked", top

    if len(candidates) > 1 and (top.score - candidates[1].score) < options.ambiguity_margin:
        if _has_strong_auto_accept_evidence(top, top.score - candidates[1].score):
            return "linked", top
        return "ambiguous", top

    return "linked", top


def _has_trusted_exact_priority(top: CandidateResult, candidates: list[CandidateResult]) -> bool:
    top_rank = trusted_exact_rank(top)
    if top_rank <= 0 or top.score < 0.55:
        return False
    if len(candidates) < 2:
        return True
    return top_rank > trusted_exact_rank(candidates[1])


def _has_strong_auto_accept_evidence(candidate: CandidateResult, margin: float) -> bool:
    reasons = set(candidate.reasons)
    has_context = bool(reasons & {"context_keyword_support", "description_overlap_support"})
    if candidate.match_source == "canonical_match" and has_context and candidate.score >= 0.80 and margin >= 0.045:
        return True
    if (
        candidate.match_source == "alias_match"
        and "context_keyword_support" in reasons
        and "description_overlap_support" in reasons
        and candidate.score >= 0.85
        and margin >= 0.05
    ):
        return True
    if (
        candidate.match_source in _FUZZY_SOURCES
        and "fuzzy_context_validated" in reasons
        and candidate.score >= 0.78
        and margin >= 0.05
    ):
        return True
    return False


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
    leading_candidates = candidates[:5]
    same_name_count = sum(
        1
        for candidate in leading_candidates
        if normalize(candidate.entity.canonical_name) == top_name
    )
    has_close_different_name = any(
        normalize(candidate.entity.canonical_name) != top_name and top.score - candidate.score < 0.08
        for candidate in leading_candidates
    )
    if same_name_count >= 2 and not has_close_different_name:
        if "duplicate_canonical_auto_accept" not in top.reasons:
            top.reasons.append("duplicate_canonical_auto_accept")
        return True
    return False
