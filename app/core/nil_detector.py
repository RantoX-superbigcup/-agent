from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.request import MentionInput, LinkOptions


def decide(
    candidates: list[CandidateResult],
    options: LinkOptions,
) -> tuple[str, CandidateResult | None]:
    """Return (status, top_candidate). status: linked | nil | ambiguous"""
    if not candidates:
        return "nil", None
    top = candidates[0]
    if options.enable_nil and top.score < options.nil_threshold:
        return "nil", top
    if len(candidates) > 1 and (top.score - candidates[1].score) < 0.08:
        return "ambiguous", top
    return "linked", top
