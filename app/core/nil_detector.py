from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.request import LinkOptions

# 精确命中豁免 NIL 检测
_NIL_EXEMPT = {"canonical_match", "alias_match", "short_name_match"}


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

    # 歧义检测
    if len(candidates) > 1 and (top.score - candidates[1].score) < 0.08:
        return "ambiguous", top

    return "linked", top
