from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
from app.core.candidate import CandidateResult
from app.models.entity import Entity
from app.models.request import MentionInput
from app.storage.index import normalize


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


def _description_overlap(context: str, entity: Entity) -> float:
    desc = normalize(str(entity.description))
    ctx = normalize(context)
    if not ctx or not desc:
        return 0.0
    ctx_chars = {c for c in ctx if "\u4e00" <= c <= "\u9fff"}
    if not ctx_chars:
        return 0.0
    matched = sum(1 for c in ctx_chars if c in desc)
    return min(1.0, matched / max(4, len(ctx_chars)))


def rescore(
    candidate: CandidateResult,
    mention: MentionInput,
    context: str,
    alias_prior: Optional[AliasPrior] = None,
) -> CandidateResult:
    entity = candidate.entity
    hits = _keyword_hits(context, entity.keywords)
    ctx_score = min(1.0, len(hits) / max(1, min(len(entity.keywords), 4))) if entity.keywords else 0.0
    ctx_score = max(ctx_score, _description_overlap(context, entity))
    canonical_bonus = 0.03 if normalize(mention.surface_form) == normalize(entity.canonical_name) else 0.0
    prior_bonus = 0.22 * (alias_prior.score(mention.surface_form, entity.entity_id) if alias_prior else 0.0)
    final = min(1.0, 0.62 * candidate.score + 0.23 * ctx_score + canonical_bonus + prior_bonus)
    candidate.score = round(final, 3)
    return candidate
