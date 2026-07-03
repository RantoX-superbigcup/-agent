from __future__ import annotations
from app.core.candidate import CandidateResult
from app.models.response import EvidenceItem
from app.models.enums import EvidenceType
from app.storage.index import normalize


def build_evidence(candidate: CandidateResult, context: str) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    source = candidate.match_source
    if source == "canonical_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.canonical_match,
            detail=f"surface_form 命中标准名：{candidate.matched_name}",
        ))
    elif source == "alias_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.alias_match,
            detail=f"surface_form 命中别名：{candidate.matched_name}",
        ))
    elif source == "short_name_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.short_name_match,
            detail=f"surface_form 命中简称：{candidate.matched_name}",
        ))
    elif source == "semantic_match":
        items.append(EvidenceItem(
            evidence_type=EvidenceType.semantic_match,
            detail=f"语义向量召回：{candidate.matched_name}，score={candidate.score}",
        ))
    entity = candidate.entity
    kw_hits = [kw for kw in entity.keywords if kw and kw in context]
    if kw_hits:
        items.append(EvidenceItem(
            evidence_type=EvidenceType.context_match,
            detail=f"上下文包含关键词：{'、'.join(kw_hits[:5])}",
        ))
    return items
