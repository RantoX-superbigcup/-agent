"""Core dataclasses used by the entity linking workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MentionRecord:
    mention_id: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    entity_type: Optional[str] = None
    sentence: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBaseEntity:
    entity_id: str
    canonical_name: str
    aliases: list[str]
    entity_type: str
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBaseSnapshot:
    kb_id: str
    version: str
    entities: list[KnowledgeBaseEntity]


@dataclass
class LinkOptions:
    top_k_candidates: int = 5
    nil_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    enable_coreference: bool = True
    return_candidates: bool = True


@dataclass
class CandidateScore:
    entity_id: str
    canonical_name: str
    entity_type: str
    score: float
    alias_similarity: float
    matched_alias: Optional[str] = None
    overlapping_keywords: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    normalized_mention: str
    matched_alias: Optional[str]
    context_snippet: str
    overlapping_keywords: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


@dataclass
class LinkDecision:
    mention_id: str
    text: str
    entity_type: Optional[str]
    linked_entity_id: Optional[str]
    canonical_name: Optional[str]
    status: str
    confidence: float
    needs_review: bool
    candidates: list[CandidateScore]
    evidence: EvidenceRecord
    coreference_source_mention_id: Optional[str] = None
