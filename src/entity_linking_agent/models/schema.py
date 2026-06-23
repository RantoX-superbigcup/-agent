"""API request and response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from entity_linking_agent.core.contracts import KnowledgeBaseEntity, LinkOptions, MentionRecord


class MentionView(BaseModel):
    mention_id: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    entity_type: Optional[str] = None
    sentence: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_core(self) -> MentionRecord:
        return MentionRecord(
            mention_id=self.mention_id,
            text=self.text,
            start=self.start,
            end=self.end,
            entity_type=self.entity_type,
            sentence=self.sentence,
            metadata=self.metadata,
        )


class LinkOptionsView(BaseModel):
    top_k_candidates: int = 5
    nil_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    enable_coreference: bool = True
    return_candidates: bool = True

    def to_core(self) -> LinkOptions:
        return LinkOptions(
            top_k_candidates=self.top_k_candidates,
            nil_threshold=self.nil_threshold,
            ambiguity_margin=self.ambiguity_margin,
            enable_coreference=self.enable_coreference,
            return_candidates=self.return_candidates,
        )


class KnowledgeBaseEntityView(BaseModel):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_core(self) -> KnowledgeBaseEntity:
        return KnowledgeBaseEntity(
            entity_id=self.entity_id,
            canonical_name=self.canonical_name,
            aliases=self.aliases,
            entity_type=self.entity_type,
            keywords=self.keywords,
            metadata=self.metadata,
        )


class InlineKnowledgeBaseView(BaseModel):
    kb_id: str = "inline"
    entities: list[KnowledgeBaseEntityView]


class LinkRequestView(BaseModel):
    text: str
    mentions: list[MentionView]
    knowledge_base_id: str = "sample-energy-v1"
    inline_knowledge_base: Optional[InlineKnowledgeBaseView] = None
    options: LinkOptionsView = Field(default_factory=LinkOptionsView)
    trace_id: Optional[str] = None

    def to_service_kwargs(self) -> dict[str, Any]:
        inline_entities = None
        knowledge_base_id = self.knowledge_base_id

        if self.inline_knowledge_base is not None:
            inline_entities = [item.to_core() for item in self.inline_knowledge_base.entities]
            knowledge_base_id = self.inline_knowledge_base.kb_id

        return {
            "text": self.text,
            "mentions": [item.to_core() for item in self.mentions],
            "knowledge_base_id": knowledge_base_id,
            "inline_entities": inline_entities,
            "options": self.options.to_core(),
            "trace_id": self.trace_id,
        }


class CandidateView(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    score: float
    alias_similarity: float
    matched_alias: Optional[str] = None
    overlapping_keywords: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class EvidenceView(BaseModel):
    normalized_mention: str
    matched_alias: Optional[str] = None
    context_snippet: str
    overlapping_keywords: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class LinkResultView(BaseModel):
    mention_id: str
    text: str
    entity_type: Optional[str] = None
    linked_entity_id: Optional[str] = None
    canonical_name: Optional[str] = None
    status: str
    confidence: float
    needs_review: bool
    candidates: list[CandidateView] = Field(default_factory=list)
    evidence: EvidenceView
    coreference_source_mention_id: Optional[str] = None


class SummaryView(BaseModel):
    total_mentions: int
    linked: int
    ambiguous: int
    nil: int
    review_required: int


class DecisionLogItemView(BaseModel):
    trace_id: str
    mention_id: str
    status: str
    linked_entity_id: Optional[str] = None
    confidence: float
    needs_review: bool


class LinkResponseView(BaseModel):
    trace_id: str
    kb_id: str
    kb_version: str
    trace_persisted: bool = True
    workflow_engine: str = "langgraph"
    graph_nodes: list[str] = Field(default_factory=list)
    route_decision: str = "unknown"
    validation_errors: list[str] = Field(default_factory=list)
    node_events: list[dict[str, Any]] = Field(default_factory=list)
    results: list[LinkResultView]
    summary: SummaryView
    decision_log: list[DecisionLogItemView]


class BatchLinkRequestView(BaseModel):
    items: list[LinkRequestView]


class BatchLinkResponseView(BaseModel):
    batch_trace_id: str
    items: list[LinkResponseView]
