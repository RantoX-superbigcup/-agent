from typing import Optional
from pydantic import BaseModel, Field
from app.models.enums import LinkStatus, EvidenceType, EntityType


class EntityRef(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType


class CandidateItem(BaseModel):
    entity_id: str
    canonical_name: str
    score: float


class EvidenceItem(BaseModel):
    evidence_type: EvidenceType
    detail: str


class CoreferenceInfo(BaseModel):
    resolved_from: str
    chain_id: str


class LinkResult(BaseModel):
    mention_id: str
    surface_form: str
    link_status: LinkStatus
    entity: Optional[EntityRef] = None
    confidence: float = 0.0
    candidates: list[CandidateItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    coreference: Optional[CoreferenceInfo] = None


class CoreferenceChain(BaseModel):
    chain_id: str
    mention_ids: list[str]
    entity_id: str


class LinkSummary(BaseModel):
    total_mentions: int
    linked_count: int
    nil_count: int
    ambiguous_count: int = 0
    review_count: int = 0


class LinkTrace(BaseModel):
    linker_version: str
    kb_id: str
    kb_version: str
    options_used: dict


class LinkResponse(BaseModel):
    schema_version: str = "v1"
    request_id: str
    status: str
    results: list[LinkResult] = Field(default_factory=list)
    coreference_chains: list[CoreferenceChain] = Field(default_factory=list)
    summary: Optional[LinkSummary] = None
    trace: Optional[LinkTrace] = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    schema_version: str = "v1"
    request_id: str
    status: str = "error"
    error: ErrorDetail
