from typing import Optional
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    content: str
    language: str = "zh"


class MentionInput(BaseModel):
    mention_id: str
    surface_form: str
    start_offset: int
    end_offset: int


class KnowledgeBaseRef(BaseModel):
    kb_id: str
    kb_version: str


class LinkOptions(BaseModel):
    top_k: int = 5
    nil_threshold: float = 0.6
    enable_nil: bool = True
    enable_coreference: bool = True
    return_candidates: bool = True
    return_evidence: bool = True
    linker_version: str = "v1"


class LinkRequest(BaseModel):
    schema_version: str = "v1"
    request_id: str
    text: TextInput
    mentions: list[MentionInput]
    knowledge_base: KnowledgeBaseRef
    options: LinkOptions = Field(default_factory=LinkOptions)
