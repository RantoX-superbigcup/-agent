from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.models.enums import MentionType


def _normalize_mention_type_value(value: Any) -> MentionType:
    if isinstance(value, MentionType):
        return value
    if value in (None, "", "UNKNOWN", "未识别"):
        return MentionType.UNKNOWN

    raw = str(value).strip()
    upper = raw.upper()
    if upper in MentionType._value2member_map_:
        return MentionType(upper)

    mapping = {
        "人": MentionType.PERSON,
        "人物": MentionType.PERSON,
        "导演": MentionType.PERSON,
        "演员": MentionType.PERSON,
        "组织": MentionType.ORG,
        "机构": MentionType.ORG,
        "公司": MentionType.ORG,
        "企业": MentionType.ORG,
        "地点": MentionType.LOC,
        "地名": MentionType.LOC,
        "城市": MentionType.LOC,
        "作品": MentionType.OTHER,
        "电影": MentionType.OTHER,
        "书籍": MentionType.OTHER,
    }
    return mapping.get(raw, MentionType.UNKNOWN)


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    language: str = "zh"


class MentionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str
    surface_form: str
    start_offset: int
    end_offset: int


class KnowledgeBaseRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str
    kb_version: str


class LinkOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = 5
    nil_threshold: float = 0.6
    ambiguity_margin: float = 0.08
    auto_calibrate: bool = True
    enable_llm_rerank: bool = True
    enable_nil: bool = True
    enable_coreference: bool = True
    return_candidates: bool = True
    return_evidence: bool = True
    linker_version: str = "v1"


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "v1"
    request_id: str
    text: TextInput
    mentions: list[MentionInput]
    knowledge_base: KnowledgeBaseRef
    options: LinkOptions = Field(default_factory=LinkOptions)


class MentionHint(BaseModel):
    mention_type: MentionType = Field(
        default=MentionType.UNKNOWN,
        validation_alias=AliasChoices("mention_type", "entity_type", "type"),
    )

    @field_validator("mention_type", mode="before")
    @classmethod
    def _normalize_mention_type(cls, value: Any) -> MentionType:
        return _normalize_mention_type_value(value)


class WorkflowMentionInput(MentionInput):
    mention_type: MentionType = Field(
        default=MentionType.UNKNOWN,
        validation_alias=AliasChoices("mention_type", "entity_type", "type"),
    )

    @field_validator("mention_type", mode="before")
    @classmethod
    def _normalize_mention_type(cls, value: Any) -> MentionType:
        return _normalize_mention_type_value(value)

    @classmethod
    def from_public(
        cls,
        mention: MentionInput,
        hint: MentionHint | None = None,
    ) -> WorkflowMentionInput:
        hint = hint or MentionHint()
        return cls(
            mention_id=mention.mention_id,
            surface_form=mention.surface_form,
            start_offset=mention.start_offset,
            end_offset=mention.end_offset,
            mention_type=hint.mention_type,
        )


class WorkflowLinkRequest(LinkRequest):
    mentions: list[WorkflowMentionInput]

    @classmethod
    def from_public(
        cls,
        request: LinkRequest,
        mention_hints: dict[str, MentionHint] | None = None,
    ) -> WorkflowLinkRequest:
        mention_hints = mention_hints or {}
        return cls(
            schema_version=request.schema_version,
            request_id=request.request_id,
            text=request.text.model_copy(deep=True),
            mentions=[
                WorkflowMentionInput.from_public(
                    mention,
                    mention_hints.get(mention.mention_id),
                )
                for mention in request.mentions
            ],
            knowledge_base=request.knowledge_base.model_copy(deep=True),
            options=request.options.model_copy(deep=True),
        )

    def to_public(self) -> LinkRequest:
        return LinkRequest(
            schema_version=self.schema_version,
            request_id=self.request_id,
            text=self.text.model_copy(deep=True),
            mentions=[
                MentionInput(
                    mention_id=mention.mention_id,
                    surface_form=mention.surface_form,
                    start_offset=mention.start_offset,
                    end_offset=mention.end_offset,
                )
                for mention in self.mentions
            ],
            knowledge_base=self.knowledge_base.model_copy(deep=True),
            options=self.options.model_copy(deep=True),
        )
