from typing import Optional
from pydantic import BaseModel, Field
from app.models.enums import EntityType


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    former_names: list[str] = Field(default_factory=list)
    description: str = ""
    parent_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    kb_id: str
    kb_version: str
    description: str = ""
    entity_count: int = 0
