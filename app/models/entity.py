from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator
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


class KBPackage(BaseModel):
    """一步导入：知识库元信息 + 实体列表。"""
    kb_id: str
    kb_version: str
    description: str = ""
    entities: list[Entity] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_nested_package(cls, data: Any) -> Any:
        """兼容误传成 {"entities": KBPackage} 的手动导入请求。"""
        if not isinstance(data, dict):
            return data
        nested = data.get("entities")
        if not isinstance(nested, dict) or not isinstance(nested.get("entities"), list):
            return data

        merged = dict(nested)
        for key in ("kb_id", "kb_version", "description"):
            outer_value = data.get(key)
            if outer_value not in (None, ""):
                merged[key] = outer_value
        merged["entities"] = nested["entities"]
        return merged

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "KBPackage":
        """同批次 entity_id 不可重复。"""
        seen: set[str] = set()
        for e in self.entities:
            if e.entity_id in seen:
                raise ValueError(f"DUPLICATE_ENTITY_ID:{e.entity_id}")
            seen.add(e.entity_id)
        return self
