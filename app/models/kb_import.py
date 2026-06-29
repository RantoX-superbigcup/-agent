from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.entity import Entity, KBPackage


SourceType = Literal["auto", "kb_package", "entities", "ccks_kb_data", "pdf", "text"]


class KBFileImportRequest(BaseModel):
    """Import a local file and convert it into the project KBPackage format."""

    file_path: str = Field(min_length=1, description="Local file path on the server machine.")
    kb_id: Optional[str] = Field(default=None, description="Override KB id. Defaults to file stem.")
    kb_version: str = "v1"
    description: Optional[str] = None
    source_type: SourceType = "auto"
    import_to_store: bool = True
    include_entities: bool = False
    preview_limit: int = Field(default=5, ge=0, le=50)
    use_llm: bool = False
    max_text_chars: int = Field(default=12000, ge=1000, le=100000)


class KBFileImportResponse(BaseModel):
    status: str
    imported: bool
    source_type: str
    file_path: str
    kb_id: str
    kb_version: str
    description: str = ""
    entity_count: int
    warnings: list[str] = Field(default_factory=list)
    entities_preview: list[Entity] = Field(default_factory=list)
    package: Optional[KBPackage] = None

