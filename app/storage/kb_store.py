from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.models.entity import Entity, KnowledgeBase


class KBStore:
    """JSON file-backed knowledge base storage."""

    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, kb_id: str) -> Path:
        return self.kb_dir / f"{kb_id}.meta.json"

    def _entities_path(self, kb_id: str) -> Path:
        return self.kb_dir / f"{kb_id}.entities.json"

    def exists(self, kb_id: str) -> bool:
        return self._meta_path(kb_id).exists()

    def create(self, kb_id: str, kb_version: str, description: str) -> KnowledgeBase:
        kb = KnowledgeBase(kb_id=kb_id, kb_version=kb_version, description=description)
        self._meta_path(kb_id).write_text(kb.model_dump_json(), encoding="utf-8")
        self._entities_path(kb_id).write_text("[]", encoding="utf-8")
        return kb

    def get_meta(self, kb_id: str) -> Optional[KnowledgeBase]:
        path = self._meta_path(kb_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        entities = self.load_entities(kb_id)
        data["entity_count"] = len(entities)
        return KnowledgeBase(**data)

    def list_all(self) -> list[KnowledgeBase]:
        result = []
        for meta_file in sorted(self.kb_dir.glob("*.meta.json")):
            kb_id = meta_file.stem.replace(".meta", "")
            kb = self.get_meta(kb_id)
            if kb:
                result.append(kb)
        return result

    def load_entities(self, kb_id: str) -> list[Entity]:
        path = self._entities_path(kb_id)
        if not path.exists():
            return []
        return [Entity(**item) for item in json.loads(path.read_text(encoding="utf-8"))]

    def import_entities(self, kb_id: str, entities: list[Entity]) -> int:
        existing = self.load_entities(kb_id)
        existing_ids = {e.entity_id for e in existing}
        new_entities = [e for e in entities if e.entity_id not in existing_ids]
        all_entities = existing + new_entities
        self._entities_path(kb_id).write_text(
            json.dumps([e.model_dump() for e in all_entities], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return len(new_entities)
