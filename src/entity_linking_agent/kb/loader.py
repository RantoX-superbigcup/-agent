"""Knowledge base loading."""

from __future__ import annotations

import json
from typing import Optional

from entity_linking_agent.config import AppConfig
from entity_linking_agent.core.contracts import KnowledgeBaseEntity, KnowledgeBaseSnapshot
from entity_linking_agent.kb.ccks2019 import load_ccks2019_entities


class KnowledgeBaseLoader:
    """Loads built-in or inline knowledge bases."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._builtin_paths = {
            config.default_kb_id: config.default_kb_path,
            config.ccks2019_kb_id: config.ccks2019_kb_path,
        }
        self._snapshot_cache: dict[str, KnowledgeBaseSnapshot] = {}

    def load(
        self,
        knowledge_base_id: Optional[str],
        inline_entities: Optional[list[KnowledgeBaseEntity]] = None,
    ) -> KnowledgeBaseSnapshot:
        if inline_entities is not None:
            kb_id = knowledge_base_id or "inline"
            return KnowledgeBaseSnapshot(kb_id=kb_id, version="inline", entities=inline_entities)

        kb_id = knowledge_base_id or self.config.default_kb_id
        kb_path = self._builtin_paths.get(kb_id)
        if kb_path is None:
            raise ValueError(f"Unknown knowledge_base_id: {kb_id}")
        if kb_id in self._snapshot_cache:
            return self._snapshot_cache[kb_id]

        if kb_id == self.config.ccks2019_kb_id:
            entities = load_ccks2019_entities(kb_path)
            snapshot = KnowledgeBaseSnapshot(
                kb_id=kb_id,
                version="ccks2019",
                entities=entities,
            )
            self._snapshot_cache[kb_id] = snapshot
            return snapshot

        payload = json.loads(kb_path.read_text(encoding="utf-8"))
        entities = [
            KnowledgeBaseEntity(
                entity_id=item["entity_id"],
                canonical_name=item["canonical_name"],
                aliases=item.get("aliases", []),
                entity_type=item["entity_type"],
                keywords=item.get("keywords", []),
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("entities", [])
        ]

        snapshot = KnowledgeBaseSnapshot(
            kb_id=payload.get("kb_id", kb_id),
            version=payload.get("version", kb_id),
            entities=entities,
        )
        self._snapshot_cache[kb_id] = snapshot
        return snapshot

    def list_builtin_kbs(self) -> dict[str, str]:
        return {kb_id: str(path) for kb_id, path in self._builtin_paths.items()}
