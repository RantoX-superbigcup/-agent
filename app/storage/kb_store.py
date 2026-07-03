from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from app.models.entity import Entity, KnowledgeBase

if TYPE_CHECKING:
    from app.core.embedder import Embedder

logger = logging.getLogger("entity_link_agent")


class KBStore:
    def __init__(self, kb_dir: Path, index_dir: Optional[Path] = None, embedder: Optional["Embedder"] = None) -> None:
        self.kb_dir = kb_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = index_dir
        self.embedder = embedder

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

    def import_full(self, kb_id: str, kb_version: str, description: str, entities: list[Entity]) -> KnowledgeBase:
        """一步导入：同时写入元信息和实体，幂等覆盖。"""
        kb = KnowledgeBase(kb_id=kb_id, kb_version=kb_version, description=description, entity_count=len(entities))
        self._meta_path(kb_id).write_text(kb.model_dump_json(exclude={"entity_count"}), encoding="utf-8")
        self._entities_path(kb_id).write_text(
            json.dumps([e.model_dump() for e in entities], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._invalidate_index(kb_id)
        if self.embedder and self.index_dir and entities:
            self._rebuild_index(kb_id, entities)
        logger.info("知识库 '%s' 一步导入完成: %d 个实体", kb_id, len(entities))
        return kb

    def get_meta(self, kb_id: str) -> Optional[KnowledgeBase]:
        path = self._meta_path(kb_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        data["entity_count"] = len(self.load_entities(kb_id))
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
        """全量替换知识库中的实体，并使旧向量索引失效（下次链接时自动重建）。"""
        self._entities_path(kb_id).write_text(
            json.dumps([e.model_dump() for e in entities], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._invalidate_index(kb_id)
        if self.embedder and self.index_dir and entities:
            self._rebuild_index(kb_id, entities)
        logger.info("知识库 '%s' 导入完成: %d 个实体", kb_id, len(entities))
        return len(entities)

    def _invalidate_index(self, kb_id: str) -> None:
        if not self.index_dir:
            return
        for suffix in (".faiss", ".ids.json"):
            p = self.index_dir / f"{kb_id}{suffix}"
            if p.exists():
                p.unlink()
                logger.info("已删除旧索引文件: %s", p.name)

    def _rebuild_index(self, kb_id: str, entities: list[Entity]) -> None:
        from app.storage.vector_index import VectorIndex
        logger.info("正在构建向量索引 kb=%s entities=%d...", kb_id, len(entities))
        texts = [
            f"{e.description} {' '.join(e.keywords)}".strip()
            for e in entities
        ]
        vectors = self.embedder.encode(texts)
        VectorIndex(kb_id, self.index_dir).build([e.entity_id for e in entities], vectors)
        logger.info("向量索引构建完成: %s (%d 条向量)", kb_id, len(entities))
