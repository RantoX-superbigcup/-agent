from __future__ import annotations
from app.models.entity import Entity, KnowledgeBase
from app.storage.kb_store import KBStore


class KBService:
    def __init__(self, store: KBStore) -> None:
        self.store = store

    def create(self, kb_id: str, kb_version: str, description: str) -> KnowledgeBase:
        return self.store.create(kb_id, kb_version, description)

    def exists(self, kb_id: str) -> bool:
        return self.store.exists(kb_id)

    def get(self, kb_id: str) -> KnowledgeBase | None:
        return self.store.get_meta(kb_id)

    def list_all(self) -> list[KnowledgeBase]:
        return self.store.list_all()

    def import_entities(self, kb_id: str, entities: list[Entity]) -> int:
        return self.store.import_entities(kb_id, entities)
