from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.entity import Entity

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


class CreateKBRequest(BaseModel):
    kb_id: str
    kb_version: str
    description: str = ""


class ImportEntitiesRequest(BaseModel):
    entities: list[Entity]


def _svc():
    from app.dependencies import get_kb_service
    return get_kb_service()


@router.post("", status_code=201)
def create_kb(body: CreateKBRequest) -> dict:
    svc = _svc()
    if svc.exists(body.kb_id):
        raise HTTPException(status_code=409, detail=f"KB {body.kb_id} already exists")
    kb = svc.create(body.kb_id, body.kb_version, body.description)
    return {"kb_id": kb.kb_id, "kb_version": kb.kb_version, "entity_count": 0, "status": "created"}


@router.post("/{kb_id}/entities")
def import_entities(kb_id: str, body: ImportEntitiesRequest) -> dict:
    svc = _svc()
    if not svc.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB {kb_id} not found")
    kb = svc.get(kb_id)
    count = svc.import_entities(kb_id, body.entities)
    return {"kb_id": kb_id, "kb_version": kb.kb_version if kb else "", "imported_count": count, "status": "success"}


@router.get("")
def list_kbs() -> dict:
    return {"knowledge_bases": [kb.model_dump() for kb in _svc().list_all()]}


@router.get("/{kb_id}")
def get_kb(kb_id: str) -> dict:
    kb = _svc().get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"KB {kb_id} not found")
    return kb.model_dump()
