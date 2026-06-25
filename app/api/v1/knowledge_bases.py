from fastapi import APIRouter, HTTPException
from app.models.entity import KBPackage

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


def _svc():
    from app.dependencies import get_kb_service
    return get_kb_service()


@router.post("", status_code=201)
def import_kb(body: KBPackage) -> dict:
    """导入知识库（含实体）。kb_id 已存在时覆盖更新。"""
    svc = _svc()
    existed = svc.exists(body.kb_id)
    try:
        kb = svc.import_full(body.kb_id, body.kb_version, body.description, body.entities)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "kb_id": kb.kb_id,
        "kb_version": kb.kb_version,
        "entity_count": len(body.entities),
        "status": "overwritten" if existed else "created",
    }


@router.get("")
def list_kbs() -> dict:
    return {"knowledge_bases": [kb.model_dump() for kb in _svc().list_all()]}


@router.get("/{kb_id}")
def get_kb(kb_id: str) -> dict:
    kb = _svc().get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"KB {kb_id} not found")
    return kb.model_dump()
