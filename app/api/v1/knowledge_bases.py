from pathlib import Path
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.models.entity import KBPackage
from app.models.kb_import import KBFileImportRequest, KBFileImportResponse
from app.services.kb_file_importer import KBFileImportError

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


def _svc():
    from app.dependencies import get_kb_service
    return get_kb_service()


def _importer():
    from app.dependencies import get_kb_file_importer
    return get_kb_file_importer()


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


@router.post("/import-file", response_model=KBFileImportResponse)
def import_kb_from_file(body: KBFileImportRequest):
    try:
        return _importer().import_file(body)
    except KBFileImportError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": exc.code,
                "message": exc.message,
                "warnings": exc.warnings,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "IMPORT_FAILED",
                "message": str(exc),
                "warnings": [],
            },
        )


@router.post("/import-upload", response_model=KBFileImportResponse)
async def import_kb_from_upload(
    file: UploadFile = File(...),
    kb_id: str | None = Form(default=None),
    kb_version: str = Form(default="v1"),
    description: str | None = Form(default=None),
    source_type: str = Form(default="auto"),
    import_to_store: bool = Form(default=True),
    include_entities: bool = Form(default=False),
    preview_limit: int = Form(default=5),
    use_llm: bool = Form(default=False),
):
    original_name = file.filename or "upload"
    suffix = Path(original_name).suffix
    effective_source_type = source_type
    if source_type == "auto" and "kb_data" in Path(original_name).name.lower():
        effective_source_type = "ccks_kb_data"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)

        body = KBFileImportRequest(
            file_path=str(tmp_path),
            kb_id=kb_id,
            kb_version=kb_version,
            description=description or f"Uploaded from {original_name}",
            source_type=effective_source_type,  # type: ignore[arg-type]
            import_to_store=import_to_store,
            include_entities=include_entities,
            preview_limit=preview_limit,
            use_llm=use_llm,
        )
        return _importer().import_file(body)
    except KBFileImportError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": exc.code,
                "message": exc.message,
                "warnings": exc.warnings,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "IMPORT_UPLOAD_FAILED",
                "message": str(exc),
                "warnings": [],
            },
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.get("")
def list_kbs() -> dict:
    return {"knowledge_bases": [kb.model_dump() for kb in _svc().list_all()]}


@router.get("/{kb_id}")
def get_kb(kb_id: str) -> dict:
    kb = _svc().get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail=f"KB {kb_id} not found")
    return kb.model_dump()
