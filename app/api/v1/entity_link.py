from fastapi import APIRouter, HTTPException
from app.models.request import LinkRequest
from app.models.response import LinkResponse

router = APIRouter(prefix="/api/v1", tags=["entity-link"])


def _svc():
    from app.dependencies import get_link_service
    return get_link_service()


@router.post("/entity-link", response_model=LinkResponse)
def entity_link(request: LinkRequest):
    try:
        return _svc().link(request)
    except ValueError as exc:
        msg = str(exc)
        parts = msg.split(":", 1)
        code = parts[0] if len(parts) == 2 else "INTERNAL_ERROR"
        detail = parts[1] if len(parts) == 2 else msg
        raise HTTPException(
            status_code=400,
            detail={"schema_version": "v1", "request_id": request.request_id,
                    "status": "error", "error": {"code": code, "message": detail}},
        )
