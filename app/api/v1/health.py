from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "entity-link-agent", "version": "v1"}
