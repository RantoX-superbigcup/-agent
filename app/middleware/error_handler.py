from fastapi import Request
from fastapi.responses import JSONResponse


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "schema_version": "v1",
            "request_id": "",
            "status": "error",
            "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
        },
    )
