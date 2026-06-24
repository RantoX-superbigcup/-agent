from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.router import api_router
from app.middleware.error_handler import global_error_handler
from app.middleware.logging import log_requests

_ROOT = Path(__file__).parent.parent
_INDEX = (_ROOT / "static" / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="Entity Link Agent", version="v1")
app.add_exception_handler(Exception, global_error_handler)
app.add_middleware(BaseHTTPMiddleware, dispatch=log_requests)
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def index():
    return HTMLResponse(_INDEX)


app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")
