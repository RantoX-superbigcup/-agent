import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.router import api_router
from app.middleware.error_handler import global_error_handler
from app.middleware.logging import log_requests

# 配置应用日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

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
