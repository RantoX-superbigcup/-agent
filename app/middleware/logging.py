import time
import logging
from fastapi import Request

logger = logging.getLogger("entity_link_agent")


async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, elapsed)
    return response
