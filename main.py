"""Run the Topic 10 API service with a public network binding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entity_linking_agent.app import app  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("EL_HOST", "0.0.0.0"),
        port=int(os.getenv("EL_PORT", "8000")),
    )
