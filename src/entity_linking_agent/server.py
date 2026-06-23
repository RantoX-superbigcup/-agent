"""Public server launcher for the Topic 10 agent."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("EL_HOST", "0.0.0.0")
    port = int(os.getenv("EL_PORT", "8000"))
    reload_enabled = os.getenv("EL_RELOAD", "false").lower() in {"1", "true", "yes"}

    uvicorn.run(
        "entity_linking_agent.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
