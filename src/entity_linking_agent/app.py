"""FastAPI entrypoint for the Topic 10 agent."""

from fastapi import FastAPI

from entity_linking_agent.api.routes import router
from entity_linking_agent.config import load_config

config = load_config()

app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description=(
        "Entity linking and knowledge alignment service for Topic 10. "
        "The framework is designed for traceable, service-oriented data governance workloads."
    ),
)
app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": config.app_name,
        "version": config.app_version,
        "docs": "/docs",
        "terminal_agent": "python -m entity_linking_agent.cli",
    }
