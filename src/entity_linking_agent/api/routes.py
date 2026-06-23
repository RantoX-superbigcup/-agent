"""HTTP routes for the Topic 10 agent."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from entity_linking_agent.config import load_config
from entity_linking_agent.core.service import Topic10EntityLinkingService
from entity_linking_agent.models.schema import (
    BatchLinkRequestView,
    BatchLinkResponseView,
    LinkRequestView,
    LinkResponseView,
)
from entity_linking_agent.utils.tracing import build_trace_id

router = APIRouter(tags=["entity-linking"])
config = load_config()
service = Topic10EntityLinkingService(config)


@router.get("/health", tags=["meta"])
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": config.app_name,
        "version": config.app_version,
        "builtin_knowledge_bases": service.list_builtin_kbs(),
        "traces_dir": str(config.traces_dir),
    }


@router.post("/v1/link", response_model=LinkResponseView)
def link_entities(request: LinkRequestView) -> LinkResponseView:
    payload = service.link(**request.to_service_kwargs())
    return LinkResponseView.model_validate(payload)


@router.post("/v1/link/batch", response_model=BatchLinkResponseView)
def link_entities_batch(request: BatchLinkRequestView) -> BatchLinkResponseView:
    responses = [
        LinkResponseView.model_validate(service.link(**item.to_service_kwargs()))
        for item in request.items
    ]
    return BatchLinkResponseView(
        batch_trace_id=build_trace_id("t10-batch"),
        items=responses,
    )


@router.get("/v1/traces/{trace_id}", tags=["trace"])
def fetch_trace(trace_id: str) -> dict:
    try:
        return service.get_trace(trace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}") from exc
