from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.knowledge_bases import router as kb_router
from app.api.v1.entity_link import router as link_router
from app.api.v1.agent_chat import router as agent_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(kb_router)
api_router.include_router(link_router)
api_router.include_router(agent_router)
