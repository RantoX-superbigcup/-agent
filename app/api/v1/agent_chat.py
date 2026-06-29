from fastapi import APIRouter, HTTPException

from app.models.agent import AgentChatRequest, AgentChatResponse
from app.services.agent_chat_service import AgentChatError

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def _svc():
    from app.dependencies import get_agent_chat_service
    return get_agent_chat_service()


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest):
    try:
        return _svc().chat(request)
    except AgentChatError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": exc.code,
                "message": exc.message,
            },
        )
