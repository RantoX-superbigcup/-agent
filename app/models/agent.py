from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.request import LinkOptions, LinkRequest
from app.models.response import LinkResponse


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1)
    kb_id: Optional[str] = None
    kb_version: str = "v1"
    model: Optional[str] = None
    history: list[AgentMessage] = Field(default_factory=list)
    options: LinkOptions = Field(default_factory=LinkOptions)
    run_workflow: bool = True


class AgentChatResponse(BaseModel):
    status: str = "success"
    intent: str = "chat"
    reply: str
    selected_kb_id: Optional[str] = None
    selected_kb_version: Optional[str] = None
    selected_model: Optional[str] = None
    link_request: Optional[LinkRequest] = None
    link_response: Optional[LinkResponse] = None
    warnings: list[str] = Field(default_factory=list)
