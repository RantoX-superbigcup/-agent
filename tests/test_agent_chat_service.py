from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.models.agent import AgentChatRequest
from app.services.agent_chat_service import AgentChatService


class FakeStore:
    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir

    def exists(self, kb_id: str) -> bool:
        return kb_id == "kb-test"


class FakeLinkService:
    def __init__(self, kb_dir: Path) -> None:
        self.store = FakeStore(kb_dir)
        self.called = False

    def link(self, request):  # pragma: no cover - tests assert this is never reached
        self.called = True
        raise AssertionError("workflow should not be called for non-link requests")


def _service(tmp_path, monkeypatch) -> tuple[AgentChatService, FakeLinkService]:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = SimpleNamespace(
        llm_api_key="sk-test",
        llm_base_url="https://example.test/v1",
        llm_model="fake-model",
    )
    link_service = FakeLinkService(tmp_path)
    return AgentChatService(config, link_service), link_service


def test_chat_intent_does_not_run_workflow_even_with_link_payload(tmp_path, monkeypatch):
    service, link_service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        service,
        "_ask_llm",
        lambda *args, **kwargs: {
            "intent": "chat",
            "reply": "这是普通对话，不需要实体链接。",
            "link_request": {
                "text": {"content": "李导演的《断背山》很好看"},
                "mentions": ["李导演", "断背山"],
            },
        },
    )

    resp = service.chat(AgentChatRequest(message="我们聊聊项目设计", kb_id="kb-test"))

    assert resp.intent == "chat"
    assert resp.link_request is None
    assert resp.link_response is None
    assert link_service.called is False


def test_link_intent_with_incomplete_payload_is_blocked(tmp_path, monkeypatch):
    service, link_service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        service,
        "_ask_llm",
        lambda *args, **kwargs: {
            "intent": "link",
            "reply": "信息还不完整。",
            "link_request": {"text": {"content": "这里只有文本，没有实体列表"}},
        },
    )

    resp = service.chat(AgentChatRequest(message="帮我看看这句话", kb_id="kb-test"))

    assert resp.intent == "chat"
    assert resp.link_request is None
    assert resp.link_response is None
    assert "blocked_non_link_payload" in resp.warnings
    assert link_service.called is False


def test_explicit_entity_list_bypasses_llm_and_preserves_english_alias(tmp_path, monkeypatch):
    service, link_service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        service,
        "_ask_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    message = (
        "三大通信运营商移动、电信、联通所有基站统一交由中国铁塔运营，"
        "英文代号 China Tower，大幅降低基础设施重复建设成本。\n"
        "实体：移动\n电信\n联通\n中国铁塔、China Tower"
    )

    resp = service.chat(AgentChatRequest(message=message, kb_id="kb-test", run_workflow=False))

    assert resp.intent == "link"
    assert resp.link_request is not None
    assert resp.link_response is None
    assert "local_explicit_entities_parser" in resp.warnings
    assert link_service.called is False
    assert [m.surface_form for m in resp.link_request.mentions] == [
        "移动",
        "电信",
        "联通",
        "中国铁塔",
        "China Tower",
    ]
    assert resp.link_request.text.content.startswith("三大通信运营商")


def test_explicit_entity_phrase_accepts_loose_marker_and_strips_text_prefix(tmp_path, monkeypatch):
    service, link_service = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        service,
        "_ask_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    message = "文本：腾讯依靠微信和云服务实现稳定盈利；需要识别的实体包括：腾讯、微信"

    resp = service.chat(AgentChatRequest(message=message, kb_id="kb-test", run_workflow=False))

    assert resp.intent == "link"
    assert resp.link_request is not None
    assert resp.link_request.text.content == "腾讯依靠微信和云服务实现稳定盈利"
    assert [m.surface_form for m in resp.link_request.mentions] == ["腾讯", "微信"]
    assert link_service.called is False


def test_freeform_without_entity_marker_still_uses_llm(tmp_path, monkeypatch):
    service, link_service = _service(tmp_path, monkeypatch)
    called = {"llm": False}

    def fake_ask_llm(*args, **kwargs):
        called["llm"] = True
        return {"intent": "chat", "reply": "请补充需要链接的实体。", "link_request": None}

    monkeypatch.setattr(service, "_ask_llm", fake_ask_llm)

    resp = service.chat(AgentChatRequest(message="帮我分析腾讯和阿里的产业布局", kb_id="kb-test"))

    assert called["llm"] is True
    assert resp.intent == "chat"
    assert resp.link_request is None
    assert link_service.called is False
