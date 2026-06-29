from __future__ import annotations

import json
import os
import re
import uuid
import urllib.error
import urllib.request
from typing import Any

from pydantic import ValidationError

from app.config import AppConfig
from app.models.agent import AgentChatRequest, AgentChatResponse
from app.models.request import LinkRequest
from app.services.llm_provider import (
    LLMProviderConfig,
    SUPPORTED_API_KEY_ENV_NAMES,
    append_chat_completions_path,
    resolve_llm_provider,
)
from app.services.link_service import LinkService


class AgentChatError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AgentChatService:
    API_KEY_ENV_NAMES = SUPPORTED_API_KEY_ENV_NAMES
    MODEL_ENV_NAMES = ("DEEPSEEK_MODEL", "LLM_MODEL")
    BASE_URL_ENV_NAMES = (
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_API_URL",
        "LLM_BASE_URL",
    )

    def __init__(self, config: AppConfig, link_service: LinkService) -> None:
        self.config = config
        self.link_service = link_service

    def chat(self, req: AgentChatRequest) -> AgentChatResponse:
        selected_kb_id, selected_kb_version = self._select_kb(req.kb_id, req.kb_version)
        local_response = self._try_local_response(req, selected_kb_id, selected_kb_version)
        if local_response:
            return local_response

        model_override = req.model if req.model and req.model != "auto" else None
        provider = resolve_llm_provider(
            config_api_key=self.config.llm_api_key,
            config_base_url=self.config.llm_base_url,
            config_model=self.config.llm_model,
            preferred_model=model_override,
        )
        if not provider:
            raise AgentChatError(
                "LLM_NOT_CONFIGURED",
                "No LLM API key found. Supported env names: " + " / ".join(self.API_KEY_ENV_NAMES),
            )
        model = provider.model
        warnings: list[str] = [f"llm_provider={provider.provider}", f"api_key_env={provider.api_key_env}"]

        llm_payload = self._ask_llm(req, selected_kb_id, selected_kb_version, provider)
        intent = str(llm_payload.get("intent") or "chat").strip() or "chat"
        reply = str(llm_payload.get("reply") or "").strip()

        if intent == "switch_model":
            model = str(llm_payload.get("model") or model).strip() or model
            return AgentChatResponse(
                intent=intent,
                reply=reply or f"已切换到模型 {model}。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=model,
                warnings=warnings,
            )

        if intent == "switch_kb":
            kb_id = str(llm_payload.get("kb_id") or selected_kb_id or "").strip()
            kb_version = str(llm_payload.get("kb_version") or selected_kb_version or "v1").strip()
            if not kb_id or not self.link_service.store.exists(kb_id):
                return AgentChatResponse(
                    intent=intent,
                    reply=f"没有找到知识库 {kb_id or '（空）'}，请先上传或在知识库列表中选择。",
                    selected_kb_id=selected_kb_id,
                    selected_kb_version=selected_kb_version,
                    selected_model=model,
                    warnings=[f"kb_not_found={kb_id or '<empty>'}"],
                )
            kb_meta = self._get_kb_meta_lightweight(kb_id)
            kb_version = str(kb_meta.get("kb_version") or kb_version)
            return AgentChatResponse(
                intent=intent,
                reply=reply or f"已切换到知识库 {kb_id}/{kb_version}。",
                selected_kb_id=kb_id,
                selected_kb_version=kb_version,
                selected_model=model,
                warnings=warnings,
            )

        if intent == "upload_kb":
            return AgentChatResponse(
                intent=intent,
                reply=reply or "可以在对话框下方选择本地文件上传为知识库；支持 CCKS kb_data、JSON/JSONL、PDF 和文本。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=model,
                warnings=warnings,
            )

        if intent != "link":
            return AgentChatResponse(
                intent=intent,
                reply=reply or "这次输入不像实体链接任务，我不会生成 LinkRequest 或调用 workflow。你可以直接聊天，或输入“文本：...；实体：A，B”开始链接。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=model,
                warnings=warnings,
            )

        link_payload = llm_payload.get("link_request")
        if not link_payload:
            return AgentChatResponse(
                intent=intent,
                reply=reply or "我还需要文本内容和需要链接的实体 mention。可以这样输入：文本：...；实体：A，B。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=model,
                warnings=warnings,
            )

        if not self._looks_like_link_payload(link_payload):
            return AgentChatResponse(
                intent="chat",
                reply=reply or "我判断这次输入还不是完整的实体链接请求，因此没有调用 workflow。请提供原文和要链接的实体，例如：文本：...；实体：A，B。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=model,
                warnings=warnings + ["blocked_non_link_payload"],
            )

        link_request = self._build_link_request(
            link_payload,
            selected_kb_id,
            selected_kb_version,
            req,
            warnings,
        )

        link_response = None
        if req.run_workflow:
            try:
                link_response = self.link_service.link(link_request)
            except ValueError as exc:
                return AgentChatResponse(
                    status="error",
                    intent="link",
                    reply=f"我已经生成了接口 JSON，但 workflow 执行失败：{exc}",
                    selected_kb_id=link_request.knowledge_base.kb_id,
                    selected_kb_version=link_request.knowledge_base.kb_version,
                    selected_model=model,
                    link_request=link_request,
                    warnings=warnings,
                )

        if link_response and link_response.summary:
            reply = (
                reply
                or f"已将你的输入转成 LinkRequest 并执行实体链接："
                f"共 {link_response.summary.total_mentions} 个 mention，"
                f"linked={link_response.summary.linked_count}，nil={link_response.summary.nil_count}。"
            )
        else:
            reply = reply or "已将你的输入转成 LinkRequest JSON。"

        return AgentChatResponse(
            intent="link",
            reply=reply,
            selected_kb_id=link_request.knowledge_base.kb_id,
            selected_kb_version=link_request.knowledge_base.kb_version,
            selected_model=model,
            link_request=link_request,
            link_response=link_response,
            warnings=warnings,
        )

    def _ask_llm(
        self,
        req: AgentChatRequest,
        selected_kb_id: str | None,
        selected_kb_version: str | None,
        provider: LLMProviderConfig,
    ) -> dict[str, Any]:
        url = append_chat_completions_path(provider.base_url)

        body = json.dumps(
            {
                "model": provider.model,
                "messages": self._build_messages(req, selected_kb_id, selected_kb_version),
                "temperature": 0.05,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AgentChatError("LLM_HTTP_ERROR", f"LLM API returned {exc.code}: {detail}") from exc
        except Exception as exc:
            raise AgentChatError("LLM_REQUEST_FAILED", f"LLM request failed: {exc}") from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(self._strip_json_fence(content))
        except Exception as exc:
            raise AgentChatError("LLM_RESPONSE_PARSE_FAILED", f"Cannot parse LLM response as JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AgentChatError("LLM_RESPONSE_SHAPE_ERROR", "LLM response must be a JSON object.")
        return parsed

    def _try_local_response(
        self,
        req: AgentChatRequest,
        selected_kb_id: str | None,
        selected_kb_version: str | None,
    ) -> AgentChatResponse | None:
        text = req.message.strip().lower()
        help_terms = ("怎么输入", "如何输入", "怎么用", "如何使用", "帮助", "help", "示例", "例子")
        upload_terms = ("上传知识库", "导入知识库", "上传文件", "导入文件")

        if any(term in text for term in help_terms):
            return AgentChatResponse(
                intent="help",
                reply=(
                    "你可以这样输入：\n"
                    "1. 文本：李导演的《断背山》真是令人动人；实体：李导演，断背山。\n"
                    "2. 南京南站可以坐高铁到北京南站；实体：南京南站，高铁，北京南站。\n"
                    "3. 换成知识库 cck2019。\n"
                    "只有当你同时提供原文和实体时，我才会生成 LinkRequest JSON 并交给 LangGraph workflow 执行。"
                ),
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=req.model,
            )

        if any(term in text for term in upload_terms):
            return AgentChatResponse(
                intent="upload_kb",
                reply="请在对话框下方点击“选择知识库文件”，可上传 CCKS kb_data、JSON/JSONL、PDF、TXT 或 Markdown。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=req.model,
            )

        if text in {"你好", "hi", "hello"}:
            return AgentChatResponse(
                intent="chat",
                reply="你好，我是实体链接 Agent。你可以输入“文本：...；实体：A，B”开始实体链接；普通聊天我不会提交 workflow。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=req.model,
            )

        return None

    def _build_messages(self, req: AgentChatRequest, selected_kb_id: str | None, selected_kb_version: str | None) -> list[dict[str, str]]:
        kbs = self._list_kbs_lightweight()[:20]
        system = (
            "你是课题10实体链接智能体的对话理解层。你的任务是把用户自然语言转换成严格符合后端接口的 LinkRequest JSON，"
            "必要时也给用户中文回复。\n"
            "只返回 JSON object，不要返回 Markdown。\n"
            "输出 schema：{\n"
            '  "intent": "link|switch_kb|switch_model|upload_kb|chat|help",\n'
            '  "reply": "给用户看的中文回复",\n'
            '  "kb_id": "可选，切换知识库时填写",\n'
            '  "kb_version": "可选",\n'
            '  "model": "可选，切换大模型时填写",\n'
            '  "link_request": null 或 LinkRequest 对象\n'
            "}\n"
            "必须先判断 intent：只有用户明确要求做实体链接，并且同时提供原文文本和要链接的实体/mention 时，intent 才能是 link。"
            "问候、闲聊、问怎么使用、解释概念、讨论项目、切换知识库、切换模型或上传知识库，都不要生成 link_request，必须令 link_request=null。\n"
            "LinkRequest 必须满足：schema_version='v1'；request_id 可先留空；text.content 必须是用户给出的原文；"
            "text.language 默认 zh；mentions 必须是数组，每个 mention 包含 mention_id、surface_form、start_offset、end_offset；"
            "knowledge_base 使用当前选中的 kb_id/kb_version，除非用户明确要求切换；options 保持默认即可。\n"
            "如果用户说“文本是/句子是/内容是...，实体是/mention 是...”，intent=link，并提取实体为 mentions。"
            "如果用户只说换知识库，intent=switch_kb；只说换模型，intent=switch_model；说上传知识库，intent=upload_kb。"
            "不要虚构知识库 ID；如果信息不足以链接，intent=chat 并用 reply 追问缺少的文本或实体。"
        )
        context = {
            "current_kb": {"kb_id": selected_kb_id, "kb_version": selected_kb_version},
            "available_kbs": kbs,
            "current_options": req.options.model_dump(),
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "system", "content": "当前上下文：" + json.dumps(context, ensure_ascii=False)},
        ]
        for item in req.history[-8:]:
            if item.role in {"user", "assistant"} and item.content:
                messages.append({"role": item.role, "content": item.content[:1200]})
        messages.append({"role": "user", "content": req.message})
        return messages

    def _looks_like_link_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        text_obj = payload.get("text") if isinstance(payload.get("text"), dict) else {}
        content = str(text_obj.get("content") or payload.get("text_content") or "").strip()
        mentions = payload.get("mentions") or payload.get("entities") or []
        return bool(content and isinstance(mentions, list) and mentions)

    def _build_link_request(
        self,
        payload: Any,
        selected_kb_id: str | None,
        selected_kb_version: str | None,
        req: AgentChatRequest,
        warnings: list[str],
    ) -> LinkRequest:
        if not isinstance(payload, dict):
            raise AgentChatError("INVALID_LINK_REQUEST", "link_request 必须是 JSON object。")

        data = dict(payload)
        text_obj = data.get("text") if isinstance(data.get("text"), dict) else {}
        content = str(text_obj.get("content") or data.get("text_content") or "").strip()
        if not content:
            raise AgentChatError("MISSING_TEXT", "LLM 没有提取到 text.content。")

        kb_obj = data.get("knowledge_base") if isinstance(data.get("knowledge_base"), dict) else {}
        kb_id = str(kb_obj.get("kb_id") or selected_kb_id or "").strip()
        kb_version = str(kb_obj.get("kb_version") or selected_kb_version or req.kb_version or "v1").strip()
        if not kb_id:
            raise AgentChatError("MISSING_KB", "当前没有选中知识库，请先上传或选择知识库。")

        mentions_payload = data.get("mentions") or data.get("entities") or []
        mentions = self._normalize_mentions(mentions_payload, content, warnings)
        if not mentions:
            raise AgentChatError("EMPTY_MENTIONS", "LLM 没有提取到 mentions。")

        normalized = {
            "schema_version": "v1",
            "request_id": data.get("request_id") or f"agent-{uuid.uuid4().hex[:12]}",
            "text": {"content": content, "language": text_obj.get("language") or "zh"},
            "mentions": mentions,
            "knowledge_base": {"kb_id": kb_id, "kb_version": kb_version},
            "options": req.options.model_dump(),
        }
        try:
            return LinkRequest(**normalized)
        except ValidationError as exc:
            raise AgentChatError("LINK_REQUEST_VALIDATION_FAILED", str(exc)) from exc

    def _normalize_mentions(self, mentions_payload: Any, content: str, warnings: list[str]) -> list[dict[str, Any]]:
        if isinstance(mentions_payload, str):
            mentions_payload = self._split_mentions(mentions_payload)
        if not isinstance(mentions_payload, list):
            return []

        mentions: list[dict[str, Any]] = []
        cursor = 0
        for idx, item in enumerate(mentions_payload, start=1):
            if isinstance(item, str):
                surface = item.strip()
            elif isinstance(item, dict):
                surface = str(item.get("surface_form") or item.get("mention") or item.get("name") or "").strip()
            else:
                surface = ""
            if not surface:
                continue

            start = content.find(surface, cursor)
            if start < 0:
                start = content.find(surface)
            if start < 0:
                warnings.append(f"mention_not_found_in_text={surface}")
                start = 0
            end = start + len(surface)
            cursor = max(cursor, end)
            mentions.append(
                {
                    "mention_id": f"m{len(mentions) + 1}",
                    "surface_form": surface,
                    "start_offset": start,
                    "end_offset": end,
                }
            )
        return mentions

    def _select_kb(self, kb_id: str | None, kb_version: str) -> tuple[str | None, str | None]:
        if kb_id:
            kb_meta = self._get_kb_meta_lightweight(kb_id)
            return kb_id, str(kb_meta.get("kb_version") or kb_version)
        kbs = self._list_kbs_lightweight()
        if not kbs:
            return None, None
        return str(kbs[0]["kb_id"]), str(kbs[0].get("kb_version") or "v1")

    def _list_kbs_lightweight(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for meta_file in sorted(self.link_service.store.kb_dir.glob("*.meta.json")):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            kb_id = data.get("kb_id") or meta_file.stem.replace(".meta", "")
            result.append(
                {
                    "kb_id": kb_id,
                    "kb_version": data.get("kb_version") or "v1",
                    "description": data.get("description") or "",
                    "entity_count": data.get("entity_count"),
                }
            )
        return result

    def _get_kb_meta_lightweight(self, kb_id: str) -> dict[str, Any]:
        path = self.link_service.store.kb_dir / f"{kb_id}.meta.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _split_mentions(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,;|/\s\u3001\uff0c\uff1b]+", text) if part.strip()]

    def _strip_json_fence(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()
        return content

    def _first_env(self, names: tuple[str, ...]) -> str:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return ""
