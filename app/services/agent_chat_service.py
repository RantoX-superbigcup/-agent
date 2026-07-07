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
from app.models.agent import AgentChatRequest, AgentChatResponse, AgentMessage
from app.models.request import LinkRequest, MentionHint, WorkflowLinkRequest
from app.models.response import LinkResponse, LinkResult
from app.services.link_service import LinkService
from app.services.llm_provider import (
    LLMProviderConfig,
    SUPPORTED_API_KEY_ENV_NAMES,
    append_chat_completions_path,
    resolve_llm_provider,
)


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

        warnings: list[str] = [
            f"llm_provider={provider.provider}",
            f"api_key_env={provider.api_key_env}",
        ]
        llm_payload = self._ask_llm(req, selected_kb_id, selected_kb_version, provider)
        intent = str(llm_payload.get("intent") or "chat").strip() or "chat"
        reply = str(llm_payload.get("reply") or "").strip()
        selected_model = provider.model

        if intent == "switch_model":
            selected_model = str(llm_payload.get("model") or selected_model).strip() or selected_model
            return AgentChatResponse(
                intent=intent,
                reply=reply or f"已切换到模型 {selected_model}。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=selected_model,
                warnings=warnings,
            )

        if intent == "switch_kb":
            kb_id = str(llm_payload.get("kb_id") or selected_kb_id or "").strip()
            kb_version = str(llm_payload.get("kb_version") or selected_kb_version or "v1").strip()
            if not kb_id or not self.link_service.store.exists(kb_id):
                return AgentChatResponse(
                    intent=intent,
                    reply=reply or f"没有找到知识库 {kb_id or '（空）'}，请先上传或从列表中选择。",
                    selected_kb_id=selected_kb_id,
                    selected_kb_version=selected_kb_version,
                    selected_model=selected_model,
                    warnings=warnings + [f"kb_not_found={kb_id or '<empty>'}"],
                )
            kb_meta = self._get_kb_meta_lightweight(kb_id)
            kb_version = str(kb_meta.get("kb_version") or kb_version)
            return AgentChatResponse(
                intent=intent,
                reply=reply or f"已切换到知识库 {kb_id}/{kb_version}。",
                selected_kb_id=kb_id,
                selected_kb_version=kb_version,
                selected_model=selected_model,
                warnings=warnings,
            )

        if intent == "upload_kb":
            return AgentChatResponse(
                intent=intent,
                reply=reply or "可以在页面中选择本地文件上传知识库，支持 CCKS kb_data、JSON、JSONL、PDF、TXT 和 Markdown。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=selected_model,
                warnings=warnings,
            )

        if intent != "link":
            return AgentChatResponse(
                intent=intent,
                reply=reply or "这次输入不是完整的实体链接任务。我不会生成 LinkRequest，也不会触发 workflow。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=selected_model,
                warnings=warnings,
            )

        link_payload = llm_payload.get("link_request")
        if not link_payload:
            return AgentChatResponse(
                intent="link",
                reply=reply or "我还需要原文文本和要链接的实体。可以这样输入：文本：...；实体：A，B。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=selected_model,
                warnings=warnings,
            )

        if not self._looks_like_link_payload(link_payload):
            return AgentChatResponse(
                intent="chat",
                reply=reply or "当前输入还不是完整的实体链接请求，因此没有调用 workflow。请提供原文和实体列表。",
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=selected_model,
                warnings=warnings + ["blocked_non_link_payload"],
            )

        public_link_request, mention_hints = self._build_link_request(
            link_payload,
            selected_kb_id,
            selected_kb_version,
            req,
            warnings,
        )
        workflow_request, mention_type_diagnostics = self._prepare_link_request(
            public_link_request,
            mention_hints,
            warnings,
        )

        link_response = None
        if req.run_workflow:
            try:
                link_response = self.link_service.link_with_diagnostics(
                    workflow_request,
                    mention_type_diagnostics,
                )
            except ValueError as exc:
                return AgentChatResponse(
                    status="error",
                    intent="link",
                    reply=f"我已经生成了接口 JSON，但 workflow 执行失败：{exc}",
                    selected_kb_id=public_link_request.knowledge_base.kb_id,
                    selected_kb_version=public_link_request.knowledge_base.kb_version,
                    selected_model=selected_model,
                    link_request=public_link_request,
                    warnings=warnings,
                )

        if link_response and link_response.summary:
            wants_detail = self._wants_workflow_detail(req.message)
            generated_reply = (
                self._build_workflow_process_reply(link_response, local_parser=False)
                if wants_detail
                else self._build_link_completion_reply(link_response)
            )
            reply = generated_reply if wants_detail else (reply or generated_reply)
        else:
            reply = reply or "已将你的输入转换成标准 LinkRequest JSON。"

        return AgentChatResponse(
            intent="link",
            reply=reply,
            selected_kb_id=public_link_request.knowledge_base.kb_id,
            selected_kb_version=public_link_request.knowledge_base.kb_version,
            selected_model=selected_model,
            link_request=public_link_request,
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
        explicit_link_payload = self._extract_explicit_link_payload(req.message)
        if explicit_link_payload:
            return self._run_local_link_request(req, selected_kb_id, selected_kb_version, explicit_link_payload)

        text = req.message.strip().lower()
        help_terms = ("怎么输入", "如何输入", "怎么用", "如何使用", "帮助", "help", "示例", "例子")
        upload_terms = ("上传知识库", "导入知识库", "上传文件", "导入文件")

        if any(term in text for term in help_terms):
            return AgentChatResponse(
                intent="help",
                reply=(
                    "你可以这样输入：\n"
                    "1. 文本：李导演的《断背山》真是令人动人；实体：李导演，断背山。\n"
                    "2. 文本：国网推进特高压线路扩容；实体：国网。\n"
                    "3. 换成知识库 cck2019。\n"
                    "只有同时提供原文和实体时，我才会生成 LinkRequest 并提交 LangGraph workflow。"
                ),
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=req.model,
            )

        if any(term in text for term in upload_terms):
            return AgentChatResponse(
                intent="upload_kb",
                reply="请在页面下方点击“选择知识库文件”，支持 CCKS kb_data、JSON、JSONL、PDF、TXT 和 Markdown。",
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

    def _extract_explicit_link_payload(self, message: str) -> dict[str, Any] | None:
        pattern = re.compile(
            r"(?:^|[\n\r;；])\s*(?:需要(?:识别|链接|消歧)的)?(?:实体|实体列表|mentions?|mention)(?:包括|如下|为|是)?\s*[:：]\s*",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(message))
        if not matches:
            return None

        marker = matches[-1]
        content = self._strip_text_marker(message[: marker.start()])
        mentions_text = message[marker.end():].strip()
        if not content or not mentions_text:
            return None

        mentions = self._split_mentions(mentions_text)
        if not mentions:
            return None
        return {
            "text": {"content": content, "language": "zh"},
            "mentions": mentions,
        }

    def _strip_text_marker(self, text: str) -> str:
        text = text.strip().strip(";；")
        text = re.sub(r"^(?:文本|原文|句子|内容)\s*[:：]\s*", "", text)
        return text.strip().strip(";；")

    def _run_local_link_request(
        self,
        req: AgentChatRequest,
        selected_kb_id: str | None,
        selected_kb_version: str | None,
        link_payload: dict[str, Any],
    ) -> AgentChatResponse:
        warnings = ["local_explicit_entities_parser"]
        try:
            public_link_request, mention_hints = self._build_link_request(
                link_payload,
                selected_kb_id,
                selected_kb_version,
                req,
                warnings,
            )
        except AgentChatError as exc:
            return AgentChatResponse(
                status="error",
                intent="link",
                reply=exc.message,
                selected_kb_id=selected_kb_id,
                selected_kb_version=selected_kb_version,
                selected_model=req.model,
                warnings=warnings,
            )

        workflow_request, mention_type_diagnostics = self._prepare_link_request(
            public_link_request,
            mention_hints,
            warnings,
        )
        link_response = None
        if req.run_workflow:
            try:
                link_response = self.link_service.link_with_diagnostics(
                    workflow_request,
                    mention_type_diagnostics,
                )
            except ValueError as exc:
                return AgentChatResponse(
                    status="error",
                    intent="link",
                    reply=f"我已经在本地生成了 LinkRequest，但 workflow 执行失败：{exc}",
                    selected_kb_id=public_link_request.knowledge_base.kb_id,
                    selected_kb_version=public_link_request.knowledge_base.kb_version,
                    selected_model=req.model,
                    link_request=public_link_request,
                    warnings=warnings,
                )

        if link_response and link_response.summary:
            reply = (
                self._build_workflow_process_reply(link_response, local_parser=True)
                if self._wants_workflow_detail(req.message)
                else self._build_link_completion_reply(link_response)
            )
        else:
            reply = "已根据明确的实体列表在本地生成标准 LinkRequest JSON。本步没有调用前置大模型解析。"

        return AgentChatResponse(
            intent="link",
            reply=reply,
            selected_kb_id=public_link_request.knowledge_base.kb_id,
            selected_kb_version=public_link_request.knowledge_base.kb_version,
            selected_model=req.model,
            link_request=public_link_request,
            link_response=link_response,
            warnings=warnings,
        )

    def _wants_workflow_detail(self, message: str) -> bool:
        text = message.lower()
        detail_terms = (
            "详细",
            "过程",
            "流程",
            "工作流",
            "workflow",
            "节点",
            "每个实体",
            "各个实体",
            "做的操作",
            "怎么处理",
            "如何处理",
            "为什么",
            "原因",
            "解释",
            "说明",
        )
        return any(term in text for term in detail_terms)

    def _build_link_completion_reply(self, link_response: LinkResponse) -> str:
        summary = link_response.summary
        if not summary:
            return "实体链接已完成，结果已显示在右侧。"
        trace_options = link_response.trace.options_used if link_response.trace else {}
        llm_rerank_count = int(trace_options.get("llm_rerank_count") or 0)
        parts = [
            (
                f"实体链接已完成，共 {summary.total_mentions} 个 mention，"
                f"linked={summary.linked_count}，ambiguous={summary.ambiguous_count}，"
                f"nil={summary.nil_count}，review={summary.review_count}。"
            )
        ]
        if llm_rerank_count:
            parts.append(f"大模型复核参与了 {llm_rerank_count} 个实体，可在结果筛选中查看。")
        elif summary.review_count:
            parts.append("右侧可筛选需要人工复核的实体。")
        else:
            parts.append("详细候选、证据和筛选按钮都在结果区。")
        return "".join(parts)

    def _build_workflow_process_reply(self, link_response: LinkResponse, *, local_parser: bool) -> str:
        summary = link_response.summary
        if not summary:
            return "本轮未返回可展示的 workflow 摘要。"
        trace_options = link_response.trace.options_used if link_response.trace else {}
        llm_rerank_count = int(trace_options.get("llm_rerank_count") or 0)
        coref_count = sum(1 for result in link_response.results if result.coreference)
        lines = [
            "本轮处理过程如下：",
            (
                "1. 前置解析：检测到明确的“文本/实体”列表，未调用大模型抽取 mention，直接构造标准 LinkRequest。"
                if local_parser
                else "1. 前置解析：先由大模型判断是否为实体链接任务，并把自然语言转成标准 LinkRequest。"
            ),
            "2. 工作流：进入 LangGraph 节点链路，依次执行 validate、load_kb、candidate_route、rerank、llm_rerank、resolve、coreference、review_route。",
            (
                f"3. 结果统计：mention={summary.total_mentions}，linked={summary.linked_count}，"
                f"ambiguous={summary.ambiguous_count}，nil={summary.nil_count}，review={summary.review_count}。"
            ),
            (
                f"4. 大模型候选复核：本轮有 {llm_rerank_count} 个实体触发候选级复核。"
                if llm_rerank_count
                else "4. 大模型候选复核：本轮未触发候选级复核。"
            ),
        ]
        if coref_count:
            lines.append(f"5. 共指处理：本轮共处理 {coref_count} 个回指 mention。")

        lines.append("")
        lines.append("各实体关键操作：")
        for result in link_response.results:
            lines.append(self._format_result_process_line(result))
        return "\n".join(lines)

    def _format_result_process_line(self, result: LinkResult) -> str:
        if result.entity:
            target = f"{result.entity.canonical_name}({result.entity.entity_id},{result.entity.entity_type.value})"
        else:
            target = "-"
        mention_type = result.mention_type.value if getattr(result, "mention_type", None) else "UNKNOWN"
        operations = self._summarize_result_operations(result)
        return (
            f"- {result.surface_form} {result.mention_id}: mention_type={mention_type}，"
            f"status={result.link_status.value} -> {target}，confidence={result.confidence:.3f}，"
            f"操作={operations}"
        )

    def _prepare_link_request(
        self,
        link_request: LinkRequest,
        mention_hints: dict[str, MentionHint],
        warnings: list[str],
    ) -> tuple[WorkflowLinkRequest, dict[str, dict[str, Any]]]:
        workflow_request, mention_type_diagnostics = self.link_service.prepare_request(
            link_request,
            mention_hints,
        )
        if mention_type_diagnostics:
            llm_typed = sum(1 for item in mention_type_diagnostics.values() if item.get("status") == "llm")
            heuristic_typed = sum(
                1 for item in mention_type_diagnostics.values() if item.get("status") == "heuristic"
            )
            if llm_typed:
                warnings.append(f"mention_type_llm={llm_typed}")
            if heuristic_typed:
                warnings.append(f"mention_type_heuristic={heuristic_typed}")
        return workflow_request, mention_type_diagnostics

    def _summarize_result_operations(self, result: LinkResult) -> str:
        ops: list[str] = []
        for evidence in result.evidence:
            detail = evidence.detail
            kind = evidence.evidence_type.value
            if kind == "canonical_match":
                ops.append("标准名精确命中")
            elif kind == "alias_match":
                ops.append("别名精确命中")
            elif kind == "former_name_match":
                ops.append("曾用名精确命中")
            elif kind == "similarity_match":
                ops.append("语义向量召回" if "语义向量召回" in detail else "模糊召回")
            elif kind == "context_match":
                ops.append("上下文关键词支持")
            elif kind == "type_match":
                ops.append("类型一致")
            elif kind == "coreference":
                ops.append("共指回链")
            elif kind == "model_inference":
                if "大模型复核" in detail:
                    ops.append("大模型候选复核")
                elif "human_review_required" in detail:
                    ops.append("标记人工复核")
                elif "alias_prior_support" in detail:
                    ops.append("别名先验支持")
                elif "同名重复实体" in detail:
                    ops.append("同名候选去噪")
                elif "模糊召回结果已通过上下文验证" in detail:
                    ops.append("模糊召回上下文验证")
                else:
                    ops.append("模型或规则推断")
        if result.coreference:
            ops.append(f"共指来源 {result.coreference.resolved_from}")
        if result.link_status.value == "nil":
            ops.append("输出 NIL")
        if not ops and result.candidates:
            ops.append("候选召回后按分数排序")
        if not ops:
            ops.append("未获得有效候选")
        return " -> ".join(dict.fromkeys(ops))

    def _build_messages(
        self,
        req: AgentChatRequest,
        selected_kb_id: str | None,
        selected_kb_version: str | None,
    ) -> list[dict[str, str]]:
        kbs = self._list_kbs_lightweight()[:20]
        system = (
            "你是课题10实体链接智能体的对话理解层。你的任务是把用户自然语言转换成严格符合后端协议的 LinkRequest JSON，必要时给出中文回复。\n"
            "只返回 JSON object，不要返回 Markdown。\n"
            "输出 schema:\n"
            "{\n"
            '  "intent": "link|switch_kb|switch_model|upload_kb|chat|help",\n'
            '  "reply": "给用户看的中文回复",\n'
            '  "kb_id": "可选，切换知识库时填写",\n'
            '  "kb_version": "可选",\n'
            '  "model": "可选，切换模型时填写",\n'
            '  "link_request": null 或 LinkRequest 对象\n'
            "}\n"
            "只有当用户明确要求做实体链接，并且同时提供了原文和实体列表时，intent 才能是 link。\n"
            "问候、帮助、解释、切换知识库、切换模型、上传知识库时，都不要生成 link_request。\n"
            "LinkRequest 规则：schema_version 固定为 v1；request_id 可留空；text.content 使用用户原文；"
            "text.language 默认 zh；mentions 是数组，每个 mention 只能包含 mention_id、surface_form、start_offset、end_offset 四个字段；"
            "不要输出 entity_type 或其他内部字段；knowledge_base 使用当前选中的 kb_id 和 kb_version；options 保持默认即可。\n"
            "如果信息不足以链接，intent=chat，并在 reply 中指出缺少原文或实体。"
        )
        context = {
            "current_kb": {"kb_id": selected_kb_id, "kb_version": selected_kb_version},
            "available_kbs": kbs,
            "current_options": req.options.model_dump(),
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "system", "content": "当前上下文：" + json.dumps(context, ensure_ascii=False)},
        ]
        for item in req.history[-8:]:
            if isinstance(item, AgentMessage) and item.role in {"user", "assistant"} and item.content:
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
    ) -> tuple[LinkRequest, dict[str, MentionHint]]:
        if not isinstance(payload, dict):
            raise AgentChatError("INVALID_LINK_REQUEST", "link_request 必须是 JSON object。")

        data = dict(payload)
        text_obj = data.get("text") if isinstance(data.get("text"), dict) else {}
        content = str(text_obj.get("content") or data.get("text_content") or "").strip()
        if not content:
            raise AgentChatError("MISSING_TEXT", "没有提取到 text.content。")

        kb_obj = data.get("knowledge_base") if isinstance(data.get("knowledge_base"), dict) else {}
        kb_id = str(kb_obj.get("kb_id") or selected_kb_id or "").strip()
        kb_version = str(kb_obj.get("kb_version") or selected_kb_version or req.kb_version or "v1").strip()
        if not kb_id:
            raise AgentChatError("MISSING_KB", "当前没有选中知识库，请先上传或选择知识库。")

        mentions_payload = data.get("mentions") or data.get("entities") or []
        mentions, mention_hints = self._normalize_mentions(mentions_payload, content, warnings)
        if not mentions:
            raise AgentChatError("EMPTY_MENTIONS", "没有提取到 mentions。")

        normalized = {
            "schema_version": "v1",
            "request_id": data.get("request_id") or f"agent-{uuid.uuid4().hex[:12]}",
            "text": {"content": content, "language": text_obj.get("language") or "zh"},
            "mentions": mentions,
            "knowledge_base": {"kb_id": kb_id, "kb_version": kb_version},
            "options": req.options.model_dump(),
        }
        try:
            return LinkRequest(**normalized), mention_hints
        except ValidationError as exc:
            raise AgentChatError("LINK_REQUEST_VALIDATION_FAILED", str(exc)) from exc

    def _normalize_mentions(
        self,
        mentions_payload: Any,
        content: str,
        warnings: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, MentionHint]]:
        if isinstance(mentions_payload, str):
            mentions_payload = self._split_mentions(mentions_payload)
        if not isinstance(mentions_payload, list):
            return [], {}

        mentions: list[dict[str, Any]] = []
        mention_hints: dict[str, MentionHint] = {}
        cursor = 0
        for item in mentions_payload:
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

            mention_id = f"m{len(mentions) + 1}"
            mentions.append(
                {
                    "mention_id": mention_id,
                    "surface_form": surface,
                    "start_offset": start,
                    "end_offset": end,
                }
            )

            hint = self._build_mention_hint(item)
            if hint:
                mention_hints[mention_id] = hint
        return mentions, mention_hints

    def _build_mention_hint(self, item: Any) -> MentionHint | None:
        if not isinstance(item, dict):
            return None

        hint = MentionHint(
            mention_type=item.get("mention_type") or item.get("entity_type") or item.get("type"),
        )
        if hint.mention_type.value == "UNKNOWN":
            return None
        return hint

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
        parts = re.split(r"[,;|/\n\r\t、，；]+", text)
        if len(parts) == 1:
            parts = re.split(r"\s+", text)
        return [part.strip() for part in parts if part.strip()]

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
