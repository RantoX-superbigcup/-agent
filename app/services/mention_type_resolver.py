from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from app.config import AppConfig
from app.models.enums import MentionType
from app.models.request import WorkflowLinkRequest as LinkRequest, WorkflowMentionInput as MentionInput
from app.services.llm_provider import append_chat_completions_path, resolve_llm_provider
from app.storage.index import NameIndex
logger = logging.getLogger("entity_link_agent")

_PERSON_HINTS = (
    "导演", "演员", "作家", "作者", "歌手", "球员", "先生", "女士", "主席", "总统", "教授", "医生", "书记",
)
_ORG_HINTS = (
    "公司", "集团", "企业", "机构", "大学", "银行", "医院", "协会", "运营商", "商飞", "证券", "物流", "电网",
)
_LOC_HINTS = (
    "城市", "景区", "湖", "山", "河", "站", "机场", "港", "湾", "古城", "故里", "窟", "宫", "雪山", "省", "市", "县",
)
_OTHER_HINTS = (
    "电影", "电视剧", "作品", "小说", "歌曲", "理念", "方略", "项目", "产品", "模型", "系统", "专辑", "图书",
)


class MentionTypeResolver:
    def __init__(self, config: AppConfig, timeout_seconds: int = 15, max_cases: int = 8) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_cases = max_cases

    def enrich(
        self,
        request: LinkRequest,
        index: NameIndex,
    ) -> tuple[LinkRequest, dict[str, dict[str, Any]]]:
        updated_mentions: list[MentionInput] = []
        diagnostics: dict[str, dict[str, Any]] = {}
        llm_cases: list[dict[str, Any]] = []

        for mention in request.mentions:
            current = mention.mention_type
            if current.value != "UNKNOWN":
                updated_mentions.append(mention)
                diagnostics[mention.mention_id] = {
                    "status": "provided",
                    "mention_type": current.value,
                }
                continue

            exact_matches = index.lookup(mention.surface_form)
            candidate_types = sorted({entity.entity_type.value for entity in exact_matches})
            if len(candidate_types) == 1:
                exact_type = MentionType(candidate_types[0])
                updated_mentions.append(mention.model_copy(update={"mention_type": exact_type}))
                diagnostics[mention.mention_id] = {
                    "status": "exact_match",
                    "mention_type": exact_type.value,
                }
                continue

            heuristic_type = self._heuristic_type(mention, request.text.content)
            if heuristic_type != MentionType.UNKNOWN:
                updated_mentions.append(mention.model_copy(update={"mention_type": heuristic_type}))
                diagnostics[mention.mention_id] = {
                    "status": "heuristic",
                    "mention_type": heuristic_type.value,
                    "candidate_types": candidate_types,
                }
                continue

            if len(candidate_types) > 1 and len(llm_cases) < self.max_cases:
                llm_cases.append(
                    {
                        "mention_id": mention.mention_id,
                        "surface_form": mention.surface_form,
                        "context_window": self._context_window(mention, request.text.content),
                        "candidate_types": candidate_types,
                        "candidate_examples": [
                            {
                                "canonical_name": entity.canonical_name,
                                "entity_type": entity.entity_type.value,
                                "description": entity.description[:120],
                            }
                            for entity in exact_matches[:6]
                        ],
                    }
                )
            updated_mentions.append(mention)
            diagnostics[mention.mention_id] = {
                "status": "unknown",
                "mention_type": MentionType.UNKNOWN.value,
                "candidate_types": candidate_types,
            }

        if llm_cases:
            llm_choices = self._infer_by_llm(request, llm_cases)
            final_mentions: list[MentionInput] = []
            for mention in updated_mentions:
                choice = llm_choices.get(mention.mention_id)
                if not choice or choice == MentionType.UNKNOWN:
                    final_mentions.append(mention)
                    continue
                final_mentions.append(mention.model_copy(update={"mention_type": choice}))
                diagnostics[mention.mention_id] = {
                    "status": "llm",
                    "mention_type": choice.value,
                }
            updated_mentions = final_mentions

        return request.model_copy(update={"mentions": updated_mentions}), diagnostics

    def _infer_by_llm(self, request: LinkRequest, cases: list[dict[str, Any]]) -> dict[str, MentionType]:
        provider = resolve_llm_provider(
            config_api_key=getattr(self.config, "llm_api_key", ""),
            config_base_url=getattr(self.config, "llm_base_url", ""),
            config_model=getattr(self.config, "llm_model", ""),
        )
        if not provider:
            return {}

        payload = [
            {
                "role": "system",
                "content": (
                    "You are a mention type classifier for entity linking. "
                    "Classify each mention into exactly one of PERSON, ORG, LOC, OTHER, UNKNOWN "
                    "using the text context and candidate type hints. "
                    "Return strict JSON only: "
                    "{\"decisions\":[{\"mention_id\":\"m1\",\"entity_type\":\"ORG\",\"reason\":\"...\"}]}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": request.text.content[:3000],
                        "language": request.text.language,
                        "cases": cases,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        body = json.dumps(
            {
                "model": provider.model,
                "messages": payload,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = urllib.request.Request(
            append_chat_completions_path(provider.base_url),
            data=body,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_json_fence(content))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.warning("  [mention_type] LLM API returned %s: %s", exc.code, detail[:500])
            return {}
        except Exception as exc:  # pragma: no cover - network branch
            logger.warning("  [mention_type] mention type inference skipped: %s", exc)
            return {}

        decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
        if not isinstance(decisions, list):
            return {}

        result: dict[str, MentionType] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            mention_id = str(item.get("mention_id") or "").strip()
            entity_type = _normalize_mention_type(item.get("entity_type"))
            if not mention_id or entity_type == MentionType.UNKNOWN:
                continue
            result[mention_id] = entity_type
        return result

    def _heuristic_type(self, mention: MentionInput, context: str) -> MentionType:
        window = self._context_window(mention, context)
        if _contains_any(window, _PERSON_HINTS):
            return MentionType.PERSON
        if _contains_any(window, _ORG_HINTS):
            return MentionType.ORG
        if _contains_any(window, _LOC_HINTS):
            return MentionType.LOC
        if _contains_any(window, _OTHER_HINTS):
            return MentionType.OTHER
        return MentionType.UNKNOWN

    @staticmethod
    def _context_window(mention: MentionInput, context: str, radius: int = 20) -> str:
        if not context:
            return mention.surface_form
        start = max(0, mention.start_offset - radius)
        end = min(len(context), mention.end_offset + radius)
        return context[start:end]


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_mention_type(value: Any) -> MentionType:
    text = str(value or "").strip().upper()
    mapping = {
        "PERSON": MentionType.PERSON,
        "ORG": MentionType.ORG,
        "LOC": MentionType.LOC,
        "OTHER": MentionType.OTHER,
        "UNKNOWN": MentionType.UNKNOWN,
        "人": MentionType.PERSON,
        "人物": MentionType.PERSON,
        "导演": MentionType.PERSON,
        "演员": MentionType.PERSON,
        "组织": MentionType.ORG,
        "机构": MentionType.ORG,
        "公司": MentionType.ORG,
        "企业": MentionType.ORG,
        "地点": MentionType.LOC,
        "地名": MentionType.LOC,
        "城市": MentionType.LOC,
        "作品": MentionType.OTHER,
        "电影": MentionType.OTHER,
        "书籍": MentionType.OTHER,
        "未识别": MentionType.UNKNOWN,
    }
    return mapping.get(text, MentionType.UNKNOWN)


def _strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return content
