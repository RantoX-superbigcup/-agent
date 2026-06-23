"""DeepSeek API client used by the conversational terminal agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


class DeepSeekAPIError(RuntimeError):
    """Raised when DeepSeek cannot return a usable response."""


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "20")),
        )


class DeepSeekChatClient:
    """Small OpenAI-compatible chat client for DeepSeek."""

    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        self.config = config or DeepSeekConfig.from_env()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.is_configured:
            raise DeepSeekAPIError("DEEPSEEK_API_KEY is not configured")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        }
        request = urllib.request.Request(
            url=self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise DeepSeekAPIError(f"DeepSeek HTTP {exc.code}: {detail}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DeepSeekAPIError(f"DeepSeek request failed: {exc}") from exc

        try:
            return str(response_payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekAPIError(f"DeepSeek response missing message content: {response_payload}") from exc

    def analyze_turn(self, user_text: str, current_state: dict[str, Any]) -> dict[str, Any]:
        """Parse a Chinese user turn into a compact dialogue action."""

        messages = [
            {
                "role": "system",
                "content": (
                    "你是课题10实体链接Agent的对话理解模块。"
                    "只输出JSON，不要输出Markdown。"
                    "任务是理解用户输入，抽取知识库、待链接文本、mention列表和是否立即运行。"
                    "如果用户只是在问能力或闲聊，给出reply，不要编造text。"
                    "如果用户要求你自己识别实体，可以从文本中抽取重要实体mention。"
                    "如果mention是代称或带角色称谓，例如李导演、张老师，应在mention_aliases中给出可能的标准名。"
                    "例如根据《断背山》可把李导演扩展为李安，但mentions仍保留用户原文李导演。"
                    "kb_id只能是ccks2019-v1、sample-energy-v1或null。"
                    "sample-energy-v1只用于国家电网、南方电网、配电、能源等样例领域；"
                    "电影、人物、地点、作品、通用百科类文本优先选择ccks2019-v1。"
                    "输出字段固定为：action、kb_id、text、mentions、mention_aliases、run_requested、reply、confidence。"
                    "action取update、run、reply、reset、help或unknown。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_state": current_state,
                        "user_text": user_text,
                        "output_example": {
                            "action": "run",
                            "kb_id": "ccks2019-v1",
                            "text": "南京南站:坐高铁在南京南站下。南京南站",
                            "mentions": ["南京南站", "高铁"],
                            "mention_aliases": {},
                            "run_requested": True,
                            "reply": None,
                            "confidence": 0.92,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        content = self.chat(messages)
        return normalize_turn_action(extract_json_object(content))


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise DeepSeekAPIError(f"DeepSeek did not return JSON: {content}")
        parsed = json.loads(content[start : end + 1])

    if not isinstance(parsed, dict):
        raise DeepSeekAPIError(f"DeepSeek JSON must be an object: {parsed}")
    return parsed


def normalize_turn_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep LLM output predictable for the dialogue manager."""

    action = str(payload.get("action") or "unknown").strip().lower()
    if action not in {"update", "run", "reply", "reset", "help", "unknown"}:
        action = "unknown"

    kb_id = payload.get("kb_id")
    if kb_id not in {"ccks2019-v1", "sample-energy-v1"}:
        kb_id = None

    text = payload.get("text")
    text = str(text).strip() if text is not None else None
    if not text:
        text = None

    raw_mentions = payload.get("mentions") or []
    mentions: list[str] = []
    if isinstance(raw_mentions, list):
        for item in raw_mentions:
            mention = str(item).strip()
            if mention and mention not in mentions:
                mentions.append(mention)

    mention_aliases: dict[str, list[str]] = {}
    raw_aliases = payload.get("mention_aliases") or {}
    if isinstance(raw_aliases, dict):
        for mention, aliases in raw_aliases.items():
            mention_text = str(mention).strip()
            if not mention_text:
                continue
            values = aliases if isinstance(aliases, list) else [aliases]
            mention_aliases[mention_text] = []
            for item in values:
                alias = str(item).strip()
                if alias and alias != mention_text and alias not in mention_aliases[mention_text]:
                    mention_aliases[mention_text].append(alias)
            if not mention_aliases[mention_text]:
                mention_aliases.pop(mention_text, None)

    reply = payload.get("reply")
    reply = str(reply).strip() if reply is not None else None
    if not reply:
        reply = None

    try:
        confidence = float(payload.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "action": action,
        "kb_id": kb_id,
        "text": text,
        "mentions": mentions,
        "mention_aliases": mention_aliases,
        "run_requested": bool(payload.get("run_requested")) or action == "run",
        "reply": reply,
        "confidence": confidence,
    }
