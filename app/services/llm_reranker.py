from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig
from app.core import candidate as candidate_mod
from app.models.request import LinkOptions, LinkRequest
from app.services.llm_provider import append_chat_completions_path, resolve_llm_provider

logger = logging.getLogger("entity_link_agent")


@dataclass(frozen=True)
class LLMRerankChoice:
    mention_id: str
    entity_id: str
    confidence: float
    reason: str = ""


class LLMReranker:
    def __init__(self, config: AppConfig, timeout_seconds: int = 20, max_cases: int = 6) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_cases = max_cases

    def rerank(
        self,
        request: LinkRequest,
        candidates_by_id: dict[str, list[candidate_mod.CandidateResult]],
        options: LinkOptions,
    ) -> dict[str, LLMRerankChoice]:
        if not options.enable_llm_rerank:
            return {}

        cases = self._build_cases(request, candidates_by_id, options)
        if not cases:
            return {}

        provider = resolve_llm_provider(
            config_api_key=getattr(self.config, "llm_api_key", ""),
            config_base_url=getattr(self.config, "llm_base_url", ""),
            config_model=getattr(self.config, "llm_model", ""),
        )
        if not provider:
            logger.info("  [llm_rerank] LLM API key not configured, skip model rerank")
            return {}

        payload = self._request_payload(request, cases)
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
            logger.warning("  [llm_rerank] LLM API returned %s: %s", exc.code, detail[:500])
            return {}
        except Exception as exc:
            logger.warning("  [llm_rerank] LLM rerank skipped: %s", exc)
            return {}

        choices = self._parse_choices(parsed, candidates_by_id)
        if choices:
            logger.info("  [llm_rerank] accepted %d model rerank choice(s)", len(choices))
        return choices

    def _build_cases(
        self,
        request: LinkRequest,
        candidates_by_id: dict[str, list[candidate_mod.CandidateResult]],
        options: LinkOptions,
    ) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        mentions_by_id = {mention.mention_id: mention for mention in request.mentions}
        for mention_id, candidates in candidates_by_id.items():
            if len(candidates) < 2 or not self._needs_llm_rerank(candidates, options):
                continue
            mention = mentions_by_id.get(mention_id)
            if mention is None:
                continue
            cases.append(
                {
                    "mention_id": mention_id,
                    "surface_form": mention.surface_form,
                    "mention_type": mention.entity_type.value if mention.entity_type else None,
                    "candidates": [
                        {
                            "entity_id": candidate.entity.entity_id,
                            "canonical_name": candidate.entity.canonical_name,
                            "entity_type": candidate.entity.entity_type.value,
                            "score": candidate.score,
                            "match_source": candidate.match_source,
                            "aliases": candidate.entity.aliases[:6],
                            "former_names": candidate.entity.former_names[:4],
                            "keywords": candidate.entity.keywords[:8],
                            "description": candidate.entity.description[:180],
                        }
                        for candidate in candidates[: min(options.top_k, 5)]
                    ],
                }
            )
            if len(cases) >= self.max_cases:
                break
        return cases

    @staticmethod
    def _needs_llm_rerank(candidates: list[candidate_mod.CandidateResult], options: LinkOptions) -> bool:
        top, second = candidates[0], candidates[1]
        margin = top.score - second.score
        if margin < max(options.ambiguity_margin, 0.08):
            return True
        if top.score < options.nil_threshold + 0.12:
            return True
        if top.match_source == "similarity_match" and second.score >= options.nil_threshold:
            return True
        return False

    def _request_payload(self, request: LinkRequest, cases: list[dict[str, Any]]) -> list[dict[str, str]]:
        system = (
            "You are an entity linking reranker. Choose the best entity_id for each mention from the provided "
            "candidate list only. Use the original text, aliases, former_names, keywords and descriptions. "
            "Return strict JSON only: {\"decisions\":[{\"mention_id\":\"...\",\"entity_id\":\"...\","
            "\"confidence\":0.0,\"reason\":\"short Chinese reason\"}]}. "
            "If no candidate is suitable, set entity_id to \"NIL\"."
        )
        user = {
            "text": request.text.content[:3000],
            "language": request.text.language,
            "knowledge_base": request.knowledge_base.model_dump(),
            "cases": cases,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

    @staticmethod
    def _parse_choices(
        parsed: Any,
        candidates_by_id: dict[str, list[candidate_mod.CandidateResult]],
    ) -> dict[str, LLMRerankChoice]:
        decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
        if not isinstance(decisions, list):
            return {}

        result: dict[str, LLMRerankChoice] = {}
        for item in decisions:
            if not isinstance(item, dict):
                continue
            mention_id = str(item.get("mention_id") or "").strip()
            entity_id = str(item.get("entity_id") or "").strip()
            if not mention_id or not entity_id or entity_id.upper() == "NIL":
                continue
            candidate_ids = {candidate.entity.entity_id for candidate in candidates_by_id.get(mention_id, [])}
            if entity_id not in candidate_ids:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.55:
                continue
            result[mention_id] = LLMRerankChoice(
                mention_id=mention_id,
                entity_id=entity_id,
                confidence=max(0.0, min(1.0, confidence)),
                reason=str(item.get("reason") or "").strip()[:160],
            )
        return result


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()
