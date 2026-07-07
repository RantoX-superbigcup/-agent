from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig
from app.core import candidate as candidate_mod
from app.models.request import LinkOptions, WorkflowLinkRequest as LinkRequest
from app.services.llm_provider import (
    SUPPORTED_API_KEY_ENV_NAMES,
    append_chat_completions_path,
    resolve_llm_provider,
)

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
        self.last_diagnostics: dict[str, Any] = {
            "status": "not_run",
            "reason": "not_started",
        }

    def rerank(
        self,
        request: LinkRequest,
        candidates_by_id: dict[str, list[candidate_mod.CandidateResult]],
        options: LinkOptions,
    ) -> dict[str, LLMRerankChoice]:
        self._record("started", "building_cases")
        if not options.enable_llm_rerank:
            self._record("disabled", "enable_llm_rerank_false")
            return {}

        cases = self._build_cases(request, candidates_by_id, options)
        if not cases:
            self._record("skipped", "no_risky_candidates", case_count=0)
            return {}

        provider = resolve_llm_provider(
            config_api_key=getattr(self.config, "llm_api_key", ""),
            config_base_url=getattr(self.config, "llm_base_url", ""),
            config_model=getattr(self.config, "llm_model", ""),
        )
        if not provider:
            self._record(
                "skipped",
                "provider_not_configured",
                case_count=len(cases),
                supported_api_key_envs=list(SUPPORTED_API_KEY_ENV_NAMES),
            )
            logger.warning("  [llm_rerank] provider_not_configured, skip model rerank")
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
            self._record(
                "failed",
                "http_error",
                case_count=len(cases),
                provider=provider.provider,
                model=provider.model,
                api_key_env=provider.api_key_env,
                base_url=provider.base_url,
                http_status=exc.code,
                detail=detail[:500],
            )
            logger.warning("  [llm_rerank] LLM API returned %s: %s", exc.code, detail[:500])
            return {}
        except Exception as exc:
            self._record(
                "failed",
                "request_or_parse_error",
                case_count=len(cases),
                provider=provider.provider,
                model=provider.model,
                api_key_env=provider.api_key_env,
                base_url=provider.base_url,
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            logger.warning("  [llm_rerank] LLM rerank skipped: %s", exc)
            return {}

        choices, parse_diagnostics = self._parse_choices_with_diagnostics(parsed, candidates_by_id)
        self._record(
            "accepted" if choices else "filtered",
            "accepted_choices" if choices else "all_model_decisions_filtered",
            case_count=len(cases),
            provider=provider.provider,
            model=provider.model,
            api_key_env=provider.api_key_env,
            base_url=provider.base_url,
            accepted_count=len(choices),
            parse=parse_diagnostics,
        )
        if choices:
            logger.info("  [llm_rerank] accepted %d model rerank choice(s)", len(choices))
        else:
            logger.warning("  [llm_rerank] all model decisions filtered: %s", parse_diagnostics)
        return choices

    def _record(self, status: str, reason: str, **extra: Any) -> None:
        self.last_diagnostics = {
            "status": status,
            "reason": reason,
            **extra,
        }

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
                    "mention_type": (
                        mention.mention_type.value
                        if mention.mention_type and mention.mention_type.value != "UNKNOWN"
                        else None
                    ),
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
        if candidate_mod.trusted_exact_rank(top) == 0 and second.score >= options.nil_threshold:
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
        choices, _ = LLMReranker._parse_choices_with_diagnostics(parsed, candidates_by_id)
        return choices

    @staticmethod
    def _parse_choices_with_diagnostics(
        parsed: Any,
        candidates_by_id: dict[str, list[candidate_mod.CandidateResult]],
    ) -> tuple[dict[str, LLMRerankChoice], dict[str, Any]]:
        decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
        if not isinstance(decisions, list):
            return {}, {"total_decisions": 0, "accepted": 0, "skipped": {"invalid_decisions_format": 1}}

        result: dict[str, LLMRerankChoice] = {}
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for item in decisions:
            if not isinstance(item, dict):
                skip("non_object_decision")
                continue
            mention_id = str(item.get("mention_id") or "").strip()
            entity_id = str(item.get("entity_id") or "").strip()
            if not mention_id or not entity_id:
                skip("missing_mention_or_entity_id")
                continue
            if entity_id.upper() == "NIL":
                skip("nil_decision")
                continue
            mention_candidates = candidates_by_id.get(mention_id, [])
            if not mention_candidates:
                skip("unknown_mention_id")
                continue
            candidate_ids = {candidate.entity.entity_id for candidate in mention_candidates}
            if entity_id not in candidate_ids:
                skip("entity_id_not_in_candidates")
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.55:
                skip("confidence_below_0_55")
                continue
            result[mention_id] = LLMRerankChoice(
                mention_id=mention_id,
                entity_id=entity_id,
                confidence=max(0.0, min(1.0, confidence)),
                reason=str(item.get("reason") or "").strip()[:160],
            )
        return result, {
            "total_decisions": len(decisions),
            "accepted": len(result),
            "skipped": skipped,
        }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()
