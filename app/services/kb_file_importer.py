from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import get_config
from app.models.entity import Entity, KBPackage
from app.models.enums import EntityType
from app.models.kb_import import KBFileImportRequest, KBFileImportResponse
from app.services.llm_provider import (
    SUPPORTED_API_KEY_ENV_NAMES,
    append_chat_completions_path,
    resolve_llm_provider,
)
from app.storage.kb_store import KBStore


class KBFileImportError(ValueError):
    def __init__(self, code: str, message: str, warnings: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.warnings = warnings or []


class KBFileImporter:
    STRUCTURED_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    API_KEY_ENV_NAMES = SUPPORTED_API_KEY_ENV_NAMES
    MODEL_ENV_NAMES = ("DEEPSEEK_MODEL", "LLM_MODEL")
    BASE_URL_ENV_NAMES = (
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_API_URL",
        "LLM_BASE_URL",
    )

    def __init__(self, store: KBStore) -> None:
        self.store = store

    def import_file(self, req: KBFileImportRequest) -> KBFileImportResponse:
        path = self._resolve_path(req.file_path)
        source_type = self._detect_source_type(path, req.source_type)
        warnings: list[str] = []

        try:
            package = self._convert(path, source_type, req, warnings)
        except ValidationError as exc:
            raise KBFileImportError("INVALID_KB_FORMAT", str(exc), warnings) from exc

        existed = self.store.exists(package.kb_id)
        if req.import_to_store:
            self.store.import_full(package.kb_id, package.kb_version, package.description, package.entities)
            status = "overwritten" if existed else "created"
        else:
            status = "converted"

        preview = package.entities[: req.preview_limit] if req.preview_limit else []
        return KBFileImportResponse(
            status=status,
            imported=req.import_to_store,
            source_type=source_type,
            file_path=str(path),
            kb_id=package.kb_id,
            kb_version=package.kb_version,
            description=package.description,
            entity_count=len(package.entities),
            warnings=warnings,
            entities_preview=preview,
            package=package if req.include_entities else None,
        )

    def _convert(
        self,
        path: Path,
        source_type: str,
        req: KBFileImportRequest,
        warnings: list[str],
    ) -> KBPackage:
        if source_type in {"kb_package", "entities", "ccks_kb_data"}:
            prefer_jsonl = source_type == "ccks_kb_data" or "kb_data" in path.name.lower()
            payload = self._read_structured_payload(path, prefer_jsonl=prefer_jsonl)
            return self._package_from_payload(payload, path, source_type, req, warnings)

        if source_type == "pdf":
            text = self._extract_pdf_text(path, req.max_text_chars, warnings)
            return self._package_from_unstructured_text(text, path, source_type, req, warnings)

        if source_type == "text":
            text = self._read_text_with_fallback(path)[: req.max_text_chars]
            return self._package_from_unstructured_text(text, path, source_type, req, warnings)

        raise KBFileImportError("UNSUPPORTED_SOURCE_TYPE", f"Unsupported source_type: {source_type}", warnings)

    def _package_from_payload(
        self,
        payload: Any,
        path: Path,
        source_type: str,
        req: KBFileImportRequest,
        warnings: list[str],
    ) -> KBPackage:
        if source_type == "kb_package" or self._looks_like_kb_package(payload):
            if not isinstance(payload, dict):
                raise KBFileImportError("UNSUPPORTED_JSON_SHAPE", "KBPackage payload must be a JSON object.", warnings)
            data = dict(payload)
            if req.kb_id:
                data["kb_id"] = req.kb_id
            if req.kb_version:
                data["kb_version"] = req.kb_version
            if req.description is not None:
                data["description"] = req.description
            return KBPackage(**data)

        entities_payload, forced_type = self._extract_entities_payload(payload)

        if not isinstance(entities_payload, list):
            raise KBFileImportError(
                "UNSUPPORTED_JSON_SHAPE",
                "JSON must be a KBPackage object, an entity list, a CCKS kb_data list/jsonl file, "
                'or an object like {"kb_data": [...]}.',
                warnings,
            )

        is_ccks = (
            source_type == "ccks_kb_data"
            or forced_type == "ccks_kb_data"
            or any(self._looks_like_ccks_record(item) for item in entities_payload[:20])
        )
        if is_ccks:
            entities = [self._ccks_record_to_entity(item, idx, warnings) for idx, item in enumerate(entities_payload)]
            inferred = "ccks_kb_data"
        else:
            entities = [self._entity_from_mapping(item, idx, warnings) for idx, item in enumerate(entities_payload)]
            inferred = "entities"

        warnings.append(f"detected_source_type={inferred}")
        return KBPackage(
            kb_id=req.kb_id or self._default_kb_id(path),
            kb_version=req.kb_version,
            description=req.description or f"Imported from {path.name}",
            entities=entities,
        )

    def _extract_entities_payload(self, payload: Any) -> tuple[Any, str | None]:
        if isinstance(payload, dict):
            if "kb_data" in payload:
                return payload["kb_data"], "ccks_kb_data"
            if "entities" in payload:
                return payload["entities"], None
            if self._looks_like_ccks_record(payload):
                return [payload], "ccks_kb_data"
        return payload, None

    def _package_from_unstructured_text(
        self,
        text: str,
        path: Path,
        source_type: str,
        req: KBFileImportRequest,
        warnings: list[str],
    ) -> KBPackage:
        embedded = self._try_parse_embedded_json(text)
        if embedded is not None:
            warnings.append("parsed_embedded_json_from_text")
            return self._package_from_payload(embedded, path, "auto", req, warnings)

        if not req.use_llm:
            raise KBFileImportError(
                "UNSTRUCTURED_TEXT_REQUIRES_LLM",
                "PDF/text is unstructured. Set use_llm=true and configure an LLM API key "
                "environment variable to extract entities.",
                warnings,
            )

        payload = self._extract_entities_with_llm(text, path, req, warnings)
        return self._package_from_payload(payload, path, "entities", req, warnings)

    def _read_structured_payload(self, path: Path, prefer_jsonl: bool = False) -> Any:
        if prefer_jsonl:
            jsonl_payload = self._try_read_jsonl_payload(path)
            if jsonl_payload is not None:
                return jsonl_payload

        text = self._read_text_with_fallback(path).strip()
        if not text:
            raise KBFileImportError("EMPTY_FILE", "The input file is empty.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise KBFileImportError(
                    "INVALID_JSON_OR_JSONL",
                    f"Cannot parse JSON or JSONL. First invalid line: {line_no}. {exc}",
                ) from exc
        if not records:
            raise KBFileImportError("EMPTY_JSONL", "No JSONL records found.")
        return records

    def _try_read_jsonl_payload(self, path: Path) -> list[Any] | None:
        for encoding in self.STRUCTURED_ENCODINGS:
            records: list[Any] = []
            try:
                with path.open("r", encoding=encoding) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
                return records if records else None
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError:
                return None
        return None

    def _read_text_with_fallback(self, path: Path) -> str:
        raw = path.read_bytes()
        errors = []
        for encoding in self.STRUCTURED_ENCODINGS:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
        raise KBFileImportError("TEXT_DECODE_FAILED", "Cannot decode text file. Tried utf-8/gb18030/gbk. " + "; ".join(errors))

    def _extract_pdf_text(self, path: Path, max_chars: int, warnings: list[str]) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise KBFileImportError(
                "PDF_READER_NOT_INSTALLED",
                "PDF support requires pypdf. Run: python -m pip install pypdf",
                warnings,
            ) from exc

        try:
            reader = PdfReader(str(path))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
                if sum(len(c) for c in chunks) >= max_chars:
                    break
            text = "\n".join(chunks).strip()[:max_chars]
        except Exception as exc:
            raise KBFileImportError("PDF_READ_FAILED", f"Failed to read PDF: {exc}", warnings) from exc

        if not text:
            raise KBFileImportError(
                "PDF_TEXT_EMPTY",
                "No selectable text was extracted from the PDF. It may be a scanned PDF and needs OCR first.",
                warnings,
            )
        warnings.append(f"pdf_text_chars={len(text)}")
        return text

    def _extract_entities_with_llm(
        self,
        text: str,
        path: Path,
        req: KBFileImportRequest,
        warnings: list[str],
    ) -> Any:
        config = get_config()
        provider = resolve_llm_provider(
            config_api_key=config.llm_api_key,
            config_base_url=config.llm_base_url,
            config_model=config.llm_model,
        )
        if not provider:
            raise KBFileImportError(
                "LLM_NOT_CONFIGURED",
                "Missing LLM API key. Supported env names: "
                + ", ".join(self.API_KEY_ENV_NAMES),
                warnings,
            )

        url = append_chat_completions_path(provider.base_url)

        system = (
            "You convert Chinese or English documents into an entity-linking knowledge base. "
            "Return strict JSON only. The JSON must be {\"entities\":[...]}. "
            "Each entity must have entity_id, canonical_name, entity_type, aliases, former_names, "
            "description, parent_ids, keywords. entity_type must be ORG, PERSON, LOC, or OTHER."
        )
        user = (
            f"Source file: {path.name}\n"
            f"Target kb_id: {req.kb_id or self._default_kb_id(path)}\n"
            "Extract stable named entities suitable for a knowledge base. "
            "Do not invent facts. If uncertain, omit the entity.\n\n"
            f"{text[: req.max_text_chars]}"
        )
        body = json.dumps(
            {
                "model": provider.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KBFileImportError("LLM_HTTP_ERROR", f"LLM API returned {exc.code}: {detail}", warnings) from exc
        except Exception as exc:
            raise KBFileImportError("LLM_REQUEST_FAILED", f"LLM request failed: {exc}", warnings) from exc

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(self._strip_json_fence(content))
        except Exception as exc:
            raise KBFileImportError("LLM_RESPONSE_PARSE_FAILED", f"Cannot parse LLM response as JSON: {exc}", warnings) from exc

        warnings.append(f"entities_extracted_by_llm provider={provider.provider} api_key_env={provider.api_key_env}")
        return parsed.get("entities", parsed)

    def _entity_from_mapping(self, item: Any, idx: int, warnings: list[str]) -> Entity:
        if not isinstance(item, dict):
            raise KBFileImportError("INVALID_ENTITY_ITEM", f"Entity item at index {idx} is not an object.", warnings)
        data = dict(item)
        data.setdefault("entity_id", f"E{idx + 1:06d}")
        data.setdefault("canonical_name", data.get("name") or data.get("subject") or data.get("label") or "")
        data["entity_type"] = self._map_entity_type(data.get("entity_type") or data.get("type") or data.get("subject_type"), data)
        data["aliases"] = self._as_str_list(data.get("aliases") or data.get("alias"))
        data["former_names"] = self._as_str_list(data.get("former_names") or data.get("former_name"))
        data.setdefault("description", "")
        data["parent_ids"] = self._as_str_list(data.get("parent_ids"))
        data["keywords"] = self._as_str_list(data.get("keywords"))
        if not data["canonical_name"]:
            raise KBFileImportError("MISSING_ENTITY_NAME", f"Entity item at index {idx} has no canonical_name/name/subject.", warnings)
        return Entity(**data)

    def _ccks_record_to_entity(self, item: Any, idx: int, warnings: list[str]) -> Entity:
        if not isinstance(item, dict):
            raise KBFileImportError("INVALID_CCKS_RECORD", f"CCKS record at index {idx} is not an object.", warnings)

        subject = str(item.get("subject") or item.get("name") or item.get("label") or "").strip()
        if not subject:
            raise KBFileImportError("MISSING_CCKS_SUBJECT", f"CCKS record at index {idx} has no subject.", warnings)

        facts = self._normalize_facts(item.get("data") or item.get("attrs") or [])
        fact_text = " ".join(f"{predicate}:{obj}" for predicate, obj in facts[:12])
        description = self._description_from_facts(facts) or str(item.get("description") or "").strip()

        aliases = self._as_str_list(item.get("alias") or item.get("aliases"))
        aliases.extend(self._aliases_from_facts(facts))
        aliases = [alias for alias in dict.fromkeys(aliases) if alias and alias != subject]

        keyword_text = " ".join([subject, " ".join(aliases), description, fact_text])
        entity_type = self._map_entity_type(
            item.get("type") or item.get("subject_type"),
            {**item, "canonical_name": subject, "description": description, "facts": facts},
        )

        return Entity(
            entity_id=str(item.get("subject_id") or item.get("entity_id") or item.get("id") or f"ccks-{idx + 1:06d}"),
            canonical_name=subject,
            entity_type=entity_type,
            aliases=aliases,
            former_names=[],
            description=(description or fact_text)[:1200],
            parent_ids=[],
            keywords=self._keywords_from_text(keyword_text)[:20],
        )

    def _normalize_facts(self, raw_data: Any) -> list[tuple[str, str]]:
        if isinstance(raw_data, dict):
            raw_data = [{"predicate": key, "object": value} for key, value in raw_data.items()]
        if not isinstance(raw_data, list):
            return []

        facts: list[tuple[str, str]] = []
        for fact in raw_data:
            if not isinstance(fact, dict):
                continue
            predicate = str(fact.get("predicate") or fact.get("property") or fact.get("key") or "").strip()
            obj = fact.get("object", fact.get("value", ""))
            if isinstance(obj, (list, tuple, set)):
                obj_text = " | ".join(str(x) for x in obj if x is not None)
            elif isinstance(obj, dict):
                obj_text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            else:
                obj_text = str(obj or "")
            obj_text = obj_text.strip()
            if predicate or obj_text:
                facts.append((predicate, obj_text))
        return facts

    def _description_from_facts(self, facts: list[tuple[str, str]]) -> str:
        preferred = (
            "\u6458\u8981",
            "\u4e49\u9879\u63cf\u8ff0",
            "\u63cf\u8ff0",
            "\u7b80\u4ecb",
            "\u4ecb\u7ecd",
            "description",
            "summary",
        )
        for predicate, obj in facts:
            low = predicate.lower()
            if obj and any(key.lower() in low for key in preferred):
                return obj
        long_values = [obj for _, obj in facts if len(obj) >= 20]
        return long_values[0] if long_values else ""

    def _aliases_from_facts(self, facts: list[tuple[str, str]]) -> list[str]:
        alias_predicates = (
            "\u522b\u540d",
            "\u522b\u79f0",
            "\u4e2d\u6587\u540d",
            "\u4e2d\u6587\u540d\u79f0",
            "\u4e2d\u6587\u5b66\u540d",
            "\u5916\u6587\u540d",
            "\u5916\u6587\u540d\u79f0",
            "\u82f1\u6587\u540d",
            "\u62c9\u4e01\u5b66\u540d",
        )
        aliases: list[str] = []
        for predicate, obj in facts:
            if any(key in predicate for key in alias_predicates):
                aliases.extend(self._as_str_list(obj))
        return list(dict.fromkeys(aliases))

    def _map_entity_type(self, raw_type: Any, context: dict[str, Any]) -> EntityType:
        pieces = self._as_str_list(raw_type)
        pieces.extend(
            self._as_str_list(
                [
                    context.get("canonical_name"),
                    context.get("subject"),
                    context.get("description"),
                ]
            )
        )
        facts = context.get("facts")
        if isinstance(facts, list):
            pieces.extend(f"{predicate} {obj}" for predicate, obj in facts[:20])
        text = " ".join(pieces).lower()

        if self._contains_any(
            text,
            (
                "person",
                "people",
                "human",
                "actor",
                "director",
                "singer",
                "writer",
                "\u4eba\u7269",
                "\u4eba\u540d",
                "\u5bfc\u6f14",
                "\u6f14\u5458",
                "\u4f5c\u5bb6",
                "\u6b4c\u624b",
                "\u51fa\u751f",
                "\u804c\u4e1a",
            ),
        ):
            return EntityType.PERSON

        if self._contains_any(
            text,
            (
                "loc",
                "location",
                "place",
                "geo",
                "city",
                "country",
                "province",
                "\u5730\u70b9",
                "\u5730\u7406",
                "\u57ce\u5e02",
                "\u56fd\u5bb6",
                "\u7701",
                "\u53bf",
                "\u533a",
                "\u5c71",
                "\u6cb3",
                "\u4f4d\u4e8e",
                "\u5730\u5740",
            ),
        ):
            return EntityType.LOC

        if self._contains_any(
            text,
            (
                "org",
                "organization",
                "company",
                "enterprise",
                "university",
                "school",
                "bank",
                "hospital",
                "\u673a\u6784",
                "\u516c\u53f8",
                "\u4f01\u4e1a",
                "\u96c6\u56e2",
                "\u5927\u5b66",
                "\u5b66\u6821",
                "\u94f6\u884c",
                "\u533b\u9662",
                "\u534f\u4f1a",
                "\u7f51\u7ad9",
                "\u653f\u5e9c",
            ),
        ):
            return EntityType.ORG

        return EntityType.OTHER

    def _keywords_from_text(self, text: str) -> list[str]:
        tokens = re.split(r"[\s,;|/()\[\]{}<>\u3001\uff0c\u3002\uff1b\uff1a\uff08\uff09\u300a\u300b]+", text)
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip()
            if 2 <= len(token) <= 32 and token not in seen:
                seen.add(token)
                result.append(token)
        return result

    def _as_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = re.split(r"[,;|/\u3001\uff0c\uff1b]+", value)
            return [part.strip() for part in parts if part.strip()]
        if isinstance(value, (list, tuple, set)):
            result: list[str] = []
            for item in value:
                result.extend(self._as_str_list(item))
            return list(dict.fromkeys(result))
        text = str(value).strip()
        return [text] if text else []

    def _resolve_path(self, file_path: str) -> Path:
        raw = Path(file_path).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(Path.cwd() / raw)
        for path in candidates:
            path = path.resolve()
            if path.exists():
                if not path.is_file():
                    raise KBFileImportError("PATH_NOT_FILE", f"Path is not a file: {path}")
                return path
        raise KBFileImportError("FILE_NOT_FOUND", f"File not found: {file_path}")

    def _detect_source_type(self, path: Path, requested: str) -> str:
        if requested != "auto":
            return requested

        name = path.name.lower()
        suffix = path.suffix.lower()
        if name == "kb_data" or "kb_data" in name:
            return "ccks_kb_data"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".txt", ".md"}:
            return "text"
        if suffix in {".json", ".jsonl"}:
            return "entities"
        raise KBFileImportError(
            "UNSUPPORTED_FILE_EXTENSION",
            f"Unsupported file extension '{suffix or '<none>'}'. Supported: .json, .jsonl, .pdf, .txt, .md, or CCKS kb_data.",
        )

    def _looks_like_kb_package(self, payload: Any) -> bool:
        return isinstance(payload, dict) and {"kb_id", "kb_version", "entities"}.issubset(payload.keys())

    def _looks_like_ccks_record(self, item: Any) -> bool:
        return isinstance(item, dict) and bool(item.get("subject") or item.get("subject_id")) and (
            "data" in item or "alias" in item or "type" in item
        )

    def _try_parse_embedded_json(self, text: str) -> Any | None:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def _strip_json_fence(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()
        return content

    def _default_kb_id(self, path: Path) -> str:
        text = re.sub(r"[^0-9A-Za-z_\-]+", "-", path.stem).strip("-").lower()
        return text or "imported-kb"

    def _first_env(self, names: tuple[str, ...]) -> str:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return ""

    def _contains_any(self, text: str, needles: tuple[str, ...]) -> bool:
        return any(needle in text for needle in needles)
