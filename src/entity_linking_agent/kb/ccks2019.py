"""CCKS2019 entity linking dataset adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from entity_linking_agent.core.contracts import KnowledgeBaseEntity, MentionRecord
from entity_linking_agent.utils.text import normalize_text

_SPLIT_PATTERN = re.compile(r"[、,，/；;|｜\s]+")


def load_ccks2019_entities(
    kb_path: Path,
    limit: Optional[int] = None,
    subject_ids: Optional[set[str]] = None,
    alias_texts: Optional[set[str]] = None,
) -> list[KnowledgeBaseEntity]:
    entities: list[KnowledgeBaseEntity] = []
    normalized_alias_texts = {normalize_text(text) for text in alias_texts or set()}

    with kb_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            subject_id = str(item["subject_id"])
            aliases = [item.get("subject", ""), *item.get("alias", [])]
            alias_matched = bool(
                normalized_alias_texts
                and any(normalize_text(alias) in normalized_alias_texts for alias in aliases)
            )
            id_matched = subject_ids is not None and subject_id in subject_ids
            has_filters = subject_ids is not None or bool(normalized_alias_texts)
            if has_filters and not id_matched and not alias_matched:
                continue
            entities.append(convert_ccks2019_entity(item))
            if limit is not None and len(entities) >= limit:
                break

    return entities


def convert_ccks2019_entity(item: dict) -> KnowledgeBaseEntity:
    subject = item.get("subject", "")
    aliases = _dedupe([subject, *item.get("alias", [])])
    entity_type = _first_or_default(item.get("type", []), "Thing")
    keywords: list[str] = []
    description_parts: list[str] = []
    metadata = {
        "source": "ccks2019",
        "type": item.get("type", []),
        "data": item.get("data", []),
    }

    for record in item.get("data", []):
        predicate = record.get("predicate", "")
        value = record.get("object", "")
        if value:
            description_parts.append(f"{predicate}:{value}")
        if predicate in {"义项描述", "标签", "中文名", "中文名称", "别称", "外文名", "摘要"}:
            keywords.extend(_split_keywords(value))
    metadata["description"] = " ".join(description_parts)

    return KnowledgeBaseEntity(
        entity_id=str(item["subject_id"]),
        canonical_name=subject,
        aliases=aliases,
        entity_type=entity_type,
        keywords=_dedupe(keywords)[:32],
        metadata=metadata,
    )


def iter_ccks2019_documents(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def convert_ccks2019_mentions(document: dict) -> list[MentionRecord]:
    mentions: list[MentionRecord] = []
    text = document["text"]
    for index, mention in enumerate(document.get("mention_data", [])):
        start = int(mention.get("offset", 0))
        mention_text = mention["mention"]
        mentions.append(
            MentionRecord(
                mention_id=f"{document['text_id']}:{index}",
                text=mention_text,
                start=start,
                end=start + len(mention_text),
                sentence=text,
                metadata={
                    "text_id": document["text_id"],
                    "gold_kb_id": mention.get("kb_id"),
                },
            )
        )
    return mentions


def _split_keywords(value: str) -> list[str]:
    if not value:
        return []
    parts = [part.strip() for part in _SPLIT_PATTERN.split(value) if part.strip()]
    if len(value) <= 24:
        parts.append(value)
    return parts


def _first_or_default(values: list[str], default: str) -> str:
    return values[0] if values else default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
