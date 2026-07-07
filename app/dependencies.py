from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from app.config import get_config
from app.storage.kb_store import KBStore
from app.services.kb_service import KBService
from app.services.kb_file_importer import KBFileImporter
from app.services.link_service import LinkService
from app.services.agent_chat_service import AgentChatService
from app.core.scorer import AliasPrior

logger = logging.getLogger("entity_link_agent")


def _looks_like_local_model_path(model_name: str) -> bool:
    value = model_name.strip()
    if not value:
        return False
    if value.startswith((".", "..", "/", "\\")):
        return True
    if value.startswith(("data/", "data\\", "models/", "models\\")):
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    return Path(value).is_absolute()


def _legacy_local_model_path_candidates(model_name: str) -> list[Path]:
    """
    Return compatibility candidates for older local embedding layouts.

    We historically had two different conventions:
    1. config.yaml expects: data/models/BAAI/bge-small-zh-v1___5
    2. scripts/download_model.py previously saved to:
       data/models/BAAI___bge-small-zh-v1.5

    This helper lets runtime resolve either location.
    """
    original = Path(model_name)
    candidates: list[Path] = [original]

    parts = original.parts
    if len(parts) >= 4 and parts[-3] == "models":
        provider = parts[-2]
        model_leaf = parts[-1]
        merged_leaf = f"{provider}___{model_leaf.replace('___', '.')}"
        merged_path = original.parent.parent / merged_leaf
        if merged_path not in candidates:
            candidates.append(merged_path)

    name = original.name
    if "___" in name:
        split_name = name.replace("___", ".")
        dotted_path = original.with_name(split_name)
        if dotted_path not in candidates:
            candidates.append(dotted_path)

    return candidates


def resolve_local_embedding_model_path(model_name: str) -> Path | None:
    if not _looks_like_local_model_path(model_name):
        return None
    for candidate in _legacy_local_model_path_candidates(model_name):
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_embedder():
    config = get_config()
    model_path = Path(config.embedding_model)
    looks_like_local_path = _looks_like_local_model_path(config.embedding_model)
    if looks_like_local_path:
        resolved_local_path = resolve_local_embedding_model_path(config.embedding_model)
        if resolved_local_path is None:
            searched = ", ".join(str(path) for path in _legacy_local_model_path_candidates(config.embedding_model))
            logger.warning(
                "Embedding model path not found, vector retrieval disabled. searched=[%s]",
                searched,
            )
            return None
        model_path = resolved_local_path
    try:
        from app.core.embedder import Embedder
        return Embedder(str(model_path) if looks_like_local_path else config.embedding_model, config.embedding_device)
    except Exception as exc:
        logger.warning("Embedding model unavailable, vector retrieval disabled: %s", exc)
        return None


@lru_cache(maxsize=1)
def get_kb_service() -> KBService:
    config = get_config()
    store = KBStore(config.kb_dir, index_dir=config.index_dir)
    return KBService(store)


@lru_cache(maxsize=1)
def get_kb_file_importer() -> KBFileImporter:
    config = get_config()
    # Importing or converting a KB file does not need to eagerly warm up the
    # embedding model. If vector search is enabled, the link path can rebuild
    # the vector index lazily on first real retrieval.
    store = KBStore(config.kb_dir, index_dir=config.index_dir, embedder=None)
    return KBFileImporter(store)


@lru_cache(maxsize=1)
def get_link_service() -> LinkService:
    config = get_config()
    embedder = get_embedder()
    store = KBStore(config.kb_dir, index_dir=config.index_dir, embedder=embedder)
    prior = AliasPrior.load(Path("data/models/ccks2019_alias_prior.json"))
    return LinkService(store, config, prior, embedder=embedder)


@lru_cache(maxsize=1)
def get_agent_chat_service() -> AgentChatService:
    config = get_config()
    return AgentChatService(config, get_link_service())
