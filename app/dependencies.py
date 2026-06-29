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


@lru_cache(maxsize=1)
def get_embedder():
    config = get_config()
    model_path = Path(config.embedding_model)
    looks_like_local_path = model_path.is_absolute() or "/" in config.embedding_model or "\\" in config.embedding_model
    if looks_like_local_path and not model_path.exists():
        logger.warning("Embedding model path not found, vector retrieval disabled: %s", model_path)
        return None
    try:
        from app.core.embedder import Embedder
        return Embedder(config.embedding_model, config.embedding_device)
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
    embedder = get_embedder()
    store = KBStore(config.kb_dir, index_dir=config.index_dir, embedder=embedder)
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
