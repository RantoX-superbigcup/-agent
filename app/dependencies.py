from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from app.config import get_config
from app.storage.kb_store import KBStore
from app.services.kb_service import KBService
from app.services.link_service import LinkService
from app.core.scorer import AliasPrior


@lru_cache(maxsize=1)
def get_embedder():
    config = get_config()
    try:
        from app.core.embedder import Embedder
        return Embedder(config.embedding_model, config.embedding_device)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_kb_service() -> KBService:
    config = get_config()
    store = KBStore(config.kb_dir, index_dir=config.index_dir)
    return KBService(store)


@lru_cache(maxsize=1)
def get_link_service() -> LinkService:
    config = get_config()
    embedder = get_embedder()
    store = KBStore(config.kb_dir, index_dir=config.index_dir, embedder=embedder)
    prior = AliasPrior.load(Path("data/models/ccks2019_alias_prior.json"))
    return LinkService(store, config, prior, embedder=embedder)
