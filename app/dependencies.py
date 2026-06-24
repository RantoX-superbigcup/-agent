from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from app.config import get_config
from app.storage.kb_store import KBStore
from app.services.kb_service import KBService
from app.services.link_service import LinkService
from app.core.scorer import AliasPrior


@lru_cache(maxsize=1)
def get_kb_service() -> KBService:
    config = get_config()
    return KBService(KBStore(config.kb_dir))


@lru_cache(maxsize=1)
def get_link_service() -> LinkService:
    config = get_config()
    store = KBStore(config.kb_dir)
    prior = AliasPrior.load(Path("data/models/ccks2019_alias_prior.json"))
    return LinkService(store, config, prior)
