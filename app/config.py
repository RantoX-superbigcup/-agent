from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"
    kb_dir: str = "data/knowledge_bases"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


class AppConfig:
    def __init__(self, settings: Settings, yaml_cfg: dict) -> None:
        self.settings = settings
        self.kb_dir = Path(settings.kb_dir)
        linker = yaml_cfg.get("linker", {})
        self.default_top_k: int = linker.get("default_top_k", 5)
        self.nil_threshold: float = linker.get("nil_threshold", 0.6)
        self.ambiguity_margin: float = linker.get("ambiguity_margin", 0.08)
        self.linker_version: str = linker.get("version", "v1")
        coref = yaml_cfg.get("coreference", {})
        self.coreference_terms: set[str] = set(coref.get("trigger_terms", []))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    settings = Settings()
    config_path = Path("config.yaml")
    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return AppConfig(settings, yaml_cfg)
