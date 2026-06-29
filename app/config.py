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
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


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
        emb = yaml_cfg.get("embedding", {})
        self.embedding_model: str = emb.get("model_name", "BAAI/bge-small-zh-v1.5")
        self.embedding_device: str = emb.get("device", "cuda")
        self.index_dir: Path = Path(emb.get("index_dir", "data/vector_index"))
        self.top_k_retrieve: int = emb.get("top_k_retrieve", 20)
        self.llm_api_key: str = settings.llm_api_key or settings.deepseek_api_key
        self.llm_base_url: str = settings.llm_base_url or settings.deepseek_base_url
        self.llm_model: str = settings.llm_model or settings.deepseek_model


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    settings = Settings()
    config_path = Path("config.yaml")
    yaml_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return AppConfig(settings, yaml_cfg)
