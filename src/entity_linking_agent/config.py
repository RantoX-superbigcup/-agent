"""Configuration helpers."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AppConfig:
    app_name: str
    app_version: str
    default_kb_id: str
    default_kb_path: Path
    ccks2019_kb_id: str
    ccks2019_kb_path: Path
    alias_prior_path: Path
    traces_dir: Path
    trace_prefix: str
    default_top_k_candidates: int


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    default_kb_relpath = os.getenv("EL_DEFAULT_KB_PATH", "data/kb/sample_kb.json")
    default_ccks_path = project_root.parent.parent / "ccks2019_el" / "kb_data"

    return AppConfig(
        app_name=os.getenv("EL_APP_NAME", "Topic 10 Entity Linking Agent"),
        app_version=os.getenv("EL_APP_VERSION", "0.1.0"),
        default_kb_id=os.getenv("EL_DEFAULT_KB_ID", "sample-energy-v1"),
        default_kb_path=project_root / default_kb_relpath,
        ccks2019_kb_id=os.getenv("EL_CCKS2019_KB_ID", "ccks2019-v1"),
        ccks2019_kb_path=Path(os.getenv("EL_CCKS2019_KB_PATH", str(default_ccks_path))),
        alias_prior_path=project_root / os.getenv("EL_ALIAS_PRIOR_PATH", "data/models/ccks2019_alias_prior.json"),
        traces_dir=project_root / os.getenv("EL_TRACES_DIR", "artifacts/traces"),
        trace_prefix=os.getenv("EL_TRACE_PREFIX", "t10"),
        default_top_k_candidates=int(os.getenv("EL_DEFAULT_TOP_K", "5")),
    )
