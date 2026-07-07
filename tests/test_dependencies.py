from pathlib import Path

from app.dependencies import resolve_local_embedding_model_path


def test_resolve_local_embedding_model_path_supports_configured_layout(tmp_path: Path):
    model_dir = tmp_path / "data" / "models" / "BAAI" / "bge-small-zh-v1___5"
    model_dir.mkdir(parents=True)

    resolved = resolve_local_embedding_model_path(str(model_dir))

    assert resolved == model_dir


def test_resolve_local_embedding_model_path_supports_legacy_download_layout(tmp_path: Path):
    configured_path = tmp_path / "data" / "models" / "BAAI" / "bge-small-zh-v1___5"
    legacy_path = tmp_path / "data" / "models" / "BAAI___bge-small-zh-v1.5"
    legacy_path.mkdir(parents=True)

    resolved = resolve_local_embedding_model_path(str(configured_path))

    assert resolved == legacy_path
