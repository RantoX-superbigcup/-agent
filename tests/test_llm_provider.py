from app.services.llm_provider import resolve_llm_provider


def _clear_llm_env(monkeypatch):
    names = [
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_resolve_dashscope_from_qwen_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")

    provider = resolve_llm_provider()

    assert provider is not None
    assert provider.provider == "dashscope"
    assert provider.api_key_env == "QWEN_API_KEY"
    assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert provider.model == "qwen-plus"


def test_resolve_provider_model_override(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-turbo")

    provider = resolve_llm_provider(preferred_model="qwen-max")

    assert provider is not None
    assert provider.provider == "dashscope"
    assert provider.model == "qwen-max"


def test_resolve_openai_key(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = resolve_llm_provider()

    assert provider is not None
    assert provider.provider == "openai"
    assert provider.base_url == "https://api.openai.com/v1"


def test_llm_provider_override_changes_priority(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen")
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")

    provider = resolve_llm_provider()

    assert provider is not None
    assert provider.provider == "dashscope"
    assert provider.api_key_env == "QWEN_API_KEY"


def test_qwen_model_prefers_dashscope_key_when_multiple_keys_exist(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen")

    provider = resolve_llm_provider(preferred_model="qwen-plus")

    assert provider is not None
    assert provider.provider == "dashscope"
    assert provider.api_key_env == "QWEN_API_KEY"
    assert provider.model == "qwen-plus"
    assert "dashscope.aliyuncs.com" in provider.base_url


def test_qwen_model_is_not_sent_to_forced_deepseek_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    provider = resolve_llm_provider(preferred_model="qwen-plus")

    assert provider is not None
    assert provider.provider == "deepseek"
    assert provider.model == "deepseek-chat"
    assert provider.model_env == "deepseek:default_model"


def test_config_deepseek_ignores_incompatible_qwen_model(monkeypatch):
    _clear_llm_env(monkeypatch)

    provider = resolve_llm_provider(
        config_api_key="sk-deepseek",
        config_base_url="https://api.deepseek.com",
        config_model="deepseek-v4-flash",
        preferred_model="qwen-plus",
    )

    assert provider is not None
    assert provider.provider == "config"
    assert provider.model == "deepseek-v4-flash"
    assert provider.model_env == "config.llm_model"
