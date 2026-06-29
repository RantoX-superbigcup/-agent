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
