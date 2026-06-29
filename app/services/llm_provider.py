from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...]
    model_envs: tuple[str, ...]
    default_base_url: str
    default_model: str


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    api_key: str
    api_key_env: str
    base_url: str
    base_url_env: str
    model: str
    model_env: str


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="generic",
        key_envs=("LLM_API_KEY",),
        base_url_envs=("LLM_BASE_URL", "LLM_API_BASE", "LLM_API_URL"),
        model_envs=("LLM_MODEL",),
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    ProviderSpec(
        name="deepseek",
        key_envs=("DEEPSEEK_API_KEY", "DEEPSEEK_APIKEY", "DEEPSEEK_KEY", "DEEPSEEK_API_TOKEN"),
        base_url_envs=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_BASE", "DEEPSEEK_API_URL"),
        model_envs=("DEEPSEEK_MODEL",),
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
    ),
    ProviderSpec(
        name="dashscope",
        key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        base_url_envs=("DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "DASHSCOPE_API_BASE", "QWEN_API_BASE"),
        model_envs=("DASHSCOPE_MODEL", "QWEN_MODEL"),
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
    ),
    ProviderSpec(
        name="openai",
        key_envs=("OPENAI_API_KEY",),
        base_url_envs=("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_URL"),
        model_envs=("OPENAI_MODEL",),
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
    ProviderSpec(
        name="siliconflow",
        key_envs=("SILICONFLOW_API_KEY", "SILICON_API_KEY", "SF_API_KEY"),
        base_url_envs=("SILICONFLOW_BASE_URL", "SILICON_API_BASE", "SF_BASE_URL"),
        model_envs=("SILICONFLOW_MODEL", "SILICON_MODEL", "SF_MODEL"),
        default_base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
    ),
    ProviderSpec(
        name="moonshot",
        key_envs=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        base_url_envs=("MOONSHOT_BASE_URL", "KIMI_BASE_URL"),
        model_envs=("MOONSHOT_MODEL", "KIMI_MODEL"),
        default_base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
    ),
    ProviderSpec(
        name="zhipu",
        key_envs=("ZHIPU_API_KEY", "ZHIPUAI_API_KEY", "GLM_API_KEY"),
        base_url_envs=("ZHIPU_BASE_URL", "ZHIPUAI_BASE_URL", "GLM_BASE_URL"),
        model_envs=("ZHIPU_MODEL", "ZHIPUAI_MODEL", "GLM_MODEL"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
    ),
    ProviderSpec(
        name="ark",
        key_envs=("ARK_API_KEY", "VOLCENGINE_API_KEY", "VOLC_API_KEY"),
        base_url_envs=("ARK_BASE_URL", "VOLCENGINE_BASE_URL", "VOLC_BASE_URL"),
        model_envs=("ARK_MODEL", "VOLCENGINE_MODEL", "VOLC_MODEL"),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-lite-4k",
    ),
)

SUPPORTED_API_KEY_ENV_NAMES = tuple(env for spec in PROVIDER_SPECS for env in spec.key_envs)


def resolve_llm_provider(
    *,
    config_api_key: str = "",
    config_base_url: str = "",
    config_model: str = "",
    preferred_model: Optional[str] = None,
) -> Optional[LLMProviderConfig]:
    """Resolve an OpenAI-compatible provider from env vars, with safe provider defaults."""

    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    specs = _ordered_specs(provider_name)

    for spec in specs:
        key_env, api_key = _first_env(spec.key_envs)
        if not api_key:
            continue
        base_env, base_url = _first_env(spec.base_url_envs)
        model_env, model = _first_env(spec.model_envs)
        return LLMProviderConfig(
            provider=spec.name,
            api_key=api_key,
            api_key_env=key_env,
            base_url=base_url or spec.default_base_url,
            base_url_env=base_env or f"{spec.name}:default_base_url",
            model=preferred_model or model or spec.default_model,
            model_env="request.model" if preferred_model else (model_env or f"{spec.name}:default_model"),
        )

    if config_api_key:
        return LLMProviderConfig(
            provider="config",
            api_key=config_api_key,
            api_key_env="config.llm_api_key",
            base_url=config_base_url or "https://api.deepseek.com",
            base_url_env="config.llm_base_url",
            model=preferred_model or config_model or "deepseek-chat",
            model_env="request.model" if preferred_model else "config.llm_model",
        )
    return None


def append_chat_completions_path(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return url


def _ordered_specs(provider_name: str) -> tuple[ProviderSpec, ...]:
    if not provider_name:
        return PROVIDER_SPECS
    selected = tuple(spec for spec in PROVIDER_SPECS if spec.name == provider_name)
    rest = tuple(spec for spec in PROVIDER_SPECS if spec.name != provider_name)
    return selected + rest


def _first_env(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return name, value
    return "", ""
