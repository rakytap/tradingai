from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_PROPOSAL_DIR = Path("./data/kg_proposals")
MAX_RELATION_LIST = 12

SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "groq"})


@lru_cache
def get_llm_provider() -> str:
    raw = os.getenv("TRADINGAI_LLM_PROVIDER", "openai").strip().lower()
    if raw in SUPPORTED_LLM_PROVIDERS:
        return raw
    return "openai"


@lru_cache
def get_openai_api_key() -> str | None:
    raw = os.getenv("OPENAI_API_KEY", "").strip()
    return raw or None


@lru_cache
def get_groq_api_key() -> str | None:
    raw = os.getenv("GROQ_API_KEY", "").strip()
    return raw or None


@lru_cache
def get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")


@lru_cache
def get_groq_base_url() -> str:
    return os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL).rstrip("/")


@lru_cache
def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


@lru_cache
def get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def resolve_llm_settings(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    """Resolve provider credentials for OpenAI-compatible Chat Completions."""
    selected = (provider or get_llm_provider()).strip().lower()
    if selected not in SUPPORTED_LLM_PROVIDERS:
        selected = "openai"

    if selected == "groq":
        key = api_key if api_key is not None else get_groq_api_key()
        url = (base_url or get_groq_base_url()).rstrip("/")
        selected_model = model or get_groq_model()
        key_env = "GROQ_API_KEY"
    else:
        key = api_key if api_key is not None else get_openai_api_key()
        url = (base_url or get_openai_base_url()).rstrip("/")
        selected_model = model or get_openai_model()
        key_env = "OPENAI_API_KEY"

    return {
        "provider": selected,
        "api_key": key or "",
        "base_url": url,
        "model": selected_model,
        "key_env": key_env,
    }


@lru_cache
def get_kg_proposal_dir() -> Path:
    raw = os.getenv("TRADINGAI_KG_PROPOSAL_DIR", str(DEFAULT_PROPOSAL_DIR))
    return Path(raw)


def clear_llm_config_cache() -> None:
    get_llm_provider.cache_clear()
    get_openai_api_key.cache_clear()
    get_groq_api_key.cache_clear()
    get_openai_base_url.cache_clear()
    get_groq_base_url.cache_clear()
    get_openai_model.cache_clear()
    get_groq_model.cache_clear()
    get_kg_proposal_dir.cache_clear()
