from __future__ import annotations

import json
import re

import httpx

from tradingai.news.kg_curator.config import resolve_llm_settings
from tradingai.news.kg_curator.prompt import SYSTEM_PROMPT, build_user_prompt


class LLMCuratorError(RuntimeError):
    """Raised when the LLM curator cannot complete a proposal."""


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def chat_completion_json(
    evidence: dict,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str | None = None,
    timeout: float = 60.0,
) -> tuple[dict, str]:
    settings = resolve_llm_settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    key = settings["api_key"]
    if not key:
        raise LLMCuratorError(
            f"{settings['key_env']} is not set for provider '{settings['provider']}'. "
            "Configure it in .env to propose graph edges."
        )
    url_base = settings["base_url"]
    selected_model = settings["model"]
    payload = {
        "model": selected_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(evidence)},
        ],
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{url_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if response.status_code >= 400:
        raise LLMCuratorError(
            f"LLM request failed ({settings['provider']} {response.status_code}): "
            f"{response.text[:400]}"
        )
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMCuratorError(f"Unexpected LLM response shape: {body}") from exc
    try:
        parsed = _extract_json(content)
    except json.JSONDecodeError as exc:
        raise LLMCuratorError(f"LLM returned non-JSON content: {content[:300]}") from exc
    if not isinstance(parsed, dict):
        raise LLMCuratorError("LLM JSON root must be an object")
    return parsed, selected_model
