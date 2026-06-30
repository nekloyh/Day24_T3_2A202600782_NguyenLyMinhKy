"""Provider-neutral chat wrapper with a safe NVIDIA rate-limit fallback."""

from __future__ import annotations

import json
import time
from typing import Any

from config import (
    ACTIVE_LLM_PROVIDER,
    FALLBACK_LLM_API_KEY,
    FALLBACK_LLM_BASE_URL,
    FALLBACK_LLM_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    NVIDIA_RATE_LIMIT_COOLDOWN_SECONDS,
    ANSWER_TEMPERATURE,
)


_nvidia_rate_limited_until = 0.0


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    json_mode: bool = False,
    temperature: float | None = None,
) -> str | None:
    """Return text from the active provider, degrading cleanly on NVIDIA 429s."""
    global _nvidia_rate_limited_until
    if not LLM_API_KEY:
        return None

    primary_is_available = not (
        ACTIVE_LLM_PROVIDER == "nvidia" and time.monotonic() < _nvidia_rate_limited_until
    )
    if primary_is_available:
        response, rate_limited = _request_with_retry(
            LLM_API_KEY,
            LLM_BASE_URL,
            LLM_MODEL,
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            json_mode=json_mode,
            temperature=temperature,
        )
        if response:
            return response
        if rate_limited and ACTIVE_LLM_PROVIDER == "nvidia":
            _nvidia_rate_limited_until = time.monotonic() + NVIDIA_RATE_LIMIT_COOLDOWN_SECONDS
            print("  Warning: NVIDIA DeepSeek rate limited; using configured fallback provider")

    if _has_distinct_fallback():
        response, _ = _request_with_retry(
            FALLBACK_LLM_API_KEY,
            FALLBACK_LLM_BASE_URL,
            FALLBACK_LLM_MODEL,
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            json_mode=json_mode,
            temperature=temperature,
        )
        return response
    return None


def chat_json(system_prompt: str, user_prompt: str, *, max_tokens: int) -> dict[str, Any] | None:
    """Request JSON and validate that the model returned an object."""
    content = chat_completion(system_prompt, user_prompt, max_tokens=max_tokens, json_mode=True)
    if not content:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _request_with_retry(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    json_mode: bool,
    temperature: float | None,
) -> tuple[str | None, bool]:
    """Execute a bounded retry loop and return ``(content, was_rate_limited)``."""
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
            request: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": ANSWER_TEMPERATURE if temperature is None else temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content
            return (content.strip() if content else None), False
        except Exception as exc:
            rate_limited = getattr(exc, "status_code", None) == 429 or "429" in str(exc)
            if rate_limited and attempt < LLM_MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            print(f"  Warning: LLM request failed for {model}: {exc}")
            return None, rate_limited
    return None, False


def _has_distinct_fallback() -> bool:
    return bool(FALLBACK_LLM_API_KEY) and (
        FALLBACK_LLM_API_KEY != LLM_API_KEY
        or FALLBACK_LLM_BASE_URL != LLM_BASE_URL
        or FALLBACK_LLM_MODEL != LLM_MODEL
    )
