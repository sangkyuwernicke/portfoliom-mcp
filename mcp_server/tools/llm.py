"""LLM access (OpenRouter, OpenAI-compatible API).

Every LLM call in the server goes through :func:`call_llm` (free text) or
:func:`call_llm_structured` (Pydantic-validated JSON). Retries on 429 / 5xx /
connection errors are handled by the OpenAI SDK (``LLM_MAX_RETRIES``); the
local circuit breaker stops hammering the API after repeated failures.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from mcp_server.tools.resilience import CircuitOpenError, Timeout, circuit_llm

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-6-luna")
LLM_MAX_OUTPUT_TOKENS_DEFAULT = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "16000"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Optional reasoning effort override (low|medium|high); unset uses the model default.
LLM_EFFORT = os.getenv("LLM_EFFORT", "").strip() or None

ModelT = TypeVar("ModelT", bound=BaseModel)

_client: OpenAI | None = None
_client_lock = threading.Lock()


class LLMRefusalError(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


def get_client() -> OpenAI:
    """Shared client. Credentials resolve from OPENROUTER_API_KEY."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise openai.OpenAIError("OPENROUTER_API_KEY is not set")
            _client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                timeout=Timeout.LLM,
                max_retries=LLM_MAX_RETRIES,
            )
        return _client


def _request_kwargs(system: str, user: str, model: str | None, max_tokens: int | None) -> dict:
    kwargs: dict = {
        "model": model or OPENROUTER_MODEL,
        "max_tokens": max_tokens or LLM_MAX_OUTPUT_TOKENS_DEFAULT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if LLM_EFFORT:
        kwargs["extra_body"] = {"reasoning": {"effort": LLM_EFFORT}}
    return kwargs


def _check_stop(response) -> None:
    choice = response.choices[0]
    refusal = getattr(choice.message, "refusal", None)
    if refusal or choice.finish_reason == "content_filter":
        raise LLMRefusalError(f"model declined the request ({refusal or 'content_filter'})")
    if choice.finish_reason == "length":
        logger.warning(
            "LLM response truncated by max_tokens (model=%s). Raise LLM_MAX_OUTPUT_TOKENS.", response.model
        )


def call_llm(system: str, user: str, *, model: str | None = None, max_tokens: int | None = None) -> str:
    """Single-turn text completion. Returns the message text."""

    def _do_request():
        return get_client().chat.completions.create(**_request_kwargs(system, user, model, max_tokens))

    response = circuit_llm.call(_do_request)
    _check_stop(response)
    return (response.choices[0].message.content or "").strip()


def call_llm_structured(
    system: str,
    user: str,
    output_format: type[ModelT],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> ModelT:
    """Single-turn completion constrained to ``output_format`` (validated Pydantic instance)."""

    def _do_request():
        return get_client().beta.chat.completions.parse(
            **_request_kwargs(system, user, model, max_tokens), response_format=output_format
        )

    response = circuit_llm.call(_do_request)
    _check_stop(response)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output")
    return parsed


def summarize_text(text: str, max_sentences: int = 6, model: str | None = None) -> str:
    """텍스트 요약. 실패하면 빈 문자열을 반환해 리포트 생성은 계속 진행됩니다."""
    if not text or not text.strip():
        return ""

    system = (
        f"You are a concise financial analyst. Summarize in {max_sentences} sentences (bullet-ready). "
        "Focus on drivers, risks, guidance, and near-term catalysts."
    )
    try:
        return call_llm(system, text[:8000], model=model, max_tokens=2048)
    except CircuitOpenError:
        logger.warning("LLM circuit open, skipping summarization")
    except LLMRefusalError as e:
        logger.warning("LLM summarization refused: %s", e)
    except openai.OpenAIError as e:  # API errors and missing credentials
        logger.warning("LLM summarization failed: %s", e)
    return ""


def summarize_items(lines: List[str], max_sentences: int = 6) -> str:
    """리스트 항목 요약."""
    text = "\n".join(f"- {ln}" for ln in lines if ln)
    return summarize_text(text, max_sentences=max_sentences)
