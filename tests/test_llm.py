"""Offline tests for the OpenRouter LLM wrapper (no network access)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_server.tools import llm


def _message(*texts: str, finish_reason: str = "stop", refusal: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="".join(texts), refusal=refusal, parsed=None),
                finish_reason=finish_reason,
            )
        ],
        model="openai/gpt-6-luna",
    )


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture
def fake_client(monkeypatch):
    def install(response):
        completions = FakeCompletions(response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        )
        monkeypatch.setattr(llm, "_client", client)
        llm.circuit_llm.reset()
        return completions

    return install


def test_call_llm_uses_openrouter_model(fake_client):
    completions = fake_client(_message("Hello ", "world"))
    assert llm.call_llm("sys", "hi") == "Hello world"
    sent = completions.calls[0]
    assert sent["model"] == "openai/gpt-6-luna"
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert "temperature" not in sent


def test_refusal_raises(fake_client):
    fake_client(_message("", refusal="cyber"))
    with pytest.raises(llm.LLMRefusalError, match="cyber"):
        llm.call_llm("sys", "hi")


def test_summarize_swallows_refusal(fake_client):
    fake_client(_message("", refusal="cyber"))
    assert llm.summarize_items(["a", "b"]) == ""


def test_summarize_skips_empty_input(fake_client):
    messages = fake_client(_message("unused"))
    assert llm.summarize_text("   ") == ""
    assert messages.calls == []


def test_structured_output(fake_client):
    from mcp_server.tools.news_sentiment import _call_llm_sentiment, _LLMSentiment

    parsed = _LLMSentiment(overall_sentiment="bullish", sentiment_score=1.7, key_themes=["AI"],
                           summary="Strong demand.", confidence=0.8)
    response = _message("{}")
    response.choices[0].message.parsed = parsed
    messages = fake_client(response)

    out = _call_llm_sentiment([{"title": "Chip demand surges", "snippet": "..."}])
    assert out["overall_sentiment"] == "bullish"
    assert out["sentiment_score"] == 1.0  # clamped
    assert messages.calls[0]["response_format"] is _LLMSentiment
