from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine

import httpx
import pytest

from src import config, llm_client

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"}}
MESSAGES = [{"role": "system", "content": "Rules"},
            {"role": "user", "content": [{"type": "text", "text": "Read"}, IMAGE]}]
REAL_CLIENT = httpx.AsyncClient


def _sse(*events: dict) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _call(monkeypatch, provider: str, replies: list[httpx.Response]) -> tuple[list[httpx.Request], Coroutine]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return replies[len(requests) - 1]

    monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm_client, "BACKOFF_SECONDS", [0, 0, 0])
    monkeypatch.setenv("CODEX_RELAY_BASE_URL", "https://codex.test/v1/")
    monkeypatch.setenv("GEMINI_RELAY_BASE_URL", "https://gemini.test/v1")
    return requests, llm_client.call_llm("m", MESSAGES, temperature=0.2, api_key="k", provider=provider)


def test_codex_relay_sends_responses_request_and_retries_server_errors(monkeypatch):
    stream = _sse({"type": "response.output_text.delta", "delta": "an"},
                  {"type": "response.output_text.delta", "delta": "swer"},
                  {"type": "response.completed", "response": {"status": "completed"}})
    requests, call = _call(monkeypatch, "codex-relay", [httpx.Response(502), httpx.Response(200, text=stream)])
    assert asyncio.run(call) == "answer"
    assert [str(r.url) for r in requests] == ["https://codex.test/v1/responses"] * 2
    assert requests[1].headers["authorization"] == "Bearer k"
    body = json.loads(requests[1].content)
    # This relay rejects temperature and a string input.
    assert "temperature" not in body and body["stream"] is True and body["store"] is False
    assert body["instructions"] == "Rules"
    assert body["input"] == [{"role": "user", "content": [
        {"type": "input_text", "text": "Read"},
        {"type": "input_image", "image_url": "data:image/png;base64,QUJD", "detail": "high"}]}]


def test_gemini_relay_sends_messages_request_and_rejects_truncation(monkeypatch):
    stream = _sse({"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hidden"}},
                  {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "answer"}},
                  {"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
    requests, call = _call(monkeypatch, "gemini-relay", [httpx.Response(200, text=stream)])
    assert asyncio.run(call) == "answer"
    assert str(requests[0].url) == "https://gemini.test/v1/messages"
    body = json.loads(requests[0].content)
    assert body["system"] == "Rules" and body["temperature"] == 0.2 and body["max_tokens"] >= 1024
    assert body["messages"][0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}

    # Truncated, rejected and unfinished answers are never retried or returned.
    for reply in (httpx.Response(200, text=_sse({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}})),
                  httpx.Response(200, text=_sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "half"}})),
                  httpx.Response(401)):
        requests, call = _call(monkeypatch, "gemini-relay", [reply])
        with pytest.raises(ValueError):
            asyncio.run(call)
        assert len(requests) == 1


def test_stage_model_override_selects_provider_and_requires_its_credentials(monkeypatch):
    monkeypatch.setenv("PAPERS_DEEP_ANALYSIS_MODEL", "codex-relay:gpt-test")
    models = config.load_config()["models"]
    assert (models["deep_analysis"]["provider"], models["deep_analysis"]["model_id"]) == ("codex-relay", "gpt-test")
    assert models["relevance_filter"]["provider"] == "openrouter"

    monkeypatch.setenv("CODEX_RELAY_API_KEY", "k")
    monkeypatch.delenv("CODEX_RELAY_BASE_URL", raising=False)
    with pytest.raises(EnvironmentError):
        config.provider_api_key("codex-relay")
    monkeypatch.setenv("PAPERS_DEEP_ANALYSIS_MODEL", "unknown:gpt-test")
    with pytest.raises(ValueError):
        config.load_config()
