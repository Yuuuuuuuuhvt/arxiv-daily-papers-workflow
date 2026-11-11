"""Model relays that speak OpenAI Responses or Anthropic Messages.

A relay reports token usage but no per-request cost. These calls therefore
cannot run under the USD run budget in budget.py, and OpenRouter's
file-parser plugins do not exist here: papers reach a relay as text and
inline images only.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from .config import PROVIDER_ENV, get_env

# Anthropic Messages requires max_tokens, and thinking models spend it before
# they write the answer, so a small value returns empty content.
DEFAULT_MAX_OUTPUT_TOKENS = 32768


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Both protocols take the system prompt as a top-level field."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _parts(content: str | list[dict]) -> list[dict]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def _responses_request(model: str, messages: list[dict], temperature: float,
                       max_tokens: int | None) -> tuple[str, dict]:
    def convert(part: dict) -> dict:
        if part["type"] == "text":
            return {"type": "input_text", "text": part["text"]}
        if part["type"] == "image_url":
            image = part["image_url"]
            return {"type": "input_image", "image_url": image["url"],
                    "detail": image.get("detail", "auto")}
        raise ValueError(f"relay providers do not accept {part['type']} content")

    system, turns = _split_system(messages)
    # temperature is not sent: this relay rejects it with
    # 400 "Unsupported parameter: temperature". input must be a message list.
    payload: dict = {
        "model": model, "stream": True, "store": False, "instructions": system,
        "input": [{"role": m["role"], "content": [convert(p) for p in _parts(m["content"])]}
                  for m in turns],
    }
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    return "/responses", payload


def _read_responses_event(event: dict) -> tuple[str, bool]:
    kind = event.get("type")
    if kind == "response.output_text.delta":
        return event.get("delta", ""), False
    if kind == "response.completed":
        return "", True
    if kind == "response.incomplete":
        raise ValueError("Response truncated (response.incomplete)")
    if kind in ("response.failed", "error"):
        raise ValueError(f"relay stream reported {kind}")
    return "", False


def _messages_request(model: str, messages: list[dict], temperature: float,
                      max_tokens: int | None) -> tuple[str, dict]:
    def convert(part: dict) -> dict:
        if part["type"] == "text":
            return {"type": "text", "text": part["text"]}
        if part["type"] == "image_url":
            # Callers send images as data URLs: data:<media type>;base64,<data>.
            header, data = part["image_url"]["url"].split(",", 1)
            media_type = header.removeprefix("data:").removesuffix(";base64")
            return {"type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data}}
        raise ValueError(f"relay providers do not accept {part['type']} content")

    system, turns = _split_system(messages)
    return "/messages", {
        "model": model, "stream": True, "system": system, "temperature": temperature,
        "max_tokens": max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        "messages": [{"role": m["role"], "content": [convert(p) for p in _parts(m["content"])]}
                     for m in turns],
    }


def _read_messages_event(event: dict) -> tuple[str, bool]:
    kind = event.get("type")
    if kind == "content_block_delta":
        delta = event.get("delta", {})
        # Thinking deltas are not part of the answer.
        return (delta.get("text", "") if delta.get("type") == "text_delta" else ""), False
    if kind == "message_delta":
        reason = event.get("delta", {}).get("stop_reason")
        if reason == "max_tokens":
            raise ValueError("Response truncated (stop_reason=max_tokens)")
        return "", reason in ("end_turn", "stop_sequence")
    if kind == "error":
        raise ValueError("relay stream reported error")
    return "", False


Builder = Callable[[str, list[dict], float, "int | None"], tuple[str, dict]]
Reader = Callable[[dict], tuple[str, bool]]
_PROTOCOLS: dict[str, tuple[Builder, Reader]] = {
    "codex-relay": (_responses_request, _read_responses_event),
    "gemini-relay": (_messages_request, _read_messages_event),
}


async def relay_completion(
    client: httpx.AsyncClient,
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    api_key: str,
) -> str:
    """Send one streamed request and return the complete answer text.

    Streaming is mandatory on the Codex relay and keeps long analyses inside
    the relay's idle timeout. 429 and 5xx raise httpx.HTTPStatusError for the
    caller's retry loop; every other failure is a ValueError because resending
    the same request would fail the same way.
    """
    build, read = _PROTOCOLS[provider]
    path, payload = build(model, messages, temperature, max_tokens)
    url = get_env(PROVIDER_ENV[provider][1]).rstrip("/") + path
    # Both relays accept a bearer key; anthropic-version is ignored by Responses.
    headers = {"Authorization": f"Bearer {api_key}", "anthropic-version": "2023-06-01"}
    parts: list[str] = []
    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            await resp.aread()
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"status {resp.status_code}", request=resp.request, response=resp
                )
            # The body is deliberately not logged or embedded here.
            raise ValueError(f"relay rejected the request with status {resp.status_code}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue  # event names, blank separators and keep-alive comments
            event = json.loads(line[5:])
            if not isinstance(event, dict):
                continue
            text, finished = read(event)
            parts.append(text)
            if finished:
                content = "".join(parts)
                if not content:
                    raise ValueError("Empty content in response")
                return content
    raise ValueError("relay stream ended before the response completed")
