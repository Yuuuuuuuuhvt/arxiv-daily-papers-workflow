from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging

import httpx

from .budget import BudgetExceeded, RunBudget, current_budget, parse_cost

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 3
BACKOFF_SECONDS = [2, 4, 8]

# native bills input tokens, cloudflare-ai is free, pdf-text is its
# compatibility alias. Any other engine (e.g. mistral-ocr) is billed per page.
FREE_OR_TOKEN_PDF_ENGINES = {"native", "cloudflare-ai", "pdf-text"}
# Content types whose cost is covered by the input-token reservation.
BILLED_AS_TOKENS_CONTENT = {"text", "file", "image_url"}


def _validate_budgeted_request(
    budget: RunBudget,
    model: str,
    messages: list[dict],
    max_tokens: int | None,
    plugins: list[dict] | None,
) -> int:
    """Reject requests whose cost the ledger cannot bound; return max_tokens."""
    if budget.policy(model) is None:
        raise ValueError(f"model {model} has no budget policy")
    if max_tokens is None:
        max_tokens = budget.max_output_tokens
    elif not 0 < max_tokens <= budget.max_output_tokens:
        raise ValueError(
            f"max_tokens {max_tokens} outside 1..{budget.max_output_tokens}"
        )
    for plugin in plugins or []:
        engine = (plugin.get("pdf") or {}).get("engine")
        if plugin.get("id") != "file-parser" or engine not in FREE_OR_TOKEN_PDF_ENGINES:
            raise ValueError("only file-parser with native/cloudflare-ai/pdf-text is allowed")
    has_file = False
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) or content is None:
            continue
        for part in content:
            kind = part.get("type")
            if kind not in BILLED_AS_TOKENS_CONTENT:
                raise ValueError(f"unbudgeted content type: {kind}")
            has_file = has_file or kind == "file"
            if kind == "image_url":
                if not budget.policy(model)["supports_images"]:
                    raise ValueError("model has no verified image budget policy")
                url = (part.get("image_url") or {}).get("url", "")
                if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                    raise ValueError("budgeted images must be inline PNG or JPEG")
                try:
                    if not base64.b64decode(url.split(",", 1)[1], validate=True):
                        raise ValueError("empty image")
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("invalid inline image") from exc
    if has_file and not plugins:
        # Without an explicit engine OpenRouter may fall back to paid OCR.
        raise ValueError("file content requires an explicit file-parser engine")
    return max_tokens


def _finish_reason(data: dict) -> str | None:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


def _account_response(budget: RunBudget, rid: int, resp: httpx.Response) -> str | None:
    """Settle from usage.cost when valid, otherwise keep the reservation."""
    try:
        data = resp.json()
    except ValueError:
        data = None
    data = data if isinstance(data, dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    meta = {
        "http_status": resp.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response_id": data.get("id") if isinstance(data.get("id"), str) else None,
        "finish_reason": _finish_reason(data),
    }
    cost = parse_cost(usage.get("cost"))
    if cost is None:
        budget.mark_uncertain(rid, **meta)
    else:
        budget.settle(rid, cost, **meta)
    return meta["finish_reason"]


class StreamFailure(RuntimeError):
    """The SSE stream ended in a state that cannot be trusted as a full answer.

    It is deliberately not a transport error, so it is never retried."""


def _has_file_content(messages: list[dict]) -> bool:
    return any(
        part.get("type") == "file"
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 429 or resp.status_code >= 500:
        # The body is deliberately not logged or embedded here.
        raise httpx.HTTPStatusError(
            f"status {resp.status_code}", request=resp.request, response=resp
        )
    resp.raise_for_status()


async def _stream_completion(
    client: httpx.AsyncClient,
    budget: RunBudget | None,
    rid: int | None,
    headers: dict,
    payload: dict,
) -> str:
    """Read one SSE chat completion and return its aggregated content.

    Streaming exists because a long non-streaming PDF request can die with a
    ReadError before any response arrives, which loses the generation id and
    leaves the full worst-case reservation frozen. Here the id reaches the
    ledger with the first chunk, so a broken stream can still be audited.

    The reservation is settled only after the [DONE] event, because usage.cost
    arrives in the last chunks and an earlier value could undercount. Any
    exception before that point leaves the reservation to the caller's
    mark_uncertain. A stream that ends badly but whose [DONE] arrived with a
    valid usage.cost is still settled, then raises."""
    async with client.stream("POST", OPENROUTER_BASE, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            await resp.aread()
            if rid is not None:
                _account_response(budget, rid, resp)
            _raise_for_status(resp)
            raise StreamFailure(f"unexpected status {resp.status_code}")

        parts: list[str] = []
        usage: dict = {}
        generation_id: str | None = None
        finish_reason: str | None = None
        done = error_event = malformed = False
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue  # blank separators and ": keep-alive" comments
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                chunk = None
            if not isinstance(chunk, dict):
                malformed = True
                continue
            if "error" in chunk:
                error_event = True  # keep reading: usage may still follow
            chunk_id = chunk.get("id")
            if generation_id is None and isinstance(chunk_id, str):
                generation_id = chunk_id
                if rid is not None:
                    budget.annotate(rid, response_id=generation_id)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    parts.append(delta["content"])
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]

    if rid is not None:
        meta = {
            "http_status": 200,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "response_id": generation_id,
            "finish_reason": finish_reason,
        }
        cost = parse_cost(usage.get("cost")) if done else None
        if cost is None:
            budget.mark_uncertain(rid, **meta)
        else:
            budget.settle(rid, cost, **meta)

    if not done:
        raise StreamFailure("stream ended without [DONE]")
    if error_event:
        raise StreamFailure("stream carried an error event")
    if malformed:
        raise StreamFailure("stream carried an unparsable event")
    if not usage:
        raise StreamFailure("stream carried no final usage")
    if finish_reason is None or finish_reason == "error":
        raise StreamFailure(f"stream finish_reason is {finish_reason}")
    if finish_reason == "length":
        raise ValueError("Response truncated (finish_reason=length)")
    content = "".join(parts)
    if not content:
        raise ValueError("Empty content in response")
    return content


async def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int | None = None,
    api_key: str = "",
    plugins: list[dict] | None = None,
    stream: bool = False,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/arxiv-daily-agent",
        "X-Title": "arXiv Daily Papers Agent",
    }
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # Complete documents must fail on context overflow, not lose middle sections.
        "transforms": [],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if plugins is not None:
        payload["plugins"] = plugins

    stream = stream or _has_file_content(messages)
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    budget = current_budget.get()
    image_count = sum(
        part.get("type") == "image_url"
        for message in messages if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    if budget is not None:
        payload["max_tokens"] = _validate_budgeted_request(
            budget, model, messages, max_tokens, plugins
        )
        policy = budget.policy(model)
        payload["provider"] = {
            "require_parameters": True,
            "max_price": {
                "prompt": float(policy["prompt_price_per_million"]),
                "completion": float(policy["completion_price_per_million"]),
                "request": 0,
            },
        }
        if image_count:
            payload["provider"]["max_price"]["image"] = float(policy["image_price_per_image"])

    last_exc: Exception | None = None
    attempts = 0
    for attempt in range(MAX_RETRIES):
        attempts = attempt + 1
        # Every attempt, retries included, needs its own reservation. The
        # reservation sits outside the try so BudgetExceeded is never retried.
        rid = (
            await budget.reserve(model, payload["max_tokens"], attempt + 1, image_count=image_count)
            if budget is not None
            else None
        )
        try:
            async with httpx.AsyncClient(timeout=480) as client:
                if stream:
                    return await _stream_completion(client, budget, rid, headers, payload)
                resp = await client.post(
                    OPENROUTER_BASE, headers=headers, json=payload
                )
                finish_reason = None
                if rid is not None:
                    finish_reason = _account_response(budget, rid, resp)
                _raise_for_status(resp)
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("Empty content in response")
                if finish_reason == "length":
                    # Truncated output is unusable and a retry would only
                    # spend more of the budget on the same limit.
                    raise ValueError("Response truncated (finish_reason=length)")
                return content
        except BudgetExceeded:
            raise
        except Exception as exc:
            last_exc = exc
            # TransportError covers ReadError, which the old tuple missed.
            # StreamFailure and ValueError are deliberately not retryable.
            retryable = isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError))
            if (
                budget is not None
                and isinstance(exc, httpx.HTTPStatusError)
                and 400 <= exc.response.status_code < 500
                and exc.response.status_code != 429
            ):
                # Client errors repeat identically; retrying only burns a
                # conservative reservation.
                retryable = False
            if retryable and attempt < MAX_RETRIES - 1:
                wait = BACKOFF_SECONDS[attempt]
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %ds: %s: %s",
                    attempt + 1, MAX_RETRIES, wait,
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "LLM call failed after %d attempt(s), not retrying: %s: %s",
                    attempts, type(exc).__name__, exc,
                )
                break
        finally:
            # Transport errors, timeouts and cancellation reach here with the
            # reservation still open; keep it all as uncertain spend. This is a
            # no-op once the response was settled.
            if rid is not None:
                try:
                    budget.mark_uncertain(rid)
                except BudgetExceeded:
                    # The budget is already stopped, so the next reserve()
                    # raises; do not mask the exception already in flight.
                    pass

    raise RuntimeError(f"LLM call failed after {attempts} attempt(s)") from last_exc
