from __future__ import annotations

import asyncio
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
BILLED_AS_TOKENS_CONTENT = {"text", "file"}


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


async def call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int | None = None,
    api_key: str = "",
    plugins: list[dict] | None = None,
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
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if plugins is not None:
        payload["plugins"] = plugins

    budget = current_budget.get()
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

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        # Every attempt, retries included, needs its own reservation. The
        # reservation sits outside the try so BudgetExceeded is never retried.
        rid = (
            await budget.reserve(model, payload["max_tokens"], attempt + 1)
            if budget is not None
            else None
        )
        try:
            async with httpx.AsyncClient(timeout=480) as client:
                resp = await client.post(
                    OPENROUTER_BASE, headers=headers, json=payload
                )
                finish_reason = None
                if rid is not None:
                    finish_reason = _account_response(budget, rid, resp)
                if resp.status_code == 429 or resp.status_code >= 500:
                    # The body is deliberately not logged or embedded here.
                    raise httpx.HTTPStatusError(
                        f"status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
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
            retryable = isinstance(exc, (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError))
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
            elif attempt < MAX_RETRIES - 1 and not retryable:
                logger.error("LLM call non-retryable error: %s: %s", type(exc).__name__, exc)
                break
            else:
                logger.error("LLM call failed after %d attempts: %s: %s", MAX_RETRIES, type(exc).__name__, exc)
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

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries") from last_exc
