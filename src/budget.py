"""Per-run USD budget ledger for OpenRouter calls.

This is a local, single-process guard. It is not an OpenRouter account-level
limit, which would need a management key. Every attempt (including retries)
reserves its worst-case cost before it is sent, so the sum of known spend,
uncertain spend and in-flight reservations never exceeds the limit.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MILLION = Decimal(1_000_000)
# Round every money amount up to 1e-8 USD so rounding can only over-count.
_QUANTUM = Decimal("0.00000001")
# annotate() accepts only these ledger fields, each capped in length.
_ANNOTATION_KEYS = frozenset(
    {"http_status", "prompt_tokens", "completion_tokens", "response_id", "finish_reason"}
)
_MAX_ANNOTATION_CHARS = 128


class BudgetExceeded(RuntimeError):
    """Raised before sending a request that could push the run over budget."""


current_budget: contextvars.ContextVar[RunBudget | None] = contextvars.ContextVar(
    "current_budget", default=None
)


def _ceil(value: Decimal) -> Decimal:
    return value.quantize(_QUANTUM, rounding=ROUND_CEILING)


def parse_cost(value: Any) -> Decimal | None:
    """Return a valid non-negative cost, or None when it cannot be trusted."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        cost = Decimal(str(value))
        if not cost.is_finite() or cost < 0:
            return None
        return _ceil(cost)
    except (InvalidOperation, ValueError):
        # Absurdly large values overflow quantize; treat them as unknown cost.
        return None


def _decimal(name: str, value: Any) -> Decimal:
    """Parse a finite, non-negative configuration amount or raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{name} is not a number: {value!r}") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be finite and non-negative: {value!r}")
    return amount


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class RunBudget:
    def __init__(
        self,
        limit_usd: float | str | Decimal,
        ledger_path: str | Path,
        model_policies: dict[str, dict],
        max_output_tokens: int = 8192,
    ) -> None:
        self.limit = _decimal("limit_usd", limit_usd)
        if self.limit == 0:
            raise ValueError("limit_usd must be positive")
        self.max_output_tokens = _positive_int("max_output_tokens", max_output_tokens)
        self.policies = {}
        for model, p in model_policies.items():
            try:
                self.policies[model] = {
                    "context_length": _positive_int(
                        f"{model}.context_length", p["context_length"]),
                    "prompt_price_per_million": _decimal(
                        f"{model}.prompt_price_per_million", p["prompt_price_per_million"]),
                    "completion_price_per_million": _decimal(
                        f"{model}.completion_price_per_million",
                        p["completion_price_per_million"]),
                }
            except KeyError as exc:
                raise ValueError(f"policy for {model} lacks {exc}") from None
        self.ledger_path = Path(ledger_path)
        self.spent = Decimal(0)
        self.uncertain = Decimal(0)
        self.reserved = Decimal(0)
        self.stopped = False
        self.stop_reason: str | None = None
        self._ledger_failed = False
        self._requests: dict[int, dict] = {}
        self._open: dict[int, Decimal] = {}  # in-flight reservation amounts
        self._next_id = 1
        self._cond = asyncio.Condition()
        self._wake_tasks: set[asyncio.Task] = set()

        # O_EXCL makes creation atomic: an existing ledger means a previous run
        # already spent from this budget, and two processes cannot both
        # initialise the same path. Reusing a path would bypass the limit.
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.ledger_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            self._write(fd)
        except OSError:
            self.ledger_path.unlink(missing_ok=True)
            raise

    # ---- public API -------------------------------------------------------

    def policy(self, model: str) -> dict | None:
        return self.policies.get(model)

    def worst_case_cost(self, model: str, max_tokens: int) -> Decimal:
        p = self.policies[model]
        return _ceil(
            p["context_length"] * p["prompt_price_per_million"] / _MILLION
            + max_tokens * p["completion_price_per_million"] / _MILLION
        )

    def snapshot(self) -> dict:
        available = max(self.limit - self.spent - self.uncertain - self.reserved, Decimal(0))
        return {
            "limit_usd": float(self.limit),
            "spent_usd": float(self.spent),
            "uncertain_usd": float(self.uncertain),
            "reserved_usd": float(self.reserved),
            "available_usd": float(available),
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "requests": [dict(r) for r in self._requests.values()],
        }

    async def reserve(self, model: str, max_tokens: int, attempt: int = 1) -> int:
        """Reserve the worst-case cost of one attempt, waiting for in-flight
        requests to settle when the remaining headroom is too small."""
        worst = self.worst_case_cost(model, max_tokens)
        async with self._cond:
            while True:
                if self.stopped:
                    raise BudgetExceeded(self.stop_reason or "budget stopped")
                if self._ledger_failed:
                    self._stop("ledger write failed")
                    raise BudgetExceeded(self.stop_reason)
                if self.spent + self.uncertain + self.reserved + worst <= self.limit:
                    break
                if not self._open:
                    # Nothing in flight can free headroom any more.
                    self._stop(
                        f"worst-case cost {worst} USD does not fit the remaining budget"
                    )
                    self._cond.notify_all()
                    raise BudgetExceeded(self.stop_reason)
                await self._cond.wait()

            rid = self._next_id
            self._next_id += 1
            self._open[rid] = worst
            self.reserved += worst
            self._requests[rid] = {
                "id": rid,
                "model": model,
                "attempt": attempt,
                "max_tokens": max_tokens,
                "reserved_usd": float(worst),
                "status": "reserved",
                "http_status": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "response_id": None,
                "cost_usd": None,
                "finish_reason": None,
            }
            if not self._persist():
                self._open.pop(rid)
                self.reserved -= worst
                del self._requests[rid]
                self._stop("ledger write failed")
                self._cond.notify_all()
                raise BudgetExceeded(self.stop_reason)
            return rid

    # settle/mark_uncertain are synchronous so that they still complete in a
    # cancellation handler; they never await between reading and writing the
    # counters, which makes them atomic on the event loop.
    def settle(self, rid: int, cost: Decimal, **meta: Any) -> None:
        """Record the real cost. Raises BudgetExceeded, after recording, when
        the cost passes its own reservation, the total limit, or the ledger
        cannot be written."""
        amount = self._open.pop(rid, None)
        if amount is None:
            return
        self.reserved -= amount
        self.spent += cost
        self._requests[rid].update(meta, status="settled", cost_usd=float(cost))
        # A cost above the request's own reservation means the price caps did
        # not hold, so no later estimate can be trusted either.
        if cost > amount:
            self._stop(f"request {rid} cost {cost} USD exceeded its reservation {amount} USD")
        elif self.spent + self.uncertain > self.limit:
            self._stop("actual cost exceeded the budget")
        persisted = self._persist()
        if not persisted:
            self._stop("ledger write failed")
        self._wake()
        if self.stopped:
            raise BudgetExceeded(self.stop_reason)

    def annotate(self, rid: int, **meta: Any) -> None:
        """Save bounded metadata, such as an early response_id, for an open
        request. Reserved, spent and uncertain amounts are not touched, so a
        later settle or mark_uncertain keeps working on the same reservation.

        Only whitelisted keys with short strings or integers are stored; the
        ledger must never receive payloads, error bodies or reasoning text.
        Raises BudgetExceeded when the ledger cannot be written, which also
        refuses every later reserve()."""
        if rid not in self._open:
            return
        unknown = set(meta) - _ANNOTATION_KEYS
        if unknown:
            raise ValueError(f"unsupported annotation keys: {sorted(unknown)}")
        for key, value in meta.items():
            if isinstance(value, str):
                value = value[:_MAX_ANNOTATION_CHARS]
            elif isinstance(value, bool) or not isinstance(value, int):
                value = None
            self._requests[rid][key] = value
        if not self._persist():
            self._stop("ledger write failed")
            raise BudgetExceeded(self.stop_reason)

    def mark_uncertain(self, rid: int, **meta: Any) -> None:
        """Keep the full reservation as spend because the real cost is unknown."""
        amount = self._open.pop(rid, None)
        if amount is None:
            return
        self.reserved -= amount
        self.uncertain += amount
        self._requests[rid].update(meta, status="uncertain")
        if not self._persist():
            self._stop("ledger write failed")
        self._wake()
        if self.stopped and self.stop_reason == "ledger write failed":
            raise BudgetExceeded(self.stop_reason)

    # ---- internals --------------------------------------------------------

    def _stop(self, reason: str) -> None:
        if not self.stopped:
            self.stopped = True
            self.stop_reason = reason
            logger.error("Budget stopped: %s", reason)
            self._persist()

    def _persist(self) -> bool:
        """Atomically rewrite the ledger. Returns False when writing fails."""
        tmp = self.ledger_path.with_name(self.ledger_path.name + ".tmp")
        try:
            self._write(os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600))
            os.replace(tmp, self.ledger_path)
            return True
        except OSError as exc:
            self._ledger_failed = True
            logger.error("Budget ledger write failed: %s", exc)
            return False

    def _write(self, fd: int) -> None:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _wake(self) -> None:
        async def notify() -> None:
            async with self._cond:
                self._cond.notify_all()

        task = asyncio.get_running_loop().create_task(notify())
        self._wake_tasks.add(task)
        task.add_done_callback(self._wake_tasks.discard)
