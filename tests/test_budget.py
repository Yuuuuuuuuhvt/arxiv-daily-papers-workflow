from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal

import httpx
import pytest

from src import llm_client
from src.budget import BudgetExceeded, RunBudget, current_budget, parse_cost

MODEL = "vendor/model"
# Worst case per attempt with max_tokens=100: 1000*1/1e6 + 100*2/1e6 = 0.0012
POLICIES = {MODEL: {"context_length": 1000, "prompt_price_per_million": 1,
                    "completion_price_per_million": 2}}


def make_budget(tmp_path, limit, name="ledger.json"):
    return RunBudget(limit, tmp_path / name, POLICIES, max_output_tokens=100)


def test_concurrent_reserve_waits_then_proceeds_after_settle(tmp_path):
    async def scenario():
        budget = make_budget(tmp_path, "0.002")  # fits one attempt, not two
        first = await budget.reserve(MODEL, 100)
        second = asyncio.create_task(budget.reserve(MODEL, 100))
        await asyncio.sleep(0.05)
        assert not second.done()  # waits for the in-flight request
        budget.settle(first, Decimal("0.0001"))
        rid = await asyncio.wait_for(second, 1)
        budget.settle(rid, Decimal("0.0001"))
        snap = budget.snapshot()
        assert snap["spent_usd"] == pytest.approx(0.0002)
        assert snap["reserved_usd"] == 0
        json.dumps(snap)

        # An existing ledger must not be reusable by a restarted run, and it
        # is created owner-only.
        with pytest.raises(FileExistsError):
            make_budget(tmp_path, "0.002")
        assert (tmp_path / "ledger.json").stat().st_mode & 0o777 == 0o600

        # Invalid configuration and absurd costs are rejected, not accepted.
        for bad in ("nan", "inf", -1, 0, True):
            with pytest.raises(ValueError):
                make_budget(tmp_path, bad, "bad.json")
        for field, bad in (("context_length", 0), ("context_length", 1.5),
                           ("prompt_price_per_million", float("nan")),
                           ("completion_price_per_million", -1)):
            with pytest.raises(ValueError):
                RunBudget(1, tmp_path / "bad.json",
                          {MODEL: {**POLICIES[MODEL], field: bad}})
        with pytest.raises(ValueError):
            RunBudget(1, tmp_path / "bad.json", POLICIES, max_output_tokens=0)
        assert parse_cost("1e400") is None and parse_cost(1e30) is None
        assert parse_cost(True) is None and parse_cost(float("nan")) is None

    asyncio.run(scenario())


def test_unknown_cost_and_cancel_keep_reservation(tmp_path, monkeypatch):
    async def scenario():
        budget = make_budget(tmp_path, "0.0030")
        rid = await budget.reserve(MODEL, 100)
        budget.mark_uncertain(rid)
        assert budget.snapshot()["uncertain_usd"] == pytest.approx(0.0012)

        # A cancelled client call must also keep its reservation.
        token = current_budget.set(budget)
        try:
            task = asyncio.create_task(llm_client.call_llm(
                MODEL, [{"role": "user", "content": "x"}], max_tokens=100))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            current_budget.reset(token)
        snap = budget.snapshot()
        assert snap["uncertain_usd"] == pytest.approx(0.0024)
        assert snap["spent_usd"] == 0
        # 0.0006 left is below one worst case, so the next attempt is refused.
        with pytest.raises(BudgetExceeded):
            await budget.reserve(MODEL, 100)
        assert budget.snapshot()["stopped"]

        # A cost above the request's own reservation stops the budget at once,
        # raises, and wakes a waiter, although the total limit is not reached.
        b2 = make_budget(tmp_path, "1", "excess.json")
        first = await b2.reserve(MODEL, 100)
        b2.limit = Decimal("0.0024")  # room for exactly two reservations
        second = await b2.reserve(MODEL, 100)
        waiter = asyncio.create_task(b2.reserve(MODEL, 100))
        await asyncio.sleep(0.05)
        with pytest.raises(BudgetExceeded):
            b2.settle(first, Decimal("0.002"))
        with pytest.raises(BudgetExceeded):
            await asyncio.wait_for(waiter, 1)
        assert b2.snapshot()["stopped"]
        b2.mark_uncertain(second)

        # A ledger write failure during settlement stops the budget and raises.
        b3 = make_budget(tmp_path, "1", "writefail.json")
        rid = await b3.reserve(MODEL, 100)
        def fail(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(BudgetExceeded):
            b3.settle(rid, Decimal("0.0001"))
        monkeypatch.undo()
        with pytest.raises(BudgetExceeded):
            await b3.reserve(MODEL, 100)

    # The cancelled call blocks on the real network stack, so give it a
    # transport that never answers.
    class Hang(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(10)

    real = httpx.AsyncClient
    llm_client.httpx.AsyncClient = lambda **kw: real(transport=Hang(), **kw)
    try:
        asyncio.run(scenario())
    finally:
        llm_client.httpx.AsyncClient = real


def test_over_budget_rejected_before_sending(tmp_path):
    sent = []

    class Spy(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            sent.append(request)
            return httpx.Response(200, json={})

    async def scenario():
        budget = make_budget(tmp_path, "0.001")  # below one worst case
        token = current_budget.set(budget)
        try:
            with pytest.raises(BudgetExceeded):
                await llm_client.call_llm(MODEL, [{"role": "user", "content": "x"}])
            # Unpolicied models and paid plugins are refused before sending too.
            with pytest.raises(ValueError):
                await llm_client.call_llm("other/model", [{"role": "user", "content": "x"}])
            with pytest.raises(ValueError):
                await llm_client.call_llm(
                    MODEL, [{"role": "user", "content": "x"}],
                    plugins=[{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}])
        finally:
            current_budget.reset(token)

    real = httpx.AsyncClient
    llm_client.httpx.AsyncClient = lambda **kw: real(transport=Spy(), **kw)
    try:
        asyncio.run(scenario())
    finally:
        llm_client.httpx.AsyncClient = real
    assert sent == []


def test_client_payload_price_caps_and_retry_accounting(tmp_path, monkeypatch):
    requests: list[dict] = []
    mode = {"special": None}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if mode["special"] == "400":
            return httpx.Response(400, json={"error": "bad", "usage": {"cost": 0}})
        if mode["special"] == "length":
            return httpx.Response(200, json={
                "id": "gen-3", "usage": {"cost": 0.0001},
                "choices": [{"finish_reason": "length", "message": {"content": "cut"}}]})
        if len(requests) == 1:
            return httpx.Response(503, json={"usage": {"cost": 0.0005}, "id": "gen-1"})
        return httpx.Response(200, json={
            "id": "gen-2", "usage": {"cost": 0.0007, "prompt_tokens": 10,
                                     "completion_tokens": 5},
            "choices": [{"message": {"content": "ok"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm_client, "BACKOFF_SECONDS", [0, 0, 0])

    async def scenario():
        budget = make_budget(tmp_path, "1")
        token = current_budget.set(budget)
        try:
            plugins = [{"id": "file-parser", "pdf": {"engine": "native"}}]
            out = await llm_client.call_llm(
                MODEL, [{"role": "user", "content": "secret prompt"}], plugins=plugins)
        finally:
            current_budget.reset(token)
        return out, budget

    out, budget = asyncio.run(scenario())
    assert out == "ok"
    body = requests[0]
    assert body["max_tokens"] == 100  # defaults to max_output_tokens
    assert body["provider"] == {
        "require_parameters": True,
        "max_price": {"prompt": 1.0, "completion": 2.0, "request": 0}}
    snap = budget.snapshot()
    # Both attempts settle from usage.cost; the 503 attempt is not refunded.
    assert snap["spent_usd"] == pytest.approx(0.0012)
    assert [r["http_status"] for r in snap["requests"]] == [503, 200]
    assert snap["requests"][1]["response_id"] == "gen-2"
    assert "secret prompt" not in (tmp_path / "ledger.json").read_text()

    # A 4xx other than 429 is sent once; a truncated answer is rejected but
    # its cost is still settled and its finish_reason recorded.
    async def run(kind, name):
        mode["special"] = kind
        requests.clear()
        budget = make_budget(tmp_path, "1", name)
        token = current_budget.set(budget)
        try:
            with pytest.raises(RuntimeError):
                await llm_client.call_llm(MODEL, [{"role": "user", "content": "x"}])
        finally:
            current_budget.reset(token)
        return budget.snapshot()

    snap = asyncio.run(run("400", "four.json"))
    assert len(requests) == 1 and snap["requests"][0]["http_status"] == 400
    snap = asyncio.run(run("length", "length.json"))
    assert len(requests) == 1
    assert snap["requests"][0]["finish_reason"] == "length"
    assert snap["spent_usd"] == pytest.approx(0.0001)


def test_streaming_file_call_ledger_protection(tmp_path, monkeypatch):
    def sse(*events: dict | str) -> bytes:
        lines = [": OPENROUTER PROCESSING\n"]
        for event in events:
            lines.append(f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n")
        return "".join(lines).encode()

    def delta(text, finish=None, gen="gen-s"):
        return {"id": gen, "choices": [{"delta": {"content": text}, "finish_reason": finish}]}

    usage = {"id": "gen-s", "choices": [], "usage": {"cost": 0.0004, "prompt_tokens": 7,
                                                      "completion_tokens": 3}}

    class BrokenAfterFirstChunk(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse(delta("par"))
            raise httpx.ReadError("connection reset")

    bodies = {
        "ok": sse(delta("hel"), delta("lo", "stop"), usage, "[DONE]"),
        "error_event": sse(delta("x"), {"error": {"message": "secret upstream text"}},
                           usage, "[DONE]"),
        "no_usage": sse(delta("x", "stop"), "[DONE]"),
        "no_done": sse(delta("x", "stop"), usage),
    }
    mode = {"kind": "ok"}
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if mode["kind"] == "broken":
            return httpx.Response(200, stream=BrokenAfterFirstChunk())
        return httpx.Response(200, content=bodies[mode["kind"]])

    real = httpx.AsyncClient
    monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm_client, "BACKOFF_SECONDS", [0, 0, 0])
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "secret prompt"},
        {"type": "file", "file": {"filename": "p.pdf", "file_data": "data:;base64,AAAA"}}]}]
    plugins = [{"id": "file-parser", "pdf": {"engine": "native"}}]

    async def run(kind, name):
        mode["kind"] = kind
        requests.clear()
        budget = make_budget(tmp_path, "1", name)
        token = current_budget.set(budget)
        try:
            try:
                out = await llm_client.call_llm(MODEL, messages, plugins=plugins)
            except Exception as exc:
                out = exc
        finally:
            current_budget.reset(token)
        return out, budget.snapshot()

    async def annotate_contract():
        budget = make_budget(tmp_path, "1", "annotate.json")
        rid = await budget.reserve(MODEL, 100)
        before = budget.snapshot()
        budget.annotate(rid, response_id="g" * 500, finish_reason={"payload": "x"})
        after = budget.snapshot()
        assert after["reserved_usd"] == before["reserved_usd"]
        assert len(after["requests"][0]["response_id"]) == 128
        assert after["requests"][0]["finish_reason"] is None
        with pytest.raises(ValueError):
            budget.annotate(rid, messages="secret")
        # A failed ledger write must close every later request.
        monkeypatch.setattr(os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("full")))
        with pytest.raises(BudgetExceeded):
            budget.annotate(rid, response_id="gen-x")
        monkeypatch.undo()
        monkeypatch.setattr(llm_client.httpx, "AsyncClient",
                            lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        with pytest.raises(BudgetExceeded):
            await budget.reserve(MODEL, 100)

    # Normal stream: chunks are joined and the final usage.cost is settled.
    out, snap = asyncio.run(run("ok", "ok.json"))
    assert out == "hello"
    body = requests[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 100 and body["provider"]["max_price"]["request"] == 0
    assert snap["spent_usd"] == pytest.approx(0.0004) and snap["reserved_usd"] == 0
    assert snap["requests"][0]["response_id"] == "gen-s"
    assert snap["requests"][0]["finish_reason"] == "stop"

    # The id is recorded before the connection breaks; each retry reserves again
    # and stays uncertain, and the error text counts the real attempts.
    out, snap = asyncio.run(run("broken", "broken.json"))
    assert isinstance(out, RuntimeError) and "3 attempt(s)" in str(out)
    assert len(requests) == 3 and snap["spent_usd"] == 0
    assert [r["status"] for r in snap["requests"]] == ["uncertain"] * 3
    assert {r["response_id"] for r in snap["requests"]} == {"gen-s"}
    assert snap["uncertain_usd"] == pytest.approx(3 * 0.0012)

    # Error event, missing usage and missing [DONE] fail on the first attempt.
    # Only a [DONE]-terminated stream with a valid cost is settled.
    out, snap = asyncio.run(run("error_event", "event.json"))
    assert isinstance(out, RuntimeError) and "1 attempt(s)" in str(out)
    assert len(requests) == 1 and snap["spent_usd"] == pytest.approx(0.0004)
    out, snap = asyncio.run(run("no_usage", "nousage.json"))
    assert isinstance(out, RuntimeError) and len(requests) == 1
    assert snap["requests"][0]["status"] == "uncertain"
    out, snap = asyncio.run(run("no_done", "nodone.json"))
    assert isinstance(out, RuntimeError) and len(requests) == 1
    assert snap["requests"][0]["status"] == "uncertain" and snap["spent_usd"] == 0
    for name in ("broken", "event", "nousage", "nodone"):
        text = (tmp_path / f"{name}.json").read_text()
        assert "secret" not in text

    asyncio.run(annotate_contract())
