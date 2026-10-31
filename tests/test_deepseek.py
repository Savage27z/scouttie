"""DeepSeek client tests against a mocked transport -- no network, no spend."""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import make_account
from scout.config import Config, Secrets, TargetSpec, load_config
from scout.enrich import enrich_pending
from scout.funnel import deepseek as ds
from scout.funnel import llm
from scout.funnel.llm import LLMFatalError, LLMUnavailable
from scout.runner import Runner

TOOL = llm.CLASSIFY_TOOL
GOOD_ARGS = {"is_project": True, "confidence": 0.9, "category": "defi", "chain": "solana", "reason": "ok"}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ds.time, "sleep", lambda _s: None)


def ok_body(args, name="classify_account", finish="tool_calls"):
    return {"choices": [{"finish_reason": finish, "message": {"tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def make_client(handler, **kw) -> ds.DeepSeekClient:
    http = httpx.Client(base_url=ds.BASE_URL, headers={"Authorization": "Bearer k"},
                        transport=httpx.MockTransport(handler))
    return ds.DeepSeekClient("k", client=http, **kw)


def call(client):
    return client.call_tool(model="deepseek-flash", system="SYS", user="USR", tool=TOOL, max_tokens=300)


# ---------------------------------------------------------------- request shape


def test_request_disables_thinking_and_forces_the_named_tool():
    """DeepSeek returns HTTP 400 for a named tool_choice while thinking is on (its
    default). Getting this wrong fails every call, so pin it."""
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=ok_body(GOOD_ARGS))

    assert call(make_client(handler)) == GOOD_ARGS
    b = seen["body"]
    assert seen["path"] == "/chat/completions" and seen["auth"] == "Bearer k"
    assert b["thinking"] == {"type": "disabled"}
    assert b["tool_choice"] == {"type": "function", "function": {"name": "classify_account"}}
    assert b["tools"][0]["type"] == "function"
    assert b["tools"][0]["function"]["parameters"] == TOOL["input_schema"]
    assert "strict" not in b["tools"][0]["function"]  # strict mode is Beta; we validate ourselves
    assert b["model"] == "deepseek-flash" and b["temperature"] == 0 and b["stream"] is False
    assert [m["role"] for m in b["messages"]] == ["system", "user"]
    assert b["messages"][0]["content"] == "SYS" and b["messages"][1]["content"] == "USR"


# --------------------------------------------------------------- response parsing


def test_arguments_string_is_parsed_into_a_dict():
    assert call(make_client(lambda r: httpx.Response(200, json=ok_body({"a": 1})))) == {"a": 1}


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {},
        {"choices": [{"finish_reason": "stop", "message": {"content": "free text, no tool"}}]},
        ok_body(GOOD_ARGS, name="some_other_tool"),
        ok_body(GOOD_ARGS, finish="length"),  # truncated -> JSON may be cut off
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"function": {"name": "classify_account", "arguments": "{not json"}}]}}]},
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"function": {"name": "classify_account", "arguments": "[1,2]"}}]}}]},  # JSON but not an object
    ],
)
def test_unusable_responses_return_none_instead_of_raising(body):
    assert call(make_client(lambda r: httpx.Response(200, json=body))) is None


# ------------------------------------------------------------------- failures


@pytest.mark.parametrize("status,hint", [
    (401, "DEEPSEEK_API_KEY"), (402, "balance"), (403, "forbidden"),
    (404, "deepseek-flash"), (400, "rejected"), (422, "rejected"),
])
def test_unrecoverable_statuses_are_fatal_and_not_retried(status, hint):
    n = {"calls": 0}

    def handler(_r):
        n["calls"] += 1
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(LLMFatalError, match=hint):
        call(make_client(handler))
    assert n["calls"] == 1


def test_rate_limit_and_5xx_are_retried_then_succeed():
    n = {"calls": 0}

    def handler(_r):
        n["calls"] += 1
        if n["calls"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        if n["calls"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=ok_body(GOOD_ARGS))

    assert call(make_client(handler)) == GOOD_ARGS and n["calls"] == 3


def test_persistent_outage_raises_unavailable_not_fatal():
    with pytest.raises(LLMUnavailable):
        call(make_client(lambda r: httpx.Response(500), max_retries=2))


def test_network_error_is_retried():
    n = {"calls": 0}

    def handler(_r):
        n["calls"] += 1
        if n["calls"] < 2:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=ok_body(GOOD_ARGS))

    assert call(make_client(handler)) == GOOD_ARGS


def test_missing_key_rejected_at_construction():
    with pytest.raises(LLMFatalError):
        ds.DeepSeekClient("")


def test_api_key_never_appears_in_error_messages():
    h = lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})
    http = httpx.Client(base_url=ds.BASE_URL, transport=httpx.MockTransport(h))
    with pytest.raises(LLMFatalError) as ei:
        ds.DeepSeekClient("sk-SECRET123", client=http).call_tool(
            model="m", system="s", user="u", tool=TOOL)
    assert "sk-SECRET123" not in str(ei.value)


# ---------------------------------------------- fatal errors through the stages


class FatalClient:
    def call_tool(self, **_kw):
        raise LLMFatalError("DeepSeek HTTP 402: insufficient balance")


def test_classify_lets_fatal_errors_escape_but_swallows_ordinary_ones():
    with pytest.raises(LLMFatalError):
        llm.classify(FatalClient(), "m", make_account(1))

    class Flaky:
        def call_tool(self, **_kw):
            raise LLMUnavailable("503")

    assert llm.classify(Flaky(), "m", make_account(1)) is None


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return Config(secrets=Secrets(), targets=(TargetSpec("solana"),), filters=base.filters,
                  db_path=tmp_path / "x.db")


def test_enrich_propagates_fatal_and_leaves_candidates_queued(db, cfg):
    db.upsert_account(make_account(1, url="https://x.xyz", bio="b"))
    db.upsert_candidate("1", convergence_count=1, score=8.0)
    with pytest.raises(LLMFatalError):
        enrich_pending(db, cfg, FatalClient())
    c = db.get_candidate("1")
    assert c["state"] == "new" and c["llm_is_project"] is None  # not lost, not mislabelled


def test_runner_disables_llm_on_fatal_error_and_keeps_running(db, cfg, caplog):
    class NoProvider:
        def get_followers(self, *a, **k):
            raise AssertionError("no targets are due in this test")

    db.upsert_account(make_account(1, url="https://x.xyz", bio="b"))
    db.upsert_candidate("1", convergence_count=1, score=8.0)
    r = Runner(db, cfg, NoProvider(), llm_client=FatalClient(), expand=None)
    with caplog.at_level("ERROR"):
        r.cycle()  # must not raise
    assert r.llm_client is None  # disabled until restart
    assert "LLM disabled" in caplog.text and "insufficient balance" in caplog.text
    r.cycle()  # second cycle: no repeated API hammering, still alive
    assert db.get_candidate("1")["state"] == "new"
