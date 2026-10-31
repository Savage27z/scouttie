from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from conftest import make_account
from scout import routing
from scout.alerts import telegram as tgm
from scout.config import Config, Secrets, load_config


class FakeTG(tgm.Telegram):
    """Telegram with the network replaced by an in-memory log."""

    def __init__(self, chat_id="42"):
        self.chat_ids = [int(p) for p in str(chat_id).split(",")]
        self.chat_id = self.chat_ids[0]
        self.sent = []
        self.answers = []
        self.cleared = []
        self._n = 0

    def send(self, text, *, account_id=None):
        self._n += 1
        self.sent.append((text, account_id))
        return self._n

    def answer_callback(self, callback_id, text):
        self.answers.append((callback_id, text))

    def clear_buttons(self, message_id, chat_id=None):
        self.cleared.append(message_id)
        self.cleared_in = chat_id


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return Config(secrets=Secrets(), targets=base.targets, filters=base.filters, db_path=tmp_path / "x.db")


def ready(db, i, conv=1, conf=0.7, **acct):
    db.upsert_account(make_account(i, name=f"Proj {i}", bio="b", url="https://p.xyz", **acct))
    db.upsert_candidate(str(i), convergence_count=conv, score=8.0, state="ready",
                        llm_is_project=1, llm_confidence=conf, llm_category="defi",
                        llm_chain="base", llm_summary="A DEX.")


# ------------------------------------------------------------ formatting


def test_stranger_text_is_html_escaped():
    evil = {"username": "x", "name": "<b>Pwn</b>", "bio": "<a href='http://evil'>click</a>",
            "llm_summary": "</i><script>alert(1)</script>", "url": "https://a.b/?q=<x>",
            "followers_count": 1, "created_at": "2026-01-01", "score": 8}
    text = tgm.format_alert(evil, kind="instant")
    assert "<script>" not in text and "<a href='http://evil'>" not in text
    assert "&lt;script&gt;" in text and "&lt;b&gt;Pwn&lt;/b&gt;" in text


def test_message_never_exceeds_telegram_limit():
    big = {"username": "x", "name": "n", "llm_summary": "y" * 20000, "followers_count": 1, "score": 1}
    assert len(tgm.format_alert(big)) <= 4096


# ------------------------------------------------------------- callbacks


def cb(data, chat=42, mid=7):
    return {"id": "cbid", "data": data, "message": {"chat": {"id": chat}, "message_id": mid}}


@pytest.mark.parametrize("data,state", [("i:5", "interested"), ("n:5", "not_project"), ("m:5", "muted")])
def test_button_presses_write_state(db, data, state):
    ready(db, 5)
    tg = FakeTG()
    assert tgm.handle_callback(db, tg, cb(data)) == state
    assert db.get_candidate("5")["state"] == state
    assert tg.cleared == [7]  # buttons removed so it can't be double-pressed


def test_callback_from_foreign_chat_is_ignored(db):
    ready(db, 5)
    tg = FakeTG()
    assert tgm.handle_callback(db, tg, cb("i:5", chat=999)) is None
    assert db.get_candidate("5")["state"] == "ready"


@pytest.mark.parametrize("data", ["", "x:5", "i:abc", "i:5; DROP TABLE candidates", "i:" + "9" * 40, None])
def test_malformed_callback_data_ignored(db, data):
    ready(db, 5)
    assert tgm.handle_callback(db, FakeTG(), cb(data)) is None
    assert db.get_candidate("5")["state"] == "ready"


def test_callback_for_unknown_candidate(db):
    tg = FakeTG()
    assert tgm.handle_callback(db, tg, cb("i:12345")) is None
    assert tg.answers  # user gets feedback rather than a silent spinner


def _real_tg(chat_ids, handler):
    return tgm.Telegram("TOK", chat_ids, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_chat_id_list_is_parsed_and_deduplicated():
    tg = _real_tg("42, 7,42", lambda r: httpx.Response(200, json={"ok": True, "result": {}}))
    assert tg.chat_ids == [42, 7] and tg.chat_id == 42


@pytest.mark.parametrize("bad", ["abc", "42,x", ",,"])
def test_bad_chat_id_list_rejected(bad):
    with pytest.raises(tgm.TelegramError):
        _real_tg(bad, lambda r: httpx.Response(200, json={"ok": True, "result": {}}))


def test_alert_goes_to_every_allowed_chat(monkeypatch):
    monkeypatch.setattr(tgm.time, "sleep", lambda s: None)
    seen = []

    def handler(req):
        import json
        seen.append(json.loads(req.content)["chat_id"])
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(seen)}})

    tg = _real_tg("42,7", handler)
    assert tg.send("hi", account_id="5") == 1  # returns the primary chat's message id
    assert seen == [42, 7]


def test_failed_send_to_extra_chat_does_not_fail_or_resend_the_alert(monkeypatch):
    monkeypatch.setattr(tgm.time, "sleep", lambda s: None)
    import json
    calls = []

    def handler(req):
        chat = json.loads(req.content)["chat_id"]
        calls.append(chat)
        if chat == 7:  # the second person never pressed Start
            return httpx.Response(400, json={"ok": False, "description": "chat not found"})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})

    tg = _real_tg("42,7", handler)
    assert tg.send("hi") == 9 and calls == [42, 7]


def test_failed_send_to_primary_raises_before_touching_extras(monkeypatch):
    monkeypatch.setattr(tgm.time, "sleep", lambda s: None)
    import json
    calls = []

    def handler(req):
        calls.append(json.loads(req.content)["chat_id"])
        return httpx.Response(400, json={"ok": False, "description": "chat not found"})

    with pytest.raises(tgm.TelegramError):
        _real_tg("42,7", handler).send("hi")
    assert calls == [42]  # no partial delivery that a retry would duplicate


def test_button_press_from_second_allowed_chat_is_honoured(db):
    ready(db, 5)
    tg = FakeTG("42,7")
    assert tgm.handle_callback(db, tg, cb("i:5", chat=7, mid=11)) == "interested"
    assert tg.cleared == [11] and tg.cleared_in == 7  # buttons cleared in the chat that pressed
    ready(db, 6)
    assert tgm.handle_callback(db, tg, cb("i:6", chat=999)) is None  # still not just anyone


def test_token_never_leaks_into_error_message():
    def boom(_req):
        raise httpx.ConnectError("failed for https://api.telegram.org/botSECRETTOKEN/sendMessage")

    t = tgm.Telegram("SECRETTOKEN", "42", client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(tgm.TelegramError) as ei:
        t._call("sendMessage", chat_id=42)
    assert "SECRETTOKEN" not in str(ei.value) and ei.value.__cause__ is None


# --------------------------------------------------------------- routing


def test_instant_on_convergence_or_high_confidence_else_digest(db, cfg):
    ready(db, 1, conv=2, conf=0.5)   # converged -> instant
    ready(db, 2, conv=1, conf=0.95)  # very confident -> instant
    ready(db, 3, conv=1, conf=0.7)   # neither -> digest
    tg = FakeTG()
    assert routing.route_instant(db, cfg, tg) == 2
    states = {a: db.get_candidate(a)["state"] for a in "123"}
    assert states == {"1": "alerted", "2": "alerted", "3": "ready"}


def test_instant_hourly_cap_spills_to_digest_without_dropping(db, cfg):
    cfg.filters["alerts"]["max_instant_per_hour"] = 2
    for i in range(1, 6):
        ready(db, i, conv=3)
    tg = FakeTG()
    assert routing.route_instant(db, cfg, tg) == 2
    assert db.candidate_state_counts() == {"alerted": 2, "ready": 3}
    # the spill-over is delivered by the digest
    routing.send_digest(db, cfg, tg)
    assert db.candidate_state_counts() == {"alerted": 2, "digested": 3}


def test_failed_send_does_not_lose_the_lead(db, cfg):
    ready(db, 1, conv=3)

    class Down(FakeTG):
        def send(self, *a, **k):
            raise tgm.TelegramError("down")

    with pytest.raises(tgm.TelegramError):
        routing.route_instant(db, cfg, Down())
    assert db.get_candidate("1")["state"] == "ready"  # still eligible next cycle


def test_digest_sends_once_per_day_after_the_configured_hour(db, cfg):
    cfg.filters["alerts"]["digest_hour_utc"] = 7
    early = datetime(2026, 10, 29, 6, 0, tzinfo=timezone.utc)
    later = datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc)
    assert not routing.digest_due(db, cfg, early)
    assert routing.digest_due(db, cfg, later)
    ready(db, 1)
    tg = FakeTG()
    assert routing.send_digest(db, cfg, tg) == 1
    assert not routing.digest_due(db, cfg, later)  # already sent today
    assert len(tg.sent) == 2  # header + 1 item


def test_empty_digest_sends_nothing_but_marks_the_day(db, cfg):
    tg = FakeTG()
    assert routing.send_digest(db, cfg, tg) == 0 and tg.sent == []
    assert db.usage_today("digest_sent") == 1


# ------------------------------------------------------- has_own_contracts


@pytest.mark.parametrize("val,needle", [
    ("yes", "✅"), ("no", "⛔"), ("unclear", "❔"),
])
def test_alert_shows_contracts_verdict(val, needle):
    text = tgm.format_alert({"username": "x", "name": "n", "llm_summary": "s", "llm_contracts": val,
                             "followers_count": 1, "score": 1})
    assert needle in text and "Own smart contracts" in text


def test_alert_omits_contracts_line_when_unknown_or_tampered():
    base = {"username": "x", "name": "n", "followers_count": 1, "score": 1}
    assert "Own smart contracts" not in tgm.format_alert(base)
    # an unexpected DB value must not be echoed into the message
    evil = tgm.format_alert({**base, "llm_contracts": "<script>alert(1)</script>"})
    assert "Own smart contracts" not in evil and "<script>" not in evil


# ------------------------------------------------------------- migration


def test_existing_v1_database_is_upgraded_in_place(tmp_path):
    import sqlite3
    from scout.db import Database

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE candidates (account_id TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL,
            convergence_count INTEGER NOT NULL DEFAULT 1, score REAL, llm_is_project INTEGER,
            llm_confidence REAL, llm_category TEXT, llm_chain TEXT, llm_summary TEXT, llm_stack TEXT,
            state TEXT NOT NULL DEFAULT 'new', alerted_at TEXT, updated_at TEXT NOT NULL);
        INSERT INTO candidates (account_id, first_seen_at, updated_at, state, llm_summary)
            VALUES ('7', 'x', 'x', 'ready', 'kept');
        PRAGMA user_version = 1;
    """)
    old.commit(); old.close()

    with Database(path) as db:
        row = db.get_candidate("7")
        assert row["llm_summary"] == "kept" and row["llm_contracts"] is None  # data survives, column added
        db.upsert_candidate("7", llm_contracts="yes")
        assert db.get_candidate("7")["llm_contracts"] == "yes"
        assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    with Database(path):  # opening again must be a no-op, not a duplicate-column error
        pass


def test_fresh_database_has_the_column_and_is_current(tmp_path):
    from scout.db import Database, SCHEMA_VERSION

    with Database(tmp_path / "new.db") as db:
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(candidates)")}
        assert "llm_contracts" in cols and SCHEMA_VERSION == 2
