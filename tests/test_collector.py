from __future__ import annotations

import pytest

from conftest import FakeProvider, make_account
from scout import collector
from scout.collector import poll_target


def accounts(start: int, n: int):
    """ids start..start+n-1, in order."""
    return [make_account(i) for i in range(start, start + n)]


def seed(db, spec, provider):
    """Cold start so subsequent polls are real."""
    r = poll_target(db, provider, spec)
    assert r.cold_start
    return r


def test_cold_start_emits_nothing_and_records_boundary(db, registered):
    p = FakeProvider(accounts(1000, 50))  # newest first: 1000 is newest
    r = poll_target(db, p, registered)
    assert r.cold_start and r.new_accounts == []
    assert r.pages_fetched == 1
    tid = db.get_target("solana")["id"]
    assert db.seen_count(tid) == 20  # one page of page_size=20
    assert db.get_target("solana")["cold_started"] == 1
    assert db.stage_counts() == {}  # nothing judged


def test_no_new_followers(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    r = poll_target(db, p, registered)
    assert not r.cold_start and r.new_accounts == []
    assert r.pages_fetched == 1  # stops immediately: page 1 is all known


def test_detects_new_at_top_and_stops_at_boundary(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    p.prepend([make_account(2001), make_account(2002), make_account(2003)])
    r = poll_target(db, p, registered)
    assert [a["id"] for a in r.new_accounts] == ["2001", "2002", "2003"]
    assert r.pages_fetched == 1
    # written through: accounts + follow events exist
    assert db.get_account("2001") is not None
    assert db.targets_followed("2001") == ["solana"]


def test_same_follower_not_reported_twice(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    p.prepend([make_account(2001)])
    assert len(poll_target(db, p, registered).new_accounts) == 1
    assert poll_target(db, p, registered).new_accounts == []


def test_overflow_paginates_until_known_id(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    p.prepend(accounts(5000, 45))  # 45 new = 2 full pages + 5
    r = poll_target(db, p, registered)
    assert len(r.new_accounts) == 45
    assert r.pages_fetched == 3  # 20 + 20 + (5 new + 15 known)
    assert not r.hit_page_cap


def test_page_cap_flagged_and_interval_shortened(db, spec):
    spec = type(spec)(username="solana", page_size=20, max_pages_per_poll=2)
    db.upsert_target("solana", 3600, True)
    p = FakeProvider(accounts(1000, 50))
    seed(db, spec, p)
    p.prepend(accounts(5000, 200))  # far more than 2 pages
    r = poll_target(db, p, spec)
    assert r.hit_page_cap and r.pages_fetched == 2
    assert len(r.new_accounts) == 40
    assert r.next_interval_s < 3600


def test_mid_poll_failure_records_nothing(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    tid = db.get_target("solana")["id"]
    before = db.seen_count(tid)

    p.prepend(accounts(5000, 45))
    p.fail_on_call = p.calls + 2  # page 2 of the next poll blows up
    with pytest.raises(RuntimeError):
        poll_target(db, p, registered)

    assert db.seen_count(tid) == before  # page 1 was NOT half-committed
    assert db.get_account("5000") is None
    # the failed poll still logged its error and the credits it burned
    row = db.conn.execute("SELECT error, credits_spent FROM poll_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["error"] and row["credits_spent"] > 0

    # recovery: the next poll picks up everything
    p.fail_on_call = None
    r = poll_target(db, p, registered)
    assert len(r.new_accounts) == 45


def test_duplicate_across_pages_deduped(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    dup = make_account(7000)
    # id 7000 appears on both page 1 and page 2 (cursor drift)
    p.prepend(accounts(5000, 19) + [dup] + [dup] + accounts(5100, 18))
    r = poll_target(db, p, registered)
    ids = [a["id"] for a in r.new_accounts]
    assert ids.count("7000") == 1


def test_seen_set_is_bounded(db, registered):
    p = FakeProvider(accounts(1000, 50))
    seed(db, registered, p)
    tid = db.get_target("solana")["id"]
    for batch in range(5):
        p.prepend(accounts(10_000 + batch * 100, 10))
        poll_target(db, p, registered, seen_keep=30)
    assert db.seen_count(tid) <= 30


def test_interval_slows_when_quiet_and_speeds_up_when_busy():
    from scout.config import TargetSpec

    spec = TargetSpec(username="x", page_size=200)
    # 2 new in 1h -> very quiet -> long interval (clamped to max)
    slow, _ = collector._next_interval(spec, 3600, 2, 1.0, None, False)
    assert slow == spec.max_poll_interval_s
    # 600 new in 1h -> busy -> want ~150 per poll -> 15 min (the floor)
    fast, _ = collector._next_interval(spec, 3600, 600, 1.0, None, False)
    assert fast == spec.min_poll_interval_s
    # zero new followers -> stays at max, doesn't divide by zero
    idle, rate = collector._next_interval(spec, 3600, 0, 1.0, None, False)
    assert idle == spec.max_poll_interval_s and rate == 0


def test_poll_interval_bounds_come_from_config_and_are_validated():
    from scout.config import ConfigError, TargetSpec

    fast = TargetSpec(username="x", page_size=200, min_poll_interval_s=300, max_poll_interval_s=1800)
    quiet, _ = collector._next_interval(fast, 3600, 1, 1.0, None, False)
    assert quiet == 1800  # a tighter cap is honoured: latency is the user's call
    assert TargetSpec(username="y").max_poll_interval_s == 7200  # default matches the 1-2h promise
    with pytest.raises(ConfigError):
        TargetSpec(username="z", min_poll_interval_s=900, max_poll_interval_s=300)
