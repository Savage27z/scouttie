from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from conftest import FakeProvider, make_account
from scout import pipeline
from scout.collector import poll_target
from scout.config import Config, Secrets, TargetSpec, load_config
from scout.db import Database


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return Config(secrets=Secrets(), targets=base.targets, filters=base.filters, db_path=tmp_path / "x.db")


def project(i, **over):
    return make_account(
        i, name="Orbit Labs", bio="Building the liquidity layer. Mainnet soon.",
        url="https://orbit-labs.xyz", followers_count=40, following_count=30,
        created_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds"),
        **over,
    )


def test_end_to_end_poll_then_judge(db, cfg):
    spec = TargetSpec(username="solana", page_size=20)
    db.upsert_target("solana", 3600, True)
    p = FakeProvider([make_account(i, bio="old follower") for i in range(1000, 1040)])
    poll_target(db, p, spec)  # cold start

    p.prepend([project(5001), make_account(5002, bio="gm", followers_count=1, statuses_count=0)])
    r = poll_target(db, p, spec)
    decisions = {a["id"]: pipeline.judge(db, cfg, a) for a in r.new_accounts}

    assert decisions["5001"].passed_gate and decisions["5001"].score >= 5
    assert not decisions["5002"].passed_gate and decisions["5002"].reason.startswith("stage1")
    assert db.get_candidate("5001")["state"] == "new"
    assert db.get_candidate("5002") is None  # rejects are NOT candidates...
    assert db.stage_counts()["stage1"] == {"passed": 1, "failed": 1}  # ...but are recorded


def test_rejections_are_stored_for_later_audit(db, cfg):
    db.upsert_target("solana", 3600, True)
    acc = make_account(9, bio="airdrops", following_count=4000)
    db.upsert_account(acc)
    pipeline.judge(db, cfg, acc)
    row = db.conn.execute("SELECT * FROM verdicts WHERE account_id='9' AND stage='stage1'").fetchone()
    assert row["passed"] == 0 and row["reason"] == "following>2000"


def test_rejudge_does_not_reset_candidate_state(db, cfg):
    db.upsert_target("solana", 3600, True)
    acc = project(7)
    db.upsert_account(acc)
    pipeline.judge(db, cfg, acc)
    db.set_candidate_state("7", "interested")
    pipeline.judge(db, cfg, acc)
    assert db.get_candidate("7")["state"] == "interested"


def test_convergence_lifts_candidate_count(db, cfg):
    t1 = db.upsert_target("solana", 3600, True)
    t2 = db.upsert_target("base", 3600, True)
    acc = project(8)
    db.upsert_account(acc)
    db.record_follow_event("8", t1)
    d1 = pipeline.judge(db, cfg, acc)
    db.record_follow_event("8", t2)
    d2 = pipeline.judge(db, cfg, acc)
    assert d1.convergence_count == 1 and d2.convergence_count == 2
    assert d2.score > d1.score
    assert db.get_candidate("8")["convergence_count"] == 2


def test_convergence_window_anchored_at_latest_event_not_now(db):
    """An account whose two follows were months ago must still read as converged
    when replayed -- otherwise replay disagrees with what live would have said."""
    t1 = db.upsert_target("solana", 3600, True)
    t2 = db.upsert_target("base", 3600, True)
    old = "2026-01-01T00:00:00+00:00"
    older = "2025-12-31T12:00:00+00:00"
    db.conn.execute("INSERT INTO follow_events VALUES ('x', ?, ?, 0)", (t1, older))
    db.conn.execute("INSERT INTO follow_events VALUES ('x', ?, ?, 0)", (t2, old))
    assert db.convergence("x", 72)[0] == 2
    # events further apart than the window do NOT converge
    db.conn.execute("INSERT INTO follow_events VALUES ('y', ?, '2025-01-01T00:00:00+00:00', 0)", (t1,))
    db.conn.execute("INSERT INTO follow_events VALUES ('y', ?, '2025-06-01T00:00:00+00:00', 0)", (t2,))
    assert db.convergence("y", 72)[0] == 1


def test_replay_dry_run_reports_flips_without_writing(db, cfg):
    db.upsert_target("solana", 3600, True)
    borderline = make_account(
        20, name="Sparrow", bio="shipping soon", url="https://github.com/sparrow",
        followers_count=40, following_count=60,
        created_at=(datetime.now(timezone.utc) - timedelta(days=20)).isoformat(timespec="seconds"),
    )
    db.upsert_account(borderline)
    db.upsert_account(project(21))
    for row in db.iter_accounts():
        pipeline.judge(db, cfg, row)
    before = db.stage_counts()

    # loosen the threshold: borderline account should flip IN
    cfg.filters["stage2"]["threshold"] = 1.0
    rep = pipeline.replay(db, cfg, apply=False)
    assert rep.total == 2 and rep.gate_pass == 2 and rep.previously_gate_pass == 1
    assert [f[0] for f in rep.flipped_in] == ["20"]
    assert db.stage_counts() == before  # dry-run wrote nothing

    # tighten it: the project flips OUT
    cfg.filters["stage2"]["threshold"] = 50.0
    rep = pipeline.replay(db, cfg, apply=False)
    assert rep.gate_pass == 0 and len(rep.flipped_out) == 1

    # apply persists
    pipeline.replay(db, cfg, apply=True)
    assert db.stage_counts()["stage2"]["passed"] == 0
