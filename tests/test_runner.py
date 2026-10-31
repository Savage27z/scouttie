from __future__ import annotations

import pytest

from conftest import FakeProvider, make_account
from scout import cli
from scout.config import Config, ConfigError, Secrets, TargetSpec, load_config
from scout.providers.base import ProviderError, ProviderUnavailable, UserNotFound
from scout.runner import Runner, bootstrap
from datetime import datetime, timedelta, timezone


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    return Config(
        secrets=Secrets(), filters=base.filters, db_path=tmp_path / "x.db",
        targets=(TargetSpec("solana", page_size=20), TargetSpec("base", page_size=20),
                 TargetSpec("parked", active=False)),
    )


def good_project(i):
    return make_account(
        i, name="Orbit Labs", bio="Building the liquidity layer. Mainnet soon.",
        url="https://orbit-labs.xyz", followers_count=40, following_count=30,
        created_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat(timespec="seconds"),
    )


class MultiProvider(FakeProvider):
    """One FakeProvider per target handle."""

    def __init__(self, per_target):
        self.per = per_target

    def get_followers(self, username, cursor=None, page_size=200):
        return self.per[username].get_followers(username, cursor, page_size)

    def resolve_user(self, username):
        return self.per[username].resolve_user(username)


def test_bootstrap_registers_and_resolves_only_active_targets(db, cfg):
    p = MultiProvider({"solana": FakeProvider([]), "base": FakeProvider([])})
    assert bootstrap(db, cfg, p) == ["solana", "base"]
    assert db.get_target("solana")["user_id"] == "1"
    assert db.get_target("parked")["user_id"] is None  # inactive: no API call wasted


def test_bootstrap_fails_loudly_on_unresolvable_active_handle(db, cfg):
    class Missing(MultiProvider):
        def resolve_user(self, username):
            raise UserNotFound("nope")

    with pytest.raises(ConfigError, match="could not be resolved"):
        bootstrap(db, cfg, Missing({}))


def test_full_cycle_cold_start_then_detect_and_converge(db, cfg):
    sol = FakeProvider([make_account(i) for i in range(1000, 1040)])
    base = FakeProvider([make_account(i) for i in range(2000, 2040)])
    p = MultiProvider({"solana": sol, "base": base})
    bootstrap(db, cfg, p)
    r = Runner(db, cfg, p)

    out = r.cycle()  # cold start both
    assert out["polled"] == 2 and out["new"] == 0

    # force both due again, then a project follows both ecosystems
    db.conn.execute("UPDATE targets SET last_polled_at = '2000-01-01T00:00:00+00:00'")
    sol.prepend([good_project(7777)])
    base.prepend([good_project(7777)])
    out = r.cycle()
    assert out["new"] == 2 and out["gate"] == 2
    assert db.get_candidate("7777")["convergence_count"] == 2  # converged across both
    assert set(db.targets_followed("7777")) == {"solana", "base"}


def test_not_due_targets_are_not_polled(db, cfg):
    p = MultiProvider({"solana": FakeProvider([make_account(1)]), "base": FakeProvider([make_account(2)])})
    bootstrap(db, cfg, p)
    r = Runner(db, cfg, p)
    assert r.cycle()["polled"] == 2
    assert r.cycle()["polled"] == 0  # just polled; interval hasn't elapsed


def test_transient_failure_backs_off_without_crashing_or_corrupting(db, cfg):
    class Flaky(MultiProvider):
        def get_followers(self, username, cursor=None, page_size=200):
            if username == "solana":
                raise ProviderUnavailable("503")
            return super().get_followers(username, cursor, page_size)

    p = Flaky({"solana": FakeProvider([]), "base": FakeProvider([make_account(2)])})
    bootstrap(db, cfg, p)
    r = Runner(db, cfg, p)
    out = r.cycle()
    assert out["polled"] == 1  # base still polled despite solana failing
    assert r._fails["solana"] == 1 and r._retry_at["solana"] > 0
    assert db.get_target("solana")["cold_started"] == 0  # no half-state recorded


def test_fatal_provider_error_propagates(db, cfg):
    class NoCredits(MultiProvider):
        def get_followers(self, *a, **k):
            raise ProviderError("out of credits (402)")

    p = NoCredits({"solana": FakeProvider([]), "base": FakeProvider([])})
    bootstrap(db, cfg, p)
    with pytest.raises(ProviderError):
        Runner(db, cfg, p).cycle()


def test_cli_parser_exposes_all_commands():
    p = cli.build_parser()
    for cmd in ("measure", "poll-once", "run", "status", "replay", "test-alert", "test-llm"):
        assert p.parse_args([cmd]).cmd == cmd
    assert p.parse_args(["replay", "--apply"]).apply is True


def test_status_runs_on_a_fresh_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCOUT_DB_PATH", str(tmp_path / "fresh.db"))
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "last 24h" in out and "candidates" in out


def test_missing_credentials_give_a_clear_error_not_a_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SCOUT_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setattr(cli, "load_config", lambda: Config(
        secrets=Secrets(), targets=(TargetSpec("solana"),), filters=load_config().filters, db_path=tmp_path / "x.db"))
    assert cli.main(["measure"]) == 1
    assert "TWITTERAPI_KEY" in capsys.readouterr().err


def test_flaky_button_poll_does_not_block_alert_delivery(db, cfg, monkeypatch):
    from scout import runner as runner_mod
    from scout.alerts.telegram import TelegramError

    class NoProvider:
        def get_followers(self, *a, **k):
            raise AssertionError("nothing due")

    calls = []
    monkeypatch.setattr(runner_mod, "poll_callbacks", lambda *a, **k: (_ for _ in ()).throw(TelegramError("getUpdates: ConnectTimeout")))
    monkeypatch.setattr(runner_mod, "route_instant", lambda *a, **k: calls.append("instant") or 0)
    monkeypatch.setattr(runner_mod, "digest_due", lambda *a, **k: calls.append("digest_check") or False)
    Runner(db, cfg, NoProvider(), telegram=object(), expand=None).cycle()
    assert calls == ["instant", "digest_check"]  # both still ran despite the failed button poll
