"""The long-running loop: poll due targets, judge, enrich, alert.

Single-threaded on purpose. Volumes are small (a handful of polls an hour) and one
thread means no locking around SQLite and no interleaved Telegram sends.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from .alerts.telegram import Telegram, TelegramError, poll_callbacks
from .collector import PollResult, poll_target
from .config import Config, ConfigError
from .db import Database, parse_ts
from .enrich import enrich_pending
from .funnel.llm import LLMFatalError
from .pipeline import Decision, judge
from .providers.base import ProviderError, ProviderUnavailable, UserNotFound, XProvider
from .routing import digest_due, route_instant, send_digest
from .webfetch import make_expander

log = logging.getLogger(__name__)

BACKOFF_BASE_S = 300
BACKOFF_MAX_S = 3600
TICK_S = 30


def bootstrap(db: Database, cfg: Config, provider: XProvider) -> list[str]:
    """Register targets and resolve their ids. A bad handle on an ACTIVE target
    fails loudly here instead of producing a silently dead target."""
    registered = []
    for spec in cfg.targets:
        tid = db.upsert_target(spec.username, spec.poll_interval_s, spec.active)
        row = db.get_target(spec.username)
        if spec.active and not row["user_id"]:
            try:
                user = provider.resolve_user(spec.username)
            except UserNotFound as exc:
                raise ConfigError(f"target @{spec.username} could not be resolved: {exc}") from exc
            db.set_target_user_id(tid, user.id)
            log.info("resolved @%s -> %s (%s followers)", spec.username, user.id, user.followers_count)
        if spec.active:
            registered.append(spec.username)
    return registered


def judge_new(db: Database, cfg: Config, result: PollResult, expand: Any = None) -> list[Decision]:
    return [judge(db, cfg, acc, expand=expand) for acc in result.new_accounts]


def target_due(row: Any, now: datetime) -> bool:
    last = parse_ts(row["last_polled_at"])
    return last is None or (now - last).total_seconds() >= int(row["poll_interval_s"])


class Runner:
    def __init__(
        self,
        db: Database,
        cfg: Config,
        provider: XProvider,
        *,
        llm_client: Any = None,
        telegram: Telegram | None = None,
        expand: Any = "default",
    ):
        self.db, self.cfg, self.provider = db, cfg, provider
        self.llm_client, self.tg = llm_client, telegram
        self._expand = make_expander() if expand == "default" else expand
        self._fails: dict[str, int] = {}
        self._retry_at: dict[str, float] = {}
        self._tg_offset: int | None = None
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg)

    def cycle(self) -> dict[str, int]:
        """One pass over everything. Returns counters for logging/tests."""
        now = datetime.now(timezone.utc)
        out = {"polled": 0, "new": 0, "gate": 0, "instant": 0, "digest": 0}
        specs = {s.username: s for s in self.cfg.active_targets()}

        for row in self.db.list_targets(active_only=True):
            spec = specs.get(row["username"])
            if spec is None or not target_due(row, now):
                continue
            if time.monotonic() < self._retry_at.get(spec.username, 0):
                continue
            try:
                result = poll_target(self.db, self.provider, spec)
            except ProviderUnavailable as exc:
                n = self._fails[spec.username] = self._fails.get(spec.username, 0) + 1
                delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (n - 1))
                self._retry_at[spec.username] = time.monotonic() + delay
                log.warning("@%s unavailable (%s); retry in %ds", spec.username, exc, delay)
                continue
            # Any other ProviderError (bad key, out of credits) is fatal: retrying
            # cannot fix it and would only hide it. Let it propagate.
            self._fails.pop(spec.username, None)
            decisions = judge_new(self.db, self.cfg, result, self._expand)
            out["polled"] += 1
            out["new"] += len(result.new_accounts)
            out["gate"] += sum(d.passed_gate for d in decisions)
            log.info(
                "@%s: %d new, %d past gate, %d page(s), %.0f credits, next in %ds%s",
                spec.username, len(result.new_accounts), sum(d.passed_gate for d in decisions),
                result.pages_fetched, result.credits_spent, result.next_interval_s,
                " (cold start)" if result.cold_start else "",
            )

        if self.llm_client is not None:
            try:
                enrich_pending(self.db, self.cfg, self.llm_client)
            except LLMFatalError as exc:
                # Retrying cannot fix a bad key / empty balance / rejected request, and
                # hammering the API would only hide it. Stop the LLM stage until restart;
                # polling and delivery of already-ready leads carry on, and unclassified
                # candidates simply wait in the queue.
                log.error("LLM disabled until restart: %s", exc)
                self.llm_client = None
        else:
            self._warn_once("llm", "DEEPSEEK_API_KEY not set (or LLM disabled): candidates will queue unclassified")

        if self.tg is not None:
            # Independent steps: a flaky button-poll (getUpdates) must never stop an
            # alert from going out in the same cycle. Failed sends leave leads
            # 'ready', so they are simply retried next cycle.
            try:
                self._tg_offset = poll_callbacks(self.db, self.tg, self._tg_offset)
            except TelegramError as exc:
                log.warning("telegram (button poll): %s", exc)
            try:
                out["instant"] = route_instant(self.db, self.cfg, self.tg)
            except TelegramError as exc:
                log.warning("telegram (instant alerts): %s", exc)
            try:
                if digest_due(self.db, self.cfg):
                    out["digest"] = send_digest(self.db, self.cfg, self.tg)
            except TelegramError as exc:
                log.warning("telegram (digest): %s", exc)
        else:
            self._warn_once("tg", "Telegram not configured: nothing will be delivered")
        return out

    def run_forever(self) -> None:
        log.info("running; %d active target(s)", len(self.cfg.active_targets()))
        while True:
            try:
                self.cycle()
            except ProviderError as exc:
                log.error("fatal provider error, stopping: %s", exc)
                raise
            time.sleep(TICK_S)
