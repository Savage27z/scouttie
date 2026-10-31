"""Decides which 'ready' candidates alert instantly and which wait for the digest.

Over-alerting is the most likely way this project fails, so digest is the default
and instant is the narrow exception, with a hard hourly cap. Candidates that miss
the cap simply stay 'ready' and fall into the digest -- nothing is dropped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .alerts.telegram import Telegram, format_alert
from .config import Config
from .db import Database, utcnow

log = logging.getLogger(__name__)


def _ready(db: Database) -> list[dict[str, Any]]:
    out = []
    for r in db.candidates_by_state("ready", limit=500):
        d = dict(r)
        d["followed"] = db.targets_followed(d["account_id"])
        out.append(d)
    return out


def is_instant(c: dict[str, Any], cfg: Config) -> bool:
    a = cfg.alerts()
    # A project with no contracts of its own is not an audit lead, however many
    # ecosystems it followed. It may still reach the digest (see send_digest).
    if c.get("llm_contracts") == "no":
        return False
    min_targets = int(cfg.stage3().get("min_targets", 2))
    converged = bool(a.get("instant_on_convergence", True)) and c["convergence_count"] >= min_targets
    confident = (c.get("llm_confidence") or 0) >= float(a.get("instant_min_confidence", 0.85))
    return converged or confident


def instant_sent_last_hour(db: Database) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    return db.conn.execute(
        "SELECT COUNT(*) AS c FROM candidates WHERE state = 'alerted' AND alerted_at >= ?",
        (cutoff,),
    ).fetchone()["c"]


def instant_sent_today(db: Database) -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return db.conn.execute(
        "SELECT COUNT(*) AS c FROM candidates WHERE state = 'alerted' AND alerted_at >= ?",
        (start.isoformat(timespec="seconds"),),
    ).fetchone()["c"]


def route_instant(db: Database, cfg: Config, tg: Telegram) -> int:
    cap = int(cfg.alerts().get("max_instant_per_hour", 6))
    day_cap = int(cfg.alerts().get("max_instant_per_day", 6))
    sent = 0
    for c in _ready(db):
        if instant_sent_last_hour(db) >= cap:
            log.info("instant cap (%d/h) reached; remaining candidates wait for the digest", cap)
            break
        if instant_sent_today(db) >= day_cap:
            log.info("instant cap (%d/day) reached; remaining candidates wait for the digest", day_cap)
            break
        if not is_instant(c, cfg):
            continue
        tg.send(format_alert(c, kind="instant"), account_id=c["account_id"])
        # Mark AFTER a successful send so a Telegram outage never loses a lead.
        db.upsert_candidate(c["account_id"], state="alerted", alerted_at=utcnow())
        sent += 1
    return sent


def digest_due(db: Database, cfg: Config, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    if now.hour < int(cfg.alerts().get("digest_hour_utc", 7)):
        return False
    return db.usage_today("digest_sent") == 0


def _digest_rank(c: dict[str, Any]) -> tuple:
    """Best leads first: own contracts, then convergence, then model confidence."""
    contracts = {"yes": 0, "unclear": 1, None: 1}.get(c.get("llm_contracts"), 1)
    return (contracts, -(c.get("convergence_count") or 0), -(c.get("llm_confidence") or 0),
            -(c.get("score") or 0))


def send_digest(db: Database, cfg: Config, tg: Telegram) -> int:
    """Send the best few 'ready' leads; shelve the rest silently.

    The digest is a short list, never a dump of the queue. Everything not sent is kept
    (state 'shelved') so it stays queryable, but it never reaches Telegram.
    """
    a = cfg.alerts()
    max_items = int(a.get("digest_max_items", 10))
    skip_no = bool(a.get("digest_skip_no_contracts", True))
    items = sorted(_ready(db), key=_digest_rank)
    # Mark the day as done even when empty, so we don't re-check all day.
    db.bump_usage("digest_sent")
    if not items:
        return 0
    eligible = [c for c in items if not (skip_no and c.get("llm_contracts") == "no")]
    chosen = eligible[:max_items]
    chosen_ids = {c["account_id"] for c in chosen}
    shelved = [c for c in items if c["account_id"] not in chosen_ids]
    for c in shelved:
        db.upsert_candidate(c["account_id"], state="shelved")
    if not chosen:
        return 0
    note = f", {len(shelved)} lower-priority shelved" if shelved else ""
    tg.send(f"🗞 <b>Daily digest</b> — top {len(chosen)} of {len(items)}{note}")
    sent = 0
    for c in chosen:
        tg.send(format_alert(c, kind="digest"), account_id=c["account_id"])
        db.upsert_candidate(c["account_id"], state="digested", alerted_at=utcnow())
        sent += 1
    return sent
