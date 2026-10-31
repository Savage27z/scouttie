"""Poll one target for new followers.

The provider returns followers newest-first, so detecting new ones means reading
page 1 and paginating only until a page contains an id we have already seen. Cost
therefore tracks real new-follower volume, not list size.

Invariants worth knowing about:

* A poll is ALL-OR-NOTHING. Pages are fetched into memory first and only written
  to the database once every page succeeded. Recording pages 1-2 as "seen" and then
  failing on page 3 would make the next poll stop at page 1 and permanently lose
  everything on page 3+.
* The first poll of a target is a COLD START: page 1 is recorded as seen and
  nothing is emitted. Otherwise the first poll would report the 200 newest existing
  followers as brand new.
* Hitting the page cap means older new followers were not fetched and cannot be
  recovered by later polls (they sit below the new "seen" boundary). That is a
  data-loss event, so it is logged loudly and shortens the poll interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import TargetSpec
from .db import Database, parse_ts
from .providers.base import Account, XProvider

log = logging.getLogger(__name__)

SEEN_KEEP_PER_TARGET = 100_000

# Adaptive interval bounds. Detection latency of an hour or two is fine here --
# this is lead generation, not trading.
TARGET_PAGE_FILL = 0.75
RATE_SMOOTHING = 0.5  # weight of the newest observation


@dataclass
class PollResult:
    target: str
    new_accounts: list[Account] = field(default_factory=list)
    pages_fetched: int = 0
    items_returned: int = 0
    credits_spent: float = 0.0
    hit_page_cap: bool = False
    cold_start: bool = False
    next_interval_s: int = 3600
    observed_new_per_hour: float | None = None


def _next_interval(
    spec: TargetSpec,
    current_s: int,
    new_count: int,
    elapsed_h: float | None,
    previous_rate: float | None,
    hit_cap: bool,
) -> tuple[int, float | None]:
    """Steer the interval so an expected poll fills ~75% of one page."""
    if hit_cap:
        # We fell behind. Halve the interval; the rate we measured is a floor.
        return max(spec.min_poll_interval_s, current_s // 2), previous_rate

    if elapsed_h is None or elapsed_h <= 0:
        return current_s, previous_rate

    observed = new_count / elapsed_h
    rate = (
        observed
        if previous_rate is None
        else RATE_SMOOTHING * observed + (1 - RATE_SMOOTHING) * previous_rate
    )
    if rate <= 0:
        return spec.max_poll_interval_s, rate
    wanted_h = (TARGET_PAGE_FILL * spec.page_size) / rate
    interval = int(min(spec.max_poll_interval_s, max(spec.min_poll_interval_s, wanted_h * 3600)))
    return interval, rate


def poll_target(
    db: Database,
    provider: XProvider,
    spec: TargetSpec,
    *,
    seen_keep: int = SEEN_KEEP_PER_TARGET,
) -> PollResult:
    row = db.get_target(spec.username)
    if row is None:
        raise RuntimeError(f"target @{spec.username} not registered; run startup first")
    tid = int(row["id"])
    result = PollResult(target=spec.username, next_interval_s=int(row["poll_interval_s"]))
    cold = not row["cold_started"]
    result.cold_start = cold

    poll_id = db.start_poll(tid)
    fetched: list[Account] = []  # every account seen this poll, newest first
    new_accounts: list[Account] = []
    emitted: set[str] = set()
    cursor: str | None = None
    pages = 0
    credits = 0.0
    items = 0
    hit_cap = False

    try:
        while True:
            page = provider.get_followers(spec.username, cursor, spec.page_size)
            pages += 1
            credits += page.credits_spent
            items += page.raw_count
            fetched.extend(page.accounts)

            if cold:
                break  # one page is enough to establish the boundary

            ids = [a["id"] for a in page.accounts]
            unseen = db.filter_unseen(tid, ids)
            for a in page.accounts:
                # Cursor drift can return the same account on two pages.
                if a["id"] in unseen and a["id"] not in emitted:
                    emitted.add(a["id"])
                    new_accounts.append(a)

            reached_known = len(unseen) < len(ids)
            if reached_known or not page.has_next_page:
                break
            if pages >= spec.max_pages_per_poll:
                hit_cap = True
                log.error(
                    "@%s: hit page cap (%d pages) with every page entirely new -- "
                    "older new followers were NOT fetched and cannot be recovered. "
                    "Raise max_pages_per_poll or poll more often.",
                    spec.username,
                    pages,
                )
                break
            cursor = page.next_cursor
    except Exception as exc:
        # Nothing was recorded, so the next poll retries from scratch. We still
        # log the credits already burned so spend reporting stays honest.
        db.finish_poll(poll_id, pages, items, 0, credits, hit_page_cap=False, error=str(exc)[:500])
        raise

    now = datetime.now(timezone.utc)
    last = parse_ts(row["last_polled_at"])
    elapsed_h = ((now - last).total_seconds() / 3600.0) if last and not cold else None

    interval, rate = _next_interval(
        spec,
        int(row["poll_interval_s"]),
        len(new_accounts),
        elapsed_h,
        row["observed_new_per_hour"],
        hit_cap,
    )

    with db.tx():
        if cold:
            db.record_seen(tid, [a["id"] for a in fetched])
            db.mark_cold_started(tid)
        else:
            for pos, acc in enumerate(new_accounts):
                db.upsert_account(acc)
                db.record_follow_event(acc["id"], tid, pos)
            # Record every id we fetched (not only the new ones) so re-reading the
            # same boundary page next time is cheap and idempotent.
            db.record_seen(tid, [a["id"] for a in fetched])
        db.prune_seen(tid, seen_keep)
        db.mark_polled(tid, rate, interval)
        db.finish_poll(
            poll_id,
            pages,
            items,
            len(new_accounts),
            credits,
            hit_page_cap=hit_cap,
        )

    result.new_accounts = new_accounts
    result.pages_fetched = pages
    result.items_returned = items
    result.credits_spent = credits
    result.hit_page_cap = hit_cap
    result.next_interval_s = interval
    result.observed_new_per_hour = rate
    return result
