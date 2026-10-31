"""SQLite state. Schema, migrations, and the queries the funnel needs.

Design notes:

* `seen_followers` is a BOUNDED rolling set, not a full follower list. Because we
  only ever read the newest end of a follower list, we never need an account's
  historical millions. Pruning keeps the most recent N ids per target, which at
  realistic volumes is still weeks of memory.
* Every funnel stage writes a row to `verdicts`, including rejections. That is what
  makes `scout replay` possible: thresholds can be re-tuned offline against real
  observed accounts, with no API calls.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS targets (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    username                  TEXT NOT NULL UNIQUE,
    user_id                   TEXT,
    poll_interval_s           INTEGER NOT NULL DEFAULT 3600,
    observed_new_per_hour     REAL,
    last_polled_at            TEXT,
    cold_started              INTEGER NOT NULL DEFAULT 0,
    active                    INTEGER NOT NULL DEFAULT 1
);

-- Bounded rolling set of recently-seen follower ids. Pruned by recency.
CREATE TABLE IF NOT EXISTS seen_followers (
    target_id      INTEGER NOT NULL,
    follower_id    TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL,
    PRIMARY KEY (target_id, follower_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_seen_prune ON seen_followers (target_id, first_seen_at);

CREATE TABLE IF NOT EXISTS accounts (
    id                  TEXT PRIMARY KEY,
    username            TEXT,
    name                TEXT,
    bio                 TEXT,
    url                 TEXT,
    followers_count     INTEGER,
    following_count     INTEGER,
    statuses_count      INTEGER,
    favourites_count    INTEGER,
    created_at          TEXT,
    verified            INTEGER,
    protected           INTEGER,
    has_default_avatar  INTEGER,
    profile_image_url   TEXT,
    raw_json            TEXT,
    first_seen_at       TEXT NOT NULL,
    last_refreshed_at   TEXT NOT NULL
);

-- One row per (account, stage). Rejections are kept deliberately: they are the
-- evidence base for tuning, and let you audit what the filter threw away.
CREATE TABLE IF NOT EXISTS verdicts (
    account_id       TEXT NOT NULL,
    stage            TEXT NOT NULL,
    passed           INTEGER NOT NULL,
    score            REAL,
    reason           TEXT,
    components_json  TEXT,
    decided_at       TEXT NOT NULL,
    PRIMARY KEY (account_id, stage)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_verdicts_stage ON verdicts (stage, passed);

-- Which tracked targets each account was observed following, and when detected.
CREATE TABLE IF NOT EXISTS follow_events (
    account_id   TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    detected_at  TEXT NOT NULL,
    page_pos     INTEGER,
    PRIMARY KEY (account_id, target_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_follow_events_time ON follow_events (detected_at);

CREATE TABLE IF NOT EXISTS candidates (
    account_id            TEXT PRIMARY KEY,
    first_seen_at         TEXT NOT NULL,
    convergence_count     INTEGER NOT NULL DEFAULT 1,
    score                 REAL,
    llm_is_project        INTEGER,
    llm_confidence        REAL,
    llm_category          TEXT,
    llm_chain             TEXT,
    llm_summary           TEXT,
    llm_stack             TEXT,
    llm_contracts         TEXT,
    state                 TEXT NOT NULL DEFAULT 'new',
    alerted_at            TEXT,
    updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_state ON candidates (state);

CREATE TABLE IF NOT EXISTS poll_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id       INTEGER,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    pages_fetched   INTEGER NOT NULL DEFAULT 0,
    items_returned  INTEGER NOT NULL DEFAULT 0,
    new_followers   INTEGER NOT NULL DEFAULT 0,
    credits_spent   REAL NOT NULL DEFAULT 0,
    hit_page_cap    INTEGER NOT NULL DEFAULT 0,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_poll_log_time ON poll_log (started_at);

-- Daily counters for the LLM spend guards.
CREATE TABLE IF NOT EXISTS usage_counters (
    day    TEXT NOT NULL,
    kind   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, kind)
) WITHOUT ROWID;
"""


def utcnow() -> str:
    """ISO-8601 UTC timestamp. Stored as text; SQLite has no native datetime."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _hours_ago(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(
        timespec="seconds"
    )


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if current < 2:
            # v2: candidates.llm_contracts. CREATE TABLE IF NOT EXISTS leaves an existing
            # v1 table untouched, so add the column in place (guarded: fresh DBs have it).
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(candidates)")}
            if "llm_contracts" not in cols:
                self.conn.execute("ALTER TABLE candidates ADD COLUMN llm_contracts TEXT")
        if current < SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. isolation_level=None means autocommit otherwise."""
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ---------------------------------------------------------------- targets

    def upsert_target(self, username: str, poll_interval_s: int, active: bool) -> int:
        # On conflict we deliberately do NOT overwrite poll_interval_s: the
        # collector adapts it from observed volume, and a restart must not reset
        # that learning back to the config's cold-start value.
        self.conn.execute(
            """
            INSERT INTO targets (username, poll_interval_s, active)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET active = excluded.active
            """,
            (username, poll_interval_s, int(active)),
        )
        row = self.conn.execute(
            "SELECT id FROM targets WHERE username = ?", (username,)
        ).fetchone()
        return int(row["id"])

    def set_target_user_id(self, target_id: int, user_id: str) -> None:
        self.conn.execute(
            "UPDATE targets SET user_id = ? WHERE id = ?", (user_id, target_id)
        )

    def get_target(self, username: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM targets WHERE username = ?", (username,)
        ).fetchone()

    def list_targets(self, active_only: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT * FROM targets"
        if active_only:
            sql += " WHERE active = 1"
        return list(self.conn.execute(sql + " ORDER BY username"))

    def mark_polled(
        self, target_id: int, observed_new_per_hour: float | None, poll_interval_s: int
    ) -> None:
        self.conn.execute(
            """
            UPDATE targets
               SET last_polled_at = ?,
                   observed_new_per_hour = COALESCE(?, observed_new_per_hour),
                   poll_interval_s = ?
             WHERE id = ?
            """,
            (utcnow(), observed_new_per_hour, poll_interval_s, target_id),
        )

    def mark_cold_started(self, target_id: int) -> None:
        self.conn.execute(
            "UPDATE targets SET cold_started = 1 WHERE id = ?", (target_id,)
        )

    # --------------------------------------------------------- seen followers

    def filter_unseen(self, target_id: int, follower_ids: Iterable[str]) -> set[str]:
        """Return the subset of ids not already recorded for this target."""
        ids = list(follower_ids)
        if not ids:
            return set()
        unseen: set[str] = set(ids)
        # Chunk to stay well under SQLite's variable limit.
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT follower_id FROM seen_followers "
                f"WHERE target_id = ? AND follower_id IN ({placeholders})",
                (target_id, *chunk),
            )
            for row in rows:
                unseen.discard(row["follower_id"])
        return unseen

    def record_seen(self, target_id: int, follower_ids: Iterable[str]) -> int:
        now = utcnow()
        rows = [(target_id, fid, now) for fid in follower_ids]
        if not rows:
            return 0
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO seen_followers (target_id, follower_id, first_seen_at) "
            "VALUES (?, ?, ?)",
            rows,
        )
        return cur.rowcount or 0

    def prune_seen(self, target_id: int, keep: int) -> int:
        """Keep only the `keep` most recently seen ids for a target."""
        total = self.seen_count(target_id)
        if total <= keep:
            return 0
        cur = self.conn.execute(
            """
            DELETE FROM seen_followers
             WHERE target_id = ?
               AND follower_id IN (
                   SELECT follower_id FROM seen_followers
                    WHERE target_id = ?
                    ORDER BY first_seen_at ASC, follower_id ASC
                    LIMIT ?
               )
            """,
            (target_id, target_id, total - keep),
        )
        return cur.rowcount or 0

    def seen_count(self, target_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS c FROM seen_followers WHERE target_id = ?", (target_id,)
        ).fetchone()["c"]

    # --------------------------------------------------------------- accounts

    def upsert_account(self, account: dict[str, Any]) -> None:
        now = utcnow()
        self.conn.execute(
            """
            INSERT INTO accounts (
                id, username, name, bio, url, followers_count, following_count,
                statuses_count, favourites_count, created_at, verified, protected,
                has_default_avatar, profile_image_url, raw_json,
                first_seen_at, last_refreshed_at
            ) VALUES (
                :id, :username, :name, :bio, :url, :followers_count, :following_count,
                :statuses_count, :favourites_count, :created_at, :verified, :protected,
                :has_default_avatar, :profile_image_url, :raw_json, :now, :now
            )
            ON CONFLICT(id) DO UPDATE SET
                username           = excluded.username,
                name               = excluded.name,
                bio                = excluded.bio,
                url                = excluded.url,
                followers_count    = excluded.followers_count,
                following_count    = excluded.following_count,
                statuses_count     = excluded.statuses_count,
                favourites_count   = excluded.favourites_count,
                verified           = excluded.verified,
                protected          = excluded.protected,
                has_default_avatar = excluded.has_default_avatar,
                profile_image_url  = excluded.profile_image_url,
                raw_json           = excluded.raw_json,
                last_refreshed_at  = excluded.last_refreshed_at
            """,
            {**account, "now": now},
        )

    def set_account_url(self, account_id: str, url: str) -> None:
        """Store a resolved profile link. The original t.co stays in raw_json."""
        self.conn.execute("UPDATE accounts SET url = ? WHERE id = ?", (url, account_id))

    def get_account(self, account_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()

    def iter_accounts(self, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM accounts ORDER BY first_seen_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return list(self.conn.execute(sql))

    # --------------------------------------------------------------- verdicts

    def record_verdict(
        self,
        account_id: str,
        stage: str,
        passed: bool,
        score: float | None = None,
        reason: str | None = None,
        components: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO verdicts
                (account_id, stage, passed, score, reason, components_json, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, stage) DO UPDATE SET
                passed          = excluded.passed,
                score           = excluded.score,
                reason          = excluded.reason,
                components_json = excluded.components_json,
                decided_at      = excluded.decided_at
            """,
            (
                account_id,
                stage,
                int(passed),
                score,
                reason,
                json.dumps(components or {}, separators=(",", ":")),
                utcnow(),
            ),
        )

    def clear_verdicts(self, stages: Iterable[str] | None = None) -> int:
        """Used by `replay` to re-score history under new thresholds."""
        if stages is None:
            cur = self.conn.execute("DELETE FROM verdicts")
        else:
            stage_list = list(stages)
            placeholders = ",".join("?" * len(stage_list))
            cur = self.conn.execute(
                f"DELETE FROM verdicts WHERE stage IN ({placeholders})", stage_list
            )
        return cur.rowcount or 0

    def stage_counts(self) -> dict[str, dict[str, int]]:
        rows = self.conn.execute(
            "SELECT stage, passed, COUNT(*) AS c FROM verdicts GROUP BY stage, passed"
        )
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            bucket = out.setdefault(row["stage"], {"passed": 0, "failed": 0})
            bucket["passed" if row["passed"] else "failed"] += row["c"]
        return out

    # ---------------------------------------------------------- follow events

    def record_follow_event(
        self, account_id: str, target_id: int, page_pos: int | None = None
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO follow_events "
            "(account_id, target_id, detected_at, page_pos) VALUES (?, ?, ?, ?)",
            (account_id, target_id, utcnow(), page_pos),
        )

    def convergence(self, account_id: str, window_hours: int) -> tuple[int, list[str]]:
        """How many distinct tracked targets this account followed inside the window.

        The window is anchored at the account's LATEST event, not at wall-clock now.
        Live, the latest event is the one just detected so the two are identical; but
        anchoring this way makes `replay` over historical data give the same answer
        the live pipeline would have given, instead of silently dropping old events.

        Individuals tend to follow one ecosystem; projects doing launch prep follow
        several. Airdrop farmers also do, so treat it as a strong signal, not proof.
        """
        rows = list(
            self.conn.execute(
                """
                SELECT t.username AS username, fe.detected_at AS at
                  FROM follow_events fe
                  JOIN targets t ON t.id = fe.target_id
                 WHERE fe.account_id = ?
                 ORDER BY fe.detected_at
                """,
                (account_id,),
            )
        )
        if not rows:
            return 0, []
        anchor = parse_ts(rows[-1]["at"])
        if anchor is None:
            return len(rows), [r["username"] for r in rows]
        cutoff = anchor - timedelta(hours=window_hours)
        names = [
            r["username"]
            for r in rows
            if (parse_ts(r["at"]) or anchor) >= cutoff
        ]
        return len(names), names

    def targets_followed(self, account_id: str) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT t.username AS username FROM follow_events fe
              JOIN targets t ON t.id = fe.target_id
             WHERE fe.account_id = ? ORDER BY fe.detected_at
            """,
            (account_id,),
        )
        return [r["username"] for r in rows]

    # ------------------------------------------------------------- candidates

    def upsert_candidate(self, account_id: str, **fields: Any) -> None:
        now = utcnow()
        existing = self.conn.execute(
            "SELECT account_id FROM candidates WHERE account_id = ?", (account_id,)
        ).fetchone()
        if existing is None:
            cols = ["account_id", "first_seen_at", "updated_at", *fields.keys()]
            vals = [account_id, now, now, *fields.values()]
            placeholders = ",".join("?" * len(cols))
            self.conn.execute(
                f"INSERT INTO candidates ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )
        elif fields:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            self.conn.execute(
                f"UPDATE candidates SET {assignments}, updated_at = ? "
                f"WHERE account_id = ?",
                [*fields.values(), now, account_id],
            )

    def get_candidate(self, account_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM candidates WHERE account_id = ?", (account_id,)
        ).fetchone()

    def candidates_by_state(self, state: str, limit: int = 200) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT c.*, a.username, a.name, a.bio, a.url, a.followers_count,
                       a.following_count, a.created_at
                  FROM candidates c JOIN accounts a ON a.id = c.account_id
                 WHERE c.state = ? ORDER BY c.convergence_count DESC, c.score DESC
                 LIMIT ?
                """,
                (state, limit),
            )
        )

    def set_candidate_state(self, account_id: str, state: str) -> None:
        self.conn.execute(
            "UPDATE candidates SET state = ?, updated_at = ? WHERE account_id = ?",
            (state, utcnow(), account_id),
        )

    def candidate_state_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) AS c FROM candidates GROUP BY state"
        )
        return {r["state"]: r["c"] for r in rows}

    # --------------------------------------------------------------- poll log

    def start_poll(self, target_id: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO poll_log (target_id, started_at) VALUES (?, ?)",
            (target_id, utcnow()),
        )
        return int(cur.lastrowid)

    def finish_poll(
        self,
        poll_id: int,
        pages_fetched: int,
        items_returned: int,
        new_followers: int,
        credits_spent: float,
        hit_page_cap: bool = False,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE poll_log
               SET finished_at = ?, pages_fetched = ?, items_returned = ?,
                   new_followers = ?, credits_spent = ?, hit_page_cap = ?, error = ?
             WHERE id = ?
            """,
            (
                utcnow(),
                pages_fetched,
                items_returned,
                new_followers,
                credits_spent,
                int(hit_page_cap),
                error,
                poll_id,
            ),
        )

    def spend_since(self, hours: int = 24) -> dict[str, float]:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(credits_spent), 0) AS credits,
                   COALESCE(SUM(items_returned), 0) AS items,
                   COALESCE(SUM(new_followers), 0)  AS new_followers,
                   COUNT(*) AS polls,
                   COALESCE(SUM(hit_page_cap), 0)   AS capped
              FROM poll_log WHERE started_at >= ?
            """,
            (_hours_ago(hours),),
        ).fetchone()
        return {
            "credits": float(row["credits"]),
            "usd": float(row["credits"]) / 100_000.0,
            "items": int(row["items"]),
            "new_followers": int(row["new_followers"]),
            "polls": int(row["polls"]),
            "capped": int(row["capped"]),
        }

    def recent_new_rate(self, target_id: int, hours: int = 24) -> float | None:
        """Observed new followers per hour, from completed polls in the window."""
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(new_followers), 0) AS n,
                   MIN(started_at) AS first_at, MAX(started_at) AS last_at,
                   COUNT(*) AS polls
              FROM poll_log
             WHERE target_id = ? AND started_at >= ? AND finished_at IS NOT NULL
               AND error IS NULL
            """,
            (target_id, _hours_ago(hours)),
        ).fetchone()
        if not row or row["polls"] < 2:
            return None
        first, last = parse_ts(row["first_at"]), parse_ts(row["last_at"])
        if not first or not last:
            return None
        span_h = (last - first).total_seconds() / 3600.0
        if span_h <= 0:
            return None
        return float(row["n"]) / span_h

    # --------------------------------------------------------------- counters

    def bump_usage(self, kind: str, amount: int = 1) -> int:
        day = datetime.now(timezone.utc).date().isoformat()
        self.conn.execute(
            """
            INSERT INTO usage_counters (day, kind, count) VALUES (?, ?, ?)
            ON CONFLICT(day, kind) DO UPDATE SET count = count + excluded.count
            """,
            (day, kind, amount),
        )
        return self.usage_today(kind)

    def usage_today(self, kind: str) -> int:
        day = datetime.now(timezone.utc).date().isoformat()
        row = self.conn.execute(
            "SELECT count FROM usage_counters WHERE day = ? AND kind = ?", (day, kind)
        ).fetchone()
        return int(row["count"]) if row else 0
