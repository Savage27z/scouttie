"""Runs LLM stages 4-5 over candidates that passed the free gate.

Spend is bounded by per-day counters from filters.yaml. When a cap is hit the work
is deferred to the next day (candidates stay 'new'); it is never silently skipped
and never overspends.

Candidate states:
  new         passed stage 1-2, awaiting LLM
  rejected    LLM says not a project (silent, kept for audit)
  ready       LLM says project; awaiting alert routing
  alerted     sent as an instant alert
  digested    sent in the daily digest
  interested / not_project / muted     set by the human via Telegram buttons
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .config import Config
from .db import Database
from .funnel import llm
from .webfetch import fetch_text

log = logging.getLogger(__name__)

Fetcher = Callable[[str], "str | None"]


def pending_candidates(db: Database, limit: int) -> list[dict[str, Any]]:
    rows = db.conn.execute(
        """
        SELECT c.account_id, c.convergence_count, c.score,
               a.username, a.name, a.bio, a.url, a.followers_count,
               a.following_count, a.statuses_count, a.created_at
          FROM candidates c JOIN accounts a ON a.id = c.account_id
         WHERE c.state = 'new' AND c.llm_is_project IS NULL
         ORDER BY c.convergence_count DESC, c.score DESC
         LIMIT ?
        """,
        (limit,),
    )
    return [dict(r) for r in rows]


def enrich_pending(
    db: Database,
    cfg: Config,
    client: Any,
    *,
    fetcher: Fetcher = fetch_text,
    batch: int = 25,
) -> dict[str, int]:
    """Classify (stage 4) then summarise (stage 5) pending candidates."""
    lcfg = cfg.llm()
    classify_model = lcfg.get("classify_model", "deepseek-flash")
    summarize_model = lcfg.get("summarize_model", "deepseek-flash")
    max_classify = int(lcfg.get("max_classify_per_day", 600))
    max_summarize = int(lcfg.get("max_summarize_per_day", 80))
    min_conf = float(lcfg.get("min_confidence_to_summarize", 0.6))

    suppress_memes = bool(lcfg.get("suppress_memecoins", True))
    # Only suppress when the model is sure. A shaky memecoin call must not hide a lead
    # (live data: a crowd-launch platform was labelled memecoin at 0.55).
    meme_min_conf = float(lcfg.get("memecoin_suppress_min_confidence", 0.7))
    stats = {"classified": 0, "projects": 0, "summarized": 0, "deferred_cap": 0,
             "failed": 0, "memecoins_suppressed": 0}

    for cand in pending_candidates(db, batch):
        if db.usage_today("classify") >= max_classify:
            stats["deferred_cap"] += 1
            log.warning("daily classify cap (%d) reached; deferring remaining candidates", max_classify)
            break

        # Count the attempt before the call so a crash loop cannot overspend.
        db.bump_usage("classify")
        result = llm.classify(client, classify_model, cand)
        if result is None:
            stats["failed"] += 1
            continue  # stays 'new'; retried next cycle
        stats["classified"] += 1

        fields: dict[str, Any] = {
            "llm_is_project": int(result.is_project),
            "llm_confidence": result.confidence,
            "llm_category": result.category,
            "llm_chain": result.chain,
        }

        if not result.is_project:
            db.upsert_candidate(cand["account_id"], state="rejected", **fields)
            continue

        if suppress_memes and result.category == "memecoin" and result.confidence >= meme_min_conf:
            # Stored with its LLM verdict for audit, but never alerted or summarised.
            stats["memecoins_suppressed"] += 1
            db.upsert_candidate(cand["account_id"], state="rejected", **fields)
            continue

        stats["projects"] += 1
        summary = None
        if result.confidence >= min_conf and db.usage_today("summarize") < max_summarize:
            db.bump_usage("summarize")
            site = fetcher(cand["url"]) if cand.get("url") else None
            summary = llm.summarize(client, summarize_model, cand, site)
            if summary:
                stats["summarized"] += 1
        if summary:
            fields["llm_summary"] = summary.summary
            fields["llm_stack"] = summary.stack_hint
            fields["llm_contracts"] = summary.has_own_contracts
        db.upsert_candidate(cand["account_id"], state="ready", **fields)

    return stats
