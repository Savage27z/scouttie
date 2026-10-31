"""Runs the free funnel stages (1-3) over accounts and records every verdict.

Stages 4-5 (LLM) are separate because they cost money and are rate-capped; this
module is pure and offline, which is what makes `replay` free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import Config
from .db import Database
from .funnel import stage1_hard, stage2_score, stage3_convergence
from .funnel.common import as_dict, classify_url


Expander = Callable[[str], "str | None"]


def expand_link(db: Database, cfg: Config, a: dict[str, Any], expand: Expander | None, persist: bool) -> dict[str, Any]:
    """Swap a t.co profile link for its destination.

    Must happen BEFORE stage 1: a brand-new project with a real domain but <10
    followers is only exempt from the follower floor once we can see the domain.
    Skipped when stage 1 will reject for reasons a resolved link cannot change, so
    we don't make HTTP requests for protected accounts, default avatars or farmers.
    """
    url = a.get("url") or ""
    if expand is None or classify_url(url, cfg.stage2()) != "short":
        return a
    if a.get("protected") or a.get("has_default_avatar"):
        return a
    max_following = cfg.stage1().get("max_following")
    if max_following is not None and int(a.get("following_count") or 0) > int(max_following):
        return a
    dest = expand(url)
    if not dest:
        return a
    if persist:
        db.set_account_url(str(a["id"]), dest)
    return {**a, "url": dest}


@dataclass
class Decision:
    account_id: str
    username: str = ""
    passed_gate: bool = False
    stage1_passed: bool = False
    stage2_passed: bool = False
    score: float | None = None
    reason: str | None = None
    convergence_count: int = 0
    targets: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


def judge(
    db: Database,
    cfg: Config,
    account: Any,
    *,
    persist: bool = True,
    now: datetime | None = None,
    expand: Expander | None = None,
) -> Decision:
    """Stages 1-3 for one account. With persist=False nothing is written (replay
    dry-run). Stage 3 runs first because its count feeds stage 2's bonus."""
    a = expand_link(db, cfg, as_dict(account), expand, persist)
    aid = str(a["id"])
    filters = cfg.filters

    v3, targets = stage3_convergence.evaluate(db, aid, filters)
    count = len(targets)
    d = Decision(aid, a.get("username") or "", convergence_count=count, targets=targets)

    v1 = stage1_hard.evaluate(a, filters)
    d.stage1_passed = v1.passed
    if persist:
        db.record_verdict(aid, "stage3", v3.passed, v3.score, v3.reason, v3.components)
        db.record_verdict(aid, "stage1", v1.passed, v1.score, v1.reason, v1.components)
    if not v1.passed:
        d.reason = f"stage1: {v1.reason}"
        return d

    v2 = stage2_score.evaluate(a, filters, convergence_count=max(count, 1), now=now)
    d.stage2_passed = v2.passed
    d.score = v2.score
    d.components = v2.components
    if persist:
        db.record_verdict(aid, "stage2", v2.passed, v2.score, v2.reason, v2.components)
    if not v2.passed:
        d.reason = f"stage2: {v2.reason}"
        return d

    d.passed_gate = True
    if persist:
        # upsert never resets `state`, so a re-judged account that was already
        # alerted/muted/rejected keeps its status.
        db.upsert_candidate(aid, convergence_count=count, score=v2.score)
    return d


@dataclass
class ReplayReport:
    total: int = 0
    stage1_pass: int = 0
    gate_pass: int = 0
    previously_gate_pass: int = 0
    flipped_in: list[tuple[str, str, float | None]] = field(default_factory=list)
    flipped_out: list[tuple[str, str, float | None]] = field(default_factory=list)


def replay(db: Database, cfg: Config, *, apply: bool = False, limit: int | None = None) -> ReplayReport:
    """Re-score stored accounts under the CURRENT filters.yaml. No API calls.

    Dry-run by default: reports how many accounts would pass and which flip, so a
    threshold change can be judged before it changes live alert volume.
    """
    rep = ReplayReport()
    prior = {
        r["account_id"]: bool(r["passed"])
        for r in db.conn.execute("SELECT account_id, passed FROM verdicts WHERE stage = 'stage2'")
    }
    for row in db.iter_accounts(limit):
        d = judge(db, cfg, row, persist=apply)
        rep.total += 1
        rep.stage1_pass += d.stage1_passed
        rep.gate_pass += d.passed_gate
        was = prior.get(d.account_id, False)
        rep.previously_gate_pass += was
        if d.passed_gate and not was:
            rep.flipped_in.append((d.account_id, d.username, d.score))
        elif was and not d.passed_gate:
            rep.flipped_out.append((d.account_id, d.username, d.score))
    return rep
