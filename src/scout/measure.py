"""Phase 0 measurement: verify the assumptions the design rests on, with real data.

Three things I could not know from documentation alone:
  1. Is the follower list really newest-first? (page-1 polling is only valid if so)
  2. How many new followers per hour does each target get? (sizes the funnel)
  3. How much of a real page survives stage 1/2, and does the provider return usable
     website URLs or unexpanded t.co links? (the strongest signal depends on it)
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .funnel import stage1_hard, stage2_score
from .funnel.common import age_days, classify_url
from .providers.base import Account


@dataclass(frozen=True)
class OrderingResult:
    verdict: str  # confirmed | violated | inconclusive_no_new | inconclusive_no_overlap
    new_count: int
    overlap: int
    detail: str

    @property
    def confirmed(self) -> bool:
        return self.verdict == "confirmed"


def check_ordering(first: list[str], second: list[str]) -> OrderingResult:
    """Given two snapshots of the same list taken some time apart, decide whether
    new entries arrived at the FRONT and the rest kept their relative order.

    That is the property page-1 polling depends on. A follower can legitimately
    vanish between snapshots (unfollow), so we test relative order of the common
    ids rather than demanding an exact shifted copy.
    """
    a_pos = {fid: i for i, fid in enumerate(first)}
    common_idx = [i for i, fid in enumerate(second) if fid in a_pos]
    if not common_idx:
        return OrderingResult(
            "inconclusive_no_overlap", len(second), 0,
            "snapshots share no ids: more arrived than one page holds. Poll sooner or "
            "use a larger window; this is also what a page-cap overflow looks like.",
        )
    first_common = common_idx[0]
    new_ids = [fid for fid in second if fid not in a_pos]
    if not new_ids:
        return OrderingResult("inconclusive_no_new", 0, len(common_idx),
                              "no new followers in the interval; wait longer and retry.")

    # New entries must all sit before the first previously-seen entry.
    scattered = [fid for i, fid in enumerate(second) if fid not in a_pos and i > first_common]
    # The previously-seen entries must keep their old relative order.
    order = [a_pos[second[i]] for i in common_idx]
    reordered = any(x > y for x, y in zip(order, order[1:]))

    if scattered or reordered:
        why = []
        if scattered:
            why.append(f"{len(scattered)} new id(s) appear AFTER known ones")
        if reordered:
            why.append("known ids changed relative order")
        return OrderingResult("violated", len(new_ids), len(common_idx), "; ".join(why))
    return OrderingResult("confirmed", len(new_ids), len(common_idx),
                          f"{len(new_ids)} new at the front, {len(common_idx)} known in original order")


@dataclass
class PageStats:
    total: int = 0
    stage1_pass: int = 0
    gate_pass: int = 0
    url_kinds: Counter = field(default_factory=Counter)
    stage1_reasons: Counter = field(default_factory=Counter)
    missing_created_at: int = 0
    default_avatars: int = 0
    median_age_days: float | None = None
    median_followers: float | None = None

    def pct(self, n: int) -> float:
        return round(100.0 * n / self.total, 1) if self.total else 0.0


def analyse_page(accounts: list[Account], filters: dict[str, Any]) -> PageStats:
    s = PageStats(total=len(accounts))
    s2 = filters.get("stage2", {})
    ages, followers = [], []
    for a in accounts:
        s.url_kinds[classify_url(a.get("url"), s2)] += 1
        s.missing_created_at += a.get("created_at") is None
        s.default_avatars += bool(a.get("has_default_avatar"))
        age = age_days(a.get("created_at"))
        if age is not None:
            ages.append(age)
        followers.append(int(a.get("followers_count") or 0))

        v1 = stage1_hard.evaluate(a, filters)
        if not v1.passed:
            s.stage1_reasons[v1.reason or "?"] += 1
            continue
        s.stage1_pass += 1
        if stage2_score.evaluate(a, filters).passed:
            s.gate_pass += 1
    s.median_age_days = round(statistics.median(ages), 1) if ages else None
    s.median_followers = statistics.median(followers) if followers else None
    return s


def render(stats: PageStats, *, target: str) -> str:
    lines = [
        f"=== @{target}: one page of {stats.total} newest followers ===",
        f"  stage 1 survival : {stats.stage1_pass}/{stats.total} ({stats.pct(stats.stage1_pass)}%)   [plan assumed 10-20%]",
        f"  past stage 2     : {stats.gate_pass}/{stats.total} ({stats.pct(stats.gate_pass)}%)   [goes on to the LLM]",
        f"  median age       : {stats.median_age_days} days   median followers: {stats.median_followers}",
        f"  default avatars  : {stats.default_avatars}   missing created_at: {stats.missing_created_at}",
        "  profile link kinds:",
    ]
    for kind, n in stats.url_kinds.most_common():
        lines.append(f"      {kind:<11} {n:>4} ({stats.pct(n)}%)")
    if stats.url_kinds.get("short", 0) and stats.url_kinds["short"] >= 0.5 * max(1, stats.total - stats.url_kinds.get("none", 0)):
        lines.append("  !! Most links are unexpanded t.co shorteners: the 'custom domain' signal is blind.")
        lines.append("     Resolve them (HEAD request) or use /twitter/user/info per candidate before scoring.")
    lines.append("  stage 1 rejection reasons:")
    for reason, n in stats.stage1_reasons.most_common(8):
        lines.append(f"      {reason:<24} {n:>4}")
    return "\n".join(lines)


def expand_links(accounts: list[Account], filters: dict[str, Any], expand: Any) -> int:
    """Resolve t.co profile links in place so the page can be analysed truthfully.
    Mirrors the live pipeline's skip rules, so the numbers match what it would do."""
    s1, s2 = filters.get("stage1", {}), filters.get("stage2", {})
    cap = s1.get("max_following")
    done = 0
    for a in accounts:
        if classify_url(a.get("url"), s2) != "short" or a.get("protected") or a.get("has_default_avatar"):
            continue
        if cap is not None and int(a.get("following_count") or 0) > int(cap):
            continue
        dest = expand(a["url"])
        if dest:
            a["url"] = dest
            done += 1
    return done
