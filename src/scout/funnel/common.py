"""Shared helpers for the funnel stages: verdicts, URL classification, term matching.

Matching is word-bounded on purpose. Plain substring matching misfires badly on
bios ("pgm " matching "gm ", "fi" inside "defi") and silently skews scores in ways
that are hard to notice later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class Verdict:
    passed: bool
    score: float | None = None
    reason: str | None = None
    components: dict[str, float] = field(default_factory=dict)


def as_dict(account: Any) -> dict[str, Any]:
    """Accept either a plain dict or a sqlite3.Row."""
    if isinstance(account, Mapping):
        return dict(account)
    return {k: account[k] for k in account.keys()}


# ------------------------------------------------------------------ matching


def _term_re(term: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![a-z0-9]){re.escape(term.strip().lower())}(?![a-z0-9])", re.IGNORECASE
    )


def contains_term(text: str, term: str) -> bool:
    return bool(text) and _term_re(term).search(text) is not None


def count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if contains_term(text, t))


_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def split_camel(name: str) -> str:
    """'UniswapLabs' -> 'Uniswap Labs' so brand tokens can be word-matched."""
    return _CAMEL_SPLIT.sub(" ", name or "")


# ---------------------------------------------------------------------- URLs


def host_of(url: str | None) -> str:
    if not url:
        return ""
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        host = urlparse(raw).hostname or ""
    except ValueError:
        return ""
    host = host.lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def _host_matches(host: str, domains: list[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def classify_url(url: str | None, stage2_cfg: dict[str, Any]) -> str:
    """none | short | aggregator | code_host | custom

    'short' means an unexpanded shortener (t.co): we cannot tell what it points at,
    which is different from knowing it is a Linktree. Keep those apart.
    """
    host = host_of(url)
    if not host or "." not in host:
        return "none"
    if _host_matches(host, stage2_cfg.get("short_url_domains", [])):
        return "short"
    if _host_matches(host, stage2_cfg.get("link_aggregator_domains", [])):
        return "aggregator"
    if _host_matches(host, stage2_cfg.get("code_host_domains", [])):
        return "code_host"
    return "custom"


_TLDS = (
    "com|io|xyz|fi|finance|org|app|network|dev|ai|sol|so|gg|exchange|money|cash|"
    "tech|co|net|build|fun|lol|wtf|zone|systems|labs|capital|fund|trade|bet|vc"
)
_DOMAIN_IN_TEXT = re.compile(
    rf"(?<![@\w./-])((?:[a-z0-9][a-z0-9-]*\.)+(?:{_TLDS}))(?![\w-])", re.IGNORECASE
)


def domains_in_text(text: str) -> list[str]:
    """Domain-looking tokens in a bio, excluding emails and @handles."""
    if not text:
        return []
    return [m.group(1).lower() for m in _DOMAIN_IN_TEXT.finditer(text)]


_SOL_ADDR = re.compile(r"(?<![1-9A-HJ-NP-Za-km-z])[1-9A-HJ-NP-Za-km-z]{32,44}(?![1-9A-HJ-NP-Za-km-z])")
_EVM_ADDR = re.compile(r"\b0x[0-9a-fA-F]{40}\b")


def has_contract_address(text: str) -> bool:
    """A Solana (base58) or EVM address pasted in a bio: a strong token-page tell.
    Long all-letter words can match base58, so require a digit or mixed case."""
    if not text:
        return False
    if _EVM_ADDR.search(text):
        return True
    for m in _SOL_ADDR.finditer(text):
        tok = m.group(0)
        if any(c.isdigit() for c in tok) and any(c.isupper() for c in tok) and any(c.islower() for c in tok):
            return True
    return False


# --------------------------------------------------------------------- dates


def age_days(created_at: str | None, now: datetime | None = None) -> float | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 86400.0
