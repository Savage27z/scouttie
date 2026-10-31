"""Stage 2 -- project-shape scoring. Free.

Score-based rather than hard-drop, because legitimate new project accounts are
young and small. Every component is stored in the verdict so thresholds can be
retuned against real data with `scout replay`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .common import (
    Verdict,
    age_days,
    as_dict,
    classify_url,
    contains_term,
    count_terms,
    domains_in_text,
    has_contract_address,
    host_of,
    split_camel,
)

_TICKER = re.compile(r"\$[A-Za-z]{2,10}\b")
_CAMEL_INTERNAL = re.compile(r"[a-z][A-Z]")
_HUMAN_NAME = re.compile(r"^[A-Z][a-z]{1,}(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]{1,})$")


def _brand_shaped(name: str, tokens: list[str]) -> bool:
    spaced = split_camel(name)
    if any(contains_term(spaced, t) for t in tokens):
        return True
    # single token with internal capitals: "SolFlare", "DeFiLlama"
    return " " not in name.strip() and bool(_CAMEL_INTERNAL.search(name))


def _human_shaped(name: str, tokens: list[str]) -> bool:
    name = (name or "").strip()
    if not _HUMAN_NAME.match(name):
        return False
    return not any(contains_term(name, t) for t in tokens)


def evaluate(
    account: Any,
    filters: dict[str, Any],
    *,
    convergence_count: int = 1,
    now: datetime | None = None,
) -> Verdict:
    a = as_dict(account)
    s2 = filters.get("stage2", {})
    w = s2.get("weights", {})

    name = a.get("name") or ""
    bio = a.get("bio") or ""
    url = a.get("url") or ""
    followers = int(a.get("followers_count") or 0)
    following = int(a.get("following_count") or 0)

    parts: dict[str, float] = {}

    # --- website: the strongest discriminator ------------------------------
    kind = classify_url(url, s2)
    if kind == "custom":
        parts["custom_domain_url"] = w.get("custom_domain_url", 4.0)
    elif kind == "aggregator":
        parts["link_aggregator_url"] = w.get("link_aggregator_url", -1.0)
    elif kind == "short":
        parts["unresolved_short_url"] = w.get("unresolved_short_url", 1.5)
    elif kind == "code_host":
        parts["code_host_url"] = w.get("code_host_url", 1.0)

    # A domain mentioned in the bio, distinct from the profile link host.
    link_host = host_of(url)
    aggregators = s2.get("link_aggregator_domains", [])
    bio_domains = [
        d
        for d in domains_in_text(bio)
        if d != link_host and not any(d == g or d.endswith("." + g) for g in aggregators)
    ]
    if bio_domains and kind != "custom":
        parts["domain_in_bio"] = w.get("domain_in_bio", 2.0)

    # --- bio keywords -------------------------------------------------------
    kw_hits = count_terms(bio, s2.get("bio_keywords", []))
    if kw_hits:
        parts["bio_keywords"] = min(
            kw_hits * w.get("bio_keyword_each", 1.0), w.get("bio_keyword_cap", 3.0)
        )

    # --- name shape ---------------------------------------------------------
    tokens = s2.get("brand_name_tokens", [])
    if _brand_shaped(name, tokens):
        parts["brand_shaped_name"] = w.get("brand_shaped_name", 2.0)
    if _TICKER.search(name):
        parts["ticker_in_name"] = w.get("ticker_in_name", 1.5)
    if _human_shaped(name, tokens):
        parts["human_name_shape"] = w.get("human_name_shape", -2.0)

    # --- age and audience: weak positives, never cutoffs --------------------
    age = age_days(a.get("created_at"), now)
    if age is not None and age <= s2.get("young_account_max_days", 90):
        parts["young_account"] = w.get("young_account", 1.5)
    if following >= 0 and followers / max(following, 1) >= s2.get(
        "healthy_follower_ratio_min", 1.0
    ):
        parts["healthy_follower_ratio"] = w.get("healthy_follower_ratio", 1.0)

    # --- memecoin / token-page markers --------------------------------------
    if has_contract_address(bio) or count_terms(bio, s2.get("memecoin_terms", [])):
        parts["memecoin_marker"] = w.get("memecoin_marker", -2.5)

    # --- personal markers ---------------------------------------------------
    marker_hits = count_terms(bio, s2.get("personal_markers", []))
    if marker_hits:
        parts["personal_markers"] = max(
            marker_hits * w.get("personal_marker_each", -2.5),
            w.get("personal_marker_cap", -5.0),
        )

    # --- convergence: a bonus, never a bypass -------------------------------
    extra = max(0, convergence_count - 1)
    if extra:
        parts["convergence_bonus"] = min(
            extra * w.get("convergence_bonus_per_extra_target", 2.5),
            w.get("convergence_bonus_cap", 5.0),
        )

    score = round(sum(parts.values()), 2)
    threshold = float(s2.get("threshold", 5.0))
    passed = score >= threshold
    return Verdict(
        passed,
        score,
        None if passed else f"score {score} < {threshold}",
        {k: round(v, 2) for k, v in parts.items()},
    )
