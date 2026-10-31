"""Stage 1 -- hard structural filters. Free.

Kills bots, lurkers and airdrop farmers. Deliberately has NO maximum account age
and no meaningful follower floor for accounts that link a real custom domain:
young + small + branded is the target, and thin-profile early teams are the best
leads because nobody else has found them yet.
"""

from __future__ import annotations

from typing import Any

from .common import Verdict, as_dict, classify_url


def evaluate(account: Any, filters: dict[str, Any]) -> Verdict:
    a = as_dict(account)
    s1 = filters.get("stage1", {})
    s2 = filters.get("stage2", {})

    bio = (a.get("bio") or "").strip()
    url = (a.get("url") or "").strip()
    followers = int(a.get("followers_count") or 0)
    following = int(a.get("following_count") or 0)
    statuses = int(a.get("statuses_count") or 0)
    url_kind = classify_url(url, s2)

    def reject(reason: str) -> Verdict:
        return Verdict(False, None, reason, {"url_kind": 0.0})

    if s1.get("drop_protected", True) and a.get("protected"):
        return reject("protected")
    if s1.get("drop_default_avatar", True) and a.get("has_default_avatar"):
        return reject("default_avatar")
    if s1.get("drop_if_no_bio_and_no_url", True) and not bio and not url:
        return reject("no_bio_no_url")

    max_following = s1.get("max_following")
    if max_following is not None and following > int(max_following):
        return reject(f"following>{max_following}")

    exempt = bool(s1.get("follower_floors_exempt_if_custom_domain", True)) and (
        url_kind == "custom"
    )
    if not exempt:
        min_followers = int(s1.get("min_followers", 0))
        if followers < min_followers:
            return reject(f"followers<{min_followers}")
        if statuses == 0 and followers < int(s1.get("min_followers_if_no_posts", 0)):
            return reject("silent_low_followers")

    return Verdict(True, None, "exempt_custom_domain" if exempt else None, {})
