"""Stage 3 -- convergence. Free.

How many distinct tracked ecosystems did this account follow inside the window?

This is a strong signal but NOT a definitive one. Airdrop farmers routinely follow
several ecosystem accounts at once, so convergence is fed into stage 2 as a score
bonus rather than letting it bypass scoring. It is also what gates *instant*
alerts, together with the LLM's confirmation that the account is a project.
"""

from __future__ import annotations

from typing import Any

from ..db import Database
from .common import Verdict


def evaluate(db: Database, account_id: str, filters: dict[str, Any]) -> tuple[Verdict, list[str]]:
    s3 = filters.get("stage3", {})
    window = int(s3.get("window_hours", 72))
    needed = int(s3.get("min_targets", 2))
    count, names = db.convergence(account_id, window)
    passed = count >= needed
    return (
        Verdict(
            passed,
            float(count),
            None if passed else f"{count} target(s) < {needed}",
            {"targets_in_window": float(count)},
        ),
        names,
    )
