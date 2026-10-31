"""Provider interface.

Everything vendor-specific lives behind this protocol so the X data source can be
swapped (twitterapi.io today, something else if pricing or behaviour changes)
without touching the collector or funnel.

The funnel only ever sees `Account` dicts in the normalized shape below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """The provider failed in a way retrying will not fix (bad key, bad handle)."""


class ProviderUnavailable(ProviderError):
    """Transient failure after retries (network, 5xx, rate limit)."""


class UserNotFound(ProviderError):
    """The handle does not exist, or is suspended/unavailable."""


# Normalized account shape. Keys match the `accounts` table columns that
# `Database.upsert_account` expects.
Account = dict[str, Any]


@dataclass(frozen=True)
class FollowerPage:
    """One page of followers, newest first."""

    accounts: list[Account]
    next_cursor: str | None
    has_next_page: bool
    credits_spent: float = 0.0
    raw_count: int = 0
    # Anything the adapter wants to surface for `scout measure` diagnostics.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedUser:
    id: str
    username: str
    name: str
    followers_count: int


class XProvider(Protocol):
    def resolve_user(self, username: str) -> ResolvedUser:
        """Validate a handle and return its numeric id. Raises UserNotFound."""

    def get_followers(
        self, username: str, cursor: str | None = None, page_size: int = 200
    ) -> FollowerPage:
        """One page of followers, sorted newest-follow-first."""

    def close(self) -> None: ...
