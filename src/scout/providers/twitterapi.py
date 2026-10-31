"""twitterapi.io adapter.

Verified against the vendor docs (not yet against live responses -- `scout measure`
is the live check):

* GET /twitter/user/followers  -- newest-follow-first, pageSize 20-200, cursor paging.
  Billed per follower returned: 1 credit/item at 200, 2 at 100-199, 3 at 20-99,
  with a 60-credit minimum per call. 100,000 credits = $1.
* GET /twitter/user/info       -- always HTTP 200; check `status` in the body.
  NOTE the field names differ from the followers endpoint (`followers`/`following`/
  `createdAt` vs `followers_count`/`following_count`/`created_at`), so both shapes
  are normalized here.

QPS scales with account balance (free: one request per 5 seconds), so the client
throttles itself. Set SCOUT_MIN_REQUEST_INTERVAL_S lower once you hold a balance.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .base import (
    Account,
    FollowerPage,
    ProviderError,
    ProviderUnavailable,
    ResolvedUser,
    UserNotFound,
)

log = logging.getLogger(__name__)

BASE_URL = "https://api.twitterapi.io"
MIN_CALL_CREDITS = 60


def estimate_credits(items_returned: int) -> float:
    """Credits billed for one followers call, per the vendor's tiered pricing."""
    if items_returned <= 0:
        return float(MIN_CALL_CREDITS)
    if items_returned >= 200:
        per_item = 1
    elif items_returned >= 100:
        per_item = 2
    else:
        per_item = 3
    return float(max(MIN_CALL_CREDITS, per_item * items_returned))


def parse_x_datetime(value: Any) -> str | None:
    """Parse either ISO-8601 or the classic X format into ISO-8601 UTC.

    The two endpoints disagree on format, and a silent parse failure would turn
    every account into 'age unknown', quietly disabling the young-account signal.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)  # "Wed Oct 10 20:19:24 +0000 2018"
        except (TypeError, ValueError):
            try:
                dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _is_default_avatar(url: str | None) -> bool:
    return bool(url) and "default_profile" in url


def normalize_follower(raw: dict[str, Any]) -> Account | None:
    """Followers-endpoint profile -> normalized account. None if there is no id."""
    uid = raw.get("id") or raw.get("id_str")
    if not uid:
        return None
    avatar = raw.get("profile_image_url_https") or raw.get("profilePicture")
    return {
        "id": str(uid),
        "username": raw.get("userName") or raw.get("screen_name") or "",
        "name": raw.get("name") or "",
        "bio": raw.get("description") or "",
        "url": raw.get("url") or "",
        "followers_count": _int(raw.get("followers_count", raw.get("followers"))),
        "following_count": _int(
            raw.get("following_count", raw.get("friends_count", raw.get("following")))
        ),
        "statuses_count": _int(raw.get("statuses_count", raw.get("statusesCount"))),
        "favourites_count": _int(raw.get("favourites_count", raw.get("favouritesCount"))),
        "created_at": parse_x_datetime(raw.get("created_at") or raw.get("createdAt")),
        "verified": int(bool(raw.get("verified") or raw.get("isBlueVerified"))),
        "protected": int(bool(raw.get("protected"))),
        "has_default_avatar": int(_is_default_avatar(avatar)),
        "profile_image_url": avatar or "",
        "raw_json": json.dumps(raw, separators=(",", ":"), default=str),
    }


class TwitterApiIo:
    def __init__(
        self,
        api_key: str,
        min_request_interval_s: float = 5.0,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise ProviderError("TWITTERAPI_KEY is not set")
        self._min_interval = max(0.0, min_request_interval_s)
        self._max_retries = max(1, max_retries)
        self._last_request = 0.0
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            headers={"X-API-Key": api_key, "Accept": "application/json"},
            timeout=timeout_s,
        )

    # ----------------------------------------------------------------- http

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            try:
                resp = self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning("request error on %s (attempt %d): %s", path, attempt, exc)
                self._backoff(attempt)
                continue
            finally:
                self._last_request = time.monotonic()

            if resp.status_code in (401, 403):
                raise ProviderError(
                    f"auth rejected ({resp.status_code}) -- check TWITTERAPI_KEY"
                )
            if resp.status_code == 402:
                raise ProviderError("out of credits (402) -- top up twitterapi.io")
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = ProviderUnavailable(f"HTTP {resp.status_code} from {path}")
                log.warning("HTTP %d on %s (attempt %d)", resp.status_code, path, attempt)
                self._backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                continue
            if resp.status_code >= 400:
                raise ProviderError(f"HTTP {resp.status_code} from {path}: {resp.text[:200]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise ProviderError(f"non-JSON response from {path}") from exc
        raise ProviderUnavailable(
            f"{path} failed after {self._max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> None:
        delay = 2.0**attempt
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(min(delay, 60.0))

    # ------------------------------------------------------------ interface

    def resolve_user(self, username: str) -> ResolvedUser:
        body = self._get("/twitter/user/info", {"userName": username})
        if body.get("status") != "success":
            msg = str(body.get("msg") or body.get("message") or "unknown error")
            if "not found" in msg.lower():
                raise UserNotFound(f"@{username}: {msg}")
            raise ProviderError(f"@{username}: {msg}")
        data = body.get("data") or {}
        if data.get("unavailable"):
            reason = data.get("unavailableReason") or data.get("message") or "unavailable"
            raise UserNotFound(f"@{username} is unavailable ({reason})")
        if not data.get("id"):
            raise ProviderError(f"@{username}: response had no id")
        return ResolvedUser(
            id=str(data["id"]),
            username=str(data.get("userName") or username),
            name=str(data.get("name") or ""),
            followers_count=_int(data.get("followers")),
        )

    def get_followers(
        self, username: str, cursor: str | None = None, page_size: int = 200
    ) -> FollowerPage:
        if not 20 <= page_size <= 200:
            raise ValueError("page_size must be between 20 and 200")
        body = self._get(
            "/twitter/user/followers",
            {"userName": username, "cursor": cursor or "", "pageSize": page_size},
        )
        if body.get("status") not in (None, "success"):
            raise ProviderError(
                f"@{username} followers: {body.get('msg') or body.get('message')}"
            )
        raw_items = body.get("followers") or []
        accounts: list[Account] = []
        for item in raw_items:
            norm = normalize_follower(item)
            if norm is not None:
                accounts.append(norm)
        next_cursor = body.get("next_cursor") or None
        has_next = bool(body.get("has_next_page")) and bool(next_cursor)
        return FollowerPage(
            accounts=accounts,
            next_cursor=next_cursor if has_next else None,
            has_next_page=has_next,
            credits_spent=estimate_credits(len(raw_items)),
            raw_count=len(raw_items),
            meta={"dropped_no_id": len(raw_items) - len(accounts)},
        )

    def close(self) -> None:
        self._client.close()
