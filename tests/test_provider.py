"""Adapter tests against a mocked HTTP transport -- no network, no credits."""

from __future__ import annotations

import httpx
import pytest

from scout.providers import twitterapi as ta
from scout.providers.base import ProviderError, ProviderUnavailable, UserNotFound


def make_provider(handler, **kw) -> ta.TwitterApiIo:
    client = httpx.Client(
        base_url=ta.BASE_URL,
        headers={"X-API-Key": "k"},
        transport=httpx.MockTransport(handler),
    )
    return ta.TwitterApiIo("k", min_request_interval_s=0, client=client, **kw)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ta.time, "sleep", lambda _s: None)


def follower(i: int, **over) -> dict:
    base = {
        "id": str(i),
        "userName": f"user{i}",
        "name": f"User {i}",
        "description": "building the protocol",
        "url": "https://example.xyz",
        "followers_count": 120,
        "following_count": 80,
        "statuses_count": 14,
        "favourites_count": 3,
        "created_at": "Wed Oct 10 20:19:24 +0000 2018",
        "protected": False,
        "profile_image_url_https": "https://pbs.twimg.com/profile_images/1/x.jpg",
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ dates


@pytest.mark.parametrize(
    "value",
    [
        "Wed Oct 10 20:19:24 +0000 2018",  # classic X format
        "2018-10-10T20:19:24.000Z",  # ISO with Z
        "2018-10-10T20:19:24+00:00",  # ISO with offset
    ],
)
def test_parse_both_endpoint_date_formats(value):
    assert ta.parse_x_datetime(value) == "2018-10-10T20:19:24+00:00"


def test_parse_garbage_date_is_none_not_crash():
    assert ta.parse_x_datetime("not a date") is None
    assert ta.parse_x_datetime(None) is None
    assert ta.parse_x_datetime("") is None


# ----------------------------------------------------------- normalization


def test_normalize_followers_endpoint_shape():
    acc = ta.normalize_follower(follower(7))
    assert acc["id"] == "7" and acc["username"] == "user7"
    assert acc["followers_count"] == 120 and acc["following_count"] == 80
    assert acc["created_at"] == "2018-10-10T20:19:24+00:00"
    assert acc["has_default_avatar"] == 0 and acc["protected"] == 0


def test_normalize_user_info_field_names_also_handled():
    acc = ta.normalize_follower(
        {"id": "9", "userName": "x", "followers": 5, "following": 6,
         "createdAt": "2020-01-01T00:00:00.000Z", "statusesCount": 2,
         "isBlueVerified": True}
    )
    assert acc["followers_count"] == 5 and acc["following_count"] == 6
    assert acc["statuses_count"] == 2 and acc["verified"] == 1
    assert acc["created_at"] == "2020-01-01T00:00:00+00:00"


def test_default_avatar_detected():
    acc = ta.normalize_follower(
        follower(1, profile_image_url_https="https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png")
    )
    assert acc["has_default_avatar"] == 1


def test_missing_id_is_skipped():
    assert ta.normalize_follower({"userName": "ghost"}) is None


# ----------------------------------------------------------------- credits


@pytest.mark.parametrize(
    "n,expected",
    [(0, 60), (5, 60), (20, 60), (21, 63), (99, 297), (100, 200), (199, 398), (200, 200)],
)
def test_credit_estimate_matches_vendor_tiers(n, expected):
    assert ta.estimate_credits(n) == expected


# -------------------------------------------------------------- followers


def test_get_followers_parses_page_and_cursor():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/twitter/user/followers"
        assert req.url.params["userName"] == "solana"
        assert req.url.params["pageSize"] == "200"
        assert req.headers["X-API-Key"] == "k"
        return httpx.Response(
            200,
            json={"status": "success", "followers": [follower(1), follower(2)],
                  "has_next_page": True, "next_cursor": "abc"},
        )

    page = make_provider(handler).get_followers("solana")
    assert [a["id"] for a in page.accounts] == ["1", "2"]
    assert page.has_next_page and page.next_cursor == "abc"
    assert page.credits_spent == 60  # 2 items hits the 60-credit floor


def test_last_page_has_no_cursor():
    def handler(_req):
        return httpx.Response(
            200, json={"status": "success", "followers": [follower(1)],
                       "has_next_page": False, "next_cursor": ""},
        )

    page = make_provider(handler).get_followers("solana")
    assert page.next_cursor is None and not page.has_next_page


def test_has_next_without_cursor_is_treated_as_end():
    # Defensive: never loop forever on a malformed pagination envelope.
    def handler(_req):
        return httpx.Response(
            200, json={"status": "success", "followers": [follower(1)],
                       "has_next_page": True, "next_cursor": ""},
        )

    assert make_provider(handler).get_followers("solana").has_next_page is False


def test_page_size_bounds_enforced_client_side():
    p = make_provider(lambda _r: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        p.get_followers("solana", page_size=10)
    with pytest.raises(ValueError):
        p.get_followers("solana", page_size=500)


# -------------------------------------------------------------- resolution


def test_resolve_user_ok():
    def handler(req):
        assert req.url.path == "/twitter/user/info"
        return httpx.Response(
            200, json={"status": "success",
                       "data": {"id": "123", "userName": "solana", "name": "Solana", "followers": 3_000_000}},
        )

    u = make_provider(handler).resolve_user("solana")
    assert u.id == "123" and u.followers_count == 3_000_000


def test_resolve_user_not_found_is_http_200_error_body():
    def handler(_req):
        return httpx.Response(200, json={"status": "error", "msg": "user not found"})

    with pytest.raises(UserNotFound):
        make_provider(handler).resolve_user("nope")


def test_resolve_suspended_user():
    def handler(_req):
        return httpx.Response(
            200, json={"status": "success",
                       "data": {"unavailable": True, "unavailableReason": "Suspended"}},
        )

    with pytest.raises(UserNotFound, match="Suspended"):
        make_provider(handler).resolve_user("banned")


# ------------------------------------------------------------ http failures


def test_retries_transient_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(_req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"status": "success", "followers": [], "has_next_page": False})

    make_provider(handler).get_followers("solana")
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    p = make_provider(lambda _r: httpx.Response(500), max_retries=2)
    with pytest.raises(ProviderUnavailable):
        p.get_followers("solana")


def test_bad_key_fails_fast_without_retry():
    calls = {"n": 0}

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(401)

    with pytest.raises(ProviderError, match="auth"):
        make_provider(handler).get_followers("solana")
    assert calls["n"] == 1


def test_out_of_credits_is_not_retried():
    calls = {"n": 0}

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(402)

    with pytest.raises(ProviderError, match="credits"):
        make_provider(handler).get_followers("solana")
    assert calls["n"] == 1


def test_missing_api_key_rejected_at_construction():
    with pytest.raises(ProviderError):
        ta.TwitterApiIo("")
