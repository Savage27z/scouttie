from __future__ import annotations

import json

import pytest

from scout.config import TargetSpec
from scout.db import Database
from scout.providers.base import Account, FollowerPage, ResolvedUser


def make_account(i: int, **over) -> Account:
    acc: Account = {
        "id": str(i),
        "username": f"user{i}",
        "name": f"User {i}",
        "bio": "",
        "url": "",
        "followers_count": 100,
        "following_count": 100,
        "statuses_count": 10,
        "favourites_count": 10,
        "created_at": "2024-01-01T00:00:00+00:00",
        "verified": 0,
        "protected": 0,
        "has_default_avatar": 0,
        "profile_image_url": "https://pbs.twimg.com/x.jpg",
        "raw_json": "{}",
    }
    acc.update(over)
    acc["raw_json"] = json.dumps(acc.get("raw_json", "{}"))
    return acc


class FakeProvider:
    """In-memory follower list, newest first. Paginates by integer cursor."""

    def __init__(self, followers: list[Account]):
        self.followers = list(followers)
        self.calls = 0
        self.fail_on_call: int | None = None

    def prepend(self, accounts: list[Account]) -> None:
        """New followers arrive at the FRONT of a newest-first list."""
        self.followers = list(accounts) + self.followers

    def resolve_user(self, username: str) -> ResolvedUser:
        return ResolvedUser(id="1", username=username, name=username, followers_count=len(self.followers))

    def get_followers(self, username, cursor=None, page_size=200) -> FollowerPage:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("boom")
        start = int(cursor or 0)
        chunk = self.followers[start : start + page_size]
        end = start + page_size
        has_next = end < len(self.followers)
        return FollowerPage(
            accounts=chunk,
            next_cursor=str(end) if has_next else None,
            has_next_page=has_next,
            credits_spent=60.0,
            raw_count=len(chunk),
        )

    def close(self) -> None:  # pragma: no cover
        pass


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    yield d
    d.close()


@pytest.fixture
def spec():
    return TargetSpec(username="solana", page_size=20, max_pages_per_poll=5)


@pytest.fixture
def registered(db, spec):
    db.upsert_target(spec.username, 3600, True)
    return spec
