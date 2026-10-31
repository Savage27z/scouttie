from __future__ import annotations

import pytest

from conftest import make_account
from scout import measure
from scout.config import load_config

FILTERS = load_config().filters


def ids(*xs):
    return [str(x) for x in xs]


def test_confirmed_when_new_arrive_at_front_in_order():
    first = ids(10, 9, 8, 7, 6)
    second = ids(13, 12, 11, 10, 9, 8, 7)
    r = measure.check_ordering(first, second)
    assert r.confirmed and r.new_count == 3 and r.overlap == 4


def test_confirmed_tolerates_an_unfollow_between_snapshots():
    first = ids(10, 9, 8, 7, 6)
    second = ids(12, 11, 10, 8, 7, 6)  # 9 unfollowed
    assert measure.check_ordering(first, second).confirmed


def test_violated_when_new_ids_are_scattered_through_the_list():
    """If the vendor ordered by something else (e.g. user id), new entries land
    mid-list. This is the go/no-go failure the whole approach hinges on."""
    first = ids(10, 9, 8, 7, 6)
    second = ids(10, 9, 99, 8, 7, 6)
    r = measure.check_ordering(first, second)
    assert r.verdict == "violated" and "AFTER" in r.detail


def test_violated_when_known_ids_reorder():
    r = measure.check_ordering(ids(1, 2, 3, 4), ids(9, 3, 2, 1))
    assert r.verdict == "violated" and "relative order" in r.detail


def test_no_new_followers_is_inconclusive_not_confirmed():
    r = measure.check_ordering(ids(3, 2, 1), ids(3, 2, 1))
    assert r.verdict == "inconclusive_no_new" and not r.confirmed


def test_no_overlap_is_inconclusive_not_violated():
    r = measure.check_ordering(ids(3, 2, 1), ids(9, 8, 7))
    assert r.verdict == "inconclusive_no_overlap" and not r.confirmed


def test_analyse_page_counts_and_flags_blind_t_co_signal():
    page = [make_account(i, bio="hello", url="https://t.co/abc") for i in range(1, 11)]
    page += [make_account(100, bio="x", url="https://good.xyz", followers_count=2)]
    stats = measure.analyse_page(page, FILTERS)
    assert stats.total == 11 and stats.url_kinds["short"] == 10 and stats.url_kinds["custom"] == 1
    out = measure.render(stats, target="solana")
    assert "t.co" in out and "blind" in out


def test_analyse_page_empty_is_safe():
    stats = measure.analyse_page([], FILTERS)
    assert stats.total == 0 and stats.pct(0) == 0.0
    assert "0 newest" in measure.render(stats, target="x")
