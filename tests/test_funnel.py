"""Funnel tests over SYNTHETIC profiles.

These check scoring mechanics and known edge cases. They do not measure real-world
precision -- the hand-labelled set in test_labelled_profiles is invented, so a pass
here means "the rules do what I intended", not "the rules find real projects".
Replace/extend the fixtures with real accounts captured by `scout measure`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import yaml

from conftest import make_account
from scout import config as cfgmod
from scout.funnel import stage1_hard, stage2_score, stage3_convergence
from scout.funnel.common import (
    classify_url,
    contains_term,
    domains_in_text,
    host_of,
    split_camel,
)

FILTERS = yaml.safe_load(cfgmod.DEFAULT_FILTERS.read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)


def days_ago(n: int) -> str:
    return (NOW - timedelta(days=n)).isoformat(timespec="seconds")


def run(acc, convergence: int = 1):
    """Stage 1 then stage 2, the way the pipeline does."""
    v1 = stage1_hard.evaluate(acc, FILTERS)
    if not v1.passed:
        return v1, None
    return v1, stage2_score.evaluate(acc, FILTERS, convergence_count=convergence, now=NOW)


# ------------------------------------------------------------------ helpers


def test_word_bounded_matching_avoids_substring_false_positives():
    assert contains_term("building on solana", "building")
    assert not contains_term("pgm trader", "gm")  # 'gm' inside a word
    assert not contains_term("defi protocols", "fi")
    assert contains_term("zk rollup", "zk")
    assert contains_term("Views My Own", "views my own")  # case-insensitive


def test_split_camel():
    assert split_camel("UniswapLabs") == "Uniswap Labs"
    assert split_camel("solana") == "solana"


@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://jupiter.exchange", "custom"),
        ("jupiter.exchange", "custom"),
        ("https://www.myproject.xyz/docs", "custom"),
        ("https://docs.myproject.io", "custom"),
        ("https://linktr.ee/someone", "aggregator"),
        ("https://x.com/someone", "aggregator"),
        ("https://t.co/AbC123", "short"),
        ("https://github.com/org", "code_host"),
        ("", "none"),
        ("not a url", "none"),
    ],
)
def test_classify_url(url, kind):
    assert classify_url(url, FILTERS["stage2"]) == kind


def test_host_of_strips_www_and_scheme():
    assert host_of("https://www.Example.com/path?q=1") == "example.com"
    assert host_of(None) == ""


def test_domains_in_bio_ignores_emails_and_handles():
    assert domains_in_text("docs at acme.finance, ping me@acme.com") == ["acme.finance"]
    assert domains_in_text("follow @acme.io for news") == []
    assert domains_in_text("no domain here") == []


# ------------------------------------------------------------------ stage 1


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"protected": 1, "bio": "x"}, "protected"),
        ({"has_default_avatar": 1, "bio": "x"}, "default_avatar"),
        ({"bio": "", "url": ""}, "no_bio_no_url"),
        ({"bio": "airdrop hunter", "following_count": 4500}, "following>2000"),
        ({"bio": "hello", "followers_count": 3}, "followers<10"),
        ({"bio": "hello", "followers_count": 30, "statuses_count": 0}, "silent_low_followers"),
    ],
)
def test_stage1_rejects(over, reason):
    v = stage1_hard.evaluate(make_account(1, **over), FILTERS)
    assert not v.passed and v.reason == reason


def test_stage1_passes_ordinary_account():
    assert stage1_hard.evaluate(make_account(1, bio="building a protocol"), FILTERS).passed


def test_new_project_with_custom_domain_survives_tiny_follower_count():
    """The core product requirement: a project that has just followed the ecosystem
    often has <10 followers. A real domain must exempt it from the follower floors."""
    acc = make_account(1, bio="Coming soon", url="https://newproto.xyz",
                       followers_count=2, statuses_count=0)
    v = stage1_hard.evaluate(acc, FILTERS)
    assert v.passed and v.reason == "exempt_custom_domain"


def test_exemption_does_not_apply_to_linktree_or_bots():
    acc = make_account(1, bio="gm", url="https://linktr.ee/x", followers_count=2)
    assert not stage1_hard.evaluate(acc, FILTERS).passed


def test_exemption_does_not_override_farmer_following_cap():
    acc = make_account(1, bio="x", url="https://farm.xyz", following_count=3000)
    assert not stage1_hard.evaluate(acc, FILTERS).passed


# ------------------------------------------------------------------ stage 2


def test_real_project_scores_well_above_threshold():
    acc = make_account(
        1, name="Meridian Protocol", bio="Non-custodial perps on Base. Mainnet live. docs.meridian.fi",
        url="https://meridian.fi", followers_count=900, following_count=120,
        created_at=days_ago(30),
    )
    _, v2 = run(acc)
    assert v2.passed and v2.score >= 9, v2


def test_thin_brand_new_project_still_passes():
    """Best lead type: tiny, new, but has a domain. Must not be filtered out."""
    acc = make_account(
        1, name="Orbit Labs", bio="Building something. Soon.", url="https://orbit-labs.xyz",
        followers_count=4, following_count=12, statuses_count=1, created_at=days_ago(3),
    )
    v1, v2 = run(acc)
    assert v1.passed and v2.passed, (v1, v2)


def test_individual_with_personal_site_does_not_pass_on_site_alone():
    acc = make_account(
        1, name="John Smith", bio="Engineer. he/him. Views my own.", url="https://johnsmith.com",
        followers_count=300, following_count=400, created_at=days_ago(2000),
    )
    _, v2 = run(acc)
    assert not v2.passed, v2


def test_trader_with_linktree_fails():
    acc = make_account(
        1, name="crypto_chad", bio="degen trader. NFA. dyor", url="https://linktr.ee/chad",
        followers_count=500, following_count=900, created_at=days_ago(800),
    )
    _, v2 = run(acc)
    assert not v2.passed and v2.components.get("personal_markers", 0) < 0


def test_airdrop_farmer_rejected_at_stage1():
    acc = make_account(1, name="Airdrop King", bio="airdrops", following_count=3800)
    v1, v2 = run(acc)
    assert not v1.passed and v2 is None


def test_personal_markers_capped():
    acc = make_account(
        1, bio="he/him degen trader shitposter views my own personal account nfa dyor",
    )
    _, v2 = run(acc)
    assert v2.components["personal_markers"] == FILTERS["stage2"]["weights"]["personal_marker_cap"]


def test_keywords_capped():
    acc = make_account(1, bio="protocol mainnet testnet sdk onchain rollup staking oracle vault")
    _, v2 = run(acc)
    assert v2.components["bio_keywords"] == FILTERS["stage2"]["weights"]["bio_keyword_cap"]


def test_young_account_is_positive_not_negative():
    young = make_account(1, bio="protocol", created_at=days_ago(10))
    old = make_account(2, bio="protocol", created_at=days_ago(3000))
    _, vy = run(young)
    _, vo = run(old)
    assert vy.score > vo.score


def test_unknown_creation_date_does_not_crash():
    _, v = run(make_account(1, bio="protocol mainnet", created_at=None))
    assert "young_account" not in v.components


def test_unresolved_t_co_url_is_weak_positive_not_zero_not_custom():
    _, v = run(make_account(1, bio="hello world", url="https://t.co/AbC123"))
    assert v.components.get("unresolved_short_url") == 1.5
    assert "custom_domain_url" not in v.components


# -------------------------------------------------------------- convergence


def test_convergence_bonus_can_lift_a_borderline_account_over_threshold():
    acc = make_account(1, name="Sparrow", bio="shipping soon", url="https://github.com/sparrow",
                       followers_count=40, following_count=60, created_at=days_ago(20))
    _, single = run(acc, convergence=1)
    _, triple = run(acc, convergence=3)
    assert triple.score > single.score
    assert triple.components["convergence_bonus"] == 5.0  # capped (2 extra * 2.5)


def test_convergence_alone_does_not_rescue_a_farmer_profile():
    """Farmers follow every ecosystem account. The bonus must not be a bypass."""
    acc = make_account(
        1, name="Alex Doe", bio="degen. NFA. he/him", url="",
        followers_count=60, following_count=900, created_at=days_ago(40),
    )
    _, v = run(acc, convergence=4)
    assert not v.passed, v


def test_stage3_counts_distinct_targets(db):
    t1 = db.upsert_target("solana", 3600, True)
    t2 = db.upsert_target("base", 3600, True)
    db.upsert_account(make_account(1, bio="x"))
    db.record_follow_event("1", t1)
    v, names = stage3_convergence.evaluate(db, "1", FILTERS)
    assert not v.passed and names == ["solana"]
    db.record_follow_event("1", t2)
    v, names = stage3_convergence.evaluate(db, "1", FILTERS)
    assert v.passed and v.score == 2.0 and set(names) == {"solana", "base"}


# ------------------------------------------------- hand-labelled mini corpus


PROJECTS = [
    dict(name="Meridian Protocol", bio="Perps on Base. Mainnet live.", url="https://meridian.fi", followers_count=900, following_count=120, created_at=days_ago(30)),
    dict(name="Orbit Labs", bio="Building the liquidity layer for Solana.", url="https://orbit-labs.xyz", followers_count=40, following_count=30, created_at=days_ago(8)),
    dict(name="Kestrel", bio="Restaking infrastructure. Audited. docs.kestrel.so", url="https://kestrel.so", followers_count=300, following_count=90, created_at=days_ago(60)),
    dict(name="SolForge", bio="Permissionless launchpad on Solana. Testnet now live.", url="https://solforge.app", followers_count=150, following_count=40, created_at=days_ago(15)),
    dict(name="Lumen Finance", bio="Non-custodial lending. Smart contracts audited.", url="https://lumen.finance", followers_count=75, following_count=22, created_at=days_ago(5)),
    dict(name="Halcyon Network", bio="zk rollup for payments. SDK + docs below.", url="https://halcyon.network", followers_count=500, following_count=100, created_at=days_ago(75)),
    dict(name="PrismDAO", bio="Onchain governance tooling. Mainnet soon.", url="https://prismdao.org", followers_count=60, following_count=50, created_at=days_ago(20)),
    dict(name="Tessera", bio="Oracle network. Building in public.", url="https://tessera.dev", followers_count=220, following_count=70, created_at=days_ago(45)),
]

NOT_PROJECTS = [
    dict(name="Sarah Connor", bio="she/her. coffee. views my own", url="", followers_count=200, following_count=500, created_at=days_ago(2500)),
    dict(name="crypto_wolf", bio="degen trader. NFA. dyor", url="https://linktr.ee/wolf", followers_count=800, following_count=1200, created_at=days_ago(900)),
    dict(name="Mike Johnson", bio="Dad. Engineer. Opinions are my own.", url="https://mikejohnson.com", followers_count=150, following_count=300, created_at=days_ago(3000)),
    dict(name="Airdrop Hunter", bio="airdrops daily", url="", followers_count=500, following_count=3900, created_at=days_ago(100)),
    dict(name="🔥 gm gm 🔥", bio="gm", url="", followers_count=1, following_count=4, statuses_count=0, created_at=days_ago(2)),
    dict(name="NFT Whale", bio="trader | NFA | personal account", url="https://x.com/whale", followers_count=2000, following_count=800, created_at=days_ago(1200)),
    dict(name="Priya Patel", bio="she/her. product designer. views my own", url="https://priya.design", followers_count=400, following_count=600, created_at=days_ago(1800)),
    dict(name="Dave", bio="anon. shitposter.", url="", followers_count=90, following_count=700, created_at=days_ago(600)),
]


def test_labelled_profiles():
    """Mechanics check on invented data. See module docstring."""
    missed = []
    for i, p in enumerate(PROJECTS):
        v1, v2 = run(make_account(100 + i, **p))
        if not (v1.passed and v2 and v2.passed):
            missed.append((p["name"], v1.reason, v2 and v2.score))
    leaked = []
    for i, p in enumerate(NOT_PROJECTS):
        v1, v2 = run(make_account(200 + i, **p))
        if v1.passed and v2 and v2.passed:
            leaked.append((p["name"], v2.score, v2.components))
    assert not missed, f"projects wrongly filtered: {missed}"
    assert not leaked, f"non-projects leaked through: {leaked}"


# ----------------------------------------------------------------- memecoins


from scout.funnel.common import has_contract_address  # noqa: E402


@pytest.mark.parametrize(
    "bio,expected",
    [
        ("CA: 7rkqDbQgTtvqJg4Q5z3BS4pDZgZdXx7gysENetRppump", True),  # real-shaped Solana mint
        ("buy 0x1234567890abcdef1234567890ABCDEF12345678 now", True),  # EVM address
        ("tx 0x" + "ab" * 32, False),  # a 64-hex tx hash is not an address
        ("Perps on Base. Mainnet live.", False),
        ("Infrastructureforpermissionlesssettlementnetworks", False),  # long word, no digits/mixed case
        ("", False),
    ],
)
def test_contract_address_detection(bio, expected):
    assert has_contract_address(bio) is expected


def test_memecoin_markers_are_penalised_not_dropped():
    token = make_account(1, name="Fomo coin", bio="Born from a simple doodle. CA: 7rkqDbQgTtvqJg4Q5z3BS4pDZgZdXx7gysENetRppump",
                         url="https://fomocoin.space", followers_count=17, created_at=days_ago(4))
    clean = make_account(2, name="Fomo coin", bio="Born from a simple doodle.",
                         url="https://fomocoin.space", followers_count=17, created_at=days_ago(4))
    _, vt = run(token)
    _, vc = run(clean)
    assert vt.components["memecoin_marker"] == -2.5
    assert vt.score == vc.score - 2.5


def test_real_project_listing_its_token_still_passes():
    """A genuine protocol that also publishes its token address must survive: the
    marker is a penalty, and the domain + keywords outweigh it."""
    acc = make_account(1, name="Meridian Protocol", url="https://meridian.fi", followers_count=900,
                       following_count=120, created_at=days_ago(30),
                       bio="Non-custodial perps on Base. Mainnet live. CA: 7rkqDbQgTtvqJg4Q5z3BS4pDZgZdXx7gysENetRppump")
    v1, v2 = run(acc)
    assert v2.passed and "memecoin_marker" in v2.components


def test_pump_fun_link_is_not_credited_as_a_project_website():
    assert classify_url("https://pump.fun/join/zero", FILTERS["stage2"]) == "aggregator"


def test_dex_that_swaps_memecoins_is_not_penalised_as_a_memecoin():
    """Regression from live data: FastSwap DEX's bio says 'instant memecoin swap'."""
    acc = make_account(1, name="FastSwap DEX", url="https://fastswap-dex.vercel.app",
                       bio="Next-Gen Solana DEX Aggregator. Best routing across Raydium, Orca & Meteora. Lowest slippage & instant memecoin swap",
                       followers_count=7, created_at=days_ago(1000))
    _, v = run(acc)
    assert "memecoin_marker" not in v.components and v.passed


def test_state_abbreviation_ca_is_not_a_memecoin_marker():
    acc = make_account(1, name="Orbit Labs", bio="Building the liquidity layer. Los Angeles, CA", url="https://orbit-labs.xyz")
    _, v = run(acc)
    assert "memecoin_marker" not in v.components
