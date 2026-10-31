"""scout -- command line entry point.

    scout measure --target solana [--wait-minutes 30]   Phase 0: verify assumptions
    scout poll-once [--target X] [--llm] [--send]       one pass; prints, sends nothing by default
    scout run                                           the long-running service
    scout status                                        spend, funnel counts, candidates
    scout replay [--apply]                              re-score stored accounts, no API calls
    scout test-alert                                    send one synthetic alert to Telegram
    scout test-llm                                      verify DeepSeek key/model with 3 synthetic profiles
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import measure as measure_mod
from .alerts.telegram import Telegram, TelegramError, format_alert
from .collector import poll_target
from .config import PROJECT_ROOT, Config, ConfigError, load_config
from .db import Database
from .funnel import llm as llm_mod
from .funnel.deepseek import DeepSeekClient
from .funnel.llm import LLMFatalError
from .pipeline import replay as replay_history
from .providers.base import ProviderError
from .providers.twitterapi import TwitterApiIo
from .runner import Runner, bootstrap, judge_new
from .webfetch import make_expander

DAILY_BUDGET_USD = 0.50  # status warns above this; the plan expects ~$0.17


def _provider(cfg: Config) -> TwitterApiIo:
    cfg.secrets.require("twitterapi_key")
    interval = float(os.getenv("SCOUT_MIN_REQUEST_INTERVAL_S", "5.0"))
    return TwitterApiIo(cfg.secrets.twitterapi_key, min_request_interval_s=interval)


def _llm_client(cfg: Config):
    if not cfg.secrets.deepseek_api_key:
        return None
    return DeepSeekClient(cfg.secrets.deepseek_api_key)


def _telegram(cfg: Config) -> Telegram | None:
    if not (cfg.secrets.telegram_token and cfg.secrets.telegram_chat_id):
        return None
    return Telegram(cfg.secrets.telegram_token, cfg.secrets.telegram_chat_id)


# ------------------------------------------------------------------ measure


def cmd_measure(args: argparse.Namespace, cfg: Config) -> int:
    provider = _provider(cfg)
    try:
        page_size = 200
        first = provider.get_followers(args.target, None, page_size)
        if not args.no_expand:
            n = measure_mod.expand_links(first.accounts, cfg.filters, make_expander())
            print(f"(resolved {n} t.co profile link(s); --no-expand to skip)")
        stats = measure_mod.analyse_page(first.accounts, cfg.filters)
        print(measure_mod.render(stats, target=args.target))
        print(f"  credits this call: {first.credits_spent:.0f} (~${first.credits_spent / 100_000:.4f})")

        cap_dir = PROJECT_ROOT / "data" / "captures"
        cap_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cap = cap_dir / f"{args.target}-{stamp}.json"
        cap.write_text(json.dumps([json.loads(a["raw_json"]) for a in first.accounts]), encoding="utf-8")
        print(f"  raw page saved   : {cap.relative_to(PROJECT_ROOT)}  (build your labelled test corpus from this)")

        if args.wait_minutes > 0:
            print(f"\nwaiting {args.wait_minutes} min for a second snapshot to verify ordering...")
            t0 = time.monotonic()
            time.sleep(args.wait_minutes * 60)
            second = provider.get_followers(args.target, None, page_size)
            hours = (time.monotonic() - t0) / 3600
            res = measure_mod.check_ordering(
                [a["id"] for a in first.accounts], [a["id"] for a in second.accounts]
            )
            print(f"\nORDERING: {res.verdict.upper()} -- {res.detail}")
            if res.new_count and res.overlap:
                print(f"  new followers/hour: ~{res.new_count / hours:.0f}")
            elif res.verdict == "inconclusive_no_overlap":
                print(f"  new followers/hour: > {len(second.accounts) / hours:.0f} (page overflowed)")
            if res.verdict == "violated":
                print("\nNO-GO: not newest-first. Page-1 polling is invalid; the design needs re-costing.")
                return 2
            if not res.confirmed:
                print("\nNot yet proven. Re-run with a different --wait-minutes.")
                return 3
            print("\nGO: newest-first confirmed on live data.")
        else:
            print("\n(ordering NOT yet verified -- re-run with --wait-minutes 30)")
    finally:
        provider.close()
    return 0


# ---------------------------------------------------------------- poll-once


def cmd_poll_once(args: argparse.Namespace, cfg: Config) -> int:
    provider = _provider(cfg)
    with Database(cfg.db_path) as db:
        try:
            bootstrap(db, cfg, provider)
            specs = [s for s in cfg.active_targets() if not args.target or s.username == args.target]
            if not specs:
                print(f"no active target matches {args.target!r}")
                return 1
            for spec in specs:
                result = poll_target(db, provider, spec)
                if result.cold_start:
                    print(f"@{spec.username}: cold start -- recorded the current boundary, "
                          f"nothing to report. Run again later.")
                    continue
                decisions = judge_new(db, cfg, result, make_expander())
                gate = [d for d in decisions if d.passed_gate]
                print(f"@{spec.username}: {len(result.new_accounts)} new, {len(gate)} past the gate "
                      f"({result.pages_fetched} page(s), {result.credits_spent:.0f} credits)")
                for d in sorted(gate, key=lambda d: -(d.score or 0)):
                    print(f"   {d.score:>5}  @{d.username:<20} {json.dumps(d.components)}")
            if args.llm:
                from .enrich import enrich_pending
                client = _llm_client(cfg)
                if client is None:
                    print("--llm needs DEEPSEEK_API_KEY")
                else:
                    print("LLM:", enrich_pending(db, cfg, client))
            if args.send:
                from .routing import route_instant
                tg = _telegram(cfg)
                print("instant alerts sent:", route_instant(db, cfg, tg) if tg else "Telegram not configured")
        finally:
            provider.close()
    return 0


# --------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    provider = _provider(cfg)
    with Database(cfg.db_path) as db:
        try:
            bootstrap(db, cfg, provider)
            Runner(db, cfg, provider, llm_client=_llm_client(cfg), telegram=_telegram(cfg)).run_forever()
        except KeyboardInterrupt:
            print("stopped")
        finally:
            provider.close()
    return 0


# ------------------------------------------------------------------ status


def cmd_status(_args: argparse.Namespace, cfg: Config) -> int:
    with Database(cfg.db_path) as db:
        s = db.spend_since(24)
        print(f"last 24h: {s['polls']} polls, {s['items']} profiles read, {s['new_followers']} new, "
              f"{s['credits']:.0f} credits = ${s['usd']:.4f}")
        if s["capped"]:
            print(f"  !! {s['capped']} poll(s) hit the page cap -- followers were MISSED. "
                  f"Raise max_pages_per_poll or poll more often.")
        if s["usd"] > DAILY_BUDGET_USD:
            print(f"  !! over the ${DAILY_BUDGET_USD:.2f}/day budget the plan expects")
        print(f"LLM today: {db.usage_today('classify')} classified, {db.usage_today('summarize')} summarised")
        print("\ntargets:")
        for t in db.list_targets(active_only=False):
            rate = f"{t['observed_new_per_hour']:.0f}/h" if t["observed_new_per_hour"] is not None else "n/a"
            print(f"  @{t['username']:<16} active={t['active']} interval={t['poll_interval_s']}s "
                  f"rate={rate} seen={db.seen_count(t['id'])} cold_started={t['cold_started']}")
        print("\nfunnel (all time):")
        for stage, c in sorted(db.stage_counts().items()):
            print(f"  {stage}: {c['passed']} passed / {c['failed']} failed")
        print("\ncandidates:", db.candidate_state_counts() or "none yet")
        done = {k: v for k, v in db.candidate_state_counts().items() if k in ("interested", "not_project")}
        if done:
            n = sum(done.values())
            print(f"  human precision so far: {done.get('interested', 0)}/{n} marked interested")
    return 0


# ------------------------------------------------------------------ replay


def cmd_replay(args: argparse.Namespace, cfg: Config) -> int:
    with Database(cfg.db_path) as db:
        rep = replay_history(db, cfg, apply=args.apply, limit=args.limit)
        mode = "APPLIED" if args.apply else "dry-run (nothing written)"
        print(f"replay [{mode}] over {rep.total} stored accounts, using current config/filters.yaml")
        print(f"  stage 1 pass: {rep.stage1_pass}   past gate: {rep.gate_pass}   (was {rep.previously_gate_pass})")
        for label, items in (("would NEWLY pass", rep.flipped_in), ("would NO LONGER pass", rep.flipped_out)):
            print(f"  {label}: {len(items)}")
            for _id, user, score in items[:15]:
                print(f"      @{user} (score {score})")
    return 0


# -------------------------------------------------------------- test-alert


def cmd_test_alert(_args: argparse.Namespace, cfg: Config) -> int:
    cfg.secrets.require("telegram_token", "telegram_chat_id")
    tg = _telegram(cfg)
    demo = {
        "username": "example", "name": "Example Protocol", "llm_summary":
        "SYNTHETIC test alert. A perps DEX on Base. Buttons below should respond but change nothing.",
        "llm_category": "defi", "llm_chain": "base", "llm_stack": "solidity_evm", "llm_confidence": 0.93,
        "llm_contracts": "yes",
        "followed": ["solana", "base"], "url": "https://example.com", "followers_count": 120,
        "created_at": "2026-10-20", "score": 9.5,
    }
    mid = tg.send(format_alert(demo, kind="instant"), account_id="0")
    print(f"sent test alert (message {mid})")
    return 0


# ---------------------------------------------------------------- test-llm


# Synthetic profiles with an obvious expected answer. Not real accounts.
_LLM_PROBES = [
    ("protocol", {"name": "Meridian Protocol", "username": "meridian_fi",
                  "bio": "Non-custodial perps on Base. Mainnet live. Audited. docs.meridian.fi",
                  "url": "https://meridian.fi", "followers_count": 900, "following_count": 120,
                  "statuses_count": 80, "created_at": "2026-09-20"}, ("project", "not memecoin")),
    ("person", {"name": "Sarah Connor", "username": "sconnor", "bio": "she/her. coffee. product designer. views my own",
                "url": "", "followers_count": 200, "following_count": 500, "statuses_count": 900,
                "created_at": "2019-03-02"}, ("not a project", "")),
    ("memecoin", {"name": "GTAPE", "username": "gtape_sol", "bio": "One monkey. One city. Zero risk management. $GTAPE",
                  "url": "https://gtape.top", "followers_count": 9, "following_count": 14,
                  "statuses_count": 3, "created_at": "2026-10-25"}, ("memecoin", "")),
]


def cmd_test_llm(_args: argparse.Namespace, cfg: Config) -> int:
    cfg.secrets.require("deepseek_api_key")
    client = _llm_client(cfg)
    model = cfg.llm().get("classify_model", "deepseek-flash")
    print(f"model: {model}   (3 calls, a fraction of a cent)\n")
    try:
        for label, acct, expect in _LLM_PROBES:
            r = llm_mod.classify(client, model, acct)
            if r is None:
                print(f"  {label:<9} -> no usable answer (see warnings above)")
                continue
            print(f"  {label:<9} -> is_project={r.is_project!s:<5} conf={r.confidence:.2f} "
                  f"category={r.category:<14} chain={r.chain:<9} | {r.reason}")
            print(f"  {'':<9}    expected: {expect[0]}")
    finally:
        client.close()
    return 0


# -------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="Phase 0: measure real volumes and verify newest-first ordering")
    m.add_argument("--target", default="solana")
    m.add_argument("--wait-minutes", type=float, default=0, help="take a second snapshot after N minutes to verify ordering")
    m.add_argument("--no-expand", action="store_true", help="do not resolve t.co profile links")
    m.set_defaults(fn=cmd_measure)

    o = sub.add_parser("poll-once", help="one polling pass; prints what passed the gate")
    o.add_argument("--target")
    o.add_argument("--llm", action="store_true", help="also run LLM stages (costs money)")
    o.add_argument("--send", action="store_true", help="also send instant alerts to Telegram")
    o.set_defaults(fn=cmd_poll_once)

    sub.add_parser("run", help="run the service").set_defaults(fn=cmd_run)
    sub.add_parser("status", help="spend, funnel counts, candidates").set_defaults(fn=cmd_status)

    r = sub.add_parser("replay", help="re-score stored accounts under current filters (no API calls)")
    r.add_argument("--apply", action="store_true", help="persist the new verdicts (default: dry run)")
    r.add_argument("--limit", type=int)
    r.set_defaults(fn=cmd_replay)

    sub.add_parser("test-alert", help="send one synthetic alert").set_defaults(fn=cmd_test_alert)
    sub.add_parser("test-llm", help="classify 3 synthetic profiles to verify the DeepSeek setup").set_defaults(fn=cmd_test_llm)
    return p


def main(argv: list[str] | None = None) -> int:
    # Real account names contain emoji; the default Windows console codepage crashes on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
        logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines include request URLs
        return int(args.fn(args, cfg) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 1
    except LLMFatalError as exc:
        print(f"llm error: {exc}", file=sys.stderr)
        return 1
    except TelegramError as exc:
        print(f"telegram error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
