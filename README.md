# CT Follow Scout

Finds brand-new crypto projects the day they appear, by watching who starts
**following** ecosystem accounts (@solana, @base, @ethereum, ...). New projects
follow the ecosystem they build on almost immediately; the ecosystems themselves
rarely follow anyone, so the inbound side is where the signal is.

It is a **scout**, not an auditor: it finds and qualifies leads. Auditing happens in
separate Claude Code sessions.

## Setup

```bash
pip install -e .[dev]
cp .env.example .env      # then fill in TWITTERAPI_KEY (and the others when ready)
pytest                    # ~200 tests, no network, no credits
```

## Phase 0 -- verify the assumptions on live data (do this first, ~$0.01)

The design rests on one vendor behaviour -- that follower lists come back
newest-first -- and on volume/survival numbers that are currently estimates.

```bash
scout measure --target solana --wait-minutes 30
```

Prints stage-1/2 survival on one real page, the mix of profile-link kinds, the
observed new-followers/hour, and a **GO / NO-GO** verdict on ordering. Exit code 0 =
confirmed, 2 = ordering violated (stop; the design needs re-costing), 3 = not yet
conclusive (try a different `--wait-minutes`). The raw page is saved to
`data/captures/` -- use it to build a real labelled test corpus.

Check the "profile link kinds" block: if most links are unexpanded `t.co`, the
strongest signal (a custom domain in the profile) is blind and needs a resolver.

## Day to day

```bash
scout poll-once                 # one pass; prints what cleared the free filters. Sends nothing.
scout poll-once --llm           # + classify/summarise survivors (costs money)
scout replay                    # re-score stored accounts under current config/filters.yaml (free)
scout replay --apply
scout status                    # spend, funnel counts, candidates, human precision
scout test-alert                # one synthetic Telegram message
scout test-llm                  # verify the DeepSeek key/model on 3 synthetic profiles (<1 cent)
scout run                       # the long-running service (VPS)
```

The first poll of any target is a **cold start**: it records the current boundary and
reports nothing, so existing followers are not mistaken for new ones.

## Tuning

Everything tunable is in `config/filters.yaml` -- thresholds are **untuned guesses**
until real accounts are stored. Loop: let it collect, edit the file, `scout replay`,
read what flips, repeat. Telegram buttons record your verdicts
(`interested` / `not_project`) and `scout status` reports precision from them.

## Layout

```
config/targets.yaml    accounts to watch + poll settings
config/filters.yaml    every threshold, keyword list, LLM cap, alert rule
src/scout/
  providers/           X data source behind a swappable interface (twitterapi.io)
  collector.py         poll newest-first, stop at the first known id; all-or-nothing
  funnel/              stage 1 hard filters, 2 scoring, 3 convergence, LLM 4-5
  pipeline.py          stages 1-3 + replay        enrich.py   stages 4-5 with daily caps
  funnel/deepseek.py   DeepSeek client (thinking disabled, forced tool call, fatal-vs-retry errors)
  routing.py           instant vs digest          alerts/     Telegram
  webfetch.py          SSRF-guarded site fetcher  runner.py   the loop
```
