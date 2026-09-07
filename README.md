# real-token-meter

[![CI](https://github.com/anhminhzui-dev/real-token-meter/actions/workflows/ci.yml/badge.svg)](https://github.com/anhminhzui-dev/real-token-meter/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/badge/licence-evaluation--only-blue)](LICENSE)

**Count non-padding tokens before comparing training throughput or cost.**

This offline meter computes throughput and synthetic cost over real tokens, refuses untrustworthy steps, and names the denominator of every aggregate. Here, real tokens means `batch_size * seq_len - pad_tokens`; it does not measure learning progress or useful gradient updates.

```text
step log + policy → validate counts, time and rate → per-step + run-level checks
                  → counted-token totals → denominator-bound report
```

Both padding budgets matter: individually acceptable steps can still exceed the aggregate run budget. Refused rows remain visible but never enter the reported totals. Deterministic summaries bind the input hashes and omit timestamps and local paths.

The comparison below uses invented logs and explicitly synthetic hourly rates. There is no trainer, hardware or provider integration, and no real efficiency gain is claimed.

## Why raw tokens per second misleads

Raw throughput is `batch_size * seq_len / wall_seconds`. It counts padding as work. A run that
pads every sequence out to a fixed length and a run that packs sequences to length can post the
same raw figure while processing different quantities of non-padding tokens. This arithmetic alone cannot establish which run learns more. So the
numerator here is `batch_size * seq_len - pad_tokens` — that is what **real** means in this
repository — and the headline figure is cost per million real tokens.

The demonstration is the `compare` subcommand. `run-padded` is 73.24% padding but turns its
steps over quickly, so on cost per million real tokens it can look *cheaper* than the
well-packed run. The padding budget is the only thing standing between a reviewer and that
conclusion:

```
$ PYTHONPATH=src python -m real_token_meter.cli compare --policy fixtures/policy.json --a fixtures/log_clean.jsonl --b fixtures/log_padded.jsonl
POLICY: throughput-v1  sha=454ed24eb74bb918  step_pad_max=0.35  run_pad_max=0.20  min_steps=4
RUN A: run-packed  GO    0.285193 per million real tokens over 6 steps counted  (6 read, 0 refused)
RUN B: run-padded  HOLD  MISSING (0 counted of 5 read)  (5 read, 5 refused)
RANK 1: run-packed  0.285193 per million real tokens over 6 steps counted
RANK 2: run-padded  REFUSED, not ranked on cost (0 counted of 5 read)
CHEAPEST: not computed - run-padded refused, so no ratio exists
VERDICT: HOLD (1 of 2 runs refused)
$ echo $?
2
```

Switch the padding guard off — which is exactly what the falsifier test does — and the meter
cheerfully hands the win to the inflated run:

```
RANK 1: run-padded  0.202758 per million real tokens over 5 steps counted
RANK 2: run-packed  0.285193 per million real tokens over 6 steps counted
CHEAPEST: run-padded, 1.41x cheaper per million real tokens than run-packed, over 5 steps counted against 6 steps counted
VERDICT: GO
```

Both numbers above are an illustrative computation over the five and six rows in `fixtures/`,
which are invented for this repository — not a measured efficiency win for any real training
run. What the comparison demonstrates is the mechanism: the same fixture bytes flip from HOLD to
a policy-violating GO the moment the padding guard is switched off, which is why that switch is a test and
not a claim.

A guard that has never been shown to miss something certifies nothing, so that failure is a
test rather than a paragraph.

## What it refuses

| Code | What triggers it |
|---|---|
| `MALFORMED_ROW` | the line is not a JSON object — a truncated writer, not a slow step |
| `MISSING_FIELD` | a field the policy requires is absent, its value is not a number, or a token count is a non-finite number (`Infinity`, `-Infinity`) that overflows an integer cast |
| `DUPLICATE_STEP` | the same step number appears twice in one run |
| `NEGATIVE_OR_ZERO_TIME` | `wall_seconds` is zero, negative, `NaN`, or infinite — a divide-by-zero, or a division by nonsense, dressed as speed |
| `BAD_RATE` | `gpu_hour_rate_synthetic` is negative, `NaN`, or infinite — an untrusted rate is never allowed into the cost arithmetic |
| `BAD_TOKEN_COUNTS` | `batch_size` or `seq_len` is zero or negative, padding is negative, or padding is at least the step's whole token count |
| `PADDING_INFLATED` | the step's padding ratio is over the policy's per-step budget |

Two run-level codes sit above those. `TOO_FEW_STEPS` refuses a run too short to mean anything,
and `RUN_PADDING_OVER_BUDGET` catches the case a per-step budget cannot see: a fleet of steps
each just under the step budget whose aggregate is still over the run budget. That is why the
policy carries two thresholds rather than one. Three conditions halt a run outright rather than
scoring it — a policy or log row without the `"synthetic": true` marker, two different run ids
inside one file, and a `compare` of a file against itself.

Refused steps are excluded from every aggregate but never dropped from the report: they are
printed, counted and named, and the run's verdict is HOLD. `--out` writes a `summary.json`
whose every metric key states its own denominator (`real_tokens_over_counted_steps`,
`cost_per_million_real_tokens_synthetic_rate`); it carries no timestamp and no path, so two
runs over the same inputs write identical bytes.

## Try it in 60 seconds

```
$ python -m pytest -q
..........................                                               [100%]
26 passed in 0.16s        # run 2026-09-07
```

26 of 26: one per refusal code, one proving the run-level padding budget fires on a run whose
every individual step is under the step budget, one per halt condition, the padding falsifier
above, a test that walks the whole rendered report and fails on any token count printed without
its denominator, a public-clean scan that plants five forbidden shapes and requires each to fire
before a clean tree is allowed to count for anything, ten parametrized cases — `NaN`, `+inf`,
`-inf`, zero and negative wall time; negative, `NaN` and `+inf`/`-inf` rates — proving an
untrusted cost input is refused rather than admitted as a `GO`, one proving a non-numeric string
in a numeric field is a named `MISSING_FIELD` rather than a crash, one proving an infinite token
count is the same named refusal rather than an uncaught `OverflowError`, and a second falsifier
that switches the two cost-input guards off and proves the meter then hands back a `GO` with a
`NaN` cost and a `GO` with a negative cost — the exact two admissions a review of this repository
found before this guard existed.

## Scope and integration

Synthetic logs make the accounting reproducible: padded tokens, non-padding tokens, elapsed time and configured rates are kept distinct. Displayed costs are fixture arithmetic, not measurements of named hardware or a real training run.

Budget thresholds and minimum-step requirements are policy inputs. Connect representative logs and actual rates before using the output for a purchase decision. Non-padding throughput measures work volume, not learning quality. The tool runs offline without loading a model.

## Integration path

Ask for one week of real training logs and the cost number the team is judged on. Then wire
this in as the step the training loop cannot skip: real tokens per second and cost per million
real tokens on every run, refusals visible instead of averaged away, thresholds set from your
own padding distribution rather than mine. From there the same shape extends outward — spend
per million real tokens by data source, by sequence-length bucket, by model size — because once
the denominator is honest, every downstream comparison is worth making. What I would not ship
is a dashboard whose numbers have never been shown to be wrong.

## Project context

Problem definition, architecture and acceptance review: **Minh Vo**, with AI-assisted implementation. This focused tool belongs to a broader body of data, assessment and training-systems work described in the [research overview](https://github.com/anhminhzui-dev#research-engineering-the-evidence-behind-ai-judgement). Its runnable scope is the mechanism documented here.

## Licence

Source-available, evaluation-only — read it, run it, quote it in a review; see `LICENSE`.
