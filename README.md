# real-token-meter

> "Demonstrated expertise with Python, including deep familiarity with machine learning frameworks such as scikit-learn, TensorFlow, or PyTorch" — micro1, Machine Learning Engineer (Contractor), first requirement of the posting at himalayas.app/companies/micro1/jobs/machine-learning-engineer-7037967333

Built for this posting, in a day, to show the shape of what I would do on day one.

## To the micro1 reviewer

Frameworks are the easy half. The half that decides whether a training budget was spent well
is the arithmetic underneath them, and the most common way that arithmetic lies is padding.
This tool reads a per-step training log and reports throughput and cost over **real** tokens,
refuses the steps whose numbers cannot be trusted, and prints every figure next to the
denominator it was computed over. Standard library only, Python 3.10 or newer, nothing to
install except `pytest`. The one command a reviewer runs:

```
$ PYTHONPATH=src python -m real_token_meter.cli meter --policy fixtures/policy.json --log fixtures/log_bad.jsonl
```

It exits 2 and names six refusals. The clean fixture exits 0. Everything under `fixtures/` is
invented for this repository, and the rate is a made-up number: the log field is called
`gpu_hour_rate_synthetic` so it cannot be mistaken for a price anyone charges.

## Why raw tokens per second misleads

Raw throughput is `batch_size * seq_len / wall_seconds`. It counts padding as work. A run that
pads every sequence out to a fixed length and a run that packs sequences to length can post the
same raw figure while one of them is buying far less learning per unit of spend. So the
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

A guard that has never been shown to miss something certifies nothing, so that failure is a
test rather than a paragraph.

## What it refuses

| Code | What triggers it |
|---|---|
| `MALFORMED_ROW` | the line is not a JSON object — a truncated writer, not a slow step |
| `MISSING_FIELD` | a field the policy requires is absent, or its value is not a number |
| `DUPLICATE_STEP` | the same step number appears twice in one run |
| `NEGATIVE_OR_ZERO_TIME` | `wall_seconds` is zero or negative — a divide-by-zero dressed as speed |
| `BAD_TOKEN_COUNTS` | padding is negative, or is at least the step's whole token count |
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

## Tests

```
$ python -m pytest -q
.............                                                            [100%]
13 passed in 0.12s
```

13 of 13: one per refusal code, one proving the run-level padding budget fires on a run whose
every individual step is under the step budget, one per halt condition, the falsifier above, a
test that walks the whole rendered report and fails on any token count printed without its
denominator, and a public-clean scan that plants five forbidden shapes and requires each to
fire before a clean tree is allowed to count for anything.

## What this is not

**No accuracy is claimed here and none is computable from what ships here.** Nothing in this
repository measures a real training run, a real model or a real machine: no hardware is named
and no price is real. There is no network code path — a test greps `src/` for the
network-capable imports and fails on a hit — and no model is loaded, so the arithmetic is
deterministic by construction rather than by promise. Every threshold is a design constant of
this project, not a validated operating point: the step budget of 0.35, the run budget of 0.20
and the minimum of 4 steps are policy inputs a team sets from its own data, not findings.

## What I would do on day one at micro1

Ask for one week of real training logs and the cost number the team is judged on. Then wire
this in as the step the training loop cannot skip: real tokens per second and cost per million
real tokens on every run, refusals visible instead of averaged away, thresholds set from your
own padding distribution rather than mine. From there the same shape extends outward — spend
per million real tokens by data source, by sequence-length bucket, by model size — because once
the denominator is honest, every downstream comparison is worth making. What I would not ship
is a dashboard whose numbers have never been shown to be wrong.

## Licence

Source-available, evaluation-only — read it, run it, quote it in a review; see `LICENSE`.
