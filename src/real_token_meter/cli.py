"""Command line and rendering. Exit 0 = GO, 2 = HOLD, 1 = bad usage or a crash.

A crash is not a pass: an unhandled exception still prints a HOLD line before it leaves.
Every printed token count carries a denominator phrase from meter.DENOMINATOR_PHRASES; a test
walks the rendered report and fails on any token count that stands on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .meter import DEFAULT_CHECKS, HaltError, RunReport, Step, load_policy, meter_run

EXIT_GO, EXIT_USAGE, EXIT_HOLD = 0, 1, 2


def num(value: float | None, digits: int = 2) -> str:
    return "MISSING" if value is None else f"{value:,.{digits}f}"


def pct(value: float | None) -> str:
    return "MISSING" if value is None else f"{value * 100:.2f}%"


def cost_phrase(report: RunReport) -> str:
    """The headline figure, never printed without the steps it was computed over."""
    if report.cost_per_million_real is None:
        return f"MISSING ({report.steps_counted} counted of {report.steps_read} read)"
    return (f"{report.cost_per_million_real:.6f} per million real tokens "
            f"over {report.steps_counted} steps counted")


def render_policy(report: RunReport) -> str:
    p = report.policy
    return (f"POLICY: {p.policy_id}  sha={p.sha256[:16]}  step_pad_max={p.max_step_padding_ratio:.2f}"
            f"  run_pad_max={p.max_run_padding_ratio:.2f}  min_steps={p.min_steps}")


def render_step(step: Step) -> str:
    if step.ok:
        return (f"step {step.step:>3}  ok       raw {num(step.raw_tps)} tok/s  "
                f"real {num(step.real_tps)} tok/s  pad {pct(step.padding_ratio)} of "
                f"{step.total_tokens:,} tokens over 1 step counted  "
                f"cost/Mreal {num(step.cost_per_million_real, 6)}")
    line = f"step {step.step:>3}  REFUSED  {','.join(step.codes)}"
    if step.total_tokens > 0:
        line += (f"  pad {pct(step.padding_ratio)} of {step.total_tokens:,} tokens, "
                 "not counted in the run")
    return line


def render_run(report: RunReport) -> str:
    lines = [f"RUN: {report.run_id}  log_sha={report.log_sha256[:16]}", render_policy(report),
             f"STEPS: {report.steps_read} read, {report.steps_counted} counted, "
             f"{report.steps_refused} refused"]
    if report.steps_counted == 0:
        blank = f"MISSING ({report.steps_counted} counted of {report.steps_read} read)"
        lines += [f"TOKENS: {blank}", f"THROUGHPUT: {blank}", f"COST: {blank}"]
    else:
        over = f"over {report.steps_counted} steps counted"
        lines += [
            f"TOKENS: {report.total_tokens:,} total tokens {over}; {report.real_tokens:,} "
            f"real tokens {over}; padding {pct(report.padding_ratio)} of total",
            f"THROUGHPUT: raw {num(report.raw_tps)} tok/s, real {num(report.real_tps)} tok/s "
            f"({num(report.wall_seconds)} wall seconds {over})",
            f"COST: {report.cost:.6f} at the synthetic rate {over}; {cost_phrase(report)}",
        ]
    lines.append("PER STEP:")
    lines += ["  " + render_step(step) for step in report.steps]
    counts = report.code_counts
    named = " ".join(f"{code}={n}" for code, n in sorted(counts.items()))
    lines.append("REFUSALS: " + (named if counts else "none"))
    if report.verdict == "GO":
        lines.append("VERDICT: GO")
    else:
        detail = f"{report.steps_refused} refused of {report.steps_read} steps read"
        if report.run_codes:
            detail += "; run codes: " + ",".join(report.run_codes)
        lines.append(f"VERDICT: HOLD ({detail})")
    return "\n".join(lines)


def render_compare(left: RunReport, right: RunReport) -> str:
    """Rank two runs on cost per million REAL tokens. A refused run is never ranked on cost."""
    if left.run_id == right.run_id:
        raise HaltError("DUPLICATE_RUN_ID", left.run_id)
    width = max(len(left.run_id), len(right.run_id))
    lines = [render_policy(left)]
    for label, report in (("A", left), ("B", right)):
        lines.append(f"RUN {label}: {report.run_id:<{width}}  {report.verdict:<4}  "
                     f"{cost_phrase(report)}  ({report.steps_read} read, "
                     f"{report.steps_refused} refused)")
    ranked = sorted((r for r in (left, right)
                     if r.verdict == "GO" and r.cost_per_million_real is not None),
                    key=lambda r: (r.cost_per_million_real, r.run_id))
    refused = sorted((r for r in (left, right) if r.run_id not in {x.run_id for x in ranked}),
                     key=lambda r: r.run_id)
    for place, report in enumerate(ranked, 1):
        lines.append(f"RANK {place}: {report.run_id:<{width}}  {cost_phrase(report)}")
    for place, report in enumerate(refused, len(ranked) + 1):
        lines.append(f"RANK {place}: {report.run_id:<{width}}  REFUSED, not ranked on cost "
                     f"({report.steps_counted} counted of {report.steps_read} read)")
    if len(ranked) == 2:
        best, worst = ranked
        factor = worst.cost_per_million_real / best.cost_per_million_real
        if abs(factor - 1.0) < 1e-12:
            lines.append(f"CHEAPEST: TIE - {best.run_id} and {worst.run_id} at the same cost per "
                         f"million real tokens, over {best.steps_counted} and "
                         f"{worst.steps_counted} steps counted")
        else:
            lines.append(f"CHEAPEST: {best.run_id}, {factor:.2f}x cheaper per million real tokens "
                         f"than {worst.run_id}, over {best.steps_counted} steps counted against "
                         f"{worst.steps_counted} steps counted")
    else:
        names = ", ".join(r.run_id for r in refused)
        lines.append(f"CHEAPEST: not computed - {names} refused, so no ratio exists")
    holds = sum(1 for r in (left, right) if r.verdict == "HOLD")
    lines.append("VERDICT: GO" if holds == 0 else f"VERDICT: HOLD ({holds} of 2 runs refused)")
    return "\n".join(lines)


def summary_document(report: RunReport) -> dict[str, Any]:
    """Content only: no timestamp, no host, no path, so two runs write identical bytes.
    Every metric key names the denominator the metric was computed over."""
    return {"version": __version__, "run_id": report.run_id, "verdict": report.verdict,
            "log_sha256": report.log_sha256, "policy_id": report.policy.policy_id,
            "policy_sha256": report.policy.sha256, "steps_read": report.steps_read,
            "steps_counted": report.steps_counted, "steps_refused": report.steps_refused,
            "total_tokens_over_counted_steps": report.total_tokens,
            "real_tokens_over_counted_steps": report.real_tokens,
            "raw_tokens_per_second_over_counted_steps": report.raw_tps,
            "real_tokens_per_second_over_counted_steps": report.real_tps,
            "padding_ratio_over_counted_steps": report.padding_ratio,
            "cost_per_million_real_tokens_synthetic_rate": report.cost_per_million_real,
            "codes": report.code_counts}


def write_summary(out_dir: str, report: RunReport) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary_document(report), indent=2, sort_keys=True) + "\n"
    (out / "summary.json").write_text(text, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="real-token-meter",
        description="Training throughput and cost over REAL tokens, padding excluded, with "
                    "every figure printed next to its denominator.")
    parser.add_argument("--version", action="version", version=__version__)
    verbs = parser.add_subparsers(dest="verb", required=True)
    meter_verb = verbs.add_parser("meter", help="meter one run log against one policy")
    meter_verb.add_argument("--log", required=True)
    meter_verb.add_argument("--out", default=None, help="directory for summary.json")
    compare_verb = verbs.add_parser("compare", help="rank two run logs by cost per million real")
    compare_verb.add_argument("--a", required=True)
    compare_verb.add_argument("--b", required=True)
    for verb in (meter_verb, compare_verb):
        verb.add_argument("--policy", required=True)
        verb.add_argument("--allow-nonsynthetic", action="store_true",
                          help="permit inputs without the synthetic marker (off by default)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_GO if exc.code in (0, None) else EXIT_USAGE
    try:
        policy = load_policy(args.policy, args.allow_nonsynthetic)
        if args.verb == "meter":
            report = meter_run(args.log, policy, DEFAULT_CHECKS, args.allow_nonsynthetic)
            print(render_run(report))
            if args.out:
                write_summary(args.out, report)
            return EXIT_GO if report.verdict == "GO" else EXIT_HOLD
        left = meter_run(args.a, policy, DEFAULT_CHECKS, args.allow_nonsynthetic)
        right = meter_run(args.b, policy, DEFAULT_CHECKS, args.allow_nonsynthetic)
        print(render_compare(left, right))
        return EXIT_GO if left.verdict == right.verdict == "GO" else EXIT_HOLD
    except HaltError as halt:
        print(f"HALT: {halt}\nVERDICT: HOLD (halt: {halt.code})")
        return EXIT_HOLD
    except Exception as exc:  # a crash must refuse out loud, never pass quietly
        print(f"VERDICT: HOLD (crash: {type(exc).__name__})")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
