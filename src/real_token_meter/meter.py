"""Throughput and cost over REAL tokens, with a refusal for every number that cannot be trusted.

Raw tokens per second counts padding as work. A run that pads every sequence to a fixed length
and a run that packs to length can post the same raw figure while one of them buys far less
learning per unit of spend. So the numerator here is batch_size * seq_len - pad_tokens, and any
step whose padding exceeds the policy budget is refused rather than averaged in.

No network, no clock, no model, no randomness: the same log bytes and the same policy bytes
always produce the same report, which is why every report carries the hash of what it read.
Every figure goes through ratio(), so a figure with no denominator is MISSING, never 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

SECONDS_PER_HOUR = 3600.0
PER_MILLION = 1_000_000.0
REQUIRED_POLICY_FIELDS = ("policy_id", "max_step_padding_ratio", "max_run_padding_ratio",
                          "min_steps", "required_fields")
STEP_CODES = ("MALFORMED_ROW", "MISSING_FIELD", "DUPLICATE_STEP", "NEGATIVE_OR_ZERO_TIME",
              "BAD_TOKEN_COUNTS", "PADDING_INFLATED")
RUN_CODES = ("TOO_FEW_STEPS", "RUN_PADDING_OVER_BUDGET")
CODES = STEP_CODES + RUN_CODES
# The switchable guards. "padding" owns both padding rules: the per-step budget, and the run
# budget that a fleet of just-under-budget steps would otherwise slip past together.
DEFAULT_CHECKS = ("missing_field", "duplicate_step", "negative_or_zero_time",
                  "bad_token_counts", "padding")
# The only phrases allowed to stand next to a printed token count. A test enforces it.
DENOMINATOR_PHRASES = ("steps counted", "step counted", "not counted in the run")


class HaltError(Exception):
    """A run-level refusal: the run stops and the verdict is HOLD, never a silent pass."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ratio(numerator: float, denominator: float) -> float | None:
    """No denominator, no number. Every figure in this package is built here."""
    return None if denominator <= 0 else numerator / denominator


@dataclass(frozen=True)
class Policy:
    policy_id: str
    max_step_padding_ratio: float
    max_run_padding_ratio: float
    min_steps: int
    required_fields: tuple[str, ...]
    sha256: str


def load_policy(path: str | Path, allow_nonsynthetic: bool = False) -> Policy:
    raw = Path(path).read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HaltError("MALFORMED_POLICY_FILE", type(exc).__name__) from exc
    if not isinstance(doc, dict):
        raise HaltError("MALFORMED_POLICY_FILE", "top level is not an object")
    if doc.get("synthetic") is not True and not allow_nonsynthetic:
        raise HaltError("NOT_SYNTHETIC", "the policy file carries no synthetic marker")
    for name in REQUIRED_POLICY_FIELDS:
        if name not in doc:
            raise HaltError("MISSING_POLICY_FIELD", name)
    return Policy(str(doc["policy_id"]), float(doc["max_step_padding_ratio"]),
                  float(doc["max_run_padding_ratio"]), int(doc["min_steps"]),
                  tuple(str(name) for name in doc["required_fields"]), sha256_bytes(raw))


@dataclass(frozen=True)
class Step:
    index: int
    step: str
    run_id: str = ""
    total_tokens: int = 0
    pad_tokens: int = 0
    wall_seconds: float = 0.0
    rate: float = 0.0
    codes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool: return not self.codes

    @property
    def real_tokens(self) -> int: return self.total_tokens - self.pad_tokens

    @property
    def padding_ratio(self) -> float | None: return ratio(self.pad_tokens, self.total_tokens)

    @property
    def raw_tps(self) -> float | None: return ratio(self.total_tokens, self.wall_seconds)

    @property
    def real_tps(self) -> float | None: return ratio(self.real_tokens, self.wall_seconds)

    @property
    def cost(self) -> float: return self.wall_seconds / SECONDS_PER_HOUR * self.rate

    @property
    def cost_per_million_real(self) -> float | None:
        return ratio(self.cost, self.real_tokens / PER_MILLION)


def check_row(index: int, line: str, policy: Policy, seen: set[str], checks: Sequence[str],
              allow_nonsynthetic: bool = False) -> Step:
    """One log line to one Step. A row that cannot be trusted is refused, never repaired."""
    try:
        row: Any = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("row is not an object")
    except (json.JSONDecodeError, ValueError):
        return Step(index=index, step="?", codes=("MALFORMED_ROW",))
    if row.get("synthetic") is not True and not allow_nonsynthetic:
        raise HaltError("NOT_SYNTHETIC", f"row {index} carries no synthetic marker")
    step_id, run_id = str(row.get("step", "?")), str(row.get("run_id", ""))
    if "missing_field" in checks:
        if [name for name in policy.required_fields if name not in row]:
            return Step(index, step_id, run_id, codes=("MISSING_FIELD",))
    try:
        total = int(row["batch_size"]) * int(row["seq_len"])
        pad, wall = int(row["pad_tokens"]), float(row["wall_seconds"])
        rate = float(row["gpu_hour_rate_synthetic"])
    except (KeyError, TypeError, ValueError):
        return Step(index, step_id, run_id, codes=("MISSING_FIELD",))
    codes: list[str] = []
    if "duplicate_step" in checks and step_id in seen:
        codes.append("DUPLICATE_STEP")
    if "negative_or_zero_time" in checks and wall <= 0:
        codes.append("NEGATIVE_OR_ZERO_TIME")
    if "bad_token_counts" in checks and (total <= 0 or pad < 0 or pad >= total):
        codes.append("BAD_TOKEN_COUNTS")
    elif "padding" in checks and total > 0 and pad / total > policy.max_step_padding_ratio:
        codes.append("PADDING_INFLATED")
    seen.add(step_id)
    return Step(index, step_id, run_id, total, pad, wall, rate, tuple(codes))


@dataclass(frozen=True)
class RunReport:
    """Every aggregate below is summed over COUNTED steps only; refused steps are reported,
    never averaged in, and `steps_counted` is the denominator printed beside every figure."""

    run_id: str
    log_sha256: str
    policy: Policy
    steps: tuple[Step, ...]
    run_codes: tuple[str, ...] = ()

    def summed(self, field: str) -> Any:
        return sum(getattr(step, field) for step in self.counted)

    @property
    def counted(self) -> tuple[Step, ...]: return tuple(s for s in self.steps if s.ok)

    @property
    def steps_read(self) -> int: return len(self.steps)

    @property
    def steps_counted(self) -> int: return len(self.counted)

    @property
    def steps_refused(self) -> int: return self.steps_read - self.steps_counted

    @property
    def total_tokens(self) -> int: return self.summed("total_tokens")

    @property
    def pad_tokens(self) -> int: return self.summed("pad_tokens")

    @property
    def real_tokens(self) -> int: return self.summed("real_tokens")

    @property
    def wall_seconds(self) -> float: return self.summed("wall_seconds")

    @property
    def cost(self) -> float: return self.summed("cost")

    @property
    def padding_ratio(self) -> float | None: return ratio(self.pad_tokens, self.total_tokens)

    @property
    def raw_tps(self) -> float | None: return ratio(self.total_tokens, self.wall_seconds)

    @property
    def real_tps(self) -> float | None: return ratio(self.real_tokens, self.wall_seconds)

    @property
    def cost_per_million_real(self) -> float | None:
        return ratio(self.cost, self.real_tokens / PER_MILLION)

    @property
    def code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for code in [c for step in self.steps for c in step.codes] + list(self.run_codes):
            counts[code] = counts.get(code, 0) + 1
        return counts

    @property
    def verdict(self) -> str:
        return "HOLD" if self.steps_refused or self.run_codes else "GO"


def meter_run(path: str | Path, policy: Policy, checks: Sequence[str] = DEFAULT_CHECKS,
              allow_nonsynthetic: bool = False) -> RunReport:
    """Read one run log, refuse what cannot be trusted, aggregate only what survived."""
    raw = Path(path).read_bytes()
    seen: set[str] = set()
    steps = tuple(check_row(index, line, policy, seen, checks, allow_nonsynthetic)
                  for index, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1)
                  if line.strip())
    labels = {step.run_id for step in steps if step.run_id}
    if len(labels) > 1:
        raise HaltError("MIXED_RUN_IDS", ", ".join(sorted(labels)))
    report = RunReport(labels.pop() if labels else "unnamed-run", sha256_bytes(raw), policy, steps)
    codes: list[str] = []
    if report.steps_counted < policy.min_steps:
        codes.append("TOO_FEW_STEPS")
    run_pad = report.padding_ratio
    if "padding" in checks and run_pad is not None and run_pad > policy.max_run_padding_ratio:
        codes.append("RUN_PADDING_OVER_BUDGET")
    return replace(report, run_codes=tuple(codes))
