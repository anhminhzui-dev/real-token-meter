"""Every rule gets a test, and the padding rule gets a test that makes the meter miss.

A guard that has never been shown to miss something certifies nothing, so
test_falsifier_padding_check_disabled_lets_the_inflated_run_win_the_compare switches the
padding guard off through the checks= seam and asserts the padding-inflated run then takes
rank 1 on cost per million real tokens. The public-clean scanner is treated the same way:
five forbidden shapes are planted in memory and each must fire before the clean result over
the shipped tree is allowed to mean anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from real_token_meter.cli import main, render_compare, render_run, summary_document, write_summary
from real_token_meter.meter import (
    CODES,
    DEFAULT_CHECKS,
    DENOMINATOR_PHRASES,
    HaltError,
    load_policy,
    meter_run,
)

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "fixtures" / "policy.json"
CLEAN = ROOT / "fixtures" / "log_clean.jsonl"
BAD = ROOT / "fixtures" / "log_bad.jsonl"
PADDED = ROOT / "fixtures" / "log_padded.jsonl"
NO_PADDING_GUARD = tuple(check for check in DEFAULT_CHECKS if check != "padding")
EXPECTED_BAD = {
    5: ("MALFORMED_ROW",),
    6: ("MISSING_FIELD",),
    7: ("NEGATIVE_OR_ZERO_TIME",),
    8: ("DUPLICATE_STEP",),
    9: ("BAD_TOKEN_COUNTS",),
    10: ("PADDING_INFLATED",),
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "runs"}
# Shapes, never names: no machine, drive, user or person is written out here.
PUBLIC_CLEAN = (
    ("drive_letter_root", re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")),
    ("windows_user_home", re.compile(r"[\\/][Uu]sers[\\/]")),
    ("posix_user_home", re.compile(r"[\\/]home[\\/][A-Za-z0-9._-]+")),
    ("accuracy_claim", re.compile(r"\b(?:MAE|accuracy|band)\s*[:=]?\s*[0-9]", re.IGNORECASE)),
    ("consumer_mailbox", re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|outlook|yahoo)\.[A-Za-z]{2,}")),
)
# Built by concatenation so the forbidden literal never appears in this file.
PLANTS = (
    ("drive_letter_root", "D" + ":" + "\\" + "work" + "\\" + "notes.txt"),
    ("windows_user_home", "\\" + "Users" + "\\" + "someone" + "\\"),
    ("posix_user_home", "/" + "home" + "/someone/notes"),
    ("accuracy_claim", "acc" + "uracy = 0.93"),
    ("consumer_mailbox", "someone" + "@" + "gmail" + ".com"),
)
NETWORK_TOKENS = ("urllib", "socket", "http.client", "requests", "urlopen", "subprocess")


def policy():
    return load_policy(POLICY)


def by_index(report, index):
    return next(step for step in report.steps if step.index == index)


def row(step, pad=200, wall=2.0, run_id="run-tmp", **extra):
    body = {
        "synthetic": True,
        "run_id": run_id,
        "step": step,
        "batch_size": 8,
        "seq_len": 512,
        "pad_tokens": pad,
        "wall_seconds": wall,
        "gpu_hour_rate_synthetic": 2.0,
    }
    body.update(extra)
    return json.dumps(body)


def write_log(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_clean_run_counts_every_step_and_returns_go():
    report = meter_run(CLEAN, policy())
    assert report.verdict == "GO"
    assert (report.steps_read, report.steps_counted, report.steps_refused) == (6, 6, 0)
    assert (report.total_tokens, report.real_tokens) == (24576, 23376)
    assert report.raw_tps == pytest.approx(2048.0)
    assert report.real_tps == pytest.approx(1948.0)
    assert report.padding_ratio == pytest.approx(0.048828, abs=1e-5)
    assert report.cost_per_million_real == pytest.approx(0.28519, abs=1e-4)


def test_seeded_bad_fixture_holds_and_names_exactly_one_code_per_seeded_row():
    report = meter_run(BAD, policy())
    assert report.verdict == "HOLD"
    assert (report.steps_read, report.steps_counted, report.steps_refused) == (10, 4, 6)
    for index, codes in EXPECTED_BAD.items():
        assert by_index(report, index).codes == codes, index
    assert report.run_codes == ()
    assert report.code_counts == {code: 1 for codes in EXPECTED_BAD.values() for code in codes}
    assert len(CODES) == 8 and set(report.code_counts) <= set(CODES)


def test_falsifier_padding_check_disabled_lets_the_inflated_run_win_the_compare():
    guarded = render_compare(meter_run(CLEAN, policy()), meter_run(PADDED, policy()))
    assert "RANK 1: run-packed" in guarded
    assert "run-padded  REFUSED, not ranked on cost (0 counted of 5 read)" in guarded
    assert "CHEAPEST: not computed" in guarded
    left = meter_run(CLEAN, policy(), NO_PADDING_GUARD)
    right = meter_run(PADDED, policy(), NO_PADDING_GUARD)
    unguarded = render_compare(left, right)
    assert right.verdict == "GO" and right.steps_counted == 5
    assert "RANK 1: run-padded" in unguarded
    assert "CHEAPEST: run-padded, 1.41x cheaper per million real tokens than run-packed" in unguarded


def test_run_padding_budget_fires_when_every_step_is_individually_under_budget(tmp_path):
    path = write_log(tmp_path, "creep.jsonl", [row(n, pad=1200) for n in range(1, 5)])
    report = meter_run(path, policy())
    assert report.steps_refused == 0
    assert report.padding_ratio == pytest.approx(0.29297, abs=1e-4)
    assert report.run_codes == ("RUN_PADDING_OVER_BUDGET",) and report.verdict == "HOLD"


def test_too_few_counted_steps_holds_the_run(tmp_path):
    path = write_log(tmp_path, "short.jsonl", [row(1), row(2)])
    report = meter_run(path, policy())
    assert report.run_codes == ("TOO_FEW_STEPS",) and report.verdict == "HOLD"


def test_a_row_without_the_synthetic_marker_halts_the_run(tmp_path):
    body = json.loads(row(1))
    del body["synthetic"]
    path = write_log(tmp_path, "real.jsonl", [json.dumps(body)])
    with pytest.raises(HaltError) as caught:
        meter_run(path, policy())
    assert caught.value.code == "NOT_SYNTHETIC"
    assert meter_run(path, policy(), DEFAULT_CHECKS, allow_nonsynthetic=True).verdict == "HOLD"


def test_two_run_ids_in_one_file_halt_rather_than_merge(tmp_path):
    path = write_log(tmp_path, "mixed.jsonl", [row(1, run_id="run-a"), row(2, run_id="run-b")])
    with pytest.raises(HaltError) as caught:
        meter_run(path, policy())
    assert caught.value.code == "MIXED_RUN_IDS"


def test_the_report_never_prints_a_token_count_without_its_denominator():
    reports = [meter_run(path, policy()) for path in (CLEAN, BAD, PADDED)]
    text = "\n".join([render_run(r) for r in reports] + [render_compare(reports[0], reports[2])])
    lines_with_tokens = [line for line in text.splitlines() if "tokens" in line]
    assert len(lines_with_tokens) >= 12
    for line in lines_with_tokens:
        assert any(phrase in line for phrase in DENOMINATOR_PHRASES), line


def test_the_summary_names_its_denominators_and_is_byte_identical_across_runs(tmp_path):
    for name in ("first", "second"):
        write_summary(str(tmp_path / name), meter_run(BAD, policy()))
    first = (tmp_path / "first" / "summary.json").read_bytes()
    assert first == (tmp_path / "second" / "summary.json").read_bytes()
    document = summary_document(meter_run(BAD, policy()))
    assert document["steps_read"] == 10 and document["steps_counted"] == 4
    assert document["codes"]["PADDING_INFLATED"] == 1
    assert "PADDING_INFLATED" in first.decode("utf-8")
    for key in document:
        if "tokens" in key and "per_million" not in key:
            assert key.endswith("over_counted_steps"), key


def test_compare_refuses_to_rank_one_file_against_itself():
    with pytest.raises(HaltError) as caught:
        render_compare(meter_run(CLEAN, policy()), meter_run(CLEAN, policy()))
    assert caught.value.code == "DUPLICATE_RUN_ID"


def test_public_clean_scanner_fires_on_every_planted_shape_then_the_tree_is_clean():
    def scan(text):
        return sorted({name for name, pattern in PUBLIC_CLEAN if pattern.search(text)})

    for expected, planted in PLANTS:
        assert scan(planted) == [expected], planted
    findings = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            findings[path.name] = scan(path.read_text(encoding="utf-8", errors="ignore"))
    assert {name: hits for name, hits in findings.items() if hits} == {}
    assert len(findings) >= 11


def test_no_network_capable_import_exists_anywhere_under_src():
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in NETWORK_TOKENS:
            assert token not in text, (path.name, token)


def test_cli_returns_zero_on_go_two_on_hold_and_one_on_a_crash(capsys, tmp_path):
    assert main(["meter", "--policy", str(POLICY), "--log", str(CLEAN)]) == 0
    assert "VERDICT: GO" in capsys.readouterr().out
    code = main(
        ["meter", "--policy", str(POLICY), "--log", str(BAD), "--out", str(tmp_path / "runs")]
    )
    assert code == 2 and "VERDICT: HOLD (6 refused of 10 steps read)" in capsys.readouterr().out
    code = main(["compare", "--policy", str(POLICY), "--a", str(CLEAN), "--b", str(PADDED)])
    assert code == 2 and "VERDICT: HOLD (1 of 2 runs refused)" in capsys.readouterr().out
    assert main(["meter", "--policy", str(tmp_path / "nope.json"), "--log", str(CLEAN)]) == 1
    assert "VERDICT: HOLD (crash: FileNotFoundError)" in capsys.readouterr().out
