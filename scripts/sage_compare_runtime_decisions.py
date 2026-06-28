#!/usr/bin/env python3
"""
Compare in-process SAGE decision events against Python scheduler decisions.

The C++ runtime hook emits one decision JSONL event per llama-completion run.
This gate proves that event matches the frozen Python policy for the same task
metadata and tensor summaries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_candidate_probe_fit import expression_mask
from sage_runtime_scheduler import (
    DEFAULT_EXPRESSION,
    make_decisions,
    load_records,
    parse_classes,
)


EXACT_FIELDS = (
    "proxy_gate_accept",
    "verifier_needed",
    "verifier_covered",
    "verifier_accept",
    "action",
    "reason",
)


@dataclass
class DecisionMismatch:
    task_id: str
    field: str
    runtime_value: Any
    expected_value: Any
    abs_error: float | None = None


@dataclass
class DecisionCompareReport:
    runtime_json: list[str]
    captures: int
    runtime_decisions: int
    expected_decisions: int
    matched_decisions: int
    missing_runtime_decisions: int
    extra_runtime_decisions: int
    mismatches: list[DecisionMismatch]
    abs_tol: float
    passed: bool
    created_at: str


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def latest_ffn_sum(capture: dict[str, Any]) -> float | None:
    value: float | None = None
    for tensor in capture.get("tensors", []):
        if tensor.get("name") == "ffn_norm-0":
            value = float(tensor.get("sum", 0.0))
    return value


def decision_by_task(captures: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int, int]:
    out: dict[str, dict[str, Any]] = {}
    total = 0
    extras = 0
    for capture in captures:
        task = capture.get("task") if isinstance(capture.get("task"), dict) else {}
        task_id = str(task.get("task_id", f"capture-{capture.get('index', 0)}"))
        decisions = capture.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        total += len(decisions)
        if len(decisions) > 1:
            extras += len(decisions) - 1
        if decisions:
            out[task_id] = decisions[-1]
    return out, total, extras


def close(a: Any, b: Any, abs_tol: float) -> tuple[bool, float]:
    if a is None and b is None:
        return True, 0.0
    if a is None or b is None:
        return False, math.inf
    af = float(a)
    bf = float(b)
    if math.isnan(af) and math.isnan(bf):
        return True, 0.0
    err = abs(af - bf)
    return err <= abs_tol, err


def load_raw_captures(paths: list[Path]) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"runtime JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for capture in payload.get("captures", []):
            if isinstance(capture, dict):
                item = dict(capture)
                item["_source"] = str(path.resolve())
                captures.append(item)
    return captures


def compare(args: argparse.Namespace) -> DecisionCompareReport:
    paths = [Path(item) for item in args.runtime_json]
    records, captures = load_records(paths, args.label, args.label_quality)
    raw_captures = load_raw_captures(paths)
    if len(raw_captures) != len(captures):
        fail("raw captures and scheduler captures differ; check task metadata filters")
    verifier_mask = expression_mask(records, args.expression)
    expected = make_decisions(
        records=records,
        captures=captures,
        verifier_mask=verifier_mask,
        verifier_classes=parse_classes(args.verifier_token_classes),
        proxy_max_entropy=args.proxy_max_entropy,
        proxy_min_margin=args.proxy_min_margin,
    )
    expected_by_task = {item.task_id: item for item in expected}
    runtime_by_task, runtime_decisions, extra_runtime_decisions = decision_by_task(raw_captures)
    mismatches: list[DecisionMismatch] = []
    missing = 0
    matched = 0

    capture_by_task: dict[str, dict[str, Any]] = {}
    for capture in raw_captures:
        task = capture.get("task") if isinstance(capture.get("task"), dict) else {}
        task_id = str(task.get("task_id", f"capture-{capture.get('index', 0)}"))
        capture_by_task[task_id] = capture

    for task_id, expected_decision in expected_by_task.items():
        runtime = runtime_by_task.get(task_id)
        if runtime is None:
            missing += 1
            mismatches.append(DecisionMismatch(task_id, "<missing>", None, asdict(expected_decision)))
            continue
        matched += 1
        if str(runtime.get("candidate_id", "")) != task_id:
            mismatches.append(DecisionMismatch(task_id, "candidate_id", runtime.get("candidate_id"), task_id))
        for field in EXACT_FIELDS:
            expected_value = getattr(expected_decision, field)
            runtime_value = runtime.get(field)
            if runtime_value != expected_value:
                mismatches.append(DecisionMismatch(task_id, field, runtime_value, expected_value))
        expected_sum = latest_ffn_sum(capture_by_task[task_id])
        ok, err = close(runtime.get("ffn_norm_0_sum"), expected_sum, args.abs_tol)
        if not ok:
            mismatches.append(
                DecisionMismatch(
                    task_id=task_id,
                    field="ffn_norm_0_sum",
                    runtime_value=runtime.get("ffn_norm_0_sum"),
                    expected_value=expected_sum,
                    abs_error=err,
                )
            )

    extra_task_ids = sorted(set(runtime_by_task) - set(expected_by_task))
    for task_id in extra_task_ids:
        mismatches.append(DecisionMismatch(task_id, "<extra>", runtime_by_task[task_id], None))

    return DecisionCompareReport(
        runtime_json=[str(path.resolve()) for path in paths],
        captures=len(raw_captures),
        runtime_decisions=runtime_decisions,
        expected_decisions=len(expected),
        matched_decisions=matched,
        missing_runtime_decisions=missing,
        extra_runtime_decisions=extra_runtime_decisions + len(extra_task_ids),
        mismatches=mismatches,
        abs_tol=args.abs_tol,
        passed=not mismatches,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def print_report(report: DecisionCompareReport) -> None:
    print("# SAGE Runtime Decision Parity")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    print(f"| captures | {report.captures} |")
    print(f"| runtime_decisions | {report.runtime_decisions} |")
    print(f"| expected_decisions | {report.expected_decisions} |")
    print(f"| matched_decisions | {report.matched_decisions} |")
    print(f"| missing_runtime_decisions | {report.missing_runtime_decisions} |")
    print(f"| extra_runtime_decisions | {report.extra_runtime_decisions} |")
    print(f"| mismatches | {len(report.mismatches)} |")
    print(f"| passed | {report.passed} |")
    if report.mismatches:
        print()
        print("First mismatches:")
        for mismatch in report.mismatches[:10]:
            suffix = "" if mismatch.abs_error is None else f" abs_error={mismatch.abs_error:.6g}"
            print(f"- {mismatch.task_id} {mismatch.field}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare in-process SAGE decisions against Python scheduler decisions.")
    parser.add_argument("--runtime-json", nargs="+", required=True)
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = compare(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_report(report)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    if args.require_match and not report.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
