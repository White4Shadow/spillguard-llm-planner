#!/usr/bin/env python3
"""
Evaluate a frozen SAGE proxy gate against oracle replay labels.

This intentionally does not fit thresholds. Use it for validation sets after a
candidate policy has already been selected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GateSummary:
    records: int
    matches: int
    mismatches: int
    accepted: int
    rejected: int
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    accepted_error_rate: float
    bad_catch_rate: float
    good_reject_rate: float
    accepted_proxy_rate: float | None
    total_error_rate: float | None
    oracle_call_rate: float | None


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("task"), dict):
                rows.append(row)
    return rows


def gate_accepts(task: dict[str, Any], max_entropy: float, min_margin: float) -> bool:
    entropy = float(task.get("proxy_entropy", 0.0))
    margin = float(task.get("proxy_margin", 0.0))
    return entropy <= max_entropy or margin >= min_margin


def summarize(rows: list[dict[str, Any]], max_entropy: float, min_margin: float, stats_total: int) -> GateSummary:
    accepted_indices = {idx for idx, row in enumerate(rows) if gate_accepts(row["task"], max_entropy, min_margin)}
    accepted_rows = [row for idx, row in enumerate(rows) if idx in accepted_indices]
    rejected_rows = [row for idx, row in enumerate(rows) if idx not in accepted_indices]
    matches = sum(1 for row in rows if bool(row.get("replay_top1_token_match", False)))
    mismatches = len(rows) - matches
    true_accepts = sum(1 for row in accepted_rows if bool(row.get("replay_top1_token_match", False)))
    false_accepts = len(accepted_rows) - true_accepts
    true_rejects = sum(1 for row in rejected_rows if not bool(row.get("replay_top1_token_match", False)))
    false_rejects = len(rejected_rows) - true_rejects
    accepted_error_rate = false_accepts / len(accepted_rows) if accepted_rows else 0.0
    bad_catch_rate = true_rejects / mismatches if mismatches else 0.0
    good_reject_rate = false_rejects / matches if matches else 0.0
    accepted_proxy_rate = None
    total_error_rate = None
    oracle_call_rate = None
    if stats_total > 0:
        if stats_total < len(rows):
            fail("--stats-total must be >= replay row count")
        accepted_proxy_rate = len(accepted_rows) / stats_total
        total_error_rate = false_accepts / stats_total
        oracle_call_rate = (stats_total - len(accepted_rows)) / stats_total
    return GateSummary(
        records=len(rows),
        matches=matches,
        mismatches=mismatches,
        accepted=len(accepted_rows),
        rejected=len(rejected_rows),
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        accepted_error_rate=accepted_error_rate,
        bad_catch_rate=bad_catch_rate,
        good_reject_rate=good_reject_rate,
        accepted_proxy_rate=accepted_proxy_rate,
        total_error_rate=total_error_rate,
        oracle_call_rate=oracle_call_rate,
    )


def print_summary(summary: GateSummary, max_entropy: float, min_margin: float) -> None:
    print("# SAGE Frozen Gate Evaluation")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    print(f"| Gate | entropy <= {max_entropy:.6g} OR margin >= {min_margin:.6g} |")
    for key, value in asdict(summary).items():
        if isinstance(value, float):
            print(f"| {key} | {value:.6f} |")
        elif value is None:
            print(f"| {key} |  |")
        else:
            print(f"| {key} | {value} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen SAGE proxy gate on replay JSON files.")
    parser.add_argument("--replay-json", nargs="+", required=True)
    parser.add_argument("--max-entropy", type=float, required=True)
    parser.add_argument("--min-margin", type=float, required=True)
    parser.add_argument("--stats-total", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows = load_rows([Path(item) for item in args.replay_json])
    if not rows:
        fail("no replay rows found")
    summary = summarize(rows, args.max_entropy, args.min_margin, args.stats_total)
    payload = {
        "params": {
            "replay_json": args.replay_json,
            "max_entropy": args.max_entropy,
            "min_margin": args.min_margin,
            "stats_total": args.stats_total,
        },
        "summary": asdict(summary),
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_summary(summary, args.max_entropy, args.min_margin)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
