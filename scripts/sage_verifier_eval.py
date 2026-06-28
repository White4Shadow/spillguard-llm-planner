#!/usr/bin/env python3
"""
Evaluate a frozen SAGE sparse-verifier rule on probe captures.

This script intentionally does not fit thresholds. Use it after a verifier rule
family has already been selected and frozen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sage_candidate_probe_fit import evaluate_expression, load_records, load_replay_labels, summarize


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def ensure_files(paths: list[str], label: str) -> list[Path]:
    out = [Path(item) for item in paths]
    for path in out:
        if not path.is_file():
            fail(f"{label} not found: {path}")
    return out


def print_summary(payload: dict) -> None:
    result = payload["result"]
    print("# SAGE Frozen Sparse Verifier Evaluation")
    print()
    print(f"Rule: `{payload['params']['expression']}`")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key in (
        "total",
        "accepted",
        "rejected",
        "true_accepts",
        "false_accepts",
        "true_rejects",
        "false_rejects",
        "precision",
        "accepted_error_rate",
        "bad_catch_rate",
        "good_reject_rate",
        "skip_rate",
    ):
        value = result[key]
        if isinstance(value, float):
            print(f"| {key} | {value:.6f} |")
        else:
            print(f"| {key} | {value} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen SAGE sparse-verifier expression.")
    parser.add_argument("--probe-json", nargs="+", required=True)
    parser.add_argument("--replay-json", nargs="+", default=[])
    parser.add_argument("--expression", required=True)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    probe_paths = ensure_files(args.probe_json, "probe JSON")
    replay_paths = ensure_files(args.replay_json, "replay JSON") if args.replay_json else []
    replay_labels = load_replay_labels(replay_paths, args.label) if replay_paths else None
    records = load_records(probe_paths, args.label, args.label_quality, replay_labels)
    if not records:
        fail("no candidate records found")
    result = evaluate_expression(records, args.expression)
    payload = {
        "params": {
            "probe_json": args.probe_json,
            "replay_json": args.replay_json,
            "expression": args.expression,
            "label": args.label,
            "label_quality": args.label_quality,
        },
        "summary": summarize(records),
        "result": asdict(result),
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_summary(payload)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
