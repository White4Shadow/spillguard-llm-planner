#!/usr/bin/env python3
"""
Compare the model-free C++ SAGE policy executable against replay/probe rows.

This validates the persistent-runtime policy path without loading a model:
Python prepares the same proxy and FFN-sentinel inputs used by replay
validation, then `llama-sage-policy-check` calls `common_sage_decide()`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_candidate_probe_fit import CandidateRecord, load_records, load_replay_labels
from sage_policy_report import load_replay_rows, proxy_accepts, replay_row_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "tools" / "llama.cpp-src" / "build-live-migration-cuda" / "bin" / "Release" / "llama-sage-policy-check.exe"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
FROZEN_ENTROPY_ACCEPT = 0.040003470206379677
FROZEN_FFN_SUM_ACCEPT = -72.150276000000005


@dataclass
class ParityRow:
    index: int
    task_id: str
    prompt_index: int
    step_index: int
    token_class: str
    proxy_token: str
    oracle_token: str
    top1_match: bool
    proxy_entropy: float
    proxy_margin: float
    has_ffn_norm_0: bool
    ffn_norm_0_sum: float | None
    expected_action: str
    expected_reason: str
    expected_selected_token: str
    expected_false_accept: bool
    cpp_action: str
    cpp_reason: str
    cpp_selected_token: str
    cpp_false_accept: bool | None
    action_match: bool
    reason_match: bool
    selected_token_match: bool
    false_accept_match: bool
    proxy_gate_accept_match: bool
    verifier_needed_match: bool
    verifier_covered_match: bool
    verifier_accept_match: bool


@dataclass
class ParitySummary:
    replay_rows: int
    checked_rows: int
    probe_records: int
    cpp_mode: str
    cpp_invocations: int
    proxy_gate_accepts: int
    verifier_needed: int
    verifier_covered: int
    verifier_missing: int
    expected_accept_proxy: int
    expected_oracle_fallback: int
    final_false_accepts: int
    action_matches: int
    reason_matches: int
    selected_token_matches: int
    false_accept_matches: int
    proxy_gate_accept_matches: int
    verifier_needed_matches: int
    verifier_covered_matches: int
    verifier_accept_matches: int
    first_mismatch_index: int | None
    first_mismatch_task_id: str
    meets_required_matches: bool


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        paths.extend(matches if matches else [Path(value)])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def ensure_files(paths: list[Path], label: str) -> list[Path]:
    if not paths:
        fail(f"no {label} files provided")
    for path in paths:
        if not path.is_file():
            fail(f"{label} not found: {path}")
    return paths


def record_key(record: CandidateRecord) -> tuple[str, str, int, str]:
    return (str(record.task_id), str(record.prompt), int(record.step_index), str(record.proxy_token))


def probe_by_key(records: list[CandidateRecord]) -> dict[tuple[str, str, int, str], CandidateRecord]:
    out: dict[tuple[str, str, int, str], CandidateRecord] = {}
    for record in records:
        out[record_key(record)] = record
    return out


def expected_decision(
    *,
    task: dict[str, Any],
    ffn_sum: float | None,
    proxy_max_entropy: float,
    proxy_min_margin: float,
    verifier_classes: set[str],
) -> dict[str, Any]:
    proxy_gate_accept = proxy_accepts(task, proxy_max_entropy, proxy_min_margin)
    verifier_needed = proxy_gate_accept and str(task.get("token_class", "")) in verifier_classes
    verifier_covered = verifier_needed and ffn_sum is not None
    verifier_accept = False
    if verifier_covered:
        verifier_accept = float(task.get("proxy_entropy", 0.0)) <= FROZEN_ENTROPY_ACCEPT or float(ffn_sum) >= FROZEN_FFN_SUM_ACCEPT
    action = "oracle_fallback"
    reason = "proxy_gate_reject"
    if proxy_gate_accept and not verifier_needed:
        action = "accept_proxy"
        reason = "proxy_gate_non_verifier_class"
    elif verifier_needed and not verifier_covered:
        reason = "missing_runtime_signal"
    elif verifier_needed and verifier_accept:
        action = "accept_proxy"
        reason = "runtime_verifier_accept"
    elif verifier_needed:
        reason = "runtime_verifier_reject"
    return {
        "proxy_gate_accept": proxy_gate_accept,
        "verifier_needed": verifier_needed,
        "verifier_covered": verifier_covered,
        "verifier_accept": verifier_accept if verifier_covered else None,
        "action": action,
        "reason": reason,
    }


def replay_label_match(row: dict[str, Any], task: dict[str, Any], label: str) -> bool:
    replay_key = f"replay_top1_{label}_match"
    if replay_key in row:
        return bool(row.get(replay_key, False))
    return bool(task.get(f"top1_{label}_match", False))


def replay_oracle_token(row: dict[str, Any], task: dict[str, Any]) -> str:
    if "oracle_token" in row:
        return str(row.get("oracle_token", ""))
    return str(task.get("oracle_token", ""))


def cpp_input_row(row: dict[str, Any], task: dict[str, Any], ffn_sum: float | None, label: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": str(task.get("task_id", "")),
        "token_class": str(task.get("token_class", "")),
        "proxy_token": str(task.get("proxy_token", "")),
        "oracle_token": replay_oracle_token(row, task),
        "top1_match": replay_label_match(row, task, label),
        "proxy_entropy": float(task.get("proxy_entropy", 0.0)),
        "proxy_margin": float(task.get("proxy_margin", 0.0)),
    }
    if ffn_sum is not None:
        payload["ffn_norm_0_sum"] = float(ffn_sum)
        payload["ffn_norm_0_sequence"] = int(task.get("step_index", 0))
    return payload


def run_cpp_policy(exe: Path, row: dict[str, Any], task: dict[str, Any], ffn_sum: float | None, label: str, timeout: int) -> dict[str, Any]:
    return run_cpp_policy_batch(exe, [(row, task, ffn_sum)], label, timeout)[0]


def run_cpp_policy_batch(
    exe: Path,
    tasks: list[tuple[dict[str, Any], dict[str, Any], float | None]],
    label: str,
    timeout: int,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="sage-cpp-policy-") as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / "input.jsonl"
        out_path = tmp_path / "output.jsonl"
        in_path.write_text(
            "\n".join(json.dumps(cpp_input_row(row, task, ffn_sum, label), ensure_ascii=False) for row, task, ffn_sum in tasks) + "\n",
            encoding="utf-8",
        )
        command = [str(exe), "--jsonl-in", str(in_path), "--jsonl-out", str(out_path)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            fail("C++ policy executable timed out in JSONL batch mode")
        if completed.returncode != 0:
            fail(f"C++ policy executable failed in JSONL batch mode: rc={completed.returncode} stderr={completed.stderr.strip()}")
        if not out_path.is_file():
            fail("C++ policy executable did not write JSONL output")
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(tasks):
            fail(f"C++ policy JSONL output row count mismatch: got {len(rows)}, expected {len(tasks)}")
        return rows


def parse_classes(raw: str) -> set[str]:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or {"punct", "capitalized"}


def compare_bool(cpp: dict[str, Any], expected: dict[str, Any], key: str) -> bool:
    return bool(cpp.get(key, False)) == bool(expected.get(key, False))


def run_parity(args: argparse.Namespace) -> tuple[ParitySummary, list[ParityRow]]:
    exe = Path(args.policy_check_exe)
    if not exe.is_file():
        fail(f"policy check executable not found: {exe}")
    replay_paths = ensure_files(expand_paths(args.replay_json), "replay JSON")
    probe_paths = ensure_files(expand_paths(args.probe_json), "probe JSON")

    replay_rows = load_replay_rows(replay_paths)
    if args.limit > 0:
        replay_rows = replay_rows[args.offset : args.offset + args.limit]
    elif args.offset:
        replay_rows = replay_rows[args.offset :]
    if not replay_rows:
        fail("no replay rows selected")

    replay_labels = load_replay_labels(replay_paths, args.label)
    records = load_records(probe_paths, args.label, args.label_quality, replay_labels)
    probes = probe_by_key(records)
    verifier_classes = parse_classes(args.verifier_token_classes)

    prepared: list[tuple[dict[str, Any], dict[str, Any], float | None, dict[str, Any]]] = []
    for index, row in enumerate(replay_rows):
        task = row["task"]
        key = replay_row_key(task)
        probe = probes.get(key)
        ffn_sum = None
        if probe is not None and "ffn_norm_0.sum" in probe.features:
            ffn_sum = float(probe.features["ffn_norm_0.sum"])
        expected = expected_decision(
            task=task,
            ffn_sum=ffn_sum,
            proxy_max_entropy=args.proxy_max_entropy,
            proxy_min_margin=args.proxy_min_margin,
            verifier_classes=verifier_classes,
        )
        prepared.append((row, task, ffn_sum, expected))

    if args.per_row_process:
        cpp_rows = [run_cpp_policy(exe, row, task, ffn_sum, args.label, args.timeout) for row, task, ffn_sum, _ in prepared]
        cpp_mode = "per_row_process"
        cpp_invocations = len(cpp_rows)
    else:
        cpp_rows = run_cpp_policy_batch(exe, [(row, task, ffn_sum) for row, task, ffn_sum, _ in prepared], args.label, args.timeout)
        cpp_mode = "jsonl_batch"
        cpp_invocations = 1

    rows: list[ParityRow] = []
    for index, ((row, task, ffn_sum, expected), cpp) in enumerate(zip(prepared, cpp_rows)):
        proxy_token = str(task.get("proxy_token", ""))
        oracle_token = replay_oracle_token(row, task)
        top1_match = replay_label_match(row, task, args.label)
        expected_selected_token = proxy_token if expected["action"] == "accept_proxy" else oracle_token
        expected_false_accept = expected["action"] == "accept_proxy" and not top1_match
        cpp_false_accept = cpp.get("false_accept")
        verifier_accept_match = cpp.get("verifier_accept") == expected["verifier_accept"]
        rows.append(
            ParityRow(
                index=index,
                task_id=str(task.get("task_id", "")),
                prompt_index=int(task.get("prompt_index", 0)),
                step_index=int(task.get("step_index", 0)),
                token_class=str(task.get("token_class", "")),
                proxy_token=proxy_token,
                oracle_token=oracle_token,
                top1_match=top1_match,
                proxy_entropy=float(task.get("proxy_entropy", 0.0)),
                proxy_margin=float(task.get("proxy_margin", 0.0)),
                has_ffn_norm_0=ffn_sum is not None,
                ffn_norm_0_sum=ffn_sum,
                expected_action=str(expected["action"]),
                expected_reason=str(expected["reason"]),
                expected_selected_token=expected_selected_token,
                expected_false_accept=expected_false_accept,
                cpp_action=str(cpp.get("action", "")),
                cpp_reason=str(cpp.get("reason", "")),
                cpp_selected_token=str(cpp.get("selected_token", "")),
                cpp_false_accept=bool(cpp_false_accept) if isinstance(cpp_false_accept, bool) else None,
                action_match=str(cpp.get("action", "")) == expected["action"],
                reason_match=str(cpp.get("reason", "")) == expected["reason"],
                selected_token_match=str(cpp.get("selected_token", "")) == expected_selected_token,
                false_accept_match=isinstance(cpp_false_accept, bool) and bool(cpp_false_accept) == expected_false_accept,
                proxy_gate_accept_match=compare_bool(cpp, expected, "proxy_gate_accept"),
                verifier_needed_match=compare_bool(cpp, expected, "verifier_needed"),
                verifier_covered_match=compare_bool(cpp, expected, "verifier_covered"),
                verifier_accept_match=verifier_accept_match,
            )
        )

    first_mismatch = next(
        (
            item
            for item in rows
            if not (
                item.action_match
                and item.reason_match
                and item.selected_token_match
                and item.false_accept_match
                and item.proxy_gate_accept_match
                and item.verifier_needed_match
                and item.verifier_covered_match
                and item.verifier_accept_match
            )
        ),
        None,
    )
    summary = ParitySummary(
        replay_rows=len(replay_rows),
        checked_rows=len(rows),
        probe_records=len(records),
        cpp_mode=cpp_mode,
        cpp_invocations=cpp_invocations,
        proxy_gate_accepts=sum(1 for item in rows if item.expected_action == "accept_proxy" or item.expected_reason != "proxy_gate_reject"),
        verifier_needed=sum(1 for item in rows if item.expected_reason in {"runtime_verifier_accept", "runtime_verifier_reject", "missing_runtime_signal"}),
        verifier_covered=sum(1 for item in rows if item.has_ffn_norm_0 and item.expected_reason in {"runtime_verifier_accept", "runtime_verifier_reject"}),
        verifier_missing=sum(1 for item in rows if item.expected_reason == "missing_runtime_signal"),
        expected_accept_proxy=sum(1 for item in rows if item.expected_action == "accept_proxy"),
        expected_oracle_fallback=sum(1 for item in rows if item.expected_action == "oracle_fallback"),
        final_false_accepts=sum(1 for item in rows if item.expected_false_accept),
        action_matches=sum(1 for item in rows if item.action_match),
        reason_matches=sum(1 for item in rows if item.reason_match),
        selected_token_matches=sum(1 for item in rows if item.selected_token_match),
        false_accept_matches=sum(1 for item in rows if item.false_accept_match),
        proxy_gate_accept_matches=sum(1 for item in rows if item.proxy_gate_accept_match),
        verifier_needed_matches=sum(1 for item in rows if item.verifier_needed_match),
        verifier_covered_matches=sum(1 for item in rows if item.verifier_covered_match),
        verifier_accept_matches=sum(1 for item in rows if item.verifier_accept_match),
        first_mismatch_index=first_mismatch.index if first_mismatch is not None else None,
        first_mismatch_task_id=first_mismatch.task_id if first_mismatch is not None else "",
        meets_required_matches=first_mismatch is None and len(rows) > 0,
    )
    return summary, rows


def print_report(summary: ParitySummary) -> None:
    print("# SAGE C++ Policy Parity")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key, value in asdict(summary).items():
        if isinstance(value, bool):
            print(f"| {key} | {value} |")
        else:
            print(f"| {key} | {value} |")


def self_test() -> int:
    task = {
        "task_id": "self",
        "token_class": "punct",
        "proxy_entropy": 0.5,
        "proxy_margin": 0.1,
    }
    decision = expected_decision(
        task=task,
        ffn_sum=-50.0,
        proxy_max_entropy=0.6,
        proxy_min_margin=1.5,
        verifier_classes={"punct", "capitalized"},
    )
    if decision["action"] != "accept_proxy" or decision["reason"] != "runtime_verifier_accept":
        fail("self-test expected verifier accept", 1)
    decision = expected_decision(
        task=task,
        ffn_sum=-100.0,
        proxy_max_entropy=0.6,
        proxy_min_margin=1.5,
        verifier_classes={"punct", "capitalized"},
    )
    if decision["action"] != "oracle_fallback" or decision["reason"] != "runtime_verifier_reject":
        fail("self-test expected verifier reject", 1)
    print("SAGE C++ policy parity self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare llama-sage-policy-check against replay/probe rows.")
    parser.add_argument("--replay-json", nargs="+", default=[])
    parser.add_argument("--probe-json", nargs="+", default=[])
    parser.add_argument("--policy-check-exe", default=str(DEFAULT_EXE))
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--per-row-process", action="store_true", help="invoke C++ executable once per row instead of JSONL batch mode")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--tag", default="cpp-policy-parity")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.replay_json:
        parser.error("--replay-json is required unless --self-test is used")
    if not args.probe_json:
        parser.error("--probe-json is required unless --self-test is used")
    if args.offset < 0 or args.limit < 0:
        parser.error("--offset and --limit must be non-negative")

    summary, rows = run_parity(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-sage-cpp-policy-parity-{args.tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "summary": asdict(summary),
        "rows": [asdict(item) for item in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(summary)
        print()
        print(f"wrote: {out_path.resolve()}")
    if args.require_pass and not summary.meets_required_matches:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
