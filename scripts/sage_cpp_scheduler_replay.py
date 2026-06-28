#!/usr/bin/env python3
"""
Validate the persistent C++ SAGE scheduler replay skeleton.

The C++ executable is still model-free: Python prepares replay/probe rows with
proxy/oracle tokens and the FFN-sentinel statistic, then one long-lived C++
process applies `common_sage_decide()`, selects the emitted token, and maintains
generated text per prompt stream.
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

from sage_candidate_probe_fit import load_records, load_replay_labels
from sage_cpp_policy_parity import (
    expected_decision,
    parse_classes,
    probe_by_key,
    replay_label_match,
    replay_oracle_token,
)
from sage_policy_report import load_replay_rows, proxy_accepts, replay_row_key


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "tools" / "llama.cpp-src" / "build-live-migration-cuda" / "bin" / "Release" / "llama-sage-scheduler-replay.exe"
DEFAULT_OUT_DIR = ROOT / "benchmarks"


@dataclass
class SchedulerReplayRow:
    index: int
    task_id: str
    prompt_index: int
    step_index: int
    token_class: str
    proxy_token: str
    oracle_token: str
    expected_action: str
    cpp_action: str
    expected_reason: str
    cpp_reason: str
    expected_selected_token: str
    cpp_selected_token: str
    expected_false_accept: bool
    cpp_false_accept: bool | None
    expected_prefix_before: str
    cpp_prefix_before: str
    expected_generated_text_after: str
    cpp_generated_text_after: str
    action_match: bool
    reason_match: bool
    selected_token_match: bool
    false_accept_match: bool
    prefix_match: bool
    generated_text_match: bool


@dataclass
class SchedulerReplaySummary:
    replay_rows: int
    checked_rows: int
    cpp_invocations: int
    prompt_count: int
    proxy_gate_accepts: int
    verifier_needed: int
    verifier_covered: int
    expected_accept_proxy: int
    expected_oracle_fallback: int
    final_false_accepts: int
    action_matches: int
    reason_matches: int
    selected_token_matches: int
    false_accept_matches: int
    prefix_matches: int
    generated_text_matches: int
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


def cpp_input_row(row: dict[str, Any], task: dict[str, Any], ffn_sum: float | None, label: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": str(task.get("task_id", "")),
        "prompt_index": int(task.get("prompt_index", 0)),
        "step_index": int(task.get("step_index", 0)),
        "prompt": str(task.get("prompt", "")),
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


def run_cpp_scheduler(exe: Path, rows: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sage-cpp-scheduler-") as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / "input.jsonl"
        out_path = tmp_path / "output.json"
        in_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        command = [str(exe), "--jsonl-in", str(in_path), "--json-out", str(out_path)]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            fail("C++ scheduler replay executable timed out")
        if completed.returncode != 0:
            fail(f"C++ scheduler replay executable failed: rc={completed.returncode} stderr={completed.stderr.strip()}")
        if not out_path.is_file():
            fail("C++ scheduler replay executable did not write JSON output")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            fail("C++ scheduler replay executable produced non-object JSON")
        return payload


def run_replay(args: argparse.Namespace) -> tuple[SchedulerReplaySummary, list[SchedulerReplayRow], dict[str, Any]]:
    exe = Path(args.scheduler_replay_exe)
    if not exe.is_file():
        fail(f"scheduler replay executable not found: {exe}")
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
    cpp_inputs: list[dict[str, Any]] = []
    for row in replay_rows:
        task = row["task"]
        probe = probes.get(replay_row_key(task))
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
        cpp_inputs.append(cpp_input_row(row, task, ffn_sum, args.label))

    cpp_payload = run_cpp_scheduler(exe, cpp_inputs, args.timeout)
    cpp_decisions = cpp_payload.get("decisions", [])
    if len(cpp_decisions) != len(prepared):
        fail(f"C++ scheduler decision count mismatch: got {len(cpp_decisions)}, expected {len(prepared)}")

    prefixes: dict[int, str] = {}
    rows: list[SchedulerReplayRow] = []
    for index, ((row, task, _, expected), cpp) in enumerate(zip(prepared, cpp_decisions)):
        prompt_index = int(task.get("prompt_index", 0))
        prefix_before = prefixes.get(prompt_index, "")
        proxy_token = str(task.get("proxy_token", ""))
        oracle_token = replay_oracle_token(row, task)
        top1_match = replay_label_match(row, task, args.label)
        expected_selected = proxy_token if expected["action"] == "accept_proxy" else oracle_token
        expected_generated = prefix_before + expected_selected
        prefixes[prompt_index] = expected_generated
        expected_false_accept = expected["action"] == "accept_proxy" and not top1_match
        cpp_false_accept = cpp.get("false_accept")
        rows.append(
            SchedulerReplayRow(
                index=index,
                task_id=str(task.get("task_id", "")),
                prompt_index=prompt_index,
                step_index=int(task.get("step_index", 0)),
                token_class=str(task.get("token_class", "")),
                proxy_token=proxy_token,
                oracle_token=oracle_token,
                expected_action=str(expected["action"]),
                cpp_action=str(cpp.get("action", "")),
                expected_reason=str(expected["reason"]),
                cpp_reason=str(cpp.get("reason", "")),
                expected_selected_token=expected_selected,
                cpp_selected_token=str(cpp.get("selected_token", "")),
                expected_false_accept=expected_false_accept,
                cpp_false_accept=bool(cpp_false_accept) if isinstance(cpp_false_accept, bool) else None,
                expected_prefix_before=prefix_before,
                cpp_prefix_before=str(cpp.get("prefix_before", "")),
                expected_generated_text_after=expected_generated,
                cpp_generated_text_after=str(cpp.get("generated_text_after", "")),
                action_match=str(cpp.get("action", "")) == expected["action"],
                reason_match=str(cpp.get("reason", "")) == expected["reason"],
                selected_token_match=str(cpp.get("selected_token", "")) == expected_selected,
                false_accept_match=isinstance(cpp_false_accept, bool) and bool(cpp_false_accept) == expected_false_accept,
                prefix_match=str(cpp.get("prefix_before", "")) == prefix_before,
                generated_text_match=str(cpp.get("generated_text_after", "")) == expected_generated,
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
                and item.prefix_match
                and item.generated_text_match
            )
        ),
        None,
    )
    summary = SchedulerReplaySummary(
        replay_rows=len(replay_rows),
        checked_rows=len(rows),
        cpp_invocations=1,
        prompt_count=len(prefixes),
        proxy_gate_accepts=sum(1 for _, task, _, _ in prepared if proxy_accepts(task, args.proxy_max_entropy, args.proxy_min_margin)),
        verifier_needed=sum(1 for item in rows if item.expected_reason in {"runtime_verifier_accept", "runtime_verifier_reject", "missing_runtime_signal"}),
        verifier_covered=sum(1 for _, _, ffn_sum, expected in prepared if ffn_sum is not None and expected["verifier_needed"]),
        expected_accept_proxy=sum(1 for item in rows if item.expected_action == "accept_proxy"),
        expected_oracle_fallback=sum(1 for item in rows if item.expected_action == "oracle_fallback"),
        final_false_accepts=sum(1 for item in rows if item.expected_false_accept),
        action_matches=sum(1 for item in rows if item.action_match),
        reason_matches=sum(1 for item in rows if item.reason_match),
        selected_token_matches=sum(1 for item in rows if item.selected_token_match),
        false_accept_matches=sum(1 for item in rows if item.false_accept_match),
        prefix_matches=sum(1 for item in rows if item.prefix_match),
        generated_text_matches=sum(1 for item in rows if item.generated_text_match),
        first_mismatch_index=first_mismatch.index if first_mismatch is not None else None,
        first_mismatch_task_id=first_mismatch.task_id if first_mismatch is not None else "",
        meets_required_matches=first_mismatch is None and len(rows) > 0,
    )
    return summary, rows, cpp_payload


def print_report(summary: SchedulerReplaySummary) -> None:
    print("# SAGE C++ Scheduler Replay")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key, value in asdict(summary).items():
        print(f"| {key} | {value} |")


def self_test() -> int:
    task = {
        "task_id": "self",
        "token_class": "whitespace",
        "proxy_entropy": 0.5,
        "proxy_margin": 0.1,
    }
    decision = expected_decision(
        task=task,
        ffn_sum=None,
        proxy_max_entropy=0.6,
        proxy_min_margin=1.5,
        verifier_classes={"punct", "capitalized"},
    )
    if decision["action"] != "accept_proxy":
        fail("self-test expected accept_proxy", 1)
    print("SAGE C++ scheduler replay self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate llama-sage-scheduler-replay against replay/probe rows.")
    parser.add_argument("--replay-json", nargs="+", default=[])
    parser.add_argument("--probe-json", nargs="+", default=[])
    parser.add_argument("--scheduler-replay-exe", default=str(DEFAULT_EXE))
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--tag", default="cpp-scheduler-replay")
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

    summary, rows, cpp_payload = run_replay(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-sage-cpp-scheduler-replay-{args.tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "summary": asdict(summary),
        "rows": [asdict(item) for item in rows],
        "cpp_summary": cpp_payload.get("summary", {}),
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
