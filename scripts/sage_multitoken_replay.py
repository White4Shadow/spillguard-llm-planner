#!/usr/bin/env python3
"""
Replay a multi-token SAGE scheduler trace from proxy/oracle replay rows.

This is a deterministic bridge between aggregate policy reports and a future
live scheduler loop. It walks candidate rows in prompt/step order and emits the
scheduler action that would be taken for each candidate:

- reject to oracle when the proxy gate fails;
- accept proxy directly for accepted non-verifier classes;
- use C++ decision events or Python runtime-tensor decisions for verifier rows;
- fall back to oracle if a required verifier signal is missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_policy_report import BYTES_PER_GIB, path_call_ms
from sage_runtime_scheduler import (
    DEFAULT_EXPRESSION,
    make_decisions,
    load_records,
    parse_classes,
)
from sage_candidate_probe_fit import expression_mask


@dataclass
class ReplayDecision:
    task_id: str
    prompt_index: int
    prompt: str
    step_index: int
    token_class: str
    proxy_token: str
    oracle_token: str
    replay_top1_match: bool
    proxy_gate_accept: bool
    verifier_needed: bool
    verifier_covered: bool
    action: str
    reason: str
    decision_source: str
    selected_token: str
    false_accept: bool


@dataclass
class ReplaySummary:
    stats_total: int
    replay_rows: int
    prompt_count: int
    proxy_gate_accepts: int
    proxy_false_accepts: int
    verifier_needed: int
    verifier_covered: int
    verifier_missing: int
    cpp_decisions_used: int
    python_decisions_used: int
    direct_proxy_accepts: int
    final_accepts: int
    oracle_fallbacks: int
    final_false_accepts: int
    prompts_with_false_accept: int
    final_accepted_error_rate: float
    final_total_error_rate: float
    verifier_coverage_rate: float
    accepted_proxy_rate: float
    verifier_call_rate: float
    oracle_call_rate: float
    expected_ms_per_token: float
    effective_tps: float
    meets_target_tps: bool
    meets_accepted_error: bool
    meets_total_error: bool
    meets_verifier_coverage: bool
    meets_all_gates: bool


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


def load_task_id_filter(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.is_file():
        fail(f"tasks filter JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    return {str(item.get("task_id", "")) for item in raw_tasks if isinstance(item, dict) and item.get("task_id")}


def load_replay_rows(paths: list[Path], task_id_filter: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"replay JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("task"), dict):
                if task_id_filter and str(row["task"].get("task_id", "")) not in task_id_filter:
                    continue
                item = dict(row)
                item["_source"] = str(path.resolve())
                rows.append(item)
    rows.sort(
        key=lambda row: (
            str(row["task"].get("prompt", "")),
            int(row["task"].get("step_index", 0)),
            str(row["task"].get("task_id", "")),
        )
    )
    return rows


def row_key_from_task(task: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(task.get("task_id", "")),
        str(task.get("prompt", "")),
        int(task.get("step_index", 0)),
        str(task.get("proxy_token", "")),
    )


def proxy_gate_accepts(task: dict[str, Any], max_entropy: float, min_margin: float) -> bool:
    return float(task.get("proxy_entropy", 0.0)) <= max_entropy or float(task.get("proxy_margin", 0.0)) >= min_margin


def load_cpp_decisions(paths: list[Path]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    out: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            fail(f"runtime JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for capture in payload.get("captures", []):
            task = capture.get("task")
            if not isinstance(task, dict):
                continue
            decisions = capture.get("decisions", [])
            if isinstance(decisions, list) and decisions:
                out[row_key_from_task(task)] = dict(decisions[-1])
    return out


def load_python_decisions(args: argparse.Namespace, paths: list[Path]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    if not paths:
        return {}
    records, captures = load_records(paths, args.label, args.label_quality)
    if not records:
        return {}
    verifier_mask = expression_mask(records, args.expression)
    decisions = make_decisions(
        records=records,
        captures=captures,
        verifier_mask=verifier_mask,
        verifier_classes=parse_classes(args.verifier_token_classes),
        proxy_max_entropy=args.proxy_max_entropy,
        proxy_min_margin=args.proxy_min_margin,
    )
    out: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    # Use exact task keys from captures because prompt text is not stored in SchedulerDecision.
    for capture, decision in zip(captures, decisions):
        task = capture.get("task")
        if isinstance(task, dict):
            out[row_key_from_task(task)] = asdict(decision)
    return out


def choose_runtime_decision(
    *,
    key: tuple[str, str, int, str],
    cpp_decisions: dict[tuple[str, str, int, str], dict[str, Any]],
    python_decisions: dict[tuple[str, str, int, str], dict[str, Any]],
    source: str,
) -> tuple[dict[str, Any] | None, str]:
    if source == "cpp":
        return cpp_decisions.get(key), "cpp" if key in cpp_decisions else "missing"
    if source == "python":
        return python_decisions.get(key), "python" if key in python_decisions else "missing"
    if key in cpp_decisions:
        return cpp_decisions[key], "cpp"
    if key in python_decisions:
        return python_decisions[key], "python"
    return None, "missing"


def replay(args: argparse.Namespace) -> tuple[ReplaySummary, list[ReplayDecision]]:
    replay_paths = expand_paths(args.replay_json)
    runtime_paths = expand_paths(args.runtime_json)
    rows = load_replay_rows(replay_paths, load_task_id_filter(args.tasks_filter_json))
    if not rows:
        fail("no replay rows found")
    stats_total = args.stats_total or len(rows)
    if stats_total < len(rows):
        fail("--stats-total must be >= replay row count")

    verifier_classes = parse_classes(args.verifier_token_classes)
    cpp_decisions = load_cpp_decisions(runtime_paths)
    python_decisions = load_python_decisions(args, runtime_paths)

    decisions: list[ReplayDecision] = []
    for row in rows:
        task = row["task"]
        key = row_key_from_task(task)
        proxy_accept = proxy_gate_accepts(task, args.proxy_max_entropy, args.proxy_min_margin)
        verifier_needed = proxy_accept and str(task.get("token_class", "")) in verifier_classes
        replay_match = bool(row.get(f"replay_top1_{args.label}_match", False))
        action = "oracle_fallback"
        reason = "proxy_gate_reject"
        decision_source = "proxy_gate"
        verifier_covered = False

        if proxy_accept and not verifier_needed:
            action = "accept_proxy"
            reason = "proxy_gate_non_verifier_class"
            decision_source = "proxy_gate"
        elif verifier_needed:
            runtime_decision, decision_source = choose_runtime_decision(
                key=key,
                cpp_decisions=cpp_decisions,
                python_decisions=python_decisions,
                source=args.decision_source,
            )
            if runtime_decision is not None:
                verifier_covered = bool(runtime_decision.get("verifier_covered", False))
                action = str(runtime_decision.get("action", "oracle_fallback"))
                reason = str(runtime_decision.get("reason", "runtime_decision"))
            else:
                reason = "missing_runtime_signal"

        proxy_token = str(task.get("proxy_token", ""))
        oracle_token = str(row.get("oracle_token", task.get("replay_oracle_token", task.get("oracle_token", ""))))
        selected = proxy_token if action == "accept_proxy" else oracle_token
        decisions.append(
            ReplayDecision(
                task_id=str(task.get("task_id", "")),
                prompt_index=int(task.get("prompt_index", 0)),
                prompt=str(task.get("prompt", "")),
                step_index=int(task.get("step_index", 0)),
                token_class=str(task.get("token_class", "")),
                proxy_token=proxy_token,
                oracle_token=oracle_token,
                replay_top1_match=replay_match,
                proxy_gate_accept=proxy_accept,
                verifier_needed=verifier_needed,
                verifier_covered=verifier_covered,
                action=action,
                reason=reason,
                decision_source=decision_source,
                selected_token=selected,
                false_accept=action == "accept_proxy" and not replay_match,
            )
        )

    summary = summarize(decisions, rows, args, stats_total)
    return summary, decisions


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def summarize(decisions: list[ReplayDecision], rows: list[dict[str, Any]], args: argparse.Namespace, stats_total: int) -> ReplaySummary:
    prompts = {item.prompt for item in decisions}
    proxy_gate_accepts = sum(1 for item in decisions if item.proxy_gate_accept)
    proxy_false_accepts = sum(1 for item in decisions if item.proxy_gate_accept and not item.replay_top1_match)
    verifier_needed = sum(1 for item in decisions if item.verifier_needed)
    verifier_covered = sum(1 for item in decisions if item.verifier_covered)
    verifier_missing = verifier_needed - verifier_covered
    cpp_decisions_used = sum(1 for item in decisions if item.decision_source == "cpp")
    python_decisions_used = sum(1 for item in decisions if item.decision_source == "python")
    direct_proxy_accepts = sum(1 for item in decisions if item.reason == "proxy_gate_non_verifier_class")
    final_accepts = sum(1 for item in decisions if item.action == "accept_proxy")
    oracle_fallbacks = sum(1 for item in decisions if item.action == "oracle_fallback")
    final_false_accepts = sum(1 for item in decisions if item.false_accept)
    prompts_with_false_accept = len({item.prompt_index for item in decisions if item.false_accept})

    proxy_ms = 1000.0 / args.proxy_tps
    verifier_ms = path_call_ms(args.params_b, args.quant_bpw, args.verifier_active_percent, args.pcie_gbps, args.verifier_compute_ms, args.verifier_fixed_ms)
    oracle_ms = path_call_ms(args.params_b, args.quant_bpw, args.oracle_active_percent, args.pcie_gbps, args.oracle_compute_ms, args.oracle_fixed_ms)
    verifier_call_rate = verifier_needed / stats_total
    oracle_call_rate = (stats_total - final_accepts) / stats_total
    expected_ms = proxy_ms + verifier_call_rate * verifier_ms + oracle_call_rate * oracle_ms
    effective_tps = 1000.0 / expected_ms
    accepted_error = final_false_accepts / final_accepts if final_accepts else 0.0
    total_error = final_false_accepts / stats_total
    coverage = verifier_covered / verifier_needed if verifier_needed else 1.0
    meets_tps = effective_tps >= args.target_tps
    meets_accepted = accepted_error <= args.max_accepted_error
    meets_total = total_error <= args.max_total_error
    meets_coverage = not args.require_full_verifier_coverage or verifier_missing == 0
    return ReplaySummary(
        stats_total=stats_total,
        replay_rows=len(rows),
        prompt_count=len(prompts),
        proxy_gate_accepts=proxy_gate_accepts,
        proxy_false_accepts=proxy_false_accepts,
        verifier_needed=verifier_needed,
        verifier_covered=verifier_covered,
        verifier_missing=verifier_missing,
        cpp_decisions_used=cpp_decisions_used,
        python_decisions_used=python_decisions_used,
        direct_proxy_accepts=direct_proxy_accepts,
        final_accepts=final_accepts,
        oracle_fallbacks=oracle_fallbacks,
        final_false_accepts=final_false_accepts,
        prompts_with_false_accept=prompts_with_false_accept,
        final_accepted_error_rate=accepted_error,
        final_total_error_rate=total_error,
        verifier_coverage_rate=coverage,
        accepted_proxy_rate=final_accepts / stats_total,
        verifier_call_rate=verifier_call_rate,
        oracle_call_rate=oracle_call_rate,
        expected_ms_per_token=expected_ms,
        effective_tps=effective_tps,
        meets_target_tps=meets_tps,
        meets_accepted_error=meets_accepted,
        meets_total_error=meets_total,
        meets_verifier_coverage=meets_coverage,
        meets_all_gates=meets_tps and meets_accepted and meets_total and meets_coverage,
    )


def print_report(summary: ReplaySummary, decisions: list[ReplayDecision], top: int) -> None:
    print("# SAGE Multi-Token Scheduler Replay")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key, value in asdict(summary).items():
        if isinstance(value, bool):
            print(f"| {key} | {value} |")
        elif isinstance(value, float):
            print(f"| {key} | {value:.6f} |")
        else:
            print(f"| {key} | {value} |")
    if top > 0:
        print()
        print("| Prompt | Step | Task | Action | Source | Match | Proxy | Oracle |")
        print("| ---: | ---: | --- | --- | --- | --- | --- | --- |")
        for item in decisions[:top]:
            proxy = item.proxy_token.replace("|", "\\|")
            oracle = item.oracle_token.replace("|", "\\|")
            print(
                f"| {item.prompt_index} | {item.step_index} | {item.task_id} | {item.action} "
                f"| {item.decision_source} | {item.replay_top1_match} | `{proxy}` | `{oracle}` |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a multi-token SAGE scheduler trace from replay rows and runtime decisions.")
    parser.add_argument("--replay-json", nargs="+", required=True)
    parser.add_argument("--runtime-json", nargs="+", default=[])
    parser.add_argument("--tasks-filter-json", default="", help="optional tasks JSON; only replay rows with those task_id values are used")
    parser.add_argument("--decision-source", choices=["prefer-cpp", "cpp", "python"], default="prefer-cpp")
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--stats-total", type=int, default=0)
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--max-accepted-error", type=float, default=0.05)
    parser.add_argument("--max-total-error", type=float, default=0.02)
    parser.add_argument("--proxy-tps", type=float, default=25.0)
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--verifier-active-percent", type=float, default=1.0)
    parser.add_argument("--oracle-active-percent", type=float, default=10.0)
    parser.add_argument("--verifier-compute-ms", type=float, default=10.0)
    parser.add_argument("--verifier-fixed-ms", type=float, default=5.0)
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--require-full-verifier-coverage", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.stats_total < 0:
        parser.error("--stats-total must be non-negative")
    if args.proxy_tps <= 0:
        parser.error("--proxy-tps must be positive")
    summary, decisions = replay(args)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "summary": asdict(summary),
        "decisions": [asdict(item) for item in decisions],
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(summary, decisions, args.top)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    if args.require_pass and not summary.meets_all_gates:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
