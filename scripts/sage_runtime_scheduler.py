#!/usr/bin/env python3
"""
Turn SAGE runtime tensor captures into scheduler decisions.

This is the first scheduler-facing bridge after the runtime/debug parity gate:
given normal llama-completion captures with task metadata and tensor summaries,
it applies the frozen proxy gate plus the frozen FFN-sentinel verifier rule and
emits explicit accept/fallback decisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_candidate_probe_fit import CandidateRecord, expression_mask, features_for_capture, label_quality_matches


DEFAULT_EXPRESSION = "(proxy.entropy <= 0.040003470206379677) OR (ffn_norm_0.sum >= -72.150276000000005)"


@dataclass
class SchedulerDecision:
    source: str
    capture_index: int
    task_id: str
    prompt_index: int
    step_index: int
    token_class: str
    proxy_token: str
    oracle_token: str
    top1_match: bool
    proxy_margin: float
    proxy_entropy: float
    tensor_count: int
    proxy_gate_accept: bool
    verifier_needed: bool
    verifier_covered: bool
    verifier_accept: bool | None
    action: str
    reason: str
    false_accept: bool


@dataclass
class SchedulerSummary:
    captures: int
    proxy_gate_accepts: int
    verifier_needed: int
    verifier_covered: int
    verifier_missing: int
    verifier_accepts: int
    verifier_rejects: int
    accepted_proxy: int
    oracle_fallbacks: int
    false_accepts: int
    true_accepts: int
    true_rejects: int
    false_rejects: int
    accepted_error_rate: float
    oracle_fallback_rate: float
    verifier_coverage_rate: float
    meets_accepted_error: bool
    meets_verifier_coverage: bool
    meets_all_gates: bool


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def parse_classes(raw: str) -> set[str]:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or {"punct", "capitalized"}


def label_match(task: dict[str, Any], label: str) -> bool:
    replay_key = f"replay_top1_{label}_match"
    if replay_key in task:
        return bool(task.get(replay_key, False))
    return bool(task.get(f"top1_{label}_match", False))


def oracle_token(task: dict[str, Any]) -> str:
    if "replay_oracle_token" in task:
        return str(task.get("replay_oracle_token", ""))
    return str(task.get("oracle_token", ""))


def load_records(paths: list[Path], label: str, label_quality: str) -> tuple[list[CandidateRecord], list[dict[str, Any]]]:
    records: list[CandidateRecord] = []
    captures: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"runtime JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for capture in payload.get("captures", []):
            task = capture.get("task")
            if not isinstance(task, dict):
                continue
            if not label_quality_matches(task, label_quality):
                continue
            task_id = str(task.get("task_id", f"capture-{len(records) + 1:04d}"))
            features = features_for_capture(capture, task)
            records.append(
                CandidateRecord(
                    task_id=task_id,
                    prompt_index=int(task.get("prompt_index", 0)),
                    prompt=str(task.get("prompt", capture.get("prompt", ""))),
                    step_index=int(task.get("step_index", 0)),
                    proxy_token=str(task.get("proxy_token", "")),
                    oracle_token=oracle_token(task),
                    token_class=str(task.get("token_class", "")),
                    top1_match=label_match(task, label),
                    label_source="oracle-replay" if f"replay_top1_{label}_match" in task else "continuation",
                    label_quality=str(task.get("label_quality", "")),
                    proxy_margin=float(task.get("proxy_margin", 0.0)),
                    proxy_entropy=float(task.get("proxy_entropy", 0.0)),
                    tensor_count=int(capture.get("tensor_count", 0)),
                    features=features,
                )
            )
            item = dict(capture)
            item["_source"] = str(path.resolve())
            captures.append(item)
    return records, captures


def proxy_gate_accepts(record: CandidateRecord, max_entropy: float, min_margin: float) -> bool:
    return record.proxy_entropy <= max_entropy or record.proxy_margin >= min_margin


def make_decisions(
    *,
    records: list[CandidateRecord],
    captures: list[dict[str, Any]],
    verifier_mask: tuple[bool, ...],
    verifier_classes: set[str],
    proxy_max_entropy: float,
    proxy_min_margin: float,
) -> list[SchedulerDecision]:
    decisions: list[SchedulerDecision] = []
    for record, capture, verifier_accept in zip(records, captures, verifier_mask):
        proxy_accept = proxy_gate_accepts(record, proxy_max_entropy, proxy_min_margin)
        verifier_needed = proxy_accept and record.token_class in verifier_classes
        verifier_covered = verifier_needed and record.tensor_count > 0
        action = "oracle_fallback"
        reason = "proxy_gate_reject"
        final_verifier_accept: bool | None = None
        if proxy_accept and not verifier_needed:
            action = "accept_proxy"
            reason = "proxy_gate_non_verifier_class"
        elif verifier_needed and not verifier_covered:
            reason = "missing_runtime_signal"
        elif verifier_needed:
            final_verifier_accept = bool(verifier_accept)
            if final_verifier_accept:
                action = "accept_proxy"
                reason = "runtime_verifier_accept"
            else:
                reason = "runtime_verifier_reject"

        decisions.append(
            SchedulerDecision(
                source=str(capture.get("_source", "")),
                capture_index=int(capture.get("index", 0)),
                task_id=record.task_id,
                prompt_index=record.prompt_index,
                step_index=record.step_index,
                token_class=record.token_class,
                proxy_token=record.proxy_token,
                oracle_token=record.oracle_token,
                top1_match=record.top1_match,
                proxy_margin=record.proxy_margin,
                proxy_entropy=record.proxy_entropy,
                tensor_count=record.tensor_count,
                proxy_gate_accept=proxy_accept,
                verifier_needed=verifier_needed,
                verifier_covered=verifier_covered,
                verifier_accept=final_verifier_accept,
                action=action,
                reason=reason,
                false_accept=action == "accept_proxy" and not record.top1_match,
            )
        )
    return decisions


def summarize(decisions: list[SchedulerDecision], max_accepted_error: float, require_full_verifier_coverage: bool) -> SchedulerSummary:
    captures = len(decisions)
    proxy_gate_accepts = sum(1 for item in decisions if item.proxy_gate_accept)
    verifier_needed = sum(1 for item in decisions if item.verifier_needed)
    verifier_covered = sum(1 for item in decisions if item.verifier_covered)
    verifier_missing = verifier_needed - verifier_covered
    verifier_accepts = sum(1 for item in decisions if item.verifier_accept is True)
    verifier_rejects = sum(1 for item in decisions if item.verifier_accept is False)
    accepted_proxy = sum(1 for item in decisions if item.action == "accept_proxy")
    oracle_fallbacks = sum(1 for item in decisions if item.action == "oracle_fallback")
    false_accepts = sum(1 for item in decisions if item.false_accept)
    true_accepts = sum(1 for item in decisions if item.action == "accept_proxy" and item.top1_match)
    true_rejects = sum(1 for item in decisions if item.verifier_accept is False and not item.top1_match)
    false_rejects = sum(1 for item in decisions if item.verifier_accept is False and item.top1_match)
    accepted_error_rate = false_accepts / accepted_proxy if accepted_proxy else 0.0
    oracle_fallback_rate = oracle_fallbacks / captures if captures else 0.0
    verifier_coverage_rate = verifier_covered / verifier_needed if verifier_needed else 1.0
    meets_accepted_error = accepted_error_rate <= max_accepted_error
    meets_verifier_coverage = not require_full_verifier_coverage or verifier_missing == 0
    return SchedulerSummary(
        captures=captures,
        proxy_gate_accepts=proxy_gate_accepts,
        verifier_needed=verifier_needed,
        verifier_covered=verifier_covered,
        verifier_missing=verifier_missing,
        verifier_accepts=verifier_accepts,
        verifier_rejects=verifier_rejects,
        accepted_proxy=accepted_proxy,
        oracle_fallbacks=oracle_fallbacks,
        false_accepts=false_accepts,
        true_accepts=true_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        accepted_error_rate=accepted_error_rate,
        oracle_fallback_rate=oracle_fallback_rate,
        verifier_coverage_rate=verifier_coverage_rate,
        meets_accepted_error=meets_accepted_error,
        meets_verifier_coverage=meets_verifier_coverage,
        meets_all_gates=meets_accepted_error and meets_verifier_coverage,
    )


def print_report(summary: SchedulerSummary, decisions: list[SchedulerDecision], top: int) -> None:
    print("# SAGE Runtime Scheduler Decisions")
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
        print("| Task | Class | Action | Reason | Match | Proxy | Oracle | ffn? |")
        print("| --- | --- | --- | --- | --- | --- | --- | ---: |")
        for item in decisions[:top]:
            verifier = "" if item.verifier_accept is None else str(item.verifier_accept)
            proxy = item.proxy_token.replace("|", "\\|")
            oracle = item.oracle_token.replace("|", "\\|")
            print(
                f"| {item.task_id} | {item.token_class} | {item.action} | {item.reason} "
                f"| {item.top1_match} | `{proxy}` | `{oracle}` | {verifier} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply frozen SAGE scheduler decisions to runtime captures.")
    parser.add_argument("--runtime-json", nargs="+", required=True)
    parser.add_argument("--expression", default=DEFAULT_EXPRESSION)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--max-accepted-error", type=float, default=0.05)
    parser.add_argument("--require-full-verifier-coverage", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    paths = [Path(item) for item in args.runtime_json]
    records, captures = load_records(paths, args.label, args.label_quality)
    if not records:
        parser.error("no runtime captures with task metadata found")
    verifier_mask = expression_mask(records, args.expression)
    decisions = make_decisions(
        records=records,
        captures=captures,
        verifier_mask=verifier_mask,
        verifier_classes=parse_classes(args.verifier_token_classes),
        proxy_max_entropy=args.proxy_max_entropy,
        proxy_min_margin=args.proxy_min_margin,
    )
    summary = summarize(decisions, args.max_accepted_error, args.require_full_verifier_coverage)
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
