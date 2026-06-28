#!/usr/bin/env python3
"""
Report end-to-end SAGE policy quality and throughput from measured artifacts.

This combines:
- replay labels for proxy/oracle candidate tasks;
- the frozen proxy gate;
- optional frozen sparse-verifier probe captures;
- an active-byte throughput model.

Rows that need a verifier but are missing from the probe capture are treated as
oracle fallbacks for quality accounting. The report therefore stays conservative
when a probe run is partial.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_candidate_probe_fit import expression_mask, load_records, load_replay_labels


BYTES_PER_GIB = 1024**3


@dataclass
class PolicySummary:
    stats_total: int
    replay_rows: int
    proxy_accepted: int
    proxy_false_accepts: int
    proxy_accepted_error_rate: float
    proxy_total_error_rate: float
    verifier_needed: int
    verifier_covered: int
    verifier_missing: int
    verifier_coverage_rate: float
    verifier_accepted: int
    verifier_false_accepts: int
    verifier_true_rejects: int
    verifier_false_rejects: int
    verifier_bad_catch_rate: float
    verifier_good_reject_rate: float
    final_accepted: int
    final_false_accepts: int
    final_accepted_error_rate: float
    final_total_error_rate: float
    accepted_proxy_rate: float
    verifier_call_rate: float
    oracle_call_rate: float
    proxy_ms_per_token: float
    verifier_call_ms: float
    oracle_call_ms: float
    expected_ms_per_token: float
    effective_tps: float
    meets_target_tps: bool
    meets_accepted_error: bool
    meets_total_error: bool
    meets_verifier_coverage: bool
    meets_all_gates: bool


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def ensure_files(paths: list[str], label: str) -> list[Path]:
    out = [Path(item) for item in paths]
    for path in out:
        if not path.is_file():
            fail(f"{label} not found: {path}")
    return out


def parse_classes(raw: str) -> set[str]:
    classes = {part.strip() for part in raw.split(",") if part.strip()}
    return classes or {"punct", "capitalized"}


def replay_row_key(task: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(task.get("task_id", "")),
        str(task.get("prompt", "")),
        int(task.get("step_index", 0)),
        str(task.get("proxy_token", "")),
    )


def record_key(record: Any) -> tuple[str, str, int, str]:
    return (str(record.task_id), str(record.prompt), int(record.step_index), str(record.proxy_token))


def load_replay_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("task"), dict):
                rows.append(row)
    return rows


def proxy_accepts(task: dict[str, Any], max_entropy: float, min_margin: float) -> bool:
    return float(task.get("proxy_entropy", 0.0)) <= max_entropy or float(task.get("proxy_margin", 0.0)) >= min_margin


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def path_call_ms(params_b: float, quant_bpw: float, active_percent: float, pcie_gbps: float, compute_ms: float, fixed_ms: float) -> float:
    dense_gib = dense_weight_gib(params_b, quant_bpw)
    active_gib = dense_gib * active_percent / 100.0
    transfer_ms = active_gib / (pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)) * 1000.0
    return transfer_ms + compute_ms + fixed_ms


def summarize(args: argparse.Namespace) -> PolicySummary:
    replay_paths = ensure_files(args.replay_json, "replay JSON")
    replay_rows = load_replay_rows(replay_paths)
    if not replay_rows:
        fail("no replay rows found")
    if args.stats_total < len(replay_rows):
        fail("--stats-total must be >= replay row count")

    verifier_classes = parse_classes(args.verifier_token_classes)
    accepted_rows = [row for row in replay_rows if proxy_accepts(row["task"], args.proxy_max_entropy, args.proxy_min_margin)]
    proxy_false_accepts = sum(1 for row in accepted_rows if not bool(row.get("replay_top1_token_match", False)))
    verifier_needed_rows = [row for row in accepted_rows if str(row["task"].get("token_class", "")) in verifier_classes]
    other_accepted_rows = [row for row in accepted_rows if str(row["task"].get("token_class", "")) not in verifier_classes]
    other_false_accepts = sum(1 for row in other_accepted_rows if not bool(row.get("replay_top1_token_match", False)))

    verifier_by_key: dict[tuple[str, str, int, str], tuple[bool, bool]] = {}
    if args.probe_json:
        probe_paths = ensure_files(args.probe_json, "probe JSON")
        replay_labels = load_replay_labels(replay_paths, args.label)
        records = load_records(probe_paths, args.label, args.label_quality, replay_labels)
        if not records:
            fail("no verifier records found")
        mask = expression_mask(records, args.expression)
        verifier_by_key = {record_key(record): (accepted, bool(record.top1_match)) for record, accepted in zip(records, mask)}

    verifier_covered = 0
    verifier_missing = 0
    verifier_accepted = 0
    verifier_false_accepts = 0
    verifier_true_rejects = 0
    verifier_false_rejects = 0
    verifier_bad_total = sum(1 for row in verifier_needed_rows if not bool(row.get("replay_top1_token_match", False)))
    verifier_good_total = len(verifier_needed_rows) - verifier_bad_total

    for row in verifier_needed_rows:
        key = replay_row_key(row["task"])
        item = verifier_by_key.get(key)
        if item is None:
            verifier_missing += 1
            continue
        verifier_covered += 1
        accepted, match = item
        if accepted:
            verifier_accepted += 1
            if not match:
                verifier_false_accepts += 1
        else:
            if match:
                verifier_false_rejects += 1
            else:
                verifier_true_rejects += 1

    final_accepted = len(other_accepted_rows) + verifier_accepted
    final_false_accepts = other_false_accepts + verifier_false_accepts

    proxy_ms = 1000.0 / args.proxy_tps
    verifier_ms = path_call_ms(args.params_b, args.quant_bpw, args.verifier_active_percent, args.pcie_gbps, args.verifier_compute_ms, args.verifier_fixed_ms)
    oracle_ms = path_call_ms(args.params_b, args.quant_bpw, args.oracle_active_percent, args.pcie_gbps, args.oracle_compute_ms, args.oracle_fixed_ms)
    verifier_call_rate = len(verifier_needed_rows) / args.stats_total
    oracle_call_rate = (args.stats_total - final_accepted) / args.stats_total
    expected_ms = proxy_ms + verifier_call_rate * verifier_ms + oracle_call_rate * oracle_ms
    effective_tps = 1000.0 / expected_ms

    meets_target_tps = effective_tps >= args.target_tps
    meets_accepted_error = (final_false_accepts / final_accepted if final_accepted else 0.0) <= args.max_accepted_error
    meets_total_error = (final_false_accepts / args.stats_total) <= args.max_total_error
    meets_verifier_coverage = not args.require_full_verifier_coverage or verifier_missing == 0

    return PolicySummary(
        stats_total=args.stats_total,
        replay_rows=len(replay_rows),
        proxy_accepted=len(accepted_rows),
        proxy_false_accepts=proxy_false_accepts,
        proxy_accepted_error_rate=proxy_false_accepts / len(accepted_rows) if accepted_rows else 0.0,
        proxy_total_error_rate=proxy_false_accepts / args.stats_total,
        verifier_needed=len(verifier_needed_rows),
        verifier_covered=verifier_covered,
        verifier_missing=verifier_missing,
        verifier_coverage_rate=verifier_covered / len(verifier_needed_rows) if verifier_needed_rows else 1.0,
        verifier_accepted=verifier_accepted,
        verifier_false_accepts=verifier_false_accepts,
        verifier_true_rejects=verifier_true_rejects,
        verifier_false_rejects=verifier_false_rejects,
        verifier_bad_catch_rate=verifier_true_rejects / verifier_bad_total if verifier_bad_total else 0.0,
        verifier_good_reject_rate=verifier_false_rejects / verifier_good_total if verifier_good_total else 0.0,
        final_accepted=final_accepted,
        final_false_accepts=final_false_accepts,
        final_accepted_error_rate=final_false_accepts / final_accepted if final_accepted else 0.0,
        final_total_error_rate=final_false_accepts / args.stats_total,
        accepted_proxy_rate=final_accepted / args.stats_total,
        verifier_call_rate=verifier_call_rate,
        oracle_call_rate=oracle_call_rate,
        proxy_ms_per_token=proxy_ms,
        verifier_call_ms=verifier_ms,
        oracle_call_ms=oracle_ms,
        expected_ms_per_token=expected_ms,
        effective_tps=effective_tps,
        meets_target_tps=meets_target_tps,
        meets_accepted_error=meets_accepted_error,
        meets_total_error=meets_total_error,
        meets_verifier_coverage=meets_verifier_coverage,
        meets_all_gates=meets_target_tps and meets_accepted_error and meets_total_error and meets_verifier_coverage,
    )


def print_report(summary: PolicySummary) -> None:
    print("# SAGE Policy Report")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Report SAGE proxy + verifier + oracle policy metrics.")
    parser.add_argument("--replay-json", nargs="+", required=True)
    parser.add_argument("--probe-json", nargs="+", default=[])
    parser.add_argument("--expression", default="(proxy.entropy <= 0.040003470206379677) OR (ffn_norm_0.sum >= -72.150276000000005)")
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="any")
    parser.add_argument("--stats-total", type=int, required=True)
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.stats_total <= 0:
        parser.error("--stats-total must be positive")
    if args.proxy_tps <= 0:
        parser.error("--proxy-tps must be positive")
    summary = summarize(args)
    payload = {"params": vars(args), "summary": asdict(summary)}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(summary)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    if args.require_pass and not summary.meets_all_gates:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
