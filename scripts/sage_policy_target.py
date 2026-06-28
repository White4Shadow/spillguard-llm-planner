#!/usr/bin/env python3
"""
Evaluate a two-stage SAGE policy target.

Proxy-only routing was not safe enough on local Gemma 12B -> 31B data. The next
question is sharper:

If a proxy-only rule proposes "probably safe" tokens, how accurate and how cheap
must a sparse oracle verifier be to make the system both fast and useful?

This script models:

1. proxy runs for every token;
2. proxy-side rule selects candidate tokens;
3. sparse verifier runs only for candidates;
4. verifier catches some bad candidates and may false-reject some good ones;
5. full oracle path handles non-candidates and verifier rejects.

It does not prove sparse verification works. It gives the implementation target
that a real sparse FFN/attention verifier must hit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sage_router_fit import expand_inputs, load_records, oracle_ms_per_call, token_class


@dataclass
class CandidateStats:
    total: int
    candidates: int
    candidate_rate: float
    candidate_good: int
    candidate_bad: int
    rejected: int
    rejected_good: int
    rejected_bad: int
    candidate_precision: float
    candidate_error_rate: float


@dataclass
class PolicyRow:
    verifier_catch_bad_rate: float
    verifier_false_reject_good_rate: float
    verifier_call_rate: float
    oracle_call_rate: float
    proxy_ms_per_token: float
    verifier_ms_per_call: float
    oracle_ms_per_call: float
    expected_ms_per_token: float
    effective_tps: float
    accepted_proxy_rate: float
    accepted_proxy_error_rate: float
    total_error_rate: float
    meets_target_tps: bool
    meets_total_error: bool
    meets_accepted_error: bool
    meets_all: bool


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def candidate_filter(record: object, args: argparse.Namespace) -> bool:
    cls = token_class(getattr(record, "proxy_token"))
    if args.candidate_class != "any" and cls != args.candidate_class:
        return False
    if args.margin_threshold is not None and getattr(record, "proxy_margin") < args.margin_threshold:
        return False
    if args.entropy_threshold is not None and getattr(record, "proxy_entropy") > args.entropy_threshold:
        return False
    return True


def make_candidate_stats(records: list[object], args: argparse.Namespace, label: str) -> CandidateStats:
    if args.stats_total >= 0 or args.stats_candidates >= 0 or args.stats_candidate_good >= 0 or args.stats_candidate_bad >= 0:
        values = [args.stats_total, args.stats_candidates, args.stats_candidate_good, args.stats_candidate_bad]
        if any(value < 0 for value in values):
            raise ValueError("all --stats-* values must be provided together")
        if args.stats_candidates != args.stats_candidate_good + args.stats_candidate_bad:
            raise ValueError("--stats-candidates must equal --stats-candidate-good + --stats-candidate-bad")
        if args.stats_candidates > args.stats_total:
            raise ValueError("--stats-candidates must be <= --stats-total")
        rejected = args.stats_total - args.stats_candidates
        return CandidateStats(
            total=args.stats_total,
            candidates=args.stats_candidates,
            candidate_rate=args.stats_candidates / args.stats_total if args.stats_total else 0.0,
            candidate_good=args.stats_candidate_good,
            candidate_bad=args.stats_candidate_bad,
            rejected=rejected,
            rejected_good=0,
            rejected_bad=0,
            candidate_precision=args.stats_candidate_good / args.stats_candidates if args.stats_candidates else 1.0,
            candidate_error_rate=args.stats_candidate_bad / args.stats_candidates if args.stats_candidates else 0.0,
        )

    candidates = [record for record in records if candidate_filter(record, args)]
    rejected = [record for record in records if not candidate_filter(record, args)]
    candidate_good = sum(1 for record in candidates if getattr(record, f"top1_{label}_match"))
    candidate_bad = len(candidates) - candidate_good
    rejected_good = sum(1 for record in rejected if getattr(record, f"top1_{label}_match"))
    rejected_bad = len(rejected) - rejected_good
    total = len(records)
    return CandidateStats(
        total=total,
        candidates=len(candidates),
        candidate_rate=len(candidates) / total if total else 0.0,
        candidate_good=candidate_good,
        candidate_bad=candidate_bad,
        rejected=len(rejected),
        rejected_good=rejected_good,
        rejected_bad=rejected_bad,
        candidate_precision=candidate_good / len(candidates) if candidates else 1.0,
        candidate_error_rate=candidate_bad / len(candidates) if candidates else 0.0,
    )


def verifier_ms_per_call(args: argparse.Namespace) -> float:
    return oracle_ms_per_call(
        params_b=args.params_b,
        quant_bpw=args.quant_bpw,
        active_percent=args.verifier_active_percent,
        pcie_gbps=args.pcie_gbps,
        oracle_compute_ms=args.verifier_compute_ms,
        oracle_fixed_ms=args.verifier_fixed_ms,
    )


def make_rows(stats: CandidateStats, args: argparse.Namespace) -> list[PolicyRow]:
    proxy_ms = 1000.0 / args.proxy_tps
    verifier_ms = verifier_ms_per_call(args)
    oracle_ms = oracle_ms_per_call(
        params_b=args.params_b,
        quant_bpw=args.quant_bpw,
        active_percent=args.oracle_active_percent,
        pcie_gbps=args.pcie_gbps,
        oracle_compute_ms=args.oracle_compute_ms,
        oracle_fixed_ms=args.oracle_fixed_ms,
    )
    rows: list[PolicyRow] = []
    for catch_bad in args.catch_bad_rates:
        for false_reject in args.false_reject_good_rates:
            bad_caught = stats.candidate_bad * catch_bad
            bad_uncaught = stats.candidate_bad - bad_caught
            good_false_rejected = stats.candidate_good * false_reject
            verifier_calls = stats.candidates
            oracle_calls = stats.rejected + bad_caught + good_false_rejected
            accepted_proxy = max(0.0, stats.candidates - bad_caught - good_false_rejected)
            verifier_call_rate = verifier_calls / stats.total if stats.total else 0.0
            oracle_call_rate = oracle_calls / stats.total if stats.total else 0.0
            accepted_proxy_rate = accepted_proxy / stats.total if stats.total else 0.0
            accepted_error = bad_uncaught / accepted_proxy if accepted_proxy > 0 else 0.0
            total_error = bad_uncaught / stats.total if stats.total else 0.0
            expected_ms = proxy_ms + verifier_call_rate * verifier_ms + oracle_call_rate * oracle_ms
            effective_tps = 1000.0 / expected_ms if expected_ms > 0 else 0.0
            meets_target = effective_tps >= args.target_tps
            meets_total_error = total_error <= args.max_total_error_rate
            meets_accepted_error = accepted_error <= args.max_accepted_error_rate
            rows.append(
                PolicyRow(
                    verifier_catch_bad_rate=catch_bad,
                    verifier_false_reject_good_rate=false_reject,
                    verifier_call_rate=verifier_call_rate,
                    oracle_call_rate=oracle_call_rate,
                    proxy_ms_per_token=proxy_ms,
                    verifier_ms_per_call=verifier_ms,
                    oracle_ms_per_call=oracle_ms,
                    expected_ms_per_token=expected_ms,
                    effective_tps=effective_tps,
                    accepted_proxy_rate=accepted_proxy_rate,
                    accepted_proxy_error_rate=accepted_error,
                    total_error_rate=total_error,
                    meets_target_tps=meets_target,
                    meets_total_error=meets_total_error,
                    meets_accepted_error=meets_accepted_error,
                    meets_all=meets_target and meets_total_error and meets_accepted_error,
                )
            )
    rows.sort(
        key=lambda row: (
            row.meets_all,
            row.effective_tps,
            -row.accepted_proxy_error_rate,
            -row.total_error_rate,
        ),
        reverse=True,
    )
    return rows


def print_rows(title: str, rows: Iterable[PolicyRow], top: int) -> None:
    print(f"## {title}")
    print()
    print("| Catch bad | False reject good | Verifier calls | Oracle calls | Accepted proxy | Accepted error | Total error | Tok/s | OK |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in list(rows)[:top]:
        print(
            f"| {row.verifier_catch_bad_rate:.0%} | {row.verifier_false_reject_good_rate:.0%} "
            f"| {row.verifier_call_rate:.1%} | {row.oracle_call_rate:.1%} | {row.accepted_proxy_rate:.1%} "
            f"| {row.accepted_proxy_error_rate:.1%} | {row.total_error_rate:.1%} "
            f"| {row.effective_tps:.2f} | {row.meets_all} |"
        )


def print_report(stats: CandidateStats, rows: list[PolicyRow], args: argparse.Namespace) -> None:
    print("# SAGE Two-Stage Policy Target")
    print()
    print("## Candidate Rule")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    print(f"| Candidate class | {args.candidate_class} |")
    print(f"| Margin threshold | {'' if args.margin_threshold is None else args.margin_threshold} |")
    print(f"| Entropy threshold | {'' if args.entropy_threshold is None else args.entropy_threshold} |")
    print()
    print("## Candidate Evidence")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Records | {stats.total} |")
    print(f"| Candidates | {stats.candidates} ({stats.candidate_rate:.1%}) |")
    print(f"| Candidate good/bad | {stats.candidate_good}/{stats.candidate_bad} |")
    print(f"| Candidate precision | {stats.candidate_precision:.1%} |")
    print(f"| Candidate error rate | {stats.candidate_error_rate:.1%} |")
    print(f"| Rejected records | {stats.rejected} |")
    print()
    print("## Cost Assumptions")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Target | {args.target_tps:.2f} tok/s |")
    print(f"| Proxy speed | {args.proxy_tps:.2f} tok/s |")
    print(f"| Sparse verifier active percent | {args.verifier_active_percent:.2f}% |")
    print(f"| Oracle active percent | {args.oracle_active_percent:.2f}% |")
    print(f"| Max accepted-proxy error | {args.max_accepted_error_rate:.1%} |")
    print(f"| Max total error | {args.max_total_error_rate:.1%} |")
    print()
    passing = [row for row in rows if row.meets_all]
    print_rows("Passing Rows", passing, args.top)
    if not passing:
        print("No row met speed and error targets under these assumptions.")
        print()
    fastest = sorted(rows, key=lambda row: row.effective_tps, reverse=True)
    print_rows("Fastest Rows", fastest, args.top)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAGE proxy + sparse verifier policy targets.")
    parser.add_argument("--logprob-json", nargs="+", required=True)
    parser.add_argument("--ignore-prefix-steps", type=int, default=0)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--candidate-class", default="any", choices=["any", "control", "punct", "whitespace", "number", "capitalized", "word"])
    parser.add_argument("--margin-threshold", type=float, default=None)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--stats-total", type=int, default=-1, help="override total records for externally measured candidate stats")
    parser.add_argument("--stats-candidates", type=int, default=-1, help="override candidate count for externally measured candidate stats")
    parser.add_argument("--stats-candidate-good", type=int, default=-1, help="override good candidate count")
    parser.add_argument("--stats-candidate-bad", type=int, default=-1, help="override bad candidate count")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--proxy-tps", type=float, default=25.0)
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--verifier-active-percent", type=float, default=2.0)
    parser.add_argument("--verifier-compute-ms", type=float, default=5.0)
    parser.add_argument("--verifier-fixed-ms", type=float, default=2.0)
    parser.add_argument("--oracle-active-percent", type=float, default=10.0)
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--catch-bad-rates", type=parse_float_list, default=parse_float_list("0,0.5,0.8,0.9,0.95,1.0"))
    parser.add_argument("--false-reject-good-rates", type=parse_float_list, default=parse_float_list("0,0.05,0.1,0.2"))
    parser.add_argument("--max-accepted-error-rate", type=float, default=0.05)
    parser.add_argument("--max-total-error-rate", type=float, default=0.02)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.ignore_prefix_steps < 0:
        parser.error("--ignore-prefix-steps must be non-negative")
    for attr in ("target_tps", "proxy_tps", "params_b", "quant_bpw", "pcie_gbps"):
        if getattr(args, attr) <= 0:
            parser.error(f"--{attr.replace('_', '-')} must be positive")
    for attr in ("verifier_active_percent", "oracle_active_percent"):
        value = getattr(args, attr)
        if not 0 <= value <= 100:
            parser.error(f"--{attr.replace('_', '-')} must be in [0, 100]")
    for attr in ("max_accepted_error_rate", "max_total_error_rate"):
        value = getattr(args, attr)
        if not 0 <= value <= 1:
            parser.error(f"--{attr.replace('_', '-')} must be in [0, 1]")

    try:
        paths = expand_inputs(args.logprob_json)
        records = load_records(paths, args.ignore_prefix_steps, args.label)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not records:
        parser.error("no records after filtering")

    try:
        stats = make_candidate_stats(records, args, args.label)
    except ValueError as exc:
        parser.error(str(exc))
    rows = make_rows(stats, args)
    payload = {
        "params": {
            "files": [str(path) for path in paths],
            "ignore_prefix_steps": args.ignore_prefix_steps,
            "label": args.label,
            "candidate_class": args.candidate_class,
            "margin_threshold": args.margin_threshold,
            "entropy_threshold": args.entropy_threshold,
            "target_tps": args.target_tps,
            "proxy_tps": args.proxy_tps,
            "params_b": args.params_b,
            "quant_bpw": args.quant_bpw,
            "pcie_gbps": args.pcie_gbps,
            "verifier_active_percent": args.verifier_active_percent,
            "oracle_active_percent": args.oracle_active_percent,
            "max_accepted_error_rate": args.max_accepted_error_rate,
            "max_total_error_rate": args.max_total_error_rate,
        },
        "candidate_stats": asdict(stats),
        "rows": [asdict(row) for row in rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(stats, rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
