#!/usr/bin/env python3
"""
Evaluate SAGE proxy/oracle agreement results against a throughput target.

The agreement harness tells us how often a proxy and oracle produce the same
greedy output. This script converts that into a harder systems question:

If a perfect router could know exactly which proxy outputs are safe to skip,
would that skip rate be enough to reach the target tokens/sec?

If the answer is no, the architecture needs a better proxy, a sparse oracle,
larger accepted draft batches, or a different model family before CUDA work is
justified.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BYTES_PER_GIB = 1024**3


@dataclass
class AgreementStats:
    files: list[str]
    rows: int
    exact_matches: int
    normalized_matches: int
    exact_match_rate: float
    normalized_match_rate: float
    proxy_output_tokens: int
    common_prefix_tokens: int
    common_prefix_token_rate: float
    proxy_total_sec: float
    oracle_total_sec: float


@dataclass
class RouterTarget:
    target_tps: float
    proxy_tps: float
    params_b: float
    quant_bpw: float
    dense_weight_gib: float
    active_percent: float
    oracle_active_gib: float
    pcie_gbps: float
    oracle_transfer_ms: float
    oracle_compute_ms: float
    oracle_fixed_ms: float
    oracle_ms_per_call: float
    max_oracle_call_rate_for_target: float
    required_safe_skip_rate: float
    observed_safe_skip_upper_bound: float
    observed_required_oracle_call_rate: float
    observed_upper_bound_tps: float
    target_reachable_by_perfect_router: bool
    trust_proxy_all_error_rate: float


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def output_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def common_prefix_len(left: list[str], right: list[str]) -> int:
    total = 0
    for a, b in zip(left, right):
        if a != b:
            break
        total += 1
    return total


def expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(value))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str], float, float]:
    rows: list[dict[str, Any]] = []
    files: list[str] = []
    proxy_total = 0.0
    oracle_total = 0.0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_rows = payload.get("rows", [])
        if not isinstance(file_rows, list):
            raise ValueError(f"{path} does not contain a rows list")
        rows.extend(file_rows)
        files.append(str(path))
        summary = payload.get("summary", {})
        proxy_total += float(summary.get("proxy_total_sec", 0.0))
        oracle_total += float(summary.get("oracle_total_sec", 0.0))
    return rows, files, proxy_total, oracle_total


def agreement_stats(paths: list[Path]) -> AgreementStats:
    rows, files, proxy_total, oracle_total = load_rows(paths)
    exact = 0
    normalized = 0
    proxy_token_count = 0
    prefix_count = 0

    for row in rows:
        proxy_output = str(row.get("proxy", {}).get("output", ""))
        oracle_output = str(row.get("oracle", {}).get("output", ""))
        exact_match = proxy_output == oracle_output
        normalized_match = normalize(proxy_output) == normalize(oracle_output)
        exact += int(exact_match)
        normalized += int(normalized_match)

        proxy_tokens = output_tokens(proxy_output)
        oracle_tokens = output_tokens(oracle_output)
        proxy_token_count += len(proxy_tokens)
        prefix_count += common_prefix_len(proxy_tokens, oracle_tokens)

    total = len(rows)
    return AgreementStats(
        files=files,
        rows=total,
        exact_matches=exact,
        normalized_matches=normalized,
        exact_match_rate=exact / total if total else 0.0,
        normalized_match_rate=normalized / total if total else 0.0,
        proxy_output_tokens=proxy_token_count,
        common_prefix_tokens=prefix_count,
        common_prefix_token_rate=prefix_count / proxy_token_count if proxy_token_count else 0.0,
        proxy_total_sec=proxy_total,
        oracle_total_sec=oracle_total,
    )


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def router_target(
    *,
    stats: AgreementStats,
    target_tps: float,
    proxy_tps: float,
    params_b: float,
    quant_bpw: float,
    active_percent: float,
    pcie_gbps: float,
    oracle_compute_ms: float,
    oracle_fixed_ms: float,
    match_field: str,
) -> RouterTarget:
    if target_tps <= 0:
        raise ValueError("target_tps must be positive")
    if proxy_tps <= 0:
        raise ValueError("proxy_tps must be positive")
    if params_b <= 0:
        raise ValueError("params_b must be positive")
    if quant_bpw <= 0:
        raise ValueError("quant_bpw must be positive")
    if not 0 <= active_percent <= 100:
        raise ValueError("active_percent must be in [0, 100]")
    if pcie_gbps <= 0:
        raise ValueError("pcie_gbps must be positive")

    match_rate = stats.normalized_match_rate if match_field == "normalized" else stats.exact_match_rate
    dense_gib = dense_weight_gib(params_b, quant_bpw)
    active_gib = dense_gib * active_percent / 100.0
    transfer_ms = active_gib / (pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)) * 1000.0
    oracle_ms = transfer_ms + oracle_compute_ms + oracle_fixed_ms
    proxy_ms = 1000.0 / proxy_tps
    target_ms = 1000.0 / target_tps

    if oracle_ms <= 0:
        max_call_rate = 1.0
    else:
        max_call_rate = (target_ms - proxy_ms) / oracle_ms
    max_call_rate = max(0.0, min(1.0, max_call_rate))
    required_skip = 1.0 - max_call_rate

    # This is an optimistic upper bound: a real router cannot know matches in
    # advance unless it has predictive features strong enough to separate them.
    observed_safe_skip = match_rate
    observed_call_rate = 1.0 - observed_safe_skip
    observed_ms = proxy_ms + observed_call_rate * oracle_ms
    observed_tps = 1000.0 / observed_ms if observed_ms > 0 else 0.0

    return RouterTarget(
        target_tps=target_tps,
        proxy_tps=proxy_tps,
        params_b=params_b,
        quant_bpw=quant_bpw,
        dense_weight_gib=dense_gib,
        active_percent=active_percent,
        oracle_active_gib=active_gib,
        pcie_gbps=pcie_gbps,
        oracle_transfer_ms=transfer_ms,
        oracle_compute_ms=oracle_compute_ms,
        oracle_fixed_ms=oracle_fixed_ms,
        oracle_ms_per_call=oracle_ms,
        max_oracle_call_rate_for_target=max_call_rate,
        required_safe_skip_rate=required_skip,
        observed_safe_skip_upper_bound=observed_safe_skip,
        observed_required_oracle_call_rate=observed_call_rate,
        observed_upper_bound_tps=observed_tps,
        target_reachable_by_perfect_router=observed_safe_skip >= required_skip,
        trust_proxy_all_error_rate=1.0 - match_rate,
    )


def print_markdown(stats: AgreementStats, target: RouterTarget, match_field: str) -> None:
    print("# SAGE Router Feasibility")
    print()
    print(f"- Agreement files: `{len(stats.files)}`")
    print(f"- Rows: `{stats.rows}`")
    print(f"- Match field: `{match_field}`")
    print()
    print("## Agreement Evidence")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Exact matches | {stats.exact_matches}/{stats.rows} ({stats.exact_match_rate:.1%}) |")
    print(f"| Normalized matches | {stats.normalized_matches}/{stats.rows} ({stats.normalized_match_rate:.1%}) |")
    print(f"| Common prefix tokens | {stats.common_prefix_tokens}/{stats.proxy_output_tokens} ({stats.common_prefix_token_rate:.1%}) |")
    print(f"| Proxy total wall time | {stats.proxy_total_sec:.1f} s |")
    print(f"| Oracle total wall time | {stats.oracle_total_sec:.1f} s |")
    print()
    print("## Throughput Gate")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Target speed | {target.target_tps:.2f} tok/s |")
    print(f"| Proxy speed assumption | {target.proxy_tps:.2f} tok/s |")
    print(f"| Dense giant size | {target.dense_weight_gib:.2f} GiB |")
    print(f"| Active oracle per call | {target.oracle_active_gib:.2f} GiB ({target.active_percent:.1f}%) |")
    print(f"| Oracle transfer per call | {target.oracle_transfer_ms:.1f} ms |")
    print(f"| Oracle total per call | {target.oracle_ms_per_call:.1f} ms |")
    print(f"| Max oracle call rate for target | {target.max_oracle_call_rate_for_target:.1%} |")
    print(f"| Required safe skip rate | {target.required_safe_skip_rate:.1%} |")
    print(f"| Observed perfect-router skip upper bound | {target.observed_safe_skip_upper_bound:.1%} |")
    print(f"| Observed perfect-router speed upper bound | {target.observed_upper_bound_tps:.2f} tok/s |")
    print(f"| Target reachable by perfect router | {target.target_reachable_by_perfect_router} |")
    print(f"| Error if proxy is trusted for all tokens | {target.trust_proxy_all_error_rate:.1%} |")
    print()
    if target.target_reachable_by_perfect_router:
        print("Result: this agreement set is not ruled out by the byte budget, but a real router still has to predict the matching cases before calling the oracle.")
    else:
        print("Result: this agreement set is ruled out for the target under these byte assumptions. Improve agreement, reduce active oracle bytes, increase proxy speed, or use a different sparse verification path.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAGE agreement JSON against a throughput/router target.")
    parser.add_argument("--agreement-json", nargs="+", required=True, help="one or more agreement JSON files or glob patterns")
    parser.add_argument("--match-field", choices=["exact", "normalized"], default="normalized")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--proxy-tps", type=float, default=25.0)
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--active-percent", type=float, default=10.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        paths = expand_inputs(args.agreement_json)
        stats = agreement_stats(paths)
        target = router_target(
            stats=stats,
            target_tps=args.target_tps,
            proxy_tps=args.proxy_tps,
            params_b=args.params_b,
            quant_bpw=args.quant_bpw,
            active_percent=args.active_percent,
            pcie_gbps=args.pcie_gbps,
            oracle_compute_ms=args.oracle_compute_ms,
            oracle_fixed_ms=args.oracle_fixed_ms,
            match_field=args.match_field,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps({"agreement": asdict(stats), "router_target": asdict(target)}, indent=2))
    else:
        print_markdown(stats, target, args.match_field)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
