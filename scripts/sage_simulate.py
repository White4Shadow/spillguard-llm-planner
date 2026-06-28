#!/usr/bin/env python3
"""
Throughput simulator for the SAGE-100 architecture.

SAGE assumes a fast resident proxy handles every token and a giant oracle is
called only for uncertain tokens. When the oracle is called, it should touch only
a sparse active subset of the 100B+ model. This script estimates the resulting
tokens/sec from byte movement and fixed overhead assumptions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


BYTES_PER_GIB = 1024**3


@dataclass
class SimulationRow:
    target_tps: float
    proxy_tps: float
    params_b: float
    quant_bpw: float
    dense_weight_gib: float
    pcie_gbps: float
    oracle_call_rate: float
    active_percent: float
    oracle_active_gib: float
    proxy_ms_per_token: float
    oracle_transfer_ms: float
    oracle_compute_ms: float
    oracle_fixed_ms: float
    expected_ms_per_token: float
    effective_tps: float
    meets_target: bool


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def simulate_row(
    *,
    target_tps: float,
    proxy_tps: float,
    params_b: float,
    quant_bpw: float,
    pcie_gbps: float,
    oracle_call_rate: float,
    active_percent: float,
    oracle_compute_ms: float,
    oracle_fixed_ms: float,
) -> SimulationRow:
    if proxy_tps <= 0:
        raise ValueError("proxy_tps must be positive")
    if pcie_gbps <= 0:
        raise ValueError("pcie_gbps must be positive")
    if not 0 <= oracle_call_rate <= 1:
        raise ValueError("oracle_call_rate must be in [0, 1]")
    if not 0 <= active_percent <= 100:
        raise ValueError("active_percent must be in [0, 100]")

    dense_gib = dense_weight_gib(params_b, quant_bpw)
    active_gib = dense_gib * active_percent / 100.0
    transfer_ms = active_gib / (pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)) * 1000.0
    proxy_ms = 1000.0 / proxy_tps
    oracle_ms = transfer_ms + oracle_compute_ms + oracle_fixed_ms
    expected_ms = proxy_ms + oracle_call_rate * oracle_ms
    effective_tps = 1000.0 / expected_ms

    return SimulationRow(
        target_tps=target_tps,
        proxy_tps=proxy_tps,
        params_b=params_b,
        quant_bpw=quant_bpw,
        dense_weight_gib=dense_gib,
        pcie_gbps=pcie_gbps,
        oracle_call_rate=oracle_call_rate,
        active_percent=active_percent,
        oracle_active_gib=active_gib,
        proxy_ms_per_token=proxy_ms,
        oracle_transfer_ms=transfer_ms,
        oracle_compute_ms=oracle_compute_ms,
        oracle_fixed_ms=oracle_fixed_ms,
        expected_ms_per_token=expected_ms,
        effective_tps=effective_tps,
        meets_target=effective_tps >= target_tps,
    )


def make_rows(args: argparse.Namespace) -> list[SimulationRow]:
    rows: list[SimulationRow] = []
    for call_rate in args.oracle_call_rates:
        for active_percent in args.active_percents:
            rows.append(
                simulate_row(
                    target_tps=args.target_tps,
                    proxy_tps=args.proxy_tps,
                    params_b=args.params_b,
                    quant_bpw=args.quant_bpw,
                    pcie_gbps=args.pcie_gbps,
                    oracle_call_rate=call_rate,
                    active_percent=active_percent,
                    oracle_compute_ms=args.oracle_compute_ms,
                    oracle_fixed_ms=args.oracle_fixed_ms,
                )
            )
    return rows


def print_table(rows: list[SimulationRow]) -> None:
    print("| Oracle calls | Active giant | Active GiB | Oracle transfer | Expected tok/s | Meets target |")
    print("| ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            f"| {row.oracle_call_rate:.0%} "
            f"| {row.active_percent:.1f}% "
            f"| {row.oracle_active_gib:.2f} "
            f"| {row.oracle_transfer_ms:.1f} ms "
            f"| {row.effective_tps:.2f} "
            f"| {row.meets_target} |"
        )


def write_csv(rows: list[SimulationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate SAGE-100 throughput from oracle sparsity and call rates.")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--proxy-tps", type=float, default=25.0, help="resident proxy generation speed")
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0, help="sustained PCIe bandwidth in decimal GB/s")
    parser.add_argument(
        "--oracle-call-rates",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.25,0.5"),
        help="comma-separated fraction of tokens that call oracle",
    )
    parser.add_argument(
        "--active-percents",
        type=parse_float_list,
        default=parse_float_list("1,2,5,10,15,20"),
        help="comma-separated active percent of giant model touched per oracle call",
    )
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = make_rows(args)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print_table(rows)

    if args.csv_out:
        write_csv(rows, Path(args.csv_out))
        print(f"\nwrote: {Path(args.csv_out).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
