#!/usr/bin/env python3
"""
Active-byte budget calculator for the SAGE-100 research track.

This does not benchmark a model. It converts a target tokens/sec objective into
the maximum amount of giant-model data that can be moved per token or per oracle
call before PCIe transfer alone makes the target impossible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


BYTES_PER_GIB = 1024**3


@dataclass
class SageBudget:
    target_tps: float
    ms_per_token: float
    pcie_gbps: float
    transfer_fraction: float
    transfer_budget_ms_per_token: float
    transfer_budget_gib_per_token: float
    params_b: float
    quant_bpw: float
    dense_weight_gib: float
    oracle_call_rate: float
    transfer_budget_gib_per_oracle_call: float
    max_active_model_percent_per_oracle_call: float
    dense_streaming_tps_upper_bound: float
    dense_streaming_meets_target: bool
    gpu_vram_gib: float
    runtime_overhead_gib: float
    kv_cache_gib: float
    dense_weights_fit_in_vram: bool


def compute_budget(
    *,
    target_tps: float,
    pcie_gbps: float,
    transfer_fraction: float,
    params_b: float,
    quant_bpw: float,
    oracle_call_rate: float,
    gpu_vram_gib: float,
    runtime_overhead_gib: float,
    kv_cache_gib: float,
) -> SageBudget:
    if target_tps <= 0:
        raise ValueError("target_tps must be positive")
    if pcie_gbps <= 0:
        raise ValueError("pcie_gbps must be positive")
    if not 0 < transfer_fraction <= 1:
        raise ValueError("transfer_fraction must be in (0, 1]")
    if params_b <= 0:
        raise ValueError("params_b must be positive")
    if quant_bpw <= 0:
        raise ValueError("quant_bpw must be positive")
    if not 0 < oracle_call_rate <= 1:
        raise ValueError("oracle_call_rate must be in (0, 1]")

    ms_per_token = 1000.0 / target_tps
    transfer_budget_ms = ms_per_token * transfer_fraction
    transfer_budget_gib = pcie_gbps * (transfer_budget_ms / 1000.0) * (1_000_000_000 / BYTES_PER_GIB)
    dense_weight_gib = params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB
    per_oracle_call_gib = transfer_budget_gib / oracle_call_rate
    active_percent = 100.0 * per_oracle_call_gib / dense_weight_gib
    dense_tps_upper = pcie_gbps * (1_000_000_000 / BYTES_PER_GIB) / dense_weight_gib
    dense_fit = dense_weight_gib + runtime_overhead_gib + kv_cache_gib <= gpu_vram_gib

    return SageBudget(
        target_tps=target_tps,
        ms_per_token=ms_per_token,
        pcie_gbps=pcie_gbps,
        transfer_fraction=transfer_fraction,
        transfer_budget_ms_per_token=transfer_budget_ms,
        transfer_budget_gib_per_token=transfer_budget_gib,
        params_b=params_b,
        quant_bpw=quant_bpw,
        dense_weight_gib=dense_weight_gib,
        oracle_call_rate=oracle_call_rate,
        transfer_budget_gib_per_oracle_call=per_oracle_call_gib,
        max_active_model_percent_per_oracle_call=active_percent,
        dense_streaming_tps_upper_bound=dense_tps_upper,
        dense_streaming_meets_target=dense_tps_upper >= target_tps,
        gpu_vram_gib=gpu_vram_gib,
        runtime_overhead_gib=runtime_overhead_gib,
        kv_cache_gib=kv_cache_gib,
        dense_weights_fit_in_vram=dense_fit,
    )


def print_markdown(budget: SageBudget) -> None:
    print("# SAGE Active-Byte Budget")
    print()
    print(f"- Target speed: `{budget.target_tps:.2f} tok/s` (`{budget.ms_per_token:.2f} ms/token`)")
    print(f"- PCIe bandwidth assumption: `{budget.pcie_gbps:.2f} GB/s`")
    print(f"- Transfer time budget: `{budget.transfer_fraction:.0%}` of token time")
    print(f"- Model size: `{budget.params_b:.1f}B` params at `{budget.quant_bpw:.2f} bpw`")
    print(f"- Oracle call rate: `{budget.oracle_call_rate:.0%}` of tokens")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Dense weight size | {budget.dense_weight_gib:.2f} GiB |")
    print(f"| Transfer budget per token | {budget.transfer_budget_gib_per_token:.2f} GiB |")
    print(f"| Transfer budget per oracle call | {budget.transfer_budget_gib_per_oracle_call:.2f} GiB |")
    print(f"| Max active model per oracle call | {budget.max_active_model_percent_per_oracle_call:.2f}% |")
    print(f"| Dense PCIe streaming upper bound | {budget.dense_streaming_tps_upper_bound:.2f} tok/s |")
    print(f"| Dense weights plus KV/overhead fit in VRAM | {budget.dense_weights_fit_in_vram} |")
    print()
    if budget.dense_streaming_meets_target and budget.dense_weights_fit_in_vram:
        print("Dense execution is not ruled out by this byte budget, but compute still needs benchmarking.")
    else:
        print("Dense per-token streaming is ruled out for the target. The giant model must be sparse, cached, or called only occasionally.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute SAGE-100 active-byte feasibility budgets.")
    parser.add_argument("--target-tps", type=float, default=7.0, help="target generated tokens/sec")
    parser.add_argument("--pcie-gbps", type=float, default=24.0, help="sustained PCIe bandwidth assumption in decimal GB/s")
    parser.add_argument("--transfer-fraction", type=float, default=0.30, help="fraction of token time allowed for transfers")
    parser.add_argument("--params-b", type=float, default=100.0, help="model parameter count in billions")
    parser.add_argument("--quant-bpw", type=float, default=2.0, help="effective quantized bits per weight")
    parser.add_argument("--oracle-call-rate", type=float, default=0.25, help="fraction of tokens that invoke the giant oracle")
    parser.add_argument("--gpu-vram-gib", type=float, default=12.0, help="available GPU VRAM in GiB")
    parser.add_argument("--runtime-overhead-gib", type=float, default=2.0, help="CUDA/runtime/activation overhead in GiB")
    parser.add_argument("--kv-cache-gib", type=float, default=1.0, help="KV cache budget in GiB")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args()

    try:
        budget = compute_budget(
            target_tps=args.target_tps,
            pcie_gbps=args.pcie_gbps,
            transfer_fraction=args.transfer_fraction,
            params_b=args.params_b,
            quant_bpw=args.quant_bpw,
            oracle_call_rate=args.oracle_call_rate,
            gpu_vram_gib=args.gpu_vram_gib,
            runtime_overhead_gib=args.runtime_overhead_gib,
            kv_cache_gib=args.kv_cache_gib,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(asdict(budget), indent=2, sort_keys=True))
    else:
        print_markdown(budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
