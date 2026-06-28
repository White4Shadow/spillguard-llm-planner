#!/usr/bin/env python3
"""
Compute the measured overlap/prefetch budget needed for SAGE to reach 10 tok/s.

This is an optimization target, not a runtime implementation. It uses measured
projection artifacts to quantify exactly what must be hidden by async prefetch,
CUDA overlap, proxy speedup, or lower fallback rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail(f"expected JSON object: {path}")
    return payload


def float_value(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(item.get(key, default))
    except (TypeError, ValueError):
        return default


def bool_value(item: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(item.get(key, default))


def scenario_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    for item in scenarios:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    projection_path = Path(args.projection_json)
    projection = load_json(projection_path)
    sparse_model = projection.get("sparse_fallback_model", {})
    sparse_model = sparse_model if isinstance(sparse_model, dict) else {}
    sparse_summary = sparse_model.get("summary", {})
    sparse_summary = sparse_summary if isinstance(sparse_summary, dict) else {}

    full = scenario_by_name(projection, args.full_scenario)
    gpu = scenario_by_name(projection, args.gpu_scenario)
    if not full:
        fail(f"full scenario not found: {args.full_scenario}")
    if not gpu:
        fail(f"GPU-only scenario not found: {args.gpu_scenario}")

    target_ms = 1000.0 / args.target_tps
    proxy_ms = float_value(full, "proxy_ms_per_token")
    verifier_ms = float_value(full, "verifier_call_ms")
    fallback_rate = float_value(full, "oracle_call_rate")
    full_fallback_ms = float_value(full, "oracle_call_ms")
    gpu_fallback_ms = float_value(gpu, "oracle_call_ms")
    full_expected_ms = float_value(full, "expected_ms_per_token")
    gpu_expected_ms = float_value(gpu, "expected_ms_per_token")
    max_fallback_ms = (target_ms - proxy_ms - verifier_ms) / fallback_rate if fallback_rate > 0 else 0.0
    required_full_hidden_ms = max(0.0, full_fallback_ms - max_fallback_ms)
    required_gpu_hidden_ms = max(0.0, gpu_fallback_ms - max_fallback_ms)
    proxy_ms_for_gpu_target = target_ms - verifier_ms - fallback_rate * gpu_fallback_ms
    required_proxy_reduction_ms = max(0.0, proxy_ms - proxy_ms_for_gpu_target)
    max_fallback_rate_gpu = (
        (target_ms - proxy_ms - verifier_ms) / gpu_fallback_ms
        if gpu_fallback_ms > 0 and target_ms > proxy_ms + verifier_ms
        else 0.0
    )
    required_fallback_rate_reduction_pp = max(0.0, (fallback_rate - max_fallback_rate_gpu) * 100.0)

    host_read_ms = float_value(sparse_summary, "host_read_ms")
    pcie_ms = float_value(sparse_summary, "pcie_transfer_ms")
    kernel_ms = float_value(sparse_summary, "cuda_kernel_ms")
    hidden_host_read_tps = float_value(gpu, "effective_tps")
    full_tps = float_value(full, "effective_tps")
    overlap_target_passed = (
        full_tps >= args.floor_tps
        and full_tps < args.target_tps
        and hidden_host_read_tps < args.target_tps
        and required_gpu_hidden_ms > 0
        and required_gpu_hidden_ms <= pcie_ms + kernel_ms
        and required_proxy_reduction_ms <= args.max_proxy_reduction_ms
        and bool_value(sparse_summary, "component_replay_complete")
        and not bool_value(sparse_summary, "transformer_integrated", True)
    )

    return {
        "schema": "sage-overlap-budget-v0",
        "status": "measured_overlap_target_not_runtime_integrated",
        "inputs": {
            "projection_json": str(projection_path.resolve()),
            "full_scenario": args.full_scenario,
            "gpu_scenario": args.gpu_scenario,
        },
        "summary": {
            "target_tps": args.target_tps,
            "floor_tps": args.floor_tps,
            "overlap_target_passed": overlap_target_passed,
            "runtime_integrated": False,
            "transformer_integrated": False,
            "full_replay_tps": full_tps,
            "gpu_only_tps": hidden_host_read_tps,
            "full_expected_ms_per_token": full_expected_ms,
            "gpu_only_expected_ms_per_token": gpu_expected_ms,
            "proxy_ms_per_token": proxy_ms,
            "verifier_ms_per_token": verifier_ms,
            "fallback_rate": fallback_rate,
            "full_fallback_ms": full_fallback_ms,
            "gpu_fallback_ms": gpu_fallback_ms,
            "host_read_ms": host_read_ms,
            "pcie_transfer_ms": pcie_ms,
            "cuda_kernel_ms": kernel_ms,
            "max_fallback_ms_for_target": max_fallback_ms,
            "required_full_hidden_ms_per_fallback": required_full_hidden_ms,
            "required_gpu_hidden_ms_per_fallback": required_gpu_hidden_ms,
            "required_gpu_hidden_ms_per_token": required_gpu_hidden_ms * fallback_rate,
            "proxy_ms_for_gpu_target": proxy_ms_for_gpu_target,
            "required_proxy_reduction_ms_per_token": required_proxy_reduction_ms,
            "max_fallback_rate_with_gpu_only": max_fallback_rate_gpu,
            "required_fallback_rate_reduction_percentage_points": required_fallback_rate_reduction_pp,
            "measured_pcie_plus_kernel_ms": pcie_ms + kernel_ms,
            "optimization_routes": [
                "hide host reads with background sparse-page prefetch",
                "hide at least the required GPU fallback time with CUDA stream overlap",
                "reduce proxy latency by the required per-token delta",
                "reduce shortlist fallback rate below the measured hard120 rate",
            ],
        },
    }


def print_markdown(payload: dict[str, Any]) -> None:
    item = payload["summary"]
    print("# SAGE Overlap Budget")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Full replay speed: `{item['full_replay_tps']:.3f}` tok/s")
    print(f"- Host-read hidden speed: `{item['gpu_only_tps']:.3f}` tok/s")
    print(f"- Target: `{item['target_tps']:.1f}` tok/s")
    print(f"- Max fallback for target: `{item['max_fallback_ms_for_target']:.3f}` ms")
    print(f"- Extra GPU fallback to hide: `{item['required_gpu_hidden_ms_per_fallback']:.3f}` ms")
    print(f"- Proxy latency reduction alternative: `{item['required_proxy_reduction_ms_per_token']:.3f}` ms/token")
    print(f"- Fallback-rate reduction alternative: `{item['required_fallback_rate_reduction_percentage_points']:.3f}` points")
    print(f"- Runtime integrated: `{item['runtime_integrated']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute SAGE overlap/prefetch budget from measured artifacts.")
    parser.add_argument(
        "--projection-json",
        default="benchmarks/sage-runtime-projection-format-scaffold-measured-sparse-fallback.json",
    )
    parser.add_argument(
        "--full-scenario",
        default="format_scaffold_hard120_measured_proxy_measured_sparse_replay",
    )
    parser.add_argument(
        "--gpu-scenario",
        default="format_scaffold_hard120_measured_proxy_measured_sparse_gpu_only",
    )
    parser.add_argument("--target-tps", type=float, default=10.0)
    parser.add_argument("--floor-tps", type=float, default=7.0)
    parser.add_argument("--max-proxy-reduction-ms", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.target_tps <= 0 or args.floor_tps <= 0:
        parser.error("--target-tps and --floor-tps must be positive")
    if args.max_proxy_reduction_ms < 0:
        parser.error("--max-proxy-reduction-ms must be non-negative")
    payload = build_payload(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_markdown(payload)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
