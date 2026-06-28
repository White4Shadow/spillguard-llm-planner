#!/usr/bin/env python3
"""
Compute SAGE runtime budget from the resident pinned page-cache replay smoke.

This is a target-setting artifact. It uses measured proxy/verifier/fallback
rates from the runtime projection and measured per-replay GPU time from the
resident page cache smoke to estimate how small the active oracle page set must
be for 10 tok/s. It does not claim transformer integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3


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


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("summary", payload)
    return raw if isinstance(raw, dict) else {}


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


def bytes_to_gib(n_bytes: float) -> float:
    return n_bytes / BYTES_PER_GIB


def gib_to_bytes(gib: float) -> int:
    return int(gib * BYTES_PER_GIB)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    projection_path = Path(args.projection_json)
    page_cache_path = Path(args.page_cache_json)
    page_ledger_path = Path(args.page_ledger_json)
    projection = load_json(projection_path)
    page_cache = load_json(page_cache_path)
    page_cache_summary = summary(page_cache)
    page_ledger = load_json(page_ledger_path)
    page_ledger_summary = summary(page_ledger)

    scenario = scenario_by_name(projection, args.scenario)
    if not scenario:
        fail(f"scenario not found: {args.scenario}")
    if page_cache.get("schema") != "sage-oracle-page-cuda-page-cache-smoke-v0":
        fail("expected sage-oracle-page-cuda-page-cache-smoke-v0 input")
    if page_cache.get("status") != "measured_resident_pinned_page_cache_replay_touch_not_transformer":
        fail("page-cache artifact has unexpected status")

    proxy_ms = float_value(scenario, "proxy_ms_per_token")
    verifier_rate = float_value(scenario, "verifier_call_rate", 1.0)
    verifier_call_ms = float_value(scenario, "verifier_call_ms")
    verifier_ms = verifier_rate * verifier_call_ms
    fallback_rate = float_value(scenario, "oracle_call_rate")
    measured_replay_ms = float_value(page_cache_summary, "per_replay_gpu_ms")
    measured_replay_gib = float_value(page_cache_summary, "staged_gib_per_replay")
    measured_replay_bytes = float_value(page_cache_summary, "staged_bytes_per_replay")
    ms_per_gib = measured_replay_ms / measured_replay_gib if measured_replay_gib > 0 else 0.0
    expected_ms = proxy_ms + verifier_ms + fallback_rate * measured_replay_ms
    effective_tps = 1000.0 / expected_ms if expected_ms > 0 else 0.0

    target_ms = 1000.0 / args.target_tps
    floor_ms = 1000.0 / args.floor_tps
    max_fallback_ms_for_target = (
        (target_ms - proxy_ms - verifier_ms) / fallback_rate
        if fallback_rate > 0
        else float("inf")
    )
    max_active_gib_for_target = (
        max_fallback_ms_for_target / ms_per_gib
        if ms_per_gib > 0 and max_fallback_ms_for_target > 0
        else 0.0
    )
    max_active_bytes_for_target = gib_to_bytes(max_active_gib_for_target)
    required_active_gib_reduction = max(0.0, measured_replay_gib - max_active_gib_for_target)
    required_active_byte_reduction = max(0, int(measured_replay_bytes - max_active_bytes_for_target))
    required_active_reduction_percent = (
        required_active_gib_reduction / measured_replay_gib * 100.0
        if measured_replay_gib > 0
        else 0.0
    )
    max_fallback_rate_for_target = (
        (target_ms - proxy_ms - verifier_ms) / measured_replay_ms
        if measured_replay_ms > 0 and target_ms > proxy_ms + verifier_ms
        else 0.0
    )
    required_fallback_rate_reduction_pp = max(0.0, (fallback_rate - max_fallback_rate_for_target) * 100.0)
    proxy_ms_for_target = target_ms - verifier_ms - fallback_rate * measured_replay_ms
    required_proxy_reduction_ms = max(0.0, proxy_ms - proxy_ms_for_target)
    slack_to_floor_ms = floor_ms - expected_ms

    ref_active_percent = float_value(page_ledger_summary, "active_percent_of_reference_100b")
    max_ref_active_percent_for_target = (
        ref_active_percent * max_active_gib_for_target / measured_replay_gib
        if measured_replay_gib > 0
        else 0.0
    )
    selected_pages = int(float_value(page_ledger_summary, "selected_pages"))
    max_pages_for_target = int(selected_pages * max_active_gib_for_target / measured_replay_gib) if measured_replay_gib > 0 else 0

    cache_evidence = page_cache.get("runtime_ledger_evidence", {})
    cache_evidence = cache_evidence if isinstance(cache_evidence, dict) else {}
    reduction_target_identified = (
        effective_tps >= args.floor_tps
        and effective_tps < args.target_tps
        and measured_replay_ms > 0
        and measured_replay_gib > 0
        and max_active_gib_for_target > 0
        and max_active_gib_for_target < measured_replay_gib
        and required_active_gib_reduction > 0
        and bool_value(page_cache_summary, "cache_replay_saves_host_read")
        and bool_value(page_cache_summary, "stage_byte_match")
        and bool_value(page_cache_summary, "byte_budget_respected")
        and page_cache_summary.get("sparse_transformer_status") == "not_implemented"
        and bool_value(cache_evidence, "resident_pinned_page_cache")
        and not bool_value(cache_evidence, "transformer_layer_math", True)
    )
    measured_plan_meets_target = (
        effective_tps >= args.target_tps
        and measured_replay_ms > 0
        and measured_replay_gib > 0
        and max_active_gib_for_target >= measured_replay_gib
        and required_active_gib_reduction <= 0
        and bool_value(page_cache_summary, "cache_replay_saves_host_read")
        and bool_value(page_cache_summary, "stage_byte_match")
        and bool_value(page_cache_summary, "byte_budget_respected")
        and page_cache_summary.get("sparse_transformer_status") == "not_implemented"
        and bool_value(cache_evidence, "resident_pinned_page_cache")
        and not bool_value(cache_evidence, "transformer_layer_math", True)
    )
    budget_passed = reduction_target_identified or measured_plan_meets_target

    return {
        "schema": "sage-page-cache-budget-v0",
        "status": "measured_page_cache_budget_target_not_transformer_integrated",
        "inputs": {
            "projection_json": str(projection_path.resolve()),
            "page_cache_json": str(page_cache_path.resolve()),
            "page_ledger_json": str(page_ledger_path.resolve()),
            "scenario": args.scenario,
        },
        "summary": {
            "target_tps": args.target_tps,
            "floor_tps": args.floor_tps,
            "budget_target_passed": budget_passed,
            "reduction_target_identified": reduction_target_identified,
            "measured_plan_meets_target_tps": measured_plan_meets_target,
            "runtime_integrated": False,
            "transformer_integrated": False,
            "effective_tps_with_page_cache": effective_tps,
            "expected_ms_per_token_with_page_cache": expected_ms,
            "slack_ms_to_floor_tps": slack_to_floor_ms,
            "proxy_ms_per_token": proxy_ms,
            "verifier_ms_per_token": verifier_ms,
            "fallback_rate": fallback_rate,
            "measured_page_cache_replay_ms": measured_replay_ms,
            "measured_page_cache_replay_gib": measured_replay_gib,
            "measured_page_cache_replay_bytes": int(measured_replay_bytes),
            "measured_page_cache_ms_per_gib": ms_per_gib,
            "max_fallback_ms_for_target": max_fallback_ms_for_target,
            "max_active_gib_for_target": max_active_gib_for_target,
            "max_active_bytes_for_target": max_active_bytes_for_target,
            "required_active_gib_reduction": required_active_gib_reduction,
            "required_active_byte_reduction": required_active_byte_reduction,
            "required_active_reduction_percent": required_active_reduction_percent,
            "current_active_percent_of_reference_100b": ref_active_percent,
            "max_active_percent_of_reference_100b_for_target": max_ref_active_percent_for_target,
            "selected_pages": selected_pages,
            "max_pages_for_target_at_same_page_mix": max_pages_for_target,
            "max_fallback_rate_for_target": max_fallback_rate_for_target,
            "required_fallback_rate_reduction_percentage_points": required_fallback_rate_reduction_pp,
            "proxy_ms_for_target": proxy_ms_for_target,
            "required_proxy_reduction_ms_per_token": required_proxy_reduction_ms,
            "optimization_routes": [
                "reduce selected active pages to the max active GiB target",
                "replace byte-touch replay with faster sparse dequant/matmul kernels",
                "reduce shortlist fallback rate below the measured hard120 rate",
                "reduce proxy latency by the required per-token delta",
            ],
        },
    }


def print_markdown(payload: dict[str, Any]) -> None:
    item = payload["summary"]
    print("# SAGE Page-Cache Budget")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Page-cache projection: `{item['effective_tps_with_page_cache']:.3f}` tok/s")
    print(f"- Target: `{item['target_tps']:.1f}` tok/s")
    print(f"- Measured replay: `{item['measured_page_cache_replay_ms']:.3f}` ms for `{item['measured_page_cache_replay_gib']:.3f}` GiB")
    print(f"- Max active bytes for target: `{item['max_active_gib_for_target']:.3f}` GiB")
    print(f"- Required active-byte reduction: `{item['required_active_reduction_percent']:.1f}%`")
    print(f"- Max fallback rate for target: `{item['max_fallback_rate_for_target'] * 100.0:.2f}%`")
    print(f"- Runtime integrated: `{item['runtime_integrated']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute SAGE budget target from measured resident page-cache replay.")
    parser.add_argument(
        "--projection-json",
        default="benchmarks/sage-runtime-projection-format-scaffold-measured-sparse-fallback.json",
    )
    parser.add_argument(
        "--page-cache-json",
        default="benchmarks/sage-oracle-page-cuda-page-cache-gemma31b-full-replay3.json",
    )
    parser.add_argument(
        "--page-ledger-json",
        default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json",
    )
    parser.add_argument(
        "--scenario",
        default="format_scaffold_hard120_measured_proxy_measured_sparse_replay",
    )
    parser.add_argument("--target-tps", type=float, default=10.0)
    parser.add_argument("--floor-tps", type=float, default=7.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.target_tps <= 0 or args.floor_tps <= 0:
        parser.error("--target-tps and --floor-tps must be positive")
    payload = build_payload(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
