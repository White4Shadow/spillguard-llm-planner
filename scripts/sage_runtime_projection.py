#!/usr/bin/env python3
"""
Project SAGE persistent-runtime speed from measured policy and live artifacts.

The live loop is intentionally conservative and currently pays server swaps and
subprocess launches. This script separates those costs from the active-byte
policy model so the next runtime target is measurable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BYTES_PER_GIB = 1024**3


@dataclass
class Scenario:
    name: str
    description: str
    proxy_ms_per_token: float
    verifier_call_rate: float
    verifier_call_ms: float
    oracle_call_rate: float
    oracle_call_ms: float
    fixed_extra_ms_per_token: float
    expected_ms_per_token: float
    effective_tps: float
    meets_target_tps: bool
    meets_upper_target_tps: bool


@dataclass
class ActiveLimit:
    name: str
    target_tps: float
    proxy_ms_per_token: float
    verifier_call_rate: float
    verifier_call_ms: float
    oracle_call_rate: float
    oracle_compute_ms: float
    oracle_fixed_ms: float
    max_oracle_call_ms: float
    max_oracle_transfer_ms: float
    max_oracle_active_percent: float
    feasible: bool


@dataclass
class FallbackLimit:
    name: str
    target_tps: float
    proxy_ms_per_token: float
    verifier_call_ms: float
    fallback_rate: float
    configured_fallback_ms: float
    max_fallback_ms: float
    meets_configured_fallback_ms: bool


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


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def transfer_ms_per_active_percent(params_b: float, quant_bpw: float, pcie_gbps: float) -> float:
    dense_gib = dense_weight_gib(params_b, quant_bpw)
    active_gib = dense_gib / 100.0
    return active_gib / (pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)) * 1000.0


def ms_from_tps(tps: float) -> float:
    if tps <= 0:
        fail("tps values must be positive")
    return 1000.0 / tps


def scenario(
    *,
    name: str,
    description: str,
    proxy_ms: float,
    verifier_rate: float,
    verifier_ms: float,
    oracle_rate: float,
    oracle_ms: float,
    fixed_extra_ms: float,
    target_tps: float,
    upper_target_tps: float,
) -> Scenario:
    expected_ms = proxy_ms + verifier_rate * verifier_ms + oracle_rate * oracle_ms + fixed_extra_ms
    effective_tps = 1000.0 / expected_ms if expected_ms > 0 else 0.0
    return Scenario(
        name=name,
        description=description,
        proxy_ms_per_token=proxy_ms,
        verifier_call_rate=verifier_rate,
        verifier_call_ms=verifier_ms,
        oracle_call_rate=oracle_rate,
        oracle_call_ms=oracle_ms,
        fixed_extra_ms_per_token=fixed_extra_ms,
        expected_ms_per_token=expected_ms,
        effective_tps=effective_tps,
        meets_target_tps=effective_tps >= target_tps,
        meets_upper_target_tps=effective_tps >= upper_target_tps,
    )


def active_limit(
    *,
    name: str,
    target_tps: float,
    proxy_ms: float,
    verifier_rate: float,
    verifier_ms: float,
    oracle_rate: float,
    oracle_compute_ms: float,
    oracle_fixed_ms: float,
    ms_per_active_percent: float,
) -> ActiveLimit:
    target_ms = ms_from_tps(target_tps)
    if oracle_rate <= 0:
        max_call = float("inf")
        max_transfer = float("inf")
        max_active = float("inf")
        feasible = True
    else:
        max_call = (target_ms - proxy_ms - verifier_rate * verifier_ms) / oracle_rate
        max_transfer = max_call - oracle_compute_ms - oracle_fixed_ms
        max_active = max_transfer / ms_per_active_percent if ms_per_active_percent > 0 else 0.0
        feasible = max_call > 0 and max_transfer >= 0 and max_active >= 0
    return ActiveLimit(
        name=name,
        target_tps=target_tps,
        proxy_ms_per_token=proxy_ms,
        verifier_call_rate=verifier_rate,
        verifier_call_ms=verifier_ms,
        oracle_call_rate=oracle_rate,
        oracle_compute_ms=oracle_compute_ms,
        oracle_fixed_ms=oracle_fixed_ms,
        max_oracle_call_ms=max_call,
        max_oracle_transfer_ms=max_transfer,
        max_oracle_active_percent=max_active,
        feasible=feasible,
    )


def float_from(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def measured_candidate_verifier_ms(
    verifier_summary: dict[str, Any],
    candidate_rows_per_step: float,
    *,
    include_host_read: bool,
) -> float:
    measured_rows = max(1.0, float_from(verifier_summary, "candidate_tokens", 1.0))
    measured_ms = float_from(verifier_summary, "h2d_ms", 0.0) + float_from(verifier_summary, "kernel_ms", 0.0)
    if include_host_read:
        measured_ms += float_from(verifier_summary, "host_read_ms", 0.0)
    return measured_ms * candidate_rows_per_step / measured_rows


def shortlist_label(path_text: str) -> str:
    stem = Path(path_text).stem.lower()
    if "validation80e" in stem:
        return "validation80e"
    if "hard120" in stem:
        return "hard120"
    return stem.replace("sage-proxy-shortlist-", "").replace("-", "_")


def fallback_limit(
    *,
    name: str,
    target_tps: float,
    proxy_ms: float,
    verifier_ms: float,
    fallback_rate: float,
    configured_fallback_ms: float,
) -> FallbackLimit:
    target_ms = ms_from_tps(target_tps)
    if fallback_rate <= 0:
        max_fallback_ms = float("inf")
        meets_configured = True
    else:
        max_fallback_ms = (target_ms - proxy_ms - verifier_ms) / fallback_rate
        meets_configured = configured_fallback_ms <= max_fallback_ms
    return FallbackLimit(
        name=name,
        target_tps=target_tps,
        proxy_ms_per_token=proxy_ms,
        verifier_call_ms=verifier_ms,
        fallback_rate=fallback_rate,
        configured_fallback_ms=configured_fallback_ms,
        max_fallback_ms=max_fallback_ms,
        meets_configured_fallback_ms=meets_configured,
    )


def make_projection(args: argparse.Namespace) -> dict[str, Any]:
    policy_payload = load_json(Path(args.policy_json))
    policy = policy_payload.get("summary", {})
    policy_params = policy_payload.get("params", {})
    if not isinstance(policy, dict):
        fail("policy JSON has no summary object")
    if not isinstance(policy_params, dict):
        policy_params = {}

    live_summary: dict[str, Any] = {}
    if args.live_report_json:
        live_payload = load_json(Path(args.live_report_json))
        raw = live_payload.get("summary", {})
        live_summary = dict(raw) if isinstance(raw, dict) else {}

    candidate_verifier_summary: dict[str, Any] = {}
    if args.candidate_verifier_json:
        raw = load_json(Path(args.candidate_verifier_json)).get("summary", {})
        candidate_verifier_summary = dict(raw) if isinstance(raw, dict) else {}

    sparse_runtime_step_summary: dict[str, Any] = {}
    if args.sparse_runtime_step_json:
        raw = load_json(Path(args.sparse_runtime_step_json)).get("summary", {})
        sparse_runtime_step_summary = dict(raw) if isinstance(raw, dict) else {}

    shortlist_payloads: list[dict[str, Any]] = []
    for path_text in args.format_shortlist_json:
        payload = load_json(Path(path_text))
        raw = payload.get("summary", {})
        if isinstance(raw, dict):
            shortlist_payloads.append({"path": path_text, "summary": dict(raw)})

    stats_total = max(1.0, float_from(policy, "stats_total", 1.0))
    verifier_rate = float_from(policy, "verifier_call_rate", float_from(policy, "verifier_needed", 0.0) / stats_total)
    oracle_rate = float_from(policy, "oracle_call_rate", (stats_total - float_from(policy, "final_accepted", 0.0)) / stats_total)
    policy_proxy_ms = float_from(policy, "proxy_ms_per_token", ms_from_tps(args.proxy_tps))
    policy_verifier_ms = float_from(policy, "verifier_call_ms", args.verifier_call_ms)
    policy_oracle_ms = float_from(policy, "oracle_call_ms", args.oracle_call_ms)
    policy_expected_ms = float_from(policy, "expected_ms_per_token", policy_proxy_ms + verifier_rate * policy_verifier_ms + oracle_rate * policy_oracle_ms)

    live_steps = max(1.0, float_from(live_summary, "steps", 1.0))
    live_proxy_ms = 1000.0 * float_from(live_summary, "proxy_sec", 0.0) / live_steps
    live_accounted_ms = 1000.0 * float_from(live_summary, "accounted_sec", 0.0) / live_steps
    live_elapsed_ms = 1000.0 * float_from(live_summary, "elapsed_sec", 0.0) / live_steps
    live_overhead_ms = max(0.0, live_elapsed_ms - live_accounted_ms)

    proxy_ms = ms_from_tps(args.proxy_tps)
    if args.use_live_proxy and live_proxy_ms > 0:
        proxy_ms = live_proxy_ms

    scenarios = [
        scenario(
            name="live_python_observed",
            description="Measured Python live loop with server swaps and subprocesses.",
            proxy_ms=live_proxy_ms,
            verifier_rate=0.0,
            verifier_ms=0.0,
            oracle_rate=0.0,
            oracle_ms=0.0,
            fixed_extra_ms=live_elapsed_ms - live_proxy_ms if live_elapsed_ms > live_proxy_ms else live_elapsed_ms,
            target_tps=args.target_tps,
            upper_target_tps=args.upper_target_tps,
        ),
        scenario(
            name="live_no_outer_overhead_same_requests",
            description="Live request timings with outer orchestration overhead removed only.",
            proxy_ms=live_proxy_ms,
            verifier_rate=0.0,
            verifier_ms=0.0,
            oracle_rate=0.0,
            oracle_ms=0.0,
            fixed_extra_ms=live_accounted_ms - live_proxy_ms if live_accounted_ms > live_proxy_ms else live_accounted_ms,
            target_tps=args.target_tps,
            upper_target_tps=args.upper_target_tps,
        ),
        scenario(
            name="persistent_active_byte_policy",
            description="Policy report active-byte model with configured proxy speed.",
            proxy_ms=policy_proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_ms=policy_oracle_ms,
            fixed_extra_ms=0.0,
            target_tps=args.target_tps,
            upper_target_tps=args.upper_target_tps,
        ),
        scenario(
            name="persistent_active_byte_measured_proxy",
            description="Policy active-byte model but using measured live proxy request time.",
            proxy_ms=live_proxy_ms if live_proxy_ms > 0 else proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_ms=policy_oracle_ms,
            fixed_extra_ms=0.0,
            target_tps=args.target_tps,
            upper_target_tps=args.upper_target_tps,
        ),
        scenario(
            name="persistent_active_byte_configured_proxy",
            description="Policy active-byte model with --proxy-tps.",
            proxy_ms=proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_ms=policy_oracle_ms,
            fixed_extra_ms=0.0,
            target_tps=args.target_tps,
            upper_target_tps=args.upper_target_tps,
        ),
    ]

    format_shortlist_scenarios: list[Scenario] = []
    fallback_limits: list[FallbackLimit] = []
    sparse_fallback_component_ms = float_from(sparse_runtime_step_summary, "component_measured_ms", 0.0)
    sparse_fallback_gpu_ms = (
        float_from(sparse_runtime_step_summary, "pcie_transfer_ms", 0.0)
        + float_from(sparse_runtime_step_summary, "cuda_kernel_ms", 0.0)
    )
    for item in shortlist_payloads:
        shortlist = item["summary"]
        label = shortlist_label(str(item["path"]))
        fallback_rate = float_from(shortlist, "best_eval_fallback_rate_for_exact", 1.0)
        rows_per_step = float_from(shortlist, "best_eval_candidate_rows_per_step", 0.0)
        verifier_ms = measured_candidate_verifier_ms(
            candidate_verifier_summary,
            rows_per_step,
            include_host_read=args.include_candidate_host_read,
        )
        for proxy_label, proxy_value in (
            ("configured_proxy", proxy_ms),
            ("measured_proxy", live_proxy_ms if live_proxy_ms > 0 else proxy_ms),
        ):
            format_shortlist_scenarios.append(
                scenario(
                    name=f"format_scaffold_{label}_{proxy_label}",
                    description=(
                        "Proxy top-k plus format/prompt scaffold, Q6_K candidate verification every token, "
                        "and exact fallback only for shortlist misses."
                    ),
                    proxy_ms=proxy_value,
                    verifier_rate=1.0,
                    verifier_ms=verifier_ms,
                    oracle_rate=fallback_rate,
                    oracle_ms=args.exact_fallback_ms,
                    fixed_extra_ms=0.0,
                    target_tps=args.target_tps,
                    upper_target_tps=args.upper_target_tps,
                )
            )
            for target in (args.target_tps, args.upper_target_tps):
                fallback_limits.append(
                    fallback_limit(
                        name=f"format_scaffold_{label}_{proxy_label}",
                        target_tps=target,
                        proxy_ms=proxy_value,
                        verifier_ms=verifier_ms,
                        fallback_rate=fallback_rate,
                        configured_fallback_ms=args.exact_fallback_ms,
                    )
                )
            if sparse_fallback_component_ms > 0:
                sparse_name = f"format_scaffold_{label}_{proxy_label}_measured_sparse_replay"
                format_shortlist_scenarios.append(
                    scenario(
                        name=sparse_name,
                        description=(
                            "Proxy top-k plus format/prompt scaffold, Q6_K candidate verification every token, "
                            "and measured sparse-oracle component replay for shortlist misses."
                        ),
                        proxy_ms=proxy_value,
                        verifier_rate=1.0,
                        verifier_ms=verifier_ms,
                        oracle_rate=fallback_rate,
                        oracle_ms=sparse_fallback_component_ms,
                        fixed_extra_ms=0.0,
                        target_tps=args.target_tps,
                        upper_target_tps=args.upper_target_tps,
                    )
                )
                for target in (args.target_tps, args.upper_target_tps):
                    fallback_limits.append(
                        fallback_limit(
                            name=sparse_name,
                            target_tps=target,
                            proxy_ms=proxy_value,
                            verifier_ms=verifier_ms,
                            fallback_rate=fallback_rate,
                            configured_fallback_ms=sparse_fallback_component_ms,
                        )
                    )
            if sparse_fallback_gpu_ms > 0:
                sparse_gpu_name = f"format_scaffold_{label}_{proxy_label}_measured_sparse_gpu_only"
                format_shortlist_scenarios.append(
                    scenario(
                        name=sparse_gpu_name,
                        description=(
                            "Proxy top-k plus format/prompt scaffold, Q6_K candidate verification every token, "
                            "and measured sparse-oracle H2D+kernel time for shortlist misses, excluding host reads."
                        ),
                        proxy_ms=proxy_value,
                        verifier_rate=1.0,
                        verifier_ms=verifier_ms,
                        oracle_rate=fallback_rate,
                        oracle_ms=sparse_fallback_gpu_ms,
                        fixed_extra_ms=0.0,
                        target_tps=args.target_tps,
                        upper_target_tps=args.upper_target_tps,
                    )
                )
                for target in (args.target_tps, args.upper_target_tps):
                    fallback_limits.append(
                        fallback_limit(
                            name=sparse_gpu_name,
                            target_tps=target,
                            proxy_ms=proxy_value,
                            verifier_ms=verifier_ms,
                            fallback_rate=fallback_rate,
                            configured_fallback_ms=sparse_fallback_gpu_ms,
                        )
                    )
    scenarios.extend(format_shortlist_scenarios)

    params_b = float_from(policy_params, "params_b", args.params_b)
    quant_bpw = float_from(policy_params, "quant_bpw", args.quant_bpw)
    pcie_gbps = float_from(policy_params, "pcie_gbps", args.pcie_gbps)
    oracle_compute_ms = float_from(policy_params, "oracle_compute_ms", args.oracle_compute_ms)
    oracle_fixed_ms = float_from(policy_params, "oracle_fixed_ms", args.oracle_fixed_ms)
    ms_per_active = transfer_ms_per_active_percent(params_b, quant_bpw, pcie_gbps)

    limits = [
        active_limit(
            name="measured_proxy_target",
            target_tps=args.target_tps,
            proxy_ms=live_proxy_ms if live_proxy_ms > 0 else proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_compute_ms=oracle_compute_ms,
            oracle_fixed_ms=oracle_fixed_ms,
            ms_per_active_percent=ms_per_active,
        ),
        active_limit(
            name="configured_proxy_target",
            target_tps=args.target_tps,
            proxy_ms=proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_compute_ms=oracle_compute_ms,
            oracle_fixed_ms=oracle_fixed_ms,
            ms_per_active_percent=ms_per_active,
        ),
        active_limit(
            name="configured_proxy_upper_target",
            target_tps=args.upper_target_tps,
            proxy_ms=proxy_ms,
            verifier_rate=verifier_rate,
            verifier_ms=policy_verifier_ms,
            oracle_rate=oracle_rate,
            oracle_compute_ms=oracle_compute_ms,
            oracle_fixed_ms=oracle_fixed_ms,
            ms_per_active_percent=ms_per_active,
        ),
    ]

    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "policy": {
            "stats_total": stats_total,
            "verifier_call_rate": verifier_rate,
            "oracle_call_rate": oracle_rate,
            "policy_expected_ms_per_token": policy_expected_ms,
            "policy_effective_tps": float_from(policy, "effective_tps", 1000.0 / policy_expected_ms),
            "final_total_error_rate": float_from(policy, "final_total_error_rate", 0.0),
            "final_accepted_error_rate": float_from(policy, "final_accepted_error_rate", 0.0),
        },
        "live": {
            "steps": live_steps,
            "proxy_ms_per_token": live_proxy_ms,
            "accounted_ms_per_token": live_accounted_ms,
            "elapsed_ms_per_token": live_elapsed_ms,
            "outer_overhead_ms_per_token": live_overhead_ms,
        },
        "active_byte_model": {
            "params_b": params_b,
            "quant_bpw": quant_bpw,
            "pcie_gbps": pcie_gbps,
            "dense_weight_gib": dense_weight_gib(params_b, quant_bpw),
            "transfer_ms_per_active_percent": ms_per_active,
        },
        "format_scaffold_model": {
            "candidate_verifier_json": args.candidate_verifier_json,
            "format_shortlist_json": args.format_shortlist_json,
            "include_candidate_host_read": args.include_candidate_host_read,
            "exact_fallback_ms": args.exact_fallback_ms,
            "candidate_verifier": candidate_verifier_summary,
            "shortlists": shortlist_payloads,
        },
        "sparse_fallback_model": {
            "sparse_runtime_step_json": args.sparse_runtime_step_json,
            "component_measured_ms": sparse_fallback_component_ms,
            "gpu_only_measured_ms": sparse_fallback_gpu_ms,
            "summary": sparse_runtime_step_summary,
        },
        "scenarios": [asdict(item) for item in scenarios],
        "limits": [asdict(item) for item in limits],
        "fallback_limits": [asdict(item) for item in fallback_limits],
    }


def print_projection(payload: dict[str, Any]) -> None:
    print("# SAGE Persistent Runtime Projection")
    print()
    print("| Scenario | ms/token | tok/s | 7 tok/s | 10 tok/s |")
    print("| --- | ---: | ---: | --- | --- |")
    for item in payload["scenarios"]:
        print(
            f"| {item['name']} | {item['expected_ms_per_token']:.3f} | "
            f"{item['effective_tps']:.3f} | {item['meets_target_tps']} | {item['meets_upper_target_tps']} |"
        )
    print()
    print("| Limit | Target | Max oracle call ms | Max oracle active % | Feasible |")
    print("| --- | ---: | ---: | ---: | --- |")
    for item in payload["limits"]:
        active = item["max_oracle_active_percent"]
        active_text = "inf" if active == float("inf") else f"{active:.3f}"
        print(
            f"| {item['name']} | {item['target_tps']:.1f} | "
            f"{item['max_oracle_call_ms']:.3f} | {active_text} | {item['feasible']} |"
        )
    if payload.get("fallback_limits"):
        print()
        print("| Fallback Limit | Target | Fallback rate | Configured fallback ms | Max fallback ms | Fits |")
        print("| --- | ---: | ---: | ---: | ---: | --- |")
        for item in payload["fallback_limits"]:
            max_ms = item["max_fallback_ms"]
            max_text = "inf" if max_ms == float("inf") else f"{max_ms:.3f}"
            print(
                f"| {item['name']} | {item['target_tps']:.1f} | "
                f"{item['fallback_rate']:.2%} | {item['configured_fallback_ms']:.3f} | "
                f"{max_text} | {item['meets_configured_fallback_ms']} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Project SAGE persistent-runtime speed from measured artifacts.")
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--live-report-json", default="")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--upper-target-tps", type=float, default=10.0)
    parser.add_argument("--proxy-tps", type=float, default=25.0)
    parser.add_argument("--use-live-proxy", action="store_true")
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--verifier-call-ms", type=float, default=25.4166666667)
    parser.add_argument("--oracle-call-ms", type=float, default=119.1666666667)
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--candidate-verifier-json", default="")
    parser.add_argument("--sparse-runtime-step-json", default="")
    parser.add_argument("--format-shortlist-json", nargs="*", default=[])
    parser.add_argument(
        "--include-candidate-host-read",
        action="store_true",
        help="include measured GGUF host-row read time in candidate verifier projection",
    )
    parser.add_argument(
        "--exact-fallback-ms",
        type=float,
        default=270.0,
        help="assumed exact fallback latency for one token when shortlist verification misses",
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.target_tps <= 0 or args.upper_target_tps <= 0 or args.proxy_tps <= 0:
        parser.error("target and proxy tps values must be positive")
    if args.exact_fallback_ms < 0:
        parser.error("--exact-fallback-ms must be non-negative")
    payload = make_projection(args)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_projection(payload)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
