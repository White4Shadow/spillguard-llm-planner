#!/usr/bin/env python3
"""
Annotate a live SAGE trace with hot/warm/cold oracle KV byte accounting.

This does not integrate compressed KV into llama.cpp attention. It takes runtime
token counts from a live trace and applies the measured SAGE KV tier policy so
the active-byte ledger can account for KV bytes alongside oracle weight pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sage_kv_ledger import bytes_to_gib


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"expected JSON object in {path}")
    return payload


def int_value(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    value = mapping.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_tier(ledger_payload: dict[str, Any], name: str) -> dict[str, Any]:
    oracle = ledger_payload.get("oracle", {})
    tiers = oracle.get("tiers", []) if isinstance(oracle, dict) else []
    if not isinstance(tiers, list):
        return {}
    for tier in tiers:
        if isinstance(tier, dict) and tier.get("name") == name:
            return tier
    return {}


def parse_sweep_tokens(raw: str) -> list[int]:
    tokens: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError:
            fail(f"invalid integer in --context-sweep-tokens: {stripped!r}")
        if value < 0:
            fail("--context-sweep-tokens values must be non-negative")
        tokens.append(value)
    return sorted(set(tokens))


class KVPolicy:
    def __init__(self, ledger_payload: dict[str, Any]) -> None:
        params = ledger_payload.get("params", {})
        oracle = ledger_payload.get("oracle", {})
        full_context = oracle.get("full_precision_context", {}) if isinstance(oracle, dict) else {}
        hot_tier = find_tier(ledger_payload, "hot")
        warm_tier = find_tier(ledger_payload, "warm")
        cold_tier = find_tier(ledger_payload, "cold")

        self.context_tokens = int_value(params, "context_tokens")
        self.hot_recent_tokens = int_value(params, "hot_recent_tokens")
        self.sink_tokens = int_value(params, "sink_tokens")
        self.warm_max_tokens = int_value(params, "warm_max_tokens")
        self.warm_bits = float_value(params, "warm_bits")
        self.cold_summary_dim = int_value(params, "cold_summary_dim")
        self.cold_summary_bits = float_value(params, "cold_summary_bits")
        self.full_bytes_per_token = float_value(full_context, "bytes_per_token")
        self.hot_bytes_per_token = float_value(hot_tier, "bytes_per_token", self.full_bytes_per_token)
        self.warm_bytes_per_token = float_value(warm_tier, "bytes_per_token")
        self.cold_bytes_per_token = float_value(cold_tier, "bytes_per_token")

        if self.context_tokens <= 0:
            fail("KV ledger context_tokens must be positive")
        if self.full_bytes_per_token <= 0:
            fail("KV ledger must provide full precision bytes_per_token")
        if self.warm_bytes_per_token <= 0:
            fail("KV ledger must provide warm bytes_per_token")
        if self.hot_recent_tokens < 0 or self.sink_tokens < 0 or self.warm_max_tokens < 0:
            fail("KV ledger token tier sizes must be non-negative")

    def account(self, tokens: int) -> dict[str, Any]:
        if tokens < 0:
            fail("KV token count must be non-negative")
        sink_tokens = min(tokens, self.sink_tokens)
        remaining = max(0, tokens - sink_tokens)
        recent_tokens = min(remaining, self.hot_recent_tokens)
        warm_tokens = min(max(0, remaining - recent_tokens), self.warm_max_tokens)
        cold_tokens = max(0, remaining - recent_tokens - warm_tokens)
        hot_tokens = sink_tokens + recent_tokens

        full_bytes = int(round(tokens * self.full_bytes_per_token))
        hot_bytes = int(round(hot_tokens * self.hot_bytes_per_token))
        warm_bytes = int(round(warm_tokens * self.warm_bytes_per_token))
        cold_bytes = int(round(cold_tokens * self.cold_bytes_per_token))
        tier_total = hot_bytes + warm_bytes + cold_bytes
        saved_bytes = max(0, full_bytes - tier_total)
        return {
            "tokens": tokens,
            "sink_tokens": sink_tokens,
            "recent_tokens": recent_tokens,
            "hot_tokens": hot_tokens,
            "warm_tokens": warm_tokens,
            "cold_tokens": cold_tokens,
            "full_precision_bytes": full_bytes,
            "hot_kv_bytes": hot_bytes,
            "warm_kv_bytes": warm_bytes,
            "cold_kv_bytes": cold_bytes,
            "tier_total_bytes": tier_total,
            "tier_total_gib": bytes_to_gib(tier_total),
            "saved_bytes_vs_full_precision": saved_bytes,
            "saved_percent_vs_full_precision": 100.0 * saved_bytes / full_bytes if full_bytes > 0 else 0.0,
            "kv_byte_status": "tiered_accounting_from_runtime_token_count",
        }


def step_kv_tokens(step: dict[str, Any], key: str) -> int:
    ledger = step.get("ledger", {})
    if isinstance(ledger, dict) and key in ledger:
        return int_value(ledger, key)
    if key == "proxy_kv_tokens":
        proxy = step.get("proxy", {})
        if isinstance(proxy, dict):
            return int_value(proxy, "position")
    if key == "oracle_kv_tokens":
        oracle = step.get("oracle", {})
        if isinstance(oracle, dict):
            return int_value(oracle, "position")
    return 0


def make_step_annotation(step: dict[str, Any], policy: KVPolicy) -> dict[str, Any]:
    ledger = step.get("ledger", {})
    ledger = ledger if isinstance(ledger, dict) else {}
    proxy_tokens = step_kv_tokens(step, "proxy_kv_tokens")
    oracle_tokens = step_kv_tokens(step, "oracle_kv_tokens")
    proxy_full_bytes = int(round(proxy_tokens * policy.full_bytes_per_token))
    oracle_account = policy.account(oracle_tokens)
    return {
        "step_index": int_value(step, "step_index"),
        "action": step.get("action", ""),
        "selected_source": step.get("selected_source", ""),
        "selected_token": step.get("selected_token", ""),
        "source_ledger_kv_status": ledger.get("kv_byte_status", ""),
        "proxy_kv_tokens": proxy_tokens,
        "proxy_full_precision_kv_bytes": proxy_full_bytes,
        "proxy_full_precision_kv_gib": bytes_to_gib(proxy_full_bytes),
        "oracle_kv_tokens": oracle_tokens,
        "oracle_kv_accounting": oracle_account,
    }


def max_int(items: list[dict[str, Any]], path: tuple[str, ...]) -> int:
    best = 0
    for item in items:
        current: Any = item
        for key in path:
            if not isinstance(current, dict):
                current = {}
                break
            current = current.get(key, {})
        try:
            best = max(best, int(current))
        except (TypeError, ValueError):
            pass
    return best


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    live_trace_path = Path(args.live_trace_json)
    kv_ledger_path = Path(args.kv_ledger_json)
    kv_tier_smoke_path = Path(args.kv_tier_smoke_json)
    live_trace = load_json(live_trace_path)
    kv_ledger = load_json(kv_ledger_path)
    kv_tier_smoke = load_json(kv_tier_smoke_path)
    if kv_ledger.get("schema") != "sage-kv-ledger-v0":
        fail("expected sage-kv-ledger-v0 ledger artifact")
    if kv_tier_smoke.get("schema") != "sage-kv-tier-pack-smoke-v0":
        fail("expected sage-kv-tier-pack-smoke-v0 pack artifact")

    policy = KVPolicy(kv_ledger)
    steps_raw = live_trace.get("steps", [])
    if not isinstance(steps_raw, list):
        fail("live trace steps must be a JSON array")
    steps = [step for step in steps_raw if isinstance(step, dict)]
    annotations = [make_step_annotation(step, policy) for step in steps]
    sweep_tokens = parse_sweep_tokens(args.context_sweep_tokens)
    if args.include_context_limit and policy.context_tokens not in sweep_tokens:
        sweep_tokens.append(policy.context_tokens)
        sweep_tokens = sorted(set(sweep_tokens))
    context_sweep = [
        {
            "source": "configured_context_sweep",
            "oracle_kv_accounting": policy.account(tokens),
        }
        for tokens in sweep_tokens
    ]

    summary = live_trace.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    fallback_steps = sum(1 for row in annotations if row["action"] == "oracle_fallback")
    proxy_accept_steps = sum(1 for row in annotations if row["action"] == "accept_proxy")
    max_oracle_full_bytes = max_int(annotations + context_sweep, ("oracle_kv_accounting", "full_precision_bytes"))
    max_oracle_tier_bytes = max_int(annotations + context_sweep, ("oracle_kv_accounting", "tier_total_bytes"))
    max_oracle_warm_bytes = max_int(annotations + context_sweep, ("oracle_kv_accounting", "warm_kv_bytes"))
    max_oracle_saved_percent = max(
        (
            float_value(row.get("oracle_kv_accounting", {}), "saved_percent_vs_full_precision")
            for row in annotations + context_sweep
            if isinstance(row.get("oracle_kv_accounting"), dict)
        ),
        default=0.0,
    )
    cuda_pack = kv_tier_smoke.get("cuda_pack", {})
    cuda_pack = cuda_pack if isinstance(cuda_pack, dict) else {}
    smoke_summary = kv_tier_smoke.get("summary", {})
    smoke_summary = smoke_summary if isinstance(smoke_summary, dict) else {}
    return {
        "schema": "sage-kv-runtime-ledger-v0",
        "status": "runtime_token_accounting_with_measured_pack_smoke_not_attention_integrated",
        "sources": {
            "live_trace_json": str(live_trace_path),
            "kv_ledger_json": str(kv_ledger_path),
            "kv_tier_smoke_json": str(kv_tier_smoke_path),
        },
        "runtime_trace": {
            "ledger_schema": summary.get("ledger_schema", ""),
            "kv_byte_status": summary.get("kv_byte_status", ""),
            "generated_tokens": int_value(summary, "generated_tokens"),
            "proxy_accepts": int_value(summary, "proxy_accepts"),
            "oracle_fallbacks": int_value(summary, "oracle_fallbacks"),
        },
        "policy": {
            "context_tokens": policy.context_tokens,
            "hot_recent_tokens": policy.hot_recent_tokens,
            "sink_tokens": policy.sink_tokens,
            "warm_max_tokens": policy.warm_max_tokens,
            "warm_bits": policy.warm_bits,
            "cold_summary_dim": policy.cold_summary_dim,
            "cold_summary_bits": policy.cold_summary_bits,
            "full_precision_bytes_per_token": policy.full_bytes_per_token,
            "warm_bytes_per_token": policy.warm_bytes_per_token,
            "cold_bytes_per_token": policy.cold_bytes_per_token,
        },
        "measured_pack_evidence": {
            "schema": kv_tier_smoke.get("schema", ""),
            "status": kv_tier_smoke.get("status", ""),
            "compression_ratio_vs_fp16": float_value(smoke_summary, "compression_ratio_vs_fp16"),
            "bytes_match_plan": bool(smoke_summary.get("bytes_match_plan", False)),
            "checksums_match": bool(smoke_summary.get("checksums_match", False)),
            "cuda_enabled": bool(cuda_pack.get("enabled", False)),
            "cuda_packed_matches_cpu": bool(cuda_pack.get("packed_matches_cpu", False)),
            "cuda_pack_ms": float_value(cuda_pack, "pack_ms"),
            "cuda_unpack_ms": float_value(cuda_pack, "unpack_ms"),
            "cuda_scaled_full_warm_pack_ms": float_value(cuda_pack, "scaled_full_warm_pack_ms"),
            "cuda_scaled_full_warm_unpack_ms": float_value(cuda_pack, "scaled_full_warm_unpack_ms"),
        },
        "summary": {
            "annotated_steps": len(annotations),
            "runtime_oracle_fallback_steps": fallback_steps,
            "runtime_proxy_accept_steps": proxy_accept_steps,
            "max_runtime_proxy_kv_tokens": max_int(annotations, ("proxy_kv_tokens",)),
            "max_runtime_oracle_kv_tokens": max_int(annotations, ("oracle_kv_tokens",)),
            "context_sweep_rows": len(context_sweep),
            "context_sweep_exercises_warm_kv": any(
                row["oracle_kv_accounting"]["warm_tokens"] > 0 for row in context_sweep
            ),
            "max_oracle_full_precision_bytes": max_oracle_full_bytes,
            "max_oracle_full_precision_gib": bytes_to_gib(max_oracle_full_bytes),
            "max_oracle_tier_total_bytes": max_oracle_tier_bytes,
            "max_oracle_tier_total_gib": bytes_to_gib(max_oracle_tier_bytes),
            "max_oracle_warm_kv_bytes": max_oracle_warm_bytes,
            "max_oracle_warm_kv_gib": bytes_to_gib(max_oracle_warm_bytes),
            "max_oracle_saved_percent_vs_full_precision": max_oracle_saved_percent,
            "attention_integration": False,
            "kv_byte_status": "tiered_runtime_accounting_not_attention_integrated",
        },
        "step_ledgers": annotations,
        "context_sweep": context_sweep,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    evidence = payload["measured_pack_evidence"]
    print("# SAGE KV Runtime Ledger")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Annotated runtime steps: `{summary['annotated_steps']}`")
    print(f"- Runtime oracle fallbacks: `{summary['runtime_oracle_fallback_steps']}`")
    print(f"- Context sweep warm KV: `{summary['context_sweep_exercises_warm_kv']}`")
    print(f"- Max oracle full precision KV: `{summary['max_oracle_full_precision_gib']:.3f} GiB`")
    print(f"- Max oracle tiered KV: `{summary['max_oracle_tier_total_gib']:.3f} GiB`")
    print(f"- Max saved vs full precision: `{summary['max_oracle_saved_percent_vs_full_precision']:.1f}%`")
    print(f"- CUDA pack evidence: `{evidence['cuda_pack_ms']:.4f} ms`, matches CPU `{evidence['cuda_packed_matches_cpu']}`")
    print(f"- Attention integration: `{summary['attention_integration']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate live SAGE traces with tiered KV byte accounting.")
    parser.add_argument("--live-trace-json", default="benchmarks/sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json")
    parser.add_argument("--kv-ledger-json", default="benchmarks/sage-kv-ledger-gemma31b-ctx4096-hot528-warm2bit.json")
    parser.add_argument(
        "--kv-tier-smoke-json",
        default="benchmarks/sage-kv-tier-pack-smoke-gemma31b-ctx4096-hot528-warm2bit-sample8.json",
    )
    parser.add_argument("--context-sweep-tokens", default="1,5,16,528,529,1024")
    parser.add_argument("--include-context-limit", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    payload = make_payload(args)
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
