#!/usr/bin/env python3
"""
Plan a SAGE hot/warm/cold KV byte ledger from GGUF tensor shapes.

This is a planning artifact, not a runtime KV implementation. It estimates how
many bytes the proxy/oracle KV tiers would use so the active-byte runtime can
budget KV alongside sparse weight pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import ModelIndex, parse_gguf


BYTES_PER_GIB = 1024**3


@dataclass
class LayerKVShape:
    layer: int
    k_dim: int
    v_dim: int


@dataclass
class KVTier:
    name: str
    tokens: int
    k_bits: float
    v_bits: float
    bytes_per_token: float
    total_bytes: int
    location: str
    mode: str


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def bytes_to_gib(n_bytes: int | float) -> float:
    return float(n_bytes) / BYTES_PER_GIB


def tensor_kv_dim(shape: list[int], hidden_dim: int) -> int:
    if not shape:
        return 0
    if len(shape) == 1:
        return int(shape[0])
    candidates = [int(dim) for dim in shape if int(dim) != hidden_dim]
    if candidates:
        return max(candidates)
    return int(shape[-1])


def infer_hidden_dim(index: ModelIndex) -> int:
    for tensor in index.tensors:
        if tensor.name == "token_embd.weight" and tensor.shape:
            return int(tensor.shape[0])
    for tensor in index.tensors:
        if tensor.layer is not None and tensor.name.endswith(".attn_q.weight") and tensor.shape:
            return max(int(dim) for dim in tensor.shape)
    return 0


def infer_layer_kv_shapes(index: ModelIndex) -> list[LayerKVShape]:
    hidden_dim = infer_hidden_dim(index)
    by_layer: dict[int, dict[str, int]] = {}
    for tensor in index.tensors:
        if tensor.layer is None:
            continue
        if tensor.name.endswith(".attn_k.weight"):
            by_layer.setdefault(tensor.layer, {})["k"] = tensor_kv_dim(tensor.shape, hidden_dim)
        elif tensor.name.endswith(".attn_v.weight"):
            by_layer.setdefault(tensor.layer, {})["v"] = tensor_kv_dim(tensor.shape, hidden_dim)
    shapes: list[LayerKVShape] = []
    for layer, dims in sorted(by_layer.items()):
        k_dim = int(dims.get("k", 0))
        v_dim = int(dims.get("v", k_dim))
        if k_dim > 0 or v_dim > 0:
            shapes.append(LayerKVShape(layer=layer, k_dim=k_dim, v_dim=v_dim))
    if not shapes:
        fail(f"could not infer KV shapes from {index.path}")
    return shapes


def bytes_per_token(shapes: list[LayerKVShape], k_bits: float, v_bits: float) -> float:
    total_bits = 0.0
    for shape in shapes:
        total_bits += shape.k_dim * k_bits
        total_bits += shape.v_dim * v_bits
    return total_bits / 8.0


def make_tier(
    *,
    name: str,
    tokens: int,
    shapes: list[LayerKVShape],
    k_bits: float,
    v_bits: float,
    location: str,
    mode: str,
) -> KVTier:
    bpt = bytes_per_token(shapes, k_bits, v_bits)
    return KVTier(
        name=name,
        tokens=tokens,
        k_bits=k_bits,
        v_bits=v_bits,
        bytes_per_token=bpt,
        total_bytes=int(round(tokens * bpt)),
        location=location,
        mode=mode,
    )


def make_cold_tier(tokens: int, summary_dim: int, summary_bits: float) -> KVTier:
    bpt = summary_dim * summary_bits / 8.0
    return KVTier(
        name="cold",
        tokens=tokens,
        k_bits=summary_bits,
        v_bits=summary_bits,
        bytes_per_token=bpt,
        total_bytes=int(round(tokens * bpt)),
        location="disk_or_ram",
        mode="summary_vectors",
    )


def model_payload(index: ModelIndex, context_tokens: int, hot_recent_tokens: int, sink_tokens: int, warm_max_tokens: int, warm_bits: float, cold_summary_dim: int, cold_summary_bits: float) -> dict[str, Any]:
    shapes = infer_layer_kv_shapes(index)
    hot_tokens = min(context_tokens, hot_recent_tokens + sink_tokens)
    warm_tokens = min(max(0, context_tokens - hot_tokens), warm_max_tokens)
    cold_tokens = max(0, context_tokens - hot_tokens - warm_tokens)
    tiers = [
        make_tier(
            name="hot",
            tokens=hot_tokens,
            shapes=shapes,
            k_bits=16.0,
            v_bits=16.0,
            location="vram",
            mode="full_precision_recent_plus_sinks",
        ),
        make_tier(
            name="warm",
            tokens=warm_tokens,
            shapes=shapes,
            k_bits=warm_bits,
            v_bits=warm_bits,
            location="ram",
            mode="quantized_kv",
        ),
        make_cold_tier(cold_tokens, cold_summary_dim, cold_summary_bits),
    ]
    full_context_bpt = bytes_per_token(shapes, 16.0, 16.0)
    full_context_bytes = int(round(context_tokens * full_context_bpt))
    tier_bytes = sum(tier.total_bytes for tier in tiers)
    return {
        "model": {
            "path": index.path,
            "name": Path(index.path).name,
            "architecture": index.metadata.get("general.architecture", "unknown"),
            "layer_count": index.layer_count,
            "tensor_bytes": index.total_tensor_bytes,
        },
        "kv_shape": {
            "layers": len(shapes),
            "sum_k_dim": sum(shape.k_dim for shape in shapes),
            "sum_v_dim": sum(shape.v_dim for shape in shapes),
            "mean_k_dim": sum(shape.k_dim for shape in shapes) / len(shapes),
            "mean_v_dim": sum(shape.v_dim for shape in shapes) / len(shapes),
            "first_layers": [asdict(shape) for shape in shapes[:8]],
        },
        "full_precision_context": {
            "tokens": context_tokens,
            "bytes_per_token": full_context_bpt,
            "total_bytes": full_context_bytes,
            "total_gib": bytes_to_gib(full_context_bytes),
        },
        "tiers": [asdict(tier) for tier in tiers],
        "summary": {
            "context_tokens": context_tokens,
            "hot_tokens": hot_tokens,
            "warm_tokens": warm_tokens,
            "cold_tokens": cold_tokens,
            "tier_total_bytes": tier_bytes,
            "tier_total_gib": bytes_to_gib(tier_bytes),
            "full_precision_total_bytes": full_context_bytes,
            "full_precision_total_gib": bytes_to_gib(full_context_bytes),
            "saved_bytes_vs_full_precision": max(0, full_context_bytes - tier_bytes),
            "saved_percent_vs_full_precision": (
                100.0 * max(0, full_context_bytes - tier_bytes) / full_context_bytes
                if full_context_bytes > 0
                else 0.0
            ),
        },
    }


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    proxy_path = Path(args.proxy_model) if args.proxy_model else Path(args.oracle_model)
    oracle_path = Path(args.oracle_model)
    if not proxy_path.is_file():
        fail(f"proxy model not found: {proxy_path}")
    if not oracle_path.is_file():
        fail(f"oracle model not found: {oracle_path}")
    if args.context_tokens <= 0:
        fail("--context-tokens must be positive")
    if args.hot_recent_tokens < 0 or args.sink_tokens < 0 or args.warm_max_tokens < 0:
        fail("token tier sizes must be non-negative")
    if args.warm_bits <= 0 or args.cold_summary_bits <= 0:
        fail("bit widths must be positive")
    if args.cold_summary_dim < 0:
        fail("--cold-summary-dim must be non-negative")

    proxy = model_payload(
        parse_gguf(proxy_path),
        args.context_tokens,
        args.hot_recent_tokens,
        args.sink_tokens,
        args.warm_max_tokens,
        args.warm_bits,
        args.cold_summary_dim,
        args.cold_summary_bits,
    )
    oracle = model_payload(
        parse_gguf(oracle_path),
        args.context_tokens,
        args.hot_recent_tokens,
        args.sink_tokens,
        args.warm_max_tokens,
        args.warm_bits,
        args.cold_summary_dim,
        args.cold_summary_bits,
    )
    oracle_summary = oracle["summary"]
    hot_bytes = next(tier["total_bytes"] for tier in oracle["tiers"] if tier["name"] == "hot")
    warm_bytes = next(tier["total_bytes"] for tier in oracle["tiers"] if tier["name"] == "warm")
    cold_bytes = next(tier["total_bytes"] for tier in oracle["tiers"] if tier["name"] == "cold")
    return {
        "schema": "sage-kv-ledger-v0",
        "status": "plan_only_not_runtime_integrated",
        "params": {
            "context_tokens": args.context_tokens,
            "hot_recent_tokens": args.hot_recent_tokens,
            "sink_tokens": args.sink_tokens,
            "warm_max_tokens": args.warm_max_tokens,
            "warm_bits": args.warm_bits,
            "cold_summary_dim": args.cold_summary_dim,
            "cold_summary_bits": args.cold_summary_bits,
            "oracle_hot_vram_budget_gib": args.oracle_hot_vram_budget_gib,
        },
        "proxy": proxy,
        "oracle": oracle,
        "summary": {
            "oracle_tier_total_bytes": oracle_summary["tier_total_bytes"],
            "oracle_tier_total_gib": oracle_summary["tier_total_gib"],
            "oracle_full_precision_total_gib": oracle_summary["full_precision_total_gib"],
            "oracle_saved_percent_vs_full_precision": oracle_summary["saved_percent_vs_full_precision"],
            "oracle_hot_bytes": hot_bytes,
            "oracle_hot_gib": bytes_to_gib(hot_bytes),
            "oracle_warm_bytes": warm_bytes,
            "oracle_warm_gib": bytes_to_gib(warm_bytes),
            "oracle_cold_bytes": cold_bytes,
            "oracle_cold_gib": bytes_to_gib(cold_bytes),
            "oracle_hot_fits_budget": bytes_to_gib(hot_bytes) <= args.oracle_hot_vram_budget_gib,
        },
        "runtime_ledger_fields": {
            "proxy_kv_bytes": proxy["summary"]["tier_total_bytes"],
            "oracle_hot_kv_bytes": hot_bytes,
            "oracle_warm_kv_bytes": warm_bytes,
            "oracle_cold_kv_bytes": cold_bytes,
            "kv_byte_status": "planned_not_runtime_integrated",
        },
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    oracle = payload["oracle"]
    print("# SAGE KV Ledger Plan")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Oracle model: `{oracle['model']['name']}`")
    print(f"- Context tokens: `{oracle['summary']['context_tokens']}`")
    print(f"- Hot/warm/cold tokens: `{oracle['summary']['hot_tokens']}` / `{oracle['summary']['warm_tokens']}` / `{oracle['summary']['cold_tokens']}`")
    print(f"- Oracle tier total: `{summary['oracle_tier_total_gib']:.3f} GiB`")
    print(f"- Full precision baseline: `{summary['oracle_full_precision_total_gib']:.3f} GiB`")
    print(f"- Saved vs full precision: `{summary['oracle_saved_percent_vs_full_precision']:.1f}%`")
    print(f"- Hot KV fits budget: `{summary['oracle_hot_fits_budget']}`")
    print()
    print("| Tier | Tokens | Location | Mode | Bytes |")
    print("| --- | ---: | --- | --- | ---: |")
    for tier in oracle["tiers"]:
        print(f"| {tier['name']} | {tier['tokens']} | {tier['location']} | {tier['mode']} | {bytes_to_gib(tier['total_bytes']):.3f} GiB |")
    print()
    print("## KV Shape")
    print()
    print(f"- Layers: `{oracle['kv_shape']['layers']}`")
    print(f"- Sum K dim: `{oracle['kv_shape']['sum_k_dim']}`")
    print(f"- Sum V dim: `{oracle['kv_shape']['sum_v_dim']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan hot/warm/cold SAGE KV byte tiers from GGUF tensor shapes.")
    parser.add_argument("--oracle-model", required=True)
    parser.add_argument("--proxy-model", default="")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--hot-recent-tokens", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--warm-max-tokens", type=int, default=3584)
    parser.add_argument("--warm-bits", type=float, default=2.0)
    parser.add_argument("--cold-summary-dim", type=int, default=256)
    parser.add_argument("--cold-summary-bits", type=float, default=8.0)
    parser.add_argument("--oracle-hot-vram-budget-gib", type=float, default=0.8)
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
