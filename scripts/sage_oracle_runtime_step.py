#!/usr/bin/env python3
"""
Assemble a measured SAGE sparse-oracle runtime-step replay artifact.

This script does not implement transformer execution inside llama.cpp. It joins
the measured component artifacts into the per-token ledger shape the runtime
needs: block-page transport, CUDA page consumption, sparse Q6_K candidate
verification, and exact fallback when the candidate set misses the oracle top
token.
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
        fail(f"missing input JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path} did not contain a JSON object")
    return payload


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("summary", {})
    return item if isinstance(item, dict) else {}


def int_value(item: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(item.get(key, default))
    except (TypeError, ValueError):
        return default


def float_value(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(item.get(key, default))
    except (TypeError, ValueError):
        return default


def bool_value(item: dict[str, Any], key: str, default: bool = False) -> bool:
    value = item.get(key, default)
    return bool(value)


def bytes_to_gib(value: int | float) -> float:
    return float(value) / BYTES_PER_GIB


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    page_ledger_path = Path(args.page_ledger_json)
    page_kernel_path = Path(args.page_cuda_kernel_json)
    candidate_path = Path(args.candidate_verifier_json)
    kv_runtime_path = Path(args.kv_runtime_ledger_json) if args.kv_runtime_ledger_json else None

    page_ledger = load_json(page_ledger_path)
    page_kernel = load_json(page_kernel_path)
    candidate = load_json(candidate_path)
    kv_runtime = load_json(kv_runtime_path) if kv_runtime_path else {}

    page_summary = summary(page_ledger)
    page_kernel_summary = summary(page_kernel)
    candidate_summary = summary(candidate)
    kv_summary = summary(kv_runtime)

    if page_ledger.get("schema") != "sage-oracle-page-ledger-v0":
        fail("page ledger schema mismatch")
    if page_kernel.get("schema") != "sage-oracle-page-cuda-kernel-smoke-v0":
        fail("page CUDA kernel schema mismatch")
    if candidate.get("schema") != "sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0":
        fail("candidate verifier schema mismatch")

    sparse_page_bytes = int_value(page_summary, "selected_bytes")
    candidate_bytes = int_value(candidate_summary, "candidate_bytes")
    exact_fallback_bytes = int_value(page_summary, "exact_fallback_bytes")
    active_sparse_step_bytes = sparse_page_bytes + candidate_bytes
    reference_100b = page_ledger.get("reference_100b", {})
    reference_dense_bytes = int(float_value(reference_100b, "dense_weight_gib") * BYTES_PER_GIB)
    active_reference_percent = (
        100.0 * active_sparse_step_bytes / reference_dense_bytes if reference_dense_bytes > 0 else 0.0
    )
    max_page_buffer = int_value(page_kernel_summary, "max_live_buffer_bytes")
    candidate_stage_buffer = int_value(candidate_summary, "allocated_stage_buffer_bytes")
    max_measured_device_stage_bytes = max(max_page_buffer, candidate_stage_buffer)
    pcie_transfer_ms = float_value(page_kernel_summary, "h2d_ms") + float_value(candidate_summary, "h2d_ms")
    cuda_kernel_ms = float_value(page_kernel_summary, "kernel_ms") + float_value(candidate_summary, "kernel_ms")
    host_read_ms = float_value(page_kernel_summary, "host_read_ms") + float_value(candidate_summary, "host_read_ms")
    candidate_covers_global_top1 = bool_value(candidate_summary, "candidate_contains_llamacpp_global_top1")
    exact_fallback_required = not candidate_covers_global_top1
    component_replay_complete = (
        sparse_page_bytes > 0
        and candidate_bytes > 0
        and page_kernel.get("status") == "measured_cuda_kernel_touch_not_transformer"
        and candidate.get("status") == "measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits"
        and bool_value(page_kernel_summary, "stage_byte_match")
        and bool_value(page_kernel_summary, "byte_budget_respected")
        and bool_value(candidate_summary, "llamacpp_logit_checks_passed")
        and bool_value(candidate_summary, "cpu_score_checks_passed")
    )

    return {
        "schema": "sage-sparse-oracle-runtime-step-v0",
        "status": "measured_component_replay_not_transformer_integrated",
        "inputs": {
            "page_ledger_json": str(page_ledger_path.resolve()),
            "page_cuda_kernel_json": str(page_kernel_path.resolve()),
            "candidate_verifier_json": str(candidate_path.resolve()),
            "kv_runtime_ledger_json": str(kv_runtime_path.resolve()) if kv_runtime_path else "",
        },
        "summary": {
            "component_replay_complete": component_replay_complete,
            "transformer_integrated": False,
            "llama_cpp_live_integrated": False,
            "page_plan_status": page_ledger.get("status", ""),
            "page_cuda_status": page_kernel.get("status", ""),
            "candidate_status": candidate.get("status", ""),
            "selected_pages": int_value(page_summary, "selected_pages"),
            "stage_count": int_value(page_summary, "stage_count"),
            "sparse_page_bytes": sparse_page_bytes,
            "sparse_page_gib": bytes_to_gib(sparse_page_bytes),
            "candidate_rows": int_value(candidate_summary, "candidate_tokens"),
            "candidate_bytes": candidate_bytes,
            "candidate_gib": bytes_to_gib(candidate_bytes),
            "active_sparse_step_bytes": active_sparse_step_bytes,
            "active_sparse_step_gib": bytes_to_gib(active_sparse_step_bytes),
            "active_sparse_step_percent_reference_100b": active_reference_percent,
            "max_measured_device_stage_bytes": max_measured_device_stage_bytes,
            "max_measured_device_stage_gib": bytes_to_gib(max_measured_device_stage_bytes),
            "page_h2d_ms": float_value(page_kernel_summary, "h2d_ms"),
            "page_kernel_ms": float_value(page_kernel_summary, "kernel_ms"),
            "candidate_h2d_ms": float_value(candidate_summary, "h2d_ms"),
            "candidate_kernel_ms": float_value(candidate_summary, "kernel_ms"),
            "host_read_ms": host_read_ms,
            "pcie_transfer_ms": pcie_transfer_ms,
            "cuda_kernel_ms": cuda_kernel_ms,
            "component_measured_ms": host_read_ms + pcie_transfer_ms + cuda_kernel_ms,
            "candidate_source_kind": candidate_summary.get("candidate_source_kind", ""),
            "candidate_contains_llamacpp_global_top1": candidate_covers_global_top1,
            "candidate_global_top1_coverage_status": candidate_summary.get(
                "candidate_global_top1_coverage_status",
                "",
            ),
            "exact_fallback_required": exact_fallback_required,
            "exact_fallback_bytes": exact_fallback_bytes,
            "exact_fallback_gib": bytes_to_gib(exact_fallback_bytes),
            "kv_runtime_accounting_status": kv_runtime.get("status", ""),
            "kv_runtime_attention_integrated": bool_value(kv_summary, "attention_integration", False),
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_candidate_verifier_replay",
            "oracle_active_bytes": active_sparse_step_bytes,
            "oracle_page_active_bytes": sparse_page_bytes,
            "verifier_active_bytes": candidate_bytes,
            "oracle_active_percent_of_reference_100b": active_reference_percent,
            "oracle_blocks": page_ledger.get("runtime_ledger_template", {}).get("oracle_blocks", []),
            "gpu_staged_bytes_total": active_sparse_step_bytes,
            "max_live_gpu_stage_bytes": max_measured_device_stage_bytes,
            "host_pinned_bytes": int_value(page_kernel.get("runtime_ledger_evidence", {}), "host_pinned_bytes"),
            "pcie_transfer_ms": pcie_transfer_ms,
            "cuda_kernel_ms": cuda_kernel_ms,
            "candidate_rows": int_value(candidate_summary, "candidate_tokens"),
            "candidate_scoring_status": candidate_summary.get("candidate_scoring_status", ""),
            "candidate_contains_llamacpp_global_top1": candidate_covers_global_top1,
            "exact_fallback_required": exact_fallback_required,
            "exact_fallback_bytes": exact_fallback_bytes,
            "transformer_integrated": False,
            "llama_cpp_live_integrated": False,
        },
        "component_evidence": {
            "page_ledger": {
                "schema": page_ledger.get("schema", ""),
                "status": page_ledger.get("status", ""),
                "selected_bytes": sparse_page_bytes,
                "active_percent_of_reference_100b": float_value(page_summary, "active_percent_of_reference_100b"),
            },
            "page_cuda_kernel": {
                "schema": page_kernel.get("schema", ""),
                "status": page_kernel.get("status", ""),
                "stage_byte_match": bool_value(page_kernel_summary, "stage_byte_match"),
                "byte_budget_respected": bool_value(page_kernel_summary, "byte_budget_respected"),
                "kernel_output_nonzero": bool_value(page_kernel_summary, "kernel_output_nonzero"),
            },
            "candidate_verifier": {
                "schema": candidate.get("schema", ""),
                "status": candidate.get("status", ""),
                "llamacpp_logit_checks_passed": bool_value(candidate_summary, "llamacpp_logit_checks_passed"),
                "cpu_score_checks_passed": bool_value(candidate_summary, "cpu_score_checks_passed"),
                "llamacpp_candidate_top1_match": bool_value(candidate_summary, "llamacpp_candidate_top1_match"),
            },
        },
    }


def print_markdown(payload: dict[str, Any]) -> None:
    item = payload["summary"]
    print("# SAGE Sparse Oracle Runtime-Step Replay")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Component replay complete: `{item['component_replay_complete']}`")
    print(f"- Sparse page bytes: `{item['sparse_page_bytes']}` (`{item['sparse_page_gib']:.6f} GiB`)")
    print(f"- Candidate bytes: `{item['candidate_bytes']}` (`{item['candidate_gib']:.6f} GiB`)")
    print(f"- Active sparse step: `{item['active_sparse_step_percent_reference_100b']:.4f}%` of 100B 2-bit reference")
    print(f"- Max measured device stage: `{item['max_measured_device_stage_gib']:.6f} GiB`")
    print(f"- H2D time: `{item['pcie_transfer_ms']:.4f} ms`")
    print(f"- CUDA kernel time: `{item['cuda_kernel_ms']:.4f} ms`")
    print(f"- Exact fallback required: `{item['exact_fallback_required']}`")
    print(f"- Transformer integrated: `{item['transformer_integrated']}`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a measured SAGE sparse-oracle runtime-step replay.")
    parser.add_argument("--page-ledger-json", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--page-cuda-kernel-json", default="benchmarks/sage-oracle-page-cuda-kernel-gemma31b-full.json")
    parser.add_argument(
        "--candidate-verifier-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-candidate-verifier-gemma12proxy-france-top10-fallback-result-norm.json",
    )
    parser.add_argument("--kv-runtime-ledger-json", default="benchmarks/sage-kv-runtime-ledger-qwen05b-arithmetic-gemma31b-plan.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

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
