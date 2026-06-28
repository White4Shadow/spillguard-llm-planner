#!/usr/bin/env python3
"""
Create a SAGE sparse-oracle page ledger ranked by measured activation signal.

The baseline pager is byte-budget aware, but it does not know which selected
pages preserve the strongest measured row scores for a concrete activation.
This planner consumes a fuller real-activation matvec artifact, ranks GGUF
layer/component pages by measured row-score signal, and emits the same
sage-oracle-page-ledger-v0 schema used by the CUDA staging/matvec smokes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sage_block_plan import PlannedBlock, component_summary
from sage_gguf_blocks import parse_gguf
from sage_oracle_pager import (
    BYTES_PER_GIB,
    bytes_to_gib,
    dense_weight_gib,
    make_pages,
    make_stages,
    plan_status,
)


TENSOR_BLOCK_RE = re.compile(r"^blk\.(\d+)\.([^.]+)\.")


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


def float_value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def tensor_block_key(name: str) -> str | None:
    match = TENSOR_BLOCK_RE.match(name)
    if not match:
        return None
    layer = int(match.group(1))
    suffix = match.group(2)
    if suffix.startswith("attn"):
        component = "attention"
    elif suffix.startswith("ffn"):
        component = "ffn"
    elif "norm" in suffix:
        component = "norm"
    else:
        component = "other"
    return f"blk.{layer}.{component}"


def extract_signal(signal_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    block_signal: dict[str, dict[str, Any]] = {}
    for stage in signal_payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for tensor in stage.get("top_scores", []):
            if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
                continue
            block_key = tensor_block_key(tensor["name"])
            if block_key is None:
                continue
            entry = block_signal.setdefault(
                block_key,
                {
                    "score_sum": 0.0,
                    "max_abs_score": 0.0,
                    "row_count": 0,
                    "tensor_names": set(),
                    "top_rows": [],
                },
            )
            entry["tensor_names"].add(tensor["name"])
            rows = tensor.get("top_scores", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                score = float_value(row, "score")
                abs_score = abs(score)
                entry["score_sum"] += abs_score
                entry["max_abs_score"] = max(float(entry["max_abs_score"]), abs_score)
                entry["row_count"] += 1
                entry["top_rows"].append(
                    {
                        "tensor": tensor["name"],
                        "row": row.get("row"),
                        "score": score,
                        "abs_score": abs_score,
                    }
                )

    for entry in block_signal.values():
        entry["tensor_names"] = sorted(entry["tensor_names"])
        entry["top_rows"] = sorted(entry["top_rows"], key=lambda item: item["abs_score"], reverse=True)[:16]
    return block_signal


def validate_signal_payload(payload: dict[str, Any]) -> None:
    allowed_schemas = {
        "sage-oracle-page-cuda-real-activation-matvec-smoke-v0",
        "sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0",
    }
    if payload.get("schema") not in allowed_schemas:
        fail("signal artifact must be a real-activation matvec artifact")
    summary = payload.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    if summary.get("activation_mode") != "real_tensor_values_jsonl":
        fail("signal artifact must use a real tensor-values activation")
    if summary.get("row_score_capture_status") != "measured_per_row_scores":
        fail("signal artifact must include per-row top score capture")


def make_signal_plan(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")
    signal_path = Path(args.signal_json)
    signal_payload = load_json(signal_path)
    validate_signal_payload(signal_payload)

    index = parse_gguf(model_path)
    model_bytes = int(index.total_tensor_bytes)
    budget_bytes = int(args.budget_gib * BYTES_PER_GIB)
    block_signal = extract_signal(signal_payload)

    candidates: list[PlannedBlock] = []
    signal_by_key: dict[str, dict[str, Any]] = {}
    for group in index.groups:
        if group.layer is None and args.include_global == "none":
            continue
        if group.layer is None and args.include_global != "all" and group.component != args.include_global:
            continue
        signal = block_signal.get(group.key, {})
        measured_score = float_value(signal, "score_sum")
        signal_score = measured_score
        if group.component == "norm" and args.include_norms:
            signal_score = max(signal_score, args.norm_score)
        elif signal_score <= 0.0 and args.drop_unsignaled:
            continue
        rank_score = signal_score / (bytes_to_gib(group.n_bytes) ** args.byte_penalty) if group.n_bytes > 0 else 0.0
        signal_by_key[group.key] = {
            "measured_score_sum": measured_score,
            "score_sum": signal_score,
            "rank_score": rank_score,
            "max_abs_score": float_value(signal, "max_abs_score"),
            "row_count": int(signal.get("row_count", 0)) if isinstance(signal, dict) else 0,
            "tensor_names": signal.get("tensor_names", []) if isinstance(signal, dict) else [],
            "top_rows": signal.get("top_rows", []) if isinstance(signal, dict) else [],
        }
        candidates.append(
            PlannedBlock(
                key=group.key,
                layer=group.layer,
                component=group.component,
                n_tensors=group.n_tensors,
                n_bytes=group.n_bytes,
                score=rank_score,
            )
        )

    selected: list[PlannedBlock] = []
    selected_keys: set[str] = set()
    used_bytes = 0

    def select(block: PlannedBlock) -> None:
        nonlocal used_bytes
        selected.append(block)
        selected_keys.add(block.key)
        used_bytes += block.n_bytes

    if args.include_norms:
        for block in sorted([item for item in candidates if item.component == "norm"], key=lambda item: (item.layer or -1, item.key)):
            if used_bytes + block.n_bytes > budget_bytes:
                fail("norm pages alone exceed the active-byte budget")
            select(block)

    ranked = sorted(
        [item for item in candidates if item.key not in selected_keys],
        key=lambda item: (
            item.score,
            float_value(signal_by_key.get(item.key, {}), "score_sum"),
            -item.n_bytes,
            item.key,
        ),
        reverse=True,
    )
    for block in ranked:
        if args.drop_unsignaled and float_value(signal_by_key.get(block.key, {}), "score_sum") <= 0.0:
            continue
        if used_bytes + block.n_bytes <= budget_bytes:
            select(block)

    pages = make_pages(model_path, selected, model_bytes, args.max_tensor_names_per_page)
    stage_buffer_bytes = int(args.stage_buffer_gib * BYTES_PER_GIB)
    stages = make_stages(pages, stage_buffer_bytes, args.pcie_gbps)
    active_percent_model = 100.0 * used_bytes / model_bytes if model_bytes > 0 else 0.0
    reference_dense_gib = dense_weight_gib(args.reference_params_b, args.reference_quant_bpw)
    active_percent_reference = bytes_to_gib(used_bytes) / reference_dense_gib * 100.0 if reference_dense_gib > 0 else 0.0
    total_transfer = sum(stage.transfer_ms for stage in stages)
    max_stage_bytes = max((stage.n_bytes for stage in stages), default=0)
    components = {
        component: {"pages": count, "bytes": n_bytes}
        for component, (count, n_bytes) in component_summary(selected).items()
    }
    oracle_blocks = [page.block_key for page in pages]
    page_payloads = [asdict(page) for page in pages]
    for page in page_payloads:
        signal = signal_by_key.get(str(page["block_key"]), {})
        page["signal_score_sum"] = float_value(signal, "score_sum")
        page["signal_rank_score"] = float_value(signal, "rank_score")
        page["signal_max_abs_score"] = float_value(signal, "max_abs_score")
        page["signal_row_count"] = int(signal.get("row_count", 0)) if isinstance(signal, dict) else 0
        page["signal_tensor_names"] = signal.get("tensor_names", []) if isinstance(signal, dict) else []
        page["signal_top_rows"] = signal.get("top_rows", []) if isinstance(signal, dict) else []

    return {
        "schema": "sage-oracle-page-ledger-v0",
        "status": "plan_only_not_executed_signal_aware",
        "selection_policy": {
            "name": "measured_real_activation_signal_rank",
            "signal_json": str(signal_path.resolve()),
            "byte_penalty": args.byte_penalty,
            "include_norms": args.include_norms,
            "drop_unsignaled": args.drop_unsignaled,
            "norm_score": args.norm_score,
            "measured_signal_blocks": len(block_signal),
            "selected_measured_signal_blocks": sum(
                1 for block in selected if float_value(signal_by_key.get(block.key, {}), "measured_score_sum") > 0.0
            ),
        },
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "tensor_bytes": model_bytes,
            "tensor_gib": bytes_to_gib(model_bytes),
            "layer_count": index.layer_count,
            "metadata": index.metadata,
        },
        "reference_100b": {
            "params_b": args.reference_params_b,
            "quant_bpw": args.reference_quant_bpw,
            "dense_weight_gib": reference_dense_gib,
            "active_percent_of_reference": active_percent_reference,
        },
        "budget": {
            "target_tps": args.target_tps,
            "budget_gib": args.budget_gib,
            "budget_bytes": budget_bytes,
            "stage_buffer_gib": args.stage_buffer_gib,
            "stage_buffer_bytes": stage_buffer_bytes,
            "pcie_gbps": args.pcie_gbps,
            "max_active_percent_7tps": args.max_active_percent_7tps,
            "max_active_percent_10tps": args.max_active_percent_10tps,
            "status": plan_status(
                active_percent_reference,
                args.target_tps,
                args.max_active_percent_7tps,
                args.max_active_percent_10tps,
            ),
        },
        "summary": {
            "selected_pages": len(pages),
            "selected_bytes": used_bytes,
            "selected_gib": bytes_to_gib(used_bytes),
            "active_percent_of_model": active_percent_model,
            "active_percent_of_reference_100b": active_percent_reference,
            "stage_count": len(stages),
            "max_stage_bytes": max_stage_bytes,
            "max_stage_gib": bytes_to_gib(max_stage_bytes),
            "estimated_transfer_ms": total_transfer,
            "exact_fallback_bytes": model_bytes,
            "exact_fallback_gib": bytes_to_gib(model_bytes),
            "component_bytes": components,
            "selected_signal_score_sum": sum(float_value(signal_by_key.get(block.key, {}), "score_sum") for block in selected),
        },
        "runtime_ledger_template": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_signal_aware_page_plan",
            "oracle_active_bytes": used_bytes,
            "oracle_active_percent_of_model": active_percent_model,
            "oracle_active_percent_of_reference_100b": active_percent_reference,
            "oracle_blocks": oracle_blocks,
            "gpu_staged_bytes": max_stage_bytes,
            "host_pinned_bytes": used_bytes,
            "pcie_transfer_ms": total_transfer,
            "pcie_transfer_status": "estimated_not_measured",
            "kv_byte_status": "not_implemented",
        },
        "exact_fallback_ledger_template": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "exact_dense_fallback",
            "oracle_active_bytes": model_bytes,
            "oracle_active_percent_of_model": 100.0,
            "oracle_blocks": ["dense_model"],
            "pcie_transfer_status": "not_estimated",
        },
        "pages": page_payloads,
        "stages": [asdict(stage) for stage in stages],
    }


def print_markdown(payload: dict[str, Any], top_pages: int) -> None:
    summary = payload["summary"]
    budget = payload["budget"]
    policy = payload["selection_policy"]
    print("# SAGE Signal-Aware Sparse Oracle Page Ledger")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- Policy: `{policy['name']}`")
    print(f"- Selected pages: `{summary['selected_pages']}`")
    print(f"- Selected bytes: `{summary['selected_gib']:.3f} GiB`")
    print(f"- Active percent of 100B reference: `{summary['active_percent_of_reference_100b']:.2f}%`")
    print(f"- Budget status: `{budget['status']}`")
    print(f"- Measured signal blocks: `{policy['measured_signal_blocks']}`")
    print(f"- Selected measured signal blocks: `{policy['selected_measured_signal_blocks']}`")
    print()
    print("## Top Selected Signal Pages")
    print()
    print("| Page | Block | Component | Bytes | Signal |")
    print("| ---: | --- | --- | ---: | ---: |")
    pages = sorted(payload["pages"], key=lambda item: item.get("signal_score_sum", 0.0), reverse=True)
    for page in pages[:top_pages]:
        print(
            f"| {page['page_id']} | {page['block_key']} | {page['component']} | "
            f"{bytes_to_gib(page['n_bytes']):.3f} GiB | {page['signal_score_sum']:.3f} |"
        )
    print()
    print("## Stages")
    print()
    print("| Stage | Buffer | Pages | Bytes | Transfer |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(f"| {stage['stage_index']} | {stage['buffer']} | {stage['n_pages']} | {bytes_to_gib(stage['n_bytes']):.3f} GiB | {stage['transfer_ms']:.2f} ms |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a signal-aware sparse oracle page ledger.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--signal-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-ffn-norm0-full.json",
    )
    parser.add_argument("--budget-gib", type=float, default=1.18)
    parser.add_argument("--stage-buffer-gib", type=float, default=0.50)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--target-tps", type=float, default=10.0)
    parser.add_argument("--reference-params-b", type=float, default=100.0)
    parser.add_argument("--reference-quant-bpw", type=float, default=2.0)
    parser.add_argument("--max-active-percent-7tps", type=float, default=10.0)
    parser.add_argument("--max-active-percent-10tps", type=float, default=5.07)
    parser.add_argument("--byte-penalty", type=float, default=0.0, help="0 ranks by signal sum; 1 ranks by signal density")
    parser.add_argument("--include-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-score", type=float, default=1.0e-6)
    parser.add_argument("--drop-unsignaled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-global", choices=["none", "embedding", "output", "all"], default="none")
    parser.add_argument("--max-tensor-names-per-page", type=int, default=8)
    parser.add_argument("--top-pages", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.budget_gib <= 0:
        parser.error("--budget-gib must be positive")
    if args.stage_buffer_gib <= 0:
        parser.error("--stage-buffer-gib must be positive")
    if args.target_tps <= 0:
        parser.error("--target-tps must be positive")
    if args.byte_penalty < 0.0:
        parser.error("--byte-penalty must be non-negative")
    if args.max_tensor_names_per_page < 0:
        parser.error("--max-tensor-names-per-page must be non-negative")

    payload = make_signal_plan(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_markdown(payload, args.top_pages)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
