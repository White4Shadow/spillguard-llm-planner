#!/usr/bin/env python3
"""
Create a SAGE sparse-oracle page ledger from a GGUF block plan.

This is the bridge between "we selected active blocks" and "a runtime can prove
which giant-model bytes it touched." It does not execute the model. It emits the
page/stage ledger that the future CUDA/llama.cpp pager must match.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_block_plan import PlannedBlock, component_summary, make_plan
from sage_gguf_blocks import parse_gguf


BYTES_PER_GIB = 1024**3


@dataclass
class PageEntry:
    page_id: int
    block_key: str
    layer: int | None
    component: str
    n_tensors: int
    n_bytes: int
    active_percent_of_model: float
    tensor_names: list[str]


@dataclass
class StageEntry:
    stage_index: int
    buffer: str
    page_ids: list[int]
    n_pages: int
    n_bytes: int
    transfer_ms: float


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def bytes_to_gib(n_bytes: int | float) -> float:
    return float(n_bytes) / BYTES_PER_GIB


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def transfer_ms(n_bytes: int, pcie_gbps: float) -> float:
    if pcie_gbps <= 0:
        fail("--pcie-gbps must be positive")
    gib_per_second = pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)
    return bytes_to_gib(n_bytes) / gib_per_second * 1000.0


def tensor_names_for_blocks(model_path: Path, selected: list[PlannedBlock], max_names_per_page: int) -> dict[str, list[str]]:
    index = parse_gguf(model_path)
    selected_keys = {block.key for block in selected}
    names: dict[str, list[str]] = {key: [] for key in selected_keys}
    for tensor in index.tensors:
        key = f"blk.{tensor.layer}.{tensor.component}" if tensor.layer is not None else f"global.{tensor.component}"
        if key in names and len(names[key]) < max_names_per_page:
            names[key].append(tensor.name)
    return names


def make_pages(model_path: Path, selected: list[PlannedBlock], model_bytes: int, max_names_per_page: int) -> list[PageEntry]:
    tensor_names = tensor_names_for_blocks(model_path, selected, max_names_per_page)
    pages: list[PageEntry] = []
    for page_id, block in enumerate(sorted(selected, key=lambda item: (999999 if item.layer is None else item.layer, item.component, item.key))):
        pages.append(
            PageEntry(
                page_id=page_id,
                block_key=block.key,
                layer=block.layer,
                component=block.component,
                n_tensors=block.n_tensors,
                n_bytes=block.n_bytes,
                active_percent_of_model=100.0 * block.n_bytes / model_bytes if model_bytes > 0 else 0.0,
                tensor_names=tensor_names.get(block.key, []),
            )
        )
    return pages


def make_stages(pages: list[PageEntry], stage_buffer_bytes: int, pcie_gbps: float) -> list[StageEntry]:
    if stage_buffer_bytes <= 0:
        fail("--stage-buffer-gib must be positive")
    stages: list[StageEntry] = []
    current_ids: list[int] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current_ids, current_bytes
        if not current_ids:
            return
        stage_index = len(stages)
        stages.append(
            StageEntry(
                stage_index=stage_index,
                buffer="A" if stage_index % 2 == 0 else "B",
                page_ids=current_ids,
                n_pages=len(current_ids),
                n_bytes=current_bytes,
                transfer_ms=transfer_ms(current_bytes, pcie_gbps),
            )
        )
        current_ids = []
        current_bytes = 0

    for page in pages:
        if page.n_bytes > stage_buffer_bytes:
            flush()
            stage_index = len(stages)
            stages.append(
                StageEntry(
                    stage_index=stage_index,
                    buffer="A" if stage_index % 2 == 0 else "B",
                    page_ids=[page.page_id],
                    n_pages=1,
                    n_bytes=page.n_bytes,
                    transfer_ms=transfer_ms(page.n_bytes, pcie_gbps),
                )
            )
            continue
        if current_ids and current_bytes + page.n_bytes > stage_buffer_bytes:
            flush()
        current_ids.append(page.page_id)
        current_bytes += page.n_bytes
    flush()
    return stages


def plan_status(active_percent: float, target_tps: float, max_active_7tps: float, max_active_10tps: float) -> str:
    if target_tps >= 10.0:
        return "within_10tps_budget" if active_percent <= max_active_10tps else "exceeds_10tps_budget"
    if target_tps >= 7.0:
        return "within_7tps_budget" if active_percent <= max_active_7tps else "exceeds_7tps_budget"
    return "informational"


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")

    plan = make_plan(
        model=model_path,
        budget_gib=args.budget_gib,
        policy=args.policy,
        boundary_layers=args.boundary_layers,
        global_mode=args.include_global,
        ffn_min_share=args.ffn_min_share,
        attention_max_share=args.attention_max_share,
    )
    index = parse_gguf(model_path)
    model_bytes = int(index.total_tensor_bytes)
    pages = make_pages(model_path, plan.selected, model_bytes, args.max_tensor_names_per_page)
    stage_buffer_bytes = int(args.stage_buffer_gib * BYTES_PER_GIB)
    stages = make_stages(pages, stage_buffer_bytes, args.pcie_gbps)
    active_bytes = sum(page.n_bytes for page in pages)
    active_percent_model = 100.0 * active_bytes / model_bytes if model_bytes > 0 else 0.0
    reference_dense_gib = dense_weight_gib(args.reference_params_b, args.reference_quant_bpw)
    active_percent_reference = bytes_to_gib(active_bytes) / reference_dense_gib * 100.0 if reference_dense_gib > 0 else 0.0
    total_transfer = sum(stage.transfer_ms for stage in stages)
    max_stage_bytes = max((stage.n_bytes for stage in stages), default=0)

    components = {
        component: {"pages": count, "bytes": n_bytes}
        for component, (count, n_bytes) in component_summary(plan.selected).items()
    }
    oracle_blocks = [page.block_key for page in pages]
    return {
        "schema": "sage-oracle-page-ledger-v0",
        "status": "plan_only_not_executed",
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
            "budget_bytes": int(args.budget_gib * BYTES_PER_GIB),
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
            "selected_bytes": active_bytes,
            "selected_gib": bytes_to_gib(active_bytes),
            "active_percent_of_model": active_percent_model,
            "active_percent_of_reference_100b": active_percent_reference,
            "stage_count": len(stages),
            "max_stage_bytes": max_stage_bytes,
            "max_stage_gib": bytes_to_gib(max_stage_bytes),
            "estimated_transfer_ms": total_transfer,
            "exact_fallback_bytes": model_bytes,
            "exact_fallback_gib": bytes_to_gib(model_bytes),
            "component_bytes": components,
        },
        "runtime_ledger_template": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_plan",
            "oracle_active_bytes": active_bytes,
            "oracle_active_percent_of_model": active_percent_model,
            "oracle_active_percent_of_reference_100b": active_percent_reference,
            "oracle_blocks": oracle_blocks,
            "gpu_staged_bytes": max_stage_bytes,
            "host_pinned_bytes": active_bytes,
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
        "pages": [asdict(page) for page in pages],
        "stages": [asdict(stage) for stage in stages],
    }


def print_markdown(payload: dict[str, Any], top_pages: int) -> None:
    summary = payload["summary"]
    budget = payload["budget"]
    print("# SAGE Sparse Oracle Page Ledger")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- Selected pages: `{summary['selected_pages']}`")
    print(f"- Selected bytes: `{summary['selected_gib']:.3f} GiB`")
    print(f"- Active percent of local model: `{summary['active_percent_of_model']:.2f}%`")
    print(f"- Active percent of 100B reference: `{summary['active_percent_of_reference_100b']:.2f}%`")
    print(f"- Stage count: `{summary['stage_count']}`")
    print(f"- Estimated transfer: `{summary['estimated_transfer_ms']:.2f} ms`")
    print(f"- Budget status: `{budget['status']}`")
    print()
    print("## Components")
    print()
    print("| Component | Pages | Bytes |")
    print("| --- | ---: | ---: |")
    for component, item in sorted(summary["component_bytes"].items(), key=lambda kv: kv[1]["bytes"], reverse=True):
        print(f"| {component} | {item['pages']} | {bytes_to_gib(item['bytes']):.3f} GiB |")
    print()
    print("## First Pages")
    print()
    print("| Page | Block | Component | Layer | Bytes |")
    print("| ---: | --- | --- | ---: | ---: |")
    for page in payload["pages"][:top_pages]:
        layer = "" if page["layer"] is None else page["layer"]
        print(f"| {page['page_id']} | {page['block_key']} | {page['component']} | {layer} | {bytes_to_gib(page['n_bytes']):.3f} GiB |")
    print()
    print("## Stages")
    print()
    print("| Stage | Buffer | Pages | Bytes | Transfer |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(f"| {stage['stage_index']} | {stage['buffer']} | {stage['n_pages']} | {bytes_to_gib(stage['n_bytes']):.3f} GiB | {stage['transfer_ms']:.2f} ms |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sparse oracle page ledger from a GGUF block plan.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-gib", type=float, default=2.33, help="active oracle byte budget for the local GGUF")
    parser.add_argument("--policy", choices=["attention-first", "balanced", "boundary", "ffn-first"], default="balanced")
    parser.add_argument("--boundary-layers", type=int, default=4)
    parser.add_argument("--include-global", choices=["none", "embedding", "output", "all"], default="none")
    parser.add_argument("--ffn-min-share", type=float, default=0.0)
    parser.add_argument("--attention-max-share", type=float, default=1.0)
    parser.add_argument("--stage-buffer-gib", type=float, default=0.75)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--reference-params-b", type=float, default=100.0)
    parser.add_argument("--reference-quant-bpw", type=float, default=2.0)
    parser.add_argument("--max-active-percent-7tps", type=float, default=10.0)
    parser.add_argument("--max-active-percent-10tps", type=float, default=5.0)
    parser.add_argument("--max-tensor-names-per-page", type=int, default=8)
    parser.add_argument("--top-pages", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.budget_gib <= 0:
        parser.error("--budget-gib must be positive")
    if args.stage_buffer_gib <= 0:
        parser.error("--stage-buffer-gib must be positive")
    if args.target_tps <= 0:
        parser.error("--target-tps must be positive")
    if args.max_tensor_names_per_page < 0:
        parser.error("--max-tensor-names-per-page must be non-negative")

    payload = make_payload(args)
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
