#!/usr/bin/env python3
"""
Execute a bounded SAGE sparse-oracle page staging smoke.

This does not run a sparse transformer forward pass. It proves the next lower
level of the pager contract: the selected GGUF block pages can be resolved to
real tensor byte ranges and streamed through fixed-size staging buffers with a
measured byte/latency ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from sage_gguf_blocks import (
    DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    TensorInfo,
    parse_gguf,
    read_string,
    read_u32,
    read_u64,
    read_value,
)


BYTES_PER_GIB = 1024**3


@dataclass
class StagedPage:
    page_id: int
    block_key: str
    planned_bytes: int
    staged_bytes: int
    n_tensors: int
    read_calls: int
    elapsed_ms: float
    crc32: str


@dataclass
class StagedStage:
    stage_index: int
    buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    read_calls: int
    elapsed_ms: float
    throughput_gib_s: float
    page_ids: list[int]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def bytes_to_gib(n_bytes: int | float) -> float:
    return float(n_bytes) / BYTES_PER_GIB


def align_offset(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


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


def gguf_data_start(path: Path) -> int:
    with path.open("rb") as handle:
        magic = read_u32(handle)
        if magic != GGUF_MAGIC:
            fail(f"{path} is not a GGUF file")
        _version = read_u32(handle)
        tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)
        alignment = DEFAULT_ALIGNMENT

        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            value = read_value(handle, value_type, keep_arrays=False)
            if key == "general.alignment":
                alignment = int(value)

        for _ in range(tensor_count):
            _name = read_string(handle)
            n_dims = read_u32(handle)
            for _ in range(n_dims):
                read_u64(handle)
            read_u32(handle)
            read_u64(handle)

        return align_offset(handle.tell(), alignment)


def block_key_for_tensor(tensor: TensorInfo) -> str:
    return f"blk.{tensor.layer}.{tensor.component}" if tensor.layer is not None else f"global.{tensor.component}"


def group_tensors_by_block(tensors: list[TensorInfo]) -> dict[str, list[TensorInfo]]:
    groups: dict[str, list[TensorInfo]] = {}
    for tensor in tensors:
        groups.setdefault(block_key_for_tensor(tensor), []).append(tensor)
    for items in groups.values():
        items.sort(key=lambda tensor: tensor.offset)
    return groups


def resolve_model_path(args: argparse.Namespace, ledger: dict[str, Any]) -> Path:
    if args.model:
        return Path(args.model)
    model = ledger.get("model", {})
    if not isinstance(model, dict):
        fail("page ledger is missing model metadata; pass --model")
    path = str(model.get("path", ""))
    if not path:
        fail("page ledger is missing model.path; pass --model")
    return Path(path)


def selected_stages(ledger: dict[str, Any], limit_stages: int, max_bytes: int) -> list[dict[str, Any]]:
    raw_stages = ledger.get("stages", [])
    if not isinstance(raw_stages, list):
        fail("page ledger stages must be a list")

    selected: list[dict[str, Any]] = []
    staged_bytes = 0
    for stage in raw_stages:
        if not isinstance(stage, dict):
            continue
        if limit_stages > 0 and len(selected) >= limit_stages:
            break
        n_bytes = int(stage.get("n_bytes", 0))
        if max_bytes > 0 and selected and staged_bytes + n_bytes > max_bytes:
            break
        if max_bytes > 0 and not selected and n_bytes > max_bytes:
            fail("first planned stage exceeds --max-bytes; raise the limit or lower the page plan stage size")
        selected.append(stage)
        staged_bytes += n_bytes
    return selected


def read_tensor_into_buffer(
    handle: BinaryIO,
    tensor: TensorInfo,
    data_start: int,
    buffer_view: memoryview,
    write_offset: int,
) -> int:
    target = buffer_view[write_offset : write_offset + tensor.n_bytes]
    handle.seek(data_start + tensor.offset)
    n_read = handle.readinto(target)
    if n_read != tensor.n_bytes:
        fail(f"short read for tensor {tensor.name}: {n_read} != {tensor.n_bytes}")
    return n_read


def stage_pages(
    *,
    model_path: Path,
    ledger: dict[str, Any],
    stages: list[dict[str, Any]],
    buffer_count: int,
    stage_buffer_bytes: int,
) -> tuple[list[StagedStage], list[StagedPage]]:
    index = parse_gguf(model_path)
    data_start = gguf_data_start(model_path)
    file_size = model_path.stat().st_size
    groups = group_tensors_by_block(index.tensors)
    pages = ledger.get("pages", [])
    if not isinstance(pages, list):
        fail("page ledger pages must be a list")
    page_by_id = {
        int(page.get("page_id")): page
        for page in pages
        if isinstance(page, dict) and page.get("page_id") is not None
    }

    max_tensor_end = max((data_start + tensor.offset + tensor.n_bytes for tensor in index.tensors), default=data_start)
    if max_tensor_end > file_size:
        fail(f"tensor table points past file end: {max_tensor_end} > {file_size}")

    buffers = [bytearray(stage_buffer_bytes) for _ in range(buffer_count)]
    buffer_views = [memoryview(buffer) for buffer in buffers]
    staged_stages: list[StagedStage] = []
    staged_pages: list[StagedPage] = []

    with model_path.open("rb", buffering=0) as handle:
        for planned_stage in stages:
            stage_index = int(planned_stage.get("stage_index", len(staged_stages)))
            buffer_index = stage_index % buffer_count
            buffer_view = buffer_views[buffer_index]
            page_ids = [int(page_id) for page_id in planned_stage.get("page_ids", [])]
            planned_stage_bytes = int(planned_stage.get("n_bytes", 0))
            if planned_stage_bytes > stage_buffer_bytes:
                fail(
                    f"stage {stage_index} requires {planned_stage_bytes} bytes, "
                    f"but stage buffer holds {stage_buffer_bytes}"
                )

            stage_start = time.perf_counter()
            stage_offset = 0
            stage_reads = 0
            stage_page_records: list[StagedPage] = []

            for page_id in page_ids:
                page = page_by_id.get(page_id)
                if not page:
                    fail(f"stage {stage_index} references missing page {page_id}")
                block_key = str(page.get("block_key", ""))
                tensors = groups.get(block_key, [])
                if not tensors:
                    fail(f"no GGUF tensors found for block page {block_key}")
                planned_page_bytes = int(page.get("n_bytes", 0))
                actual_page_bytes = sum(tensor.n_bytes for tensor in tensors)
                if planned_page_bytes != actual_page_bytes:
                    fail(
                        f"page {page_id} byte mismatch for {block_key}: "
                        f"ledger={planned_page_bytes}, tensors={actual_page_bytes}"
                    )
                if stage_offset + actual_page_bytes > stage_buffer_bytes:
                    fail(f"stage {stage_index} overflows staging buffer at page {page_id}")

                page_start = time.perf_counter()
                page_crc = 0
                page_reads = 0
                page_offset = stage_offset
                for tensor in tensors:
                    read_tensor_into_buffer(handle, tensor, data_start, buffer_view, stage_offset)
                    segment = buffer_view[stage_offset : stage_offset + tensor.n_bytes]
                    page_crc = zlib.crc32(segment, page_crc)
                    stage_offset += tensor.n_bytes
                    page_reads += 1
                page_elapsed_ms = (time.perf_counter() - page_start) * 1000.0
                page_record = StagedPage(
                    page_id=page_id,
                    block_key=block_key,
                    planned_bytes=planned_page_bytes,
                    staged_bytes=stage_offset - page_offset,
                    n_tensors=len(tensors),
                    read_calls=page_reads,
                    elapsed_ms=page_elapsed_ms,
                    crc32=f"{page_crc & 0xFFFFFFFF:08x}",
                )
                stage_page_records.append(page_record)
                stage_reads += page_reads

            stage_elapsed_ms = (time.perf_counter() - stage_start) * 1000.0
            staged_bytes = stage_offset
            throughput = bytes_to_gib(staged_bytes) / (stage_elapsed_ms / 1000.0) if stage_elapsed_ms > 0 else 0.0
            staged_stages.append(
                StagedStage(
                    stage_index=stage_index,
                    buffer=str(planned_stage.get("buffer", chr(ord("A") + buffer_index))),
                    planned_bytes=planned_stage_bytes,
                    staged_bytes=staged_bytes,
                    n_pages=len(page_ids),
                    read_calls=stage_reads,
                    elapsed_ms=stage_elapsed_ms,
                    throughput_gib_s=throughput,
                    page_ids=page_ids,
                )
            )
            staged_pages.extend(stage_page_records)

    for view in buffer_views:
        view.release()

    return staged_stages, staged_pages


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = Path(args.page_ledger)
    ledger = load_json(ledger_path)
    if ledger.get("schema") != "sage-oracle-page-ledger-v0":
        fail("expected sage-oracle-page-ledger-v0 input")

    model_path = resolve_model_path(args, ledger)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")

    budget = ledger.get("budget", {})
    if not isinstance(budget, dict):
        fail("page ledger is missing budget object")
    stage_buffer_bytes = int(args.stage_buffer_gib * BYTES_PER_GIB) if args.stage_buffer_gib > 0 else int(budget.get("stage_buffer_bytes", 0))
    if stage_buffer_bytes <= 0:
        fail("stage buffer bytes must be positive")
    if args.buffer_count <= 0:
        fail("--buffer-count must be positive")
    max_bytes = int(args.max_gib * BYTES_PER_GIB) if args.max_gib > 0 else 0
    stages = selected_stages(ledger, args.limit_stages, max_bytes)
    if not stages:
        fail("no stages selected")

    started = time.perf_counter()
    staged_stages, staged_pages = stage_pages(
        model_path=model_path,
        ledger=ledger,
        stages=stages,
        buffer_count=args.buffer_count,
        stage_buffer_bytes=stage_buffer_bytes,
    )
    total_elapsed_ms = (time.perf_counter() - started) * 1000.0

    staged_bytes = sum(stage.staged_bytes for stage in staged_stages)
    planned_bytes = sum(stage.planned_bytes for stage in staged_stages)
    max_stage_bytes = max((stage.staged_bytes for stage in staged_stages), default=0)
    staging_elapsed_ms = sum(stage.elapsed_ms for stage in staged_stages)
    all_stage_bytes_match = all(stage.planned_bytes == stage.staged_bytes for stage in staged_stages)
    all_page_bytes_match = all(page.planned_bytes == page.staged_bytes for page in staged_pages)
    total_throughput = bytes_to_gib(staged_bytes) / (total_elapsed_ms / 1000.0) if total_elapsed_ms > 0 else 0.0
    staging_throughput = bytes_to_gib(staged_bytes) / (staging_elapsed_ms / 1000.0) if staging_elapsed_ms > 0 else 0.0

    return {
        "schema": "sage-oracle-page-staging-v0",
        "status": "measured_host_staging_not_cuda",
        "source_page_ledger": str(ledger_path.resolve()),
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "file_bytes": model_path.stat().st_size,
        },
        "limits": {
            "limit_stages": args.limit_stages,
            "max_gib": args.max_gib,
            "buffer_count": args.buffer_count,
            "stage_buffer_bytes": stage_buffer_bytes,
            "stage_buffer_gib": bytes_to_gib(stage_buffer_bytes),
            "allocated_buffer_bytes": stage_buffer_bytes * args.buffer_count,
            "allocated_buffer_gib": bytes_to_gib(stage_buffer_bytes * args.buffer_count),
        },
        "summary": {
            "stages_staged": len(staged_stages),
            "pages_staged": len(staged_pages),
            "planned_bytes": planned_bytes,
            "staged_bytes": staged_bytes,
            "staged_gib": bytes_to_gib(staged_bytes),
            "max_live_buffer_bytes": max_stage_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_stage_bytes),
            "read_calls": sum(stage.read_calls for stage in staged_stages),
            "elapsed_ms": total_elapsed_ms,
            "staging_elapsed_ms": staging_elapsed_ms,
            "total_throughput_gib_s": total_throughput,
            "staging_throughput_gib_s": staging_throughput,
            "stage_byte_match": all_stage_bytes_match,
            "page_byte_match": all_page_bytes_match,
            "byte_budget_respected": max_stage_bytes <= stage_buffer_bytes,
            "pcie_transfer_status": "not_measured_cpu_file_to_host_staging_only",
            "cuda_execution_status": "not_implemented",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_staging_smoke",
            "oracle_active_bytes": staged_bytes,
            "oracle_blocks": [page.block_key for page in staged_pages],
            "gpu_staged_bytes": max_stage_bytes,
            "host_staged_bytes": staged_bytes,
            "pcie_transfer_status": "not_measured_cpu_file_to_host_staging_only",
        },
        "stages": [asdict(stage) for stage in staged_stages],
        "pages": [asdict(page) for page in staged_pages],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    limits = payload["limits"]
    print("# SAGE Oracle Page Staging Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- Stages staged: `{summary['stages_staged']}`")
    print(f"- Pages staged: `{summary['pages_staged']}`")
    print(f"- Bytes staged: `{summary['staged_gib']:.3f} GiB`")
    print(f"- Max live buffer: `{summary['max_live_buffer_gib']:.3f} GiB` / `{limits['stage_buffer_gib']:.3f} GiB`")
    print(f"- Elapsed: `{summary['elapsed_ms']:.2f} ms`")
    print(f"- Staging elapsed: `{summary['staging_elapsed_ms']:.2f} ms`")
    print(f"- Staging throughput: `{summary['staging_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Total throughput including setup: `{summary['total_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Byte match: `stage={summary['stage_byte_match']}`, `page={summary['page_byte_match']}`")
    print(f"- CUDA execution: `{summary['cuda_execution_status']}`")
    print()
    print("## Staged Stages")
    print()
    print("| Stage | Buffer | Pages | Bytes | Elapsed | Throughput |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(
            f"| {stage['stage_index']} | {stage['buffer']} | {stage['n_pages']} | "
            f"{bytes_to_gib(stage['staged_bytes']):.3f} GiB | {stage['elapsed_ms']:.2f} ms | "
            f"{stage['throughput_gib_s']:.2f} GiB/s |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded GGUF sparse-oracle page staging smoke.")
    parser.add_argument("--page-ledger", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--model", default="", help="override model path from the page ledger")
    parser.add_argument("--limit-stages", type=int, default=1, help="number of planned stages to execute; 0 means all")
    parser.add_argument("--max-gib", type=float, default=0.0, help="optional cap on selected planned stage bytes")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="override stage buffer size; default uses page ledger")
    parser.add_argument("--buffer-count", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.limit_stages < 0:
        parser.error("--limit-stages must be non-negative")
    if args.max_gib < 0:
        parser.error("--max-gib must be non-negative")
    if args.stage_buffer_gib < 0:
        parser.error("--stage-buffer-gib must be non-negative")
    if args.buffer_count <= 0:
        parser.error("--buffer-count must be positive")

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
