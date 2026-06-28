#!/usr/bin/env python3
"""
Measure CUDA H2D/kernel overlap for SAGE sparse-oracle page staging.

This smoke is deliberately narrow. It does not implement transformer math.
It answers one transport question for the active-byte oracle path: can the
selected GGUF page bytes be copied to the GPU while a previous staged page
buffer is being consumed by a CUDA kernel?
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf
from sage_oracle_cuda_kernel_smoke import CudaDriver, KERNEL_SOURCE, NvrtcRuntime
from sage_oracle_cuda_staging import (
    CUDA_MEMCPY_DEVICE_TO_HOST,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    CudaRuntime,
    make_host_view,
)
from sage_oracle_pager_staging import (
    BYTES_PER_GIB,
    bytes_to_gib,
    gguf_data_start,
    group_tensors_by_block,
    load_json,
    resolve_model_path,
    selected_stages,
)


CUDA_SUCCESS = 0


@dataclass
class HostStage:
    stage_index: int
    buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    read_calls: int
    host_read_ms: float
    crc32: str
    page_ids: list[int]


@dataclass
class OverlapStage:
    stage_index: int
    host_buffer: str
    device_buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    read_calls: int
    host_read_ms: float
    h2d_ms: float
    kernel_ms: float
    kernel_grid: int
    kernel_block: int
    page_ids: list[int]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def bind_overlap_cuda(cuda: CudaRuntime) -> None:
    cuda.lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    cuda.lib.cudaStreamWaitEvent.restype = ctypes.c_int


def stream_wait_event(cuda: CudaRuntime, stream: ctypes.c_void_p, event: ctypes.c_void_p, what: str) -> None:
    cuda.check(cuda.lib.cudaStreamWaitEvent(stream, event, 0), what)


def event_record(cuda: CudaRuntime, event: ctypes.c_void_p, stream: ctypes.c_void_p, what: str) -> None:
    cuda.check(cuda.lib.cudaEventRecord(event, stream), what)


def event_sync(cuda: CudaRuntime, event: ctypes.c_void_p, what: str) -> None:
    cuda.check(cuda.lib.cudaEventSynchronize(event), what)


def event_elapsed_ms(cuda: CudaRuntime, start: ctypes.c_void_p, stop: ctypes.c_void_p, what: str) -> float:
    elapsed = ctypes.c_float()
    cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop), what)
    return float(elapsed.value)


def read_stage_into_pinned_host(
    *,
    handle: Any,
    host_view: memoryview,
    planned_stage: dict[str, Any],
    stage_index: int,
    data_start: int,
    groups: dict[str, Any],
    page_by_id: dict[int, dict[str, Any]],
) -> tuple[HostStage, int]:
    page_ids = [int(page_id) for page_id in planned_stage.get("page_ids", [])]
    planned_stage_bytes = int(planned_stage.get("n_bytes", 0))
    if planned_stage_bytes <= 0:
        fail(f"stage {stage_index} has no planned bytes")
    if planned_stage_bytes > len(host_view):
        fail(f"stage {stage_index} requires {planned_stage_bytes} bytes but host buffer has {len(host_view)}")

    stage_offset = 0
    stage_reads = 0
    stage_crc = 0
    started = time.perf_counter()

    for page_id in page_ids:
        page = page_by_id.get(page_id)
        if not page:
            fail(f"stage {stage_index} references missing page {page_id}")
        block_key = str(page.get("block_key", ""))
        tensors = groups.get(block_key, [])
        if not tensors:
            fail(f"no GGUF tensors found for block page {block_key}")
        planned_page_bytes = int(page.get("n_bytes", 0))
        actual_page_bytes = sum(int(tensor.n_bytes) for tensor in tensors)
        if planned_page_bytes != actual_page_bytes:
            fail(f"page {page_id} byte mismatch for {block_key}: ledger={planned_page_bytes}, tensors={actual_page_bytes}")
        if stage_offset + actual_page_bytes > len(host_view):
            fail(f"stage {stage_index} overflows host buffer at page {page_id}")

        for tensor in tensors:
            target = host_view[stage_offset : stage_offset + tensor.n_bytes]
            handle.seek(data_start + tensor.offset)
            n_read = handle.readinto(target)
            if n_read != tensor.n_bytes:
                fail(f"short read for tensor {tensor.name}: {n_read} != {tensor.n_bytes}")
            stage_crc = zlib.crc32(target, stage_crc)
            stage_offset += tensor.n_bytes
            stage_reads += 1

    host_read_ms = (time.perf_counter() - started) * 1000.0
    return (
        HostStage(
            stage_index=stage_index,
            buffer=str(planned_stage.get("buffer", f"H{stage_index}")),
            planned_bytes=planned_stage_bytes,
            staged_bytes=stage_offset,
            n_pages=len(page_ids),
            read_calls=stage_reads,
            host_read_ms=host_read_ms,
            crc32=f"{stage_crc & 0xFFFFFFFF:08x}",
            page_ids=page_ids,
        ),
        stage_offset,
    )


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
    if args.device_buffer_count < 2:
        fail("--device-buffer-count must be at least 2 for overlap")

    max_bytes = int(args.max_gib * BYTES_PER_GIB) if args.max_gib > 0 else 0
    stages = selected_stages(ledger, args.limit_stages, max_bytes)
    if len(stages) < 2:
        fail("at least two stages are required to measure overlap")

    index = parse_gguf(model_path)
    data_start = gguf_data_start(model_path)
    groups = group_tensors_by_block(index.tensors)
    pages = ledger.get("pages", [])
    if not isinstance(pages, list):
        fail("page ledger pages must be a list")
    page_by_id = {
        int(page.get("page_id")): page
        for page in pages
        if isinstance(page, dict) and page.get("page_id") is not None
    }

    cuda = CudaRuntime(args.cudart)
    bind_overlap_cuda(cuda)
    driver = CudaDriver()
    device_count = cuda.device_count()
    if args.device < 0 or args.device >= device_count:
        fail(f"--device must be between 0 and {device_count - 1}")
    cuda.set_device(args.device)
    free_before, total_vram = cuda.mem_info()
    driver.require_current_context()
    arch = args.arch or driver.compute_arch(args.device, real=True)
    nvrtc = NvrtcRuntime(args.nvrtc)
    module_image, compile_log, module_image_kind = nvrtc.compile_image(KERNEL_SOURCE, arch)
    module, function = driver.load_function(module_image, b"sage_page_byte_sum")

    host_ptrs: list[ctypes.c_void_p] = []
    host_views: list[memoryview] = []
    host_stages: list[HostStage] = []
    host_stage_bytes: list[int] = []
    device_ptrs: list[ctypes.c_void_p] = []
    transfer_stream = ctypes.c_void_p()
    compute_stream = ctypes.c_void_p()
    output_ptr = ctypes.c_void_p()
    h2d_start_events: list[ctypes.c_void_p] = []
    h2d_stop_events: list[ctypes.c_void_p] = []
    kernel_start_events: list[ctypes.c_void_p] = []
    kernel_stop_events: list[ctypes.c_void_p] = []
    gpu_start_event = ctypes.c_void_p()
    gpu_stop_event = ctypes.c_void_p()
    overlap_stages: list[OverlapStage] = []

    try:
        host_read_started = time.perf_counter()
        with model_path.open("rb", buffering=0) as handle:
            for index_in_plan, planned_stage in enumerate(stages):
                stage_index = int(planned_stage.get("stage_index", index_in_plan))
                planned_bytes = int(planned_stage.get("n_bytes", 0))
                if planned_bytes > stage_buffer_bytes:
                    fail(f"stage {stage_index} requires {planned_bytes} bytes, but stage buffer holds {stage_buffer_bytes}")
                host_ptr = cuda.host_alloc(planned_bytes)
                host_view = make_host_view(host_ptr, planned_bytes)
                host_stage, staged_bytes = read_stage_into_pinned_host(
                    handle=handle,
                    host_view=host_view,
                    planned_stage=planned_stage,
                    stage_index=stage_index,
                    data_start=data_start,
                    groups=groups,
                    page_by_id=page_by_id,
                )
                host_ptrs.append(host_ptr)
                host_views.append(host_view)
                host_stages.append(host_stage)
                host_stage_bytes.append(staged_bytes)
        host_read_elapsed_ms = (time.perf_counter() - host_read_started) * 1000.0

        max_stage_bytes = max(host_stage_bytes)
        for _ in range(args.device_buffer_count):
            device_ptrs.append(cuda.device_alloc(max_stage_bytes))
        transfer_stream = cuda.stream_create()
        compute_stream = cuda.stream_create()
        output_ptr = cuda.device_alloc(ctypes.sizeof(ctypes.c_uint64))
        cuda.memset_async(output_ptr, 0, ctypes.sizeof(ctypes.c_uint64), compute_stream)
        cuda.device_synchronize()

        for _ in host_stages:
            h2d_start_events.append(cuda.event_create())
            h2d_stop_events.append(cuda.event_create())
            kernel_start_events.append(cuda.event_create())
            kernel_stop_events.append(cuda.event_create())
        gpu_start_event = cuda.event_create()
        gpu_stop_event = cuda.event_create()

        gpu_host_start = time.perf_counter()
        event_record(cuda, gpu_start_event, transfer_stream, "cudaEventRecord(gpu-start)")
        for i, host_stage in enumerate(host_stages):
            device_buffer_index = i % args.device_buffer_count
            if i >= args.device_buffer_count:
                stream_wait_event(
                    cuda,
                    transfer_stream,
                    kernel_stop_events[i - args.device_buffer_count],
                    f"cudaStreamWaitEvent(reuse-buffer-{device_buffer_index})",
                )

            event_record(cuda, h2d_start_events[i], transfer_stream, f"cudaEventRecord(h2d-start-{i})")
            cuda.check(
                cuda.lib.cudaMemcpyAsync(
                    device_ptrs[device_buffer_index],
                    host_ptrs[i],
                    host_stage.staged_bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    transfer_stream,
                ),
                f"cudaMemcpyAsync(H2D stage {i})",
            )
            event_record(cuda, h2d_stop_events[i], transfer_stream, f"cudaEventRecord(h2d-stop-{i})")

            stream_wait_event(cuda, compute_stream, h2d_stop_events[i], f"cudaStreamWaitEvent(h2d-done-{i})")
            event_record(cuda, kernel_start_events[i], compute_stream, f"cudaEventRecord(kernel-start-{i})")
            block = args.block_size
            grid = min(args.max_grid, max(1, (host_stage.staged_bytes + block - 1) // block))
            driver.launch_byte_sum(
                function,
                device_ptrs[device_buffer_index],
                host_stage.staged_bytes,
                output_ptr,
                compute_stream,
                grid,
                block,
            )
            event_record(cuda, kernel_stop_events[i], compute_stream, f"cudaEventRecord(kernel-stop-{i})")
            overlap_stages.append(
                OverlapStage(
                    stage_index=host_stage.stage_index,
                    host_buffer=host_stage.buffer,
                    device_buffer=chr(ord("A") + device_buffer_index),
                    planned_bytes=host_stage.planned_bytes,
                    staged_bytes=host_stage.staged_bytes,
                    n_pages=host_stage.n_pages,
                    read_calls=host_stage.read_calls,
                    host_read_ms=host_stage.host_read_ms,
                    h2d_ms=0.0,
                    kernel_ms=0.0,
                    kernel_grid=grid,
                    kernel_block=block,
                    page_ids=host_stage.page_ids,
                )
            )
        event_record(cuda, gpu_stop_event, compute_stream, "cudaEventRecord(gpu-stop)")
        event_sync(cuda, gpu_stop_event, "cudaEventSynchronize(gpu-stop)")
        gpu_wall_ms = (time.perf_counter() - gpu_host_start) * 1000.0

        for i, stage in enumerate(overlap_stages):
            stage.h2d_ms = event_elapsed_ms(cuda, h2d_start_events[i], h2d_stop_events[i], f"cudaEventElapsedTime(h2d-{i})")
            stage.kernel_ms = event_elapsed_ms(
                cuda,
                kernel_start_events[i],
                kernel_stop_events[i],
                f"cudaEventElapsedTime(kernel-{i})",
            )
        overlapped_gpu_ms = event_elapsed_ms(cuda, gpu_start_event, gpu_stop_event, "cudaEventElapsedTime(gpu-window)")

        output_value = ctypes.c_uint64()
        cuda.check(
            cuda.lib.cudaMemcpy(
                ctypes.cast(ctypes.byref(output_value), ctypes.c_void_p),
                output_ptr,
                ctypes.sizeof(output_value),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H output)",
        )
        free_after, _total_after = cuda.mem_info()
    finally:
        if output_ptr and output_ptr.value:
            cuda.device_free(output_ptr)
        for event in [gpu_stop_event, gpu_start_event]:
            if event and event.value:
                cuda.event_destroy(event)
        for events in [kernel_stop_events, kernel_start_events, h2d_stop_events, h2d_start_events]:
            for event in events:
                cuda.event_destroy(event)
        if compute_stream and compute_stream.value:
            cuda.stream_destroy(compute_stream)
        if transfer_stream and transfer_stream.value:
            cuda.stream_destroy(transfer_stream)
        for ptr in device_ptrs:
            cuda.device_free(ptr)
        for view in host_views:
            view.release()
        for ptr in host_ptrs:
            cuda.host_free(ptr)
        driver.unload_module(module)

    staged_bytes = sum(stage.staged_bytes for stage in overlap_stages)
    planned_bytes = sum(stage.planned_bytes for stage in overlap_stages)
    h2d_ms = sum(stage.h2d_ms for stage in overlap_stages)
    kernel_ms = sum(stage.kernel_ms for stage in overlap_stages)
    host_read_ms = sum(stage.host_read_ms for stage in overlap_stages)
    sequential_gpu_ms = h2d_ms + kernel_ms
    overlap_savings_ms = sequential_gpu_ms - overlapped_gpu_ms
    overlap_savings_pct = (overlap_savings_ms / sequential_gpu_ms * 100.0) if sequential_gpu_ms > 0 else 0.0
    stage_byte_match = all(stage.planned_bytes == stage.staged_bytes for stage in overlap_stages)
    max_live_buffer_bytes = max((stage.staged_bytes for stage in overlap_stages), default=0)
    max_device_live_bytes = max_live_buffer_bytes * args.device_buffer_count
    output_nonzero = int(output_value.value) != 0

    return {
        "schema": "sage-oracle-page-cuda-overlap-smoke-v0",
        "status": "measured_cuda_double_buffer_overlap_touch_not_transformer",
        "source_page_ledger": str(ledger_path.resolve()),
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "file_bytes": model_path.stat().st_size,
        },
        "cuda": {
            "runtime_dll": str(cuda.path),
            "runtime_version": cuda.runtime_version(),
            "nvrtc_dll": str(nvrtc.path),
            "device": args.device,
            "device_count": device_count,
            "arch": arch,
            "module_image_kind": module_image_kind,
            "vram_total_bytes": total_vram,
            "vram_free_before_bytes": free_before,
            "vram_free_after_bytes": free_after,
        },
        "limits": {
            "stage_buffer_bytes": stage_buffer_bytes,
            "device_buffer_count": args.device_buffer_count,
            "max_grid": args.max_grid,
            "block_size": args.block_size,
        },
        "stages": [asdict(stage) for stage in overlap_stages],
        "host_stages": [asdict(stage) for stage in host_stages],
        "compile_log": compile_log,
        "runtime_ledger_evidence": {
            "oracle_mode": "sparse_page_cuda_overlap_smoke",
            "transport": "pinned_host_h2d_double_buffer",
            "kernel": "byte_sum_touch",
            "transformer_layer_math": False,
            "host_read_overlap_measured": False,
            "gpu_h2d_kernel_overlap_measured": True,
        },
        "summary": {
            "stages_staged": len(overlap_stages),
            "planned_bytes": planned_bytes,
            "staged_bytes": staged_bytes,
            "staged_gib": bytes_to_gib(staged_bytes),
            "host_pinned_bytes": sum(host_stage_bytes),
            "host_pinned_gib": bytes_to_gib(sum(host_stage_bytes)),
            "max_live_buffer_bytes": max_live_buffer_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_buffer_bytes),
            "max_device_live_bytes": max_device_live_bytes,
            "max_device_live_gib": bytes_to_gib(max_device_live_bytes),
            "read_calls": sum(stage.read_calls for stage in overlap_stages),
            "host_read_ms": host_read_ms,
            "host_read_elapsed_ms": host_read_elapsed_ms,
            "h2d_ms": h2d_ms,
            "kernel_ms": kernel_ms,
            "sequential_gpu_ms": sequential_gpu_ms,
            "overlapped_gpu_ms": overlapped_gpu_ms,
            "overlapped_gpu_wall_ms": gpu_wall_ms,
            "overlap_savings_ms": overlap_savings_ms,
            "overlap_savings_pct": overlap_savings_pct,
            "h2d_throughput_gib_s": bytes_to_gib(staged_bytes) / (h2d_ms / 1000.0) if h2d_ms > 0 else 0.0,
            "overlapped_gpu_throughput_gib_s": bytes_to_gib(staged_bytes) / (overlapped_gpu_ms / 1000.0)
            if overlapped_gpu_ms > 0
            else 0.0,
            "kernel_touch_throughput_gib_s": bytes_to_gib(staged_bytes) / (kernel_ms / 1000.0) if kernel_ms > 0 else 0.0,
            "kernel_output_u64": int(output_value.value),
            "kernel_output_nonzero": output_nonzero,
            "stage_byte_match": stage_byte_match,
            "byte_budget_respected": staged_bytes <= int(ledger.get("summary", {}).get("selected_bytes", staged_bytes)),
            "pcie_transfer_status": "measured_cuda_h2d_from_prestaged_pinned_host",
            "cuda_overlap_status": "measured_two_stream_double_buffer_cuda_events",
            "cuda_kernel_status": "measured_byte_sum_touch_kernel",
            "host_read_overlap_status": "not_measured_prestaged_pinned_host_buffers",
            "sparse_transformer_status": "not_implemented",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure CUDA H2D/kernel overlap for SAGE oracle page stages.")
    parser.add_argument(
        "--page-ledger",
        default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json",
        help="SAGE oracle page ledger JSON",
    )
    parser.add_argument("--model", default="", help="Override model path from the page ledger")
    parser.add_argument("--json-out", default="", help="Optional output JSON path")
    parser.add_argument("--cudart", default="", help="Path to cudart DLL")
    parser.add_argument("--nvrtc", default="", help="Path to nvrtc DLL")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--arch", default="", help="NVRTC architecture override such as sm_86")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="Override planned stage buffer size")
    parser.add_argument("--limit-stages", type=int, default=0, help="Limit number of planned stages")
    parser.add_argument("--max-gib", type=float, default=0.0, help="Stop after this many GiB of planned stages")
    parser.add_argument("--device-buffer-count", type=int, default=2, help="Number of device buffers used for the overlap ring")
    parser.add_argument("--block-size", type=int, default=256, help="CUDA block size")
    parser.add_argument("--max-grid", type=int, default=65535, help="CUDA grid cap")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = make_payload(args)
    text = json.dumps(payload, indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
