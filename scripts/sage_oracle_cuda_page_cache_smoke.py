#!/usr/bin/env python3
"""
Measure resident pinned page-cache replay for SAGE sparse-oracle pages.

This smoke builds a pinned host cache for the selected GGUF page stages once,
then replays several sparse-oracle fallback passes from that cache through CUDA
H2D plus a byte-touch kernel. It proves cache reuse mechanics and amortized
transport timing. It is not transformer execution.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf
from sage_oracle_cuda_kernel_smoke import CudaDriver, KERNEL_SOURCE, NvrtcRuntime
from sage_oracle_cuda_overlap_smoke import (
    HostStage,
    bind_overlap_cuda,
    event_elapsed_ms,
    event_record,
    event_sync,
    read_stage_into_pinned_host,
    stream_wait_event,
)
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


@dataclass
class CacheReplayStage:
    replay_index: int
    stage_index: int
    host_cache_slot: int
    device_buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    h2d_ms: float
    kernel_ms: float
    kernel_grid: int
    kernel_block: int
    page_ids: list[int]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = Path(args.page_ledger)
    ledger = load_json(ledger_path)
    if ledger.get("schema") != "sage-oracle-page-ledger-v0":
        fail("expected sage-oracle-page-ledger-v0 input")
    model_path = resolve_model_path(args, ledger)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")
    if args.replays < 1:
        fail("--replays must be positive")

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
        fail("at least two stages are required to measure cache replay overlap")

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
    replay_start_events: list[ctypes.c_void_p] = []
    replay_stop_events: list[ctypes.c_void_p] = []
    total_start_event = ctypes.c_void_p()
    total_stop_event = ctypes.c_void_p()
    replay_stages: list[CacheReplayStage] = []

    try:
        cache_build_started = time.perf_counter()
        with model_path.open("rb", buffering=0) as handle:
            for index_in_plan, planned_stage in enumerate(stages):
                stage_index = int(planned_stage.get("stage_index", index_in_plan))
                planned_bytes = int(planned_stage.get("n_bytes", 0))
                if planned_bytes <= 0:
                    fail(f"stage {stage_index} has no planned bytes")
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
        cache_build_ms = (time.perf_counter() - cache_build_started) * 1000.0

        max_stage_bytes = max(host_stage_bytes)
        for _ in range(args.device_buffer_count):
            device_ptrs.append(cuda.device_alloc(max_stage_bytes))
        transfer_stream = cuda.stream_create()
        compute_stream = cuda.stream_create()
        output_ptr = cuda.device_alloc(ctypes.sizeof(ctypes.c_uint64))
        cuda.memset_async(output_ptr, 0, ctypes.sizeof(ctypes.c_uint64), compute_stream)
        cuda.device_synchronize()

        total_ops = args.replays * len(host_stages)
        for _ in range(total_ops):
            h2d_start_events.append(cuda.event_create())
            h2d_stop_events.append(cuda.event_create())
            kernel_start_events.append(cuda.event_create())
            kernel_stop_events.append(cuda.event_create())
        for _ in range(args.replays):
            replay_start_events.append(cuda.event_create())
            replay_stop_events.append(cuda.event_create())
        total_start_event = cuda.event_create()
        total_stop_event = cuda.event_create()

        replay_wall_started = time.perf_counter()
        event_record(cuda, total_start_event, transfer_stream, "cudaEventRecord(cache-replay-start)")
        for replay_index in range(args.replays):
            event_record(cuda, replay_start_events[replay_index], transfer_stream, f"cudaEventRecord(replay-start-{replay_index})")
            for stage_pos, host_stage in enumerate(host_stages):
                op_index = replay_index * len(host_stages) + stage_pos
                device_buffer_index = op_index % args.device_buffer_count
                if op_index >= args.device_buffer_count:
                    stream_wait_event(
                        cuda,
                        transfer_stream,
                        kernel_stop_events[op_index - args.device_buffer_count],
                        f"cudaStreamWaitEvent(reuse-device-buffer-{device_buffer_index})",
                    )

                event_record(cuda, h2d_start_events[op_index], transfer_stream, f"cudaEventRecord(h2d-start-{op_index})")
                cuda.check(
                    cuda.lib.cudaMemcpyAsync(
                        device_ptrs[device_buffer_index],
                        host_ptrs[stage_pos],
                        host_stage.staged_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        transfer_stream,
                    ),
                    f"cudaMemcpyAsync(H2D replay {replay_index} stage {stage_pos})",
                )
                event_record(cuda, h2d_stop_events[op_index], transfer_stream, f"cudaEventRecord(h2d-stop-{op_index})")

                stream_wait_event(cuda, compute_stream, h2d_stop_events[op_index], f"cudaStreamWaitEvent(h2d-done-{op_index})")
                event_record(cuda, kernel_start_events[op_index], compute_stream, f"cudaEventRecord(kernel-start-{op_index})")
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
                event_record(cuda, kernel_stop_events[op_index], compute_stream, f"cudaEventRecord(kernel-stop-{op_index})")
                replay_stages.append(
                    CacheReplayStage(
                        replay_index=replay_index,
                        stage_index=host_stage.stage_index,
                        host_cache_slot=stage_pos,
                        device_buffer=chr(ord("A") + device_buffer_index),
                        planned_bytes=host_stage.planned_bytes,
                        staged_bytes=host_stage.staged_bytes,
                        n_pages=host_stage.n_pages,
                        h2d_ms=0.0,
                        kernel_ms=0.0,
                        kernel_grid=grid,
                        kernel_block=block,
                        page_ids=host_stage.page_ids,
                    )
                )
            stream_wait_event(
                cuda,
                transfer_stream,
                kernel_stop_events[(replay_index + 1) * len(host_stages) - 1],
                f"cudaStreamWaitEvent(replay-done-{replay_index})",
            )
            event_record(cuda, replay_stop_events[replay_index], transfer_stream, f"cudaEventRecord(replay-stop-{replay_index})")

        event_record(cuda, total_stop_event, transfer_stream, "cudaEventRecord(cache-replay-stop)")
        event_sync(cuda, total_stop_event, "cudaEventSynchronize(cache-replay-stop)")
        replay_wall_ms = (time.perf_counter() - replay_wall_started) * 1000.0
        total_replay_gpu_ms = event_elapsed_ms(cuda, total_start_event, total_stop_event, "cudaEventElapsedTime(cache-replay-total)")

        for op_index, stage in enumerate(replay_stages):
            stage.h2d_ms = event_elapsed_ms(cuda, h2d_start_events[op_index], h2d_stop_events[op_index], f"cudaEventElapsedTime(h2d-{op_index})")
            stage.kernel_ms = event_elapsed_ms(
                cuda,
                kernel_start_events[op_index],
                kernel_stop_events[op_index],
                f"cudaEventElapsedTime(kernel-{op_index})",
            )
        replay_gpu_ms = [
            event_elapsed_ms(cuda, replay_start_events[i], replay_stop_events[i], f"cudaEventElapsedTime(replay-{i})")
            for i in range(args.replays)
        ]

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
        for event in [total_stop_event, total_start_event]:
            if event and event.value:
                cuda.event_destroy(event)
        for events in [replay_stop_events, replay_start_events, kernel_stop_events, kernel_start_events, h2d_stop_events, h2d_start_events]:
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

    cache_bytes = sum(host_stage_bytes)
    staged_bytes_per_replay = sum(stage.staged_bytes for stage in host_stages)
    planned_bytes_per_replay = sum(stage.planned_bytes for stage in host_stages)
    h2d_ms_total = sum(stage.h2d_ms for stage in replay_stages)
    kernel_ms_total = sum(stage.kernel_ms for stage in replay_stages)
    sequential_gpu_ms_total = h2d_ms_total + kernel_ms_total
    per_replay_h2d_ms = h2d_ms_total / args.replays
    per_replay_kernel_ms = kernel_ms_total / args.replays
    per_replay_sequential_gpu_ms = sequential_gpu_ms_total / args.replays
    per_replay_gpu_ms = total_replay_gpu_ms / args.replays
    amortized_total_ms = (cache_build_ms + total_replay_gpu_ms) / args.replays
    cache_build_per_replay_ms = cache_build_ms / args.replays
    cache_hits = max(0, args.replays - 1) * len(host_stages)
    cache_hit_bytes = max(0, args.replays - 1) * staged_bytes_per_replay
    stage_byte_match = all(stage.planned_bytes == stage.staged_bytes for stage in host_stages)
    output_nonzero = int(output_value.value) != 0

    return {
        "schema": "sage-oracle-page-cuda-page-cache-smoke-v0",
        "status": "measured_resident_pinned_page_cache_replay_touch_not_transformer",
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
            "replays": args.replays,
            "max_grid": args.max_grid,
            "block_size": args.block_size,
        },
        "cache_stages": [asdict(stage) for stage in host_stages],
        "replay_stages": [asdict(stage) for stage in replay_stages],
        "replay_gpu_ms": replay_gpu_ms,
        "compile_log": compile_log,
        "runtime_ledger_evidence": {
            "oracle_mode": "sparse_page_cuda_page_cache_smoke",
            "transport": "resident_pinned_host_page_cache_then_h2d",
            "kernel": "byte_sum_touch",
            "transformer_layer_math": False,
            "resident_pinned_page_cache": True,
            "cache_replay_measured": True,
            "gpu_h2d_kernel_overlap_measured": True,
        },
        "summary": {
            "stages_cached": len(host_stages),
            "replays": args.replays,
            "planned_bytes_per_replay": planned_bytes_per_replay,
            "staged_bytes_per_replay": staged_bytes_per_replay,
            "staged_gib_per_replay": bytes_to_gib(staged_bytes_per_replay),
            "cache_bytes": cache_bytes,
            "cache_gib": bytes_to_gib(cache_bytes),
            "cache_build_ms": cache_build_ms,
            "cache_build_per_replay_ms": cache_build_per_replay_ms,
            "cache_hits": cache_hits,
            "cache_hit_bytes": cache_hit_bytes,
            "cache_hit_gib": bytes_to_gib(cache_hit_bytes),
            "h2d_ms_total": h2d_ms_total,
            "kernel_ms_total": kernel_ms_total,
            "sequential_gpu_ms_total": sequential_gpu_ms_total,
            "total_replay_gpu_ms": total_replay_gpu_ms,
            "replay_wall_ms": replay_wall_ms,
            "per_replay_h2d_ms": per_replay_h2d_ms,
            "per_replay_kernel_ms": per_replay_kernel_ms,
            "per_replay_sequential_gpu_ms": per_replay_sequential_gpu_ms,
            "per_replay_gpu_ms": per_replay_gpu_ms,
            "amortized_total_ms": amortized_total_ms,
            "cache_replay_saves_host_read": cache_hits > 0,
            "h2d_throughput_gib_s": bytes_to_gib(staged_bytes_per_replay * args.replays) / (h2d_ms_total / 1000.0)
            if h2d_ms_total > 0
            else 0.0,
            "kernel_touch_throughput_gib_s": bytes_to_gib(staged_bytes_per_replay * args.replays) / (kernel_ms_total / 1000.0)
            if kernel_ms_total > 0
            else 0.0,
            "replay_throughput_gib_s": bytes_to_gib(staged_bytes_per_replay * args.replays) / (total_replay_gpu_ms / 1000.0)
            if total_replay_gpu_ms > 0
            else 0.0,
            "kernel_output_u64": int(output_value.value),
            "kernel_output_nonzero": output_nonzero,
            "stage_byte_match": stage_byte_match,
            "byte_budget_respected": staged_bytes_per_replay <= int(ledger.get("summary", {}).get("selected_bytes", staged_bytes_per_replay)),
            "cache_status": "measured_resident_pinned_host_page_cache",
            "pcie_transfer_status": "measured_cuda_h2d_from_resident_pinned_page_cache",
            "cuda_overlap_status": "measured_two_stream_double_buffer_cuda_events",
            "cuda_kernel_status": "measured_byte_sum_touch_kernel",
            "sparse_transformer_status": "not_implemented",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure resident pinned page-cache replay for SAGE oracle page stages.")
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
    parser.add_argument("--device-buffer-count", type=int, default=2, help="CUDA device buffers used for the replay ring")
    parser.add_argument("--replays", type=int, default=3, help="How many fallback replays to run from the resident page cache")
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
