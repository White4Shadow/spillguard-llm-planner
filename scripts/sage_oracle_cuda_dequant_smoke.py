#!/usr/bin/env python3
"""
Run a CUDA Q4_0 dequantization smoke over staged SAGE oracle pages.

This proves the staged Gemma GGUF pages can be interpreted as quantized tensor
blocks by CUDA code. It still does not perform transformer matmul or score
candidate tokens.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf
from sage_oracle_cuda_kernel_smoke import CudaDriver, NvrtcRuntime, fail
from sage_oracle_cuda_staging import CudaRuntime
from sage_oracle_pager_staging import (
    BYTES_PER_GIB,
    bytes_to_gib,
    gguf_data_start,
    group_tensors_by_block,
    load_json,
    resolve_model_path,
    selected_stages,
)


KERNEL_SOURCE = r"""
__device__ float sage_half_to_float(unsigned short h) {
    unsigned int sign = ((unsigned int) h & 0x8000u) << 16;
    unsigned int exp = ((unsigned int) h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int) h & 0x03ffu;
    unsigned int out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            unsigned int f_exp = exp + (127u - 15u);
            out = sign | (f_exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        unsigned int f_exp = exp + (127u - 15u);
        out = sign | (f_exp << 23) | (mant << 13);
    }

    return __uint_as_float(out);
}

extern "C" __global__
void sage_q4_0_dequant_reduce(const unsigned char * data, unsigned long long n_blocks, float * out_sum, float * out_abs) {
    unsigned int tid = threadIdx.x;
    unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + tid;
    unsigned long long stride = (unsigned long long) blockDim.x * gridDim.x;
    float sum_acc = 0.0f;
    float abs_acc = 0.0f;

    while (i < n_blocks) {
        const unsigned char * block = data + i * 18ull;
        unsigned short hb = ((unsigned short) block[0]) | (((unsigned short) block[1]) << 8);
        float d = sage_half_to_float(hb);

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned char q = block[2 + j];
            int lo = (int) (q & 0x0f) - 8;
            int hi = (int) (q >> 4) - 8;
            float vlo = d * (float) lo;
            float vhi = d * (float) hi;
            sum_acc += vlo + vhi;
            abs_acc += fabsf(vlo) + fabsf(vhi);
        }
        i += stride;
    }

    atomicAdd(out_sum, sum_acc);
    atomicAdd(out_abs, abs_acc);
}
"""


@dataclass
class DequantStage:
    stage_index: int
    buffer: str
    q4_0_bytes: int
    q4_0_tensors: int
    q4_0_blocks: int
    q4_0_values: int
    host_read_ms: float
    h2d_ms: float
    dequant_ms: float
    dequant_throughput_gib_s: float
    page_ids: list[int]


def launch_q4_0_reduce(
    driver: CudaDriver,
    function: ctypes.c_void_p,
    device_ptr: ctypes.c_void_p,
    n_blocks: int,
    out_sum: ctypes.c_void_p,
    out_abs: ctypes.c_void_p,
    stream: ctypes.c_void_p,
    grid: int,
    block: int,
) -> None:
    data_arg = ctypes.c_uint64(int(device_ptr.value))
    blocks_arg = ctypes.c_uint64(n_blocks)
    sum_arg = ctypes.c_uint64(int(out_sum.value))
    abs_arg = ctypes.c_uint64(int(out_abs.value))
    params = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(data_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(blocks_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(sum_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(abs_arg), ctypes.c_void_p),
    )
    driver.check(
        driver.lib.cuLaunchKernel(
            function,
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream,
            params,
            None,
        ),
        "cuLaunchKernel(q4_0_dequant)",
    )


def make_host_view(ptr: ctypes.c_void_p, n_bytes: int) -> memoryview:
    array_type = ctypes.c_ubyte * n_bytes
    array = array_type.from_address(int(ptr.value))
    return memoryview(array).cast("B")


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = Path(args.page_ledger)
    ledger = load_json(ledger_path)
    if ledger.get("schema") != "sage-oracle-page-ledger-v0":
        fail("expected sage-oracle-page-ledger-v0 input")
    model_path = resolve_model_path(args, ledger)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")

    cuda = CudaRuntime(args.cudart)
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
    module, function = driver.load_function(module_image, b"sage_q4_0_dequant_reduce")

    budget = ledger.get("budget", {})
    if not isinstance(budget, dict):
        fail("page ledger is missing budget object")
    stage_buffer_bytes = int(args.stage_buffer_gib * BYTES_PER_GIB) if args.stage_buffer_gib > 0 else int(budget.get("stage_buffer_bytes", 0))
    if stage_buffer_bytes <= 0:
        fail("stage buffer bytes must be positive")
    max_bytes = int(args.max_gib * BYTES_PER_GIB) if args.max_gib > 0 else 0
    stages = selected_stages(ledger, args.limit_stages, max_bytes)
    if not stages:
        fail("no stages selected")

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

    host_ptrs: list[ctypes.c_void_p] = []
    device_ptrs: list[ctypes.c_void_p] = []
    streams: list[ctypes.c_void_p] = []
    start_events: list[ctypes.c_void_p] = []
    stop_events: list[ctypes.c_void_p] = []
    host_views: list[memoryview] = []
    out_sum = ctypes.c_void_p()
    out_abs = ctypes.c_void_p()

    try:
        for _ in range(args.buffer_count):
            host_ptr = cuda.host_alloc(stage_buffer_bytes)
            device_ptr = cuda.device_alloc(stage_buffer_bytes)
            stream = cuda.stream_create()
            start_event = cuda.event_create()
            stop_event = cuda.event_create()
            host_ptrs.append(host_ptr)
            device_ptrs.append(device_ptr)
            streams.append(stream)
            start_events.append(start_event)
            stop_events.append(stop_event)
            host_views.append(make_host_view(host_ptr, stage_buffer_bytes))
        out_sum = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        out_abs = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        cuda.memset_async(out_sum, 0, ctypes.sizeof(ctypes.c_float), streams[0])
        cuda.memset_async(out_abs, 0, ctypes.sizeof(ctypes.c_float), streams[0])
        cuda.device_synchronize()

        dequant_stages: list[DequantStage] = []
        started = time.perf_counter()
        skipped_f32_bytes = 0
        skipped_tensors = 0

        with model_path.open("rb", buffering=0) as handle:
            for planned_stage in stages:
                stage_index = int(planned_stage.get("stage_index", len(dequant_stages)))
                buffer_index = stage_index % args.buffer_count
                host_view = host_views[buffer_index]
                page_ids = [int(page_id) for page_id in planned_stage.get("page_ids", [])]

                stage_offset = 0
                q4_0_tensors = 0
                host_start = time.perf_counter()
                for page_id in page_ids:
                    page = page_by_id.get(page_id)
                    if not page:
                        fail(f"stage {stage_index} references missing page {page_id}")
                    block_key = str(page.get("block_key", ""))
                    tensors = groups.get(block_key, [])
                    if not tensors:
                        fail(f"no GGUF tensors found for block page {block_key}")
                    for tensor in tensors:
                        if tensor.tensor_type != "Q4_0":
                            skipped_f32_bytes += tensor.n_bytes
                            skipped_tensors += 1
                            continue
                        if tensor.n_bytes % 18 != 0:
                            fail(f"Q4_0 tensor byte size is not block-aligned: {tensor.name}")
                        if stage_offset + tensor.n_bytes > stage_buffer_bytes:
                            fail(f"stage {stage_index} Q4_0 bytes overflow staging buffer")
                        target = host_view[stage_offset : stage_offset + tensor.n_bytes]
                        handle.seek(data_start + tensor.offset)
                        n_read = handle.readinto(target)
                        if n_read != tensor.n_bytes:
                            fail(f"short read for tensor {tensor.name}: {n_read} != {tensor.n_bytes}")
                        stage_offset += tensor.n_bytes
                        q4_0_tensors += 1
                host_read_ms = (time.perf_counter() - host_start) * 1000.0
                if stage_offset == 0:
                    continue

                stream = streams[buffer_index]
                start_event = start_events[buffer_index]
                stop_event = stop_events[buffer_index]
                h2d_ms = cuda.memcpy_h2d_timed(
                    device_ptrs[buffer_index],
                    host_ptrs[buffer_index],
                    stage_offset,
                    stream,
                    start_event,
                    stop_event,
                )

                n_blocks = stage_offset // 18
                block = args.block_size
                grid = min(args.max_grid, max(1, (n_blocks + block - 1) // block))
                cuda.lib.cudaEventRecord(start_event, stream)
                launch_q4_0_reduce(
                    driver,
                    function,
                    device_ptrs[buffer_index],
                    n_blocks,
                    out_sum,
                    out_abs,
                    stream,
                    grid,
                    block,
                )
                cuda.lib.cudaEventRecord(stop_event, stream)
                cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(q4_0_dequant)")
                elapsed = ctypes.c_float()
                cuda.check(
                    cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event),
                    "cudaEventElapsedTime(q4_0_dequant)",
                )
                dequant_ms = float(elapsed.value)
                dequant_stages.append(
                    DequantStage(
                        stage_index=stage_index,
                        buffer=str(planned_stage.get("buffer", chr(ord("A") + buffer_index))),
                        q4_0_bytes=stage_offset,
                        q4_0_tensors=q4_0_tensors,
                        q4_0_blocks=n_blocks,
                        q4_0_values=n_blocks * 32,
                        host_read_ms=host_read_ms,
                        h2d_ms=h2d_ms,
                        dequant_ms=dequant_ms,
                        dequant_throughput_gib_s=bytes_to_gib(stage_offset) / (dequant_ms / 1000.0)
                        if dequant_ms > 0
                        else 0.0,
                        page_ids=page_ids,
                    )
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sum_out = ctypes.c_float()
        abs_out = ctypes.c_float()
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(sum_out), ctypes.c_void_p), out_sum, ctypes.sizeof(sum_out))
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(abs_out), ctypes.c_void_p), out_abs, ctypes.sizeof(abs_out))
        free_after, _total_after = cuda.mem_info()
    finally:
        if out_abs and out_abs.value:
            cuda.device_free(out_abs)
        if out_sum and out_sum.value:
            cuda.device_free(out_sum)
        for view in host_views:
            view.release()
        for event in stop_events:
            cuda.event_destroy(event)
        for event in start_events:
            cuda.event_destroy(event)
        for stream in streams:
            cuda.stream_destroy(stream)
        for ptr in device_ptrs:
            cuda.device_free(ptr)
        for ptr in host_ptrs:
            cuda.host_free(ptr)
        driver.unload_module(module)

    q4_0_bytes = sum(stage.q4_0_bytes for stage in dequant_stages)
    q4_0_blocks = sum(stage.q4_0_blocks for stage in dequant_stages)
    q4_0_values = sum(stage.q4_0_values for stage in dequant_stages)
    h2d_ms = sum(stage.h2d_ms for stage in dequant_stages)
    dequant_ms = sum(stage.dequant_ms for stage in dequant_stages)
    host_read_ms = sum(stage.host_read_ms for stage in dequant_stages)
    max_live_bytes = max((stage.q4_0_bytes for stage in dequant_stages), default=0)

    return {
        "schema": "sage-oracle-page-cuda-dequant-smoke-v0",
        "status": "measured_cuda_q4_0_dequant_not_matmul",
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
            "vram_free_after_dequant_bytes": free_after,
        },
        "limits": {
            "limit_stages": args.limit_stages,
            "max_gib": args.max_gib,
            "buffer_count": args.buffer_count,
            "stage_buffer_bytes": stage_buffer_bytes,
            "stage_buffer_gib": bytes_to_gib(stage_buffer_bytes),
            "allocated_pinned_host_bytes": stage_buffer_bytes * args.buffer_count,
            "allocated_device_buffer_bytes": stage_buffer_bytes * args.buffer_count,
            "kernel_block_size": args.block_size,
            "kernel_max_grid": args.max_grid,
        },
        "summary": {
            "stages_staged": len(dequant_stages),
            "q4_0_tensors": sum(stage.q4_0_tensors for stage in dequant_stages),
            "q4_0_bytes": q4_0_bytes,
            "q4_0_gib": bytes_to_gib(q4_0_bytes),
            "q4_0_blocks": q4_0_blocks,
            "q4_0_values": q4_0_values,
            "skipped_non_q4_0_tensors": skipped_tensors,
            "skipped_non_q4_0_bytes": skipped_f32_bytes,
            "max_live_buffer_bytes": max_live_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_bytes),
            "elapsed_ms": elapsed_ms,
            "host_read_ms": host_read_ms,
            "h2d_ms": h2d_ms,
            "dequant_ms": dequant_ms,
            "h2d_throughput_gib_s": bytes_to_gib(q4_0_bytes) / (h2d_ms / 1000.0) if h2d_ms > 0 else 0.0,
            "dequant_throughput_gib_s": bytes_to_gib(q4_0_bytes) / (dequant_ms / 1000.0) if dequant_ms > 0 else 0.0,
            "dequant_sum": float(sum_out.value),
            "dequant_abs_sum": float(abs_out.value),
            "dequant_output_nonzero": float(abs_out.value) > 0.0,
            "byte_budget_respected": max_live_bytes <= stage_buffer_bytes,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_kernel_status": "measured_q4_0_dequant_reduce_kernel",
            "sparse_matmul_status": "not_implemented",
            "sparse_transformer_status": "not_implemented",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_cuda_q4_0_dequant_smoke",
            "oracle_active_bytes": q4_0_bytes,
            "gpu_staged_bytes": max_live_bytes,
            "host_pinned_bytes": stage_buffer_bytes * args.buffer_count,
            "pcie_transfer_ms": h2d_ms,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_dequant_ms": dequant_ms,
            "cuda_kernel_status": "measured_q4_0_dequant_reduce_kernel",
        },
        "compile_log": compile_log,
        "stages": [asdict(stage) for stage in dequant_stages],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    limits = payload["limits"]
    cuda = payload["cuda"]
    print("# SAGE Oracle CUDA Q4_0 Dequant Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- CUDA arch: `{cuda['arch']}`")
    print(f"- Stages staged: `{summary['stages_staged']}`")
    print(f"- Q4_0 tensors: `{summary['q4_0_tensors']}`")
    print(f"- Q4_0 bytes: `{summary['q4_0_gib']:.3f} GiB`")
    print(f"- Dequantized values: `{summary['q4_0_values']}`")
    print(f"- Max live device buffer: `{summary['max_live_buffer_gib']:.3f} GiB` / `{limits['stage_buffer_gib']:.3f} GiB`")
    print(f"- H2D time: `{summary['h2d_ms']:.2f} ms`")
    print(f"- Dequant time: `{summary['dequant_ms']:.2f} ms`")
    print(f"- Dequant throughput: `{summary['dequant_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Dequant abs sum nonzero: `{summary['dequant_output_nonzero']}`")
    print(f"- Sparse matmul: `{summary['sparse_matmul_status']}`")
    print()
    print("## Dequant Stages")
    print()
    print("| Stage | Buffer | Q4_0 tensors | Q4_0 bytes | H2D | Dequant | Throughput |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(
            f"| {stage['stage_index']} | {stage['buffer']} | {stage['q4_0_tensors']} | "
            f"{bytes_to_gib(stage['q4_0_bytes']):.3f} GiB | {stage['h2d_ms']:.2f} ms | "
            f"{stage['dequant_ms']:.2f} ms | {stage['dequant_throughput_gib_s']:.2f} GiB/s |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA Q4_0 dequantization smoke over staged GGUF oracle pages.")
    parser.add_argument("--page-ledger", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--model", default="", help="override model path from the page ledger")
    parser.add_argument("--limit-stages", type=int, default=0, help="number of planned stages to execute; 0 means all")
    parser.add_argument("--max-gib", type=float, default=0.0, help="optional cap on selected planned stage bytes")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="override stage buffer size; default uses page ledger")
    parser.add_argument("--buffer-count", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="", help="NVRTC architecture such as sm_86; default queries CUDA driver")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-grid", type=int, default=4096)
    parser.add_argument("--cudart", default="", help="explicit path to cudart DLL")
    parser.add_argument("--nvrtc", default="", help="explicit path to nvrtc DLL")
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
    if args.block_size <= 0 or args.block_size > 1024:
        parser.error("--block-size must be between 1 and 1024")
    if args.max_grid <= 0:
        parser.error("--max-grid must be positive")

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
