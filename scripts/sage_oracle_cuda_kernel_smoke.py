#!/usr/bin/env python3
"""
Launch a tiny CUDA kernel over staged SAGE oracle pages.

This smoke proves that planned GGUF oracle pages can be copied into bounded CUDA
buffers and consumed by a GPU kernel. It is not transformer math and does not
score candidate tokens.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf
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


NVRTC_SUCCESS = 0
CUDA_SUCCESS = 0


KERNEL_SOURCE = r"""
extern "C" __global__
void sage_page_byte_sum(const unsigned char * data, unsigned long long n, unsigned long long * out) {
    __shared__ unsigned long long local[256];
    unsigned int tid = threadIdx.x;
    unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + tid;
    unsigned long long stride = (unsigned long long) blockDim.x * gridDim.x;
    unsigned long long acc = 0;

    while (i < n) {
        acc += (unsigned long long) data[i];
        i += stride;
    }

    local[tid] = acc;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local[tid] += local[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, local[0]);
    }
}
"""


@dataclass
class KernelStage:
    stage_index: int
    buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    read_calls: int
    host_read_ms: float
    h2d_ms: float
    kernel_ms: float
    kernel_grid: int
    kernel_block: int
    kernel_throughput_gib_s: float
    page_ids: list[int]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def candidate_nvrtc_paths() -> list[Path]:
    paths: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        paths.append(Path(cuda_path) / "bin" / "x64" / "nvrtc64_130_0.dll")
        paths.append(Path(cuda_path) / "bin" / "nvrtc64_130_0.dll")
    roots = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"),
    ]
    for root in roots:
        paths.append(root / "bin" / "x64" / "nvrtc64_130_0.dll")
        paths.append(root / "bin" / "nvrtc64_130_0.dll")
    return paths


class NvrtcRuntime:
    def __init__(self, explicit_path: str = "") -> None:
        self.path = self._resolve_path(explicit_path)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(self.path.parent))
        self.lib = ctypes.WinDLL(str(self.path))
        self._bind()

    def _resolve_path(self, explicit_path: str) -> Path:
        if explicit_path:
            path = Path(explicit_path)
            if not path.is_file():
                fail(f"NVRTC DLL not found: {path}")
            return path
        for path in candidate_nvrtc_paths():
            if path.is_file():
                return path
        fail("could not locate NVRTC DLL; pass --nvrtc")

    def _bind(self) -> None:
        lib = self.lib
        lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.nvrtcCreateProgram.restype = ctypes.c_int
        lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcCompileProgram.restype = ctypes.c_int
        lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
        lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetProgramLog.restype = ctypes.c_int
        lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetPTXSize.restype = ctypes.c_int
        lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetPTX.restype = ctypes.c_int
        self.has_cubin = hasattr(lib, "nvrtcGetCUBINSize") and hasattr(lib, "nvrtcGetCUBIN")
        if self.has_cubin:
            lib.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            lib.nvrtcGetCUBINSize.restype = ctypes.c_int
            lib.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.nvrtcGetCUBIN.restype = ctypes.c_int
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.nvrtcDestroyProgram.restype = ctypes.c_int

    def check(self, code: int, what: str) -> None:
        if code == NVRTC_SUCCESS:
            return
        raw = self.lib.nvrtcGetErrorString(code)
        message = raw.decode("utf-8", errors="replace") if raw else f"nvrtc error {code}"
        fail(f"{what} failed: {message} ({code})")

    def compile_image(self, source: str, arch: str) -> tuple[bytes, str, str]:
        program = ctypes.c_void_p()
        self.check(
            self.lib.nvrtcCreateProgram(
                ctypes.byref(program),
                source.encode("utf-8"),
                b"sage_page_byte_sum.cu",
                0,
                None,
                None,
            ),
            "nvrtcCreateProgram",
        )
        options = [
            f"--gpu-architecture={arch}".encode("utf-8"),
            b"--std=c++17",
        ]
        option_array = (ctypes.c_char_p * len(options))(*options)
        compile_code = self.lib.nvrtcCompileProgram(program, len(options), option_array)

        log_size = ctypes.c_size_t()
        self.check(self.lib.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), "nvrtcGetProgramLogSize")
        log_buffer = ctypes.create_string_buffer(max(1, int(log_size.value)))
        self.check(self.lib.nvrtcGetProgramLog(program, log_buffer), "nvrtcGetProgramLog")
        log = log_buffer.value.decode("utf-8", errors="replace")
        if compile_code != NVRTC_SUCCESS:
            self.check(compile_code, f"nvrtcCompileProgram log:\n{log}")

        if arch.startswith("sm_") and self.has_cubin:
            image_size = ctypes.c_size_t()
            self.check(self.lib.nvrtcGetCUBINSize(program, ctypes.byref(image_size)), "nvrtcGetCUBINSize")
            image_buffer = ctypes.create_string_buffer(int(image_size.value))
            self.check(self.lib.nvrtcGetCUBIN(program, image_buffer), "nvrtcGetCUBIN")
            image_kind = "cubin"
        else:
            image_size = ctypes.c_size_t()
            self.check(self.lib.nvrtcGetPTXSize(program, ctypes.byref(image_size)), "nvrtcGetPTXSize")
            image_buffer = ctypes.create_string_buffer(int(image_size.value))
            self.check(self.lib.nvrtcGetPTX(program, image_buffer), "nvrtcGetPTX")
            image_kind = "ptx"
        self.check(self.lib.nvrtcDestroyProgram(ctypes.byref(program)), "nvrtcDestroyProgram")
        return bytes(image_buffer.raw), log, image_kind


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.WinDLL("nvcuda.dll")
        self._bind()
        self.check(self.lib.cuInit(0), "cuInit")

    def _bind(self) -> None:
        lib = self.lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuGetErrorString.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.cuDeviceComputeCapability.restype = ctypes.c_int
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuModuleLoadData.restype = ctypes.c_int
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleUnload.restype = ctypes.c_int
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuModuleGetFunction.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cuLaunchKernel.restype = ctypes.c_int

    def check(self, code: int, what: str) -> None:
        if code == CUDA_SUCCESS:
            return
        raw = ctypes.c_char_p()
        self.lib.cuGetErrorString(code, ctypes.byref(raw))
        message = raw.value.decode("utf-8", errors="replace") if raw.value else f"driver error {code}"
        fail(f"{what} failed: {message} ({code})")

    def compute_arch(self, device_index: int, *, real: bool = False) -> str:
        device = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(device), device_index), "cuDeviceGet")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self.check(
            self.lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device),
            "cuDeviceComputeCapability",
        )
        prefix = "sm" if real else "compute"
        return f"{prefix}_{major.value}{minor.value}"

    def require_current_context(self) -> None:
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if not ctx.value:
            fail("CUDA driver has no current context after runtime initialization")

    def load_function(self, ptx: bytes, name: bytes) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        module = ctypes.c_void_p()
        ptx_buffer = ctypes.create_string_buffer(ptx)
        self.check(self.lib.cuModuleLoadData(ctypes.byref(module), ctypes.cast(ptx_buffer, ctypes.c_void_p)), "cuModuleLoadData")
        function = ctypes.c_void_p()
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, name), "cuModuleGetFunction")
        return module, function

    def unload_module(self, module: ctypes.c_void_p) -> None:
        if module and module.value:
            self.check(self.lib.cuModuleUnload(module), "cuModuleUnload")

    def launch_byte_sum(
        self,
        function: ctypes.c_void_p,
        device_ptr: ctypes.c_void_p,
        n_bytes: int,
        output_ptr: ctypes.c_void_p,
        stream: ctypes.c_void_p,
        grid: int,
        block: int,
    ) -> None:
        data_arg = ctypes.c_uint64(int(device_ptr.value))
        n_arg = ctypes.c_uint64(n_bytes)
        out_arg = ctypes.c_uint64(int(output_ptr.value))
        params = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(data_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(out_arg), ctypes.c_void_p),
        )
        self.check(
            self.lib.cuLaunchKernel(
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
            "cuLaunchKernel",
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
    module, function = driver.load_function(module_image, b"sage_page_byte_sum")

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
    output_ptr = ctypes.c_void_p()

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
        output_ptr = cuda.device_alloc(ctypes.sizeof(ctypes.c_uint64))
        cuda.memset_async(output_ptr, 0, ctypes.sizeof(ctypes.c_uint64), streams[0])

        kernel_stages: list[KernelStage] = []
        started = time.perf_counter()
        with model_path.open("rb", buffering=0) as handle:
            for planned_stage in stages:
                stage_index = int(planned_stage.get("stage_index", len(kernel_stages)))
                buffer_index = stage_index % args.buffer_count
                host_view = host_views[buffer_index]
                page_ids = [int(page_id) for page_id in planned_stage.get("page_ids", [])]
                planned_stage_bytes = int(planned_stage.get("n_bytes", 0))
                if planned_stage_bytes > stage_buffer_bytes:
                    fail(
                        f"stage {stage_index} requires {planned_stage_bytes} bytes, "
                        f"but stage buffer holds {stage_buffer_bytes}"
                    )

                stage_offset = 0
                stage_reads = 0
                host_read_start = time.perf_counter()
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
                    for tensor in tensors:
                        target = host_view[stage_offset : stage_offset + tensor.n_bytes]
                        handle.seek(data_start + tensor.offset)
                        n_read = handle.readinto(target)
                        if n_read != tensor.n_bytes:
                            fail(f"short read for tensor {tensor.name}: {n_read} != {tensor.n_bytes}")
                        stage_offset += tensor.n_bytes
                        stage_reads += 1
                host_read_ms = (time.perf_counter() - host_read_start) * 1000.0

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

                block = args.block_size
                grid = min(args.max_grid, max(1, (stage_offset + block - 1) // block))
                cuda.lib.cudaEventRecord(start_event, stream)
                driver.launch_byte_sum(
                    function,
                    device_ptrs[buffer_index],
                    stage_offset,
                    output_ptr,
                    stream,
                    grid,
                    block,
                )
                cuda.lib.cudaEventRecord(stop_event, stream)
                cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(kernel)")
                kernel_elapsed = ctypes.c_float()
                cuda.check(
                    cuda.lib.cudaEventElapsedTime(ctypes.byref(kernel_elapsed), start_event, stop_event),
                    "cudaEventElapsedTime(kernel)",
                )
                kernel_ms = float(kernel_elapsed.value)
                kernel_stages.append(
                    KernelStage(
                        stage_index=stage_index,
                        buffer=str(planned_stage.get("buffer", chr(ord("A") + buffer_index))),
                        planned_bytes=planned_stage_bytes,
                        staged_bytes=stage_offset,
                        n_pages=len(page_ids),
                        read_calls=stage_reads,
                        host_read_ms=host_read_ms,
                        h2d_ms=h2d_ms,
                        kernel_ms=kernel_ms,
                        kernel_grid=grid,
                        kernel_block=block,
                        kernel_throughput_gib_s=bytes_to_gib(stage_offset) / (kernel_ms / 1000.0) if kernel_ms > 0 else 0.0,
                        page_ids=page_ids,
                    )
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output_value = ctypes.c_uint64()
        cuda.memcpy_d2h(
            ctypes.cast(ctypes.byref(output_value), ctypes.c_void_p),
            output_ptr,
            ctypes.sizeof(output_value),
        )
        free_after, _total_after = cuda.mem_info()
    finally:
        if output_ptr and output_ptr.value:
            cuda.device_free(output_ptr)
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

    staged_bytes = sum(stage.staged_bytes for stage in kernel_stages)
    planned_bytes = sum(stage.planned_bytes for stage in kernel_stages)
    h2d_ms = sum(stage.h2d_ms for stage in kernel_stages)
    kernel_ms = sum(stage.kernel_ms for stage in kernel_stages)
    host_read_ms = sum(stage.host_read_ms for stage in kernel_stages)
    max_live_bytes = max((stage.staged_bytes for stage in kernel_stages), default=0)
    stage_byte_match = all(stage.planned_bytes == stage.staged_bytes for stage in kernel_stages)
    return {
        "schema": "sage-oracle-page-cuda-kernel-smoke-v0",
        "status": "measured_cuda_kernel_touch_not_transformer",
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
            "vram_free_after_kernel_bytes": free_after,
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
            "stages_staged": len(kernel_stages),
            "planned_bytes": planned_bytes,
            "staged_bytes": staged_bytes,
            "staged_gib": bytes_to_gib(staged_bytes),
            "max_live_buffer_bytes": max_live_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_bytes),
            "read_calls": sum(stage.read_calls for stage in kernel_stages),
            "elapsed_ms": elapsed_ms,
            "host_read_ms": host_read_ms,
            "h2d_ms": h2d_ms,
            "kernel_ms": kernel_ms,
            "h2d_throughput_gib_s": bytes_to_gib(staged_bytes) / (h2d_ms / 1000.0) if h2d_ms > 0 else 0.0,
            "kernel_touch_throughput_gib_s": bytes_to_gib(staged_bytes) / (kernel_ms / 1000.0) if kernel_ms > 0 else 0.0,
            "kernel_output_u64": int(output_value.value),
            "kernel_output_nonzero": int(output_value.value) != 0,
            "stage_byte_match": stage_byte_match,
            "byte_budget_respected": max_live_bytes <= stage_buffer_bytes,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_kernel_status": "measured_byte_sum_touch_kernel",
            "sparse_transformer_status": "not_implemented",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_cuda_kernel_smoke",
            "oracle_active_bytes": staged_bytes,
            "gpu_staged_bytes": max_live_bytes,
            "host_pinned_bytes": stage_buffer_bytes * args.buffer_count,
            "pcie_transfer_ms": h2d_ms,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_kernel_ms": kernel_ms,
            "cuda_kernel_status": "measured_byte_sum_touch_kernel",
        },
        "compile_log": compile_log,
        "stages": [asdict(stage) for stage in kernel_stages],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    limits = payload["limits"]
    cuda = payload["cuda"]
    print("# SAGE Oracle CUDA Kernel Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- CUDA arch: `{cuda['arch']}`")
    print(f"- Stages staged: `{summary['stages_staged']}`")
    print(f"- Bytes touched: `{summary['staged_gib']:.3f} GiB`")
    print(f"- Max live device buffer: `{summary['max_live_buffer_gib']:.3f} GiB` / `{limits['stage_buffer_gib']:.3f} GiB`")
    print(f"- H2D time: `{summary['h2d_ms']:.2f} ms`")
    print(f"- Kernel time: `{summary['kernel_ms']:.2f} ms`")
    print(f"- Kernel touch throughput: `{summary['kernel_touch_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Kernel output nonzero: `{summary['kernel_output_nonzero']}`")
    print(f"- Sparse transformer: `{summary['sparse_transformer_status']}`")
    print()
    print("## Kernel Stages")
    print()
    print("| Stage | Buffer | Pages | Bytes | H2D | Kernel | Kernel throughput |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(
            f"| {stage['stage_index']} | {stage['buffer']} | {stage['n_pages']} | "
            f"{bytes_to_gib(stage['staged_bytes']):.3f} GiB | {stage['h2d_ms']:.2f} ms | "
            f"{stage['kernel_ms']:.2f} ms | {stage['kernel_throughput_gib_s']:.2f} GiB/s |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a CUDA kernel over staged GGUF sparse-oracle pages.")
    parser.add_argument("--page-ledger", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--model", default="", help="override model path from the page ledger")
    parser.add_argument("--limit-stages", type=int, default=0, help="number of planned stages to execute; 0 means all")
    parser.add_argument("--max-gib", type=float, default=0.0, help="optional cap on selected planned stage bytes")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="override stage buffer size; default uses page ledger")
    parser.add_argument("--buffer-count", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="", help="NVRTC architecture such as compute_86; default queries CUDA driver")
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
