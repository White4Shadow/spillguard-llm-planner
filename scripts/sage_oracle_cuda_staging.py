#!/usr/bin/env python3
"""
Measure SAGE sparse-oracle page staging through real CUDA device buffers.

This is still not sparse oracle execution. It proves the next transport layer:
selected GGUF pages can be read into pinned host buffers and copied into bounded
CUDA device staging buffers, with measured H2D transfer time.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_oracle_pager_staging import (
    BYTES_PER_GIB,
    bytes_to_gib,
    gguf_data_start,
    group_tensors_by_block,
    load_json,
    resolve_model_path,
    selected_stages,
)
from sage_gguf_blocks import parse_gguf


CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_HOST_ALLOC_DEFAULT = 0


@dataclass
class CudaStagedPage:
    page_id: int
    block_key: str
    planned_bytes: int
    staged_bytes: int
    n_tensors: int
    read_calls: int
    host_read_ms: float
    crc32: str


@dataclass
class CudaStagedStage:
    stage_index: int
    buffer: str
    planned_bytes: int
    staged_bytes: int
    n_pages: int
    read_calls: int
    host_read_ms: float
    h2d_ms: float
    total_stage_ms: float
    host_read_throughput_gib_s: float
    h2d_throughput_gib_s: float
    page_ids: list[int]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def candidate_cudart_paths() -> list[Path]:
    paths: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        paths.extend(
            [
                Path(cuda_path) / "bin" / "x64" / "cudart64_13.dll",
                Path(cuda_path) / "bin" / "cudart64_13.dll",
            ]
        )
    roots = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"),
    ]
    for root in roots:
        paths.extend(
            [
                root / "bin" / "x64" / "cudart64_13.dll",
                root / "bin" / "cudart64_13.dll",
                root / "bin" / "x64" / "cudart64_12.dll",
                root / "bin" / "cudart64_12.dll",
            ]
        )
    return paths


class CudaRuntime:
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
                fail(f"CUDA runtime DLL not found: {path}")
            return path
        for path in candidate_cudart_paths():
            if path.is_file():
                return path
        fail("could not locate cudart DLL; pass --cudart")

    def _bind(self) -> None:
        lib = self.lib
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        lib.cudaMemGetInfo.restype = ctypes.c_int
        lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        lib.cudaHostAlloc.restype = ctypes.c_int
        lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        lib.cudaFreeHost.restype = ctypes.c_int
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamCreate.restype = ctypes.c_int
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaStreamDestroy.restype = ctypes.c_int
        lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaEventCreate.restype = ctypes.c_int
        lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaEventDestroy.restype = ctypes.c_int
        lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaEventRecord.restype = ctypes.c_int
        lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaEventSynchronize.restype = ctypes.c_int
        lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        lib.cudaEventElapsedTime.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.cudaMemcpy.restype = ctypes.c_int
        lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        lib.cudaMemsetAsync.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

    def check(self, code: int, what: str) -> None:
        if code == CUDA_SUCCESS:
            return
        raw = self.lib.cudaGetErrorString(code)
        message = raw.decode("utf-8", errors="replace") if raw else f"cuda error {code}"
        fail(f"{what} failed: {message} ({code})")

    def runtime_version(self) -> int:
        version = ctypes.c_int()
        self.check(self.lib.cudaRuntimeGetVersion(ctypes.byref(version)), "cudaRuntimeGetVersion")
        return int(version.value)

    def device_count(self) -> int:
        count = ctypes.c_int()
        self.check(self.lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        return int(count.value)

    def set_device(self, device: int) -> None:
        self.check(self.lib.cudaSetDevice(device), "cudaSetDevice")

    def mem_info(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
        return int(free.value), int(total.value)

    def host_alloc(self, n_bytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaHostAlloc(ctypes.byref(ptr), n_bytes, CUDA_HOST_ALLOC_DEFAULT), "cudaHostAlloc")
        return ptr

    def host_free(self, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            self.check(self.lib.cudaFreeHost(ptr), "cudaFreeHost")

    def device_alloc(self, n_bytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), n_bytes), "cudaMalloc")
        return ptr

    def device_free(self, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            self.check(self.lib.cudaFree(ptr), "cudaFree")

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self.check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def event_create(self) -> ctypes.c_void_p:
        event = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(event)), "cudaEventCreate")
        return event

    def event_destroy(self, event: ctypes.c_void_p) -> None:
        if event and event.value:
            self.check(self.lib.cudaEventDestroy(event), "cudaEventDestroy")

    def memcpy_h2d_timed(
        self,
        dst: ctypes.c_void_p,
        src: ctypes.c_void_p,
        n_bytes: int,
        stream: ctypes.c_void_p,
        start: ctypes.c_void_p,
        stop: ctypes.c_void_p,
    ) -> float:
        self.check(self.lib.cudaEventRecord(start, stream), "cudaEventRecord(start)")
        self.check(
            self.lib.cudaMemcpyAsync(dst, src, n_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, stream),
            "cudaMemcpyAsync(H2D)",
        )
        self.check(self.lib.cudaEventRecord(stop, stream), "cudaEventRecord(stop)")
        self.check(self.lib.cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")
        elapsed = ctypes.c_float()
        self.check(self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop), "cudaEventElapsedTime")
        return float(elapsed.value)

    def memset_async(self, dst: ctypes.c_void_p, value: int, n_bytes: int, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaMemsetAsync(dst, value, n_bytes, stream), "cudaMemsetAsync")

    def memcpy_d2h(self, dst: ctypes.c_void_p, src: ctypes.c_void_p, n_bytes: int) -> None:
        self.check(self.lib.cudaMemcpy(dst, src, n_bytes, CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy(D2H)")

    def device_synchronize(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def make_host_view(ptr: ctypes.c_void_p, n_bytes: int) -> memoryview:
    array_type = ctypes.c_ubyte * n_bytes
    array = array_type.from_address(int(ptr.value))
    return memoryview(array).cast("B")


def read_tensor_into_host_view(
    handle: Any,
    host_view: memoryview,
    write_offset: int,
    absolute_offset: int,
    n_bytes: int,
) -> int:
    target = host_view[write_offset : write_offset + n_bytes]
    handle.seek(absolute_offset)
    n_read = handle.readinto(target)
    if n_read != n_bytes:
        fail(f"short read at offset {absolute_offset}: {n_read} != {n_bytes}")
    return n_read


def run_cuda_staging(args: argparse.Namespace, ledger: dict[str, Any], model_path: Path) -> dict[str, Any]:
    cuda = CudaRuntime(args.cudart)
    device_count = cuda.device_count()
    if device_count <= 0:
        fail("no CUDA devices found")
    if args.device < 0 or args.device >= device_count:
        fail(f"--device must be between 0 and {device_count - 1}")
    cuda.set_device(args.device)
    version = cuda.runtime_version()
    free_before, total_vram = cuda.mem_info()

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

        staged_stages: list[CudaStagedStage] = []
        staged_pages: list[CudaStagedPage] = []
        started = time.perf_counter()

        with model_path.open("rb", buffering=0) as handle:
            for planned_stage in stages:
                stage_wall_start = time.perf_counter()
                stage_index = int(planned_stage.get("stage_index", len(staged_stages)))
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
                stage_read_ms = 0.0
                stage_page_records: list[CudaStagedPage] = []

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

                    page_offset = stage_offset
                    page_crc = 0
                    page_reads = 0
                    page_start = time.perf_counter()
                    for tensor in tensors:
                        absolute_offset = data_start + tensor.offset
                        read_tensor_into_host_view(
                            handle,
                            host_view,
                            stage_offset,
                            absolute_offset,
                            tensor.n_bytes,
                        )
                        segment = host_view[stage_offset : stage_offset + tensor.n_bytes]
                        page_crc = zlib.crc32(segment, page_crc)
                        stage_offset += tensor.n_bytes
                        page_reads += 1
                    page_read_ms = (time.perf_counter() - page_start) * 1000.0
                    stage_read_ms += page_read_ms
                    stage_reads += page_reads
                    staged_bytes = stage_offset - page_offset
                    stage_page_records.append(
                        CudaStagedPage(
                            page_id=page_id,
                            block_key=block_key,
                            planned_bytes=planned_page_bytes,
                            staged_bytes=staged_bytes,
                            n_tensors=len(tensors),
                            read_calls=page_reads,
                            host_read_ms=page_read_ms,
                            crc32=f"{page_crc & 0xFFFFFFFF:08x}",
                        )
                    )

                h2d_ms = cuda.memcpy_h2d_timed(
                    device_ptrs[buffer_index],
                    host_ptrs[buffer_index],
                    stage_offset,
                    streams[buffer_index],
                    start_events[buffer_index],
                    stop_events[buffer_index],
                )
                total_stage_ms = (time.perf_counter() - stage_wall_start) * 1000.0
                staged_stages.append(
                    CudaStagedStage(
                        stage_index=stage_index,
                        buffer=str(planned_stage.get("buffer", chr(ord("A") + buffer_index))),
                        planned_bytes=planned_stage_bytes,
                        staged_bytes=stage_offset,
                        n_pages=len(page_ids),
                        read_calls=stage_reads,
                        host_read_ms=stage_read_ms,
                        h2d_ms=h2d_ms,
                        total_stage_ms=total_stage_ms,
                        host_read_throughput_gib_s=bytes_to_gib(stage_offset) / (stage_read_ms / 1000.0)
                        if stage_read_ms > 0
                        else 0.0,
                        h2d_throughput_gib_s=bytes_to_gib(stage_offset) / (h2d_ms / 1000.0) if h2d_ms > 0 else 0.0,
                        page_ids=page_ids,
                    )
                )
                staged_pages.extend(stage_page_records)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        free_after, _total_after = cuda.mem_info()
    finally:
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

    staged_bytes_total = sum(stage.staged_bytes for stage in staged_stages)
    planned_bytes_total = sum(stage.planned_bytes for stage in staged_stages)
    h2d_ms_total = sum(stage.h2d_ms for stage in staged_stages)
    host_read_ms_total = sum(stage.host_read_ms for stage in staged_stages)
    max_live_bytes = max((stage.staged_bytes for stage in staged_stages), default=0)
    stage_byte_match = all(stage.planned_bytes == stage.staged_bytes for stage in staged_stages)
    page_byte_match = all(page.planned_bytes == page.staged_bytes for page in staged_pages)

    return {
        "schema": "sage-oracle-page-cuda-staging-v0",
        "status": "measured_cuda_h2d_not_sparse_compute",
        "source_page_ledger": str(Path(args.page_ledger).resolve()),
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "file_bytes": model_path.stat().st_size,
        },
        "cuda": {
            "runtime_dll": str(cuda.path),
            "runtime_version": version,
            "device": args.device,
            "device_count": device_count,
            "vram_total_bytes": total_vram,
            "vram_free_before_bytes": free_before,
            "vram_free_after_staging_bytes": free_after,
        },
        "limits": {
            "limit_stages": args.limit_stages,
            "max_gib": args.max_gib,
            "buffer_count": args.buffer_count,
            "stage_buffer_bytes": stage_buffer_bytes,
            "stage_buffer_gib": bytes_to_gib(stage_buffer_bytes),
            "allocated_pinned_host_bytes": stage_buffer_bytes * args.buffer_count,
            "allocated_device_buffer_bytes": stage_buffer_bytes * args.buffer_count,
        },
        "summary": {
            "stages_staged": len(staged_stages),
            "pages_staged": len(staged_pages),
            "planned_bytes": planned_bytes_total,
            "staged_bytes": staged_bytes_total,
            "staged_gib": bytes_to_gib(staged_bytes_total),
            "h2d_bytes": staged_bytes_total,
            "max_live_buffer_bytes": max_live_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_bytes),
            "read_calls": sum(stage.read_calls for stage in staged_stages),
            "elapsed_ms": elapsed_ms,
            "host_read_ms": host_read_ms_total,
            "h2d_ms": h2d_ms_total,
            "host_read_throughput_gib_s": bytes_to_gib(staged_bytes_total) / (host_read_ms_total / 1000.0)
            if host_read_ms_total > 0
            else 0.0,
            "h2d_throughput_gib_s": bytes_to_gib(staged_bytes_total) / (h2d_ms_total / 1000.0)
            if h2d_ms_total > 0
            else 0.0,
            "stage_byte_match": stage_byte_match,
            "page_byte_match": page_byte_match,
            "byte_budget_respected": max_live_bytes <= stage_buffer_bytes,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_execution_status": "not_implemented_sparse_compute",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_cuda_staging_smoke",
            "oracle_active_bytes": staged_bytes_total,
            "oracle_blocks": [page.block_key for page in staged_pages],
            "gpu_staged_bytes": max_live_bytes,
            "host_pinned_bytes": stage_buffer_bytes * args.buffer_count,
            "pcie_transfer_ms": h2d_ms_total,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
        },
        "stages": [asdict(stage) for stage in staged_stages],
        "pages": [asdict(page) for page in staged_pages],
    }


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    ledger = load_json(Path(args.page_ledger))
    if ledger.get("schema") != "sage-oracle-page-ledger-v0":
        fail("expected sage-oracle-page-ledger-v0 input")
    model_path = resolve_model_path(args, ledger)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")
    return run_cuda_staging(args, ledger, model_path)


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    limits = payload["limits"]
    cuda = payload["cuda"]
    print("# SAGE Oracle CUDA Staging Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- CUDA runtime: `{cuda['runtime_version']}`")
    print(f"- Stages staged: `{summary['stages_staged']}`")
    print(f"- Pages staged: `{summary['pages_staged']}`")
    print(f"- Bytes staged: `{summary['staged_gib']:.3f} GiB`")
    print(f"- Max live device buffer: `{summary['max_live_buffer_gib']:.3f} GiB` / `{limits['stage_buffer_gib']:.3f} GiB`")
    print(f"- Host read time: `{summary['host_read_ms']:.2f} ms`")
    print(f"- CUDA H2D time: `{summary['h2d_ms']:.2f} ms`")
    print(f"- CUDA H2D throughput: `{summary['h2d_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Byte match: `stage={summary['stage_byte_match']}`, `page={summary['page_byte_match']}`")
    print(f"- Sparse compute: `{summary['cuda_execution_status']}`")
    print()
    print("## CUDA Stages")
    print()
    print("| Stage | Buffer | Pages | Bytes | Host read | H2D | H2D throughput |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(
            f"| {stage['stage_index']} | {stage['buffer']} | {stage['n_pages']} | "
            f"{bytes_to_gib(stage['staged_bytes']):.3f} GiB | {stage['host_read_ms']:.2f} ms | "
            f"{stage['h2d_ms']:.2f} ms | {stage['h2d_throughput_gib_s']:.2f} GiB/s |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure GGUF sparse-oracle page staging through CUDA buffers.")
    parser.add_argument("--page-ledger", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--model", default="", help="override model path from the page ledger")
    parser.add_argument("--limit-stages", type=int, default=0, help="number of planned stages to execute; 0 means all")
    parser.add_argument("--max-gib", type=float, default=0.0, help="optional cap on selected planned stage bytes")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="override stage buffer size; default uses page ledger")
    parser.add_argument("--buffer-count", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudart", default="", help="explicit path to cudart DLL")
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
