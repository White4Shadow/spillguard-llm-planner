#!/usr/bin/env python3
"""
Measure a bounded hot/warm/cold KV tier packing smoke from real GGUF KV shapes.

This is not runtime-integrated attention. It verifies the byte mechanics behind
the SAGE KV plan: infer the oracle KV dimensions from GGUF tensors, generate a
bounded synthetic full-precision warm-KV sample, pack it to 2-bit codes, unpack
the codes, and scale the measured bytes to the planned warm tier.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf
from sage_kv_ledger import bytes_per_token, bytes_to_gib, infer_layer_kv_shapes
from sage_oracle_cuda_kernel_smoke import CudaDriver, NvrtcRuntime
from sage_oracle_cuda_staging import CudaRuntime, make_host_view


KERNEL_SOURCE = r"""
extern "C" __global__
void sage_kv_pack2(const unsigned char * src, unsigned char * dst, unsigned long long n_values) {
    unsigned long long out_i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long out_n = (n_values + 3ULL) / 4ULL;
    if (out_i >= out_n) {
        return;
    }
    unsigned long long value_i = out_i * 4ULL;
    unsigned char packed = 0;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        unsigned long long current = value_i + (unsigned long long) lane;
        if (current < n_values) {
            unsigned char code = src[current * 2ULL + 1ULL] & 3U;
            packed |= (unsigned char) (code << (lane * 2));
        }
    }
    dst[out_i] = packed;
}

extern "C" __global__
void sage_kv_unpack2(const unsigned char * packed, unsigned char * dst, unsigned long long n_values) {
    unsigned long long value_i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (value_i >= n_values) {
        return;
    }
    unsigned char byte = packed[value_i / 4ULL];
    unsigned int shift = (unsigned int) ((value_i & 3ULL) * 2ULL);
    dst[value_i] = (byte >> shift) & 3U;
}
"""


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def make_pattern(n_bytes: int, seed: int) -> bytearray:
    data = bytearray(n_bytes)
    state = seed & 0xFFFFFFFF
    for index in range(n_bytes):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        data[index] = (state >> 24) & 0xFF
    return data


def pack_2bit_from_fp16_bytes(fp16_bytes: bytearray) -> tuple[bytearray, int]:
    if len(fp16_bytes) % 2 != 0:
        fail("fp16 byte sample must contain complete 2-byte values")
    n_values = len(fp16_bytes) // 2
    out = bytearray((n_values + 3) // 4)
    checksum = 0
    out_index = 0
    for value_index in range(0, n_values, 4):
        packed = 0
        for lane in range(4):
            current = value_index + lane
            if current >= n_values:
                break
            code = fp16_bytes[current * 2 + 1] & 0x03
            packed |= code << (lane * 2)
            checksum = (checksum + code + current) & 0xFFFFFFFF
        out[out_index] = packed
        out_index += 1
    return out, checksum


def unpack_2bit_codes(packed: bytearray, n_values: int) -> tuple[bytearray, int]:
    out = bytearray(n_values)
    checksum = 0
    value_index = 0
    for packed_byte in packed:
        for lane in range(4):
            if value_index >= n_values:
                break
            code = (packed_byte >> (lane * 2)) & 0x03
            out[value_index] = code
            checksum = (checksum + code + value_index) & 0xFFFFFFFF
            value_index += 1
    return out, checksum


def timed(callable_obj: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callable_obj()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def event_elapsed_ms(cuda: CudaRuntime, start_event: ctypes.c_void_p, stop_event: ctypes.c_void_p) -> float:
    elapsed = ctypes.c_float()
    cuda.check(cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event), "cudaEventElapsedTime")
    return float(elapsed.value)


def launch_cuda_kernel(
    driver: CudaDriver,
    function: ctypes.c_void_p,
    a_ptr: ctypes.c_void_p,
    b_ptr: ctypes.c_void_p,
    n_values: int,
    stream: ctypes.c_void_p,
    grid: int,
    block: int,
    label: str,
) -> None:
    a_arg = ctypes.c_uint64(int(a_ptr.value))
    b_arg = ctypes.c_uint64(int(b_ptr.value))
    n_arg = ctypes.c_uint64(n_values)
    params = (ctypes.c_void_p * 3)(
        ctypes.cast(ctypes.byref(a_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(b_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
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
        f"cuLaunchKernel({label})",
    )


def run_cuda_pack_smoke(
    args: argparse.Namespace,
    sample: bytearray,
    cpu_packed: bytearray,
    sample_values: int,
    sample_full_bytes: int,
    sample_packed_bytes: int,
    sample_tokens: int,
    warm_tokens: int,
) -> dict[str, Any]:
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
    module, pack_function = driver.load_function(module_image, b"sage_kv_pack2")
    _module_2, unpack_function = module, ctypes.c_void_p()
    driver.check(driver.lib.cuModuleGetFunction(ctypes.byref(unpack_function), module, b"sage_kv_unpack2"), "cuModuleGetFunction(unpack)")

    host_ptr = ctypes.c_void_p()
    device_src = ctypes.c_void_p()
    device_packed = ctypes.c_void_p()
    device_unpacked = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    try:
        host_ptr = cuda.host_alloc(sample_full_bytes)
        host_view = make_host_view(host_ptr, sample_full_bytes)
        host_view[:] = sample
        device_src = cuda.device_alloc(sample_full_bytes)
        device_packed = cuda.device_alloc(sample_packed_bytes)
        device_unpacked = cuda.device_alloc(sample_values)
        stream = cuda.stream_create()
        start_event = cuda.event_create()
        stop_event = cuda.event_create()

        h2d_ms = cuda.memcpy_h2d_timed(device_src, host_ptr, sample_full_bytes, stream, start_event, stop_event)

        block = int(args.cuda_block_size)
        pack_grid = (sample_packed_bytes + block - 1) // block
        unpack_grid = (sample_values + block - 1) // block

        cuda.check(cuda.lib.cudaEventRecord(start_event, stream), "cudaEventRecord(pack start)")
        launch_cuda_kernel(
            driver,
            pack_function,
            device_src,
            device_packed,
            sample_values,
            stream,
            pack_grid,
            block,
            "sage_kv_pack2",
        )
        cuda.check(cuda.lib.cudaEventRecord(stop_event, stream), "cudaEventRecord(pack stop)")
        cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(pack)")
        pack_ms = event_elapsed_ms(cuda, start_event, stop_event)

        cuda.check(cuda.lib.cudaEventRecord(start_event, stream), "cudaEventRecord(unpack start)")
        launch_cuda_kernel(
            driver,
            unpack_function,
            device_packed,
            device_unpacked,
            sample_values,
            stream,
            unpack_grid,
            block,
            "sage_kv_unpack2",
        )
        cuda.check(cuda.lib.cudaEventRecord(stop_event, stream), "cudaEventRecord(unpack stop)")
        cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(unpack)")
        unpack_ms = event_elapsed_ms(cuda, start_event, stop_event)

        packed_out = (ctypes.c_ubyte * sample_packed_bytes)()
        cuda.memcpy_d2h(ctypes.cast(packed_out, ctypes.c_void_p), device_packed, sample_packed_bytes)
        packed_bytes = bytes(packed_out)
        match = packed_bytes == bytes(cpu_packed)
        checksum = sum(packed_bytes) & 0xFFFFFFFF

        free_after, _total_after = cuda.mem_info()
        return {
            "enabled": True,
            "device": args.device,
            "arch": arch,
            "module_image_kind": module_image_kind,
            "compile_log": compile_log.strip(),
            "free_vram_before": free_before,
            "free_vram_after": free_after,
            "total_vram": total_vram,
            "h2d_ms": h2d_ms,
            "pack_ms": pack_ms,
            "unpack_ms": unpack_ms,
            "scaled_full_warm_pack_ms": pack_ms * warm_tokens / sample_tokens if sample_tokens else 0.0,
            "scaled_full_warm_unpack_ms": unpack_ms * warm_tokens / sample_tokens if sample_tokens else 0.0,
            "pack_input_gib_per_sec": bytes_to_gib(sample_full_bytes) / max(pack_ms / 1000.0, 1.0e-9),
            "unpack_output_gib_per_sec": bytes_to_gib(sample_values) / max(unpack_ms / 1000.0, 1.0e-9),
            "packed_matches_cpu": match,
            "packed_checksum": checksum,
            "pack_grid": pack_grid,
            "unpack_grid": unpack_grid,
            "block": block,
        }
    finally:
        if start_event and start_event.value:
            cuda.event_destroy(start_event)
        if stop_event and stop_event.value:
            cuda.event_destroy(stop_event)
        if stream and stream.value:
            cuda.stream_destroy(stream)
        if device_unpacked and device_unpacked.value:
            cuda.device_free(device_unpacked)
        if device_packed and device_packed.value:
            cuda.device_free(device_packed)
        if device_src and device_src.value:
            cuda.device_free(device_src)
        if host_ptr and host_ptr.value:
            cuda.host_free(host_ptr)
        if module and module.value:
            driver.unload_module(module)


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.oracle_model)
    if not model_path.is_file():
        fail(f"oracle model not found: {model_path}")
    if args.context_tokens <= 0:
        fail("--context-tokens must be positive")
    if args.hot_recent_tokens < 0 or args.sink_tokens < 0 or args.warm_max_tokens < 0:
        fail("token tier sizes must be non-negative")
    if args.warm_bits != 2:
        fail("this smoke currently measures exactly 2-bit warm KV packing")
    if args.sample_tokens <= 0:
        fail("--sample-tokens must be positive")

    index = parse_gguf(model_path)
    shapes = infer_layer_kv_shapes(index)
    full_bpt = int(round(bytes_per_token(shapes, 16.0, 16.0)))
    packed_bpt = int(round(bytes_per_token(shapes, float(args.warm_bits), float(args.warm_bits))))

    hot_tokens = min(args.context_tokens, args.hot_recent_tokens + args.sink_tokens)
    warm_tokens = min(max(0, args.context_tokens - hot_tokens), args.warm_max_tokens)
    cold_tokens = max(0, args.context_tokens - hot_tokens - warm_tokens)
    sample_tokens = min(args.sample_tokens, max(1, warm_tokens if warm_tokens > 0 else args.sample_tokens))

    sample_full_bytes = sample_tokens * full_bpt
    sample_values = sample_full_bytes // 2
    expected_sample_packed_bytes = (sample_values + 3) // 4
    expected_full_warm_packed_bytes = warm_tokens * packed_bpt
    full_precision_context_bytes = args.context_tokens * full_bpt
    tier_total_bytes = hot_tokens * full_bpt + expected_full_warm_packed_bytes

    sample, generate_ms = timed(lambda: make_pattern(sample_full_bytes, args.seed))
    (packed, pack_checksum), pack_ms = timed(lambda: pack_2bit_from_fp16_bytes(sample))
    (unpacked, unpack_checksum), unpack_ms = timed(lambda: unpack_2bit_codes(packed, sample_values))
    _ = unpacked[:1]
    cuda_pack: dict[str, Any] = {"enabled": False}
    if args.cuda_pack:
        cuda_pack = run_cuda_pack_smoke(
            args,
            sample,
            packed,
            sample_values,
            sample_full_bytes,
            len(packed),
            sample_tokens,
            warm_tokens,
        )

    bytes_match_plan = len(packed) == expected_sample_packed_bytes
    compression_ratio = sample_full_bytes / max(len(packed), 1)
    pack_mib_s = sample_full_bytes / max(pack_ms, 1.0e-9) / 1000.0 / 1024.0
    unpack_mib_s = len(packed) / max(unpack_ms, 1.0e-9) / 1000.0 / 1024.0
    scaled_pack_ms = pack_ms * warm_tokens / sample_tokens if sample_tokens else 0.0
    scaled_unpack_ms = unpack_ms * warm_tokens / sample_tokens if sample_tokens else 0.0

    return {
        "schema": "sage-kv-tier-pack-smoke-v0",
        "status": "measured_synthetic_2bit_warm_kv_pack_not_runtime_integrated",
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "architecture": index.metadata.get("general.architecture", "unknown"),
            "layer_count": index.layer_count,
        },
        "params": {
            "context_tokens": args.context_tokens,
            "hot_recent_tokens": args.hot_recent_tokens,
            "sink_tokens": args.sink_tokens,
            "warm_max_tokens": args.warm_max_tokens,
            "warm_bits": args.warm_bits,
            "sample_tokens": sample_tokens,
            "seed": args.seed,
        },
        "kv_shape": {
            "layers": len(shapes),
            "sum_k_dim": sum(shape.k_dim for shape in shapes),
            "sum_v_dim": sum(shape.v_dim for shape in shapes),
            "full_precision_bytes_per_token": full_bpt,
            "packed_warm_bytes_per_token": packed_bpt,
            "first_layers": [shape.__dict__ for shape in shapes[:8]],
        },
        "summary": {
            "hot_tokens": hot_tokens,
            "warm_tokens": warm_tokens,
            "cold_tokens": cold_tokens,
            "sample_tokens": sample_tokens,
            "sample_full_bytes": sample_full_bytes,
            "sample_packed_bytes": len(packed),
            "sample_expected_packed_bytes": expected_sample_packed_bytes,
            "bytes_match_plan": bytes_match_plan,
            "compression_ratio_vs_fp16": compression_ratio,
            "full_precision_context_bytes": full_precision_context_bytes,
            "full_precision_context_gib": bytes_to_gib(full_precision_context_bytes),
            "estimated_hot_bytes": hot_tokens * full_bpt,
            "estimated_warm_packed_bytes": expected_full_warm_packed_bytes,
            "estimated_tier_total_bytes": tier_total_bytes,
            "estimated_tier_total_gib": bytes_to_gib(tier_total_bytes),
            "estimated_saved_percent_vs_full_precision": (
                100.0 * max(0, full_precision_context_bytes - tier_total_bytes) / full_precision_context_bytes
                if full_precision_context_bytes > 0
                else 0.0
            ),
            "generate_ms": generate_ms,
            "pack_ms": pack_ms,
            "unpack_ms": unpack_ms,
            "scaled_full_warm_pack_ms": scaled_pack_ms,
            "scaled_full_warm_unpack_ms": scaled_unpack_ms,
            "pack_input_mib_per_sec": pack_mib_s,
            "unpack_input_mib_per_sec": unpack_mib_s,
            "pack_checksum": pack_checksum,
            "unpack_checksum": unpack_checksum,
            "checksums_match": pack_checksum == unpack_checksum,
            "kv_byte_status": "measured_synthetic_pack_not_runtime_integrated",
        },
        "cuda_pack": cuda_pack,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# SAGE KV Tier Pack Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- Sample tokens: `{summary['sample_tokens']}`")
    print(f"- Sample full bytes: `{summary['sample_full_bytes']}`")
    print(f"- Sample packed bytes: `{summary['sample_packed_bytes']}`")
    print(f"- Compression ratio: `{summary['compression_ratio_vs_fp16']:.2f}x`")
    print(f"- Pack/unpack: `{summary['pack_ms']:.2f} ms` / `{summary['unpack_ms']:.2f} ms`")
    cuda_pack = payload.get("cuda_pack", {})
    if isinstance(cuda_pack, dict) and cuda_pack.get("enabled"):
        print(f"- CUDA pack/unpack: `{cuda_pack['pack_ms']:.4f} ms` / `{cuda_pack['unpack_ms']:.4f} ms`")
        print(f"- CUDA packed matches CPU: `{cuda_pack['packed_matches_cpu']}`")
    print(f"- Scaled warm pack/unpack: `{summary['scaled_full_warm_pack_ms']:.2f} ms` / `{summary['scaled_full_warm_unpack_ms']:.2f} ms`")
    print(f"- Estimated tier total: `{summary['estimated_tier_total_gib']:.3f} GiB`")
    print(f"- Saved vs full precision: `{summary['estimated_saved_percent_vs_full_precision']:.1f}%`")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure bounded 2-bit warm-KV packing from GGUF KV shapes.")
    parser.add_argument("--oracle-model", required=True)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--hot-recent-tokens", type=int, default=512)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--warm-max-tokens", type=int, default=3584)
    parser.add_argument("--warm-bits", type=int, default=2)
    parser.add_argument("--sample-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cuda-pack", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudart", default="")
    parser.add_argument("--nvrtc", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--cuda-block-size", type=int, default=256)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cuda_block_size <= 0:
        parser.error("--cuda-block-size must be positive")

    payload = make_payload(args)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
