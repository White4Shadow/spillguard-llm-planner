#!/usr/bin/env python3
"""
Run a CUDA Q4_0 matvec/scoring smoke over staged SAGE oracle pages.

This SAGE smoke combines real selected GGUF Q4_0 weights with an activation
vector and produces output scores. By default the activation is deterministic
synthetic data. Passing --activation-jsonl switches the kernel to a captured
runtime tensor vector, which is closer to the oracle path but still not full
transformer inference or real candidate scoring.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import TensorInfo, parse_gguf
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

__device__ float sage_synthetic_activation(int col) {
    int v = (col * 17 + 13) % 127;
    return ((float) v - 63.0f) * 0.001953125f;
}

__device__ float sage_activation(int col, const float * activation, int activation_n, int use_real_activation) {
    if (use_real_activation != 0) {
        return col >= 0 && col < activation_n ? activation[col] : 0.0f;
    }
    return sage_synthetic_activation(col);
}

extern "C" __global__
void sage_q4_0_matvec_score(
    const unsigned char * data,
    int n0,
    int n1,
    const float * activation,
    int activation_n,
    int use_real_activation,
    float * out_sum,
    float * out_abs,
    float * row_scores,
    int row_scores_base,
    int write_row_scores
) {
    __shared__ float local[256];
    int row = (int) blockIdx.x;
    int tid = (int) threadIdx.x;
    int blocks_per_row = n0 / 32;
    const unsigned char * row_data = data + (unsigned long long) row * (unsigned long long) blocks_per_row * 18ull;
    float acc = 0.0f;

    if (row < n1) {
        for (int qb = tid; qb < blocks_per_row; qb += (int) blockDim.x) {
            const unsigned char * block = row_data + (unsigned long long) qb * 18ull;
            unsigned short hb = ((unsigned short) block[0]) | (((unsigned short) block[1]) << 8);
            float d = sage_half_to_float(hb);

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                unsigned char q = block[2 + j];
                int col = qb * 32 + j * 2;
                float vlo = d * (float) ((int) (q & 0x0f) - 8);
                float vhi = d * (float) ((int) (q >> 4) - 8);
                acc += vlo * sage_activation(col, activation, activation_n, use_real_activation);
                acc += vhi * sage_activation(col + 1, activation, activation_n, use_real_activation);
            }
        }
    }

    local[tid] = acc;
    __syncthreads();

    for (int offset = (int) blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local[tid] += local[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0 && row < n1) {
        float score = local[0];
        atomicAdd(out_sum, score);
        atomicAdd(out_abs, fabsf(score));
        if (write_row_scores != 0 && row_scores != 0) {
            row_scores[row_scores_base + row] = score;
        }
    }
}
"""


@dataclass
class MatvecTensor:
    name: str
    component: str
    n0: int
    n1: int
    n_bytes: int
    offset_in_stage: int


@dataclass
class RowScore:
    row: int
    score: float
    abs_score: float


@dataclass
class TensorTopScores:
    name: str
    component: str
    n0: int
    n1: int
    score_offset: int
    top_scores: list[RowScore]


@dataclass
class CpuScoreCheck:
    tensor: str
    row: int
    cuda_score: float
    cpu_score: float
    abs_error: float
    rel_error: float
    passed: bool


@dataclass
class MatvecStage:
    stage_index: int
    buffer: str
    q4_0_bytes: int
    q4_0_tensors: int
    q4_0_values: int
    output_scores: int
    host_read_ms: float
    h2d_ms: float
    matvec_ms: float
    matvec_throughput_gib_s: float
    page_ids: list[int]
    tensors: list[MatvecTensor]
    top_scores: list[TensorTopScores]
    cpu_score_checks: list[CpuScoreCheck]


@dataclass
class ActivationVector:
    source_jsonl: str
    record_index: int
    sequence: int
    name: str
    dtype: str
    op: str
    shape: list[int]
    i1: int | None
    axis: str
    count: int
    emitted: int
    truncated: bool
    value_count: int


def make_host_view(ptr: ctypes.c_void_p, n_bytes: int) -> memoryview:
    array_type = ctypes.c_ubyte * n_bytes
    array = array_type.from_address(int(ptr.value))
    return memoryview(array).cast("B")


def load_activation_vector(path: Path, tensor_name: str, record_index: int) -> tuple[ActivationVector, list[float]]:
    if record_index < 0:
        fail("--activation-record-index must be non-negative")
    matched = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid activation JSONL at {path}:{line_number}: {exc}")
            if tensor_name and record.get("name") != tensor_name:
                continue
            if matched != record_index:
                matched += 1
                continue

            values = record.get("values")
            if record.get("dtype") != "f32":
                fail(f"activation record must be f32, got {record.get('dtype')!r}")
            if record.get("axis") != "i0":
                fail(f"activation record must use axis i0, got {record.get('axis')!r}")
            if bool(record.get("truncated")):
                fail("activation record is truncated; capture with a larger --tensor-values-limit")
            if not isinstance(values, list) or not values:
                fail("activation record has no values")
            emitted = int(record.get("emitted", len(values)))
            if emitted != len(values):
                fail(f"activation emitted count {emitted} does not match value count {len(values)}")
            clean_values = [float(value) for value in values]
            meta = ActivationVector(
                source_jsonl=str(path.resolve()),
                record_index=record_index,
                sequence=int(record.get("sequence", -1)),
                name=str(record.get("name", "")),
                dtype=str(record.get("dtype", "")),
                op=str(record.get("op", "")),
                shape=[int(dim) for dim in record.get("shape", [])],
                i1=int(record["i1"]) if record.get("i1") is not None else None,
                axis=str(record.get("axis", "")),
                count=int(record.get("count", len(clean_values))),
                emitted=emitted,
                truncated=bool(record.get("truncated")),
                value_count=len(clean_values),
            )
            return meta, clean_values
    name_part = f" named {tensor_name!r}" if tensor_name else ""
    fail(f"no activation record{name_part} at index {record_index} found in {path}")


def synthetic_activation(col: int) -> float:
    value = (col * 17 + 13) % 127
    return (float(value) - 63.0) / 512.0


def activation_at(col: int, activation_values: list[float], use_real_activation: bool) -> float:
    if use_real_activation:
        return activation_values[col] if 0 <= col < len(activation_values) else 0.0
    return synthetic_activation(col)


def half_to_float(data: memoryview, offset: int) -> float:
    return float(struct.unpack("<e", bytes(data[offset : offset + 2]))[0])


def cpu_q4_0_row_score(
    host_view: memoryview,
    tensor: MatvecTensor,
    row: int,
    activation_values: list[float],
    use_real_activation: bool,
) -> float:
    if row < 0 or row >= tensor.n1:
        fail(f"row {row} out of range for {tensor.name}")
    blocks_per_row = tensor.n0 // 32
    row_offset = tensor.offset_in_stage + row * blocks_per_row * 18
    acc = 0.0
    for qb in range(blocks_per_row):
        block_offset = row_offset + qb * 18
        d = half_to_float(host_view, block_offset)
        for j in range(16):
            q = int(host_view[block_offset + 2 + j])
            col = qb * 32 + j * 2
            vlo = d * float((q & 0x0F) - 8)
            vhi = d * float((q >> 4) - 8)
            acc += vlo * activation_at(col, activation_values, use_real_activation)
            acc += vhi * activation_at(col + 1, activation_values, use_real_activation)
    return acc


def top_scores_for_tensor(scores: list[float], tensor: MatvecTensor, score_offset: int, top_k: int) -> TensorTopScores:
    ranked = sorted(range(tensor.n1), key=lambda row: scores[score_offset + row], reverse=True)[:top_k]
    return TensorTopScores(
        name=tensor.name,
        component=tensor.component,
        n0=tensor.n0,
        n1=tensor.n1,
        score_offset=score_offset,
        top_scores=[
            RowScore(
                row=int(row),
                score=float(scores[score_offset + row]),
                abs_score=abs(float(scores[score_offset + row])),
            )
            for row in ranked
        ],
    )


def make_cpu_score_checks(
    host_view: memoryview,
    stage_tensors: list[MatvecTensor],
    stage_scores: list[float],
    activation_values: list[float],
    use_real_activation: bool,
    max_checks: int,
) -> list[CpuScoreCheck]:
    if max_checks <= 0:
        return []
    checks: list[CpuScoreCheck] = []
    score_offset = 0
    for tensor in stage_tensors:
        if len(checks) >= max_checks:
            break
        best_row = max(range(tensor.n1), key=lambda row: stage_scores[score_offset + row])
        cuda_score = float(stage_scores[score_offset + best_row])
        cpu_score = float(cpu_q4_0_row_score(host_view, tensor, best_row, activation_values, use_real_activation))
        abs_error = abs(cpu_score - cuda_score)
        rel_error = abs_error / max(abs(cpu_score), abs(cuda_score), 1.0)
        passed = math.isfinite(abs_error) and abs_error <= max(
            1.0e-3,
            2.0e-4 * max(abs(cpu_score), abs(cuda_score), 1.0),
        )
        checks.append(
            CpuScoreCheck(
                tensor=tensor.name,
                row=int(best_row),
                cuda_score=cuda_score,
                cpu_score=cpu_score,
                abs_error=float(abs_error),
                rel_error=float(rel_error),
                passed=passed,
            )
        )
        score_offset += tensor.n1
    return checks


def launch_matvec(
    driver: CudaDriver,
    function: ctypes.c_void_p,
    device_base: ctypes.c_void_p,
    tensor: MatvecTensor,
    activation_device: ctypes.c_void_p,
    activation_len: int,
    use_real_activation: bool,
    out_sum: ctypes.c_void_p,
    out_abs: ctypes.c_void_p,
    row_scores: ctypes.c_void_p,
    row_scores_base: int,
    write_row_scores: bool,
    stream: ctypes.c_void_p,
    block: int,
) -> None:
    data_arg = ctypes.c_uint64(int(device_base.value) + tensor.offset_in_stage)
    n0_arg = ctypes.c_int(tensor.n0)
    n1_arg = ctypes.c_int(tensor.n1)
    activation_arg = ctypes.c_uint64(int(activation_device.value) if activation_device and activation_device.value else 0)
    activation_n_arg = ctypes.c_int(activation_len)
    use_real_activation_arg = ctypes.c_int(1 if use_real_activation else 0)
    sum_arg = ctypes.c_uint64(int(out_sum.value))
    abs_arg = ctypes.c_uint64(int(out_abs.value))
    row_scores_arg = ctypes.c_uint64(int(row_scores.value) if row_scores and row_scores.value else 0)
    row_scores_base_arg = ctypes.c_int(row_scores_base)
    write_row_scores_arg = ctypes.c_int(1 if write_row_scores else 0)
    params = (ctypes.c_void_p * 11)(
        ctypes.cast(ctypes.byref(data_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n0_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n1_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(activation_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(activation_n_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(use_real_activation_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(sum_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(abs_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(row_scores_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(row_scores_base_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(write_row_scores_arg), ctypes.c_void_p),
    )
    driver.check(
        driver.lib.cuLaunchKernel(
            function,
            tensor.n1,
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
        "cuLaunchKernel(q4_0_matvec)",
    )


def tensor_key(tensor: TensorInfo) -> str:
    return f"blk.{tensor.layer}.{tensor.component}" if tensor.layer is not None else f"global.{tensor.component}"


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
    module, function = driver.load_function(module_image, b"sage_q4_0_matvec_score")

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

    activation_meta: ActivationVector | None = None
    activation_values: list[float] = []
    if args.activation_jsonl:
        activation_meta, activation_values = load_activation_vector(
            Path(args.activation_jsonl),
            args.activation_name,
            args.activation_record_index,
        )
    use_real_activation = activation_meta is not None
    activation_len = len(activation_values)

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
    activation_device = ctypes.c_void_p()
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
        if use_real_activation:
            activation_array = (ctypes.c_float * activation_len)(*activation_values)
            activation_n_bytes = ctypes.sizeof(activation_array)
            activation_device = cuda.device_alloc(activation_n_bytes)
            cuda.check(
                cuda.lib.cudaMemcpy(
                    activation_device,
                    ctypes.cast(activation_array, ctypes.c_void_p),
                    activation_n_bytes,
                    1,
                ),
                "cudaMemcpy(activation H2D)",
            )
        out_sum = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        out_abs = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        cuda.memset_async(out_sum, 0, ctypes.sizeof(ctypes.c_float), streams[0])
        cuda.memset_async(out_abs, 0, ctypes.sizeof(ctypes.c_float), streams[0])
        cuda.device_synchronize()

        matvec_stages: list[MatvecStage] = []
        skipped_tensors = 0
        skipped_bytes = 0
        skipped_width_tensors = 0
        skipped_width_bytes = 0
        started = time.perf_counter()

        with model_path.open("rb", buffering=0) as handle:
            for planned_stage in stages:
                stage_index = int(planned_stage.get("stage_index", len(matvec_stages)))
                buffer_index = stage_index % args.buffer_count
                host_view = host_views[buffer_index]
                page_ids = [int(page_id) for page_id in planned_stage.get("page_ids", [])]

                stage_offset = 0
                stage_tensors: list[MatvecTensor] = []
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
                        if tensor.tensor_type != "Q4_0" or len(tensor.shape) != 2:
                            skipped_tensors += 1
                            skipped_bytes += tensor.n_bytes
                            continue
                        n0, n1 = int(tensor.shape[0]), int(tensor.shape[1])
                        if use_real_activation and n0 != activation_len:
                            skipped_width_tensors += 1
                            skipped_width_bytes += tensor.n_bytes
                            continue
                        if n0 % 32 != 0 or tensor.n_bytes != n1 * (n0 // 32) * 18:
                            fail(f"unsupported Q4_0 matrix layout for {tensor.name}: shape={tensor.shape}")
                        if stage_offset + tensor.n_bytes > stage_buffer_bytes:
                            fail(f"stage {stage_index} Q4_0 bytes overflow staging buffer")
                        target = host_view[stage_offset : stage_offset + tensor.n_bytes]
                        handle.seek(data_start + tensor.offset)
                        n_read = handle.readinto(target)
                        if n_read != tensor.n_bytes:
                            fail(f"short read for tensor {tensor.name}: {n_read} != {tensor.n_bytes}")
                        stage_tensors.append(
                            MatvecTensor(
                                name=tensor.name,
                                component=tensor.component,
                                n0=n0,
                                n1=n1,
                                n_bytes=tensor.n_bytes,
                                offset_in_stage=stage_offset,
                            )
                        )
                        stage_offset += tensor.n_bytes
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

                cuda.lib.cudaEventRecord(start_event, stream)
                output_scores = sum(t.n1 for t in stage_tensors)
                write_row_scores = args.score_top_k > 0 or args.cpu_check_rows > 0
                row_scores_device = ctypes.c_void_p()
                stage_scores: list[float] = []
                if write_row_scores:
                    row_scores_device = cuda.device_alloc(output_scores * ctypes.sizeof(ctypes.c_float))
                    cuda.memset_async(row_scores_device, 0, output_scores * ctypes.sizeof(ctypes.c_float), stream)
                row_scores_base = 0
                for tensor in stage_tensors:
                    launch_matvec(
                        driver,
                        function,
                        device_ptrs[buffer_index],
                        tensor,
                        activation_device,
                        activation_len,
                        use_real_activation,
                        out_sum,
                        out_abs,
                        row_scores_device,
                        row_scores_base,
                        write_row_scores,
                        stream,
                        args.block_size,
                    )
                    row_scores_base += tensor.n1
                cuda.lib.cudaEventRecord(stop_event, stream)
                cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(q4_0_matvec)")
                elapsed = ctypes.c_float()
                cuda.check(
                    cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event),
                    "cudaEventElapsedTime(q4_0_matvec)",
                )
                matvec_ms = float(elapsed.value)
                top_scores: list[TensorTopScores] = []
                cpu_checks: list[CpuScoreCheck] = []
                if write_row_scores:
                    host_scores = (ctypes.c_float * output_scores)()
                    cuda.memcpy_d2h(
                        ctypes.cast(host_scores, ctypes.c_void_p),
                        row_scores_device,
                        output_scores * ctypes.sizeof(ctypes.c_float),
                    )
                    stage_scores = [float(host_scores[i]) for i in range(output_scores)]
                    if args.score_top_k > 0:
                        score_offset = 0
                        for tensor in stage_tensors[: args.max_score_tensors]:
                            top_scores.append(top_scores_for_tensor(stage_scores, tensor, score_offset, args.score_top_k))
                            score_offset += tensor.n1
                    if args.cpu_check_rows > 0:
                        cpu_checks = make_cpu_score_checks(
                            host_view,
                            stage_tensors,
                            stage_scores,
                            activation_values,
                            use_real_activation,
                            args.cpu_check_rows,
                        )
                    cuda.device_free(row_scores_device)
                q4_0_values = sum(t.n0 * t.n1 for t in stage_tensors)
                matvec_stages.append(
                    MatvecStage(
                        stage_index=stage_index,
                        buffer=str(planned_stage.get("buffer", chr(ord("A") + buffer_index))),
                        q4_0_bytes=stage_offset,
                        q4_0_tensors=len(stage_tensors),
                        q4_0_values=q4_0_values,
                        output_scores=output_scores,
                        host_read_ms=host_read_ms,
                        h2d_ms=h2d_ms,
                        matvec_ms=matvec_ms,
                        matvec_throughput_gib_s=bytes_to_gib(stage_offset) / (matvec_ms / 1000.0)
                        if matvec_ms > 0
                        else 0.0,
                        page_ids=page_ids,
                        tensors=stage_tensors[: args.max_tensor_records],
                        top_scores=top_scores,
                        cpu_score_checks=cpu_checks,
                    )
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        sum_out = ctypes.c_float()
        abs_out = ctypes.c_float()
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(sum_out), ctypes.c_void_p), out_sum, ctypes.sizeof(sum_out))
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(abs_out), ctypes.c_void_p), out_abs, ctypes.sizeof(abs_out))
        free_after, _total_after = cuda.mem_info()
    finally:
        if activation_device and activation_device.value:
            cuda.device_free(activation_device)
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

    q4_0_bytes = sum(stage.q4_0_bytes for stage in matvec_stages)
    q4_0_values = sum(stage.q4_0_values for stage in matvec_stages)
    output_scores = sum(stage.output_scores for stage in matvec_stages)
    h2d_ms = sum(stage.h2d_ms for stage in matvec_stages)
    matvec_ms = sum(stage.matvec_ms for stage in matvec_stages)
    host_read_ms = sum(stage.host_read_ms for stage in matvec_stages)
    max_live_bytes = max((stage.q4_0_bytes for stage in matvec_stages), default=0)
    if use_real_activation and not matvec_stages:
        fail(f"no Q4_0 matrices matched activation width {activation_len}")
    top_score_tensors = sum(len(stage.top_scores) for stage in matvec_stages)
    top_score_rows = sum(len(tensor.top_scores) for stage in matvec_stages for tensor in stage.top_scores)
    cpu_score_checks = [check for stage in matvec_stages for check in stage.cpu_score_checks]
    cpu_score_checks_passed = bool(cpu_score_checks) and all(check.passed for check in cpu_score_checks)
    max_cpu_abs_error = max((check.abs_error for check in cpu_score_checks), default=0.0)
    max_cpu_rel_error = max((check.rel_error for check in cpu_score_checks), default=0.0)
    ranked_scores_enabled = args.score_top_k > 0

    if use_real_activation and ranked_scores_enabled:
        schema = "sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0"
        status = "measured_cuda_q4_0_real_activation_ranked_scores_not_oracle_logits"
    elif use_real_activation:
        schema = "sage-oracle-page-cuda-real-activation-matvec-smoke-v0"
        status = "measured_cuda_q4_0_matvec_real_activation_not_oracle_logits"
    else:
        schema = "sage-oracle-page-cuda-matvec-smoke-v0"
        status = "measured_cuda_q4_0_matvec_synthetic_activation_not_transformer"
    cuda_kernel_status = (
        "measured_q4_0_matvec_real_activation_kernel"
        if use_real_activation
        else "measured_q4_0_matvec_synthetic_activation_kernel"
    )
    if use_real_activation and ranked_scores_enabled:
        oracle_mode = "sparse_page_cuda_q4_0_real_activation_ranked_matvec_smoke"
    elif use_real_activation:
        oracle_mode = "sparse_page_cuda_q4_0_real_activation_matvec_smoke"
    else:
        oracle_mode = "sparse_page_cuda_q4_0_matvec_smoke"

    return {
        "schema": schema,
        "status": status,
        "source_page_ledger": str(ledger_path.resolve()),
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "file_bytes": model_path.stat().st_size,
        },
        "activation": asdict(activation_meta)
        if activation_meta is not None
        else {
            "mode": "synthetic",
            "formula": "x[col]=(((col*17+13)%127)-63)/512",
            "value_count": 0,
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
            "vram_free_after_matvec_bytes": free_after,
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
            "max_tensor_records_per_stage": args.max_tensor_records,
            "score_top_k": args.score_top_k,
            "max_score_tensors_per_stage": args.max_score_tensors,
            "cpu_check_rows_per_stage": args.cpu_check_rows,
        },
        "summary": {
            "stages_staged": len(matvec_stages),
            "q4_0_tensors": sum(stage.q4_0_tensors for stage in matvec_stages),
            "q4_0_bytes": q4_0_bytes,
            "q4_0_gib": bytes_to_gib(q4_0_bytes),
            "q4_0_values": q4_0_values,
            "output_scores": output_scores,
            "skipped_non_q4_0_or_non_2d_tensors": skipped_tensors,
            "skipped_non_q4_0_or_non_2d_bytes": skipped_bytes,
            "skipped_activation_width_mismatch_tensors": skipped_width_tensors,
            "skipped_activation_width_mismatch_bytes": skipped_width_bytes,
            "max_live_buffer_bytes": max_live_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_bytes),
            "elapsed_ms": elapsed_ms,
            "host_read_ms": host_read_ms,
            "h2d_ms": h2d_ms,
            "matvec_ms": matvec_ms,
            "h2d_throughput_gib_s": bytes_to_gib(q4_0_bytes) / (h2d_ms / 1000.0) if h2d_ms > 0 else 0.0,
            "matvec_weight_throughput_gib_s": bytes_to_gib(q4_0_bytes) / (matvec_ms / 1000.0)
            if matvec_ms > 0
            else 0.0,
            "activation_mode": "real_tensor_values_jsonl" if use_real_activation else "synthetic_formula",
            "activation_width": activation_len,
            "synthetic_activation": "x[col]=(((col*17+13)%127)-63)/512" if not use_real_activation else "",
            "score_sum": float(sum_out.value),
            "score_abs_sum": float(abs_out.value),
            "score_output_nonzero": float(abs_out.value) > 0.0,
            "row_score_capture_status": "measured_per_row_scores" if (args.score_top_k > 0 or args.cpu_check_rows > 0) else "not_requested",
            "top_score_tensors": top_score_tensors,
            "top_score_rows": top_score_rows,
            "top_score_rank_by": "score_desc" if args.score_top_k > 0 else "",
            "cpu_score_checks": len(cpu_score_checks),
            "cpu_score_checks_passed": cpu_score_checks_passed,
            "max_cpu_score_abs_error": max_cpu_abs_error,
            "max_cpu_score_rel_error": max_cpu_rel_error,
            "byte_budget_respected": max_live_bytes <= stage_buffer_bytes,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_kernel_status": cuda_kernel_status,
            "real_activation_status": "measured_tensor_values_jsonl" if use_real_activation else "not_implemented",
            "candidate_scoring_status": "ranked_projection_rows_not_candidate_tokens" if ranked_scores_enabled else "not_implemented",
            "sparse_transformer_status": "not_implemented",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": oracle_mode,
            "oracle_active_bytes": q4_0_bytes,
            "gpu_staged_bytes": max_live_bytes,
            "host_pinned_bytes": stage_buffer_bytes * args.buffer_count,
            "pcie_transfer_ms": h2d_ms,
            "pcie_transfer_status": "measured_cuda_h2d_from_pinned_host",
            "cuda_matvec_ms": matvec_ms,
            "cuda_kernel_status": cuda_kernel_status,
            "activation_mode": "real_tensor_values_jsonl" if use_real_activation else "synthetic_formula",
            "activation_width": activation_len,
            "row_score_capture_status": "measured_per_row_scores" if (args.score_top_k > 0 or args.cpu_check_rows > 0) else "not_requested",
            "candidate_scoring_status": "ranked_projection_rows_not_candidate_tokens" if ranked_scores_enabled else "not_implemented",
        },
        "compile_log": compile_log,
        "stages": [asdict(stage) for stage in matvec_stages],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    limits = payload["limits"]
    cuda = payload["cuda"]
    print("# SAGE Oracle CUDA Q4_0 Matvec Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Model: `{payload['model']['name']}`")
    print(f"- Activation mode: `{summary['activation_mode']}`")
    if summary["activation_mode"] == "real_tensor_values_jsonl":
        print(f"- Activation source: `{payload['activation']['name']}` width `{summary['activation_width']}`")
    print(f"- CUDA arch: `{cuda['arch']}`")
    print(f"- Stages staged: `{summary['stages_staged']}`")
    print(f"- Q4_0 tensors: `{summary['q4_0_tensors']}`")
    print(f"- Q4_0 bytes: `{summary['q4_0_gib']:.3f} GiB`")
    print(f"- Matvec values: `{summary['q4_0_values']}`")
    print(f"- Output scores: `{summary['output_scores']}`")
    print(f"- Width-mismatch Q4_0 tensors skipped: `{summary['skipped_activation_width_mismatch_tensors']}`")
    print(f"- Max live device buffer: `{summary['max_live_buffer_gib']:.3f} GiB` / `{limits['stage_buffer_gib']:.3f} GiB`")
    print(f"- H2D time: `{summary['h2d_ms']:.2f} ms`")
    print(f"- Matvec time: `{summary['matvec_ms']:.2f} ms`")
    print(f"- Matvec weight throughput: `{summary['matvec_weight_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Score abs sum nonzero: `{summary['score_output_nonzero']}`")
    print(f"- Row score capture: `{summary['row_score_capture_status']}`")
    if summary["row_score_capture_status"] == "measured_per_row_scores":
        print(f"- Top-score tensors: `{summary['top_score_tensors']}`")
        print(f"- CPU score checks: `{summary['cpu_score_checks']}` passed `{summary['cpu_score_checks_passed']}`")
        print(f"- Max CPU abs error: `{summary['max_cpu_score_abs_error']:.6g}`")
    print(f"- Candidate scoring: `{summary['candidate_scoring_status']}`")
    print()
    print("## Matvec Stages")
    print()
    print("| Stage | Buffer | Q4_0 tensors | Q4_0 bytes | Scores | H2D | Matvec | Throughput |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for stage in payload["stages"]:
        print(
            f"| {stage['stage_index']} | {stage['buffer']} | {stage['q4_0_tensors']} | "
            f"{bytes_to_gib(stage['q4_0_bytes']):.3f} GiB | {stage['output_scores']} | "
            f"{stage['h2d_ms']:.2f} ms | {stage['matvec_ms']:.2f} ms | "
            f"{stage['matvec_throughput_gib_s']:.2f} GiB/s |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA Q4_0 matvec smoke over staged GGUF oracle pages.")
    parser.add_argument("--page-ledger", default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json")
    parser.add_argument("--model", default="", help="override model path from the page ledger")
    parser.add_argument("--limit-stages", type=int, default=0, help="number of planned stages to execute; 0 means all")
    parser.add_argument("--max-gib", type=float, default=0.0, help="optional cap on selected planned stage bytes")
    parser.add_argument("--stage-buffer-gib", type=float, default=0.0, help="override stage buffer size; default uses page ledger")
    parser.add_argument("--buffer-count", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="", help="NVRTC architecture such as sm_86; default queries CUDA driver")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-tensor-records", type=int, default=8)
    parser.add_argument("--score-top-k", type=int, default=0, help="capture top-k row scores per recorded tensor")
    parser.add_argument("--max-score-tensors", type=int, default=8, help="max tensors per stage to include in top-score records")
    parser.add_argument("--cpu-check-rows", type=int, default=0, help="CPU-check this many top rows per stage")
    parser.add_argument("--activation-jsonl", default="", help="optional tensor-values JSONL file captured from llama-debug")
    parser.add_argument("--activation-name", default="", help="optional tensor name filter inside --activation-jsonl")
    parser.add_argument("--activation-record-index", type=int, default=0)
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
    if args.max_tensor_records < 0:
        parser.error("--max-tensor-records must be non-negative")
    if args.score_top_k < 0:
        parser.error("--score-top-k must be non-negative")
    if args.max_score_tensors < 0:
        parser.error("--max-score-tensors must be non-negative")
    if args.cpu_check_rows < 0:
        parser.error("--cpu-check-rows must be non-negative")
    if args.activation_record_index < 0:
        parser.error("--activation-record-index must be non-negative")

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
