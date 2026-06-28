#!/usr/bin/env python3
"""
Run a CUDA Q6_K vocabulary-projection smoke over Gemma's tied token embedding.

Gemma 31B does not expose a separate Q4_0 output.weight tensor in this GGUF.
The tied vocabulary projection is token_embd.weight in Q6_K format. This smoke
pages that matrix in bounded chunks, multiplies rows by a captured activation
vector, emits top token ids, and CPU-checks sampled scores against raw GGUF
bytes. It is still not true oracle logits unless the activation is the final
post-norm hidden state from the same forward pass.
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
from sage_oracle_cuda_matvec_smoke import ActivationVector, half_to_float, load_activation_vector
from sage_oracle_cuda_staging import CudaRuntime
from sage_oracle_pager_staging import BYTES_PER_GIB, bytes_to_gib, gguf_data_start


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

__device__ float sage_q6_k_value(const unsigned char * block, int i) {
    const unsigned char * ql = block;
    const unsigned char * qh = block + 128;
    const signed char * scales = (const signed char *) (block + 192);
    unsigned short hb = ((unsigned short) block[208]) | (((unsigned short) block[209]) << 8);
    float d = sage_half_to_float(hb);

    int half = i >= 128 ? 1 : 0;
    int local = i - half * 128;
    int ql_base = half * 64;
    int qh_base = half * 32;
    int sc_base = half * 8;
    int l;
    int q;
    int sc;

    if (local < 32) {
        l = local;
        q = (int) ((ql[ql_base + l] & 0x0f) | (((qh[qh_base + l] >> 0) & 3) << 4));
        sc = (int) scales[sc_base + l / 16 + 0];
    } else if (local < 64) {
        l = local - 32;
        q = (int) ((ql[ql_base + l + 32] & 0x0f) | (((qh[qh_base + l] >> 2) & 3) << 4));
        sc = (int) scales[sc_base + l / 16 + 2];
    } else if (local < 96) {
        l = local - 64;
        q = (int) ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4));
        sc = (int) scales[sc_base + l / 16 + 4];
    } else {
        l = local - 96;
        q = (int) ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4));
        sc = (int) scales[sc_base + l / 16 + 6];
    }

    return d * (float) sc * (float) (q - 32);
}

extern "C" __global__
void sage_q6_k_vocab_score(
    const unsigned char * data,
    int n0,
    int n_rows,
    const float * activation,
    float * row_scores,
    float * out_sum,
    float * out_abs
) {
    __shared__ float local[256];
    int row = (int) blockIdx.x;
    int tid = (int) threadIdx.x;
    int blocks_per_row = n0 / 256;
    const unsigned char * row_data = data + (unsigned long long) row * (unsigned long long) blocks_per_row * 210ull;
    float acc = 0.0f;

    if (row < n_rows && tid < 256) {
        for (int qb = 0; qb < blocks_per_row; ++qb) {
            int col = qb * 256 + tid;
            const unsigned char * block = row_data + (unsigned long long) qb * 210ull;
            acc += sage_q6_k_value(block, tid) * activation[col];
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

    if (tid == 0 && row < n_rows) {
        float score = local[0];
        row_scores[row] = score;
        atomicAdd(out_sum, score);
        atomicAdd(out_abs, fabsf(score));
    }
}
"""


@dataclass
class TokenScore:
    token_id: int
    score: float
    abs_score: float
    raw_score: float


@dataclass
class CpuTokenCheck:
    token_id: int
    cuda_score: float
    cpu_score: float
    abs_error: float
    rel_error: float
    passed: bool


@dataclass
class VocabChunk:
    chunk_index: int
    row_start: int
    n_rows: int
    chunk_bytes: int
    host_read_ms: float
    h2d_ms: float
    kernel_ms: float
    top_scores: list[TokenScore]


@dataclass
class LogitScoreCheck:
    token_id: int
    sage_score: float
    llama_logit: float
    abs_error: float
    rel_error: float


@dataclass
class LogitComparison:
    loaded_logits: int
    top_k: int
    sage_top_ids: list[int]
    llama_top_ids: list[int]
    overlap_count: int
    overlap_rate: float
    top1_match: bool
    compared_llama_top: int
    max_abs_error: float
    mean_abs_error: float
    passed: bool
    score_checks: list[LogitScoreCheck]


def make_host_view(ptr: ctypes.c_void_p, n_bytes: int) -> memoryview:
    array_type = ctypes.c_ubyte * n_bytes
    array = array_type.from_address(int(ptr.value))
    return memoryview(array).cast("B")


def find_tensor(index_tensors: list[TensorInfo], name: str) -> TensorInfo:
    for tensor in index_tensors:
        if tensor.name == name:
            return tensor
    fail(f"tensor not found: {name}")


def q6_k_row_bytes(n0: int) -> int:
    if n0 % 256 != 0:
        fail(f"Q6_K row width must be divisible by 256, got {n0}")
    return (n0 // 256) * 210


def q6_k_value(block: memoryview, i: int) -> float:
    ql_offset = 0
    qh_offset = 128
    sc_offset = 192
    d = half_to_float(block, 208)
    half = 1 if i >= 128 else 0
    local = i - half * 128
    ql_base = ql_offset + half * 64
    qh_base = qh_offset + half * 32
    sc_base = sc_offset + half * 8

    if local < 32:
        l = local
        q = (int(block[ql_base + l]) & 0x0F) | (((int(block[qh_base + l]) >> 0) & 3) << 4)
        sc = ctypes.c_int8(int(block[sc_base + l // 16 + 0])).value
    elif local < 64:
        l = local - 32
        q = (int(block[ql_base + l + 32]) & 0x0F) | (((int(block[qh_base + l]) >> 2) & 3) << 4)
        sc = ctypes.c_int8(int(block[sc_base + l // 16 + 2])).value
    elif local < 96:
        l = local - 64
        q = (int(block[ql_base + l]) >> 4) | (((int(block[qh_base + l]) >> 4) & 3) << 4)
        sc = ctypes.c_int8(int(block[sc_base + l // 16 + 4])).value
    else:
        l = local - 96
        q = (int(block[ql_base + l + 32]) >> 4) | (((int(block[qh_base + l]) >> 6) & 3) << 4)
        sc = ctypes.c_int8(int(block[sc_base + l // 16 + 6])).value

    return d * float(sc) * float(q - 32)


def cpu_q6_k_row_score(row_data: bytes, n0: int, activation: list[float]) -> float:
    view = memoryview(row_data).cast("B")
    blocks_per_row = n0 // 256
    acc = 0.0
    for qb in range(blocks_per_row):
        block = view[qb * 210 : (qb + 1) * 210]
        for i in range(256):
            acc += q6_k_value(block, i) * activation[qb * 256 + i]
    view.release()
    return float(acc)


def launch_vocab_kernel(
    driver: CudaDriver,
    function: ctypes.c_void_p,
    device_data: ctypes.c_void_p,
    n0: int,
    n_rows: int,
    activation_device: ctypes.c_void_p,
    row_scores: ctypes.c_void_p,
    out_sum: ctypes.c_void_p,
    out_abs: ctypes.c_void_p,
    stream: ctypes.c_void_p,
) -> None:
    data_arg = ctypes.c_uint64(int(device_data.value))
    n0_arg = ctypes.c_int(n0)
    n_rows_arg = ctypes.c_int(n_rows)
    activation_arg = ctypes.c_uint64(int(activation_device.value))
    scores_arg = ctypes.c_uint64(int(row_scores.value))
    sum_arg = ctypes.c_uint64(int(out_sum.value))
    abs_arg = ctypes.c_uint64(int(out_abs.value))
    params = (ctypes.c_void_p * 7)(
        ctypes.cast(ctypes.byref(data_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n0_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_rows_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(activation_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(scores_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(sum_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(abs_arg), ctypes.c_void_p),
    )
    driver.check(
        driver.lib.cuLaunchKernel(
            function,
            n_rows,
            1,
            1,
            256,
            1,
            1,
            0,
            stream,
            params,
            None,
        ),
        "cuLaunchKernel(q6_k_vocab)",
    )


def top_scores(
    raw_scores: list[float],
    row_start: int,
    top_k: int,
    rank_scores: list[float] | None = None,
) -> list[TokenScore]:
    scores = raw_scores if rank_scores is None else rank_scores
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        TokenScore(
            token_id=row_start + int(i),
            score=float(scores[i]),
            abs_score=abs(float(scores[i])),
            raw_score=float(raw_scores[i]),
        )
        for i in ranked
    ]


def merge_top(existing: list[TokenScore], incoming: list[TokenScore], top_k: int) -> list[TokenScore]:
    return sorted(existing + incoming, key=lambda item: item.score, reverse=True)[:top_k]


def metadata_final_logit_softcap(metadata: dict[str, Any]) -> float:
    for key, value in metadata.items():
        if key.endswith(".final_logit_softcapping"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def metadata_suppress_tokens(metadata: dict[str, Any]) -> set[int]:
    value = metadata.get("tokenizer.ggml.suppress_tokens")
    if not isinstance(value, list):
        return set()
    tokens: set[int] = set()
    for item in value:
        try:
            tokens.add(int(item))
        except (TypeError, ValueError):
            continue
    return tokens


def apply_logit_transforms(
    raw_scores: list[float],
    row_start: int,
    softcap: float,
    suppress_tokens: set[int],
) -> list[float]:
    transformed: list[float] = []
    for idx, score in enumerate(raw_scores):
        value = float(score)
        if softcap > 0.0:
            value = softcap * math.tanh(value / softcap)
        if row_start + idx in suppress_tokens:
            value = -math.inf
        transformed.append(value)
    return transformed


def load_llamacpp_logits(path: Path, expected_rows: int) -> list[float]:
    if not path.is_file():
        fail(f"llama.cpp logits file not found: {path}")
    if path.suffix.lower() == ".bin":
        payload = path.read_bytes()
        expected_bytes = expected_rows * 4
        if len(payload) != expected_bytes:
            fail(f"logits bin size mismatch: {len(payload)} != {expected_bytes}")
        return list(struct.unpack(f"<{expected_rows}f", payload))

    logits = [0.0] * expected_rows
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if ":" not in line:
                continue
            idx_text, value_text = line.split(":", 1)
            try:
                idx = int(idx_text.strip())
                value = float(value_text.strip())
            except ValueError as exc:
                fail(f"invalid logits text line {line_no}: {line.strip()} ({exc})")
            if idx < 0 or idx >= expected_rows:
                fail(f"logits token id out of range on line {line_no}: {idx}")
            logits[idx] = value
            seen += 1
    if seen != expected_rows:
        fail(f"logits text count mismatch: {seen} != {expected_rows}")
    return logits


def logit_top_scores(logits: list[float], top_k: int) -> list[TokenScore]:
    def rank_value(idx: int) -> float:
        value = logits[idx]
        return -math.inf if math.isnan(value) else value

    ranked = sorted(range(len(logits)), key=rank_value, reverse=True)[:top_k]
    return [
        TokenScore(token_id=int(idx), score=float(logits[idx]), abs_score=abs(float(logits[idx])), raw_score=float(logits[idx]))
        for idx in ranked
    ]


def compare_logits(
    sage_top: list[TokenScore],
    llama_top: list[TokenScore],
    tracked_scores: dict[int, TokenScore],
    loaded_logits: int,
    top_k: int,
    max_abs_error: float,
) -> LogitComparison:
    sage_top_ids = [item.token_id for item in sage_top[:top_k]]
    llama_top_ids = [item.token_id for item in llama_top[:top_k]]
    overlap_count = len(set(sage_top_ids) & set(llama_top_ids))
    overlap_rate = overlap_count / max(len(llama_top_ids), 1)
    score_checks: list[LogitScoreCheck] = []

    for llama_item in llama_top[:top_k]:
        sage_item = tracked_scores.get(llama_item.token_id)
        if sage_item is None:
            continue
        if not math.isfinite(llama_item.score) or not math.isfinite(sage_item.score):
            continue
        abs_error = abs(sage_item.score - llama_item.score)
        rel_error = abs_error / max(abs(sage_item.score), abs(llama_item.score), 1.0)
        score_checks.append(
            LogitScoreCheck(
                token_id=llama_item.token_id,
                sage_score=sage_item.score,
                llama_logit=llama_item.score,
                abs_error=float(abs_error),
                rel_error=float(rel_error),
            )
        )

    max_error = max((item.abs_error for item in score_checks), default=0.0)
    mean_error = sum(item.abs_error for item in score_checks) / len(score_checks) if score_checks else 0.0
    top1_match = bool(sage_top_ids and llama_top_ids and sage_top_ids[0] == llama_top_ids[0])
    passed = (
        top1_match
        and overlap_count == len(llama_top_ids)
        and len(score_checks) == len(llama_top_ids)
        and max_error <= max_abs_error
    )
    return LogitComparison(
        loaded_logits=loaded_logits,
        top_k=top_k,
        sage_top_ids=sage_top_ids,
        llama_top_ids=llama_top_ids,
        overlap_count=overlap_count,
        overlap_rate=float(overlap_rate),
        top1_match=top1_match,
        compared_llama_top=len(score_checks),
        max_abs_error=float(max_error),
        mean_abs_error=float(mean_error),
        passed=passed,
        score_checks=score_checks,
    )


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model)
    if not model_path.is_file():
        fail(f"model not found: {model_path}")
    activation_meta, activation_values = load_activation_vector(
        Path(args.activation_jsonl),
        args.activation_name,
        args.activation_record_index,
    )

    index = parse_gguf(model_path, keep_metadata_arrays=True)
    tensor = find_tensor(index.tensors, args.tensor_name)
    if tensor.tensor_type != "Q6_K":
        fail(f"{tensor.name} must be Q6_K, got {tensor.tensor_type}")
    if len(tensor.shape) != 2:
        fail(f"{tensor.name} must be 2D, got shape={tensor.shape}")
    n0, n1 = int(tensor.shape[0]), int(tensor.shape[1])
    if len(activation_values) != n0:
        fail(f"activation width {len(activation_values)} does not match {tensor.name} width {n0}")
    row_bytes = q6_k_row_bytes(n0)
    if tensor.n_bytes != row_bytes * n1:
        fail(f"unexpected Q6_K bytes for {tensor.name}: {tensor.n_bytes} != {row_bytes * n1}")

    softcap = metadata_final_logit_softcap(index.metadata) if not args.no_logit_transforms else 0.0
    suppress_tokens = metadata_suppress_tokens(index.metadata) if not args.no_logit_transforms else set()
    logits_path = Path(args.llamacpp_logits_bin or args.llamacpp_logits_txt) if (
        args.llamacpp_logits_bin or args.llamacpp_logits_txt
    ) else None
    llama_logits = load_llamacpp_logits(logits_path, n1) if logits_path else []
    llama_top = logit_top_scores(llama_logits, args.logit_top_k) if llama_logits else []
    tracked_token_ids = {item.token_id for item in llama_top}
    tracked_scores: dict[int, TokenScore] = {}

    stage_buffer_bytes = int(args.stage_buffer_gib * BYTES_PER_GIB)
    if stage_buffer_bytes < row_bytes:
        fail("--stage-buffer-gib is too small for one Q6_K row")
    rows_per_chunk = max(1, min(n1, stage_buffer_bytes // row_bytes))
    if args.limit_rows > 0:
        n1_effective = min(n1, args.limit_rows)
    else:
        n1_effective = n1
    chunk_count = math.ceil(n1_effective / rows_per_chunk)

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
    module, function = driver.load_function(module_image, b"sage_q6_k_vocab_score")

    host_ptr = ctypes.c_void_p()
    device_ptr = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    activation_device = ctypes.c_void_p()
    row_scores_device = ctypes.c_void_p()
    out_sum = ctypes.c_void_p()
    out_abs = ctypes.c_void_p()
    host_view: memoryview | None = None

    chunks: list[VocabChunk] = []
    global_top: list[TokenScore] = []

    try:
        host_ptr = cuda.host_alloc(stage_buffer_bytes)
        device_ptr = cuda.device_alloc(stage_buffer_bytes)
        stream = cuda.stream_create()
        start_event = cuda.event_create()
        stop_event = cuda.event_create()
        host_view = make_host_view(host_ptr, stage_buffer_bytes)
        max_scores_bytes = rows_per_chunk * ctypes.sizeof(ctypes.c_float)
        row_scores_device = cuda.device_alloc(max_scores_bytes)
        out_sum = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        out_abs = cuda.device_alloc(ctypes.sizeof(ctypes.c_float))
        activation_array = (ctypes.c_float * len(activation_values))(*activation_values)
        activation_device = cuda.device_alloc(ctypes.sizeof(activation_array))
        cuda.check(
            cuda.lib.cudaMemcpy(
                activation_device,
                ctypes.cast(activation_array, ctypes.c_void_p),
                ctypes.sizeof(activation_array),
                1,
            ),
            "cudaMemcpy(activation H2D)",
        )
        cuda.memset_async(out_sum, 0, ctypes.sizeof(ctypes.c_float), stream)
        cuda.memset_async(out_abs, 0, ctypes.sizeof(ctypes.c_float), stream)
        cuda.device_synchronize()

        data_start = gguf_data_start(model_path)
        with model_path.open("rb", buffering=0) as handle:
            for chunk_index in range(chunk_count):
                row_start = chunk_index * rows_per_chunk
                n_rows = min(rows_per_chunk, n1_effective - row_start)
                chunk_bytes = n_rows * row_bytes
                read_start = time.perf_counter()
                handle.seek(data_start + tensor.offset + row_start * row_bytes)
                n_read = handle.readinto(host_view[:chunk_bytes])
                if n_read != chunk_bytes:
                    fail(f"short read for {tensor.name} chunk {chunk_index}: {n_read} != {chunk_bytes}")
                host_read_ms = (time.perf_counter() - read_start) * 1000.0
                h2d_ms = cuda.memcpy_h2d_timed(device_ptr, host_ptr, chunk_bytes, stream, start_event, stop_event)
                cuda.memset_async(row_scores_device, 0, n_rows * ctypes.sizeof(ctypes.c_float), stream)
                cuda.lib.cudaEventRecord(start_event, stream)
                launch_vocab_kernel(
                    driver,
                    function,
                    device_ptr,
                    n0,
                    n_rows,
                    activation_device,
                    row_scores_device,
                    out_sum,
                    out_abs,
                    stream,
                )
                cuda.lib.cudaEventRecord(stop_event, stream)
                cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(q6_k_vocab)")
                elapsed = ctypes.c_float()
                cuda.check(
                    cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event),
                    "cudaEventElapsedTime(q6_k_vocab)",
                )
                kernel_ms = float(elapsed.value)
                host_scores = (ctypes.c_float * n_rows)()
                cuda.memcpy_d2h(
                    ctypes.cast(host_scores, ctypes.c_void_p),
                    row_scores_device,
                    n_rows * ctypes.sizeof(ctypes.c_float),
                )
                scores = [float(host_scores[i]) for i in range(n_rows)]
                rank_scores = apply_logit_transforms(scores, row_start, softcap, suppress_tokens)
                for token_id in tracked_token_ids:
                    if row_start <= token_id < row_start + n_rows:
                        local_id = token_id - row_start
                        tracked_scores[token_id] = TokenScore(
                            token_id=token_id,
                            score=float(rank_scores[local_id]),
                            abs_score=abs(float(rank_scores[local_id])),
                            raw_score=float(scores[local_id]),
                        )
                chunk_top = top_scores(scores, row_start, args.top_k, rank_scores)
                global_top = merge_top(global_top, chunk_top, args.top_k)
                chunks.append(
                    VocabChunk(
                        chunk_index=chunk_index,
                        row_start=row_start,
                        n_rows=n_rows,
                        chunk_bytes=chunk_bytes,
                        host_read_ms=host_read_ms,
                        h2d_ms=h2d_ms,
                        kernel_ms=kernel_ms,
                        top_scores=chunk_top,
                    )
                )

        sum_out = ctypes.c_float()
        abs_out = ctypes.c_float()
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(sum_out), ctypes.c_void_p), out_sum, ctypes.sizeof(sum_out))
        cuda.memcpy_d2h(ctypes.cast(ctypes.byref(abs_out), ctypes.c_void_p), out_abs, ctypes.sizeof(abs_out))
        free_after, _ = cuda.mem_info()
    finally:
        if host_view is not None:
            host_view.release()
        if out_abs and out_abs.value:
            cuda.device_free(out_abs)
        if out_sum and out_sum.value:
            cuda.device_free(out_sum)
        if row_scores_device and row_scores_device.value:
            cuda.device_free(row_scores_device)
        if activation_device and activation_device.value:
            cuda.device_free(activation_device)
        if stop_event and stop_event.value:
            cuda.event_destroy(stop_event)
        if start_event and start_event.value:
            cuda.event_destroy(start_event)
        if stream and stream.value:
            cuda.stream_destroy(stream)
        if device_ptr and device_ptr.value:
            cuda.device_free(device_ptr)
        if host_ptr and host_ptr.value:
            cuda.host_free(host_ptr)
        driver.unload_module(module)

    cpu_checks: list[CpuTokenCheck] = []
    data_start = gguf_data_start(model_path)
    with model_path.open("rb", buffering=0) as handle:
        for item in global_top[: args.cpu_check_top_k]:
            handle.seek(data_start + tensor.offset + item.token_id * row_bytes)
            row_data = handle.read(row_bytes)
            if len(row_data) != row_bytes:
                fail(f"short CPU check read for token {item.token_id}")
            cpu_score = cpu_q6_k_row_score(row_data, n0, activation_values)
            abs_error = abs(cpu_score - item.raw_score)
            rel_error = abs_error / max(abs(cpu_score), abs(item.raw_score), 1.0)
            passed = abs_error <= max(1.0e-3, 2.0e-4 * max(abs(cpu_score), abs(item.raw_score), 1.0))
            cpu_checks.append(
                CpuTokenCheck(
                    token_id=item.token_id,
                    cuda_score=item.raw_score,
                    cpu_score=cpu_score,
                    abs_error=float(abs_error),
                    rel_error=float(rel_error),
                    passed=passed,
                )
            )

    total_bytes = sum(chunk.chunk_bytes for chunk in chunks)
    h2d_ms = sum(chunk.h2d_ms for chunk in chunks)
    kernel_ms = sum(chunk.kernel_ms for chunk in chunks)
    host_read_ms = sum(chunk.host_read_ms for chunk in chunks)
    max_live_bytes = max((chunk.chunk_bytes for chunk in chunks), default=0)
    cpu_checks_passed = bool(cpu_checks) and all(check.passed for check in cpu_checks)
    max_cpu_abs_error = max((check.abs_error for check in cpu_checks), default=0.0)
    logit_comparison = (
        compare_logits(
            global_top,
            llama_top,
            tracked_scores,
            len(llama_logits),
            args.logit_top_k,
            args.logit_max_abs_error,
        )
        if llama_top
        else None
    )
    status = (
        "measured_q6_k_tied_vocab_projection_with_llamacpp_logit_compare"
        if logit_comparison is not None
        else "measured_q6_k_tied_vocab_projection_not_true_logits"
    )
    candidate_scoring_status = (
        "vocab_projection_compared_to_llamacpp_logits"
        if logit_comparison is not None
        else "vocab_token_scores_from_captured_activation_not_oracle_logits"
    )
    true_logit_status = (
        "compared_against_llamacpp_logits"
        if logit_comparison is not None
        else "not_implemented_final_hidden_state_missing"
    )

    return {
        "schema": "sage-oracle-page-cuda-q6-k-vocab-projection-smoke-v0",
        "status": status,
        "model": {
            "path": str(model_path.resolve()),
            "name": model_path.name,
            "file_bytes": model_path.stat().st_size,
        },
        "tensor": asdict(tensor),
        "activation": asdict(activation_meta),
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
            "stage_buffer_gib": bytes_to_gib(stage_buffer_bytes),
            "rows_per_chunk": rows_per_chunk,
            "limit_rows": args.limit_rows,
            "top_k": args.top_k,
            "cpu_check_top_k": args.cpu_check_top_k,
        },
        "summary": {
            "tensor_type": tensor.tensor_type,
            "n0": n0,
            "vocab_rows_scored": n1_effective,
            "row_bytes": row_bytes,
            "chunks": len(chunks),
            "staged_bytes": total_bytes,
            "staged_gib": bytes_to_gib(total_bytes),
            "max_live_buffer_bytes": max_live_bytes,
            "max_live_buffer_gib": bytes_to_gib(max_live_bytes),
            "host_read_ms": host_read_ms,
            "h2d_ms": h2d_ms,
            "kernel_ms": kernel_ms,
            "kernel_weight_throughput_gib_s": bytes_to_gib(total_bytes) / (kernel_ms / 1000.0)
            if kernel_ms > 0
            else 0.0,
            "top_tokens": len(global_top),
            "cpu_score_checks": len(cpu_checks),
            "cpu_score_checks_passed": cpu_checks_passed,
            "max_cpu_score_abs_error": max_cpu_abs_error,
            "score_sum": float(sum_out.value),
            "score_abs_sum": float(abs_out.value),
            "score_output_nonzero": float(abs_out.value) > 0.0,
            "byte_budget_respected": max_live_bytes <= stage_buffer_bytes,
            "activation_mode": "real_tensor_values_jsonl",
            "activation_width": len(activation_values),
            "candidate_scoring_status": candidate_scoring_status,
            "true_logit_status": true_logit_status,
            "logit_transform_softcap": softcap,
            "suppress_token_count": len(suppress_tokens),
            "llamacpp_logit_compare_passed": logit_comparison.passed if logit_comparison else False,
            "llamacpp_logit_top1_match": logit_comparison.top1_match if logit_comparison else False,
            "llamacpp_logit_overlap_rate": logit_comparison.overlap_rate if logit_comparison else 0.0,
            "llamacpp_logit_max_abs_error": logit_comparison.max_abs_error if logit_comparison else 0.0,
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_page_cuda_q6_k_vocab_projection_smoke",
            "oracle_active_bytes": total_bytes,
            "gpu_staged_bytes": max_live_bytes,
            "host_pinned_bytes": stage_buffer_bytes,
            "pcie_transfer_ms": h2d_ms,
            "cuda_kernel_ms": kernel_ms,
            "candidate_scoring_status": candidate_scoring_status,
            "true_logit_status": true_logit_status,
        },
        "logit_transform": {
            "softcap": softcap,
            "suppress_tokens": sorted(suppress_tokens),
            "disabled": bool(args.no_logit_transforms),
        },
        "llamacpp_logits": {
            "path": str(logits_path.resolve()) if logits_path else "",
            "loaded_logits": len(llama_logits),
        },
        "llamacpp_logit_comparison": asdict(logit_comparison) if logit_comparison else None,
        "top_tokens": [asdict(item) for item in global_top],
        "cpu_score_checks": [asdict(item) for item in cpu_checks],
        "chunks": [asdict(chunk) for chunk in chunks],
        "compile_log": compile_log,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# SAGE Q6_K Vocab Projection Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Tensor: `{payload['tensor']['name']}` `{summary['tensor_type']}`")
    print(f"- Vocab rows scored: `{summary['vocab_rows_scored']}`")
    print(f"- Chunks: `{summary['chunks']}`")
    print(f"- Staged bytes: `{summary['staged_gib']:.3f} GiB`")
    print(f"- Max live buffer: `{summary['max_live_buffer_gib']:.3f} GiB`")
    print(f"- H2D time: `{summary['h2d_ms']:.2f} ms`")
    print(f"- Kernel time: `{summary['kernel_ms']:.2f} ms`")
    print(f"- Kernel throughput: `{summary['kernel_weight_throughput_gib_s']:.2f} GiB/s`")
    print(f"- Top tokens: `{summary['top_tokens']}`")
    print(f"- CPU checks: `{summary['cpu_score_checks']}` passed `{summary['cpu_score_checks_passed']}`")
    print(f"- Candidate scoring: `{summary['candidate_scoring_status']}`")
    if payload["llamacpp_logit_comparison"]:
        comparison = payload["llamacpp_logit_comparison"]
        print(f"- llama.cpp top-1 match: `{comparison['top1_match']}`")
        print(f"- llama.cpp overlap@{comparison['top_k']}: `{comparison['overlap_count']}/{comparison['top_k']}`")
        print(f"- llama.cpp max abs error: `{comparison['max_abs_error']:.6g}`")
    print()
    print("## Top Tokens")
    print()
    print("| Rank | Token id | Score |")
    print("| ---: | ---: | ---: |")
    for rank, item in enumerate(payload["top_tokens"], 1):
        print(f"| {rank} | {item['token_id']} | {item['score']:.6g} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA Q6_K vocab projection smoke.")
    parser.add_argument("--model", default="models/gemma-4-31b-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf")
    parser.add_argument("--tensor-name", default="token_embd.weight")
    parser.add_argument("--activation-jsonl", default="benchmarks/sage-gemma31b-ffn-norm0-values-5376.jsonl")
    parser.add_argument("--activation-name", default="ffn_norm-0")
    parser.add_argument("--activation-record-index", type=int, default=0)
    parser.add_argument("--stage-buffer-gib", type=float, default=0.75)
    parser.add_argument("--limit-rows", type=int, default=0, help="debug cap; 0 scores the full vocab tensor")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cpu-check-top-k", type=int, default=8)
    parser.add_argument("--llamacpp-logits-bin", default="", help="llama-debug .bin logits to compare against")
    parser.add_argument("--llamacpp-logits-txt", default="", help="llama-debug .txt logits to compare against")
    parser.add_argument("--logit-top-k", type=int, default=10)
    parser.add_argument("--logit-max-abs-error", type=float, default=0.05)
    parser.add_argument("--no-logit-transforms", action="store_true", help="do not apply GGUF softcap/suppress-token transforms")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="", help="NVRTC architecture such as sm_86; default queries CUDA driver")
    parser.add_argument("--cudart", default="", help="explicit path to cudart DLL")
    parser.add_argument("--nvrtc", default="", help="explicit path to nvrtc DLL")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.stage_buffer_gib <= 0:
        parser.error("--stage-buffer-gib must be positive")
    if args.limit_rows < 0:
        parser.error("--limit-rows must be non-negative")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.cpu_check_top_k < 0:
        parser.error("--cpu-check-top-k must be non-negative")
    if args.activation_record_index < 0:
        parser.error("--activation-record-index must be non-negative")
    if args.llamacpp_logits_bin and args.llamacpp_logits_txt:
        parser.error("pass only one of --llamacpp-logits-bin or --llamacpp-logits-txt")
    if args.logit_top_k <= 0:
        parser.error("--logit-top-k must be positive")
    if args.logit_max_abs_error < 0:
        parser.error("--logit-max-abs-error must be non-negative")

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
