#!/usr/bin/env python3
"""
Score a sparse set of Gemma Q6_K vocabulary rows against a captured hidden state.

This is the candidate-verifier form of the Q6_K vocab smoke. Instead of paging
the full tied vocabulary matrix, it reads only selected token rows, packs them
into a small pinned host/device buffer, runs the same CUDA Q6_K row-score kernel,
and compares the selected scores to llama.cpp logits when available.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import TensorInfo, parse_gguf
from sage_oracle_cuda_kernel_smoke import CudaDriver, NvrtcRuntime, fail
from sage_oracle_cuda_staging import CudaRuntime
from sage_oracle_cuda_vocab_smoke import (
    KERNEL_SOURCE,
    cpu_q6_k_row_score,
    find_tensor,
    load_llamacpp_logits,
    logit_top_scores,
    make_host_view,
    metadata_final_logit_softcap,
    metadata_suppress_tokens,
    q6_k_row_bytes,
    launch_vocab_kernel,
)
from sage_oracle_cuda_matvec_smoke import load_activation_vector
from sage_oracle_pager_staging import BYTES_PER_GIB, bytes_to_gib, gguf_data_start


@dataclass
class CandidateScore:
    token_id: int
    packed_row: int
    raw_score: float
    sage_logit: float
    llama_logit: float | None
    abs_error: float | None
    rel_error: float | None
    cpu_score: float
    cpu_abs_error: float
    cpu_passed: bool


def parse_token_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    token_ids: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            token_id = int(part)
        except ValueError as exc:
            fail(f"invalid token id {part!r}: {exc}")
        token_ids.append(token_id)
    return token_ids


def unique_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_live_trace_candidates(
    path_text: str,
    step_index: int,
    source: str,
    max_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    if not path_text:
        return [], {}
    path = Path(path_text)
    if not path.is_file():
        fail(f"candidate live trace not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid candidate live trace JSON {path}: {exc}")
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        fail(f"{path} does not contain a steps list")
    if step_index < 0 or step_index >= len(steps):
        fail(f"--candidate-live-step-index {step_index} out of range for {len(steps)} steps")
    step = steps[step_index]
    if not isinstance(step, dict):
        fail(f"step {step_index} in {path} is not an object")
    row = step.get(source)
    if not isinstance(row, dict):
        fail(f"step {step_index} has no object source {source!r}")
    shortlist = row.get("candidate_shortlist")
    if not isinstance(shortlist, dict):
        fail(f"step {step_index} source {source!r} has no candidate_shortlist object")
    ids_raw = shortlist.get("candidate_token_ids")
    if not isinstance(ids_raw, list):
        fail(f"step {step_index} source {source!r} candidate_shortlist has no candidate_token_ids list")
    expected_rows = shortlist.get("candidate_rows")
    if expected_rows is not None and int(expected_rows) != len(ids_raw):
        fail(f"candidate_rows={expected_rows} does not match candidate_token_ids={len(ids_raw)}")

    token_ids: list[int] = []
    for offset, value in enumerate(ids_raw):
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            fail(f"invalid live trace token id at offset {offset}: {value!r}: {exc}")
        token_ids.append(token_id)
    limited_ids = token_ids[:max_rows] if max_rows > 0 else token_ids
    top_ids_raw = row.get("logit_top_ids", [])
    top_ids: list[int] = []
    if isinstance(top_ids_raw, list):
        for value in top_ids_raw:
            try:
                top_ids.append(int(value))
            except (TypeError, ValueError):
                top_ids = []
                break

    return limited_ids, {
        "path": str(path.resolve()),
        "step_index": step_index,
        "step_token_index": step.get("token_index"),
        "source": source,
        "schema": shortlist.get("schema", ""),
        "shortlist_source": shortlist.get("source", ""),
        "candidate_status": shortlist.get("candidate_status", ""),
        "candidate_rows_reported": int(shortlist.get("candidate_rows", len(ids_raw))),
        "candidate_rows_loaded": len(token_ids),
        "candidate_rows_used": len(limited_ids),
        "top_ids_prefix_match": bool(top_ids) and top_ids[: len(limited_ids)] == limited_ids[: len(top_ids)],
        "tokenizer_scope": "row_id_bridge_unvalidated_tokenizer_equivalence",
    }


def load_logprob_candidates(
    path_text: str,
    row_index: int,
    step_index: int,
    side: str,
    max_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    if not path_text:
        return [], {}
    path = Path(path_text)
    if not path.is_file():
        fail(f"candidate logprob JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid candidate logprob JSON {path}: {exc}")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        fail(f"{path} does not contain a rows list")
    if row_index < 0 or row_index >= len(rows):
        fail(f"--candidate-logprob-row-index {row_index} out of range for {len(rows)} rows")
    row = rows[row_index]
    if not isinstance(row, dict):
        fail(f"row {row_index} in {path} is not an object")
    source = row.get(side)
    if not isinstance(source, dict):
        fail(f"row {row_index} has no object side {side!r}")
    steps = source.get("steps", [])
    if not isinstance(steps, list):
        fail(f"row {row_index} side {side!r} does not contain a steps list")
    if step_index < 0 or step_index >= len(steps):
        fail(f"--candidate-logprob-step-index {step_index} out of range for {len(steps)} steps")
    step = steps[step_index]
    if not isinstance(step, dict):
        fail(f"row {row_index} side {side!r} step {step_index} is not an object")
    top_logprobs = step.get("top_logprobs", [])
    if not isinstance(top_logprobs, list):
        fail(f"row {row_index} side {side!r} step {step_index} has no top_logprobs list")

    token_ids: list[int] = []
    token_texts: list[str] = []
    for offset, item in enumerate(top_logprobs):
        if not isinstance(item, dict):
            fail(f"top_logprobs[{offset}] is not an object")
        try:
            token_id = int(item["id"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid logprob token id at offset {offset}: {exc}")
        token_ids.append(token_id)
        token_texts.append(str(item.get("token", "")))
    limited_ids = token_ids[:max_rows] if max_rows > 0 else token_ids

    oracle_top1_id: int | None = None
    oracle_top1_token = ""
    oracle = row.get("oracle")
    if isinstance(oracle, dict):
        oracle_steps = oracle.get("steps", [])
        if isinstance(oracle_steps, list) and 0 <= step_index < len(oracle_steps):
            oracle_step = oracle_steps[step_index]
            if isinstance(oracle_step, dict):
                try:
                    oracle_top1_id = int(oracle_step["token_id"])
                except (KeyError, TypeError, ValueError):
                    oracle_top1_id = None
                oracle_top1_token = str(oracle_step.get("token", ""))

    return limited_ids, {
        "path": str(path.resolve()),
        "row_index": row_index,
        "step_index": step_index,
        "side": side,
        "prompt": str(row.get("prompt", "")),
        "model": str(source.get("model", "")),
        "candidate_rows_loaded": len(token_ids),
        "candidate_rows_used": len(limited_ids),
        "candidate_tokens": token_texts[: len(limited_ids)],
        "source_token_id": step.get("token_id"),
        "source_token": step.get("token", ""),
        "oracle_top1_id": oracle_top1_id,
        "oracle_top1_token": oracle_top1_token,
        "oracle_top1_covered": oracle_top1_id in limited_ids if oracle_top1_id is not None else None,
        "tokenizer_scope": "same_family_logprob_capture" if "gemma" in str(path).lower() else "logprob_capture",
    }


def transform_logit(raw_score: float, token_id: int, softcap: float, suppress_tokens: set[int]) -> float:
    value = float(raw_score)
    if softcap > 0.0:
        value = softcap * math.tanh(value / softcap)
    if token_id in suppress_tokens:
        value = -math.inf
    return value


def launch_candidate_rows(args: argparse.Namespace) -> dict[str, Any]:
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
    n0, n_vocab = int(tensor.shape[0]), int(tensor.shape[1])
    if len(activation_values) != n0:
        fail(f"activation width {len(activation_values)} does not match {tensor.name} width {n0}")
    row_bytes = q6_k_row_bytes(n0)

    logits_path = Path(args.llamacpp_logits_bin or args.llamacpp_logits_txt) if (
        args.llamacpp_logits_bin or args.llamacpp_logits_txt
    ) else None
    llama_logits = load_llamacpp_logits(logits_path, n_vocab) if logits_path else []

    explicit_candidate_ids = parse_token_ids(args.token_ids)
    live_candidate_ids, live_candidate_source = load_live_trace_candidates(
        args.candidate_live_trace_json,
        args.candidate_live_step_index,
        args.candidate_live_source,
        args.candidate_live_max_rows,
    )
    logprob_candidate_ids, logprob_candidate_source = load_logprob_candidates(
        args.candidate_logprob_json,
        args.candidate_logprob_row_index,
        args.candidate_logprob_step_index,
        args.candidate_logprob_side,
        args.candidate_logprob_max_rows,
    )
    logit_candidate_ids: list[int] = []
    if llama_logits and args.top_k_from_logits > 0:
        logit_candidate_ids = [item.token_id for item in logit_top_scores(llama_logits, args.top_k_from_logits)]
    candidate_ids = explicit_candidate_ids + live_candidate_ids + logprob_candidate_ids + logit_candidate_ids
    candidate_ids = unique_preserve_order(candidate_ids)
    if not candidate_ids:
        fail(
            "no candidate token ids selected; pass --token-ids, --candidate-live-trace-json, "
            "--candidate-logprob-json, or --top-k-from-logits with logits"
        )
    for token_id in candidate_ids:
        if token_id < 0 or token_id >= n_vocab:
            fail(f"candidate token id out of range: {token_id}")

    softcap = metadata_final_logit_softcap(index.metadata) if not args.no_logit_transforms else 0.0
    suppress_tokens = metadata_suppress_tokens(index.metadata) if not args.no_logit_transforms else set()
    candidate_bytes = len(candidate_ids) * row_bytes
    stage_buffer_bytes = max(candidate_bytes, int(args.min_stage_buffer_mib * 1024 * 1024))
    stage_buffer_bytes = min(stage_buffer_bytes, int(args.stage_buffer_gib * BYTES_PER_GIB))
    if candidate_bytes > stage_buffer_bytes:
        fail(f"candidate rows need {candidate_bytes} bytes but stage buffer holds {stage_buffer_bytes}")

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
    row_datas: list[bytes] = []

    try:
        host_ptr = cuda.host_alloc(stage_buffer_bytes)
        device_ptr = cuda.device_alloc(stage_buffer_bytes)
        stream = cuda.stream_create()
        start_event = cuda.event_create()
        stop_event = cuda.event_create()
        host_view = make_host_view(host_ptr, stage_buffer_bytes)
        row_scores_device = cuda.device_alloc(len(candidate_ids) * ctypes.sizeof(ctypes.c_float))
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

        data_start = gguf_data_start(model_path)
        host_read_start = time.perf_counter()
        with model_path.open("rb", buffering=0) as handle:
            for packed_row, token_id in enumerate(candidate_ids):
                offset = packed_row * row_bytes
                handle.seek(data_start + tensor.offset + token_id * row_bytes)
                target = host_view[offset : offset + row_bytes]
                n_read = handle.readinto(target)
                if n_read != row_bytes:
                    fail(f"short read for token {token_id}: {n_read} != {row_bytes}")
                row_datas.append(bytes(target))
        host_read_ms = (time.perf_counter() - host_read_start) * 1000.0

        cuda.memset_async(out_sum, 0, ctypes.sizeof(ctypes.c_float), stream)
        cuda.memset_async(out_abs, 0, ctypes.sizeof(ctypes.c_float), stream)
        h2d_ms = cuda.memcpy_h2d_timed(device_ptr, host_ptr, candidate_bytes, stream, start_event, stop_event)
        cuda.memset_async(row_scores_device, 0, len(candidate_ids) * ctypes.sizeof(ctypes.c_float), stream)
        cuda.lib.cudaEventRecord(start_event, stream)
        launch_vocab_kernel(
            driver,
            function,
            device_ptr,
            n0,
            len(candidate_ids),
            activation_device,
            row_scores_device,
            out_sum,
            out_abs,
            stream,
        )
        cuda.lib.cudaEventRecord(stop_event, stream)
        cuda.check(cuda.lib.cudaEventSynchronize(stop_event), "cudaEventSynchronize(q6_k_candidate)")
        elapsed = ctypes.c_float()
        cuda.check(
            cuda.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event),
            "cudaEventElapsedTime(q6_k_candidate)",
        )
        kernel_ms = float(elapsed.value)
        host_scores = (ctypes.c_float * len(candidate_ids))()
        cuda.memcpy_d2h(
            ctypes.cast(host_scores, ctypes.c_void_p),
            row_scores_device,
            len(candidate_ids) * ctypes.sizeof(ctypes.c_float),
        )
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

    candidates: list[CandidateScore] = []
    for packed_row, token_id in enumerate(candidate_ids):
        raw_score = float(host_scores[packed_row])
        sage_logit = transform_logit(raw_score, token_id, softcap, suppress_tokens)
        llama_logit = float(llama_logits[token_id]) if llama_logits else None
        if llama_logit is None or not math.isfinite(llama_logit) or not math.isfinite(sage_logit):
            abs_error = None
            rel_error = None
        else:
            abs_error = abs(sage_logit - llama_logit)
            rel_error = abs_error / max(abs(sage_logit), abs(llama_logit), 1.0)
        cpu_score = cpu_q6_k_row_score(row_datas[packed_row], n0, activation_values)
        cpu_abs_error = abs(cpu_score - raw_score)
        cpu_passed = cpu_abs_error <= max(1.0e-3, 2.0e-4 * max(abs(cpu_score), abs(raw_score), 1.0))
        candidates.append(
            CandidateScore(
                token_id=token_id,
                packed_row=packed_row,
                raw_score=raw_score,
                sage_logit=sage_logit,
                llama_logit=llama_logit,
                abs_error=None if abs_error is None else float(abs_error),
                rel_error=None if rel_error is None else float(rel_error),
                cpu_score=cpu_score,
                cpu_abs_error=float(cpu_abs_error),
                cpu_passed=cpu_passed,
            )
        )

    finite_errors = [item.abs_error for item in candidates if item.abs_error is not None]
    ranked = sorted(candidates, key=lambda item: item.sage_logit, reverse=True)
    llama_ranked = (
        sorted(candidates, key=lambda item: -math.inf if item.llama_logit is None else item.llama_logit, reverse=True)
        if llama_logits
        else []
    )
    candidate_top1_match = bool(
        ranked and llama_ranked and ranked[0].token_id == llama_ranked[0].token_id
    )
    all_logit_checks_passed = bool(finite_errors) and max(finite_errors) <= args.logit_max_abs_error
    cpu_checks_passed = all(item.cpu_passed for item in candidates)
    active_percent_vocab_tensor = 100.0 * candidate_bytes / max(tensor.n_bytes, 1)
    active_percent_reference_100b_2bit = 100.0 * candidate_bytes / max(int(100_000_000_000 * 2 / 8), 1)
    global_top_scores = logit_top_scores(llama_logits, 1) if llama_logits else []
    global_top1_id = global_top_scores[0].token_id if global_top_scores else None
    global_top1_logit = global_top_scores[0].score if global_top_scores else None
    candidate_contains_global_top1 = global_top1_id in candidate_ids if global_top1_id is not None else False
    candidate_global_top1_coverage_status = (
        "candidate_covers_global_top1"
        if candidate_contains_global_top1
        else "candidate_misses_global_top1_exact_fallback_required"
        if llama_logits
        else "global_top1_unavailable"
    )
    if live_candidate_ids and not explicit_candidate_ids and not logprob_candidate_ids and not logit_candidate_ids:
        candidate_source_kind = "live_proxy_shortlist"
    elif logprob_candidate_ids and not explicit_candidate_ids and not live_candidate_ids and not logit_candidate_ids:
        candidate_source_kind = f"logprob_{args.candidate_logprob_side}_top_k"
    elif live_candidate_ids or logprob_candidate_ids:
        candidate_source_kind = "mixed_candidate_sources"
    else:
        candidate_source_kind = "explicit_or_logit_top_k"

    return {
        "schema": "sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0",
        "status": "measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits"
        if llama_logits
        else "measured_sparse_q6_k_candidate_rows_no_logit_compare",
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
            "stage_buffer_gib": args.stage_buffer_gib,
            "min_stage_buffer_mib": args.min_stage_buffer_mib,
            "stage_buffer_bytes": stage_buffer_bytes,
            "top_k_from_logits": args.top_k_from_logits,
            "logit_max_abs_error": args.logit_max_abs_error,
            "candidate_live_trace_json": args.candidate_live_trace_json,
            "candidate_live_step_index": args.candidate_live_step_index,
            "candidate_live_source": args.candidate_live_source,
            "candidate_live_max_rows": args.candidate_live_max_rows,
            "candidate_logprob_json": args.candidate_logprob_json,
            "candidate_logprob_row_index": args.candidate_logprob_row_index,
            "candidate_logprob_step_index": args.candidate_logprob_step_index,
            "candidate_logprob_side": args.candidate_logprob_side,
            "candidate_logprob_max_rows": args.candidate_logprob_max_rows,
        },
        "summary": {
            "tensor_type": tensor.tensor_type,
            "n0": n0,
            "candidate_tokens": len(candidate_ids),
            "candidate_bytes": candidate_bytes,
            "candidate_gib": bytes_to_gib(candidate_bytes),
            "row_bytes": row_bytes,
            "candidate_source_kind": candidate_source_kind,
            "explicit_candidate_rows": len(explicit_candidate_ids),
            "live_trace_candidate_rows": len(live_candidate_ids),
            "logprob_candidate_rows": len(logprob_candidate_ids),
            "logit_top_k_candidate_rows": len(logit_candidate_ids),
            "live_trace_candidate_status": live_candidate_source.get("candidate_status", ""),
            "live_trace_top_ids_prefix_match": bool(live_candidate_source.get("top_ids_prefix_match", False)),
            "live_trace_tokenizer_scope": live_candidate_source.get("tokenizer_scope", ""),
            "logprob_prompt": logprob_candidate_source.get("prompt", ""),
            "logprob_source_side": logprob_candidate_source.get("side", ""),
            "logprob_source_token_id": logprob_candidate_source.get("source_token_id"),
            "logprob_oracle_top1_id": logprob_candidate_source.get("oracle_top1_id"),
            "logprob_oracle_top1_covered": bool(logprob_candidate_source.get("oracle_top1_covered", False)),
            "logprob_tokenizer_scope": logprob_candidate_source.get("tokenizer_scope", ""),
            "max_live_buffer_bytes": candidate_bytes,
            "max_live_buffer_gib": bytes_to_gib(candidate_bytes),
            "allocated_stage_buffer_bytes": stage_buffer_bytes,
            "host_read_ms": host_read_ms,
            "h2d_ms": h2d_ms,
            "kernel_ms": kernel_ms,
            "candidate_rows_per_ms": len(candidate_ids) / kernel_ms if kernel_ms > 0 else 0.0,
            "active_percent_vocab_tensor": active_percent_vocab_tensor,
            "active_percent_reference_100b_2bit": active_percent_reference_100b_2bit,
            "logit_transform_softcap": softcap,
            "suppress_token_count": len(suppress_tokens),
            "llamacpp_logit_checks": len(finite_errors),
            "llamacpp_logit_max_abs_error": max(finite_errors, default=0.0),
            "llamacpp_logit_mean_abs_error": sum(finite_errors) / len(finite_errors) if finite_errors else 0.0,
            "llamacpp_candidate_top1_match": candidate_top1_match,
            "llamacpp_global_top1_id": global_top1_id,
            "llamacpp_global_top1_logit": global_top1_logit,
            "candidate_contains_llamacpp_global_top1": candidate_contains_global_top1,
            "candidate_global_top1_coverage_status": candidate_global_top1_coverage_status,
            "llamacpp_logit_checks_passed": all_logit_checks_passed,
            "cpu_score_checks": len(candidates),
            "cpu_score_checks_passed": cpu_checks_passed,
            "max_cpu_score_abs_error": max((item.cpu_abs_error for item in candidates), default=0.0),
            "candidate_scoring_status": "sparse_candidate_rows_compared_to_llamacpp_logits"
            if llama_logits
            else "sparse_candidate_rows_no_logit_compare",
            "true_logit_status": "candidate_rows_compared_against_llamacpp_logits"
            if llama_logits
            else "candidate_rows_not_compared_to_logits",
        },
        "runtime_ledger_evidence": {
            "schema": "sage-active-byte-ledger-v0",
            "oracle_mode": "sparse_q6_k_candidate_verifier_smoke",
            "oracle_active_bytes": candidate_bytes,
            "gpu_staged_bytes": candidate_bytes,
            "host_pinned_bytes": stage_buffer_bytes,
            "pcie_transfer_ms": h2d_ms,
            "cuda_kernel_ms": kernel_ms,
            "candidate_scoring_status": "sparse_candidate_rows_compared_to_llamacpp_logits"
            if llama_logits
            else "sparse_candidate_rows_no_logit_compare",
        },
        "llamacpp_logits": {
            "path": str(logits_path.resolve()) if logits_path else "",
            "loaded_logits": len(llama_logits),
        },
        "candidate_source": {
            "kind": candidate_source_kind,
            "explicit_rows": len(explicit_candidate_ids),
            "live_trace": live_candidate_source,
            "logprob": logprob_candidate_source,
            "logit_top_k_rows": len(logit_candidate_ids),
            "candidate_order": "explicit_then_live_trace_then_logprob_then_logit_top_k_unique",
            "note": (
                "Live trace and logprob token ids are consumed as sparse row ids. Tokenizer equivalence is proven "
                "only when the candidate source, activation, logits, and GGUF model come from the same tokenizer/model family."
            ),
        },
        "candidate_token_ids": candidate_ids,
        "ranked_candidate_token_ids": [item.token_id for item in ranked],
        "llamacpp_ranked_candidate_token_ids": [item.token_id for item in llama_ranked],
        "candidate_scores": [asdict(item) for item in candidates],
        "compile_log": compile_log,
    }


def print_markdown(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("# SAGE Q6_K Candidate Verifier Smoke")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Candidate source: `{summary['candidate_source_kind']}`")
    print(f"- Candidate tokens: `{summary['candidate_tokens']}`")
    print(f"- Candidate bytes: `{summary['candidate_bytes']}` (`{summary['candidate_gib']:.6f} GiB`)")
    print(f"- Active vocab tensor share: `{summary['active_percent_vocab_tensor']:.6f}%`")
    print(f"- H2D time: `{summary['h2d_ms']:.4f} ms`")
    print(f"- Kernel time: `{summary['kernel_ms']:.4f} ms`")
    print(f"- llama.cpp checks: `{summary['llamacpp_logit_checks']}` passed `{summary['llamacpp_logit_checks_passed']}`")
    print(f"- Candidate top-1 match: `{summary['llamacpp_candidate_top1_match']}`")
    print(f"- Candidate covers global top-1: `{summary['candidate_contains_llamacpp_global_top1']}`")
    print()
    print("| Rank | Token id | SAGE logit | llama.cpp logit | Abs error |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    by_id = {item["token_id"]: item for item in payload["candidate_scores"]}
    for rank, token_id in enumerate(payload["ranked_candidate_token_ids"], 1):
        item = by_id[token_id]
        llama_logit = "" if item["llama_logit"] is None else f"{item['llama_logit']:.6g}"
        abs_error = "" if item["abs_error"] is None else f"{item['abs_error']:.6g}"
        print(f"| {rank} | {token_id} | {item['sage_logit']:.6g} | {llama_logit} | {abs_error} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA Q6_K candidate-row verifier smoke.")
    parser.add_argument("--model", default="models/gemma-4-31b-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf")
    parser.add_argument("--tensor-name", default="token_embd.weight")
    parser.add_argument("--activation-jsonl", default="benchmarks/sage-gemma31b-result-norm-values-5376.jsonl")
    parser.add_argument("--activation-name", default="result_norm")
    parser.add_argument("--activation-record-index", type=int, default=0)
    parser.add_argument("--token-ids", default="", help="comma-separated candidate token ids")
    parser.add_argument(
        "--candidate-live-trace-json",
        default="",
        help="load candidate token ids from a sage-live-proxy-shortlist-v0 trace",
    )
    parser.add_argument("--candidate-live-step-index", type=int, default=0)
    parser.add_argument("--candidate-live-source", default="proxy", help="step object to read, usually proxy")
    parser.add_argument("--candidate-live-max-rows", type=int, default=0, help="0 means use every live candidate row")
    parser.add_argument(
        "--candidate-logprob-json",
        default="",
        help="load candidate token ids from a sage_agreement/logprob capture rows[*].<side>.steps[*].top_logprobs",
    )
    parser.add_argument("--candidate-logprob-row-index", type=int, default=0)
    parser.add_argument("--candidate-logprob-step-index", type=int, default=0)
    parser.add_argument("--candidate-logprob-side", default="proxy", choices=["proxy", "oracle"])
    parser.add_argument("--candidate-logprob-max-rows", type=int, default=0, help="0 means use every logprob row")
    parser.add_argument("--top-k-from-logits", type=int, default=0, help="add top-k ids from llama.cpp logits")
    parser.add_argument("--llamacpp-logits-bin", default="")
    parser.add_argument("--llamacpp-logits-txt", default="")
    parser.add_argument("--logit-max-abs-error", type=float, default=0.05)
    parser.add_argument("--stage-buffer-gib", type=float, default=0.05)
    parser.add_argument("--min-stage-buffer-mib", type=float, default=1.0)
    parser.add_argument("--no-logit-transforms", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="")
    parser.add_argument("--cudart", default="")
    parser.add_argument("--nvrtc", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.activation_record_index < 0:
        parser.error("--activation-record-index must be non-negative")
    if args.candidate_live_step_index < 0:
        parser.error("--candidate-live-step-index must be non-negative")
    if args.candidate_live_max_rows < 0:
        parser.error("--candidate-live-max-rows must be non-negative")
    if args.candidate_logprob_row_index < 0:
        parser.error("--candidate-logprob-row-index must be non-negative")
    if args.candidate_logprob_step_index < 0:
        parser.error("--candidate-logprob-step-index must be non-negative")
    if args.candidate_logprob_max_rows < 0:
        parser.error("--candidate-logprob-max-rows must be non-negative")
    if args.top_k_from_logits < 0:
        parser.error("--top-k-from-logits must be non-negative")
    if args.llamacpp_logits_bin and args.llamacpp_logits_txt:
        parser.error("pass only one of --llamacpp-logits-bin or --llamacpp-logits-txt")
    if args.logit_max_abs_error < 0:
        parser.error("--logit-max-abs-error must be non-negative")
    if args.stage_buffer_gib <= 0:
        parser.error("--stage-buffer-gib must be positive")
    if args.min_stage_buffer_mib <= 0:
        parser.error("--min-stage-buffer-mib must be positive")

    payload = launch_candidate_rows(args)
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
