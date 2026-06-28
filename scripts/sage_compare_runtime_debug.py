#!/usr/bin/env python3
"""
Compare SAGE debug JSONL captures against non-debug runtime captures.

This is the gate between "we can measure a verifier signal" and "the same signal
is available in normal generation." It compares tensor records by prompt index,
tensor name, and occurrence, then checks shape/type metadata and scalar stats.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FLOAT_FIELDS = ("sum", "mean", "min_value", "max_value")
EXACT_FIELDS = ("dtype", "op", "shape", "count", "nan_count")


@dataclass
class TensorMismatch:
    prompt_index: int
    tensor_key: str
    field: str
    debug_value: Any
    runtime_value: Any
    abs_error: float | None = None
    rel_error: float | None = None


@dataclass
class CompareReport:
    debug_json: list[str]
    runtime_json: list[str]
    prompt_pairs: int
    debug_tensor_records: int
    runtime_tensor_records: int
    matched_tensor_records: int
    missing_in_runtime: int
    extra_in_runtime: int
    mismatches: list[TensorMismatch]
    abs_tol: float
    rel_tol: float
    passed: bool
    created_at: str


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_capture_files(paths: list[Path], label: str) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"{label} JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for capture in payload.get("captures", []):
            item = dict(capture)
            item["_source"] = str(path.resolve())
            captures.append(item)
    return captures


def tensor_key(tensor: dict[str, Any]) -> str:
    return f"{tensor.get('name')}#{tensor.get('occurrence', 0)}"


def tensor_map(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tensor in capture.get("tensors", []):
        key = tensor_key(tensor)
        if key in out:
            fail(f"duplicate tensor key in {capture.get('_source', '<unknown>')} prompt {capture.get('index')}: {key}")
        out[key] = tensor
    return out


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def float_close(debug_value: Any, runtime_value: Any, abs_tol: float, rel_tol: float) -> tuple[bool, float, float]:
    a = to_float(debug_value)
    b = to_float(runtime_value)
    if math.isnan(a) and math.isnan(b):
        return True, 0.0, 0.0
    abs_error = abs(a - b)
    denom = max(abs(a), abs(b), 1.0)
    rel_error = abs_error / denom
    return abs_error <= abs_tol or rel_error <= rel_tol, abs_error, rel_error


def compare_captures(
    *,
    debug_captures: list[dict[str, Any]],
    runtime_captures: list[dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> CompareReport:
    prompt_pairs = min(len(debug_captures), len(runtime_captures))
    mismatches: list[TensorMismatch] = []
    missing_in_runtime = 0
    extra_in_runtime = 0
    matched_tensor_records = 0
    debug_tensor_records = sum(len(capture.get("tensors", [])) for capture in debug_captures)
    runtime_tensor_records = sum(len(capture.get("tensors", [])) for capture in runtime_captures)

    if len(debug_captures) != len(runtime_captures):
        mismatches.append(
            TensorMismatch(
                prompt_index=-1,
                tensor_key="<capture-count>",
                field="captures",
                debug_value=len(debug_captures),
                runtime_value=len(runtime_captures),
            )
        )

    for debug_capture, runtime_capture in zip(debug_captures, runtime_captures):
        prompt_index = int(debug_capture.get("index", 0))
        if debug_capture.get("prompt") != runtime_capture.get("prompt"):
            mismatches.append(
                TensorMismatch(
                    prompt_index=prompt_index,
                    tensor_key="<prompt>",
                    field="prompt",
                    debug_value=debug_capture.get("prompt"),
                    runtime_value=runtime_capture.get("prompt"),
                )
            )
        if "rendered_prompt" in debug_capture and "rendered_prompt" in runtime_capture:
            if debug_capture.get("rendered_prompt") != runtime_capture.get("rendered_prompt"):
                mismatches.append(
                    TensorMismatch(
                        prompt_index=prompt_index,
                        tensor_key="<rendered-prompt>",
                        field="rendered_prompt",
                        debug_value=debug_capture.get("rendered_prompt"),
                        runtime_value=runtime_capture.get("rendered_prompt"),
                    )
                )

        debug_tensors = tensor_map(debug_capture)
        runtime_tensors = tensor_map(runtime_capture)
        debug_keys = set(debug_tensors)
        runtime_keys = set(runtime_tensors)

        for key in sorted(debug_keys - runtime_keys):
            missing_in_runtime += 1
            mismatches.append(
                TensorMismatch(
                    prompt_index=prompt_index,
                    tensor_key=key,
                    field="<missing>",
                    debug_value=debug_tensors[key],
                    runtime_value=None,
                )
            )
        for key in sorted(runtime_keys - debug_keys):
            extra_in_runtime += 1
            mismatches.append(
                TensorMismatch(
                    prompt_index=prompt_index,
                    tensor_key=key,
                    field="<extra>",
                    debug_value=None,
                    runtime_value=runtime_tensors[key],
                )
            )

        for key in sorted(debug_keys & runtime_keys):
            matched_tensor_records += 1
            debug_tensor = debug_tensors[key]
            runtime_tensor = runtime_tensors[key]
            for field in EXACT_FIELDS:
                if debug_tensor.get(field) != runtime_tensor.get(field):
                    mismatches.append(
                        TensorMismatch(
                            prompt_index=prompt_index,
                            tensor_key=key,
                            field=field,
                            debug_value=debug_tensor.get(field),
                            runtime_value=runtime_tensor.get(field),
                        )
                    )
            for field in FLOAT_FIELDS:
                close, abs_error, rel_error = float_close(debug_tensor.get(field), runtime_tensor.get(field), abs_tol, rel_tol)
                if not close:
                    mismatches.append(
                        TensorMismatch(
                            prompt_index=prompt_index,
                            tensor_key=key,
                            field=field,
                            debug_value=debug_tensor.get(field),
                            runtime_value=runtime_tensor.get(field),
                            abs_error=abs_error,
                            rel_error=rel_error,
                        )
                    )

    return CompareReport(
        debug_json=[],
        runtime_json=[],
        prompt_pairs=prompt_pairs,
        debug_tensor_records=debug_tensor_records,
        runtime_tensor_records=runtime_tensor_records,
        matched_tensor_records=matched_tensor_records,
        missing_in_runtime=missing_in_runtime,
        extra_in_runtime=extra_in_runtime,
        mismatches=mismatches,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        passed=not mismatches,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SAGE debug and runtime tensor-stat captures.")
    parser.add_argument("--debug-json", nargs="+", required=True, help="sage_probe_capture.py JSON output")
    parser.add_argument("--runtime-json", nargs="+", required=True, help="sage_runtime_capture.py JSON output")
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--require-match", action="store_true")
    args = parser.parse_args()

    debug_paths = [Path(path) for path in args.debug_json]
    runtime_paths = [Path(path) for path in args.runtime_json]
    debug_captures = load_capture_files(debug_paths, "debug")
    runtime_captures = load_capture_files(runtime_paths, "runtime")

    report = compare_captures(
        debug_captures=debug_captures,
        runtime_captures=runtime_captures,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )
    report.debug_json = [str(path.resolve()) for path in debug_paths]
    report.runtime_json = [str(path.resolve()) for path in runtime_paths]

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"wrote: {out_path.resolve()}")

    print(f"prompt pairs:             {report.prompt_pairs}")
    print(f"debug tensor records:     {report.debug_tensor_records}")
    print(f"runtime tensor records:   {report.runtime_tensor_records}")
    print(f"matched tensor records:   {report.matched_tensor_records}")
    print(f"missing in runtime:       {report.missing_in_runtime}")
    print(f"extra in runtime:         {report.extra_in_runtime}")
    print(f"mismatches:               {len(report.mismatches)}")
    print(f"result:                   {'pass' if report.passed else 'fail'}")

    if report.mismatches:
        print("first mismatches:")
        for mismatch in report.mismatches[:10]:
            suffix = ""
            if mismatch.abs_error is not None:
                suffix = f" abs_error={mismatch.abs_error:.6g} rel_error={mismatch.rel_error:.6g}"
            print(f"  prompt={mismatch.prompt_index} tensor={mismatch.tensor_key} field={mismatch.field}{suffix}")

    if args.require_match and not report.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
