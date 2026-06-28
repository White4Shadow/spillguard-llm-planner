#!/usr/bin/env python3
"""
Run the SAGE debug-vs-runtime tensor-stat parity gate.

This wraps the three commands needed to prove that a normal generation binary
sees the same verifier signal as llama-debug on identical prompts:

1. sage_probe_capture.py with --tensor-stats-jsonl
2. sage_runtime_capture.py with --sage-* runtime hook arguments
3. sage_compare_runtime_debug.py with --require-match
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text)
    return out.strip("-")[:80] or "run"


def add_bool_flag(command: list[str], name: str, enabled: bool) -> None:
    command.append(f"--{name}" if enabled else f"--no-{name}")


def add_prompt_source(command: list[str], args: argparse.Namespace) -> None:
    if args.tasks_json:
        command.extend(["--tasks-json", args.tasks_json])
    elif args.prompts:
        command.extend(["--prompts", args.prompts])
    else:
        command.extend(["--prompt", args.prompt])


def run_command(command: list[str], dry_run: bool) -> None:
    print("gate:", " ".join(command), flush=True)
    if dry_run:
        return
    proc = subprocess.run(command, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def build_debug_command(args: argparse.Namespace, debug_json: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sage_probe_capture.py"),
        "--model",
        args.model,
        "--mode",
        args.mode,
        "--tensor-filter",
        args.tensor_filter,
        "--tensor-stats-jsonl",
        "--ngl",
        str(args.ngl),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--timeout",
        str(args.timeout),
        "--tag",
        f"{args.tag}-debug",
        "--json-out",
        str(debug_json),
    ]
    add_prompt_source(command, args)
    add_bool_flag(command, "chat-enable-thinking", args.chat_enable_thinking)
    add_bool_flag(command, "no-warmup", args.no_warmup)
    if args.append_text:
        command.extend(["--append-text", args.append_text])
    if args.gemma4_thought_prefix:
        command.append("--gemma4-thought-prefix")
    if args.offset > 0:
        command.extend(["--offset", str(args.offset)])
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.llama_debug:
        command.extend(["--llama-debug", args.llama_debug])
    if args.threads > 0:
        command.extend(["--threads", str(args.threads)])
    if args.cuda_bin:
        command.extend(["--cuda-bin", args.cuda_bin])
    for extra_arg in args.debug_extra_arg:
        command.extend(["--extra-arg", extra_arg])
    return command


def build_runtime_command(args: argparse.Namespace, runtime_json: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sage_runtime_capture.py"),
        "--model",
        args.model,
        "--mode",
        args.mode,
        "--tensor-filter",
        args.tensor_filter,
        "--tokens",
        str(args.tokens),
        "--ngl",
        str(args.ngl),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--timeout",
        str(args.timeout),
        "--tag",
        f"{args.tag}-runtime",
        "--json-out",
        str(runtime_json),
    ]
    add_prompt_source(command, args)
    add_bool_flag(command, "chat-enable-thinking", args.chat_enable_thinking)
    add_bool_flag(command, "no-warmup", args.no_warmup)
    if args.capture_decisions:
        command.append("--capture-decisions")
    if args.append_text:
        command.extend(["--append-text", args.append_text])
    if args.gemma4_thought_prefix:
        command.append("--gemma4-thought-prefix")
    if args.offset > 0:
        command.extend(["--offset", str(args.offset)])
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.llama_completion:
        command.extend(["--llama-completion", args.llama_completion])
    if args.threads > 0:
        command.extend(["--threads", str(args.threads)])
    if args.cuda_bin:
        command.extend(["--cuda-bin", args.cuda_bin])
    for extra_arg in args.runtime_extra_arg:
        command.extend(["--extra-arg", extra_arg])
    return command


def build_compare_command(args: argparse.Namespace, debug_json: Path, runtime_json: Path, compare_json: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sage_compare_runtime_debug.py"),
        "--debug-json",
        str(debug_json),
        "--runtime-json",
        str(runtime_json),
        "--abs-tol",
        str(args.abs_tol),
        "--rel-tol",
        str(args.rel_tol),
        "--json-out",
        str(compare_json),
    ]
    if not args.allow_mismatch:
        command.append("--require-match")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a paired SAGE debug/runtime parity gate.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-debug", default="")
    parser.add_argument("--llama-completion", default="")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--tasks-json", default="")
    parser.add_argument("--mode", choices=["raw", "gemma4-chat"], default="raw")
    parser.add_argument("--chat-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-text", default="")
    parser.add_argument("--gemma4-thought-prefix", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tensor-filter", required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--ngl", default="0")
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ubatch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-decisions", action="store_true", help="include in-process SAGE decision events in the runtime capture")
    parser.add_argument("--cuda-bin", default="")
    parser.add_argument("--tag", default="runtime-parity")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--debug-json-out", default="")
    parser.add_argument("--runtime-json-out", default="")
    parser.add_argument("--compare-json-out", default="")
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--debug-extra-arg", action="append", default=[])
    parser.add_argument("--runtime-extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = safe_name(args.tag)
    debug_json = Path(args.debug_json_out) if args.debug_json_out else out_dir / f"{timestamp}-sage-runtime-gate-{run_tag}-debug.json"
    runtime_json = Path(args.runtime_json_out) if args.runtime_json_out else out_dir / f"{timestamp}-sage-runtime-gate-{run_tag}-runtime.json"
    compare_json = Path(args.compare_json_out) if args.compare_json_out else out_dir / f"{timestamp}-sage-runtime-gate-{run_tag}-compare.json"
    for path in (debug_json, runtime_json, compare_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    run_command(build_debug_command(args, debug_json), args.dry_run)
    run_command(build_runtime_command(args, runtime_json), args.dry_run)
    run_command(build_compare_command(args, debug_json, runtime_json, compare_json), args.dry_run)

    result = {
        "debug_json": str(debug_json.resolve()),
        "runtime_json": str(runtime_json.resolve()),
        "compare_json": str(compare_json.resolve()),
    }
    print("gate outputs:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
