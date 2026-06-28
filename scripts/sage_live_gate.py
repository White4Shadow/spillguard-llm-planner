#!/usr/bin/env python3
"""
Run the Gemma-scale SAGE live-vs-replay gate.

This script is a guarded runner around:

1. sage_live_loop.py seeded from candidate replay JSONs;
2. sage_live_replay_compare.py against the validated multi-token replay trace.

It exists to make the next proof step reproducible. Use --print-only first to
inspect the exact commands without launching model processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARKS = ROOT / "benchmarks"
DEFAULT_REPLAY_GLOB = str(DEFAULT_BENCHMARKS / "*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json")
DEFAULT_REPLAY_TRACE = DEFAULT_BENCHMARKS / "20260627-sage-multitoken-replay-validation80e-full-cpp-decisions-promptkey.json"
DEFAULT_PROXY = ROOT / "models" / "gemma-4-12b-it-qat-q4_0-gguf" / "gemma-4-12b-it-qat-q4_0.gguf"
DEFAULT_ORACLE = ROOT / "models" / "gemma-4-31b-it-qat-q4_0-gguf" / "gemma-4-31B_q4_0-it.gguf"
DEFAULT_QWEN = ROOT / "models" / "qwen2.5-0.5b-instruct-gguf" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_LLAMA_SERVER = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-server.exe"
DEFAULT_LLAMA_COMPLETION = ROOT / "tools" / "llama.cpp-src" / "build-live-migration-cuda" / "bin" / "Release" / "llama-completion.exe"


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        paths.extend(matches if matches else [Path(value)])
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return sorted(unique, key=lambda item: str(item))


def quote_command(command: list[str]) -> str:
    parts: list[str] = []
    for item in command:
        if not item:
            parts.append('""')
        elif any(ch.isspace() for ch in item) or any(ch in item for ch in "`\"'"):
            escaped = item.replace('"', '`"')
            parts.append(f'"{escaped}"')
        else:
            parts.append(item)
    return " ".join(parts)


def live_json_path(args: argparse.Namespace) -> Path:
    if args.live_json:
        return Path(args.live_json).resolve()
    return (Path(args.out_dir) / f"{args.tag}.json").resolve()


def compare_json_path(args: argparse.Namespace) -> Path:
    if args.compare_json:
        return Path(args.compare_json).resolve()
    if "live-" in args.tag:
        stem = args.tag.replace("live-", "live-compare-", 1)
    else:
        stem = f"{args.tag}-compare"
    return (Path(args.out_dir) / f"{stem}.json").resolve()


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    replay_paths = expand_paths(args.seed_replay_json)
    if not replay_paths:
        fail("no seed replay JSON files found")
    for path in replay_paths:
        ensure_file(path, "seed replay JSON")
    ensure_file(Path(args.replay_trace), "replay trace JSON")
    ensure_file(Path(args.proxy_model), "proxy model")
    ensure_file(Path(args.oracle_model), "oracle model")
    ensure_file(Path(args.verifier_model), "verifier model")
    ensure_file(Path(args.llama_server), "llama-server")
    ensure_file(Path(args.llama_completion), "llama-completion")
    return replay_paths


def live_command(args: argparse.Namespace, replay_paths: list[Path], live_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sage_live_loop.py"),
        "--llama-server",
        str(Path(args.llama_server).resolve()),
        "--llama-completion",
        str(Path(args.llama_completion).resolve()),
        "--seed-replay-json",
        *[str(path) for path in replay_paths],
        "--seed-prompt-index",
        str(args.prompt_index),
        "--seed-replay-offset",
        str(args.replay_offset),
        "--proxy-model",
        str(Path(args.proxy_model).resolve()),
        "--oracle-model",
        str(Path(args.oracle_model).resolve()),
        "--verifier-model",
        str(Path(args.verifier_model).resolve()),
        "--proxy-ngl",
        str(args.proxy_ngl),
        "--oracle-ngl",
        str(args.oracle_ngl),
        "--verifier-ngl",
        str(args.verifier_ngl),
        "--ctx-size",
        str(args.ctx_size),
        "--verifier-ctx-size",
        str(args.verifier_ctx_size),
        "--threads",
        str(args.threads),
        "--verifier-threads",
        str(args.verifier_threads),
        "--ready-timeout",
        str(args.ready_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--verifier-timeout",
        str(args.verifier_timeout),
        "--json-out",
        str(live_path),
        "--tag",
        args.tag,
    ]
    if args.max_live_tokens > 0:
        command.extend(["--tokens", str(args.max_live_tokens)])
    else:
        command.append("--tokens-from-replay")
    if args.no_seed_apply_params:
        command.append("--no-seed-apply-params")
    if args.proxy_base_url:
        command.extend(["--proxy-base-url", args.proxy_base_url])
    if args.oracle_base_url:
        command.extend(["--oracle-base-url", args.oracle_base_url])
    if args.no_pause_proxy_for_verifier:
        command.append("--no-pause-proxy-for-verifier")
    return command


def compare_command(args: argparse.Namespace, live_path: Path, compare_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "sage_live_replay_compare.py"),
        "--live-json",
        str(live_path),
        "--replay-json",
        str(Path(args.replay_trace).resolve()),
        "--prompt-index",
        str(args.prompt_index),
        "--replay-offset",
        str(args.replay_offset),
        "--json-out",
        str(compare_path),
        "--tag",
        args.tag,
    ]
    if args.max_live_tokens > 0:
        command.extend(["--max-steps", str(args.max_live_tokens)])
    if args.require_pass:
        command.append("--require-pass")
    if args.require_proxy_token_match:
        command.append("--require-proxy-token-match")
    if args.no_require_verifier_coverage_match:
        command.append("--no-require-verifier-coverage-match")
    return command


def run_command(command: list[str], label: str) -> int:
    print()
    print(f"live-gate: running {label}", flush=True)
    print(quote_command(command), flush=True)
    proc = subprocess.run(command, cwd=ROOT, check=False)
    return int(proc.returncode)


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the seeded SAGE live-loop gate and replay comparator.")
    parser.add_argument("--seed-replay-json", nargs="+", default=[DEFAULT_REPLAY_GLOB])
    parser.add_argument("--replay-trace", default=str(DEFAULT_REPLAY_TRACE))
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--replay-offset", type=int, default=0)
    parser.add_argument("--max-live-tokens", type=int, default=0, help="0 uses all replay rows for the prompt")
    parser.add_argument("--proxy-model", default=str(DEFAULT_PROXY))
    parser.add_argument("--oracle-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--verifier-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--qwen-smoke", action="store_true", help="use Qwen 0.5B for all model roles")
    parser.add_argument("--llama-server", default=str(DEFAULT_LLAMA_SERVER))
    parser.add_argument("--llama-completion", default=str(DEFAULT_LLAMA_COMPLETION))
    parser.add_argument("--proxy-ngl", default="all")
    parser.add_argument("--oracle-ngl", default="38")
    parser.add_argument("--verifier-ngl", default="38")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--verifier-ctx-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--verifier-threads", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--verifier-timeout", type=int, default=1800)
    parser.add_argument("--proxy-base-url", default="")
    parser.add_argument("--oracle-base-url", default="")
    parser.add_argument("--no-seed-apply-params", action="store_true")
    parser.add_argument("--no-pause-proxy-for-verifier", action="store_true")
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-proxy-token-match", action="store_true")
    parser.add_argument("--no-require-verifier-coverage-match", action="store_true")
    parser.add_argument("--skip-live", action="store_true", help="only run comparator against existing --live-json")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_BENCHMARKS))
    parser.add_argument("--live-json", default="")
    parser.add_argument("--compare-json", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.qwen_smoke:
        qwen = str(DEFAULT_QWEN)
        args.proxy_model = qwen
        args.oracle_model = qwen
        args.verifier_model = qwen
        args.proxy_ngl = "0"
        args.oracle_ngl = "0"
        args.verifier_ngl = "0"
        args.ctx_size = min(args.ctx_size, 64)
        args.verifier_ctx_size = min(args.verifier_ctx_size, 64)
        args.threads = min(args.threads, 4)
        args.verifier_threads = 4 if args.verifier_threads == 0 else min(args.verifier_threads, 4)
    if args.prompt_index < 1:
        parser.error("--prompt-index must be >= 1")
    if args.replay_offset < 0:
        parser.error("--replay-offset must be non-negative")
    if args.max_live_tokens < 0:
        parser.error("--max-live-tokens must be non-negative")
    if not args.tag:
        model_tag = "qwen" if args.qwen_smoke else "gemma"
        args.tag = f"{model_tag}-live-validation80e-p{args.prompt_index:03d}"

    replay_paths = validate_inputs(args)
    live_path = live_json_path(args)
    compare_path = compare_json_path(args)
    live_cmd = live_command(args, replay_paths, live_path)
    compare_cmd = compare_command(args, live_path, compare_path)

    if args.print_only:
        print("seed replay JSONs:")
        for path in replay_paths:
            print(f"  {path}")
        print()
        print("live command:")
        print(quote_command(live_cmd))
        print()
        print("compare command:")
        print(quote_command(compare_cmd))
        return 0

    if not args.skip_live:
        live_code = run_command(live_cmd, "live loop")
        if live_code != 0:
            return live_code
    compare_code = run_command(compare_cmd, "live/replay comparator")
    live_summary = load_summary(live_path)
    compare_summary = load_summary(compare_path)
    print()
    print("SAGE live gate summary")
    print(f"  live_json: {live_path}")
    print(f"  compare_json: {compare_path}")
    if live_summary:
        print(f"  live steps: {live_summary.get('steps')}")
        print(f"  live tok/s: {live_summary.get('tokens_per_sec')}")
        print(f"  live proxy accepts: {live_summary.get('proxy_accepts')}")
        print(f"  live oracle fallbacks: {live_summary.get('oracle_fallbacks')}")
    if compare_summary:
        print(f"  compared steps: {compare_summary.get('compared_steps')}")
        print(f"  action matches: {compare_summary.get('action_matches')}")
        print(f"  selected-token matches: {compare_summary.get('selected_token_matches')}")
        print(f"  first mismatch: {compare_summary.get('first_mismatch_index')} {compare_summary.get('first_mismatch_field')}")
        print(f"  pass: {compare_summary.get('meets_required_matches')}")
    return compare_code


if __name__ == "__main__":
    raise SystemExit(main())
