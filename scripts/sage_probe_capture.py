#!/usr/bin/env python3
"""
Capture sparse-verifier tensor signals from llama.cpp.

This is the measurement bridge between the SAGE verifier manifest and a learned
accept/reject verifier. It runs a patched llama-debug binary with a manifest
tensor filter, captures compact tensor summaries from the callback output or a
JSONL sidecar, and writes structured JSON that can later be joined with
proxy/oracle labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_verifier_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_DEBUG = ROOT / "tools" / "llama.cpp-src" / "build-live-migration-cuda" / "bin" / "Release" / "llama-debug.exe"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LOG_DIR = DEFAULT_OUT_DIR / "sage-probe-logs"
DEFAULT_CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64")

TENSOR_RE = re.compile(
    r"common_debug_cb_eval:\s*(?P<name>[^=]+?)\s*=\s*"
    r"\((?P<dtype>[^)]+)\)\s*(?P<op>[A-Z0-9_]+)\(.*\)\s*=\s*\{(?P<shape>[^}]*)\}"
)
SUM_RE = re.compile(r"^\s*sum\s*=\s*(?P<value>[-+0-9.eE]+)\s*$")
FLOAT_RE = r"[-+0-9.eEinfnanINFNAN]+"
STATS_RE = re.compile(
    rf"^\s*stats:\s*count\s*=\s*(?P<count>\d+),\s*sum\s*=\s*(?P<sum>{FLOAT_RE}),\s*"
    rf"mean\s*=\s*(?P<mean>{FLOAT_RE}),\s*min\s*=\s*(?P<min>{FLOAT_RE}),\s*"
    rf"max\s*=\s*(?P<max>{FLOAT_RE}),\s*nan_count\s*=\s*(?P<nan_count>\d+)\s*$"
)


@dataclass
class TensorSignal:
    occurrence: int
    name: str
    dtype: str
    op: str
    shape: list[int]
    sum: float
    count: int | None = None
    mean: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    nan_count: int | None = None


@dataclass
class PromptCapture:
    index: int
    prompt: str
    rendered_prompt: str
    task: dict[str, Any] | None
    elapsed_sec: float
    returncode: int
    log_path: str
    stats_path: str
    tensor_count: int
    tensors: list[TensorSignal]


@dataclass
class ProbeCapture:
    model: str
    llama_debug: str
    mode: str
    chat_enable_thinking: bool
    append_text: str
    gemma4_thought_prefix: bool
    tensor_stats: bool
    tensor_stats_jsonl: bool
    tensor_filter: str
    n_gpu_layers: str
    ctx_size: int
    batch_size: int
    ubatch_size: int
    no_warmup: bool
    prompt_offset: int
    prompt_limit: int
    created_at: str
    captures: list[PromptCapture]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text)
    return out.strip("-")[:80] or "run"


def load_prompts(path: Path | None, prompt: str, offset: int, limit: int) -> list[str]:
    if path is None:
        prompts = [prompt]
    else:
        raw = path.read_text(encoding="utf-8").splitlines()
        prompts = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    if offset > 0:
        prompts = prompts[offset:]
    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def load_tasks(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    tasks: list[dict[str, Any]] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        if "prompt" not in item:
            continue
        tasks.append(dict(item))
    if offset > 0:
        tasks = tasks[offset:]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def render_prompt(prompt: str, mode: str, chat_enable_thinking: bool) -> str:
    if mode == "raw":
        return prompt
    if mode == "gemma4-chat":
        # llama.cpp Gemma4 chat template, single user turn, no tools/system.
        # Verified against llama-cli --conversation --single-turn --verbose-prompt.
        pieces: list[str] = []
        if chat_enable_thinking:
            pieces.append("<|turn>system\n<|think|>\n<turn|>\n")
        pieces.append(f"<|turn>user\n{prompt.strip()}<turn|>\n")
        pieces.append("<|turn>model\n")
        if not chat_enable_thinking:
            pieces.append("<|channel>thought\n<channel|>")
        return "".join(pieces)
    fail(f"unsupported mode: {mode}")
    return prompt


def rendered_prompt_with_prefix(args: argparse.Namespace, prompt: str, append_text: str | None = None) -> str:
    rendered = render_prompt(prompt, args.mode, args.chat_enable_thinking)
    append_value = args.append_text if append_text is None else append_text
    if args.gemma4_thought_prefix:
        if args.mode != "gemma4-chat":
            fail("--gemma4-thought-prefix requires --mode gemma4-chat")
        rendered += "<|channel>thought\n"
    if append_value:
        rendered += append_value
    return rendered


def parse_shape(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_tensor_signals(output: str) -> list[TensorSignal]:
    signals: list[TensorSignal] = []
    pending: dict[str, object] | None = None
    occurrences: dict[str, int] = {}
    for line in output.splitlines():
        match = TENSOR_RE.search(line)
        if match:
            name = match.group("name").strip()
            pending = {
                "name": name,
                "dtype": match.group("dtype").strip(),
                "op": match.group("op").strip(),
                "shape": parse_shape(match.group("shape")),
            }
            continue
        sum_match = SUM_RE.match(line)
        if sum_match and pending is not None:
            name = str(pending["name"])
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            signals.append(
                TensorSignal(
                    occurrence=occurrence,
                    name=name,
                    dtype=str(pending["dtype"]),
                    op=str(pending["op"]),
                    shape=list(pending["shape"]),  # type: ignore[arg-type]
                    sum=float(sum_match.group("value")),
                )
            )
            pending = None
            continue
        stats_match = STATS_RE.match(line)
        if stats_match and pending is not None:
            name = str(pending["name"])
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            signals.append(
                TensorSignal(
                    occurrence=occurrence,
                    name=name,
                    dtype=str(pending["dtype"]),
                    op=str(pending["op"]),
                    shape=list(pending["shape"]),  # type: ignore[arg-type]
                    sum=float(stats_match.group("sum")),
                    count=int(stats_match.group("count")),
                    mean=float(stats_match.group("mean")),
                    min_value=float(stats_match.group("min")),
                    max_value=float(stats_match.group("max")),
                    nan_count=int(stats_match.group("nan_count")),
                )
            )
            pending = None
    return signals


def json_number_to_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def parse_tensor_signals_jsonl(path: Path) -> list[TensorSignal]:
    signals: list[TensorSignal] = []
    occurrences: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid tensor stats JSONL at {path}:{line_no}: {exc}")

            name = str(record["name"]).strip()
            occurrence = occurrences.get(name, 0)
            occurrences[name] = occurrence + 1
            signals.append(
                TensorSignal(
                    occurrence=occurrence,
                    name=name,
                    dtype=str(record["dtype"]),
                    op=str(record["op"]),
                    shape=[int(value) for value in record["shape"]],
                    sum=json_number_to_float(record["sum"]),
                    count=int(record["count"]),
                    mean=json_number_to_float(record["mean"]),
                    min_value=json_number_to_float(record["min"]),
                    max_value=json_number_to_float(record["max"]),
                    nan_count=int(record["nan_count"]),
                )
            )
    return signals


def tensor_filter_from_args(args: argparse.Namespace) -> str:
    if args.tensor_filter:
        return args.tensor_filter
    if args.manifest_json:
        payload = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
        value = str(payload.get("debug_tensor_filter_regex", ""))
        if not value:
            fail(f"manifest has no debug_tensor_filter_regex: {args.manifest_json}")
        return value

    manifest = build_manifest(
        model=Path(args.model),
        policy=args.policy,
        layer_order=args.layer_order,
        params_b=args.params_b,
        quant_bpw=args.quant_bpw,
        active_percent=args.active_percent,
    )
    if not manifest.debug_tensor_filter_regex:
        fail("manifest produced an empty tensor filter")
    return manifest.debug_tensor_filter_regex


def build_command(
    args: argparse.Namespace,
    llama_debug: Path,
    model: Path,
    prompt: str,
    tensor_filter: str,
    tensor_stats_output: Path | None,
) -> list[str]:
    command = [
        str(llama_debug),
        "-m",
        str(model),
        "-p",
        prompt,
        "-ngl",
        str(args.ngl),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--tensor-filter",
        tensor_filter,
        "--verbose",
    ]
    if args.no_warmup:
        command.append("--no-warmup")
    if args.tensor_stats or tensor_stats_output is not None:
        command.append("--tensor-stats")
    if tensor_stats_output is not None:
        command.extend(["--tensor-stats-output", str(tensor_stats_output)])
    if args.threads > 0:
        command.extend(["-t", str(args.threads), "-tb", str(args.threads)])
    command.extend(args.extra_arg)
    return command


def build_env(cuda_bin: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_bin is not None and cuda_bin.is_dir():
        env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    return env


def run_prompt(
    *,
    args: argparse.Namespace,
    llama_debug: Path,
    model: Path,
    prompt: str,
    index: int,
    task: dict[str, Any] | None,
    append_text: str | None,
    tensor_filter: str,
    log_dir: Path,
    env: dict[str, str],
) -> PromptCapture:
    rendered_prompt = rendered_prompt_with_prefix(args, prompt, append_text)
    stats_path = log_dir / f"prompt-{index:03d}-tensor-stats.jsonl" if args.tensor_stats_jsonl else None
    command = build_command(args, llama_debug, model, rendered_prompt, tensor_filter, stats_path)
    if args.dry_run:
        print(" ".join(command))
        return PromptCapture(
            index=index,
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            task=task,
            elapsed_sec=0.0,
            returncode=0,
            log_path="",
            stats_path=str(stats_path.resolve()) if stats_path is not None else "",
            tensor_count=0,
            tensors=[],
        )

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = proc.stdout or ""
    log_path = log_dir / f"prompt-{index:03d}.log"
    log_path.write_text(output, encoding="utf-8")
    if stats_path is not None and stats_path.is_file():
        signals = parse_tensor_signals_jsonl(stats_path)
    else:
        signals = parse_tensor_signals(output)
    return PromptCapture(
        index=index,
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        task=task,
        elapsed_sec=elapsed,
        returncode=proc.returncode,
        log_path=str(log_path.resolve()),
        stats_path=str(stats_path.resolve()) if stats_path is not None else "",
        tensor_count=len(signals),
        tensors=signals,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SAGE sparse-verifier tensor signals with llama-debug.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-debug", default=str(DEFAULT_LLAMA_DEBUG))
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--prompts")
    parser.add_argument("--tasks-json", default="", help="candidate task JSON from sage_candidate_tasks.py")
    parser.add_argument("--mode", choices=["raw", "gemma4-chat"], default="raw")
    parser.add_argument("--chat-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-text", default="", help="append raw text after chat/prefix rendering before capture")
    parser.add_argument("--gemma4-thought-prefix", action="store_true", help="append <|channel>thought newline for Gemma4 content-token capture")
    parser.add_argument("--offset", type=int, default=0, help="skip this many prompts before applying --limit")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tensor-filter", default="", help="manual llama-debug tensor-filter regex")
    parser.add_argument("--manifest-json", default="", help="manifest JSON containing debug_tensor_filter_regex")
    parser.add_argument("--policy", choices=["ffn-sentinel", "attention-sentinel", "hybrid"], default="hybrid")
    parser.add_argument("--layer-order", choices=["boundary", "early", "late", "middle"], default="boundary")
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--active-percent", type=float, default=1.0)
    parser.add_argument("--ngl", default="0")
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ubatch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-stats", action="store_true", help="pass patched llama-debug --tensor-stats and parse compact stats lines")
    parser.add_argument("--tensor-stats-jsonl", action="store_true", help="use patched llama-debug --tensor-stats-output JSONL sidecar")
    parser.add_argument("--cuda-bin", default=str(DEFAULT_CUDA_BIN))
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.offset < 0:
        parser.error("--offset must be non-negative")
    llama_debug = ensure_file(Path(args.llama_debug), "llama-debug")
    model = ensure_file(Path(args.model), "model")
    if args.tasks_json:
        tasks_path = ensure_file(Path(args.tasks_json), "tasks JSON")
        tasks = load_tasks(tasks_path, args.offset, args.limit)
    else:
        prompt_path = Path(args.prompts) if args.prompts else None
        prompts = load_prompts(prompt_path, args.prompt, args.offset, args.limit)
        tasks = [{"prompt": prompt, "append_text": args.append_text} for prompt in prompts]
    if not tasks:
        fail("no prompts to capture")
    tensor_filter = tensor_filter_from_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = safe_name(args.tag or f"{Path(args.model).stem}-{args.policy}-{args.active_percent:g}pct")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) / f"{timestamp}-{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cuda_bin = Path(args.cuda_bin) if args.cuda_bin else None
    env = build_env(cuda_bin)

    captures: list[PromptCapture] = []
    for index, task in enumerate(tasks, start=1):
        prompt = str(task.get("prompt", ""))
        append_text = str(task.get("append_text", args.append_text))
        task_id = str(task.get("task_id", f"prompt-{index:03d}"))
        print(f"probe: prompt {index}/{len(tasks)} ({task_id})")
        capture = run_prompt(
            args=args,
            llama_debug=llama_debug,
            model=model,
            prompt=prompt,
            index=index,
            task=task,
            append_text=append_text,
            tensor_filter=tensor_filter,
            log_dir=log_dir,
            env=env,
        )
        captures.append(capture)
        if capture.returncode != 0:
            fail(f"llama-debug failed for prompt {index}; see {capture.log_path}", code=capture.returncode)

    payload = ProbeCapture(
        model=str(model),
        llama_debug=str(llama_debug),
        mode=args.mode,
        chat_enable_thinking=args.chat_enable_thinking,
        append_text=args.append_text,
        gemma4_thought_prefix=args.gemma4_thought_prefix,
        tensor_stats=args.tensor_stats,
        tensor_stats_jsonl=args.tensor_stats_jsonl,
        tensor_filter=tensor_filter,
        n_gpu_layers=str(args.ngl),
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        no_warmup=args.no_warmup,
        prompt_offset=args.offset,
        prompt_limit=args.limit,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        captures=captures,
    )
    text = json.dumps(asdict(payload), indent=2)
    json_out = Path(args.json_out) if args.json_out else out_dir / f"{timestamp}-sage-probe-{run_tag}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(text, encoding="utf-8")
    total = sum(capture.tensor_count for capture in captures)
    print(f"wrote: {json_out.resolve()}")
    print(f"captured tensor summaries: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
