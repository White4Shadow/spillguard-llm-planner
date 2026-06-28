#!/usr/bin/env python3
"""
Capture SAGE tensor-stat signals from a normal llama.cpp generation binary.

Unlike sage_probe_capture.py, this does not use llama-debug. It exercises the
non-debug runtime hook added by patches/llama.cpp/0006-*.patch through
llama-completion and writes parsed JSON summaries for downstream policy checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_COMPLETION = ROOT / "tools" / "llama.cpp-src" / "build-live-migration-cuda" / "bin" / "Release" / "llama-completion.exe"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LOG_DIR = DEFAULT_OUT_DIR / "sage-runtime-logs"
DEFAULT_CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64")


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
class RuntimeCapture:
    index: int
    prompt: str
    rendered_prompt: str
    task: dict[str, Any] | None
    elapsed_sec: float
    returncode: int
    log_path: str
    stats_path: str
    decision_path: str
    tensor_count: int
    tensors: list[TensorSignal]
    decision_count: int
    decisions: list[dict[str, Any]]


@dataclass
class RuntimeCaptureRun:
    model: str
    llama_completion: str
    mode: str
    chat_enable_thinking: bool
    append_text: str
    gemma4_thought_prefix: bool
    tensor_filter: str
    n_gpu_layers: str
    ctx_size: int
    batch_size: int
    ubatch_size: int
    tokens: int
    no_conversation: bool
    no_warmup: bool
    capture_decisions: bool
    prompt_offset: int
    prompt_limit: int
    created_at: str
    captures: list[RuntimeCapture]


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
                record: dict[str, Any] = json.loads(line)
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


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {path}:{line_no}: {exc}")
            if isinstance(record, dict):
                records.append(record)
    return records


def build_env(cuda_bin: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_bin is not None and cuda_bin.is_dir():
        env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    return env


def build_command(
    *,
    args: argparse.Namespace,
    llama_completion: Path,
    model: Path,
    rendered_prompt: str,
    stats_path: Path,
    decision_path: Path | None,
    task: dict[str, Any] | None,
    index: int,
) -> list[str]:
    command = [
        str(llama_completion),
        "-m",
        str(model),
        "-p",
        rendered_prompt,
        "-n",
        str(args.tokens),
        "-ngl",
        str(args.ngl),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--sage-tensor-filter",
        args.tensor_filter,
        "--sage-tensor-stats-output",
        str(stats_path),
    ]
    if decision_path is not None:
        task_id = str((task or {}).get("task_id", f"prompt-{index:03d}"))
        token_class = str((task or {}).get("token_class", ""))
        proxy_entropy = float((task or {}).get("proxy_entropy", 0.0))
        proxy_margin = float((task or {}).get("proxy_margin", 0.0))
        command.extend(
            [
                "--sage-decision-output",
                str(decision_path),
                "--sage-candidate-id",
                task_id,
                "--sage-token-class",
                token_class,
                "--sage-proxy-entropy",
                str(proxy_entropy),
                "--sage-proxy-margin",
                str(proxy_margin),
            ]
        )
    if args.no_conversation:
        command.append("-no-cnv")
    if args.no_warmup:
        command.append("--no-warmup")
    if args.threads > 0:
        command.extend(["-t", str(args.threads), "-tb", str(args.threads)])
    command.extend(args.extra_arg)
    return command


def run_prompt(
    *,
    args: argparse.Namespace,
    llama_completion: Path,
    model: Path,
    prompt: str,
    rendered_prompt: str,
    index: int,
    task: dict[str, Any] | None,
    log_dir: Path,
    env: dict[str, str],
) -> RuntimeCapture:
    stats_path = log_dir / f"prompt-{index:03d}-sage-runtime.jsonl"
    decision_path = log_dir / f"prompt-{index:03d}-sage-decision.jsonl" if args.capture_decisions else None
    command = build_command(
        args=args,
        llama_completion=llama_completion,
        model=model,
        rendered_prompt=rendered_prompt,
        stats_path=stats_path,
        decision_path=decision_path,
        task=task,
        index=index,
    )
    if args.dry_run:
        print(" ".join(command))
        return RuntimeCapture(
            index=index,
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            task=task,
            elapsed_sec=0.0,
            returncode=0,
            log_path="",
            stats_path=str(stats_path.resolve()),
            decision_path=str(decision_path.resolve()) if decision_path is not None else "",
            tensor_count=0,
            tensors=[],
            decision_count=0,
            decisions=[],
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
    signals = parse_tensor_signals_jsonl(stats_path) if stats_path.is_file() else []
    decisions = parse_jsonl(decision_path) if decision_path is not None and decision_path.is_file() else []
    return RuntimeCapture(
        index=index,
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        task=task,
        elapsed_sec=elapsed,
        returncode=proc.returncode,
        log_path=str(log_path.resolve()),
        stats_path=str(stats_path.resolve()),
        decision_path=str(decision_path.resolve()) if decision_path is not None else "",
        tensor_count=len(signals),
        tensors=signals,
        decision_count=len(decisions),
        decisions=decisions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SAGE runtime tensor signals with llama-completion.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-completion", default=str(DEFAULT_LLAMA_COMPLETION))
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--prompts")
    parser.add_argument("--tasks-json", default="", help="candidate task JSON from sage_candidate_tasks.py")
    parser.add_argument("--mode", choices=["raw", "gemma4-chat"], default="raw")
    parser.add_argument("--chat-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-text", default="", help="append raw text after chat/prefix rendering before capture")
    parser.add_argument("--gemma4-thought-prefix", action="store_true", help="append <|channel>thought newline for Gemma4 content-token capture")
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
    parser.add_argument("--no-conversation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-decisions", action="store_true", help="capture in-process SAGE scheduler decision JSONL")
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
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    llama_completion = ensure_file(Path(args.llama_completion), "llama-completion")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = safe_name(args.tag or f"{Path(args.model).stem}-runtime")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) / f"{timestamp}-{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cuda_bin = Path(args.cuda_bin) if args.cuda_bin else None
    env = build_env(cuda_bin)

    captures: list[RuntimeCapture] = []
    for index, task in enumerate(tasks, start=1):
        prompt = str(task.get("prompt", ""))
        append_text = str(task.get("append_text", args.append_text))
        task_id = str(task.get("task_id", f"prompt-{index:03d}"))
        rendered_prompt = rendered_prompt_with_prefix(args, prompt, append_text)
        print(f"runtime: prompt {index}/{len(tasks)} ({task_id})")
        capture = run_prompt(
            args=args,
            llama_completion=llama_completion,
            model=model,
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            index=index,
            task=task,
            log_dir=log_dir,
            env=env,
        )
        captures.append(capture)
        if capture.returncode != 0:
            fail(f"llama-completion failed for prompt {index}; see {capture.log_path}", code=capture.returncode)

    payload = RuntimeCaptureRun(
        model=str(model),
        llama_completion=str(llama_completion),
        mode=args.mode,
        chat_enable_thinking=args.chat_enable_thinking,
        append_text=args.append_text,
        gemma4_thought_prefix=args.gemma4_thought_prefix,
        tensor_filter=args.tensor_filter,
        n_gpu_layers=str(args.ngl),
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        tokens=args.tokens,
        no_conversation=args.no_conversation,
        no_warmup=args.no_warmup,
        capture_decisions=args.capture_decisions,
        prompt_offset=args.offset,
        prompt_limit=args.limit,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        captures=captures,
    )
    text = json.dumps(asdict(payload), indent=2)
    json_out = Path(args.json_out) if args.json_out else out_dir / f"{timestamp}-sage-runtime-{run_tag}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(text, encoding="utf-8")
    total = sum(capture.tensor_count for capture in captures)
    decision_total = sum(capture.decision_count for capture in captures)
    print(f"wrote: {json_out.resolve()}")
    print(f"captured tensor summaries: {total}")
    if args.capture_decisions:
        print(f"captured scheduler decisions: {decision_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
