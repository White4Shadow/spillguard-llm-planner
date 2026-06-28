#!/usr/bin/env python3
"""
Replay oracle labels for SAGE router candidates on the proxy prefix.

The original proxy/oracle logprob calibration compares two independently
generated continuations. Once they diverge, later labels are weak. This script
fixes that by asking the oracle to predict the candidate token from:

    rendered chat prompt + thought-channel prefix + proxy-generated prefix

The output can replace exploratory labels in candidate verifier fitting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-server.exe"
DEFAULT_ORACLE = ROOT / "models" / "gemma-4-31b-it-qat-q4_0-gguf" / "gemma-4-31B_q4_0-it.gguf"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LOG_DIR = DEFAULT_OUT_DIR / "sage-replay-logs"


@dataclass
class TopToken:
    id: int
    token: str
    logprob: float


@dataclass
class ReplayRow:
    task: dict[str, Any]
    rendered_prompt: str
    oracle_token_id: int
    oracle_token: str
    oracle_logprob: float
    oracle_top_logprobs: list[TopToken]
    oracle_margin: float
    oracle_entropy: float
    proxy_token_in_oracle_topk: bool
    replay_top1_token_match: bool
    replay_top1_id_match: bool
    original_top1_token_match: bool
    original_label_quality: str
    elapsed_sec: float


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


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body) if body else {}


def wait_for_server(base_url: str, timeout: int) -> None:
    deadline = time.perf_counter() + timeout
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            http_json("GET", f"{base_url}/health", timeout=2)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready: {last_error}")


def server_args(
    *,
    server: Path,
    model: Path,
    n_gpu_layers: str,
    ctx_size: int,
    threads: int,
    host: str,
    port: int,
    cache_type: str,
    cache_prompt: bool,
) -> list[str]:
    args = [
        str(server),
        "-m",
        str(model),
        "-c",
        str(ctx_size),
        "-ngl",
        n_gpu_layers,
        "-t",
        str(threads),
        "-ctk",
        cache_type,
        "-ctv",
        cache_type,
        "--host",
        host,
        "--port",
        str(port),
        "--no-warmup",
        "--no-webui",
        "--log-disable",
    ]
    args.append("--cache-prompt" if cache_prompt else "--no-cache-prompt")
    return args


class ManagedServer:
    def __init__(
        self,
        *,
        server: Path,
        model: Path,
        n_gpu_layers: str,
        ctx_size: int,
        threads: int,
        host: str,
        port: int,
        cache_type: str,
        cache_prompt: bool,
        log_dir: Path,
        label: str,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: subprocess.Popen[bytes] | None = None
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        base = f"{stamp}-{safe_name(label)}"
        self.stdout_log = log_dir / f"{base}.out.log"
        self.stderr_log = log_dir / f"{base}.err.log"
        self.argv = server_args(
            server=server,
            model=model,
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
            threads=threads,
            host=host,
            port=port,
            cache_type=cache_type,
            cache_prompt=cache_prompt,
        )

    def start(self, ready_timeout: int) -> None:
        stdout = self.stdout_log.open("wb")
        stderr = self.stderr_log.open("wb")
        try:
            self.process = subprocess.Popen(self.argv, cwd=ROOT, stdout=stdout, stderr=stderr)
            wait_for_server(self.base_url, ready_timeout)
        except Exception:
            stdout.close()
            stderr.close()
            self.stop()
            raise
        self._stdout_handle = stdout
        self._stderr_handle = stderr

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)
        for attr in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, attr, None)
            if handle is not None and not handle.closed:
                handle.close()


def load_tasks(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    tasks = [dict(item) for item in raw_tasks if isinstance(item, dict) and "prompt" in item]
    if offset > 0:
        tasks = tasks[offset:]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def render_prompt(prompt: str, mode: str, chat_enable_thinking: bool, gemma4_thought_prefix: bool, append_text: str) -> str:
    if mode == "raw":
        rendered = prompt
    elif mode == "gemma4-chat":
        pieces: list[str] = []
        if chat_enable_thinking:
            pieces.append("<|turn>system\n<|think|>\n<turn|>\n")
        pieces.append(f"<|turn>user\n{prompt.strip()}<turn|>\n")
        pieces.append("<|turn>model\n")
        if not chat_enable_thinking:
            pieces.append("<|channel>thought\n<channel|>")
        rendered = "".join(pieces)
    else:
        fail(f"unsupported mode: {mode}")
    if gemma4_thought_prefix:
        if mode != "gemma4-chat":
            fail("--gemma4-thought-prefix requires --mode gemma4-chat")
        rendered += "<|channel>thought\n"
    return rendered + append_text


def topk_entropy(top: list[TopToken]) -> float:
    if not top:
        return 0.0
    probs = [math.exp(item.logprob) for item in top]
    total = sum(probs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for prob in probs:
        p = prob / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def parse_step(raw: dict[str, Any]) -> tuple[int, str, float, list[TopToken], float, float]:
    top = [
        TopToken(
            id=int(item.get("id", -1)),
            token=str(item.get("token", "")),
            logprob=float(item.get("logprob", 0.0)),
        )
        for item in raw.get("top_logprobs", [])
    ]
    if top:
        first = top[0]
        margin = first.logprob - top[1].logprob if len(top) > 1 else math.inf
        token_id = int(raw.get("id", first.id))
        token = str(raw.get("token", first.token))
        logprob = float(raw.get("logprob", first.logprob))
    else:
        token_id = int(raw.get("id", -1))
        token = str(raw.get("token", ""))
        logprob = float(raw.get("logprob", 0.0))
        margin = 0.0
    return token_id, token, logprob, top, margin, topk_entropy(top)


def replay_task(base_url: str, model: Path, task: dict[str, Any], args: argparse.Namespace) -> ReplayRow:
    rendered = render_prompt(
        prompt=str(task.get("prompt", "")),
        mode=args.mode,
        chat_enable_thinking=args.chat_enable_thinking,
        gemma4_thought_prefix=args.gemma4_thought_prefix,
        append_text=str(task.get("append_text", "")),
    )
    payload = {
        "prompt": rendered,
        "n_predict": 1,
        "temperature": 0,
        "top_k": 1,
        "n_probs": args.top_k,
        "stream": False,
        "cache_prompt": args.cache_prompt,
    }
    if args.completion_special is not None:
        payload["special"] = args.completion_special
    start = time.perf_counter()
    response = http_json("POST", f"{base_url}/completion", payload, timeout=args.request_timeout)
    elapsed = time.perf_counter() - start
    steps_raw = response.get("completion_probabilities", [])
    if not steps_raw:
        fail(f"oracle replay returned no completion_probabilities for {task.get('task_id', '<unknown>')}")
    token_id, token, logprob, top, margin, entropy = parse_step(steps_raw[0])
    proxy_token = str(task.get("proxy_token", ""))
    proxy_token_id = int(task.get("proxy_token_id", -999999)) if "proxy_token_id" in task else None
    return ReplayRow(
        task=task,
        rendered_prompt=rendered,
        oracle_token_id=token_id,
        oracle_token=token,
        oracle_logprob=logprob,
        oracle_top_logprobs=top,
        oracle_margin=margin,
        oracle_entropy=entropy,
        proxy_token_in_oracle_topk=any(item.token == proxy_token for item in top),
        replay_top1_token_match=token == proxy_token,
        replay_top1_id_match=(proxy_token_id == token_id) if proxy_token_id is not None else False,
        original_top1_token_match=bool(task.get("top1_token_match", False)),
        original_label_quality=str(task.get("label_quality", "")),
        elapsed_sec=elapsed,
    )


def summarize(rows: list[ReplayRow]) -> dict[str, Any]:
    total = len(rows)
    matches = sum(1 for row in rows if row.replay_top1_token_match)
    original_matches = sum(1 for row in rows if row.original_top1_token_match)
    changed = sum(1 for row in rows if row.replay_top1_token_match != row.original_top1_token_match)
    same_prefix = [row for row in rows if row.original_label_quality == "same-prefix"]
    diverged = [row for row in rows if row.original_label_quality == "diverged-prefix"]
    return {
        "tasks": total,
        "replay_top1_token_matches": matches,
        "replay_top1_token_match_rate": matches / total if total else 0.0,
        "original_top1_token_matches": original_matches,
        "original_top1_token_match_rate": original_matches / total if total else 0.0,
        "label_changes": changed,
        "proxy_token_in_oracle_topk": sum(1 for row in rows if row.proxy_token_in_oracle_topk),
        "same_prefix_tasks": len(same_prefix),
        "same_prefix_replay_matches": sum(1 for row in same_prefix if row.replay_top1_token_match),
        "diverged_prefix_tasks": len(diverged),
        "diverged_prefix_replay_matches": sum(1 for row in diverged if row.replay_top1_token_match),
        "mean_oracle_margin": sum(row.oracle_margin for row in rows) / total if total else 0.0,
        "mean_elapsed_sec": sum(row.elapsed_sec for row in rows) / total if total else 0.0,
    }


def write_output(rows: list[ReplayRow], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.json_out) if args.json_out else out_dir / f"{stamp}-sage-candidate-replay-{safe_name(args.tag)}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {
            "tasks_json": str(Path(args.tasks_json).resolve()),
            "oracle_model": str(Path(args.oracle_model).resolve()),
            "oracle_ngl": args.oracle_ngl,
            "top_k": args.top_k,
            "mode": args.mode,
            "chat_enable_thinking": args.chat_enable_thinking,
            "gemma4_thought_prefix": args.gemma4_thought_prefix,
            "completion_special": args.completion_special,
            "cache_prompt": args.cache_prompt,
            "offset": args.offset,
            "limit": args.limit,
        },
        "summary": summarize(rows),
        "rows": [asdict(row) for row in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(rows: list[ReplayRow], out_path: Path) -> None:
    summary = summarize(rows)
    print("# SAGE Candidate Oracle Replay")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"| {key} | {value:.3f} |")
        else:
            print(f"| {key} | {value} |")
    print()
    print(f"wrote: {out_path.resolve()}")
    print()
    print("| Task | Quality | Proxy | Replay oracle | Match | Original |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows[:20]:
        task_id = str(row.task.get("task_id", ""))
        proxy = str(row.task.get("proxy_token", "")).replace("|", "\\|")
        oracle = row.oracle_token.replace("|", "\\|")
        print(
            f"| {task_id} | {row.original_label_quality} | `{proxy}` | `{oracle}` "
            f"| {row.replay_top1_token_match} | {row.original_top1_token_match} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay oracle labels for SAGE router candidates on proxy prefixes.")
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--llama-server", default=str(DEFAULT_SERVER))
    parser.add_argument("--oracle-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--oracle-ngl", default="38")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["raw", "gemma4-chat"], default="gemma4-chat")
    parser.add_argument("--chat-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gemma4-thought-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--completion-special", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache-type", default="q8_0")
    parser.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--tag", default="gemma31b")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")

    tasks_path = ensure_file(Path(args.tasks_json), "tasks JSON")
    server = ensure_file(Path(args.llama_server), "llama-server")
    oracle_model = ensure_file(Path(args.oracle_model), "oracle model")
    tasks = load_tasks(tasks_path, args.offset, args.limit)
    if not tasks:
        parser.error("no tasks to replay")

    port = args.port if args.port else find_free_port(args.host)
    if args.dry_run:
        print("oracle server command:")
        print(
            json.dumps(
                server_args(
                    server=server,
                    model=oracle_model,
                    n_gpu_layers=args.oracle_ngl,
                    ctx_size=args.ctx_size,
                    threads=args.threads,
                    host=args.host,
                    port=port,
                    cache_type=args.cache_type,
                    cache_prompt=args.cache_prompt,
                ),
                indent=2,
            )
        )
        print("first rendered prompt:")
        print(render_prompt(str(tasks[0].get("prompt", "")), args.mode, args.chat_enable_thinking, args.gemma4_thought_prefix, str(tasks[0].get("append_text", ""))))
        return 0

    managed = ManagedServer(
        server=server,
        model=oracle_model,
        n_gpu_layers=args.oracle_ngl,
        ctx_size=args.ctx_size,
        threads=args.threads,
        host=args.host,
        port=port,
        cache_type=args.cache_type,
        cache_prompt=args.cache_prompt,
        log_dir=Path(args.log_dir),
        label="oracle-replay",
    )
    print(f"replay: starting oracle server on {managed.base_url}", flush=True)
    print(f"replay: server logs {managed.stdout_log} | {managed.stderr_log}", flush=True)
    managed.start(args.ready_timeout)
    rows: list[ReplayRow] = []
    try:
        for index, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id", f"task-{index:04d}"))
            print(f"replay: task {index}/{len(tasks)} ({task_id})", flush=True)
            rows.append(replay_task(managed.base_url, oracle_model, task, args))
    finally:
        managed.stop()

    out_path = write_output(rows, args)
    print_summary(rows, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
