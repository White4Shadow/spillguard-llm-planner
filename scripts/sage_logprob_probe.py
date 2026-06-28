#!/usr/bin/env python3
"""
Top-k logprob agreement probe for SAGE-100.

This script uses llama-server's /completion endpoint with n_probs to compare a
resident proxy model against a larger oracle at the token-distribution level.
That is a stronger gate than text-only agreement:

- top-1 token match
- top-k token/id overlap
- proxy confidence margin
- approximate entropy over returned top-k probabilities

The script starts one local server per model sequentially so it does not require
both proxy and oracle to fit in VRAM at the same time.
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
DEFAULT_PROXY = ROOT / "models" / "gemma-4-12b-it-qat-q4_0-gguf" / "gemma-4-12b-it-qat-q4_0.gguf"
DEFAULT_ORACLE = ROOT / "models" / "gemma-4-31b-it-qat-q4_0-gguf" / "gemma-4-31B_q4_0-it.gguf"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LOG_DIR = DEFAULT_OUT_DIR / "sage-logprob-logs"


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In Python, a list comprehension is used to",
    "The main reason GPUs are useful for neural networks is",
    "When a program runs out of GPU memory, it can",
    "A transformer attention head computes",
    "The derivative of x squared is",
    "To sort a list in ascending order,",
    "The opposite of hot is",
    "A good unit test should",
    "The first step in debugging a crash is",
]


@dataclass
class TopToken:
    id: int
    token: str
    logprob: float


@dataclass
class ModelStep:
    token_id: int
    token: str
    logprob: float
    top_logprobs: list[TopToken]
    top1_margin: float
    topk_entropy: float


@dataclass
class ModelProbe:
    model: str
    n_gpu_layers: str
    elapsed_sec: float
    content: str
    timings: dict[str, Any]
    steps: list[ModelStep]


@dataclass
class StepComparison:
    index: int
    top1_id_match: bool
    top1_token_match: bool
    topk_id_overlap: int
    topk_token_overlap: int
    topk_id_jaccard: float
    topk_token_jaccard: float
    proxy_margin: float
    proxy_entropy: float
    oracle_entropy: float


@dataclass
class ProbeRow:
    prompt: str
    proxy: ModelProbe
    oracle: ModelProbe
    steps: list[StepComparison]


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def load_prompts(path: Path | None, offset: int, limit: int) -> list[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    else:
        raw = path.read_text(encoding="utf-8").splitlines()
        prompts = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    if offset > 0:
        prompts = prompts[offset:]
    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def safe_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text)
    return out.strip("-")[:80] or "run"


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
) -> list[str]:
    return [
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


def parse_step(raw: dict[str, Any]) -> ModelStep:
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
        margin = 0.0
        token_id = int(raw.get("id", -1))
        token = str(raw.get("token", ""))
        logprob = float(raw.get("logprob", 0.0))

    return ModelStep(
        token_id=token_id,
        token=token,
        logprob=logprob,
        top_logprobs=top,
        top1_margin=margin,
        topk_entropy=topk_entropy(top),
    )


def probe_prompt(
    *,
    base_url: str,
    model: Path,
    n_gpu_layers: str,
    prompt: str,
    tokens: int,
    top_k: int,
    mode: str,
    request_timeout: int,
) -> ModelProbe:
    start = time.perf_counter()
    if mode == "chat":
        payload = {
            "model": model.stem,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top_k,
            "stream": False,
        }
        response = http_json("POST", f"{base_url}/v1/chat/completions", payload, timeout=request_timeout)
        choice = response.get("choices", [{}])[0]
        content = str(choice.get("message", {}).get("content", ""))
        steps_raw = choice.get("logprobs", {}).get("content", [])
    else:
        payload = {
            "prompt": prompt,
            "n_predict": tokens,
            "temperature": 0,
            "top_k": 1,
            "n_probs": top_k,
            "stream": False,
            "cache_prompt": False,
        }
        response = http_json("POST", f"{base_url}/completion", payload, timeout=request_timeout)
        content = str(response.get("content", ""))
        steps_raw = response.get("completion_probabilities", [])
    elapsed = time.perf_counter() - start
    steps = [parse_step(item) for item in steps_raw]
    return ModelProbe(
        model=str(model.resolve()),
        n_gpu_layers=n_gpu_layers,
        elapsed_sec=elapsed,
        content=content,
        timings=response.get("timings", {}),
        steps=steps,
    )


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

    def __enter__(self) -> "ManagedServer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()


def run_model_probes(
    *,
    server: Path,
    model: Path,
    n_gpu_layers: str,
    prompts: list[str],
    tokens: int,
    top_k: int,
    mode: str,
    ctx_size: int,
    threads: int,
    cache_type: str,
    host: str,
    port: int,
    log_dir: Path,
    label: str,
    ready_timeout: int,
    request_timeout: int,
) -> list[ModelProbe]:
    managed = ManagedServer(
        server=server,
        model=model,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        threads=threads,
        host=host,
        port=port,
        cache_type=cache_type,
        log_dir=log_dir,
        label=label,
    )
    print(f"logprob: starting {label} server on {managed.base_url}", flush=True)
    print(f"logprob: server logs {managed.stdout_log} | {managed.stderr_log}", flush=True)
    managed.start(ready_timeout)
    try:
        results: list[ModelProbe] = []
        for idx, prompt in enumerate(prompts, start=1):
            print(f"logprob: {label} prompt {idx}/{len(prompts)}", flush=True)
            results.append(
                probe_prompt(
                    base_url=managed.base_url,
                    model=model,
                    n_gpu_layers=n_gpu_layers,
                    prompt=prompt,
                    tokens=tokens,
                    top_k=top_k,
                    mode=mode,
                    request_timeout=request_timeout,
                )
            )
        return results
    finally:
        managed.stop()


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def compare_steps(proxy: ModelStep, oracle: ModelStep, index: int) -> StepComparison:
    proxy_ids = {item.id for item in proxy.top_logprobs}
    oracle_ids = {item.id for item in oracle.top_logprobs}
    proxy_tokens = {item.token for item in proxy.top_logprobs}
    oracle_tokens = {item.token for item in oracle.top_logprobs}
    return StepComparison(
        index=index,
        top1_id_match=proxy.token_id == oracle.token_id,
        top1_token_match=proxy.token == oracle.token,
        topk_id_overlap=len(proxy_ids & oracle_ids),
        topk_token_overlap=len(proxy_tokens & oracle_tokens),
        topk_id_jaccard=jaccard(proxy_ids, oracle_ids),
        topk_token_jaccard=jaccard(proxy_tokens, oracle_tokens),
        proxy_margin=proxy.top1_margin,
        proxy_entropy=proxy.topk_entropy,
        oracle_entropy=oracle.topk_entropy,
    )


def make_rows(prompts: list[str], proxy: list[ModelProbe], oracle: list[ModelProbe]) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    for prompt, proxy_probe, oracle_probe in zip(prompts, proxy, oracle):
        step_count = min(len(proxy_probe.steps), len(oracle_probe.steps))
        comparisons = [
            compare_steps(proxy_probe.steps[idx], oracle_probe.steps[idx], idx)
            for idx in range(step_count)
        ]
        rows.append(ProbeRow(prompt=prompt, proxy=proxy_probe, oracle=oracle_probe, steps=comparisons))
    return rows


def summarize(rows: list[ProbeRow], ignore_prefix_steps: int = 0) -> dict[str, Any]:
    comparisons = [
        step
        for row in rows
        for step in row.steps
        if step.index >= ignore_prefix_steps
    ]
    total = len(comparisons)
    top1_id = sum(1 for step in comparisons if step.top1_id_match)
    top1_token = sum(1 for step in comparisons if step.top1_token_match)
    return {
        "prompts": len(rows),
        "ignored_prefix_steps_per_prompt": ignore_prefix_steps,
        "steps": total,
        "top1_id_matches": top1_id,
        "top1_id_match_rate": top1_id / total if total else 0.0,
        "top1_token_matches": top1_token,
        "top1_token_match_rate": top1_token / total if total else 0.0,
        "mean_topk_id_jaccard": sum(step.topk_id_jaccard for step in comparisons) / total if total else 0.0,
        "mean_topk_token_jaccard": sum(step.topk_token_jaccard for step in comparisons) / total if total else 0.0,
        "mean_proxy_margin": sum(step.proxy_margin for step in comparisons) / total if total else 0.0,
        "mean_proxy_entropy": sum(step.proxy_entropy for step in comparisons) / total if total else 0.0,
        "mean_oracle_entropy": sum(step.oracle_entropy for step in comparisons) / total if total else 0.0,
        "proxy_total_sec": sum(row.proxy.elapsed_sec for row in rows),
        "oracle_total_sec": sum(row.oracle.elapsed_sec for row in rows),
    }


def write_output(rows: list[ProbeRow], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{stamp}-sage-logprob-{args.tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {
            "proxy_model": str(Path(args.proxy_model).resolve()),
            "oracle_model": str(Path(args.oracle_model).resolve()),
            "proxy_ngl": args.proxy_ngl,
            "oracle_ngl": args.oracle_ngl,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "mode": args.mode,
            "prompts": str(Path(args.prompts).resolve()) if args.prompts else "",
            "offset": args.offset,
            "limit": args.limit,
            "ctx_size": args.ctx_size,
            "threads": args.threads,
            "cache_type": args.cache_type,
        },
        "summary": summarize(rows, args.ignore_prefix_steps),
        "rows": [asdict(row) for row in rows],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(rows: list[ProbeRow], out_path: Path, ignore_prefix_steps: int = 0) -> None:
    summary = summarize(rows, ignore_prefix_steps)
    print()
    print("SAGE logprob summary")
    print(f"  prompts: {summary['prompts']}")
    if ignore_prefix_steps:
        print(f"  ignored prefix steps per prompt: {ignore_prefix_steps}")
    print(f"  compared steps: {summary['steps']}")
    print(f"  top1 id match: {summary['top1_id_matches']}/{summary['steps']} ({summary['top1_id_match_rate']:.1%})")
    print(f"  top1 token match: {summary['top1_token_matches']}/{summary['steps']} ({summary['top1_token_match_rate']:.1%})")
    print(f"  mean top-k token Jaccard: {summary['mean_topk_token_jaccard']:.3f}")
    print(f"  mean proxy margin: {summary['mean_proxy_margin']:.3f}")
    print(f"  wrote: {out_path}")
    print()
    for row in rows:
        visible_steps = [step for step in row.steps if step.index >= ignore_prefix_steps]
        if not visible_steps:
            print(f"?? {row.prompt}")
            continue
        step = visible_steps[0]
        mark = "==" if step.top1_token_match else "!="
        proxy_top = row.proxy.steps[step.index].token if len(row.proxy.steps) > step.index else ""
        oracle_top = row.oracle.steps[step.index].token if len(row.oracle.steps) > step.index else ""
        print(f"{mark} {row.prompt}")
        print(f"  proxy top1 : {proxy_top!r}")
        print(f"  oracle top1: {oracle_top!r}")
        print(f"  top-k token Jaccard: {step.topk_token_jaccard:.3f}, proxy margin: {step.proxy_margin:.3f}")


def self_test(out_dir: Path) -> int:
    proxy = ModelProbe(
        model="mock-proxy",
        n_gpu_layers="all",
        elapsed_sec=0.1,
        content=" Paris",
        timings={},
        steps=[
            ModelStep(
                token_id=1,
                token=" Paris",
                logprob=-0.1,
                top_logprobs=[TopToken(1, " Paris", -0.1), TopToken(2, " Lyon", -3.0)],
                top1_margin=2.9,
                topk_entropy=0.2,
            )
        ],
    )
    oracle = ModelProbe(
        model="mock-oracle",
        n_gpu_layers="38",
        elapsed_sec=0.2,
        content=" Paris",
        timings={},
        steps=[
            ModelStep(
                token_id=1,
                token=" Paris",
                logprob=-0.2,
                top_logprobs=[TopToken(1, " Paris", -0.2), TopToken(3, " Marseille", -3.4)],
                top1_margin=3.2,
                topk_entropy=0.18,
            )
        ],
    )
    rows = make_rows(["The capital of France is"], [proxy], [oracle])
    args = argparse.Namespace(
        out_dir=str(out_dir),
        tag="self-test",
        proxy_model="mock-proxy",
        oracle_model="mock-oracle",
        proxy_ngl="all",
        oracle_ngl="38",
        tokens=1,
        top_k=2,
        mode="raw",
        prompts="",
        offset=0,
        limit=1,
        ignore_prefix_steps=0,
        ctx_size=512,
        threads=1,
        cache_type="q8_0",
    )
    out_path = write_output(rows, args)
    print_summary(rows, out_path, args.ignore_prefix_steps)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare proxy/oracle top-k logprobs using llama-server.")
    parser.add_argument("--llama-server", default=str(DEFAULT_SERVER))
    parser.add_argument("--proxy-model", default=str(DEFAULT_PROXY))
    parser.add_argument("--oracle-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--proxy-ngl", default="all")
    parser.add_argument("--oracle-ngl", default="38")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--ignore-prefix-steps", type=int, default=0, help="ignore the first N generated steps in summary metrics")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache-type", default="q8_0")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free local port")
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tag", default="gemma12b-to-31b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test(Path(args.out_dir))
    if args.tokens <= 0:
        fail("--tokens must be positive")
    if args.top_k <= 0:
        fail("--top-k must be positive")
    if args.ignore_prefix_steps < 0:
        fail("--ignore-prefix-steps must be non-negative")
    if args.offset < 0:
        fail("--offset must be non-negative")

    server = ensure_file(Path(args.llama_server), "llama-server")
    proxy_model = ensure_file(Path(args.proxy_model), "proxy model")
    oracle_model = ensure_file(Path(args.oracle_model), "oracle model")
    prompt_path = Path(args.prompts).resolve() if args.prompts else None
    if prompt_path is not None:
        ensure_file(prompt_path, "prompts file")
    prompts = load_prompts(prompt_path, args.offset, args.limit)
    if not prompts:
        fail("no prompts to evaluate")

    port = args.port if args.port else find_free_port(args.host)
    if args.dry_run:
        print("proxy server command:")
        print(
            json.dumps(
                server_args(
                    server=server,
                    model=proxy_model,
                    n_gpu_layers=args.proxy_ngl,
                    ctx_size=args.ctx_size,
                    threads=args.threads,
                    host=args.host,
                    port=port,
                    cache_type=args.cache_type,
                ),
                indent=2,
            )
        )
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
                ),
                indent=2,
            )
        )
        return 0

    proxy = run_model_probes(
        server=server,
        model=proxy_model,
        n_gpu_layers=args.proxy_ngl,
        prompts=prompts,
        tokens=args.tokens,
        top_k=args.top_k,
        mode=args.mode,
        ctx_size=args.ctx_size,
        threads=args.threads,
        cache_type=args.cache_type,
        host=args.host,
        port=port,
        log_dir=Path(args.log_dir),
        label="proxy",
        ready_timeout=args.ready_timeout,
        request_timeout=args.request_timeout,
    )
    # Reuse the same port after the proxy server has stopped.
    oracle = run_model_probes(
        server=server,
        model=oracle_model,
        n_gpu_layers=args.oracle_ngl,
        prompts=prompts,
        tokens=args.tokens,
        top_k=args.top_k,
        mode=args.mode,
        ctx_size=args.ctx_size,
        threads=args.threads,
        cache_type=args.cache_type,
        host=args.host,
        port=port,
        log_dir=Path(args.log_dir),
        label="oracle",
        ready_timeout=args.ready_timeout,
        request_timeout=args.request_timeout,
    )
    rows = make_rows(prompts, proxy, oracle)
    out_path = write_output(rows, args)
    print_summary(rows, out_path, args.ignore_prefix_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
