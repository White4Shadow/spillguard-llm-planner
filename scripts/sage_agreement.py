#!/usr/bin/env python3
"""
Proxy/oracle agreement harness for the SAGE-100 research track.

The first SAGE gate is simple: can a smaller resident proxy model predict the
same next token as a larger oracle often enough that the oracle could be skipped
on many tokens? This script measures a coarse version of that using
llama-completion greedy generation. For instruction-tuned GGUFs, it can also
measure single-turn llama-cli chat output with --mode chat.

It intentionally uses normal llama.cpp binaries instead of Python model bindings
so it works with the same local GGUF files used by the rest of this repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-completion.exe"
DEFAULT_CHAT_CLI = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-cli.exe"
DEFAULT_PROXY = ROOT / "models" / "gemma-4-12b-it-qat-q4_0-gguf" / "gemma-4-12b-it-qat-q4_0.gguf"
DEFAULT_ORACLE = ROOT / "models" / "gemma-4-31b-it-qat-q4_0-gguf" / "gemma-4-31B_q4_0-it.gguf"
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LOG_DIR = DEFAULT_OUT_DIR / "sage-agreement-logs"


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


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class GenerationResult:
    model: str
    n_gpu_layers: str
    returncode: int
    elapsed_sec: float
    output: str
    stderr_tail: str
    stdout_log: str = ""
    stderr_log: str = ""
    llama_log: str = ""


@dataclass
class AgreementRow:
    prompt: str
    proxy: GenerationResult
    oracle: GenerationResult
    exact_match: bool
    normalized_match: bool


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def clean_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r\n", "\n").strip()


def extract_generation(text: str, prompt: str, mode: str) -> str:
    text = ANSI_RE.sub("", text).replace("\r\n", "\n")
    if mode != "chat":
        return text.strip()

    marker = f"> {prompt}"
    marker_idx = text.rfind(marker)
    if marker_idx >= 0:
        text = text[marker_idx + len(marker) :]

    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    generated: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[ Prompt:") or stripped == "Exiting..." or stripped == ">":
            break
        generated.append(line)
    return "\n".join(generated).strip()


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tail(text: str, max_chars: int = 2000) -> str:
    text = clean_output(text)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def file_tail(path: Path, max_bytes: int = 20_000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def safe_log_name(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug[:80] or "run"


def load_prompts(path: Path | None, limit: int) -> list[str]:
    if path is None:
        prompts = DEFAULT_PROMPTS
    else:
        raw = path.read_text(encoding="utf-8").splitlines()
        prompts = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def llama_command(
    *,
    cli: Path,
    model: Path,
    prompt: str,
    n_gpu_layers: str,
    tokens: int,
    ctx_size: int,
    threads: int,
    cache_type: str,
    mode: str,
    log_file: Path | None = None,
) -> list[str]:
    argv = [
        str(cli),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(tokens),
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
        "--temp",
        "0",
        "--top-k",
        "1",
        "--seed",
        "1",
    ]
    if mode == "chat":
        argv.extend(["--single-turn", "--no-display-prompt", "--simple-io", "--no-warmup", "--no-perf"])
    else:
        argv.extend(["--no-conversation", "--no-display-prompt", "--simple-io", "--no-warmup", "--no-perf"])
    if log_file is not None:
        argv.extend(["--log-file", str(log_file)])
    return argv


def run_generation(
    *,
    cli: Path,
    model: Path,
    prompt: str,
    n_gpu_layers: str,
    tokens: int,
    ctx_size: int,
    threads: int,
    cache_type: str,
    mode: str,
    timeout: int,
    progress_label: str,
    progress_interval: int,
    log_dir: Path,
) -> GenerationResult:
    argv = llama_command(
        cli=cli,
        model=model,
        prompt=prompt,
        n_gpu_layers=n_gpu_layers,
        tokens=tokens,
        ctx_size=ctx_size,
        threads=threads,
        cache_type=cache_type,
        mode=mode,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_base = f"{stamp}-{safe_log_name(progress_label)}"
    stdout_log = log_dir / f"{log_base}.out.log"
    stderr_log = log_dir / f"{log_base}.err.log"
    llama_log = log_dir / f"{log_base}.llama.log"
    argv.extend(["--log-file", str(llama_log)])

    print(f"agreement: starting {progress_label}", flush=True)
    print(f"agreement: child logs {stdout_log} | {stderr_log} | {llama_log}", flush=True)
    start = time.perf_counter()
    timed_out = False
    with stdout_log.open("wb") as stdout_file, stderr_log.open("wb") as stderr_file:
        proc = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        next_progress = max(1, progress_interval)
        while proc.poll() is None:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                timed_out = True
                proc.kill()
                proc.wait()
                break
            if elapsed >= next_progress:
                print(f"agreement: still running {progress_label} after {elapsed:.0f}s", flush=True)
                next_progress += max(1, progress_interval)
            time.sleep(1)

    elapsed = time.perf_counter() - start
    stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
    stderr = file_tail(stderr_log)
    if timed_out:
        return GenerationResult(
            model=str(model),
            n_gpu_layers=n_gpu_layers,
            returncode=-1,
            elapsed_sec=elapsed,
            output=extract_generation(stdout, prompt, mode),
            stderr_tail=f"timeout after {timeout}s\n{tail(stderr)}",
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            llama_log=str(llama_log),
        )

    print(f"agreement: finished {progress_label} in {elapsed:.1f}s rc={proc.returncode}", flush=True)
    return GenerationResult(
        model=str(model),
        n_gpu_layers=n_gpu_layers,
        returncode=proc.returncode if proc.returncode is not None else -1,
        elapsed_sec=elapsed,
        output=extract_generation(stdout, prompt, mode),
        stderr_tail=tail(stderr),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        llama_log=str(llama_log),
    )


def evaluate(
    *,
    cli: Path,
    proxy_model: Path,
    oracle_model: Path,
    prompts: Iterable[str],
    proxy_ngl: str,
    oracle_ngl: str,
    tokens: int,
    ctx_size: int,
    threads: int,
    cache_type: str,
    mode: str,
    timeout: int,
    progress_interval: int,
    log_dir: Path,
) -> list[AgreementRow]:
    rows: list[AgreementRow] = []
    for idx, prompt in enumerate(prompts, start=1):
        print(f"agreement: prompt {idx}: {prompt[:80]}", flush=True)
        proxy = run_generation(
            cli=cli,
            model=proxy_model,
            prompt=prompt,
            n_gpu_layers=proxy_ngl,
            tokens=tokens,
            ctx_size=ctx_size,
            threads=threads,
            cache_type=cache_type,
            mode=mode,
            timeout=timeout,
            progress_label=f"proxy prompt {idx}",
            progress_interval=progress_interval,
            log_dir=log_dir,
        )
        oracle = run_generation(
            cli=cli,
            model=oracle_model,
            prompt=prompt,
            n_gpu_layers=oracle_ngl,
            tokens=tokens,
            ctx_size=ctx_size,
            threads=threads,
            cache_type=cache_type,
            mode=mode,
            timeout=timeout,
            progress_label=f"oracle prompt {idx}",
            progress_interval=progress_interval,
            log_dir=log_dir,
        )
        rows.append(
            AgreementRow(
                prompt=prompt,
                proxy=proxy,
                oracle=oracle,
                exact_match=proxy.output == oracle.output,
                normalized_match=normalize_for_match(proxy.output) == normalize_for_match(oracle.output),
            )
        )
    return rows


def write_outputs(rows: list[AgreementRow], out_dir: Path, tag: str, args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{stamp}-sage-agreement-{tag}.json"
    exact = sum(1 for row in rows if row.exact_match)
    normalized = sum(1 for row in rows if row.normalized_match)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {
            "proxy_model": str(Path(args.proxy_model).resolve()),
            "oracle_model": str(Path(args.oracle_model).resolve()),
            "proxy_ngl": args.proxy_ngl,
            "oracle_ngl": args.oracle_ngl,
            "tokens": args.tokens,
            "ctx_size": args.ctx_size,
            "threads": args.threads,
            "cache_type": args.cache_type,
            "mode": args.mode,
            "log_dir": str(Path(args.log_dir).resolve()),
        },
        "summary": {
            "prompts": len(rows),
            "exact_matches": exact,
            "exact_match_rate": exact / len(rows) if rows else 0.0,
            "normalized_matches": normalized,
            "normalized_match_rate": normalized / len(rows) if rows else 0.0,
            "proxy_total_sec": sum(row.proxy.elapsed_sec for row in rows),
            "oracle_total_sec": sum(row.oracle.elapsed_sec for row in rows),
        },
        "rows": [asdict(row) for row in rows],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(rows: list[AgreementRow], out_path: Path) -> None:
    exact = sum(1 for row in rows if row.exact_match)
    normalized = sum(1 for row in rows if row.normalized_match)
    total = len(rows)
    print()
    print("SAGE agreement summary")
    print(f"  prompts: {total}")
    print(f"  exact matches: {exact}/{total} ({exact / total:.1%})" if total else "  exact matches: n/a")
    print(
        f"  normalized matches: {normalized}/{total} ({normalized / total:.1%})"
        if total
        else "  normalized matches: n/a"
    )
    print(f"  wrote: {out_path}")
    print()
    for row in rows:
        mark = "==" if row.normalized_match else "!="
        print(f"{mark} {row.prompt}")
        print(f"  proxy : {row.proxy.output!r}")
        print(f"  oracle: {row.oracle.output!r}")


def self_test(out_dir: Path) -> int:
    proxy_ok = GenerationResult(
        model="mock-proxy",
        n_gpu_layers="all",
        returncode=0,
        elapsed_sec=0.01,
        output=" Paris",
        stderr_tail="",
    )
    oracle_ok = GenerationResult(
        model="mock-oracle",
        n_gpu_layers="all",
        returncode=0,
        elapsed_sec=0.02,
        output=" Paris",
        stderr_tail="",
    )
    proxy_miss = GenerationResult(
        model="mock-proxy",
        n_gpu_layers="all",
        returncode=0,
        elapsed_sec=0.01,
        output=" fast",
        stderr_tail="",
    )
    oracle_miss = GenerationResult(
        model="mock-oracle",
        n_gpu_layers="all",
        returncode=0,
        elapsed_sec=0.02,
        output=" slowly",
        stderr_tail="",
    )
    rows = [
        AgreementRow(
            prompt="The capital of France is",
            proxy=proxy_ok,
            oracle=oracle_ok,
            exact_match=True,
            normalized_match=True,
        ),
        AgreementRow(
            prompt="A 100B model on a 12GB GPU must run",
            proxy=proxy_miss,
            oracle=oracle_miss,
            exact_match=False,
            normalized_match=False,
        ),
    ]
    args = argparse.Namespace(
        proxy_model="mock-proxy",
        oracle_model="mock-oracle",
        proxy_ngl="all",
        oracle_ngl="all",
        tokens=1,
        ctx_size=512,
        threads=1,
        cache_type="q8_0",
        mode="raw",
        log_dir=str(DEFAULT_LOG_DIR),
    )
    out_path = write_outputs(rows, out_dir, "self-test", args)
    print_summary(rows, out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure greedy next-token agreement between a proxy and oracle GGUF model.")
    parser.add_argument("--llama-cli", default=str(DEFAULT_CLI), help="path to llama-completion.exe for --mode raw")
    parser.add_argument("--llama-chat-cli", default=str(DEFAULT_CHAT_CLI), help="path to llama-cli.exe for --mode chat")
    parser.add_argument("--mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--proxy-model", default=str(DEFAULT_PROXY))
    parser.add_argument("--oracle-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--proxy-ngl", default="all")
    parser.add_argument("--oracle-ngl", default="38")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache-type", default="q8_0")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tag", default="gemma12b-to-31b")
    parser.add_argument("--dry-run", action="store_true", help="print commands that would run, but do not load models")
    parser.add_argument("--self-test", action="store_true", help="run a deterministic mock agreement test")
    args = parser.parse_args()

    if args.self_test:
        return self_test(Path(args.out_dir))

    cli = ensure_file(
        Path(args.llama_chat_cli if args.mode == "chat" else args.llama_cli),
        "llama-cli" if args.mode == "chat" else "llama-completion",
    )
    proxy_model = ensure_file(Path(args.proxy_model), "proxy model")
    oracle_model = ensure_file(Path(args.oracle_model), "oracle model")
    prompt_path = Path(args.prompts).resolve() if args.prompts else None
    if prompt_path is not None:
        ensure_file(prompt_path, "prompts file")
    if args.tokens <= 0:
        fail("--tokens must be positive")
    if args.progress_interval <= 0:
        fail("--progress-interval must be positive")

    prompts = load_prompts(prompt_path, args.limit)
    if not prompts:
        fail("no prompts to evaluate")

    if args.dry_run:
        first_prompt = prompts[0]
        proxy_cmd = llama_command(
            cli=cli,
            model=proxy_model,
            prompt=first_prompt,
            n_gpu_layers=args.proxy_ngl,
            tokens=args.tokens,
            ctx_size=args.ctx_size,
            threads=args.threads,
            cache_type=args.cache_type,
            mode=args.mode,
        )
        oracle_cmd = llama_command(
            cli=cli,
            model=oracle_model,
            prompt=first_prompt,
            n_gpu_layers=args.oracle_ngl,
            tokens=args.tokens,
            ctx_size=args.ctx_size,
            threads=args.threads,
            cache_type=args.cache_type,
            mode=args.mode,
        )
        print("proxy command:")
        print(json.dumps(proxy_cmd, indent=2))
        print("oracle command:")
        print(json.dumps(oracle_cmd, indent=2))
        return 0

    rows = evaluate(
        cli=cli,
        proxy_model=proxy_model,
        oracle_model=oracle_model,
        prompts=prompts,
        proxy_ngl=args.proxy_ngl,
        oracle_ngl=args.oracle_ngl,
        tokens=args.tokens,
        ctx_size=args.ctx_size,
        threads=args.threads,
        cache_type=args.cache_type,
        mode=args.mode,
        timeout=args.timeout,
        progress_interval=args.progress_interval,
        log_dir=Path(args.log_dir),
    )
    out_path = write_outputs(rows, Path(args.out_dir), args.tag, args)
    print_summary(rows, out_path)

    if any(row.proxy.returncode != 0 or row.oracle.returncode != 0 for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
