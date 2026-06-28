#!/usr/bin/env python3
"""
Live multi-token SAGE loop prototype.

This is the first non-replay driver for SAGE-100. It asks a proxy model for
the next token, applies the frozen proxy gate, optionally calls the patched
llama.cpp one-shot verifier, and falls back to an oracle model when the token
is rejected.

The default managed mode is intentionally memory-safe for one-consumer-GPU
machines: it keeps the proxy server alive, but if the oracle is a different
model it stops the proxy server before launching the oracle server on the same
port. That is slow, but it proves live control flow without requiring both
models to fit at once. For serious runs, pass --proxy-base-url and
--oracle-base-url to use externally managed endpoints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_candidate_tasks import token_class
from sage_logprob_probe import (
    DEFAULT_ORACLE,
    DEFAULT_PROXY,
    DEFAULT_SERVER,
    ManagedServer,
    find_free_port,
    http_json,
    parse_step,
    safe_name,
)
from sage_runtime_capture import DEFAULT_CUDA_BIN, DEFAULT_LLAMA_COMPLETION, build_env


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"
DEFAULT_LIVE_LOG_DIR = DEFAULT_OUT_DIR / "sage-live-logs"
DEFAULT_VERIFIER_LOG_DIR = DEFAULT_OUT_DIR / "sage-live-verifier-logs"


@dataclass
class LiveModelStep:
    token_id: int
    token: str
    logprob: float
    top1_margin: float
    topk_entropy: float
    top_logprobs: list[dict[str, Any]]
    elapsed_sec: float
    timings: dict[str, Any]


@dataclass
class LiveDecision:
    step_index: int
    prefix_before: str
    proxy: LiveModelStep
    token_class: str
    candidate_eligible: bool
    proxy_gate_accept: bool
    verifier_needed: bool
    verifier_decision: dict[str, Any] | None
    oracle: LiveModelStep | None
    action: str
    reason: str
    selected_token: str
    selected_token_id: int
    elapsed_sec: float


@dataclass
class LiveSummary:
    steps: int
    candidate_eligible: int
    proxy_gate_accepts: int
    verifier_needed: int
    verifier_calls: int
    verifier_accepts: int
    verifier_rejects: int
    proxy_accepts: int
    oracle_fallbacks: int
    final_text: str
    elapsed_sec: float
    tokens_per_sec: float


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def parse_classes(raw: str) -> set[str]:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or {"punct", "capitalized"}


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
    return unique


def load_replay_seed(paths: list[Path], prompt_index: int, replay_offset: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    params: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            fail(f"seed replay JSON not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not params and isinstance(payload.get("params"), dict):
            params = dict(payload["params"])
        for row in payload.get("rows", []):
            if not isinstance(row, dict) or not isinstance(row.get("task"), dict):
                continue
            task = row["task"]
            if int(task.get("prompt_index", -1)) == prompt_index:
                rows.append(dict(row))
    rows.sort(key=lambda row: (int(row["task"].get("step_index", 0)), str(row["task"].get("task_id", ""))))
    if replay_offset < 0:
        fail("--seed-replay-offset must be non-negative")
    rows = rows[replay_offset:]
    if not rows:
        fail(f"no seed replay rows for prompt_index={prompt_index}")
    first_task = rows[0]["task"]
    prompt = str(first_task.get("prompt", ""))
    if not prompt:
        fail("seed replay row has no prompt")
    prompt_rows = [row for row in rows if str(row["task"].get("prompt", "")) == prompt]
    skipped_rows = [row for row in rows if str(row["task"].get("prompt", "")) != prompt]
    return {
        "prompt": prompt,
        "rows": prompt_rows,
        "skipped_prompt_index_collision_rows": skipped_rows,
        "params": params,
        "prompt_index": prompt_index,
        "replay_offset": replay_offset,
    }


def apply_replay_seed(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.seed_replay_json:
        return None
    seed = load_replay_seed(expand_paths(args.seed_replay_json), args.seed_prompt_index, args.seed_replay_offset)
    args.prompt = str(seed["prompt"])
    if args.tokens_from_replay:
        args.tokens = len(seed["rows"])
    if args.seed_apply_params:
        params = seed["params"]
        if "mode" in params:
            args.mode = str(params["mode"])
        if "chat_enable_thinking" in params:
            args.chat_enable_thinking = bool(params["chat_enable_thinking"])
        if "gemma4_thought_prefix" in params:
            args.gemma4_thought_prefix = bool(params["gemma4_thought_prefix"])
        if "cache_prompt" in params:
            args.cache_prompt = bool(params["cache_prompt"])
        if "top_k" in params:
            args.top_k = int(params["top_k"])
    return seed


def proxy_gate_accepts(step: LiveModelStep, max_entropy: float, min_margin: float) -> bool:
    return step.topk_entropy <= max_entropy or step.top1_margin >= min_margin


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


def step_to_live(raw_step: Any, elapsed_sec: float, timings: dict[str, Any]) -> LiveModelStep:
    parsed = parse_step(raw_step)
    return LiveModelStep(
        token_id=parsed.token_id,
        token=parsed.token,
        logprob=parsed.logprob,
        top1_margin=parsed.top1_margin,
        topk_entropy=parsed.topk_entropy,
        top_logprobs=[asdict(item) for item in parsed.top_logprobs],
        elapsed_sec=elapsed_sec,
        timings=timings,
    )


def completion_next(
    *,
    base_url: str,
    rendered_prompt: str,
    top_k: int,
    request_timeout: int,
    cache_prompt: bool,
    completion_special: bool | None,
) -> LiveModelStep:
    payload: dict[str, Any] = {
        "prompt": rendered_prompt,
        "n_predict": 1,
        "temperature": 0,
        "top_k": 1,
        "n_probs": top_k,
        "stream": False,
        "cache_prompt": cache_prompt,
    }
    if completion_special is not None:
        payload["special"] = completion_special
    start = time.perf_counter()
    response = http_json("POST", f"{base_url}/completion", payload, timeout=request_timeout)
    elapsed = time.perf_counter() - start
    steps_raw = response.get("completion_probabilities", [])
    if not steps_raw:
        fail("completion endpoint returned no completion_probabilities")
    return step_to_live(steps_raw[0], elapsed, response.get("timings", {}))


class EndpointManager:
    def __init__(self, args: argparse.Namespace, proxy_model: Path, oracle_model: Path, port: int) -> None:
        self.args = args
        self.proxy_model = proxy_model
        self.oracle_model = oracle_model
        self.port = port
        self.proxy_server: ManagedServer | None = None
        self.oracle_server: ManagedServer | None = None

    def _start_server(self, model: Path, ngl: str, label: str) -> ManagedServer:
        server = ManagedServer(
            server=Path(self.args.llama_server).resolve(),
            model=model,
            n_gpu_layers=ngl,
            ctx_size=self.args.ctx_size,
            threads=self.args.threads,
            host=self.args.host,
            port=self.port,
            cache_type=self.args.cache_type,
            log_dir=Path(self.args.log_dir),
            label=label,
        )
        print(f"live: starting {label} server on {server.base_url}", flush=True)
        print(f"live: server logs {server.stdout_log} | {server.stderr_log}", flush=True)
        server.start(self.args.ready_timeout)
        return server

    def proxy_base_url(self) -> str:
        if self.args.proxy_base_url:
            return str(self.args.proxy_base_url).rstrip("/")
        if self.proxy_server is None:
            self.proxy_server = self._start_server(self.proxy_model, self.args.proxy_ngl, "proxy")
        return self.proxy_server.base_url

    def oracle_base_url(self) -> str:
        if self.args.oracle_base_url:
            return str(self.args.oracle_base_url).rstrip("/")
        if self.proxy_model.resolve() == self.oracle_model.resolve() and self.args.proxy_base_url:
            return str(self.args.proxy_base_url).rstrip("/")
        if self.proxy_model.resolve() == self.oracle_model.resolve():
            return self.proxy_base_url()
        if self.proxy_server is not None:
            self.proxy_server.stop()
            self.proxy_server = None
        self.oracle_server = self._start_server(self.oracle_model, self.args.oracle_ngl, "oracle")
        return self.oracle_server.base_url

    def finish_oracle_call(self) -> None:
        if self.oracle_server is not None:
            self.oracle_server.stop()
            self.oracle_server = None

    def pause_proxy_for_child_process(self) -> None:
        if self.args.proxy_base_url:
            return
        if self.proxy_server is not None:
            self.proxy_server.stop()
            self.proxy_server = None

    def stop(self) -> None:
        if self.proxy_server is not None:
            self.proxy_server.stop()
            self.proxy_server = None
        if self.oracle_server is not None:
            self.oracle_server.stop()
            self.oracle_server = None


def run_verifier_decision(
    *,
    args: argparse.Namespace,
    verifier_model: Path,
    prompt: str,
    prefix: str,
    proxy_step: LiveModelStep,
    cls: str,
    step_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    task_id = f"live-{step_index:04d}"
    task = {
        "task_id": task_id,
        "prompt_index": 1,
        "prompt": prompt,
        "step_index": step_index,
        "append_text": prefix,
        "proxy_token_id": proxy_step.token_id,
        "proxy_token": proxy_step.token,
        "proxy_margin": proxy_step.top1_margin if math.isfinite(proxy_step.top1_margin) else 999.0,
        "proxy_entropy": proxy_step.topk_entropy,
        "token_class": cls,
        "label_quality": "live",
    }
    tasks_path = run_dir / f"{task_id}-task.json"
    stats_path = run_dir / f"{task_id}-sage-runtime.jsonl"
    decision_path = run_dir / f"{task_id}-sage-decision.jsonl"
    tasks_path.write_text(json.dumps({"tasks": [task]}, indent=2), encoding="utf-8")
    command = [
        str(Path(args.llama_completion).resolve()),
        "-m",
        str(verifier_model),
        "-p",
        render_prompt(prompt, args.mode, args.chat_enable_thinking, args.gemma4_thought_prefix, prefix),
        "-n",
        "1",
        "-ngl",
        str(args.verifier_ngl),
        "-c",
        str(args.verifier_ctx_size),
        "-b",
        str(args.verifier_batch_size),
        "-ub",
        str(args.verifier_ubatch_size),
        "--sage-tensor-filter",
        args.verifier_tensor_filter,
        "--sage-tensor-stats-output",
        str(stats_path),
        "--sage-decision-output",
        str(decision_path),
        "--sage-candidate-id",
        task_id,
        "--sage-token-class",
        cls,
        "--sage-proxy-entropy",
        str(task["proxy_entropy"]),
        "--sage-proxy-margin",
        str(task["proxy_margin"]),
    ]
    if args.no_conversation:
        command.append("-no-cnv")
    if args.no_warmup:
        command.append("--no-warmup")
    if args.verifier_threads > 0:
        command.extend(["-t", str(args.verifier_threads), "-tb", str(args.verifier_threads)])

    log_path = run_dir / f"{task_id}-runtime-capture.log"
    cuda_bin = Path(args.cuda_bin) if args.cuda_bin else None
    env = build_env(cuda_bin)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = proc.wait(timeout=args.verifier_timeout + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait(timeout=30)
            elapsed = time.perf_counter() - start
            return {
                "action": "oracle_fallback",
                "reason": "verifier_process_timeout",
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "log_path": str(log_path.resolve()),
                "verifier_covered": False,
            }
    elapsed = time.perf_counter() - start
    if returncode != 0:
        return {
            "action": "oracle_fallback",
            "reason": "verifier_process_failed",
            "returncode": returncode,
            "elapsed_sec": elapsed,
            "log_path": str(log_path.resolve()),
            "verifier_covered": False,
        }
    decisions = []
    if decision_path.is_file():
        with decision_path.open("r", encoding="utf-8") as handle:
            decisions = [json.loads(line) for line in handle if line.strip()]
    decision = dict(decisions[-1]) if decisions else {}
    if not decision:
        decision = {
            "action": "oracle_fallback",
            "reason": "missing_verifier_decision",
            "verifier_covered": False,
        }
    decision["elapsed_sec"] = elapsed
    decision["task_json"] = str(tasks_path.resolve())
    decision["stats_jsonl"] = str(stats_path.resolve())
    decision["decision_jsonl"] = str(decision_path.resolve())
    decision["capture_log_path"] = str(log_path.resolve())
    return decision


def query_oracle(
    *,
    args: argparse.Namespace,
    endpoints: EndpointManager,
    prompt: str,
    prefix: str,
) -> LiveModelStep:
    rendered = render_prompt(prompt, args.mode, args.chat_enable_thinking, args.gemma4_thought_prefix, prefix)
    try:
        return completion_next(
            base_url=endpoints.oracle_base_url(),
            rendered_prompt=rendered,
            top_k=args.top_k,
            request_timeout=args.request_timeout,
            cache_prompt=args.cache_prompt,
            completion_special=args.completion_special,
        )
    finally:
        endpoints.finish_oracle_call()


def run_live(args: argparse.Namespace) -> tuple[LiveSummary, list[LiveDecision], Path]:
    proxy_model = ensure_file(Path(args.proxy_model), "proxy model")
    oracle_model = ensure_file(Path(args.oracle_model), "oracle model")
    verifier_model = ensure_file(Path(args.verifier_model), "verifier model")
    ensure_file(Path(args.llama_server), "llama-server")
    ensure_file(Path(args.llama_completion), "llama-completion")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = safe_name(args.tag or "live-loop")
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.verifier_log_dir) / f"{timestamp}-{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = args.port if args.port else find_free_port(args.host)
    endpoints = EndpointManager(args, proxy_model, oracle_model, port)
    candidate_classes = parse_classes(args.candidate_token_classes)
    verifier_classes = parse_classes(args.verifier_token_classes)

    prefix = ""
    decisions: list[LiveDecision] = []
    started = time.perf_counter()
    try:
        for step_index in range(args.tokens):
            step_started = time.perf_counter()
            rendered = render_prompt(args.prompt, args.mode, args.chat_enable_thinking, args.gemma4_thought_prefix, prefix)
            proxy_step = completion_next(
                base_url=endpoints.proxy_base_url(),
                rendered_prompt=rendered,
                top_k=args.top_k,
                request_timeout=args.request_timeout,
                cache_prompt=args.cache_prompt,
                completion_special=args.completion_special,
            )
            cls = token_class(proxy_step.token)
            candidate_eligible = cls in candidate_classes
            proxy_accept = candidate_eligible and proxy_gate_accepts(proxy_step, args.proxy_max_entropy, args.proxy_min_margin)
            verifier_needed = proxy_accept and cls in verifier_classes
            verifier_decision: dict[str, Any] | None = None
            oracle_step: LiveModelStep | None = None
            action = "accept_proxy"
            reason = "proxy_gate_non_verifier_class"

            if not candidate_eligible:
                action = "oracle_fallback"
                reason = "non_candidate_class"
            elif not proxy_accept:
                action = "oracle_fallback"
                reason = "proxy_gate_reject"
            elif verifier_needed:
                if args.pause_proxy_for_verifier:
                    endpoints.pause_proxy_for_child_process()
                verifier_decision = run_verifier_decision(
                    args=args,
                    verifier_model=verifier_model,
                    prompt=args.prompt,
                    prefix=prefix,
                    proxy_step=proxy_step,
                    cls=cls,
                    step_index=step_index,
                    run_dir=run_dir,
                )
                action = str(verifier_decision.get("action", "oracle_fallback"))
                reason = str(verifier_decision.get("reason", "runtime_verifier_decision"))

            if action == "oracle_fallback":
                oracle_step = query_oracle(args=args, endpoints=endpoints, prompt=args.prompt, prefix=prefix)
                selected_token = oracle_step.token
                selected_token_id = oracle_step.token_id
            else:
                selected_token = proxy_step.token
                selected_token_id = proxy_step.token_id

            decisions.append(
                LiveDecision(
                    step_index=step_index,
                    prefix_before=prefix,
                    proxy=proxy_step,
                    token_class=cls,
                    candidate_eligible=candidate_eligible,
                    proxy_gate_accept=proxy_accept,
                    verifier_needed=verifier_needed,
                    verifier_decision=verifier_decision,
                    oracle=oracle_step,
                    action=action,
                    reason=reason,
                    selected_token=selected_token,
                    selected_token_id=selected_token_id,
                    elapsed_sec=time.perf_counter() - step_started,
                )
            )
            prefix += selected_token
            print(
                f"live: step {step_index + 1}/{args.tokens} {action} "
                f"class={cls} token={selected_token!r}",
                flush=True,
            )
    finally:
        endpoints.stop()

    elapsed = time.perf_counter() - started
    summary = summarize(decisions, prefix, elapsed)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"{timestamp}-sage-live-loop-{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "seed": args._seed_info if hasattr(args, "_seed_info") else None,
        "summary": asdict(summary),
        "decisions": [asdict(item) for item in decisions],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary, decisions, out_path


def summarize(decisions: list[LiveDecision], final_text: str, elapsed_sec: float) -> LiveSummary:
    steps = len(decisions)
    verifier_calls = sum(1 for item in decisions if item.verifier_decision is not None)
    verifier_accepts = sum(1 for item in decisions if item.verifier_decision and item.verifier_decision.get("action") == "accept_proxy")
    verifier_rejects = sum(1 for item in decisions if item.verifier_decision and item.verifier_decision.get("action") == "oracle_fallback")
    proxy_accepts = sum(1 for item in decisions if item.action == "accept_proxy")
    oracle_fallbacks = sum(1 for item in decisions if item.action == "oracle_fallback")
    return LiveSummary(
        steps=steps,
        proxy_gate_accepts=sum(1 for item in decisions if item.proxy_gate_accept),
        candidate_eligible=sum(1 for item in decisions if item.candidate_eligible),
        verifier_needed=sum(1 for item in decisions if item.verifier_needed),
        verifier_calls=verifier_calls,
        verifier_accepts=verifier_accepts,
        verifier_rejects=verifier_rejects,
        proxy_accepts=proxy_accepts,
        oracle_fallbacks=oracle_fallbacks,
        final_text=final_text,
        elapsed_sec=elapsed_sec,
        tokens_per_sec=steps / elapsed_sec if elapsed_sec > 0 else 0.0,
    )


def mock_step(token: str, token_id: int, margin: float, entropy: float) -> LiveModelStep:
    return LiveModelStep(
        token_id=token_id,
        token=token,
        logprob=-0.1,
        top1_margin=margin,
        topk_entropy=entropy,
        top_logprobs=[
            {"id": token_id, "token": token, "logprob": -0.1},
            {"id": token_id + 1, "token": " alt", "logprob": -0.1 - margin},
        ],
        elapsed_sec=0.001,
        timings={},
    )


def self_test(out_dir: Path) -> int:
    started = time.perf_counter()
    proxy_steps = [
        mock_step(" Paris", 1, 2.0, 0.02),
        mock_step(".", 2, 1.8, 0.40),
        mock_step(" 1", 3, 1.7, 0.20),
    ]
    oracle_steps = [mock_step("!", 20, 1.1, 0.50)]
    verifier_actions = [
        {"action": "accept_proxy", "reason": "runtime_verifier_accept", "verifier_covered": True},
        {"action": "oracle_fallback", "reason": "runtime_verifier_reject", "verifier_covered": True},
    ]
    prefix = ""
    decisions: list[LiveDecision] = []
    verifier_index = 0
    oracle_index = 0
    candidate_classes = {"punct", "capitalized", "number"}
    verifier_classes = {"punct", "capitalized"}
    for idx, proxy_step in enumerate(proxy_steps):
        cls = token_class(proxy_step.token)
        candidate_eligible = cls in candidate_classes
        proxy_accept = candidate_eligible and proxy_gate_accepts(proxy_step, 0.6, 1.5)
        verifier_needed = proxy_accept and cls in verifier_classes
        verifier_decision: dict[str, Any] | None = None
        action = "accept_proxy"
        reason = "proxy_gate_non_verifier_class"
        oracle_step = None
        if not candidate_eligible:
            action = "oracle_fallback"
            reason = "non_candidate_class"
        elif not proxy_accept:
            action = "oracle_fallback"
            reason = "proxy_gate_reject"
        elif verifier_needed:
            verifier_decision = verifier_actions[verifier_index]
            verifier_index += 1
            action = str(verifier_decision["action"])
            reason = str(verifier_decision["reason"])
        if action == "oracle_fallback":
            oracle_step = oracle_steps[oracle_index]
            oracle_index += 1
            selected = oracle_step.token
            selected_id = oracle_step.token_id
        else:
            selected = proxy_step.token
            selected_id = proxy_step.token_id
        decisions.append(
            LiveDecision(
                step_index=idx,
                prefix_before=prefix,
                proxy=proxy_step,
                token_class=cls,
                candidate_eligible=candidate_eligible,
                proxy_gate_accept=proxy_accept,
                verifier_needed=verifier_needed,
                verifier_decision=verifier_decision,
                oracle=oracle_step,
                action=action,
                reason=reason,
                selected_token=selected,
                selected_token_id=selected_id,
                elapsed_sec=0.001,
            )
        )
        prefix += selected

    summary = summarize(decisions, prefix, time.perf_counter() - started)
    if summary.final_text != " Paris! 1":
        fail(f"self-test final_text mismatch: {summary.final_text!r}", code=1)
    if summary.proxy_accepts != 2 or summary.oracle_fallbacks != 1:
        fail("self-test summary mismatch", code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sage-live-loop-self-test.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {"self_test": True},
        "summary": asdict(summary),
        "decisions": [asdict(item) for item in decisions],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SAGE live loop self-test passed")
    print(f"  final text: {summary.final_text!r}")
    print(f"  wrote: {out_path.resolve()}")
    return 0


def print_summary(summary: LiveSummary, out_path: Path) -> None:
    print()
    print("SAGE live loop summary")
    print(f"  steps: {summary.steps}")
    print(f"  proxy accepts: {summary.proxy_accepts}")
    print(f"  oracle fallbacks: {summary.oracle_fallbacks}")
    print(f"  verifier calls: {summary.verifier_calls}")
    print(f"  elapsed sec: {summary.elapsed_sec:.3f}")
    print(f"  observed tok/s: {summary.tokens_per_sec:.3f}")
    print(f"  final text: {summary.final_text!r}")
    print(f"  wrote: {out_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live SAGE proxy/verifier/oracle token loop.")
    parser.add_argument("--llama-server", default=str(DEFAULT_SERVER))
    parser.add_argument("--llama-completion", default=str(DEFAULT_LLAMA_COMPLETION))
    parser.add_argument("--proxy-model", default=str(DEFAULT_PROXY))
    parser.add_argument("--oracle-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--verifier-model", default=str(DEFAULT_ORACLE))
    parser.add_argument("--seed-replay-json", nargs="+", default=[], help="candidate replay JSON(s) used to seed prompt and token count")
    parser.add_argument("--seed-prompt-index", type=int, default=1)
    parser.add_argument("--seed-replay-offset", type=int, default=0)
    parser.add_argument("--tokens-from-replay", action="store_true", help="set --tokens to the number of seeded replay rows")
    parser.add_argument("--seed-apply-params", action=argparse.BooleanOptionalAction, default=True, help="copy mode/top-k/cache prompt settings from seed replay params")
    parser.add_argument("--print-seed-only", action="store_true", help="print resolved replay seed and exit")
    parser.add_argument("--proxy-base-url", default="", help="external proxy llama-server base URL")
    parser.add_argument("--oracle-base-url", default="", help="external oracle llama-server base URL")
    parser.add_argument("--proxy-ngl", default="all")
    parser.add_argument("--oracle-ngl", default="38")
    parser.add_argument("--verifier-ngl", default="38")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=["raw", "gemma4-chat"], default="raw")
    parser.add_argument("--chat-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gemma4-thought-prefix", action="store_true")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache-type", default="q8_0")
    parser.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--completion-special", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--proxy-max-entropy", type=float, default=0.6)
    parser.add_argument("--proxy-min-margin", type=float, default=1.5)
    parser.add_argument("--candidate-token-classes", default="punct,whitespace,number,capitalized")
    parser.add_argument("--verifier-token-classes", default="punct,capitalized")
    parser.add_argument("--verifier-tensor-filter", default="ffn_norm-0")
    parser.add_argument("--verifier-ctx-size", type=int, default=256)
    parser.add_argument("--verifier-batch-size", type=int, default=128)
    parser.add_argument("--verifier-ubatch-size", type=int, default=8)
    parser.add_argument("--verifier-threads", type=int, default=0)
    parser.add_argument("--verifier-timeout", type=int, default=1800)
    parser.add_argument("--pause-proxy-for-verifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-conversation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--log-dir", default=str(DEFAULT_LIVE_LOG_DIR))
    parser.add_argument("--verifier-log-dir", default=str(DEFAULT_VERIFIER_LOG_DIR))
    parser.add_argument("--cuda-bin", default=str(DEFAULT_CUDA_BIN))
    parser.add_argument("--tag", default="live-loop")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test(Path(args.out_dir))
    seed_info = apply_replay_seed(args)
    if seed_info is not None:
        args._seed_info = {
            "prompt": seed_info["prompt"],
            "prompt_index": seed_info["prompt_index"],
            "replay_offset": seed_info["replay_offset"],
            "rows": len(seed_info["rows"]),
            "skipped_prompt_index_collision_rows": len(seed_info["skipped_prompt_index_collision_rows"]),
            "task_ids": [str(row["task"].get("task_id", "")) for row in seed_info["rows"][: args.tokens]],
            "step_indices": [int(row["task"].get("step_index", 0)) for row in seed_info["rows"][: args.tokens]],
        }
    if args.print_seed_only:
        if seed_info is None:
            parser.error("--print-seed-only requires --seed-replay-json")
        print(json.dumps(args._seed_info, indent=2))
        return 0
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    summary, _decisions, out_path = run_live(args)
    print_summary(summary, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
