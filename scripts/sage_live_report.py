#!/usr/bin/env python3
"""
Summarize SAGE live-loop gate artifacts.

The live gate currently proves correctness before speed: it may restart
llama-server or launch llama-completion subprocesses during a token step.
This report separates model request time from orchestration overhead so the
next production target is explicit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"


@dataclass
class StepTiming:
    live_json: str
    step_index: int
    action: str
    reason: str
    token_class: str
    selected_token: str
    candidate_eligible: bool
    proxy_gate_accept: bool
    verifier_needed: bool
    verifier_covered: bool | None
    proxy_sec: float
    verifier_sec: float
    oracle_sec: float
    accounted_sec: float
    step_sec: float
    overhead_sec: float


@dataclass
class RunSummary:
    live_json: str
    compare_json: str
    prompt: str
    seed_prompt_index: int
    seed_rows: int
    skipped_prompt_index_collision_rows: int
    steps: int
    candidate_eligible: int
    proxy_accepts: int
    oracle_fallbacks: int
    verifier_calls: int
    verifier_rejects: int
    final_text: str
    elapsed_sec: float
    observed_tps: float
    proxy_sec: float
    verifier_sec: float
    oracle_sec: float
    accounted_sec: float
    overhead_sec: float
    overhead_rate: float
    compare_steps: int
    action_matches: int
    selected_token_matches: int
    verifier_coverage_matches: int
    compare_pass: bool | None


@dataclass
class ReportSummary:
    runs: int
    steps: int
    proxy_accepts: int
    oracle_fallbacks: int
    verifier_calls: int
    elapsed_sec: float
    observed_tps: float
    proxy_sec: float
    verifier_sec: float
    oracle_sec: float
    accounted_sec: float
    overhead_sec: float
    overhead_rate: float
    compare_steps: int
    action_matches: int
    selected_token_matches: int
    verifier_coverage_matches: int
    all_compares_passed: bool


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail(f"expected JSON object: {path}")
    return payload


def maybe_compare_path(live_path: Path) -> Path:
    name = live_path.name
    if "live-validation" in name:
        return live_path.with_name(name.replace("live-validation", "live-compare-validation", 1))
    return live_path.with_suffix(".compare.json")


def elapsed_from_step(step: dict[str, Any], key: str) -> float:
    value = step.get(key)
    if isinstance(value, dict):
        return float(value.get("elapsed_sec", 0.0))
    return 0.0


def verifier_elapsed(step: dict[str, Any]) -> float:
    decision = step.get("verifier_decision")
    if isinstance(decision, dict):
        return float(decision.get("elapsed_sec", 0.0))
    return 0.0


def verifier_covered(step: dict[str, Any]) -> bool | None:
    decision = step.get("verifier_decision")
    if isinstance(decision, dict):
        return bool(decision.get("verifier_covered", False))
    return None


def selected_token(step: dict[str, Any]) -> str:
    return str(step.get("selected_token", ""))


def summarize_live(live_path: Path, compare_path: Path | None) -> tuple[RunSummary, list[StepTiming]]:
    payload = load_json(live_path)
    summary = payload.get("summary", {})
    params = payload.get("params", {})
    seed = payload.get("seed", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(params, dict):
        params = {}
    if not isinstance(seed, dict):
        seed = {}
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        fail(f"live JSON has no decisions list: {live_path}")

    steps: list[StepTiming] = []
    for raw in decisions:
        if not isinstance(raw, dict):
            continue
        proxy_sec = elapsed_from_step(raw, "proxy")
        verifier_sec = verifier_elapsed(raw)
        oracle_sec = elapsed_from_step(raw, "oracle")
        step_sec = float(raw.get("elapsed_sec", 0.0))
        accounted = proxy_sec + verifier_sec + oracle_sec
        steps.append(
            StepTiming(
                live_json=str(live_path.resolve()),
                step_index=int(raw.get("step_index", 0)),
                action=str(raw.get("action", "")),
                reason=str(raw.get("reason", "")),
                token_class=str(raw.get("token_class", "")),
                selected_token=selected_token(raw),
                candidate_eligible=bool(raw.get("candidate_eligible", False)),
                proxy_gate_accept=bool(raw.get("proxy_gate_accept", False)),
                verifier_needed=bool(raw.get("verifier_needed", False)),
                verifier_covered=verifier_covered(raw),
                proxy_sec=proxy_sec,
                verifier_sec=verifier_sec,
                oracle_sec=oracle_sec,
                accounted_sec=accounted,
                step_sec=step_sec,
                overhead_sec=max(0.0, step_sec - accounted),
            )
        )

    compare_summary: dict[str, Any] = {}
    compare_path_str = ""
    if compare_path is not None and compare_path.is_file():
        compare_payload = load_json(compare_path)
        raw_summary = compare_payload.get("summary", {})
        compare_summary = dict(raw_summary) if isinstance(raw_summary, dict) else {}
        compare_path_str = str(compare_path.resolve())

    proxy_sec = sum(item.proxy_sec for item in steps)
    verifier_sec = sum(item.verifier_sec for item in steps)
    oracle_sec = sum(item.oracle_sec for item in steps)
    accounted_sec = proxy_sec + verifier_sec + oracle_sec
    elapsed_sec = float(summary.get("elapsed_sec", sum(item.step_sec for item in steps)))
    overhead_sec = max(0.0, elapsed_sec - accounted_sec)
    run = RunSummary(
        live_json=str(live_path.resolve()),
        compare_json=compare_path_str,
        prompt=str(params.get("prompt", "")),
        seed_prompt_index=int(seed.get("prompt_index", params.get("seed_prompt_index", 0) or 0)),
        seed_rows=int(seed.get("rows", 0) or 0),
        skipped_prompt_index_collision_rows=int(seed.get("skipped_prompt_index_collision_rows", 0) or 0),
        steps=int(summary.get("steps", len(steps))),
        candidate_eligible=int(summary.get("candidate_eligible", sum(1 for item in steps if item.candidate_eligible))),
        proxy_accepts=int(summary.get("proxy_accepts", sum(1 for item in steps if item.action == "accept_proxy"))),
        oracle_fallbacks=int(summary.get("oracle_fallbacks", sum(1 for item in steps if item.action == "oracle_fallback"))),
        verifier_calls=int(summary.get("verifier_calls", sum(1 for item in steps if item.verifier_covered is not None))),
        verifier_rejects=int(summary.get("verifier_rejects", sum(1 for item in steps if item.reason == "runtime_verifier_reject"))),
        final_text=str(summary.get("final_text", "")),
        elapsed_sec=elapsed_sec,
        observed_tps=float(summary.get("tokens_per_sec", (len(steps) / elapsed_sec if elapsed_sec > 0 else 0.0))),
        proxy_sec=proxy_sec,
        verifier_sec=verifier_sec,
        oracle_sec=oracle_sec,
        accounted_sec=accounted_sec,
        overhead_sec=overhead_sec,
        overhead_rate=overhead_sec / elapsed_sec if elapsed_sec > 0 else 0.0,
        compare_steps=int(compare_summary.get("compared_steps", 0) or 0),
        action_matches=int(compare_summary.get("action_matches", 0) or 0),
        selected_token_matches=int(compare_summary.get("selected_token_matches", 0) or 0),
        verifier_coverage_matches=int(compare_summary.get("verifier_coverage_matches", 0) or 0),
        compare_pass=bool(compare_summary.get("meets_required_matches")) if compare_summary else None,
    )
    return run, steps


def combine(runs: list[RunSummary]) -> ReportSummary:
    steps = sum(item.steps for item in runs)
    elapsed = sum(item.elapsed_sec for item in runs)
    proxy = sum(item.proxy_sec for item in runs)
    verifier = sum(item.verifier_sec for item in runs)
    oracle = sum(item.oracle_sec for item in runs)
    accounted = proxy + verifier + oracle
    overhead = max(0.0, elapsed - accounted)
    compare_steps = sum(item.compare_steps for item in runs)
    return ReportSummary(
        runs=len(runs),
        steps=steps,
        proxy_accepts=sum(item.proxy_accepts for item in runs),
        oracle_fallbacks=sum(item.oracle_fallbacks for item in runs),
        verifier_calls=sum(item.verifier_calls for item in runs),
        elapsed_sec=elapsed,
        observed_tps=steps / elapsed if elapsed > 0 else 0.0,
        proxy_sec=proxy,
        verifier_sec=verifier,
        oracle_sec=oracle,
        accounted_sec=accounted,
        overhead_sec=overhead,
        overhead_rate=overhead / elapsed if elapsed > 0 else 0.0,
        compare_steps=compare_steps,
        action_matches=sum(item.action_matches for item in runs),
        selected_token_matches=sum(item.selected_token_matches for item in runs),
        verifier_coverage_matches=sum(item.verifier_coverage_matches for item in runs),
        all_compares_passed=all(item.compare_pass is True for item in runs if item.compare_pass is not None),
    )


def print_report(summary: ReportSummary, runs: list[RunSummary], steps: list[StepTiming], top_steps: int) -> None:
    print("# SAGE Live Gate Report")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key, value in asdict(summary).items():
        if isinstance(value, bool):
            print(f"| {key} | {value} |")
        elif isinstance(value, float):
            print(f"| {key} | {value:.6f} |")
        else:
            print(f"| {key} | {value} |")
    print()
    print("| Run | Steps | Pass | t/s | Proxy s | Verifier s | Oracle s | Overhead s |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        print(
            f"| {Path(run.live_json).name} | {run.steps} | {run.compare_pass} | "
            f"{run.observed_tps:.4f} | {run.proxy_sec:.3f} | {run.verifier_sec:.3f} | "
            f"{run.oracle_sec:.3f} | {run.overhead_sec:.3f} |"
        )
    if top_steps > 0:
        print()
        print("| Run | Step | Action | Token | Proxy s | Verifier s | Oracle s | Overhead s |")
        print("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for item in steps[:top_steps]:
            token = item.selected_token.replace("|", "\\|")
            print(
                f"| {Path(item.live_json).name} | {item.step_index} | {item.action} | `{token}` | "
                f"{item.proxy_sec:.3f} | {item.verifier_sec:.3f} | {item.oracle_sec:.3f} | {item.overhead_sec:.3f} |"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SAGE live-loop gate timings and replay matches.")
    parser.add_argument("--live-json", nargs="+", required=True)
    parser.add_argument("--compare-json", nargs="*", default=[], help="optional compare JSONs aligned with --live-json")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--tag", default="live-report")
    parser.add_argument("--top-steps", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    live_paths = expand_paths(args.live_json)
    if not live_paths:
        fail("no live JSON files found")
    compare_paths = expand_paths(args.compare_json) if args.compare_json else []
    runs: list[RunSummary] = []
    all_steps: list[StepTiming] = []
    for index, live_path in enumerate(live_paths):
        compare_path = compare_paths[index] if index < len(compare_paths) else maybe_compare_path(live_path)
        run, steps = summarize_live(live_path, compare_path)
        runs.append(run)
        all_steps.extend(steps)
    summary = combine(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-sage-live-report-{args.tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "summary": asdict(summary),
        "runs": [asdict(item) for item in runs],
        "steps": [asdict(item) for item in all_steps],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(summary, runs, all_steps, args.top_steps)
        print()
        print(f"wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
