#!/usr/bin/env python3
"""
Build router-candidate sparse-probe tasks from SAGE logprob probes.

The normal logprob probe records full proxy and oracle continuations. For later
steps those continuations may already have diverged, so this task builder marks
whether the prefix before a candidate step still matched. Candidate captures can
then choose strict same-prefix labels or broader exploratory labels explicitly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"


@dataclass
class CandidateTask:
    task_id: str
    source: str
    prompt_index: int
    prompt: str
    step_index: int
    append_text: str
    proxy_token_id: int
    proxy_token: str
    oracle_token_id: int
    oracle_token: str
    proxy_margin: float
    proxy_entropy: float
    topk_token_jaccard: float
    top1_token_match: bool
    top1_id_match: bool
    token_class: str
    prefix_token_match_before_step: bool
    prefix_id_match_before_step: bool
    label_quality: str


def expand_inputs(values: list[str]) -> list[Path]:
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


def token_class(token: str) -> str:
    stripped = token.strip()
    if not stripped:
        return "whitespace"
    if re.fullmatch(r"<\|.*\|>", stripped):
        return "control"
    if re.fullmatch(r"[\W_]+", stripped):
        return "punct"
    if stripped.isdigit():
        return "number"
    if stripped[:1].isupper():
        return "capitalized"
    return "word"


def step_token(steps: list[dict[str, Any]], index: int) -> str:
    if 0 <= index < len(steps):
        return str(steps[index].get("token", ""))
    return ""


def step_token_id(steps: list[dict[str, Any]], index: int) -> int:
    if 0 <= index < len(steps):
        return int(steps[index].get("token_id", steps[index].get("id", -1)))
    return -1


def candidate_matches(step: dict[str, Any], cls: str, args: argparse.Namespace) -> bool:
    classes = candidate_classes(args)
    if "any" not in classes and cls not in classes:
        return False
    if float(step.get("proxy_margin", 0.0)) < args.margin_threshold:
        return False
    if args.max_entropy >= 0 and float(step.get("proxy_entropy", 0.0)) > args.max_entropy:
        return False
    return True


def candidate_classes(args: argparse.Namespace) -> set[str]:
    raw = args.candidate_classes or args.candidate_class
    classes = {part.strip() for part in raw.split(",") if part.strip()}
    valid = {"any", "whitespace", "control", "punct", "number", "capitalized", "word"}
    invalid = sorted(classes - valid)
    if invalid:
        raise ValueError(f"invalid candidate class: {invalid[0]}")
    return classes or {"punct"}


def build_tasks(paths: list[Path], args: argparse.Namespace) -> list[CandidateTask]:
    tasks: list[CandidateTask] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for prompt_index, row in enumerate(payload.get("rows", []), start=1):
            prompt = str(row.get("prompt", ""))
            proxy_steps = row.get("proxy", {}).get("steps", [])
            oracle_steps = row.get("oracle", {}).get("steps", [])
            prefix_token_ok = True
            prefix_id_ok = True

            for step in row.get("steps", []):
                index = int(step.get("index", 0))
                if index < args.ignore_prefix_steps:
                    continue

                proxy_token = step_token(proxy_steps, index)
                oracle_token = step_token(oracle_steps, index)
                proxy_token_id = step_token_id(proxy_steps, index)
                oracle_token_id = step_token_id(oracle_steps, index)
                cls = token_class(proxy_token)
                top1_token_match = bool(step.get("top1_token_match", False))
                top1_id_match = bool(step.get("top1_id_match", False))

                if candidate_matches(step, cls, args):
                    if not args.require_matching_prefix or prefix_token_ok:
                        append_text = "".join(step_token(proxy_steps, prior) for prior in range(args.ignore_prefix_steps, index))
                        label_quality = "same-prefix" if prefix_token_ok else "diverged-prefix"
                        task_number = len(tasks) + 1
                        tasks.append(
                            CandidateTask(
                                task_id=f"cand-{task_number:04d}",
                                source=str(path),
                                prompt_index=prompt_index,
                                prompt=prompt,
                                step_index=index,
                                append_text=append_text,
                                proxy_token_id=proxy_token_id,
                                proxy_token=proxy_token,
                                oracle_token_id=oracle_token_id,
                                oracle_token=oracle_token,
                                proxy_margin=float(step.get("proxy_margin", 0.0)),
                                proxy_entropy=float(step.get("proxy_entropy", 0.0)),
                                topk_token_jaccard=float(step.get("topk_token_jaccard", 0.0)),
                                top1_token_match=top1_token_match,
                                top1_id_match=top1_id_match,
                                token_class=cls,
                                prefix_token_match_before_step=prefix_token_ok,
                                prefix_id_match_before_step=prefix_id_ok,
                                label_quality=label_quality,
                            )
                        )

                prefix_token_ok = prefix_token_ok and top1_token_match
                prefix_id_ok = prefix_id_ok and top1_id_match
    return tasks


def summarize(tasks: list[CandidateTask]) -> dict[str, Any]:
    same_prefix = [task for task in tasks if task.prefix_token_match_before_step]
    return {
        "tasks": len(tasks),
        "top1_token_matches": sum(1 for task in tasks if task.top1_token_match),
        "top1_token_match_rate": sum(1 for task in tasks if task.top1_token_match) / len(tasks) if tasks else 0.0,
        "same_prefix_tasks": len(same_prefix),
        "same_prefix_top1_token_matches": sum(1 for task in same_prefix if task.top1_token_match),
        "same_prefix_top1_token_match_rate": (
            sum(1 for task in same_prefix if task.top1_token_match) / len(same_prefix) if same_prefix else 0.0
        ),
        "mean_proxy_margin": sum(task.proxy_margin for task in tasks) / len(tasks) if tasks else 0.0,
        "mean_proxy_entropy": sum(task.proxy_entropy for task in tasks) / len(tasks) if tasks else 0.0,
    }


def write_output(tasks: list[CandidateTask], paths: list[Path], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"{args.candidate_class}-margin-{args.margin_threshold:g}"
    out_path = out_dir / f"{stamp}-sage-candidate-tasks-{tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {
            "sources": [str(path) for path in paths],
            "ignore_prefix_steps": args.ignore_prefix_steps,
            "candidate_class": args.candidate_class,
            "candidate_classes": sorted(candidate_classes(args)),
            "margin_threshold": args.margin_threshold,
            "max_entropy": args.max_entropy,
            "require_matching_prefix": args.require_matching_prefix,
            "render_mode": args.render_mode,
            "gemma4_thought_prefix": args.gemma4_thought_prefix,
        },
        "summary": summarize(tasks),
        "tasks": [asdict(task) for task in tasks],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(tasks: list[CandidateTask], out_path: Path) -> None:
    summary = summarize(tasks)
    print("# SAGE Candidate Tasks")
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
    print("| Task | Step | Prefix | Match | Margin | Token | Prompt |")
    print("| --- | ---: | --- | --- | ---: | --- | --- |")
    for task in tasks[:20]:
        prompt = task.prompt.replace("|", "\\|")
        token = task.proxy_token.replace("|", "\\|")
        print(
            f"| {task.task_id} | {task.step_index} | {task.label_quality} | {task.top1_token_match} "
            f"| {task.proxy_margin:.3f} | `{token}` | {prompt} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SAGE sparse-probe tasks from router candidate steps.")
    parser.add_argument("--logprob-json", nargs="+", required=True)
    parser.add_argument("--ignore-prefix-steps", type=int, default=3)
    parser.add_argument("--candidate-class", choices=["any", "whitespace", "control", "punct", "number", "capitalized", "word"], default="punct")
    parser.add_argument("--candidate-classes", default="", help="comma-separated candidate classes; overrides --candidate-class")
    parser.add_argument("--margin-threshold", type=float, default=0.644)
    parser.add_argument("--max-entropy", type=float, default=-1.0, help="negative disables entropy filtering")
    parser.add_argument("--require-matching-prefix", action="store_true")
    parser.add_argument("--render-mode", choices=["gemma4-chat"], default="gemma4-chat")
    parser.add_argument("--gemma4-thought-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.ignore_prefix_steps < 0:
        parser.error("--ignore-prefix-steps must be non-negative")
    try:
        candidate_classes(args)
    except ValueError as exc:
        parser.error(str(exc))
    paths = expand_inputs(args.logprob_json)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error(f"file not found: {missing[0]}")
    tasks = build_tasks(paths, args)
    out_path = write_output(tasks, paths, args)
    print_summary(tasks, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
