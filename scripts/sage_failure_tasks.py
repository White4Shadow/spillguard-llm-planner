#!/usr/bin/env python3
"""
Extract SAGE replay failures into sparse-probe task JSON.

Use this after a frozen proxy gate has been evaluated. The output keeps the
original candidate task identity so later sparse captures can still join
against the replay JSON with sage_candidate_probe_fit.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"
VALID_CLASSES = {"any", "whitespace", "control", "punct", "number", "capitalized", "word"}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


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


def parse_classes(raw: str) -> set[str]:
    classes = {part.strip() for part in raw.split(",") if part.strip()}
    if not classes:
        return {"any"}
    invalid = sorted(classes - VALID_CLASSES)
    if invalid:
        fail(f"invalid token class: {invalid[0]}")
    return classes


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if isinstance(row, dict) and isinstance(row.get("task"), dict):
                item = dict(row)
                item["_replay_source"] = str(path)
                rows.append(item)
    return rows


def gate_accepts(task: dict[str, Any], max_entropy: float, min_margin: float) -> bool:
    entropy = float(task.get("proxy_entropy", 0.0))
    margin = float(task.get("proxy_margin", 0.0))
    return entropy <= max_entropy or margin >= min_margin


def row_matches_kind(row: dict[str, Any], kind: str, max_entropy: float, min_margin: float, label: str) -> bool:
    task = row["task"]
    accepted = gate_accepts(task, max_entropy, min_margin)
    replay_match = bool(row.get(f"replay_top1_{label}_match", False))
    if kind == "false-accepts":
        return accepted and not replay_match
    if kind == "true-accepts":
        return accepted and replay_match
    if kind == "all-mismatches":
        return not replay_match
    if kind == "accepted":
        return accepted
    if kind == "all":
        return True
    fail(f"unsupported kind: {kind}")
    return False


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    classes = parse_classes(args.token_classes)
    out: list[dict[str, Any]] = []
    per_class: Counter[str] = Counter()
    for row in rows:
        task = row["task"]
        cls = str(task.get("token_class", ""))
        if "any" not in classes and cls not in classes:
            continue
        if args.label_quality != "any" and str(task.get("label_quality", "")) != args.label_quality:
            continue
        if row_matches_kind(row, args.kind, args.max_entropy, args.min_margin, args.label):
            if args.max_per_class > 0 and per_class[cls] >= args.max_per_class:
                continue
            out.append(row)
            per_class[cls] += 1
            if args.max_tasks > 0 and len(out) >= args.max_tasks:
                break
    return out


def task_from_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    task = dict(row["task"])
    replay_match = bool(row.get("replay_top1_token_match", False))
    task["failure_id"] = f"fail-{index:04d}"
    task["failure_replay_source"] = str(row.get("_replay_source", ""))
    task["failure_kind"] = "replay-match" if replay_match else "replay-false-accept"
    task["replay_oracle_token_id"] = int(row.get("oracle_token_id", -1))
    task["replay_oracle_token"] = str(row.get("oracle_token", ""))
    task["replay_oracle_margin"] = float(row.get("oracle_margin", 0.0))
    task["replay_oracle_entropy"] = float(row.get("oracle_entropy", 0.0))
    task["replay_top1_token_match"] = bool(row.get("replay_top1_token_match", False))
    task["replay_top1_id_match"] = bool(row.get("replay_top1_id_match", False))
    task["original_top1_token_match"] = bool(row.get("original_top1_token_match", False))
    task["proxy_token_in_oracle_topk"] = bool(row.get("proxy_token_in_oracle_topk", False))
    return task


def summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    false_accepts = [
        row
        for row in rows
        if gate_accepts(row["task"], args.max_entropy, args.min_margin)
        and not bool(row.get(f"replay_top1_{args.label}_match", False))
    ]
    by_class = Counter(str(row["task"].get("token_class", "")) for row in selected)
    by_quality = Counter(str(row["task"].get("label_quality", "")) for row in selected)
    selected_matches = sum(1 for row in selected if bool(row.get(f"replay_top1_{args.label}_match", False)))
    return {
        "replay_rows": len(rows),
        "selected_tasks": len(selected),
        "selected_replay_matches": selected_matches,
        "selected_replay_mismatches": len(selected) - selected_matches,
        "selected_by_class": dict(by_class),
        "selected_by_label_quality": dict(by_quality),
        "gate_false_accepts_all_classes": len(false_accepts),
        "max_entropy": args.max_entropy,
        "min_margin": args.min_margin,
        "max_tasks": args.max_tasks,
        "max_per_class": args.max_per_class,
    }


def write_output(rows: list[dict[str, Any]], selected: list[dict[str, Any]], paths: list[Path], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"{args.kind}-e{args.max_entropy:g}-m{args.min_margin:g}"
    out_path = out_dir / f"{stamp}-sage-failure-tasks-{tag}.json"
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": {
            "replay_json": [str(path) for path in paths],
            "kind": args.kind,
            "label": args.label,
            "max_entropy": args.max_entropy,
            "min_margin": args.min_margin,
            "token_classes": sorted(parse_classes(args.token_classes)),
            "label_quality": args.label_quality,
            "max_tasks": args.max_tasks,
            "max_per_class": args.max_per_class,
            "render_mode": args.render_mode,
            "gemma4_thought_prefix": args.gemma4_thought_prefix,
        },
        "summary": summarize(rows, selected, args),
        "tasks": [task_from_row(row, index) for index, row in enumerate(selected, start=1)],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_summary(selected: list[dict[str, Any]], out_path: Path) -> None:
    print("# SAGE Failure Tasks")
    print()
    print(f"wrote: {out_path.resolve()}")
    print()
    print("| Failure | Task | Class | Quality | Margin | Entropy | Proxy | Replay oracle | Prompt |")
    print("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |")
    for index, row in enumerate(selected[:25], start=1):
        task = row["task"]
        prompt = str(task.get("prompt", "")).replace("|", "\\|")
        proxy_token = str(task.get("proxy_token", "")).replace("|", "\\|")
        oracle_token = str(row.get("oracle_token", "")).replace("|", "\\|")
        print(
            f"| fail-{index:04d} | {task.get('task_id', '')} | {task.get('token_class', '')} "
            f"| {task.get('label_quality', '')} | {float(task.get('proxy_margin', 0.0)):.3f} "
            f"| {float(task.get('proxy_entropy', 0.0)):.3f} | `{proxy_token}` | `{oracle_token}` | {prompt} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen-gate replay failures into SAGE sparse-probe tasks.")
    parser.add_argument("--replay-json", nargs="+", required=True)
    parser.add_argument("--max-entropy", type=float, default=0.6)
    parser.add_argument("--min-margin", type=float, default=1.5)
    parser.add_argument("--token-classes", default="any")
    parser.add_argument("--label-quality", choices=["any", "same-prefix", "diverged-prefix"], default="any")
    parser.add_argument("--kind", choices=["false-accepts", "true-accepts", "all-mismatches", "accepted", "all"], default="false-accepts")
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--max-tasks", type=int, default=0, help="maximum selected tasks; 0 keeps all")
    parser.add_argument("--max-per-class", type=int, default=0, help="maximum selected tasks per token class; 0 keeps all")
    parser.add_argument("--render-mode", choices=["gemma4-chat"], default="gemma4-chat")
    parser.add_argument("--gemma4-thought-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.max_tasks < 0:
        parser.error("--max-tasks must be non-negative")
    if args.max_per_class < 0:
        parser.error("--max-per-class must be non-negative")

    paths = expand_inputs(args.replay_json)
    rows = load_rows(paths)
    if not rows:
        fail("no replay rows found")
    selected = selected_rows(rows, args)
    out_path = write_output(rows, selected, paths, args)
    print_summary(selected, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
