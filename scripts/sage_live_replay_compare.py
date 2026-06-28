#!/usr/bin/env python3
"""
Compare a SAGE live-loop trace against a scheduler replay trace.

The live loop proves that proxy generation, verifier decisions, and oracle
fallback can happen on newly generated tokens. This comparator checks whether
that live behavior still matches a replay trace for the same prompt region.
It is intentionally strict about action and token drift, because any mismatch
means the live prefix may diverge from the validated replay accounting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "benchmarks"


@dataclass
class CompareRow:
    live_index: int
    replay_index: int
    replay_task_id: str
    replay_prompt_index: int
    replay_step_index: int
    live_token_class: str
    replay_token_class: str
    live_action: str
    replay_action: str
    live_reason: str
    replay_reason: str
    live_proxy_token: str
    replay_proxy_token: str
    live_selected_token: str
    replay_selected_token: str
    action_match: bool
    reason_match: bool
    token_class_match: bool
    proxy_token_match: bool
    selected_token_match: bool
    verifier_coverage_match: bool
    live_verifier_covered: bool | None
    replay_verifier_covered: bool | None


@dataclass
class CompareSummary:
    live_steps: int
    replay_steps_available: int
    replay_selection: str
    replay_prompt_index: int
    replay_offset: int
    compared_steps: int
    action_matches: int
    reason_matches: int
    token_class_matches: int
    proxy_token_matches: int
    selected_token_matches: int
    verifier_coverage_matches: int
    first_mismatch_index: int | None
    first_mismatch_field: str
    prefix_aligned_steps: int
    require_action_match: bool
    require_selected_token_match: bool
    require_proxy_token_match: bool
    require_verifier_coverage_match: bool
    meets_required_matches: bool


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        fail(f"expected JSON object: {path}")
    return payload


def replay_decisions_for_prompt(payload: dict[str, Any], prompt_index: int) -> list[dict[str, Any]]:
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        fail("replay JSON has no decisions list")
    rows = [
        dict(item)
        for item in decisions
        if isinstance(item, dict) and int(item.get("prompt_index", -1)) == prompt_index
    ]
    rows.sort(key=lambda row: (int(row.get("step_index", 0)), str(row.get("task_id", ""))))
    return rows


def replay_decisions_for_task_ids(payload: dict[str, Any], task_ids: list[str]) -> list[dict[str, Any]]:
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        fail("replay JSON has no decisions list")
    by_id = {str(item.get("task_id", "")): dict(item) for item in decisions if isinstance(item, dict)}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        fail(f"replay JSON is missing seeded task_id: {missing[0]}")
    return [by_id[task_id] for task_id in task_ids]


def live_decisions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        fail("live JSON has no decisions list")
    return [dict(item) for item in decisions if isinstance(item, dict)]


def live_seed_task_ids(payload: dict[str, Any]) -> list[str]:
    seed = payload.get("seed")
    if not isinstance(seed, dict):
        return []
    task_ids = seed.get("task_ids", [])
    if not isinstance(task_ids, list):
        return []
    return [str(item) for item in task_ids if str(item)]


def live_proxy_token(row: dict[str, Any]) -> str:
    proxy = row.get("proxy", {})
    if isinstance(proxy, dict):
        return str(proxy.get("token", ""))
    return ""


def live_verifier_covered(row: dict[str, Any]) -> bool | None:
    decision = row.get("verifier_decision")
    if not isinstance(decision, dict):
        return None
    return bool(decision.get("verifier_covered", False))


def replay_verifier_covered(row: dict[str, Any]) -> bool | None:
    if not bool(row.get("verifier_needed", False)):
        return None
    return bool(row.get("verifier_covered", False))


def compare_rows(live: dict[str, Any], replay: dict[str, Any], live_index: int, replay_index: int) -> CompareRow:
    live_covered = live_verifier_covered(live)
    replay_covered = replay_verifier_covered(replay)
    verifier_match = live_covered == replay_covered
    if live_covered is None and replay_covered is None:
        verifier_match = True
    live_proxy = live_proxy_token(live)
    replay_proxy = str(replay.get("proxy_token", ""))
    live_selected = str(live.get("selected_token", ""))
    replay_selected = str(replay.get("selected_token", ""))
    live_action = str(live.get("action", ""))
    replay_action = str(replay.get("action", ""))
    live_reason = str(live.get("reason", ""))
    replay_reason = str(replay.get("reason", ""))
    live_cls = str(live.get("token_class", ""))
    replay_cls = str(replay.get("token_class", ""))
    return CompareRow(
        live_index=live_index,
        replay_index=replay_index,
        replay_task_id=str(replay.get("task_id", "")),
        replay_prompt_index=int(replay.get("prompt_index", 0)),
        replay_step_index=int(replay.get("step_index", 0)),
        live_token_class=live_cls,
        replay_token_class=replay_cls,
        live_action=live_action,
        replay_action=replay_action,
        live_reason=live_reason,
        replay_reason=replay_reason,
        live_proxy_token=live_proxy,
        replay_proxy_token=replay_proxy,
        live_selected_token=live_selected,
        replay_selected_token=replay_selected,
        action_match=live_action == replay_action,
        reason_match=live_reason == replay_reason,
        token_class_match=live_cls == replay_cls,
        proxy_token_match=live_proxy == replay_proxy,
        selected_token_match=live_selected == replay_selected,
        verifier_coverage_match=verifier_match,
        live_verifier_covered=live_covered,
        replay_verifier_covered=replay_covered,
    )


def first_mismatch(rows: list[CompareRow], args: argparse.Namespace) -> tuple[int | None, str]:
    fields: list[tuple[str, str]] = []
    if args.require_action_match:
        fields.append(("action", "action_match"))
    if args.require_selected_token_match:
        fields.append(("selected_token", "selected_token_match"))
    if args.require_proxy_token_match:
        fields.append(("proxy_token", "proxy_token_match"))
    if args.require_verifier_coverage_match:
        fields.append(("verifier_coverage", "verifier_coverage_match"))
    for row in rows:
        row_dict = asdict(row)
        for label, attr in fields:
            if not bool(row_dict[attr]):
                return row.live_index, label
    return None, ""


def summarize(
    *,
    rows: list[CompareRow],
    live_count: int,
    replay_count: int,
    replay_selection: str,
    args: argparse.Namespace,
) -> CompareSummary:
    mismatch_index, mismatch_field = first_mismatch(rows, args)
    prefix_aligned = len(rows) if mismatch_index is None else mismatch_index
    action_matches = sum(1 for row in rows if row.action_match)
    selected_matches = sum(1 for row in rows if row.selected_token_match)
    proxy_matches = sum(1 for row in rows if row.proxy_token_match)
    verifier_matches = sum(1 for row in rows if row.verifier_coverage_match)
    meets = True
    if args.require_action_match:
        meets = meets and action_matches == len(rows)
    if args.require_selected_token_match:
        meets = meets and selected_matches == len(rows)
    if args.require_proxy_token_match:
        meets = meets and proxy_matches == len(rows)
    if args.require_verifier_coverage_match:
        meets = meets and verifier_matches == len(rows)
    meets = meets and live_count == len(rows)
    return CompareSummary(
        live_steps=live_count,
        replay_steps_available=replay_count,
        replay_selection=replay_selection,
        replay_prompt_index=args.prompt_index,
        replay_offset=args.replay_offset,
        compared_steps=len(rows),
        action_matches=action_matches,
        reason_matches=sum(1 for row in rows if row.reason_match),
        token_class_matches=sum(1 for row in rows if row.token_class_match),
        proxy_token_matches=proxy_matches,
        selected_token_matches=selected_matches,
        verifier_coverage_matches=verifier_matches,
        first_mismatch_index=mismatch_index,
        first_mismatch_field=mismatch_field,
        prefix_aligned_steps=prefix_aligned,
        require_action_match=args.require_action_match,
        require_selected_token_match=args.require_selected_token_match,
        require_proxy_token_match=args.require_proxy_token_match,
        require_verifier_coverage_match=args.require_verifier_coverage_match,
        meets_required_matches=meets,
    )


def compare(args: argparse.Namespace) -> tuple[CompareSummary, list[CompareRow]]:
    live_payload = load_json(Path(args.live_json))
    replay_payload = load_json(Path(args.replay_json))
    live_rows = live_decisions(live_payload)
    seeded_task_ids = live_seed_task_ids(live_payload) if args.use_live_seed_task_ids else []
    replay_selection = "prompt_index"
    if seeded_task_ids:
        replay_rows_all = replay_decisions_for_task_ids(replay_payload, seeded_task_ids)
        replay_rows = replay_rows_all
        replay_selection = "live_seed_task_ids"
    else:
        replay_rows_all = replay_decisions_for_prompt(replay_payload, args.prompt_index)
        if args.replay_offset < 0:
            fail("--replay-offset must be non-negative")
        replay_rows = replay_rows_all[args.replay_offset:]
    if args.max_steps > 0:
        live_rows = live_rows[: args.max_steps]
        replay_rows = replay_rows[: args.max_steps]
    compared = min(len(live_rows), len(replay_rows))
    if compared == 0:
        fail("no comparable live/replay rows")
    rows = [
        compare_rows(live_rows[index], replay_rows[index], index, (args.replay_offset + index) if replay_selection == "prompt_index" else index)
        for index in range(compared)
    ]
    summary = summarize(rows=rows, live_count=len(live_rows), replay_count=len(replay_rows_all), replay_selection=replay_selection, args=args)
    return summary, rows


def write_payload(summary: CompareSummary, rows: list[CompareRow], args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.json_out:
        out_path = Path(args.json_out)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"{stamp}-sage-live-replay-compare-{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "params": vars(args),
        "summary": asdict(summary),
        "rows": [asdict(row) for row in rows],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def print_report(summary: CompareSummary, rows: list[CompareRow], out_path: Path, top: int) -> None:
    print("# SAGE Live vs Replay Compare")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key, value in asdict(summary).items():
        if isinstance(value, bool):
            print(f"| {key} | {value} |")
        else:
            print(f"| {key} | {value} |")
    if top > 0:
        print()
        print("| Live | Replay task | Action | Selected | Proxy | Verifier |")
        print("| ---: | --- | --- | --- | --- | --- |")
        for row in rows[:top]:
            print(
                f"| {row.live_index} | {row.replay_task_id} | {row.action_match} "
                f"| {row.selected_token_match} | {row.proxy_token_match} "
                f"| {row.verifier_coverage_match} |"
            )
    print()
    print(f"wrote: {out_path.resolve()}")


def self_test(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    live = {
        "decisions": [
            {
                "step_index": 0,
                "token_class": "capitalized",
                "proxy": {"token": " Paris"},
                "action": "accept_proxy",
                "reason": "runtime_verifier_accept",
                "selected_token": " Paris",
                "verifier_decision": {"verifier_covered": True},
            },
            {
                "step_index": 1,
                "token_class": "punct",
                "proxy": {"token": "."},
                "action": "oracle_fallback",
                "reason": "runtime_verifier_reject",
                "selected_token": "!",
                "verifier_decision": {"verifier_covered": True},
            },
        ]
    }
    replay = {
        "decisions": [
            {
                "task_id": "cand-a",
                "prompt_index": 1,
                "step_index": 3,
                "token_class": "capitalized",
                "proxy_token": " Paris",
                "selected_token": " Paris",
                "action": "accept_proxy",
                "reason": "runtime_verifier_accept",
                "verifier_needed": True,
                "verifier_covered": True,
            },
            {
                "task_id": "cand-b",
                "prompt_index": 1,
                "step_index": 4,
                "token_class": "punct",
                "proxy_token": ".",
                "selected_token": "!",
                "action": "oracle_fallback",
                "reason": "runtime_verifier_reject",
                "verifier_needed": True,
                "verifier_covered": True,
            },
        ]
    }
    live_path = out_dir / "sage-live-replay-compare-self-test-live.json"
    replay_path = out_dir / "sage-live-replay-compare-self-test-replay.json"
    live_path.write_text(json.dumps(live, indent=2), encoding="utf-8")
    replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    args = argparse.Namespace(
        live_json=str(live_path),
        replay_json=str(replay_path),
        prompt_index=1,
        replay_offset=0,
        max_steps=0,
        require_action_match=True,
        require_selected_token_match=True,
        require_proxy_token_match=True,
        require_verifier_coverage_match=True,
        use_live_seed_task_ids=True,
        require_pass=True,
        out_dir=str(out_dir),
        json_out=str(out_dir / "sage-live-replay-compare-self-test.json"),
        tag="self-test",
        top=5,
    )
    summary, rows = compare(args)
    if not summary.meets_required_matches:
        fail("self-test expected all required matches", code=1)
    out_path = write_payload(summary, rows, args)
    print("SAGE live/replay compare self-test passed")
    print(f"  compared steps: {summary.compared_steps}")
    print(f"  wrote: {out_path.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SAGE live-loop decisions against a replay trace.")
    parser.add_argument("--live-json", required=False, default="")
    parser.add_argument("--replay-json", required=False, default="")
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--replay-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--require-action-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-selected-token-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-proxy-token-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-verifier-coverage-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-live-seed-task-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--tag", default="live-vs-replay")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test(Path(args.out_dir))
    if not args.live_json:
        parser.error("--live-json is required")
    if not args.replay_json:
        parser.error("--replay-json is required")
    if args.prompt_index < 1:
        parser.error("--prompt-index must be >= 1")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")

    summary, rows = compare(args)
    out_path = write_payload(summary, rows, args)
    if args.json:
        payload = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "params": vars(args),
            "summary": asdict(summary),
            "rows": [asdict(row) for row in rows],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_report(summary, rows, out_path, args.top)
    if args.require_pass and not summary.meets_required_matches:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
