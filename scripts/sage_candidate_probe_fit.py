#!/usr/bin/env python3
"""
Fit sparse verifier rules from candidate-token probe captures.

This is stricter than the fixed-step probe fitter: every capture should carry a
candidate task with the proxy token, oracle label, prefix-quality flag, and the
proxy prefix appended before the probe was captured.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_probe_fit import tensor_features


@dataclass
class CandidateRecord:
    task_id: str
    prompt_index: int
    prompt: str
    step_index: int
    proxy_token: str
    oracle_token: str
    token_class: str
    top1_match: bool
    label_source: str
    label_quality: str
    proxy_margin: float
    proxy_entropy: float
    tensor_count: int
    features: dict[str, float]


@dataclass
class Predicate:
    expression: str
    feature: str
    direction: str
    threshold: float
    mask: tuple[bool, ...]


@dataclass
class RuleResult:
    expression: str
    accepted: int
    rejected: int
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    total: int
    precision: float
    accepted_error_rate: float
    bad_catch_rate: float
    good_reject_rate: float
    skip_rate: float


@dataclass
class SplitRuleResult:
    expression: str
    train: RuleResult
    holdout: RuleResult


def fail(message: str, code: int = 2) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def label_value(task: dict[str, Any], label: str) -> bool:
    key = f"top1_{label}_match"
    return bool(task.get(key, False))


def replay_lookup_key(task: dict[str, Any]) -> str:
    """Stable candidate identity across chunked task files."""
    parts = [
        str(task.get("task_id", "")),
        str(task.get("prompt", "")),
        int(task.get("step_index", 0)),
        str(task.get("append_text", "")),
        str(task.get("proxy_token_id", "")),
        str(task.get("proxy_token", "")),
    ]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def load_replay_labels(paths: list[Path], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    id_counts: dict[str, int] = {}
    key = f"replay_top1_{label}_match"
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            task = row.get("task", {})
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("task_id", ""))
            if not task_id:
                continue
            value = {
                "match": bool(row.get(key, False)),
                "oracle_token": str(row.get("oracle_token", "")),
                "oracle_token_id": int(row.get("oracle_token_id", -1)),
                "proxy_token_in_oracle_topk": bool(row.get("proxy_token_in_oracle_topk", False)),
            }
            out[replay_lookup_key(task)] = value
            by_id[task_id] = value
            id_counts[task_id] = id_counts.get(task_id, 0) + 1
    for task_id, count in id_counts.items():
        if count == 1:
            out[f"id:{task_id}"] = by_id[task_id]
    return out


def label_quality_matches(task: dict[str, Any], label_quality: str) -> bool:
    quality = str(task.get("label_quality", ""))
    if label_quality == "any":
        return True
    if label_quality == "same-prefix":
        return quality == "same-prefix" or bool(task.get("prefix_token_match_before_step", False))
    if label_quality == "diverged-prefix":
        return quality == "diverged-prefix"
    return False


def features_for_capture(capture: dict[str, Any], task: dict[str, Any]) -> dict[str, float]:
    features = tensor_features(capture.get("tensors", []))
    features["proxy.margin"] = float(task.get("proxy_margin", 0.0))
    features["proxy.entropy"] = float(task.get("proxy_entropy", 0.0))
    features["step.index"] = float(task.get("step_index", 0))
    features["capture.tensor_count"] = float(capture.get("tensor_count", 0))
    cls = str(task.get("token_class", ""))
    for value in ("whitespace", "control", "punct", "number", "capitalized", "word"):
        features[f"token_class.{value}"] = 1.0 if cls == value else 0.0
    return features


def load_records(paths: list[Path], label: str, label_quality: str, replay_labels: dict[str, dict[str, Any]] | None = None) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for path in paths:
        payload = load_json(path)
        for capture in payload.get("captures", []):
            task = capture.get("task")
            if not isinstance(task, dict):
                continue
            if not label_quality_matches(task, label_quality):
                continue
            task_id = str(task.get("task_id", f"capture-{len(records) + 1:04d}"))
            replay = None
            if replay_labels is not None:
                replay = replay_labels.get(replay_lookup_key(task))
                if replay is None:
                    replay = replay_labels.get(f"id:{task_id}")
            if replay_labels is not None and replay is None:
                continue
            records.append(
                CandidateRecord(
                    task_id=task_id,
                    prompt_index=int(task.get("prompt_index", 0)),
                    prompt=str(task.get("prompt", capture.get("prompt", ""))),
                    step_index=int(task.get("step_index", 0)),
                    proxy_token=str(task.get("proxy_token", "")),
                    oracle_token=str(replay.get("oracle_token", "")) if replay is not None else str(task.get("oracle_token", "")),
                    token_class=str(task.get("token_class", "")),
                    top1_match=bool(replay["match"]) if replay is not None else label_value(task, label),
                    label_source="oracle-replay" if replay is not None else "continuation",
                    label_quality=str(task.get("label_quality", "")),
                    proxy_margin=float(task.get("proxy_margin", 0.0)),
                    proxy_entropy=float(task.get("proxy_entropy", 0.0)),
                    tensor_count=int(capture.get("tensor_count", 0)),
                    features=features_for_capture(capture, task),
                )
            )
    return records


def finite_values(records: list[CandidateRecord]) -> dict[str, list[float]]:
    keys = sorted({key for record in records for key in record.features})
    out: dict[str, list[float]] = {}
    for key in keys:
        values = sorted({record.features[key] for record in records if key in record.features and math.isfinite(record.features[key])})
        if values:
            out[key] = values
    return out


def make_predicates(records: list[CandidateRecord]) -> list[Predicate]:
    predicates: list[Predicate] = []
    for feature, values in finite_values(records).items():
        for threshold in values:
            for direction in ("<=", ">="):
                if direction == "<=":
                    mask = tuple(record.features.get(feature, math.inf) <= threshold for record in records)
                else:
                    mask = tuple(record.features.get(feature, -math.inf) >= threshold for record in records)
                predicates.append(
                    Predicate(
                        expression=f"{feature} {direction} {threshold:.17g}",
                        feature=feature,
                        direction=direction,
                        threshold=threshold,
                        mask=mask,
                    )
                )
    return predicates


def evaluate_mask(records: list[CandidateRecord], expression: str, mask: tuple[bool, ...]) -> RuleResult:
    accepted_indices = [idx for idx, accepted in enumerate(mask) if accepted]
    rejected_indices = [idx for idx, accepted in enumerate(mask) if not accepted]
    true_accepts = sum(1 for idx in accepted_indices if records[idx].top1_match)
    false_accepts = len(accepted_indices) - true_accepts
    true_rejects = sum(1 for idx in rejected_indices if not records[idx].top1_match)
    false_rejects = len(rejected_indices) - true_rejects
    bad_total = sum(1 for record in records if not record.top1_match)
    good_total = sum(1 for record in records if record.top1_match)
    accepted = len(accepted_indices)
    total = len(records)
    return RuleResult(
        expression=expression,
        accepted=accepted,
        rejected=len(rejected_indices),
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        total=total,
        precision=true_accepts / accepted if accepted else 1.0,
        accepted_error_rate=false_accepts / accepted if accepted else 0.0,
        bad_catch_rate=true_rejects / bad_total if bad_total else 0.0,
        good_reject_rate=false_rejects / good_total if good_total else 0.0,
        skip_rate=accepted / total if total else 0.0,
    )


def expression_mask(records: list[CandidateRecord], expression: str) -> tuple[bool, ...]:
    and_marker = ") AND ("
    or_marker = ") OR ("
    if expression.startswith("(") and expression.endswith(")") and and_marker in expression:
        split_at = expression.find(and_marker)
        left = expression[1:split_at]
        right = expression[split_at + len(and_marker) : -1]
        left_mask = expression_mask(records, left)
        right_mask = expression_mask(records, right)
        return tuple(left and right for left, right in zip(left_mask, right_mask))
    if expression.startswith("(") and expression.endswith(")") and or_marker in expression:
        split_at = expression.find(or_marker)
        left = expression[1:split_at]
        right = expression[split_at + len(or_marker) : -1]
        left_mask = expression_mask(records, left)
        right_mask = expression_mask(records, right)
        return tuple(left or right for left, right in zip(left_mask, right_mask))

    match = re.fullmatch(r"(?P<feature>\S+)\s+(?P<direction><=|>=)\s+(?P<threshold>[-+0-9.eE]+)", expression)
    if match is None:
        fail(f"cannot parse rule expression: {expression}")
    feature = match.group("feature")
    direction = match.group("direction")
    threshold = float(match.group("threshold"))
    if direction == "<=":
        return tuple(record.features.get(feature, math.inf) <= threshold for record in records)
    return tuple(record.features.get(feature, -math.inf) >= threshold for record in records)


def evaluate_expression(records: list[CandidateRecord], expression: str) -> RuleResult:
    return evaluate_mask(records, expression, expression_mask(records, expression))


def rank_key(row: RuleResult) -> tuple[float, float, float, float]:
    return (row.skip_rate, row.bad_catch_rate, -row.good_reject_rate, row.precision)


def fit_rules(records: list[CandidateRecord], max_accepted_error: float, pair_base_top: int, include_pairs: bool) -> list[RuleResult]:
    predicates = make_predicates(records)
    single_results = [evaluate_mask(records, predicate.expression, predicate.mask) for predicate in predicates]
    results = list(single_results)
    if not include_pairs or pair_base_top <= 1:
        return results

    ranked_predicates = [
        item[0]
        for item in sorted(
            zip(predicates, single_results),
            key=lambda item: (
                item[1].accepted_error_rate <= max_accepted_error,
                item[1].skip_rate,
                item[1].bad_catch_rate,
                -item[1].good_reject_rate,
            ),
            reverse=True,
        )[:pair_base_top]
    ]
    seen_masks = {result.expression for result in results}
    for left_index, left in enumerate(ranked_predicates):
        for right in ranked_predicates[left_index + 1 :]:
            and_mask = tuple(a and b for a, b in zip(left.mask, right.mask))
            or_mask = tuple(a or b for a, b in zip(left.mask, right.mask))
            for op, mask in (("AND", and_mask), ("OR", or_mask)):
                expression = f"({left.expression}) {op} ({right.expression})"
                if expression in seen_masks:
                    continue
                seen_masks.add(expression)
                results.append(evaluate_mask(records, expression, mask))
    return results


def summarize(records: list[CandidateRecord]) -> dict[str, Any]:
    matches = sum(1 for record in records if record.top1_match)
    qualities: dict[str, int] = {}
    sources: dict[str, int] = {}
    prompt_indices = sorted({record.prompt_index for record in records})
    for record in records:
        qualities[record.label_quality] = qualities.get(record.label_quality, 0) + 1
        sources[record.label_source] = sources.get(record.label_source, 0) + 1
    return {
        "records": len(records),
        "matches": matches,
        "mismatches": len(records) - matches,
        "match_rate": matches / len(records) if records else 0.0,
        "mean_proxy_margin": sum(record.proxy_margin for record in records) / len(records) if records else 0.0,
        "mean_proxy_entropy": sum(record.proxy_entropy for record in records) / len(records) if records else 0.0,
        "mean_tensor_count": sum(record.tensor_count for record in records) / len(records) if records else 0.0,
        "features": len({key for record in records for key in record.features}),
        "label_quality": qualities,
        "label_source": sources,
        "prompt_count": len(prompt_indices),
        "prompt_indices": prompt_indices,
    }


def split_records(records: list[CandidateRecord], modulus: int, remainder: int) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    holdout = [record for record in records if record.prompt_index % modulus == remainder]
    train = [record for record in records if record.prompt_index % modulus != remainder]
    return train, holdout


def parse_prompt_indices(value: str) -> set[int]:
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def split_records_by_prompt_indices(records: list[CandidateRecord], prompt_indices: set[int]) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    holdout = [record for record in records if record.prompt_index in prompt_indices]
    train = [record for record in records if record.prompt_index not in prompt_indices]
    return train, holdout


def fit_holdout(
    records: list[CandidateRecord],
    *,
    modulus: int = 0,
    remainder: int = 0,
    prompt_indices: set[int] | None = None,
    max_accepted_error: float,
    pair_base_top: int,
    include_pairs: bool,
) -> tuple[list[CandidateRecord], list[CandidateRecord], list[SplitRuleResult]]:
    if prompt_indices:
        train, holdout = split_records_by_prompt_indices(records, prompt_indices)
    else:
        train, holdout = split_records(records, modulus, remainder)
    if not train:
        fail("holdout split leaves no training records")
    if not holdout:
        fail("holdout split leaves no holdout records")

    train_results = fit_rules(train, max_accepted_error, pair_base_top, include_pairs)
    train_ranked = sorted(
        [row for row in train_results if row.accepted_error_rate <= max_accepted_error],
        key=rank_key,
        reverse=True,
    )
    if not train_ranked:
        train_ranked = sorted(train_results, key=rank_key, reverse=True)
    split_results = [
        SplitRuleResult(
            expression=row.expression,
            train=row,
            holdout=evaluate_expression(holdout, row.expression),
        )
        for row in train_ranked
    ]
    return train, holdout, split_results


def print_records(records: list[CandidateRecord]) -> None:
    print("| Task | Step | Source | Quality | Match | Margin | Proxy | Oracle | Prompt |")
    print("| --- | ---: | --- | --- | --- | ---: | --- | --- | --- |")
    for record in records:
        prompt = record.prompt.replace("|", "\\|")
        proxy = record.proxy_token.replace("|", "\\|")
        oracle = record.oracle_token.replace("|", "\\|")
        print(
            f"| {record.task_id} | {record.step_index} | {record.label_source} | {record.label_quality} | {record.top1_match} "
            f"| {record.proxy_margin:.3f} | `{proxy}` | `{oracle}` | {prompt} |"
        )


def print_rules(results: list[RuleResult], top: int) -> None:
    print("| Rule | Accept err | Bad catch | Good reject | Accepted | Skip |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in results[:top]:
        expression = row.expression.replace("|", "\\|")
        print(
            f"| `{expression}` | {row.accepted_error_rate:.1%} | {row.bad_catch_rate:.1%} "
            f"| {row.good_reject_rate:.1%} | {row.accepted}/{row.total} | {row.skip_rate:.1%} |"
        )


def print_split_rules(results: list[SplitRuleResult], top: int) -> None:
    print("| Rule | Train skip | Train err | Holdout skip | Holdout err | Holdout bad catch | Holdout accepted |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results[:top]:
        expression = row.expression.replace("|", "\\|")
        print(
            f"| `{expression}` | {row.train.skip_rate:.1%} | {row.train.accepted_error_rate:.1%} "
            f"| {row.holdout.skip_rate:.1%} | {row.holdout.accepted_error_rate:.1%} "
            f"| {row.holdout.bad_catch_rate:.1%} | {row.holdout.accepted}/{row.holdout.total} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit SAGE candidate-token sparse verifier rules.")
    parser.add_argument("--probe-json", nargs="+", required=True)
    parser.add_argument("--replay-json", nargs="+", default=[], help="optional oracle replay JSON files from sage_candidate_replay.py")
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--label-quality", choices=["same-prefix", "diverged-prefix", "any"], default="same-prefix")
    parser.add_argument("--max-accepted-error", type=float, default=0.05)
    parser.add_argument("--pair-base-top", type=int, default=80)
    parser.add_argument("--holdout-modulus", type=int, default=0, help="if >1, hold out records where prompt_index %% modulus == remainder")
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--holdout-prompt-indices", default="", help="comma-separated prompt_index values to hold out")
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not 0 <= args.max_accepted_error <= 1:
        parser.error("--max-accepted-error must be in [0, 1]")
    if args.pair_base_top < 0:
        parser.error("--pair-base-top must be non-negative")
    if args.holdout_modulus < 0:
        parser.error("--holdout-modulus must be non-negative")
    if args.holdout_modulus == 1:
        parser.error("--holdout-modulus must be 0 or greater than 1")
    if args.holdout_modulus > 1 and not 0 <= args.holdout_remainder < args.holdout_modulus:
        parser.error("--holdout-remainder must be in [0, holdout-modulus)")
    try:
        holdout_prompt_indices = parse_prompt_indices(args.holdout_prompt_indices) if args.holdout_prompt_indices else set()
    except ValueError as exc:
        parser.error(f"invalid --holdout-prompt-indices: {exc}")
    if holdout_prompt_indices and args.holdout_modulus > 1:
        parser.error("use either --holdout-prompt-indices or --holdout-modulus, not both")
    replay_labels = load_replay_labels([Path(item) for item in args.replay_json], args.label) if args.replay_json else None
    records = load_records([Path(item) for item in args.probe_json], args.label, args.label_quality, replay_labels)
    if not records:
        parser.error("no candidate records found; capture with sage_probe_capture.py --tasks-json first")

    summary = summarize(records)
    if args.holdout_modulus > 1 or holdout_prompt_indices:
        train, holdout, split_results = fit_holdout(
            records,
            modulus=args.holdout_modulus,
            remainder=args.holdout_remainder,
            prompt_indices=holdout_prompt_indices,
            max_accepted_error=args.max_accepted_error,
            pair_base_top=args.pair_base_top,
            include_pairs=not args.no_pairs,
        )
        holdout_safe_by_acceptance = sorted(
            [row for row in split_results if row.holdout.accepted_error_rate <= args.max_accepted_error],
            key=lambda row: (row.holdout.skip_rate, row.holdout.bad_catch_rate, -row.holdout.good_reject_rate, row.train.skip_rate),
            reverse=True,
        )
        payload = {
            "params": {
                "probe_json": args.probe_json,
                "replay_json": args.replay_json,
                "label": args.label,
                "label_quality": args.label_quality,
                "max_accepted_error": args.max_accepted_error,
                "pair_base_top": args.pair_base_top,
                "include_pairs": not args.no_pairs,
                "holdout_modulus": args.holdout_modulus,
                "holdout_remainder": args.holdout_remainder,
                "holdout_prompt_indices": sorted(holdout_prompt_indices),
            },
            "summary": summary,
            "train_summary": summarize(train),
            "holdout_summary": summarize(holdout),
            "records": [asdict(record) for record in records],
            "heldout_by_acceptance": [asdict(row) for row in holdout_safe_by_acceptance],
            "split_results": [asdict(row) for row in split_results],
        }
        if args.json or args.json_out:
            text = json.dumps(payload, indent=2)
            if args.json:
                print(text)
            if args.json_out:
                out = Path(args.json_out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
                print(f"wrote: {out.resolve()}")
        else:
            print("# SAGE Candidate Sparse Probe Held-Out Fit")
            print()
            print("| Split | Records | Matches | Mismatches | Prompts |")
            print("| --- | ---: | ---: | ---: | ---: |")
            for label, rows in (("all", records), ("train", train), ("holdout", holdout)):
                item = summarize(rows)
                print(f"| {label} | {item['records']} | {item['matches']} | {item['mismatches']} | {item['prompt_count']} |")
            print()
            print(f"## Best Held-Out Rules Under {args.max_accepted_error:.1%} Accepted Error")
            print()
            print_split_rules(holdout_safe_by_acceptance, args.top)
        return 0

    results = fit_rules(records, args.max_accepted_error, args.pair_base_top, not args.no_pairs)
    safe_by_acceptance = sorted(
        [row for row in results if row.accepted_error_rate <= args.max_accepted_error],
        key=rank_key,
        reverse=True,
    )
    bad_catch_ranked = sorted(
        results,
        key=lambda row: (row.accepted_error_rate <= args.max_accepted_error, row.bad_catch_rate, row.skip_rate, -row.good_reject_rate),
        reverse=True,
    )
    payload = {
        "params": {
            "probe_json": args.probe_json,
            "replay_json": args.replay_json,
            "label": args.label,
            "label_quality": args.label_quality,
            "max_accepted_error": args.max_accepted_error,
            "pair_base_top": args.pair_base_top,
            "include_pairs": not args.no_pairs,
        },
        "summary": summary,
        "records": [asdict(record) for record in records],
        "safe_by_acceptance": [asdict(row) for row in safe_by_acceptance],
        "bad_catch_ranked": [asdict(row) for row in bad_catch_ranked],
    }

    if args.json or args.json_out:
        text = json.dumps(payload, indent=2)
        if args.json:
            print(text)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote: {out.resolve()}")
    else:
        print("# SAGE Candidate Sparse Probe Fit")
        print()
        print("| Metric | Value |")
        print("| --- | ---: |")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"| {key} | {value:.3f} |")
            else:
                print(f"| {key} | {value} |")
        print()
        print_records(records)
        print()
        if summary["matches"] == 0 or summary["mismatches"] == 0:
            print("Dataset has only one label class after filtering; use it for capture validation, not verifier claims.")
        else:
            print(f"## Best Acceptance Rules Under {args.max_accepted_error:.1%} Accepted Error")
            print()
            print_rules(safe_by_acceptance, args.top)
            print()
            print("## Best Bad-Catch Rules")
            print()
            print_rules(bad_catch_ranked, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
