#!/usr/bin/env python3
"""
Fit transparent proxy-only routing rules from SAGE logprob probes.

A real SAGE runtime must decide whether to trust the proxy before it calls the
oracle. Therefore this script uses only proxy-side features:

- proxy top-1 margin
- proxy top-k entropy
- generated step index

The oracle is used only as the label during offline evaluation. The output is a
set of threshold rules with skip rate, false-accept rate, oracle-call rate, and
estimated throughput.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


BYTES_PER_GIB = 1024**3


@dataclass
class StepRecord:
    source: str
    prompt: str
    step_index: int
    proxy_token: str
    oracle_token: str
    proxy_margin: float
    proxy_entropy: float
    topk_token_jaccard: float
    top1_token_match: bool
    top1_id_match: bool


@dataclass
class RuleResult:
    rule: str
    margin_threshold: float | None
    entropy_threshold: float | None
    accepted: int
    rejected: int
    true_accepts: int
    false_accepts: int
    total: int
    skip_rate: float
    oracle_call_rate: float
    precision: float
    recall: float
    accepted_error_rate: float
    total_error_rate: float
    effective_tps: float
    meets_target_tps: bool
    max_false_accept_rate: float
    passes_safety: bool


def expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(value))
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


def load_records(paths: list[Path], ignore_prefix_steps: int, label: str) -> list[StepRecord]:
    records: list[StepRecord] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            prompt = str(row.get("prompt", ""))
            proxy_steps = row.get("proxy", {}).get("steps", [])
            oracle_steps = row.get("oracle", {}).get("steps", [])
            for step in row.get("steps", []):
                idx = int(step.get("index", 0))
                if idx < ignore_prefix_steps:
                    continue
                proxy_token = ""
                oracle_token = ""
                if idx < len(proxy_steps):
                    proxy_token = str(proxy_steps[idx].get("token", ""))
                if idx < len(oracle_steps):
                    oracle_token = str(oracle_steps[idx].get("token", ""))
                records.append(
                    StepRecord(
                        source=str(path),
                        prompt=prompt,
                        step_index=idx,
                        proxy_token=proxy_token,
                        oracle_token=oracle_token,
                        proxy_margin=float(step.get("proxy_margin", 0.0)),
                        proxy_entropy=float(step.get("proxy_entropy", 0.0)),
                        topk_token_jaccard=float(step.get("topk_token_jaccard", 0.0)),
                        top1_token_match=bool(step.get("top1_token_match", False)),
                        top1_id_match=bool(step.get("top1_id_match", False)),
                    )
                )
    if label not in {"token", "id"}:
        raise ValueError("label must be token or id")
    return records


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def oracle_ms_per_call(
    *,
    params_b: float,
    quant_bpw: float,
    active_percent: float,
    pcie_gbps: float,
    oracle_compute_ms: float,
    oracle_fixed_ms: float,
) -> float:
    dense_gib = dense_weight_gib(params_b, quant_bpw)
    active_gib = dense_gib * active_percent / 100.0
    transfer_ms = active_gib / (pcie_gbps * (1_000_000_000 / BYTES_PER_GIB)) * 1000.0
    return transfer_ms + oracle_compute_ms + oracle_fixed_ms


def evaluate_rule(
    *,
    records: list[StepRecord],
    label: str,
    rule: str,
    margin_threshold: float | None,
    entropy_threshold: float | None,
    accept_fn: Callable[[StepRecord], bool],
    target_tps: float,
    proxy_tps: float,
    oracle_ms: float,
    max_false_accept_rate: float,
) -> RuleResult:
    accepted_records = [record for record in records if accept_fn(record)]
    accepted = len(accepted_records)
    total = len(records)
    positives = sum(1 for record in records if getattr(record, f"top1_{label}_match"))
    true_accepts = sum(1 for record in accepted_records if getattr(record, f"top1_{label}_match"))
    false_accepts = accepted - true_accepts
    rejected = total - accepted
    skip_rate = accepted / total if total else 0.0
    oracle_call_rate = 1.0 - skip_rate
    precision = true_accepts / accepted if accepted else 1.0
    recall = true_accepts / positives if positives else 0.0
    accepted_error_rate = false_accepts / accepted if accepted else 0.0
    total_error_rate = false_accepts / total if total else 0.0
    expected_ms = (1000.0 / proxy_tps) + oracle_call_rate * oracle_ms
    effective_tps = 1000.0 / expected_ms if expected_ms > 0 else 0.0

    return RuleResult(
        rule=rule,
        margin_threshold=margin_threshold,
        entropy_threshold=entropy_threshold,
        accepted=accepted,
        rejected=rejected,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        total=total,
        skip_rate=skip_rate,
        oracle_call_rate=oracle_call_rate,
        precision=precision,
        recall=recall,
        accepted_error_rate=accepted_error_rate,
        total_error_rate=total_error_rate,
        effective_tps=effective_tps,
        meets_target_tps=effective_tps >= target_tps,
        max_false_accept_rate=max_false_accept_rate,
        passes_safety=accepted_error_rate <= max_false_accept_rate,
    )


def finite_values(values: list[float]) -> list[float]:
    out = sorted({value for value in values if math.isfinite(value)})
    if 0.0 not in out:
        out.insert(0, 0.0)
    return out


def fit_rules(
    *,
    records: list[StepRecord],
    label: str,
    target_tps: float,
    proxy_tps: float,
    oracle_ms: float,
    max_false_accept_rate: float,
) -> list[RuleResult]:
    margins = finite_values([record.proxy_margin for record in records])
    entropies = finite_values([record.proxy_entropy for record in records])
    classes = sorted({token_class(record.proxy_token) for record in records})
    results: list[RuleResult] = []

    for margin in margins:
        results.append(
            evaluate_rule(
                records=records,
                label=label,
                rule="margin>=",
                margin_threshold=margin,
                entropy_threshold=None,
                accept_fn=lambda record, m=margin: record.proxy_margin >= m,
                target_tps=target_tps,
                proxy_tps=proxy_tps,
                oracle_ms=oracle_ms,
                max_false_accept_rate=max_false_accept_rate,
            )
        )

    for entropy in entropies:
        results.append(
            evaluate_rule(
                records=records,
                label=label,
                rule="entropy<=",
                margin_threshold=None,
                entropy_threshold=entropy,
                accept_fn=lambda record, e=entropy: record.proxy_entropy <= e,
                target_tps=target_tps,
                proxy_tps=proxy_tps,
                oracle_ms=oracle_ms,
                max_false_accept_rate=max_false_accept_rate,
            )
        )

    for margin in margins:
        for entropy in entropies:
            results.append(
                evaluate_rule(
                    records=records,
                    label=label,
                    rule="margin>= AND entropy<=",
                    margin_threshold=margin,
                    entropy_threshold=entropy,
                    accept_fn=lambda record, m=margin, e=entropy: record.proxy_margin >= m and record.proxy_entropy <= e,
                    target_tps=target_tps,
                    proxy_tps=proxy_tps,
                    oracle_ms=oracle_ms,
                    max_false_accept_rate=max_false_accept_rate,
                )
            )

    for cls in classes:
        results.append(
            evaluate_rule(
                records=records,
                label=label,
                rule=f"class=={cls}",
                margin_threshold=None,
                entropy_threshold=None,
                accept_fn=lambda record, c=cls: token_class(record.proxy_token) == c,
                target_tps=target_tps,
                proxy_tps=proxy_tps,
                oracle_ms=oracle_ms,
                max_false_accept_rate=max_false_accept_rate,
            )
        )
        for margin in margins:
            results.append(
                evaluate_rule(
                    records=records,
                    label=label,
                    rule=f"class=={cls} AND margin>=",
                    margin_threshold=margin,
                    entropy_threshold=None,
                    accept_fn=lambda record, c=cls, m=margin: token_class(record.proxy_token) == c and record.proxy_margin >= m,
                    target_tps=target_tps,
                    proxy_tps=proxy_tps,
                    oracle_ms=oracle_ms,
                    max_false_accept_rate=max_false_accept_rate,
                )
            )
        for entropy in entropies:
            results.append(
                evaluate_rule(
                    records=records,
                    label=label,
                    rule=f"class=={cls} AND entropy<=",
                    margin_threshold=None,
                    entropy_threshold=entropy,
                    accept_fn=lambda record, c=cls, e=entropy: token_class(record.proxy_token) == c and record.proxy_entropy <= e,
                    target_tps=target_tps,
                    proxy_tps=proxy_tps,
                    oracle_ms=oracle_ms,
                    max_false_accept_rate=max_false_accept_rate,
                )
            )

    return results


def summarize_records(records: list[StepRecord], label: str) -> dict[str, Any]:
    positives = sum(1 for record in records if getattr(record, f"top1_{label}_match"))
    classes: dict[str, dict[str, int]] = {}
    for record in records:
        cls = token_class(record.proxy_token)
        bucket = classes.setdefault(cls, {"total": 0, "matches": 0})
        bucket["total"] += 1
        bucket["matches"] += int(getattr(record, f"top1_{label}_match"))
    return {
        "records": len(records),
        "label": label,
        "matches": positives,
        "match_rate": positives / len(records) if records else 0.0,
        "mean_margin": sum(record.proxy_margin for record in records) / len(records) if records else 0.0,
        "mean_entropy": sum(record.proxy_entropy for record in records) / len(records) if records else 0.0,
        "mean_topk_token_jaccard": sum(record.topk_token_jaccard for record in records) / len(records) if records else 0.0,
        "token_classes": {
            cls: {
                "total": data["total"],
                "matches": data["matches"],
                "match_rate": data["matches"] / data["total"] if data["total"] else 0.0,
            }
            for cls, data in sorted(classes.items())
        },
    }


def rule_sort_key(row: RuleResult) -> tuple[float, float, float]:
    return (row.effective_tps, row.skip_rate, row.precision)


def print_rule_table(rows: list[RuleResult], top: int) -> None:
    print("| Rule | Margin | Entropy | Skip | Precision | Accepted error | Oracle calls | Tok/s | Target | Safe |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows[:top]:
        margin = "" if row.margin_threshold is None else f"{row.margin_threshold:.3f}"
        entropy = "" if row.entropy_threshold is None else f"{row.entropy_threshold:.3f}"
        print(
            f"| {row.rule} | {margin} | {entropy} | {row.skip_rate:.1%} | {row.precision:.1%} "
            f"| {row.accepted_error_rate:.1%} | {row.oracle_call_rate:.1%} | {row.effective_tps:.2f} "
            f"| {row.meets_target_tps} | {row.passes_safety} |"
        )


def print_result(summary: dict[str, Any], results: list[RuleResult], top: int) -> None:
    print("# SAGE Proxy Router Fit")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| Records | {summary['records']} |")
    print(f"| Label | {summary['label']} |")
    print(f"| Top-1 matches | {summary['matches']} ({summary['match_rate']:.1%}) |")
    print(f"| Mean proxy margin | {summary['mean_margin']:.3f} |")
    print(f"| Mean proxy entropy | {summary['mean_entropy']:.3f} |")
    print(f"| Mean top-k token Jaccard | {summary['mean_topk_token_jaccard']:.3f} |")
    print()
    safe = sorted([row for row in results if row.passes_safety], key=rule_sort_key, reverse=True)
    target_safe = [row for row in safe if row.meets_target_tps]
    fastest = sorted(results, key=rule_sort_key, reverse=True)
    print("## Best Safe Rules")
    print()
    print_rule_table(safe, top)
    print()
    print("## Safe Rules That Meet Target")
    print()
    if target_safe:
        print_rule_table(target_safe, top)
    else:
        print("No rule met both the safety and throughput target.")
    print()
    print("## Fastest Rules")
    print()
    print_rule_table(fastest, top)
    print()
    print("## Token Classes")
    print()
    print("| Class | Total | Matches | Match rate |")
    print("| --- | ---: | ---: | ---: |")
    for cls, data in summary["token_classes"].items():
        print(f"| {cls} | {data['total']} | {data['matches']} | {data['match_rate']:.1%} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit proxy-only SAGE routing thresholds from logprob probe JSON.")
    parser.add_argument("--logprob-json", nargs="+", required=True, help="one or more sage_logprob_probe JSON files or globs")
    parser.add_argument("--ignore-prefix-steps", type=int, default=0)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--proxy-tps", type=float, default=25.0)
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--active-percent", type=float, default=10.0)
    parser.add_argument("--pcie-gbps", type=float, default=24.0)
    parser.add_argument("--oracle-compute-ms", type=float, default=10.0)
    parser.add_argument("--oracle-fixed-ms", type=float, default=5.0)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.ignore_prefix_steps < 0:
        parser.error("--ignore-prefix-steps must be non-negative")
    if args.proxy_tps <= 0:
        parser.error("--proxy-tps must be positive")
    if args.target_tps <= 0:
        parser.error("--target-tps must be positive")
    if not 0 <= args.active_percent <= 100:
        parser.error("--active-percent must be in [0, 100]")
    if not 0 <= args.max_false_accept_rate <= 1:
        parser.error("--max-false-accept-rate must be in [0, 1]")

    try:
        paths = expand_inputs(args.logprob_json)
        records = load_records(paths, args.ignore_prefix_steps, args.label)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not records:
        parser.error("no records after filtering")

    oracle_ms = oracle_ms_per_call(
        params_b=args.params_b,
        quant_bpw=args.quant_bpw,
        active_percent=args.active_percent,
        pcie_gbps=args.pcie_gbps,
        oracle_compute_ms=args.oracle_compute_ms,
        oracle_fixed_ms=args.oracle_fixed_ms,
    )
    results = fit_rules(
        records=records,
        label=args.label,
        target_tps=args.target_tps,
        proxy_tps=args.proxy_tps,
        oracle_ms=oracle_ms,
        max_false_accept_rate=args.max_false_accept_rate,
    )
    summary = summarize_records(records, args.label)
    payload = {
        "params": {
            "files": [str(path) for path in paths],
            "ignore_prefix_steps": args.ignore_prefix_steps,
            "target_tps": args.target_tps,
            "proxy_tps": args.proxy_tps,
            "params_b": args.params_b,
            "quant_bpw": args.quant_bpw,
            "active_percent": args.active_percent,
            "oracle_ms_per_call": oracle_ms,
            "max_false_accept_rate": args.max_false_accept_rate,
        },
        "summary": summary,
        "results": [asdict(row) for row in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_result(summary, results, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
