#!/usr/bin/env python3
"""
Join sparse probe captures with proxy/oracle labels and fit first verifier rules.

The goal is deliberately modest: test whether cheap tensor summaries from the
oracle-side sparse probe contain any obvious signal for proxy/oracle agreement.
If the joined labels have only one class, the script reports that the dataset is
not fit-worthy instead of pretending a verifier exists.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class JoinedRecord:
    prompt: str
    step_index: int
    proxy_token: str
    oracle_token: str
    top1_match: bool
    proxy_margin: float
    proxy_entropy: float
    tensor_count: int
    features: dict[str, float]


@dataclass
class RuleResult:
    feature: str
    direction: str
    threshold: float
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


def fail(message: str, code: int = 2) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= max(int(value), 1)
    return out


def tensor_features(tensors: list[dict[str, Any]]) -> dict[str, float]:
    features: dict[str, float] = {}
    latest: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        latest[str(tensor.get("name", ""))] = tensor

    for name, tensor in latest.items():
        if not name:
            continue
        shape = [int(item) for item in tensor.get("shape", [])]
        total = float(tensor.get("sum", 0.0))
        n_values = product(shape) if shape else 1
        n_tokens = max(shape[1], 1) if len(shape) > 1 else 1
        safe = name.replace("-", "_")
        features[f"{safe}.sum"] = total
        features[f"{safe}.mean"] = float(tensor.get("mean")) if tensor.get("mean") is not None else total / n_values
        features[f"{safe}.per_token_sum"] = total / n_tokens
        if tensor.get("count") is not None:
            features[f"{safe}.count"] = float(tensor.get("count", 0))
        if tensor.get("min_value") is not None:
            features[f"{safe}.min"] = float(tensor.get("min_value", 0.0))
        if tensor.get("max_value") is not None:
            features[f"{safe}.max"] = float(tensor.get("max_value", 0.0))
        if tensor.get("nan_count") is not None:
            features[f"{safe}.nan_count"] = float(tensor.get("nan_count", 0))

    if "ffn_out-0" in latest and "l_out-0" in latest:
        ffn = float(latest["ffn_out-0"].get("sum", 0.0))
        out = float(latest["l_out-0"].get("sum", 0.0))
        features["layer0.residual_delta_sum"] = out - ffn
        features["layer0.ffn_to_lout_ratio"] = ffn / out if out else 0.0
    if "attn_out-0" in latest and "ffn_out-0" in latest:
        attn = float(latest["attn_out-0"].get("sum", 0.0))
        ffn = float(latest["ffn_out-0"].get("sum", 0.0))
        features["layer0.ffn_minus_attn_sum"] = ffn - attn
    return features


def load_probe_by_prompt(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_prompt: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        for capture in payload.get("captures", []):
            prompt = str(capture.get("prompt", ""))
            if prompt:
                by_prompt[prompt] = capture
    return by_prompt


def load_joined_records(logprob_path: Path, probe_paths: list[Path], step_index: int, label: str) -> list[JoinedRecord]:
    payload = load_json(logprob_path)
    probes = load_probe_by_prompt(probe_paths)
    records: list[JoinedRecord] = []
    for row in payload.get("rows", []):
        prompt = str(row.get("prompt", ""))
        probe = probes.get(prompt)
        if probe is None:
            continue
        steps = row.get("steps", [])
        proxy_steps = row.get("proxy", {}).get("steps", [])
        oracle_steps = row.get("oracle", {}).get("steps", [])
        if step_index >= len(steps):
            continue
        step = steps[step_index]
        proxy_token = str(proxy_steps[step_index].get("token", "")) if step_index < len(proxy_steps) else ""
        oracle_token = str(oracle_steps[step_index].get("token", "")) if step_index < len(oracle_steps) else ""
        match_key = f"top1_{label}_match"
        records.append(
            JoinedRecord(
                prompt=prompt,
                step_index=step_index,
                proxy_token=proxy_token,
                oracle_token=oracle_token,
                top1_match=bool(step.get(match_key, False)),
                proxy_margin=float(step.get("proxy_margin", 0.0)),
                proxy_entropy=float(step.get("proxy_entropy", 0.0)),
                tensor_count=int(probe.get("tensor_count", 0)),
                features=tensor_features(probe.get("tensors", [])),
            )
        )
    return records


def finite_feature_values(records: list[JoinedRecord]) -> dict[str, list[float]]:
    keys = sorted({key for record in records for key in record.features})
    out: dict[str, list[float]] = {}
    for key in keys:
        values = sorted({record.features[key] for record in records if key in record.features and math.isfinite(record.features[key])})
        if values:
            out[key] = values
    return out


def evaluate_rule(records: list[JoinedRecord], feature: str, direction: str, threshold: float, accept_fn: Callable[[float], bool]) -> RuleResult:
    accepted_records = [record for record in records if feature in record.features and accept_fn(record.features[feature])]
    rejected_records = [record for record in records if record not in accepted_records]
    true_accepts = sum(1 for record in accepted_records if record.top1_match)
    false_accepts = len(accepted_records) - true_accepts
    true_rejects = sum(1 for record in rejected_records if not record.top1_match)
    false_rejects = len(rejected_records) - true_rejects
    bad_total = sum(1 for record in records if not record.top1_match)
    good_total = sum(1 for record in records if record.top1_match)
    accepted = len(accepted_records)
    return RuleResult(
        feature=feature,
        direction=direction,
        threshold=threshold,
        accepted=accepted,
        rejected=len(rejected_records),
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        total=len(records),
        precision=true_accepts / accepted if accepted else 1.0,
        accepted_error_rate=false_accepts / accepted if accepted else 0.0,
        bad_catch_rate=true_rejects / bad_total if bad_total else 0.0,
        good_reject_rate=false_rejects / good_total if good_total else 0.0,
    )


def fit_rules(records: list[JoinedRecord]) -> list[RuleResult]:
    results: list[RuleResult] = []
    for feature, values in finite_feature_values(records).items():
        for threshold in values:
            results.append(evaluate_rule(records, feature, "<=", threshold, lambda value, t=threshold: value <= t))
            results.append(evaluate_rule(records, feature, ">=", threshold, lambda value, t=threshold: value >= t))
    return results


def summarize(records: list[JoinedRecord]) -> dict[str, Any]:
    matches = sum(1 for record in records if record.top1_match)
    return {
        "records": len(records),
        "matches": matches,
        "mismatches": len(records) - matches,
        "match_rate": matches / len(records) if records else 0.0,
        "mean_proxy_margin": sum(record.proxy_margin for record in records) / len(records) if records else 0.0,
        "mean_tensor_count": sum(record.tensor_count for record in records) / len(records) if records else 0.0,
        "features": len({key for record in records for key in record.features}),
    }


def print_records(records: list[JoinedRecord]) -> None:
    print("| Prompt | Match | Proxy | Oracle | Margin | Tensors |")
    print("| --- | --- | --- | --- | ---: | ---: |")
    for record in records:
        prompt = record.prompt.replace("|", "\\|")
        proxy = record.proxy_token.replace("|", "\\|")
        oracle = record.oracle_token.replace("|", "\\|")
        print(f"| {prompt} | {record.top1_match} | `{proxy}` | `{oracle}` | {record.proxy_margin:.3f} | {record.tensor_count} |")


def print_rules(results: list[RuleResult], top: int) -> None:
    print("| Feature | Dir | Threshold | Accept err | Bad catch | Good reject | Accepted | Skip |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results[:top]:
        skip_rate = row.accepted / row.total if row.total else 0.0
        print(
            f"| `{row.feature}` | {row.direction} | {row.threshold:.6g} | {row.accepted_error_rate:.1%} "
            f"| {row.bad_catch_rate:.1%} | {row.good_reject_rate:.1%} | {row.accepted}/{row.total} | {skip_rate:.1%} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit first sparse-probe verifier rules against proxy/oracle labels.")
    parser.add_argument("--logprob-json", required=True)
    parser.add_argument("--probe-json", nargs="+", required=True)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--label", choices=["token", "id"], default="token")
    parser.add_argument("--max-accepted-error", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.step_index < 0:
        parser.error("--step-index must be non-negative")
    if not 0 <= args.max_accepted_error <= 1:
        parser.error("--max-accepted-error must be in [0, 1]")
    records = load_joined_records(Path(args.logprob_json), [Path(item) for item in args.probe_json], args.step_index, args.label)
    if not records:
        parser.error("no joined records; prompts or step index do not match")

    summary = summarize(records)
    results = fit_rules(records)
    viable = summary["matches"] > 0 and summary["mismatches"] > 0
    ranked = sorted(
        results,
        key=lambda row: (
            row.accepted_error_rate <= args.max_accepted_error,
            row.bad_catch_rate,
            row.accepted / row.total if row.total else 0.0,
            -row.good_reject_rate,
            row.precision,
        ),
        reverse=True,
    )
    safe_by_acceptance = sorted(
        [row for row in results if row.accepted_error_rate <= args.max_accepted_error],
        key=lambda row: (row.accepted / row.total if row.total else 0.0, row.bad_catch_rate, -row.good_reject_rate, row.precision),
        reverse=True,
    )
    payload = {
        "summary": summary,
        "params": {
            "logprob_json": args.logprob_json,
            "probe_json": args.probe_json,
            "step_index": args.step_index,
            "label": args.label,
            "max_accepted_error": args.max_accepted_error,
        },
        "viable_for_rule_fit": viable,
        "records": [asdict(record) for record in records],
        "results": [asdict(row) for row in ranked],
        "safe_by_acceptance": [asdict(row) for row in safe_by_acceptance],
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
        print("# SAGE Sparse Probe Fit")
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
        if viable:
            print("## Best Bad-Catch Rules")
            print()
            print_rules(ranked, args.top)
            print()
            print(f"## Best Acceptance Rules Under {args.max_accepted_error:.1%} Accepted Error")
            print()
            print_rules(safe_by_acceptance, args.top)
        else:
            print("Dataset is not viable for verifier fitting because it contains only one label class.")
            print("Collect chat-aligned captures or a larger raw set with both proxy/oracle matches and mismatches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
