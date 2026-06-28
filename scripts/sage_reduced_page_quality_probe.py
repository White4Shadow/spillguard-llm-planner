#!/usr/bin/env python3
"""
Compare a reduced real-activation page plan against a fuller real-activation
page plan.

This is a quality probe for SAGE page selection. It does not prove token-level
candidate acceptance. It checks whether the reduced plan preserves row-score
signals from the fuller sparse matvec artifact for the same captured activation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"expected JSON object in {path}")
    return payload


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("summary", payload)
    return raw if isinstance(raw, dict) else {}


def int_value(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def float_value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def bytes_to_gib(n_bytes: int | float) -> float:
    return float(n_bytes) / BYTES_PER_GIB


def stage_tensors(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for stage in payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for tensor in stage.get("tensors", []):
            if isinstance(tensor, dict) and isinstance(tensor.get("name"), str):
                tensors[tensor["name"]] = tensor
    return tensors


def scored_tensors(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    scored: dict[str, list[dict[str, Any]]] = {}
    for stage in payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for tensor in stage.get("top_scores", []):
            if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
                continue
            rows = tensor.get("top_scores", [])
            scored[tensor["name"]] = [row for row in rows if isinstance(row, dict)]
    return scored


def global_top_rows(scored: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor_name, top_rows in scored.items():
        for row in top_rows:
            row_id = int_value(row, "row", -1)
            score = float_value(row, "score")
            rows.append(
                {
                    "tensor": tensor_name,
                    "row": row_id,
                    "score": score,
                    "abs_score": abs(score),
                }
            )
    return sorted(rows, key=lambda item: item["abs_score"], reverse=True)


def top_row_map(scored: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for tensor_name, rows in scored.items():
        for row in rows:
            result[(tensor_name, int_value(row, "row", -1))] = float_value(row, "score")
    return result


def validate_real_activation_artifact(payload: dict[str, Any], label: str) -> None:
    allowed_schemas = {
        "sage-oracle-page-cuda-real-activation-matvec-smoke-v0",
        "sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0",
    }
    if payload.get("schema") not in allowed_schemas:
        fail(f"{label} schema is not a real-activation matvec artifact")
    payload_summary = summary(payload)
    if payload_summary.get("activation_mode") != "real_tensor_values_jsonl":
        fail(f"{label} does not use a real tensor-values activation")
    if payload_summary.get("row_score_capture_status") != "measured_per_row_scores":
        fail(f"{label} does not include per-row top score capture")


def build_probe(args: argparse.Namespace) -> dict[str, Any]:
    full_path = Path(args.full_matvec_json)
    reduced_path = Path(args.reduced_matvec_json)
    full = load_json(full_path)
    reduced = load_json(reduced_path)
    validate_real_activation_artifact(full, "full")
    validate_real_activation_artifact(reduced, "reduced")

    full_summary = summary(full)
    reduced_summary = summary(reduced)
    full_activation = full.get("activation", {})
    full_activation = full_activation if isinstance(full_activation, dict) else {}
    reduced_activation = reduced.get("activation", {})
    reduced_activation = reduced_activation if isinstance(reduced_activation, dict) else {}
    activation_match = (
        full_activation.get("source_jsonl") == reduced_activation.get("source_jsonl")
        and full_activation.get("name") == reduced_activation.get("name")
        and full_activation.get("record_index") == reduced_activation.get("record_index")
        and int_value(full_activation, "value_count") == int_value(reduced_activation, "value_count")
    )

    full_selected = stage_tensors(full)
    reduced_selected = stage_tensors(reduced)
    full_scored = scored_tensors(full)
    reduced_scored = scored_tensors(reduced)
    shared_scored_names = sorted(set(full_scored) & set(reduced_scored))

    top1_matches = 0
    topk_overlaps: list[float] = []
    max_score_abs_delta = 0.0
    shared_details: list[dict[str, Any]] = []
    reduced_score_map = top_row_map(reduced_scored)

    for name in shared_scored_names:
        full_rows = full_scored[name]
        reduced_rows = reduced_scored[name]
        full_top_rows = [int_value(row, "row", -1) for row in full_rows[: args.top_k]]
        reduced_top_rows = [int_value(row, "row", -1) for row in reduced_rows[: args.top_k]]
        full_set = set(full_top_rows)
        reduced_set = set(reduced_top_rows)
        overlap = len(full_set & reduced_set) / max(len(full_set), 1)
        topk_overlaps.append(overlap)
        top1_match = bool(full_top_rows and reduced_top_rows and full_top_rows[0] == reduced_top_rows[0])
        if top1_match:
            top1_matches += 1
        row_deltas: list[float] = []
        for row in full_rows:
            key = (name, int_value(row, "row", -1))
            if key in reduced_score_map:
                delta = abs(float_value(row, "score") - reduced_score_map[key])
                row_deltas.append(delta)
                max_score_abs_delta = max(max_score_abs_delta, delta)
        shared_details.append(
            {
                "tensor": name,
                "full_top1_row": full_top_rows[0] if full_top_rows else None,
                "reduced_top1_row": reduced_top_rows[0] if reduced_top_rows else None,
                "top1_match": top1_match,
                "top_k_overlap": overlap,
                "max_shared_row_score_abs_delta": max(row_deltas, default=0.0),
            }
        )

    full_global = global_top_rows(full_scored)
    reduced_keys = set(top_row_map(reduced_scored))
    retention: dict[str, dict[str, Any]] = {}
    for n_rows in args.global_top_n:
        selected = full_global[:n_rows]
        retained = [row for row in selected if (row["tensor"], row["row"]) in reduced_keys]
        retention[str(n_rows)] = {
            "full_rows": len(selected),
            "retained_rows": len(retained),
            "retention_rate": len(retained) / max(len(selected), 1),
        }

    full_bytes = int_value(full_summary, "q4_0_bytes")
    reduced_bytes = int_value(reduced_summary, "q4_0_bytes")
    shared_top1_match_rate = top1_matches / max(len(shared_scored_names), 1)
    shared_topk_overlap_mean = sum(topk_overlaps) / max(len(topk_overlaps), 1)
    top20_retention = retention.get("20", {}).get("retention_rate", 0.0)
    reduced_signal_consistent = (
        activation_match
        and len(shared_scored_names) > 0
        and shared_top1_match_rate >= args.min_shared_top1_match_rate
        and shared_topk_overlap_mean >= args.min_shared_topk_overlap
        and max_score_abs_delta <= args.max_shared_score_abs_delta
    )
    selection_needs_signal_optimization = top20_retention < args.min_global_top20_retention

    payload = {
        "schema": "sage-reduced-page-quality-probe-v0",
        "status": "measured_reduced_real_activation_signal_overlap_not_token_decisions",
        "inputs": {
            "full_matvec_json": str(full_path.resolve()),
            "reduced_matvec_json": str(reduced_path.resolve()),
            "top_k": args.top_k,
            "global_top_n": args.global_top_n,
        },
        "activation": {
            "full_name": full_activation.get("name", ""),
            "reduced_name": reduced_activation.get("name", ""),
            "full_source_jsonl": full_activation.get("source_jsonl", ""),
            "reduced_source_jsonl": reduced_activation.get("source_jsonl", ""),
            "record_index_match": full_activation.get("record_index") == reduced_activation.get("record_index"),
            "value_count_match": int_value(full_activation, "value_count")
            == int_value(reduced_activation, "value_count"),
            "activation_match": activation_match,
        },
        "summary": {
            "full_q4_0_bytes": full_bytes,
            "full_q4_0_gib": bytes_to_gib(full_bytes),
            "reduced_q4_0_bytes": reduced_bytes,
            "reduced_q4_0_gib": bytes_to_gib(reduced_bytes),
            "reduced_vs_full_q4_0_percent": 100.0 * reduced_bytes / max(full_bytes, 1),
            "full_selected_tensors": len(full_selected),
            "reduced_selected_tensors": len(reduced_selected),
            "reduced_selected_subset_of_full_selected": set(reduced_selected).issubset(set(full_selected)),
            "full_scored_tensors": len(full_scored),
            "reduced_scored_tensors": len(reduced_scored),
            "shared_scored_tensors": len(shared_scored_names),
            "shared_top1_matches": top1_matches,
            "shared_top1_match_rate": shared_top1_match_rate,
            "shared_topk_overlap_mean": shared_topk_overlap_mean,
            "shared_topk_overlap_min": min(topk_overlaps, default=0.0),
            "max_shared_score_abs_delta": max_score_abs_delta,
            "global_top_retention": retention,
            "global_top20_retention_rate": top20_retention,
            "reduced_signal_consistent_with_full_shared_tensors": reduced_signal_consistent,
            "selection_needs_signal_aware_optimization": selection_needs_signal_optimization,
            "token_decision_integrated": False,
            "candidate_token_quality_proven": False,
            "quality_status": (
                "shared_tensor_rows_match_but_global_signal_retention_low"
                if selection_needs_signal_optimization
                else "shared_tensor_rows_match_and_global_signal_retention_meets_probe_floor"
            ),
        },
        "shared_tensor_details": shared_details,
        "top_missing_full_rows": [
            row for row in full_global[: args.report_missing_top_n] if (row["tensor"], row["row"]) not in reduced_keys
        ],
        "next_step": (
            "Use signal-aware page selection so the 1.18 GiB plan retains more of the full-plan global "
            "real-activation row signal, then map retained rows to candidate-token decisions."
        ),
    }
    return payload


def print_markdown(payload: dict[str, Any]) -> None:
    summary_payload = summary(payload)
    print("# SAGE Reduced Page Quality Probe")
    print()
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Full Q4_0: `{summary_payload['full_q4_0_gib']:.3f} GiB`")
    print(f"- Reduced Q4_0: `{summary_payload['reduced_q4_0_gib']:.3f} GiB`")
    print(f"- Reduced/full bytes: `{summary_payload['reduced_vs_full_q4_0_percent']:.1f}%`")
    print(f"- Shared scored tensors: `{summary_payload['shared_scored_tensors']}`")
    print(f"- Shared top-1 match rate: `{summary_payload['shared_top1_match_rate']:.1%}`")
    print(f"- Shared top-k overlap mean: `{summary_payload['shared_topk_overlap_mean']:.1%}`")
    print(f"- Global top-20 retention: `{summary_payload['global_top20_retention_rate']:.1%}`")
    print(f"- Signal consistency: `{summary_payload['reduced_signal_consistent_with_full_shared_tensors']}`")
    print(f"- Token decision integrated: `{summary_payload['token_decision_integrated']}`")
    print()
    print("## Retention")
    print()
    print("| Full global top N | Retained rows | Retention |")
    print("| ---: | ---: | ---: |")
    for label, values in summary_payload["global_top_retention"].items():
        print(f"| {label} | {values['retained_rows']}/{values['full_rows']} | {values['retention_rate']:.1%} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare reduced SAGE page row signals against a fuller page plan.")
    parser.add_argument(
        "--full-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-ffn-norm0-full.json",
    )
    parser.add_argument(
        "--reduced-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-matvec-gemma31b-balanced-1180mib-ffn-norm0.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--global-top-n", type=int, nargs="+", default=[10, 20, 50, 100, 140])
    parser.add_argument("--report-missing-top-n", type=int, default=20)
    parser.add_argument("--min-shared-top1-match-rate", type=float, default=0.99)
    parser.add_argument("--min-shared-topk-overlap", type=float, default=0.99)
    parser.add_argument("--max-shared-score-abs-delta", type=float, default=1.0e-5)
    parser.add_argument("--min-global-top20-retention", type=float, default=0.80)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if any(value <= 0 for value in args.global_top_n):
        parser.error("--global-top-n values must be positive")
    if args.report_missing_top_n < 0:
        parser.error("--report-missing-top-n must be non-negative")

    payload = build_probe(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_markdown(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
