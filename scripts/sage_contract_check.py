#!/usr/bin/env python3
"""
Check the current SAGE-100 evidence against the active-byte oracle contract.

This script is intentionally stricter than the individual smoke tests. It does
not decide that the 100B goal is done just because replay and small-model live
checks pass. Instead, it separates:

- implemented and measured runtime pieces;
- modeled 100B active-byte assumptions;
- missing production gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BYTES_PER_GIB = 1024**3


@dataclass
class Gate:
    stage: str
    name: str
    passed: bool
    evidence: str
    next_step: str


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            fail(f"missing JSON artifact: {path}")
        return {}
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


def nested_summary(path: Path, *, required: bool = True) -> dict[str, Any]:
    return summary(load_json(path, required=required))


def bool_value(mapping: dict[str, Any], key: str, default: bool = False) -> bool:
    value = mapping.get(key, default)
    return bool(value)


def int_value(mapping: dict[str, Any], key: str, default: int = 0) -> int:
    value = mapping.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bytes_to_gib(n_bytes: int | float) -> float:
    return float(n_bytes) / BYTES_PER_GIB


def scenario_by_name(projection: dict[str, Any], name: str) -> dict[str, Any]:
    scenarios = projection.get("scenarios", [])
    if not isinstance(scenarios, list):
        return {}
    for item in scenarios:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def limit_by_name(projection: dict[str, Any], name: str) -> dict[str, Any]:
    limits = projection.get("limits", [])
    if not isinstance(limits, list):
        return {}
    for item in limits:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def fallback_limit_by_name_target(projection: dict[str, Any], name: str, target_tps: float) -> dict[str, Any]:
    limits = projection.get("fallback_limits", [])
    if not isinstance(limits, list):
        return {}
    for item in limits:
        if (
            isinstance(item, dict)
            and item.get("name") == name
            and abs(float_value(item, "target_tps") - target_tps) <= 1.0e-9
        ):
            return item
    return {}


def check_contract(args: argparse.Namespace) -> dict[str, Any]:
    policy = nested_summary(Path(args.policy_json))
    parity = nested_summary(Path(args.policy_parity_json))
    scheduler = nested_summary(Path(args.scheduler_replay_json))
    proxy = nested_summary(Path(args.proxy_live_json))
    dual_payload = load_json(Path(args.dual_live_json))
    dual = summary(dual_payload)
    dual_steps_raw = dual_payload.get("steps", [])
    dual_steps = dual_steps_raw if isinstance(dual_steps_raw, list) else []
    page_ledger_payload = load_json(Path(args.oracle_page_ledger_json), required=False)
    page_ledger_summary = summary(page_ledger_payload)
    page_staging_payload = load_json(Path(args.oracle_page_staging_json), required=False)
    page_staging_summary = summary(page_staging_payload)
    page_cuda_staging_payload = load_json(Path(args.oracle_page_cuda_staging_json), required=False)
    page_cuda_staging_summary = summary(page_cuda_staging_payload)
    page_cuda_kernel_payload = load_json(Path(args.oracle_page_cuda_kernel_json), required=False)
    page_cuda_kernel_summary = summary(page_cuda_kernel_payload)
    page_cuda_overlap_payload = load_json(Path(args.oracle_page_cuda_overlap_json), required=False)
    page_cuda_overlap_summary = summary(page_cuda_overlap_payload)
    page_cuda_prefetch_overlap_payload = load_json(Path(args.oracle_page_cuda_prefetch_overlap_json), required=False)
    page_cuda_prefetch_overlap_summary = summary(page_cuda_prefetch_overlap_payload)
    page_cuda_page_cache_payload = load_json(Path(args.oracle_page_cuda_page_cache_json), required=False)
    page_cuda_page_cache_summary = summary(page_cuda_page_cache_payload)
    page_cuda_dequant_payload = load_json(Path(args.oracle_page_cuda_dequant_json), required=False)
    page_cuda_dequant_summary = summary(page_cuda_dequant_payload)
    page_cuda_matvec_payload = load_json(Path(args.oracle_page_cuda_matvec_json), required=False)
    page_cuda_matvec_summary = summary(page_cuda_matvec_payload)
    reduced_page_cuda_dequant_payload = load_json(Path(args.reduced_oracle_page_cuda_dequant_json), required=False)
    reduced_page_cuda_dequant_summary = summary(reduced_page_cuda_dequant_payload)
    reduced_page_cuda_matvec_payload = load_json(Path(args.reduced_oracle_page_cuda_matvec_json), required=False)
    reduced_page_cuda_matvec_summary = summary(reduced_page_cuda_matvec_payload)
    reduced_page_cuda_real_activation_matvec_payload = load_json(
        Path(args.reduced_oracle_page_cuda_real_activation_matvec_json),
        required=False,
    )
    reduced_page_cuda_real_activation_matvec_summary = summary(reduced_page_cuda_real_activation_matvec_payload)
    page_cuda_real_activation_matvec_payload = load_json(
        Path(args.oracle_page_cuda_real_activation_matvec_json),
        required=False,
    )
    page_cuda_real_activation_matvec_summary = summary(page_cuda_real_activation_matvec_payload)
    page_cuda_real_activation_ranked_matvec_payload = load_json(
        Path(args.oracle_page_cuda_real_activation_ranked_matvec_json),
        required=False,
    )
    page_cuda_real_activation_ranked_matvec_summary = summary(page_cuda_real_activation_ranked_matvec_payload)
    page_cuda_vocab_projection_payload = load_json(Path(args.oracle_page_cuda_vocab_projection_json), required=False)
    page_cuda_vocab_projection_summary = summary(page_cuda_vocab_projection_payload)
    page_cuda_vocab_logit_compare_payload = load_json(Path(args.oracle_page_cuda_vocab_logit_compare_json), required=False)
    page_cuda_vocab_logit_compare_summary = summary(page_cuda_vocab_logit_compare_payload)
    page_cuda_candidate_verifier_payload = load_json(Path(args.oracle_page_cuda_candidate_verifier_json), required=False)
    page_cuda_candidate_verifier_summary = summary(page_cuda_candidate_verifier_payload)
    page_cuda_live_candidate_verifier_payload = load_json(
        Path(args.oracle_page_cuda_live_candidate_verifier_json),
        required=False,
    )
    page_cuda_live_candidate_verifier_summary = summary(page_cuda_live_candidate_verifier_payload)
    page_cuda_proxy_fallback_verifier_payload = load_json(
        Path(args.oracle_page_cuda_proxy_fallback_verifier_json),
        required=False,
    )
    page_cuda_proxy_fallback_verifier_summary = summary(page_cuda_proxy_fallback_verifier_payload)
    sparse_oracle_runtime_step_payload = load_json(Path(args.sparse_oracle_runtime_step_json), required=False)
    sparse_oracle_runtime_step_summary = summary(sparse_oracle_runtime_step_payload)
    proxy_shortlist_validation_payload = load_json(Path(args.proxy_shortlist_validation_json), required=False)
    proxy_shortlist_validation_summary = summary(proxy_shortlist_validation_payload)
    proxy_shortlist_hard_payload = load_json(Path(args.proxy_shortlist_hard_json), required=False)
    proxy_shortlist_hard_summary = summary(proxy_shortlist_hard_payload)
    kv_ledger_payload = load_json(Path(args.kv_ledger_json), required=False)
    kv_ledger_summary = summary(kv_ledger_payload)
    kv_tier_smoke_payload = load_json(Path(args.kv_tier_smoke_json), required=False)
    kv_tier_smoke_summary = summary(kv_tier_smoke_payload)
    kv_tier_smoke_cuda = kv_tier_smoke_payload.get("cuda_pack", {})
    kv_tier_smoke_cuda = kv_tier_smoke_cuda if isinstance(kv_tier_smoke_cuda, dict) else {}
    kv_runtime_ledger_payload = load_json(Path(args.kv_runtime_ledger_json), required=False)
    kv_runtime_ledger_summary = summary(kv_runtime_ledger_payload)
    kv_runtime_pack_evidence = kv_runtime_ledger_payload.get("measured_pack_evidence", {})
    kv_runtime_pack_evidence = kv_runtime_pack_evidence if isinstance(kv_runtime_pack_evidence, dict) else {}
    projection_payload = load_json(Path(args.projection_json))
    projection_policy = projection_payload.get("policy", {})
    projection_policy = projection_policy if isinstance(projection_policy, dict) else {}
    format_projection_payload = load_json(Path(args.format_projection_json), required=False)
    sparse_fallback_projection_payload = load_json(Path(args.sparse_fallback_projection_json), required=False)
    overlap_budget_payload = load_json(Path(args.overlap_budget_json), required=False)
    overlap_budget_summary = summary(overlap_budget_payload)
    page_cache_budget_payload = load_json(Path(args.page_cache_budget_json), required=False)
    page_cache_budget_summary = summary(page_cache_budget_payload)
    reduced_page_cache_budget_payload = load_json(Path(args.reduced_page_cache_budget_json), required=False)
    reduced_page_cache_budget_summary = summary(reduced_page_cache_budget_payload)
    reduced_page_quality_payload = load_json(Path(args.reduced_page_quality_json), required=False)
    reduced_page_quality_summary = summary(reduced_page_quality_payload)
    signal_aware_page_cache_budget_payload = load_json(Path(args.signal_aware_page_cache_budget_json), required=False)
    signal_aware_page_cache_budget_summary = summary(signal_aware_page_cache_budget_payload)
    signal_aware_page_quality_payload = load_json(Path(args.signal_aware_page_quality_json), required=False)
    signal_aware_page_quality_summary = summary(signal_aware_page_quality_payload)
    signal_aware_cross_activation_quality_payload = load_json(
        Path(args.signal_aware_cross_activation_quality_json),
        required=False,
    )
    signal_aware_cross_activation_quality_summary = summary(signal_aware_cross_activation_quality_payload)

    gates: list[Gate] = []

    parity_rows = int_value(parity, "checked_rows")
    parity_replay_rows = int_value(parity, "replay_rows")
    parity_passed = (
        parity_rows > 0
        and parity_rows == parity_replay_rows
        and bool_value(parity, "meets_required_matches")
    )
    gates.append(
        Gate(
            stage="stage1_scheduler",
            name="cxx_policy_parity",
            passed=parity_passed,
            evidence=(
                f"{parity_rows}/{parity_replay_rows} replay rows checked; "
                f"required matches={bool_value(parity, 'meets_required_matches')}; "
                f"final false accepts={int_value(parity, 'final_false_accepts')}"
            ),
            next_step="Keep this as a regression gate for any policy change.",
        )
    )

    scheduler_rows = int_value(scheduler, "checked_rows")
    scheduler_replay_rows = int_value(scheduler, "replay_rows")
    scheduler_passed = (
        scheduler_rows > 0
        and scheduler_rows == scheduler_replay_rows
        and bool_value(scheduler, "meets_required_matches")
    )
    gates.append(
        Gate(
            stage="stage1_scheduler",
            name="cxx_scheduler_replay",
            passed=scheduler_passed,
            evidence=(
                f"{scheduler_rows}/{scheduler_replay_rows} rows; "
                f"required matches={bool_value(scheduler, 'meets_required_matches')}; "
                f"prompt streams={int_value(scheduler, 'prompt_count')}"
            ),
            next_step="Keep prefix/generated-text parity before adding paged oracle state.",
        )
    )

    proxy_tokens = int_value(proxy, "generated_tokens")
    proxy_tps = float_value(proxy, "tokens_per_sec")
    gates.append(
        Gate(
            stage="stage1_scheduler",
            name="resident_proxy_live",
            passed=proxy_tokens > 0 and proxy_tps > 0,
            evidence=f"{proxy_tokens} generated tokens at {proxy_tps:.2f} tok/s on the small proxy smoke.",
            next_step="Measure a useful proxy that can stay under roughly 40 ms/token.",
        )
    )

    dual_tokens = int_value(dual, "generated_tokens")
    dual_proxy_accepts = int_value(dual, "proxy_accepts")
    dual_oracle_fallbacks = int_value(dual, "oracle_fallbacks")
    gates.append(
        Gate(
            stage="stage1_scheduler",
            name="dual_context_live_sync",
            passed=dual_tokens > 0 and dual_proxy_accepts > 0 and dual_oracle_fallbacks > 0,
            evidence=(
                f"{dual_tokens} tokens; proxy accepts={dual_proxy_accepts}; "
                f"oracle fallbacks={dual_oracle_fallbacks}; final text={dual.get('final_text', '')!r}"
            ),
            next_step="Replace the same-model smoke oracle with the sparse sentinel/oracle path.",
        )
    )

    required_ledger_fields = {
        "schema",
        "oracle_mode",
        "oracle_active_bytes",
        "verifier_active_bytes",
        "proxy_kv_tokens",
        "oracle_kv_tokens",
        "kv_byte_status",
        "total_step_ms",
    }
    ledgers = [
        step.get("ledger")
        for step in dual_steps
        if isinstance(step, dict) and isinstance(step.get("ledger"), dict)
    ]
    ledger_fields_present = all(required_ledger_fields.issubset(set(ledger)) for ledger in ledgers)
    ledger_schemas_ok = all(ledger.get("schema") == "sage-active-byte-ledger-v0" for ledger in ledgers)
    accept_zero_oracle = any(
        isinstance(step, dict)
        and step.get("action") == "accept_proxy"
        and isinstance(step.get("ledger"), dict)
        and int_value(step["ledger"], "oracle_active_bytes") == 0
        for step in dual_steps
    )
    fallback_exact_oracle = any(
        isinstance(step, dict)
        and step.get("action") == "oracle_fallback"
        and isinstance(step.get("ledger"), dict)
        and step["ledger"].get("oracle_mode") == "exact_resident_context"
        and int_value(step["ledger"], "oracle_active_bytes") > 0
        for step in dual_steps
    )
    ledger_passed = (
        dual_tokens > 0
        and len(ledgers) >= dual_tokens
        and dual.get("ledger_schema") == "sage-active-byte-ledger-v0"
        and ledger_fields_present
        and ledger_schemas_ok
        and accept_zero_oracle
        and fallback_exact_oracle
    )
    gates.append(
        Gate(
            stage="stage1_scheduler",
            name="active_byte_ledger_trace",
            passed=ledger_passed,
            evidence=(
                f"{len(ledgers)}/{dual_tokens} generated steps have ledgers; "
                f"accept zero-oracle={accept_zero_oracle}; "
                f"fallback exact-oracle={fallback_exact_oracle}; "
                f"schema={dual.get('ledger_schema', '')}"
            ),
            next_step="Replace exact-resident oracle byte accounting with sparse GGUF block pages.",
        )
    )

    policy_modeled_passed = (
        bool_value(policy, "meets_all_gates")
        and float_value(policy, "effective_tps") >= args.target_tps
        and float_value(policy, "final_total_error_rate") <= args.max_total_error
        and float_value(policy, "final_accepted_error_rate") <= args.max_accepted_error
    )
    gates.append(
        Gate(
            stage="stage2_policy",
            name="modeled_sparse_verifier_policy",
            passed=policy_modeled_passed,
            evidence=(
                f"modeled {float_value(policy, 'effective_tps'):.2f} tok/s; "
                f"total error={float_value(policy, 'final_total_error_rate'):.2%}; "
                f"accepted error={float_value(policy, 'final_accepted_error_rate'):.2%}; "
                f"verifier coverage={float_value(policy, 'verifier_coverage_rate'):.2%}"
            ),
            next_step="Turn the verifier statistic into an in-process hot-path call, not a sidecar measurement.",
        )
    )

    measured_proxy_scenario = scenario_by_name(projection_payload, "persistent_active_byte_measured_proxy")
    measured_proxy_tps = float_value(measured_proxy_scenario, "effective_tps")
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="measured_proxy_runtime_budget",
            passed=measured_proxy_tps >= args.target_tps,
            evidence=(
                f"persistent active-byte model with measured proxy speed gives "
                f"{measured_proxy_tps:.2f} tok/s."
            ),
            next_step="Proxy path must get faster or the oracle active percent must drop below the configured model.",
        )
    )

    format_hard_measured = scenario_by_name(format_projection_payload, "format_scaffold_hard120_measured_proxy")
    format_hard_limit_7 = fallback_limit_by_name_target(
        format_projection_payload,
        "format_scaffold_hard120_measured_proxy",
        args.target_tps,
    )
    format_projection_model = format_projection_payload.get("format_scaffold_model", {})
    format_projection_model = format_projection_model if isinstance(format_projection_model, dict) else {}
    format_projection_passed = (
        float_value(format_hard_measured, "effective_tps") >= args.target_tps
        and bool_value(format_hard_measured, "meets_target_tps")
        and bool_value(format_hard_limit_7, "meets_configured_fallback_ms")
        and float_value(format_hard_limit_7, "configured_fallback_ms") > 0
        and float_value(format_hard_limit_7, "max_fallback_ms") >= float_value(format_hard_limit_7, "configured_fallback_ms")
        and bool_value(format_projection_model, "include_candidate_host_read")
    )
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="format_scaffold_runtime_budget_projection",
            passed=format_projection_passed,
            evidence=(
                f"hard120 measured-proxy projection gives "
                f"{float_value(format_hard_measured, 'effective_tps'):.2f} tok/s; "
                f"fallback rate={float_value(format_hard_limit_7, 'fallback_rate'):.2%}; "
                f"configured fallback={float_value(format_hard_limit_7, 'configured_fallback_ms'):.1f} ms; "
                f"max fallback for {args.target_tps:.0f} tok/s="
                f"{float_value(format_hard_limit_7, 'max_fallback_ms'):.1f} ms"
            ),
            next_step=(
                "Turn this projection into a persistent runtime measurement with the proxy, "
                "Q6_K candidate verifier, and exact fallback all resident or warm."
            ),
        )
    )

    sparse_fallback_scenario = scenario_by_name(
        sparse_fallback_projection_payload,
        "format_scaffold_hard120_measured_proxy_measured_sparse_replay",
    )
    sparse_fallback_limit = fallback_limit_by_name_target(
        sparse_fallback_projection_payload,
        "format_scaffold_hard120_measured_proxy_measured_sparse_replay",
        args.target_tps,
    )
    sparse_fallback_model = sparse_fallback_projection_payload.get("sparse_fallback_model", {})
    sparse_fallback_summary = sparse_fallback_model.get("summary", {}) if isinstance(sparse_fallback_model, dict) else {}
    sparse_fallback_projection_passed = (
        float_value(sparse_fallback_scenario, "effective_tps") >= args.target_tps
        and bool_value(sparse_fallback_scenario, "meets_target_tps")
        and not bool_value(sparse_fallback_scenario, "meets_upper_target_tps")
        and float_value(sparse_fallback_scenario, "oracle_call_ms") > 0
        and float_value(sparse_fallback_model, "component_measured_ms") > 0
        and bool_value(sparse_fallback_limit, "meets_configured_fallback_ms")
        and isinstance(sparse_fallback_summary, dict)
        and bool_value(sparse_fallback_summary, "component_replay_complete")
        and not bool_value(sparse_fallback_summary, "transformer_integrated", True)
        and bool_value(sparse_fallback_summary, "exact_fallback_required")
    )
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="format_scaffold_measured_sparse_fallback_projection",
            passed=sparse_fallback_projection_passed,
            evidence=(
                f"hard120 measured-proxy with measured sparse fallback gives "
                f"{float_value(sparse_fallback_scenario, 'effective_tps'):.2f} tok/s; "
                f"fallback rate={float_value(sparse_fallback_scenario, 'oracle_call_rate'):.2%}; "
                f"fallback_ms={float_value(sparse_fallback_scenario, 'oracle_call_ms'):.1f}; "
                f"max fallback for {args.target_tps:.0f} tok/s="
                f"{float_value(sparse_fallback_limit, 'max_fallback_ms'):.1f} ms; "
                f"transformer_integrated={bool_value(sparse_fallback_summary, 'transformer_integrated', True)}"
            ),
            next_step=(
                "Turn the measured sparse fallback replay into a live sparse transformer step; "
                "this projection has budget room for 7 tok/s but not 10 tok/s."
            ),
        )
    )

    overlap_budget_passed = (
        overlap_budget_payload.get("schema") == "sage-overlap-budget-v0"
        and overlap_budget_payload.get("status") == "measured_overlap_target_not_runtime_integrated"
        and bool_value(overlap_budget_summary, "overlap_target_passed")
        and not bool_value(overlap_budget_summary, "runtime_integrated", True)
        and not bool_value(overlap_budget_summary, "transformer_integrated", True)
        and float_value(overlap_budget_summary, "full_replay_tps") >= args.target_tps
        and float_value(overlap_budget_summary, "full_replay_tps") < float_value(overlap_budget_summary, "target_tps")
        and float_value(overlap_budget_summary, "gpu_only_tps") < float_value(overlap_budget_summary, "target_tps")
        and float_value(overlap_budget_summary, "required_gpu_hidden_ms_per_fallback") > 0
        and float_value(overlap_budget_summary, "required_gpu_hidden_ms_per_fallback")
        <= float_value(overlap_budget_summary, "measured_pcie_plus_kernel_ms")
        and float_value(overlap_budget_summary, "required_proxy_reduction_ms_per_token")
        <= args.max_overlap_proxy_reduction_ms
        and float_value(overlap_budget_summary, "required_fallback_rate_reduction_percentage_points") > 0
    )
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="overlap_prefetch_10tps_budget_target",
            passed=overlap_budget_passed,
            evidence=(
                f"full={float_value(overlap_budget_summary, 'full_replay_tps'):.2f} tok/s; "
                f"host-hidden={float_value(overlap_budget_summary, 'gpu_only_tps'):.2f} tok/s; "
                f"target={float_value(overlap_budget_summary, 'target_tps'):.1f}; "
                f"hide_gpu={float_value(overlap_budget_summary, 'required_gpu_hidden_ms_per_fallback'):.1f} ms/fallback; "
                f"proxy_delta={float_value(overlap_budget_summary, 'required_proxy_reduction_ms_per_token'):.2f} ms/token; "
                f"fallback_delta={float_value(overlap_budget_summary, 'required_fallback_rate_reduction_percentage_points'):.2f} pp"
            ),
            next_step=(
                "Implement async page prefetch/CUDA overlap or reduce proxy/fallback latency by this measured amount "
                "before claiming the 10 tok/s target."
            ),
        )
    )

    page_cache_budget_passed = (
        page_cache_budget_payload.get("schema") == "sage-page-cache-budget-v0"
        and page_cache_budget_payload.get("status") == "measured_page_cache_budget_target_not_transformer_integrated"
        and bool_value(page_cache_budget_summary, "budget_target_passed")
        and not bool_value(page_cache_budget_summary, "runtime_integrated", True)
        and not bool_value(page_cache_budget_summary, "transformer_integrated", True)
        and float_value(page_cache_budget_summary, "effective_tps_with_page_cache") >= args.target_tps
        and float_value(page_cache_budget_summary, "effective_tps_with_page_cache")
        < float_value(page_cache_budget_summary, "target_tps")
        and float_value(page_cache_budget_summary, "measured_page_cache_replay_ms") > 0
        and float_value(page_cache_budget_summary, "measured_page_cache_replay_gib") > 0
        and float_value(page_cache_budget_summary, "max_active_gib_for_target") > 0
        and float_value(page_cache_budget_summary, "max_active_gib_for_target")
        < float_value(page_cache_budget_summary, "measured_page_cache_replay_gib")
        and float_value(page_cache_budget_summary, "required_active_reduction_percent") > 0
        and float_value(page_cache_budget_summary, "required_proxy_reduction_ms_per_token")
        <= args.max_overlap_proxy_reduction_ms
        and float_value(page_cache_budget_summary, "required_fallback_rate_reduction_percentage_points") > 0
    )
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="resident_page_cache_10tps_budget_target",
            passed=page_cache_budget_passed,
            evidence=(
                f"cache={float_value(page_cache_budget_summary, 'effective_tps_with_page_cache'):.2f} tok/s; "
                f"target={float_value(page_cache_budget_summary, 'target_tps'):.1f}; "
                f"replay={float_value(page_cache_budget_summary, 'measured_page_cache_replay_ms'):.1f} ms/"
                f"{float_value(page_cache_budget_summary, 'measured_page_cache_replay_gib'):.3f} GiB; "
                f"max_active={float_value(page_cache_budget_summary, 'max_active_gib_for_target'):.3f} GiB; "
                f"active_delta={float_value(page_cache_budget_summary, 'required_active_reduction_percent'):.1f}%; "
                f"fallback_delta={float_value(page_cache_budget_summary, 'required_fallback_rate_reduction_percentage_points'):.2f} pp"
            ),
            next_step=(
                "Reduce the resident active page set toward the measured max-active GiB target or lower fallback rate "
                "before claiming 10 tok/s."
            ),
        )
    )

    reduced_page_cache_budget_passed = (
        reduced_page_cache_budget_payload.get("schema") == "sage-page-cache-budget-v0"
        and reduced_page_cache_budget_payload.get("status")
        == "measured_page_cache_budget_target_not_transformer_integrated"
        and bool_value(reduced_page_cache_budget_summary, "budget_target_passed")
        and bool_value(reduced_page_cache_budget_summary, "measured_plan_meets_target_tps")
        and not bool_value(reduced_page_cache_budget_summary, "runtime_integrated", True)
        and not bool_value(reduced_page_cache_budget_summary, "transformer_integrated", True)
        and float_value(reduced_page_cache_budget_summary, "effective_tps_with_page_cache")
        >= float_value(reduced_page_cache_budget_summary, "target_tps")
        and float_value(reduced_page_cache_budget_summary, "measured_page_cache_replay_ms") > 0
        and float_value(reduced_page_cache_budget_summary, "measured_page_cache_replay_gib") > 0
        and float_value(reduced_page_cache_budget_summary, "measured_page_cache_replay_gib")
        <= float_value(reduced_page_cache_budget_summary, "max_active_gib_for_target")
        and float_value(reduced_page_cache_budget_summary, "current_active_percent_of_reference_100b")
        <= float_value(reduced_page_cache_budget_summary, "max_active_percent_of_reference_100b_for_target")
    )
    gates.append(
        Gate(
            stage="stage2_runtime",
            name="reduced_resident_page_cache_10tps_projection",
            passed=reduced_page_cache_budget_passed,
            evidence=(
                f"cache={float_value(reduced_page_cache_budget_summary, 'effective_tps_with_page_cache'):.2f} tok/s; "
                f"target={float_value(reduced_page_cache_budget_summary, 'target_tps'):.1f}; "
                f"replay={float_value(reduced_page_cache_budget_summary, 'measured_page_cache_replay_ms'):.1f} ms/"
                f"{float_value(reduced_page_cache_budget_summary, 'measured_page_cache_replay_gib'):.3f} GiB; "
                f"active_ref={float_value(reduced_page_cache_budget_summary, 'current_active_percent_of_reference_100b'):.2f}%; "
                f"max_ref={float_value(reduced_page_cache_budget_summary, 'max_active_percent_of_reference_100b_for_target'):.2f}%"
            ),
            next_step=(
                "Turn this reduced resident-cache projection into sparse dequant/matmul execution with quality checks."
            ),
        )
    )

    configured_target = limit_by_name(projection_payload, "configured_proxy_target")
    configured_upper = limit_by_name(projection_payload, "configured_proxy_upper_target")
    target_active = float_value(configured_target, "max_oracle_active_percent")
    upper_active = float_value(configured_upper, "max_oracle_active_percent")
    active_budget_passed = target_active >= args.oracle_active_percent_7tps and upper_active >= args.oracle_active_percent_10tps
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="active_byte_budget_window",
            passed=active_budget_passed,
            evidence=(
                f"configured proxy permits {target_active:.2f}% active oracle at 7 tok/s "
                f"and {upper_active:.2f}% at 10 tok/s."
            ),
            next_step="Implement a GGUF block-paged oracle that proves those active percentages as measured bytes.",
        )
    )

    page_ledger_pages = page_ledger_payload.get("pages", [])
    page_ledger_stages = page_ledger_payload.get("stages", [])
    page_ledger_budget = page_ledger_payload.get("budget", {})
    page_ledger_template = page_ledger_payload.get("runtime_ledger_template", {})
    page_ledger_active_ref = float_value(page_ledger_summary, "active_percent_of_reference_100b")
    page_ledger_passed = (
        page_ledger_payload.get("schema") == "sage-oracle-page-ledger-v0"
        and page_ledger_payload.get("status") == "plan_only_not_executed"
        and isinstance(page_ledger_pages, list)
        and len(page_ledger_pages) > 0
        and isinstance(page_ledger_stages, list)
        and len(page_ledger_stages) > 0
        and isinstance(page_ledger_budget, dict)
        and str(page_ledger_budget.get("status", "")).startswith("within_")
        and isinstance(page_ledger_template, dict)
        and page_ledger_template.get("oracle_mode") == "sparse_page_plan"
        and page_ledger_active_ref > 0
        and page_ledger_active_ref <= args.oracle_active_percent_7tps
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_page_ledger_plan",
            passed=page_ledger_passed,
            evidence=(
                f"schema={page_ledger_payload.get('schema', '')}; "
                f"pages={len(page_ledger_pages) if isinstance(page_ledger_pages, list) else 0}; "
                f"stages={len(page_ledger_stages) if isinstance(page_ledger_stages, list) else 0}; "
                f"active reference={page_ledger_active_ref:.2f}%; "
                f"status={page_ledger_budget.get('status', '') if isinstance(page_ledger_budget, dict) else ''}"
            ),
            next_step="Make the page plan executable with pinned host pages and GPU staging buffers.",
        )
    )

    page_staging_limits = page_staging_payload.get("limits", {})
    page_staging_evidence = page_staging_payload.get("runtime_ledger_evidence", {})
    staging_status = page_staging_payload.get("status", "")
    page_staging_passed = (
        page_staging_payload.get("schema") == "sage-oracle-page-staging-v0"
        and staging_status == "measured_host_staging_not_cuda"
        and int_value(page_staging_summary, "stages_staged") > 0
        and int_value(page_staging_summary, "pages_staged") > 0
        and int_value(page_staging_summary, "staged_bytes") > 0
        and bool_value(page_staging_summary, "stage_byte_match")
        and bool_value(page_staging_summary, "page_byte_match")
        and bool_value(page_staging_summary, "byte_budget_respected")
        and page_staging_summary.get("cuda_execution_status") == "not_implemented"
        and isinstance(page_staging_limits, dict)
        and int_value(page_staging_limits, "stage_buffer_bytes") > 0
        and isinstance(page_staging_evidence, dict)
        and page_staging_evidence.get("oracle_mode") == "sparse_page_staging_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_page_staging_smoke",
            passed=page_staging_passed,
            evidence=(
                f"status={staging_status}; "
                f"staged={float_value(page_staging_summary, 'staged_gib'):.3f} GiB; "
                f"stages={int_value(page_staging_summary, 'stages_staged')}; "
                f"pages={int_value(page_staging_summary, 'pages_staged')}; "
                f"max live={float_value(page_staging_summary, 'max_live_buffer_gib'):.3f} GiB; "
                f"staging throughput={float_value(page_staging_summary, 'staging_throughput_gib_s'):.2f} GiB/s"
            ),
            next_step="Replace CPU file-to-host staging with pinned host pages, CUDA H2D streams, and sparse compute.",
        )
    )

    cuda_staging_evidence = page_cuda_staging_payload.get("runtime_ledger_evidence", {})
    cuda_staging_status = page_cuda_staging_payload.get("status", "")
    page_cuda_staging_passed = (
        page_cuda_staging_payload.get("schema") == "sage-oracle-page-cuda-staging-v0"
        and cuda_staging_status == "measured_cuda_h2d_not_sparse_compute"
        and int_value(page_cuda_staging_summary, "stages_staged") > 0
        and int_value(page_cuda_staging_summary, "pages_staged") > 0
        and int_value(page_cuda_staging_summary, "h2d_bytes") > 0
        and float_value(page_cuda_staging_summary, "h2d_ms") > 0
        and bool_value(page_cuda_staging_summary, "stage_byte_match")
        and bool_value(page_cuda_staging_summary, "page_byte_match")
        and bool_value(page_cuda_staging_summary, "byte_budget_respected")
        and page_cuda_staging_summary.get("pcie_transfer_status") == "measured_cuda_h2d_from_pinned_host"
        and page_cuda_staging_summary.get("cuda_execution_status") == "not_implemented_sparse_compute"
        and isinstance(cuda_staging_evidence, dict)
        and cuda_staging_evidence.get("oracle_mode") == "sparse_page_cuda_staging_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_staging_smoke",
            passed=page_cuda_staging_passed,
            evidence=(
                f"status={cuda_staging_status}; "
                f"staged={float_value(page_cuda_staging_summary, 'staged_gib'):.3f} GiB; "
                f"stages={int_value(page_cuda_staging_summary, 'stages_staged')}; "
                f"pages={int_value(page_cuda_staging_summary, 'pages_staged')}; "
                f"h2d={float_value(page_cuda_staging_summary, 'h2d_ms'):.2f} ms; "
                f"h2d throughput={float_value(page_cuda_staging_summary, 'h2d_throughput_gib_s'):.2f} GiB/s"
            ),
            next_step="Add CUDA stream overlap with compute and sparse page kernels before claiming executable oracle inference.",
        )
    )

    cuda_kernel_evidence = page_cuda_kernel_payload.get("runtime_ledger_evidence", {})
    cuda_kernel_status = page_cuda_kernel_payload.get("status", "")
    page_cuda_kernel_passed = (
        page_cuda_kernel_payload.get("schema") == "sage-oracle-page-cuda-kernel-smoke-v0"
        and cuda_kernel_status == "measured_cuda_kernel_touch_not_transformer"
        and int_value(page_cuda_kernel_summary, "stages_staged") > 0
        and int_value(page_cuda_kernel_summary, "staged_bytes") > 0
        and float_value(page_cuda_kernel_summary, "kernel_ms") > 0
        and bool_value(page_cuda_kernel_summary, "stage_byte_match")
        and bool_value(page_cuda_kernel_summary, "byte_budget_respected")
        and bool_value(page_cuda_kernel_summary, "kernel_output_nonzero")
        and page_cuda_kernel_summary.get("cuda_kernel_status") == "measured_byte_sum_touch_kernel"
        and page_cuda_kernel_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(cuda_kernel_evidence, dict)
        and cuda_kernel_evidence.get("oracle_mode") == "sparse_page_cuda_kernel_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_kernel_smoke",
            passed=page_cuda_kernel_passed,
            evidence=(
                f"status={cuda_kernel_status}; "
                f"touched={float_value(page_cuda_kernel_summary, 'staged_gib'):.3f} GiB; "
                f"stages={int_value(page_cuda_kernel_summary, 'stages_staged')}; "
                f"h2d={float_value(page_cuda_kernel_summary, 'h2d_ms'):.2f} ms; "
                f"kernel={float_value(page_cuda_kernel_summary, 'kernel_ms'):.2f} ms; "
                f"touch throughput={float_value(page_cuda_kernel_summary, 'kernel_touch_throughput_gib_s'):.2f} GiB/s"
            ),
            next_step="Replace the byte-touch kernel with sparse dequant/matmul kernels tied to candidate scoring.",
        )
    )

    cuda_overlap_evidence = page_cuda_overlap_payload.get("runtime_ledger_evidence", {})
    cuda_overlap_status = page_cuda_overlap_payload.get("status", "")
    page_cuda_overlap_passed = (
        page_cuda_overlap_payload.get("schema") == "sage-oracle-page-cuda-overlap-smoke-v0"
        and cuda_overlap_status == "measured_cuda_double_buffer_overlap_touch_not_transformer"
        and int_value(page_cuda_overlap_summary, "stages_staged") >= 2
        and int_value(page_cuda_overlap_summary, "staged_bytes") > 0
        and int_value(page_cuda_overlap_summary, "staged_bytes") == int_value(page_cuda_kernel_summary, "staged_bytes")
        and float_value(page_cuda_overlap_summary, "h2d_ms") > 0
        and float_value(page_cuda_overlap_summary, "kernel_ms") > 0
        and float_value(page_cuda_overlap_summary, "sequential_gpu_ms") > 0
        and float_value(page_cuda_overlap_summary, "overlapped_gpu_ms") > 0
        and float_value(page_cuda_overlap_summary, "overlap_savings_ms") >= 0
        and bool_value(page_cuda_overlap_summary, "stage_byte_match")
        and bool_value(page_cuda_overlap_summary, "byte_budget_respected")
        and bool_value(page_cuda_overlap_summary, "kernel_output_nonzero")
        and page_cuda_overlap_summary.get("cuda_overlap_status") == "measured_two_stream_double_buffer_cuda_events"
        and page_cuda_overlap_summary.get("cuda_kernel_status") == "measured_byte_sum_touch_kernel"
        and page_cuda_overlap_summary.get("host_read_overlap_status") == "not_measured_prestaged_pinned_host_buffers"
        and page_cuda_overlap_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(cuda_overlap_evidence, dict)
        and cuda_overlap_evidence.get("oracle_mode") == "sparse_page_cuda_overlap_smoke"
        and bool_value(cuda_overlap_evidence, "gpu_h2d_kernel_overlap_measured")
        and not bool_value(cuda_overlap_evidence, "transformer_layer_math")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_overlap_smoke",
            passed=page_cuda_overlap_passed,
            evidence=(
                f"status={cuda_overlap_status}; "
                f"touched={float_value(page_cuda_overlap_summary, 'staged_gib'):.3f} GiB; "
                f"stages={int_value(page_cuda_overlap_summary, 'stages_staged')}; "
                f"sequential_gpu={float_value(page_cuda_overlap_summary, 'sequential_gpu_ms'):.2f} ms; "
                f"overlapped_gpu={float_value(page_cuda_overlap_summary, 'overlapped_gpu_ms'):.2f} ms; "
                f"savings={float_value(page_cuda_overlap_summary, 'overlap_savings_ms'):.2f} ms "
                f"({float_value(page_cuda_overlap_summary, 'overlap_savings_pct'):.1f}%)"
            ),
            next_step="Move from byte-touch overlap to sparse dequant/matmul overlap and background host-page prefetch.",
        )
    )

    cuda_prefetch_overlap_evidence = page_cuda_prefetch_overlap_payload.get("runtime_ledger_evidence", {})
    cuda_prefetch_overlap_status = page_cuda_prefetch_overlap_payload.get("status", "")
    page_cuda_prefetch_overlap_passed = (
        page_cuda_prefetch_overlap_payload.get("schema") == "sage-oracle-page-cuda-prefetch-overlap-smoke-v0"
        and cuda_prefetch_overlap_status == "measured_host_prefetch_cuda_overlap_touch_not_transformer"
        and int_value(page_cuda_prefetch_overlap_summary, "stages_staged") >= 2
        and int_value(page_cuda_prefetch_overlap_summary, "staged_bytes") > 0
        and int_value(page_cuda_prefetch_overlap_summary, "staged_bytes") == int_value(page_cuda_kernel_summary, "staged_bytes")
        and float_value(page_cuda_prefetch_overlap_summary, "host_read_ms") > 0
        and float_value(page_cuda_prefetch_overlap_summary, "h2d_ms") > 0
        and float_value(page_cuda_prefetch_overlap_summary, "kernel_ms") > 0
        and float_value(page_cuda_prefetch_overlap_summary, "sequential_total_ms") > 0
        and float_value(page_cuda_prefetch_overlap_summary, "pipeline_wall_ms") > 0
        and float_value(page_cuda_prefetch_overlap_summary, "pipeline_savings_ms") >= 0
        and bool_value(page_cuda_prefetch_overlap_summary, "stage_byte_match")
        and bool_value(page_cuda_prefetch_overlap_summary, "byte_budget_respected")
        and bool_value(page_cuda_prefetch_overlap_summary, "kernel_output_nonzero")
        and page_cuda_prefetch_overlap_summary.get("host_read_overlap_status")
        == "measured_single_worker_background_prefetch"
        and page_cuda_prefetch_overlap_summary.get("cuda_overlap_status")
        == "measured_two_stream_double_buffer_cuda_events"
        and page_cuda_prefetch_overlap_summary.get("cuda_kernel_status") == "measured_byte_sum_touch_kernel"
        and page_cuda_prefetch_overlap_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(cuda_prefetch_overlap_evidence, dict)
        and cuda_prefetch_overlap_evidence.get("oracle_mode") == "sparse_page_cuda_prefetch_overlap_smoke"
        and bool_value(cuda_prefetch_overlap_evidence, "host_read_overlap_measured")
        and bool_value(cuda_prefetch_overlap_evidence, "gpu_h2d_kernel_overlap_measured")
        and not bool_value(cuda_prefetch_overlap_evidence, "transformer_layer_math")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_prefetch_overlap_smoke",
            passed=page_cuda_prefetch_overlap_passed,
            evidence=(
                f"status={cuda_prefetch_overlap_status}; "
                f"touched={float_value(page_cuda_prefetch_overlap_summary, 'staged_gib'):.3f} GiB; "
                f"stages={int_value(page_cuda_prefetch_overlap_summary, 'stages_staged')}; "
                f"host_read={float_value(page_cuda_prefetch_overlap_summary, 'host_read_ms'):.2f} ms; "
                f"sequential_total={float_value(page_cuda_prefetch_overlap_summary, 'sequential_total_ms'):.2f} ms; "
                f"pipeline_wall={float_value(page_cuda_prefetch_overlap_summary, 'pipeline_wall_ms'):.2f} ms; "
                f"savings={float_value(page_cuda_prefetch_overlap_summary, 'pipeline_savings_ms'):.2f} ms "
                f"({float_value(page_cuda_prefetch_overlap_summary, 'pipeline_savings_pct'):.1f}%)"
            ),
            next_step=(
                "Replace byte-touch work with sparse dequant/matmul kernels and replace ad hoc GGUF reads "
                "with a resident pinned page cache."
            ),
        )
    )

    cuda_page_cache_evidence = page_cuda_page_cache_payload.get("runtime_ledger_evidence", {})
    cuda_page_cache_status = page_cuda_page_cache_payload.get("status", "")
    page_cuda_page_cache_passed = (
        page_cuda_page_cache_payload.get("schema") == "sage-oracle-page-cuda-page-cache-smoke-v0"
        and cuda_page_cache_status == "measured_resident_pinned_page_cache_replay_touch_not_transformer"
        and int_value(page_cuda_page_cache_summary, "stages_cached") >= 2
        and int_value(page_cuda_page_cache_summary, "replays") >= 2
        and int_value(page_cuda_page_cache_summary, "staged_bytes_per_replay") > 0
        and int_value(page_cuda_page_cache_summary, "staged_bytes_per_replay")
        == int_value(page_cuda_kernel_summary, "staged_bytes")
        and float_value(page_cuda_page_cache_summary, "cache_build_ms") > 0
        and float_value(page_cuda_page_cache_summary, "per_replay_gpu_ms") > 0
        and float_value(page_cuda_page_cache_summary, "per_replay_h2d_ms") > 0
        and float_value(page_cuda_page_cache_summary, "per_replay_kernel_ms") > 0
        and int_value(page_cuda_page_cache_summary, "cache_hits") > 0
        and int_value(page_cuda_page_cache_summary, "cache_hit_bytes") > 0
        and bool_value(page_cuda_page_cache_summary, "cache_replay_saves_host_read")
        and bool_value(page_cuda_page_cache_summary, "stage_byte_match")
        and bool_value(page_cuda_page_cache_summary, "byte_budget_respected")
        and bool_value(page_cuda_page_cache_summary, "kernel_output_nonzero")
        and page_cuda_page_cache_summary.get("cache_status") == "measured_resident_pinned_host_page_cache"
        and page_cuda_page_cache_summary.get("pcie_transfer_status")
        == "measured_cuda_h2d_from_resident_pinned_page_cache"
        and page_cuda_page_cache_summary.get("cuda_overlap_status") == "measured_two_stream_double_buffer_cuda_events"
        and page_cuda_page_cache_summary.get("cuda_kernel_status") == "measured_byte_sum_touch_kernel"
        and page_cuda_page_cache_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(cuda_page_cache_evidence, dict)
        and cuda_page_cache_evidence.get("oracle_mode") == "sparse_page_cuda_page_cache_smoke"
        and bool_value(cuda_page_cache_evidence, "resident_pinned_page_cache")
        and bool_value(cuda_page_cache_evidence, "cache_replay_measured")
        and bool_value(cuda_page_cache_evidence, "gpu_h2d_kernel_overlap_measured")
        and not bool_value(cuda_page_cache_evidence, "transformer_layer_math")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_page_cache_smoke",
            passed=page_cuda_page_cache_passed,
            evidence=(
                f"status={cuda_page_cache_status}; "
                f"cached={float_value(page_cuda_page_cache_summary, 'cache_gib'):.3f} GiB; "
                f"replays={int_value(page_cuda_page_cache_summary, 'replays')}; "
                f"cache_build={float_value(page_cuda_page_cache_summary, 'cache_build_ms'):.2f} ms; "
                f"per_replay_gpu={float_value(page_cuda_page_cache_summary, 'per_replay_gpu_ms'):.2f} ms; "
                f"cache_hits={int_value(page_cuda_page_cache_summary, 'cache_hits')}; "
                f"hit_bytes={float_value(page_cuda_page_cache_summary, 'cache_hit_gib'):.3f} GiB"
            ),
            next_step=(
                "Turn the resident pinned page cache into a real sparse oracle cache with dequant/matmul kernels "
                "and eviction policy."
            ),
        )
    )

    cuda_dequant_evidence = page_cuda_dequant_payload.get("runtime_ledger_evidence", {})
    cuda_dequant_status = page_cuda_dequant_payload.get("status", "")
    page_cuda_dequant_passed = (
        page_cuda_dequant_payload.get("schema") == "sage-oracle-page-cuda-dequant-smoke-v0"
        and cuda_dequant_status == "measured_cuda_q4_0_dequant_not_matmul"
        and int_value(page_cuda_dequant_summary, "stages_staged") > 0
        and int_value(page_cuda_dequant_summary, "q4_0_tensors") > 0
        and int_value(page_cuda_dequant_summary, "q4_0_bytes") > 0
        and int_value(page_cuda_dequant_summary, "q4_0_values") > 0
        and float_value(page_cuda_dequant_summary, "dequant_ms") > 0
        and bool_value(page_cuda_dequant_summary, "dequant_output_nonzero")
        and bool_value(page_cuda_dequant_summary, "byte_budget_respected")
        and page_cuda_dequant_summary.get("cuda_kernel_status") == "measured_q4_0_dequant_reduce_kernel"
        and page_cuda_dequant_summary.get("sparse_matmul_status") == "not_implemented"
        and isinstance(cuda_dequant_evidence, dict)
        and cuda_dequant_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_dequant_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_dequant_smoke",
            passed=page_cuda_dequant_passed,
            evidence=(
                f"status={cuda_dequant_status}; "
                f"q4_0={float_value(page_cuda_dequant_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(page_cuda_dequant_summary, 'q4_0_tensors')}; "
                f"values={int_value(page_cuda_dequant_summary, 'q4_0_values')}; "
                f"h2d={float_value(page_cuda_dequant_summary, 'h2d_ms'):.2f} ms; "
                f"dequant={float_value(page_cuda_dequant_summary, 'dequant_ms'):.2f} ms"
            ),
            next_step="Replace dequant reduction with sparse Q4_0 matmul/candidate scoring kernels.",
        )
    )

    cuda_matvec_evidence = page_cuda_matvec_payload.get("runtime_ledger_evidence", {})
    cuda_matvec_status = page_cuda_matvec_payload.get("status", "")
    page_cuda_matvec_passed = (
        page_cuda_matvec_payload.get("schema") == "sage-oracle-page-cuda-matvec-smoke-v0"
        and cuda_matvec_status == "measured_cuda_q4_0_matvec_synthetic_activation_not_transformer"
        and int_value(page_cuda_matvec_summary, "stages_staged") > 0
        and int_value(page_cuda_matvec_summary, "q4_0_tensors") > 0
        and int_value(page_cuda_matvec_summary, "q4_0_bytes") > 0
        and int_value(page_cuda_matvec_summary, "q4_0_values") > 0
        and int_value(page_cuda_matvec_summary, "output_scores") > 0
        and float_value(page_cuda_matvec_summary, "matvec_ms") > 0
        and bool_value(page_cuda_matvec_summary, "score_output_nonzero")
        and bool_value(page_cuda_matvec_summary, "byte_budget_respected")
        and page_cuda_matvec_summary.get("cuda_kernel_status") == "measured_q4_0_matvec_synthetic_activation_kernel"
        and page_cuda_matvec_summary.get("candidate_scoring_status") == "not_implemented"
        and isinstance(cuda_matvec_evidence, dict)
        and cuda_matvec_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_matvec_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_matvec_smoke",
            passed=page_cuda_matvec_passed,
            evidence=(
                f"status={cuda_matvec_status}; "
                f"q4_0={float_value(page_cuda_matvec_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(page_cuda_matvec_summary, 'q4_0_tensors')}; "
                f"scores={int_value(page_cuda_matvec_summary, 'output_scores')}; "
                f"h2d={float_value(page_cuda_matvec_summary, 'h2d_ms'):.2f} ms; "
                f"matvec={float_value(page_cuda_matvec_summary, 'matvec_ms'):.2f} ms"
            ),
            next_step="Replace synthetic activations with live hidden states and compare sparse scores to exact oracle logits.",
        )
    )

    reduced_cuda_dequant_evidence = reduced_page_cuda_dequant_payload.get("runtime_ledger_evidence", {})
    reduced_cuda_dequant_status = reduced_page_cuda_dequant_payload.get("status", "")
    reduced_page_cuda_dequant_passed = (
        reduced_page_cuda_dequant_payload.get("schema") == "sage-oracle-page-cuda-dequant-smoke-v0"
        and reduced_cuda_dequant_status == "measured_cuda_q4_0_dequant_not_matmul"
        and int_value(reduced_page_cuda_dequant_summary, "stages_staged") > 0
        and int_value(reduced_page_cuda_dequant_summary, "q4_0_tensors") > 0
        and int_value(reduced_page_cuda_dequant_summary, "q4_0_bytes") > 0
        and float_value(reduced_page_cuda_dequant_summary, "q4_0_gib") <= 1.18
        and float_value(reduced_page_cuda_dequant_summary, "max_live_buffer_gib") <= 0.50
        and float_value(reduced_page_cuda_dequant_summary, "h2d_ms") > 0
        and float_value(reduced_page_cuda_dequant_summary, "dequant_ms") > 0
        and bool_value(reduced_page_cuda_dequant_summary, "dequant_output_nonzero")
        and bool_value(reduced_page_cuda_dequant_summary, "byte_budget_respected")
        and reduced_page_cuda_dequant_summary.get("cuda_kernel_status") == "measured_q4_0_dequant_reduce_kernel"
        and reduced_page_cuda_dequant_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(reduced_cuda_dequant_evidence, dict)
        and reduced_cuda_dequant_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_dequant_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="reduced_sparse_oracle_cuda_dequant_10tps_plan",
            passed=reduced_page_cuda_dequant_passed,
            evidence=(
                f"status={reduced_cuda_dequant_status}; "
                f"q4_0={float_value(reduced_page_cuda_dequant_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(reduced_page_cuda_dequant_summary, 'q4_0_tensors')}; "
                f"max_live={float_value(reduced_page_cuda_dequant_summary, 'max_live_buffer_gib'):.3f} GiB; "
                f"h2d={float_value(reduced_page_cuda_dequant_summary, 'h2d_ms'):.2f} ms; "
                f"dequant={float_value(reduced_page_cuda_dequant_summary, 'dequant_ms'):.2f} ms"
            ),
            next_step="Use the reduced page set for real-activation sparse matvec and candidate-quality checks.",
        )
    )

    reduced_cuda_matvec_evidence = reduced_page_cuda_matvec_payload.get("runtime_ledger_evidence", {})
    reduced_cuda_matvec_status = reduced_page_cuda_matvec_payload.get("status", "")
    reduced_page_cuda_matvec_passed = (
        reduced_page_cuda_matvec_payload.get("schema") == "sage-oracle-page-cuda-matvec-smoke-v0"
        and reduced_cuda_matvec_status == "measured_cuda_q4_0_matvec_synthetic_activation_not_transformer"
        and int_value(reduced_page_cuda_matvec_summary, "stages_staged") > 0
        and int_value(reduced_page_cuda_matvec_summary, "q4_0_tensors") > 0
        and int_value(reduced_page_cuda_matvec_summary, "q4_0_bytes") > 0
        and int_value(reduced_page_cuda_matvec_summary, "output_scores") > 0
        and float_value(reduced_page_cuda_matvec_summary, "q4_0_gib") <= 1.18
        and float_value(reduced_page_cuda_matvec_summary, "max_live_buffer_gib") <= 0.50
        and float_value(reduced_page_cuda_matvec_summary, "h2d_ms") > 0
        and float_value(reduced_page_cuda_matvec_summary, "matvec_ms") > 0
        and bool_value(reduced_page_cuda_matvec_summary, "score_output_nonzero")
        and bool_value(reduced_page_cuda_matvec_summary, "byte_budget_respected")
        and reduced_page_cuda_matvec_summary.get("row_score_capture_status") == "measured_per_row_scores"
        and int_value(reduced_page_cuda_matvec_summary, "cpu_score_checks") > 0
        and bool_value(reduced_page_cuda_matvec_summary, "cpu_score_checks_passed")
        and float_value(reduced_page_cuda_matvec_summary, "max_cpu_score_abs_error") <= 1.0e-3
        and reduced_page_cuda_matvec_summary.get("cuda_kernel_status")
        == "measured_q4_0_matvec_synthetic_activation_kernel"
        and reduced_page_cuda_matvec_summary.get("candidate_scoring_status")
        == "ranked_projection_rows_not_candidate_tokens"
        and reduced_page_cuda_matvec_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(reduced_cuda_matvec_evidence, dict)
        and reduced_cuda_matvec_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_matvec_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="reduced_sparse_oracle_cuda_matvec_10tps_plan",
            passed=reduced_page_cuda_matvec_passed,
            evidence=(
                f"status={reduced_cuda_matvec_status}; "
                f"q4_0={float_value(reduced_page_cuda_matvec_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(reduced_page_cuda_matvec_summary, 'q4_0_tensors')}; "
                f"scores={int_value(reduced_page_cuda_matvec_summary, 'output_scores')}; "
                f"max_live={float_value(reduced_page_cuda_matvec_summary, 'max_live_buffer_gib'):.3f} GiB; "
                f"h2d={float_value(reduced_page_cuda_matvec_summary, 'h2d_ms'):.2f} ms; "
                f"matvec={float_value(reduced_page_cuda_matvec_summary, 'matvec_ms'):.2f} ms"
            ),
            next_step="Replace synthetic activations on this reduced plan with captured hidden states and oracle-logit checks.",
        )
    )

    reduced_real_matvec_evidence = reduced_page_cuda_real_activation_matvec_payload.get("runtime_ledger_evidence", {})
    reduced_real_matvec_activation = reduced_page_cuda_real_activation_matvec_payload.get("activation", {})
    reduced_real_matvec_activation = reduced_real_matvec_activation if isinstance(reduced_real_matvec_activation, dict) else {}
    reduced_real_matvec_status = reduced_page_cuda_real_activation_matvec_payload.get("status", "")
    reduced_real_matvec_width = int_value(reduced_page_cuda_real_activation_matvec_summary, "activation_width")
    reduced_real_matvec_passed = (
        reduced_page_cuda_real_activation_matvec_payload.get("schema")
        == "sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0"
        and reduced_real_matvec_status == "measured_cuda_q4_0_real_activation_ranked_scores_not_oracle_logits"
        and int_value(reduced_page_cuda_real_activation_matvec_summary, "stages_staged") > 0
        and int_value(reduced_page_cuda_real_activation_matvec_summary, "q4_0_tensors") > 0
        and int_value(reduced_page_cuda_real_activation_matvec_summary, "q4_0_bytes") > 0
        and int_value(reduced_page_cuda_real_activation_matvec_summary, "output_scores") > 0
        and reduced_real_matvec_width > 0
        and int_value(reduced_real_matvec_activation, "value_count") == reduced_real_matvec_width
        and float_value(reduced_page_cuda_real_activation_matvec_summary, "q4_0_gib") <= 1.18
        and float_value(reduced_page_cuda_real_activation_matvec_summary, "max_live_buffer_gib") <= 0.50
        and float_value(reduced_page_cuda_real_activation_matvec_summary, "h2d_ms") > 0
        and float_value(reduced_page_cuda_real_activation_matvec_summary, "matvec_ms") > 0
        and bool_value(reduced_page_cuda_real_activation_matvec_summary, "score_output_nonzero")
        and bool_value(reduced_page_cuda_real_activation_matvec_summary, "byte_budget_respected")
        and reduced_page_cuda_real_activation_matvec_summary.get("row_score_capture_status")
        == "measured_per_row_scores"
        and int_value(reduced_page_cuda_real_activation_matvec_summary, "cpu_score_checks") > 0
        and bool_value(reduced_page_cuda_real_activation_matvec_summary, "cpu_score_checks_passed")
        and float_value(reduced_page_cuda_real_activation_matvec_summary, "max_cpu_score_abs_error") <= 1.0e-3
        and reduced_page_cuda_real_activation_matvec_summary.get("cuda_kernel_status")
        == "measured_q4_0_matvec_real_activation_kernel"
        and reduced_page_cuda_real_activation_matvec_summary.get("real_activation_status")
        == "measured_tensor_values_jsonl"
        and reduced_page_cuda_real_activation_matvec_summary.get("candidate_scoring_status")
        == "ranked_projection_rows_not_candidate_tokens"
        and reduced_page_cuda_real_activation_matvec_summary.get("sparse_transformer_status") == "not_implemented"
        and isinstance(reduced_real_matvec_evidence, dict)
        and reduced_real_matvec_evidence.get("oracle_mode")
        == "sparse_page_cuda_q4_0_real_activation_ranked_matvec_smoke"
        and reduced_real_matvec_evidence.get("activation_mode") == "real_tensor_values_jsonl"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="reduced_sparse_oracle_cuda_real_activation_matvec_10tps_plan",
            passed=reduced_real_matvec_passed,
            evidence=(
                f"status={reduced_real_matvec_status}; "
                f"activation_width={reduced_real_matvec_width}; "
                f"q4_0={float_value(reduced_page_cuda_real_activation_matvec_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(reduced_page_cuda_real_activation_matvec_summary, 'q4_0_tensors')}; "
                f"scores={int_value(reduced_page_cuda_real_activation_matvec_summary, 'output_scores')}; "
                f"h2d={float_value(reduced_page_cuda_real_activation_matvec_summary, 'h2d_ms'):.2f} ms; "
                f"matvec={float_value(reduced_page_cuda_real_activation_matvec_summary, 'matvec_ms'):.2f} ms; "
                f"cpu_checks={int_value(reduced_page_cuda_real_activation_matvec_summary, 'cpu_score_checks')}"
            ),
            next_step="Map these reduced real-activation row scores to candidate-token decisions and compare to exact logits.",
        )
    )

    quality_retention = reduced_page_quality_summary.get("global_top_retention", {})
    quality_retention = quality_retention if isinstance(quality_retention, dict) else {}
    top20_retention = quality_retention.get("20", {})
    top20_retention = top20_retention if isinstance(top20_retention, dict) else {}
    reduced_page_quality_passed = (
        reduced_page_quality_payload.get("schema") == "sage-reduced-page-quality-probe-v0"
        and reduced_page_quality_payload.get("status")
        == "measured_reduced_real_activation_signal_overlap_not_token_decisions"
        and bool_value(reduced_page_quality_payload.get("activation", {}), "activation_match")
        and bool_value(reduced_page_quality_summary, "reduced_signal_consistent_with_full_shared_tensors")
        and not bool_value(reduced_page_quality_summary, "token_decision_integrated", True)
        and not bool_value(reduced_page_quality_summary, "candidate_token_quality_proven", True)
        and float_value(reduced_page_quality_summary, "reduced_vs_full_q4_0_percent") > 0
        and float_value(reduced_page_quality_summary, "reduced_vs_full_q4_0_percent") < 100
        and int_value(reduced_page_quality_summary, "shared_scored_tensors") > 0
        and float_value(reduced_page_quality_summary, "shared_top1_match_rate") >= 0.99
        and float_value(reduced_page_quality_summary, "shared_topk_overlap_mean") >= 0.99
        and float_value(reduced_page_quality_summary, "max_shared_score_abs_delta") <= 1.0e-5
        and float_value(top20_retention, "retention_rate") > 0
        and bool_value(reduced_page_quality_summary, "selection_needs_signal_aware_optimization")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="reduced_page_signal_quality_probe",
            passed=reduced_page_quality_passed,
            evidence=(
                f"reduced={float_value(reduced_page_quality_summary, 'reduced_q4_0_gib'):.3f} GiB vs "
                f"full={float_value(reduced_page_quality_summary, 'full_q4_0_gib'):.3f} GiB; "
                f"shared_top1={float_value(reduced_page_quality_summary, 'shared_top1_match_rate'):.1%}; "
                f"top20_retention={float_value(top20_retention, 'retention_rate'):.1%}; "
                f"needs_signal_opt={bool_value(reduced_page_quality_summary, 'selection_needs_signal_aware_optimization')}"
            ),
            next_step=(
                "Improve page selection with signal-aware ranking, then map retained rows to candidate-token decisions."
            ),
        )
    )

    signal_quality_retention = signal_aware_page_quality_summary.get("global_top_retention", {})
    signal_quality_retention = signal_quality_retention if isinstance(signal_quality_retention, dict) else {}
    signal_top20_retention = signal_quality_retention.get("20", {})
    signal_top20_retention = signal_top20_retention if isinstance(signal_top20_retention, dict) else {}
    signal_top50_retention = signal_quality_retention.get("50", {})
    signal_top50_retention = signal_top50_retention if isinstance(signal_top50_retention, dict) else {}
    signal_aware_page_quality_passed = (
        signal_aware_page_quality_payload.get("schema") == "sage-reduced-page-quality-probe-v0"
        and signal_aware_page_quality_payload.get("status")
        == "measured_reduced_real_activation_signal_overlap_not_token_decisions"
        and bool_value(signal_aware_page_quality_payload.get("activation", {}), "activation_match")
        and bool_value(signal_aware_page_quality_summary, "reduced_signal_consistent_with_full_shared_tensors")
        and not bool_value(signal_aware_page_quality_summary, "token_decision_integrated", True)
        and not bool_value(signal_aware_page_quality_summary, "candidate_token_quality_proven", True)
        and float_value(signal_aware_page_quality_summary, "reduced_vs_full_q4_0_percent") > 0
        and float_value(signal_aware_page_quality_summary, "reduced_vs_full_q4_0_percent") < 100
        and int_value(signal_aware_page_quality_summary, "shared_scored_tensors") > 0
        and float_value(signal_aware_page_quality_summary, "shared_top1_match_rate") >= 0.99
        and float_value(signal_aware_page_quality_summary, "shared_topk_overlap_mean") >= 0.99
        and float_value(signal_aware_page_quality_summary, "max_shared_score_abs_delta") <= 1.0e-5
        and float_value(signal_top20_retention, "retention_rate") >= 0.80
        and float_value(signal_top50_retention, "retention_rate") >= 0.90
        and float_value(signal_top20_retention, "retention_rate") > float_value(top20_retention, "retention_rate")
        and not bool_value(signal_aware_page_quality_summary, "selection_needs_signal_aware_optimization", True)
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="signal_aware_page_selector_retention_probe",
            passed=signal_aware_page_quality_passed,
            evidence=(
                f"signal-aware={float_value(signal_aware_page_quality_summary, 'reduced_q4_0_gib'):.3f} GiB vs "
                f"full={float_value(signal_aware_page_quality_summary, 'full_q4_0_gib'):.3f} GiB; "
                f"top20_retention={float_value(signal_top20_retention, 'retention_rate'):.1%} "
                f"vs baseline={float_value(top20_retention, 'retention_rate'):.1%}; "
                f"top50_retention={float_value(signal_top50_retention, 'retention_rate'):.1%}; "
                f"shared_top1={float_value(signal_aware_page_quality_summary, 'shared_top1_match_rate'):.1%}"
            ),
            next_step=(
                "Validate the signal-aware selector on more activations, then feed retained rows into token-candidate "
                "decisions instead of treating row retention as final quality."
            ),
        )
    )

    cross_activation = signal_aware_cross_activation_quality_payload.get("activation", {})
    cross_activation = cross_activation if isinstance(cross_activation, dict) else {}
    cross_retention = signal_aware_cross_activation_quality_summary.get("global_top_retention", {})
    cross_retention = cross_retention if isinstance(cross_retention, dict) else {}
    cross_top20_retention = cross_retention.get("20", {})
    cross_top20_retention = cross_top20_retention if isinstance(cross_top20_retention, dict) else {}
    cross_top50_retention = cross_retention.get("50", {})
    cross_top50_retention = cross_top50_retention if isinstance(cross_top50_retention, dict) else {}
    signal_aware_cross_activation_passed = (
        signal_aware_cross_activation_quality_payload.get("schema") == "sage-reduced-page-quality-probe-v0"
        and signal_aware_cross_activation_quality_payload.get("status")
        == "measured_reduced_real_activation_signal_overlap_not_token_decisions"
        and cross_activation.get("full_name") == "result_norm"
        and cross_activation.get("reduced_name") == "result_norm"
        and bool_value(cross_activation, "activation_match")
        and bool_value(signal_aware_cross_activation_quality_summary, "reduced_signal_consistent_with_full_shared_tensors")
        and not bool_value(signal_aware_cross_activation_quality_summary, "token_decision_integrated", True)
        and not bool_value(signal_aware_cross_activation_quality_summary, "candidate_token_quality_proven", True)
        and int_value(signal_aware_cross_activation_quality_summary, "shared_scored_tensors") > 0
        and float_value(signal_aware_cross_activation_quality_summary, "shared_top1_match_rate") >= 0.99
        and float_value(signal_aware_cross_activation_quality_summary, "shared_topk_overlap_mean") >= 0.99
        and float_value(signal_aware_cross_activation_quality_summary, "max_shared_score_abs_delta") <= 1.0e-5
        and float_value(cross_top20_retention, "retention_rate") >= 0.90
        and float_value(cross_top50_retention, "retention_rate") >= 0.80
        and float_value(cross_top20_retention, "retention_rate") > float_value(top20_retention, "retention_rate")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="signal_aware_cross_activation_retention_probe",
            passed=signal_aware_cross_activation_passed,
            evidence=(
                f"activation={cross_activation.get('full_name', '')}; "
                f"signal-aware={float_value(signal_aware_cross_activation_quality_summary, 'reduced_q4_0_gib'):.3f} GiB vs "
                f"full={float_value(signal_aware_cross_activation_quality_summary, 'full_q4_0_gib'):.3f} GiB; "
                f"top20_retention={float_value(cross_top20_retention, 'retention_rate'):.1%}; "
                f"top50_retention={float_value(cross_top50_retention, 'retention_rate'):.1%}; "
                f"shared_top1={float_value(signal_aware_cross_activation_quality_summary, 'shared_top1_match_rate'):.1%}"
            ),
            next_step=(
                "Repeat this cross-activation check across multiple prompts/layers and convert retained rows into "
                "candidate-token decisions before treating the selector as quality-proven."
            ),
        )
    )

    signal_aware_page_cache_budget_passed = (
        signal_aware_page_cache_budget_payload.get("schema") == "sage-page-cache-budget-v0"
        and signal_aware_page_cache_budget_payload.get("status")
        == "measured_page_cache_budget_target_not_transformer_integrated"
        and bool_value(signal_aware_page_cache_budget_summary, "budget_target_passed")
        and bool_value(signal_aware_page_cache_budget_summary, "measured_plan_meets_target_tps")
        and not bool_value(signal_aware_page_cache_budget_summary, "runtime_integrated", True)
        and not bool_value(signal_aware_page_cache_budget_summary, "transformer_integrated", True)
        and float_value(signal_aware_page_cache_budget_summary, "effective_tps_with_page_cache")
        >= float_value(signal_aware_page_cache_budget_summary, "target_tps")
        and float_value(signal_aware_page_cache_budget_summary, "measured_page_cache_replay_ms") > 0
        and float_value(signal_aware_page_cache_budget_summary, "measured_page_cache_replay_gib") > 0
        and float_value(signal_aware_page_cache_budget_summary, "measured_page_cache_replay_gib")
        <= float_value(signal_aware_page_cache_budget_summary, "max_active_gib_for_target")
        and float_value(signal_aware_page_cache_budget_summary, "current_active_percent_of_reference_100b")
        <= float_value(signal_aware_page_cache_budget_summary, "max_active_percent_of_reference_100b_for_target")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="signal_aware_page_cache_10tps_projection",
            passed=signal_aware_page_cache_budget_passed,
            evidence=(
                f"cache={float_value(signal_aware_page_cache_budget_summary, 'effective_tps_with_page_cache'):.2f} tok/s; "
                f"target={float_value(signal_aware_page_cache_budget_summary, 'target_tps'):.1f}; "
                f"replay={float_value(signal_aware_page_cache_budget_summary, 'measured_page_cache_replay_ms'):.1f} ms/"
                f"{float_value(signal_aware_page_cache_budget_summary, 'measured_page_cache_replay_gib'):.3f} GiB; "
                f"active_ref={float_value(signal_aware_page_cache_budget_summary, 'current_active_percent_of_reference_100b'):.2f}%; "
                f"max_ref={float_value(signal_aware_page_cache_budget_summary, 'max_active_percent_of_reference_100b_for_target'):.2f}%"
            ),
            next_step=(
                "Integrate the signal-aware page cache with sparse transformer math and runtime fallback decisions."
            ),
        )
    )

    real_matvec_evidence = page_cuda_real_activation_matvec_payload.get("runtime_ledger_evidence", {})
    real_matvec_activation = page_cuda_real_activation_matvec_payload.get("activation", {})
    real_matvec_activation = real_matvec_activation if isinstance(real_matvec_activation, dict) else {}
    real_matvec_status = page_cuda_real_activation_matvec_payload.get("status", "")
    real_matvec_width = int_value(page_cuda_real_activation_matvec_summary, "activation_width")
    real_matvec_passed = (
        page_cuda_real_activation_matvec_payload.get("schema")
        == "sage-oracle-page-cuda-real-activation-matvec-smoke-v0"
        and real_matvec_status == "measured_cuda_q4_0_matvec_real_activation_not_oracle_logits"
        and int_value(page_cuda_real_activation_matvec_summary, "stages_staged") > 0
        and int_value(page_cuda_real_activation_matvec_summary, "q4_0_tensors") > 0
        and int_value(page_cuda_real_activation_matvec_summary, "q4_0_bytes") > 0
        and int_value(page_cuda_real_activation_matvec_summary, "q4_0_values") > 0
        and int_value(page_cuda_real_activation_matvec_summary, "output_scores") > 0
        and real_matvec_width > 0
        and int_value(real_matvec_activation, "value_count") == real_matvec_width
        and float_value(page_cuda_real_activation_matvec_summary, "matvec_ms") > 0
        and bool_value(page_cuda_real_activation_matvec_summary, "score_output_nonzero")
        and bool_value(page_cuda_real_activation_matvec_summary, "byte_budget_respected")
        and page_cuda_real_activation_matvec_summary.get("cuda_kernel_status")
        == "measured_q4_0_matvec_real_activation_kernel"
        and page_cuda_real_activation_matvec_summary.get("real_activation_status") == "measured_tensor_values_jsonl"
        and page_cuda_real_activation_matvec_summary.get("candidate_scoring_status") == "not_implemented"
        and isinstance(real_matvec_evidence, dict)
        and real_matvec_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_real_activation_matvec_smoke"
        and real_matvec_evidence.get("activation_mode") == "real_tensor_values_jsonl"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_real_activation_matvec_smoke",
            passed=real_matvec_passed,
            evidence=(
                f"status={real_matvec_status}; "
                f"activation_width={real_matvec_width}; "
                f"q4_0={float_value(page_cuda_real_activation_matvec_summary, 'q4_0_gib'):.3f} GiB; "
                f"tensors={int_value(page_cuda_real_activation_matvec_summary, 'q4_0_tensors')}; "
                f"scores={int_value(page_cuda_real_activation_matvec_summary, 'output_scores')}; "
                f"h2d={float_value(page_cuda_real_activation_matvec_summary, 'h2d_ms'):.2f} ms; "
                f"matvec={float_value(page_cuda_real_activation_matvec_summary, 'matvec_ms'):.2f} ms"
            ),
            next_step="Use real activation scores for candidate ranking and compare them to exact oracle logits.",
        )
    )

    ranked_matvec_evidence = page_cuda_real_activation_ranked_matvec_payload.get("runtime_ledger_evidence", {})
    ranked_matvec_activation = page_cuda_real_activation_ranked_matvec_payload.get("activation", {})
    ranked_matvec_activation = ranked_matvec_activation if isinstance(ranked_matvec_activation, dict) else {}
    ranked_matvec_status = page_cuda_real_activation_ranked_matvec_payload.get("status", "")
    ranked_matvec_width = int_value(page_cuda_real_activation_ranked_matvec_summary, "activation_width")
    ranked_matvec_passed = (
        page_cuda_real_activation_ranked_matvec_payload.get("schema")
        == "sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0"
        and ranked_matvec_status == "measured_cuda_q4_0_real_activation_ranked_scores_not_oracle_logits"
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "stages_staged") > 0
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "q4_0_tensors") > 0
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "q4_0_bytes") > 0
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "output_scores") > 0
        and ranked_matvec_width > 0
        and int_value(ranked_matvec_activation, "value_count") == ranked_matvec_width
        and page_cuda_real_activation_ranked_matvec_summary.get("row_score_capture_status")
        == "measured_per_row_scores"
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "top_score_tensors") > 0
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "top_score_rows") > 0
        and int_value(page_cuda_real_activation_ranked_matvec_summary, "cpu_score_checks") > 0
        and bool_value(page_cuda_real_activation_ranked_matvec_summary, "cpu_score_checks_passed")
        and float_value(page_cuda_real_activation_ranked_matvec_summary, "max_cpu_score_abs_error") <= 1.0e-3
        and float_value(page_cuda_real_activation_ranked_matvec_summary, "matvec_ms") > 0
        and bool_value(page_cuda_real_activation_ranked_matvec_summary, "score_output_nonzero")
        and bool_value(page_cuda_real_activation_ranked_matvec_summary, "byte_budget_respected")
        and page_cuda_real_activation_ranked_matvec_summary.get("cuda_kernel_status")
        == "measured_q4_0_matvec_real_activation_kernel"
        and page_cuda_real_activation_ranked_matvec_summary.get("real_activation_status") == "measured_tensor_values_jsonl"
        and page_cuda_real_activation_ranked_matvec_summary.get("candidate_scoring_status")
        == "ranked_projection_rows_not_candidate_tokens"
        and isinstance(ranked_matvec_evidence, dict)
        and ranked_matvec_evidence.get("oracle_mode") == "sparse_page_cuda_q4_0_real_activation_ranked_matvec_smoke"
        and ranked_matvec_evidence.get("activation_mode") == "real_tensor_values_jsonl"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_real_activation_ranked_matvec_smoke",
            passed=ranked_matvec_passed,
            evidence=(
                f"status={ranked_matvec_status}; "
                f"top_tensors={int_value(page_cuda_real_activation_ranked_matvec_summary, 'top_score_tensors')}; "
                f"top_rows={int_value(page_cuda_real_activation_ranked_matvec_summary, 'top_score_rows')}; "
                f"cpu_checks={int_value(page_cuda_real_activation_ranked_matvec_summary, 'cpu_score_checks')}; "
                f"max_abs_err={float_value(page_cuda_real_activation_ranked_matvec_summary, 'max_cpu_score_abs_error'):.3g}; "
                f"matvec={float_value(page_cuda_real_activation_ranked_matvec_summary, 'matvec_ms'):.2f} ms"
            ),
            next_step="Feed ranked projection rows into candidate-token scoring and compare against exact oracle logits.",
        )
    )

    vocab_projection_evidence = page_cuda_vocab_projection_payload.get("runtime_ledger_evidence", {})
    vocab_projection_status = page_cuda_vocab_projection_payload.get("status", "")
    vocab_projection_passed = (
        page_cuda_vocab_projection_payload.get("schema") == "sage-oracle-page-cuda-q6-k-vocab-projection-smoke-v0"
        and vocab_projection_status == "measured_q6_k_tied_vocab_projection_not_true_logits"
        and page_cuda_vocab_projection_summary.get("tensor_type") == "Q6_K"
        and int_value(page_cuda_vocab_projection_summary, "vocab_rows_scored") > 0
        and int_value(page_cuda_vocab_projection_summary, "chunks") > 0
        and int_value(page_cuda_vocab_projection_summary, "staged_bytes") > 0
        and int_value(page_cuda_vocab_projection_summary, "top_tokens") > 0
        and int_value(page_cuda_vocab_projection_summary, "cpu_score_checks") > 0
        and bool_value(page_cuda_vocab_projection_summary, "cpu_score_checks_passed")
        and float_value(page_cuda_vocab_projection_summary, "max_cpu_score_abs_error") <= 1.0e-3
        and float_value(page_cuda_vocab_projection_summary, "kernel_ms") > 0
        and bool_value(page_cuda_vocab_projection_summary, "score_output_nonzero")
        and bool_value(page_cuda_vocab_projection_summary, "byte_budget_respected")
        and page_cuda_vocab_projection_summary.get("candidate_scoring_status")
        == "vocab_token_scores_from_captured_activation_not_oracle_logits"
        and page_cuda_vocab_projection_summary.get("true_logit_status") == "not_implemented_final_hidden_state_missing"
        and isinstance(vocab_projection_evidence, dict)
        and vocab_projection_evidence.get("oracle_mode") == "sparse_page_cuda_q6_k_vocab_projection_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_q6k_vocab_projection_smoke",
            passed=vocab_projection_passed,
            evidence=(
                f"status={vocab_projection_status}; "
                f"rows={int_value(page_cuda_vocab_projection_summary, 'vocab_rows_scored')}; "
                f"chunks={int_value(page_cuda_vocab_projection_summary, 'chunks')}; "
                f"staged={float_value(page_cuda_vocab_projection_summary, 'staged_gib'):.3f} GiB; "
                f"top_tokens={int_value(page_cuda_vocab_projection_summary, 'top_tokens')}; "
                f"cpu_checks={int_value(page_cuda_vocab_projection_summary, 'cpu_score_checks')}; "
                f"kernel={float_value(page_cuda_vocab_projection_summary, 'kernel_ms'):.2f} ms"
            ),
            next_step="Capture the final post-norm hidden state and compare Q6_K vocab scores to exact llama.cpp logits.",
        )
    )

    vocab_logit_compare_evidence = page_cuda_vocab_logit_compare_payload.get("runtime_ledger_evidence", {})
    vocab_logit_compare_status = page_cuda_vocab_logit_compare_payload.get("status", "")
    vocab_logit_compare_payload_detail = page_cuda_vocab_logit_compare_payload.get("llamacpp_logit_comparison", {})
    vocab_logit_compare = vocab_logit_compare_payload_detail if isinstance(vocab_logit_compare_payload_detail, dict) else {}
    vocab_logit_compare_passed = (
        page_cuda_vocab_logit_compare_payload.get("schema") == "sage-oracle-page-cuda-q6-k-vocab-projection-smoke-v0"
        and vocab_logit_compare_status == "measured_q6_k_tied_vocab_projection_with_llamacpp_logit_compare"
        and page_cuda_vocab_logit_compare_summary.get("tensor_type") == "Q6_K"
        and int_value(page_cuda_vocab_logit_compare_summary, "vocab_rows_scored") > 0
        and int_value(page_cuda_vocab_logit_compare_summary, "chunks") > 0
        and int_value(page_cuda_vocab_logit_compare_summary, "staged_bytes") > 0
        and int_value(page_cuda_vocab_logit_compare_summary, "top_tokens") > 0
        and bool_value(page_cuda_vocab_logit_compare_summary, "cpu_score_checks_passed")
        and bool_value(page_cuda_vocab_logit_compare_summary, "llamacpp_logit_compare_passed")
        and bool_value(page_cuda_vocab_logit_compare_summary, "llamacpp_logit_top1_match")
        and float_value(page_cuda_vocab_logit_compare_summary, "llamacpp_logit_overlap_rate") >= 1.0
        and float_value(page_cuda_vocab_logit_compare_summary, "llamacpp_logit_max_abs_error") <= args.logit_max_abs_error
        and page_cuda_vocab_logit_compare_summary.get("candidate_scoring_status")
        == "vocab_projection_compared_to_llamacpp_logits"
        and page_cuda_vocab_logit_compare_summary.get("true_logit_status") == "compared_against_llamacpp_logits"
        and isinstance(vocab_logit_compare_evidence, dict)
        and vocab_logit_compare_evidence.get("oracle_mode") == "sparse_page_cuda_q6_k_vocab_projection_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_q6k_vocab_logit_compare",
            passed=vocab_logit_compare_passed,
            evidence=(
                f"status={vocab_logit_compare_status}; "
                f"rows={int_value(page_cuda_vocab_logit_compare_summary, 'vocab_rows_scored')}; "
                f"top1={bool_value(page_cuda_vocab_logit_compare_summary, 'llamacpp_logit_top1_match')}; "
                f"overlap={int_value(vocab_logit_compare, 'overlap_count')}/"
                f"{int_value(vocab_logit_compare, 'top_k')}; "
                f"max_abs_err={float_value(page_cuda_vocab_logit_compare_summary, 'llamacpp_logit_max_abs_error'):.4g}; "
                f"kernel={float_value(page_cuda_vocab_logit_compare_summary, 'kernel_ms'):.2f} ms"
            ),
            next_step="Use this exact logit agreement path as the token-candidate verifier inside a paged oracle loop.",
        )
    )

    candidate_verifier_evidence = page_cuda_candidate_verifier_payload.get("runtime_ledger_evidence", {})
    candidate_verifier_status = page_cuda_candidate_verifier_payload.get("status", "")
    candidate_verifier_passed = (
        page_cuda_candidate_verifier_payload.get("schema")
        == "sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0"
        and candidate_verifier_status == "measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_candidate_verifier_summary.get("tensor_type") == "Q6_K"
        and int_value(page_cuda_candidate_verifier_summary, "candidate_tokens") >= args.min_candidate_verifier_tokens
        and int_value(page_cuda_candidate_verifier_summary, "candidate_bytes") > 0
        and float_value(page_cuda_candidate_verifier_summary, "active_percent_vocab_tensor")
        <= args.max_candidate_verifier_vocab_percent
        and float_value(page_cuda_candidate_verifier_summary, "h2d_ms") > 0
        and float_value(page_cuda_candidate_verifier_summary, "kernel_ms") > 0
        and bool_value(page_cuda_candidate_verifier_summary, "cpu_score_checks_passed")
        and bool_value(page_cuda_candidate_verifier_summary, "llamacpp_logit_checks_passed")
        and bool_value(page_cuda_candidate_verifier_summary, "llamacpp_candidate_top1_match")
        and float_value(page_cuda_candidate_verifier_summary, "llamacpp_logit_max_abs_error")
        <= args.logit_max_abs_error
        and page_cuda_candidate_verifier_summary.get("candidate_scoring_status")
        == "sparse_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_candidate_verifier_summary.get("true_logit_status")
        == "candidate_rows_compared_against_llamacpp_logits"
        and isinstance(candidate_verifier_evidence, dict)
        and candidate_verifier_evidence.get("oracle_mode") == "sparse_q6_k_candidate_verifier_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_cuda_q6k_candidate_verifier",
            passed=candidate_verifier_passed,
            evidence=(
                f"status={candidate_verifier_status}; "
                f"tokens={int_value(page_cuda_candidate_verifier_summary, 'candidate_tokens')}; "
                f"bytes={int_value(page_cuda_candidate_verifier_summary, 'candidate_bytes')}; "
                f"active_vocab={float_value(page_cuda_candidate_verifier_summary, 'active_percent_vocab_tensor'):.4f}%; "
                f"top1={bool_value(page_cuda_candidate_verifier_summary, 'llamacpp_candidate_top1_match')}; "
                f"max_abs_err={float_value(page_cuda_candidate_verifier_summary, 'llamacpp_logit_max_abs_error'):.4g}; "
                f"kernel={float_value(page_cuda_candidate_verifier_summary, 'kernel_ms'):.4f} ms"
            ),
            next_step="Wire candidate-row scoring into the live scheduler with proxy-proposed token shortlists.",
        )
    )

    live_candidate_verifier_evidence = page_cuda_live_candidate_verifier_payload.get("runtime_ledger_evidence", {})
    live_candidate_source = page_cuda_live_candidate_verifier_payload.get("candidate_source", {})
    live_candidate_trace = live_candidate_source.get("live_trace", {}) if isinstance(live_candidate_source, dict) else {}
    live_candidate_verifier_status = page_cuda_live_candidate_verifier_payload.get("status", "")
    live_candidate_verifier_passed = (
        page_cuda_live_candidate_verifier_payload.get("schema")
        == "sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0"
        and live_candidate_verifier_status == "measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_live_candidate_verifier_summary.get("tensor_type") == "Q6_K"
        and page_cuda_live_candidate_verifier_summary.get("candidate_source_kind") == "live_proxy_shortlist"
        and int_value(page_cuda_live_candidate_verifier_summary, "candidate_tokens") >= args.min_live_proxy_top_k
        and int_value(page_cuda_live_candidate_verifier_summary, "live_trace_candidate_rows") >= args.min_live_proxy_top_k
        and int_value(page_cuda_live_candidate_verifier_summary, "candidate_bytes") > 0
        and float_value(page_cuda_live_candidate_verifier_summary, "active_percent_vocab_tensor")
        <= args.max_candidate_verifier_vocab_percent
        and float_value(page_cuda_live_candidate_verifier_summary, "h2d_ms") > 0
        and float_value(page_cuda_live_candidate_verifier_summary, "kernel_ms") > 0
        and bool_value(page_cuda_live_candidate_verifier_summary, "cpu_score_checks_passed")
        and bool_value(page_cuda_live_candidate_verifier_summary, "llamacpp_logit_checks_passed")
        and bool_value(page_cuda_live_candidate_verifier_summary, "llamacpp_candidate_top1_match")
        and float_value(page_cuda_live_candidate_verifier_summary, "llamacpp_logit_max_abs_error")
        <= args.logit_max_abs_error
        and page_cuda_live_candidate_verifier_summary.get("candidate_scoring_status")
        == "sparse_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_live_candidate_verifier_summary.get("true_logit_status")
        == "candidate_rows_compared_against_llamacpp_logits"
        and isinstance(live_candidate_trace, dict)
        and live_candidate_trace.get("schema") == "sage-live-proxy-shortlist-v0"
        and live_candidate_trace.get("candidate_status") == "live_logit_top_k_only"
        and bool_value(page_cuda_live_candidate_verifier_summary, "live_trace_top_ids_prefix_match")
        and isinstance(live_candidate_verifier_evidence, dict)
        and live_candidate_verifier_evidence.get("oracle_mode") == "sparse_q6_k_candidate_verifier_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="live_proxy_shortlist_cuda_q6k_verifier_bridge",
            passed=live_candidate_verifier_passed,
            evidence=(
                f"status={live_candidate_verifier_status}; "
                f"source={page_cuda_live_candidate_verifier_summary.get('candidate_source_kind', '')}; "
                f"live_rows={int_value(page_cuda_live_candidate_verifier_summary, 'live_trace_candidate_rows')}; "
                f"bytes={int_value(page_cuda_live_candidate_verifier_summary, 'candidate_bytes')}; "
                f"prefix={bool_value(page_cuda_live_candidate_verifier_summary, 'live_trace_top_ids_prefix_match')}; "
                f"top1={bool_value(page_cuda_live_candidate_verifier_summary, 'llamacpp_candidate_top1_match')}; "
                f"max_abs_err={float_value(page_cuda_live_candidate_verifier_summary, 'llamacpp_logit_max_abs_error'):.4g}; "
                f"kernel={float_value(page_cuda_live_candidate_verifier_summary, 'kernel_ms'):.4f} ms"
            ),
            next_step=(
                "Move this bridge into the live scheduler so proxy-proposed rows are verified before exact "
                "oracle fallback instead of only replayed as an offline smoke."
            ),
        )
    )

    proxy_fallback_evidence = page_cuda_proxy_fallback_verifier_payload.get("runtime_ledger_evidence", {})
    proxy_fallback_status = page_cuda_proxy_fallback_verifier_payload.get("status", "")
    proxy_fallback_passed = (
        page_cuda_proxy_fallback_verifier_payload.get("schema")
        == "sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0"
        and proxy_fallback_status == "measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_proxy_fallback_verifier_summary.get("tensor_type") == "Q6_K"
        and page_cuda_proxy_fallback_verifier_summary.get("candidate_source_kind") == "logprob_proxy_top_k"
        and page_cuda_proxy_fallback_verifier_summary.get("logprob_source_side") == "proxy"
        and page_cuda_proxy_fallback_verifier_summary.get("logprob_tokenizer_scope") == "same_family_logprob_capture"
        and int_value(page_cuda_proxy_fallback_verifier_summary, "candidate_tokens") >= args.min_live_proxy_top_k
        and int_value(page_cuda_proxy_fallback_verifier_summary, "logprob_candidate_rows") >= args.min_live_proxy_top_k
        and int_value(page_cuda_proxy_fallback_verifier_summary, "candidate_bytes") > 0
        and float_value(page_cuda_proxy_fallback_verifier_summary, "active_percent_vocab_tensor")
        <= args.max_candidate_verifier_vocab_percent
        and float_value(page_cuda_proxy_fallback_verifier_summary, "h2d_ms") > 0
        and float_value(page_cuda_proxy_fallback_verifier_summary, "kernel_ms") > 0
        and bool_value(page_cuda_proxy_fallback_verifier_summary, "cpu_score_checks_passed")
        and bool_value(page_cuda_proxy_fallback_verifier_summary, "llamacpp_logit_checks_passed")
        and bool_value(page_cuda_proxy_fallback_verifier_summary, "llamacpp_candidate_top1_match")
        and not bool_value(page_cuda_proxy_fallback_verifier_summary, "candidate_contains_llamacpp_global_top1")
        and not bool_value(page_cuda_proxy_fallback_verifier_summary, "logprob_oracle_top1_covered")
        and int_value(page_cuda_proxy_fallback_verifier_summary, "llamacpp_global_top1_id")
        == int_value(page_cuda_proxy_fallback_verifier_summary, "logprob_oracle_top1_id")
        and page_cuda_proxy_fallback_verifier_summary.get("candidate_global_top1_coverage_status")
        == "candidate_misses_global_top1_exact_fallback_required"
        and float_value(page_cuda_proxy_fallback_verifier_summary, "llamacpp_logit_max_abs_error")
        <= args.logit_max_abs_error
        and page_cuda_proxy_fallback_verifier_summary.get("candidate_scoring_status")
        == "sparse_candidate_rows_compared_to_llamacpp_logits"
        and page_cuda_proxy_fallback_verifier_summary.get("true_logit_status")
        == "candidate_rows_compared_against_llamacpp_logits"
        and isinstance(proxy_fallback_evidence, dict)
        and proxy_fallback_evidence.get("oracle_mode") == "sparse_q6_k_candidate_verifier_smoke"
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="same_prompt_proxy_shortlist_fallback_smoke",
            passed=proxy_fallback_passed,
            evidence=(
                f"status={proxy_fallback_status}; "
                f"prompt={page_cuda_proxy_fallback_verifier_summary.get('logprob_prompt', '')!r}; "
                f"rows={int_value(page_cuda_proxy_fallback_verifier_summary, 'logprob_candidate_rows')}; "
                f"global_top1={int_value(page_cuda_proxy_fallback_verifier_summary, 'llamacpp_global_top1_id')}; "
                f"source_top1={int_value(page_cuda_proxy_fallback_verifier_summary, 'logprob_source_token_id')}; "
                f"covered={bool_value(page_cuda_proxy_fallback_verifier_summary, 'candidate_contains_llamacpp_global_top1')}; "
                f"bytes={int_value(page_cuda_proxy_fallback_verifier_summary, 'candidate_bytes')}; "
                f"kernel={float_value(page_cuda_proxy_fallback_verifier_summary, 'kernel_ms'):.4f} ms"
            ),
            next_step=(
                "Move this reject/fallback condition into the live scheduler: proxy top-k misses must fall back "
                "unless the format/prompt scaffold covers the oracle top token."
            ),
        )
    )

    def shortlist_passed(payload: dict[str, Any], item: dict[str, Any]) -> bool:
        return (
            payload.get("schema") == "sage-proxy-shortlist-coverage-v0"
            and payload.get("status") == "measured_proxy_shortlist_coverage"
            and "prompt" in str(item.get("best_eval_coverage_name", ""))
            and float_value(item, "best_eval_coverage_rate") >= args.min_proxy_shortlist_format_coverage
            and float_value(item, "best_eval_fallback_rate_for_exact") <= args.max_proxy_shortlist_fallback
            and float_value(item, "best_eval_candidate_rows_per_step") <= args.max_proxy_shortlist_candidate_rows
            and int_value(item, "position_rescue_steps") > 0
            and int_value(item, "prompt_piece_tokens_max") > 0
        )

    validation_shortlist_passed = shortlist_passed(proxy_shortlist_validation_payload, proxy_shortlist_validation_summary)
    hard_shortlist_passed = shortlist_passed(proxy_shortlist_hard_payload, proxy_shortlist_hard_summary)
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="proxy_format_scaffold_shortlist_coverage",
            passed=validation_shortlist_passed and hard_shortlist_passed,
            evidence=(
                "validation "
                f"{float_value(proxy_shortlist_validation_summary, 'best_eval_coverage_rate'):.2%} "
                f"coverage/{float_value(proxy_shortlist_validation_summary, 'best_eval_fallback_rate_for_exact'):.2%} "
                f"fallback at {float_value(proxy_shortlist_validation_summary, 'best_eval_candidate_rows_per_step'):.1f} rows; "
                "hard "
                f"{float_value(proxy_shortlist_hard_summary, 'best_eval_coverage_rate'):.2%} "
                f"coverage/{float_value(proxy_shortlist_hard_summary, 'best_eval_fallback_rate_for_exact'):.2%} "
                f"fallback at {float_value(proxy_shortlist_hard_summary, 'best_eval_candidate_rows_per_step'):.1f} rows"
            ),
            next_step=(
                "Use the proxy top-k plus position/prompt-piece scaffold as the candidate producer for the "
                "CUDA Q6_K verifier, then measure end-to-end fallback cost."
            ),
        )
    )

    live_proxy_shortlist_rows = []
    for step in dual_steps:
        if not isinstance(step, dict):
            continue
        proxy_row = step.get("proxy")
        if not isinstance(proxy_row, dict):
            continue
        top_ids = proxy_row.get("logit_top_ids", [])
        top_logits = proxy_row.get("logit_top_logits", [])
        shortlist = proxy_row.get("candidate_shortlist", {})
        if isinstance(top_ids, list) and isinstance(top_logits, list) and isinstance(shortlist, dict):
            live_proxy_shortlist_rows.append((proxy_row, top_ids, top_logits, shortlist))
    live_proxy_shortlist_passed = (
        int_value(dual, "logit_top_k") >= args.min_live_proxy_top_k
        and dual.get("candidate_shortlist_status") == "live_logit_top_k_only"
        and len(live_proxy_shortlist_rows) >= dual_tokens
        and all(len(top_ids) >= args.min_live_proxy_top_k for _proxy, top_ids, _top_logits, _shortlist in live_proxy_shortlist_rows)
        and all(len(top_logits) == len(top_ids) for _proxy, top_ids, top_logits, _shortlist in live_proxy_shortlist_rows)
        and all(int_value(shortlist, "candidate_rows") == len(top_ids) for _proxy, top_ids, _top_logits, shortlist in live_proxy_shortlist_rows)
        and all(shortlist.get("schema") == "sage-live-proxy-shortlist-v0" for _proxy, _top_ids, _top_logits, shortlist in live_proxy_shortlist_rows)
        and all(shortlist.get("candidate_status") == "live_logit_top_k_only" for _proxy, _top_ids, _top_logits, shortlist in live_proxy_shortlist_rows)
        and all(top_ids and int(top_ids[0]) == int_value(proxy_row, "logit_top1_id") for proxy_row, top_ids, _top_logits, _shortlist in live_proxy_shortlist_rows)
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="live_proxy_shortlist_trace",
            passed=live_proxy_shortlist_passed,
            evidence=(
                f"rows={len(live_proxy_shortlist_rows)}/{dual_tokens}; "
                f"logit_top_k={int_value(dual, 'logit_top_k')}; "
                f"status={dual.get('candidate_shortlist_status', '')}; "
                f"min_rows={min((len(top_ids) for _proxy, top_ids, _top_logits, _shortlist in live_proxy_shortlist_rows), default=0)}"
            ),
            next_step="Feed this live top-k candidate set into the CUDA Q6_K verifier instead of only recording it.",
        )
    )

    sparse_runtime_ledger = sparse_oracle_runtime_step_payload.get("runtime_ledger_evidence", {})
    sparse_runtime_step_passed = (
        sparse_oracle_runtime_step_payload.get("schema") == "sage-sparse-oracle-runtime-step-v0"
        and sparse_oracle_runtime_step_payload.get("status")
        == "measured_component_replay_not_transformer_integrated"
        and bool_value(sparse_oracle_runtime_step_summary, "component_replay_complete")
        and not bool_value(sparse_oracle_runtime_step_summary, "transformer_integrated", True)
        and not bool_value(sparse_oracle_runtime_step_summary, "llama_cpp_live_integrated", True)
        and int_value(sparse_oracle_runtime_step_summary, "sparse_page_bytes") > 0
        and int_value(sparse_oracle_runtime_step_summary, "candidate_bytes") > 0
        and int_value(sparse_oracle_runtime_step_summary, "active_sparse_step_bytes") > 0
        and float_value(sparse_oracle_runtime_step_summary, "active_sparse_step_percent_reference_100b")
        <= args.oracle_active_percent_7tps
        and int_value(sparse_oracle_runtime_step_summary, "max_measured_device_stage_bytes") > 0
        and float_value(sparse_oracle_runtime_step_summary, "pcie_transfer_ms") > 0
        and float_value(sparse_oracle_runtime_step_summary, "cuda_kernel_ms") > 0
        and bool_value(sparse_oracle_runtime_step_summary, "exact_fallback_required")
        and sparse_oracle_runtime_step_summary.get("candidate_global_top1_coverage_status")
        == "candidate_misses_global_top1_exact_fallback_required"
        and isinstance(sparse_runtime_ledger, dict)
        and sparse_runtime_ledger.get("oracle_mode") == "sparse_page_candidate_verifier_replay"
        and int_value(sparse_runtime_ledger, "oracle_page_active_bytes")
        == int_value(sparse_oracle_runtime_step_summary, "sparse_page_bytes")
        and int_value(sparse_runtime_ledger, "verifier_active_bytes")
        == int_value(sparse_oracle_runtime_step_summary, "candidate_bytes")
    )
    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_runtime_step_replay",
            passed=sparse_runtime_step_passed,
            evidence=(
                f"status={sparse_oracle_runtime_step_payload.get('status', '')}; "
                f"page_bytes={int_value(sparse_oracle_runtime_step_summary, 'sparse_page_bytes')}; "
                f"candidate_bytes={int_value(sparse_oracle_runtime_step_summary, 'candidate_bytes')}; "
                f"active_ref={float_value(sparse_oracle_runtime_step_summary, 'active_sparse_step_percent_reference_100b'):.4f}%; "
                f"max_stage={float_value(sparse_oracle_runtime_step_summary, 'max_measured_device_stage_gib'):.3f} GiB; "
                f"h2d={float_value(sparse_oracle_runtime_step_summary, 'pcie_transfer_ms'):.2f} ms; "
                f"kernel={float_value(sparse_oracle_runtime_step_summary, 'cuda_kernel_ms'):.2f} ms; "
                f"fallback={bool_value(sparse_oracle_runtime_step_summary, 'exact_fallback_required')}"
            ),
            next_step=(
                "Replace this measured component replay with a live llama.cpp scheduler step that executes "
                "the sparse page math and candidate verifier before falling back."
            ),
        )
    )

    gates.append(
        Gate(
            stage="stage3_oracle_pager",
            name="sparse_oracle_pager_implemented",
            passed=False,
            evidence="No artifact proves block-paged 100B oracle execution yet.",
            next_step="Add pinned-host GGUF block pages, GPU staging buffers, and exact fallback telemetry.",
        )
    )

    kv_runtime_fields = kv_ledger_payload.get("runtime_ledger_fields", {})
    kv_plan_passed = (
        kv_ledger_payload.get("schema") == "sage-kv-ledger-v0"
        and kv_ledger_payload.get("status") == "plan_only_not_runtime_integrated"
        and isinstance(kv_runtime_fields, dict)
        and int_value(kv_runtime_fields, "oracle_hot_kv_bytes") > 0
        and int_value(kv_runtime_fields, "oracle_warm_kv_bytes") >= 0
        and kv_runtime_fields.get("kv_byte_status") == "planned_not_runtime_integrated"
        and bool_value(kv_ledger_summary, "oracle_hot_fits_budget")
        and float_value(kv_ledger_summary, "oracle_saved_percent_vs_full_precision") > 0
    )
    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="kv_ledger_plan",
            passed=kv_plan_passed,
            evidence=(
                f"schema={kv_ledger_payload.get('schema', '')}; "
                f"oracle tier={float_value(kv_ledger_summary, 'oracle_tier_total_gib'):.3f} GiB; "
                f"hot={float_value(kv_ledger_summary, 'oracle_hot_gib'):.3f} GiB; "
                f"saved={float_value(kv_ledger_summary, 'oracle_saved_percent_vs_full_precision'):.1f}%; "
                f"hot fits={bool_value(kv_ledger_summary, 'oracle_hot_fits_budget')}"
            ),
            next_step="Feed planned KV byte tiers into live runtime telemetry instead of null KV fields.",
        )
    )

    kv_pack_passed = (
        kv_tier_smoke_payload.get("schema") == "sage-kv-tier-pack-smoke-v0"
        and kv_tier_smoke_payload.get("status") == "measured_synthetic_2bit_warm_kv_pack_not_runtime_integrated"
        and int_value(kv_tier_smoke_summary, "sample_tokens") > 0
        and int_value(kv_tier_smoke_summary, "sample_packed_bytes") > 0
        and bool_value(kv_tier_smoke_summary, "bytes_match_plan")
        and bool_value(kv_tier_smoke_summary, "checksums_match")
        and float_value(kv_tier_smoke_summary, "compression_ratio_vs_fp16") >= 7.9
        and int_value(kv_tier_smoke_summary, "estimated_warm_packed_bytes")
        == int_value(kv_ledger_summary, "oracle_warm_bytes")
        and bool_value(kv_tier_smoke_cuda, "enabled")
        and bool_value(kv_tier_smoke_cuda, "packed_matches_cpu")
        and float_value(kv_tier_smoke_cuda, "pack_ms") > 0
        and float_value(kv_tier_smoke_cuda, "unpack_ms") > 0
    )
    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="kv_warm_2bit_cuda_pack_smoke",
            passed=kv_pack_passed,
            evidence=(
                f"sample={int_value(kv_tier_smoke_summary, 'sample_tokens')} tokens; "
                f"ratio={float_value(kv_tier_smoke_summary, 'compression_ratio_vs_fp16'):.2f}x; "
                f"warm={bytes_to_gib(int_value(kv_tier_smoke_summary, 'estimated_warm_packed_bytes')):.3f} GiB; "
                f"cuda_pack={float_value(kv_tier_smoke_cuda, 'pack_ms'):.4f} ms; "
                f"cuda_unpack={float_value(kv_tier_smoke_cuda, 'unpack_ms'):.4f} ms; "
                f"matches={bool_value(kv_tier_smoke_cuda, 'packed_matches_cpu')}"
            ),
            next_step="Integrate this native KV packing with real llama.cpp KV tensors and attention reads.",
        )
    )

    def ledger_has_runtime_kv(ledger: dict[str, Any]) -> bool:
        status = str(ledger.get("kv_byte_status", ""))
        if status == "estimated_full_precision_runtime":
            return int_value(ledger, "proxy_kv_bytes") > 0 and int_value(ledger, "proxy_kv_bytes_per_token_estimate") > 0
        if status == "tiered_runtime_accounting_not_attention_integrated":
            return (
                int_value(ledger, "proxy_kv_bytes") > 0
                and int_value(ledger, "proxy_kv_bytes_per_token_estimate") > 0
                and "kv_attention_integration" in ledger
                and not bool_value(ledger, "kv_attention_integration", True)
            )
        return False

    runtime_kv_estimate = any(
        isinstance(step, dict)
        and isinstance(step.get("ledger"), dict)
        and ledger_has_runtime_kv(step["ledger"])
        for step in dual_steps
    )
    runtime_oracle_kv_estimate = any(
        isinstance(step, dict)
        and isinstance(step.get("ledger"), dict)
        and step.get("action") == "oracle_fallback"
        and int_value(step["ledger"], "oracle_hot_kv_bytes") > 0
        and int_value(step["ledger"], "oracle_kv_bytes_per_token_estimate") > 0
        for step in dual_steps
    )
    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="runtime_kv_estimate_trace",
            passed=runtime_kv_estimate and runtime_oracle_kv_estimate,
            evidence=(
                f"proxy estimate={runtime_kv_estimate}; "
                f"oracle fallback estimate={runtime_oracle_kv_estimate}; "
                f"summary status={dual.get('kv_byte_status', '')}"
            ),
            next_step="Keep this as the base runtime KV field gate while packed KV tensor telemetry is added.",
        )
    )

    live_tiered_kv_trace = any(
        isinstance(step, dict)
        and isinstance(step.get("ledger"), dict)
        and step.get("action") == "oracle_fallback"
        and step["ledger"].get("kv_byte_status") == "tiered_runtime_accounting_not_attention_integrated"
        and int_value(step["ledger"], "oracle_full_precision_kv_bytes") > 0
        and int_value(step["ledger"], "oracle_tiered_kv_bytes") > 0
        and int_value(step["ledger"], "oracle_hot_kv_tokens") > 0
        and "oracle_kv_saved_percent_vs_full_precision" in step["ledger"]
        and not bool_value(step["ledger"], "kv_attention_integration", True)
        for step in dual_steps
    )
    summary_tiered_kv_trace = (
        dual.get("kv_byte_status") == "tiered_runtime_accounting_not_attention_integrated"
        and not bool_value(dual, "kv_attention_integration", True)
        and int_value(dual, "kv_warm_bits") > 0
        and int_value(dual, "kv_hot_recent_tokens") >= 0
    )
    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="live_tiered_kv_trace",
            passed=live_tiered_kv_trace and summary_tiered_kv_trace,
            evidence=(
                f"summary status={dual.get('kv_byte_status', '')}; "
                f"fallback tiered fields={live_tiered_kv_trace}; "
                f"warm_bits={int_value(dual, 'kv_warm_bits')}; "
                f"attention_integration={bool_value(dual, 'kv_attention_integration', True)}"
            ),
            next_step="Replace accounting fields with real packed KV tensor allocation and attention-read telemetry.",
        )
    )

    kv_runtime_accounting_passed = (
        kv_runtime_ledger_payload.get("schema") == "sage-kv-runtime-ledger-v0"
        and kv_runtime_ledger_payload.get("status")
        == "runtime_token_accounting_with_measured_pack_smoke_not_attention_integrated"
        and int_value(kv_runtime_ledger_summary, "annotated_steps") > 0
        and int_value(kv_runtime_ledger_summary, "runtime_oracle_fallback_steps") > 0
        and bool_value(kv_runtime_ledger_summary, "context_sweep_exercises_warm_kv")
        and int_value(kv_runtime_ledger_summary, "max_oracle_tier_total_bytes") > 0
        and int_value(kv_runtime_ledger_summary, "max_oracle_warm_kv_bytes") > 0
        and float_value(kv_runtime_ledger_summary, "max_oracle_saved_percent_vs_full_precision") > 0
        and not bool_value(kv_runtime_ledger_summary, "attention_integration", True)
        and bool_value(kv_runtime_pack_evidence, "cuda_enabled")
        and bool_value(kv_runtime_pack_evidence, "cuda_packed_matches_cpu")
    )
    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="runtime_tiered_kv_accounting",
            passed=kv_runtime_accounting_passed,
            evidence=(
                f"annotated={int_value(kv_runtime_ledger_summary, 'annotated_steps')} steps; "
                f"fallbacks={int_value(kv_runtime_ledger_summary, 'runtime_oracle_fallback_steps')}; "
                f"tier={float_value(kv_runtime_ledger_summary, 'max_oracle_tier_total_gib'):.3f} GiB; "
                f"warm={float_value(kv_runtime_ledger_summary, 'max_oracle_warm_kv_gib'):.3f} GiB; "
                f"saved={float_value(kv_runtime_ledger_summary, 'max_oracle_saved_percent_vs_full_precision'):.1f}%; "
                f"cuda_pack={float_value(kv_runtime_pack_evidence, 'cuda_pack_ms'):.4f} ms"
            ),
            next_step=(
                "Move this tiered accounting into llama-sage-dual-live and bind the fields to "
                "real packed KV tensors."
            ),
        )
    )

    gates.append(
        Gate(
            stage="stage4_kv_ledger",
            name="kv_ledger_implemented",
            passed=False,
            evidence=(
                "Tiered KV accounting exists in the live trace and offline context-sweep artifact, "
                "but llama.cpp still does not emit measured compressed KV bytes from real KV tensors."
            ),
            next_step="Integrate tiered KV storage and measured hot/warm/cold byte fields inside llama.cpp attention.",
        )
    )

    passed = sum(1 for gate in gates if gate.passed)
    failed = len(gates) - passed
    complete = failed == 0
    return {
        "complete": complete,
        "passed_gates": passed,
        "failed_gates": failed,
        "target_tps": args.target_tps,
        "max_total_error": args.max_total_error,
        "max_accepted_error": args.max_accepted_error,
        "projection_policy": projection_policy,
        "gates": [asdict(gate) for gate in gates],
    }


def print_markdown(payload: dict[str, Any]) -> None:
    print("# SAGE Active-Byte Contract Check")
    print()
    print(f"- Complete: `{payload['complete']}`")
    print(f"- Passed gates: `{payload['passed_gates']}`")
    print(f"- Failed gates: `{payload['failed_gates']}`")
    print()
    print("| Stage | Gate | Result | Evidence | Next step |")
    print("| --- | --- | --- | --- | --- |")
    for gate in payload["gates"]:
        result = "pass" if gate["passed"] else "fail"
        evidence = str(gate["evidence"]).replace("|", "\\|")
        next_step = str(gate["next_step"]).replace("|", "\\|")
        print(f"| {gate['stage']} | {gate['name']} | {result} | {evidence} | {next_step} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAGE-100 contract evidence.")
    parser.add_argument(
        "--policy-json",
        default="benchmarks/20260627-154000-sage-policy-report-validation80e-frozen-ffn1-stats.json",
    )
    parser.add_argument(
        "--policy-parity-json",
        default="benchmarks/sage-cpp-policy-parity-validation80e.json",
    )
    parser.add_argument(
        "--scheduler-replay-json",
        default="benchmarks/sage-cpp-scheduler-replay-validation80e.json",
    )
    parser.add_argument(
        "--proxy-live-json",
        default="benchmarks/sage-proxy-live-qwen05b-smoke.json",
    )
    parser.add_argument(
        "--dual-live-json",
        default="benchmarks/sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json",
    )
    parser.add_argument(
        "--projection-json",
        default="benchmarks/sage-runtime-projection-validation80e-live.json",
    )
    parser.add_argument(
        "--format-projection-json",
        default="benchmarks/sage-runtime-projection-format-scaffold-localfallback270.json",
    )
    parser.add_argument(
        "--sparse-fallback-projection-json",
        default="benchmarks/sage-runtime-projection-format-scaffold-measured-sparse-fallback.json",
    )
    parser.add_argument(
        "--overlap-budget-json",
        default="benchmarks/sage-overlap-budget-hard120-measured-sparse-fallback-10tps.json",
    )
    parser.add_argument(
        "--page-cache-budget-json",
        default="benchmarks/sage-page-cache-budget-hard120-resident-cache-10tps.json",
    )
    parser.add_argument(
        "--reduced-page-cache-budget-json",
        default="benchmarks/sage-page-cache-budget-hard120-resident-cache-1180mib-10tps.json",
    )
    parser.add_argument(
        "--reduced-page-quality-json",
        default="benchmarks/sage-reduced-page-quality-gemma31b-1180mib-vs-full-ffn-norm0.json",
    )
    parser.add_argument(
        "--signal-aware-page-cache-budget-json",
        default="benchmarks/sage-page-cache-budget-hard120-signal-aware-1180mib-10tps.json",
    )
    parser.add_argument(
        "--signal-aware-page-quality-json",
        default="benchmarks/sage-reduced-page-quality-gemma31b-signal-aware-1180mib-vs-full-ffn-norm0.json",
    )
    parser.add_argument(
        "--signal-aware-cross-activation-quality-json",
        default="benchmarks/sage-reduced-page-quality-gemma31b-signal-aware-1180mib-vs-full-result-norm.json",
    )
    parser.add_argument(
        "--oracle-page-ledger-json",
        default="benchmarks/sage-oracle-page-ledger-gemma31b-balanced-2330mib.json",
    )
    parser.add_argument(
        "--oracle-page-staging-json",
        default="benchmarks/sage-oracle-page-staging-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-staging-json",
        default="benchmarks/sage-oracle-page-cuda-staging-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-kernel-json",
        default="benchmarks/sage-oracle-page-cuda-kernel-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-overlap-json",
        default="benchmarks/sage-oracle-page-cuda-overlap-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-prefetch-overlap-json",
        default="benchmarks/sage-oracle-page-cuda-prefetch-overlap-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-page-cache-json",
        default="benchmarks/sage-oracle-page-cuda-page-cache-gemma31b-full-replay3.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-dequant-json",
        default="benchmarks/sage-oracle-page-cuda-dequant-gemma31b-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-matvec-gemma31b-full.json",
    )
    parser.add_argument(
        "--reduced-oracle-page-cuda-dequant-json",
        default="benchmarks/sage-oracle-page-cuda-dequant-gemma31b-balanced-1180mib.json",
    )
    parser.add_argument(
        "--reduced-oracle-page-cuda-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-matvec-gemma31b-balanced-1180mib.json",
    )
    parser.add_argument(
        "--reduced-oracle-page-cuda-real-activation-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-matvec-gemma31b-balanced-1180mib-ffn-norm0.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-real-activation-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-matvec-gemma31b-ffn-norm0-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-real-activation-ranked-matvec-json",
        default="benchmarks/sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-ffn-norm0-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-vocab-projection-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-vocab-projection-gemma31b-ffn-norm0-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-vocab-logit-compare-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-vocab-logit-compare-gemma31b-result-norm-full.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-candidate-verifier-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-candidate-verifier-gemma31b-result-norm-top64.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-live-candidate-verifier-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-candidate-verifier-live-topk-qwen-trace-gemma31b-result-norm.json",
    )
    parser.add_argument(
        "--oracle-page-cuda-proxy-fallback-verifier-json",
        default="benchmarks/sage-oracle-page-cuda-q6k-candidate-verifier-gemma12proxy-france-top10-fallback-result-norm.json",
    )
    parser.add_argument(
        "--sparse-oracle-runtime-step-json",
        default="benchmarks/sage-sparse-oracle-runtime-step-gemma31b-page-q6k-fallback-replay.json",
    )
    parser.add_argument(
        "--proxy-shortlist-validation-json",
        default="benchmarks/sage-proxy-shortlist-format-scaffold-validation80e-k10-pos8-prompt16.json",
    )
    parser.add_argument(
        "--proxy-shortlist-hard-json",
        default="benchmarks/sage-proxy-shortlist-format-scaffold-hard120-k10-pos8-prompt16.json",
    )
    parser.add_argument(
        "--kv-ledger-json",
        default="benchmarks/sage-kv-ledger-gemma31b-ctx4096-hot528-warm2bit.json",
    )
    parser.add_argument(
        "--kv-tier-smoke-json",
        default="benchmarks/sage-kv-tier-pack-smoke-gemma31b-ctx4096-hot528-warm2bit-sample8.json",
    )
    parser.add_argument(
        "--kv-runtime-ledger-json",
        default="benchmarks/sage-kv-runtime-ledger-qwen05b-arithmetic-gemma31b-plan.json",
    )
    parser.add_argument("--target-tps", type=float, default=7.0)
    parser.add_argument("--max-total-error", type=float, default=0.02)
    parser.add_argument("--max-accepted-error", type=float, default=0.05)
    parser.add_argument("--oracle-active-percent-7tps", type=float, default=10.0)
    parser.add_argument("--oracle-active-percent-10tps", type=float, default=5.0)
    parser.add_argument("--logit-max-abs-error", type=float, default=0.05)
    parser.add_argument("--min-candidate-verifier-tokens", type=int, default=32)
    parser.add_argument("--max-candidate-verifier-vocab-percent", type=float, default=0.1)
    parser.add_argument("--min-proxy-shortlist-format-coverage", type=float, default=0.95)
    parser.add_argument("--max-proxy-shortlist-fallback", type=float, default=0.05)
    parser.add_argument("--max-proxy-shortlist-candidate-rows", type=float, default=64.0)
    parser.add_argument("--min-live-proxy-top-k", type=int, default=10)
    parser.add_argument("--max-overlap-proxy-reduction-ms", type=float, default=5.0)
    parser.add_argument("--require-complete", action="store_true", help="exit non-zero unless every contract gate passes")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    payload = check_contract(args)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_markdown(payload)
        if args.json_out:
            print()
            print(f"wrote: {Path(args.json_out).resolve()}")
    return 1 if args.require_complete and not payload["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
