#!/usr/bin/env python3
"""
Plan tiny sparse-verifier budgets against a real GGUF tensor layout.

The policy target says the sparse verifier likely needs to touch around 0.5-1%
of a 100B 2-bit model. This tool converts that percentage into GiB and asks:

- how many real Gemma 31B blocks fit in the same byte budget?
- is a full layer possible?
- which static block policies fit?
- what is the minimum budget for one FFN, three attention sentinels, or one
  complete layer?

It is a design/falsification tool. It does not execute sparse verification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sage_block_plan import make_plan
from sage_gguf_blocks import GroupInfo, parse_gguf


BYTES_PER_GIB = 1024**3


@dataclass
class ComponentFit:
    component: str
    groups_fit: int
    bytes_fit: int
    total_groups: int


@dataclass
class PolicyFit:
    policy: str
    used_bytes: int
    selected_count: int
    ffn_blocks: int
    attention_blocks: int
    norm_blocks: int
    other_blocks: int


@dataclass
class BudgetFit:
    active_percent: float
    budget_gib: float
    budget_bytes: int
    local_model_tensor_share: float
    full_layers_fit: int
    component_fits: list[ComponentFit]
    policy_fits: list[PolicyFit]


def parse_float_list(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def parse_str_list(value: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one value")
    return out


def dense_weight_gib(params_b: float, quant_bpw: float) -> float:
    return params_b * 1_000_000_000 * quant_bpw / 8.0 / BYTES_PER_GIB


def human_gib(n_bytes: int | float) -> str:
    return f"{n_bytes / BYTES_PER_GIB:.2f} GiB"


def component_fits(groups: list[GroupInfo], budget_bytes: int) -> list[ComponentFit]:
    by_component: dict[str, list[GroupInfo]] = {}
    for group in groups:
        by_component.setdefault(group.component, []).append(group)

    fits: list[ComponentFit] = []
    for component, component_groups in by_component.items():
        used = 0
        count = 0
        for group in sorted(component_groups, key=lambda item: item.n_bytes):
            if used + group.n_bytes > budget_bytes:
                break
            used += group.n_bytes
            count += 1
        fits.append(ComponentFit(component=component, groups_fit=count, bytes_fit=used, total_groups=len(component_groups)))
    fits.sort(key=lambda item: item.bytes_fit, reverse=True)
    return fits


def full_layers_fit(groups: list[GroupInfo], budget_bytes: int) -> int:
    layer_bytes: dict[int, int] = {}
    for group in groups:
        if group.layer is not None:
            layer_bytes[group.layer] = layer_bytes.get(group.layer, 0) + group.n_bytes
    used = 0
    count = 0
    for _, n_bytes in sorted(layer_bytes.items(), key=lambda item: item[1]):
        if used + n_bytes > budget_bytes:
            break
        used += n_bytes
        count += 1
    return count


def policy_fit(model: Path, budget_gib: float, policy: str) -> PolicyFit:
    plan = make_plan(
        model=model,
        budget_gib=budget_gib,
        policy=policy,
        boundary_layers=4,
        global_mode="none",
        ffn_min_share=0.0,
        attention_max_share=1.0,
    )
    counts = {"ffn": 0, "attention": 0, "norm": 0, "other": 0}
    for block in plan.selected:
        if block.component in counts:
            counts[block.component] += 1
        else:
            counts["other"] += 1
    return PolicyFit(
        policy=policy,
        used_bytes=plan.used_bytes,
        selected_count=plan.selected_count,
        ffn_blocks=counts["ffn"],
        attention_blocks=counts["attention"],
        norm_blocks=counts["norm"],
        other_blocks=counts["other"],
    )


def make_budget_fit(
    *,
    model: Path,
    groups: list[GroupInfo],
    tensor_bytes: int,
    giant_dense_gib: float,
    active_percent: float,
    policies: list[str],
) -> BudgetFit:
    budget_gib = giant_dense_gib * active_percent / 100.0
    budget_bytes = int(budget_gib * BYTES_PER_GIB)
    return BudgetFit(
        active_percent=active_percent,
        budget_gib=budget_gib,
        budget_bytes=budget_bytes,
        local_model_tensor_share=budget_bytes / tensor_bytes if tensor_bytes else 0.0,
        full_layers_fit=full_layers_fit(groups, budget_bytes),
        component_fits=component_fits(groups, budget_bytes),
        policy_fits=[policy_fit(model, budget_gib, policy) for policy in policies],
    )


def min_combo_budgets(groups: list[GroupInfo], giant_dense_gib: float) -> dict[str, dict[str, float]]:
    by_component: dict[str, list[int]] = {}
    layer_bytes: dict[int, int] = {}
    for group in groups:
        by_component.setdefault(group.component, []).append(group.n_bytes)
        if group.layer is not None:
            layer_bytes[group.layer] = layer_bytes.get(group.layer, 0) + group.n_bytes

    def entry(n_bytes: int) -> dict[str, float]:
        gib = n_bytes / BYTES_PER_GIB
        return {"gib": gib, "active_percent_of_giant": 100.0 * gib / giant_dense_gib if giant_dense_gib else 0.0}

    ffn = sorted(by_component.get("ffn", []))
    attn = sorted(by_component.get("attention", []))
    layers = sorted(layer_bytes.values())
    combos: dict[str, dict[str, float]] = {}
    if ffn:
        combos["one_ffn_group"] = entry(ffn[0])
    if attn:
        combos["one_attention_group"] = entry(attn[0])
    if len(attn) >= 3:
        combos["three_attention_groups"] = entry(sum(attn[:3]))
    if ffn and attn:
        combos["one_ffn_plus_one_attention"] = entry(ffn[0] + attn[0])
    if layers:
        combos["one_full_layer"] = entry(layers[0])
    return combos


def print_report(model: Path, giant_dense_gib: float, fits: list[BudgetFit], combos: dict[str, dict[str, float]]) -> None:
    print("# SAGE Sparse Verifier Plan")
    print()
    print(f"- Local model: `{model.name}`")
    print(f"- Giant dense reference: `{giant_dense_gib:.2f} GiB`")
    print()
    print("## Minimum Useful Building Blocks")
    print()
    print("| Candidate | GiB | Active % of giant |")
    print("| --- | ---: | ---: |")
    for name, data in combos.items():
        print(f"| {name} | {data['gib']:.3f} | {data['active_percent_of_giant']:.2f}% |")
    print()
    print("## Budget Fits")
    print()
    print("| Active % | Budget | Local tensor share | Full layers | FFN groups | Attention groups |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for fit in fits:
        components = {item.component: item for item in fit.component_fits}
        ffn = components.get("ffn")
        attn = components.get("attention")
        print(
            f"| {fit.active_percent:.2f}% | {fit.budget_gib:.3f} GiB | {fit.local_model_tensor_share:.1%} "
            f"| {fit.full_layers_fit} | {0 if ffn is None else ffn.groups_fit} | {0 if attn is None else attn.groups_fit} |"
        )
    print()
    print("## Static Policy Fits")
    print()
    print("| Active % | Policy | Used | FFN | Attention | Norm | Other |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for fit in fits:
        for policy in fit.policy_fits:
            print(
                f"| {fit.active_percent:.2f}% | {policy.policy} | {human_gib(policy.used_bytes)} "
                f"| {policy.ffn_blocks} | {policy.attention_blocks} | {policy.norm_blocks} | {policy.other_blocks} |"
            )
    print()
    print("## Interpretation")
    print()
    print("A 1% 100B-equivalent verifier budget is smaller than one full Gemma 31B layer. It can fit roughly one FFN group or three attention groups. A useful verifier at this budget therefore cannot be a normal partial forward pass; it needs to be a micro-verifier such as boundary sentinels, selected FFN probes, low-rank logit correction, or another learned check.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan tiny sparse-verifier budgets against a GGUF model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--params-b", type=float, default=100.0)
    parser.add_argument("--quant-bpw", type=float, default=2.0)
    parser.add_argument("--active-percents", type=parse_float_list, default=parse_float_list("0.5,1,2,4"))
    parser.add_argument("--policies", type=parse_str_list, default=parse_str_list("ffn-first,attention-first,boundary"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.params_b <= 0:
        parser.error("--params-b must be positive")
    if args.quant_bpw <= 0:
        parser.error("--quant-bpw must be positive")
    for percent in args.active_percents:
        if percent < 0:
            parser.error("--active-percents must be non-negative")

    model = Path(args.model)
    index = parse_gguf(model)
    giant_dense_gib = dense_weight_gib(args.params_b, args.quant_bpw)
    fits = [
        make_budget_fit(
            model=model,
            groups=index.groups,
            tensor_bytes=index.total_tensor_bytes,
            giant_dense_gib=giant_dense_gib,
            active_percent=percent,
            policies=args.policies,
        )
        for percent in args.active_percents
    ]
    combos = min_combo_budgets(index.groups, giant_dense_gib)
    if args.json:
        print(json.dumps({"model": str(model.resolve()), "giant_dense_gib": giant_dense_gib, "combos": combos, "fits": [asdict(fit) for fit in fits]}, indent=2))
    else:
        print_report(model, giant_dense_gib, fits, combos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
