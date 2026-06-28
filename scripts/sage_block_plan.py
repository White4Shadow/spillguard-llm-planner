#!/usr/bin/env python3
"""
SAGE active-block planner.

The block indexer tells us how large model components are. This planner turns
that into a concrete sparse-oracle candidate: which blocks would be active under
a fixed byte budget?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sage_gguf_blocks import GroupInfo, parse_gguf


COMPONENT_WEIGHTS = {
    "balanced": {
        "attention": 4.0,
        "ffn": 2.5,
        "norm": 100.0,
        "rope": 100.0,
        "embedding": 0.5,
        "output": 0.5,
        "other": 1.0,
    },
    "attention-first": {
        "attention": 8.0,
        "ffn": 1.0,
        "norm": 100.0,
        "rope": 100.0,
        "embedding": 0.5,
        "output": 0.5,
        "other": 1.0,
    },
    "ffn-first": {
        "attention": 1.5,
        "ffn": 8.0,
        "norm": 100.0,
        "rope": 100.0,
        "embedding": 0.5,
        "output": 0.5,
        "other": 1.0,
    },
    "boundary": {
        "attention": 3.0,
        "ffn": 3.0,
        "norm": 100.0,
        "rope": 100.0,
        "embedding": 0.5,
        "output": 0.5,
        "other": 1.0,
    },
}


@dataclass
class PlannedBlock:
    key: str
    layer: int | None
    component: str
    n_tensors: int
    n_bytes: int
    score: float


@dataclass
class BlockPlan:
    model: str
    policy: str
    budget_bytes: int
    used_bytes: int
    ffn_min_share: float
    attention_max_share: float
    selected_count: int
    omitted_count: int
    selected: list[PlannedBlock]
    omitted: list[PlannedBlock]


def human_gib(size: int | float) -> str:
    return f"{size / (1024**3):.2f} GiB"


def is_boundary(layer: int | None, layer_count: int, boundary_layers: int) -> bool:
    if layer is None:
        return False
    return layer < boundary_layers or layer >= layer_count - boundary_layers


def include_global(group: GroupInfo, mode: str) -> bool:
    if group.layer is not None:
        return True
    if mode == "all":
        return True
    if mode == "none":
        return False
    return group.component == mode


def score_group(group: GroupInfo, policy: str, layer_count: int, boundary_layers: int) -> float:
    weights = COMPONENT_WEIGHTS[policy]
    score = weights.get(group.component, weights["other"])
    boundary = is_boundary(group.layer, layer_count, boundary_layers)
    if boundary:
        if policy == "boundary":
            score *= 8.0
        else:
            score *= 2.0
    if group.layer is not None and layer_count > 1:
        # Mild edge preference: early and late layers are often more fragile.
        center = (layer_count - 1) / 2.0
        distance = abs(group.layer - center) / max(center, 1.0)
        score *= 1.0 + 0.25 * distance
    return score


def make_plan(
    *,
    model: Path,
    budget_gib: float,
    policy: str,
    boundary_layers: int,
    global_mode: str,
    ffn_min_share: float,
    attention_max_share: float,
) -> BlockPlan:
    if not 0.0 <= ffn_min_share <= 1.0:
        raise ValueError("ffn_min_share must be in [0, 1]")
    if not 0.0 <= attention_max_share <= 1.0:
        raise ValueError("attention_max_share must be in [0, 1]")
    index = parse_gguf(model)
    budget_bytes = int(budget_gib * 1024**3)
    ffn_min_bytes = int(budget_bytes * ffn_min_share)
    attention_max_bytes = int(budget_bytes * attention_max_share)
    candidates: list[PlannedBlock] = []
    for group in index.groups:
        if not include_global(group, global_mode):
            continue
        candidates.append(
            PlannedBlock(
                key=group.key,
                layer=group.layer,
                component=group.component,
                n_tensors=group.n_tensors,
                n_bytes=group.n_bytes,
                score=score_group(group, policy, index.layer_count, boundary_layers),
            )
        )

    candidates.sort(key=lambda item: (item.score / max(item.n_bytes, 1), item.score, -item.n_bytes), reverse=True)
    selected: list[PlannedBlock] = []
    selected_ids: set[str] = set()
    used = 0
    attention_used = 0
    ffn_used = 0

    def can_select(block: PlannedBlock) -> bool:
        if block.key in selected_ids:
            return False
        if used + block.n_bytes > budget_bytes:
            return False
        if block.component == "attention" and attention_used + block.n_bytes > attention_max_bytes:
            return False
        return True

    def select(block: PlannedBlock) -> None:
        nonlocal used, attention_used, ffn_used
        selected.append(block)
        selected_ids.add(block.key)
        used += block.n_bytes
        if block.component == "attention":
            attention_used += block.n_bytes
        elif block.component == "ffn":
            ffn_used += block.n_bytes

    if ffn_min_bytes > 0:
        for block in [item for item in candidates if item.component == "ffn"]:
            if ffn_used >= ffn_min_bytes:
                break
            if can_select(block):
                select(block)

    for block in candidates:
        if can_select(block):
            select(block)

    omitted = [block for block in candidates if block.key not in selected_ids]

    return BlockPlan(
        model=str(model.resolve()),
        policy=policy,
        budget_bytes=budget_bytes,
        used_bytes=used,
        ffn_min_share=ffn_min_share,
        attention_max_share=attention_max_share,
        selected_count=len(selected),
        omitted_count=len(omitted),
        selected=selected,
        omitted=omitted,
    )


def component_summary(blocks: list[PlannedBlock]) -> dict[str, tuple[int, int]]:
    summary: dict[str, tuple[int, int]] = {}
    for block in blocks:
        count, n_bytes = summary.get(block.component, (0, 0))
        summary[block.component] = (count + 1, n_bytes + block.n_bytes)
    return summary


def print_plan(plan: BlockPlan, top: int) -> None:
    print("# SAGE Active Block Plan")
    print()
    print(f"- Model: `{Path(plan.model).name}`")
    print(f"- Policy: `{plan.policy}`")
    print(f"- Budget: `{human_gib(plan.budget_bytes)}`")
    print(f"- Used: `{human_gib(plan.used_bytes)}`")
    print(f"- FFN minimum share: `{plan.ffn_min_share:.0%}`")
    print(f"- Attention maximum share: `{plan.attention_max_share:.0%}`")
    print(f"- Selected blocks: `{plan.selected_count}`")
    print(f"- Omitted candidate blocks: `{plan.omitted_count}`")
    print()
    print("## Selected Component Summary")
    print()
    print("| Component | Blocks | Bytes |")
    print("| --- | ---: | ---: |")
    for component, (count, n_bytes) in sorted(component_summary(plan.selected).items(), key=lambda item: item[1][1], reverse=True):
        print(f"| {component} | {count} | {human_gib(n_bytes)} |")
    print()
    print("## Selected Blocks")
    print()
    print("| Block | Component | Layer | Bytes | Score |")
    print("| --- | --- | ---: | ---: | ---: |")
    for block in sorted(plan.selected, key=lambda item: (999999 if item.layer is None else item.layer, item.component))[:top]:
        layer = "" if block.layer is None else str(block.layer)
        print(f"| {block.key} | {block.component} | {layer} | {human_gib(block.n_bytes)} | {block.score:.2f} |")
    print()
    print("## Largest Omitted Blocks")
    print()
    print("| Block | Component | Layer | Bytes | Score |")
    print("| --- | --- | ---: | ---: | ---: |")
    for block in sorted(plan.omitted, key=lambda item: item.n_bytes, reverse=True)[:top]:
        layer = "" if block.layer is None else str(block.layer)
        print(f"| {block.key} | {block.component} | {layer} | {human_gib(block.n_bytes)} | {block.score:.2f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan active SAGE oracle blocks under a byte budget.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-gib", type=float, default=3.83)
    parser.add_argument("--policy", choices=sorted(COMPONENT_WEIGHTS), default="balanced")
    parser.add_argument("--boundary-layers", type=int, default=4)
    parser.add_argument("--include-global", choices=["none", "embedding", "output", "all"], default="none")
    parser.add_argument("--ffn-min-share", type=float, default=0.0, help="reserve at least this fraction of the byte budget for FFN blocks when possible")
    parser.add_argument("--attention-max-share", type=float, default=1.0, help="cap attention blocks at this fraction of the byte budget")
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    try:
        plan = make_plan(
            model=Path(args.model),
            budget_gib=args.budget_gib,
            policy=args.policy,
            boundary_layers=args.boundary_layers,
            global_mode=args.include_global,
            ffn_min_share=args.ffn_min_share,
            attention_max_share=args.attention_max_share,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json or args.json_out:
        text = json.dumps(asdict(plan), indent=2)
        if args.json:
            print(text)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote: {out.resolve()}")
    else:
        print_plan(plan, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
