#!/usr/bin/env python3
"""
Build a concrete sparse-verifier manifest from a GGUF tensor table.

The SAGE verifier is intentionally much smaller than a partial forward pass.
This tool maps a 100B-equivalent active-byte budget onto real local GGUF tensor
groups and emits the exact layer/component blocks that a llama.cpp prototype
would need to page, pin, or probe.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sage_gguf_blocks import GroupInfo, ModelIndex, TensorInfo, parse_gguf


GRAPH_PROBE_POINTS = [
    "attn_norm",
    "attn_out",
    "ffn_norm",
    "ffn_norm_1",
    "ffn_norm_2",
    "ffn_out",
    "ffn_mlp",
    "ffn_moe",
    "l_out",
    "result_output",
]


@dataclass
class TensorRef:
    name: str
    tensor_type: str
    shape: list[int]
    n_bytes: int


@dataclass
class SelectedGroup:
    key: str
    layer: int | None
    component: str
    role: str
    n_tensors: int
    n_bytes: int
    tensors: list[TensorRef]


@dataclass
class VerifierManifest:
    model: str
    architecture: str
    policy: str
    layer_order: str
    reference_params_b: float
    reference_quant_bpw: float
    reference_weight_bytes: int
    active_percent: float
    budget_bytes: int
    used_bytes: int
    selected_count: int
    selected_layers: list[int]
    selected_groups: list[SelectedGroup]
    graph_probe_points: list[str]
    debug_tensor_filters: list[str]
    debug_tensor_filter_regex: str
    runtime_contract: list[str]


def human_gib(size: int | float) -> str:
    return f"{size / (1024**3):.2f} GiB"


def reference_weight_bytes(params_b: float, quant_bpw: float) -> int:
    return int(params_b * 1_000_000_000 * quant_bpw / 8.0)


def layer_sequence(layer_count: int, mode: str) -> list[int]:
    layers = list(range(layer_count))
    if mode == "early":
        return layers
    if mode == "late":
        return list(reversed(layers))
    if mode == "middle":
        center = (layer_count - 1) / 2.0
        return sorted(layers, key=lambda layer: (abs(layer - center), layer))
    if mode == "boundary":
        ordered: list[int] = []
        left = 0
        right = layer_count - 1
        while left <= right:
            ordered.append(left)
            if right != left:
                ordered.append(right)
            left += 1
            right -= 1
        return ordered
    raise ValueError(f"unknown layer order: {mode}")


def groups_by_key(index: ModelIndex) -> dict[tuple[int | None, str], GroupInfo]:
    return {(group.layer, group.component): group for group in index.groups}


def tensors_for_group(index: ModelIndex, group: GroupInfo) -> list[TensorInfo]:
    return [
        tensor
        for tensor in index.tensors
        if tensor.layer == group.layer and tensor.component == group.component
    ]


def group_to_selected(index: ModelIndex, group: GroupInfo, role: str) -> SelectedGroup:
    return SelectedGroup(
        key=group.key,
        layer=group.layer,
        component=group.component,
        role=role,
        n_tensors=group.n_tensors,
        n_bytes=group.n_bytes,
        tensors=[
            TensorRef(
                name=tensor.name,
                tensor_type=tensor.tensor_type,
                shape=tensor.shape,
                n_bytes=tensor.n_bytes,
            )
            for tensor in tensors_for_group(index, group)
        ],
    )


def component_package(
    group_map: dict[tuple[int | None, str], GroupInfo],
    layer: int,
    component: str,
    selected_keys: set[str],
) -> list[tuple[GroupInfo, str]]:
    component_group = group_map.get((layer, component))
    if component_group is None or component_group.key in selected_keys:
        return []

    package: list[tuple[GroupInfo, str]] = []
    norm_group = group_map.get((layer, "norm"))
    if norm_group is not None and norm_group.key not in selected_keys:
        package.append((norm_group, f"{component}-normalizer"))
    package.append((component_group, f"{component}-sentinel"))
    return package


def add_package(
    *,
    index: ModelIndex,
    selected: list[SelectedGroup],
    selected_keys: set[str],
    used_bytes: int,
    budget_bytes: int,
    group_map: dict[tuple[int | None, str], GroupInfo],
    layer: int,
    component: str,
) -> int:
    package = component_package(group_map, layer, component, selected_keys)
    if not package:
        return used_bytes

    package_bytes = sum(group.n_bytes for group, _role in package)
    if used_bytes + package_bytes > budget_bytes:
        return used_bytes

    for group, role in package:
        selected.append(group_to_selected(index, group, role))
        selected_keys.add(group.key)
        used_bytes += group.n_bytes
    return used_bytes


def build_manifest(
    *,
    model: Path,
    policy: str,
    layer_order: str,
    params_b: float,
    quant_bpw: float,
    active_percent: float,
) -> VerifierManifest:
    index = parse_gguf(model)
    ref_bytes = reference_weight_bytes(params_b, quant_bpw)
    budget_bytes = int(ref_bytes * active_percent / 100.0)
    order = layer_sequence(index.layer_count, layer_order)
    group_map = groups_by_key(index)
    selected: list[SelectedGroup] = []
    selected_keys: set[str] = set()
    used_bytes = 0

    def try_component(layer: int, component: str) -> None:
        nonlocal used_bytes
        used_bytes = add_package(
            index=index,
            selected=selected,
            selected_keys=selected_keys,
            used_bytes=used_bytes,
            budget_bytes=budget_bytes,
            group_map=group_map,
            layer=layer,
            component=component,
        )

    if policy == "ffn-sentinel":
        for layer in order:
            try_component(layer, "ffn")
    elif policy == "attention-sentinel":
        for layer in order:
            try_component(layer, "attention")
    elif policy == "hybrid":
        # The first hybrid prototype spends enough bytes on one FFN sentinel,
        # then uses the remaining budget on attention sentinels. If anything is
        # still free, additional FFN sentinels are added in the same layer order.
        for layer in order:
            before = used_bytes
            try_component(layer, "ffn")
            if used_bytes != before:
                break
        for layer in order:
            try_component(layer, "attention")
        for layer in order:
            try_component(layer, "ffn")
    else:
        raise ValueError(f"unknown policy: {policy}")

    selected_layers = sorted({group.layer for group in selected if group.layer is not None})
    debug_tensor_filters = build_debug_tensor_filters(selected)
    debug_tensor_filter_regex = combine_debug_tensor_filters(debug_tensor_filters)
    runtime_contract = [
        "Run the resident proxy first and create verifier candidates only for routed tokens.",
        "Load only the selected GGUF tensor groups plus their normalizers for the verifier path.",
        "Probe llama.cpp graph nodes at the listed hook names for the selected layers.",
        "Score the candidate with a small verifier head or learned threshold; do not load the full LM head.",
        "If the verifier rejects or is unavailable, fall back to the larger sparse oracle or exact slow path.",
    ]
    return VerifierManifest(
        model=str(model.resolve()),
        architecture=str(index.metadata.get("general.architecture", "unknown")),
        policy=policy,
        layer_order=layer_order,
        reference_params_b=params_b,
        reference_quant_bpw=quant_bpw,
        reference_weight_bytes=ref_bytes,
        active_percent=active_percent,
        budget_bytes=budget_bytes,
        used_bytes=used_bytes,
        selected_count=len(selected),
        selected_layers=selected_layers,
        selected_groups=selected,
        graph_probe_points=GRAPH_PROBE_POINTS,
        debug_tensor_filters=debug_tensor_filters,
        debug_tensor_filter_regex=debug_tensor_filter_regex,
        runtime_contract=runtime_contract,
    )


def build_debug_tensor_filters(groups: list[SelectedGroup]) -> list[str]:
    by_layer: dict[int, set[str]] = {}
    for group in groups:
        if group.layer is None:
            continue
        by_layer.setdefault(group.layer, set()).add(group.component)

    filters: list[str] = []
    for layer in sorted(by_layer):
        components = by_layer[layer]
        if "attention" in components:
            filters.extend([f"attn_norm-{layer}$", f"attn_out-{layer}$"])
        if "ffn" in components:
            filters.extend(
                [
                    f"ffn_norm-{layer}$",
                    f"ffn_norm_1-{layer}$",
                    f"ffn_norm_2-{layer}$",
                    f"ffn_out-{layer}$",
                    f"ffn_mlp-{layer}$",
                    f"ffn_moe-{layer}$",
                ]
            )
        if "attention" in components or "ffn" in components:
            filters.append(f"l_out-{layer}$")
    return filters


def combine_debug_tensor_filters(filters: list[str]) -> str:
    if not filters:
        return ""
    parts = [item[:-1] if item.endswith("$") else item for item in filters]
    return "(" + "|".join(parts) + ")$"


def component_summary(groups: list[SelectedGroup]) -> dict[str, tuple[int, int]]:
    summary: dict[str, tuple[int, int]] = {}
    for group in groups:
        count, n_bytes = summary.get(group.component, (0, 0))
        summary[group.component] = (count + 1, n_bytes + group.n_bytes)
    return summary


def print_manifest(manifest: VerifierManifest, top_tensors: int) -> None:
    print("# SAGE Sparse Verifier Manifest")
    print()
    print(f"- Model: `{Path(manifest.model).name}`")
    print(f"- Architecture: `{manifest.architecture}`")
    print(f"- Policy: `{manifest.policy}`")
    print(f"- Layer order: `{manifest.layer_order}`")
    print(
        f"- Reference model: `{manifest.reference_params_b:g}B` at "
        f"`{manifest.reference_quant_bpw:g}` bpw (`{human_gib(manifest.reference_weight_bytes)}`)"
    )
    print(f"- Active verifier budget: `{manifest.active_percent:g}%` (`{human_gib(manifest.budget_bytes)}`)")
    print(f"- Used by manifest: `{human_gib(manifest.used_bytes)}`")
    print(f"- Selected layers: `{', '.join(str(layer) for layer in manifest.selected_layers)}`")
    print()
    print("## Component Summary")
    print()
    print("| Component | Groups | Bytes |")
    print("| --- | ---: | ---: |")
    for component, (count, n_bytes) in sorted(component_summary(manifest.selected_groups).items()):
        print(f"| {component} | {count} | {human_gib(n_bytes)} |")
    print()
    print("## Selected Groups")
    print()
    print("| Group | Role | Layer | Tensors | Bytes |")
    print("| --- | --- | ---: | ---: | ---: |")
    for group in manifest.selected_groups:
        layer = "" if group.layer is None else str(group.layer)
        print(f"| {group.key} | {group.role} | {layer} | {group.n_tensors} | {human_gib(group.n_bytes)} |")
    print()
    print("## Tensor Names")
    print()
    for group in manifest.selected_groups:
        print(f"### {group.key}")
        for tensor in group.tensors[:top_tensors]:
            shape = "x".join(str(dim) for dim in tensor.shape)
            print(f"- `{tensor.name}` `{tensor.tensor_type}` `{shape}` `{human_gib(tensor.n_bytes)}`")
        if len(group.tensors) > top_tensors:
            print(f"- ... {len(group.tensors) - top_tensors} more tensors")
        print()
    print("## llama.cpp Probe Points")
    print()
    print(", ".join(f"`{name}`" for name in manifest.graph_probe_points))
    print()
    print("## llama-debug Tensor Filters")
    print()
    if manifest.debug_tensor_filter_regex:
        print(f"Use one filter argument: `--tensor-filter \"{manifest.debug_tensor_filter_regex}\"`")
        print()
    for item in manifest.debug_tensor_filters:
        print(f"- `{item}`")
    print()
    print("## Runtime Contract")
    print()
    for item in manifest.runtime_contract:
        print(f"- {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a SAGE sparse-verifier manifest from a GGUF model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=["ffn-sentinel", "attention-sentinel", "hybrid"], default="hybrid")
    parser.add_argument("--layer-order", choices=["boundary", "early", "late", "middle"], default="boundary")
    parser.add_argument("--params-b", type=float, default=100.0, help="reference giant model size in billions of parameters")
    parser.add_argument("--quant-bpw", type=float, default=2.0, help="reference giant model quantization in bits per weight")
    parser.add_argument("--active-percent", type=float, default=1.0, help="verifier budget as percent of reference giant weights")
    parser.add_argument("--top-tensors", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.active_percent <= 0:
        parser.error("--active-percent must be positive")

    manifest = build_manifest(
        model=Path(args.model),
        policy=args.policy,
        layer_order=args.layer_order,
        params_b=args.params_b,
        quant_bpw=args.quant_bpw,
        active_percent=args.active_percent,
    )

    if args.json or args.json_out:
        text = json.dumps(asdict(manifest), indent=2)
        if args.json:
            print(text)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote: {out.resolve()}")
    else:
        print_manifest(manifest, args.top_tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
