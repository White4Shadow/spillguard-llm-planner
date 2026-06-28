#!/usr/bin/env python3
"""
GGUF block indexer for SAGE-100.

SAGE treats a giant model as addressable blocks rather than one monolithic
weight blob. This tool reads the GGUF tensor table with only the Python standard
library, groups tensors by layer and component, and reports the byte sizes that
an active-byte scheduler would need to page or keep hot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Any


GGUF_MAGIC = 0x46554747
DEFAULT_ALIGNMENT = 32
QK_K = 256

VALUE_TYPE_NAMES = {
    0: "UINT8",
    1: "INT8",
    2: "UINT16",
    3: "INT16",
    4: "UINT32",
    5: "INT32",
    6: "FLOAT32",
    7: "BOOL",
    8: "STRING",
    9: "ARRAY",
    10: "UINT64",
    11: "INT64",
    12: "FLOAT64",
}

SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

QUANT_SIZES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 2 + 16),
    3: ("Q4_1", 32, 2 + 2 + 16),
    6: ("Q5_0", 32, 2 + 4 + 16),
    7: ("Q5_1", 32, 2 + 2 + 4 + 16),
    8: ("Q8_0", 32, 2 + 32),
    9: ("Q8_1", 32, 4 + 4 + 32),
    10: ("Q2_K", 256, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: ("Q3_K", 256, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", 256, 2 + 2 + QK_K // 2 + 12),
    13: ("Q5_K", 256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", 256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    15: ("Q8_K", 256, 4 + QK_K + QK_K // 8),
    16: ("IQ2_XXS", 256, 2 + QK_K // 4),
    17: ("IQ2_XS", 256, 2 + QK_K // 4 + QK_K // 32),
    18: ("IQ3_XXS", 256, 2 + QK_K // 4 + QK_K // 8),
    19: ("IQ1_S", 256, 2 + QK_K // 8 + QK_K // 16),
    20: ("IQ4_NL", 32, 2 + 16),
    21: ("IQ3_S", 256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: ("IQ2_S", 256, 2 + QK_K // 4 + QK_K // 16),
    23: ("IQ4_XS", 256, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 2 + 4 * 13),
    35: ("TQ2_0", 256, 2 + 64),
    39: ("MXFP4", 32, 1 + 16),
    40: ("NVFP4", 64, 4 + 32),
    41: ("Q1_0", 128, 2 + 16),
}

LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass
class TensorInfo:
    name: str
    layer: int | None
    component: str
    tensor_type: str
    shape: list[int]
    n_elements: int
    n_bytes: int
    offset: int


@dataclass
class GroupInfo:
    key: str
    layer: int | None
    component: str
    n_tensors: int
    n_bytes: int


@dataclass
class ModelIndex:
    path: str
    version: int
    tensor_count: int
    metadata_count: int
    alignment: int
    metadata: dict[str, Any]
    total_tensor_bytes: int
    layer_count: int
    global_bytes: int
    tensors: list[TensorInfo]
    groups: list[GroupInfo]


def read_exact(handle: BinaryIO, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError("unexpected end of GGUF file")
    return data


def read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_scalar(handle: BinaryIO, value_type: int) -> Any:
    fmt = SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported scalar value type {value_type}")
    return struct.unpack("<" + fmt, read_exact(handle, struct.calcsize(fmt)))[0]


def read_string(handle: BinaryIO) -> str:
    length = read_u64(handle)
    return read_exact(handle, length).decode("utf-8", errors="replace")


def read_value(handle: BinaryIO, value_type: int, keep_arrays: bool = False) -> Any:
    if value_type == 8:
        return read_string(handle)
    if value_type == 9:
        subtype = read_u32(handle)
        length = read_u64(handle)
        values: list[Any] = []
        keep = keep_arrays and length <= 128
        for _ in range(length):
            value = read_value(handle, subtype, keep_arrays=False)
            if keep:
                values.append(value)
        if keep:
            return values
        return {"type": VALUE_TYPE_NAMES.get(subtype, str(subtype)), "count": length}
    return read_scalar(handle, value_type)


def tensor_nbytes(shape: list[int], tensor_type: int) -> int:
    try:
        _name, block_size, type_size = QUANT_SIZES[tensor_type]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor type id {tensor_type}") from exc
    n_elements = math.prod(shape)
    return math.ceil(n_elements / block_size) * type_size


def classify_component(name: str, suffix: str | None) -> str:
    text = suffix if suffix is not None else name
    if "attn" in text:
        return "attention"
    if "ffn" in text:
        return "ffn"
    if "norm" in text or "scale" in text:
        return "norm"
    if "token_embd" in text:
        return "embedding"
    if text.startswith("output"):
        return "output"
    if "rope" in text:
        return "rope"
    return "other"


def parse_gguf(path: Path, keep_metadata_arrays: bool = False) -> ModelIndex:
    tensors: list[TensorInfo] = []
    metadata: dict[str, Any] = {}
    with path.open("rb") as handle:
        magic = read_u32(handle)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32(handle)
        tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)

        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            metadata[key] = read_value(handle, value_type, keep_arrays=keep_metadata_arrays)

        tensor_specs: list[tuple[str, list[int], int, int]] = []
        for _ in range(tensor_count):
            name = read_string(handle)
            n_dims = read_u32(handle)
            shape = [read_u64(handle) for _ in range(n_dims)]
            tensor_type = read_u32(handle)
            offset = read_u64(handle)
            tensor_specs.append((name, shape, tensor_type, offset))

        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        for name, shape, tensor_type, offset in tensor_specs:
            match = LAYER_RE.match(name)
            layer = int(match.group(1)) if match else None
            suffix = match.group(2) if match else None
            qname = QUANT_SIZES.get(tensor_type, (f"TYPE_{tensor_type}", 1, 1))[0]
            tensors.append(
                TensorInfo(
                    name=name,
                    layer=layer,
                    component=classify_component(name, suffix),
                    tensor_type=qname,
                    shape=shape,
                    n_elements=math.prod(shape),
                    n_bytes=tensor_nbytes(shape, tensor_type),
                    offset=offset,
                )
            )

    group_map: dict[tuple[int | None, str], list[TensorInfo]] = {}
    for tensor in tensors:
        group_map.setdefault((tensor.layer, tensor.component), []).append(tensor)

    groups: list[GroupInfo] = []
    for (layer, component), items in group_map.items():
        key = f"blk.{layer}.{component}" if layer is not None else f"global.{component}"
        groups.append(
            GroupInfo(
                key=key,
                layer=layer,
                component=component,
                n_tensors=len(items),
                n_bytes=sum(item.n_bytes for item in items),
            )
        )
    groups.sort(key=lambda group: (999999 if group.layer is None else group.layer, group.component))

    layer_ids = sorted({tensor.layer for tensor in tensors if tensor.layer is not None})
    global_bytes = sum(tensor.n_bytes for tensor in tensors if tensor.layer is None)
    total = sum(tensor.n_bytes for tensor in tensors)
    public_metadata = {
        key: metadata[key]
        for key in sorted(metadata)
        if key in {"general.architecture", "general.name", "general.file_type", "general.quantization_version"}
        or key.endswith(".block_count")
        or key.endswith(".context_length")
        or key.endswith(".final_logit_softcapping")
        or key == "tokenizer.ggml.suppress_tokens"
    }
    return ModelIndex(
        path=str(path.resolve()),
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        alignment=alignment,
        metadata=public_metadata,
        total_tensor_bytes=total,
        layer_count=len(layer_ids),
        global_bytes=global_bytes,
        tensors=tensors,
        groups=groups,
    )


def human_gib(size: int | float) -> str:
    return f"{size / (1024**3):.2f} GiB"


def print_summary(index: ModelIndex, budget_gib: float) -> None:
    layer_groups = [group for group in index.groups if group.layer is not None]
    layer_totals: dict[int, int] = {}
    for group in layer_groups:
        assert group.layer is not None
        layer_totals[group.layer] = layer_totals.get(group.layer, 0) + group.n_bytes

    sorted_layers = sorted(layer_totals.items())
    avg_layer = sum(layer_totals.values()) / len(layer_totals) if layer_totals else 0
    budget_bytes = int(budget_gib * 1024**3)
    greedy_layers = 0
    greedy_bytes = 0
    for _layer, n_bytes in sorted(layer_totals.items(), key=lambda item: item[1]):
        if greedy_bytes + n_bytes > budget_bytes:
            break
        greedy_layers += 1
        greedy_bytes += n_bytes

    print("# SAGE GGUF Block Index")
    print()
    print(f"- Model: `{Path(index.path).name}`")
    print(f"- Architecture: `{index.metadata.get('general.architecture', 'unknown')}`")
    print(f"- Tensors: `{index.tensor_count}`")
    print(f"- Layer count: `{index.layer_count}`")
    print(f"- Tensor bytes: `{human_gib(index.total_tensor_bytes)}`")
    print(f"- Global/non-layer bytes: `{human_gib(index.global_bytes)}`")
    print(f"- Average repeating layer bytes: `{human_gib(avg_layer)}`")
    print(f"- Active budget: `{budget_gib:.2f} GiB`")
    print(f"- Smallest full layers fitting budget: `{greedy_layers}` / `{index.layer_count}` (`{human_gib(greedy_bytes)}`)")
    print()
    print("## Component Bytes")
    print()
    component_totals: dict[str, int] = {}
    for group in index.groups:
        component_totals[group.component] = component_totals.get(group.component, 0) + group.n_bytes
    print("| Component | Bytes | Share |")
    print("| --- | ---: | ---: |")
    for component, n_bytes in sorted(component_totals.items(), key=lambda item: item[1], reverse=True):
        share = 100.0 * n_bytes / index.total_tensor_bytes if index.total_tensor_bytes else 0.0
        print(f"| {component} | {human_gib(n_bytes)} | {share:.1f}% |")
    print()
    print("## Budget Fit By Component")
    print()
    print("| Component | Groups fit | Bytes fit | Total groups |")
    print("| --- | ---: | ---: | ---: |")
    for component in sorted(component_totals, key=lambda key: component_totals[key], reverse=True):
        component_groups = sorted(
            [group for group in index.groups if group.component == component],
            key=lambda group: group.n_bytes,
        )
        fit_count = 0
        fit_bytes = 0
        for group in component_groups:
            if fit_bytes + group.n_bytes > budget_bytes:
                break
            fit_count += 1
            fit_bytes += group.n_bytes
        print(f"| {component} | {fit_count} | {human_gib(fit_bytes)} | {len(component_groups)} |")
    print()
    print("## Largest Groups")
    print()
    print("| Group | Tensors | Bytes |")
    print("| --- | ---: | ---: |")
    for group in sorted(index.groups, key=lambda item: item.n_bytes, reverse=True)[:20]:
        print(f"| {group.key} | {group.n_tensors} | {human_gib(group.n_bytes)} |")
    print()
    print("## First Layers")
    print()
    print("| Layer | Total | Attention | FFN | Norm/Other |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for layer, total in sorted_layers[:12]:
        attention = sum(g.n_bytes for g in layer_groups if g.layer == layer and g.component == "attention")
        ffn = sum(g.n_bytes for g in layer_groups if g.layer == layer and g.component == "ffn")
        rest = total - attention - ffn
        print(f"| {layer} | {human_gib(total)} | {human_gib(attention)} | {human_gib(ffn)} | {human_gib(rest)} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index GGUF tensors into SAGE layer/component blocks.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-gib", type=float, default=3.83, help="active-byte budget to compare against")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    index = parse_gguf(Path(args.model))
    if args.json or args.json_out:
        payload = asdict(index)
        text = json.dumps(payload, indent=2)
        if args.json:
            print(text)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote: {out.resolve()}")
    else:
        print_summary(index, args.budget_gib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
