#!/usr/bin/env python3
"""
Evaluate whether a resident proxy's top-k tokens form a useful oracle shortlist.

The Q6_K candidate verifier can score a small set of token rows cheaply, but it
only helps if the oracle's preferred token is usually inside the shortlist. This
script measures that coverage from existing sage_logprob_probe artifacts and can
optionally learn a tiny static rescue set from train-split misses.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sage_gguf_blocks import parse_gguf, read_u32, read_u64, read_string, read_value
from sage_oracle_cuda_kernel_smoke import fail
from sage_oracle_cuda_vocab_smoke import find_tensor, q6_k_row_bytes
from sage_oracle_pager_staging import bytes_to_gib

SPIECE_MARKER = "\u2581"
TOKEN_NORM_RE = re.compile(r"[a-z0-9+-]{2,24}")
PROMPT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+-]*")
TOKEN_STRIP = "\"`*_()[]{}.,:;!?"


@dataclass
class Coverage:
    name: str
    steps: int
    covered: int
    missed: int
    coverage_rate: float
    fallback_rate_for_exact: float
    candidate_rows_per_step: float
    candidate_bytes_per_step: float
    active_percent_vocab_tensor: float
    active_percent_reference_100b_2bit: float


@dataclass
class SplitSummary:
    name: str
    prompts: int
    steps: int
    proxy_top1_matches: int
    proxy_top1_match_rate: float
    coverage: list[Coverage]


def expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        paths.extend(matches if matches else [Path(value)])
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


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


def safe_top_ids(step: dict[str, Any], k: int) -> list[int]:
    ids: list[int] = []
    for item in step.get("top_logprobs", [])[:k]:
        try:
            token_id = int(item.get("id", -1))
        except (TypeError, ValueError):
            continue
        if token_id >= 0:
            ids.append(token_id)
    return ids


def token_text_norm(token: str) -> str:
    text = token.replace(SPIECE_MARKER, " ").strip().lower()
    return text.strip(TOKEN_STRIP)


def load_gguf_vocab_tokens(path: Path) -> list[str]:
    with path.open("rb") as handle:
        _magic = read_u32(handle)
        _version = read_u32(handle)
        _tensor_count = read_u64(handle)
        metadata_count = read_u64(handle)
        for _ in range(metadata_count):
            key = read_string(handle)
            value_type = read_u32(handle)
            if key == "tokenizer.ggml.tokens":
                _subtype = read_u32(handle)
                length = read_u64(handle)
                return [read_string(handle) for _ in range(length)]
            read_value(handle, value_type, keep_arrays=False)
    fail(f"tokenizer.ggml.tokens not found in GGUF metadata: {path}")
    return []


def token_rank(token_id: int, token: str) -> tuple[int, int, int]:
    body = token.replace(SPIECE_MARKER, "")
    return (
        0 if token.startswith(SPIECE_MARKER) else 1,
        0 if body[:1].isupper() else 1,
        token_id,
    )


def build_prompt_piece_index(vocab_tokens: list[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for token_id, token in enumerate(vocab_tokens):
        key = token_text_norm(token)
        if key and TOKEN_NORM_RE.fullmatch(key):
            grouped.setdefault(key, []).append((token_id, token))
    return {
        key: [token_id for token_id, _token in sorted(items, key=lambda item: token_rank(item[0], item[1]))]
        for key, items in grouped.items()
    }


def prompt_pieces(prompt: str) -> list[str]:
    pieces: list[str] = []
    for word in (item.lower() for item in PROMPT_WORD_RE.findall(prompt)):
        if len(word) < 2:
            continue
        pieces.append(word)
        for n_chars in (3, 4, 5, 6, 7, 8, 9, 10, 12):
            if len(word) > n_chars:
                pieces.append(word[:n_chars])
        for n_chars in (3, 4, 5, 6):
            if len(word) > n_chars + 2:
                pieces.append(word[-n_chars:])
    return list(dict.fromkeys(pieces))


def prompt_piece_ids(
    prompt: str,
    prompt_piece_index: dict[str, list[int]] | None,
    max_ids: int,
    ids_per_piece: int,
) -> list[int]:
    if not prompt_piece_index or max_ids <= 0:
        return []
    ids: list[int] = []
    for piece in prompt_pieces(prompt):
        ids.extend(prompt_piece_index.get(piece, [])[:ids_per_piece])
    return list(dict.fromkeys(ids))[:max_ids]


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            fail(f"missing logprob JSON: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            if isinstance(row, dict):
                row["_source"] = str(path)
                rows.append(row)
    if not rows:
        fail("no rows found in logprob JSON inputs")
    return rows


def split_rows(rows: list[dict[str, Any]], train_mod: int, train_remainders: set[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_mod <= 1:
        return rows, rows
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index % train_mod in train_remainders:
            train.append(row)
        else:
            eval_rows.append(row)
    if not train or not eval_rows:
        fail("train/eval split produced an empty side")
    return train, eval_rows


def iter_steps(rows: list[dict[str, Any]], ignore_prefix_steps: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_index, row in enumerate(rows):
        proxy_steps = row.get("proxy", {}).get("steps", [])
        oracle_steps = row.get("oracle", {}).get("steps", [])
        limit = min(len(proxy_steps), len(oracle_steps))
        for step_index in range(ignore_prefix_steps, limit):
            proxy_step = proxy_steps[step_index]
            oracle_step = oracle_steps[step_index]
            try:
                proxy_id = int(proxy_step.get("token_id", -1))
                oracle_id = int(oracle_step.get("token_id", -1))
            except (TypeError, ValueError):
                continue
            if proxy_id < 0 or oracle_id < 0:
                continue
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": str(row.get("prompt", "")),
                    "step_index": step_index,
                    "proxy_id": proxy_id,
                    "oracle_id": oracle_id,
                    "proxy_token": str(proxy_step.get("token", "")),
                    "oracle_token": str(oracle_step.get("token", "")),
                    "proxy_top_ids_by_k": proxy_step,
                }
            )
    return records


def coverage_for(
    *,
    name: str,
    records: list[dict[str, Any]],
    k: int,
    static_rescue_ids: list[int],
    position_rescue_by_step: dict[int, list[int]],
    prompt_piece_index: dict[str, list[int]] | None,
    prompt_piece_count: int,
    prompt_piece_ids_per_piece: int,
    row_bytes: int,
    vocab_tensor_bytes: int,
) -> Coverage:
    covered = 0
    total_candidate_rows = 0
    rescue = list(static_rescue_ids)
    for record in records:
        candidate_ids = safe_top_ids(record["proxy_top_ids_by_k"], k)
        candidate_ids.extend(rescue)
        candidate_ids.extend(position_rescue_by_step.get(int(record["step_index"]), []))
        candidate_ids.extend(
            prompt_piece_ids(
                str(record["prompt"]),
                prompt_piece_index,
                prompt_piece_count,
                prompt_piece_ids_per_piece,
            )
        )
        unique_ids = list(dict.fromkeys(candidate_ids))
        total_candidate_rows += len(unique_ids)
        if record["oracle_id"] in unique_ids:
            covered += 1
    steps = len(records)
    candidate_rows_per_step = total_candidate_rows / steps if steps else 0.0
    candidate_bytes = candidate_rows_per_step * row_bytes
    return Coverage(
        name=name,
        steps=steps,
        covered=covered,
        missed=steps - covered,
        coverage_rate=covered / steps if steps else 0.0,
        fallback_rate_for_exact=1.0 - covered / steps if steps else 0.0,
        candidate_rows_per_step=candidate_rows_per_step,
        candidate_bytes_per_step=candidate_bytes,
        active_percent_vocab_tensor=100.0 * candidate_bytes / max(vocab_tensor_bytes, 1),
        active_percent_reference_100b_2bit=100.0 * candidate_bytes / max(int(100_000_000_000 * 2 / 8), 1),
    )


def learn_static_rescue_ids(records: list[dict[str, Any]], proxy_k: int, rescue_count: int) -> list[int]:
    misses: Counter[int] = Counter()
    for record in records:
        proxy_ids = safe_top_ids(record["proxy_top_ids_by_k"], proxy_k)
        oracle_id = int(record["oracle_id"])
        if oracle_id not in proxy_ids:
            misses[oracle_id] += 1
    return [token_id for token_id, _count in misses.most_common(rescue_count)]


def learn_position_rescue_ids(records: list[dict[str, Any]], rescue_count: int) -> dict[int, list[int]]:
    if rescue_count <= 0:
        return {}
    by_step: dict[int, Counter[int]] = {}
    for record in records:
        by_step.setdefault(int(record["step_index"]), Counter())[int(record["oracle_id"])] += 1
    return {
        step: [token_id for token_id, _count in counts.most_common(rescue_count)]
        for step, counts in by_step.items()
    }


def summarize_split(
    *,
    name: str,
    rows: list[dict[str, Any]],
    k_values: list[int],
    static_rescue_ids: list[int],
    position_rescue_by_step: dict[int, list[int]],
    prompt_piece_index: dict[str, list[int]] | None,
    prompt_piece_count: int,
    prompt_piece_ids_per_piece: int,
    row_bytes: int,
    vocab_tensor_bytes: int,
    ignore_prefix_steps: int,
) -> SplitSummary:
    records = iter_steps(rows, ignore_prefix_steps)
    prompts = len(rows)
    proxy_matches = sum(1 for record in records if record["proxy_id"] == record["oracle_id"])
    coverage = [
        coverage_for(
            name=f"proxy_top_{k}",
            records=records,
            k=k,
            static_rescue_ids=[],
            position_rescue_by_step={},
            prompt_piece_index=None,
            prompt_piece_count=0,
            prompt_piece_ids_per_piece=prompt_piece_ids_per_piece,
            row_bytes=row_bytes,
            vocab_tensor_bytes=vocab_tensor_bytes,
        )
        for k in k_values
    ]
    if static_rescue_ids:
        for k in k_values:
            coverage.append(
                coverage_for(
                    name=f"proxy_top_{k}_plus_static_{len(static_rescue_ids)}",
                    records=records,
                    k=k,
                    static_rescue_ids=static_rescue_ids,
                    position_rescue_by_step={},
                    prompt_piece_index=None,
                    prompt_piece_count=0,
                    prompt_piece_ids_per_piece=prompt_piece_ids_per_piece,
                    row_bytes=row_bytes,
                    vocab_tensor_bytes=vocab_tensor_bytes,
                )
            )
    if position_rescue_by_step:
        for k in k_values:
            coverage.append(
                coverage_for(
                    name=f"proxy_top_{k}_plus_pos_{max(len(ids) for ids in position_rescue_by_step.values())}",
                    records=records,
                    k=k,
                    static_rescue_ids=[],
                    position_rescue_by_step=position_rescue_by_step,
                    prompt_piece_index=None,
                    prompt_piece_count=0,
                    prompt_piece_ids_per_piece=prompt_piece_ids_per_piece,
                    row_bytes=row_bytes,
                    vocab_tensor_bytes=vocab_tensor_bytes,
                )
            )
    if position_rescue_by_step and prompt_piece_count > 0:
        position_count = max(len(ids) for ids in position_rescue_by_step.values())
        for k in k_values:
            coverage.append(
                coverage_for(
                    name=f"proxy_top_{k}_plus_pos_{position_count}_prompt_{prompt_piece_count}",
                    records=records,
                    k=k,
                    static_rescue_ids=[],
                    position_rescue_by_step=position_rescue_by_step,
                    prompt_piece_index=prompt_piece_index,
                    prompt_piece_count=prompt_piece_count,
                    prompt_piece_ids_per_piece=prompt_piece_ids_per_piece,
                    row_bytes=row_bytes,
                    vocab_tensor_bytes=vocab_tensor_bytes,
                )
            )
    return SplitSummary(
        name=name,
        prompts=prompts,
        steps=len(records),
        proxy_top1_matches=proxy_matches,
        proxy_top1_match_rate=proxy_matches / len(records) if records else 0.0,
        coverage=coverage,
    )


def class_breakdown(
    rows: list[dict[str, Any]],
    ignore_prefix_steps: int,
    k: int,
    static_rescue_ids: list[int],
    position_rescue_by_step: dict[int, list[int]],
    prompt_piece_index: dict[str, list[int]] | None,
    prompt_piece_count: int,
    prompt_piece_ids_per_piece: int,
) -> dict[str, Any]:
    records = iter_steps(rows, ignore_prefix_steps)
    by_class: dict[str, dict[str, int]] = {}
    for record in records:
        cls = token_class(record["proxy_token"])
        bucket = by_class.setdefault(cls, {"steps": 0, "covered": 0, "proxy_top1_matches": 0})
        bucket["steps"] += 1
        if record["proxy_id"] == record["oracle_id"]:
            bucket["proxy_top1_matches"] += 1
        ids = safe_top_ids(record["proxy_top_ids_by_k"], k)
        ids.extend(static_rescue_ids)
        ids.extend(position_rescue_by_step.get(int(record["step_index"]), []))
        ids.extend(
            prompt_piece_ids(
                str(record["prompt"]),
                prompt_piece_index,
                prompt_piece_count,
                prompt_piece_ids_per_piece,
            )
        )
        if record["oracle_id"] in set(ids):
            bucket["covered"] += 1
    return {
        cls: {
            **values,
            "coverage_rate": values["covered"] / values["steps"] if values["steps"] else 0.0,
            "proxy_top1_match_rate": values["proxy_top1_matches"] / values["steps"] if values["steps"] else 0.0,
        }
        for cls, values in sorted(by_class.items())
    }


def make_payload(args: argparse.Namespace) -> dict[str, Any]:
    paths = expand_inputs(args.logprob_json)
    rows = load_rows(paths)
    k_values = [int(item) for item in args.k_values.split(",") if item.strip()]
    if not k_values or any(k <= 0 for k in k_values):
        fail("--k-values must contain positive integers")

    model_path = Path(args.model)
    index = parse_gguf(model_path)
    tensor = find_tensor(index.tensors, args.tensor_name)
    if tensor.tensor_type != "Q6_K":
        fail(f"{tensor.name} must be Q6_K, got {tensor.tensor_type}")
    row_bytes = q6_k_row_bytes(int(tensor.shape[0]))
    prompt_piece_index = (
        build_prompt_piece_index(load_gguf_vocab_tokens(model_path)) if args.prompt_piece_count > 0 else None
    )

    train_remainders = {int(item) for item in args.train_remainders.split(",") if item.strip()}
    train_rows, eval_rows = split_rows(rows, args.train_mod, train_remainders)
    train_records = iter_steps(train_rows, args.ignore_prefix_steps)
    eval_records = iter_steps(eval_rows, args.ignore_prefix_steps)
    static_rescue_ids = learn_static_rescue_ids(train_records, args.proxy_k_for_rescue, args.static_rescue_count)
    position_rescue_by_step = learn_position_rescue_ids(train_records, args.position_rescue_count)

    train_summary = summarize_split(
        name="train",
        rows=train_rows,
        k_values=k_values,
        static_rescue_ids=static_rescue_ids,
        position_rescue_by_step=position_rescue_by_step,
        prompt_piece_index=prompt_piece_index,
        prompt_piece_count=args.prompt_piece_count,
        prompt_piece_ids_per_piece=args.prompt_piece_ids_per_piece,
        row_bytes=row_bytes,
        vocab_tensor_bytes=tensor.n_bytes,
        ignore_prefix_steps=args.ignore_prefix_steps,
    )
    eval_summary = summarize_split(
        name="eval",
        rows=eval_rows,
        k_values=k_values,
        static_rescue_ids=static_rescue_ids,
        position_rescue_by_step=position_rescue_by_step,
        prompt_piece_index=prompt_piece_index,
        prompt_piece_count=args.prompt_piece_count,
        prompt_piece_ids_per_piece=args.prompt_piece_ids_per_piece,
        row_bytes=row_bytes,
        vocab_tensor_bytes=tensor.n_bytes,
        ignore_prefix_steps=args.ignore_prefix_steps,
    )
    full_summary = summarize_split(
        name="all",
        rows=rows,
        k_values=k_values,
        static_rescue_ids=static_rescue_ids,
        position_rescue_by_step=position_rescue_by_step,
        prompt_piece_index=prompt_piece_index,
        prompt_piece_count=args.prompt_piece_count,
        prompt_piece_ids_per_piece=args.prompt_piece_ids_per_piece,
        row_bytes=row_bytes,
        vocab_tensor_bytes=tensor.n_bytes,
        ignore_prefix_steps=args.ignore_prefix_steps,
    )

    eval_best = max(eval_summary.coverage, key=lambda item: (item.coverage_rate, -item.candidate_rows_per_step))
    rescue_token_counts = Counter(record["oracle_id"] for record in train_records)
    static_rescue = [
        {
            "token_id": token_id,
            "train_oracle_count": rescue_token_counts.get(token_id, 0),
        }
        for token_id in static_rescue_ids
    ]

    return {
        "schema": "sage-proxy-shortlist-coverage-v0",
        "status": "measured_proxy_shortlist_coverage",
        "source_logprob_json": [str(path) for path in paths],
        "model": {
            "path": str(Path(args.model).resolve()),
            "tensor_name": tensor.name,
            "tensor_type": tensor.tensor_type,
            "tensor_bytes": tensor.n_bytes,
            "row_bytes": row_bytes,
        },
        "params": {
            "ignore_prefix_steps": args.ignore_prefix_steps,
            "k_values": k_values,
            "train_mod": args.train_mod,
            "train_remainders": sorted(train_remainders),
            "proxy_k_for_rescue": args.proxy_k_for_rescue,
            "static_rescue_count": args.static_rescue_count,
            "position_rescue_count": args.position_rescue_count,
            "prompt_piece_count": args.prompt_piece_count,
            "prompt_piece_ids_per_piece": args.prompt_piece_ids_per_piece,
        },
        "summary": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_steps": len(train_records),
            "eval_steps": len(eval_records),
            "static_rescue_tokens": len(static_rescue_ids),
            "position_rescue_steps": len(position_rescue_by_step),
            "prompt_piece_tokens_max": args.prompt_piece_count,
            "best_eval_coverage_name": eval_best.name,
            "best_eval_coverage_rate": eval_best.coverage_rate,
            "best_eval_fallback_rate_for_exact": eval_best.fallback_rate_for_exact,
            "best_eval_candidate_rows_per_step": eval_best.candidate_rows_per_step,
            "best_eval_candidate_bytes_per_step": eval_best.candidate_bytes_per_step,
            "best_eval_active_percent_vocab_tensor": eval_best.active_percent_vocab_tensor,
            "proxy_shortlist_status": "insufficient_without_rescue"
            if next((c for c in eval_summary.coverage if c.name == f"proxy_top_{max(k_values)}"), eval_best).coverage_rate < 0.8
            else "usable_without_rescue",
        },
        "static_rescue": static_rescue,
        "position_rescue_by_step": {
            str(step): token_ids for step, token_ids in sorted(position_rescue_by_step.items())
        },
        "splits": [asdict(train_summary), asdict(eval_summary), asdict(full_summary)],
        "class_breakdown_eval": class_breakdown(
            eval_rows,
            args.ignore_prefix_steps,
            max(k_values),
            static_rescue_ids,
            position_rescue_by_step,
            prompt_piece_index,
            args.prompt_piece_count,
            args.prompt_piece_ids_per_piece,
        ),
    }


def print_markdown(payload: dict[str, Any]) -> None:
    print("# SAGE Proxy Shortlist Coverage")
    print()
    summary = payload["summary"]
    print(f"- Schema: `{payload['schema']}`")
    print(f"- Status: `{payload['status']}`")
    print(f"- Train/eval steps: `{summary['train_steps']}` / `{summary['eval_steps']}`")
    print(f"- Static rescue tokens: `{summary['static_rescue_tokens']}`")
    print(f"- Position rescue steps: `{summary['position_rescue_steps']}`")
    print(f"- Prompt-piece tokens max: `{summary['prompt_piece_tokens_max']}`")
    print(f"- Best eval coverage: `{summary['best_eval_coverage_name']}` = `{summary['best_eval_coverage_rate']:.2%}`")
    print(f"- Exact fallback rate at best eval coverage: `{summary['best_eval_fallback_rate_for_exact']:.2%}`")
    print()
    for split in payload["splits"]:
        print(f"## {split['name'].title()}")
        print()
        print(f"- Steps: `{split['steps']}`")
        print(f"- Proxy top-1 match rate: `{split['proxy_top1_match_rate']:.2%}`")
        print()
        print("| Shortlist | Coverage | Fallback for Exact | Rows/Step | Bytes/Step | Vocab % |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in split["coverage"]:
            print(
                f"| {row['name']} | {row['coverage_rate']:.2%} | "
                f"{row['fallback_rate_for_exact']:.2%} | {row['candidate_rows_per_step']:.2f} | "
                f"{row['candidate_bytes_per_step']:.0f} | {row['active_percent_vocab_tensor']:.4f}% |"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate proxy shortlist coverage against oracle top-1 tokens.")
    parser.add_argument("--logprob-json", nargs="+", required=True)
    parser.add_argument("--model", default="models/gemma-4-31b-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf")
    parser.add_argument("--tensor-name", default="token_embd.weight")
    parser.add_argument("--ignore-prefix-steps", type=int, default=3)
    parser.add_argument("--k-values", default="1,2,4,8,10")
    parser.add_argument("--train-mod", type=int, default=2)
    parser.add_argument("--train-remainders", default="0")
    parser.add_argument("--proxy-k-for-rescue", type=int, default=10)
    parser.add_argument("--static-rescue-count", type=int, default=16)
    parser.add_argument("--position-rescue-count", type=int, default=0)
    parser.add_argument("--prompt-piece-count", type=int, default=0)
    parser.add_argument("--prompt-piece-ids-per-piece", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.ignore_prefix_steps < 0:
        parser.error("--ignore-prefix-steps must be non-negative")
    if args.train_mod <= 0:
        parser.error("--train-mod must be positive")
    if args.proxy_k_for_rescue <= 0:
        parser.error("--proxy-k-for-rescue must be positive")
    if args.static_rescue_count < 0:
        parser.error("--static-rescue-count must be non-negative")
    if args.position_rescue_count < 0:
        parser.error("--position-rescue-count must be non-negative")
    if args.prompt_piece_count < 0:
        parser.error("--prompt-piece-count must be non-negative")
    if args.prompt_piece_ids_per_piece <= 0:
        parser.error("--prompt-piece-ids-per-piece must be positive")

    payload = make_payload(args)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
