#!/usr/bin/env python3
"""
Production-oriented helper for weak-hardware llama.cpp inference.

The tool profiles GGUF models with llama-bench, stores local evidence, chooses
spill-aware placements, and emits run commands for llama-cli / llama-server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-bench.exe"
DEFAULT_CLI = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-cli.exe"
DEFAULT_SERVER = ROOT / "tools" / "llama.cpp-b9804-cuda124" / "llama-server.exe"
DEFAULT_PHASE_SCRIPT = ROOT / "scripts" / "phase-split-llama.ps1"
DEFAULT_MODELS = ROOT / "models"
DEFAULT_PROFILES = ROOT / "profiles"
DEFAULT_BENCHMARKS = ROOT / "benchmarks"
DEFAULT_SLOT_DIR = ROOT / "phase-cache"
CACHE_INDEX_NAME = "cache-index.json"
VALID_CACHE_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def split_csv(value: str, cast=str) -> list[Any]:
    out: list[Any] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def slug_model(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9._-]+", "-", stem)
    return stem[:120].strip("-") or "model"


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def quote_ps(argv: Iterable[str]) -> str:
    # Good enough for Windows PowerShell commands whose arguments do not depend
    # on shell expansion. Prefer double quotes for paths with spaces.
    rendered: list[str] = []
    for arg in argv:
        if re.search(r"\s", arg) or any(ch in arg for ch in "`'\"&()[]{};"):
            rendered.append('"' + arg.replace('"', '`"') + '"')
        else:
            rendered.append(arg)
    return " ".join(rendered)


def run(argv: list[str], timeout: int | None = None) -> CommandResult:
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return CommandResult(argv, completed.returncode, completed.stdout, completed.stderr)


def extract_json_array(text: str) -> list[Any]:
    match = re.search(r"(?s)\[\s*\{.*\}\s*\]", text)
    if not match:
        raise ValueError("no JSON array found in command output")
    return json.loads(match.group(0))


def detect_gpu() -> dict[str, Any]:
    try:
        result = run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ], timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "error": result.stderr.strip() or result.stdout.strip()}

    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {"available": False, "raw": first}
    return {
        "available": True,
        "name": parts[0],
        "memory_total_mib": int(float(parts[1])),
        "memory_free_mib": int(float(parts[2])),
        "driver": parts[3],
    }


def detect_ram() -> dict[str, Any]:
    if platform.system().lower() == "windows":
        result = run([
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "$cs = Get-CimInstance Win32_ComputerSystem; "
            "$os = Get-CimInstance Win32_OperatingSystem; "
            "[pscustomobject]@{"
            "TotalPhysicalMemory = [int64]$cs.TotalPhysicalMemory; "
            "FreePhysicalMemory = ([int64]$os.FreePhysicalMemory * 1024)"
            "} | ConvertTo-Json -Compress",
        ], timeout=10)
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout.strip())
                total = int(payload["TotalPhysicalMemory"])
                free = int(payload["FreePhysicalMemory"])
                return {
                    "total_bytes": total,
                    "total_gib": round(total / 1024**3, 2),
                    "free_bytes": free,
                    "free_gib": round(free / 1024**3, 2),
                    "free_ratio": round(free / total, 4) if total else None,
                }
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
    return {}


def detect_hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": detect_gpu(),
        "ram": detect_ram(),
    }


def list_models(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("*.gguf")):
        rows.append({
            "path": str(path.resolve()),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "size": human_bytes(path.stat().st_size),
        })
    return rows


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_prompt(prompt: str = "", prompt_file: str = "") -> str:
    if prompt_file:
        return Path(prompt_file).resolve().read_text(encoding="utf-8")
    return prompt


def estimate_prompt_tokens(prompt: str, fallback: int = 512) -> int:
    if not prompt:
        return fallback
    return max(1, math.ceil(len(prompt) / 4))


def stable_hash(payload: Any, length: int = 20) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def model_identity(model: Path) -> dict[str, Any]:
    stat = model.stat()
    return {
        "path": str(model.resolve()),
        "name": model.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_index_path(slot_dir: Path) -> Path:
    return ensure_dir(slot_dir) / CACHE_INDEX_NAME


def load_cache_index(slot_dir: Path) -> dict[str, Any]:
    path = cache_index_path(slot_dir)
    if not path.is_file():
        return {"schema_version": 1, "updated_at": utc_now(), "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"cache index is not valid JSON: {path}: {exc}")
    payload.setdefault("schema_version", 1)
    payload.setdefault("updated_at", utc_now())
    payload.setdefault("entries", [])
    return payload


def save_cache_index(slot_dir: Path, payload: dict[str, Any]) -> Path:
    payload["updated_at"] = utc_now()
    path = cache_index_path(slot_dir)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def slot_path(slot_dir: Path, slot_file: str) -> Path:
    path = Path(slot_file)
    if path.is_absolute():
        return path
    return slot_dir.resolve() / slot_file


def refresh_cache_index(slot_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    for entry in payload.get("entries", []):
        path = slot_path(slot_dir, str(entry.get("slot_file", "")))
        entry["slot_path"] = str(path)
        if path.is_file():
            entry["status"] = "available"
            entry["slot_size_bytes"] = path.stat().st_size
        elif entry.get("status") != "planned":
            entry["status"] = "missing"
            entry.pop("slot_size_bytes", None)
        entry["checked_at"] = now
    return payload


def cache_lookup(
    index: dict[str, Any],
    slot_dir: Path,
    model: Path,
    prompt_hash: str,
    context: int,
    cache_type: str,
    prefill_ngl: int,
    decode_ngl: int,
) -> dict[str, Any] | None:
    model_id = model_identity(model)
    for entry in index.get("entries", []):
        if entry.get("model_path") != model_id["path"]:
            continue
        if entry.get("model_size_bytes") != model_id["size_bytes"]:
            continue
        if entry.get("prompt_hash") != prompt_hash:
            continue
        if int(entry.get("context", -1)) != context:
            continue
        if entry.get("cache_type") != cache_type:
            continue
        if int(entry.get("prefill_gpu_layers", -9999)) != prefill_ngl:
            continue
        if int(entry.get("decode_gpu_layers", -9999)) != decode_ngl:
            continue
        if slot_path(slot_dir, str(entry.get("slot_file", ""))).is_file():
            entry["status"] = "available"
            return entry
    return None


def planned_cache_entry(
    model: Path,
    prompt: str,
    context: int,
    cache_type: str,
    prefill_ngl: int,
    decode_ngl: int,
    expected_reuses: int,
) -> dict[str, Any]:
    model_id = model_identity(model)
    p_hash = prompt_digest(prompt)
    key = stable_hash({
        "model": model_id,
        "prompt_hash": p_hash,
        "context": context,
        "cache_type": cache_type,
        "prefill_ngl": prefill_ngl,
        "decode_ngl": decode_ngl,
    })
    return {
        "cache_key": key,
        "status": "planned",
        "model_path": model_id["path"],
        "model_name": model_id["name"],
        "model_size_bytes": model_id["size_bytes"],
        "prompt_hash": p_hash,
        "prompt_chars": len(prompt),
        "estimated_prompt_tokens": estimate_prompt_tokens(prompt),
        "context": context,
        "cache_type": cache_type,
        "prefill_gpu_layers": prefill_ngl,
        "decode_gpu_layers": decode_ngl,
        "slot_file": f"adaptive-{key}.bin",
        "expected_reuses": expected_reuses,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def upsert_cache_entry(index: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    entries = index.setdefault("entries", [])
    for existing in entries:
        if existing.get("cache_key") == entry["cache_key"]:
            existing.update({k: v for k, v in entry.items() if k not in {"created_at"}})
            existing["updated_at"] = utc_now()
            return existing
    entries.append(entry)
    return entry


def profile_path(profile_dir: Path, model: Path) -> Path:
    return profile_dir / f"{slug_model(model)}.profile.json"


def load_profile(profile_dir: Path, model: Path) -> dict[str, Any] | None:
    path = profile_path(profile_dir, model)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile(profile_dir: Path, model: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(profile_dir)
    path = profile_path(profile_dir, model)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def rows_by_run(rows: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        ngl = int(row.get("n_gpu_layers", 0))
        type_k = str(row.get("type_k", ""))
        type_v = str(row.get("type_v", ""))
        key = (ngl, type_k, type_v)
        grouped.setdefault(key, {})
        if int(row.get("n_gen", 0) or 0) > 0:
            grouped[key]["tg"] = row
        if int(row.get("n_prompt", 0) or 0) > 0:
            grouped[key]["pp"] = row
    return grouped


def recommend_from_rows(rows: list[dict[str, Any]], mode: str = "chat") -> dict[str, Any]:
    grouped = rows_by_run(rows)
    candidates: list[dict[str, Any]] = []
    for (ngl, type_k, type_v), parts in grouped.items():
        tg = parts.get("tg")
        pp = parts.get("pp")
        if not tg:
            continue
        tg_ts = float(tg.get("avg_ts", 0.0) or 0.0)
        pp_ts = float(pp.get("avg_ts", 0.0) or 0.0) if pp else 0.0
        if mode == "prefill":
            score = pp_ts
        elif mode == "balanced":
            score = (tg_ts * 0.75) + (min(pp_ts, 1000.0) / 1000.0 * tg_ts * 0.25)
        else:
            score = tg_ts
        candidates.append({
            "n_gpu_layers": ngl,
            "cache_type_k": type_k,
            "cache_type_v": type_v,
            "generation_tps": tg_ts,
            "prompt_tps": pp_ts,
            "score": score,
            "tg_row": tg,
            "pp_row": pp,
        })

    if not candidates:
        fail("profile does not contain successful generation benchmark rows")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best_decode = candidates[0]
    prefill_candidates = [c for c in candidates if c["prompt_tps"] > 0]
    best_prefill = max(prefill_candidates, key=lambda item: item["prompt_tps"]) if prefill_candidates else best_decode

    full_offload = next((c for c in candidates if c["n_gpu_layers"] == -1), None)
    cpu_only = next((c for c in candidates if c["n_gpu_layers"] == 0), None)
    spill_warning = None
    if full_offload and best_decode["n_gpu_layers"] != -1:
        if full_offload["generation_tps"] < best_decode["generation_tps"] * 0.8:
            spill_warning = (
                "Full offload is slower than the selected partial placement; "
                "avoid -ngl -1 for this model on this hardware."
            )

    return {
        "mode": mode,
        "decode": slim_candidate(best_decode),
        "prefill": slim_candidate(best_prefill),
        "cpu_only": slim_candidate(cpu_only) if cpu_only else None,
        "full_offload": slim_candidate(full_offload) if full_offload else None,
        "spill_warning": spill_warning,
        "top": [slim_candidate(c) for c in candidates[:8]],
    }


def slim_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "n_gpu_layers": candidate["n_gpu_layers"],
        "cache_type_k": candidate["cache_type_k"],
        "cache_type_v": candidate["cache_type_v"],
        "generation_tps": round(candidate["generation_tps"], 4),
        "prompt_tps": round(candidate["prompt_tps"], 4),
        "score": round(candidate["score"], 4),
    }


def write_report(profile: dict[str, Any], report_path: Path) -> None:
    model = profile["model"]["path"]
    recommendation = profile.get("recommendation") or recommend_from_rows(profile["benchmarks"], "chat")
    lines = [
        "# Weak LLM Profile Report",
        "",
        f"Model: `{model}`",
        f"Profile created: {profile.get('created_at')}",
        f"llama.cpp build: {profile.get('llama_cpp', {}).get('build', 'unknown')}",
        "",
        "## Hardware",
        "",
        "```json",
        json.dumps(profile.get("hardware", {}), indent=2),
        "```",
        "",
        "## Recommendation",
        "",
        f"Decode `-ngl`: `{recommendation['decode']['n_gpu_layers']}`",
        f"KV cache: `{recommendation['decode']['cache_type_k']}/{recommendation['decode']['cache_type_v']}`",
        f"Generation: `{recommendation['decode']['generation_tps']}` tok/s",
        f"Prompt: `{recommendation['decode']['prompt_tps']}` tok/s",
        "",
    ]
    if recommendation.get("spill_warning"):
        lines.extend(["Warning:", recommendation["spill_warning"], ""])
    lines.extend([
        "## Top Candidates",
        "",
        "| rank | ngl | cache | generation tok/s | prompt tok/s |",
        "| ---: | ---: | --- | ---: | ---: |",
    ])
    for idx, cand in enumerate(recommendation["top"], start=1):
        lines.append(
            f"| {idx} | {cand['n_gpu_layers']} | {cand['cache_type_k']}/{cand['cache_type_v']} | "
            f"{cand['generation_tps']} | {cand['prompt_tps']} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_profile(args: argparse.Namespace) -> None:
    bench = ensure_file(Path(args.llama_bench), "llama-bench")
    model = ensure_file(Path(args.model), "model")
    profile_dir = ensure_dir(Path(args.profile_dir))
    benchmark_dir = ensure_dir(Path(args.benchmark_dir))

    gpu_layers = split_csv(args.gpu_layers, int)
    cache_types = split_csv(args.cache_types, str)
    for cache in cache_types:
        if cache not in VALID_CACHE_TYPES:
            fail(f"invalid cache type: {cache}")

    raw_runs: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for cache in cache_types:
        for ngl in gpu_layers:
            argv = [
                str(bench),
                "-m", str(model),
                "-p", str(args.prompt_tokens),
                "-n", str(args.gen_tokens),
                "-r", str(args.repetitions),
                "-ngl", str(ngl),
                "-ctk", cache,
                "-ctv", cache,
                "-o", "json",
            ]
            print(f"profile: ngl={ngl} cache={cache}", flush=True)
            result = run(argv, timeout=args.timeout)
            raw = result.stdout + "\n" + result.stderr
            run_record = {
                "n_gpu_layers": ngl,
                "cache_type": cache,
                "returncode": result.returncode,
                "ok": False,
            }
            try:
                rows = extract_json_array(raw)
                if result.returncode == 0:
                    run_record["ok"] = True
                    benchmark_rows.extend(rows)
                else:
                    run_record["error"] = raw[-4000:]
            except Exception as exc:
                run_record["error"] = str(exc)
                run_record["raw_tail"] = raw[-4000:]
            raw_runs.append(run_record)

    if not benchmark_rows:
        fail("no successful benchmark rows were collected")

    recommendation = recommend_from_rows(benchmark_rows, args.mode)
    profile = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": {
            "path": str(model),
            "name": model.name,
            "size_bytes": model.stat().st_size,
            "size": human_bytes(model.stat().st_size),
        },
        "hardware": detect_hardware(),
        "llama_cpp": {
            "bench": str(bench),
            "build": benchmark_rows[0].get("build_number"),
            "commit": benchmark_rows[0].get("build_commit"),
        },
        "profile_params": {
            "gpu_layers": gpu_layers,
            "cache_types": cache_types,
            "prompt_tokens": args.prompt_tokens,
            "gen_tokens": args.gen_tokens,
            "repetitions": args.repetitions,
            "mode": args.mode,
        },
        "runs": raw_runs,
        "benchmarks": benchmark_rows,
        "recommendation": recommendation,
    }

    out_json = save_profile(profile_dir, model, profile)
    report = benchmark_dir / f"{now_stamp()}-{slug_model(model)}-production-profile.md"
    write_report(profile, report)
    print(f"wrote profile: {out_json}")
    print(f"wrote report: {report}")
    print(json.dumps(recommendation, indent=2))


def cmd_inventory(args: argparse.Namespace) -> None:
    payload = {
        "root": str(ROOT),
        "hardware": detect_hardware(),
        "models": list_models(Path(args.models_dir)),
        "runtimes": {
            "llama_bench": str(Path(args.llama_bench).resolve()),
            "llama_cli": str(Path(args.llama_cli).resolve()),
            "llama_server": str(Path(args.llama_server).resolve()),
            "phase_split_script": str(Path(args.phase_script).resolve()),
        },
    }
    print(json.dumps(payload, indent=2))


def profile_or_fail(profile_dir: Path, model: Path) -> dict[str, Any]:
    profile = load_profile(profile_dir, model)
    if not profile:
        fail(f"no profile found for {model}; run the profile subcommand first")
    return profile


def cmd_recommend(args: argparse.Namespace) -> None:
    model = ensure_file(Path(args.model), "model")
    profile = profile_or_fail(Path(args.profile_dir), model)
    recommendation = recommend_from_rows(profile["benchmarks"], args.mode)
    profile["recommendation"] = recommendation
    save_profile(Path(args.profile_dir), model, profile)
    print(json.dumps(recommendation, indent=2))


def selected(profile: dict[str, Any], mode: str) -> dict[str, Any]:
    recommendation = recommend_from_rows(profile["benchmarks"], mode)
    return recommendation["decode"]


def phase_selected(profile: dict[str, Any], mode: str) -> dict[str, Any]:
    recommendation = recommend_from_rows(profile["benchmarks"], mode)
    decode = recommendation["decode"]
    if decode["cache_type_k"] != decode["cache_type_v"]:
        fail("phase-split command requires the same K and V cache type")

    ranked = [recommendation["prefill"], *recommendation["top"]]
    compatible = [
        candidate for candidate in ranked
        if candidate
        if candidate["cache_type_k"] == decode["cache_type_k"]
        and candidate["cache_type_v"] == decode["cache_type_v"]
        and candidate["prompt_tps"] > 0
    ]
    prefill = max(compatible, key=lambda item: item["prompt_tps"]) if compatible else decode

    return {
        "decode": decode,
        "prefill": prefill,
        "cache_type": decode["cache_type_k"],
        "spill_warning": recommendation.get("spill_warning"),
    }


def ms_for_tokens(tokens: int, tokens_per_second: float) -> float | None:
    if tokens_per_second <= 0:
        return None
    return (tokens / tokens_per_second) * 1000.0


def memory_pressure(model: Path, hardware: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    gpu = hardware.get("gpu", {})
    ram = hardware.get("ram", {})
    total_mib = int(gpu.get("memory_total_mib") or 0)
    free_mib = int(gpu.get("memory_free_mib") or 0)
    model_mib = model.stat().st_size / 1024**2
    free_ratio = (free_mib / total_mib) if total_mib else None
    model_to_vram_ratio = (model_mib / total_mib) if total_mib else None
    ram_total_bytes = int(ram.get("total_bytes") or 0)
    ram_free_bytes = int(ram.get("free_bytes") or 0)
    ram_free_ratio = (ram_free_bytes / ram_total_bytes) if ram_total_bytes else None
    larger_than_vram = bool(total_mib and model_mib > total_mib * 0.95)
    full_offload_slow = bool(recommendation.get("spill_warning"))

    level = "unknown"
    reasons: list[str] = []
    if total_mib:
        level = "normal"
        if larger_than_vram:
            level = "high"
            reasons.append("model file is larger than available GPU memory class")
        if full_offload_slow:
            level = "high"
            reasons.append("profile shows full offload is slower than partial placement")
        if free_ratio is not None and free_ratio < 0.20:
            level = "high"
            reasons.append("current free VRAM is below 20%")
        elif free_ratio is not None and free_ratio < 0.35 and level != "high":
            level = "moderate"
            reasons.append("current free VRAM is below 35%")
    else:
        reasons.append("no NVIDIA GPU memory data available")

    if ram_free_ratio is not None:
        if ram_free_ratio < 0.15:
            level = "high"
            reasons.append("current free system RAM is below 15%")
        elif ram_free_ratio < 0.30 and level not in {"high", "unknown"}:
            level = "moderate"
            reasons.append("current free system RAM is below 30%")

    return {
        "level": level,
        "gpu_name": gpu.get("name"),
        "vram_total_mib": total_mib or None,
        "vram_free_mib": free_mib or None,
        "vram_free_ratio": round(free_ratio, 4) if free_ratio is not None else None,
        "ram_total_gib": round(ram_total_bytes / 1024**3, 2) if ram_total_bytes else None,
        "ram_free_gib": round(ram_free_bytes / 1024**3, 2) if ram_free_bytes else None,
        "ram_free_ratio": round(ram_free_ratio, 4) if ram_free_ratio is not None else None,
        "model_size_mib": round(model_mib, 2),
        "model_to_vram_ratio": round(model_to_vram_ratio, 4) if model_to_vram_ratio is not None else None,
        "larger_than_vram": larger_than_vram,
        "full_offload_slow": full_offload_slow,
        "reasons": reasons,
    }


def latest_phase_measurement(
    benchmark_dir: Path,
    model: Path,
    context: int,
    cache_type: str,
    prefill_ngl: int,
    decode_ngl: int,
) -> dict[str, Any] | None:
    if not benchmark_dir.exists():
        return None
    files = sorted(
        benchmark_dir.glob("*phase-split-result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    model_path = str(model.resolve()).lower()
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("model", "")).lower() != model_path:
            continue
        if int(data.get("context_size") or 0) != context:
            continue
        if str(data.get("cache_type", "")) != cache_type:
            continue
        if int(str(data.get("prefill_gpu_layers", "0"))) != prefill_ngl:
            continue
        if int(str(data.get("decode_gpu_layers", "0"))) != decode_ngl:
            continue

        cold = data.get("cold") or {}
        cold_elapsed_ms = data.get("cold_elapsed_ms") or cold.get("elapsed_ms")
        phase_decode_ms = data.get("phase_decode_elapsed_ms")
        cached_saving_ms = data.get("cached_decode_saving_ms")
        if cached_saving_ms is None and cold_elapsed_ms is not None and phase_decode_ms is not None:
            cached_saving_ms = float(cold_elapsed_ms) - float(phase_decode_ms)

        prefill_ms = float(data.get("prefill_elapsed_ms") or 0.0)
        save_ms = float(data.get("save_elapsed_ms") or 0.0)
        restore_ms = float(data.get("restore_elapsed_ms") or 0.0)
        cache_create_ms = prefill_ms + save_ms + restore_ms
        break_even = data.get("break_even_loaded_reuses")
        if break_even is None and cached_saving_ms and cached_saving_ms > 0:
            break_even = math.ceil(cache_create_ms / cached_saving_ms)

        return {
            "source_file": str(path.resolve()),
            "prompt_chars": data.get("prompt_chars"),
            "prefill_tokens_evaluated": data.get("prefill_tokens_evaluated"),
            "phase_decode_elapsed_ms": phase_decode_ms,
            "cold_elapsed_ms": cold_elapsed_ms,
            "cached_decode_saving_ms": cached_saving_ms,
            "cache_create_ms": cache_create_ms,
            "break_even_loaded_reuses": break_even,
        }
    return None


def estimate_break_even(
    prompt_tokens: int,
    new_tokens: int,
    decode: dict[str, Any],
    prefill: dict[str, Any],
    measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    normal_prompt_ms = ms_for_tokens(prompt_tokens, float(decode["prompt_tps"]))
    normal_generation_ms = ms_for_tokens(new_tokens, float(decode["generation_tps"])) or 0.0
    normal_total_ms = (normal_prompt_ms or 0.0) + normal_generation_ms

    if measurement and measurement.get("prefill_tokens_evaluated"):
        measured_tokens = max(1, int(measurement["prefill_tokens_evaluated"]))
        create_per_token = float(measurement["cache_create_ms"]) / measured_tokens
        saving_per_token = max(0.0, float(measurement.get("cached_decode_saving_ms") or 0.0) / measured_tokens)
        cache_create_ms = create_per_token * prompt_tokens
        cached_saving_ms = saving_per_token * prompt_tokens
        source = "phase_measurement"
    else:
        prefill_ms = ms_for_tokens(prompt_tokens, float(prefill["prompt_tps"])) or normal_total_ms
        save_restore_ms = max(250.0, prompt_tokens * 0.20)
        cache_create_ms = prefill_ms + save_restore_ms
        cached_saving_ms = max(0.0, (normal_prompt_ms or 0.0) * 0.15)
        source = "profile_estimate"

    break_even = math.ceil(cache_create_ms / cached_saving_ms) if cached_saving_ms > 0 else None
    cached_total_ms = max(0.0, normal_total_ms - cached_saving_ms)
    return {
        "source": source,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "normal_total_ms": round(normal_total_ms, 3),
        "normal_prompt_ms": round(normal_prompt_ms or 0.0, 3),
        "normal_generation_ms": round(normal_generation_ms, 3),
        "phase_cache_create_ms": round(cache_create_ms, 3),
        "phase_cached_total_ms": round(cached_total_ms, 3),
        "phase_cached_saving_ms": round(cached_saving_ms, 3),
        "break_even_reuses": break_even,
        "measurement": measurement,
    }


def phase_command_argv(
    phase_script: Path,
    server: Path,
    model: Path,
    choice: dict[str, Any],
    context: int,
    new_tokens: int,
    decode_prompt_mode: str,
    port_base: int,
    benchmark_dir: Path,
    slot_dir: Path,
    prompt: str = "",
    prompt_file: str = "",
    compare_cold: bool = False,
    slot_file: str = "",
    use_existing_slot: bool = False,
) -> list[str]:
    argv = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", str(phase_script),
        "-Server", str(server),
        "-Model", str(model),
        "-PrefillGpuLayers", str(choice["prefill"]["n_gpu_layers"]),
        "-DecodeGpuLayers", str(choice["decode"]["n_gpu_layers"]),
        "-CacheType", choice["cache_type"],
        "-ContextSize", str(context),
        "-NewTokens", str(new_tokens),
        "-DecodePromptMode", decode_prompt_mode,
        "-PortBase", str(port_base),
        "-OutDir", str(benchmark_dir.resolve()),
        "-SlotDir", str(slot_dir.resolve()),
    ]
    if slot_file:
        argv.extend(["-SlotFile", slot_file])
    if use_existing_slot:
        argv.append("-UseExistingSlot")
    if prompt:
        argv.extend(["-PromptText", prompt])
    if prompt_file:
        argv.extend(["-PromptFile", str(Path(prompt_file).resolve())])
    if compare_cold:
        argv.append("-CompareCold")
    return argv


def server_command_argv(
    server: Path,
    model: Path,
    choice: dict[str, Any],
    context: int,
    host: str,
    port: int,
    slot_dir: Path,
) -> list[str]:
    return [
        str(server),
        "-m", str(model),
        "-ngl", str(choice["n_gpu_layers"]),
        "-ctk", choice["cache_type_k"],
        "-ctv", choice["cache_type_v"],
        "-c", str(context),
        "--host", host,
        "--port", str(port),
        "--cache-prompt",
        "--slot-save-path", str(slot_dir.resolve()),
    ]


def make_adaptive_plan(args: argparse.Namespace, register_cache: bool = True) -> dict[str, Any]:
    model = ensure_file(Path(args.model), "model")
    server = ensure_file(Path(args.llama_server), "llama-server")
    phase_script = ensure_file(Path(args.phase_script), "phase-split script")
    profile = profile_or_fail(Path(args.profile_dir), model)
    recommendation = recommend_from_rows(profile["benchmarks"], args.mode)
    phase_choice = phase_selected(profile, args.mode)
    decode = phase_choice["decode"]
    prefill = phase_choice["prefill"]
    cache_type = args.cache_type or phase_choice["cache_type"]
    if args.cache_type:
        phase_choice = {**phase_choice, "cache_type": cache_type}
    prompt = read_prompt(args.prompt, args.prompt_file)
    fallback_tokens = int(profile.get("profile_params", {}).get("prompt_tokens") or 512)
    prompt_tokens = args.prompt_tokens or estimate_prompt_tokens(prompt, fallback_tokens)
    hardware = detect_hardware()
    pressure = memory_pressure(model, hardware, recommendation)
    measurement = latest_phase_measurement(
        Path(args.benchmark_dir),
        model,
        args.context,
        cache_type,
        int(prefill["n_gpu_layers"]),
        int(decode["n_gpu_layers"]),
    )
    economics = estimate_break_even(prompt_tokens, args.new_tokens, decode, prefill, measurement)

    slot_dir = ensure_dir(Path(args.slot_dir))
    index = refresh_cache_index(slot_dir, load_cache_index(slot_dir))
    prompt_hash = prompt_digest(prompt) if prompt else ""
    hit = None
    planned = None
    if prompt_hash:
        hit = cache_lookup(
            index,
            slot_dir,
            model,
            prompt_hash,
            args.context,
            cache_type,
            int(prefill["n_gpu_layers"]),
            int(decode["n_gpu_layers"]),
        )
        planned = planned_cache_entry(
            model,
            prompt,
            args.context,
            cache_type,
            int(prefill["n_gpu_layers"]),
            int(decode["n_gpu_layers"]),
            args.expected_reuses,
        )

    reasons: list[str] = []
    strategy = "normal-server"
    command: list[str]
    cache_entry = None
    break_even = economics["break_even_reuses"]

    if args.strategy == "normal":
        reasons.append("forced normal strategy")
    elif args.strategy == "phase" and prompt:
        strategy = "phase-reuse" if hit else "phase-create"
        reasons.append("forced phase strategy")
    elif hit:
        strategy = "phase-reuse"
        cache_entry = hit
        reasons.append("matching persistent slot cache is available")
    elif prompt and break_even and args.expected_reuses >= break_even and pressure["level"] in {"moderate", "high"}:
        strategy = "phase-create"
        reasons.append(f"expected reuses ({args.expected_reuses}) meet break-even ({break_even})")
        reasons.extend(pressure["reasons"])
    elif prompt and break_even and args.expected_reuses >= break_even and args.allow_phase_on_normal_pressure:
        strategy = "phase-create"
        reasons.append("phase allowed despite normal pressure and reuse meets break-even")
    else:
        if not prompt:
            reasons.append("no prompt text was provided, so a persistent prompt cache cannot be keyed")
        if break_even and args.expected_reuses < break_even:
            reasons.append(f"expected reuses ({args.expected_reuses}) are below break-even ({break_even})")
        if pressure["level"] == "normal":
            reasons.append("current VRAM pressure is normal")

    if strategy == "phase-create" and planned:
        cache_entry = upsert_cache_entry(index, planned)
        command = phase_command_argv(
            phase_script,
            server,
            model,
            phase_choice,
            args.context,
            args.new_tokens,
            args.decode_prompt_mode,
            args.port_base,
            Path(args.benchmark_dir),
            slot_dir,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            compare_cold=args.compare_cold,
            slot_file=cache_entry["slot_file"],
        )
    elif strategy == "phase-reuse" and hit:
        cache_entry = hit
        command = phase_command_argv(
            phase_script,
            server,
            model,
            phase_choice,
            args.context,
            args.new_tokens,
            args.decode_prompt_mode,
            args.port_base,
            Path(args.benchmark_dir),
            slot_dir,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            compare_cold=args.compare_cold,
            slot_file=cache_entry["slot_file"],
            use_existing_slot=True,
        )
    else:
        strategy = "normal-server"
        command = server_command_argv(
            server,
            model,
            decode,
            args.context,
            args.host,
            args.port,
            slot_dir,
        )

    if register_cache and strategy == "phase-create" and cache_entry:
        save_cache_index(slot_dir, index)
    elif register_cache:
        save_cache_index(slot_dir, index)

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "strategy": strategy,
        "reasons": reasons,
        "model": model_identity(model),
        "hardware_pressure": pressure,
        "placements": {
            "decode": decode,
            "prefill": prefill,
            "cache_type": cache_type,
        },
        "economics": economics,
        "cache": {
            "slot_dir": str(slot_dir.resolve()),
            "prompt_hash": prompt_hash,
            "hit": bool(hit),
            "entry": cache_entry,
            "index": str(cache_index_path(slot_dir)),
        },
        "command_argv": command,
        "command": quote_ps(command),
    }


def cli_command(args: argparse.Namespace, server: bool = False) -> list[str]:
    model = ensure_file(Path(args.model), "model")
    profile = profile_or_fail(Path(args.profile_dir), model)
    choice = selected(profile, args.mode)
    executable = Path(args.llama_server if server else args.llama_cli).resolve()
    ensure_file(executable, "llama-server" if server else "llama-cli")

    argv = [
        str(executable),
        "-m", str(model),
        "-ngl", str(choice["n_gpu_layers"]),
        "-ctk", choice["cache_type_k"],
        "-ctv", choice["cache_type_v"],
        "-c", str(args.context),
    ]
    if server:
        argv.extend([
            "--host", args.host,
            "--port", str(args.port),
            "--cache-prompt",
            "--slot-save-path", str(Path(args.slot_dir).resolve()),
        ])
    else:
        argv.extend(["-n", str(args.new_tokens)])
        if args.conversation:
            argv.append("-cnv")
        if args.single_turn:
            argv.append("-st")
        if args.prompt:
            argv.extend(["-p", args.prompt])
        if args.prompt_file:
            argv.extend(["-f", str(Path(args.prompt_file).resolve())])
    return argv


def cmd_generate_command(args: argparse.Namespace) -> None:
    argv = cli_command(args, server=False)
    print(quote_ps(argv))


def cmd_serve_command(args: argparse.Namespace) -> None:
    argv = cli_command(args, server=True)
    print(quote_ps(argv))


def cmd_phase_command(args: argparse.Namespace) -> None:
    model = ensure_file(Path(args.model), "model")
    server = ensure_file(Path(args.llama_server), "llama-server")
    phase_script = ensure_file(Path(args.phase_script), "phase-split script")
    profile = profile_or_fail(Path(args.profile_dir), model)
    choice = phase_selected(profile, args.mode)
    if args.cache_type:
        choice = {**choice, "cache_type": args.cache_type}
    argv = phase_command_argv(
        phase_script,
        server,
        model,
        choice,
        args.context,
        args.new_tokens,
        args.decode_prompt_mode,
        args.port_base,
        Path(args.benchmark_dir),
        Path(args.slot_dir),
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        compare_cold=args.compare_cold,
        slot_file=args.slot_file,
        use_existing_slot=args.use_existing_slot,
    )
    print(quote_ps(argv))


def cmd_plan(args: argparse.Namespace) -> None:
    plan = make_adaptive_plan(args, register_cache=not args.no_register_cache)
    print(json.dumps(plan, indent=2))


def cmd_adaptive_command(args: argparse.Namespace) -> None:
    plan = make_adaptive_plan(args, register_cache=not args.no_register_cache)
    print(plan["command"])


def cmd_cache_list(args: argparse.Namespace) -> None:
    slot_dir = ensure_dir(Path(args.slot_dir))
    index = refresh_cache_index(slot_dir, load_cache_index(slot_dir))
    if args.save:
        save_cache_index(slot_dir, index)
    print(json.dumps(index, indent=2))


def cmd_cache_prune(args: argparse.Namespace) -> None:
    slot_dir = ensure_dir(Path(args.slot_dir))
    index = refresh_cache_index(slot_dir, load_cache_index(slot_dir))
    kept = []
    removed = []
    for entry in index.get("entries", []):
        status = entry.get("status")
        should_remove = status == "missing" or (args.drop_planned and status == "planned")
        if should_remove:
            removed.append(entry)
        else:
            kept.append(entry)
    index["entries"] = kept
    save_cache_index(slot_dir, index)
    print(json.dumps({
        "removed": len(removed),
        "kept": len(kept),
        "removed_entries": removed,
        "index": str(cache_index_path(slot_dir)),
    }, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    argv = cli_command(args, server=False)
    print(f"running: {quote_ps(argv)}", file=sys.stderr)
    completed = subprocess.run(argv, cwd=ROOT)
    raise SystemExit(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile and run GGUF models with spill-aware llama.cpp settings.",
    )
    parser.set_defaults(func=lambda _: parser.print_help())
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS))
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILES))
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARKS))
    parser.add_argument("--llama-bench", default=str(DEFAULT_BENCH))
    parser.add_argument("--llama-cli", default=str(DEFAULT_CLI))
    parser.add_argument("--llama-server", default=str(DEFAULT_SERVER))
    parser.add_argument("--phase-script", default=str(DEFAULT_PHASE_SCRIPT))

    sub = parser.add_subparsers(dest="command")

    inventory = sub.add_parser("inventory", help="List hardware, runtimes, and local GGUF models.")
    inventory.set_defaults(func=cmd_inventory)

    profile = sub.add_parser("profile", help="Benchmark a model and store a local recommendation profile.")
    profile.add_argument("--model", required=True)
    profile.add_argument("--gpu-layers", default="0,20,30,36,40,44,-1")
    profile.add_argument("--cache-types", default="q8_0")
    profile.add_argument("--prompt-tokens", type=int, default=128)
    profile.add_argument("--gen-tokens", type=int, default=64)
    profile.add_argument("--repetitions", type=int, default=1)
    profile.add_argument("--timeout", type=int, default=900)
    profile.add_argument("--mode", choices=["chat", "balanced", "prefill"], default="chat")
    profile.set_defaults(func=run_profile)

    recommend = sub.add_parser("recommend", help="Print the best stored profile recommendation.")
    recommend.add_argument("--model", required=True)
    recommend.add_argument("--mode", choices=["chat", "balanced", "prefill"], default="chat")
    recommend.set_defaults(func=cmd_recommend)

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--model", required=True)
    common_run.add_argument("--mode", choices=["chat", "balanced", "prefill"], default="chat")
    common_run.add_argument("--context", type=int, default=4096)

    gen_cmd = sub.add_parser("generate-command", parents=[common_run], help="Emit a llama-cli command.")
    gen_cmd.add_argument("--new-tokens", type=int, default=512)
    gen_cmd.add_argument("--prompt", default="")
    gen_cmd.add_argument("--prompt-file", default="")
    gen_cmd.add_argument("--conversation", action="store_true", default=True)
    gen_cmd.add_argument("--no-conversation", action="store_false", dest="conversation")
    gen_cmd.add_argument("--single-turn", action="store_true", default=True)
    gen_cmd.add_argument("--multi-turn", action="store_false", dest="single_turn")
    gen_cmd.set_defaults(func=cmd_generate_command)

    serve_cmd = sub.add_parser("serve-command", parents=[common_run], help="Emit a llama-server command.")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)
    serve_cmd.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    serve_cmd.set_defaults(func=cmd_serve_command)

    phase_cmd = sub.add_parser(
        "phase-command",
        parents=[common_run],
        help="Emit a profile-derived phase-split cache-reuse command.",
    )
    phase_cmd.add_argument("--new-tokens", type=int, default=64)
    phase_cmd.add_argument("--prompt", default="")
    phase_cmd.add_argument("--prompt-file", default="")
    phase_cmd.add_argument("--cache-type", default="")
    phase_cmd.add_argument("--decode-prompt-mode", choices=["same", "empty"], default="same")
    phase_cmd.add_argument("--port-base", type=int, default=18100)
    phase_cmd.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    phase_cmd.add_argument("--slot-file", default="")
    phase_cmd.add_argument("--use-existing-slot", action="store_true")
    phase_cmd.add_argument("--compare-cold", action="store_true")
    phase_cmd.set_defaults(func=cmd_phase_command)

    adaptive = argparse.ArgumentParser(add_help=False)
    adaptive.add_argument("--model", required=True)
    adaptive.add_argument("--mode", choices=["chat", "balanced", "prefill"], default="chat")
    adaptive.add_argument("--context", type=int, default=4096)
    adaptive.add_argument("--new-tokens", type=int, default=128)
    adaptive.add_argument("--prompt", default="")
    adaptive.add_argument("--prompt-file", default="")
    adaptive.add_argument("--prompt-tokens", type=int, default=0)
    adaptive.add_argument("--expected-reuses", type=int, default=1)
    adaptive.add_argument("--cache-type", default="")
    adaptive.add_argument("--decode-prompt-mode", choices=["same", "empty"], default="same")
    adaptive.add_argument("--host", default="127.0.0.1")
    adaptive.add_argument("--port", type=int, default=8080)
    adaptive.add_argument("--port-base", type=int, default=18100)
    adaptive.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    adaptive.add_argument("--compare-cold", action="store_true")
    adaptive.add_argument("--strategy", choices=["auto", "normal", "phase"], default="auto")
    adaptive.add_argument("--allow-phase-on-normal-pressure", action="store_true")
    adaptive.add_argument("--no-register-cache", action="store_true")

    plan_cmd = sub.add_parser("plan", parents=[adaptive], help="Print an adaptive normal-vs-phase strategy plan.")
    plan_cmd.set_defaults(func=cmd_plan)

    adaptive_cmd = sub.add_parser("adaptive-command", parents=[adaptive], help="Emit the command selected by the adaptive planner.")
    adaptive_cmd.set_defaults(func=cmd_adaptive_command)

    cache_list = sub.add_parser("cache-list", help="List persistent prompt slot cache metadata.")
    cache_list.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    cache_list.add_argument("--save", action="store_true", help="Refresh statuses and persist the cache index.")
    cache_list.set_defaults(func=cmd_cache_list)

    cache_prune = sub.add_parser("cache-prune", help="Remove stale entries from the persistent cache index.")
    cache_prune.add_argument("--slot-dir", default=str(DEFAULT_SLOT_DIR))
    cache_prune.add_argument("--drop-planned", action="store_true")
    cache_prune.set_defaults(func=cmd_cache_prune)

    run_cmd = sub.add_parser("run", parents=[common_run], help="Run llama-cli with the stored recommendation.")
    run_cmd.add_argument("--new-tokens", type=int, default=128)
    run_cmd.add_argument("--prompt", default="")
    run_cmd.add_argument("--prompt-file", default="")
    run_cmd.add_argument("--conversation", action="store_true", default=True)
    run_cmd.add_argument("--no-conversation", action="store_false", dest="conversation")
    run_cmd.add_argument("--single-turn", action="store_true", default=True)
    run_cmd.add_argument("--multi-turn", action="store_false", dest="single_turn")
    run_cmd.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
