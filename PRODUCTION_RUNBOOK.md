# Weak-Hardware LLM Production Runbook

Date: 2026-06-26

This runbook turns the research checkpoint into a repeatable local workflow for this machine.

## What The Production Tool Does

The stable entry point is:

```powershell
python .\scripts\weak_llm.py --help
```

It provides:

- hardware/model inventory;
- automated `llama-bench` profiling across GPU layer counts and KV-cache types;
- stored local profiles under `profiles/`;
- spill-aware placement recommendations;
- generated `llama-cli` and `llama-server` commands;
- generated phase-split cache-reuse commands;
- adaptive normal-vs-phase planning with break-even prediction;
- persistent prompt slot cache metadata under `phase-cache/cache-index.json`;
- optional direct `llama-cli` execution.

Generated profiles and reports are ignored by git because they are machine-local measurements.

## Inventory

```powershell
python .\scripts\weak_llm.py inventory
```

This lists detected GPU memory, runtime paths, and local GGUF files.

## Profile A Model

Small smoke profile:

```powershell
python .\scripts\weak_llm.py profile `
  --model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --gpu-layers "0,-1" `
  --cache-types "q8_0" `
  --prompt-tokens 32 `
  --gen-tokens 8 `
  --repetitions 1
```

Production profile for the 31B larger-than-VRAM model:

```powershell
python .\scripts\weak_llm.py profile `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --gpu-layers "0,20,30,36,40,44,-1" `
  --cache-types "q8_0" `
  --prompt-tokens 128 `
  --gen-tokens 64 `
  --repetitions 2
```

The tool stores a JSON profile in `profiles/` and a markdown report in `benchmarks/`.

## Get Recommendation

```powershell
python .\scripts\weak_llm.py recommend `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf
```

For Gemma 4 31B Q4_0 on this RTX 3060 12 GB machine, the research baseline found:

- avoid naive `-ngl -1`;
- use around `-ngl 40`;
- use Q8 KV cache for generation-focused use.

The current local production profile selected:

- decode: `-ngl 40`, `q8_0/q8_0`, about `4.47` generation tok/s;
- prefill: `-ngl 36`, `q8_0/q8_0`, about `120.28` prompt tok/s;
- full offload: `-ngl -1`, about `1.61` generation tok/s, so it is slower than partial placement.

The production tool will rederive that from the stored profile instead of hard-coding it.

## Generate A One-Off CLI Command

```powershell
python .\scripts\weak_llm.py generate-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 512 `
  --prompt "Write a concise plan for optimizing local LLM inference."
```

Then run the printed command.

## Generate A Server Command

```powershell
python .\scripts\weak_llm.py serve-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --port 8080
```

The generated server command includes prompt cache support and a slot save directory. This is the preferred production path for repeated interactive work because prompt-cache reuse matters more than one-off CLI launch speed.

## Adaptive Planner

The unique layer is the adaptive planner:

```powershell
python .\scripts\weak_llm.py plan `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 64 `
  --prompt "Persistent local research prompt for adaptive weak hardware inference." `
  --expected-reuses 3
```

The planner:

- detects larger-than-VRAM pressure, current free VRAM, and current free system RAM;
- uses the stored profile to choose decode placement;
- uses the stored profile to choose a separate prefill placement;
- checks matching phase-split benchmark results when available;
- predicts the cache reuse break-even point;
- registers a deterministic planned slot file in `phase-cache/cache-index.json`;
- chooses `normal-server`, `phase-create`, or `phase-reuse`.

To print only the selected runnable command:

```powershell
python .\scripts\weak_llm.py adaptive-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 64 `
  --prompt "Persistent local research prompt for adaptive weak hardware inference." `
  --expected-reuses 3
```

The planner currently selected `phase-create` for Gemma 31B when expected reuse is `3`, because the measured phase result predicts break-even at `3` loaded reuses. With expected reuse `1`, it selected `normal-server`.

Manage cache metadata:

```powershell
python .\scripts\weak_llm.py cache-list --save
python .\scripts\weak_llm.py cache-prune --drop-planned
```

## Phase-Split Prototype

The unique experimental path is profile-derived through the production CLI:

```powershell
python .\scripts\weak_llm.py phase-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 64 `
  --compare-cold
```

Run the printed command. It will use the stored profile to choose compatible prefill and decode placements.

For persistent cache reuse, the phase runner also accepts `-SlotFile` and `-UseExistingSlot`. The adaptive planner emits those flags automatically when a matching slot cache is available.

The underlying command is still available directly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\phase-split-llama.ps1 `
  -Model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  -PrefillGpuLayers "36" `
  -DecodeGpuLayers "40" `
  -CacheType "q8_0" `
  -ContextSize 4096 `
  -NewTokens 64 `
  -CompareCold
```

Use this when testing repeated long-prompt workflows. It is not yet the default one-off production path because process reload overhead dominates.

## Operational Defaults For This Machine

- 12B Q4 model: full GPU offload is practical.
- 31B Q4 model: partial offload is required for useful speed.
- Default KV cache: `q8_0` for larger models unless quality testing shows a problem.
- Default context: start at 4096; increase only after profiling memory and latency.
- Avoid `-ngl -1` for larger-than-VRAM models unless profiling proves it is fastest.

## Maintenance

- Re-profile after driver, llama.cpp, or model changes.
- Keep model/runtime artifacts ignored; only scripts/docs should be committed.
- Treat profiles as local evidence, not universal truth for other machines.
