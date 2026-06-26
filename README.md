# SpillGuard LLM Planner

Adaptive weak-hardware inference planning for local GGUF models with `llama.cpp`.

SpillGuard helps run larger-than-VRAM models on consumer hardware by profiling the local machine, avoiding slow memory-spill placements, and choosing between normal `llama.cpp` serving and phase-split KV-cache reuse.

The project was developed around a Windows desktop with:

- NVIDIA GeForce RTX 3060, 12 GB VRAM
- Intel Core i7-13700KF
- 32 GB system RAM
- `llama.cpp` CUDA build `b9804-960d628f4`

The core target is practical local inference for models such as Gemma 31B Q4 GGUF that are larger than the available GPU memory.

## What This Project Does

SpillGuard:

- inventories local CPU/GPU/RAM state;
- profiles GGUF models with `llama-bench`;
- finds the best decode placement for generation speed;
- finds a separate prefill placement for prompt processing speed;
- detects VRAM and system RAM pressure;
- predicts whether phase-split prompt-cache reuse is worth using;
- manages persistent slot-cache metadata;
- emits runnable `llama-cli`, `llama-server`, and phase-split commands.

The important observation is that full GPU offload is not always best. On larger-than-VRAM models, `-ngl -1` can spill memory and become slower than a measured partial offload.

## Current Measured Result

On the RTX 3060 12 GB test machine, Gemma 31B Q4 showed:

| Placement | Purpose | Result |
| --- | --- | ---: |
| CPU only, `-ngl 0` | baseline decode | ~1.81 tok/s |
| full offload, `-ngl -1` | naive GPU decode | ~1.61 tok/s |
| partial offload, `-ngl 40` | selected decode | ~4.47 tok/s |
| partial offload, `-ngl 36` | selected prefill | ~120.28 prompt tok/s |

The planner therefore chooses:

- decode: `-ngl 40`, `q8_0/q8_0`
- prefill: `-ngl 36`, `q8_0/q8_0`
- avoid naive full offload for this model on this hardware

These values are not hard-coded. They come from local profiles under `profiles/`, which are ignored by git because they are machine-specific.

## What Is Innovative

This project does not claim to invent prefill/decode disaggregation or KV-cache reuse. Those ideas exist in larger serving systems.

The innovative angle here is adapting those ideas to a single weak consumer machine:

1. Profile the actual local hardware and GGUF model.
2. Detect that full GPU offload is slower than partial offload.
3. Choose one GPU-layer placement for prefill and another for decode.
4. Save a `llama.cpp` slot KV cache from the prefill placement.
5. Restore that slot under the decode placement.
6. Decide automatically whether normal serving, cache creation, or cache reuse is best.

This makes a larger-than-VRAM model more usable without requiring a server cluster, multi-GPU system, or custom model runtime.

## Live Layer Migration Track

The next research goal is true live layer migration inside llama.cpp while generation is running.

That work is separate from the current Python planner. It changes llama.cpp itself so repeating model layers can be loaded into independent backend buffers, copied to another device while the context is alive, and freed from the old device after the graph is reset. The patch also includes an opt-in live policy that checks VRAM/RAM after successful decode calls and moves one layer at a time.

Current artifact:

```text
patches/llama.cpp/0001-experimental-live-layer-migration.patch
```

Implementation notes:

```text
docs/live-layer-migration.md
```

This patch compiles in a CPU-only MSVC llama.cpp library build. CUDA runtime validation, VRAM-before/after proof, and production-grade policy tuning are still required before this is production-ready.

## Repository Layout

```text
.
|-- README.md
|-- docs/
|   `-- live-layer-migration.md
|-- patches/
|   `-- llama.cpp/
|       `-- 0001-experimental-live-layer-migration.patch
|-- scripts/
|   |-- weak_llm.py
|   |-- phase-split-llama.ps1
|   `-- profile-llama-fit.ps1
`-- .gitignore
```

Ignored local artifacts:

- `models/`
- `tools/`
- `benchmarks/`
- `profiles/`
- `phase-cache*/`
- Python bytecode

That keeps the public repository small and prevents model files, local binaries, benchmark logs, and machine-specific cache files from being published.

## Requirements

- Windows PowerShell
- Python 3.11 or newer
- NVIDIA GPU with `nvidia-smi` for GPU/RAM pressure reporting
- `llama.cpp` binaries:
  - `llama-bench.exe`
  - `llama-cli.exe`
  - `llama-server.exe`
- GGUF model files stored locally under `models/`

Default expected runtime location:

```text
tools/llama.cpp-b9804-cuda124/
```

Default expected model location:

```text
models/
```

You can override paths through CLI flags such as `--llama-bench`, `--llama-cli`, `--llama-server`, `--models-dir`, and `--profile-dir`.

## Quick Start

Show inventory:

```powershell
python .\scripts\weak_llm.py inventory
```

Profile a model:

```powershell
python .\scripts\weak_llm.py profile `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --gpu-layers "0,20,30,36,40,44,-1" `
  --cache-types "q8_0" `
  --prompt-tokens 128 `
  --gen-tokens 64 `
  --repetitions 2
```

Print the recommendation:

```powershell
python .\scripts\weak_llm.py recommend `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf
```

Start a normal server:

```powershell
python .\scripts\weak_llm.py serve `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --port 8080
```

Keep that PowerShell window open while using the model. When the server is ready, open:

```text
http://127.0.0.1:8080
```

If you only want to print the underlying `llama-server.exe` command without starting it, use:

```powershell
python .\scripts\weak_llm.py serve-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --port 8080
```

## Adaptive Planning

The adaptive planner decides whether to use normal `llama-server`, create a phase-split cache, or reuse an existing phase-split cache.

```powershell
python .\scripts\weak_llm.py plan `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 64 `
  --prompt "Persistent local research prompt for adaptive weak hardware inference." `
  --expected-reuses 3
```

The plan includes:

- selected strategy: `normal-server`, `phase-create`, or `phase-reuse`;
- reasons for the strategy;
- current VRAM and RAM pressure;
- selected prefill/decode placements;
- break-even reuse estimate;
- cache metadata;
- the exact command to run.

To print only the selected command:

```powershell
python .\scripts\weak_llm.py adaptive-command `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context 4096 `
  --new-tokens 64 `
  --prompt "Persistent local research prompt for adaptive weak hardware inference." `
  --expected-reuses 3
```

## Persistent Slot Cache Reuse

The phase runner can create and reuse deterministic slot files:

```powershell
python .\scripts\weak_llm.py adaptive-command `
  --model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --context 2048 `
  --new-tokens 4 `
  --prompt "Small persistent cache verification prompt." `
  --expected-reuses 2 `
  --strategy phase
```

After the slot file exists, the planner can switch to `phase-reuse` and emit a command with:

```powershell
-UseExistingSlot
```

Manage cache metadata:

```powershell
python .\scripts\weak_llm.py cache-list --save
python .\scripts\weak_llm.py cache-prune --drop-planned
```

## Direct Phase-Split Command

The lower-level phase runner is still available:

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

To reuse an existing slot:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\phase-split-llama.ps1 `
  -Model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -PrefillGpuLayers "-1" `
  -DecodeGpuLayers "-1" `
  -CacheType "q8_0" `
  -ContextSize 2048 `
  -NewTokens 4 `
  -SlotFile "adaptive-example.bin" `
  -UseExistingSlot
```

## Validation Used

The current local version was checked with:

```powershell
python -m py_compile .\scripts\weak_llm.py
```

```powershell
powershell.exe -NoProfile -Command '$tokens = $null; $errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path ''.\scripts\phase-split-llama.ps1''), [ref]$tokens, [ref]$errors) | Out-Null; if ($errors.Count) { $errors | Format-List *; exit 1 } else { ''ok'' }'
```

Runtime checks performed locally:

- inventory detects GPU and free RAM;
- Gemma 31B expected reuse `3` selects `phase-create`;
- Gemma 31B expected reuse `1` selects `normal-server`;
- Qwen small-model slot cache creation works;
- Qwen slot cache reuse works with `used_existing_slot: true`.

## Limitations

- This is a local orchestration layer around `llama.cpp`, not a new inference engine.
- The live layer migration patch is experimental and still needs CUDA runtime validation.
- Profiles are hardware-specific and should not be reused across machines.
- Long-running Gemma 31B phase-cache creation can still be slow because process startup and model loading matter.
- The planner estimates token counts from prompt length when it does not have tokenizer output.
- The project currently targets Windows PowerShell first.
