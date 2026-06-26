# True Live Layer Migration

This track targets live migration of loaded llama.cpp model layers between GPU and CPU while a context is alive. The goal is not to restart `llama-server`, not to choose `-ngl` only before launch, and not to merely schedule compute on another backend. The target is to move real model weight buffers and release the old memory.

## Current State

The repository contains an experimental llama.cpp patch:

```text
patches/llama.cpp/0001-experimental-live-layer-migration.patch
```

Patch baseline:

```text
ggml-org/llama.cpp commit 960d628f4
```

Local validation completed:

- `git diff --check` passes in the patched llama.cpp source tree.
- MSVC Release CPU-only `llama` library build passes.
- The automatic policy compiles into the patched `llama` library.
- `llama-live-migration-probe` builds and runs against the local Qwen 0.5B GGUF in a CPU-only build, exercising migration-ready load, decode, manual `llama_live_migrate_layer()`, graph reset, and continued decode.
- CUDA Toolkit 13.3 builds the patched probe for RTX 3060 architecture `86`.
- CUDA manual migration proof passes on RTX 3060: layer 23 moved from CUDA0 to CPU during generation, free GPU memory increased by 10 MiB, and decode continued.
- CUDA automatic pressure policy proof passes with an intentionally high VRAM threshold: the policy demoted layers 23, 22, 21, and 20 while decode continued.

Future hardening still left:

- Gemma 31B benchmark against the best static `-ngl` baseline.
- Long-run stress testing with larger contexts and many migration cycles.
- Cross-GPU/CUDA-version validation beyond the RTX 3060 local proof.
- Decode-latency SLO policy tuning; the built-in policy currently uses VRAM, CPU RAM, CPU utilization, and migration-copy-time thresholds.

## What Is New

Normal llama.cpp loads tensors into backend buffers grouped mainly by buffer type. That means many layers share one CUDA buffer. If one layer is copied to CPU, the old shared CUDA buffer cannot be freed, so VRAM does not actually go down.

The patch adds an opt-in migration-ready load mode:

```text
LLAMA_EXPERIMENTAL_LAYER_BUFFERS=1
```

When enabled, repeating-layer tensors are loaded into independent per-layer contexts and backend buffers. This gives each transformer layer its own releasable memory ownership unit.

The patch also adds a C API:

```c
int32_t llama_live_migrate_layer(
    struct llama_context * ctx,
    int32_t layer,
    int32_t device);
```

`device < 0` means CPU. `device >= 0` means the index in the model offload-device list.

The patch now also includes an opt-in live policy inside `llama_context`. When enabled, it checks memory, CPU utilization, and migration copy time at the end of successful `llama_decode()` calls, demotes one GPU layer to CPU under VRAM pressure, and promotes only layers it previously demoted after VRAM recovers.

## How Migration Works

At a safe point between decode calls, manual and automatic migration both use the same primitive:

1. Synchronize the backend scheduler so no GPU graph is still reading the old weights.
2. Allocate a new ggml context for the target layer.
3. Duplicate the layer tensor metadata into that context.
4. Allocate a destination backend buffer, CPU or GPU.
5. Copy all tensors from the old layer buffer to the new buffer.
6. Swap every tensor pointer in `llama_layer`.
7. Update `tensors_by_name`.
8. Release the old per-layer context and backend buffer.
9. Reset scheduler and graph reuse so the next decode builds against the new placement.

That last part matters: without graph reset, llama.cpp can reuse a graph that still assumes the previous tensor/backend placement.

## Automatic Policy

Set both variables to enable real live migration:

```powershell
$env:LLAMA_EXPERIMENTAL_LAYER_BUFFERS = "1"
$env:LLAMA_LIVE_MIGRATION = "1"
```

Optional thresholds:

```powershell
$env:LLAMA_LIVE_MIGRATION_MIN_FREE_MB = "768"
$env:LLAMA_LIVE_MIGRATION_RESTORE_FREE_MB = "2048"
$env:LLAMA_LIVE_MIGRATION_MIN_RAM_FREE_MB = "2048"
$env:LLAMA_LIVE_MIGRATION_MAX_CPU_PCT = "85"
$env:LLAMA_LIVE_MIGRATION_MAX_COPY_MS = "750"
$env:LLAMA_LIVE_MIGRATION_COPY_COOLDOWN_DECODE = "16"
$env:LLAMA_LIVE_MIGRATION_INTERVAL_DECODE = "1"
```

Policy behavior:

- if free VRAM on the first non-CPU model device drops below `LLAMA_LIVE_MIGRATION_MIN_FREE_MB`, demote one currently GPU-resident layer to CPU;
- skip demotion when CPU RAM is below `LLAMA_LIVE_MIGRATION_MIN_RAM_FREE_MB`;
- skip demotion when live CPU utilization is above `LLAMA_LIVE_MIGRATION_MAX_CPU_PCT`;
- measure each real layer migration and keep last, moving-average, max, and count stats;
- if a migration exceeds `LLAMA_LIVE_MIGRATION_MAX_COPY_MS`, pause further automatic moves for `LLAMA_LIVE_MIGRATION_COPY_COOLDOWN_DECODE` policy ticks;
- if free VRAM rises above `LLAMA_LIVE_MIGRATION_RESTORE_FREE_MB`, promote the most recently policy-demoted layer back to GPU;
- use hysteresis so the policy does not immediately bounce the same layer back and forth;
- run only after a successful decode, so the next decode builds a fresh graph with the new placement.

## Apply The Patch

From this repository root:

```powershell
git clone https://github.com/ggml-org/llama.cpp .\tools\llama.cpp-src
cd .\tools\llama.cpp-src
git checkout 960d628f4
git apply ..\..\patches\llama.cpp\0001-experimental-live-layer-migration.patch
```

If `tools/llama.cpp-src` already exists, reset it only if you do not need local changes there:

```powershell
cd .\tools\llama.cpp-src
git checkout 960d628f4
git apply ..\..\patches\llama.cpp\0001-experimental-live-layer-migration.patch
```

## Build Check Used Locally

CPU-only compile check:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  -S . `
  -B build-live-migration `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DGGML_CUDA=OFF `
  -DGGML_NATIVE=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_SERVER=OFF `
  -DLLAMA_BUILD_TESTS=OFF
```

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  --build build-live-migration `
  --config Release `
  --target llama
```

CUDA validation should use the same patch with `-DGGML_CUDA=ON` and then run a migration test against a small GGUF model before trying Gemma 31B.

## Probe Build And Run

The patch adds a probe executable:

```text
examples/live-migration-probe/live-migration-probe.cpp
```

Build it from the patched llama.cpp tree:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  -S . `
  -B build-live-migration `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DGGML_CUDA=OFF `
  -DGGML_NATIVE=OFF `
  -DLLAMA_BUILD_EXAMPLES=ON `
  -DLLAMA_BUILD_SERVER=OFF `
  -DLLAMA_BUILD_TESTS=OFF
```

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
  --build build-live-migration `
  --config Release `
  --target llama-live-migration-probe
```

CPU-only smoke run used locally:

```powershell
$env:LLAMA_EXPERIMENTAL_LAYER_BUFFERS = "1"
.\build-live-migration\bin\Release\llama-live-migration-probe.exe `
  -m C:\Users\fteki\Documents\LLM\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ngl 0 `
  -n 4 `
  --migrate-at 0 `
  --target cpu `
  "Hello"
```

Observed result:

```text
probe: migration_result=0
probe: decoded=4 migrated=true layer=23 target=cpu auto_policy=false
```

In the CPU-only build the layer is already on CPU, so this is only a runtime smoke test. The same probe is intended for CUDA validation with `-DGGML_CUDA=ON`, `-ngl` greater than zero, and `--target cpu`, where the `before/after` GPU memory lines should prove whether VRAM is actually released.

For a strict CUDA proof, use the probe's GPU checks:

```powershell
.\build-live-migration-cuda\bin\Release\llama-live-migration-probe.exe `
  -m C:\Users\fteki\Documents\LLM\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ngl 12 `
  -n 4 `
  --migrate-at 0 `
  --target cpu `
  --require-gpu `
  --expect-gpu-delta-mb 1 `
  "Hello"
```

`--require-gpu` fails the run if no non-CPU backend is available. `--expect-gpu-delta-mb` fails the run if GPU free memory does not increase by at least that many MiB after the manual layer migration. For larger models, raise the expected delta after observing the per-layer buffer size in the probe logs.

Observed local CUDA result on RTX 3060:

```text
probe: before-manual-migration gpu: CUDA0 free 10971.00 MiB / 12287.50 MiB
probe: migrating layer 23 to cpu
migrate_layer: migrated layer 23 to CPU (CPU, 9.27 MiB)
live_migrate_layer: migrated layer 23 to CPU in 2.61 ms (avg 2.61 ms, max 2.61 ms, count 1)
probe: migration_result=0
probe: after-manual-migration gpu: CUDA0 free 10981.00 MiB / 12287.50 MiB
probe: gpu_free_delta_after_migration=10.00 MiB required=1.00 MiB
probe: decoded=4 migrated=true layer=23 target=cpu auto_policy=false
```

The repository also includes a wrapper for the same strict check:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\cuda-live-migration-check.ps1 `
  -Model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -GpuLayers 12 `
  -CudaArchitecture 86 `
  -MinGpuDeltaMb 1
```

The wrapper locates CUDA Toolkit under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`, configures a CUDA build with `GGML_CUDA=ON`, builds `llama-live-migration-probe`, and fails unless the probe observes the required increase in free GPU memory.

Automatic policy proof command:

```powershell
.\build-live-migration-cuda\bin\Release\llama-live-migration-probe.exe `
  -m C:\Users\fteki\Documents\LLM\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ngl 12 `
  -n 4 `
  --auto-policy `
  "Hello"
```

Observed local result: the policy saw free VRAM below the forced threshold, demoted layers 23, 22, 21, and 20 to CPU, logged migration copy times around 2.5-2.9 ms per layer, reset the scheduler after each move, and completed decode.

## Runtime Use

The patch exposes the migration primitive for explicit control:

```c
llama_live_migrate_layer(ctx, layer_id, -1); // move layer to CPU
llama_live_migrate_layer(ctx, layer_id,  0); // move layer to first model GPU
```

The built-in automatic policy does not require a server endpoint, but a production controller could still improve decisions by adding strict decode-latency SLOs and workload-specific rules.

## Next Engineering Steps

1. Run a larger Gemma 31B Q4 benchmark on the RTX 3060 and compare against the current best static `-ngl 40` result.
2. Stress test repeated demote/promote cycles over long contexts.
3. Add a server route or debug endpoint that forces migration deterministically without a separate probe executable.
4. Extend the policy with strict decode-latency SLOs.
5. Validate the patch against another CUDA version and GPU architecture.

## Why This Is Different From The Planner

The current SpillGuard planner chooses the best startup placement and can switch processes or use slot caches. It cannot change loaded model weights inside a running llama.cpp context.

This patch changes the runtime memory model. It makes layer placement mutable after load, with the old layer buffer released. That is the innovative part needed for true weak-hardware operation.
