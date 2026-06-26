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

Not completed yet:

- CUDA build validation.
- Runtime migration test with a real GGUF model.
- VRAM-before/after proof on a CUDA backend.
- CPU utilization pressure is not handled yet; the built-in policy currently uses VRAM and CPU RAM thresholds.

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

The patch now also includes an opt-in live policy inside `llama_context`. When enabled, it checks memory at the end of successful `llama_decode()` calls, demotes one GPU layer to CPU under VRAM pressure, and promotes only layers it previously demoted after VRAM recovers.

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
$env:LLAMA_LIVE_MIGRATION_INTERVAL_DECODE = "1"
```

Policy behavior:

- if free VRAM on the first non-CPU model device drops below `LLAMA_LIVE_MIGRATION_MIN_FREE_MB`, demote one currently GPU-resident layer to CPU;
- skip demotion when CPU RAM is below `LLAMA_LIVE_MIGRATION_MIN_RAM_FREE_MB`;
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

## Runtime Use

The patch exposes the migration primitive for explicit control:

```c
llama_live_migrate_layer(ctx, layer_id, -1); // move layer to CPU
llama_live_migrate_layer(ctx, layer_id,  0); // move layer to first model GPU
```

The built-in automatic policy does not require a server endpoint, but a production controller could still improve decisions by adding decode latency, migration copy time, CPU utilization, and workload-specific rules.

## Next Engineering Steps

1. Build the patched tree with CUDA enabled.
2. Run `llama-live-migration-probe` with a CUDA build and a small GGUF model.
3. Log VRAM before and after migration to prove old CUDA memory is released.
4. Add a test executable or server route that forces migration deterministically for debugging.
5. Extend the policy with CPU utilization and latency-aware decisions.
6. Benchmark Gemma 31B Q4 on the RTX 3060 and compare against the current best static `-ngl 40` result.

## Why This Is Different From The Planner

The current SpillGuard planner chooses the best startup placement and can switch processes or use slot caches. It cannot change loaded model weights inside a running llama.cpp context.

This patch changes the runtime memory model. It makes layer placement mutable after load, with the old layer buffer released. That is the innovative part needed for true weak-hardware operation.
