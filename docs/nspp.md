# Neural Streaming with Predictive Prefetch

NSPP is the next speed-focused research track for this repository. The target is explicit:

```text
Model:    Gemma-style 31B GGUF
Hardware: RTX 3060 12 GB, i7-13700KF, 32 GB RAM
Target:   7-10 generated tokens/second
```

The current live layer migration patch proves that llama.cpp can move real layer buffers between CUDA and CPU while a context is alive. That is valuable as a memory governor, but our Gemma 31B benchmarks show that migration alone is not enough to hit the speed target.

## Current Local Evidence

Measured with `llama-bench` on the patched CUDA build:

```text
Model: gemma-4-31B_q4_0-it.gguf
Build: llama.cpp 960d628f, CUDA, RTX 3060
Test:  tg128, model-load time excluded
```

| Configuration | Result |
| --- | ---: |
| `-ngl 36` | 3.52 tok/s |
| `-ngl 38` | 3.77 tok/s |
| `-ngl 40` | 3.17 tok/s in one run, worse in later edge-of-VRAM runs |
| `-ngl 38`, flash attention forced | 3.51 tok/s |
| `-ngl 38`, 12 CPU threads | 3.55 tok/s |

Best clean result so far is about `3.7-3.8 tok/s`. The target requires roughly `2x-3x` improvement.

The live migration probe also showed that moving Gemma 31B layers from CUDA to CPU costs about `50-110 ms` per layer on this machine. That is much higher than the optimistic `15 ms` streaming estimate in the initial NSPP sketch. Therefore, per-token CPU-to-GPU layer streaming is not the primary path to `7-10 tok/s`.

## What NSPP Should Mean Here

NSPP should not be framed as "stream every missing layer and hide all transfer cost." That is unlikely to work for single-token decode on this hardware because PCIe transfer is too slow relative to useful compute.

The grounded version is:

1. Make most or all of the 31B model fit in 12 GB VRAM with mixed low-bit quantization.
2. Use speculative decoding to reduce the number of expensive target-model decode passes.
3. Keep live migration as a safety governor for VRAM pressure, not as the main speed path.
4. Add KV-cache hierarchy only after short-context speed is solved.
5. Treat sparse activation as a later custom-kernel research path.

## Component Assessment

| Component | Novel alone? | Expected speed impact | Local status |
| --- | --- | --- | --- |
| Per-layer sensitivity quantization | No | High if it allows near-full GPU residency | Not implemented |
| Double-buffered async layer streaming | No | Low to medium for single-token decode | Risky; layer moves are 50-110 ms |
| Speculative decoding | No | High if draft acceptance is good | Not implemented |
| Hierarchical KV cache | No | Medium for long context, low for short-context speed | Not implemented |
| Sparse activation | No | Potentially high, high implementation cost | Not implemented |
| Live layer migration | Uncommon in llama.cpp form | Stability/governor, not raw speed | Prototype implemented |

The possible novelty is the integrated weak-hardware runtime: low-bit placement, speculative verification, and live VRAM governance tuned for a single consumer GPU.

## Priority 1: Low-Bit Full-Fit Experiment

The fastest plausible route is to reduce the model enough that more layers stay on GPU.

Experiment:

1. Create a speed-test low-bit GGUF from the current Q4 model.
2. Benchmark whether all or nearly all layers can be offloaded.
3. Accept the experiment only if generation reaches at least `6 tok/s` before speculative decoding.

Candidate quants:

| Quant | Purpose | Risk |
| --- | --- | --- |
| `IQ3_M` or `IQ3_XS` | Better quality, may still not fully fit | Might miss speed target |
| `IQ2_M` | Strong speed/memory test | Quality loss, especially when requantized from Q4 |
| `IQ2_XS` / `IQ2_XXS` | Maximum fit pressure test | Likely quality loss |

Important: requantizing from Q4 is only a speed feasibility test. A real release should quantize from BF16/F16 with calibration or an importance matrix.

Success gate:

```text
If low-bit full/nearly-full GPU residency reaches >= 6 tok/s,
continue to speculative decoding.

If it remains below 5 tok/s,
31B at 7-10 tok/s on this exact hardware likely requires a smaller model,
heavier quality loss, custom kernels, or a different architecture.
```

## Priority 2: Speculative Decoding

Speculative decoding is the most realistic multiplier after low-bit fitting.

Expected math:

```text
Target model baseline after low-bit fit: 6 tok/s
Draft acceptance: 50-70%
Effective speedup: about 1.5x-2.5x
Possible result: 9-15 tok/s if acceptance is good
```

This requires a compatible draft model:

- same tokenizer;
- cheap enough to run fully on GPU;
- similar enough distribution for useful acceptance;
- ideally a smaller Gemma-family model.

Do not count speculative speedup before measuring acceptance rate. A poor draft model can add overhead without improving throughput.

## Priority 3: Live Migration As Governor

Live migration remains useful, but its role changes:

- start near the aggressive GPU residency limit;
- if VRAM becomes unsafe, demote one layer;
- if VRAM recovers, promote one layer;
- avoid out-of-memory crashes;
- keep a target VRAM margin such as `512-1024 MiB`.

This is a stability feature. It should not be expected to beat the fastest static placement when memory pressure is absent.

## Lower Priority: Streaming And KV Hierarchy

Double-buffered streaming is still worth studying, but not as the first speed path. The local migration timings imply that streaming whole layers every token would consume too much time.

Streaming may become useful only when:

- weights are very low-bit and packed efficiently;
- transfer uses pinned host memory;
- copy is done into preallocated GPU buffers;
- compute is batched, speculative, or grouped enough to hide transfer;
- the implementation avoids repeated allocation and graph rebuild overhead.

Hierarchical KV cache matters more for long contexts than for the current `tg128` speed target. It should be deferred until short-context generation is close to target.

## Implementation Plan

### Gate A: Quantization Feasibility

1. Fix or replace the local `llama-quantize` workflow.
2. Produce an `IQ2_M` or `IQ3_XS` speed-test GGUF.
3. Run:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-bench.exe `
  -m .\models\<low-bit-gemma31b>.gguf `
  -ngl -1 `
  -p 0 `
  -n 128 `
  -r 3 `
  -o csv
```

4. Record:
   - whether all layers fit;
   - generated tok/s;
   - free VRAM;
   - quality sanity output from `llama-cli`.

### Gate B: Speculative Feasibility

1. Pick a small draft model with matching tokenizer if possible.
2. Measure draft tok/s.
3. Implement or reuse speculative decoding in llama.cpp.
4. Measure:
   - target passes per accepted token;
   - acceptance rate;
   - effective tok/s;
   - quality drift.

### Gate C: Governor Integration

1. Keep the live migration policy disabled for speed baseline.
2. Enable it only with a VRAM margin target.
3. Measure speed loss versus avoided OOM risk.

## Claims We Should Not Publish Yet

Do not publish these as achieved results until measured locally:

- `8-35 tok/s` on Gemma 31B with NSPP;
- `<1% quality loss` from PLSQ;
- `70% PCIe latency hidden`;
- shared embeddings between arbitrary draft and target models;
- `30% compute reduction` from sparse activation;
- 8K context at target speed.

The public claim should be narrower:

```text
SpillGuard has a working live layer migration prototype.
NSPP is the next experimental track targeting 7-10 tok/s by combining
low-bit mixed precision, speculative decoding, and live VRAM governance.
```

## References

- FlexGen: <https://arxiv.org/abs/2303.06865>
- Fast Inference from Transformers via Speculative Decoding: <https://arxiv.org/abs/2211.17192>
- Speculative Decoding with Big Little Decoder: <https://arxiv.org/abs/2302.07863>
- AWQ: <https://arxiv.org/abs/2306.00978>
- GPTQ: <https://arxiv.org/abs/2210.17323>
- SmoothQuant: <https://arxiv.org/abs/2211.10438>
- H2O KV cache: <https://arxiv.org/abs/2306.14048>
- PowerInfer: <https://arxiv.org/abs/2312.12456>
- Deja Vu sparse inference: <https://arxiv.org/abs/2310.17157>
- PagedAttention / vLLM: <https://arxiv.org/abs/2309.06180>
