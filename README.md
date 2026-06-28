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

The active research track is true live layer migration inside llama.cpp while generation is running.

That work is separate from the current Python planner. It changes llama.cpp itself so repeating model layers can be loaded into independent backend buffers, copied to another device while the context is alive, and freed from the old device after the graph is reset. The patch also includes an opt-in live policy that checks VRAM, CPU RAM, CPU utilization, and migration copy time after successful decode calls and moves one layer at a time.

Current artifact:

```text
patches/llama.cpp/0001-experimental-live-layer-migration.patch
```

Implementation notes:

```text
docs/live-layer-migration.md
```

This patch compiles in CPU-only and CUDA MSVC llama.cpp builds and includes a `llama-live-migration-probe` example for runtime testing. On the RTX 3060 validation machine, the probe moved a live CUDA layer to CPU during generation, GPU free memory increased by 10 MiB, and decoding continued. The automatic pressure policy also demoted layers under a forced VRAM threshold. Production-grade long-run stress testing and Gemma 31B benchmarking are still future work.

## NSPP Speed Track

The next speed target is `7-10 tok/s` on Gemma-style 31B models with the same RTX 3060 12 GB machine. Current clean `llama-bench` measurements put Gemma 31B Q4 around `3.5-3.8 tok/s`, so live migration alone is not enough.

The grounded NSPP plan treats live migration as a VRAM safety governor and focuses the speed path on mixed low-bit quantization plus speculative decoding:

```text
docs/nspp.md
```

Unproven NSPP claims are tracked as experiment gates rather than published results.

## Consumer 100B+ Track

The next research goal is larger than NSPP: useful 100B+ local inference on consumer GPU hardware. The project treats exact dense 100B inference at interactive speed as physically unrealistic on 12 GB VRAM, so the proposed direction is **SAGE-100: Sparse Assisted Giant Execution**.

SAGE-100 uses a resident proxy model for most tokens and consults a compressed 100B+ oracle only when confidence or task policy requires it. The oracle is executed as sparse active blocks under a measured byte budget, with live migration, compressed KV, and exact slow fallback.

```text
docs/consumer-100b-research.md
docs/sage-active-byte-oracle.md
docs/sage-persistent-runtime.md
```

Current contract check:

```powershell
python .\scripts\sage_contract_check.py `
  --json-out .\benchmarks\sage-active-byte-contract-check.json
```

This check intentionally reports partial progress rather than completion. The
current result is `43` passed gates and `3` failed gates. It separates proven
C++ scheduler/live smoke gates, sparse token-row verification, proxy shortlist
coverage, live proxy top-k shortlist telemetry, the live-shortlist-to-CUDA
verifier bridge, same-prompt sparse verifier fallback detection, sparse-oracle
runtime-step component replay, measured sparse-fallback runtime projection,
overlap/prefetch budget target, measured CUDA H2D/kernel overlap smoke,
measured host-prefetch/CUDA-overlap smoke, resident pinned page-cache replay
smoke, resident page-cache 10 tok/s budget target, reduced resident page-cache
10 tok/s projection, reduced Q4_0 dequant/matvec compute smokes, reduced
real-activation matvec scoring, reduced page signal-quality probing,
signal-aware page selection, signal-aware 10 tok/s page-cache projection, live
cross-activation selector robustness, tiered KV accounting, and offline runtime-tiered KV context sweeps from the
still-missing measured proxy budget, executable 100B sparse oracle pager, and
packed KV attention integration.

Budget sanity check:

```powershell
python .\scripts\sage_budget.py --target-tps 7 --params-b 100 --quant-bpw 2 --oracle-call-rate 0.25
```

Throughput sweep:

```powershell
python .\scripts\sage_simulate.py --target-tps 7 --proxy-tps 25 --params-b 100 --quant-bpw 2
```

GGUF block index:

```powershell
python .\scripts\sage_gguf_blocks.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf
```

Sparse oracle block plan:

```powershell
python .\scripts\sage_block_plan.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy balanced
python .\scripts\sage_block_plan.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy balanced --ffn-min-share 0.55 --attention-max-share 0.35
```

Sparse oracle page ledger:

```powershell
python .\scripts\sage_oracle_pager.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --budget-gib 2.33 `
  --stage-buffer-gib 0.75 `
  --policy balanced `
  --ffn-min-share 0.55 `
  --attention-max-share 0.35 `
  --json-out .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json
```

This is a plan-only pager artifact, not sparse CUDA execution. Current result:
`79` planned pages, `2.246 GiB` active bytes, `9.65%` of a 100B 2-bit
reference, `4` staged transfers, and about `100.5 ms` estimated PCIe transfer.

Sparse oracle page staging smoke:

```powershell
python .\scripts\sage_oracle_pager_staging.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-staging-gemma31b-full.json
```

This is measured CPU file-to-host staging, not CUDA sparse execution. Current
result: all `79` planned pages from the Gemma 31B GGUF were resolved to real
tensor byte ranges and staged through bounded buffers, totaling `2.246 GiB`;
max live buffer use was `0.712 GiB` under the `0.750 GiB` stage budget, with
`1.88 GiB/s` measured staging throughput.

Sparse oracle CUDA staging smoke:

```powershell
python .\scripts\sage_oracle_cuda_staging.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-staging-gemma31b-full.json
```

This measures pinned-host to CUDA device-buffer transfer for the same planned
pages. Current result: all `79` pages and `2.246 GiB` moved through CUDA staging
buffers in `113.01 ms`, about `19.88 GiB/s` H2D throughput, with max live device
buffer use still `0.712 GiB` under the `0.750 GiB` stage budget. This proves the
transport layer, not sparse oracle compute.

Sparse oracle CUDA kernel smoke:

```powershell
python .\scripts\sage_oracle_cuda_kernel_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-kernel-gemma31b-full.json
```

This compiles a tiny CUDA byte-sum kernel with NVRTC and launches it over the
same staged page buffers. Current result: all `2.246 GiB` of selected page bytes
were touched by a GPU kernel in `18.34 ms` after `106.98 ms` of H2D transfer,
with `122.49 GiB/s` measured kernel-touch throughput. This is a kernel
consumption smoke, not transformer scoring.

Sparse oracle CUDA overlap smoke:

```powershell
python .\scripts\sage_oracle_cuda_overlap_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --json-out .\benchmarks\sage-oracle-page-cuda-overlap-gemma31b-full.json
```

This pre-stages the same selected pages into pinned host buffers, then uses two
CUDA streams and two device buffers to overlap H2D for the next stage with the
byte-touch kernel for the current stage. Current result: the full `2.246 GiB`
page plan measured `139.51 ms` as separate H2D+kernel work and `123.13 ms` as
an overlapped GPU window, saving `16.38 ms` (`11.7%`). This proves overlap
mechanics for transport plus a touch kernel, not sparse transformer execution.

Sparse oracle host-prefetch plus CUDA-overlap smoke:

```powershell
python .\scripts\sage_oracle_cuda_prefetch_overlap_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --json-out .\benchmarks\sage-oracle-page-cuda-prefetch-overlap-gemma31b-full.json
```

This keeps host staging inside the measured wall time: one background worker
reads GGUF pages into a pinned host-buffer ring while CUDA transfers and the
byte-touch kernel run on separate streams. Current result: for the same
`2.246 GiB` page plan, sequential host-read+H2D+kernel components totaled
`1233.49 ms`; the pipelined wall time was `1116.22 ms`, saving `117.27 ms`
(`9.5%`). The result also shows the problem clearly: host reads still dominate,
so production needs a resident pinned page cache and real sparse kernels.

Sparse oracle resident page-cache replay smoke:

```powershell
python .\scripts\sage_oracle_cuda_page_cache_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --replays 3 `
  --json-out .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-full-replay3.json
```

This builds the same selected page set once into resident pinned host memory,
then replays multiple fallback passes from that cache through CUDA H2D plus the
byte-touch kernel. Current result: cache build took `1464.97 ms` for
`2.246 GiB`; after that, three replay passes averaged `139.15 ms` of GPU window
per replay and reused `4.493 GiB` from cache hits. This proves the cache reuse
mechanism, but it is still transport plus touch-kernel work, not sparse
transformer execution.

Resident page-cache 10 tok/s budget target:

```powershell
python .\scripts\sage_page_cache_budget.py `
  --json-out .\benchmarks\sage-page-cache-budget-hard120-resident-cache-10tps.json
```

Using the measured resident cache replay as the fallback cost, the hard120
format-scaffold path projects to `9.72 tok/s`. To reach `10 tok/s` at the same
proxy speed and fallback rate, the active page set must shrink from `2.246 GiB`
to about `1.180 GiB`, a `47.5%` active-byte reduction, or fallback rate must
drop from `4.33%` to about `2.28%`.

Reduced resident page-cache 10 tok/s projection:

```powershell
python .\scripts\sage_oracle_pager.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --budget-gib 1.18 `
  --stage-buffer-gib 0.50 `
  --target-tps 10 `
  --max-active-percent-10tps 5.07 `
  --policy balanced `
  --ffn-min-share 0.55 `
  --attention-max-share 0.35 `
  --json-out .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json

python .\scripts\sage_oracle_cuda_page_cache_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json `
  --replays 3 `
  --json-out .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-balanced-1180mib-replay3.json

python .\scripts\sage_page_cache_budget.py `
  --page-cache-json .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-balanced-1180mib-replay3.json `
  --page-ledger-json .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json `
  --json-out .\benchmarks\sage-page-cache-budget-hard120-resident-cache-1180mib-10tps.json
```

Current result: the reduced page ledger selects `69` pages and `1.075 GiB`
active bytes, `4.62%` of a 100B 2-bit reference. The resident page-cache replay
averages `53.83 ms` per replay, and the hard120 projection reaches
`10.08 tok/s`. This is a measured transport/cache projection, not live sparse
transformer execution.

Reduced sparse oracle Q4_0 compute smokes:

```powershell
python .\scripts\sage_oracle_cuda_dequant_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-dequant-gemma31b-balanced-1180mib.json

python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json `
  --limit-stages 0 `
  --score-top-k 5 `
  --max-score-tensors 8 `
  --cpu-check-rows 16 `
  --json-out .\benchmarks\sage-oracle-page-cuda-matvec-gemma31b-balanced-1180mib.json
```

Current result: the reduced `1.073 GiB` Q4_0 subset dequantizes in `8.49 ms`
after `49.91 ms` H2D, then produces `302,336` synthetic-activation row scores
in `10.76 ms` after `48.77 ms` H2D with CPU score checks passing. This proves
the smaller 10 tok/s page plan still has working CUDA dequant/matvec mechanics;
it does not prove token quality yet.

Using the captured `ffn_norm-0` activation on the same reduced plan, the
width-matched subset is `0.715 GiB`; it produces `253,952` row scores in
`34.36 ms` after `33.13 ms` H2D, with `23` CPU score checks passing. This is
closer to a real sparse oracle, but still does not map those row scores to
candidate-token accept/reject decisions.

Reduced page signal-quality probe:

```powershell
python .\scripts\sage_reduced_page_quality_probe.py `
  --json-out .\benchmarks\sage-reduced-page-quality-gemma31b-1180mib-vs-full-ffn-norm0.json
```

Current result: compared with the fuller `1.482 GiB` real-activation matvec,
the reduced `0.715 GiB` width-matched plan is internally consistent on shared
scored tensors: `13/13` shared tensors have the same top-1 row and `100%`
top-k overlap. The problem is selection quality: the reduced plan retains only
`45%` of the fuller run's global top-20 row signals and `52%` of the top-50.
The next page selector must be signal-aware, not only byte-budget-aware.

Signal-aware page selector:

```powershell
python .\scripts\sage_signal_aware_pager.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --signal-json .\benchmarks\sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-ffn-norm0-full.json `
  --budget-gib 1.18 `
  --stage-buffer-gib 0.50 `
  --target-tps 10 `
  --max-active-percent-10tps 5.07 `
  --json-out .\benchmarks\sage-oracle-page-ledger-gemma31b-signal-aware-1180mib-10tps.json

python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-signal-aware-1180mib-10tps.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --score-top-k 5 `
  --max-score-tensors 32 `
  --cpu-check-rows 16 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-signal-aware-1180mib-ffn-norm0.json

python .\scripts\sage_reduced_page_quality_probe.py `
  --reduced-matvec-json .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-signal-aware-1180mib-ffn-norm0.json `
  --json-out .\benchmarks\sage-reduced-page-quality-gemma31b-signal-aware-1180mib-vs-full-ffn-norm0.json

python .\scripts\sage_oracle_cuda_page_cache_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-signal-aware-1180mib-10tps.json `
  --replays 3 `
  --json-out .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-signal-aware-1180mib-replay3.json

python .\scripts\sage_page_cache_budget.py `
  --page-cache-json .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-signal-aware-1180mib-replay3.json `
  --page-ledger-json .\benchmarks\sage-oracle-page-ledger-gemma31b-signal-aware-1180mib-10tps.json `
  --json-out .\benchmarks\sage-page-cache-budget-hard120-signal-aware-1180mib-10tps.json

python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --score-top-k 5 `
  --max-score-tensors 8 `
  --cpu-check-rows 4 `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-result-norm-full.json

python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-signal-aware-1180mib-10tps.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --score-top-k 5 `
  --max-score-tensors 32 `
  --cpu-check-rows 16 `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-signal-aware-1180mib-result-norm.json

python .\scripts\sage_reduced_page_quality_probe.py `
  --full-matvec-json .\benchmarks\sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-result-norm-full.json `
  --reduced-matvec-json .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-signal-aware-1180mib-result-norm.json `
  --json-out .\benchmarks\sage-reduced-page-quality-gemma31b-signal-aware-1180mib-vs-full-result-norm.json
```

Current result: the signal-aware ledger selects `70` pages and `1.173 GiB`
active bytes, `5.04%` of a 100B 2-bit reference, while staying inside the
10 tok/s active-byte window. Its real-activation matvec scores `25` Q4_0
tensors, `0.767 GiB`, with `25` CPU checks passing. The quality probe improves
global signal retention from `45%` to `100%` for top-20 rows and from `52%` to
`96%` for top-50 rows. A resident page-cache replay for this signal-aware plan
averages `56.57 ms` per fallback and projects to `10.07 tok/s`. This is the
strongest current evidence for SAGE's unique active-byte selector. A
cross-activation check keeps the same `ffn_norm-0`-selected page set and
evaluates it against the final `result_norm` vector: top-20 retention remains
`95%`, while top-50 retention is `82%`. That is better evidence than a
same-vector fit, but it is still not token-decision or transformer-integrated
proof.

Sparse oracle CUDA Q4_0 dequant smoke:

```powershell
python .\scripts\sage_oracle_cuda_dequant_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-dequant-gemma31b-full.json
```

This interprets the staged bytes as real GGUF `Q4_0` tensor blocks and runs a
CUDA dequantization/reduction kernel. Current result: `67` Q4_0 tensors,
`2.244 GiB` of quantized weights, and `4,282,908,672` dequantized values were
processed in `11.71 ms` of kernel time after `101.75 ms` of H2D transfer. This
proves CUDA can decode the staged quantized oracle pages, but it is still not
sparse matmul or candidate scoring.

Sparse oracle CUDA Q4_0 matvec smoke:

```powershell
python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-matvec-gemma31b-full.json
```

This runs real Q4_0 matrix-vector kernels over the selected GGUF tensors using a
deterministic synthetic activation vector. Current result: `67` Q4_0 matrices,
`2.244 GiB` of quantized weights, `4,282,908,672` weight values, and `628,480`
output scores were processed in `17.48 ms` of matvec kernel time after
`101.00 ms` of H2D transfer. This proves sparse page matvec mechanics, but not
live hidden-state scoring or oracle logits.

Sparse oracle CUDA Q4_0 real-activation matvec smoke:

```powershell
python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-ffn-norm0-full.json
```

This reuses the same staged Q4_0 page path, but the CUDA kernel reads a captured
Gemma `ffn_norm-0` hidden-state vector instead of the synthetic activation.
Current result: `48` width-matched Q4_0 matrices, `1.482 GiB` of quantized
weights, `2,829,582,336` weight values, and `526,336` output scores were
processed in `65.51 ms` of matvec kernel time after `86.12 ms` of H2D transfer.
This proves real activation values can drive the staged Q4_0 kernels, but it is
still not candidate ranking, full transformer composition, or oracle logit
comparison.

Sparse oracle CUDA Q4_0 ranked real-activation matvec smoke:

```powershell
python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --score-top-k 5 `
  --max-score-tensors 8 `
  --cpu-check-rows 4 `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-ranked-matvec-gemma31b-ffn-norm0-full.json
```

This keeps per-row CUDA scores instead of reducing them to totals, emits top-k
rows for selected tensors, and CPU-checks sampled top rows against the original
GGUF Q4_0 bytes. Current result: `28` tensors emitted `140` top-score rows, and
`16` CPU row checks passed with maximum absolute error `3.97e-07`. The ranked
run processed the same `48` width-matched Q4_0 matrices and `526,336` output
scores in `67.69 ms` of matvec kernel time after `80.58 ms` of H2D transfer.
This proves real projection-row ranking mechanics, but it is still not
candidate-token scoring or oracle logit comparison.

Sparse oracle CUDA Q6_K tied-vocab projection smoke:

```powershell
python .\scripts\sage_oracle_cuda_vocab_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --top-k 10 `
  --cpu-check-top-k 8 `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-vocab-projection-gemma31b-ffn-norm0-full.json
```

Gemma 31B ties the output projection to `token_embd.weight`, which is `Q6_K`
and larger than the current staging budget. This smoke pages that `1.077 GiB`
tensor in `2` chunks under a `0.750 GiB` live buffer, scores all `262,144`
token rows, and CPU-checks sampled top tokens against raw GGUF bytes. Current
result: H2D transfer `49.10 ms`, CUDA Q6_K vocab kernel `29.83 ms`, `10` top
token ids emitted, and `8/8` CPU checks passed. This is the first token-id
scoring artifact, but it is still not true oracle logits because the activation
is `ffn_norm-0`, not the final post-norm hidden state.

Sparse oracle CUDA Q6_K final-logit comparison:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-debug.exe `
  -m .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  -p "The capital of France is" `
  -ngl 38 -c 256 -b 128 -ub 8 --no-warmup `
  --save-logits `
  --logits-output-dir .\benchmarks\sage-gemma31b-logits-debug `
  --tensor-filter "result_norm$" `
  --tensor-values-output .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --tensor-values-limit 5376 `
  --tensor-values-i1 -1

python .\scripts\sage_oracle_cuda_vocab_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --llamacpp-logits-bin .\benchmarks\sage-gemma31b-logits-debug\llamacpp-gemma-4-31B_q4_0-it.bin `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-vocab-logit-compare-gemma31b-result-norm-full.json
```

This captures Gemma's final post-output-norm hidden vector and compares the
paged CUDA Q6_K vocab projection against llama.cpp's saved logits for the same
prompt. Current result: all `262,144` token rows scored in `2` chunks, max live
GPU staging buffer `0.750 GiB`, H2D transfer `56.35 ms`, CUDA kernel `29.87 ms`,
top-1 match `true`, overlap@10 `10/10`, max top-logit absolute error `0.0367`,
and `8/8` sampled raw-row CPU checks passed. This proves the paged vocab
projection can reproduce llama.cpp's final token ranking for this forward pass.

Sparse oracle CUDA Q6_K candidate verifier:

```powershell
python .\scripts\sage_oracle_cuda_candidate_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --llamacpp-logits-bin .\benchmarks\sage-gemma31b-logits-debug\llamacpp-gemma-4-31B_q4_0-it.bin `
  --top-k-from-logits 64 `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-gemma31b-result-norm-top64.json
```

This is the sparse shortlist version of the final-logit proof. Instead of
projecting all `262,144` vocab rows, it reads only selected candidate token rows
from the Q6_K matrix, packs them into a tiny staging buffer, and compares those
candidate logits to llama.cpp. Current result: `64` candidate rows, `282,240`
active bytes, only `0.0244%` of the vocab tensor, H2D transfer `0.0493 ms`,
CUDA kernel `0.1034 ms`, candidate top-1 match `true`, `64/64` llama.cpp logit
checks passed, and max absolute logit error `0.0482`. This is not yet a live
scheduler, but it is the first measured candidate-token verifier shape.

Live proxy shortlist to CUDA verifier bridge:

```powershell
python .\scripts\sage_oracle_cuda_candidate_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --llamacpp-logits-bin .\benchmarks\sage-gemma31b-logits-debug\llamacpp-gemma-4-31B_q4_0-it.bin `
  --candidate-live-trace-json .\benchmarks\sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json `
  --candidate-live-step-index 0 `
  --candidate-live-source proxy `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-live-topk-qwen-trace-gemma31b-result-norm.json
```

This consumes the `sage-live-proxy-shortlist-v0` IDs emitted by
`llama-sage-dual-live` and uses them as sparse Q6_K row IDs in the CUDA
candidate verifier. Current result: `10` live proxy candidate rows, `44,100`
active bytes, `0.003815%` of the vocab tensor, H2D transfer `0.0249 ms`, CUDA
kernel `0.1270 ms`, candidate top-1 match `true`, and `10/10` selected
llama.cpp logit checks passed with max absolute error `0.0257`. The current
artifact bridges a Qwen live trace into Gemma row IDs, so it proves runtime
plumbing and sparse scoring mechanics; tokenizer-equivalent semantic agreement
requires a live trace, activation, logits, and GGUF from the same model family.

Same-prompt proxy shortlist fallback smoke:

```powershell
python .\scripts\sage_oracle_cuda_candidate_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --llamacpp-logits-bin .\benchmarks\sage-gemma31b-logits-debug\llamacpp-gemma-4-31B_q4_0-it.bin `
  --candidate-logprob-json .\benchmarks\20260626-210118-sage-logprob-gemma12b-to-31b.json `
  --candidate-logprob-row-index 0 `
  --candidate-logprob-step-index 0 `
  --candidate-logprob-side proxy `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-gemma12proxy-france-top10-fallback-result-norm.json
```

This uses the matching Gemma 12B proxy capture for the same prompt as the saved
Gemma 31B logits: `The capital of France is`. The proxy top-k contains digits,
while the Gemma 31B global top-1 is token id `9079` (` Paris`). Current result:
`10` proxy candidate rows, `44,100` active bytes, H2D transfer `0.0293 ms`,
CUDA kernel `0.0911 ms`, `10/10` selected llama.cpp logit checks passed, but
`candidate_contains_llamacpp_global_top1` is `false`. This proves the reject
side of the runtime contract: sparse candidate scoring can be correct and still
must request exact fallback when the proxy shortlist misses the oracle winner.

Sparse oracle runtime-step replay:

```powershell
python .\scripts\sage_oracle_runtime_step.py `
  --json-out .\benchmarks\sage-sparse-oracle-runtime-step-gemma31b-page-q6k-fallback-replay.json
```

This joins measured component artifacts into one per-token oracle ledger:
Gemma 31B sparse page plan, CUDA page staging/kernel evidence, Q6_K candidate
verification, and fallback decision. Current result: `2.246490 GiB` sparse page
bytes plus `44,100` candidate bytes, `9.6488%` of a 100B 2-bit reference,
`0.711755 GiB` max measured device stage, `107.0062 ms` H2D, `18.4317 ms`
CUDA kernel time, and `exact_fallback_required: true`. This is still
`measured_component_replay_not_transformer_integrated`; it is the runtime ledger
shape that the live llama.cpp scheduler must replace with real sparse
transformer math.

Proxy format-scaffold shortlist:

```powershell
python .\scripts\sage_proxy_shortlist_eval.py `
  --logprob-json .\benchmarks\20260627-150834-sage-logprob-gemma12b-to-31b-validation80e-chat8-filtered-o000-l040.json `
                 .\benchmarks\20260627-151104-sage-logprob-gemma12b-to-31b-validation80e-chat8-filtered-o040-l040.json `
  --ignore-prefix-steps 3 `
  --k-values 1,2,4,8,10 `
  --train-mod 2 `
  --train-remainders 0 `
  --static-rescue-count 0 `
  --position-rescue-count 8 `
  --prompt-piece-count 16 `
  --prompt-piece-ids-per-piece 2 `
  --json-out .\benchmarks\sage-proxy-shortlist-format-scaffold-validation80e-k10-pos8-prompt16.json
```

This measures whether the CUDA candidate verifier can receive a realistic
shortlist without using oracle top-k as input. The current policy combines proxy
top-k, a learned per-position format scaffold, and up to `16` prompt-derived
vocab pieces. On validation80e, the best eval policy covers `97.50%` of oracle
top-1 tokens with `27.0` candidate rows per step. On hard120, it covers
`95.67%` with `26.9` rows per step. Both fit inside the already-measured
`64`-row Q6_K verifier smoke. This is still a shortlist benchmark, not a
complete runtime.

KV ledger plan:

```powershell
python .\scripts\sage_kv_ledger.py `
  --oracle-model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context-tokens 4096 `
  --hot-recent-tokens 512 `
  --sink-tokens 16 `
  --warm-max-tokens 3584 `
  --warm-bits 2 `
  --json-out .\benchmarks\sage-kv-ledger-gemma31b-ctx4096-hot528-warm2bit.json
```

This is a planning artifact, not runtime-integrated KV compression. Current
result: full precision 4K oracle KV is estimated at `3.438 GiB`; hot/warm/cold
tiers reduce that to `0.817 GiB`, with `0.443 GiB` hot VRAM KV and `76.2%`
saved versus full precision.

KV tier pack smoke:

```powershell
python .\scripts\sage_kv_tier_smoke.py `
  --oracle-model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --context-tokens 4096 `
  --hot-recent-tokens 512 `
  --sink-tokens 16 `
  --warm-max-tokens 3584 `
  --warm-bits 2 `
  --sample-tokens 8 `
  --cuda-pack `
  --json-out .\benchmarks\sage-kv-tier-pack-smoke-gemma31b-ctx4096-hot528-warm2bit-sample8.json
```

This is still not runtime-integrated attention, but it measures the warm KV
packing mechanics against Gemma's inferred KV dimensions. Current result:
`8.00x` compression versus FP16, planned warm tier `0.374 GiB`, CUDA pack
`0.0926 ms`, CUDA unpack `0.0522 ms`, and CUDA packed bytes exactly match the
CPU reference on the bounded sample.

The live `llama-sage-dual-live` smoke now emits tiered KV accounting fields from
runtime token counts. On the local Qwen 0.5B arithmetic smoke, both proxy and
oracle estimate `12288` full-precision KV bytes/token; the oracle fallback step
reports `73728` oracle full-precision KV bytes and `73728` tiered KV bytes
because the short context is still entirely hot. The trace status is
`tiered_runtime_accounting_not_attention_integrated`, so this is runtime byte
accounting, not packed KV attention execution.

KV runtime ledger accounting:

```powershell
python .\scripts\sage_kv_runtime_ledger.py `
  --live-trace-json .\benchmarks\sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json `
  --kv-ledger-json .\benchmarks\sage-kv-ledger-gemma31b-ctx4096-hot528-warm2bit.json `
  --kv-tier-smoke-json .\benchmarks\sage-kv-tier-pack-smoke-gemma31b-ctx4096-hot528-warm2bit-sample8.json `
  --include-context-limit `
  --json-out .\benchmarks\sage-kv-runtime-ledger-qwen05b-arithmetic-gemma31b-plan.json
```

This annotates live trace token counts with the Gemma 31B hot/warm/cold KV byte
policy and attaches the measured CUDA pack evidence. Current result: `2`
runtime steps annotated, `1` oracle fallback step, context sweep exercises warm
KV, max full-precision 4K oracle KV is `3.438 GiB`, tiered KV is `0.817 GiB`,
and the CUDA pack sample still matches CPU. This is byte accounting for the
runtime ledger, not compressed attention execution.

Proxy/oracle agreement smoke tests:

```powershell
python .\scripts\sage_agreement.py --mode raw --tokens 1 --limit 5
python .\scripts\sage_agreement.py --mode chat --tokens 8 --limit 5 --tag gemma12b-to-31b-chat
```

`raw` mode uses `llama-completion.exe` for non-interactive greedy continuation. `chat` mode uses `llama-cli.exe --single-turn` for instruction-style output. Child process logs are written under `benchmarks/sage-agreement-logs/`.

Top-k logprob agreement probe:

```powershell
python .\scripts\sage_logprob_probe.py --mode chat --tokens 8 --top-k 10 --limit 5 --ignore-prefix-steps 3 --tag gemma12b-to-31b-chat8-filtered
python .\scripts\sage_logprob_probe.py --prompts .\prompts\sage-calibration.txt --mode chat --tokens 8 --top-k 10 --limit 30 --ignore-prefix-steps 3 --tag gemma12b-to-31b-calib30-chat8-filtered
```

This uses `llama-server` and compares top-1 token agreement, top-k overlap, confidence margin, and approximate entropy. The `--ignore-prefix-steps 3` option ignores Gemma chat control tokens like `<|channel>`, `thought`, and the following newline.

Proxy-only router fit:

```powershell
python .\scripts\sage_router_fit.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --ignore-prefix-steps 3 --max-false-accept-rate 0.10
```

This sweeps transparent margin, entropy, and token-class rules. Current local evidence says simple proxy-only rules do not safely hit target speed at low error; SAGE needs sparse oracle verification before skipping the giant model.

Two-stage sparse verifier target:

```powershell
python .\scripts\sage_policy_target.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --ignore-prefix-steps 3 --candidate-class punct --margin-threshold 0.644 --verifier-active-percent 1 --oracle-active-percent 10
```

This estimates the catch-rate and byte-budget target for the next sparse verifier. Current local evidence says a punctuation/margin router can work only if a sparse verifier catches roughly `80%+` of bad candidates while touching about `1%` of a 100B 2-bit model, assuming the larger oracle path touches `10%`.

Sparse verifier block budget:

```powershell
python .\scripts\sage_verifier_plan.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --active-percents 0.5,1,2,4
```

This maps the 100B-equivalent verifier byte budget onto real Gemma 31B GGUF blocks. At `1%` active 100B 2-bit budget, only one FFN group or about three attention groups fit, so the verifier must be a tiny micro-verifier rather than a normal partial forward pass.

Concrete sparse verifier manifest:

```powershell
python .\scripts\sage_verifier_manifest.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy ffn-sentinel --active-percent 1
python .\scripts\sage_verifier_manifest.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy attention-sentinel --layer-order early --active-percent 1
python .\scripts\sage_verifier_manifest.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy hybrid --active-percent 2
```

This emits the exact GGUF tensor groups and llama.cpp graph probe names for the next sparse verifier prototype. On the local Gemma 31B Q4 file, `1%` fits one boundary FFN sentinel or three early attention sentinels. A `2%` boundary hybrid fits one FFN sentinel plus three attention sentinels.

Filtered llama.cpp tensor probe patch:

```text
patches/llama.cpp/0002-filter-debug-eval-callback.patch
patches/llama.cpp/0004-debug-tensor-stats-hook.patch
patches/llama.cpp/0005-debug-tensor-stats-jsonl.patch
patches/llama.cpp/0006-sage-runtime-tensor-stats-hook.patch
patches/llama.cpp/0007-sage-runtime-decision-output.patch
patches/llama.cpp/0008-sage-inprocess-policy-function.patch
patches/llama.cpp/0009-sage-policy-check-example.patch
patches/llama.cpp/0010-sage-policy-check-jsonl-batch.patch
patches/llama.cpp/0011-sage-policy-check-scheduler-fields.patch
patches/llama.cpp/0012-sage-scheduler-replay-example.patch
patches/llama.cpp/0013-sage-live-proxy-example.patch
patches/llama.cpp/0014-sage-dual-live-example.patch
patches/llama.cpp/0015-debug-tensor-values-jsonl.patch
```

This small patch makes `llama-debug --tensor-filter` reject non-matching graph nodes during the scheduler callback `ask` phase, so only selected verifier nodes are copied back. Without it, the stock debug callback can still request every tensor.

The stats hook adds `llama-debug --tensor-stats`, which prints one compact line per matched tensor:

```text
stats: count = ..., sum = ..., mean = ..., min = ..., max = ..., nan_count = ...
```

`0005-debug-tensor-stats-jsonl.patch` adds `llama-debug --tensor-stats-output PATH`, which writes the same compact statistics as JSON Lines:

```json
{"sequence":0,"name":"ffn_norm-0","dtype":"f32","op":"MUL","shape":[896,1,1,1],"count":896,"sum":-4.13,"mean":-0.0046,"min":-3.69,"max":4.07,"nan_count":0}
```

This is the first scheduler-readable SAGE FFN-sentinel signal. It still uses the debug eval callback, but it avoids full tensor-value dumps and lets `sage_probe_capture.py --tensor-stats-jsonl` consume structured records instead of scraping stdout.

`0006-sage-runtime-tensor-stats-hook.patch` moves the same JSONL signal into normal generation binaries through `common_init_from_params()`. With that patch, `llama-completion` accepts:

```powershell
--sage-tensor-filter "ffn_norm-0" --sage-tensor-stats-output .\benchmarks\sage-runtime.jsonl
```

This is the first non-debug SAGE runtime hook. It still records a sidecar for measurement, but it runs through the normal generation path and can be replaced by an in-process policy callback later.

`0007-sage-runtime-decision-output.patch` adds the first in-process decision event. With that patch, `llama-completion` also accepts:

```powershell
--sage-decision-output .\benchmarks\sage-decision.jsonl --sage-candidate-id cand-0074 --sage-token-class capitalized --sage-proxy-entropy 1.100873414272788 --sage-proxy-margin 1.616725891828537
```

The generated JSONL event applies the frozen proxy gate plus FFN-sentinel verifier rule inside the normal generation process and emits `accept_proxy` or `oracle_fallback`. This is still a one-shot verifier event, not yet the full proxy/oracle scheduler loop.

`0008-sage-inprocess-policy-function.patch` refactors the same frozen rule into `common_sage_decide()`. That keeps the existing JSONL output unchanged, but makes the policy callable by a future persistent `llama-sage` loop without waiting for callback destruction or parsing a sidecar.

`0009-sage-policy-check-example.patch` adds a tiny model-free executable for that policy:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-sage-policy-check.exe --self-test

.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-sage-policy-check.exe `
  --candidate-id manual-reject `
  --token-class capitalized `
  --proxy-entropy 0.5 `
  --proxy-margin 0.1 `
  --ffn-norm-0-sum -100 `
  --ffn-norm-0-sequence 7 `
  --expect-action oracle_fallback `
  --expect-reason runtime_verifier_reject
```

This gives the persistent-runtime work a fast C++ gate: policy changes can be tested without loading a model, starting `llama-server`, or running the Python live loop.

`0010-sage-policy-check-jsonl-batch.patch` adds persistent JSONL batch mode to that gate. The parity runner now sends all validation rows through one C++ process instead of spawning one process per row.

`0011-sage-policy-check-scheduler-fields.patch` extends JSONL mode with scheduler fields: `proxy_token`, `oracle_token`, `selected_token`, `top1_match`, and `false_accept`. That makes the C++ gate prove not only the accept/fallback action, but also the token the scheduler would emit.

`0012-sage-scheduler-replay-example.patch` adds `llama-sage-scheduler-replay`, a persistent model-free C++ scheduler skeleton. It consumes candidate rows in one process, calls `common_sage_decide()`, emits the selected token, tracks false accepts, and accumulates generated text per prompt stream.

`0013-sage-live-proxy-example.patch` adds `llama-sage-proxy-live`, the first model-backed resident proxy path. It loads one GGUF once, generates tokens in-process, computes logits-derived entropy and margin, classifies the token, and emits SAGE proxy-gate telemetry as JSON.

`0014-sage-dual-live-example.patch` adds `llama-sage-dual-live`, a two-context live scheduler skeleton. It keeps a resident proxy context and a resident oracle context, applies `common_sage_decide()` to the proxy token, selects proxy or oracle text, then feeds the selected text back into both contexts so future steps stay on the selected prefix. It now emits the `sage-active-byte-ledger-v0` trace so each step reports oracle mode, active bytes, model bytes, token positions, latency, tiered KV accounting fields, and live proxy top-k candidate rows. The KV fields are accounting-only; packed KV tensors are not used by attention yet.

`0015-debug-tensor-values-jsonl.patch` adds bounded tensor-value capture for
`llama-debug`. It enables commands such as:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-debug.exe `
  -m .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  -p "The capital of France is" `
  -ngl 38 -c 256 -b 128 -ub 8 `
  --tensor-filter "ffn_norm-0" `
  --tensor-values-output .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --tensor-values-limit 5376 `
  --tensor-values-i1 -1 `
  --no-warmup
```

That sidecar is the source for the real-activation CUDA matvec smoke. It is
bounded to one selected tensor slice and is not a general full-tensor dump.

C++ policy parity against validation80e:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
$probe = @(Get-ChildItem .\benchmarks -File -Filter '*sage-probe-validation80e-accepted-punct-cap-ffn1-stats-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_cpp_policy_parity.py `
  --replay-json @replay `
  --probe-json @probe `
  --require-pass `
  --json-out .\benchmarks\sage-cpp-policy-parity-validation80e.json `
  --tag validation80e
```

```text
checked replay rows:                 177/177
cpp mode:                            jsonl_batch
cpp invocations:                     1
probe records:                       126
verifier covered:                    126/126
final false accepts:                 1
action matches:                      177/177
reason matches:                      177/177
selected token matches:              177/177
false accept matches:                177/177
result:                              pass
```

Resident proxy smoke check on the local RTX 3060:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-sage-proxy-live.exe `
  -m .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ngl 99 `
  -n 4 `
  --json-out .\benchmarks\sage-proxy-live-qwen05b-smoke.json `
  "The capital of France is"
```

```text
generated tokens:                    4
final text:                          " Paris. It is"
short-run proxy speed:               92.23 tok/s
telemetry fields:                    token_class, proxy_entropy, proxy_margin, proxy_gate_accept
sampled-is-logit-top1:               4/4
```

Dual-context scheduler smoke checks:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-sage-dual-live.exe `
  --proxy-model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --oracle-model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ngl 99 `
  -n 2 `
  --logit-top-k 10 `
  --json-out .\benchmarks\sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json `
  "2 + 2 ="
```

```text
generated tokens:                    2
final text:                          " 4"
proxy accepts:                       1
oracle fallbacks:                    1
proxy decode calls:                  2
oracle decode calls:                 1
state sync:                          selected text fed back to both contexts
ledger schema:                       sage-active-byte-ledger-v0
accept ledger:                       oracle_mode=none, oracle_active_bytes=0
fallback ledger:                     oracle_mode=exact_resident_context, oracle_active_bytes=485452288
KV status:                           tiered_runtime_accounting_not_attention_integrated
fallback oracle full KV bytes:       73728
fallback oracle tiered KV bytes:     73728
live proxy top-k rows:               10 per generated step
live proxy shortlist schema:         sage-live-proxy-shortlist-v0
packed KV attention:                 not_implemented
```

C++ scheduler replay against validation80e:

```powershell
python .\scripts\sage_cpp_scheduler_replay.py `
  --replay-json @replay `
  --probe-json @probe `
  --require-pass `
  --json-out .\benchmarks\sage-cpp-scheduler-replay-validation80e.json `
  --tag validation80e
```

```text
checked replay rows:                 177/177
cpp invocations:                     1
prompt streams:                      40
verifier covered:                    126/126
final false accepts:                 1
action matches:                      177/177
reason matches:                      177/177
selected token matches:              177/177
false accept matches:                177/177
prefix matches:                      177/177
generated text matches:              177/177
result:                              pass
```

Small-model smoke check:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-debug.exe `
  -m .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -p "hello" `
  -ngl 0 `
  -c 32 `
  -b 32 `
  -ub 8 `
  --no-warmup `
  --tensor-filter "ffn_norm-0" `
  --tensor-stats `
  --tensor-stats-output .\benchmarks\qwen-ffn-stats.jsonl `
  --verbose
```

Sparse verifier signal capture:

```powershell
python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy hybrid --active-percent 2 --ngl 38 --batch-size 128 --ubatch-size 8 --prompts .\prompts\sage-calibration.txt --limit 5 --tensor-stats-jsonl --tag gemma31b-hybrid2
```

This runs the patched `llama-debug`, uses the manifest-generated tensor filter, and writes JSON tensor summaries under `benchmarks/`. It is the first measurement hook for training or testing whether selected FFN/attention sentinel signals can reject bad proxy tokens before falling back to the larger oracle path.

With `0004-debug-tensor-stats-hook.patch` applied and rebuilt, add `--tensor-stats` to capture compact summaries instead of full tensor dumps. With `0005-debug-tensor-stats-jsonl.patch`, prefer `--tensor-stats-jsonl`; the script creates one sidecar file per prompt and parses it directly.

Non-debug runtime signal capture:

```powershell
python .\scripts\sage_runtime_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tensor-filter "ffn_norm-0" --tokens 1 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --prompt "The capital of France is" --tag gemma31b-runtime-smoke
```

Debug-vs-runtime signal gate:

```powershell
python .\scripts\sage_compare_runtime_debug.py --debug-json .\benchmarks\20260627-160546-sage-probe-gemma31b-debug-runtime-compare-hello.json --runtime-json .\benchmarks\20260627-160604-sage-runtime-gemma31b-debug-runtime-compare-hello.json --require-match
```

Validation-shaped runtime gate:

```powershell
python .\scripts\sage_runtime_gate.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260627-151635-sage-failure-tasks-validation80e-predeclared-punct-cap-accepted.json --mode gemma4-chat --gemma4-thought-prefix --tensor-filter "ffn_norm-0" --tokens 1 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --offset 3 --limit 10 --timeout 1800 --tag validation80e-o003-l010
```

Current smoke result: Qwen 0.5B and Gemma 31B each passed a one-prompt `ffn_norm-0` comparison with `0` mismatches between debug JSONL and non-debug runtime JSONL. Gemma 31B validation-shaped subsets also passed: the first 3-task slice matched `11/11` tensor records, and the next 10-task slice matched `49/49` tensor records with `0` missing, extra, or mismatched records.

Runtime scheduler decision prototype:

```powershell
python .\scripts\sage_runtime_scheduler.py --runtime-json .\benchmarks\20260627-162541-sage-runtime-gate-validation80e-falseaccepts-l007-runtime.json --require-full-verifier-coverage --json-out .\benchmarks\20260627-sage-runtime-scheduler-validation80e-falseaccepts-l007.json
```

On the `7` known validation80e false accepts, the runtime scheduler covered `7/7` verifier calls, rejected `6/7` bad proxy candidates to oracle fallback, and left `1/7` false accept. `0007` then moves the same frozen decision into a one-shot in-process event emitted by normal `llama-completion`.

In-process decision smoke:

```powershell
python .\scripts\sage_runtime_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260627-162534-sage-failure-tasks-validation80e-predeclared-punct-cap-falseaccepts.json --mode gemma4-chat --gemma4-thought-prefix --tensor-filter "ffn_norm-0" --tokens 1 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --limit 1 --timeout 1800 --capture-decisions --tag gemma31b-decision-falseaccept-smoke
```

In-process decision parity gate:

```powershell
python .\scripts\sage_runtime_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260627-162534-sage-failure-tasks-validation80e-predeclared-punct-cap-falseaccepts.json --mode gemma4-chat --gemma4-thought-prefix --tensor-filter "ffn_norm-0" --tokens 1 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --limit 7 --timeout 1800 --capture-decisions --tag gemma31b-decision-falseaccepts-l007 --json-out .\benchmarks\20260627-sage-runtime-gemma31b-decision-falseaccepts-l007.json
python .\scripts\sage_compare_runtime_decisions.py --runtime-json .\benchmarks\20260627-sage-runtime-gemma31b-decision-falseaccepts-l007.json --json-out .\benchmarks\20260627-compare-runtime-decisions-gemma31b-falseaccepts-l007.json --require-match
```

Result: the in-process C++ events matched the Python scheduler on `7/7` false-accept tasks with `0` mismatches. The emitted actions were `6` oracle fallbacks and `1` accepted proxy token, matching the offline scheduler.

Multi-token scheduler replay:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
$probe = @(Get-ChildItem .\benchmarks -File -Filter '*sage-probe-validation80e-accepted-punct-cap-ffn1-stats-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_multitoken_replay.py --replay-json @replay --runtime-json @probe --decision-source python --stats-total 400 --require-full-verifier-coverage --require-pass --json-out .\benchmarks\20260627-sage-multitoken-replay-validation80e-full-ffn1-stats.json
$cpp = @(Get-ChildItem .\benchmarks -File -Filter '20260627-sage-runtime-gemma31b-decision-validation80e-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_compare_runtime_decisions.py --runtime-json @cpp --json-out .\benchmarks\20260627-compare-runtime-decisions-gemma31b-validation80e-full-cpp.json --require-match
python .\scripts\sage_multitoken_replay.py --replay-json @replay --runtime-json @cpp --decision-source cpp --stats-total 400 --require-full-verifier-coverage --require-pass --json-out .\benchmarks\20260627-sage-multitoken-replay-validation80e-full-cpp-decisions-promptkey.json
```

This emits a prompt/step scheduler trace instead of only aggregate accounting. On validation80e it now passes using full in-process C++ decision coverage: `126/126` C++ decisions matched the Python scheduler with `0` mismatches; the multi-token trace used `126` C++ decisions, `0` Python decisions, `101/400` final proxy accepts, `1` false accept, `0.25%` total-token error, and `7.29 tok/s` modeled throughput. The current replay script sorts/counts by prompt text because `prompt_index` is local to logprob chunks; the corrected trace reports `80` prompt texts rather than collapsing chunk-local prompt indexes to `40`.

Live multi-token loop prototype:

```powershell
python .\scripts\sage_live_loop.py --self-test

python .\scripts\sage_live_loop.py `
  --proxy-model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --oracle-model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --verifier-model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --proxy-ngl 0 `
  --oracle-ngl 0 `
  --verifier-ngl 0 `
  --prompt "The capital of France is" `
  --tokens 1 `
  --top-k 5 `
  --ctx-size 64 `
  --verifier-ctx-size 64 `
  --threads 4 `
  --verifier-threads 4 `
  --tag qwen-live-smoke-verifier-policy
```

`scripts/sage_live_loop.py` is the first live proxy/verifier/oracle control loop over newly generated tokens. In managed mode it runs a proxy `llama-server`, pauses the proxy before verifier subprocess calls to stay memory-safe on one-GPU machines, invokes patched `llama-completion` directly for `--sage-decision-output`, and writes one JSON decision trace under `benchmarks/`. The live loop now applies the validated candidate-token policy by default: only `punct,whitespace,number,capitalized` tokens are eligible for proxy acceptance; other token classes fall back to oracle. The Qwen smoke produced one verified live proxy accept with `verifier_covered: true`; this is a control-flow smoke, not a speed claim for Gemma 31B or 100B+.

Live-vs-replay comparison gate:

```powershell
python .\scripts\sage_live_gate.py --prompt-index 1 --max-live-tokens 1 --print-only
python .\scripts\sage_live_gate.py --prompt-index 1 --max-live-tokens 1
```

The gate runner validates model/replay paths, launches the seeded live loop, then launches the comparator against the validation80e replay trace. Use `--print-only` before a long run to inspect the exact commands. The equivalent lower-level commands are:

```powershell
python .\scripts\sage_live_replay_compare.py --self-test

$seedReplay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)

python .\scripts\sage_live_loop.py `
  --seed-replay-json @seedReplay `
  --seed-prompt-index 1 `
  --tokens-from-replay `
  --json-out .\benchmarks\gemma-live-validation80e-p001.json `
  --tag gemma-live-validation80e-p001

python .\scripts\sage_live_replay_compare.py `
  --live-json .\benchmarks\gemma-live-validation80e-p001.json `
  --replay-json .\benchmarks\20260627-sage-multitoken-replay-validation80e-full-cpp-decisions-promptkey.json `
  --prompt-index 1 `
  --require-pass
```

This is the next Gemma-scale gate. `--seed-replay-json` copies the prompt and Gemma chat settings from the replay artifact, `--tokens-from-replay` sets the live token count to the number of replay rows for that prompt, and the comparator checks live actions, selected tokens, and verifier coverage against the replay trace for the same prompt region. A Qwen seeded smoke intentionally failed this comparator at token `0`, proving the gate catches drift instead of silently passing unrelated live behavior.

Measured Gemma live-gate results:

```text
validation80e prompt 1:
  seeded rows:                         1
  skipped prompt-index collisions:      1
  live/replay selection:       live_seed_task_ids
  compared steps:                       1
  action/selected/verifier matches:   1/1
  result:                            pass

validation80e prompt 35, first 2 rows:
  compared steps:                       2
  verifier calls:                       1
  oracle fallbacks:                     1
  proxy accepts:                        1
  action/selected/verifier matches:   2/2
  result:                            pass
```

Live timing report:

```powershell
python .\scripts\sage_live_report.py `
  --live-json .\benchmarks\gemma-live-validation80e-p001.json .\benchmarks\gemma-live-validation80e-p035.json `
  --json-out .\benchmarks\gemma-live-report-validation80e-p001-p035.json `
  --tag validation80e-p001-p035
```

```text
combined live steps:                    3
live/replay compare result:          pass
observed live speed:                 0.059 tok/s
proxy request time:                  0.290 s
verifier request time:              12.079 s
oracle request time:                 3.537 s
orchestration/reload overhead:      35.122 s
overhead share:                       68.8%
```

Interpretation: the Python live loop is now useful as a correctness gate, not as the final speed path. The small Gemma runs match replay decisions, but most wall time is outside model request compute. The production target is therefore a persistent scheduler runtime: either move the proxy/verifier/oracle decision loop into llama.cpp, or keep long-lived proxy/verifier/oracle workers with no per-token server pause, subprocess launch, or model reload.

Persistent-runtime projection:

```powershell
python .\scripts\sage_runtime_projection.py `
  --policy-json .\benchmarks\20260627-154000-sage-policy-report-validation80e-frozen-ffn1-stats.json `
  --live-report-json .\benchmarks\gemma-live-report-validation80e-p001-p035.json `
  --json-out .\benchmarks\sage-runtime-projection-validation80e-live.json
```

```text
live Python loop:                    0.059 tok/s
same request timings without overhead: 0.189 tok/s
persistent active-byte policy model: 7.295 tok/s
active-byte model with measured proxy: 5.163 tok/s
```

This projection sharpens the next target. If the proxy path remains near the measured `96.6 ms/token`, SAGE would need an oracle path below about `3.5%` active 100B-equivalent weights just to reach `7 tok/s`, and `10 tok/s` is already impossible. If the proxy reaches `25 tok/s`, the current policy can hit `7 tok/s` with about `10%` active oracle, while `10 tok/s` requires about `5%` active oracle. See `docs/sage-persistent-runtime.md` for the implementation RFC.

Format-scaffold runtime projection:

```powershell
python .\scripts\sage_runtime_projection.py `
  --policy-json .\benchmarks\20260627-154000-sage-policy-report-validation80e-frozen-ffn1-stats.json `
  --live-report-json .\benchmarks\gemma-live-report-validation80e-p001-p035.json `
  --candidate-verifier-json .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-gemma31b-result-norm-top64.json `
  --format-shortlist-json .\benchmarks\sage-proxy-shortlist-format-scaffold-validation80e-k10-pos8-prompt16.json .\benchmarks\sage-proxy-shortlist-format-scaffold-hard120-k10-pos8-prompt16.json `
  --include-candidate-host-read `
  --exact-fallback-ms 270 `
  --json-out .\benchmarks\sage-runtime-projection-format-scaffold-localfallback270.json
```

This newer projection uses the measured Q6_K candidate verifier on every token
and exact fallback only for shortlist misses. With the measured live proxy time
and a conservative local `270 ms` fallback, hard120 projects to `9.21 tok/s`.
For the same hard120 fallback rate, `7 tok/s` can tolerate fallback up to about
`1062 ms/token`; `10 tok/s` would require fallback below about `73 ms/token`.

Measured sparse-fallback projection:

```powershell
python .\scripts\sage_runtime_projection.py `
  --policy-json .\benchmarks\20260627-154000-sage-policy-report-validation80e-frozen-ffn1-stats.json `
  --live-report-json .\benchmarks\gemma-live-report-validation80e-p001-p035.json `
  --candidate-verifier-json .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-gemma31b-result-norm-top64.json `
  --sparse-runtime-step-json .\benchmarks\sage-sparse-oracle-runtime-step-gemma31b-page-q6k-fallback-replay.json `
  --format-shortlist-json .\benchmarks\sage-proxy-shortlist-format-scaffold-validation80e-k10-pos8-prompt16.json .\benchmarks\sage-proxy-shortlist-format-scaffold-hard120-k10-pos8-prompt16.json `
  --include-candidate-host-read `
  --exact-fallback-ms 270 `
  --json-out .\benchmarks\sage-runtime-projection-format-scaffold-measured-sparse-fallback.json
```

Using the measured sparse-oracle runtime-step replay as the fallback cost,
hard120 with measured proxy speed projects to `8.07 tok/s`. This is the
conservative path because it includes host reads in the `626.5 ms` fallback
cost. The H2D+kernel-only view is `9.78 tok/s`, but the contract gates the full
component replay. Neither clears `10 tok/s`, so the next production target is
live sparse transformer math plus faster proxy scheduling, not more projection.

Overlap/prefetch budget target:

```powershell
python .\scripts\sage_overlap_budget.py `
  --json-out .\benchmarks\sage-overlap-budget-hard120-measured-sparse-fallback-10tps.json
```

For hard120 with measured proxy speed, the full measured sparse fallback path is
`8.066 tok/s`; hiding host reads raises the projection to `9.778 tok/s`, still
short of `10 tok/s`. The measured 10 tok/s target is now concrete: hide another
`52.3 ms` of GPU fallback work per fallback, reduce proxy latency by about
`2.27 ms/token`, or reduce the hard120 fallback rate by about `1.81` percentage
points. This is the next implementation target for async page prefetch,
CUDA-stream overlap, and candidate-source improvement.

The first host-prefetch pipeline smoke now proves the mechanism but not the
final speed: it saved `117.27 ms` on the full page plan, while the measured
pipeline still took `1116.22 ms` because raw GGUF host reads dominate. That
pushes the next runtime work toward a resident pinned page cache, smaller active
sets, and sparse dequant/matmul overlap.

The resident page-cache replay smoke then removes repeated host reads from the
fallback loop: after a `1464.97 ms` one-time cache build, the same page plan
replayed from pinned cache at `139.15 ms` average GPU time across three passes.
The follow-up budget artifact projects this at `9.72 tok/s`; `10 tok/s` now
requires about `1.180 GiB` active pages, `47.5%` less than the current plan, or
a fallback-rate drop to about `2.28%`.

Compact-stats validation result: rerunning the frozen `1%` FFN-sentinel verifier on the same validation80c and validation80d task sets with `sage_probe_capture.py --tensor-stats` reproduced the original full-dump policy results exactly:

```text
validation80c stats path:
  verifier coverage:                    125/125
  final false accepts:                     4
  accepted-token error:                  3.85%
  total-token error:                     1.00%
  modeled throughput:                    7.35 tok/s
  all gates:                             pass

validation80d stats path:
  verifier coverage:                    139/139
  final false accepts:                     2
  accepted-token error:                  1.79%
  total-token error:                     0.50%
  modeled throughput:                    7.43 tok/s
  all gates:                             pass

validation80e fresh stats path:
  verifier coverage:                    126/126
  final false accepts:                     1
  accepted-token error:                  0.99%
  total-token error:                     0.25%
  modeled throughput:                    7.29 tok/s
  all gates:                             pass
```

Sparse probe fit against proxy/oracle labels:

```powershell
python .\scripts\sage_probe_fit.py --logprob-json .\benchmarks\20260626-210118-sage-logprob-gemma12b-to-31b.json --probe-json .\benchmarks\20260626-214358-sage-probe-gemma31b-hybrid2-raw5.json --step-index 0
```

The first raw-prompt join is intentionally a gate, not a success claim: the available raw 5-prompt label set has `0/5` proxy/oracle top-1 matches, so it cannot train or validate a verifier. The next useful measurement is chat-aligned sparse capture or candidate-token capture with both match and mismatch labels.

Chat-aligned sparse probe:

```powershell
python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --mode gemma4-chat --gemma4-thought-prefix --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --prompts .\prompts\sage-calibration.txt --limit 5 --tag gemma31b-hybrid2-chat5-step3 --timeout 1800
python .\scripts\sage_probe_fit.py --logprob-json .\benchmarks\20260626-210528-sage-logprob-gemma12b-to-31b-chat8-filtered.json --probe-json .\benchmarks\20260626-215513-sage-probe-gemma31b-hybrid2-chat5-step3.json --step-index 3
```

This aligns the sparse probe with the first content token after Gemma4's generated thought-channel prefix. On the first 5 calibration prompts, the joined label set had `3` proxy/oracle matches and `2` mismatches. Several simple one-feature thresholds separated that tiny set with `0%` accepted error and `100%` bad-candidate catch rate. This is an early signal only; it must survive a larger calibration set before it counts as a verifier.

30-prompt chat-aligned sparse fit:

```powershell
python .\scripts\sage_probe_fit.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --probe-json .\benchmarks\20260626-215903-sage-probe-gemma31b-hybrid2-chat30-step3-00.json .\benchmarks\20260626-220101-sage-probe-gemma31b-hybrid2-chat30-step3-10.json .\benchmarks\20260626-220255-sage-probe-gemma31b-hybrid2-chat30-step3-20.json --step-index 3 --max-accepted-error 0.05
```

On 30 prompts at the same chat-aligned token step, the joined set had `20` matches and `10` mismatches. The best strict one-feature rule, `l_out_59.sum >= 32.517593`, accepted `6/30` candidates (`20%` skip) with `0%` accepted error and `100%` bad-candidate catch, but it rejected `70%` of good proxy candidates. At a looser `20%` accepted-error limit, the best rule accepted `12/30` candidates (`40%` skip) with `16.7%` accepted error and `80%` bad-candidate catch.

Interpretation: the sparse signal survived the larger calibration set, but a single-threshold verifier is too conservative for the `7-10 tok/s` goal. The next useful step is a candidate-token dataset plus a multi-feature verifier, not a claim that SAGE is solved.

Candidate-token sparse verifier:

```powershell
python .\scripts\sage_candidate_tasks.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --ignore-prefix-steps 3 --candidate-class punct --margin-threshold 0.644 --tag punct-margin0644-all

python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260626-221454-sage-candidate-tasks-punct-margin0644-all.json --mode gemma4-chat --gemma4-thought-prefix --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --limit 12 --tag candidate-punct-margin0644-fixed-00 --timeout 1800

python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260626-221454-sage-candidate-tasks-punct-margin0644-all.json --mode gemma4-chat --gemma4-thought-prefix --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --offset 12 --limit 20 --tag candidate-punct-margin0644-fixed-12 --timeout 1800

python .\scripts\sage_candidate_probe_fit.py --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json --label-quality any --max-accepted-error 0.05

python .\scripts\sage_candidate_replay.py --tasks-json .\benchmarks\20260626-223454-sage-candidate-tasks-punct-margin0644-all-v2.json --oracle-ngl 38 --top-k 10 --tag punct-margin0644-replay32

python .\scripts\sage_candidate_probe_fit.py --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json --replay-json .\benchmarks\20260626-223617-sage-candidate-replay-punct-margin0644-replay32.json --label-quality any --max-accepted-error 0.05
```

Corrected candidate-token result: the punctuation/margin router produced `32/150` candidates. Oracle replay on the proxy prefix changed the candidate label set from `26/32` matches to `30/32` matches, because several previously "bad" diverged-continuation labels became valid when judged from the proxy prefix.

With replay labels, the best current `2%` hybrid sparse rule accepts `31/32` candidates with `3.2%` accepted error and catches `1/2` bad candidates. The throughput model estimates `7.12 tok/s` with a `2%` verifier and `10%` fallback oracle path, or `7.23 tok/s` with a `1%` verifier. This is the first local measurement path that clears the `7 tok/s` budget, but it is still a small calibration set, not a production claim.

Prompt-level held-out smoke tests held out prompt groups `8,9,10,11` and `24,25,26,27`, each containing `5` good replay-labeled candidates and `1` bad one. In both splits, a rule trained on the remaining prompts accepted `5/6` held-out candidates with `0%` accepted error and caught the held-out bad candidate. Extrapolated through the same policy model, that behavior is about `7.08 tok/s` with a `2%` verifier and `10%` oracle path.

Broader punctuation+whitespace router:

```powershell
python .\scripts\sage_candidate_tasks.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --ignore-prefix-steps 3 --candidate-classes punct,whitespace --margin-threshold 0.644 --tag punct-whitespace-margin0644
python .\scripts\sage_candidate_replay.py --tasks-json .\benchmarks\20260626-224844-sage-candidate-tasks-punct-whitespace-margin0644.json --oracle-ngl 38 --top-k 10 --tag punct-whitespace-margin0644-replay43
```

Current local result: `43/150` filtered token steps became candidates, replay on the proxy prefix marked `41/43` as matching the oracle (`95.3%`), and the policy model estimates `8.00 tok/s` with no verifier and a `10%` oracle fallback, or `7.43 tok/s` with a `2%` verifier that catches both bad candidates. This is promising, but it is still a small `30`-prompt calibration set with only `2` bad replay labels.

Harder token-class expansion:

```powershell
python .\scripts\sage_candidate_tasks.py --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json --ignore-prefix-steps 3 --candidate-classes punct,whitespace,number,capitalized --margin-threshold 0.644 --tag calib30-punct-whitespace-number-capitalized
python .\scripts\sage_candidate_replay.py --tasks-json .\benchmarks\20260627-092722-sage-candidate-tasks-calib30-punct-whitespace-number-capitalized.json --oracle-ngl 38 --top-k 10 --offset 0 --limit 25 --tag calib30-pwnc-replay-o000-l025
```

The punctuation+whitespace+capitalized policy produced `75/150` candidates. Replay on the proxy prefix marked `69/75` as matching the oracle (`92.0%`): whitespace `11/11`, punctuation `30/32`, capitalized `28/32`. It fails quality without a verifier, but the policy model estimates `9.04 tok/s` with a `1%` verifier that catches `50%` of bad candidates, or `8.63 tok/s` with a `2%` verifier at the same catch rate. This is the current sharp target for SAGE: a tiny verifier must catch at least half of the remaining bad punctuation/capitalized candidates.

Full hard-120 router result: across all `120` harder prompts, the same token-class policy produced `295/600` candidates. Replay-prefix labels were `266/295` good (`90.2%`), so raw accept-all quality fails. After capturing sparse verifier signals for all `295` candidates, the best cross-chunk rule was actually proxy-side: `(proxy entropy <= 0.7938) OR (proxy margin >= 1.9020)`. It accepted `262/295`, kept accepted-token error at `4.6%`, and modeled to `9.33 tok/s` with a zero-cost proxy gate plus `10%` oracle fallback.

Fresh validation result: the hard-120 frozen proxy rule failed on `prompts/sage-validation-80.txt`, with `6.1%` accepted-token error and `2.75%` total error. A stricter diagnostic proxy gate, `(proxy entropy <= 0.6) OR (proxy margin >= 1.5)`, reached `4.55%` accepted-token error and `2.0%` total error while modeling around `9.37 tok/s`, but that is not counted as a validation pass because it was selected after seeing validation.

Second fresh validation result: the stricter rule was then predeclared and tested on `prompts/sage-validation-80b.txt`. It was still fast, modeling around `9.21 tok/s`, but missed the quality gate with `5.29%` accepted-token error and `2.25%` total error. This is now evidence that proxy thresholds alone are too brittle. The next SAGE prototype should add targeted sparse verification for surviving punctuation and capitalized-token failures before any CUDA/block-paged runtime work.

Validation-failure task extraction for the next sparse verifier:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80*pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_failure_tasks.py --replay-json @replay --max-entropy 0.6 --min-margin 1.5 --token-classes punct,capitalized --kind false-accepts --tag validation80-80b-predeclared-punct-cap-falseaccepts
```

This produced `17` targeted failure tasks locally. They became the first targeted sparse-capture set.

Targeted sparse-verifier result:

```powershell
python .\scripts\sage_failure_tasks.py --replay-json @replay --max-entropy 0.6 --min-margin 1.5 --token-classes punct,capitalized --kind true-accepts --max-per-class 17 --tag validation80-80b-predeclared-punct-cap-trueaccepts-mpc17

python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260627-115544-sage-failure-tasks-validation80-80b-predeclared-punct-cap-falseaccepts.json --mode gemma4-chat --gemma4-thought-prefix --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --tag validation-failures-hybrid2 --timeout 1800

python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --tasks-json .\benchmarks\20260627-123248-sage-failure-tasks-validation80-80b-predeclared-punct-cap-trueaccepts-mpc17.json --mode gemma4-chat --gemma4-thought-prefix --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --tag validation-trueaccepts-hybrid2 --timeout 1800
```

On `17` false accepts plus `34` true accepts, a hybrid sparse rule caught all bad validation failures in four prompt-index held-out splits with `0%` held-out accepted error. The best simple rules used `proxy.entropy` plus one layer-0 FFN/attention signal. This is promising, but still a targeted validation-failure experiment; the next proof step is a fresh held-out prompt set with the rule frozen.

Fresh frozen-verifier validation:

```powershell
python .\scripts\sage_verifier_eval.py --probe-json .\benchmarks\20260627-125555-sage-probe-validation80c-accepted-punct-cap-ffn1.json --replay-json <validation80c replay files> --expression "(proxy.entropy <= 0.040003470206379677) OR (ffn_norm_0.sum >= -72.150276000000005)" --label-quality any
```

On `prompts/sage-validation-80c.txt`, proxy-only routing failed badly: `16/144` accepted candidates were wrong (`11.1%` accepted-token error, `4.0%` total error). Applying the frozen `1%` FFN-sentinel verifier to the accepted punctuation/capitalized subset caught `12/16` bad accepts. End-to-end policy result: `4/104` accepted tokens wrong (`3.85%` accepted-token error, `1.0%` total error), modeled at about `7.35 tok/s` with a `1%` verifier path and `10%` oracle fallback.

Second frozen-verifier validation: on `prompts/sage-validation-80d.txt`, proxy-only routing again failed (`13/165` false accepts, `7.9%` accepted-token error). The same frozen `1%` FFN verifier caught `11/13` bad accepts. End-to-end policy result: `2/112` accepted tokens wrong (`1.79%` accepted-token error, `0.5%` total error), modeled at about `7.43 tok/s`.

Frozen policy report gate:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80d-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
$probe = @(Get-ChildItem .\benchmarks -File -Filter '*sage-probe-validation80d-accepted-punct-cap-ffn1-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_policy_report.py --replay-json @replay --probe-json @probe --stats-total 400 --require-full-verifier-coverage --require-pass
```

This combines replay labels, the frozen proxy gate, the frozen sparse verifier rule, and the active-byte speed model in one reproducible check. It reports proxy accepts, verifier calls, oracle fallbacks, accepted-token error, total-token error, and modeled throughput. Use the `*ffn1-stats*` probe files to validate the compact-stats path.

Current conclusion: SAGE's defensible path is no longer proxy-only routing. It is a three-stage policy: proxy gate, tiny `1%` FFN-sentinel verifier for accepted punctuation/capitalized candidates, and exact oracle fallback for everything rejected or uncertain. The compact stats hook now passes 80c, 80d, and a fresh 80e set, proving the verifier does not need full tensor dumps. The non-debug runtime hook now exposes the same tensor-stat signal during normal generation, matches the debug JSONL path on a one-prompt Gemma 31B smoke test plus 20 validation-shaped tasks, drives offline scheduler decisions that catch `6/7` known validation80e false accepts, emits in-process decisions that match the Python scheduler on `126/126` validation80e verifier tasks, and has a prompt/step multi-token replay trace that passes validation80e using only C++ verifier decisions at `7.29 tok/s` modeled throughput. The live Gemma gate now passes on small seeded prompts and proves that the replay policy can be followed by newly generated tokens. The remaining gap is runtime shape: the current Python orchestration is dominated by process/model-management overhead, so the next production milestone is a persistent SAGE runtime rather than more replay-only validation.

The larger next-gate prompt set is `prompts/sage-calibration-hard-120.txt`; the detailed run commands and pass/fail criteria are in `docs/consumer-100b-research.md`.

Router feasibility from an agreement result:

```powershell
python .\scripts\sage_router_eval.py --agreement-json .\benchmarks\20260626-204548-sage-agreement-gemma12b-to-31b-chat.json --active-percent 10
```

This computes whether the measured proxy/oracle agreement could hit the target under a perfect router. It is an upper bound, not proof that a real router exists.

## Repository Layout

```text
.
|-- README.md
|-- docs/
|   |-- consumer-100b-research.md
|   |-- live-layer-migration.md
|   |-- sage-active-byte-oracle.md
|   |-- sage-persistent-runtime.md
|   `-- nspp.md
|-- prompts/
|   |-- sage-calibration.txt
|   |-- sage-calibration-hard-120.txt
|   |-- sage-validation-80.txt
|   |-- sage-validation-80b.txt
|   |-- sage-validation-80c.txt
|   |-- sage-validation-80d.txt
|   `-- sage-validation-80e.txt
|-- patches/
|   `-- llama.cpp/
|       |-- 0001-experimental-live-layer-migration.patch
|       |-- 0002-filter-debug-eval-callback.patch
|       |-- 0003-debug-parse-special-prompt.patch
|       |-- 0004-debug-tensor-stats-hook.patch
|       |-- 0005-debug-tensor-stats-jsonl.patch
|       |-- 0006-sage-runtime-tensor-stats-hook.patch
|       |-- 0007-sage-runtime-decision-output.patch
|       |-- 0008-sage-inprocess-policy-function.patch
|       |-- 0009-sage-policy-check-example.patch
|       |-- 0010-sage-policy-check-jsonl-batch.patch
|       |-- 0011-sage-policy-check-scheduler-fields.patch
|       |-- 0012-sage-scheduler-replay-example.patch
|       |-- 0013-sage-live-proxy-example.patch
|       |-- 0014-sage-dual-live-example.patch
|       |-- 0015-debug-tensor-values-jsonl.patch
|       `-- 0016-debug-save-logits-with-tensor-capture.patch
|-- scripts/
|   |-- weak_llm.py
|   |-- sage_budget.py
|   |-- sage_simulate.py
|   |-- sage_gguf_blocks.py
|   |-- sage_block_plan.py
|   |-- sage_oracle_pager.py
|   |-- sage_signal_aware_pager.py
|   |-- sage_oracle_pager_staging.py
|   |-- sage_oracle_cuda_staging.py
|   |-- sage_oracle_cuda_kernel_smoke.py
|   |-- sage_oracle_cuda_dequant_smoke.py
|   |-- sage_oracle_cuda_matvec_smoke.py
|   |-- sage_oracle_cuda_vocab_smoke.py
|   |-- sage_oracle_cuda_candidate_smoke.py
|   |-- sage_kv_ledger.py
|   |-- sage_kv_tier_smoke.py
|   |-- sage_kv_runtime_ledger.py
|   |-- sage_agreement.py
|   |-- sage_logprob_probe.py
|   |-- sage_proxy_shortlist_eval.py
|   |-- sage_candidate_tasks.py
|   |-- sage_candidate_replay.py
|   |-- sage_failure_tasks.py
|   |-- sage_gate_eval.py
|   |-- sage_router_eval.py
|   |-- sage_router_fit.py
|   |-- sage_policy_target.py
|   |-- sage_policy_report.py
|   |-- sage_verifier_plan.py
|   |-- sage_verifier_manifest.py
|   |-- sage_probe_capture.py
|   |-- sage_runtime_capture.py
|   |-- sage_runtime_gate.py
|   |-- sage_runtime_scheduler.py
|   |-- sage_runtime_projection.py
|   |-- sage_contract_check.py
|   |-- sage_cpp_policy_parity.py
|   |-- sage_cpp_scheduler_replay.py
|   |-- sage_compare_runtime_decisions.py
|   |-- sage_multitoken_replay.py
|   |-- sage_live_loop.py
|   |-- sage_live_replay_compare.py
|   |-- sage_live_gate.py
|   |-- sage_compare_runtime_debug.py
|   |-- sage_probe_fit.py
|   |-- sage_candidate_probe_fit.py
|   |-- sage_verifier_eval.py
|   |-- phase-split-llama.ps1
|   |-- profile-llama-fit.ps1
|   `-- cuda-live-migration-check.ps1
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
