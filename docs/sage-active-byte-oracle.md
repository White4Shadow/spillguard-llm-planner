# SAGE Active-Byte Oracle Contract

This document defines the defensible SAGE-100 architecture target.

The goal is not to make dense 100B inference magically fit in 12 GB VRAM. The
goal is to make a 100B+ model useful on a consumer GPU by treating it as a
budgeted oracle that is touched only when needed, and only through a measured
active subset unless exact fallback is requested.

## Core Claim

SAGE is a consumer-GPU active-byte oracle runtime:

```text
resident proxy stream
  -> confidence and token policy
  -> tiny giant-model sentinel when needed
  -> sparse block-paged oracle when uncertain
  -> exact slow fallback when sparse evidence is not enough
```

The novelty is the control objective, not the individual ingredients. Every
token gets a byte and latency budget. The runtime must prove how many giant
model bytes, KV bytes, transfers, and fallback calls were used before claiming a
speed result.

## Prior-Art Boundary

SAGE should not claim these ideas as new:

| Area | Prior art | Boundary for SAGE |
| --- | --- | --- |
| GPU/CPU/NVMe offload | FlexGen: <https://arxiv.org/abs/2303.06865> | Offloading is known. SAGE must optimize low-latency single-user token decisions, not only batched throughput. |
| Paged KV serving | PagedAttention/vLLM: <https://arxiv.org/abs/2309.06180> | KV paging is known. SAGE must account proxy, verifier, and oracle KV separately. |
| Over-capacity device inference | PowerInfer: <https://arxiv.org/abs/2312.12456>, PowerInfer-2: <https://arxiv.org/abs/2406.06282> | Hot/cold neuron and device scheduling are known. SAGE must show a different active-byte oracle contract and exact fallback. |
| Contextual sparsity | Deja Vu: <https://arxiv.org/abs/2310.17157> | Sparse heads/MLPs are known. SAGE must bind sparsity to per-token byte budgets and quality gates. |
| Speculative decoding | Leviathan et al.: <https://arxiv.org/abs/2211.17192>, BiLD: <https://arxiv.org/abs/2302.07863> | Draft/verify is known. SAGE's proxy is a scheduler input, not the whole invention. |
| Early-exit/self-speculation | LayerSkip: <https://arxiv.org/abs/2404.16710>, CLaSp: <https://aclanthology.org/2025.acl-long.1525/> | Shared-model speculation is known. SAGE can use it only if tokenizer/model structure makes it practical. |
| KV compression | H2O: <https://arxiv.org/abs/2306.14048>, StreamingLLM: <https://arxiv.org/abs/2309.17453>, KIVI: <https://arxiv.org/abs/2402.02750>, SnapKV: <https://arxiv.org/abs/2404.14469>, InfiniGen: <https://arxiv.org/abs/2406.19707>, TurboQuant: <https://arxiv.org/abs/2504.19874> | KV compression is known. SAGE must expose KV bytes in the same ledger as weight bytes. |
| Sparse speculative verification | Sparse verification work such as <https://arxiv.org/abs/2512.21911>, <https://arxiv.org/abs/2605.19893>, <https://arxiv.org/abs/2606.24957> | Sparse verification is now an active area. SAGE must be specific to GGUF block paging and consumer-GPU byte limits. |
| Minimal runtime style | Karpathy `llm.c`: <https://github.com/karpathy/llm.c>, `llama2.c`: <https://github.com/karpathy/llama2.c> | The prototype should stay small enough to inspect and benchmark. |

## Runtime Contract

Every generated token must emit a trace with these fields:

```text
step_index
prefix_hash
proxy_token
proxy_entropy
proxy_margin
token_class
policy_action
selected_token
selected_source
verifier_active_bytes
verifier_blocks
oracle_active_bytes
oracle_blocks
oracle_mode: none | sparse | exact
proxy_kv_bytes
oracle_hot_kv_bytes
oracle_warm_kv_bytes
oracle_cold_kv_bytes
gpu_staged_bytes
host_pinned_bytes
pcie_transfer_ms
compute_ms
total_step_ms
```

Without this trace, the runtime cannot claim that it is running 100B+ in the
SAGE sense. It may only claim a smoke test.

## Active-Byte Ledger

For a 100B model at 2 bits per weight:

```text
dense weight size ~= 23.28 GiB before metadata and KV
1% active weights ~= 0.233 GiB
10% active weights ~= 2.33 GiB
```

On the local RTX 3060 class machine, the current model assumes 24 GB/s sustained
PCIe bandwidth. That makes one active percent cost about 10.42 ms of transfer
time before compute and fixed runtime overhead.

The current validation80e policy projection says:

```text
If proxy speed is 25 tok/s:
  7 tok/s permits about 10.74% active oracle.
  10 tok/s permits about 5.24% active oracle.

If proxy speed stays at the measured Gemma live-loop proxy time:
  7 tok/s permits only about 3.47% active oracle.
```

This is the hard reason the next prototype must improve both the proxy hot path
and the oracle active percentage. Improving only one side is not enough.

## Current Evidence

The current repo proves several pieces, but not the full 100B goal:

| Evidence | Status |
| --- | --- |
| C++ policy parity on validation80e | Passed: `177/177` rows and selected-token decisions matched. |
| C++ scheduler replay | Passed: one process maintains selected-token prefix state over replay rows. |
| Resident proxy live path | Passed small-model smoke: one loaded GGUF emits token, entropy, margin, class, and speed telemetry. |
| Dual-context live scheduler | Passed small-model smoke: one proxy accept and one oracle fallback with selected text fed back to both contexts. |
| Active-byte ledger trace | Passed v0 smoke: dual-live emits per-step oracle mode, active bytes, token positions, and step latency. Accepted proxy steps report `0` oracle bytes; fallback steps report exact resident-model bytes. |
| Frozen sparse-verifier policy | Passed as modeled active-byte policy on validation80e: `7.29 tok/s`, `0.25%` total-token error. |
| Format-scaffold runtime projection | Passed projection gate: hard120 measured-proxy format-scaffold path gives `9.21 tok/s` with Q6_K candidate verification every token and a `270 ms` local exact fallback for `4.33%` shortlist misses. |
| Format-scaffold measured sparse-fallback projection | Passed measured-fallback projection gate: replacing the configured fallback with the measured sparse runtime-step replay gives `8.07 tok/s` on hard120 with measured proxy speed and a `626.5 ms` conservative fallback cost. This clears `7 tok/s` but not `10 tok/s`. |
| Overlap/prefetch 10 tok/s budget target | Passed target gate: host-read hiding alone projects to `9.78 tok/s`; reaching `10 tok/s` requires hiding another `52.3 ms` of GPU fallback work, cutting proxy latency by `2.27 ms/token`, or reducing hard120 fallback rate by `1.81` percentage points. |
| Resident page-cache 10 tok/s budget target | Passed target gate: resident page-cache replay projects to `9.72 tok/s`; reaching `10 tok/s` requires shrinking active pages from `2.246 GiB` to about `1.180 GiB` (`47.5%` less), or reducing hard120 fallback rate by `2.06` percentage points. |
| Reduced resident page-cache 10 tok/s projection | Passed projection gate: a reduced ledger selects `69` pages and `1.075 GiB` active bytes (`4.62%` of a 100B 2-bit reference), replaying from resident pinned cache in `53.83 ms` and projecting to `10.08 tok/s`. This is still transport/cache evidence, not sparse transformer execution. |
| Sparse oracle page ledger | Passed plan-level gate: Gemma 31B block plan emits `79` pages, `2.246 GiB` active bytes, `9.65%` of a 100B 2-bit reference, and `4` staged transfers. |
| Sparse oracle page staging smoke | Passed measured host-staging gate: all `79` planned Gemma 31B pages resolved to real tensor byte ranges and staged `2.246 GiB` through bounded buffers at `1.88 GiB/s`; max live buffer was `0.712 GiB` under the `0.750 GiB` stage budget. |
| Sparse oracle CUDA staging smoke | Passed measured CUDA H2D gate: all `79` planned pages and `2.246 GiB` moved from pinned host buffers into CUDA staging buffers in `113.01 ms`, about `19.88 GiB/s`; sparse compute is still not implemented. |
| Sparse oracle CUDA kernel smoke | Passed byte-touch kernel gate: all `2.246 GiB` of selected page bytes were consumed by a CUDA kernel in `18.34 ms` after staging; this is not transformer scoring. |
| Sparse oracle CUDA overlap smoke | Passed two-stream overlap gate: the full `2.246 GiB` page plan measured `139.51 ms` as separate H2D+kernel work and `123.13 ms` with double-buffered H2D/kernel overlap, saving `16.38 ms` (`11.7%`). This uses pre-staged pinned host buffers and a byte-touch kernel, not sparse transformer math. |
| Sparse oracle host-prefetch CUDA-overlap smoke | Passed pipeline gate: with host reads included, the full `2.246 GiB` page plan measured `1233.49 ms` as sequential host-read+H2D+kernel components and `1116.22 ms` as a background-prefetch pipeline, saving `117.27 ms` (`9.5%`). Host reads still dominate and this remains a byte-touch smoke. |
| Sparse oracle resident page-cache replay smoke | Passed cache-reuse gate: selected pages were loaded once into `2.246 GiB` of resident pinned host cache, then replayed `3` times with `139.15 ms` average GPU replay time and `4.493 GiB` of cache-hit bytes. This is still transport plus byte-touch work, not sparse transformer execution. |
| Sparse oracle CUDA Q4_0 dequant smoke | Passed quantized-format gate: `67` Q4_0 tensors, `2.244 GiB`, and `4,282,908,672` dequantized values were processed by a CUDA kernel in `11.71 ms`; sparse matmul is still not implemented. |
| Sparse oracle CUDA Q4_0 matvec smoke | Passed synthetic-score gate: `67` real Q4_0 matrices produced `628,480` synthetic-activation output scores in `17.48 ms`; this is a mechanics baseline. |
| Reduced sparse oracle CUDA Q4_0 dequant plan | Passed reduced-plan compute gate: the `1.18 GiB`/10 tok/s page plan contains `32` Q4_0 tensors and `1.073 GiB` Q4_0 bytes, dequantized in `8.49 ms` after `49.91 ms` H2D under a `0.500 GiB` stage buffer. |
| Reduced sparse oracle CUDA Q4_0 matvec plan | Passed reduced-plan scoring gate: the same reduced page plan produced `302,336` synthetic-activation row scores in `10.76 ms` after `48.77 ms` H2D, with `32` CPU row-score checks passing. This is not token quality evidence yet. |
| Reduced sparse oracle CUDA Q4_0 real-activation matvec plan | Passed reduced-plan hidden-vector gate: using captured `ffn_norm-0`, the width-matched reduced subset is `0.715 GiB`, produces `253,952` row scores in `34.36 ms` after `33.13 ms` H2D, and passes `23` CPU checks. This still is not candidate-token quality proof. |
| Reduced page signal-quality probe | Passed measurement gate: the reduced `0.715 GiB` real-activation plan matches the fuller run's top-1 row and top-k rows for `13/13` shared scored tensors, but retains only `45%` of the fuller run's global top-20 row signals. The page selector needs signal-aware ranking before token decisions. |
| Signal-aware page selector retention probe | Passed selector gate: a measured-signal-ranked `1.173 GiB` page ledger retains `100%` of the fuller run's global top-20 row signals and `96%` of top-50, while CPU-checked CUDA scoring remains consistent. This improves the byte-only reduced selector but is still row-retention evidence, not token-decision proof. |
| Signal-aware cross-activation retention probe | Passed robustness gate: the same `ffn_norm-0`-selected signal-aware page set retains `95%` of the fuller `result_norm` top-20 row signals and `82%` of top-50. This weakens the overfit concern, but broader prompt/layer validation is still required. |
| Signal-aware page-cache 10 tok/s projection | Passed projection gate: the signal-aware page set replays from resident pinned cache in `56.57 ms` for `1.173 GiB`, remains at `5.04%` of a 100B 2-bit reference, and projects to `10.07 tok/s` under the current hard120 fallback model. This is still transport/cache evidence, not sparse transformer execution. |
| Sparse oracle CUDA Q4_0 real-activation matvec smoke | Passed hidden-state-vector gate: `48` width-matched Q4_0 matrices consumed a captured Gemma `ffn_norm-0` vector and produced `526,336` output scores in `65.51 ms`; candidate ranking and oracle logit comparison are still not implemented. |
| Sparse oracle CUDA Q4_0 ranked real-activation matvec smoke | Passed row-ranking gate: `28` tensors emitted `140` top-score rows from real activation scores, and `16` sampled rows matched CPU Q4_0 scoring with max absolute error `3.97e-07`; token-level candidate scoring is still not implemented. |
| Sparse oracle CUDA Q6_K tied-vocab projection smoke | Passed token-id scoring mechanics gate: Gemma's tied `token_embd.weight` Q6_K matrix scored all `262,144` token rows in `2` chunks under a `0.750 GiB` live buffer; `8/8` sampled top-token CPU checks passed. This older run used `ffn_norm-0`, not the final hidden state. |
| Sparse oracle CUDA Q6_K final-logit comparison | Passed exact-ranking gate for one Gemma forward pass: captured `result_norm`, paged/scored all `262,144` Q6_K vocab rows, matched llama.cpp top-1 and overlap@10 `10/10`, with max top-logit absolute error `0.0367`. |
| Sparse oracle CUDA Q6_K candidate verifier | Passed sparse shortlist gate: `64` selected vocab rows were scored from only `282,240` active bytes, `0.0244%` of the vocab tensor; candidate top-1 matched llama.cpp and `64/64` selected logits passed with max absolute error `0.0482`. |
| Live proxy shortlist to CUDA Q6_K verifier bridge | Passed bridge gate: `10` live `sage-live-proxy-shortlist-v0` token ids were consumed as sparse Q6_K candidate rows, staging only `44,100` bytes; candidate top-1 matched llama.cpp and `10/10` selected logits passed with max absolute error `0.0257`. This proves row-ID plumbing, not cross-tokenizer semantic agreement. |
| Same-prompt proxy shortlist fallback smoke | Passed fallback gate: for `The capital of France is`, Gemma 12B proxy top-k rows were scored sparsely with `44,100` active bytes and `10/10` logit checks passed, but they missed Gemma 31B global top-1 token id `9079`, proving exact fallback is required when the candidate set does not cover the oracle winner. |
| Proxy format-scaffold shortlist | Passed candidate-source gate: proxy top-k plus learned per-position format tokens plus `16` prompt-derived vocab pieces covers oracle top-1 `97.50%` on validation80e and `95.67%` on hard120, at about `27` candidate rows per step. |
| Live proxy shortlist trace | Passed live producer gate: `llama-sage-dual-live` emits `sage-live-proxy-shortlist-v0` with `10` live proxy top-k token ids per generated step, and the first id matches `logit_top1_id`. |
| Sparse oracle runtime-step replay | Passed component-replay gate: measured page-kernel evidence plus Q6_K candidate verification are combined into one active-byte ledger with `2.246490 GiB` page bytes, `44,100` candidate bytes, `9.6488%` of a 100B 2-bit reference, and exact fallback required. This is not transformer-integrated execution yet. |
| 100B sparse oracle execution | Missing. No artifact proves block-paged giant oracle execution. |
| In-process sentinel verifier in the live scheduler | Missing. The policy is callable, but the live scheduler does not yet compute the Gemma sentinel hot-path statistic. |
| KV byte ledger plan | Passed plan-level gate: Gemma 31B 4K context full KV is estimated at `3.438 GiB`; hot/warm/cold tiers reduce it to `0.817 GiB`, with `0.443 GiB` hot VRAM KV. |
| KV warm 2-bit CUDA pack smoke | Passed mechanics gate: an `8`-token Gemma KV-shaped sample packed to 2-bit at `8.00x`; CUDA pack/unpack took `0.0926 ms` / `0.0522 ms` and matched the CPU reference bytes. |
| Runtime tiered KV trace | Passed live-trace gate: `llama-sage-dual-live` emits `tiered_runtime_accounting_not_attention_integrated`, full/hot/warm/cold/tier-total KV fields, and explicit `kv_attention_integration: false`. |
| Runtime tiered KV accounting | Passed accounting gate: `sage_kv_runtime_ledger.py` annotates live trace token counts with the Gemma 31B KV tier policy, exercises warm KV in a context sweep, and carries the measured CUDA pack evidence. |
| Runtime compressed KV execution | Missing. No artifact proves llama.cpp attention reads from measured packed hot/warm/cold KV tensors yet. |

Run the current contract check:

```powershell
python .\scripts\sage_contract_check.py `
  --json-out .\benchmarks\sage-active-byte-contract-check.json
```

The expected result today is partial progress, not completion.
The current checked count is `43` passed gates and `3` failed gates.

## Next Prototype

The next useful implementation is not downloading a 100B model. It is a measured
runtime bridge:

```text
llama-sage-dual-live
  + in-process sentinel verifier
  + sparse GGUF oracle page accounting
  + measured transfer accounting
  + runtime-integrated KV byte accounting
```

Pass gate for that bridge:

```text
0 per-token process launches
0 proxy restarts during generation
proxy token telemetry present for every step
verifier statistic computed in-process
oracle fallback count measured
selected text remains synchronized across contexts
trace includes active bytes and step latency
sparse oracle fallback is measured separately from exact dense fallback
```

After that, implement the sparse oracle pager:

```text
GGUF block table
  -> active block plan
  -> page ledger with stages and transfer estimates
  -> pinned host pages
  -> GPU staging buffers
  -> sparse oracle candidate score
  -> exact fallback if sparse confidence is low
```

The first acceptable 100B-scale claim requires all of these:

```text
measured speed >= 7 tok/s
total-token error <= 2%
accepted-token error <= 5%
oracle active path <= 10% of 100B 2-bit for 7 tok/s mode
proxy/verifier/oracle byte ledger emitted per token
exact fallback still available
```

The 10 tok/s claim is stricter:

```text
measured speed >= 10 tok/s
oracle active path <= about 5% of 100B 2-bit
same quality gates
same ledger requirements
```

## Falsification Rules

Change direction if any of these remain true after the next bridge prototype:

- the proxy hot path cannot stay below roughly 40 ms/token;
- the verifier is still a full model call or sidecar process;
- the oracle path needs more than 10% active 100B weights for 7 tok/s;
- the sparse verifier only works on tuned validation-family prompts;
- exact fallback has been removed;
- KV memory grows without a bounded hot/warm/cold ledger.

## Plain-Language Summary

The project is trying to avoid asking the 100B model every question. A small
model writes most tokens. A tiny check from the big model catches risky cases.
Only when the small model and tiny check are not enough do we touch a bigger
piece of the 100B model. If that still is not enough, the system falls back to
the slow exact path.

The innovative part is making that decision with a measured byte budget on a
single consumer GPU, then proving it token by token.
