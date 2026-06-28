# SAGE Persistent Runtime RFC

This note turns the current SAGE-100 research result into an implementable
runtime plan. It is deliberately conservative: the goal is useful 100B+ local
inference on consumer hardware, not a claim that dense 100B inference can fit or
run interactively on an RTX 3060 12 GB.

The active-byte oracle contract is the stricter completion definition for this
track:

```text
docs/sage-active-byte-oracle.md
```

Use it to distinguish implemented runtime pieces from modeled speed assumptions
and still-missing 100B sparse-oracle/KV-ledger evidence.

## Current Evidence

Local validation80e policy report:

```text
artifact: benchmarks/20260627-154000-sage-policy-report-validation80e-frozen-ffn1-stats.json
stats total:                         400
verifier call rate:                31.5%
oracle call rate:                  74.75%
final accepted proxy tokens:       101/400
final false accepts:                 1
total-token error:                 0.25%
modeled speed:                     7.29 tok/s
```

Local live gate report:

```text
artifact: benchmarks/gemma-live-report-validation80e-p001-p035.json
live/replay compared steps:           3
live/replay matches:                3/3
observed live speed:              0.059 tok/s
outer orchestration overhead:      68.8%
```

Runtime projection:

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

```text
live Python loop:                    0.059 tok/s
same request timings without overhead: 0.189 tok/s
persistent active-byte policy model: 7.295 tok/s
active-byte model with measured proxy: 5.163 tok/s
format scaffold, hard120, measured proxy: 9.214 tok/s
format scaffold, hard120, measured sparse fallback: 8.066 tok/s
format scaffold, hard120, sparse H2D+kernel fallback: 9.778 tok/s
format scaffold, hard120, configured proxy: 19.252 tok/s
```

The older high-oracle-call policy gives the hard active-percent constraint:

```text
If proxy latency stays at the measured ~96.6 ms/token:
  7 tok/s requires oracle active path <= 3.47% of a 100B 2-bit model.
  10 tok/s is already impossible before oracle work.

If proxy speed reaches 25 tok/s, as assumed by the policy model:
  7 tok/s permits oracle active path <= 10.74%.
  10 tok/s requires oracle active path <= 5.24%.
```

The newer format-scaffold projection changes the runtime target. It verifies a
small Q6_K candidate set on every token and uses exact fallback only for
shortlist misses. With the measured live proxy time, hard120 fallback rate
`4.33%`, candidate verifier host-read/H2D/kernel time included, and local
fallback set to `270 ms/token`, the projection reaches `9.214 tok/s`. At that
fallback rate, `7 tok/s` can tolerate up to about `1062 ms` exact-fallback
latency, while `10 tok/s` requires fallback below about `73 ms`.

The measured sparse-fallback projection replaces the configured `270 ms` local
fallback with the `sage-sparse-oracle-runtime-step-v0` replay cost. The
conservative full component replay is `626.5 ms` per fallback and still gives
`8.066 tok/s` on hard120 with the measured proxy path. The H2D+kernel-only view
is `125.4 ms` per fallback and projects to `9.778 tok/s`, but that excludes host
reads and is not the contract gate. This tells us the architecture has measured
budget room for `7 tok/s`; reaching `10 tok/s` requires a faster proxy path,
lower fallback rate, or true overlapped sparse execution.

`scripts/sage_overlap_budget.py` now turns that miss into a concrete engineering
target. For hard120, after host reads are hidden, `10 tok/s` still requires one
of: hide another `52.3 ms` of GPU fallback work per fallback, reduce measured
proxy latency by `2.27 ms/token`, or reduce fallback rate by `1.81` percentage
points. This is the next implementation target for async page prefetch and CUDA
stream overlap.

The first CUDA overlap smoke now measures that target directly for the page
transport plus byte-touch kernel window. `scripts/sage_oracle_cuda_overlap_smoke.py`
uses pre-staged pinned host buffers, two CUDA streams, and two device staging
buffers. On the full Gemma 31B page plan it reduced the measured GPU window from
`139.51 ms` separate H2D+kernel work to `123.13 ms` overlapped work, saving
`16.38 ms` (`11.7%`). This covers part of the `52.3 ms` 10 tok/s gap, but it is
not host-read overlap and not sparse transformer execution.

`scripts/sage_oracle_cuda_prefetch_overlap_smoke.py` then keeps host staging in
the measured wall time. It uses one background GGUF reader, a pinned host-buffer
ring, two CUDA streams, and two device buffers. On the same page plan, sequential
host-read+H2D+kernel components totaled `1233.49 ms`; the pipelined wall time was
`1116.22 ms`, saving `117.27 ms` (`9.5%`). This proves the direction for
prefetch, but it also shows why production cannot read and CRC GGUF pages on
the fallback path. The next pager needs a resident pinned page cache or a much
smaller active set.

`scripts/sage_oracle_cuda_page_cache_smoke.py` measures that resident-cache
shape. It loads the selected page stages once into pinned host memory, then
replays multiple fallback passes from that cache. On the full Gemma 31B page
plan, cache build took `1464.97 ms` for `2.246 GiB`; three replay passes then
averaged `139.15 ms` GPU time each, with `4.493 GiB` served from cache hits.
This is the first measurement that separates one-time sparse-page residency
from per-fallback transport cost. The result is still above the 10 tok/s
fallback budget and still uses only a byte-touch kernel.

`scripts/sage_page_cache_budget.py` turns that into the next concrete page-plan
target. With the measured hard120 proxy/fallback rate and resident cache replay,
the format-scaffold path projects to `9.72 tok/s`. To reach `10 tok/s`, the
same page mix must shrink from `2.246 GiB` to about `1.180 GiB`, or the hard120
fallback rate must drop from `4.33%` to about `2.28%`. This makes the next
oracle-pager task sharper: find a useful sparse verifier/oracle plan around
`1.18 GiB`, not just make the current `2.246 GiB` plan faster.

## Position Against Prior Work

SAGE should not claim novelty for the individual ingredients:

- [FlexGen](https://arxiv.org/abs/2303.06865) already shows that GPU, CPU,
  and disk memory can be scheduled for large-model inference on a single
  commodity GPU, but its target is high-throughput batched inference rather than
  one-user low-latency decoding.
- [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) already treats KV
  cache memory as a paged serving resource.
- [PowerInfer-2](https://arxiv.org/abs/2406.06282) already uses fine-grained
  neuron clusters, I/O-compute pipelining, and segmented caching for
  over-capacity LLM inference on a phone-class device.
- [KIVI](https://arxiv.org/abs/2402.02750), [SnapKV](https://arxiv.org/abs/2404.14469),
  [H2O](https://arxiv.org/abs/2306.14048), [StreamingLLM](https://arxiv.org/abs/2309.17453),
  [InfiniGen](https://arxiv.org/abs/2406.19707), and
  [TurboQuant](https://arxiv.org/abs/2504.19874) already cover KV
  quantization, eviction, attention sinks, and dynamic KV management.
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
  established draft/target speculative decoding.
- Recent sparse verification work such as
  [Accelerate Speculative Decoding with Sparse Computation in Verification](https://arxiv.org/abs/2512.21911),
  [SSV](https://arxiv.org/abs/2605.19893), and
  [Dustin](https://arxiv.org/abs/2606.24957) already makes sparse verification
  an active research direction.
- [karpathy/llm.c](https://github.com/karpathy/llm.c) and
  [karpathy/llama2.c](https://github.com/karpathy/llama2.c) are useful examples
  of keeping kernels and runtime control paths small enough to reason about.

The defensible SAGE novelty is the combination: a consumer-GPU, active-byte
ledger that decides per token whether to trust a resident proxy, consult a tiny
giant-model sentinel, run a sparse oracle path, or fall back to exact slow
execution, with explicit quality gates and byte budgets.

## Runtime Shape

The production runtime should be one persistent scheduler, not a Python loop that
restarts servers or launches verifier processes per token.

```text
prompt
  |
  v
resident proxy context
  |
  v
candidate policy: entropy, margin, token class
  |
  +-- reject/unsafe --> sparse oracle or exact fallback
  |
  +-- simple safe class --> accept proxy token
  |
  +-- verifier class --> in-process FFN sentinel
                            |
                            +-- accept --> emit proxy token
                            |
                            +-- reject --> sparse oracle or exact fallback
```

### Required Runtime Objects

1. `sage_scheduler`

Owns the token loop, the active-byte ledger, quality policy, and event output. It
must emit the same decision fields currently produced by `scripts/sage_live_loop.py`
so replay comparison remains possible.

2. `sage_proxy_context`

Resident small model context. This must not restart during generation. If the
Gemma 12B proxy cannot stay below roughly `40 ms/token`, use a smaller proxy or
a draft window so the proxy path does not consume the full latency budget.

3. `sage_verifier_context`

Persistent micro-verifier path. The current `0007` patch proves the C++ decision
event, but it still runs through one-shot `llama-completion`. Patch `0008`
exposes the frozen decision policy as `common_sage_decide()`, which is the first
callable building block for a persistent loop:

```text
sage_verify_candidate(candidate, proxy_stats, prefix_state) -> accept_proxy | oracle_fallback
```

The remaining hot-path work is to compute the sentinel statistic directly and
return the result to the scheduler, not write JSONL and parse it inside the loop.
Patch `0009` adds `llama-sage-policy-check`, a model-free executable that tests
the same policy function without any Python, server, or model load.
`scripts/sage_cpp_policy_parity.py` then runs that executable over replay/probe
rows. Patch `0010` adds JSONL batch mode, so validation80e now runs through one
persistent C++ policy process and still matches `177/177` actions and reasons
with `126/126` verifier rows covered. Patch `0011` adds scheduler-level JSONL
fields, and the same validation now also matches `177/177` selected tokens and
`177/177` false-accept flags with `1` expected final false accept.
Patch `0012` adds `llama-sage-scheduler-replay`, a one-process C++ scheduler
skeleton that consumes replay candidates, selects emitted tokens, and maintains
generated text per prompt stream. On validation80e it matches `177/177` actions,
selected tokens, false-accept flags, prefixes, and generated-text states across
`40` prompt streams.
Patch `0013` adds `llama-sage-proxy-live`, the first resident model-backed proxy
path: one GGUF load, one llama.cpp context, greedy proxy generation, logits
entropy/margin telemetry, token class, and SAGE proxy-gate fields in JSON. The
local Qwen 0.5B CUDA smoke produced 4 tokens at `92.23 tok/s` in a short run.
Patch `0014` adds `llama-sage-dual-live`, a two-context live scheduler skeleton:
resident proxy context, resident oracle context, frozen SAGE proxy decision,
selected-token feedback into both contexts, and JSON showing proxy/oracle
decode counts. The local arithmetic smoke exercised both paths: `1` proxy
accept and `1` oracle fallback, ending with `" 4"`. It now also emits the
`sage-active-byte-ledger-v0` trace: proxy-accepted steps report `0` oracle
active bytes, exact fallback steps report the resident oracle model bytes, and
each step records proxy/oracle token positions plus decode/step latency. It also
emits tiered KV accounting fields from runtime token counts. The Qwen 0.5B
arithmetic smoke reports `12288` full-precision KV bytes/token, status
`tiered_runtime_accounting_not_attention_integrated`, and `73728` oracle tiered
KV bytes on the fallback step. This is live byte accounting, not packed KV
attention.

4. `sage_oracle_pager`

Block-paged giant-model executor. It should treat GGUF tensor groups as
addressable blocks, prefetched into pinned host memory and staged through one or
two GPU buffers. The policy decides the active percent per token. The current
7 tok/s budget can tolerate around `10%` active oracle at `25 tok/s` proxy speed;
the 10 tok/s budget needs about `5%` active oracle.

`scripts/sage_oracle_pager.py` is the first plan-level ledger for this object.
On the local Gemma 31B Q4 file with a `2.33 GiB` budget, the quota-controlled
plan emits `79` pages, `2.246 GiB` selected bytes, `9.65%` of a 100B 2-bit
reference, `4` staged transfers, and about `100.5 ms` estimated PCIe transfer.
This is not execution yet; it is the byte/stage contract the executable pager
must satisfy or beat.

`scripts/sage_oracle_pager_staging.py` is the next smoke step. It consumes the
page ledger, reparses the GGUF tensor table, resolves selected block pages to
real tensor byte ranges, and reads those bytes through fixed-size staging
buffers. The current full Gemma 31B smoke staged all `79` planned pages,
`2.246 GiB` total, through `4` stages. Max live buffer use was `0.712 GiB`
under the `0.750 GiB` stage budget and measured host-staging throughput was
`1.88 GiB/s`. This is still CPU file-to-host staging, not pinned CUDA H2D
transfer or sparse matrix execution.

`scripts/sage_oracle_cuda_staging.py` adds the CUDA transport smoke. It allocates
pinned host staging buffers and CUDA device staging buffers through `cudart`,
then copies the selected pages into the GPU buffers with timed H2D transfers.
The current full Gemma 31B smoke moved all `79` pages and `2.246 GiB` into CUDA
buffers in `113.01 ms`, about `19.88 GiB/s`, with max live device buffer use of
`0.712 GiB` under the `0.750 GiB` stage budget. This proves the transfer layer,
not sparse oracle computation.

`scripts/sage_oracle_cuda_kernel_smoke.py` proves that the staged bytes can be
consumed by CUDA kernels. It compiles a tiny byte-sum kernel with NVRTC, launches
it over each staged device buffer, and records kernel timing. The current full
Gemma 31B smoke touched all `2.246 GiB` of selected page bytes in `18.34 ms`
after `106.98 ms` of H2D transfer, with `122.49 GiB/s` measured kernel-touch
throughput. This still is not dequantization, matrix multiplication, attention,
or candidate scoring.

`scripts/sage_oracle_cuda_overlap_smoke.py` measures whether the page transport
can be overlapped with GPU work. It pre-stages the same selected pages in pinned
host memory, then enqueues H2D copies and the byte-touch kernel on separate CUDA
streams with event dependencies and two device buffers. The current full Gemma
31B smoke touched all `2.246 GiB` of selected page bytes, with `120.58 ms` of
summed H2D, `18.93 ms` of summed kernel work, `139.51 ms` of separate GPU work,
and a measured overlapped GPU window of `123.13 ms`. The `16.38 ms` saving is a
real CUDA overlap measurement, but the artifact explicitly reports
`host_read_overlap_status: not_measured_prestaged_pinned_host_buffers` and
`sparse_transformer_status: not_implemented`.

`scripts/sage_oracle_cuda_prefetch_overlap_smoke.py` measures the next pipeline
shape, where host staging is no longer excluded. A single background worker reads
the next stage from the GGUF into a pinned host-buffer ring while the current
stage is copied and touched by CUDA. The current full Gemma 31B smoke reports
`1100.09 ms` summed host-read time, `114.62 ms` summed H2D, `18.79 ms` summed
kernel time, `1233.49 ms` sequential components, and `1116.22 ms` measured
pipeline wall time. The `117.27 ms` saving is useful implementation evidence,
but it is not enough for the target by itself because the measured pipeline is
still dominated by raw host reads rather than reusable resident pages.

`scripts/sage_oracle_cuda_page_cache_smoke.py` moves the repeated-fallback
measurement to a resident pinned page cache. It reads the same `2.246 GiB` page
set once, then runs repeated CUDA H2D/touch-kernel replays from the cache. The
current full Gemma 31B smoke used `3` replays: cache build took `1464.97 ms`,
average per-replay H2D was `136.31 ms`, average touch-kernel time was
`20.43 ms`, and the overlapped per-replay GPU window was `139.15 ms`. This
proves cache-hit reuse and removes raw host reads from repeated fallbacks, but
the remaining per-replay cost still requires smaller active pages, faster sparse
kernels, or lower fallback rate to reach the `10 tok/s` mode.

`scripts/sage_page_cache_budget.py` derives the active-page reduction target
from that measurement. With current hard120 fallback behavior, resident cache
replay gives `9.72 tok/s`; the `10 tok/s` target allows only about `73.10 ms`
per fallback. At the measured `61.94 ms/GiB` cache replay rate, that means the
next page plan should be about `1.180 GiB`, `5.07%` of a 100B 2-bit reference,
or about `41` pages if the current page mix is kept.

The reduced-page follow-up tests that target directly. A second
`sage_oracle_pager.py` run with `--budget-gib 1.18`, `--stage-buffer-gib 0.50`,
and the same FFN/attention quota selects `69` pages and `1.075 GiB` of active
bytes, equal to `4.62%` of a 100B 2-bit reference. Its resident page-cache
replay averages `53.83 ms` for the selected page set, and
`scripts/sage_page_cache_budget.py` projects the hard120 format-scaffold path
at `10.08 tok/s`. This is a useful budget proof, but it is still cache
transport plus a byte-touch kernel; the next implementation must turn that
reduced page set into sparse dequant/matmul with candidate-quality checks.

The reduced compute smokes now exercise that page set with real GGUF Q4_0
blocks. `sage_oracle_cuda_dequant_smoke.py` processes `32` selected Q4_0
tensors, `1.073 GiB` of quantized bytes, in `8.49 ms` of dequant kernel time
after `49.91 ms` H2D. `sage_oracle_cuda_matvec_smoke.py` then runs the same
reduced set through the synthetic-activation matvec kernel, producing `302,336`
row scores in `10.76 ms` after `48.77 ms` H2D with `32` CPU score checks
passing. This confirms the smaller 10 tok/s page plan is compute-capable; it
still needs captured hidden states, candidate-token mapping, and exact oracle
logit comparison.

The same reduced ledger also runs with the captured Gemma `ffn_norm-0` vector.
The width-matched subset is `23` Q4_0 tensors and `0.715 GiB`; it produces
`253,952` row scores in `34.36 ms` after `33.13 ms` H2D, with `23` CPU score
checks passing. This is closer to the eventual sparse oracle because the input
is a real hidden vector, but the scores are still projection rows, not
candidate-token accept/reject decisions.

`scripts/sage_reduced_page_quality_probe.py` compares those reduced row signals
against the fuller `1.482 GiB` real-activation matvec artifact. The result is
mixed in the useful way: for `13` tensors scored by both runs, top-1 rows match
`100%` and top-k overlap is `100%`, so the reduced CUDA scoring is stable for
shared tensors. But the reduced selector retains only `45%` of the fuller
run's global top-20 row signals and `52%` of the top-50. The next pager should
therefore use signal-aware page ranking, not only first/last-layer byte quotas.

`scripts/sage_signal_aware_pager.py` is the first direct response to that
weakness. It consumes the fuller real-activation ranked matvec artifact and
ranks GGUF layer/component pages by measured row-score signal under the same
`1.18 GiB`/10 tok/s active-byte target. On the captured Gemma `ffn_norm-0`
vector, the signal-aware ledger selects `70` pages and `1.173 GiB` active bytes,
or `5.04%` of a 100B 2-bit reference. The matching real-activation CUDA matvec
scores `25` Q4_0 tensors and `0.767 GiB`, with `25` CPU checks passing.

The signal-aware quality probe is materially better than the byte-only reduced
plan: it retains `100%` of the fuller run's global top-20 row signals and `96%`
of top-50, versus `45%` and `52%` for the prior reduced plan. Its resident
page-cache replay averages `56.57 ms` for `1.173 GiB`, and the same hard120
fallback model projects `10.07 tok/s`. This is now the strongest measured
evidence for SAGE's active-byte selector.

The first cross-activation check keeps that `ffn_norm-0`-selected page set and
evaluates it against the final `result_norm` vector used by the Q6_K logit
comparison path. The fuller `result_norm` Q4_0 matvec again scores `48`
width-matched tensors and `1.482 GiB`; the signal-aware subset scores `25`
tensors and `0.767 GiB`. Retention stays high for the strongest rows:
`95%` of top-20 and `82%` of top-50. This reduces the risk that the selector is
only memorizing one hidden vector, but it is still not enough to claim the 100B
runtime goal because the result covers two captured activations, not a
transformer-integrated token-decision path across prompts.

`scripts/sage_oracle_cuda_dequant_smoke.py` adds the first quantized-format
compute proof. It packs only selected GGUF `Q4_0` tensors into the staging
buffers and runs a CUDA kernel that decodes Q4_0 blocks and reduces the
dequantized values. The current full Gemma 31B smoke processed `67` Q4_0
tensors, `2.244 GiB` of quantized weights, and `4,282,908,672` dequantized
values in `11.71 ms` after `101.75 ms` of H2D transfer. This still is not sparse
matmul, attention, or candidate scoring.

`scripts/sage_oracle_cuda_matvec_smoke.py` is the first matrix-vector scoring
mechanics proof. It runs Q4_0 matrix-vector kernels over the selected GGUF
tensors using a deterministic synthetic activation vector. The current full
Gemma 31B smoke processed `67` Q4_0 matrices, `2.244 GiB` of quantized weights,
`4,282,908,672` weight values, and `628,480` output scores in `17.48 ms` of
matvec kernel time after `101.00 ms` of H2D transfer. The same script can also
consume a bounded llama-debug tensor-values sidecar. With a captured Gemma
`ffn_norm-0` vector, it processed `48` width-matched Q4_0 matrices,
`1.482 GiB` of quantized weights, `2,829,582,336` weight values, and `526,336`
output scores in `65.51 ms` of matvec kernel time after `86.12 ms` of H2D
transfer. With row-score capture enabled, it emitted top-k rows for `28`
tensors, produced `140` ranked row records, and CPU-checked `16` sampled top
rows against the raw Q4_0 bytes with max absolute error `3.97e-07`. This uses
live hidden-state values and proves projection-row ranking mechanics, but it
still does not implement attention/FFN composition, candidate-token scoring, or
sparse-score comparison to oracle logits.

`scripts/sage_oracle_cuda_vocab_smoke.py` adds the first token-id scoring
mechanics proof. Gemma 31B ties its output projection to `token_embd.weight`,
which is `Q6_K` and `1.077 GiB`, larger than the `0.750 GiB` stage buffer. The
smoke pages that tensor in `2` chunks, scores all `262,144` token rows against
the captured `ffn_norm-0` vector, emits top token ids, and CPU-checks sampled
top rows from raw GGUF bytes. The current full vocab run measured `49.10 ms`
H2D transfer and `29.83 ms` CUDA kernel time, with `8/8` CPU checks passing.
This proves bounded Q6_K token-row projection, but it is still not true logits
until the activation is the final post-norm hidden state and the scores are
compared against llama.cpp logits.

The stricter final-logit run now captures `result_norm` from the prompt
`The capital of France is` and compares the paged CUDA Q6_K vocab projection
against llama.cpp's saved logits. The current artifact scores all `262,144`
rows in `2` chunks with max live GPU staging buffer `0.750 GiB`, H2D transfer
`56.35 ms`, and CUDA kernel time `29.87 ms`. It matches llama.cpp top-1 and
overlap@10 `10/10`, with max top-logit absolute error `0.0367`, while `8/8`
sampled raw-row CPU checks still pass. This is the first token-level proof that
the paged projection path can reproduce llama.cpp's final ranking for a real
Gemma hidden state.

`scripts/sage_oracle_cuda_candidate_smoke.py` narrows that proof to the shape
needed by a sparse oracle verifier. Given a candidate shortlist, it reads only
those Q6_K vocab rows, packs them into one tiny staging buffer, and scores the
selected rows against the same captured `result_norm` vector. The current
top-64 candidate run touched `282,240` active bytes, just `0.0244%` of the vocab
tensor, and measured `0.0493 ms` H2D plus `0.1034 ms` CUDA kernel time. All
`64/64` selected logits matched llama.cpp within the configured error bound,
with max absolute error `0.0482` and candidate top-1 match. This still depends
on an already-computed hidden state, but it proves the output-head verifier can
be sparse over token rows instead of full-vocab.

`scripts/sage_proxy_shortlist_eval.py` now measures the missing upstream piece:
whether a proxy can supply a small candidate set without peeking at oracle
top-k. The current best policy uses proxy top-k plus a learned per-position
format scaffold and up to `16` prompt-derived vocabulary pieces. On held-out
validation80e rows it covers oracle top-1 `97.50%` with `27.0` candidate rows
per step. On hard120 it covers `95.67%` with `26.9` rows per step. That keeps
the shortlist below the measured `64`-row Q6_K verifier budget, but still leaves
`2.50%` to `4.33%` exact fallbacks before any end-to-end runtime cost is known.

`llama-sage-dual-live` now emits the live proxy side of that handoff: each proxy
token row includes `logit_top_ids`, `logit_top_logits`, and a
`sage-live-proxy-shortlist-v0` candidate shortlist. The current Qwen arithmetic
smoke emits `10` live top-k token ids per generated step, with the first id
matching `logit_top1_id`. This is not yet CUDA verification in the live loop,
but it removes the previous gap where the C++ runtime had no candidate producer.

`scripts/sage_oracle_cuda_candidate_smoke.py` can now consume that live trace
directly via `--candidate-live-trace-json`. The current bridge artifact reads
the first Qwen live proxy shortlist and uses its `10` token ids as sparse Gemma
Q6_K row ids against the captured Gemma `result_norm` vector. It stages only
`44,100` bytes, runs the CUDA candidate kernel in `0.1270 ms`, and matches
llama.cpp's ranking among those selected rows with `10/10` logit checks passing
and max absolute error `0.0257`. Because this artifact crosses a Qwen live trace
with Gemma rows, it proves runtime row-ID plumbing and sparse scorer mechanics;
the next persistent runtime milestone is same-tokenizer live proxy shortlist
verification inside the scheduler loop.

The same script also now accepts Gemma12-to-31B logprob captures as a same-prompt
candidate source. On `The capital of France is`, the Gemma 12B proxy top-k rows
score correctly against Gemma 31B logits with only `44,100` active bytes and
`0.0911 ms` CUDA kernel time, but the shortlist misses Gemma 31B global top-1
token id `9079` (` Paris`). That artifact proves the reject side of the runtime
contract: the live scheduler must not treat sparse candidate scoring as success
unless the candidate source covers the oracle winner, and it must retain exact
fallback for misses.

`scripts/sage_oracle_runtime_step.py` now assembles these measured pieces into a
single per-token ledger replay. The current artifact combines the Gemma 31B page
plan, CUDA page-kernel smoke, same-prompt Q6_K proxy-candidate verifier, and KV
accounting reference. It reports `2.246490 GiB` of sparse page bytes plus
`44,100` candidate bytes, `9.6488%` of a 100B 2-bit reference, `0.711755 GiB`
max measured device staging, `107.0062 ms` H2D, `18.4317 ms` CUDA kernel time,
and exact fallback required. Its status is deliberately
`measured_component_replay_not_transformer_integrated`: it proves the runtime
ledger shape and byte budget, not live sparse transformer execution.

5. `sage_kv_ledger`

Separate proxy and oracle KV accounting:

- proxy KV: normal interactive KV, resident when possible;
- verifier KV: minimal or reused prefix state;
- oracle KV: hot recent/sink cache in VRAM, compressed warm cache in RAM, and
  optional cold summaries for long context.

This is where KIVI/SnapKV/H2O/StreamingLLM/InfiniGen ideas should be used, but
only behind SAGE's active-byte and quality policy.

`scripts/sage_kv_ledger.py` is the first plan-level byte ledger for this object.
On local Gemma 31B Q4 with a 4096-token context, 512 recent tokens plus 16 sink
tokens hot, and 2-bit warm KV, the planner estimates `3.438 GiB` for full
precision KV and `0.817 GiB` for the SAGE hot/warm/cold tiers. The hot VRAM tier
is `0.443 GiB`, so it fits the current `0.8 GiB` oracle-hot-KV budget.
`llama-sage-dual-live` now reports the same hot/warm/cold accounting fields from
runtime token counts, but it still does not allocate packed KV tensors or measure
compressed attention reads.

`scripts/sage_kv_tier_smoke.py` adds the first measured KV compression mechanics
artifact. It uses the same inferred Gemma KV dimensions, creates a bounded
synthetic FP16 warm-KV sample, packs it into 2-bit codes, and verifies CUDA
packed bytes against a CPU reference. On an `8`-token sample, CUDA pack/unpack
took `0.0926 ms` / `0.0522 ms`, the compression ratio was `8.00x`, and the
scaled warm tier remains `0.374 GiB`. This is still not attention over compressed
KV: the missing runtime step is wiring native pack/unpack to real llama.cpp KV
tensors and measuring hot/warm/cold attention reads.

`scripts/sage_kv_runtime_ledger.py` adds the first runtime-trace accounting
bridge. It reads `llama-sage-dual-live` token counts, applies the Gemma 31B KV
tier policy, and attaches the CUDA pack-smoke evidence. The current artifact
annotates `2` runtime steps, includes `1` oracle fallback, exercises warm KV in
a 4K context sweep, and reports `3.438 GiB` full-precision oracle KV versus
`0.817 GiB` tiered KV. This proves byte accounting for the active-byte ledger;
it still does not prove compressed KV tensors are used by attention.

## Implementation Sequence

### Stage 1: Persistent Scheduler Skeleton

Target: remove per-token server/process orchestration while keeping the current
policy behavior.

Deliverables:

- a C++ `llama-sage` example or a long-lived worker prototype;
- model-free `llama-sage-scheduler-replay` validating persistent token selection;
- one resident proxy context, currently proven by `llama-sage-proxy-live`;
- one resident oracle context synchronized by selected text, currently proven by `llama-sage-dual-live`;
- per-step v0 active-byte ledger in `llama-sage-dual-live`;
- one in-process candidate policy;
- decision JSONL matching the existing `sage_live_loop.py` schema;
- validation80e live/replay comparator pass on seeded prompts.

Pass gate:

```text
0 per-token process launches
0 proxy restarts during generation
live/replay action matches: 100% on seeded gate
live/replay selected-token matches: 100% before prefix drift
C++ scheduler replay prefix matches: 100% on validation80e
resident proxy smoke: emits token/class/entropy/margin JSON from one loaded GGUF
dual live smoke: at least one proxy accept and one oracle fallback with synchronized selected text
dual live ledger smoke: accept path reports 0 oracle bytes and fallback path reports exact resident-model bytes
```

### Stage 2: In-Process Sentinel Verifier

Target: replace `llama-completion` verifier subprocess calls with a direct
runtime call that computes only the frozen sentinel statistic.

Deliverables:

- reusable `common_sage_decide()` policy function;
- model-free `llama-sage-policy-check --self-test` regression gate;
- replay-scale `sage_cpp_policy_parity.py --require-pass` gate with one C++ JSONL batch process;
- C++ selected-token and false-accept parity against replay rows;
- no JSON sidecar in the hot path;
- direct `ffn_norm-0.sum` or equivalent sentinel result;
- parity with `0007` decision events on validation80e verifier rows;
- `scripts/sage_runtime_projection.py` updated with measured verifier call time.

Pass gate:

```text
verifier coverage: 126/126 on validation80e replay rows
C++ decision parity: 126/126
verifier call target: near the active-byte model, not seconds per call
```

### Stage 3: Sparse Oracle Pager

Target: make oracle fallback cheaper than dense 31B/100B execution.

Deliverables:

- GGUF block table from `sage_gguf_blocks.py`;
- policy-selected FFN/attention block groups;
- `sage-oracle-page-ledger-v0` artifact with pages, stages, active bytes, and exact fallback bytes;
- measured `sage-oracle-page-staging-v0` smoke that resolves page ranges and respects the stage buffer budget;
- measured `sage-oracle-page-cuda-staging-v0` smoke that copies planned pages into bounded CUDA staging buffers;
- measured `sage-oracle-page-cuda-kernel-smoke-v0` artifact proving CUDA kernels can consume the staged page bytes;
- measured `sage-oracle-page-cuda-dequant-smoke-v0` artifact proving CUDA kernels can decode staged GGUF Q4_0 blocks;
- measured `sage-oracle-page-cuda-matvec-smoke-v0` artifact proving selected Q4_0 matrices can produce synthetic activation scores;
- pinned host pages and async GPU staging buffers;
- exact fallback if the sparse oracle is uncertain.

Pass gate:

```text
7 tok/s mode: oracle active path <= 10% with 25 tok/s proxy
10 tok/s mode: oracle active path <= 5% with 25 tok/s proxy
measured total-token error <= 2%
measured accepted-token error <= 5%
```

### Stage 4: KV Ledger

Target: keep long-context behavior from breaking the active-byte budget.

Deliverables:

- hot recent/sink KV in VRAM;
- quantized warm KV in RAM;
- `sage-kv-ledger-v0` artifact with planned hot/warm/cold bytes;
- live full-precision KV estimates in `llama-sage-dual-live`;
- runtime-trace tiered KV accounting from `sage_kv_runtime_ledger.py`;
- deterministic accounting of KV transfer bytes per token;
- long-context gate based on held-out prompts, not only short validation80e.

Pass gate:

```text
KV bytes/token reported in trace
no unbounded KV growth in VRAM
long-context quality gate does not regress short-context validation
```

## Falsification Rules

Stop or change direction if any of these remain true after the relevant stage:

- proxy path cannot reach roughly `40 ms/token` or better;
- verifier remains a full model invocation measured in seconds;
- oracle fallback needs more than about `10%` active 100B-equivalent weights for
  7 tok/s, or more than about `5%` for 10 tok/s;
- sparse verifier catches validation failures only on the tuned prompt family;
- live/replay parity fails before prefix drift;
- the exact fallback is removed instead of kept as a quality escape hatch.

## Near-Term Engineering Target

The next implementation should not download a 100B model yet. The next useful
patch is a persistent `llama-sage` runtime skeleton over the existing Gemma
31B/12B artifacts:

```text
proxy token -> candidate policy -> in-process sentinel decision -> oracle fallback
```

Once this skeleton removes process orchestration and reports real proxy/verifier
latency, `scripts/sage_runtime_projection.py` tells us whether the path can still
reach the 7-10 tok/s band or whether the architecture must become more sparse.
