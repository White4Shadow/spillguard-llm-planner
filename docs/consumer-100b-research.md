# Consumer 100B+ Research Track

This track asks a harder question than the current Gemma 31B work:

```text
Can a consumer desktop with one RTX 3060 12 GB class GPU make useful
100B+ model inference possible?
```

The answer depends on the definition of "run":

- Exact dense inference is possible only slowly if most weights live in CPU RAM or NVMe.
- Fast interactive inference is possible only if the runtime avoids touching most 100B weights for most tokens.

The new research direction is therefore not "stream the whole 100B model faster." It is a sparse, budgeted, oracle-assisted runtime.

The stricter architecture contract for this track is:

```text
docs/sage-active-byte-oracle.md
```

That contract defines the per-token byte ledger and the gates that must pass
before the project can claim useful 100B+ inference on consumer hardware.

## Local Hardware Boundary

Current local target:

```text
GPU:  NVIDIA RTX 3060, 12 GB VRAM
CPU:  Intel i7-13700KF
RAM:  32 GB
Disk: NVMe SSD
Bus:  PCIe Gen4 x16 capable
```

Approximate dense weight sizes:

| Model | 4-bit | 2-bit | 1-bit before overhead |
| --- | ---: | ---: | ---: |
| 31B | 15.5 GB | 7.8 GB | 3.9 GB |
| 70B | 35 GB | 17.5 GB | 8.8 GB |
| 100B | 50 GB | 25 GB | 12.5 GB |
| 175B | 87.5 GB | 43.8 GB | 21.9 GB |

Even an ideal 1-bit 100B model is already around the full VRAM size before scales, metadata, KV cache, activations, CUDA buffers, and fragmentation. A real 100B model therefore cannot simply sit in 12 GB VRAM.

At a target of `7 tok/s`, each token has about `143 ms`. If the runtime must stream tens of GB across PCIe for every token, the target is impossible regardless of clever scheduling. The active bytes touched per token must be kept in the low single-digit GB range, and preferably far below that.

## Prior Art Summary

Relevant systems and papers:

| Area | Prior art | Useful lesson |
| --- | --- | --- |
| Offloading | FlexGen | GPU, CPU, and disk can be scheduled together, but high throughput usually relies on batching and latency tolerance. |
| Hot/cold neurons | PowerInfer | LLM activation has locality; hot neurons can stay on GPU while colder work is handled elsewhere. |
| Contextual sparsity | Deja Vu | Input-dependent head/MLP subsets can approximate dense execution and give wall-clock speedups. |
| Early exit/self-speculation | LayerSkip | Draft and verify can share compute when the same model supports early exits. |
| Speculative decoding | Leviathan et al., BiLD, EAGLE-style work | A small model can draft multiple tokens and a larger model verifies fewer expensive passes. |
| KV compression | H2O, SnapKV, KIVI, InfiniGen, StreamingLLM, TurboQuant | KV cache is a first-class memory object and should be compressed, paged, and predicted. |
| Low-bit weights | GPTQ, AWQ, SmoothQuant, HQQ, QuIP-style methods | Full-fit or near-full-fit depends on quantization quality, not just bit count. |
| Minimal kernels | Karpathy `llm.c` and `llama2.c` | Keep the experimental runtime small, measurable, and kernel-first instead of hiding behavior in a large framework. |

### Recent Literature Check, 2026-06-27

Recent systems make the novelty boundary sharper:

| Area | Recent reference | Implication for SAGE |
| --- | --- | --- |
| Consumer/device over-capacity inference | [PowerInfer-2](https://arxiv.org/abs/2406.06282) | Fine-grained neuron/cluster scheduling and I/O pipelining are already known to matter. SAGE should not claim that basic hot/cold scheduling is new. |
| Sparse speculative verification | [Dustin](https://arxiv.org/html/2606.24957v1), [SSV](https://arxiv.org/html/2605.19893v1), [sparse verification during speculative decoding](https://arxiv.org/abs/2512.21911) | Sparse verification is now an active research direction. SAGE's defensible novelty must be consumer-GPU active-byte budgeting and GGUF-block oracle execution, not sparse verification alone. |
| Self-speculation and early exit | [LayerSkip](https://arxiv.org/abs/2404.16710), [CLaSp](https://aclanthology.org/2025.acl-long.1525.pdf) | If we can avoid a second model by deriving a draft from the same model family, that may reduce memory pressure, but it usually assumes trained or structurally compatible models. |
| KV compression/offload | [KIVI](https://arxiv.org/abs/2402.02750), [InfiniGen](https://arxiv.org/abs/2406.19707), [TurboQuant](https://openreview.net/forum?id=tO3ASKZlok) | SAGE should treat KV as a schedulable ledger, not an afterthought. The giant oracle can use compressed or selective KV while the proxy keeps the normal interactive KV. |
| Minimal implementation style | [karpathy/llm.c](https://github.com/karpathy/llm.c), [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | The prototype should stay inspectable: small kernels, explicit byte budgets, transparent acceptance rules, and reproducible local traces. |

Updated novelty claim:

```text
SAGE is not "we invented sparse verification."
SAGE is a consumer-GPU runtime policy that treats a 100B+ model as a
byte-budgeted, sparsely addressable oracle over a resident proxy stream,
with exact fallback and local hardware measurements deciding when the
giant model is touched.
```

## Proposed Architecture: SAGE-100

SAGE means **Sparse Assisted Giant Execution**.

The core idea:

```text
Use a small resident model for most token steps.
Use the 100B+ model as a compressed sparse oracle.
Invoke the oracle only when the resident model is uncertain or when quality policy requires it.
When the oracle runs, execute an active subset of the giant, not the full dense model.
```

This is not exact dense 100B inference by default. It is a quality-controlled approximation with an exact slow path.

## SAGE-100 Components

### 1. Resident Proxy Model

A smaller model lives mostly or fully on the GPU:

- 7B to 12B for speed-first mode;
- 31B low-bit for quality-first mode if it fits well enough;
- full tokenizer compatibility preferred;
- always owns the interactive token stream.

The proxy produces:

- next-token candidates;
- confidence signals;
- hidden-state summaries;
- predicted active blocks for the giant model.

### 2. Compressed Giant Oracle

The 100B+ model is stored as independently addressable blocks:

- layer blocks;
- attention head blocks;
- FFN channel groups;
- output/logit correction blocks;
- quantized at mixed 1-bit, 2-bit, 3-bit, and 4-bit precision.

The oracle is not loaded as one monolithic model. It is an indexed memory object with hot, warm, and cold tiers.

```text
Hot:  GPU-resident boundary layers, high-impact heads, hot FFN groups
Warm: CPU RAM-resident quantized blocks
Cold: NVMe-resident rare blocks and full exact fallback data
```

### 3. Active-Byte Scheduler

Every token step receives a byte budget:

```text
target_ms_per_token = 1000 / target_tok_per_second
transfer_budget_ms  = target_ms_per_token * transfer_fraction
active_bytes_budget = measured_pcie_GBps * transfer_budget_ms
```

The scheduler decides whether a token can afford:

- no oracle;
- sparse oracle;
- denser oracle;
- full slow fallback.

This is the main systems idea. Instead of only deciding `-ngl`, the runtime controls how many giant-model bytes are allowed to participate in the current decision.

### 4. Uncertainty-Gated Oracle Calls

The proxy does not call the 100B oracle for every token.

Oracle call triggers:

- low top-1 probability margin;
- high entropy;
- disagreement between two cheap draft heads;
- user-selected high-accuracy mode;
- tool-call/code/math regions;
- long-range retrieval or summarization boundaries.

If the proxy is confident, the token is accepted immediately. If it is uncertain, the oracle verifies a small candidate set or a block of drafted tokens.

### 5. Sparse Oracle Verification

When the oracle is invoked, it should avoid dense execution where possible:

- use a predictor to choose active attention heads and FFN groups;
- keep globally hot neurons on GPU;
- prefetch predicted warm blocks into a pair of GPU staging buffers;
- verify only top-k candidate tokens first;
- escalate to denser execution only when sparse verification is inconclusive.

This combines the lessons of PowerInfer and Deja Vu with a strict local byte budget.

### 6. Compressed KV Ledger

The proxy keeps a normal KV cache. The giant oracle keeps a compressed, sparse KV ledger:

- recent oracle-verified tokens in GPU or CPU RAM;
- older oracle KV compressed with KIVI/TurboQuant-style methods;
- attention sinks and selected heavy hitters retained;
- optional reconstruction or rehearsal for long-context oracle calls.

The giant does not need full-precision KV for every proxy-accepted token unless exact mode is requested.

### 7. Exact Slow Path

The system must expose an honest mode:

```text
exact = run the dense 100B path, slow but faithful
sage  = run sparse/proxy/oracle path, fast but approximate
```

This protects the project from overclaiming. Fast 100B-on-consumer-GPU means approximate unless future hardware or model structure changes the memory equation.

## Why This Could Be Novel

The individual techniques are not new. The possible novelty is the combination and control objective:

```text
A consumer-GPU runtime that treats a 100B+ model as an active-byte-budgeted
oracle rather than a dense model, using proxy confidence, sparse block routing,
compressed KV, and live migration to decide how much of the giant model is
allowed to participate in each token.
```

Existing systems typically focus on one axis:

- offload dense tensors;
- quantize weights;
- compress KV;
- speculate with a draft;
- exploit sparse activations.

SAGE-100 would make those decisions together under a measured per-token byte and latency budget.

## Feasibility Gates

### Gate 1: Proxy-Oracle Agreement

Use local 12B and 31B models as a scale-down proxy:

```text
proxy:  Gemma 12B
oracle: Gemma 31B
goal:   estimate how often the proxy can safely skip oracle verification
```

Measure:

- next-token agreement;
- top-k overlap;
- entropy threshold vs correctness;
- quality loss when skipping oracle calls;
- percentage of tokens requiring oracle.

If the proxy needs oracle verification on most tokens, the 100B design will not hit the speed target.

Initial harness:

```powershell
python .\scripts\sage_agreement.py `
  --proxy-model .\models\gemma-4-12b-it-qat-q4_0-gguf\gemma-4-12b-it-qat-q4_0.gguf `
  --oracle-model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --proxy-ngl all `
  --oracle-ngl 38 `
  --tokens 1 `
  --limit 5
```

For instruction-style output, use chat mode:

```powershell
python .\scripts\sage_agreement.py `
  --mode chat `
  --tokens 8 `
  --limit 5 `
  --proxy-ngl all `
  --oracle-ngl 38 `
  --tag gemma12b-to-31b-chat
```

The harness uses `llama-completion.exe` for raw non-interactive completion and `llama-cli.exe --single-turn` for chat mode. It writes child logs under `benchmarks/sage-agreement-logs/` so stalled or failed model launches are diagnosable.

Measured local result on 2026-06-26:

```text
raw mode, 1 token, 5 prompts:
  Gemma 12B Q4 proxy -> Gemma 31B Q4 oracle
  exact agreement:      0/5
  normalized agreement: 0/5

chat mode, 8 tokens, 5 prompts:
  Gemma 12B Q4 proxy -> Gemma 31B Q4 oracle
  exact agreement:      1/5
  normalized agreement: 1/5
```

This falsifies the simplest version of SAGE where a smaller local proxy is trusted directly for most tokens. A viable design needs at least one additional mechanism: a learned agreement router, uncertainty gates, token-class policies, or sparse oracle checks before skipping the giant model.

This is still a coarse skip-rate signal, not a replacement for a logits/top-k C++ harness.

Logprob/top-k probe:

```powershell
python .\scripts\sage_logprob_probe.py `
  --mode chat `
  --tokens 8 `
  --top-k 10 `
  --limit 5 `
  --ignore-prefix-steps 3 `
  --proxy-ngl all `
  --oracle-ngl 38 `
  --tag gemma12b-to-31b-chat8-filtered
```

Larger calibration run:

```powershell
python .\scripts\sage_logprob_probe.py `
  --prompts .\prompts\sage-calibration.txt `
  --mode chat `
  --tokens 8 `
  --top-k 10 `
  --limit 30 `
  --ignore-prefix-steps 3 `
  --proxy-ngl all `
  --oracle-ngl 38 `
  --tag gemma12b-to-31b-calib30-chat8-filtered
```

Measured local result on 2026-06-26:

```text
raw mode, 1 token, 5 prompts:
  top-1 token agreement:      0/5
  mean top-k token Jaccard:   0.068
  mean proxy margin:          0.883

chat mode, 8 tokens, 5 prompts:
  unfiltered top-1 agreement: 27/40 = 67.5%
  filtered top-1 agreement:   12/25 = 48.0%
  filtered top-k Jaccard:     0.274
  filtered proxy margin:      6.868

chat mode, 8 tokens, 30 prompts:
  filtered top-1 agreement:   85/150 = 56.7%
  filtered top-k Jaccard:     0.233
  filtered proxy margin:      6.677
```

The unfiltered chat score is inflated by deterministic Gemma preamble tokens (`<|channel>`, `thought`, newline). The filtered score is more useful: the proxy and oracle often share the response format, but content diverges quickly on some prompts. That means a router should not only look at top-1 agreement; it needs margin, entropy, token class, and recent divergence features.

Router feasibility tool:

```powershell
python .\scripts\sage_router_eval.py `
  --agreement-json .\benchmarks\20260626-204548-sage-agreement-gemma12b-to-31b-chat.json `
  --target-tps 7 `
  --proxy-tps 25 `
  --params-b 100 `
  --quant-bpw 2 `
  --active-percent 10 `
  --pcie-gbps 24
```

Measured implication from the same 5-prompt chat sample:

```text
Active oracle per call:               2.33 GiB (10.0% of a 100B 2-bit model)
Oracle transfer per call:           104.2 ms
Oracle total per call:              119.2 ms
Max oracle call rate for 7 tok/s:    86.3%
Required safe skip rate:             13.7%
Observed perfect-router upper bound: 20.0%
Observed perfect-router speed:        7.39 tok/s
Trust-proxy-all error rate:           80.0%
```

This is a useful but fragile result: with a perfect router and only `10%` active giant weights per oracle call, the small sample is not ruled out. At `15%` active weights, the same evidence is ruled out (`5.65 tok/s` upper bound), and at `20%` active weights it is clearly too slow (`4.57 tok/s` upper bound). Therefore the next prototype must optimize both router precision and active-byte selection; improving only one side is not enough.

Proxy-only router fit:

```powershell
python .\scripts\sage_router_fit.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --max-false-accept-rate 0.10
```

Measured result on the 30-prompt calibration set:

```text
Records after prefix filtering:       150
Top-1 token matches:                  85/150 = 56.7%
Token-class match rates:
  whitespace:                         10/11 = 90.9%
  punctuation:                        28/40 = 70.0%
  capitalized:                        20/40 = 50.0%
  word:                               27/59 = 45.8%

With <=5% accepted-token error:
  best rule: whitespace with high margin
  skip rate:                          6.7%
  estimated speed:                    6.61 tok/s
  target reached:                     no

With <=10% accepted-token error:
  best rule: whitespace
  skip rate:                          7.3%
  estimated speed:                    6.65 tok/s
  target reached:                     no

With <=20% accepted-token error:
  best rule: punctuation with margin >= 0.644
  skip rate:                          21.3%
  accepted-token error:               18.8%
  estimated speed:                    7.48 tok/s
  target reached:                     yes, but quality risk is too high
```

Decision: proxy-only routing is not enough for a useful SAGE mode. It can hit the speed budget only by accepting too many wrong tokens. The next prototype should use the proxy router as a prefilter and add sparse oracle verification before a token is accepted.

Two-stage sparse-verifier target:

```powershell
python .\scripts\sage_policy_target.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-class punct `
  --margin-threshold 0.644 `
  --verifier-active-percent 1 `
  --oracle-active-percent 10 `
  --max-accepted-error-rate 0.05 `
  --max-total-error-rate 0.02
```

Measured target from the 30-prompt calibration set:

```text
Candidate rule:
  proxy token class:                  punctuation
  proxy margin >=                     0.644
  candidates:                         32/150 = 21.3%
  candidate good/bad:                 26/6
  candidate error before verifier:    18.8%

Assumptions:
  proxy speed:                        25 tok/s
  sparse verifier active model:        1% of 100B 2-bit
  oracle active model path:           10% of 100B 2-bit
  accepted-proxy error limit:          5%
  total error limit:                   2%

Passing target:
  verifier catches bad candidates:    >=80%
  verifier false-rejects good ones:   <=5%
  verifier call rate:                 21.3%
  oracle call rate:                   ~82-83%
  accepted proxy rate:                ~17-18%
  estimated speed:                    7.00-7.08 tok/s
```

If the verifier touches `2%` of the 100B 2-bit model while the oracle path remains at `10%`, the same policy misses the `7 tok/s` target by a small margin. If the oracle path can be reduced to `8%`, then a `2%` verifier has more room (`~7.9 tok/s` under the same catch-rate assumptions). This gives the sparse-verifier implementation a hard byte budget: it must be very small, closer to `0.5-1.0%` active giant weights unless the oracle fallback path also gets cheaper.

Verifier block-budget planner:

```powershell
python .\scripts\sage_verifier_plan.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --active-percents 0.5,1,2,4
```

Concrete verifier manifest generator:

```powershell
python .\scripts\sage_verifier_manifest.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --policy ffn-sentinel `
  --active-percent 1

python .\scripts\sage_verifier_manifest.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --policy attention-sentinel `
  --layer-order early `
  --active-percent 1

python .\scripts\sage_verifier_manifest.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --policy hybrid `
  --active-percent 2
```

Local Gemma 31B Q4 mapping for a 100B 2-bit reference:

```text
Minimum building blocks:
  one FFN group:                 0.182 GiB = 0.78% of 100B 2-bit
  one attention group:           0.069 GiB = 0.30% of 100B 2-bit
  three attention groups:        0.208 GiB = 0.89% of 100B 2-bit
  one FFN + one attention group: 0.251 GiB = 1.08% of 100B 2-bit
  one full layer:                0.251 GiB = 1.08% of 100B 2-bit

Budget fits:
  0.5% active: 0 FFN groups, 1 attention group, 0 full layers
  1.0% active: 1 FFN group, 3 attention groups, 0 full layers
  2.0% active: 2 FFN groups, 6 attention groups, 1 full layer
  4.0% active: 5 FFN groups, 13 attention groups, 3 full layers
```

Concrete manifest results:

```text
1% ffn-sentinel, boundary order:
  selected: blk.0.norm + blk.0.ffn
  used:     0.18 GiB

1% attention-sentinel, early order:
  selected: blk.0/1/2 norms + blk.0/1/2 attention
  used:     0.21 GiB

1% attention-sentinel, boundary order:
  selected: blk.0/59 norms + blk.0/59 attention
  used:     0.17 GiB
  note: late Gemma4 attention blocks are larger, so only two boundary attention sentinels fit.

2% hybrid, boundary order:
  selected: blk.0.ffn + blk.0/59/1 attention plus norms
  used:     0.42 GiB
```

The manifest also emits llama.cpp debug tensor filters for the selected layers.
For example, the `2%` hybrid manifest emits a single regex like:

```text
--tensor-filter "(attn_norm-0|attn_out-0|ffn_norm-0|ffn_norm_1-0|ffn_norm_2-0|ffn_out-0|ffn_mlp-0|ffn_moe-0|l_out-0|attn_norm-1|attn_out-1|l_out-1|attn_norm-59|attn_out-59|l_out-59)$"
```

llama.cpp patch:

```text
patches/llama.cpp/0002-filter-debug-eval-callback.patch
patches/llama.cpp/0003-debug-parse-special-prompt.patch
```

This patch changes the common debug eval callback so `--tensor-filter` rejects non-matching graph nodes during the scheduler `ask` phase. That matters because the stock callback can still request every tensor even when output printing is filtered.

The second debug patch makes `llama-debug` parse special/control tokens in prompts. This is required for chat-aligned sparse capture because rendered Gemma4 prompts contain control tokens such as `<|turn>`, `<|think|>`, `<|channel>`, and `<channel|>`.

Smoke validation on a small local Qwen model:

```powershell
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;' + $env:PATH
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-debug.exe `
  -m .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -p "hello" `
  -ngl 0 `
  -c 128 `
  -b 8 `
  -ub 8 `
  --tensor-filter "(ffn_out-0|l_out-0)$" `
  --verbose
```

Result: the patched callback completed and dumped only the targeted `ffn_out-0` and `l_out-0` graph nodes. This proves the low-overhead probe path works before attempting the heavier Gemma 31B verifier capture.

Structured capture harness:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --batch-size 128 `
  --ubatch-size 8 `
  --prompts .\prompts\sage-calibration.txt `
  --limit 5 `
  --tag gemma31b-hybrid2
```

The harness builds or reads a verifier manifest, runs patched `llama-debug`, parses matched tensor callback summaries, and writes JSON rows containing tensor name, dtype, op, shape, and sum. A smoke run on the local Qwen 0.5B model captured two summaries:

```json
[
  {"name": "ffn_out-0", "shape": [896, 1, 1, 1], "sum": 3.801703},
  {"name": "l_out-0", "shape": [896, 1, 1, 1], "sum": 3.817294}
]
```

Gemma 31B hybrid verifier smoke result:

```text
Command:
  python .\scripts\sage_probe_capture.py --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf --policy hybrid --active-percent 2 --ngl 38 --ctx-size 256 --batch-size 128 --ubatch-size 8 --prompt "The capital of France is" --tag gemma31b-hybrid2-smoke --timeout 1800

Result:
  elapsed:       8.76 sec
  tensor count:  11
  nodes:
    attn_norm-0  [5376, 6, 1, 1]  sum -3837.109375
    attn_out-0   [5376, 6, 1, 1]  sum  3934.240967
    ffn_norm-0   [5376, 6, 1, 1]  sum  -157.101517
    ffn_out-0    [5376, 6, 1, 1]  sum  -227.485428
    l_out-0      [5376, 6, 1, 1]  sum   385.453003
    attn_norm-1  [5376, 6, 1, 1]  sum  2549.001465
    attn_out-1   [5376, 6, 1, 1]  sum   932.114319
    l_out-1      [5376, 6, 1, 1]  sum   202.259109
    attn_norm-59 [5376, 6, 1, 1]  sum   326.535889
    attn_out-59  [5376, 6, 1, 1]  sum  -455.203400
    l_out-59     [5376, 6, 1, 1]  sum  -117.137016
```

This is not a trained verifier yet. It is the measurement spine needed to build one: collect sparse sentinel signals, join them with proxy/oracle agreement labels, then test whether a cheap verifier can catch bad proxy candidates at the required `80%+` bad-candidate catch rate.

First sparse-probe label join:

```powershell
python .\scripts\sage_probe_fit.py `
  --logprob-json .\benchmarks\20260626-210118-sage-logprob-gemma12b-to-31b.json `
  --probe-json .\benchmarks\20260626-214358-sage-probe-gemma31b-hybrid2-raw5.json `
  --step-index 0
```

Result:

```text
records:           5
matches:           0
mismatches:        5
mean margin:       0.883
mean tensor count: 17.6
features:          36

Decision:
  The raw-prompt label set is not viable for verifier fitting because it has
  only one class. This is still useful: it prevents overclaiming and shows that
  the next verifier dataset must use chat-aligned capture or candidate-token
  capture, where both good and bad proxy decisions exist.
```

Chat-aligned sparse probe:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --prompts .\prompts\sage-calibration.txt `
  --limit 5 `
  --tag gemma31b-hybrid2-chat5-step3 `
  --timeout 1800

python .\scripts\sage_probe_fit.py `
  --logprob-json .\benchmarks\20260626-210528-sage-logprob-gemma12b-to-31b-chat8-filtered.json `
  --probe-json .\benchmarks\20260626-215513-sage-probe-gemma31b-hybrid2-chat5-step3.json `
  --step-index 3
```

Result:

```text
records:           5
matches:           3
mismatches:        2
mean margin:       1.020
mean tensor count: 39.6
features:          36

Best tiny-sample rules:
  attn_out_0.mean >= 0.135847
  attn_out_0.per_token_sum >= 730.314
  ffn_norm_0.mean >= -0.0120904
  l_out_0.mean >= 0.0147752

Each of those rules accepted 3/5 proxy candidates with:
  accepted error: 0.0%
  bad catch rate: 100.0%
  good reject rate: 0.0%
```

Interpretation: this is the first positive sparse-signal result, but it is only `n=5`. It shows the SAGE verifier path is worth scaling to the 30-prompt calibration set; it does not yet prove the verifier target.

30-prompt chat-aligned scale-up:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --prompts .\prompts\sage-calibration.txt `
  --offset 0 `
  --limit 10 `
  --tag gemma31b-hybrid2-chat30-step3-00 `
  --timeout 1800

python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --prompts .\prompts\sage-calibration.txt `
  --offset 10 `
  --limit 10 `
  --tag gemma31b-hybrid2-chat30-step3-10 `
  --timeout 1800

python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --prompts .\prompts\sage-calibration.txt `
  --offset 20 `
  --limit 10 `
  --tag gemma31b-hybrid2-chat30-step3-20 `
  --timeout 1800

python .\scripts\sage_probe_fit.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --probe-json .\benchmarks\20260626-215903-sage-probe-gemma31b-hybrid2-chat30-step3-00.json .\benchmarks\20260626-220101-sage-probe-gemma31b-hybrid2-chat30-step3-10.json .\benchmarks\20260626-220255-sage-probe-gemma31b-hybrid2-chat30-step3-20.json `
  --step-index 3 `
  --max-accepted-error 0.05
```

Measured result:

```text
records:           30
matches:           20
mismatches:        10
match rate:        66.7%
mean margin:       1.335
mean tensor count: 41.4
features:          36

Best strict one-feature rule:
  feature:          l_out_59.sum >= 32.517593
  accepted:         6/30 = 20.0%
  accepted error:   0.0%
  bad catch rate:   100.0%
  good reject rate: 70.0%

Best looser rule under 20% accepted-token error:
  feature:          l_out_59.sum <= 17.690586
  accepted:         12/30 = 40.0%
  accepted error:   16.7%
  bad catch rate:   80.0%
  good reject rate: 50.0%
```

Interpretation: the sparse sentinel signal survived `n=30`, so the verifier path is still viable. However, a one-feature threshold is too conservative at low accepted-token error: it safely skips only `20%` of this chat-aligned step. The 7 tok/s target needs roughly `17-18%` accepted proxy rate only under optimistic `1%` verifier and `10%` oracle active-byte assumptions, so there is not enough margin yet. The next verifier must use multiple features and should be trained/evaluated on router-candidate tokens, not just one fixed step per prompt.

Router-candidate task generation:

```powershell
python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-class punct `
  --margin-threshold 0.644 `
  --tag punct-margin0644-all

python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-class punct `
  --margin-threshold 0.644 `
  --require-matching-prefix `
  --tag punct-margin0644-same-prefix
```

Task-builder result:

```text
all continuation labels:
  candidates:                  32/150 = 21.3%
  top-1 token matches:          26/32 = 81.3%
  same-prefix candidates:       21/32
  same-prefix matches:          20/21 = 95.2%

strict same-prefix labels only:
  candidates:                  21
  top-1 token matches:          20/21 = 95.2%
```

The distinction matters. Continuation labels after proxy/oracle divergence are useful for exploration, but they are not strong proof because the oracle is answering from its own previous tokens. Same-prefix labels are defensible but currently too imbalanced: only one bad candidate remains.

Corrected candidate-token sparse capture:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260626-221454-sage-candidate-tasks-punct-margin0644-all.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --limit 12 `
  --tag candidate-punct-margin0644-fixed-00 `
  --timeout 1800

python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260626-221454-sage-candidate-tasks-punct-margin0644-all.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --offset 12 `
  --limit 20 `
  --tag candidate-punct-margin0644-fixed-12 `
  --timeout 1800

python .\scripts\sage_candidate_probe_fit.py `
  --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json `
  --label-quality any `
  --max-accepted-error 0.05
```

Important correction: candidate capture must render the prompt as:

```text
Gemma4 chat template + <|channel>thought\n + previous proxy text
```

An earlier exploratory candidate capture inserted previous proxy text before the thought-channel prefix; that capture is not used for the corrected result.

Corrected candidate-probe result:

```text
strict same-prefix fit:
  records:           21
  matches/mismatches: 20/1
  best rule:         attn_norm_0.mean >= -0.174052
  accepted:          21/21 = 100%
  accepted error:    4.8%
  bad catch rate:    0.0%
  note:              label set is too imbalanced for verifier claims.

exploratory all-label fit:
  records:           32
  matches/mismatches: 26/6
  best 5%-error rule:
    (ffn_out_0.mean >= 0.0246853) OR
    (proxy.entropy >= 0.732812)
  accepted:          17/32 = 53.1%
  accepted error:    0.0%
  bad catch rate:    100.0%
  good reject rate:  34.6%
```

The fitter excludes oracle-overlap features such as proxy/oracle top-k Jaccard. Those are useful diagnostics but would leak information that a real runtime does not have before deciding whether to call the fallback oracle.

Oracle replay on proxy prefixes:

```powershell
python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-class punct `
  --margin-threshold 0.644 `
  --tag punct-margin0644-all-v2

python .\scripts\sage_candidate_replay.py `
  --tasks-json .\benchmarks\20260626-223454-sage-candidate-tasks-punct-margin0644-all-v2.json `
  --oracle-ngl 38 `
  --top-k 10 `
  --tag punct-margin0644-replay32

python .\scripts\sage_candidate_probe_fit.py `
  --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json `
  --replay-json .\benchmarks\20260626-223617-sage-candidate-replay-punct-margin0644-replay32.json `
  --label-quality any `
  --max-accepted-error 0.05
```

Replay result:

```text
tasks:                         32
replay top-1 matches:          30/32 = 93.8%
original continuation matches: 26/32 = 81.3%
label changes:                  4
proxy token in oracle top-k:    32/32
same-prefix replay matches:     20/21
diverged-prefix replay matches: 10/11
```

The replay run removes the main weakness in the previous candidate-label set: diverged-prefix candidates are now judged by the oracle from the proxy prefix. The punctuation/margin candidate router is much more viable under this stricter measurement than it looked under independent-continuation labels.

Replay-labeled sparse verifier fit:

```text
records:            32
matches/mismatches: 30/2
best 5%-error rule:
  attn_norm_0.sum >= -5173.35
accepted:           31/32 = 96.9%
accepted error:     3.2%
bad catch rate:     50.0%
good reject rate:   0.0%
```

Prompt-level held-out smoke tests:

```powershell
python .\scripts\sage_candidate_probe_fit.py `
  --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json `
  --replay-json .\benchmarks\20260626-223617-sage-candidate-replay-punct-margin0644-replay32.json `
  --label-quality any `
  --max-accepted-error 0.05 `
  --holdout-prompt-indices 8,9,10,11

python .\scripts\sage_candidate_probe_fit.py `
  --probe-json .\benchmarks\20260626-222320-sage-probe-candidate-punct-margin0644-fixed-00.json .\benchmarks\20260626-222523-sage-probe-candidate-punct-margin0644-fixed-12.json `
  --replay-json .\benchmarks\20260626-223617-sage-candidate-replay-punct-margin0644-replay32.json `
  --label-quality any `
  --max-accepted-error 0.05 `
  --holdout-prompt-indices 24,25,26,27
```

Held-out result:

```text
holdout prompts 8,9,10,11:
  train:            26 records, 25/1 good/bad
  holdout:           6 records,  5/1 good/bad
  best train rule:  attn_norm_1.mean >= 0.045847548363095236
  holdout accepted: 5/6 = 83.3%
  holdout error:    0.0%
  held-out bad catch: 100.0%

holdout prompts 24,25,26,27:
  train:            26 records, 25/1 good/bad
  holdout:           6 records,  5/1 good/bad
  best train rule:  attn_norm_0.sum >= -5173.3544920000004
  holdout accepted: 5/6 = 83.3%
  holdout error:    0.0%
  held-out bad catch: 100.0%
```

This is the first held-out check that keeps both classes in train and holdout. It is still only a smoke test because the held-out sets contain one bad candidate each.

Throughput implication for replay-labeled and held-out behavior:

```powershell
python .\scripts\sage_policy_target.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --stats-total 150 `
  --stats-candidates 32 `
  --stats-candidate-good 30 `
  --stats-candidate-bad 2 `
  --verifier-active-percent 2 `
  --oracle-active-percent 10 `
  --catch-bad-rates 0.5 `
  --false-reject-good-rates 0 `
  --max-accepted-error-rate 0.05 `
  --max-total-error-rate 0.02
```

Measured implication:

```text
all-data replay rule, 2% verifier, 10% oracle path: 7.12 tok/s, passes target
all-data replay rule, 1% verifier, 10% oracle path: 7.23 tok/s, passes target
held-out behavior,     2% verifier, 10% oracle path: 7.08 tok/s, passes target
```

Decision: this is the first local evidence chain that clears the `7 tok/s` budget under the original `10%` oracle fallback assumption:

- proxy/oracle logprob calibration;
- router-candidate extraction;
- oracle replay on the proxy prefix;
- sparse tensor capture at candidate states;
- replay-labeled verifier fit;
- active-byte throughput simulation.

This is not production proof. The dataset is still only `32` candidate tokens from `30` prompts, and the held-out checks contain only one bad candidate per split. The next gate is replay-labeled evaluation with more candidate classes, more prompts, and larger held-out sets.

Broader punctuation+whitespace candidate gate:

```powershell
python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-classes punct,whitespace `
  --margin-threshold 0.644 `
  --tag punct-whitespace-margin0644

python .\scripts\sage_candidate_replay.py `
  --tasks-json .\benchmarks\20260626-224844-sage-candidate-tasks-punct-whitespace-margin0644.json `
  --oracle-ngl 38 `
  --top-k 10 `
  --tag punct-whitespace-margin0644-replay43
```

Sparse capture was run in three chunks with offsets `0`, `15`, and `30`:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260626-224844-sage-candidate-tasks-punct-whitespace-margin0644.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --offset 0 `
  --limit 15 `
  --tag candidate-punct-whitespace-margin0644-00 `
  --timeout 1800
```

Measured replay result:

```text
candidate classes:                  punctuation + whitespace
candidates:                         43/150 = 28.7%
replay-labeled matches:             41/43 = 95.3%
original continuation matches:       36/43 = 83.7%
label changes after replay:          5
same-prefix replay matches:          30/31
diverged-prefix replay matches:      11/12

bad replay-labeled candidate cand-0011:
  prompt:                            The first step in debugging a crash is
  step:                              3
  proxy token:                       "
  oracle token:                      *
  label quality:                     same-prefix

bad replay-labeled candidate cand-0038:
  prompt:                            If disk paging starts during inference, throughput
  step:                              7
  proxy token:                       [space](
  oracle token:                      .
  label quality:                     diverged-prefix
```

All-data sparse-fit result under `<=5%` accepted-token error:

```text
best safe-by-acceptance rule:        attn_norm_0.mean >= -0.17405155729166669
accepted:                            43/43
accepted-token error:                 4.7%
bad-candidate catch:                  0.0%
good-candidate reject:                0.0%
```

This is effectively an accept-all rule for this tiny candidate set. The important observation is not that `attn_norm_0.mean` is meaningful; it is that replay-labeled punctuation+whitespace candidates were already under the `5%` accepted-token error budget in this calibration run.

Policy model implication using `150` total filtered token steps, `43` candidates, `41` good candidates, and `2` bad candidates:

```text
No verifier, 10% oracle fallback:
  accepted proxy rate:                 28.7%
  accepted-token error:                 4.7%
  total token error:                    1.3%
  oracle call rate:                    71.3%
  estimated speed:                      8.00 tok/s
  gate result:                          passes current numeric limits

2% verifier, catches no extra bad candidates:
  estimated speed:                      7.52 tok/s
  gate result:                          passes, but wastes verifier cost

2% verifier, catches both bad candidates:
  accepted proxy rate:                 27.3%
  accepted-token error:                 0.0%
  total token error:                    0.0%
  oracle call rate:                    72.7%
  estimated speed:                      7.43 tok/s
  gate result:                          passes with stricter quality
```

Prompt-level held-out smoke tests:

```text
Holdout prompts 8,9,10,11:
  train labels:                         35/36 matches
  holdout labels:                        6/7 matches
  best heldout rule:                     attn_norm_0.mean >= -0.17245699111793153
  accepted heldout:                      6/7
  accepted-token error:                  0.0%
  bad-candidate catch:                   100.0%
  good-candidate reject:                 0.0%

Holdout prompts 24,25,26,27:
  train labels:                         34/35 matches
  holdout labels:                        7/8 matches
  best heldout rule:                     attn_norm_0.sum >= -5433.8520509999998
  accepted heldout:                      7/8
  accepted-token error:                  0.0%
  bad-candidate catch:                   100.0%
  good-candidate reject:                 0.0%
```

Decision: punctuation+whitespace is a promising low-risk proxy-acceptance policy, but the sample is still too small and too easy. The next question is whether adding harder token classes can raise skip rate without requiring an unrealistically strong verifier.

Harder token-class expansion on the same 30-prompt calibration set:

```powershell
python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\20260626-211233-sage-logprob-gemma12b-to-31b-calib30-chat8-filtered.json `
  --ignore-prefix-steps 3 `
  --candidate-classes punct,whitespace,number,capitalized `
  --margin-threshold 0.644 `
  --tag calib30-punct-whitespace-number-capitalized

python .\scripts\sage_candidate_replay.py `
  --tasks-json .\benchmarks\20260627-092722-sage-candidate-tasks-calib30-punct-whitespace-number-capitalized.json `
  --oracle-ngl 38 `
  --top-k 10 `
  --offset 0 `
  --limit 25 `
  --tag calib30-pwnc-replay-o000-l025
```

Replay was run in three chunks with offsets `0`, `25`, and `50`.

Measured replay result:

```text
candidate classes:                  punctuation + whitespace + capitalized
candidates:                         75/150 = 50.0%
original continuation matches:       55/75 = 73.3%
replay-prefix matches:               69/75 = 92.0%
label changes after replay:          14
proxy token in oracle top-k:          75/75

Replay match rate by class:
  whitespace:                         11/11 = 100.0%
  punctuation:                        30/32 = 93.8%
  capitalized:                        28/32 = 87.5%
```

The six replay-labeled bad candidates were two punctuation decisions and four capitalized decisions. No whitespace candidate failed under replay labels.

Policy model implication using `150` total filtered token steps, `75` candidates, `69` good candidates, and `6` bad candidates:

```text
No verifier:
  accepted proxy rate:                 50.0%
  accepted-token error:                 8.0%
  total token error:                    4.0%
  gate result:                          fails quality limits

1% verifier, catches 50% of bad candidates:
  accepted proxy rate:                 48.0%
  accepted-token error:                 4.2%
  total token error:                    2.0%
  oracle call rate:                    52.0%
  estimated speed:                      9.04 tok/s
  gate result:                          passes current numeric limits

2% verifier, catches 50% of bad candidates:
  estimated speed:                      8.63 tok/s
  gate result:                          passes current numeric limits
```

Decision: this is now the sharper SAGE target. A capitalized-token policy can create enough skip rate for the `7-10 tok/s` band, but only if a very small sparse verifier catches at least half of the remaining bad candidates with low false rejection. This is still not production proof because it uses only the same `30` prompts; it is a concrete target for sparse capture and the hard-120 gate.

Next hard-prompt gate:

```powershell
python .\scripts\sage_logprob_probe.py `
  --prompts .\prompts\sage-calibration-hard-120.txt `
  --mode chat `
  --tokens 8 `
  --top-k 10 `
  --offset 0 `
  --limit 40 `
  --ignore-prefix-steps 3 `
  --proxy-ngl all `
  --oracle-ngl 38 `
  --tag gemma12b-to-31b-hard120-chat8-filtered-o000-l040

# Repeat with --offset 40 and --offset 80, then pass all three
# logprob JSON files to sage_candidate_tasks.py.

python .\scripts\sage_candidate_tasks.py `
  --logprob-json .\benchmarks\<hard120-o000-json> .\benchmarks\<hard120-o040-json> .\benchmarks\<hard120-o080-json> `
  --ignore-prefix-steps 3 `
  --candidate-classes punct,whitespace,number,capitalized `
  --margin-threshold 0.644 `
  --tag hard120-punct-whitespace-number-capitalized
```

Gate condition: the broader policy is still interesting only if replay-labeled candidates stay below `5%` accepted-token error or if a sparse verifier recovers that target while keeping the policy above `7 tok/s`. If the hard prompt set raises accepted-token error sharply, the design must move back toward sparse verification or narrower token classes.

Full hard-120 router result:

```text
Source:
  prompts:                             sage-calibration-hard-120.txt
  chunks:                              offset 0, 40, 80; limit 40 each
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      600
  top-1 token matches:                 332/600 = 55.3%
  mean top-k token Jaccard:             0.220
  mean proxy margin:                    7.289
```

Candidate extraction with `punct,whitespace,number,capitalized` and margin `>=0.644`:

```text
candidates:                            295/600 = 49.2%
original continuation matches:         200/295 = 67.8%
replay-prefix matches:                 266/295 = 90.2%
label changes after replay:             70
proxy token in oracle top-k:            295/295

Replay match rate by chunk:
  offset 0:                              89/100 = 89.0%
  offset 40:                             87/94 = 92.6%
  offset 80:                             90/101 = 89.1%

Replay match rate by class:
  whitespace:                            36/36 = 100.0%
  number:                                 3/3 = 100.0%
  punctuation:                          106/124 = 85.5%
  capitalized:                          121/132 = 91.7%

Replay match rate by prefix status:
  same-prefix:                          181/207 = 87.4%
  diverged-prefix:                       85/88 = 96.6%
```

Full hard-120 policy implication before sparse verification:

```text
accepted proxy rate:                    49.2%
accepted-token error:                    9.8%
total token error:                       4.8%
gate result:                             fails quality limits
```

Required verifier behavior for the full hard-120 router:

```text
1% verifier, 10% oracle fallback:
  verifier call rate:                   49.2%
  required bad-candidate catch:          about 59% minimum for total-error <=2%
  tested catch rate 66%:                 8.85 tok/s, passes
  tested catch rate 75%:                 8.81 tok/s, passes

2% verifier, 10% oracle fallback:
  tested catch rate 66%:                 8.47 tok/s, passes
  tested catch rate 75%:                 8.43 tok/s, passes
```

Cross-chunk verifier/proxy-gate fit after capturing all `295` hard-120 candidates:

```text
records:                                295
replay-labeled good/bad:                266/29
best <=5% accepted-error rule:
  (proxy.entropy <= 0.79383919530270064)
  OR (proxy.margin >= 1.9019756019115448)

accepted:                               262/295
accepted-token error:                    4.6%
bad-candidate catch:                    58.6% = 17/29
good-candidate reject:                   6.0% = 16/266
```

This best full-hard rule uses only proxy-side features, not sparse tensor features. That is important: the current strongest deployable gate is a two-stage proxy confidence gate. Sparse tensor capture remains useful as a possible quality-margin mechanism, but the full hard-120 evidence does not yet prove it is necessary.

Policy implication with the full-hard proxy gate:

```text
Zero-cost proxy gate, 10% oracle fallback:
  accepted proxy rate:                  43.7%
  accepted-token error:                  4.6%
  total token error:                     2.0%
  oracle call rate:                     56.3%
  estimated speed:                       9.33 tok/s
  gate result:                           passes current numeric limits

1% verifier cost charged anyway:
  estimated speed:                       8.64 tok/s
  gate result:                           passes current numeric limits

2% verifier cost charged anyway:
  estimated speed:                       8.28 tok/s
  gate result:                           passes current numeric limits
```

Frozen gate evaluator:

```powershell
python .\scripts\sage_gate_eval.py `
  --replay-json .\benchmarks\<validation-replay-json> `
  --max-entropy 0.79383919530270064 `
  --min-margin 1.9019756019115448 `
  --stats-total <validation-filtered-step-count>
```

The evaluator applies the frozen rule without fitting new thresholds. It should be used on fresh validation prompts before changing the rule again.

Modulo-4 held-out checks for the full-hard fit:

```text
holdout r0:
  holdout labels:                        77/84 good
  accepted:                              76/84
  accepted-token error:                   3.9%
  bad-candidate catch:                   57.1%

holdout r1:
  holdout labels:                        61/71 good
  accepted:                              60/71
  accepted-token error:                   5.0%
  bad-candidate catch:                   70.0%

holdout r2:
  holdout labels:                        67/72 good
  accepted:                              62/72
  accepted-token error:                   3.2%
  bad-candidate catch:                   60.0%

holdout r3:
  holdout labels:                        61/68 good
  accepted:                              62/68
  accepted-token error:                   4.8%
  bad-candidate catch:                   57.1%
```

Interpretation before fresh validation: the hard-120 set supported a deployable proxy-side SAGE gate at the edge of the current `2%` total-error budget, with modeled speed inside the `7-10 tok/s` target band. Because the margins were tight and the rule was selected from calibration data, the rule had to be tested on `prompts/sage-validation-80.txt` before being treated as robust.

Fresh validation result:

```text
Source:
  prompts:                             sage-validation-80.txt
  chunks:                              offset 0, 40; limit 40 each
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      400
  top-1 token matches:                 248/400 = 62.0%
  mean top-k token Jaccard:             0.218
  mean proxy margin:                    7.905
```

Candidate extraction with the frozen class prefilter:

```text
candidate classes:                     punct, whitespace, number, capitalized
margin prefilter:                      >= 0.644
candidates:                            201/400 = 50.3%
original continuation matches:         141/201 = 70.1%
replay-prefix matches:                 184/201 = 91.5%
label changes after replay:             43

Replay match rate by class:
  whitespace:                            19/19 = 100.0%
  number:                                 4/4 = 100.0%
  punctuation:                           79/90 = 87.8%
  capitalized:                           82/88 = 93.2%
```

Frozen hard-120 proxy gate evaluated on validation:

```text
gate:
  proxy_entropy <= 0.79383919530270064
  OR proxy_margin >= 1.9019756019115448

accepted:                              180/201
accepted-token error:                    6.1%
bad-candidate catch:                    35.3%
total token error:                       2.75%
accepted proxy rate:                    45.0%
estimated speed:                         9.47 tok/s
gate result:                             fails quality limits
```

Diagnostic stricter proxy gate on validation:

```text
gate:
  proxy_entropy <= 0.6
  OR proxy_margin >= 1.5

accepted:                              176/201
accepted-token error:                    4.55%
bad-candidate catch:                    52.9%
total token error:                       2.0%
accepted proxy rate:                    44.0%
estimated speed:                         about 9.37 tok/s
gate result:                             on the quality boundary
```

Decision: the hard-120-fitted proxy gate did **not** generalize cleanly. SAGE still looks feasible because a slightly stricter proxy gate on validation reaches the speed target while sitting on the error boundary, but this cannot be counted as a pass because it was tuned after looking at validation. The next design step was to freeze that stricter rule and test it on a second fresh prompt set before claiming robustness.

Second fresh validation result:

```text
Source:
  prompts:                             sage-validation-80b.txt
  chunks:                              offset 0, 40; limit 40 each
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      400
  top-1 token matches:                 180/400 = 45.0%
```

Candidate extraction with the frozen class prefilter:

```text
candidate classes:                     punct, whitespace, number, capitalized
margin prefilter:                      >= 0.644
candidates:                            205/400 = 51.3%
original continuation matches:         115/205 = 56.1%
replay-prefix matches:                 176/205 = 85.9%
label changes after replay:             63

Replay match rate by class:
  whitespace:                            25/25 = 100.0%
  punctuation:                           71/93 = 76.3%
  capitalized:                           80/87 = 92.0%
```

Predeclared conservative proxy gate evaluated on validation80b:

```text
gate:
  proxy_entropy <= 0.6
  OR proxy_margin >= 1.5

accepted:                              170/205
accepted-token error:                    5.29%
bad-candidate catch:                    69.0%
total token error:                       2.25%
accepted proxy rate:                    42.5%
estimated speed:                         9.21 tok/s
gate result:                             fails quality limits
```

False accepts under the predeclared gate were concentrated in punctuation and same-prefix cases:

```text
false accepts:                           9
  punctuation:                           8
  capitalized:                           1
  same-prefix labels:                    8
  diverged-prefix labels:                1
```

Validation-failure task extraction:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80*pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_failure_tasks.py `
  --replay-json @replay `
  --max-entropy 0.6 `
  --min-margin 1.5 `
  --token-classes punct,capitalized `
  --kind false-accepts `
  --tag validation80-80b-predeclared-punct-cap-falseaccepts
```

Local output:

```text
validation80 + validation80b false accepts: 17
  punctuation:                           15
  capitalized:                            2
```

Positive-control extraction for verifier fitting:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80*pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_failure_tasks.py `
  --replay-json @replay `
  --max-entropy 0.6 `
  --min-margin 1.5 `
  --token-classes punct,capitalized `
  --kind true-accepts `
  --max-per-class 17 `
  --tag validation80-80b-predeclared-punct-cap-trueaccepts-mpc17
```

Local output:

```text
true accepts selected:                  34
  punctuation:                           17
  capitalized:                           17
```

Hybrid sparse capture:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-115544-sage-failure-tasks-validation80-80b-predeclared-punct-cap-falseaccepts.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --tag validation-failures-hybrid2 `
  --timeout 1800

python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-123248-sage-failure-tasks-validation80-80b-predeclared-punct-cap-trueaccepts-mpc17.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --tag validation-trueaccepts-hybrid2 `
  --timeout 1800
```

Measured capture:

```text
false-accept capture:                   17 tasks, 704 tensor summaries
true-accept capture:                    34 tasks, 1474 tensor summaries
fit records:                            51
  true accepts:                         34
  false accepts:                        17
```

Best in-sample safe rule under `5%` accepted-token error:

```text
rule:
  proxy.entropy <= 0.23063930580449166
  OR ffn_norm_0.sum >= -72.150276000000005

accepted:                              27/51
accepted-token error:                    3.7%
bad-candidate catch:                    94.1%
good-candidate reject:                  23.5%
```

Best zero-error bad-catch-ranked rule:

```text
rule:
  proxy.entropy <= 0.040003470206379677
  OR ffn_norm_0.sum >= -72.150276000000005

accepted:                              25/51
accepted-token error:                    0.0%
bad-candidate catch:                   100.0%
good-candidate reject:                  26.5%
```

Modulo-4 prompt-index held-out checks:

```text
holdout r0:
  holdout labels:                         7 good / 6 bad
  accepted:                               6/13
  accepted-token error:                   0.0%
  bad-candidate catch:                  100.0%
  good-candidate reject:                 14.3%

holdout r1:
  holdout labels:                         8 good / 4 bad
  accepted:                               5/12
  accepted-token error:                   0.0%
  bad-candidate catch:                  100.0%
  good-candidate reject:                 37.5%

holdout r2:
  holdout labels:                        10 good / 2 bad
  accepted:                               7/12
  accepted-token error:                   0.0%
  bad-candidate catch:                  100.0%
  good-candidate reject:                 30.0%

holdout r3:
  holdout labels:                         9 good / 5 bad
  accepted:                               6/14
  accepted-token error:                   0.0%
  bad-candidate catch:                  100.0%
  good-candidate reject:                 33.3%
```

Interpretation: this is the strongest evidence so far that SAGE needs a tiny verifier, not more proxy-only threshold tuning. The useful signal is extremely small: the best rules use the proxy entropy plus one layer-0 FFN normalizer statistic that also fits inside a `1%` FFN-sentinel manifest. However, this is still a targeted validation-failure dataset and the rule was selected after seeing those failures. The next gate must freeze the rule family and test it on a fresh held-out prompt set.

Frozen verifier evaluation tool:

```powershell
python .\scripts\sage_verifier_eval.py `
  --probe-json .\benchmarks\20260627-115808-sage-probe-validation-failures-hybrid2.json .\benchmarks\20260627-123301-sage-probe-validation-trueaccepts-hybrid2.json `
  --replay-json @replay `
  --expression "(proxy.entropy <= 0.040003470206379677) OR (ffn_norm_0.sum >= -72.150276000000005)" `
  --label-quality any
```

Fresh frozen-rule validation result:

```text
Source:
  prompts:                             sage-validation-80c.txt
  chunks:                              offset 0, 40; limit 40 each
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      400
  top-1 token matches:                 173/400 = 43.25%
```

Candidate extraction with the frozen class prefilter:

```text
candidate classes:                     punct, whitespace, number, capitalized
margin prefilter:                      >= 0.644
candidates:                            174/400 = 43.5%
original continuation matches:          98/174 = 56.3%
replay-prefix matches:                 143/174 = 82.2%

Replay match rate by class:
  whitespace:                            19/19 = 100.0%
  punctuation:                           49/71 = 69.0%
  capitalized:                           75/84 = 89.3%
```

Predeclared proxy gate alone on validation80c:

```text
gate:
  proxy_entropy <= 0.6
  OR proxy_margin >= 1.5

accepted:                              144/174
accepted-token error:                   11.1%
bad-candidate catch:                    48.4%
total token error:                       4.0%
accepted proxy rate:                    36.0%
gate result:                             fails quality limits
```

Accepted punctuation/capitalized subset for the verifier:

```text
accepted punctuation/capitalized:       125
  true accepts:                         109
  false accepts:                         16
  false accepts by class:
    punctuation:                         13
    capitalized:                          3
```

Fresh `1%` FFN-sentinel capture:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-125540-sage-failure-tasks-validation80c-predeclared-punct-cap-accepted.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy ffn-sentinel `
  --active-percent 1 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --tag validation80c-accepted-punct-cap-ffn1 `
  --timeout 1800
```

Frozen sparse verifier on validation80c:

```text
rule:
  proxy.entropy <= 0.040003470206379677
  OR ffn_norm_0.sum >= -72.150276000000005

verifier records:                       125
verifier accepted:                       85
verifier false accepts:                   4
verifier accepted-token error:            4.7%
bad-candidate catch:                    75.0%
good-candidate reject:                  25.7%
```

End-to-end policy accounting for validation80c:

```text
proxy accepted total:                   144/400
verifier calls:                         125/400 = 31.25%
accepted after verifier:
  verifier-accepted punct/cap:           85
  unverified other classes:              19
  total accepted:                       104/400 = 26.0%
remaining false accepts:                  4

accepted-token error:                   4/104 = 3.85%
total token error:                      4/400 = 1.0%
oracle fallback calls:                 296/400 = 74.0%
modeled speed:
  1% verifier path, 10% oracle path:      7.35 tok/s
gate result:                             passes current numeric limits
```

Remaining fresh false accepts after the frozen verifier:

```text
capitalized:
  proxy " Message" vs oracle " Queue"
punctuation:
  quote vs bullet on prepared statements
  quote vs "Attention" on attention sinks
  quote vs bullet on sparse verifier prompt
```

Decision: two fresh validations proved proxy-only thresholds are not robust enough. The third fresh set shows that a frozen, tiny FFN-sentinel verifier can recover the quality budget while staying inside the `7 tok/s` speed target. This is now the strongest SAGE path: proxy gate first, `1%` FFN verifier for accepted punctuation/capitalized candidates, exact oracle fallback for everything rejected or uncertain.

Second frozen-rule validation result:

```text
Source:
  prompts:                             sage-validation-80d.txt
  chunks:                              offset 0, 40; limit 40 each
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      400
  top-1 token matches:                 215/400 = 53.75%
```

Candidate extraction with the frozen class prefilter:

```text
candidate classes:                     punct, whitespace, number, capitalized
margin prefilter:                      >= 0.644
candidates:                            198/400 = 49.5%
original continuation matches:         128/198 = 64.6%
replay-prefix matches:                 168/198 = 84.8%

Replay match rate by class:
  whitespace:                            26/26 = 100.0%
  punctuation:                           78/92 = 84.8%
  capitalized:                           64/80 = 80.0%
```

Predeclared proxy gate alone on validation80d:

```text
gate:
  proxy_entropy <= 0.6
  OR proxy_margin >= 1.5

accepted:                              165/198
accepted-token error:                    7.9%
bad-candidate catch:                    56.7%
total token error:                       3.25%
accepted proxy rate:                    41.25%
gate result:                             fails quality limits
```

Accepted punctuation/capitalized subset for the verifier:

```text
accepted punctuation/capitalized:       139
  true accepts:                         126
  false accepts:                         13
  false accepts by class:
    punctuation:                          6
    capitalized:                          7
```

Fresh `1%` FFN-sentinel capture was run in four chunks:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-133016-sage-failure-tasks-validation80d-predeclared-punct-cap-accepted.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy ffn-sentinel `
  --active-percent 1 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --offset 0 `
  --limit 35 `
  --tag validation80d-accepted-punct-cap-ffn1-o000-l035 `
  --timeout 1800
```

Frozen sparse verifier on validation80d:

```text
rule:
  proxy.entropy <= 0.040003470206379677
  OR ffn_norm_0.sum >= -72.150276000000005

verifier records:                       139
verifier accepted:                       86
verifier false accepts:                   2
verifier accepted-token error:            2.33%
bad-candidate catch:                    84.6%
good-candidate reject:                  33.3%
```

End-to-end policy accounting for validation80d:

```text
proxy accepted total:                   165/400
verifier calls:                         139/400 = 34.75%
accepted after verifier:
  verifier-accepted punct/cap:           86
  unverified other classes:              26
  total accepted:                       112/400 = 28.0%
remaining false accepts:                  2

accepted-token error:                   2/112 = 1.79%
total token error:                      2/400 = 0.5%
oracle fallback calls:                 288/400 = 72.0%
modeled speed:
  1% verifier path, 10% oracle path:      7.43 tok/s
gate result:                             passes current numeric limits
```

The same policy is now reproducible from one command:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80d-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
$probe = @(Get-ChildItem .\benchmarks -File -Filter '*sage-probe-validation80d-accepted-punct-cap-ffn1-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_policy_report.py `
  --replay-json @replay `
  --probe-json @probe `
  --stats-total 400 `
  --require-full-verifier-coverage `
  --require-pass
```

Local validation80d report output:

```text
final accepted:                         112/400
final false accepts:                      2
accepted-token error:                   1.79%
total-token error:                      0.50%
verifier coverage:                    139/139
verifier call rate:                    34.75%
oracle fallback rate:                  72.00%
effective throughput:                   7.43 tok/s
all gates:                              pass
```

Remaining fresh false accepts after the frozen verifier:

```text
punctuation:
  quote vs bullet on "The purpose of an abstract is"
capitalized:
  "Video" vs bullet on "When editing video, continuity means"
```

Decision update: validation80c and validation80d both passed with the same frozen verifier rule, and `sage_policy_report.py` now reproduces the full policy accounting. That is enough evidence to move from measurement-only work to the next implementation step: replace full `llama-debug` tensor dumping with a lightweight FFN-sentinel runtime hook that computes only compact layer-0 statistics and exposes them as structured scheduler input.

### Lightweight FFN-Sentinel Hook

Patch:

```text
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
```

The first patch adds `llama-debug --tensor-stats`. When paired with the earlier filtered-callback patch, a run such as:

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

emits compact matched-tensor statistics instead of full tensor values:

```text
common_debug_cb_eval:               ffn_norm-0 = (f32) ...
    stats: count = 896, sum = -4.13115366, mean = -0.00461066256, min = -3.69610286, max = 4.07102251, nan_count = 0
```

The second patch adds `llama-debug --tensor-stats-output PATH`, a JSONL sidecar format that carries the same signal without relying on stdout parsing:

```json
{"sequence":0,"name":"ffn_norm-0","dtype":"f32","op":"MUL","shape":[896,1,1,1],"count":896,"sum":-4.1311536571010947,"mean":-0.0046106625637289001,"min":-3.6961028575897217,"max":4.0710225105285645,"nan_count":0}
```

The third patch adds the same JSONL signal to normal generation through `common_init_from_params()`:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-completion.exe `
  -m .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -p "hello" `
  -n 1 `
  -no-cnv `
  -ngl 0 `
  -c 32 `
  -b 32 `
  -ub 8 `
  --no-warmup `
  --sage-tensor-filter "ffn_norm-0" `
  --sage-tensor-stats-output .\benchmarks\sage-runtime.jsonl
```

This is no longer a `llama-debug` path. It still writes a sidecar for measurement, but the tensor statistic is produced during normal generation, where the SAGE proxy/verifier/oracle scheduler will eventually make live decisions.

The fourth patch adds a first in-process scheduler decision event to normal generation:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-completion.exe `
  -m .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -p "hello" `
  -n 1 `
  -no-cnv `
  -ngl 0 `
  -c 32 `
  -b 32 `
  -ub 8 `
  --no-warmup `
  --sage-tensor-filter "ffn_norm-0" `
  --sage-tensor-stats-output .\benchmarks\sage-runtime.jsonl `
  --sage-decision-output .\benchmarks\sage-decision.jsonl `
  --sage-candidate-id smoke-qwen `
  --sage-token-class capitalized `
  --sage-proxy-entropy 0.5 `
  --sage-proxy-margin 2.0
```

The decision event is emitted by the C++ SAGE callback owner after the verifier run finishes. It applies the frozen rule:

```text
(proxy.entropy <= 0.040003470206379677) OR (ffn_norm_0.sum >= -72.150276000000005)
```

and writes `accept_proxy` or `oracle_fallback`. This is still one-shot verifier orchestration, not a full multi-token scheduler loop.

The fifth patch refactors the same frozen rule into a reusable C++ function:

```text
common_sage_policy_result common_sage_decide(const common_sage_policy_input & input)
```

This preserves the JSONL event path, but the policy no longer lives only in callback-destruction code. A future persistent `llama-sage` token loop can call the same rule directly after the sentinel statistic is available.

The sixth patch adds a model-free C++ policy gate:

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

This is the first model-free C++ regression gate for the persistent runtime. It proves `common_sage_decide()` can be exercised without Python, model loading, server startup, or verifier subprocess orchestration.

The seventh patch adds JSONL batch mode to the same executable. That keeps the
model-free gate but removes per-row process launch from the parity run:

```powershell
.\tools\llama.cpp-src\build-live-migration-cuda\bin\Release\llama-sage-policy-check.exe `
  --jsonl-in .\benchmarks\sage-policy-input.jsonl `
  --jsonl-out .\benchmarks\sage-policy-output.jsonl
```

Patch `0011` extends the same JSONL gate with scheduler fields: proxy token,
oracle token, selected token, top-1 match, and false-accept. The C++ policy
executable was then compared against validation80e replay/probe rows:

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
proxy gate accepts:                  147
verifier needed:                     126
verifier covered:                    126
expected final accepts:              101
expected oracle fallbacks:            76
final false accepts:                 1
action matches:                      177/177
reason matches:                      177/177
selected token matches:              177/177
false accept matches:                177/177
verifier accept matches:             177/177
result:                              pass
```

Interpretation: this is stronger than the six-case C++ smoke test. The direct
C++ policy function now matches the Python/replay policy over the same
validation80e row set that produced the `7.29 tok/s` modeled result, and the
C++ JSONL gate also proves the emitted-token choice and false-accept accounting.

Patch `0012` then adds `llama-sage-scheduler-replay`, a model-free persistent
C++ scheduler skeleton. It reads all replay candidates in one process, applies
`common_sage_decide()`, selects the emitted token, tracks false accepts, and
accumulates generated text per prompt stream. This is still not a live proxy plus
oracle model loop, but it removes the Python process boundary from the scheduler
state machine.

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

Local verification on 2026-06-27:

```text
llama-debug target build:             passed
small Qwen 0.5B stats smoke test:     passed
sage_probe_capture stats parser:      passed
0004 apply check on 0002 base:        passed
small Qwen 0.5B JSONL smoke test:     passed
sage_probe_capture --tensor-stats-jsonl: passed
0005 apply check on 0002+0004 base:   passed
llama-completion target build:        passed
small Qwen 0.5B runtime hook smoke:   passed
sage_runtime_capture parser:          passed
0006 apply check on 0002+0004+0005:   passed
llama-completion target after 0007:   passed
small Qwen 0.5B decision smoke:       passed
Gemma 31B false-accept decision smoke: passed
0007 apply check on 0002+0004+0005+0006: passed
llama-common target after 0008:       passed
llama-completion target after 0008:   passed
0008 reverse-apply check:             passed
llama-sage-policy-check target:       passed
llama-sage-policy-check self-test:    passed
0009 reverse-apply check:             passed
llama-sage-policy-check JSONL smoke:  passed
0010 reverse-apply check:             passed
0011 reverse-apply check:             passed
0012 reverse-apply check:             passed
0013 reverse-apply check:             passed
0014 reverse-apply check:             passed
sage_cpp_policy_parity self-test:     passed
validation80e C++ scheduler parity:   passed with 1 C++ process
llama-sage-scheduler-replay target:   passed
llama-sage-scheduler-replay self-test: passed
validation80e C++ replay skeleton:    passed with 1 C++ process
llama-sage-proxy-live target:         passed
llama-sage-proxy-live self-test:      passed
Qwen 0.5B resident proxy smoke:       passed on CUDA
llama-sage-dual-live target:          passed
llama-sage-dual-live self-test:       passed
Qwen 0.5B dual-context smoke:         passed on CUDA
```

Patch `0013` adds the first model-backed resident proxy example:
`llama-sage-proxy-live`. It keeps one llama.cpp model/context alive, generates
proxy tokens directly, computes entropy and margin from logits, classifies each
token, and emits SAGE proxy-gate telemetry. This removes the proxy server and
Python request loop from the proxy side of the architecture, while leaving oracle
and verifier integration as the next step.

Local Qwen 0.5B CUDA smoke on 2026-06-27:

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
sampled-is-logit-top1:               4/4
telemetry fields:                    token_class, proxy_entropy, proxy_margin, proxy_gate_accept
```

Interpretation: this does not yet run the full SAGE proxy/oracle/verifier loop,
but it proves the proxy side can be resident and scheduler-readable in C++.

Patch `0014` adds the first live two-context scheduler: `llama-sage-dual-live`.
It keeps a resident proxy context and a resident oracle context, generates a
proxy candidate, applies `common_sage_decide()`, calls the oracle only on
fallback, and then feeds the selected text back into both contexts. This is the
first C++ prototype where the scheduler state machine and both model contexts
are live in one process. The same executable now emits
`sage-active-byte-ledger-v0` per generated step: accepted proxy tokens report
`oracle_mode: none` and `oracle_active_bytes: 0`; oracle fallback reports
`oracle_mode: exact_resident_context` and the resident oracle model byte count.
The trace also records proxy/oracle token positions and step latency. KV byte
telemetry now includes tiered KV accounting from runtime token counts, with
`kv_byte_status: tiered_runtime_accounting_not_attention_integrated`. On the
local Qwen 0.5B smoke, the full-precision estimate is `12288` KV bytes/token;
the fallback step reports `73728` oracle full-precision KV bytes and `73728`
tiered KV bytes because the short context is entirely hot. This is not packed
KV attention yet.

Qwen 0.5B same-model dual-context smoke:

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
ledger schema:                       sage-active-byte-ledger-v0
accept ledger oracle bytes:          0
fallback ledger oracle bytes:        485452288
kv status:                           tiered_runtime_accounting_not_attention_integrated
kv bytes/token estimate:             12288
fallback oracle full KV bytes:       73728
fallback oracle tiered KV bytes:     73728
live proxy top-k rows:               10 per generated step
live proxy shortlist schema:         sage-live-proxy-shortlist-v0
packed KV attention:                 false
```

Interpretation: this is still not the 100B sparse oracle, but it proves the
core online control flow: accepted proxy text can advance the oracle context
without an oracle sample, and an oracle fallback can rejoin the selected prefix
without restarting either model. It also proves the first runtime byte-ledger
shape plus live tiered KV accounting fields; the next gaps are replacing exact
resident-model fallback bytes with sparse GGUF block-page bytes and replacing
accounting-only KV fields with measured packed KV tensor allocation and
attention-read bytes. It also proves the C++ runtime now produces proxy top-k
candidate rows that can be handed to the sparse Q6_K verifier; the verifier is
not called from the live loop yet.

Paired debug/runtime comparison on 2026-06-27:

```powershell
python .\scripts\sage_compare_runtime_debug.py `
  --debug-json .\benchmarks\20260627-160546-sage-probe-gemma31b-debug-runtime-compare-hello.json `
  --runtime-json .\benchmarks\20260627-160604-sage-runtime-gemma31b-debug-runtime-compare-hello.json `
  --json-out .\benchmarks\20260627-compare-gemma31b-debug-vs-runtime-hello.json `
  --require-match

python .\scripts\sage_compare_runtime_debug.py `
  --debug-json .\benchmarks\20260627-161103-sage-probe-validation80e-runtime-subset-debug.json `
  --runtime-json .\benchmarks\20260627-161140-sage-runtime-validation80e-runtime-subset.json `
  --json-out .\benchmarks\20260627-compare-gemma31b-validation80e-runtime-subset.json `
  --require-match

python .\scripts\sage_runtime_gate.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-151635-sage-failure-tasks-validation80e-predeclared-punct-cap-accepted.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --tensor-filter "ffn_norm-0" `
  --tokens 1 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --offset 3 `
  --limit 10 `
  --timeout 1800 `
  --tag validation80e-o003-l010
```

```text
Qwen 0.5B hello:
  prompt pairs:                         1
  matched tensor records:               1
  mismatches:                           0
  result:                            pass

Gemma 31B hello:
  prompt pairs:                         1
  matched tensor records:               1
  mismatches:                           0
  result:                            pass

Gemma 31B validation80e accepted subset:
  task prompts:                         3
  debug tensor records:                11
  runtime tensor records:              11
  matched tensor records:              11
  missing in runtime:                   0
  extra in runtime:                     0
  mismatches:                           0
  result:                            pass

Gemma 31B validation80e accepted subset, next slice:
  task prompts:                        10
  debug tensor records:                49
  runtime tensor records:              49
  matched tensor records:              49
  missing in runtime:                   0
  extra in runtime:                     0
  mismatches:                           0
  result:                            pass
```

Runtime scheduler prototype on 2026-06-27:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_failure_tasks.py `
  --replay-json @replay `
  --max-entropy 0.6 `
  --min-margin 1.5 `
  --token-classes punct,capitalized `
  --kind false-accepts `
  --tag validation80e-predeclared-punct-cap-falseaccepts

python .\scripts\sage_runtime_gate.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-162534-sage-failure-tasks-validation80e-predeclared-punct-cap-falseaccepts.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --tensor-filter "ffn_norm-0" `
  --tokens 1 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --limit 7 `
  --timeout 1800 `
  --tag validation80e-falseaccepts-l007

python .\scripts\sage_runtime_scheduler.py `
  --runtime-json .\benchmarks\20260627-162541-sage-runtime-gate-validation80e-falseaccepts-l007-runtime.json `
  --require-full-verifier-coverage `
  --json-out .\benchmarks\20260627-sage-runtime-scheduler-validation80e-falseaccepts-l007.json

python .\scripts\sage_runtime_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-162534-sage-failure-tasks-validation80e-predeclared-punct-cap-falseaccepts.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --tensor-filter "ffn_norm-0" `
  --tokens 1 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --limit 7 `
  --timeout 1800 `
  --capture-decisions `
  --tag gemma31b-decision-falseaccepts-l007 `
  --json-out .\benchmarks\20260627-sage-runtime-gemma31b-decision-falseaccepts-l007.json

python .\scripts\sage_compare_runtime_decisions.py `
  --runtime-json .\benchmarks\20260627-sage-runtime-gemma31b-decision-falseaccepts-l007.json `
  --json-out .\benchmarks\20260627-compare-runtime-decisions-gemma31b-falseaccepts-l007.json `
  --require-match
```

```text
False-accept parity gate:
  task prompts:                         7
  debug tensor records:                34
  runtime tensor records:              34
  matched tensor records:              34
  missing in runtime:                   0
  extra in runtime:                     0
  mismatches:                           0
  result:                            pass

Runtime scheduler on known false accepts:
  captures:                             7
  verifier covered:                   7/7
  verifier accepts:                     1
  verifier rejects:                     6
  oracle fallbacks:                     6
  false accepts left:                   1
  bad-candidate catch:               6/7

In-process C++ decision parity:
  captures:                             7
  runtime decisions:                    7
  expected Python decisions:            7
  matched decisions:                    7
  missing decisions:                    0
  extra decisions:                      0
  mismatches:                           0
  result:                            pass
```

Interpretation: `sage_runtime_scheduler.py` is the first scheduler-facing bridge from live runtime captures to decisions. It is still offline and JSONL-based, but the tensor statistic is produced by the normal generation binary, then turned into explicit `accept_proxy` or `oracle_fallback` actions with the frozen verifier expression. `0007` moves the same frozen decision into a one-shot in-process event, and `sage_compare_runtime_decisions.py` proves that the C++ event matches the Python scheduler on all `7` known validation80e false accepts. The next implementation step is a real multi-token proxy/verifier/oracle scheduler loop.

Multi-token scheduler replay on 2026-06-27:

```powershell
$replay = @(Get-ChildItem .\benchmarks -File -Filter '*sage-candidate-replay-validation80e-pwnc-replay-o*-l025.json' | Sort-Object Name | ForEach-Object FullName)
$probe = @(Get-ChildItem .\benchmarks -File -Filter '*sage-probe-validation80e-accepted-punct-cap-ffn1-stats-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_multitoken_replay.py `
  --replay-json @replay `
  --runtime-json @probe `
  --decision-source python `
  --stats-total 400 `
  --require-full-verifier-coverage `
  --require-pass `
  --json-out .\benchmarks\20260627-sage-multitoken-replay-validation80e-full-ffn1-stats.json

python .\scripts\sage_multitoken_replay.py `
  --replay-json @replay `
  --tasks-filter-json .\benchmarks\20260627-162534-sage-failure-tasks-validation80e-predeclared-punct-cap-falseaccepts.json `
  --runtime-json .\benchmarks\20260627-sage-runtime-gemma31b-decision-falseaccepts-l007.json `
  --decision-source cpp `
  --stats-total 7 `
  --require-full-verifier-coverage `
  --json-out .\benchmarks\20260627-sage-multitoken-replay-validation80e-falseaccepts-l007-cpp.json

$cpp = @(Get-ChildItem .\benchmarks -File -Filter '20260627-sage-runtime-gemma31b-decision-validation80e-o*-l035.json' | Sort-Object Name | ForEach-Object FullName)
python .\scripts\sage_compare_runtime_decisions.py `
  --runtime-json @cpp `
  --json-out .\benchmarks\20260627-compare-runtime-decisions-gemma31b-validation80e-full-cpp.json `
  --require-match

python .\scripts\sage_multitoken_replay.py `
  --replay-json @replay `
  --runtime-json @cpp `
  --decision-source cpp `
  --stats-total 400 `
  --require-full-verifier-coverage `
  --require-pass `
  --json-out .\benchmarks\20260627-sage-multitoken-replay-validation80e-full-cpp-decisions-promptkey.json
```

```text
Full validation80e measured-stat trace:
  replay rows:                         177
  prompt count:                         40
  verifier covered:                  126/126
  final accepts:                    101/400
  final false accepts:                   1
  accepted-token error:              0.99%
  total-token error:                 0.25%
  modeled throughput:                7.29 tok/s
  all gates:                         pass

C++ decision false-accept trace:
  replay rows:                           7
  verifier covered:                    7/7
  C++ decisions used:                    7
  oracle fallbacks:                      6
  false accepts left:                    1

Full validation80e C++ decision trace:
  C++ decision parity:              126/126 matched, 0 mismatches
  replay rows:                         177
  prompt texts:                          80
  verifier covered:                  126/126
  C++ decisions used:                  126
  Python decisions used:                 0
  final accepts:                    101/400
  final false accepts:                   1
  accepted-token error:              0.99%
  total-token error:                 0.25%
  modeled throughput:                7.29 tok/s
  all gates:                         pass
```

Interpretation: `sage_multitoken_replay.py` is still replay-based, but it is no longer only aggregate math. It emits the prompt/step actions the scheduler would take and proves the frozen policy survives sequence-level accounting on validation80e. The full validation80e run now uses `126` in-process C++ verifier decisions and `0` Python verifier decisions, while preserving the same quality and modeled speed as the compact-stat validation. The replay script now sorts and counts by prompt text because numeric `prompt_index` values are local to logprob chunks; this fixed the trace prompt count from `40` to `80` without changing the aggregate quality or speed metrics.

Live loop prototype on 2026-06-27:

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

```text
Qwen verified live smoke:
  live steps:                           1
  proxy gate accepts:                   1
  verifier calls:                       1
  verifier covered:                  true
  verifier action:              accept_proxy
  oracle fallbacks:                     0
  final text:                     " Paris"
  artifact: benchmarks/20260627-174029-sage-live-loop-qwen-live-smoke-verifier-policy.json
```

Interpretation: `scripts/sage_live_loop.py` is the first live proxy/verifier/oracle loop over newly generated tokens. It is not a replay trace: it gets a fresh proxy token from `llama-server`, applies the candidate-token policy and proxy gate, calls patched `llama-completion` directly for `--sage-decision-output`, and records the selected token. In managed mode it pauses the proxy server before verifier subprocess calls so a single-GPU machine does not need to hold proxy and verifier/oracle weights at once. The live policy now matches the validated candidate class boundary: only `punct,whitespace,number,capitalized` tokens are eligible for proxy acceptance by default; other token classes fall back to oracle. This Qwen run proves live control flow and C++ verifier event wiring; it does not prove Gemma 31B speed or 100B+ viability.

Live-vs-replay comparator:

```powershell
python .\scripts\sage_live_gate.py --prompt-index 1 --max-live-tokens 1 --print-only
python .\scripts\sage_live_gate.py --prompt-index 1 --max-live-tokens 1
```

The gate runner validates model/replay paths, starts the seeded live loop, then immediately runs the comparator against the validation80e C++ replay trace. Use `--print-only` first on the RTX 3060 machine; the non-print command can load Gemma 12B and Gemma 31B. The lower-level equivalent is:

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

This comparator maps live decisions onto ordered replay decisions for one prompt, then reports action matches, selected-token matches, proxy-token matches, verifier coverage matches, and the first divergence. `--seed-replay-json` copies the prompt and Gemma chat settings from the replay artifact, and `--tokens-from-replay` sets the live token count to the number of replay rows for that prompt. It is the required next gate after a Gemma-scale live run because replay validation is only useful if the live scheduler follows the same decisions before prefix drift.

Control check: the same seeded workflow was run through `scripts/sage_live_gate.py --qwen-smoke --prompt-index 1 --max-live-tokens 1 --no-require-pass` with Qwen 0.5B in place of Gemma. The comparator failed at live token `0` with `0/1` action, selected-token, proxy-token, and verifier-coverage matches. That is expected and useful: it proves the gate detects unrelated live behavior instead of passing any syntactically valid live-loop JSON.

Measured Gemma live-gate results:

```text
validation80e prompt 1:
  seed rows after prompt-text filter:    1
  skipped prompt-index collision rows:   1
  replay selection:             live_seed_task_ids
  compared steps:                       1
  action matches:                       1
  selected-token matches:               1
  verifier-coverage matches:            1
  result:                            pass

validation80e prompt 35, first two rows:
  compared steps:                       2
  verifier calls:                       1
  oracle fallbacks:                     1
  proxy accepts:                        1
  action matches:                       2
  selected-token matches:               2
  verifier-coverage matches:            2
  result:                            pass
```

The failed two-token prompt-1 attempt was useful: it exposed the chunk-local `prompt_index` collision between `cand-0001` and `cand-0094`. After switching live/replay comparison to seed task IDs and filtering seed rows by prompt text, prompt 1 passes as a valid one-row live gate instead of comparing unrelated prompts.

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
proxy accepts:                          1
oracle fallbacks:                       2
verifier calls:                         1
proxy request time:                  0.290 s
verifier request time:              12.079 s
oracle request time:                 3.537 s
orchestration/reload overhead:      35.122 s
overhead share:                       68.8%
```

Interpretation: the current live loop is a correctness instrument, not the final runtime. It proves that live proxy tokens can be routed through the same frozen replay policy and still match replay on seeded Gemma prompts, but it is intentionally conservative about memory: it pauses the proxy and launches verifier/oracle subprocesses. That makes the measured `0.059 tok/s` mostly a process/model-management number. The production direction is now specific: collapse SAGE into a persistent runtime that keeps the scheduler, proxy path, verifier path, and oracle fallback warm. A llama.cpp-native implementation is the cleanest target; a less invasive interim production target would be long-lived worker processes with pinned model state and an IPC scheduler, so token steps stop paying subprocess and reload overhead.

Persistent runtime projection:

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

7 tok/s with measured proxy:       oracle active <= 3.47%
7 tok/s with 25 tok/s proxy:       oracle active <= 10.74%
10 tok/s with 25 tok/s proxy:      oracle active <= 5.24%
format scaffold hard120, 7 tok/s: exact fallback can be <= 1062 ms/token
format scaffold hard120, 10 tok/s: exact fallback must be <= 73 ms/token
```

Decision: removing Python orchestration is necessary but not enough for the
older high-oracle-call path. The newer format-scaffold path is more promising:
candidate verification is tiny, and the hard120 fallback rate is low enough that
the measured proxy path can still project above `7 tok/s` with a local
`270 ms/token` fallback. This does not prove 100B execution because the fallback
and sparse hidden-state path are not resident or measured yet. The next
implementation must make the proxy, Q6_K candidate verifier, and fallback path
warm in one persistent runtime. The implementation RFC is
`docs/sage-persistent-runtime.md`.

Update after adding `sage-sparse-oracle-runtime-step-v0`: replacing the
configured `270 ms/token` fallback with the measured sparse component replay
still gives `8.066 tok/s` on hard120 with measured proxy speed. The GPU-only
H2D+kernel fallback view gives `9.778 tok/s`, but the contract uses the
conservative full replay including host reads. This is the first budget result
where the fallback latency comes from measured sparse page plus candidate
verifier artifacts rather than a hand-set constant.

Measured sparse-fallback projection command:

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

Overlap/prefetch budget command:

```powershell
python .\scripts\sage_overlap_budget.py `
  --json-out .\benchmarks\sage-overlap-budget-hard120-measured-sparse-fallback-10tps.json
```

Measured overlap target:

```text
full sparse replay:                 8.066 tok/s
host-read hidden view:              9.778 tok/s
target:                             10.000 tok/s
max fallback latency for target:    73.102 ms
extra GPU fallback to hide:         52.336 ms/fallback
proxy latency reduction alternate:  2.268 ms/token
fallback-rate reduction alternate:  1.808 percentage points
```

Interpretation: this narrows the 10 tok/s problem to a specific implementation
target. Simply hiding CPU host reads is almost enough but not sufficient on the
hard120 fallback rate. The next runtime experiment should use asynchronous page
prefetch and CUDA stream overlap to hide about half of the measured H2D+kernel
fallback window, or it must improve the candidate source enough to reduce
fallbacks.

First overlap measurement: `scripts/sage_oracle_cuda_overlap_smoke.py` now uses
pre-staged pinned host buffers, two CUDA streams, and two device buffers over the
same full page plan. It reduced the measured GPU window from `139.51 ms`
separate H2D+kernel work to `123.13 ms` overlapped work, saving `16.38 ms`
(`11.7%`). This is a real CUDA event measurement for transport plus a byte-touch
kernel, not host-read prefetch and not sparse transformer execution. It covers
part of the `52.3 ms/fallback` 10 tok/s gap; the remaining gap still needs
background host-page prefetch, sparse dequant/matmul overlap, lower fallback
rate, or a faster proxy.

Second overlap measurement: `scripts/sage_oracle_cuda_prefetch_overlap_smoke.py`
keeps host staging in the measured wall time. It uses one background GGUF reader,
a pinned host-buffer ring, two CUDA streams, and two device buffers. For the same
full page plan, sequential host-read+H2D+kernel components totaled `1233.49 ms`;
the pipelined wall time was `1116.22 ms`, saving `117.27 ms` (`9.5%`). This
proves the prefetch direction but also shows the current pager is still dominated
by raw GGUF reads and CRC work. A production sparse oracle needs resident pinned
pages, a much smaller active set, or both.

Third pager measurement: `scripts/sage_oracle_cuda_page_cache_smoke.py` loads
the selected pages once into a resident pinned host cache and then replays
multiple fallback passes from that cache. For the same full page plan, cache
build took `1464.97 ms` for `2.246 GiB`; three cache replay passes averaged
`139.15 ms` GPU time each and reused `4.493 GiB` as cache hits. This removes raw
host reads from repeated fallbacks, but it is still transport plus a byte-touch
kernel. The next real speed step is smaller active sets and sparse dequant/matmul
work that replaces the touch kernel.

Page-cache budget update: `scripts/sage_page_cache_budget.py` projects that
resident-cache path at `9.72 tok/s` on hard120. The strict `10 tok/s` target
allows about `73.10 ms` per fallback, which corresponds to `1.180 GiB` active
pages at the measured cache replay rate. That is about `5.07%` of a 100B 2-bit
reference, or roughly `41` pages if the current page mix is kept.

Full frozen-validation verification on 2026-06-27:

```text
validation80c compact-stats capture:
  capture files:                         4
  verifier captures:                   125/125
  tensor summaries:                    1515
  missing stats fields:                   0
  verifier accepted:                    85
  verifier false accepts:                4
  accepted-token error:               3.85%
  total-token error:                  1.00%
  effective throughput:               7.35 tok/s
  gate result:                         pass

validation80d compact-stats capture:
  capture files:                         4
  verifier captures:                   139/139
  tensor summaries:                    1641
  missing stats fields:                   0
  verifier accepted:                    86
  verifier false accepts:                2
  accepted-token error:               1.79%
  total-token error:                  0.50%
  effective throughput:               7.43 tok/s
  gate result:                         pass

validation80e fresh compact-stats capture:
  prompt overlap with 80/80b/80c/80d:    0
  proxy/oracle top-1 agreement:       205/400 = 51.25%
  candidate tasks:                    177/400
  replay-prefix matches:              155/177 = 87.6%
  proxy-only accepted:                147/400
  proxy-only false accepts:             7
  proxy-only accepted-token error:    4.76%
  proxy-only total-token error:       1.75%
  verifier captures:                  126/126
  tensor summaries:                    1440
  missing stats fields:                   0
  verifier accepted:                    80
  verifier false accepts:                1
  verifier bad-candidate catch:       85.7%
  final accepted:                      101/400
  final false accepts:                   1
  accepted-token error:               0.99%
  total-token error:                  0.25%
  effective throughput:               7.29 tok/s
  gate result:                         pass
```

The compact-stats path reproduced the full-dump verifier counts exactly on both held-out validation sets and passed one new fresh validation set. The JSONL sidecar then removes stdout scraping from future captures and gives the SAGE loop a stable record shape for `ffn_norm_0.sum`. The non-debug runtime hook now produces the same record shape during normal generation, matches debug JSONL on a one-prompt Gemma smoke test plus 20 validation-shaped tasks, drives offline scheduler decisions that catch `6/7` known validation80e false accepts, emits in-process decisions that match the Python scheduler on all `126/126` validation80e verifier tasks, has a prompt/step multi-token replay trace that passes validation80e using only C++ verifier decisions at `7.29 tok/s` modeled throughput, has a live-loop Qwen smoke that wires proxy token generation to direct C++ verifier decisions, and now passes small seeded Gemma live-vs-replay gates. The next proof step is not another Python live-loop speed run; it is removing the measured `68.8%` orchestration/reload overhead with a persistent SAGE runtime, then repeating the same live/replay gate under that runtime.

Previous first-chunk sparse-verifier behavior, for comparison:

```text
bad-candidate catch:                    63.6%
good-candidate reject:                   2.2%
accepted proxy rate:                    45.1%
accepted-token error:                    3.9%
total token error:                       1.8%
estimated speed at 1% verifier:          8.77 tok/s
gate result:                             passes current numeric limits
```

Decision: the full hard-120 router is not safe as a raw accept-all candidate policy. A second proxy-side confidence gate made it numerically viable on calibration, but two fresh validations showed that proxy-only thresholds are too brittle. Sparse captures are therefore no longer optional side evidence; they are the next mechanism to test before spending more effort on CUDA/block-paged sparse execution.

Hard-120 first chunk measured result:

```text
Source:
  prompts:                             sage-calibration-hard-120.txt
  offset/limit:                        0/40
  generated tokens per prompt:          8
  ignored Gemma control-prefix steps:   3

Proxy/oracle logprob agreement:
  filtered steps:                      200
  top-1 token matches:                 105/200 = 52.5%
  mean top-k token Jaccard:             0.240
  mean proxy margin:                    6.424
```

Candidate extraction with `punct,whitespace,number,capitalized` and margin `>=0.644`:

```text
candidates:                            100/200 = 50.0%
original continuation matches:          74/100 = 74.0%
same-prefix original matches:           60/68 = 88.2%
```

Oracle replay on the proxy prefix was run in four chunks with offsets `0`, `25`, `50`, and `75`:

```text
replay-prefix matches:                  89/100 = 89.0%
label changes after replay:             19
proxy token in oracle top-k:             100/100

Replay match rate by class:
  whitespace:                            18/18 = 100.0%
  punctuation:                           37/43 = 86.0%
  capitalized:                           34/39 = 87.2%

Replay match rate by prefix status:
  same-prefix:                           60/68 = 88.2%
  diverged-prefix:                       29/32 = 90.6%
```

Policy implication before sparse verification:

```text
accepted proxy rate:                    50.0%
accepted-token error:                   11.0%
total token error:                       5.5%
gate result:                             fails quality limits
```

Hard-chunk sparse verifier capture:

```powershell
python .\scripts\sage_probe_capture.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --tasks-json .\benchmarks\20260627-094427-sage-candidate-tasks-hard120-o000-l040-punct-whitespace-number-capitalized.json `
  --mode gemma4-chat `
  --gemma4-thought-prefix `
  --policy hybrid `
  --active-percent 2 `
  --ngl 38 `
  --ctx-size 256 `
  --batch-size 128 `
  --ubatch-size 8 `
  --offset 0 `
  --limit 25 `
  --tag hard120-o000-l040-pwnc-hybrid2-o000-l025 `
  --timeout 1800
```

Capture was run in four chunks with offsets `0`, `25`, `50`, and `75`.

Sparse verifier fit against replay labels:

```text
records:                                100
replay-labeled good/bad:                89/11
best <=5% accepted-error rule:
  (proxy.margin >= 1.1640262305736542)
  OR (layer0.residual_delta_sum <= -173.88958800000006)

accepted:                               91/100
accepted-token error:                    4.4%
bad-candidate catch:                    63.6% = 7/11
good-candidate reject:                   2.2% = 2/89
```

Policy implication with this verifier behavior:

```text
1% verifier, 10% oracle fallback:
  verifier call rate:                   50.0%
  oracle call rate:                     54.5%
  accepted proxy rate:                  45.5%
  accepted-token error:                  4.4%
  total token error:                     2.0%
  estimated speed:                       8.80 tok/s
  gate result:                           passes current numeric limits

2% verifier, 10% oracle fallback:
  estimated speed:                       8.41 tok/s
  gate result:                           passes current numeric limits
```

Modulo-4 held-out smoke tests:

```text
holdout r0:
  holdout labels:                        27/28 good
  accepted:                              27/28
  accepted-token error:                   3.7%
  bad-candidate catch:                    0.0%

holdout r1:
  holdout labels:                        21/26 good
  accepted:                              20/26
  accepted-token error:                   5.0%
  bad-candidate catch:                   80.0%

holdout r2:
  holdout labels:                        17/20 good
  accepted:                              17/20
  accepted-token error:                   0.0%
  bad-candidate catch:                  100.0%

holdout r3:
  holdout labels:                        24/26 good
  accepted:                              25/26
  accepted-token error:                   4.0%
  bad-candidate catch:                   50.0%
```

Decision: the first hard chunk shows the proposed shape can survive harder prompts: high skip rate plus a tiny verifier can meet the speed/error budget on local measurements. The weak point is generalization: one held-out split had only one bad candidate and the selected rule accepted it, so the current verifier is still too close to the calibration set. The next gate is cross-chunk verifier validation across the full hard-120 set.

Decision: the first verifier cannot be a normal partial forward pass. The `1%` budget is smaller than one complete Gemma 31B layer. The first plausible sparse verifier should be one of:

- a single boundary FFN sentinel plus cheap norms/logit adapter;
- three early attention sentinels, or two edge/boundary attention sentinels;
- a learned low-rank correction head trained from proxy/oracle logprob disagreements;
- a slightly more expensive `2%` verifier only if the fallback oracle path can be reduced below `10%`.

### Gate 2: Sparse Verification

On the 31B oracle, test whether partial verification can predict the full oracle decision:

- first N layers only;
- last M layers only using proxy hidden-state projection;
- selected attention heads;
- selected FFN groups;
- logit correction adapters.

Success target:

```text
Sparse oracle should match full oracle top-1 or top-k often enough
to avoid dense oracle execution for at least 60-80% of tokens.
```

Refined target after router calibration:

```text
For the current punctuation+whitespace+capitalized candidate rule, a sparse
verifier is useful if it catches at least 50% of bad proxy candidates at about
1% active 100B-equivalent weight cost and low false-reject rate. Higher catch
rates buy quality margin; lower catch rates fail the current total-error gate.
```

### Gate 3: Active-Byte Budget Simulator

Build a simulator that answers:

```text
Given measured PCIe bandwidth, GPU memory, layer sizes, KV size,
and oracle-call rate, can this policy reach 7-10 tok/s?
```

This should happen before implementing complex CUDA code.

Initial tool:

```powershell
python .\scripts\sage_budget.py `
  --target-tps 7 `
  --params-b 100 `
  --quant-bpw 2 `
  --oracle-call-rate 0.25 `
  --pcie-gbps 24
```

Sweep tool:

```powershell
python .\scripts\sage_simulate.py `
  --target-tps 7 `
  --proxy-tps 25 `
  --params-b 100 `
  --quant-bpw 2 `
  --oracle-call-rates 0.05,0.1,0.25,0.5 `
  --active-percents 1,2,5,10,15,20
```

This estimates the end-to-end SAGE speed if the proxy handles every token and the giant oracle is called only for a fraction of tokens.

Initial simulator observations:

- Dense per-token streaming of a 100B 2-bit model is capped around `1 tok/s` by PCIe bandwidth alone.
- A resident proxy is the speed floor and the speed ceiling: if the proxy can only do `10 tok/s`, the full system cannot reach far above that.
- With a `25 tok/s` proxy, `24 GB/s` sustained PCIe, `25%` oracle-call rate, and `10%` active giant-model blocks per oracle call, the simple simulator estimates about `14 tok/s`.
- With a slower `10 tok/s` proxy under the same oracle assumptions, the estimate drops to about `7.7 tok/s`.
- These estimates assume the sparse oracle can produce a useful verification signal without executing the whole giant model. That is the central unproven research risk.

The simulator is not proof that SAGE works. It is a falsification tool: if measured proxy speed, oracle-call rate, active block percentage, or PCIe bandwidth make the table miss the target, CUDA implementation work should stop or the architecture should change.

### Gate 4: Block-Paged Weight Runtime

Extend the live migration work from layer migration to smaller block paging:

- pinned CPU buffers;
- preallocated GPU staging buffers;
- no per-token allocation;
- async H2D copies;
- scheduler reset only when placement changes structurally;
- measurement of copy overlap.

Initial block index tool:

```powershell
python .\scripts\sage_gguf_blocks.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --budget-gib 3.83
```

Local Gemma 31B Q4 block evidence:

```text
Tensor bytes:                16.42 GiB
Global/non-layer bytes:       1.08 GiB
Repeating layers:            60
Average repeating layer:      0.26 GiB
FFN bytes:                   10.90 GiB, 66.4%
Attention bytes:              4.44 GiB, 27.0%
Active budget tested:         3.83 GiB
Full layers fitting budget:  15 / 60
FFN groups fitting budget:   21 / 60
Attention groups fitting:    53 / 60
```

This strongly supports the SAGE premise: full-layer oracle execution is too coarse, but component-level sparse execution is plausible. With the tested budget, almost all attention groups can fit, but only about one third of FFN groups can fit. Therefore the first sparse-oracle prototype should focus on FFN group selection and correction, not attention streaming.

Local Gemma 12B Q4 comparison:

```text
Tensor bytes:                 6.48 GiB
Repeating layers:            48
Average repeating layer:      0.12 GiB
FFN bytes:                    4.45 GiB, 68.7%
Attention bytes:              1.26 GiB, 19.5%
Full layers fitting budget:  32 / 48
FFN groups fitting budget:   41 / 48
Attention groups fitting:    48 / 48
```

This makes Gemma 12B a useful local proxy for designing sparse FFN routing before testing larger oracle models.

Initial block planner:

```powershell
python .\scripts\sage_block_plan.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --budget-gib 3.83 `
  --policy boundary
```

Quota-controlled block planner:

```powershell
python .\scripts\sage_block_plan.py `
  --model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  --budget-gib 3.83 `
  --policy balanced `
  --ffn-min-share 0.55 `
  --attention-max-share 0.35
```

Measured Gemma 31B candidate plans under a `3.83 GiB` active budget:

| Policy | Selected bytes | Selected blocks | Main result |
| --- | ---: | ---: | --- |
| `balanced` | 3.76 GiB | 113 | 53 attention groups + norms, no FFN |
| `attention-first` | 3.76 GiB | 113 | same as balanced, because attention blocks are smaller |
| `ffn-first` | 3.83 GiB | 86 | 18 FFN groups + 8 attention groups + norms |
| `boundary` | 3.77 GiB | 101 | 8 boundary FFN groups + 33 attention groups + norms |
| `balanced + quotas` | 3.82 GiB | 92 | 14 FFN groups, 18 attention groups, all norms |

The quota plan used `--ffn-min-share 0.55 --attention-max-share 0.35`, producing about `2.54 GiB` FFN and `1.28 GiB` attention. That is a more realistic sparse-oracle candidate than the raw `balanced` policy because it forces the expensive FFN path to participate.

Sparse oracle page-ledger prototype:

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

Measured plan-level result:

```text
schema:                             sage-oracle-page-ledger-v0
status:                             plan_only_not_executed
selected pages:                     79
selected bytes:                     2.246 GiB
active percent of 100B 2-bit ref:   9.65%
stage count:                        4
estimated PCIe transfer:            100.5 ms
budget status:                      within_7tps_budget
```

Interpretation: this is not sparse CUDA execution yet, but it is the first
runtime-shaped page contract for the oracle path. It converts static block
selection into pages, double-buffer stages, transfer estimates, a sparse ledger
template, and an exact dense fallback template. The result fits the 7 tok/s
active-byte window, but not the 10 tok/s window.

Sparse oracle page-staging smoke:

```powershell
python .\scripts\sage_oracle_pager_staging.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-staging-gemma31b-full.json
```

Measured host-staging result:

```text
schema:                             sage-oracle-page-staging-v0
status:                             measured_host_staging_not_cuda
stages staged:                      4
pages staged:                       79
bytes staged:                       2.246 GiB
max live buffer:                    0.712 GiB / 0.750 GiB
staging elapsed:                    1197.2 ms
staging throughput:                 1.88 GiB/s
CUDA execution:                     not_implemented
```

Interpretation: this proves the selected page plan can be resolved back to real
GGUF tensor byte ranges and streamed through bounded host staging buffers. It
does not prove GPU page transfer, sparse kernels, or a 100B oracle forward pass.

Sparse oracle CUDA staging smoke:

```powershell
python .\scripts\sage_oracle_cuda_staging.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-staging-gemma31b-full.json
```

Measured CUDA transport result:

```text
schema:                             sage-oracle-page-cuda-staging-v0
status:                             measured_cuda_h2d_not_sparse_compute
stages staged:                      4
pages staged:                       79
bytes staged:                       2.246 GiB
max live device buffer:             0.712 GiB / 0.750 GiB
host read time:                     1153.1 ms
CUDA H2D time:                      113.0 ms
CUDA H2D throughput:                19.88 GiB/s
sparse compute:                     not_implemented_sparse_compute
```

Interpretation: the transfer budget is no longer only a PCIe estimate. The
selected oracle pages can be staged into real CUDA buffers within about the
`100 ms` order of magnitude assumed by the planner. The slower part in this
Python smoke is reading and packing from the GGUF file; a production pager needs
pinned resident pages, async overlap, and sparse kernels.

Sparse oracle CUDA kernel-touch smoke:

```powershell
python .\scripts\sage_oracle_cuda_kernel_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-kernel-gemma31b-full.json
```

Measured kernel-touch result:

```text
schema:                             sage-oracle-page-cuda-kernel-smoke-v0
status:                             measured_cuda_kernel_touch_not_transformer
stages staged:                      4
bytes touched:                      2.246 GiB
max live device buffer:             0.712 GiB / 0.750 GiB
H2D time:                           106.98 ms
kernel time:                        18.34 ms
kernel touch throughput:            122.49 GiB/s
sparse transformer:                 not_implemented
```

Interpretation: the staged page bytes are now proven to be GPU-kernel-readable,
not merely transferable. This is still not sparse oracle inference: the next
hard step is replacing the byte-touch kernel with dequantization and block
matmul/scoring kernels that can evaluate candidate tokens.

Sparse oracle CUDA H2D/kernel overlap smoke:

```powershell
python .\scripts\sage_oracle_cuda_overlap_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --json-out .\benchmarks\sage-oracle-page-cuda-overlap-gemma31b-full.json
```

Measured overlap result:

```text
schema:                             sage-oracle-page-cuda-overlap-smoke-v0
status:                             measured_cuda_double_buffer_overlap_touch_not_transformer
stages staged:                      4
bytes touched:                      2.246 GiB
max live device buffers:            1.424 GiB across 2 buffers
summed H2D time:                    120.58 ms
summed kernel time:                 18.93 ms
separate GPU work:                  139.51 ms
overlapped GPU window:              123.13 ms
overlap saving:                     16.38 ms / 11.7%
host read overlap:                  not_measured_prestaged_pinned_host_buffers
sparse transformer:                 not_implemented
```

Interpretation: the RTX 3060 can overlap part of the sparse-oracle page H2D
window with GPU work when the host pages are already pinned. This validates the
double-buffer scheduling direction, but it does not remove the need for
background GGUF page prefetch or real sparse transformer kernels.

Sparse oracle host-prefetch plus CUDA-overlap smoke:

```powershell
python .\scripts\sage_oracle_cuda_prefetch_overlap_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --json-out .\benchmarks\sage-oracle-page-cuda-prefetch-overlap-gemma31b-full.json
```

Measured prefetch-overlap result:

```text
schema:                             sage-oracle-page-cuda-prefetch-overlap-smoke-v0
status:                             measured_host_prefetch_cuda_overlap_touch_not_transformer
stages staged:                      4
bytes touched:                      2.246 GiB
host ring capacity:                 1.500 GiB across 2 buffers
device ring capacity:               1.500 GiB across 2 buffers
summed host read:                   1100.09 ms
summed H2D time:                    114.62 ms
summed kernel time:                 18.79 ms
sequential total:                   1233.49 ms
pipeline wall time:                 1116.22 ms
pipeline saving:                    117.27 ms / 9.5%
host read overlap:                  measured_single_worker_background_prefetch
sparse transformer:                 not_implemented
```

Interpretation: background prefetch can hide some transport/touch work, but it
does not make the current fallback fast. The host read path is now the dominant
measured term. That points the next implementation toward a resident pinned page
cache, fewer active pages, and real sparse dequant/matmul overlap rather than
more Python-side file streaming.

Sparse oracle resident page-cache replay smoke:

```powershell
python .\scripts\sage_oracle_cuda_page_cache_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --replays 3 `
  --json-out .\benchmarks\sage-oracle-page-cuda-page-cache-gemma31b-full-replay3.json
```

Measured page-cache replay result:

```text
schema:                             sage-oracle-page-cuda-page-cache-smoke-v0
status:                             measured_resident_pinned_page_cache_replay_touch_not_transformer
stages cached:                      4
replays:                            3
cache size:                         2.246 GiB
cache build:                        1464.97 ms
cache hits:                         8
cache-hit bytes:                    4.493 GiB
average H2D per replay:             136.31 ms
average kernel per replay:          20.43 ms
average overlapped GPU replay:      139.15 ms
sparse transformer:                 not_implemented
```

Interpretation: the repeated fallback path no longer needs to reread selected
GGUF pages from disk if those pages stay pinned in host memory. This is the
right shape for the production pager, but the remaining per-replay transport
window is still larger than the strict `10 tok/s` fallback budget.

Resident page-cache 10 tok/s budget target:

```powershell
python .\scripts\sage_page_cache_budget.py `
  --json-out .\benchmarks\sage-page-cache-budget-hard120-resident-cache-10tps.json
```

Measured budget result:

```text
schema:                             sage-page-cache-budget-v0
status:                             measured_page_cache_budget_target_not_transformer_integrated
page-cache projection:              9.72 tok/s
target:                             10.00 tok/s
measured replay:                    139.15 ms for 2.246 GiB
max active bytes for target:        1.180 GiB
required active-byte reduction:     47.5%
max reference active percent:       5.07% of 100B 2-bit
max fallback rate for target:       2.28%
```

Interpretation: this is now a concrete optimizer target. The next page ledger
should not simply reuse the `2.246 GiB` active-byte plan; it should search for a
useful plan near `1.18 GiB`, or the candidate source must cut fallback rate
roughly in half.

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

Measured reduced-cache result:

```text
page ledger:                        69 pages
active bytes:                       1.075 GiB
reference active percent:           4.62% of 100B 2-bit
stages cached:                      3
average H2D per replay:             50.42 ms
average kernel per replay:          10.25 ms
average overlapped GPU replay:      53.83 ms
page-cache projection:              10.08 tok/s
sparse transformer:                 not_implemented
```

Interpretation: the measured page-cache transport budget can hit the strict
`10 tok/s` projection if the active oracle set is held near `1.075 GiB`. This
is not model quality proof yet. The next sparse-oracle prototype must show that
this smaller active page set can produce useful candidate/verifier evidence
before exact fallback.

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

python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-1180mib-10tps.json `
  --limit-stages 0 `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --score-top-k 5 `
  --max-score-tensors 8 `
  --cpu-check-rows 16 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-balanced-1180mib-ffn-norm0.json
```

Measured reduced-compute result:

```text
Q4_0 tensors:                       32
Q4_0 bytes:                         1.073 GiB
max live device buffer:             0.389 GiB / 0.500 GiB
dequant H2D / kernel:               49.91 ms / 8.49 ms
matvec H2D / kernel:                48.77 ms / 10.76 ms
output scores:                      302,336
CPU score checks:                   32 passed
candidate scoring:                  ranked_projection_rows_not_candidate_tokens
sparse transformer:                 not_implemented
```

Measured reduced real-activation result:

```text
activation:                         ffn_norm-0, width 5376
Q4_0 tensors:                       23
Q4_0 bytes:                         0.715 GiB
max live device buffer:             0.260 GiB / 0.500 GiB
H2D / matvec kernel:                33.13 ms / 34.36 ms
output scores:                      253,952
CPU score checks:                   23 passed
candidate scoring:                  ranked_projection_rows_not_candidate_tokens
sparse transformer:                 not_implemented
```

Interpretation: the reduced 10 tok/s page plan is no longer only a byte-touch
transport result. It can decode and score real selected Q4_0 matrices under the
smaller stage buffer. The missing proof is whether captured activations from
the real transformer make this reduced page set predictive enough to accept or
reject candidate tokens before exact fallback.

Reduced page signal-quality probe:

```powershell
python .\scripts\sage_reduced_page_quality_probe.py `
  --json-out .\benchmarks\sage-reduced-page-quality-gemma31b-1180mib-vs-full-ffn-norm0.json
```

Measured signal-quality result:

```text
full Q4_0 real-activation bytes:     1.482 GiB
reduced Q4_0 real-activation bytes:  0.715 GiB
reduced/full bytes:                  48.2%
shared scored tensors:               13
shared top-1 row match:              100.0%
shared top-k overlap mean:           100.0%
full global top-10 retained:         3/10 = 30.0%
full global top-20 retained:         9/20 = 45.0%
full global top-50 retained:         26/50 = 52.0%
token decision integrated:           false
```

Interpretation: the reduced plan is not corrupting the rows it keeps; shared
tensors reproduce the fuller run's top rows exactly. The selection policy is the
weak point: the byte-budgeted page set drops too many of the strongest global
row signals from the fuller active set. The next innovation step should be a
signal-aware page selector that still fits the `1.18 GiB` / `10 tok/s` budget.

Sparse oracle CUDA Q4_0 dequant smoke:

```powershell
python .\scripts\sage_oracle_cuda_dequant_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-dequant-gemma31b-full.json
```

Measured Q4_0 dequant result:

```text
schema:                             sage-oracle-page-cuda-dequant-smoke-v0
status:                             measured_cuda_q4_0_dequant_not_matmul
stages staged:                      4
Q4_0 tensors:                       67
Q4_0 bytes:                         2.244 GiB
dequantized values:                 4,282,908,672
max live device buffer:             0.709 GiB / 0.750 GiB
H2D time:                           101.75 ms
dequant time:                       11.71 ms
dequant throughput:                 191.65 GiB/s
sparse matmul:                      not_implemented
```

Interpretation: the active oracle pages are no longer opaque bytes. CUDA can now
decode the actual GGUF Q4_0 weight blocks selected by the page ledger. The next
hard step is sparse block matmul/candidate scoring, where dequantized weights
must interact with activations and produce a useful verifier/oracle score.

Sparse oracle CUDA Q4_0 matvec smoke:

```powershell
python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-matvec-gemma31b-full.json
```

Measured Q4_0 matvec result:

```text
schema:                             sage-oracle-page-cuda-matvec-smoke-v0
status:                             measured_cuda_q4_0_matvec_synthetic_activation_not_transformer
stages staged:                      4
Q4_0 tensors:                       67
Q4_0 bytes:                         2.244 GiB
matvec values:                      4,282,908,672
output scores:                      628,480
max live device buffer:             0.709 GiB / 0.750 GiB
H2D time:                           101.00 ms
matvec time:                        17.48 ms
matvec throughput:                  128.36 GiB/s
candidate scoring:                  not_implemented
```

Interpretation: this is the first sparse-page computation shaped like scoring:
real selected Q4_0 matrices multiply an activation-like vector and produce
scores. The activation is synthetic and deterministic, so this still is not real
oracle logits. The next implementation step is feeding live hidden states into
these sparse kernels and comparing the sparse scores against exact oracle logits
for candidate accept/reject decisions.

Sparse oracle CUDA Q4_0 real-activation matvec smoke:

```powershell
python .\scripts\sage_oracle_cuda_matvec_smoke.py `
  --page-ledger .\benchmarks\sage-oracle-page-ledger-gemma31b-balanced-2330mib.json `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --limit-stages 0 `
  --json-out .\benchmarks\sage-oracle-page-cuda-real-activation-matvec-gemma31b-ffn-norm0-full.json
```

Measured real-activation result:

```text
schema:                             sage-oracle-page-cuda-real-activation-matvec-smoke-v0
status:                             measured_cuda_q4_0_matvec_real_activation_not_oracle_logits
activation:                         ffn_norm-0, width 5376
stages staged:                      4
Q4_0 tensors:                       48
Q4_0 bytes:                         1.482 GiB
matvec values:                      2,829,582,336
output scores:                      526,336
max live device buffer:             0.473 GiB / 0.750 GiB
H2D time:                           86.12 ms
matvec time:                        65.51 ms
matvec throughput:                  22.63 GiB/s
candidate scoring:                  not_implemented
```

Interpretation: this replaces the synthetic vector with a real Gemma hidden
state captured from `llama-debug` tensor-values JSONL. It is a meaningful step
toward a sparse oracle because staged Q4_0 blocks are now multiplied by live
runtime values. The result is slower than the synthetic baseline and still only
scores width-matched matrices independently; it does not compose full
transformer layers or compare candidate logits yet.

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

Measured ranked real-activation result:

```text
schema:                             sage-oracle-page-cuda-real-activation-ranked-matvec-smoke-v0
status:                             measured_cuda_q4_0_real_activation_ranked_scores_not_oracle_logits
top-score tensors:                  28
top-score rows:                     140
CPU score checks:                   16 / 16 passed
max CPU score abs error:            3.97e-07
H2D time:                           80.58 ms
matvec time:                        67.69 ms
candidate scoring:                  ranked_projection_rows_not_candidate_tokens
```

Interpretation: the kernel now preserves per-row scores, so the runtime can see
which projection rows were most activated instead of only seeing a total score.
The CPU checks recompute sampled rows directly from the staged Q4_0 bytes and
the captured activation vector, proving the CUDA row-score values are not just
plausible but correct for the tested rows. This is still not token-level
candidate scoring, because these rows are attention/FFN projection rows rather
than final vocabulary logits.

Sparse oracle CUDA Q6_K tied-vocab projection smoke:

```powershell
python .\scripts\sage_oracle_cuda_vocab_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-ffn-norm0-values-5376.jsonl `
  --activation-name ffn_norm-0 `
  --top-k 10 `
  --cpu-check-top-k 8 `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-vocab-projection-gemma31b-ffn-norm0-full.json
```

Measured Q6_K vocab projection result:

```text
schema:                             sage-oracle-page-cuda-q6-k-vocab-projection-smoke-v0
status:                             measured_q6_k_tied_vocab_projection_not_true_logits
tensor:                             token_embd.weight Q6_K [5376, 262144]
vocab rows scored:                  262,144
chunks:                             2
staged bytes:                       1.077 GiB
max live device buffer:             0.750 GiB
H2D time:                           49.10 ms
kernel time:                        29.83 ms
top token ids:                      10
CPU score checks:                   8 / 8 passed
candidate scoring:                  vocab_token_scores_from_captured_activation_not_oracle_logits
```

Interpretation: this is the first artifact that scores vocabulary token ids
rather than internal projection rows. It also proves a second real GGUF
quantization format, Q6_K, can be paged and scored under the same weak-GPU
budget. It is not yet true oracle logits, because the activation vector is from
`ffn_norm-0`; the next proof must capture the final post-norm hidden state and
compare the top token ids against llama.cpp's logits for the same prompt.

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

Measured final-logit comparison result:

```text
schema:                             sage-oracle-page-cuda-q6-k-vocab-projection-smoke-v0
status:                             measured_q6_k_tied_vocab_projection_with_llamacpp_logit_compare
tensor:                             token_embd.weight Q6_K [5376, 262144]
activation:                         result_norm [5376, 1, 1, 1]
vocab rows scored:                  262,144
chunks:                             2
staged bytes:                       1.077 GiB
max live device buffer:             0.750 GiB
H2D time:                           56.35 ms
kernel time:                        29.87 ms
llama.cpp top-1 match:              true
llama.cpp overlap@10:               10 / 10
max top-logit abs error:            0.0367
CPU score checks:                   8 / 8 passed
```

Interpretation: this upgrades the Q6_K proof from "token-id mechanics" to
"final-token ranking agrees with llama.cpp" for one measured Gemma forward pass.
The script applies Gemma 4 final-logit softcapping from GGUF metadata before
comparing against llama.cpp's saved logits. It still is not a full sparse
100B-style oracle loop, because it computes the final vocab projection for one
captured hidden vector, but it proves the paged projection path can reproduce
llama.cpp's final top-token ordering.

Sparse oracle CUDA Q6_K candidate verifier:

```powershell
python .\scripts\sage_oracle_cuda_candidate_smoke.py `
  --activation-jsonl .\benchmarks\sage-gemma31b-result-norm-values-5376.jsonl `
  --activation-name result_norm `
  --llamacpp-logits-bin .\benchmarks\sage-gemma31b-logits-debug\llamacpp-gemma-4-31B_q4_0-it.bin `
  --top-k-from-logits 64 `
  --json-out .\benchmarks\sage-oracle-page-cuda-q6k-candidate-verifier-gemma31b-result-norm-top64.json
```

Measured sparse candidate-verifier result:

```text
schema:                             sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0
status:                             measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits
candidate token rows:               64
candidate bytes:                    282,240
active vocab tensor share:          0.0244%
active 100B 2-bit reference share:  0.00000113%
H2D time:                           0.0493 ms
kernel time:                        0.1034 ms
candidate top-1 match:              true
llama.cpp logit checks:             64 / 64 passed
max logit abs error:                0.0482
CPU score checks:                   64 / 64 passed
```

Interpretation: this is the first proof in the shape a sparse oracle scheduler
can use for token verification. Instead of full-vocab projection, a proxy or
router supplies a token shortlist, and the giant-model output head scores only
those rows. The measured active bytes are tiny compared with both the Gemma
vocab tensor and a 100B 2-bit reference. This does not yet compute the hidden
state sparsely and does not choose the shortlist; it proves that once a
shortlist exists, the Q6_K output-head verification can be sparse and still
agree with llama.cpp.

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

Measured bridge result:

```text
schema:                             sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0
status:                             measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits
candidate source:                   live_proxy_shortlist
live proxy candidate rows:          10
candidate bytes:                    44,100
active vocab tensor share:          0.003815%
H2D time:                           0.0249 ms
kernel time:                        0.1270 ms
candidate top-1 match:              true
llama.cpp logit checks:             10 / 10 passed
max logit abs error:                0.0257
```

Interpretation: this closes the offline handoff between the live C++ producer
and the sparse CUDA verifier. The live `sage-live-proxy-shortlist-v0` token ids
are now accepted as verifier candidate rows, staged into the CUDA buffer, scored
against the captured Gemma hidden state, and compared to saved llama.cpp logits.
The current artifact intentionally remains conservative: the live trace comes
from Qwen while the verifier rows are Gemma, so it proves row-ID plumbing and
sparse scoring mechanics, not tokenizer-equivalent semantic agreement. A
same-tokenizer live trace must be measured before claiming live verifier quality.

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

Measured fallback result:

```text
schema:                             sage-oracle-page-cuda-q6-k-candidate-verifier-smoke-v0
status:                             measured_sparse_q6_k_candidate_rows_compared_to_llamacpp_logits
candidate source:                   logprob_proxy_top_k
prompt:                             The capital of France is
proxy top-1 token id:                236771
Gemma 31B global top-1 token id:     9079
Gemma 31B global top-1 token:        " Paris"
candidate rows:                     10
candidate bytes:                    44,100
H2D time:                           0.0293 ms
kernel time:                        0.0911 ms
llama.cpp logit checks:             10 / 10 passed
candidate covers global top-1:      false
coverage status:                    candidate_misses_global_top1_exact_fallback_required
```

Interpretation: this is a same-prompt Gemma-family check for the reject side of
the verifier contract. The CUDA verifier scores the proxy-proposed rows
correctly, but the rows are digits while the Gemma 31B answer is ` Paris`.
Therefore sparse scoring alone is not an accept signal; the runtime must check
coverage/confidence and keep exact fallback. This is why the later
format-scaffold shortlist matters: proxy top-k alone is too brittle.

Sparse oracle runtime-step replay:

```powershell
python .\scripts\sage_oracle_runtime_step.py `
  --json-out .\benchmarks\sage-sparse-oracle-runtime-step-gemma31b-page-q6k-fallback-replay.json
```

Measured runtime-step replay result:

```text
schema:                             sage-sparse-oracle-runtime-step-v0
status:                             measured_component_replay_not_transformer_integrated
component replay complete:          true
sparse page bytes:                  2,412,150,000
sparse page GiB:                    2.246490
candidate bytes:                    44,100
active sparse step share:           9.6488% of 100B 2-bit reference
max measured device stage:          0.711755 GiB
H2D time:                           107.0062 ms
CUDA kernel time:                   18.4317 ms
exact fallback required:            true
transformer integrated:             false
```

Interpretation: this is the first single active-byte ledger that joins the page
transport path and the candidate verifier path. It keeps the sparse step inside
the `10%` 100B-equivalent byte budget and records the exact fallback decision.
It is still a component replay: the page kernel is a measured byte-touch kernel,
not Gemma transformer math, and the candidate verifier uses a captured hidden
state. The next implementation must put this ledger behind a live llama.cpp
token step.

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

python .\scripts\sage_proxy_shortlist_eval.py `
  --logprob-json .\benchmarks\20260627-094413-sage-logprob-gemma12b-to-31b-hard120-chat8-filtered-o000-l040.json `
                 .\benchmarks\20260627-101048-sage-logprob-gemma12b-to-31b-hard120-chat8-filtered-o040-l040.json `
                 .\benchmarks\20260627-101327-sage-logprob-gemma12b-to-31b-hard120-chat8-filtered-o080-l040.json `
  --ignore-prefix-steps 3 `
  --k-values 1,2,4,8,10 `
  --train-mod 2 `
  --train-remainders 0 `
  --static-rescue-count 0 `
  --position-rescue-count 8 `
  --prompt-piece-count 16 `
  --prompt-piece-ids-per-piece 2 `
  --json-out .\benchmarks\sage-proxy-shortlist-format-scaffold-hard120-k10-pos8-prompt16.json
```

Measured format-scaffold result:

```text
validation80e eval coverage:        97.50%
validation80e exact fallback:       2.50%
validation80e rows per step:        27.0
hard120 eval coverage:              95.67%
hard120 exact fallback:             4.33%
hard120 rows per step:              26.9
active vocab tensor share:          about 0.0103%
```

Interpretation: proxy top-k alone missed too many oracle top-1 tokens because
Gemma 31B often emits structured format pieces such as topic labels before the
content answer. A small per-position scaffold catches common format tokens, and
prompt-derived vocab pieces catch many topic words without using oracle logits
from the eval rows. This is a more realistic candidate source for the sparse
Q6_K verifier than the earlier oracle-derived top-64 smoke. It is still not a
complete scheduler because each miss requires exact fallback and the hidden
state path is not yet sparse or persistent.

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

Measured plan-level result:

```text
schema:                             sage-kv-ledger-v0
status:                             plan_only_not_runtime_integrated
context tokens:                     4096
hot/warm/cold tokens:               528 / 3568 / 0
full precision KV estimate:         3.438 GiB
SAGE tiered KV estimate:            0.817 GiB
hot VRAM KV:                        0.443 GiB
saved vs full precision:            76.2%
hot tier fits 0.8 GiB budget:       true
```

Interpretation: this gives the runtime a concrete KV byte target rather than
leaving KV as a vague future optimization. It still does not prove compressed KV
inside llama.cpp; the next runtime step is to feed these planned byte tiers into
the live `sage-active-byte-ledger-v0` fields and then replace estimates with
measured cache allocation/transfer bytes.

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

Measured bounded KV pack result:

```text
schema:                             sage-kv-tier-pack-smoke-v0
status:                             measured_synthetic_2bit_warm_kv_pack_not_runtime_integrated
sample tokens:                      8
sample full bytes:                  7,208,960
sample packed bytes:                  901,120
compression ratio:                  8.00x
estimated warm packed bytes:          401,899,520
estimated warm packed size:         0.374 GiB
estimated tier total:               0.817 GiB
CUDA pack time:                     0.0926 ms
CUDA unpack time:                   0.0522 ms
CUDA pack throughput:               72.50 GiB/s input
CUDA unpack throughput:             64.32 GiB/s output
CUDA packed bytes match CPU:        true
```

Interpretation: this turns the warm-KV compression plan into a native CUDA byte
mechanics measurement. It does not prove quality, attention correctness, or
runtime KV integration because the input is a deterministic synthetic KV-shaped
sample. It does prove that the proposed 2-bit warm-tier byte layout can be
packed/unpacked cheaply on the local GPU and that the packed byte count matches
the existing KV ledger. The next proof must attach this to real llama.cpp KV
tensors and measured attention reads.

KV runtime ledger accounting:

```powershell
python .\scripts\sage_kv_runtime_ledger.py `
  --live-trace-json .\benchmarks\sage-dual-live-qwen05b-arithmetic-tiered-kv-smoke.json `
  --kv-ledger-json .\benchmarks\sage-kv-ledger-gemma31b-ctx4096-hot528-warm2bit.json `
  --kv-tier-smoke-json .\benchmarks\sage-kv-tier-pack-smoke-gemma31b-ctx4096-hot528-warm2bit-sample8.json `
  --include-context-limit `
  --json-out .\benchmarks\sage-kv-runtime-ledger-qwen05b-arithmetic-gemma31b-plan.json
```

Measured runtime-accounting result:

```text
schema:                             sage-kv-runtime-ledger-v0
status:                             runtime_token_accounting_with_measured_pack_smoke_not_attention_integrated
runtime steps annotated:            2
runtime oracle fallback steps:      1
context sweep exercises warm KV:    true
max oracle full precision KV:       3.438 GiB
max oracle tiered KV:               0.817 GiB
max oracle warm KV:                 0.374 GiB
max saved vs full precision:        76.2%
CUDA pack time carried as evidence: 0.0926 ms
attention integration:              false
```

Interpretation: this closes the accounting gap between the live trace and the
KV compression plan. The runtime can now be audited with the same hot/warm/cold
byte policy used by the Gemma 31B ledger, and the artifact carries measured CUDA
packing evidence. It still does not execute attention from packed KV tensors;
that remains the next KV implementation gate.

The first practical sparse-oracle policies should be:

1. `boundary`: preserve first/last layer behavior and broad attention coverage.
2. `ffn-first`: test whether selected FFN groups can correct a proxy model enough to avoid full oracle calls.
3. A future learned router: replace static block priorities with prompt/token-conditioned FFN group selection.

The important design lesson is that a naive byte-efficient policy selects attention and starves FFN, but FFN dominates model size and likely contributes much of the missing reasoning capacity. SAGE therefore needs an explicit FFN budget or learned FFN router.

### Gate 5: 100B External Model Test

Only after the proxy/oracle and simulator gates look promising:

- choose a 70B or 100B-class GGUF;
- create mixed low-bit block index;
- run slow exact baseline;
- run SAGE sparse/proxy mode;
- measure speed, agreement, and quality.

## What Not To Claim Yet

Do not claim:

- exact 100B dense inference at 7-10 tok/s on RTX 3060;
- quality parity with the full model;
- that PCIe streaming can hide full-model movement;
- that a 100B model fits in 12 GB just because a nominal 1-bit weight count is near 12 GB;
- that KV compression solves weight bandwidth.

The defensible claim is:

```text
We are researching a sparse oracle runtime for 100B+ local inference where
the giant model is consulted selectively under a measured active-byte budget.
```

## Immediate Next Prototype

The next useful implementation should be small and measurable:

1. Treat proxy-only gates as failed; do not publish them as robust.
2. Treat the frozen `1%` FFN-sentinel verifier as the current leading path because it passed `sage-validation-80c.txt`, `sage-validation-80d.txt`, and fresh `sage-validation-80e.txt` with accepted-token error <=`5%`, total error <=`2%`, and modeled speed >=`7 tok/s`.
3. Use `scripts/sage_policy_report.py` as the required gate for every fresh validation run; a pass must report proxy accepts, verifier calls, oracle fallbacks, accepted-token error, total-token error, modeled throughput, and verifier coverage from one command.
4. Treat the compact-stats hook as validated on 80c/80d/80e; use it for future verifier captures instead of full tensor-value dumps.
5. Prefer the new `--tensor-stats-jsonl` capture path for debug experiments; it is a structured bridge between `llama-debug` and the SAGE policy code.
6. Treat the paired debug/runtime signal path as smoke-passed: Qwen hello, Gemma hello, a 3-task Gemma validation80e accepted subset, the next 10-task Gemma validation80e slice, and a 7-task false-accept slice all matched with `0` tensor-stat mismatches.
7. Treat `scripts/sage_runtime_scheduler.py` as the offline scheduler bridge: it converts runtime captures into `accept_proxy`/`oracle_fallback` actions and catches `6/7` known validation80e false accepts.
8. Treat `0007-sage-runtime-decision-output.patch` as the first in-process decision event: it emits `accept_proxy` or `oracle_fallback` from normal `llama-completion` after the verifier run, and its event matched the Python scheduler on all `7/7` known validation80e false accepts.
9. Treat `scripts/sage_multitoken_replay.py` as the current multi-token scheduler trace: it passes validation80e using full in-process C++ verifier decisions, with no Python verifier decisions.
10. Treat `scripts/sage_live_loop.py` as the first live control-flow prototype: Qwen 0.5B smoke-passes with one verified proxy accept, direct C++ decision JSONL output, and the same candidate-token eligibility boundary as the replay policy.
11. Treat `scripts/sage_live_replay_compare.py` as the live/replay parity gate. The Gemma seeded runs now pass on prompt 1 and prompt 35 after switching comparison to seed task IDs.
12. Treat `scripts/sage_live_gate.py` as the one-command Gemma gate runner. It validates paths, runs the seeded live loop, and runs the comparator; `--print-only` is the safe preflight.
13. Treat `scripts/sage_live_report.py` as the live overhead gate. The current Python loop is correctness-useful but speed-invalid because `68.8%` of wall time is orchestration/reload overhead.
14. Treat `scripts/sage_runtime_projection.py`, `docs/sage-persistent-runtime.md`, and `llama-sage-policy-check` as the next production checkpoint. The next implementation must remove process orchestration, move the sentinel verifier into the hot path, and measure whether proxy latency plus active oracle percent can still hit the `7-10 tok/s` band.
15. Only after the persistent runtime projection still passes should a 70B/100B external model test be downloaded or converted.

No CUDA kernel work or 100B download should start until the persistent-runtime numbers look plausible.

## Sources

- FlexGen: <https://arxiv.org/abs/2303.06865>
- PowerInfer: <https://arxiv.org/abs/2312.12456>
- PowerInfer-2: <https://arxiv.org/abs/2406.06282>
- PagedAttention/vLLM: <https://arxiv.org/abs/2309.06180>
- Deja Vu: <https://arxiv.org/abs/2310.17157>
- LayerSkip: <https://arxiv.org/abs/2404.16710>
- CLaSp: <https://aclanthology.org/2025.acl-long.1525/>
- Speculative sampling: <https://arxiv.org/abs/2302.01318>
- Fast speculative decoding: <https://arxiv.org/abs/2211.17192>
- Dustin sparse verification: <https://arxiv.org/html/2606.24957v1>
- SSV sparse attention verification: <https://arxiv.org/html/2605.19893v1>
- Sparse verification during speculative decoding: <https://arxiv.org/abs/2512.21911>
- H2O KV cache: <https://arxiv.org/abs/2306.14048>
- SnapKV: <https://arxiv.org/html/2404.14469v1>
- KIVI: <https://arxiv.org/abs/2402.02750>
- InfiniGen: <https://arxiv.org/abs/2406.19707>
- StreamingLLM: <https://arxiv.org/abs/2309.17453>
- TurboQuant: <https://arxiv.org/html/2504.19874v1>
- AWQ: <https://arxiv.org/abs/2306.00978>
- GPTQ: <https://arxiv.org/abs/2210.17323>
- SmoothQuant: <https://arxiv.org/abs/2211.10438>
- LLM in a flash: <https://arxiv.org/abs/2312.11514>
- KTransformers: <https://github.com/kvcache-ai/ktransformers>
- Karpathy `llm.c`: <https://github.com/karpathy/llm.c>
- Karpathy `llama2.c`: <https://github.com/karpathy/llama2.c>
