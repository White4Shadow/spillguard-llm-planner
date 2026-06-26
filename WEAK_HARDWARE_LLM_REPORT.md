# Running Larger LLMs On This Machine

Date: 2026-06-26

## Local Hardware

- CPU: 13th Gen Intel Core i7-13700KF, 16 cores / 24 logical processors.
- System RAM: about 32 GB.
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM.
- Runtime prepared: llama.cpp `b9804` CUDA 12.4 build in `tools/llama.cpp-b9804-cuda124`.

This is a strong desktop CPU with a midrange VRAM limit. The practical constraint is not whether small models run; it is how to run models whose weights plus KV cache exceed 12 GB VRAM and may approach system RAM limits.

## Existing Approaches

| Approach | What It Saves | Practical Limit |
| --- | --- | --- |
| Weight quantization: GGUF Q4/Q5/Q8, GPTQ, AWQ, EXL2, bitsandbytes | VRAM/RAM for model weights | Does not remove KV cache growth; very low bits can hurt quality or need model-specific kernels. |
| Quantization-aware training and native low-bit models: Gemma QAT, BitNet b1.58 | Quality at lower weight precision | Requires suitable model family and often specialized runtime kernels. |
| KV-cache quantization: Q8/Q4/IQ4, FP8 KV, KIVI-style schemes | Context memory | Can reduce memory, but speed/quality depends on backend and model. |
| CPU/GPU hybrid layer offload | VRAM pressure | Too much GPU offload can spill through unified memory and become slower than a deliberate partial placement. |
| CPU/disk offload: Accelerate, FlexGen-style scheduling | Total RAM/VRAM pressure | Makes otherwise impossible runs possible, but latency becomes the tradeoff. |
| MoE expert offload: KTransformers, PowerInfer-style locality | Active memory and compute for MoE models | Useful for MoE, less useful for dense Gemma-class models; needs expert prediction/caching. |
| Speculative decoding / MTP / draft models | Latency | Needs a compatible draft model or MTP head; does not solve base model memory alone. |
| Smaller distilled models plus RAG/tool use | Effective capability per GB | Does not actually run the target large model; useful fallback for production. |

Key sources used: llama.cpp documents 1.5-8 bit quantization, CUDA, and CPU+GPU hybrid inference; Microsoft BitNet documents a dedicated runtime for 1.58-bit models; Hugging Face and vLLM document quantization, offload, and KV-cache features; FlexGen/PowerInfer/KIVI/GPTQ/AWQ papers define the major research directions.

## Local Experiments

### Control: Qwen2.5 0.5B Q4_K_M

Model: `models/qwen2.5-0.5b-instruct-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf`

| Setup | Prompt Processing | Generation |
| --- | ---: | ---: |
| CPU-only, F16 KV | about 6,049 tok/s | about 67 tok/s |
| Full GPU offload, F16 KV | about 19,673 tok/s | about 390 tok/s |
| Full GPU offload, Q8 KV | about 18,164 tok/s | about 356 tok/s |

This validates the runtime and shows the expected result: small models are easy for this GPU.

### Mid-size: Gemma 4 12B QAT Q4_0

Model: `models/gemma-4-12b-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf`

File size: 6.98 GB, model size reported by llama.cpp: 6.48 GiB.

| Setup | Prompt Processing | Generation |
| --- | ---: | ---: |
| CPU-only, F16 KV | about 208 tok/s | about 3.7 tok/s |
| CPU-only, Q8 KV | about 260 tok/s | about 4.36 tok/s |
| Full GPU offload, F16 KV | about 1,285 tok/s | about 39 tok/s |
| Full GPU offload, Q8 KV | about 1,233 tok/s | about 38 tok/s |

Conclusion: 12B Q4 fits cleanly in VRAM and is comfortable on this machine.

### Larger-than-VRAM: Gemma 4 31B QAT Q4_0

Model: `models/gemma-4-31b-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf`

File size: 17.65 GB, model size reported by llama.cpp: 16.42 GiB. This cannot fit cleanly in 12 GB VRAM.

| Setup | Prompt Processing | Generation |
| --- | ---: | ---: |
| CPU-only, Q8 KV, `-ngl 0` | about 41 tok/s | about 1.55 tok/s |
| Naive full offload/spill, `-ngl -1`, Q8 KV | about 41 tok/s | about 1.40 tok/s |
| Partial offload, `-ngl 30`, Q8 KV | about 79 tok/s | about 3.02 tok/s |
| Partial offload, `-ngl 36`, Q8 KV | about 96 tok/s | about 3.39 tok/s |
| Partial offload, `-ngl 40`, Q8 KV | about 38 tok/s | about 4.00 tok/s |
| Partial offload, `-ngl 44`, Q8 KV | about 40 tok/s | about 3.34 tok/s |
| Partial offload, `-ngl 40`, F16 KV, 2048 prompt | about 35.9 tok/s | about 3.84 tok/s |
| Partial offload, `-ngl 40`, Q8 KV, 2048 prompt | about 30.9 tok/s | about 4.09 tok/s |

Conclusion: the first real win is spill avoidance. Letting llama.cpp attempt full GPU offload on a too-large model runs, but it is slower than CPU-only in this test. A measured partial-offload point around `-ngl 40` gives roughly 2.6x better generation throughput than CPU-only and roughly 2.9x better throughput than naive full-offload spill.

### Negative Result: BitNet 2B

Model: `models/bitnet-b1.58-2b-4t-gguf/ggml-model-i2_s.gguf`

Current llama.cpp `b9804` does not load this GGUF. The load error reports a removed GGUF tensor type and points toward runtime incompatibility. Microsoft also says BitNet efficiency requires `bitnet.cpp`, not standard Transformers. Treat BitNet as a separate runtime track.

### Negative Result: Broad Tensor Overrides

llama.cpp accepts tensor placement overrides with buffer types `CPU` and `CUDA0`. Single-tensor overrides work. Broad Gemma 31B attention placement crashed a CUDA assertion for some tensor groups, and narrower all-query/all-ffn-down placements did not beat layer-prefix offload. This is still promising, but it likely requires a runtime patch or a more careful tensor-layout-aware placement planner.

## Proposed Original Approach: Spill-Aware Hybrid Scheduler

The immediate better approach is not "offload as much as possible." On this machine, that is measurably worse.

Use a scheduler with four tiers:

1. **Fit probe:** benchmark a small grid of `-ngl`, KV type, context size, and batch size for each model. Choose the fastest generation setting that does not trigger unified-memory spill behavior.
2. **Spill guard:** reserve VRAM headroom instead of filling the GPU. For the 31B Q4 model, the best observed point is around `-ngl 40`, not `-ngl -1`.
3. **KV policy:** default to Q8 KV for long contexts when generation is the priority; keep F16 KV available when prompt ingestion is more important.
4. **Tensor planner track:** use GGUF tensor metadata to explore non-contiguous placement, but patch/validate llama.cpp before relying on broad `--override-tensor` strategies for Gemma 31B.

For MoE-class models such as GLM-5.2, extend the same scheduler with:

- expert residency cache: keep hot experts on GPU, cold experts in CPU RAM or NVMe;
- router-logit prefetch: predict likely next-token experts and prefetch asynchronously;
- active-parameter budget: optimize for active experts, not total model size;
- draft or MTP decoding: use a smaller model/head to reduce expensive target-model calls.

GLM-5.2 is a 744B-A40B / roughly 743B-total, 39-40B-active MoE with up to 1M context in current Z.ai/vLLM documentation. Even FP8 weights are far beyond this machine's 32 GB RAM if loaded naively. A local GLM-5.2 attempt therefore needs expert paging plus KV compression; ordinary GGUF-style dense partial offload is not enough.

This would not make a 743B-class MoE model fast on 32 GB RAM, but it is the plausible path to make partial or research-grade local inference possible where naive loading is impossible.

## Practical Recommendations

For Gemma 4 12B Q4_0:

```powershell
.\tools\llama.cpp-b9804-cuda124\llama-cli.exe `
  -m .\models\gemma-4-12b-it-qat-q4_0-gguf\gemma-4-12b-it-qat-q4_0.gguf `
  -ngl -1 -ctk q8_0 -ctv q8_0 -c 4096 -n 512
```

For Gemma 4 31B Q4_0:

```powershell
.\tools\llama.cpp-b9804-cuda124\llama-cli.exe `
  -m .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  -ngl 40 -ctk q8_0 -ctv q8_0 -c 4096 -n 512
```

Avoid `-ngl -1` for 31B on this GPU. It runs, but the benchmark shows it is slower because it spills.

To profile a new model:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\profile-llama-fit.ps1 `
  -Model .\models\gemma-4-31b-it-qat-q4_0-gguf\gemma-4-31B_q4_0-it.gguf `
  -GpuLayers "0,20,30,36,40,44,-1" `
  -CacheTypes "q8_0" `
  -PromptTokens 128 `
  -GenTokens 64
```

## Source Pointers

- llama.cpp: https://github.com/ggml-org/llama.cpp
- Google Gemma 4 12B QAT Q4_0 GGUF: https://huggingface.co/google/gemma-4-12b-it-qat-q4_0-gguf
- Google Gemma 4 31B QAT Q4_0 GGUF: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf
- Microsoft BitNet runtime: https://github.com/microsoft/BitNet
- Microsoft BitNet 2B GGUF model card: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf
- Z.ai GLM-5.2 model card: https://huggingface.co/zai-org/GLM-5.2
- Z.ai GLM-5 GitHub model table including GLM-5.2: https://github.com/zai-org/GLM-5
- vLLM GLM-5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- vLLM quantized KV-cache docs: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vLLM offload config: https://docs.vllm.ai/en/stable/api/vllm/config/offload/
- Hugging Face quantization docs: https://huggingface.co/docs/transformers/quantization
- Hugging Face Accelerate big-model inference: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- FlexGen paper: https://arxiv.org/abs/2303.06865
- PowerInfer paper: https://arxiv.org/abs/2312.12456
- KIVI KV-cache quantization: https://arxiv.org/abs/2402.02750
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
