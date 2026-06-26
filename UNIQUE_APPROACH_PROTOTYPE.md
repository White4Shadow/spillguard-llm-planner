# Unique Approach Prototype: Phase-Split SpillGuard

Date: 2026-06-26

## Problem

The first experiment showed that a larger-than-VRAM model can run on this machine, but only if we avoid naive GPU oversubscription. For Gemma 4 31B Q4_0, `-ngl 40` beat both CPU-only and naive full-offload spill.

That is useful, but it is still static tuning. The next step is a more unique approach: make placement a phase-level scheduling problem instead of one fixed launch setting.

## Idea

**Phase-Split SpillGuard** separates inference into two phases:

1. **Prefill phase:** process the long prompt with a placement optimized for prompt throughput or memory safety.
2. **Decode phase:** restore the saved KV/slot state into a second placement optimized for token generation.

This uses llama.cpp server slot save/restore as the handoff mechanism. The important property is that the saved slot can survive a process restart and be restored by a server launched with a different `-ngl` placement, as long as the model and KV cache type are compatible.

This is different from ordinary manual offload tuning because one request is no longer bound to one static model placement profile.

Related work exists: systems such as DistServe and vLLM disaggregate prefill and decode for serving. The unique claim here is narrower and more defensible: use llama.cpp slot serialization on a weak single desktop to switch CPU/GPU placement profiles between phases, specifically to avoid VRAM spill and exploit different prefill/decode optima for models that do not cleanly fit in VRAM.

## Prototype

Script:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\phase-split-llama.ps1 `
  -Model .\models\qwen2.5-0.5b-instruct-gguf\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -PrefillGpuLayers "0" `
  -DecodeGpuLayers "-1" `
  -CacheType "q8_0" `
  -NewTokens 8 `
  -CompareCold
```

What it does:

- starts a prefill `llama-server`;
- evaluates the prompt with `n_predict=0`;
- saves slot 0 to disk;
- stops the server;
- starts a decode `llama-server` with a different placement;
- restores slot 0;
- decodes new tokens using the restored prompt state;
- optionally starts a cold decode baseline.

## First Proof: Qwen2.5 0.5B

On Qwen2.5 0.5B Q4_K_M, a slot created under `-ngl 0` was restored under `-ngl -1`.

Observed scripted test:

- prefill tokens evaluated: 1501
- saved slot tokens: 1501
- restored slot tokens: 1501
- restored decode prompt time: about 9.1 ms
- cold decode prompt time: about 87.6 ms
- restored decode elapsed: about 34.8 ms
- cold decode elapsed: about 137.1 ms
- cached decode saving: about 102 ms
- one-off phase split delta including server reloads: about +1.25 s
- break-even loaded reuses: about 3 decodes

This proves that cross-placement KV/slot handoff is possible at least for the tested model and cache type.

## Larger Proof: Gemma 4 31B Q4_0

Test:

- prefill profile: `-ngl 36`, Q8 KV
- decode profile: `-ngl 40`, Q8 KV
- prompt chars: 8400, about 1502 evaluated tokens
- generated tokens: 8

Observed:

- prefill server ready: about 8.7 s
- prefill elapsed: about 6.2 s
- slot save size: about 511 MB
- decode server ready: about 7.2 s
- restore elapsed: about 0.11 s
- restored decode prompt time: about 40.3 s
- cold decode prompt time: about 43.3 s
- restored decode elapsed: about 42.8 s
- cold decode elapsed: about 45.9 s
- cached decode saving: about 3.1 s
- one-off phase split delta including server reloads: about +12.7 s
- break-even loaded reuses: about 3 decodes

This is not yet an end-to-end one-off win. It is evidence that the handoff works for the 31B larger-than-VRAM model and can reduce cached decode time, but the current process-restart prototype pays too much load overhead for a single request.

## Why It Could Matter

For Gemma 4 31B Q4_0, the measured static optima are not identical:

- prompt processing was strongest near `-ngl 36` in the short-prompt sweep;
- generation was strongest near `-ngl 40`;
- naive `-ngl -1` spilled and lost badly.

The phase-split approach can exploit that mismatch when:

- the same long prompt is reused for multiple generations;
- prefill is performed once and decode is repeated;
- two machines or two sequential local profiles are acceptable;
- a future runtime patch allows placement changes without full process restart.

## Limitations

- Sequential process restart adds large overhead for one-off prompts.
- Running two large-model servers at once is not realistic on 32 GB RAM.
- The current prototype depends on llama.cpp server slot serialization.
- The approach needs longer-context and repeated-decode benchmarking before claiming an end-to-end win on Gemma 31B.
- Tensor-level non-contiguous placement remains unstable for Gemma 31B in the current binary; broad attention overrides triggered CUDA assertions.

## Next Experiment

The next experiment is not another simple one-off run. It is either:

- repeated-decode testing with one saved long prompt, to validate the observed break-even point of about 3 reuses; or
- a llama.cpp runtime patch that changes placement profile after prefill without restarting and reloading the model.

The key metric is not only tok/s; it is break-even point:

```text
saved_prompt_time - (save_time + restore_time + extra_model_reload_time)
```

If future runtime work removes model reload, the phase split becomes substantially more attractive.

## Source Pointers

- llama.cpp server: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- llama.cpp slot/cache options are exposed through `llama-server --help` in the local `b9804` build.
- vLLM disaggregated prefill examples: https://docs.vllm.ai/
- DistServe: https://arxiv.org/abs/2401.09670
