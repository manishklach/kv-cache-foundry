# CUDA-graph serving profile on H100

## Question

Does the earlier eager-mode vLLM profile's apparent matrix-multiply dominance persist with CUDA graphs enabled? This is a diagnostic follow-up, not an optimization or an A/B speedup test.

## Setup

- Runpod Secure Cloud: one NVIDIA H100 80GB HBM3 in US-CA-2; Pod `knq7jj6hc3vqlx`; listed rate $4.39/hour; 40 GB temporary container disk.
- Device capture: 81,559 MiB, UUID `GPU-3ee0edae-03ec-7c75-1842-8c052cecfe55`, PCI bus `00000000:19:00.0`; driver 595.112.07. The end-of-run idle snapshot was 33°C and 127.23 W.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; vLLM 0.32.0; PyTorch 2.13.0+cu130; PyTorch CUDA runtime 13.0; Nsight Systems 2025.6.3.541.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, BF16, max model length 4,096, max 8 sequences, GPU memory utilization 0.55. The resolved model revision/hash was not preserved.
- CUDA graphs were enabled (`enforce_eager=False`); vLLM logged `FULL_AND_PIECEWISE`, capture sizes `[1, 2, 4, 8, 16]`, and completed graph capture. `nsys profile` used CUDA graph node tracing and the vLLM CUDA profiler start/stop API.
- Workload: eight simultaneous OpenAI-compatible chat requests, identical prompt, greedy (`temperature=0`), `max_tokens=32`. Eight warmups plus four rounds of eight measured requests; all 32 measured responses returned HTTP 200. Mean client-observed request wall time was 1.2022 s across the concurrent requests. This is not token latency or a throughput comparison.

## Nsight Systems observations

The captured GPU kernel summary contained 125.4 ms of aggregate kernel duration. Its largest kernel families were:

| Kernel name | Instances | Aggregate duration | Share of GPU kernel duration |
| --- | ---: | ---: | ---: |
| `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 1,044 | 40.009 ms | 31.9% |
| `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 2,088 | 31.073 ms | 24.8% |
| `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 1,044 | 14.675 ms | 11.7% |
| FlashAttention forward SM90 CUTLASS kernel | 1,044 | 10.478 ms | 8.3% |
| `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT` | 30 | 8.197 ms | 6.5% |

The top three `nvjet_sm90_tst_*` names account for 68.4% of the reported kernel duration. The trace confirms graph launches were exercised: `cudaGraphLaunch` was the leading CUDA API by aggregate API time (29.117 ms, 48.2% of CUDA API time). That API share is not a share of GPU kernel time.

## Interpretation and limits

This trace shows a dominant family of tiled `nvjet_sm90` kernels plus FlashAttention. It does **not** directly label those kernels as the same PyTorch `aten::mm` operations counted by the earlier Torch profile. Therefore, the old 75.35% `aten::mm` observation is not yet independently confirmed under graphs, and it would be too strong to call these Nsight rows GEMMs without kernel/source correlation. The result does make the next profiling target more concrete: identify the `nvjet_sm90` kernels by operation and correlate their time with model phases before attempting a replacement.

The previous profile used eager execution, Torch Profiler, and a different H100 Pod. This run used CUDA graphs and Nsight Systems; the workloads are similar but not a controlled profiler A/B. The Nsight `.nsys-rep` was stored on the ephemeral Pod and was not exported before cleanup; this committed report preserves the extracted aggregate rows, not the raw trace. One model, one request shape, one concurrency level, one run, and an unpinned model revision limit generalization. No latency improvement or kernel optimization is claimed.

## Cleanup

The server and Nsight capture were stopped after the trace was summarized. Pod `knq7jj6hc3vqlx` was terminated; a subsequent Runpod `get_pod` returned `404 pod not found`.
