# H100 decode linear-backend A/B: automatic versus CUTLASS

## Result in brief

The earlier profile's `nvjet_sm90_tst_*` kernels are consistent with cuBLAS/cuBLASLt decode GEMMs. In particular, the `splitK` family is identified as cuBLAS split-K GEMM for small-M decode in a vLLM-derived H100 trace discussed in [Accel-Sim issue 561](https://github.com/accel-sim/accel-sim-framework/issues/561). We tested whether vLLM's forced CUTLASS linear backend could replace that path.

It did not. The H100 Nsight traces under `linear_backend=auto` and `linear_backend=cutlass` contain the same `nvjet_sm90_tst_*` kernels with the same counts and virtually identical aggregate duration. The model was BF16; vLLM 0.32's CUTLASS linear kernel registry covers quantized kernels (FP8, INT8, NVFP4, W4A8), not general BF16 linear layers ([vLLM v0.32.0 source](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/__init__.py)). The config was accepted, but it did not switch these BF16 GEMMs.

The serving A/B found no useful gain. After excluding the same first-wave startup spike in each arm, CUTLASS averaged 1.60% lower output-token throughput across the two blocks and 1.94% higher median request latency. These differences are small and the experiment was not run in reverse order, so treat them as no demonstrated improvement, not evidence that CUTLASS is inherently slower. All measured requests returned 64 output tokens, and all four blocks had the same greedy response-content hash.

## Hardware and software

- Runpod Secure Cloud, Pod `75rl793blhf5bh`, US-CA-2, listed $4.39/hour; 40 GB ephemeral container disk. The Pod uptime at cleanup was 839 seconds (~14 minutes); rate-based cost estimate is about $1.02.
- GPU: NVIDIA H100 80GB HBM3; 81,559 MiB; UUID `GPU-d6620f1a-7d52-858b-870d-74bf3ae10d9b`; PCI bus `00000000:CB:00.0`; driver 595.91.07. Final idle snapshot: 46°C, 77.05 W.
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; host CUDA 13.2; vLLM 0.32.0; PyTorch 2.13.0+cu130, CUDA runtime 13.0; Nsight Systems 2025.6.3.541.
- Model `Qwen/Qwen3-4B-Instruct-2507`, BF16, revision `main` (commit hash not preserved), max model length 4,096, max sequences 8, GPU memory utilization 0.55. CUDA graphs enabled (`FULL_AND_PIECEWISE`, sizes 1, 2, 4, 8, 16).
- The source and raw request rows are C-only: [`linear_backend_ab.c`](../linear_backend_ab.c) and the four CSVs linked below. The C client used libcurl multi, sent eight simultaneous identical chat requests per wave, greedy decoding, `max_tokens=64`; each block had eight waves (64 requests). Eight additional requests warmed each server before the measured blocks.

## Serving results

The first measured wave in every block contained a single request taking about 4.3–4.5 seconds while the remaining seven waves completed in roughly 0.32–0.36 seconds. This first-wave spike appeared in both backends. To make the steady-state comparison explicit, the table below drops wave 0 and reports the remaining 56 requests per block. The per-request CSVs preserve all 64 requests, including the spike.

| Backend block | Requests after warm wave | Completion tokens | Median latency | p95 latency | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| auto-a | 56 | 3,584 | 337.745 ms | 347.922 ms | 1,515.442 |
| auto-b | 56 | 3,584 | 333.647 ms | 353.874 ms | 1,513.793 |
| cutlass-a | 56 | 3,584 | 342.675 ms | 362.510 ms | 1,471.951 |
| cutlass-b | 56 | 3,584 | 341.749 ms | 351.157 ms | 1,510.175 |

Combined by measured wave duration, `auto` produced 1,514.6 output tokens/s and `cutlass` 1,490.4 output tokens/s (-1.60%). The two auto blocks' median latency averaged 335.696 ms; the two CUTLASS blocks averaged 342.212 ms (+1.94%). These are small, single-Pod observations, and the order was auto followed by CUTLASS, not a randomized or reversed sequence.

For completeness, the C client's all-wave summaries (including wave 0) were:

| Block | Full block time | Output tokens/s | Median latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| auto-a | 6.898 s | 593.834 | 338.153 ms | 4,526.578 ms |
| auto-b | 2.725 s | 1,502.903 | 334.334 ms | 354.156 ms |
| cutlass-a | 6.727 s | 608.866 | 343.306 ms | 4,286.666 ms |
| cutlass-b | 2.717 s | 1,507.286 | 339.933 ms | 350.778 ms |

Every response returned HTTP 200, `prompt_tokens=21`, `completion_tokens=64`; content hash `408af883c1a56189` was identical across every request in all four blocks. Raw per-request records: [auto-a](linear-backend-auto-a-2026-11-17.csv), [auto-b](linear-backend-auto-b-2026-11-17.csv), [cutlass-a](linear-backend-cutlass-a-2026-11-17.csv), [cutlass-b](linear-backend-cutlass-b-2026-11-17.csv).

## Nsight kernel comparison

Each Nsight capture covered the two 64-request blocks for its backend, after the server was warmed. Aggregate kernel duration from `cuda_gpu_kern_sum`:

| Kernel family | Auto instances | Auto duration | CUTLASS instances | CUTLASS duration |
| --- | ---: | ---: | ---: | ---: |
| `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` | 2,088 | 79.983 ms | 2,088 | 79.952 ms |
| `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` | 4,176 | 62.019 ms | 4,176 | 61.962 ms |
| `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` | 2,088 | 29.354 ms | 2,088 | 29.368 ms |
| FlashAttention forward SM90 CUTLASS | 2,124 | 21.144 ms | 2,124 | 21.010 ms |

The top three NVJet rows sum to 171.357 ms under auto and 171.282 ms under CUTLASS (-0.04%). Their identical names and instance counts show that the forced backend did not replace the measured BF16 decode GEMM kernels. The raw `.nsys-rep` files were on the ephemeral Pod and were not exported; the aggregate rows above were extracted before cleanup.

## What this does and does not establish

- It traces the `splitK` kernel family to cuBLAS small-M decode GEMMs using a separate vLLM-derived H100 trace analysis. The exact mathematical shapes of these individual calls were not preserved in our Nsight report.
- It rejects **this configuration change** as an optimization for BF16 Qwen3-4B on vLLM 0.32. It does not reject CUTLASS for quantized models, nor does it show that cuBLAS is already optimal.
- No custom CUDA kernel was implemented or integrated, and no serving speedup is claimed.
- A useful next kernel experiment must target the actual BF16 decode shapes and compare against cuBLAS with correctness, randomized run order, and a repeatable latency/throughput gate. Do not spend another GPU run on `--linear-backend cutlass` for this BF16 model.

## Cleanup

Pod `75rl793blhf5bh` was terminated after extracting summaries and request CSVs. Runpod `get_pod` returned `404 pod not found`. The profile files and model cache on the ephemeral disk were deleted with the Pod.
