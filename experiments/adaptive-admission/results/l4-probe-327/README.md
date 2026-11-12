# Request 327: first L4 probe

## Question

The archived concurrent eviction trace has one anomalous observation at request 327: the first LRU trial generated token ID 287, while the other five trials generated token ID 220. All six reported zero cached tokens. This probe asks whether the exact prompt is near a next-token decision boundary when evaluated in isolation on a different GPU/runtime.

## Setup

- One Runpod L4, 24 GB product class; telemetry reported 22 GiB total VRAM.
- Model: `Qwen/Qwen3-4B-Instruct-2507`.
- Server image: Runpod verified `vLLM Latest`, `vllm/vllm-openai:latest`; startup log identified vLLM **0.33.0**.
- Start arguments: `--host 0.0.0.0 --port 8000 --dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.80 --max-model-len 4608`.
- The prompt was regenerated from the repository's seed-0 workload at index 327: phase 1, key 39, 4,096 deterministic token IDs plus final token ID 9327. The exact same token-ID prompt was used for all observations.
- Six sequential completion calls requested one greedy token and top-20 log probabilities: two distinct cache salts, each called once and then repeated twice. This is not a concurrency or batching experiment.

## Result

All six calls generated token ID **220**. In every call, IDs **220** and **287** were tied at log probability `-5.6221370697`; the full top-20 list was identical across calls. The vLLM server log reported an aggregate prefix-cache hit rate of 66.7% after the six calls. Per-response cached-token usage was not available from this vLLM response schema.

Median first-token latency was about 1,719.7 ms for the two first calls under fresh salts and 497.4 ms for the four repeats. This is only descriptive: the first call also paid early server warmup, and the response did not expose per-request cached tokens, so it is not a controlled latency result.

The tie supports near-tie sensitivity as a plausible explanation for the historical 287/220 variation. This run did **not** flip tokens and cannot establish the cause of the H100 anomaly. It used an L4 and vLLM 0.33.0, rather than the historical H100 and vLLM 0.32.0, and it did not recreate the trace's concurrent scheduler batches or allocator pressure.

## Hardware and cost

See [`metadata.json`](metadata.json) for the details transcribed from the Runpod Pod Details and Telemetry panels and the vLLM startup log. The pod ran in EU-RO-1 Secure Cloud with 6 vCPU on AMD EPYC 9254, 62 GB host memory, a 30 GB temporary container disk, NVIDIA driver 595.91.07, and CUDA 13.2. No `nvidia-smi` dump, GPU UUID, or PCIe link state was captured.

Runpod displayed $0.69/hour compute and $0.004/hour container storage. At about 3m48s uptime, that corresponds to roughly $0.044 before billing-system rounding. The Pod was stopped after the probe and now shows $0.00/hour. Its temporary container disk was not persistent; all result files were saved locally in this directory.

## Files

- [`probe.jsonl`](probe.jsonl): six raw streamed observations, including the top-20 token log probabilities and per-request timings.
- [`metadata.json`](metadata.json): hardware, runtime, configuration, and billing-rate metadata.
