# AMD MI350X prefix-cache calibration (2026-11-18)

## Purpose and conclusion

This short run checked whether the repository's C11 prefix-cache client and Qwen3 workload run on an AMD accelerator through ROCm/vLLM. It is a hardware and serving-path calibration, not an optimization experiment. The 16 measured requests returned the expected cache accounting: eight cold requests had zero cached tokens, and eight warm requests reported the full configured prefix as cached.

With only two observations per condition, warm-request mean TTFT was lower than cold-request mean TTFT at all four tested prefix lengths. Treat those differences as a small, noisy calibration result, not a reliable speedup. In particular, this run cannot be compared directly with the earlier H100 data: this run used a newer vLLM release and measured locally inside the Pod, while the H100 measurements traversed the Runpod HTTP proxy.

## Exact environment

- Provider: Runpod Secure Cloud
- Pod name / ID: `amd-mi350x-qwen3-prefix-calibration` / `8cjwfay1neb6vy`
- GPU catalog SKU: `AMD Instinct MI350 OAM` (catalog name `MI350X`), one GPU
- Device memory: vLLM startup reported 287.98 GiB total and 286.49 GiB free before model allocation
- Host allocation: 32 vCPUs; the create request required at least 283 GB host RAM. Runpod did not return a CPU model or the selected data-center ID (`dataCenterId` was null), so those details are unknown.
- Container image: `vllm/vllm-openai-rocm:latest`, image digest `sha256:3cb773d23f063afa86291ef75773cdd486154ddea94c213300f4ed7cc42b7af1`
- Runtime: vLLM `0.34.0+rocm100`, ROCm 10.0 image
- Model: `Qwen/Qwen3-4B-Instruct-2507`, BF16
- Server settings: eager mode, prefix caching enabled, prompt-token details enabled, max model length 4,608, GPU memory utilization 0.50
- Storage: 50 GB ephemeral container disk; no persistent or network volume
- Network: no exposed ports; client and server communicated over loopback inside the Pod
- Runpod GPU rate shown at creation: `$5.49/GPU-hour`. The Pod was terminated after the run. A `get-pod` uptime snapshot showed 8 minutes; the implied compute charge is about `$0.73`, plus container-disk charges. This is a rate-times-uptime estimate, not the final invoice.
- Runpod reported the Pod absent (HTTP 404) after termination.

## Method

The existing [`gpu_calibrate.c`](../gpu_calibrate.c) C11 client was compiled inside the Pod and ran two repeats at each shared-prefix length: 512, 1,024, 2,048, and 4,096 tokens. Each request generated one token. The warm request reused the exact token-ID prefix and cache salt within its warm sequence. The cold and warm conditions use different synthetic token streams, so the TTFT comparison is not paired on the same prompt. The harness checked that each cold request was a cache miss and each measured warm request was a full prefix hit. Timing was collected inside the Pod using the client's monotonic clock, so it excludes Runpod proxy and client-to-server network delay.

The client source was fetched from the public repository at run time; no benchmark source changes were made for this run. The Runpod command output captured `rocm-smi`, the vLLM version, the raw CSV, and selected vLLM startup diagnostics. No API key or credential was recorded in the result artifact.

## Results

| Shared prefix | Cold mean TTFT (ms) | Warm mean TTFT (ms) | Warm minus cold (ms) | Change |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 62.736 | 25.710 | -37.026 | -59.0% |
| 1,024 | 31.532 | 23.761 | -7.771 | -24.6% |
| 2,048 | 47.954 | 26.720 | -21.234 | -44.3% |
| 4,096 | 85.678 | 33.636 | -52.042 | -60.7% |

Only two observations contribute to each mean, and the cold timings vary notably between repeats at 512 and 2,048 tokens. The means are descriptive. The result establishes that the AMD ROCm runtime served this workload and that prefix-cache accounting worked; it does not demonstrate a novel optimization or establish MI350X versus H100 performance.

## Raw CSV

The exact 16 measured rows are in [`amd-mi350x-prefix-calibration.csv`](amd-mi350x-prefix-calibration.csv). Warm-up requests used internally by the client are not emitted as measured rows.

