# H100 prefix-cache calibration (2026-11-15)

## Purpose

This was a small real-GPU calibration run for the Foundry cost model: compare cold prefill with an exact repeated prefix-cache hit at four prefix lengths. It does **not** evaluate the slack-priced policy, modify vLLM, or prove a serving speedup. The CSV records client-observed streaming TTFT and total request time through the Runpod HTTP proxy.

## Exact environment

- Provider / location: Runpod, US-NE-1 Secure Cloud
- GPU: 1 x NVIDIA H100 SXM, 80 GB class (Runpod SKU)
- Host: Intel Xeon Platinum 8470, 26 vCPU, 251 GB RAM
- Container disk: 30 GB temporary
- Container image: `vllm/vllm-openai:v0.32.0`
- vLLM: 0.32.0 (confirmed by response system fingerprint `vllm-0.32.0-df1d9140`)
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Dtype: BF16; eager mode; GPU memory utilization 0.50; max model length 4,608
- Prefix caching and prompt-token details enabled
- Exact NVIDIA driver and CUDA runtime versions were not captured for this run.
- Displayed price: $4.39/GPU-hour plus $0.004/hour for the 30 GB container disk. The Pod uptime shown during teardown was 3m35s; this implies about $0.26 compute and under one cent of disk storage. This is a rate-times-uptime estimate, not a final invoice amount.

## Method

`gpu_calibrate.c` is a C11 client. It sends synthetic token-ID prompts with prefix lengths 512, 1,024, 2,048, and 4,096, followed by one generated token. For each size and each of two repeats, it sends one cold request with a unique cache salt and a warm request with an exact repeated prefix and salt. The warm response must report the full prefix as cached; cold requests must report zero cached tokens. Prompt token counts are one greater than the configured prefix because the request includes a final token after the shared prefix. The cold and warm cases use different synthetic token sequences, so their TTFT comparison is not paired on the same prompt. Only the final warm hit is recorded; a preceding request primes that warm salt when needed.

The first attempt established the 512- and 1,024-token cases but the client rejected the larger response because a vLLM SSE metadata line exceeded its fixed buffer. The client buffer was enlarged, and the calibration was rerun with run-unique cache salts. The CSV contains only the complete rerun. The prefix-cache values verified all 8/8 warm requests as full hits and all 8/8 cold requests as misses.

## Results

Mean client-observed TTFT across only two observations per cell:

| Shared prefix | Cold TTFT | Warm TTFT | Difference | Warm cached tokens |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1,061.2 ms | 1,276.0 ms | +20.2% | 512 / 512 on both repeats |
| 1,024 | 625.3 ms | 616.1 ms | -1.5% | 1,024 / 1,024 on both repeats |
| 2,048 | 1,250.7 ms | 704.7 ms | -43.7% | 2,048 / 2,048 on both repeats |
| 4,096 | 1,085.8 ms | 1,215.9 ms | +12.0% | 4,096 / 4,096 on both repeats |

The result confirms cache accounting and a working exact-prefix hit on this H100/vLLM setup. The observed TTFT is noisy and non-monotonic, with only two observations per cell; in particular, three of four warm means are not faster. These measurements are **not evidence of an optimization or a GPU performance claim**. Proxy/network overhead, vLLM scheduling, eager execution, the unpaired prompts, and the very small sample may dominate. The client reads streamed output one byte at a time and uses a fixed 256 KiB SSE-line buffer; this run exposed and fixed the earlier 8 KiB buffer limit. Follow-up calibration should use more repeats, local-in-Pod timing, randomized condition order, and production-like server settings before fitting latency costs.

## Reproduction

On Linux with GCC/Clang and `curl`:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic foundry/gpu_calibrate.c -o gpu_calibrate
./gpu_calibrate https://HOST-8000.proxy.runpod.net MODEL API_KEY 2 > foundry/results/h100_sxm_prefix_calibration.csv
```

The API key should be supplied from a secret store or environment-specific secure input and must not be committed. The temporary Runpod Pod was terminated after the measurements were saved.
