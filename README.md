# KV Cache Foundry

KV Cache Foundry develops and tests KV-cache policies for LLM inference. It includes live vLLM allocator changes, C experiments on Runpod H100 and H200 GPUs, and a C/C++ llama.cpp prototype that adjusts KV transfers in response to decode latency.

## Unified cost model prototype

The first CPU-only piece of the proposed unified decision engine is now in [`foundry/`](foundry/README.md). The C11 model provides per-GPU, per-prefix EWMA estimates for reload, recompute, and keep costs, with an explicit prior and nearest-prefix fallback. It is a prototype, not yet calibrated from live GPU measurements, and it does not yet implement cache policies. New core decision code remains in C. The current 327 anomaly has been grounded against the archived traces and has a corrected, staged probe plan in [`docs/327-probe-plan.md`](docs/327-probe-plan.md). A live GPU run is deferred: the Runpod account had $0.38 credit at the latest check, insufficient for a useful, bounded session including model startup.

| Experiment | Validation | Change | Result |
| --- | --- | --- | --- |
| [Prompt order](#h200-prompt-order) | H200 SXM | Put the shared manual before the question and enable prefix caching | 33.3% lower median time to first token with the manual first |
| [Adaptive cache admission](#h100-adaptive-cache-admission) | H100 SXM | Patch vLLM's KV cache manager to switch admission rules using recent missed-token counts | 8.8% fewer uncached prompt tokens and 6.2% lower median time to first token |
| [Cost-ranked block eviction](#h100-cost-ranked-block-eviction) | H100 SXM | Rank free KV blocks in C using estimated rebuild time and age | 3.0–5.0% fewer uncached tokens and 1.8–5.5% higher request throughput across two orders |
| [Reload versus recompute](#h200-reload-versus-recompute) | H200 SXM | Select the cheaper path using prefix length and copy backlog | 19.2% lower summed time than always reloading across the held-out synthetic grid |
| [Decode-aware KV reload](#decode-aware-kv-reload-prototype) | H200 SXM | Pace KV writes using live decode latency and allow prefix recomputation | Higher p95 first-token latency in all six settings. All 648 requests matched the reference tokens. |
| [Deadline-aware KV prefetch](#deadline-aware-kv-prefetch) | RTX PRO 6000 Blackwell | Overlap per-layer KV copies with attention and compare FIFO against deadline ordering | Layer prefetch cut median latency by about 18% for both prefix lengths. All three policies met 27/72 targets. |

## Deadline-aware KV prefetch

The CUDA benchmark compares full restore, FIFO layer prefetch, and deadline ordering on the same synthetic attention workload. On the RTX PRO 6000 Blackwell, FIFO prefetch reduced median completion latency from 53.98 to 44.51 ms at 512 tokens and from 52.01 to 42.44 ms at 4,096 tokens. Deadline ordering reached 44.12 ms and 42.41 ms, respectively. All three policies met the same 27 of 72 request targets, and all 216 requests passed bitwise attention checks. The initial H200 run used a build with a CUDA error-checking bug and is recorded as preliminary. [Benchmark code, results, and reproduction](experiments/deadline-kv-prefetch/README.md)

## H200 reload versus recompute

Restoring a short prefix from host memory can cost more than rebuilding it when the GPU copy queue is busy. The C benchmark injected a serial host-to-device copy backlog. Recompute first won at 1 GiB for 128 tokens and at 2 GiB for 512 tokens. Reload still won at the largest tested backlog, 4 GiB, for 2,048 and 4,096 tokens. On 224 held-out pairs, a calibrated selector reduced summed time from 6,005.96 ms to 4,850.48 ms. The benchmark isolates the two paths and assumes prefill can proceed independently of the copy backlog. [Methodology, raw results, and reproduction](experiments/reload-vs-recompute/README.md)

## Decode-aware KV reload prototype

A C controller and C++ llama.cpp patch pace actual host-to-device KV restore batches using the latency of a concurrent decoder. The runner also chooses when to rebuild the cached prefix. It compares stock restore, a fixed batch size, and the adaptive policy with a generated-token check. On the H200, adaptive reload raised background p95 first-token latency in all six settings. At 4,096 tokens with a 2 ms target, foreground p95 inter-token latency improved from 4.90 to 4.77 ms while background p95 first-token latency rose from 17.17 to 699.88 ms. Fixed 4 MiB batches tracked stock closely. All 648 requests matched all eight greedy reference tokens. [Results, figure, and raw logs](experiments/decode-aware-reload/h200-results.md) · [Implementation and reproduction](experiments/decode-aware-reload/README.md)

## H200 output mismatch replay

The focused audit revisited five requests from the H100 admission and eviction traces on one H200 with vLLM 0.32.0. All 40 uncached observations matched the cache-disabled output. Twelve of 20 full-prefix hits changed the greedy token, including six under stock LRU. Request 858 returned token 198 on misses and token 84 on hits under both LRU and cost-ranked eviction. The top two reported log probabilities tied in each warm run. The results support sensitivity to near-tied scores when reusing cached KV. The isolated probes leave behavior under the original eviction pressure open. [Results and reproduction](experiments/adaptive-admission/README.md#focused-h200-replay) · [Raw observations and logs](experiments/adaptive-admission/results/h200-output-audit/)

## H100 cost-ranked block eviction

A live vLLM block allocator patch calls a C ranking function when a new allocation would reuse cached blocks. It ranks uncached blocks first, then ranks cached blocks by age plus estimated rebuild time per token. Rebuild costs come from cold versus full-hit latency measurements on the earlier H100 run. The policy applies [GreedyDual-Size](https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/cao.html) to individual KV blocks.

The live benchmark ran LRU, adaptive admission, and cost-ranked eviction in forward and reverse order. Each trial served the same 512 requests at concurrency four and had 32,800 tokens of reported KV cache capacity.

| Cost-ranked eviction vs. LRU | Forward | Reverse |
| --- | ---: | ---: |
| Uncached prompt tokens | 633,056 → 614,208 | 631,168 → 599,808 |
| Requests per second | 34.54 → 36.43 | 35.28 → 35.92 |
| p95 time to first token | 206.18 → 199.89 ms | 196.55 → 203.14 ms |

Throughput improved in both trials. The p95 result changed direction. Output agreement with the first LRU trial was 507/512 and 506/512 for cost-ranked eviction, and 507/512 for the second LRU trial. The cause of these differences remains unresolved.

The whole-prefix replay separately tests the same cost model at three cache sizes across 20 synthetic traces. It also verifies that the C implementation agrees with the Python reference on every replay decision. [Reproduction steps and all six trials](experiments/adaptive-admission/README.md#concurrent-cost-ranked-block-eviction) · [Live traces and logs](experiments/adaptive-admission/cost-eviction-live-results.tar.gz) · [Replay data](experiments/adaptive-admission/cost-replay.csv)

## H100 adaptive cache admission

When many prompts use a prefix only once, admitting every prefix can evict useful KV blocks. Waiting until a prefix appears twice protects recurring prefixes, but it can also miss the only opportunity for reuse in traffic where each prefix appears exactly twice.

The engine change is in vLLM's KV cache manager. It gates insertion of newly computed blocks into the reusable cache. Two metadata-only shadow caches estimate how many prompt tokens would miss under first-hit and second-hit admission. Every 32 benchmark requests, the patch picks the rule with fewer missed tokens for the next window, keeping the current rule on ties. It uses only prior requests. A 4,096-token miss therefore counts eight times as much as a 512-token miss. vLLM continues to handle GPU block allocation, cache lookup, and LRU eviction.

Adaptive caching and reuse-based admission have precedents. The contribution here is a reproducible vLLM patch that selects between two admission rules using token-weighted misses, together with a GPU replay that shows where it helps and where it costs more. [ARC prior work](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache) · [vLLM prefix caching design](https://docs.vllm.ai/en/v0.32.0/design/prefix_caching/)

The H100 served 1,024 sequential requests per mode, using 32-token blocks and 32,800 tokens of reported KV cache capacity in both runs. The synthetic workload alternates recurring prefixes mixed with one-off traffic and groups of prefixes that appear exactly twice. Each request generates one token.

| H100 metric | Standard admission | Adaptive admission |
| --- | ---: | ---: |
| Uncached prompt tokens | 1,243,936 | 1,134,688 |
| Median time to first token | 36.12 ms | 33.90 ms |
| p95 time to first token | 76.44 ms | 76.64 ms |
| Median time to first token in one-off traffic phases | 36.04 ms | 29.52 ms |
| Median time to first token in paired traffic phases | 36.12 ms | 37.44 ms |

The generated token matched on 1,020 of 1,024 requests. The four differences are unresolved. The run used eager execution, one synthetic trace, a fixed run order, and no request concurrency. The latency figures describe that replay. The separate 20-seed simulation found fewer recomputed tokens at 16,384 and 32,768 tokens of cache capacity and more at 65,536 tokens.

Run `bash experiments/adaptive-admission/run_gpu.sh` on a compatible Linux NVIDIA GPU host with Python 3.12, Bash, and curl. The experiment directory includes the patch, workload generator, runner, simulation, and raw GPU traces with server logs. [Methodology and reproduction](experiments/adaptive-admission/README.md) · [H100 results archive](experiments/adaptive-admission/kv-admission-gpu-results.tar.gz)

## H200 prompt order

Putting a shared manual before the question let vLLM reuse a median of **3,888 prompt tokens** and reduced median time to first token from **72.4 ms to 48.3 ms** (33%) for that prompt layout. The manual-first layout scored **47/51** with caching both off and on. The question-first layout scored **51/51** in both modes.

### Results

| Prompt layout | Prefix cache | Median TTFT | p95 TTFT | Median cached tokens | Exact match |
| --- | --- | ---: | ---: | ---: | ---: |
| Question first | Off | 72.3 ms | 74.0 ms | 0 | 51/51 |
| Manual first | Off | 72.4 ms | 73.3 ms | 0 | 47/51 |
| Question first | On | 73.4 ms | 74.5 ms | 32 | 51/51 |
| Manual first | On | **48.3 ms** | **53.4 ms** | **3,888** | 47/51 |

TTFT measures the time from starting the local HTTP request to receiving the first nonempty streamed text. The latency and cache columns use 50 requests per row after the first workload request. Accuracy uses all 51. The requests were sequential and used one synthetic manual of roughly 3,900 prompt tokens. These measurements describe one run per condition.

With caching on, the manual-first layout was about 34% faster at the median than question-first, with four fewer exact matches. The cache-off results show that the accuracy difference also occurred without prefix reuse. See the [results analysis](docs/results.md) for the individual misses and what the benchmark can establish.

### Reproduce

The recorded run used one NVIDIA H200 SXM on Runpod, Qwen3-4B-Instruct-2507, and vLLM 0.32.0. On a Linux GPU machine with a compatible NVIDIA driver, Python 3.12, Bash, and curl, run:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
bash run_on_pod.sh
```

The runner installs vLLM, starts a loopback-only server for each cache mode, and writes two JSONL result files plus the server logs. Follow the [reproduction guide](docs/reproduce.md) for deployment settings, output checks, and the install workaround used on Runpod.

### H200 files

- [Benchmark client](benchmark.py) and [GPU runner](run_on_pod.sh)
- [Experimental design and metrics](docs/methodology.md)
- [Reproduction guide](docs/reproduce.md)
- [Results, accuracy misses, and limitations](docs/results.md)
- Raw requests and measurements with caching [off](data/results-off.jsonl) and [on](data/results-on.jsonl)
- [Original results archive](h200-prefix-results.tar.gz) with raw measurements, both vLLM logs, and the runner log

## License

Foundry's original work is available under [Apache 2.0](LICENSE). The llama.cpp patch includes upstream context under the [llama.cpp MIT license](experiments/decode-aware-reload/LICENSE.llama.cpp). Upstream dependencies and downloaded model weights carry their own licenses.
