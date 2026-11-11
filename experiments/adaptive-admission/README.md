# Adaptive KV prefix admission

A prefix cache with limited space can evict useful KV state when many requests use a document only once. Waiting for a second request before admitting a prefix can protect recurring documents. It also misses the only reuse when a document appears exactly twice.

## H100 engine benchmark

### Concurrent cost-ranked block eviction

The [live allocator patch](live_eviction.py) ranks free blocks with [C code](block_rank.c) when an allocation would reuse cached blocks. Uncached blocks come first. Cached blocks are ranked by age plus estimated rebuild time per token. Queue order breaks ties. vLLM retains responsibility for reference counts and cache hash eviction. This is a block-level adaptation of the whole-prefix replay below.

On one Runpod H100 SXM, each policy served 512 requests per trial at concurrency four. Both orders used the same trace, vLLM 0.32.0, Qwen3-4B-Instruct-2507, eager execution, and 32,800 tokens of reported KV cache capacity.

| Run order | Policy | Uncached tokens | Median TTFT | p95 TTFT | Requests/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Forward | LRU | 633,056 | 95.97 ms | 206.18 ms | 34.54 |
| Forward | Adaptive admission | 670,112 | 92.05 ms | 206.15 ms | 35.88 |
| Forward | Cost-ranked eviction | 614,208 | 98.52 ms | 199.89 ms | 36.43 |
| Reverse | Cost-ranked eviction | 599,808 | 96.18 ms | 203.14 ms | 35.92 |
| Reverse | Adaptive admission | 674,016 | 91.92 ms | 211.06 ms | 37.00 |
| Reverse | LRU | 631,168 | 102.19 ms | 196.55 ms | 35.28 |

Cost-ranked eviction reduced uncached tokens by 3.0% and 5.0% versus LRU and increased request rate by 5.5% and 1.8%. p95 moved in opposite directions. Throughput uses completed requests divided by the sum of the four phase wall times. Against the first LRU trial, the second LRU run matched 507/512 output tokens. The two cost trials matched 507/512 and 506/512. Concurrent arrival order can vary across runs. This test used one synthetic trace and did not isolate the reason for output differences.

Run `bash run_live_eviction.sh` on a compatible Linux NVIDIA GPU host with Python 3.12, Bash, curl, and a C compiler. Analyze the resulting [archive](cost-eviction-live-results.tar.gz) with `python3 summarize_live.py cost-eviction-live-results.tar.gz`.

### Sequential adaptive admission

The [engine patch and replay client](engine_experiment.py) modify the KV cache manager in vLLM 0.32.0. The adaptive mode gates insertion of reusable KV blocks for benchmark requests. The baseline uses the engine's normal admission rule. Both runs use real GPU KV allocation, lookup, and LRU eviction.

| Metric | Baseline | Adaptive |
| --- | ---: | ---: |
| Uncached prompt tokens reported by vLLM | 1,243,936 | 1,134,688 |
| Median time to first token | 36.12 ms | 33.90 ms |
| p95 time to first token | 76.44 ms | 76.64 ms |

The H100 run used Qwen3-4B-Instruct-2507 in eager mode with 1,024 sequential requests per condition, 32-token blocks, and 32,800 tokens of cache capacity reported by vLLM in both modes. Each synthetic prefix has a distinct salt and each request appends a unique final token. Two metadata-only shadow caches score the admission rules every 32 requests. The engine implementation remembers recently seen prefix identities, including cache hits.

During one-off traffic, median latency fell from 36.04 ms to 29.52 ms. During paired traffic it rose from 36.12 ms to 37.44 ms. The generated token matched for 1,020 of 1,024 requests. All four differences occurred on requests with more cached tokens under adaptive admission. This run did not establish the cause. The fixed run order and one synthetic replay limit the latency comparison.

On a Linux NVIDIA GPU machine with Python 3.12, Bash, and curl, run `bash run_gpu.sh`. The runner creates an isolated environment, applies the patch, and writes both per-request traces and server logs to an archive. [Recorded H100 results](kv-admission-gpu-results.tar.gz)

## Simulation

### Eviction based on H100 rebuild cost

The C replay ranks resident prefixes by estimated rebuild time per token plus an aging term, using GreedyDual-Size. For this experiment each KV token occupies the same number of bytes, so ranking by cost per token also ranks by cost per byte. This adapts an established cache policy to KV prefixes. [GreedyDual-Size paper](https://www.usenix.org/legacy/publications/library/proceedings/usits97/full_papers/cao/cao_html/cao.html)

The cost estimates use the difference between median cold and full-hit time to first token for each prefix size in the saved H100 baseline. The estimates are 1.93 ms for 1,024 tokens, 11.10 ms for 2,048, and 38.34 ms for 4,096. The 512-token cost is extrapolated to 0.97 ms because the baseline had no full hits in that bucket. These are end-to-end latency estimates that include serving overhead. The replay assumes a miss rebuilds the whole prefix.

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic cost_cache.c -o /tmp/vllm-cost-cache
python3 cost_replay.py --c-replay /tmp/vllm-cost-cache
```

The C executable reads integer prefix keys and lengths from standard input and emits missed token counts. The Python harness scores 20 synthetic traces with seeds 100–119 and checks every C decision against its reference implementation. The linear-cost control uses the same C algorithm with a constant cost per token and matches LRU on these traces.

| Cache capacity | Estimated rebuild time vs. LRU with H100 costs | Estimated rebuild time vs. LRU with adaptive admission |
| ---: | ---: | ---: |
| 16,384 tokens | 12.2% lower | 10.8% lower |
| 32,768 tokens | 12.2% lower | 22.0% lower |
| 65,536 tokens | 3.9% lower | 5.7% higher |

These are modeled rebuild costs. The C replay operates on whole prefixes. The live benchmark above ranks individual blocks and measures serving performance. [Replay results](cost-replay.csv)

### Admission based on missed tokens

This simulation compares three policies at equal cache capacity. All evict the least recently used resident prefix when space runs out.

- **LRU** admits on every miss.
- **Second-hit admission** admits a missed prefix only if its key is among the 128 most recent distinct missed keys.
- **Adaptive admission** runs both fixed policies on metadata. Every 256 requests it picks the one that missed fewer tokens in the preceding window. The serving cache retains its own state across policy changes. On ties it keeps the current rule, and it begins with first-hit admission.

Run with Python 3 and no dependencies.

```bash
python benchmark.py
```

The script writes [results.csv](results.csv) with one row per capacity, seed, and phase. Each seed alternates between two workloads twice. In the first, seven 4,096-token manuals recur among one-off documents. In the second, groups of eight documents each appear exactly twice. Each phase has 2,048 requests and each capacity uses the same 20 seeds. Prefix identities are disjoint across phases.

| Capacity in tokens | LRU missed tokens | Second-hit missed tokens | Adaptive missed tokens | Adaptive vs. LRU |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 324,929,024 | 314,168,832 | 298,635,776 | 8.1% fewer |
| 32,768 | 199,154,176 | 250,480,128 | 169,842,176 | 14.7% fewer |
| 65,536 | 156,051,968 | 250,480,128 | 165,004,800 | 5.7% more |

At 32,768 tokens, the adaptive policy saved 40.6% of the LRU misses by token count during the one-off traffic phases. It recomputed 12.4% more tokens in the paired phases because it took a window to switch. At 65,536 tokens, LRU already kept most recurring manuals, so the delay cost more than the filtering saved.

The simulation counts the full length of an uncached prefix as recomputed tokens. It uses whole, non-overlapping prefixes, a linear cost per token, and sequential requests. A real inference engine shares individual KV blocks across prompts, protects active blocks, and has runtime overhead. This experiment gives no measured GPU latency or throughput. The two simulated caches store keys and lengths only. A GPU implementation would need to account for metadata and scheduling costs.

Adaptive caching is established in prior work. The specific experiment here switches KV admission rules based on recent missed-token volume and checks how the effect changes as capacity increases. [ARC paper](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache) · [vLLM prefix cache design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
