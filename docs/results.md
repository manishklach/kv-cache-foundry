# H200 benchmark results

The benchmark ran on November 11, 2026 using Qwen3-4B-Instruct-2507 and vLLM 0.32.0. Each row contains one first workload request followed by 50 measured requests. The first request comes after a short, unrelated warmup.

| Layout | Cache | First workload TTFT | Median TTFT | p95 TTFT | Median cached tokens | Exact match |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Question first | Off | 89.7 ms | 72.3 ms | 74.0 ms | 0 | 51/51 |
| Manual first | Off | 71.4 ms | 72.4 ms | 73.3 ms | 0 | 47/51 |
| Question first | On | 89.8 ms | 73.4 ms | 74.5 ms | 32 | 51/51 |
| Manual first | On | 73.3 ms | 48.3 ms | 53.4 ms | 3,888 | 47/51 |

With the manual first, caching reduced median TTFT by 24.1 ms (33.3%) and p95 TTFT by 19.9 ms (27.1%) relative to the same layout with caching off. The question-first layout reused a median of 32 tokens, which is consistent with a short shared prefix ending before the device-specific part of the question.

## Accuracy misses

All misses occurred with the manual-first layout. Three questions were missed in both cache modes. Device numbers below are derived from the question order defined in the benchmark.

| Device | Expected | Cache off output | Cache on output |
| --- | ---: | ---: | ---: |
| 0034 | 1238 | 1308 | 1308 |
| 0136 | 1952 | 1959 | 1952 |
| 0153 | 2071 | 2064 | 2064 |
| 0187 | 2309 | 2316 | 2316 |
| 0082 | 1574 | 1574 | 1581 |

## Scope

This was a single-model experiment on one GPU using one synthetic document and sequential traffic. Layout order was fixed. It measures latency once a shared prefix has been populated. Production traffic can have competing prefixes, different lengths, and concurrent requests, so the benchmark does not estimate cache eviction, throughput under load, or latency with CUDA graphs. The two cache modes used different run identifiers, yielding 3,901 prompt tokens for cache off and 3,902 for cache on. A repeated experiment could fix the identifier across modes, alternate layout order, and use multiple manuals and repetitions.

The raw per-request measurements are available for [cache off](../data/results-off.jsonl) and [cache on](../data/results-on.jsonl). The [archive](../h200-prefix-results.tar.gz) also contains server logs that record the cache flags and runner output that identifies the GPU and vLLM version.
