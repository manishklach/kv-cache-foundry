# vLLM prefix caching on an NVIDIA H200

Putting a shared manual before the question let vLLM reuse a median of **3,888 prompt tokens** and reduced median time to first token from **72.4 ms to 48.3 ms** (33%) for that prompt layout. The manual-first layout scored **47/51** with caching both off and on. The question-first layout scored **51/51** in both modes.

## Results

| Prompt layout | Prefix cache | Median TTFT | p95 TTFT | Median cached tokens | Exact match |
| --- | --- | ---: | ---: | ---: | ---: |
| Question first | Off | 72.3 ms | 74.0 ms | 0 | 51/51 |
| Manual first | Off | 72.4 ms | 73.3 ms | 0 | 47/51 |
| Question first | On | 73.4 ms | 74.5 ms | 32 | 51/51 |
| Manual first | On | **48.3 ms** | **53.4 ms** | **3,888** | 47/51 |

TTFT measures the time from starting the local HTTP request to receiving the first nonempty streamed text. The latency and cache columns use 50 requests per row after the first workload request. Accuracy uses all 51. The requests were sequential and used one synthetic manual of roughly 3,900 prompt tokens. These measurements describe one run per condition.

With caching on, the manual-first layout was about 34% faster at the median than question-first, with four fewer exact matches. The cache-off results show that the accuracy difference also occurred without prefix reuse. See the [results analysis](docs/results.md) for the individual misses and what the benchmark can establish.

## Reproduce

The recorded run used one NVIDIA H200 SXM on Runpod, Qwen3-4B-Instruct-2507, and vLLM 0.32.0. On a Linux GPU machine with a compatible NVIDIA driver, Python 3.12, Bash, and curl, run:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
bash run_on_pod.sh
```

The runner installs vLLM, starts a loopback-only server for each cache mode, and writes two JSONL result files plus the server logs. Follow the [reproduction guide](docs/reproduce.md) for deployment settings, output checks, and the install workaround used on Runpod.

## Repository contents

- [Benchmark client](benchmark.py) and [GPU runner](run_on_pod.sh)
- [Experimental design and metrics](docs/methodology.md)
- [Reproduction guide](docs/reproduce.md)
- [Results, accuracy misses, and limitations](docs/results.md)
- Raw requests and measurements with caching [off](data/results-off.jsonl) and [on](data/results-on.jsonl)
- [Original results archive](h200-prefix-results.tar.gz) with raw measurements, both vLLM logs, and the runner log
