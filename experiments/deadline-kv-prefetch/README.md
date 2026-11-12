# Deadline-aware KV prefetch on CUDA

The benchmark restores synthetic keys and values from pinned host memory and executes 24 dependent scaled dot-product attention layers per request. Eight short and long requests arrive in a staggered burst while a separate CUDA stream runs 96 foreground attention passes. The same input tensors, compute dispatch order, and 2 MiB copy limit apply to each policy.

| Policy | Copy order | When attention can start |
| --- | --- | --- |
| Full restore | Oldest eligible request first | After the entire request is copied |
| FIFO layer prefetch | Oldest eligible request first | After the next layer is copied |
| Deadline layer prefetch | Lowest estimated slack for the next eligible layer | After the next layer is copied |

The C scheduler computes slack from the request deadline, the estimated compute remaining after the transfer, and the estimated copy time for the rest of the layer. The host polls CUDA events before marking copies and kernels complete. One transfer of up to 2 MiB is in flight. Both layer prefetch policies allow up to two layers ahead of completed compute. The background compute stream runs one layer at a time and uses the same oldest-ready ordering under all policies.

Use a CUDA compiler and a CUDA-capable Linux host. On an H100, run from this directory.

```bash
nvcc -O3 -std=c++17 -arch=sm_90 benchmark.cu -o deadline-kv
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
./deadline-kv > results.csv 2> calibration.log
```

Each policy runs nine times in rotated order. Request latency runs from scheduled arrival to the final attention layer. Foreground p95 is the time between completions of full attention passes, including the first pass. Every trial checks that each layer's output is finite and bitwise equal to the full-restore reference.

The workload isolates the scheduling rule using synthetic KV and an attention-only computation chain. It omits a language model's other layers, token sampling, and serving queues. The measurements describe attention completion and foreground pass latency on this harness.

## RTX PRO 6000 Blackwell results

The corrected benchmark compiled with CUDA 13.2.86 and warnings treated as errors. It ran on one NVIDIA RTX PRO 6000 Blackwell Server Edition with driver 595.91.07. Its `nvidia-smi` capture reports 97,887 MiB and PCIe Gen 1 x16 at capture time. All 216 measured requests passed the bitwise attention checks. See the [hardware inventory](../../docs/hardware-inventory.md) for this device record and the preliminary H200 run.

| Policy | 512-token median (ms) | 4,096-token median (ms) | Foreground p95 (ms) | Requests within target |
| --- | ---: | ---: | ---: | ---: |
| Full restore | 53.98 | 52.01 | 8.17 | 27/72 |
| FIFO layer prefetch | 44.51 | 42.44 | 8.24 | 27/72 |
| Deadline layer prefetch | 44.12 | 42.41 | 8.23 | 27/72 |

Request medians pool 36 requests per prefix length. Foreground p95 is the median of nine trial p95 values. The 512-token target was 15 ms and the 4,096-token target was 65 ms. Deadline ordering reduced short-request median latency by 0.39 ms versus FIFO and did not increase the number of requests within target.

The initial H200 run completed with a compiler warning caused by a CUDA error-checking macro shadowing a local variable. That build is preliminary. The corrected results above use an RTX PRO 6000 because Runpod could not reacquire the H200 on its host. To build for the RTX PRO 6000, use sm_120 and add -Werror all-warnings to the nvcc command.

[Corrected RTX PRO 6000 results and logs](results/rtx-pro-6000/) | [Preliminary H200 results, logs, and source snapshot](results/h200-preliminary/)
