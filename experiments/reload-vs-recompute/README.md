# KV reload versus recompute on H200

The C benchmark uses the llama.cpp b12250 C API with Qwen3-4B-Instruct-2507 Q4_K_M on one Runpod NVIDIA H200 SXM. It compares restoring a sequence's serialized KV from pinned host memory with prefilling the same token IDs from scratch. The restore path includes a controlled backlog modeled by serial host-to-device copies. The recompute path runs without that backlog, so the benchmark assumes prefill can proceed without waiting for the copy queue. The historical CSVs do not include the original `nvidia-smi` output, driver version, memory total, or CUDA version; the H200 SXM class is the hardware detail recorded in the experiment notes. See the [hardware inventory](../../docs/hardware-inventory.md).

Run `bash run_reload_c.sh` on a compatible Linux CUDA host with a C compiler, Bash, curl, and tar. The runner downloads pinned llama.cpp binaries and model weights and builds [the C benchmark](reload_vs_recompute.c).

The [extended sweep](h200-reload-extended.csv) contains 448 paired measurements across four prefix lengths and seven backlog sizes. Each condition has 16 alternating-order pairs. The selector chooses the lower median from the first eight pairs. The table uses the remaining eight.

| Prefix tokens | First tested backlog where recompute won | Reload at that backlog | Recompute |
| ---: | ---: | ---: | ---: |
| 128 | 1 GiB | 20.17 ms | 14.77 ms |
| 512 | 2 GiB | 40.61 ms | 34.73 ms |
| 2,048 | Beyond tested range | 83.93 ms at 4 GiB | 139.68 ms |
| 4,096 | Beyond tested range | 89.66 ms at 4 GiB | 287.93 ms |

On the equally weighted test grid, the selector's chosen paths summed to 4,850.48 ms across 224 held-out pairs. Always reloading summed to 6,005.96 ms and always recomputing summed to 26,741.46 ms. The selector reduced this sum by 19.2% versus always reloading. It chose recompute in five of 28 conditions. After restoration and after fresh prefill, the same suffix produced the same greedy next token for each of the four tested lengths.

The [initial sweep](h200-reload-results.csv) found no crossover through 512 MiB. Both sweeps measure isolated operations with an injected serial copy workload. The lookup uses calibration samples from the same hardware, prefix lengths, and backlog sizes as the held-out measurements. The result does not estimate latency under a production backlog distribution or prove that recompute overlaps real transfers without contention.
