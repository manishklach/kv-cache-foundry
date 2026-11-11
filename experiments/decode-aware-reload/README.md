# Decode-aware KV reload in llama.cpp

Restoring a cached prefix moves its key/value tensors from host memory to GPU memory. Stock llama.cpp submits the host restore as one group of tensor writes. This experiment gives a concurrent decode context a way to influence how much restore work is submitted at once.

The [C controller](decode_aware_reload.c) starts at 4 MiB per batch. It halves the budget when smoothed decode latency exceeds the supplied target, down to 1 MiB. It increases the budget by 1 MiB when smoothed latency drops below 80% of target, up to 16 MiB. Under pressure it can wait for a decode step, with a 2 ms timeout per batch. A new request can also recompute its prefix in 128-token chunks if calibrated prefill time is at least 10% below the estimated restore time, including pacing delays. After the foreground decoder finishes, restore batches are unbounded.

The [C++ engine patch](llama-paced-restore.patch) adds a C-callable API for host sequence restore. It batches actual tensor writes, splitting a tensor slice if needed to stay within the byte limit. It uses the engine's existing synchronous backend write API, so every submitted batch completes before the scheduler can wait or submit another one. Only the opt-in API asks for a budget. The experiment uses two contexts sharing model weights, each owned by its own thread during inference. It does not add a scheduler to llama-server or vLLM.

## Build and run

Run from this directory on Linux with CMake, a C/C++ toolchain, and the CUDA toolkit. The script pins llama.cpp to b12250. Select a CUDA architecture for the test GPU to limit compile time.

```sh
CUDA_ARCHITECTURES=90 JOBS=8 bash build.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
build/cuda/decode_aware_reload /path/to/model.gguf 2048 3.0 999 > results.csv
```

Architecture 90 applies to H100/H200. The command supplies the prefix length, target foreground inter-token latency in milliseconds, and the number of model layers to offload. Use the same model, prefix lengths, latency target, and GPU across modes. For a CPU correctness smoke test, build with `BACKEND=cpu`, run the binary under `build/cpu`, and pass `0` GPU layers.

## Measurements and correctness

Each of three trials rotates the order of stock restore, fixed 4 MiB batches, and the adaptive policy. Each mode runs twelve background requests while a separate context decodes up to 256 foreground tokens. The foreground feeds identical tokens across modes. Each background request checks eight generated tokens against an independent prefill reference. Before timing, a separate check forces 1 KiB batches to exercise splitting within tensor slices.

The CSV includes background time to first token (TTFT), foreground progress at request start and first token, and restore batch and wait counts. A summary row for each trial and mode records foreground p50/p95 and total wall time. Use the progress columns to distinguish requests whose TTFT overlapped decode from those that started after it ended. The foreground latency includes the first decode step and spans the whole foreground run.

This is a two-context inference experiment with a single background request in flight. It measures the interaction between the restore path and decode on the same device. A provider-level throughput claim would require an engine-integrated request queue and a concurrent arrival trace with enough overlapping requests to measure how many meet both latency targets.

## Validation status

A Linux CPU smoke test used the TinyLlama Stories15M Q4_0 model, a 512-token prefix, and a 1 ms target. The 3,545,264-byte KV snapshot passed the forced split-restore check. All 108 measured requests matched the eight reference tokens. The adaptive policy waited on 33 of its 36 requests and selected reload for all 36. These checks cover the restore and pacing paths. GPU performance and live selection of the recompute path remain unverified.

Model SHA-256: `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739`.
