# vLLM serving-path argmax profile

## Question

Does greedy argmax take enough GPU time in real vLLM serving to justify
integrating the standalone hierarchical CUDA argmax prototype?

## Setup

- Runpod Secure Cloud: one NVIDIA H100 80GB HBM3, Pod `htmj4ckki70ozn`,
  US-CA-2; $4.39/hour; 40 GB container disk.
- GPU: 81,559 MiB, UUID
  `GPU-7f121db6-3ac2-bb56-de2d-921fcc61805c`, PCI bus `00000000:9B:00.0`,
  compute capability 9.0, driver 595.91.07.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; Python 3.12.3,
  vLLM 0.32.0, PyTorch 2.13.0+cu130, CUDA toolkit 12.8 V12.8.93.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, BF16, max model length 4,096,
  eager mode, max 8 sequences, GPU memory utilization 0.55. The server log
  resolved the model revision to `main`; a commit hash was not captured.
- Profiler: vLLM Torch profiler, `ignore_frontend=true`, maximum 30 engine
  iterations, stack capture disabled. Profiler files stayed on the temporary
  Pod and were removed at cleanup; the extracted aggregate table is recorded
  below.
- Workload: eight simultaneous OpenAI-compatible chat requests with the same
  30-token prompt, greedy decoding, and `max_tokens=32`. All eight completed
  with HTTP 200 and returned exactly 32 completion tokens (256 in total).

## Result

The profiled aggregate reported:

| Operation | Calls | Total CUDA time | Average per call | Reported CUDA share |
| --- | ---: | ---: | ---: | ---: |
| `aten::argmax` | 30 | 134.846 µs | 4.495 µs | 0.10% |
| PyTorch reduction kernel underlying argmax | 30 | 134.846 µs | 4.495 µs | 0.10% |

The profile also recorded 28 engine iterations with an eight-request generation
batch (`execute_context_0(0)_generation_8(8)`). These are aggregated profiler
times, not end-to-end latency or a standalone microbenchmark.

The best standalone hierarchical argmax proxy from the previous H100 run was
6.275–6.318 µs for batch sizes 1–2; it was slower than the 4.495 µs average
observed for vLLM's actual `aten::argmax` in this serving trace. The custom path
also uses a two-kernel hierarchy, while the stock path here is one reduction
kernel. This candidate therefore fails the current integration gate: it is not
worth adding the custom kernel to vLLM's sampling path based on these results.

## Interpretation and limits

- This is a real vLLM serving-path profile and directly observes the production
  `aten::argmax` operation. It is **not** an A/B of the custom kernel inside
  vLLM and provides no serving throughput improvement claim.
- The Torch profiler adds overhead and is a diagnostic. vLLM recommends
  Nsight Systems for low-overhead performance-critical profiling; a follow-up
  Nsight capture would be needed before treating the exact 0.10% share as a
  production timing number.
- One H100, one model, one request shape and one concurrency level were used.
  The model revision was not pinned. This is enough to reject this standalone
  replacement for the tested setup, not to generalize to all vLLM workloads.
- A more promising follow-up would have to fuse argmax with other sampling or
  logits work so it removes operations and launches, rather than replacing an
  already fast argmax with a two-launch reduction.

## Runpod cleanup

The Pod was terminated after collecting the aggregate profile, response usage,
and device metadata. The trace itself lived only on the ephemeral container
disk. See [`docs/hardware-inventory.md`](../../../../docs/hardware-inventory.md)
for the hardware inventory entry.
