# GPU top-two margin gate

This small CUDA C benchmark tests a GPU primitive motivated by the request-327
tie: return the top two logits, a margin, and a threshold flag with a stable
token-id tie-break. It compares one CTA scanning a full vocabulary row against
a hierarchical reduction that scans the row in parallel tiles and combines the
partial top-two results.

## Why this experiment

For greedy decode, the top-two margin is a cheap signal for whether a token
choice is fragile. The kernel also makes exact ties deterministic by choosing
the lower token id. The request-327 history had equal reported scores for IDs
220 and 287; the benchmark inserts that controlled tie in its first row and
requires the result to be `(220, 287)` with margin zero.

This is a **GPU primitive benchmark**, not a vLLM integration or an inference
speedup claim. The scores are synthetic FP32 values, and the measured time is
the top-two reduction only. The hierarchical path uses two kernel launches, so
it may lose for small batches. A future runtime integration would need to fuse
the top-two/margin result into the serving engine's logits/sampling path and
measure the full token-generation path. Margin-triggered verification and
deterministic decoding have prior art; this experiment makes no novelty claim
for those general ideas.

## Implementations

- **Single CTA:** one 256-thread block per request scans all 151,936 logits,
  keeping local top-two candidates and reducing them in shared memory.
- **Hierarchical:** 4,096-logit tiles are reduced independently, then a second
  kernel combines each request's partial top-two candidates.
- Both use the same lexicographic comparator: higher score wins; on equal
  scores, the lower token id wins. The CPU oracle checks the returned IDs,
  margin, and flag for every row.
- The source is C-style CUDA (`.cu`) with no Python, C++ classes, or templates.
  NVIDIA's `nvcc` compiles CUDA kernel source using its CUDA C++ frontend.

## Build and run

On Linux with a CUDA toolkit and an NVIDIA GPU:

```sh
make clean all
./margin_gate 0.0001
```

The program tests batch sizes 1, 2, 4, 8, 16, and 32, performs 25 warmups and
200 timed repetitions per implementation, and prints CSV to stdout. It reports
kernel latency, the ratio of single-CTA latency to hierarchical latency, the
controlled tie result, and CPU-oracle agreement. The threshold is an optional
nonnegative margin value; the default is `0.0001`.

## H100 Runpod result

The first live run used one Runpod H100 SXM 80GB. For each batch size, three
separate process runs measured the same seeded logits, with 25 warmups and 200
timed calls per kernel. The table reports the median across those three runs.

| Rows | Single CTA (ms) | Hierarchical (ms) | Speedup vs. single CTA | CPU oracle |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.143794 | 0.013603 | 10.57× | 3/3 pass |
| 2 | 0.144322 | 0.013539 | 10.66× | 3/3 pass |
| 4 | 0.144287 | 0.013636 | 10.58× | 3/3 pass |
| 8 | 0.144185 | 0.014042 | 10.27× | 3/3 pass |
| 16 | 0.145347 | 0.015227 | 9.55× | 3/3 pass |
| 32 | 0.150414 | 0.023074 | 6.52× | 3/3 pass |

Every run returned the controlled top pair `(220, 287)`, a zero margin, and a
set flag. All top-two IDs, margins, and threshold flags matched the CPU oracle
for every row. The raw measurements are in
[`results/runpod-h100-2026-11-16.csv`](results/runpod-h100-2026-11-16.csv),
and the exact placement and device record is in
[`results/hardware-runpod-h100-2026-11-16.json`](results/hardware-runpod-h100-2026-11-16.json).

The 10.66× maximum is a **kernel-only comparison against this benchmark's
single-CTA reference**, which assigns one block to scan the entire 151,936-value
row. It demonstrates the benefit of splitting a large vocabulary across
parallel CTAs for this reduction. It is not a 10× inference gain and is not a
comparison against vLLM, PyTorch, FlashInfer, or another production sampling
kernel. The test uses synthetic FP32 logits and does not include model compute,
logit production, synchronization, or runtime integration.

## Interpretation and stop criteria

- A speedup here means only the hierarchical reduction is faster than this
  benchmark's single-CTA reference for that shape. It does not mean end-to-end
  inference is faster by the same factor.
- A reported speedup is usable only if both paths pass the CPU-oracle check and
  the controlled tie returns token IDs 220 then 287 with a zero margin.
- If the hierarchical path loses at batch 1 and realistic batch sizes, or its
  reduction time is insignificant beside engine sampling/synchronization,
  stop before attempting a vLLM integration.
- Results are hardware-specific. Record the exact GPU, driver, CUDA toolkit,
  compile command, and raw stdout for every run.

## Runpod record

The Runpod catalog reported Secure Cloud H100 SXM availability at LOW in
US-CA-2 and a rate of $4.39/hour. The measured Pod was `x4ji7wyn73p8vp`, in
US-CA-2, using
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. It was terminated after the
measurement; the final `get-pod` read returned 404 (`pod not found`). The
post-termination pod-billing query returned no records and `$0.00`; that is the
ledger response at query time, not an estimate of any later-posted charge.
