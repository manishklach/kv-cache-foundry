# GPU top-two margin gate

This small CUDA C benchmark tests a GPU primitive motivated by the request-327
tie: return the top two logits, a margin, and a threshold flag with a stable
token-id tie-break. It compares a one-CTA reduction, a hierarchical reduction,
and greedy argmax proxies across vocabulary tile sizes and batch sizes.

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

- **Single CTA top-two:** one 256-thread block per request scans all 151,936
  logits. The benchmark includes shared-tree and warp-shuffle reductions.
- **Hierarchical top-two:** tiles of 512, 1,024, 2,048, 4,096, 8,192, and
  16,384 logits are reduced independently, then a second kernel combines each
  request's partial top-two candidates.
- **Greedy argmax proxy:** single-CTA and hierarchical top-one paths measure
  argmax-shaped work for comparison with greedy sampling. These are standalone
  CUDA proxy kernels, not the actual PyTorch or vLLM implementation.
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
200 timed iterations per implementation, and prints CSV to stdout. Pass a
second optional integer argument to label a repeat, e.g. `./margin_gate
0.0001 2`. The threshold is an optional nonnegative margin value; the default
is `0.0001`.

## Follow-up: greedy argmax and tile sweep on H100

After the initial top-two comparison, a follow-up measured a C-style CUDA
greedy-argmax proxy, a warp-shuffle single-CTA top-two kernel, hierarchical
top-one and top-two reductions, and six tile sizes. The benchmark ran three
separate repeats on one Runpod H100 80GB HBM3. Each repeat used 25 warmups and
200 timed iterations per kernel. The table shows medians across repeats.

| Rows | Single-CTA argmax proxy (ms) | Best hierarchical argmax proxy (ms) | Argmax proxy tile | Best hierarchical top-two (ms) | Top-two tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.048832 | 0.006318 | 1,024 | 0.011068 | 1,024 |
| 2 | 0.049261 | 0.006275 | 1,024 | 0.011221 | 1,024 |
| 4 | 0.049591 | 0.006805 | 1,024 | 0.011884 | 1,024 |
| 8 | 0.049666 | 0.007272 | 2,048 | 0.013068 | 2,048 |
| 16 | 0.049759 | 0.008172 | 4,096 | 0.015497 | 4,096 |
| 32 | 0.050575 | 0.010249 | 8,192 | 0.020301 | 8,192 |

Across rows 1–32, the best hierarchical top-two path is 4.4×–4.9× the latency
of the single-CTA **argmax proxy** (top-two adds work, so this is a comparison
of unlike operations) and 10.6×–13.2× faster than the single-CTA warp-shuffle
top-two baseline. The latter is deliberately a weak large-row reference: one
CTA scans the entire vocabulary. More importantly, hierarchical greedy argmax
measures 6.4×–7.9× below the single-CTA argmax proxy in this microbenchmark.
Those figures cover only the standalone CUDA reductions. Kernel launch count,
the exact PyTorch/vLLM sampler, logits production, integration, and model
inference are not included.

The source reference inspected for vLLM 0.32.0 uses `logits.argmax(dim=-1)`
for its greedy path ([sampler.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/sample/sampler.py)).
This benchmark does **not** invoke or time that PyTorch kernel. The
`*_greedy_argmax_proxy` variants only model the argmax operation shape, so the
results cannot support a claim of superiority to production vLLM sampling.

The best top-two tile grew with batch size: 1,024 for rows 1–4, 2,048 for row
8, 4,096 for row 16, and 8,192 for row 32. This is an empirical crossover for
this H100, synthetic FP32 input, and current kernel implementation; it is not
an autotuned production policy. All 270 result rows passed the CPU oracle, and
the forced tie returned `(220, 287)` with zero margin in every top-two case.
Raw data is in [`results/runpod-h100-baseline-2026-11-16.csv`](results/runpod-h100-baseline-2026-11-16.csv).

## H100 Runpod result

This section records the initial, fixed-4,096-tile top-two-only run. The
follow-up above is a separate Pod run and is reported separately.

The initial live run used one Runpod H100 80GB HBM3. For each batch size, three
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
US-CA-2 and a rate of $4.39/hour. The initial top-two run used Pod
`x4ji7wyn73p8vp`; this follow-up used Pod `5r9e5oaxuznil7`, both in US-CA-2,
using
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. It was terminated after the
measurement; the final `get-pod` read returned 404 (`pod not found`). The
original top-two hardware record remains in
[`hardware-runpod-h100-2026-11-16.json`](results/hardware-runpod-h100-2026-11-16.json);
the follow-up used the same H100 model and toolchain, with its separate Pod ID
recorded in the repository hardware inventory.
