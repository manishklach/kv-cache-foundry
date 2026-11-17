# Throughput–output-stability frontier

This experiment asks how much serving throughput changes when batch composition changes, and how often greedy outputs change on the same H100/vLLM/model configuration. The motivating case is historical request 327: its H100 next-token top two (IDs 220 and 287) had exactly equal reported log probabilities, even with zero cached tokens. The controlled H100 pairs did not reproduce the historical token flip, so batch composition remains a hypothesis, not an established cause. See the [H100 probe record](../adaptive-admission/results/h100-probe-327/README.md).

This is an **evaluation of a known tradeoff**, not a novelty claim for deterministic inference or margin-triggered verification. LLM-42 and MarginGate are prior art. The experiment's value is a controlled, hardware-specific curve for the current Qwen/vLLM serving setup and a check of whether fragile-token rates can predict output flips or selective-check cost.

## Two research questions

1. **Throughput–stability frontier:** Across request batch sizes and orderings, what throughput does the default fast path deliver at each observed exact-output flip rate? What is the throughput cost of vLLM batch-invariant mode?
2. **Selective checks:** How many requests/tokens would a frozen margin threshold flag, how many observed flips would it catch, and what does a real verification path cost? The CPU analyzer can report flag coverage and empirical capture only when the input rows contain `verified`; it does not simulate GPU savings or treat a margin as proof of correctness.

## Pre-registered workload and controls

- **Anchor:** exact request-327 prompt and request-335 same-prefix control from the archived H100 probe. Preserve the model, tokenizer, and token IDs; record a pinned model revision for the new run.
- **Generalization set:** 24 fixed prompts: 12 code-generation/debugging tasks and 12 agent-style tool-planning tasks. Freeze the prompt text, expected answer checks, and tokenized prompt IDs before the GPU run. Keep this set in a separate fixture with provenance and license notes.
- **Modes:** stock vLLM fast path; vLLM batch-invariant mode; offline margin-threshold analysis; and, only if an actual runtime verification hook is available, a selective-verification implementation. Do not call post-hoc margin analysis an optimization.
- **Batch sizes:** 1, 2, 4, and 8 concurrent requests, with fixed request groups and rotated order. Use three repetitions per prompt/group and interleave mode runs in a predeclared randomized schedule. Wave IDs must be unique within each mode/concurrency/repetition.
- **Generation:** greedy decode, fixed model revision and server flags, fixed max output length per workload class, and request-level seed/config recorded. Repeat unchanged prompts to separate natural run variance from batch sensitivity.
- **Measurements:** completed output token IDs, first-token top-1 minus top-2 log-probability margin, TTFT, end-to-end request time, wave wall time, output tokens/s, exact-output flip rate, task checks, and verification/flag coverage. Record actual engine batch membership if available; client concurrency alone is not proof of co-batching.

The frontier is reported per workload and in aggregate. Always show the raw numerator/denominator for flips, use paired prompt/repetition comparisons, and include uncertainty intervals. Do not collapse task correctness and exact-output agreement into one metric: two different valid code answers can be semantically equivalent, and an identical greedy output can still be wrong.

## CPU gate before any GPU spend

1. Freeze the exact workload fixture, model revision, vLLM image/version, startup options, request ordering, and analysis thresholds.
2. Manually encode a few committed H100/L4 probe observations in the analyzer's TSV schema. Confirm that equal top-1/top-2 scores produce a zero margin and that request 327 is represented as a tie; the old JSONL artifacts are not accepted directly by this TSV-only tool.
3. Check the analyzer on a tiny hand-authored TSV with known output sequences, batch waves, flips, and margins. This is a software correctness check, not an inference result.
4. Dry-run the H100 server/client commands on CPU-side fixtures; check payload sizes and that the server returns token IDs, logprobs, usage, and timings before renting a GPU.
5. Proceed to a short H100 run only if the fixture and output parser preserve all expected fields and the planned run fits the available balance. Stop after the predeclared sample if the API does not expose the necessary measurements.

## C analyzer

`frontier_analyze.c` is a C11 offline analyzer. It consumes TSV rows and reports request counts, exact-output matches against a concurrency-1 baseline, output tokens/s from wave wall time, TTFT p50/p95, first-token margin tie rate, flagged-request coverage, the fraction of observed flips that were flagged, and the fraction marked verified. It requires actual measurements; it does not generate model outputs, estimate GPU timings, grade task correctness, infer batch membership, or provide a verification runtime. The first version is CPU-side analysis tooling, not yet a live-serving benchmark client.

Input columns (one request per row; do not include a header row):

```text
mode  concurrency  wave  prompt_id  repeat  wave_ms  ttft_ms  output_token_ids  top1_margin  flagged  verified
```

Fields are separated by literal tabs. `output_token_ids` is a semicolon-separated integer sequence. Use `NA` for unavailable timing/margin. `top1_margin` is logprob(top-1) minus logprob(top-2) for the first generated token, so smaller values mean a closer decision; exact ties are `0`. `flagged` is `1` when the frozen threshold selected the request for a check. `verified` is `1` only when a real check ran, otherwise `0`. `wave_ms` is the same wall time repeated on each row in a wave; it is counted once for throughput. The current tie-rate field counts exact zero-margin observations; threshold-based flagging remains a separate recorded value. The analyzer reports the fraction of observed flips that were flagged; this is a retrospective capture rate, not proof that checking would have prevented a flip.

Build and run on Linux with a C11 compiler:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  experiments/output-stability/frontier_analyze.c -lm -o frontier_analyze
./frontier_analyze measurements.tsv fast > frontier.csv
```

The baseline mode must have concurrency 1 and matching `(prompt_id, repeat)` rows. A missing match is reported as missing, not treated as a flip. Do not compare across model revisions, GPU types, or vLLM versions as if they were paired trials.

## Decision and kill criteria

- **Continue to selective-check implementation:** low-margin coverage is small, captures a materially higher share of observed flips than random/equal-coverage controls, and measured verification overhead can fit the performance budget.
- **Stop the margin policy:** low-margin tokens are common, fail to predict flips on held-out prompts, or checking costs more than the stability benefit.
- **Stop the GPU stage:** the server lacks the required per-token scores/IDs, scheduler batch composition cannot be verified for a claim that depends on it, or the first bounded run has no interpretable paired comparisons.

No speedup, correctness guarantee, or new-algorithm claim is made until an actual selective runtime path is implemented and measured end to end.
