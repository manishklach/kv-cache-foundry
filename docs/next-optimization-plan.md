# Next inference optimization: deadline-coupled KV value controller

## Recommendation

Build one online KV decision controller in the Foundry C core. Have it choose **admit, retain, evict, restore, recompute, or prefetch** using one calibrated cost model and the request's remaining latency budget.

The hypothesis is that separate cache policies can each look good while working against one another: an admission rule can keep a prefix that an eviction rule immediately removes; a prefetch can occupy GPU blocks needed by active decodes; and a restore can finish after local recomputation would have. A shared estimate of critical-path milliseconds and the GPU-memory opportunity cost could make these decisions agree.

This is a **systems integration hypothesis**, not a claim that each heuristic is new. We will validate novelty after implementing the smallest useful C prototype and comparing it against current work.

## What makes this experiment useful

The controller would score actions in milliseconds on the request's critical path:

```text
J(action) = expected critical-path ms
          + λ_memory × reserved GiB-seconds (converted to equivalent ms)
          + λ_slo × max(0, predicted TTFT - request deadline)
```

Its inputs would include prefix length and expected reuse, measured local-prefill cost, restore bandwidth and queue depth, GPU cache pressure, prefetch completion estimate, active decode pressure, and request SLO slack. The memory shadow price λ_memory converts future latency opportunity cost per GiB-second into milliseconds; λ_slo makes a predicted deadline miss more expensive in the same units. Online measurements would update action-cost distributions by GPU and prefix-size bucket. When estimates are uncertain, the controller should keep the existing vLLM decision and record the missed counterfactual instead of taking a high-risk action.

The specific exploratory idea is **slack-priced residency**: a cache block is valuable when its expected next reuse would recover scarce request slack, not just when it is likely to be reused. That could prefer a less frequently reused long prefix during decode congestion, then release or offload it when the queue clears. The score would also charge prefetch for any decode delay and for the memory it occupies before use.

## Prior art and novelty boundary

The prior-art check rules out several easy novelty claims:

- vLLM already implements exact-prefix block lookup, block reference counts, free-block ordering, and eviction in its KV cache manager ([vLLM design](https://docs.vllm.ai/en/latest/design/prefix_caching/)).
- CacheScout learns agent execution transitions to guide KV eviction and prefetching on vLLM ([paper](https://arxiv.org/abs/2608.14624)). A learned agent-aware prefetch/eviction policy alone would overlap with it.
- CacheBlend and newer systems already reuse non-prefix RAG chunks with selective recomputation. CacheWeave, posted November 2026, learns which token blocks to recompute for position-independent reuse ([CacheBlend](https://arxiv.org/abs/2405.16444), [CacheWeave](https://arxiv.org/abs/2611.01735)). A generic reordered-chunk reuse claim would overlap with this line.
- Request-order sensitivity in rolling-agent KV reuse has also been studied, including history-dependent output variation and document-aligned recomputation ([Request Order Matters](https://arxiv.org/abs/2610.05833)).

The proposed distinction is the **joint, causal cost accounting** across admission, eviction, local recompute, external restore, prefetch timing, and decode SLO slack, with per-action counterfactual measurement. That combination still needs a targeted search before we describe it as novel. This plan makes no patent or uniqueness claim.

## Staged work

### M0 — freeze inputs and baseline

1. Use the archived admission and eviction traces plus one reordered RAG/agent trace with persistent cache history.
2. Define one Trace v1 record for arrivals, prefix identity/length, local hit tokens, restore source/bytes, prefill/restore times, queue state, SLO deadline, and output comparison.
3. Keep the current vLLM policy, adaptive admission, cost-ranked eviction, and deadline prefetch as independent baselines.
4. Separate cold, warm, same-prefix, and reordered-chunk cases. Preserve request order and cache state across replay.

### M1 — implement the decision core in C

Extend [`foundry/`](../foundry/README.md) with a C11 API for cost observations, uncertainty, action scoring, and counterfactual updates. Keep the policy logic in C. Retain a conservative fallback to current vLLM behavior whenever model identity, cache hit count, or cost evidence is missing.

The first policy should only make exact-prefix decisions: whether to admit/retain a valid KV block and whether an already evicted prefix is cheaper to restore or recompute. Add prefetch only after the cost model can account for queue time, memory residency, and decode interference. This keeps the first result distinct from selective non-prefix chunk recomputation.

### M2 — CPU replay and kill gate

Run the C engine against trace replays with 3 randomized policy orders and held-out trace seeds. Record per-request decision, predicted cost, realized cost, model uncertainty, cache hits, missed tokens, output agreement, p50/p95 TTFT proxy, and bytes resident or transferred.

Do not rent a GPU unless the C replay beats the best existing baseline by at least **5%** on the pre-registered objective, has no prompt/output accounting errors, and shows no SLO regression in any held-out trace. If it fails this gate, stop and revise the mechanism before cloud spend.

### M3 — short real-GPU validation

After the CPU gate, check Runpod balance and live stock. Use one GPU and a short, fixed-duration run. Compare the unified controller against the strongest baseline on the same prompt trace, with 3 randomized repeats and identical model/runtime/hardware. Measure TTFT p50/p95, SLO goodput, cache hit tokens, restore bytes/time, prefill tokens, GPU memory pressure, and output-token agreement. Save exact hardware, driver, CUDA, image, model revision, logs, and raw observations.

The live gate is at least **5% lower p95 TTFT or higher SLO goodput at the same output agreement**, with no more than a **2%** regression in the other primary metric. Report negative results if either condition fails.

### M4 — provider economics and reproducibility

Only if the real-GPU result survives repeated runs, add a cost-per-million-generated-tokens view using measured GPU-seconds, transfer/storage cost, and the provider's actual price input. Publish the C API, traces that can be redistributed, ablations, error bars, and a vLLM adapter parity check.

## Kill criteria and risks

- **Prediction too noisy:** if measured action costs do not predict held-out realized costs, do not enable policy decisions; collect more calibration or simplify the state.
- **Benefit is just a baseline weakness:** if fixed admission or LRU matches the joint controller on held-out traces, do not claim a new optimization.
- **Prefetch hurts decode:** if decode p95 rises enough to erase the saved TTFT, disable deadline-aware prefetch and keep the admission/restore subproblem separate.
- **Output changes:** exact-prefix reuse should remain semantically valid, but existing probes show greedy outputs can be sensitive when top logits tie. Report mismatch counts and margins, and avoid claiming exact output equivalence from a single token check.
- **Integration cost:** if the C core cannot be called without a large invasive vLLM fork, first ship it as an offline C replay and measure overhead before adding runtime glue.

## Why the request-327 result matters here

The H100 probe is a correctness diagnostic, not the optimization itself. It shows request 327 has a zero top-two logprob gap on the H100 and L4, while the narrow controlled pairs did not reproduce the historical token flip. The next controller must therefore track cache state and request history as first-class trace inputs, and its evaluation must preserve the original sequence. Do not turn this near-tie observation into a cache-policy explanation without the actual scheduler-batch and trace-history evidence.

## Reading

- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [CacheScout: Learning Agent Execution for KV-Cache Management](https://arxiv.org/abs/2608.14624)
- [CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444)
- [CacheWeave: Learning to Recompute for Position-Independent KV Caching](https://arxiv.org/abs/2611.01735)
- [Request Order Matters: Cache-History Sensitivity in Selective KV-Cache Reuse](https://arxiv.org/abs/2610.05833)
