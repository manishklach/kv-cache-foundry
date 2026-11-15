# Request 327: CPU grounding and GPU probe plan

## Verified from archived traces

The archived eviction comparison contains five output differences at index 327. `gpu-1-lru` emits token 287; the other five runs emit token 220. All six report zero cached tokens. Index 327 is the first occurrence of phase 1 / key 39 in the generated sequence. The CPU audit confirms the expected request identity and that index 327 is not an admission-trace mismatch.

The checked command is:

```sh
python experiments/adaptive-admission/audit_outputs.py eviction --index 327
```

The admission check is not a silent no-op. `audit_outputs.py admission --index 327` exits with an argparse error because the requested index is absent from that archive's mismatch set. That behavior is useful validation, but the probe plan should not describe it as an empty successful result.

Zero cached tokens rules out a cache-hit explanation for this request's token difference. It does not rule out prior policy decisions changing allocator state, block placement, request scheduling, or numerical execution before index 327 arrives.

## Probe design corrections

The archived runs establish a lead, not a cause. The first GPU stage should compare same-prompt cold observations under concurrency 1 and controlled concurrent submissions, recording the top-20 next-token log probabilities and the full request metadata. A small top-two gap supports a near-tie explanation but does not by itself prove why the archived runs differ.

Thread-pool wave membership is not the same as vLLM's actual execution batch. Treat client submission overlap as a proxy. To claim a batch-composition cause, either instrument the vLLM scheduler to record actual batch membership or use a controlled request pair submitted together and report only the pairwise result.

Rotate policy order across fresh server processes. Record prompt/tokenizer identity, cache salt, model revision, GPU/driver, server configuration, process order, seed, cache-hit count, output token, top-logprob list, and timings for every observation. Keep the tokenized prompt fixed while varying only the cache salt needed to force a miss or hit. Do not infer cache effects from a run whose effective cache state was not measured.

Use a prewritten stopping rule: if repeated isolated probes never flip, move to controlled concurrent pairs; if the gap is consistently large and tokens still differ, investigate prompt equality and engine/kernel state before attributing the result to ties; use allocator tracing only if those checks leave allocator state as the remaining explanation.

## L4 first-stage result (November 12, 2026)

The first bounded GPU stage ran the exact request-327 token-ID prompt on one Runpod L4 with Qwen3-4B-Instruct-2507 and vLLM 0.33.0. It made six sequential calls: two distinct cache salts, each called once cold and twice again. All six generated token ID 220. The top two reported candidates, 220 and 287, had the same log probability (`-5.6221370697`) in every observation; the full top-20 ordering and values were identical across all six observations. The server's aggregate prefix-cache hit rate reached 66.7%; its response usage did not report a per-request cached-token count.

This supports the near-tie hypothesis as a plausible numerical sensitivity, but the run did not produce a flip and cannot explain the historical H100 result by itself. It also differs in GPU and vLLM version (L4 / v0.33.0 here versus H100 / v0.32.0 in the archived eviction trace). It was sequential and did not instrument scheduler batch membership. Raw JSONL and the hardware/runtime record are in [`results/l4-probe-327/`](../experiments/adaptive-admission/results/l4-probe-327/).

The Pod ran for about 3 minutes 48 seconds. Runpod showed $0.69/hour compute and $0.004/hour for its 30 GB temporary container disk (roughly $0.044 at the recorded uptime); it was stopped immediately after collecting the probe. No GPU billing is currently accruing.

## H100 controlled-pair result (November 15, 2026)

The H100 follow-up ran the same seed-0 request-327 token-ID prompt on one Runpod H100 SXM with vLLM 0.32.0. The probe used non-streaming single-token completions with top-20 log probabilities. It measured request 327 once with a new cache salt, then request 335 with the same prefix/cache salt as a cache-hit control. It then submitted six two-request waves concurrently: request 327 paired with adjacent trace request 326. Each request in each pair used a distinct cache salt, so the paired requests could not reuse each other's KV state. This is controlled client submission overlap, not a trace replay and not instrumentation of vLLM's actual scheduler batch membership.

Request 327 returned token **220** with **0 cached tokens**. The top two candidates, 220 and 287, were tied at `-5.6167111397`. Request 335 returned token **8** and reported **4,096 cached tokens**, matching the expected full reusable prefix. In all six concurrent pairs, request 327 returned token 220 with zero cached tokens and the same top-two tie; the adjacent request 326 returned token 220 in all six trials. The target's median whole-response client time in the concurrent pairs was about 1,120 ms, and the neighbor's was about 1,448 ms. Those times include the HTTPS Runpod proxy and are not TTFT measurements or a performance comparison.

This confirms that the near-tie carries over to the same GPU class and vLLM version as the historical anomaly. It does **not** reproduce the one historical token-287 result. The concurrent pairs are deliberately narrow and do not reproduce the preceding cache/allocator history or record the engine's actual scheduler batch. No inference algorithm was changed and no speedup is claimed. The Pod ran for about 11m19s at $4.39/hour compute plus $0.004/hour container storage, an estimated **$0.83** total from the displayed rates and uptime. It was stopped after the data was saved; Runpod then showed $0.00/hour and $0.00/day idle disk cost.

Raw observations, runner source, server logs, and the complete hardware record are in [`results/h100-probe-327/`](../experiments/adaptive-admission/results/h100-probe-327/). The runner is [`probe_327.py`](../experiments/adaptive-admission/probe_327.py).

## Next evidence step

The next informative experiment is the pre-registered batch-mate probe: reproduce the original concurrent trace through request 327, capture the actual requests co-scheduled with 327 in vLLM, widen the top-logprob record, and rotate trial order across fresh processes. That can distinguish score-tie sensitivity from batch-composition and first-run effects. Allocator tracing remains conditional on the actual batch and repeated output evidence; the current data does not justify calling this a KV policy bug.
