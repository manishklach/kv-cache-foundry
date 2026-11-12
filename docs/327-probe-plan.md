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

## Runpod funding gate

Before the run, the Runpod Billing page showed credit balance **$0.38** and spend rate **$0.00/hr**. The account had no other active GPU, CPU, storage, or endpoint resources. The targeted L4 probe completed within a few minutes; its Pod is stopped and shows $0.00/hr. A larger H100 reproduction with concurrency, request-order rotation, and pinned vLLM 0.32.0 remains a separate experiment and should be attempted only after checking the current balance and available GPU stock.

## Next GPU step

The next informative run is a short H100 reproduction with pinned vLLM 0.32.0: repeat the exact prompt at concurrency one, then submit a controlled request pair together and collect top-20 logits plus actual cache-hit counts. That directly tests whether the L4 tie carries over to the original H100 runtime and whether a concurrent pair can change the selected token. Check the current balance and stock immediately before deployment; keep the run under an explicit spend/time cap. The broader multi-hour rotation plan remains deferred.
