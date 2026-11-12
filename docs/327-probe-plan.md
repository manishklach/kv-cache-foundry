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

## Runpod funding gate

The Runpod Billing page was read on 2026-11-12: credit balance **$0.38**, spend rate **$0.00/hr**, and current GPU, vCPU, storage, and endpoint resource counts all **0**. The currently listed GPU rates include RTX A6000 at $0.59/hr, H100 SXM at $3.99/hr, and H200 SXM at $5.29/hr. The available credit is less than one hour on the least expensive listed GPU, and does not leave a safe budget for Pod startup, model availability/download, and a useful probe. No Pod was created. Do not let an experiment charge beyond the available credit without the user setting a new spend cap.

When credit is sufficient, prefer a short, targeted H100 run if the goal is to reproduce the original H100 anomaly. Runpod availability and price must be checked again at that time. Do not run the proposed multi-hour rotation/repeat plan as the first GPU stage.
