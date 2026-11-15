# Foundry synthetic CPU replay — 2026-11-15

## Result

The first slack-priced admission heuristic failed the pre-GPU gate. In the 32,768-token-cache, 20-seed synthetic replay, it was worse than the LRU-like baseline by 6.80% on the toy objective, with 6.61% more missed tokens and 11.05% more modeled deadline misses. It tied the synthetic reuse-rank policy in this setup.

| Policy | Mean missed tokens | Mean modeled deadline misses | Mean toy objective (ms) |
|---|---:|---:|---:|
| LRU-like | 4,877,081.60 | 179.70 | 65,608.9333 |
| Synthetic reuse rank | 4,877,081.60 | 179.70 | 65,608.9333 |
| Slack-priced admission | 5,199,462.40 | 199.55 | 70,067.0167 |

The gate is **failed**. No GPU run is justified by this result, and no serving speedup is claimed.

## Method

`replay.c` generated four alternating traffic phases for each seed, using a fixed linear-congruential generator. Even phases mix a hot set and one-off scan traffic; odd phases repeat groups of synthetic document keys. Cache capacity was 32,768 tokens. Each miss adds a toy cost `2 + prefix_tokens / 96` ms. Modeled deadline misses count individual misses whose toy cost exceeds a synthetic 5 ms or 80 ms slack. The toy objective sums miss cost plus three times lateness beyond request slack. The command was:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Ifoundry \
  foundry/replay.c -lm -o /tmp/foundry_replay
/tmp/foundry_replay 32768 20
```

The CSV records the output. The model does not represent vLLM scheduling, GPU memory blocks, queue interference, actual SLO distributions, or measured reload/recompute latency. The synthetic reuse-rank baseline uses a fixed reuse score; it is not the repo's existing measured-cost eviction policy. Policies run on the same deterministic trace for each seed, but there is no randomized order because the simulation has no timing state.

## Interpretation and next step

This only says that the initial admission heuristic is poor under this toy workload. It does not falsify the shared-cost-model idea: the replay does not consume measured costs, and its reuse hints are synthetic. Before tuning the heuristic, replace the toy generator with a frozen trace format containing request order, key/prefix size, inter-arrival/slack, and measured cold/warm costs. Then validate predicted action costs on held-out traces and compare against the existing admission/eviction baselines. Do not spend on a GPU until that replay shows at least 5% improvement on a pre-registered objective with no held-out SLO regression.
