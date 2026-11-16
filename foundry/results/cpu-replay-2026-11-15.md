# Foundry synthetic CPU replay — 2026-11-15

## Result

The corrected replay still fails the pre-GPU gate. This revision fixes the eviction baseline: the prior implementation labeled a reuse-ranked eviction score as LRU, so its LRU and reuse-rank rows were identical. The policies are now true LRU, an explicitly privileged oracle reuse-rank comparator, and slack-priced admission with LRU eviction. The admission policy is compared to LRU with the eviction rule held constant.

At 32,768 tokens and 20 synthetic seeds, slack-priced admission reduced the toy objective by 1.72% and missed tokens by 2.14% versus LRU, but increased modeled deadline misses by 6.38%. It remains 14.30% worse on the toy objective than the oracle reuse-rank comparator. The oracle uses the generator's true future reuse counts and is not an implementable policy. The result fails the pre-GPU gate because it does not clear the required 5% gain and regresses the deadline-miss metric.

| Policy | Mean missed tokens | Mean modeled deadline misses | Mean toy objective (ms) |
|---|---:|---:|---:|
| True LRU | 5,700,480.00 | 193.65 | 76,303.2500 |
| Oracle reuse rank (future information) | 4,877,081.60 | 179.70 | 65,608.9333 |
| Slack-priced admission with LRU eviction | 5,578,700.80 | 206.00 | 74,993.5667 |

The gate is **failed**. No GPU run is justified by this result, and no serving speedup is claimed.

## Method

`replay.c` generated four alternating traffic phases for each seed, using a fixed linear-congruential generator. Even phases mix a hot set and one-off scan traffic; odd phases repeat groups of synthetic document keys. Cache capacity was 32,768 tokens. Each miss adds a toy cost `2 + prefix_tokens / 96` ms. Modeled deadline misses count individual misses whose toy cost exceeds a synthetic 5 ms or 80 ms slack. The toy objective sums miss cost plus three times lateness beyond request slack. The command was:

```sh
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -Ifoundry \
  foundry/replay.c -lm -o /tmp/foundry_replay
/tmp/foundry_replay 32768 20
```

The CSV records the corrected output. The model does not represent vLLM scheduling, GPU memory blocks, queue interference, actual SLO distributions, or measured reload/recompute latency. The oracle reuse-rank comparator scores a resident key as `(true_future_reuses + 1) * prefix_length / (age + 1)`; its use of future reuse is deliberately disclosed and makes it an upper-bound comparator, not a deployable baseline. The slack-admission ablation uses the same true LRU eviction rule as the LRU row. Policies run on the same deterministic trace for each seed, but there is no randomized order because the simulation has no timing state.

## Interpretation and next step

This only says that the current admission heuristic does not pass this toy gate. It does not test the shared-cost-model idea: the replay does not consume measured costs, and its reuse hints are synthetic. Before tuning the heuristic, replace the toy generator with a frozen trace format containing request order, key/prefix size, inter-arrival/slack, and measured cold/warm costs. Then validate predicted action costs on held-out traces and compare against the existing admission/eviction baselines. Do not spend on a GPU until that replay shows at least 5% improvement on a pre-registered objective with no held-out SLO regression.
