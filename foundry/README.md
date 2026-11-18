# Foundry C core (prototype)

This directory starts the unified cost model proposed in the repository review. The Foundry core and new decision tools are C11 with a C API; no Python package or `pip install` distribution is planned. Existing Python and C++ vLLM experiment harnesses remain historical experiment code and are not part of the Foundry library.

The historical `experiments/adaptive-admission/probe_327.py` runner is unchanged: it resolves the model's `main` revision and follows the Runpod template's Pod-ID API-key convention. Before another probe, pin the model revision and move credentials to secure input; port that runner to C if it is brought into the new Foundry tooling.

## Current scope

`cost_model.c` stores estimates keyed by GPU name and prefix length. It updates reload, recompute, and keep costs with an exponentially weighted moving average (EWMA). A query uses an exact GPU/prefix entry, then the nearest measured prefix on the same GPU, then a caller-supplied prior. The caller can detect fallback and inspect the sample count. Missing observations are represented by `NAN` and do not change that cost field.

`foundry_choose_reload_or_recompute` scores reload and recompute in estimated critical-path milliseconds, including queue delay and a linear deadline-lateness penalty. It falls back to recompute if the key is estimated by a prior/nearest bucket, samples are below the caller's threshold, or either action is unmeasured. This conservative rule avoids acting on generic priors as though they were exact measurements.

`replay.c` is a deterministic synthetic CPU simulation comparing true LRU, a future-informed oracle reuse-rank comparator, and slack-priced admission with LRU eviction. It emits aggregate missed tokens, modeled deadline misses, and a modeled objective. This is an illustrative policy exercise: it does not ingest the archived request trace, model vLLM blocks/scheduling, calibrate costs from GPU runs, estimate confidence intervals, or establish an inference speedup. The oracle reuse counts are privileged inputs; the result must not be presented as an online controller result.

The corrected 20-seed run **failed** the 5% pre-GPU gate: slack-priced admission improved the toy objective by 1.72% versus LRU but increased modeled deadline misses by 6.38%; it was 14.30% worse on objective than the future-informed oracle comparator. Results, command, and limitations are recorded in [`results/cpu-replay-2026-11-15.md`](results/cpu-replay-2026-11-15.md) and [`results/cpu-replay-2026-11-15.csv`](results/cpu-replay-2026-11-15.csv). This is a negative result for this particular toy heuristic, not a GPU result or a test of the full shared-cost hypothesis. No GPU run is warranted until a frozen historical trace replay passes the gate.

The fixed 128-key estimator table is a prototype bound, not a production sizing claim. The nearest-prefix fallback is a guess and must remain visible to callers. No policy is integrated into vLLM.

## Real-GPU prefix-cache calibration

[`gpu_calibrate.c`](gpu_calibrate.c) is a C11 streaming client for measuring cold requests and exact-prefix hits against a vLLM OpenAI-compatible endpoint. The November 15, 2026 calibration ran on a Runpod H100 SXM with vLLM 0.32.0 and Qwen3-4B-Instruct-2507. It verified full cache hits at 512, 1,024, 2,048, and 4,096 prefix tokens. With two observations per condition, client-observed TTFT was noisy and warm requests were slower at 512 and 4,096 tokens. This validates that the cache path and client work; it does not show a speedup or evaluate the cost-model policy. See the [hardware record and analysis](results/h100-sxm-prefix-calibration.md) and [raw CSV](results/h100_sxm_prefix_calibration.csv).

The November 18, 2026 AMD run used one Runpod MI350X with the official vLLM ROCm image and vLLM 0.34.0+rocm100. The same C11 client verified all 8 cold misses and all 8 exact-prefix warm hits. Warm means were lower in this two-repeat calibration, but the result is not a speedup claim and is not directly comparable to the H100 numbers: the AMD client ran over loopback and used a newer vLLM version. See the [exact hardware record and analysis](results/amd-mi350x-prefix-calibration.md) and [raw CSV](results/amd-mi350x-prefix-calibration.csv).

Compile with a C11 compiler and run against a secured vLLM endpoint:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic foundry/gpu_calibrate.c -o gpu_calibrate
./gpu_calibrate https://HOST-8000.proxy.runpod.net MODEL API_KEY 2
```

Do not commit API keys or include them in benchmark logs.

## Build and run the C unit test and synthetic replay

With a C11 compiler:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -Ifoundry foundry/cost_model.c tests/test_cost_model.c \
  -lm -o cost_model_tests
./cost_model_tests
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -Ifoundry foundry/replay.c -lm -o foundry_replay
./foundry_replay 32768 20
```

On Windows, run from a Visual Studio Developer Command Prompt:

```powershell
cl /nologo /std:c11 /W4 /WX /TC /I foundry foundry\cost_model.c tests\test_cost_model.c /Fe:cost_model_tests.exe
.\cost_model_tests.exe
cl /nologo /std:c11 /W4 /WX /TC /I foundry foundry\replay.c /Fe:foundry_replay.exe
.\foundry_replay.exe 32768 20
```

Or run `./foundry/run_replay.ps1` in PowerShell with Clang/GCC available on PATH.

The unit test covers invalid decay, caller-supplied prior fallback, exact lookup, EWMA updates, missing observations, nearest-prefix fallback, GPU isolation, reload/recompute choice, and conservative fallback. Replay arguments are cache capacity in tokens and number of seeds.

GitHub Actions runs the C11 unit test, compiles the replay, checks that its three documented policies have distinct metrics, and validates every committed JSONL file line by line. The H100 request-327 probe artifacts were repaired to use physical newlines before adding this check.

## Next gates

1. Replace the synthetic reuse oracle with a frozen, explicitly versioned Trace v1 input and replay historical order.
2. Add model uncertainty and reject policies whose held-out prediction error exceeds the pre-registered bound.
3. Add calibration serialization in C and reproduce the measured latency table with repeated observations. Measure the 512-token bucket directly.
4. Add C replay tests against archived traces, keeping the old harness runnable.
5. Only then add admission/eviction counterfactuals and an ablation runner.

The request-327 probes and remaining evidence gaps are recorded in [`docs/327-probe-plan.md`](../docs/327-probe-plan.md). The proposed next work is a deadline-coupled C controller with a CPU replay gate before any additional GPU spend; see [`docs/next-optimization-plan.md`](../docs/next-optimization-plan.md).
