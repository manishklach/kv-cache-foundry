# Foundry C core (prototype)

This directory starts the unified cost model proposed in the repository review. New decision logic here is written in C. The existing Python vLLM experiment harness remains historical experiment code; this prototype does not add Python dependencies or replace those experiments.

## Current scope

`cost_model.c` stores estimates keyed by GPU name and prefix length. It updates reload, recompute, and keep costs with an exponentially weighted moving average (EWMA). A query uses an exact GPU/prefix entry, then the nearest measured prefix on the same GPU, then a caller-supplied prior. The caller can detect fallback and inspect the sample count. Missing observations are represented by `NAN` and do not change that cost field.

This is deliberately a small CPU-only foundation. It does not yet measure GPU costs, estimate uncertainty, interpolate between prefix sizes, calculate dollars, persist calibration data, or make admission/eviction/reload/scheduler decisions. The fixed 128-key table is a prototype bound, not a production sizing claim. The nearest-prefix fallback is a guess and must remain visible to callers.

## Build and run the C unit test

With a C11 compiler:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -Ifoundry foundry/cost_model.c tests/test_cost_model.c \
  -lm -o cost_model_tests
./cost_model_tests
```

On Windows, run from a Visual Studio Developer Command Prompt:

```powershell
cl /nologo /std:c11 /W4 /WX /TC /I foundry foundry\cost_model.c tests\test_cost_model.c /Fe:cost_model_tests.exe
.\cost_model_tests.exe
```

The test covers invalid decay, caller-supplied prior fallback, exact lookup, EWMA updates, missing observations, nearest-prefix fallback, and GPU isolation.

## Next gates

1. Add input validation and calibration serialization in C.
2. Reproduce the existing measured latency table with repeated measurements before using it as policy input. Measure the 512-token bucket directly.
3. Add C replay tests against the existing archived traces, keeping the old experiment harness runnable.
4. Only then build a C policy consumer and an ablation runner.

The request-327 probe design and its current funding constraint are recorded in [`docs/327-probe-plan.md`](../docs/327-probe-plan.md).
