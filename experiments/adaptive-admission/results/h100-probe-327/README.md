# Request 327: H100 controlled-pair probe

## Question

The concurrent eviction trace contains one request-327 output of token ID 287 while the other five runs return 220. All six historical observations report zero cached tokens. The earlier L4 probe found that 220 and 287 were tied at the top of the next-token distribution. This follow-up checks that tie on an H100 with vLLM 0.32.0 and tests six narrow concurrent request pairs.

## Result

The isolated cold request 327 returned token **220**, reported **0 cached tokens**, and assigned both token 220 and token 287 log probability **-5.6167111397**. The top-two gap was exactly zero in the returned values.

The cache-hit control, request 335, used the same cache salt and the same 4,096-token prefix as request 327, with a different final token. It returned token **8** and reported **4,096 cached tokens**, matching the expected full-prefix hit from the historical trace.

We then submitted request 327 together with adjacent trace request 326 in six two-worker client waves. Each request used a distinct cache salt, so the pair did not share KV. Request 327 returned token 220 in all six pairs, reported zero cached tokens in all six, and retained the same 220/287 top-score tie. Request 326 returned token 220 in all six. The pair's actual vLLM scheduler membership was not instrumented; client submission overlap does not prove the requests were processed in the same engine batch.

| Check | Calls | Output token | Cached tokens | Top two |
| --- | ---: | ---: | ---: | --- |
| Request 327 cold | 1 | 220 | 0 | 220 = 287 at -5.616711 |
| Request 335 same-prefix control | 1 | 8 | 4,096 | Not used for the near-tie check |
| Request 327 in concurrent pair | 6 | 220 every time | 0 every time | 220 = 287 in every call |
| Request 326 trace neighbor | 6 | 220 every time | 0 every time | Not analyzed as the target |

The measured median whole-response client time was about 1,120 ms for request 327 and 1,448 ms for request 326 in the paired calls. The client used non-streaming responses, so these are not TTFT values. They include proxy and client overhead, have only six samples, and are not evidence of an optimization or a performance comparison.

## Interpretation and limits

The near-tie carries over from the L4/vLLM 0.33.0 probe to an H100 with the historical vLLM 0.32.0 runtime. That makes score sensitivity a plausible contributor to the archived output difference. This probe did **not** reproduce token 287 and does not establish that batching caused it. It did not recreate the preceding cache and allocator history from the full trace, measure the historical scheduler batch, or rotate fresh server processes. No inference algorithm was changed and no speedup is claimed.

The next evidence step is to replay the trace through request 327 while recording actual vLLM scheduler batch membership and the top-20 logits, with trial order rotated across fresh processes. Allocator tracing remains conditional on what that evidence shows.

## Hardware and runtime

- Runpod product: **H100 SXM**, one GPU; Telemetry identified **H100 80GB HBM3** and reported 80 GiB.
- Region/cloud: **US-NE-1 Secure Cloud**.
- Host: Intel Xeon Platinum 8470, 26 vCPU, 251 GB host RAM according to Pod Details.
- GPU telemetry at the end of the run: 64/80 GiB VRAM, 0% utilization, 36 C, 127 W, 700 W power limit, P0.
- Driver **595.91.07**; Runpod Telemetry CUDA **13.2**.
- Container: `vllm/vllm-openai:v0.32.0`; startup log confirmed vLLM **0.32.0**.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, resolved revision `main` (not pinned to a commit).
- Arguments: `--dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.80 --max-model-len 4608 --enable-prompt-tokens-details`.
- vLLM logs confirm prefix caching enabled, chunked prefill with 8,192 max batched tokens, and 54.21 GiB available KV cache (394,720 KV tokens). Eager mode disabled `torch.compile`, CUDA graphs, and JIT kernel warmup.
- The 30 GB container disk was temporary and no volume was mounted. Runpod reported 8 GB used after model loading. No `nvidia-smi` capture, GPU UUID, or PCIe link state was saved.

Runpod's deployment catalog card displayed 8 vCPU / 125 GB RAM; the provisioned Pod Details displayed 26 vCPU / 251 GB RAM. The record preserves both rather than assuming the catalog summary described the actual placement.

## Cost and lifecycle

Runpod displayed $4.39/hour compute plus $0.004/hour for the 30 GB temporary disk. At the observed 11m19s uptime, the estimated total is **$0.83**; this is computed from the displayed rates, not an account-debit read. After stopping, Runpod showed total **$0.00/hour** and idle disk **$0.00/day**. The Pod is stopped, not terminated.

## Artifacts and reproduction

- [`probe.jsonl`](probe.jsonl): 14 raw observations, including token IDs, actual cached-token counts, top-20 log probabilities, and total response times.
- [`vllm-server.log`](vllm-server.log): vLLM startup and request logs. It also includes one rejected setup payload and a short API smoke request; neither is part of the 14 measured observations. Runpod private proxy peer addresses are redacted; no API key was present in the saved log.
- [`metadata.json`](metadata.json): full settings, exact hardware record, interpretation, and cost estimate.
- [`probe_327.py`](../../probe_327.py): standalone runner; uses only the Python standard library plus the local trace generator.

With the same model served from an OpenAI-compatible vLLM endpoint, the runner can be invoked as follows (the Pod's API key is formed from the Pod ID by the Runpod template; do not put credentials in the JSONL):

```powershell
python experiments/adaptive-admission/probe_327.py `
  --base-url https://<pod-id>-8000.proxy.runpod.net `
  --pod-id <pod-id> `
  --output experiments/adaptive-admission/results/h100-probe-327/probe.jsonl
```

The data is diagnostic and bounded. The pair calls check client overlap and output stability; they are not a replacement for a full trace replay with engine-level batch instrumentation.
