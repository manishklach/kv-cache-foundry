# Experimental design

Prefix caching lets vLLM reuse the attention key/value state for tokens shared at the start of multiple requests. In document QA, the order of the document and question changes how many tokens can be reused. [vLLM prefix caching design](https://docs.vllm.ai/en/v0.32.0/design/prefix_caching/)

## Workload

The client generates a synthetic manual with 256 records. Device index `i` has service code `1000 + 7 × i`. Request index `j` asks about device `(17 × j) mod 256`. Each condition uses the same 51 questions in the same order.

The client sends one user message through vLLM's streaming chat completions API. The content begins with a run identifier and layout name, followed by the manual and question in the selected order. Both layouts end with the same instruction to return only the code. The model uses temperature 0 and a 12-token output limit.

## Comparisons

The runner starts separate vLLM processes for cache off and on. Each process runs question-first followed by manual-first. The client sends a short warmup prompt before each layout, then sends one workload request at a time. The first workload request populates the shared prefix for manual-first with caching enabled. Its latency is recorded separately. A fresh UUID per client invocation reduces prefix reuse across invocations. The UUIDs differed by one token between cache modes in the recorded run.

The two cache-off rows control for prompt order. Comparing cache off and on within a layout estimates the latency effect of caching for that layout. Comparing exact matches between the two layouts tests whether the prompt rewrite changed task performance.

## Measurements

- **Time to first token (TTFT)** is wall-clock time from starting the loopback HTTP request to the first nonempty streamed text fragment. It includes HTTP and serving overhead on the Pod.
- **Median and p95 TTFT** use the 50 requests after the first workload request. The p95 uses nearest rank.
- **Cached prompt tokens** come from vLLM's streamed usage response. The summary reports the median over the same 50 requests.
- **Exact match** scores all 51 workload requests after stripping surrounding whitespace and trailing periods.
- **Per-request records** also include total latency, prompt token count, generated token count, expected answer, and returned text.
