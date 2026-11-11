"""Measure how prompt layout affects vLLM prefix reuse over streaming HTTP."""

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
import uuid


def request(args, prompt, answer):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 12,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/chat/completions", json.dumps(payload).encode(), headers
    )
    started = time.perf_counter()
    first_token = None
    fragments = []
    usage = {}
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                if fragment := (choice.get("delta") or {}).get("content"):
                    if first_token is None:
                        first_token = time.perf_counter()
                    fragments.append(fragment)
    if first_token is None or "completion_tokens" not in usage:
        raise RuntimeError("Server returned an incomplete streaming response")
    output = "".join(fragments).strip()
    return {
        "ttft_ms": (first_token - started) * 1000,
        "total_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "output_tokens": usage["completion_tokens"],
        "output": output,
        "expected": answer,
        "correct": output.rstrip(".").strip() == answer,
    }


def benchmark(args):
    records = [f"Device {i:04d}: service code {1000 + i * 7}." for i in range(256)]
    manual = "Manual:\n" + "\n".join(records)
    # Use a fresh first token sequence to prevent cache reuse across benchmark runs.
    run_id = uuid.uuid4().hex
    for layout in ("question-first", "manual-first"):
        request(args, "What color is a clear daytime sky? Reply with one word.", "blue")
        samples = []
        for index in range(args.requests):
            device = (index * 17) % len(records)
            question = f"What is the service code for device {device:04d}?"
            sections = (question, manual) if layout == "question-first" else (manual, question)
            prompt = f"{run_id} {layout}\n" + "\n\n".join(sections)
            prompt += "\n\nReturn only the service code."
            samples.append(request(args, prompt, str(1000 + device * 7)))
        warm = samples[1:]
        ttfts = sorted(sample["ttft_ms"] for sample in warm)
        cached = [sample["cached_tokens"] for sample in warm]
        print(json.dumps({
            "cache": args.cache,
            "layout": layout,
            "first_request_ms": samples[0]["ttft_ms"],
            "median_ttft_ms": statistics.median(ttfts),
            "p95_ttft_ms": ttfts[math.ceil(0.95 * len(ttfts)) - 1],
            "median_cached_tokens": None if None in cached else statistics.median(cached),
            "accuracy": sum(sample["correct"] for sample in samples) / len(samples),
            "samples": samples,
        }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", choices=("on", "off"), required=True,
                        help="Label for the server's actual prefix caching setting")
    parser.add_argument("--requests", type=int, default=51)
    args = parser.parse_args()
    if not 2 <= args.requests <= 256:
        parser.error("--requests must be between 2 and 256")
    benchmark(args)
