"""Focused request-327 replay against a vLLM OpenAI-compatible endpoint."""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from engine_experiment import workload


MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def post_json(url, api_key, body, timeout=300):
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 foundry-327-probe/1.0",
        },
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc


def wait_ready(base_url, api_key, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0 foundry-327-probe/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                payload = json.loads(response.read())
            if any(item.get("id") == MODEL for item in payload.get("data", [])):
                return
            last_error = RuntimeError(f"Expected model {MODEL}; got {payload}")
        except Exception as exc:  # noqa: BLE001 - retry only until the bounded deadline
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"vLLM endpoint did not become ready: {last_error}")


def measure(base_url, api_key, item, label, cache_salt):
    index, (phase, key, length) = item
    prompt_salt = f"admission-bench-{phase}-{key}"
    tokens = random.Random(prompt_salt).choices(range(1000, 8000), k=length)
    payload = {
        "model": MODEL,
        "prompt": tokens + [9000 + index],
        "cache_salt": cache_salt,
        "temperature": 0,
        "max_tokens": 1,
        "stream": False,
        "ignore_eos": True,
        "add_special_tokens": False,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "logprobs": 20,
    }
    started = time.perf_counter()
    with post_json(f"{base_url}/v1/completions", api_key, payload) as response:
        result = json.loads(response.read())
    elapsed_ms = (time.perf_counter() - started) * 1000
    choices = result.get("choices", [])
    if len(choices) != 1:
        raise RuntimeError(f"Expected one completion choice for {label}")
    choice = choices[0]
    usage = result.get("usage") or {}
    token_ids = choice.get("token_ids") or []
    logprobs = choice.get("logprobs") or {}
    top_logprobs = logprobs.get("top_logprobs") or []
    output_text = choice.get("text", "")
    if len(token_ids) != 1:
        raise RuntimeError(f"Incomplete response for {label}")
    if usage.get("prompt_tokens") != length + 1:
        raise RuntimeError(
            f"Prompt token count changed for {label}: {usage.get('prompt_tokens')} != {length + 1}"
        )
    if len(top_logprobs) != 1 or not top_logprobs[0]:
        raise RuntimeError(f"Expected top-20 log probabilities for {label}")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "label": label,
        "index": index,
        "phase": phase,
        "key": key,
        "prompt_tokens_expected": length + 1,
        "prompt_tokens_reported": usage["prompt_tokens"],
        "cached_tokens": details.get("cached_tokens"),
        "cache_salt": cache_salt,
        "output_token_ids": token_ids,
        "output_text": output_text,
        "ttft_ms": None,
        "total_ms": elapsed_ms,
        "top_logprobs": top_logprobs[0],
    }


def emit(output, row):
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    with open(output, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pod-id", required=True,
                        help="Used to construct the benchmark Pod's configured sk-<pod-id> API key")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ready-timeout", type=int, default=240)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    api_key = f"sk-{args.pod_id}"
    open(args.output, "w", encoding="utf-8").close()
    wait_ready(base_url, api_key, args.ready_timeout)

    trace = list(enumerate(workload(0)))
    target = trace[327]
    control = trace[335]
    neighbor = trace[326]
    if target[1] != (1, 39, 4096) or control[1] != (1, 39, 4096):
        raise RuntimeError(f"Unexpected trace rows: {target}, {control}")

    # Cold request-327 and its known same-prefix repeat control at request 335.
    for label, item, salt in (
        ("isolated-327-cold", target, "foundry-327-h100-isolated"),
        ("isolated-335-same-prefix-control", control, "foundry-327-h100-isolated"),
    ):
        emit(args.output, measure(base_url, api_key, item, label, salt))

    # Concurrent request 327 with its adjacent trace neighbor (326). Distinct cache
    # salts prevent either request from reusing KV written by the other request.
    for trial in range(1, 7):
        salt = f"foundry-327-h100-pair-{trial}"
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [
                workers.submit(measure, base_url, api_key, target,
                               f"pair-{trial}-request-327", salt + "-target"),
                workers.submit(measure, base_url, api_key, neighbor,
                               f"pair-{trial}-request-326-neighbor", salt + "-neighbor"),
            ]
            rows = [future.result() for future in futures]
        for row in rows:
            row["pair_trial"] = trial
            row["pair_max_request_ms"] = max(item["total_ms"] for item in rows)
            emit(args.output, row)


if __name__ == "__main__":
    main()
