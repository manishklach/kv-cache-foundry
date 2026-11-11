"""Patch the vLLM 0.32.0 scheduler and replay a short tokenized workload."""

import argparse
import importlib.metadata
import json
import random
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

from benchmark import Cache


class Admission:
    """Choose once per request, using two metadata-only shadow caches."""

    def __init__(self):
        self.first = Cache(32_768)
        self.second = Cache(32_768)
        self.recent = OrderedDict()
        self.decisions = {}
        self.cost = [0, 0]
        self.count = 0
        self.admit_first = True

    def allows(self, request):
        key = request.cache_salt
        if not key or not key.startswith("admission-bench-"):
            return True
        request_id = request.request_id
        if request_id in self.decisions:
            return self.decisions[request_id]
        length = request.num_prompt_tokens - 1
        allowed = self.admit_first or key in self.recent
        self.decisions[request_id] = allowed
        self.recent[key] = None
        self.recent.move_to_end(key)
        if len(self.recent) > 128:
            self.recent.popitem(last=False)
        self.cost[0] += self.first.access(key, length, True)
        self.cost[1] += self.second.access(key, length, False)
        self.count += 1
        if self.count % 32 == 0:
            if self.cost[0] != self.cost[1]:
                self.admit_first = self.cost[0] < self.cost[1]
            self.cost = [0, 0]
        return allowed

    def free(self, request_id):
        self.decisions.pop(request_id, None)


def install_patch():
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.32.0":
        raise RuntimeError("The experiment requires vLLM 0.32.0")
    path = Path(distribution.locate_file("vllm/v1/core/kv_cache_manager.py"))
    source = path.read_text(encoding="utf-8")
    edits = {
        "        self.enable_caching = enable_caching":
            "        self.enable_caching = enable_caching\n"
            "        import os\n"
            "        from engine_experiment import Admission\n"
            "        self.admission = Admission() if os.getenv('KV_ADMISSION') == 'adaptive' else None",
        "        self.coordinator.cache_blocks(request, num_tokens_to_cache)":
            "        if self.admission is None or self.admission.allows(request):\n"
            "            self.coordinator.cache_blocks(request, num_tokens_to_cache)",
        "        self.coordinator.free(request.request_id)":
            "        if self.admission is not None:\n"
            "            self.admission.free(request.request_id)\n"
            "        self.coordinator.free(request.request_id)",
        "        if self.enable_caching:\n            self.coordinator.cache_blocks(request, num_computed_tokens)":
            "        if self.enable_caching and (self.admission is None or self.admission.allows(request)):\n"
            "            self.coordinator.cache_blocks(request, num_computed_tokens)",
    }
    if all(new in source for new in edits.values()):
        return
    for old, new in edits.items():
        if source.count(old) != 1 or new in source:
            raise RuntimeError("Unexpected vLLM source or patch already applied")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")


def workload(seed):
    rng = random.Random(seed)
    for phase in range(4):
        if phase % 2 == 0:
            for index in range(256):
                if rng.random() < 0.45:
                    yield phase, rng.randrange(7), 4096
                else:
                    yield phase, index + 8, rng.choice((512, 1024, 2048))
        else:
            for group in range(16):
                documents = [(group * 8 + i, rng.choice((1024, 2048, 4096)))
                             for i in range(8)]
                for key, length in documents + documents:
                    yield phase, key, length


def open_request(request):
    try:
        return urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error


def measure(model, mode, item):
    index, (phase, key, length) = item
    salt = f"admission-bench-{phase}-{key}"
    tokens = random.Random(salt).choices(range(1000, 8000), k=length)
    payload = dict(model=model, prompt=tokens + [9000 + index], cache_salt=salt,
                   temperature=0, max_tokens=1, stream=True, ignore_eos=True,
                   add_special_tokens=False, return_token_ids=True,
                   stream_options={"include_usage": True})
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
                                 json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    token_ids = []
    with open_request(req) as response:
        for line in response:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            for choice in event.get("choices", []):
                if choice.get("token_ids"):
                    if first is None:
                        first = time.perf_counter()
                    token_ids.extend(choice["token_ids"])
                output.append(choice.get("text", ""))
            usage = event.get("usage") or usage
    if first is None or usage is None or len(token_ids) != 1:
        raise RuntimeError("Incomplete vLLM response")
    if usage["prompt_tokens"] != length + 1:
        raise RuntimeError("Server changed the supplied token count")
    details = usage.get("prompt_tokens_details") or {}
    return dict(mode=mode, index=index, phase=phase, salt=salt,
                          prompt_tokens=usage["prompt_tokens"],
                          cached_tokens=details.get("cached_tokens") or 0,
                          ttft_ms=(first - started) * 1000,
                          total_ms=(time.perf_counter() - started) * 1000,
                          output_token=token_ids[0], text="".join(output))


def run(seed, mode, model, concurrency, limit):
    requests = enumerate(workload(seed))
    with ThreadPoolExecutor(max_workers=concurrency) as workers:
        for _, group in groupby(requests, key=lambda item: item[1][0]):
            items = list(group)[:limit]
            started = time.perf_counter()
            for row in workers.map(lambda item: measure(model, mode, item), items):
                row["wall_ms"] = (time.perf_counter() - started) * 1000
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("patch", "lru", "adaptive", "cost"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests-per-phase", type=int, default=256)
    args = parser.parse_args()
    if args.concurrency < 1 or not 1 <= args.requests_per_phase <= 256:
        parser.error("Expected positive concurrency and 1 to 256 requests per phase")
    if args.mode == "patch":
        install_patch()
    else:
        run(args.seed, args.mode, args.model, args.concurrency, args.requests_per_phase)
