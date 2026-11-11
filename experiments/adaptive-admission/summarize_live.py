"""Summarize the six trials in the live cost-eviction benchmark archive."""

import argparse
import json
import math
import statistics
import tarfile
from pathlib import Path


MODES = ("lru", "adaptive", "cost")


def summarize(rows):
    latencies = sorted(row["ttft_ms"] for row in rows)
    phase_times = {}
    for row in rows:
        phase_times[row["phase"]] = max(phase_times.get(row["phase"], 0), row["wall_ms"])
    return {
        "requests": len(rows),
        "uncached": sum(row["prompt_tokens"] - row["cached_tokens"] for row in rows),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(latencies[math.ceil(0.95 * len(rows)) - 1], 2),
        "req_per_s": round(1000 * len(rows) / sum(phase_times.values()), 2),
    }


def main(path):
    with tarfile.open(path) as archive:
        trials = {}
        for trial in (1, 2):
            for mode in MODES:
                name = f"{trial}-{mode}"
                rows = [json.loads(line) for line in archive.extractfile(f"gpu-{name}.jsonl")]
                expected = [256 * phase + offset for phase in range(4) for offset in range(128)]
                if [row["index"] for row in rows] != expected:
                    raise ValueError(f"Incomplete trial {name}")
                log = archive.extractfile(f"vllm-{name}.log").read().decode()
                if "GPU KV cache size: 32,800 tokens" not in log:
                    raise ValueError(f"Unexpected cache size in trial {name}")
                if ("Cost-ranked KV block eviction enabled" in log) != (mode == "cost"):
                    raise ValueError(f"Unexpected eviction mode in trial {name}")
                trials[trial, mode] = rows
                print(name, summarize(rows))

    reference = trials[1, "lru"]
    for name, rows in trials.items():
        if any((a["salt"], a["prompt_tokens"]) != (b["salt"], b["prompt_tokens"])
               for a, b in zip(reference, rows)):
            raise ValueError(f"Prompt mismatch in trial {name}")
        matches = sum(a["output_token"] == b["output_token"] for a, b in zip(reference, rows))
        print(name, "output matches with first LRU trial:", matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
