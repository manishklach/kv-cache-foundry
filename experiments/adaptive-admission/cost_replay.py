"""Compare cost-aware prefix eviction on synthetic traces calibrated to H100 TTFT."""

import argparse
import csv
import json
import statistics
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

from benchmark import CAPACITIES, EPOCH, SEEDS, Cache, trace


HERE = Path(__file__).resolve().parent


def calibrate():
    timings = defaultdict(list)
    with tarfile.open(HERE / "kv-admission-gpu-results.tar.gz") as archive:
        for line in archive.extractfile("gpu-lru.jsonl"):
            row = json.loads(line)
            length = row["prompt_tokens"] - 1
            if row["index"] and row["cached_tokens"] in (0, length):
                timings[length, bool(row["cached_tokens"])].append(row["ttft_ms"])
    costs = {
        length: statistics.median(timings[length, False])
        - statistics.median(timings[length, True])
        for length in (1024, 2048, 4096)
    }
    # There are no full hits for 512-token prefixes in the calibration run.
    costs[512] = costs[1024] / 2
    return costs


class CostCache:
    """GreedyDual-Size with age and estimated rebuild time per resident token."""

    def __init__(self, capacity, costs):
        self.capacity = capacity
        self.costs = costs
        self.entries = {}
        self.tokens = 0
        self.age = 0

    def access(self, key, length):
        density = self.costs[length] / length
        if key in self.entries:
            self.entries.pop(key)
            self.entries[key] = (length, self.age + density)
            return 0
        while self.tokens + length > self.capacity:
            victim = min(self.entries, key=lambda item: self.entries[item][1])
            victim_length, self.age = self.entries.pop(victim)
            self.tokens -= victim_length
        self.entries[key] = (length, self.age + density)
        self.tokens += length
        return length


def native_misses(executable, capacity, costs, requests):
    keys = {}
    encoded = []
    for _, key, length in requests:
        identity = keys.setdefault(key, len(keys))
        encoded.append(f"{identity} {length}\n")
    command = [str(executable), str(capacity)]
    command.extend(str(costs[length]) for length in (512, 1024, 2048, 4096))
    result = subprocess.run(command, input="".join(encoded), text=True,
                            capture_output=True, check=True)
    misses = [int(line) for line in result.stdout.splitlines()]
    if len(misses) != len(requests):
        raise RuntimeError("C replay returned the wrong number of decisions")
    return misses


def replay(seed, capacity, costs, executable=None):
    lru, second, adaptive = (Cache(capacity) for _ in range(3))
    weighted = CostCache(capacity, costs)
    linear = CostCache(capacity, {length: length for length in costs})
    totals = defaultdict(lambda: [0, 0.0])
    shadow = [0, 0]
    admit_first = True
    requests = list(trace(seed))
    if executable:
        native_weighted = native_misses(executable, capacity, costs, requests)
        native_linear = native_misses(executable, capacity, linear.costs, requests)
    for index, (phase, key, length) in enumerate(requests, start=1):
        first_miss = lru.access(key, length, True)
        second_miss = second.access(key, length, False)
        misses = {
            "lru": first_miss,
            "token_admission": adaptive.access(key, length, admit_first),
            "linear_cost_eviction": linear.access(key, length),
            "h100_cost_eviction": weighted.access(key, length),
        }
        if executable and (native_weighted[index - 1] != misses["h100_cost_eviction"]
                           or native_linear[index - 1] != misses["linear_cost_eviction"]):
            raise RuntimeError(f"C/Python cache disagreement at request {index}")
        for policy, miss in misses.items():
            totals[phase, policy][0] += miss
            if miss:
                totals[phase, policy][1] += costs[length]
        shadow[0] += first_miss
        shadow[1] += second_miss
        if index % EPOCH == 0:
            if shadow[0] != shadow[1]:
                admit_first = shadow[0] < shadow[1]
            shadow = [0, 0]
    return [(capacity, seed, phase, policy, *result)
            for (phase, policy), result in totals.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-replay", type=Path, help="Compare native C decisions with Python")
    args = parser.parse_args()
    executable = args.c_replay.resolve() if args.c_replay else None
    costs = calibrate()
    print("Estimated rebuild milliseconds by prefix size:", costs)
    # The GPU trace calibrates costs; these independently generated traces score policies.
    rows = [row for capacity in CAPACITIES for seed in SEEDS
            for row in replay(seed + 100, capacity, costs, executable)]
    if executable:
        print("C and Python agreed on every eviction-policy hit/miss decision.")
    with (HERE / "cost-replay.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("capacity", "seed", "phase", "policy", "uncached_tokens",
                         "estimated_rebuild_ms"))
        writer.writerows(rows)
    totals = defaultdict(float)
    for capacity, _, _, policy, _, cost in rows:
        totals[capacity, policy] += cost
    for (capacity, policy), cost in totals.items():
        change = (cost / totals[capacity, "lru"] - 1) * 100
        print(f"{capacity:>6,} {policy:22s} {cost:>13,.0f} ms  {change:+.1f}% vs LRU")


if __name__ == "__main__":
    main()
