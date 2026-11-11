"""Replay KV prefix admission policies against changing synthetic traffic."""

import csv
import random
from collections import OrderedDict
from itertools import groupby
from pathlib import Path


CAPACITIES = (16_384, 32_768, 65_536)
EPOCH = 256
GHOST_CAPACITY = 128
SEEDS = range(20)


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.recent_misses = OrderedDict()
        self.tokens = 0

    def access(self, key, length, admit_first):
        if key in self.entries:
            self.entries.move_to_end(key)
            return 0

        repeated = key in self.recent_misses
        self.recent_misses[key] = None
        self.recent_misses.move_to_end(key)
        if len(self.recent_misses) > GHOST_CAPACITY:
            self.recent_misses.popitem(last=False)

        if length <= self.capacity and (admit_first or repeated):
            while self.tokens + length > self.capacity:
                _, evicted_length = self.entries.popitem(last=False)
                self.tokens -= evicted_length
            self.entries[key] = length
            self.tokens += length
        return length


def trace(seed):
    rng = random.Random(seed)
    for phase in range(4):
        if phase % 2 == 0:
            for index in range(2048):
                if rng.random() < 0.45:
                    yield phase, ("manual", phase, rng.randrange(7)), 4096
                else:
                    yield phase, ("scan", phase, index), rng.choice((512, 1024, 2048))
        else:
            # Every document is used twice, with a short delay before reuse.
            for group in range(128):
                documents = [(("pair", phase, group, i), rng.choice((1024, 2048, 4096)))
                             for i in range(8)]
                for key, length in documents + documents:
                    yield phase, key, length


def replay(seed, capacity):
    first, second, adaptive = (Cache(capacity) for _ in range(3))
    admit_first = True
    scores = [0, 0]
    rows = []
    for phase, requests in groupby(trace(seed), key=lambda request: request[0]):
        totals = [0, 0, 0]
        first_epochs = 0
        for index, (_, key, length) in enumerate(requests):
            if index % EPOCH == 0:
                first_epochs += admit_first
            misses = (
                first.access(key, length, True),
                second.access(key, length, False),
                adaptive.access(key, length, admit_first),
            )
            for policy, cost in enumerate(misses):
                totals[policy] += cost
            scores[0] += misses[0]
            scores[1] += misses[1]
            if (index + 1) % EPOCH == 0:
                if scores[0] != scores[1]:
                    admit_first = scores[0] < scores[1]
                scores = [0, 0]
        rows.append((capacity, seed, phase, *totals, first_epochs))
    return rows


def main():
    rows = [row for capacity in CAPACITIES for seed in SEEDS for row in replay(seed, capacity)]
    destination = Path(__file__).with_name("results.csv")
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("capacity", "seed", "phase", "lru_missed_tokens", "second_hit_missed_tokens",
                         "adaptive_missed_tokens", "adaptive_first_hit_epochs"))
        writer.writerows(rows)
    for capacity in CAPACITIES:
        selected = [row for row in rows if row[0] == capacity]
        for name, subset in (
            ("scan", [row for row in selected if row[2] % 2 == 0]),
            ("paired", [row for row in selected if row[2] % 2 == 1]),
            ("total", selected),
        ):
            first, second, adaptive = (sum(row[index] for row in subset) for index in (3, 4, 5))
            print(f"{capacity:>6,} {name:7} LRU {first:>11,}  "
                  f"second-hit {second:>11,}  adaptive {adaptive:>11,}")


if __name__ == "__main__":
    main()
