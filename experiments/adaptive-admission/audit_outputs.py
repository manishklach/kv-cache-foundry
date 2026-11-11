"""Audit archived output differences and replay the same tokenized workload."""

import argparse
import json
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

from engine_experiment import measure, workload


ARCHIVES = {
    "admission": ("kv-admission-gpu-results.tar.gz", "gpu-lru.jsonl", 256),
    "eviction": ("cost-eviction-live-results.tar.gz", "gpu-1-lru.jsonl", 128),
}


def emit(row):
    print(json.dumps(row), flush=True)


def load(experiment):
    name, baseline, limit = ARCHIVES[experiment]
    items = [
        item
        for _, phase in groupby(enumerate(workload(0)), key=lambda item: item[1][0])
        for item in list(phase)[:limit]
    ]
    expected = [(i, phase, f"admission-bench-{phase}-{key}", length + 1)
                for i, (phase, key, length) in items]
    runs = {}
    with tarfile.open(Path(__file__).with_name(name)) as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".jsonl"):
                with archive.extractfile(member) as source:
                    rows = [json.loads(line) for line in source]
                actual = [(r["index"], r["phase"], r["salt"], r["prompt_tokens"])
                          for r in rows]
                if actual != expected:
                    raise ValueError(f"{member.name} does not match the generated trace")
                runs[member.name] = rows
    reference = runs.pop(baseline)
    return items, reference, runs


def differences(reference, runs):
    for name, rows in sorted(runs.items()):
        for before, after in zip(reference, rows):
            if before["output_token"] != after["output_token"]:
                yield dict(run=name, index=before["index"], salt=before["salt"],
                           baseline_token=before["output_token"], output_token=after["output_token"],
                           baseline_cached=before["cached_tokens"], cached=after["cached_tokens"])


def replay(items, targets, args):
    def request(item):
        row = measure(args.model, args.mode, item,
                      logprobs=5 if item[0] in targets else None)
        if args.mode == "off" and row["cached_tokens"]:
            raise RuntimeError("The cache-disabled server reported a cache hit")
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as workers:
        if not args.isolated_only:
            for _, phase in groupby(items, key=lambda item: item[1][0]):
                for row in workers.map(request, phase):
                    if row["index"] in targets:
                        emit(dict(row, probe="trace"))

        # The wire salt isolates each pair. Token generation still uses the original salt.
        probes = [(item, f"admission-bench-audit-{uuid.uuid4().hex}")
                  for item in items if item[0] in targets]
        for probe in ("cold", "repeat"):
            def isolated(pair):
                item, salt = pair
                row = measure(args.model, args.mode, item, cache_salt=salt, logprobs=5)
                if (probe == "cold" or args.mode == "off") and row["cached_tokens"]:
                    raise RuntimeError(f"Unexpected cache hit during {probe}: {row}")
                return dict(row, probe=probe)

            for row in workers.map(isolated, probes):
                emit(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=ARCHIVES)
    parser.add_argument("--mode", choices=("off", "lru", "adaptive", "cost"),
                        help="Replay against a running server with this policy")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument("--index", type=int, action="append",
                        help="Only collect probes for this request index, repeatable")
    parser.add_argument("--isolated-only", action="store_true",
                        help="Skip the historical trace and send only cold and repeat probes")
    args = parser.parse_args()
    items, reference, runs = load(args.experiment)
    mismatches = [row for row in differences(reference, runs)
                  if args.index is None or row["index"] in args.index]
    targets = {row["index"] for row in mismatches}
    if args.index is not None and targets != set(args.index):
        parser.error("Every requested index must appear in the archived mismatches")
    if args.mode is None:
        for row in mismatches:
            emit(row)
    else:
        replay(items, targets, args)


if __name__ == "__main__":
    main()
