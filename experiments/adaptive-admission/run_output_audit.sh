#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

output=$(mktemp -d "${TMPDIR:-/tmp}/kv-output-audit.XXXXXXXX")
echo "Audit results: $output"
model=Qwen/Qwen3-4B-Instruct-2507
server_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
trap cleanup EXIT

cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -shared -fPIC block_rank.c -o block_rank.so
python engine_experiment.py patch
python live_eviction.py
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >"$output/gpu.csv"

for concurrency in 1 4; do
    for mode in off lru adaptive cost; do
      for experiment in admission eviction; do
        if [[ "$mode:$experiment" == cost:admission ]]; then continue; fi
        if [[ "${FOCUSED:-0}" == 1 && "$mode:$experiment" == adaptive:eviction ]]; then continue; fi
        caching=--enable-prefix-caching
        if [[ "$mode" == off ]]; then
            caching=--no-enable-prefix-caching
        fi
        log="$output/vllm-$experiment-$concurrency-$mode.log"
        KV_ADMISSION="$mode" vllm serve "$model" --host 127.0.0.1 \
            --max-model-len 4608 --block-size 32 --num-gpu-blocks-override 1025 \
            --gpu-memory-utilization 0.5 --max-num-seqs "$concurrency" --enforce-eager \
            "$caching" --enable-prompt-tokens-details >"$log" 2>&1 &
        server_pid=$!
        deadline=$((SECONDS + 600))
        until curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; do
            if ! kill -0 "$server_pid" 2>/dev/null || (( SECONDS >= deadline )); then
                tail -60 "$log"
                exit 1
            fi
            sleep 2
        done
        probes=()
        if [[ "${FOCUSED:-0}" == 1 ]]; then
            probes=(--isolated-only)
            if [[ "$experiment" == eviction ]]; then
                probes+=(--index 858)
            fi
        fi
        python audit_outputs.py "$experiment" --mode "$mode" --concurrency "$concurrency" \
            --model "$model" "${probes[@]}" >"$output/$experiment-$concurrency-$mode.jsonl"
        cleanup
      done
    done
done
