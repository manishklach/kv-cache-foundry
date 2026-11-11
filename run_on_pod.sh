#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
model=Qwen/Qwen3-4B-Instruct-2507
python -m pip install 'vllm==0.32.0'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
vllm --version

server_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for mode in off on; do
    cache_flag=--no-enable-prefix-caching
    if [[ "$mode" == on ]]; then
        cache_flag=--enable-prefix-caching
    fi
    vllm serve "$model" --host 127.0.0.1 --max-model-len 8192 \
        --gpu-memory-utilization 0.5 --enforce-eager \
        --enable-prompt-tokens-details "$cache_flag" >"vllm-$mode.log" 2>&1 &
    server_pid=$!
    until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -50 "vllm-$mode.log"
            exit 1
        fi
        sleep 2
    done
    python benchmark.py --model "$model" --cache "$mode" >"results-$mode.jsonl"
    cleanup
    server_pid=
done
