#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m venv --system-site-packages /tmp/kv-admission-venv
source /tmp/kv-admission-venv/bin/activate
python -m pip install vllm==0.32.0
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -shared -fPIC block_rank.c -o block_rank.so
python engine_experiment.py patch
python live_eviction.py
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
model=Qwen/Qwen3-4B-Instruct-2507
server_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
for trial in 1-lru 1-adaptive 1-cost 2-cost 2-adaptive 2-lru; do
    mode=${trial#*-}
    KV_ADMISSION="$mode" vllm serve "$model" --host 127.0.0.1 \
        --max-model-len 4608 --block-size 32 --num-gpu-blocks-override 1025 \
        --gpu-memory-utilization 0.5 --max-num-seqs 4 --enforce-eager \
        --enable-prefix-caching --enable-prompt-tokens-details \
        >"vllm-$trial.log" 2>&1 &
    server_pid=$!
    until curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -60 "vllm-$trial.log"
            exit 1
        fi
        sleep 2
    done
    python engine_experiment.py "$mode" --concurrency 4 --requests-per-phase 128 >"gpu-$trial.jsonl"
    cleanup
    server_pid=
done
tar -czf cost-eviction-live-results.tar.gz gpu-*.jsonl vllm-*.log
