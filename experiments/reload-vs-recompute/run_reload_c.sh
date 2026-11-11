#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build="$PWD/reload-build"
mkdir -p "$build/include" "$build/bin"
tag=b12250
release="https://github.com/ggml-org/llama.cpp/releases/download/$tag"
source_url="https://raw.githubusercontent.com/ggml-org/llama.cpp/$tag"
model="$build/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
model_url="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/$(basename "$model")"

if [[ ! -s "$model" ]]; then
    curl -fLsS --retry 2 "$model_url" -o "$model.part" &
    model_pid=$!
fi
for archive in "llama-$tag-bin-ubuntu-cuda-12.8-x64" "cudart-llama-$tag-bin-ubuntu-cuda-12.8-x64"; do
    curl -fLsS --retry 2 "$release/$archive.tar.gz" | tar -xz -C "$build/bin"
done
curl -fLsS "$source_url/include/llama.h" -o "$build/include/llama.h"
for header in ggml.h ggml-cpu.h ggml-backend.h ggml-alloc.h ggml-opt.h gguf.h; do
    curl -fLsS "$source_url/ggml/include/$header" -o "$build/include/$header"
done
llama_lib=$(find "$build/bin" -name 'libllama.so*' -print -quit)
cuda_lib=$(find "$build/bin" -name 'libcudart.so.12*' -print -quit)
test -n "$llama_lib" && test -n "$cuda_lib"
export LD_LIBRARY_PATH="$(dirname "$llama_lib"):$(dirname "$cuda_lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cc -std=c11 -O2 -Wall -Wextra -Werror -I "$build/include" -I /usr/local/cuda/include \
    reload_vs_recompute.c "$llama_lib" "$cuda_lib" -o "$build/reload_vs_recompute"
if [[ -n "${model_pid:-}" ]]; then
    wait "$model_pid"
    mv "$model.part" "$model"
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >&2
(cd "$(dirname "$llama_lib")" && "$build/reload_vs_recompute" "$model") >reload-results.csv
