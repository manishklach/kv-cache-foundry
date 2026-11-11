#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
build_root="${BUILD_ROOT:-$here/build}"
backend="${BACKEND:-cuda}"
jobs="${JOBS:-4}"
revision=ae4893a092e060cfbd4979600bf3c1e48a710874
src="$build_root/llama.cpp"
out="$build_root/$backend"

case "$backend" in
    cuda) cuda=ON ;;
    cpu) cuda=OFF ;;
    *) echo "BACKEND must be cuda or cpu" >&2; exit 2 ;;
esac

mkdir -p "$build_root"
if [[ ! -d "$src/.git" ]]; then
    git clone --depth 1 --branch b12250 https://github.com/ggml-org/llama.cpp.git "$src"
fi
test "$(git -C "$src" rev-parse HEAD)" = "$revision"
if ! git -C "$src" apply --reverse --check "$here/llama-paced-restore.patch" 2>/dev/null; then
    git -C "$src" apply "$here/llama-paced-restore.patch"
fi

cmake_args=()
if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
fi
cmake -S "$src" -B "$out" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA="$cuda" -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    "${cmake_args[@]}"
cmake --build "$out" --target llama --parallel "$jobs"
"${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -pthread \
    -I "$src/include" -I "$src/ggml/include" "$here/decode_aware_reload.c" \
    -L "$out/bin" -Wl,-rpath,"$out/bin" -lllama -lggml -lggml-base -lm \
    -o "$out/decode_aware_reload"
