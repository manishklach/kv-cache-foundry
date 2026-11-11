#define _POSIX_C_SOURCE 200809L
#include <cuda_runtime_api.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "llama.h"

enum { ROUNDS = 16, MAX_TOKENS = 4096, MIB = 1024 * 1024, CHUNK_MIB = 512 };

static void require(bool ok, const char *message)
{
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        exit(EXIT_FAILURE);
    }
}

static void cuda_check(cudaError_t result)
{
    require(result == cudaSuccess, cudaGetErrorString(result));
}

static double now_ms(void)
{
    struct timespec t;
    require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime failed");
    return 1000.0 * t.tv_sec + 1e-6 * t.tv_nsec;
}

static void decode(struct llama_context *ctx, llama_token *tokens, int count)
{
    require(llama_decode(ctx, llama_batch_get_one(tokens, count)) == 0,
            "llama_decode failed");
    llama_synchronize(ctx);
}

static int greedy_token(struct llama_context *ctx, int vocabulary_size)
{
    const float *logits = llama_get_logits_ith(ctx, -1);
    require(logits != NULL, "missing logits");
    int best = 0;
    for (int i = 1; i < vocabulary_size; ++i) {
        if (logits[i] > logits[best])
            best = i;
    }
    return best;
}

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double values[ROUNDS], int start)
{
    double sorted[ROUNDS / 2];
    for (int i = 0; i < ROUNDS / 2; ++i)
        sorted[i] = values[start + i];
    qsort(sorted, ROUNDS / 2, sizeof(*sorted), compare_double);
    return (sorted[ROUNDS / 4 - 1] + sorted[ROUNDS / 4]) / 2.0;
}

static double measure(struct llama_context *ctx, llama_token *tokens, int count,
                      const uint8_t *state, size_t bytes, bool reload,
                      size_t backlog, void *device, const void *host)
{
    llama_memory_clear(llama_get_memory(ctx), false);
    const double start = now_ms();
    if (reload) {
        while (backlog) {
            const size_t chunk = backlog < CHUNK_MIB * (size_t)MIB
                ? backlog : CHUNK_MIB * (size_t)MIB;
            cuda_check(cudaMemcpy(device, host, chunk, cudaMemcpyHostToDevice));
            backlog -= chunk;
        }
        require(llama_state_seq_set_data(ctx, state, bytes, 0) == bytes,
                "KV restore failed");
        llama_synchronize(ctx);
    } else {
        decode(ctx, tokens, count);
    }
    return now_ms() - start;
}

int main(int argc, char **argv)
{
    require(argc == 2, "usage: reload_vs_recompute MODEL.gguf");
    const int lengths[] = {128, 512, 2048, MAX_TOKENS};
    const int backlogs[] = {0, 64, 256, 512, 1024, 2048, 4096};
    llama_backend_init();
    require(llama_supports_gpu_offload(), "llama.cpp lacks GPU offload");
    cuda_check(cudaSetDevice(0));
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    require(model != NULL, "could not load GGUF model");
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = MAX_TOKENS + 256;
    cp.n_batch = MAX_TOKENS;
    cp.n_ubatch = 512;
    cp.offload_kqv = true;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    require(ctx != NULL, "could not create llama context");
    const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    require(vocabulary_size > 8000, "model vocabulary too small");
    void *host = NULL, *device = NULL;
    cuda_check(cudaMallocHost(&host, CHUNK_MIB * (size_t)MIB));
    cuda_check(cudaMalloc(&device, CHUNK_MIB * (size_t)MIB));
    for (size_t i = 0; i < CHUNK_MIB * (size_t)MIB; ++i)
        ((uint8_t *)host)[i] = (uint8_t)i;

    llama_token tokens[MAX_TOKENS];
    uint32_t seed = 123;
    for (int i = 0; i < MAX_TOKENS; ++i) {
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        tokens[i] = 1000 + (llama_token)(seed % 7000);
    }

    puts("tokens,backlog_mib,state_bytes,round,reload_ms,recompute_ms,selected,output_match");
    for (size_t l = 0; l < sizeof(lengths) / sizeof(lengths[0]); ++l) {
        const int length = lengths[l];
        llama_memory_clear(llama_get_memory(ctx), false);
        decode(ctx, tokens, length);
        const size_t bytes = llama_state_seq_get_size(ctx, 0);
        uint8_t *state = NULL;
        require(bytes > 0, "empty KV state");
        cuda_check(cudaMallocHost((void **)&state, bytes));
        require(llama_state_seq_get_data(ctx, state, bytes, 0) == bytes, "KV save failed");
        llama_token suffix = 37;
        decode(ctx, &suffix, 1);
        const int expected = greedy_token(ctx, vocabulary_size);
        measure(ctx, tokens, length, state, bytes, true, 0, device, host);
        decode(ctx, &suffix, 1);
        const bool match = greedy_token(ctx, vocabulary_size) == expected;
        measure(ctx, tokens, length, state, bytes, false, 0, device, host);

        for (size_t b = 0; b < sizeof(backlogs) / sizeof(backlogs[0]); ++b) {
            const size_t backlog = (size_t)backlogs[b] * MIB;
            double reload[ROUNDS], recompute[ROUNDS];
            for (int round = 0; round < ROUNDS; ++round) {
                for (int order = 0; order < 2; ++order) {
                    const bool restoring = (round + order) % 2 == 0;
                    (restoring ? reload : recompute)[round] =
                        measure(ctx, tokens, length, state, bytes, restoring,
                                backlog, device, host);
                }
            }
            const char *selected = median(reload, 0) <= median(recompute, 0)
                ? "reload" : "recompute";
            for (int round = 0; round < ROUNDS; ++round)
                printf("%d,%d,%zu,%d,%.6f,%.6f,%s,%d\n", length, backlogs[b],
                       bytes, round, reload[round], recompute[round], selected, match);
            fflush(stdout);
        }
        cuda_check(cudaFreeHost(state));
    }
    cuda_check(cudaFree(device));
    cuda_check(cudaFreeHost(host));
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
