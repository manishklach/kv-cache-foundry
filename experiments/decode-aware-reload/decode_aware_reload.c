#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml-backend.h"
#include "llama.h"

enum { MIB = 1024 * 1024, DECODE_STEPS = 256, REQUESTS = 12, TRIALS = 3, OUTPUT_STEPS = 8 };

struct governor {
    pthread_mutex_t mutex;
    pthread_cond_t tick;
    size_t bytes;
    double target_ms;
    double gap_ms;
    unsigned steps;
    unsigned batches;
    unsigned waits;
    bool done;
};

struct foreground {
    struct llama_context *ctx;
    struct governor *governor;
    double gaps[DECODE_STEPS];
    llama_token token;
};

static void check(bool ok, const char *message)
{
    if (!ok) {
        fprintf(stderr, "%s\n", message);
        exit(EXIT_FAILURE);
    }
}

static struct timespec now(void)
{
    struct timespec t;
    check(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock_gettime failed");
    return t;
}

static double ms(void)
{
    const struct timespec t = now();
    return 1000.0 * t.tv_sec + t.tv_nsec / 1e6;
}

static void decode(struct llama_context *ctx, llama_token *tokens, int count)
{
    check(llama_decode(ctx, llama_batch_get_one(tokens, count)) == 0, "llama_decode failed");
    llama_synchronize(ctx);
}

static llama_token greedy(struct llama_context *ctx, int vocab)
{
    const float *logits = llama_get_logits_ith(ctx, -1);
    check(logits != NULL, "missing logits");
    llama_token best = 0;
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > logits[best])
            best = i;
    }
    return best;
}

static void prefill(struct llama_context *ctx, llama_token *tokens, int length)
{
    for (int start = 0; start < length; start += 128) {
        const int left = length - start;
        decode(ctx, tokens + start, left < 128 ? left : 128);
    }
}

static void *run_foreground(void *arg)
{
    struct foreground *fg = arg;
    struct governor *g = fg->governor;
    double last = ms();
    for (unsigned i = 0; i < DECODE_STEPS; ++i) {
        decode(fg->ctx, &fg->token, 1);
        const double end = ms();
        fg->gaps[i] = end - last;
        last = end;
        pthread_mutex_lock(&g->mutex);
        g->gap_ms = i ? 0.75 * g->gap_ms + 0.25 * fg->gaps[i] : fg->gaps[i];
        if (g->gap_ms > g->target_ms)
            g->bytes = g->bytes > 2 * MIB ? g->bytes / 2 : MIB;
        if (g->gap_ms < 0.8 * g->target_ms)
            g->bytes = g->bytes < 15 * MIB ? g->bytes + MIB : 16 * MIB;
        ++g->steps;
        g->done = g->steps == DECODE_STEPS;
        pthread_cond_broadcast(&g->tick);
        pthread_mutex_unlock(&g->mutex);
    }
    return NULL;
}

static size_t fixed_budget(void *arg)
{
    struct governor *g = arg;
    ++g->batches;
    return 4 * MIB;
}

static size_t validation_budget(void *arg)
{
    unsigned *batches = arg;
    ++*batches;
    return 1024;
}

static size_t adaptive_budget(void *arg)
{
    struct governor *g = arg;
    ++g->batches;
    pthread_mutex_lock(&g->mutex);
    if (!g->done && g->gap_ms > g->target_ms) {
        ++g->waits;
        const unsigned step = g->steps;
        struct timespec until = now();
        until.tv_nsec += 2000000;
        if (until.tv_nsec >= 1000000000) {
            ++until.tv_sec;
            until.tv_nsec -= 1000000000;
        }
        while (g->steps == step) {
            const int rc = pthread_cond_timedwait(&g->tick, &g->mutex, &until);
            if (rc == ETIMEDOUT)
                break;
            check(rc == 0, "condition wait failed");
        }
    }
    const size_t budget = g->done ? SIZE_MAX : g->bytes;
    pthread_mutex_unlock(&g->mutex);
    return budget;
}

static bool prefer_recompute(struct governor *g, size_t bytes, double restore_ms, double prefill_ms)
{
    pthread_mutex_lock(&g->mutex);
    const bool pressured = !g->done && g->gap_ms > g->target_ms;
    // Under pressure each additional batch may wait for a token, up to 2 ms.
    const double batch_wait = pressured ? fmin(2.0, g->gap_ms) : 0.0;
    const size_t waits = (bytes - 1) / g->bytes;
    pthread_mutex_unlock(&g->mutex);
    return prefill_ms < 0.9 * (restore_ms + waits * batch_wait);
}

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static void compare_output(struct llama_context *ctx, int vocab, llama_token suffix,
                           llama_token expected[OUTPUT_STEPS], bool reference)
{
    for (int i = 0; i < OUTPUT_STEPS; ++i) {
        decode(ctx, &suffix, 1);
        suffix = greedy(ctx, vocab);
        if (reference)
            expected[i] = suffix;
        else
            check(suffix == expected[i], "greedy output mismatch");
    }
}

int main(int argc, char **argv)
{
    check(argc == 5, "usage: decode_aware_reload MODEL.gguf PREFIX_TOKENS TARGET_ITL_MS GPU_LAYERS");
    char *end;
    const long prefix = strtol(argv[2], &end, 10);
    check(*end == 0 && prefix >= 32 && prefix <= 4096, "invalid prefix length");
    const int length = (int)prefix;
    const double target = strtod(argv[3], &end);
    check(*end == 0 && isfinite(target) && target > 0, "invalid latency target");
    const long layers = strtol(argv[4], &end, 10);
    check(end != argv[4] && *end == 0 && layers >= 0 && layers <= INT32_MAX,
          "invalid GPU layer count");

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = layers;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    check(model != NULL, "model load failed");
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = length + DECODE_STEPS + OUTPUT_STEPS + 64;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = 2;
    struct llama_context *bg = llama_init_from_model(model, cp);
    struct llama_context *fg = llama_init_from_model(model, cp);
    check(bg && fg, "context initialization failed");

    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    check(vocab > 512, "vocabulary too small");
    llama_token tokens[4096];
    uint32_t seed = 123;
    for (int i = 0; i < length; ++i) {
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        tokens[i] = 256 + (llama_token)(seed % (uint32_t)(vocab - 256));
    }
    const llama_token suffix = 37;
    prefill(bg, tokens, length);
    const size_t bytes = llama_state_seq_get_size(bg, 0);
    check(bytes != 0, "empty KV snapshot");

    ggml_backend_buffer_t pinned = NULL;
    uint8_t *state = NULL;
    if (layers) {
        ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        check(gpu != NULL, "GPU backend missing");
        ggml_backend_buffer_type_t host_type = ggml_backend_dev_host_buffer_type(gpu);
        check(host_type != NULL, "GPU pinned host allocation unavailable");
        pinned = ggml_backend_buft_alloc_buffer(host_type, bytes);
        check(pinned != NULL, "pinned snapshot allocation failed");
        state = ggml_backend_buffer_get_base(pinned);
    } else {
        state = malloc(bytes);
    }
    check(state && llama_state_seq_get_data(bg, state, bytes, 0) == bytes, "snapshot failed");

    llama_token expected[OUTPUT_STEPS];
    compare_output(bg, vocab, suffix, expected, true);
    unsigned validation_batches = 0;
    llama_memory_clear(llama_get_memory(bg), false);
    check(llama_state_seq_set_data_paced(bg, state, bytes, 0,
                                       validation_budget, &validation_batches) == bytes,
          "split restore validation failed");
    check(validation_batches > 1, "split restore did not exercise multiple batches");
    compare_output(bg, vocab, suffix, expected, false);

    double restore_ms = 0, prefill_ms = 0;
    for (int i = 0; i < 5; ++i) {
        llama_memory_clear(llama_get_memory(bg), false);
        double start = ms();
        check(llama_state_seq_set_data(bg, state, bytes, 0) == bytes, "calibration restore failed");
        restore_ms += (ms() - start) / 5;
        compare_output(bg, vocab, suffix, expected, false);
        llama_memory_clear(llama_get_memory(bg), false);
        start = ms();
        prefill(bg, tokens, length);
        prefill_ms += (ms() - start) / 5;
    }
    fprintf(stderr, "snapshot=%zu bytes isolated_restore=%.3f ms isolated_prefill=%.3f ms\n",
            bytes, restore_ms, prefill_ms);

    const char *modes[] = {"stock", "fixed", "adaptive"};
    puts("trial,mode,request,path,ttft_ms,foreground_steps_at_start,foreground_steps_at_first_token,restore_batches,waits,foreground_p50_ms,foreground_p95_ms,wall_ms");
    for (int trial = 0; trial < TRIALS; ++trial) {
        for (int order = 0; order < 3; ++order) {
            const int mode = (trial + order) % 3;
            struct governor g = {.bytes = 4 * MIB, .target_ms = target};
            pthread_condattr_t attr;
            check(pthread_mutex_init(&g.mutex, NULL) == 0, "mutex init failed");
            check(pthread_condattr_init(&attr) == 0 &&
                  pthread_condattr_setclock(&attr, CLOCK_MONOTONIC) == 0 &&
                  pthread_cond_init(&g.tick, &attr) == 0, "condition init failed");
            pthread_condattr_destroy(&attr);
            struct foreground foreground = {.ctx = fg, .governor = &g, .token = suffix};
            llama_memory_clear(llama_get_memory(fg), false);
            prefill(fg, tokens, 32);
            for (int warmup = 0; warmup < 4; ++warmup)
                decode(fg, &foreground.token, 1);
            pthread_t thread;
            const double begin = ms();
            check(pthread_create(&thread, NULL, run_foreground, &foreground) == 0, "thread creation failed");
            for (int job = 0; job < REQUESTS; ++job) {
                llama_memory_clear(llama_get_memory(bg), false);
                pthread_mutex_lock(&g.mutex);
                const unsigned start_steps = g.steps;
                pthread_mutex_unlock(&g.mutex);
                g.batches = g.waits = 0;
                const double start = ms();
                const bool rebuild = mode == 2 && prefer_recompute(&g, bytes, restore_ms, prefill_ms);
                if (rebuild) {
                    prefill(bg, tokens, length);
                } else {
                    llama_kv_reload_budget_fn budget = mode == 1 ? fixed_budget : adaptive_budget;
                    const size_t restored = mode == 0
                        ? llama_state_seq_set_data(bg, state, bytes, 0)
                        : llama_state_seq_set_data_paced(bg, state, bytes, 0, budget, &g);
                    check(restored == bytes, "KV restore failed");
                }
                llama_token next = suffix;
                decode(bg, &next, 1);
                next = greedy(bg, vocab);
                const double ttft = ms() - start;
                pthread_mutex_lock(&g.mutex);
                const unsigned first_token_steps = g.steps;
                pthread_mutex_unlock(&g.mutex);
                check(next == expected[0], "first token mismatch");
                for (int i = 1; i < OUTPUT_STEPS; ++i) {
                    decode(bg, &next, 1);
                    next = greedy(bg, vocab);
                    check(next == expected[i], "greedy output mismatch");
                }
                printf("%d,%s,%d,%s,%.6f,%u,%u,%u,%u,,,\n", trial, modes[mode], job,
                       rebuild ? "recompute" : "reload", ttft, start_steps, first_token_steps,
                       g.batches, g.waits);
            }
            check(pthread_join(thread, NULL) == 0, "join failed");
            const double wall = ms() - begin;
            qsort(foreground.gaps, DECODE_STEPS, sizeof(double), compare_double);
            printf("%d,%s,,,,,,,,%.6f,%.6f,%.6f\n", trial, modes[mode],
                   foreground.gaps[DECODE_STEPS / 2 - 1],
                   foreground.gaps[(DECODE_STEPS * 95 + 99) / 100 - 1], wall);
            fflush(stdout);
            pthread_cond_destroy(&g.tick);
            pthread_mutex_destroy(&g.mutex);
        }
    }
    if (pinned)
        ggml_backend_buffer_free(pinned);
    else
        free(state);
    llama_free(fg);
    llama_free(bg);
    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
