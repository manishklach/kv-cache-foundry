/*
 * C-style CUDA microbenchmark for deterministic top-2 / margin reduction.
 *
 * The source uses CUDA's C kernel extensions and no C++ abstractions. It
 * compares one CTA per row with a hierarchical reduction that splits a large
 * vocabulary row across CTAs. Equal logits use token id as a stable tie-break.
 */
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    THREADS = 256,
    ITERATIONS = 200,
    WARMUP = 25,
    MAX_ROWS = 64,
    VOCAB_SIZE = 151936,
    WARPS = THREADS / 32
};

typedef struct {
    float value;
    int index;
} Candidate;

typedef struct {
    int top1_index;
    int top2_index;
    float top1_value;
    float top2_value;
    float margin;
    int flagged;
} Result;

static void cuda_check(cudaError_t error, const char *where)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        exit(2);
    }
}

__device__ static int candidate_precedes(Candidate a, Candidate b)
{
    if (a.index < 0)
        return 0;
    if (b.index < 0)
        return 1;
    if (a.value > b.value)
        return 1;
    if (a.value < b.value)
        return 0;
    return a.index < b.index;
}

__device__ static void insert_candidate(Candidate candidate,
                                         Candidate *first,
                                         Candidate *second)
{
    if (candidate.index < 0)
        return;
    if (candidate.index == first->index || candidate.index == second->index)
        return;
    if (candidate_precedes(candidate, *first)) {
        *second = *first;
        *first = candidate;
    } else if (candidate_precedes(candidate, *second)) {
        *second = candidate;
    }
}

__device__ static void merge_pair(Candidate a1, Candidate a2,
                                  Candidate b1, Candidate b2,
                                  Candidate *first, Candidate *second)
{
    Candidate empty = {-FLT_MAX, -1};
    *first = empty;
    *second = empty;
    insert_candidate(a1, first, second);
    insert_candidate(a2, first, second);
    insert_candidate(b1, first, second);
    insert_candidate(b2, first, second);
}

__device__ static void store_result(Result *out, Candidate first,
                                    Candidate second, float threshold)
{
    out->top1_index = first.index;
    out->top2_index = second.index;
    out->top1_value = first.value;
    out->top2_value = second.value;
    out->margin = first.value - second.value;
    out->flagged = out->margin <= threshold;
}

__device__ static void warp_reduce_top2(Candidate *first, Candidate *second)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        Candidate other_first;
        Candidate other_second;
        Candidate merged_first;
        Candidate merged_second;
        other_first.value = __shfl_down_sync(0xffffffffu, first->value, offset);
        other_first.index = __shfl_down_sync(0xffffffffu, first->index, offset);
        other_second.value = __shfl_down_sync(0xffffffffu, second->value, offset);
        other_second.index = __shfl_down_sync(0xffffffffu, second->index, offset);
        merge_pair(*first, *second, other_first, other_second,
                   &merged_first, &merged_second);
        *first = merged_first;
        *second = merged_second;
    }
}

__device__ static void warp_reduce_top1(Candidate *best)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        Candidate other;
        other.value = __shfl_down_sync(0xffffffffu, best->value, offset);
        other.index = __shfl_down_sync(0xffffffffu, best->index, offset);
        if (candidate_precedes(other, *best))
            *best = other;
    }
}

__global__ static void top2_single_cta(const float *logits, Result *out,
                                       int vocab, float threshold)
{
    __shared__ Candidate firsts[THREADS];
    __shared__ Candidate seconds[THREADS];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    Candidate first = {-FLT_MAX, -1};
    Candidate second = {-FLT_MAX, -1};
    int i;

    for (i = tid; i < vocab; i += THREADS) {
        Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
        insert_candidate(current, &first, &second);
    }
    firsts[tid] = first;
    seconds[tid] = second;
    __syncthreads();

    for (int width = THREADS / 2; width > 0; width >>= 1) {
        if (tid < width) {
            Candidate merged_first;
            Candidate merged_second;
            merge_pair(firsts[tid], seconds[tid], firsts[tid + width],
                       seconds[tid + width], &merged_first, &merged_second);
            firsts[tid] = merged_first;
            seconds[tid] = merged_second;
        }
        __syncthreads();
    }
    if (tid == 0)
        store_result(&out[row], firsts[0], seconds[0], threshold);
}

__global__ static void top2_single_warp(const float *logits, Result *out,
                                        int vocab, float threshold)
{
    __shared__ Candidate warp_firsts[WARPS];
    __shared__ Candidate warp_seconds[WARPS];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    Candidate first = {-FLT_MAX, -1};
    Candidate second = {-FLT_MAX, -1};

    for (int i = tid; i < vocab; i += THREADS) {
        Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
        insert_candidate(current, &first, &second);
    }
    warp_reduce_top2(&first, &second);
    if (lane == 0) {
        warp_firsts[warp] = first;
        warp_seconds[warp] = second;
    }
    __syncthreads();

    if (warp == 0) {
        Candidate empty = {-FLT_MAX, -1};
        if (lane < WARPS) {
            first = warp_firsts[lane];
            second = warp_seconds[lane];
        } else {
            first = empty;
            second = empty;
        }
        warp_reduce_top2(&first, &second);
        if (lane == 0)
            store_result(&out[row], first, second, threshold);
    }
}

__global__ static void greedy_argmax_proxy(const float *logits, int *out,
                                           int vocab)
{
    __shared__ Candidate warp_best[WARPS];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    Candidate best = {-FLT_MAX, -1};

    for (int i = tid; i < vocab; i += THREADS) {
        Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
        if (candidate_precedes(current, best))
            best = current;
    }
    warp_reduce_top1(&best);
    if (lane == 0)
        warp_best[warp] = best;
    __syncthreads();

    if (warp == 0) {
        Candidate empty = {-FLT_MAX, -1};
        if (lane < WARPS)
            best = warp_best[lane];
        else
            best = empty;
        warp_reduce_top1(&best);
        if (lane == 0)
            out[row] = best.index;
    }
}

__global__ static void top2_partials(const float *logits,
                                     Candidate *partial_first,
                                     Candidate *partial_second,
                                     int vocab, int tiles, int tile_size)
{
    __shared__ Candidate firsts[THREADS];
    __shared__ Candidate seconds[THREADS];
    int row = (int)blockIdx.x / tiles;
    int tile = (int)blockIdx.x - row * tiles;
    int tid = (int)threadIdx.x;
    int begin = tile * tile_size;
    int end = begin + tile_size;
    Candidate first = {-FLT_MAX, -1};
    Candidate second = {-FLT_MAX, -1};
    int i;

    if (end > vocab)
        end = vocab;
    for (i = begin + tid; i < end; i += THREADS) {
        Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
        insert_candidate(current, &first, &second);
    }
    firsts[tid] = first;
    seconds[tid] = second;
    __syncthreads();

    for (int width = THREADS / 2; width > 0; width >>= 1) {
        if (tid < width) {
            Candidate merged_first;
            Candidate merged_second;
            merge_pair(firsts[tid], seconds[tid], firsts[tid + width],
                       seconds[tid + width], &merged_first, &merged_second);
            firsts[tid] = merged_first;
            seconds[tid] = merged_second;
        }
        __syncthreads();
    }
    if (tid == 0) {
        size_t offset = (size_t)row * (size_t)tiles + (size_t)tile;
        partial_first[offset] = firsts[0];
        partial_second[offset] = seconds[0];
    }
}

__global__ static void top2_finalize(const Candidate *partial_first,
                                     const Candidate *partial_second,
                                     Result *out, int tiles, float threshold)
{
    __shared__ Candidate firsts[THREADS];
    __shared__ Candidate seconds[THREADS];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    Candidate first = {-FLT_MAX, -1};
    Candidate second = {-FLT_MAX, -1};

    for (int tile = tid; tile < tiles; tile += THREADS) {
        size_t offset = (size_t)row * (size_t)tiles + (size_t)tile;
        insert_candidate(partial_first[offset], &first, &second);
        insert_candidate(partial_second[offset], &first, &second);
    }
    firsts[tid] = first;
    seconds[tid] = second;
    __syncthreads();

    for (int width = THREADS / 2; width > 0; width >>= 1) {
        if (tid < width) {
            Candidate merged_first;
            Candidate merged_second;
            merge_pair(firsts[tid], seconds[tid], firsts[tid + width],
                       seconds[tid + width], &merged_first, &merged_second);
            firsts[tid] = merged_first;
            seconds[tid] = merged_second;
        }
        __syncthreads();
    }
    if (tid == 0)
        store_result(&out[row], firsts[0], seconds[0], threshold);
}

__global__ static void argmax_partials(const float *logits,
                                       Candidate *partial_best,
                                       int vocab, int tiles, int tile_size)
{
    __shared__ Candidate bests[THREADS];
    int row = (int)blockIdx.x / tiles;
    int tile = (int)blockIdx.x - row * tiles;
    int tid = (int)threadIdx.x;
    int begin = tile * tile_size;
    int end = begin + tile_size;
    Candidate best = {-FLT_MAX, -1};

    if (end > vocab)
        end = vocab;
    for (int i = begin + tid; i < end; i += THREADS) {
        Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
        if (candidate_precedes(current, best))
            best = current;
    }
    bests[tid] = best;
    __syncthreads();
    for (int width = THREADS / 2; width > 0; width >>= 1) {
        if (tid < width && candidate_precedes(bests[tid + width], bests[tid]))
            bests[tid] = bests[tid + width];
        __syncthreads();
    }
    if (tid == 0)
        partial_best[(size_t)row * (size_t)tiles + (size_t)tile] = bests[0];
}

__global__ static void argmax_finalize(const Candidate *partial_best,
                                       int *out, int tiles)
{
    __shared__ Candidate warp_best[WARPS];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    Candidate best = {-FLT_MAX, -1};

    for (int tile = tid; tile < tiles; tile += THREADS) {
        Candidate current = partial_best[(size_t)row * (size_t)tiles + (size_t)tile];
        if (candidate_precedes(current, best))
            best = current;
    }
    warp_reduce_top1(&best);
    if (lane == 0)
        warp_best[warp] = best;
    __syncthreads();

    if (warp == 0) {
        Candidate empty = {-FLT_MAX, -1};
        if (lane < WARPS)
            best = warp_best[lane];
        else
            best = empty;
        warp_reduce_top1(&best);
        if (lane == 0)
            out[row] = best.index;
    }
}

static int cpu_precedes(Candidate a, Candidate b)
{
    if (a.index < 0)
        return 0;
    if (b.index < 0)
        return 1;
    if (a.value != b.value)
        return a.value > b.value;
    return a.index < b.index;
}

static void cpu_insert(Candidate candidate, Candidate *first, Candidate *second)
{
    if (cpu_precedes(candidate, *first)) {
        *second = *first;
        *first = candidate;
    } else if (candidate.index != first->index && cpu_precedes(candidate, *second)) {
        *second = candidate;
    }
}

static void cpu_reference(const float *logits, Result *out, int rows,
                          int vocab, float threshold)
{
    for (int row = 0; row < rows; ++row) {
        Candidate first = {-FLT_MAX, -1};
        Candidate second = {-FLT_MAX, -1};
        for (int i = 0; i < vocab; ++i) {
            Candidate current = {logits[(size_t)row * (size_t)vocab + (size_t)i], i};
            cpu_insert(current, &first, &second);
        }
        out[row].top1_index = first.index;
        out[row].top2_index = second.index;
        out[row].top1_value = first.value;
        out[row].top2_value = second.value;
        out[row].margin = first.value - second.value;
        out[row].flagged = out[row].margin <= threshold;
    }
}

static void fill_logits(float *logits, int rows, int vocab)
{
    for (size_t i = 0; i < (size_t)rows * (size_t)vocab; ++i)
        logits[i] = (float)rand() / (float)RAND_MAX;
    if (rows > 0 && vocab > 287) {
        logits[220] = 2.0f;
        logits[287] = 2.0f;
    }
}

static double time_single(const float *d_logits, Result *d_out, int rows,
                          int vocab, float threshold)
{
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i)
        top2_single_cta<<<rows, THREADS>>>(d_logits, d_out, vocab, threshold);
    cuda_check(cudaGetLastError(), "single-CTA warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i)
        top2_single_cta<<<rows, THREADS>>>(d_logits, d_out, vocab, threshold);
    cuda_check(cudaGetLastError(), "single-CTA timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)elapsed / (double)ITERATIONS;
}

static double time_single_warp(const float *d_logits, Result *d_out, int rows,
                               int vocab, float threshold)
{
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i)
        top2_single_warp<<<rows, THREADS>>>(d_logits, d_out, vocab, threshold);
    cuda_check(cudaGetLastError(), "warp top-2 warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i)
        top2_single_warp<<<rows, THREADS>>>(d_logits, d_out, vocab, threshold);
    cuda_check(cudaGetLastError(), "warp top-2 timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)elapsed / (double)ITERATIONS;
}

static double time_argmax(const float *d_logits, int *d_out, int rows, int vocab)
{
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i)
        greedy_argmax_proxy<<<rows, THREADS>>>(d_logits, d_out, vocab);
    cuda_check(cudaGetLastError(), "argmax warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i)
        greedy_argmax_proxy<<<rows, THREADS>>>(d_logits, d_out, vocab);
    cuda_check(cudaGetLastError(), "argmax timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)elapsed / (double)ITERATIONS;
}

static double time_argmax_hierarchical(const float *d_logits,
                                       Candidate *d_partial_best,
                                       int *d_out, int rows, int vocab,
                                       int tiles, int tile_size)
{
    cudaEvent_t start, stop;
    int blocks = rows * tiles;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i) {
        argmax_partials<<<blocks, THREADS>>>(d_logits, d_partial_best,
                                             vocab, tiles, tile_size);
        argmax_finalize<<<rows, THREADS>>>(d_partial_best, d_out, tiles);
    }
    cuda_check(cudaGetLastError(), "hierarchical argmax warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i) {
        argmax_partials<<<blocks, THREADS>>>(d_logits, d_partial_best,
                                             vocab, tiles, tile_size);
        argmax_finalize<<<rows, THREADS>>>(d_partial_best, d_out, tiles);
    }
    cuda_check(cudaGetLastError(), "hierarchical argmax timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)elapsed / (double)ITERATIONS;
}

static double time_hierarchical(const float *d_logits, Candidate *d_first,
                                Candidate *d_second, Result *d_out,
                                int rows, int vocab, int tiles, int tile_size,
                                float threshold)
{
    cudaEvent_t start, stop;
    int blocks = rows * tiles;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i) {
        top2_partials<<<blocks, THREADS>>>(d_logits, d_first, d_second, vocab,
                                           tiles, tile_size);
        top2_finalize<<<rows, THREADS>>>(d_first, d_second, d_out, tiles, threshold);
    }
    cuda_check(cudaGetLastError(), "hierarchical warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i) {
        top2_partials<<<blocks, THREADS>>>(d_logits, d_first, d_second, vocab,
                                           tiles, tile_size);
        top2_finalize<<<rows, THREADS>>>(d_first, d_second, d_out, tiles, threshold);
    }
    cuda_check(cudaGetLastError(), "hierarchical timed launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)elapsed / (double)ITERATIONS;
}

static int results_match(const Result *actual, const Result *expected, int rows)
{
    for (int i = 0; i < rows; ++i) {
        if (actual[i].top1_index != expected[i].top1_index ||
            actual[i].top2_index != expected[i].top2_index ||
            actual[i].margin != expected[i].margin ||
            actual[i].flagged != expected[i].flagged)
            return 0;
    }
    return 1;
}

static void print_top2_result(int repeat, int rows, const char *variant,
                              int tile_size, int tiles, double elapsed_ms,
                              const Result *result, int correct)
{
    printf("%d,%d,%s,%d,%d,%.6f,%d,%d,%d,%.8f,%d,%d\n",
           repeat, rows, variant, tile_size, tiles, elapsed_ms,
           result[0].top1_index, result[0].top2_index,
           result[0].top1_index == 220 && result[0].top2_index == 287,
           result[0].margin, result[0].flagged, correct);
}

static void run_case(int rows, float threshold, int repeat)
{
    static const int tile_sizes[] = {512, 1024, 2048, 4096, 8192, 16384};
    size_t logits_count = (size_t)rows * (size_t)VOCAB_SIZE;
    size_t logits_bytes = logits_count * sizeof(float);
    int max_tiles = (VOCAB_SIZE + tile_sizes[0] - 1) / tile_sizes[0];
    size_t partial_count = (size_t)rows * (size_t)max_tiles;
    float *h_logits = (float *)malloc(logits_bytes);
    Result *h_expected = (Result *)malloc((size_t)rows * sizeof(*h_expected));
    Result *h_actual = (Result *)malloc((size_t)rows * sizeof(*h_actual));
    int *h_argmax = (int *)malloc((size_t)rows * sizeof(*h_argmax));
    float *d_logits = NULL;
    Result *d_out = NULL;
    int *d_argmax = NULL;
    Candidate *d_first = NULL;
    Candidate *d_second = NULL;
    Candidate *d_argmax_partials = NULL;

    if (h_logits == NULL || h_expected == NULL || h_actual == NULL || h_argmax == NULL) {
        fprintf(stderr, "host allocation failed for rows=%d\n", rows);
        exit(2);
    }
    fill_logits(h_logits, rows, VOCAB_SIZE);
    cpu_reference(h_logits, h_expected, rows, VOCAB_SIZE, threshold);
    cuda_check(cudaMalloc((void **)&d_logits, logits_bytes), "cudaMalloc(logits)");
    cuda_check(cudaMalloc((void **)&d_out, (size_t)rows * sizeof(*d_out)), "cudaMalloc(output)");
    cuda_check(cudaMalloc((void **)&d_argmax, (size_t)rows * sizeof(*d_argmax)), "cudaMalloc(argmax)");
    cuda_check(cudaMalloc((void **)&d_first, partial_count * sizeof(*d_first)), "cudaMalloc(partial first)");
    cuda_check(cudaMalloc((void **)&d_second, partial_count * sizeof(*d_second)), "cudaMalloc(partial second)");
    cuda_check(cudaMalloc((void **)&d_argmax_partials,
                          partial_count * sizeof(*d_argmax_partials)),
               "cudaMalloc(argmax partials)");
    cuda_check(cudaMemcpy(d_logits, h_logits, logits_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(logits)");

    double elapsed = time_single(d_logits, d_out, rows, VOCAB_SIZE, threshold);
    cuda_check(cudaMemcpy(h_actual, d_out, (size_t)rows * sizeof(*h_actual), cudaMemcpyDeviceToHost), "copy shared-tree result");
    print_top2_result(repeat, rows, "single_cta_shared_tree", 0, 1, elapsed,
                      h_actual, results_match(h_actual, h_expected, rows));

    elapsed = time_single_warp(d_logits, d_out, rows, VOCAB_SIZE, threshold);
    cuda_check(cudaMemcpy(h_actual, d_out, (size_t)rows * sizeof(*h_actual), cudaMemcpyDeviceToHost), "copy warp-shuffle result");
    print_top2_result(repeat, rows, "single_cta_warp_shuffle", 0, 1, elapsed,
                      h_actual, results_match(h_actual, h_expected, rows));

    elapsed = time_argmax(d_logits, d_argmax, rows, VOCAB_SIZE);
    cuda_check(cudaMemcpy(h_argmax, d_argmax, (size_t)rows * sizeof(*h_argmax), cudaMemcpyDeviceToHost), "copy argmax result");
    int argmax_correct = 1;
    for (int row = 0; row < rows; ++row) {
        if (h_argmax[row] != h_expected[row].top1_index)
            argmax_correct = 0;
    }
    printf("%d,%d,%s,%d,%d,%.6f,%d,-1,%d,-1.00000000,-1,%d\n",
           repeat, rows, "single_cta_greedy_argmax_proxy", 0, 1, elapsed,
           h_argmax[0], h_argmax[0] == 220, argmax_correct);

    for (size_t j = 0; j < sizeof(tile_sizes) / sizeof(tile_sizes[0]); ++j) {
        int tile_size = tile_sizes[j];
        int tiles = (VOCAB_SIZE + tile_size - 1) / tile_size;
        elapsed = time_hierarchical(d_logits, d_first, d_second, d_out,
                                    rows, VOCAB_SIZE, tiles, tile_size, threshold);
        cuda_check(cudaMemcpy(h_actual, d_out, (size_t)rows * sizeof(*h_actual), cudaMemcpyDeviceToHost), "copy hierarchical result");
        print_top2_result(repeat, rows, "hierarchical", tile_size, tiles, elapsed,
                          h_actual, results_match(h_actual, h_expected, rows));

        elapsed = time_argmax_hierarchical(d_logits, d_argmax_partials, d_argmax,
                                           rows, VOCAB_SIZE, tiles, tile_size);
        cuda_check(cudaMemcpy(h_argmax, d_argmax, (size_t)rows * sizeof(*h_argmax), cudaMemcpyDeviceToHost), "copy hierarchical argmax result");
        argmax_correct = 1;
        for (int row = 0; row < rows; ++row) {
            if (h_argmax[row] != h_expected[row].top1_index)
                argmax_correct = 0;
        }
        printf("%d,%d,%s,%d,%d,%.6f,%d,-1,%d,-1.00000000,-1,%d\n",
               repeat, rows, "hierarchical_greedy_argmax_proxy", tile_size,
               tiles, elapsed, h_argmax[0], h_argmax[0] == 220, argmax_correct);
    }

    cudaFree(d_argmax_partials);
    cudaFree(d_second);
    cudaFree(d_first);
    cudaFree(d_argmax);
    cudaFree(d_out);
    cudaFree(d_logits);
    free(h_argmax);
    free(h_actual);
    free(h_expected);
    free(h_logits);
}

int main(int argc, char **argv)
{
    int device_count = 0;
    float threshold = 0.0001f;
    int repeat = 1;
    int rows[] = {1, 2, 4, 8, 16, 32};
    int device;
    cudaDeviceProp properties;

    if (argc > 3) {
        fprintf(stderr, "usage: %s [nonnegative-margin-threshold] [repeat-id]\n", argv[0]);
        return 2;
    }
    if (argc > 1) {
        char *end = NULL;
        threshold = strtof(argv[1], &end);
        if (end == argv[1] || *end != '\0' || !isfinite(threshold) || threshold < 0.0f) {
            fprintf(stderr, "usage: %s [nonnegative-margin-threshold] [repeat-id]\n", argv[0]);
            return 2;
        }
    }
    if (argc > 2) {
        char *end = NULL;
        long parsed = strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || parsed < 1 || parsed > 1000000) {
            fprintf(stderr, "repeat-id must be an integer in [1, 1000000]\n");
            return 2;
        }
        repeat = (int)parsed;
    }
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 1) {
        fprintf(stderr, "no CUDA device found\n");
        return 2;
    }
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    printf("# gpu=%s compute=%d.%d vocab=%d tile_sweep=512,1024,2048,4096,8192,16384 iterations=%d warmup=%d threshold=%.8f repeat=%d\n",
           properties.name, properties.major, properties.minor, VOCAB_SIZE,
           ITERATIONS, WARMUP, threshold, repeat);
    puts("repeat,rows,variant,tile_size,tiles,latency_ms,top1_id,top2_id,tie_probe_pass,margin,flagged,correct");
    srand(327);
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
        run_case(rows[i], threshold, repeat);
    return 0;
}
