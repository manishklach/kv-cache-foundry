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
    TILE_SIZE = 4096,
    ITERATIONS = 200,
    WARMUP = 25,
    MAX_ROWS = 64,
    VOCAB_SIZE = 151936
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

__global__ static void top2_partials(const float *logits,
                                     Candidate *partial_first,
                                     Candidate *partial_second,
                                     int vocab, int tiles)
{
    __shared__ Candidate firsts[THREADS];
    __shared__ Candidate seconds[THREADS];
    int row = (int)blockIdx.x / tiles;
    int tile = (int)blockIdx.x - row * tiles;
    int tid = (int)threadIdx.x;
    int begin = tile * TILE_SIZE;
    int end = begin + TILE_SIZE;
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

static double time_hierarchical(const float *d_logits, Candidate *d_first,
                                Candidate *d_second, Result *d_out,
                                int rows, int vocab, int tiles, float threshold)
{
    cudaEvent_t start, stop;
    int blocks = rows * tiles;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    for (int i = 0; i < WARMUP; ++i) {
        top2_partials<<<blocks, THREADS>>>(d_logits, d_first, d_second, vocab, tiles);
        top2_finalize<<<rows, THREADS>>>(d_first, d_second, d_out, tiles, threshold);
    }
    cuda_check(cudaGetLastError(), "hierarchical warmup launch");
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < ITERATIONS; ++i) {
        top2_partials<<<blocks, THREADS>>>(d_logits, d_first, d_second, vocab, tiles);
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

static void run_case(int rows, float threshold)
{
    size_t logits_count = (size_t)rows * (size_t)VOCAB_SIZE;
    size_t logits_bytes = logits_count * sizeof(float);
    int tiles = (VOCAB_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    size_t partial_count = (size_t)rows * (size_t)tiles;
    float *h_logits = (float *)malloc(logits_bytes);
    Result *h_expected = (Result *)malloc((size_t)rows * sizeof(*h_expected));
    Result *h_actual = (Result *)malloc((size_t)rows * sizeof(*h_actual));
    float *d_logits = NULL;
    Result *d_out = NULL;
    Candidate *d_first = NULL;
    Candidate *d_second = NULL;
    double one_cta_ms;
    double hierarchical_ms;
    int one_ok;
    int hierarchical_ok;

    if (h_logits == NULL || h_expected == NULL || h_actual == NULL) {
        fprintf(stderr, "host allocation failed for rows=%d\n", rows);
        exit(2);
    }
    fill_logits(h_logits, rows, VOCAB_SIZE);
    cpu_reference(h_logits, h_expected, rows, VOCAB_SIZE, threshold);
    cuda_check(cudaMalloc((void **)&d_logits, logits_bytes), "cudaMalloc(logits)");
    cuda_check(cudaMalloc((void **)&d_out, (size_t)rows * sizeof(*d_out)), "cudaMalloc(output)");
    cuda_check(cudaMalloc((void **)&d_first, partial_count * sizeof(*d_first)), "cudaMalloc(partial first)");
    cuda_check(cudaMalloc((void **)&d_second, partial_count * sizeof(*d_second)), "cudaMalloc(partial second)");
    cuda_check(cudaMemcpy(d_logits, h_logits, logits_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(logits)");

    one_cta_ms = time_single(d_logits, d_out, rows, VOCAB_SIZE, threshold);
    cuda_check(cudaMemcpy(h_actual, d_out, (size_t)rows * sizeof(*h_actual), cudaMemcpyDeviceToHost), "copy single result");
    one_ok = results_match(h_actual, h_expected, rows);

    hierarchical_ms = time_hierarchical(d_logits, d_first, d_second, d_out,
                                        rows, VOCAB_SIZE, tiles, threshold);
    cuda_check(cudaMemcpy(h_actual, d_out, (size_t)rows * sizeof(*h_actual), cudaMemcpyDeviceToHost), "copy hierarchical result");
    hierarchical_ok = results_match(h_actual, h_expected, rows);

    printf("%d,%d,%d,%.6f,%.6f,%.3f,%d,%d,%d,%.8f,%d,%d\n",
           rows, VOCAB_SIZE, tiles, one_cta_ms, hierarchical_ms,
           one_cta_ms / hierarchical_ms,
           h_actual[0].top1_index, h_actual[0].top2_index,
           h_actual[0].top1_index == 220 && h_actual[0].top2_index == 287,
           h_actual[0].margin, h_actual[0].flagged,
           one_ok && hierarchical_ok);

    cudaFree(d_second);
    cudaFree(d_first);
    cudaFree(d_out);
    cudaFree(d_logits);
    free(h_actual);
    free(h_expected);
    free(h_logits);
}

int main(int argc, char **argv)
{
    int device_count = 0;
    float threshold = 0.0001f;
    int rows[] = {1, 2, 4, 8, 16, 32};
    int device;
    cudaDeviceProp properties;

    if (argc > 1) {
        char *end = NULL;
        threshold = strtof(argv[1], &end);
        if (end == argv[1] || *end != '\0' || !isfinite(threshold) || threshold < 0.0f) {
            fprintf(stderr, "usage: %s [nonnegative-margin-threshold]\n", argv[0]);
            return 2;
        }
    }
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 1) {
        fprintf(stderr, "no CUDA device found\n");
        return 2;
    }
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    printf("# gpu=%s compute=%d.%d vocab=%d tile=%d iterations=%d warmup=%d threshold=%.8f\n",
           properties.name, properties.major, properties.minor, VOCAB_SIZE,
           TILE_SIZE, ITERATIONS, WARMUP, threshold);
    puts("rows,vocab,tiles,single_cta_ms,hierarchical_ms,speedup_single_over_hierarchical,"
         "top1_id,top2_id,tie_probe_pass,margin,flagged,correct");
    srand(327);
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
        run_case(rows[i], threshold);
    return 0;
}
