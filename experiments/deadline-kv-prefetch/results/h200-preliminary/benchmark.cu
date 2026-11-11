#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "scheduler.h"

constexpr int layers = 24, heads = 8, dim = 32, threads = 128;
constexpr int requests = 8, foreground_steps = 96;
constexpr size_t chunk_bytes = 2 * 1024 * 1024;

#define CUDA(call) do {     const cudaError_t status = (call);     if (status != cudaSuccess) {         std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status));         std::exit(EXIT_FAILURE);     } } while (0)

static double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

__global__ void attention(const float *kv, const float *query, float *output, int tokens)
{
    extern __shared__ float scores[];
    __shared__ float reduction[threads];
    const int h = blockIdx.x, lane = threadIdx.x;
    const float *key = kv + h * tokens * dim;
    const float *value = kv + (heads + h) * tokens * dim;
    float maximum = -CUDART_INF_F;
    for (int t = lane; t < tokens; t += threads) {
        float dot = 0;
        for (int d = 0; d < dim; ++d)
            dot += key[t * dim + d] * query[h * dim + d];
        scores[t] = dot * rsqrtf(float(dim));
        maximum = fmaxf(maximum, scores[t]);
    }
    reduction[lane] = maximum;
    __syncthreads();
    for (int width = threads / 2; width > 0; width /= 2) {
        if (lane < width)
            reduction[lane] = fmaxf(reduction[lane], reduction[lane + width]);
        __syncthreads();
    }
    maximum = reduction[0];
    __syncthreads();
    float sum = 0;
    for (int t = lane; t < tokens; t += threads) {
        scores[t] = expf(scores[t] - maximum);
        sum += scores[t];
    }
    reduction[lane] = sum;
    __syncthreads();
    for (int width = threads / 2; width > 0; width /= 2) {
        if (lane < width)
            reduction[lane] += reduction[lane + width];
        __syncthreads();
    }
    if (lane < dim) {
        float weighted = 0;
        for (int t = 0; t < tokens; ++t)
            weighted += scores[t] * value[t * dim + lane];
        output[h * dim + lane] = weighted / reduction[0];
    }
}

struct request {
    int tokens;
    double arrival_ms, target_ms;
    size_t layer_bytes;
    float *host, *device, *output;
};

struct progress {
    size_t copied = 0;
    int computed = 0;
    double finished_ms = 0;
};

static bool ready(cudaEvent_t event)
{
    const auto status = cudaEventQuery(event);
    if (status == cudaErrorNotReady)
        return false;
    CUDA(status);
    return true;
}

static void launch(const request &job, const float *query, int layer, cudaStream_t stream)
{
    const size_t stride = job.layer_bytes / sizeof(float);
    const float *input = layer == 0 ? query : job.output + (layer - 1) * heads * dim;
    attention<<<heads, threads, job.tokens * sizeof(float), stream>>>(
        job.device + layer * stride, input, job.output + layer * heads * dim, job.tokens);
    CUDA(cudaGetLastError());
}

static void allocate(request &job, int tokens, double arrival, double target, int seed)
{
    job.tokens = tokens;
    job.arrival_ms = arrival;
    job.target_ms = target;
    job.layer_bytes = size_t(2) * heads * tokens * dim * sizeof(float);
    CUDA(cudaMallocHost(&job.host, layers * job.layer_bytes));
    CUDA(cudaMalloc(&job.device, layers * job.layer_bytes));
    CUDA(cudaMalloc(&job.output, layers * heads * dim * sizeof(float)));
    for (size_t i = 0; i < layers * job.layer_bytes / sizeof(float); ++i) {
        unsigned x = unsigned(i) + unsigned(seed) * 65537u;
        x = (x ^ (x >> 16)) * 0x45d9f3bu;
        x = (x ^ (x >> 16)) * 0x45d9f3bu;
        job.host[i] = float(int((x ^ (x >> 16)) & 255u) - 128) / 128;
    }
}

static void release(request &job)
{
    CUDA(cudaFreeHost(job.host));
    CUDA(cudaFree(job.device));
    CUDA(cudaFree(job.output));
}

static void run(const std::array<request, requests> &jobs, const request &foreground,
                const float *query, double bandwidth, double layer_ms, int mode, int trial,
                std::array<std::vector<float>, requests> &reference)
{
    cudaStream_t copy, compute, decode;
    int low, high;
    CUDA(cudaDeviceGetStreamPriorityRange(&low, &high));
    CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithPriority(&compute, cudaStreamNonBlocking, low));
    CUDA(cudaStreamCreateWithPriority(&decode, cudaStreamNonBlocking, high));
    cudaEvent_t copy_done, compute_done, decode_done;
    CUDA(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&decode_done, cudaEventDisableTiming));
    std::array<progress, requests> state{};
    std::vector<double> gaps;
    const double start = now_ms();
    double last_decode = start;
    int copying = -1, computing = -1, finished = 0;
    int decode_layer = 0;
    bool decoding = false;
    size_t submitted = 0;

    while (finished != requests || gaps.size() < foreground_steps) {
        if (copying >= 0 && ready(copy_done)) {
            state[copying].copied += submitted;
            copying = -1;
        }
        if (computing >= 0 && ready(compute_done)) {
            auto &p = state[computing];
            if (++p.computed == layers) {
                p.finished_ms = now_ms() - start;
                ++finished;
            }
            computing = -1;
        }
        if (decoding && ready(decode_done)) {
            decoding = false;
            if (++decode_layer == layers) {
                decode_layer = 0;
                const double timestamp = now_ms();
                gaps.push_back(timestamp - last_decode);
                last_decode = timestamp;
            }
        }
        if (!decoding && gaps.size() < foreground_steps) {
            launch(foreground, query, decode_layer, decode);
            CUDA(cudaEventRecord(decode_done, decode));
            decoding = true;
        }

        const double elapsed = now_ms() - start;
        if (computing < 0) {
            for (int i = 0; i < requests; ++i) {
                const auto &job = jobs[i];
                const auto &p = state[i];
                const size_t needed = job.layer_bytes * (mode == 0 ? layers : p.computed + 1);
                if (p.computed < layers && p.copied >= needed) {
                    launch(job, query, p.computed, compute);
                    CUDA(cudaEventRecord(compute_done, compute));
                    computing = i;
                    break;
                }
            }
        }

        if (copying < 0) {
            struct transfer candidates[requests]{};
            for (int i = 0; i < requests; ++i) {
                const auto &job = jobs[i];
                const auto &p = state[i];
                if (elapsed < job.arrival_ms || p.copied == job.layer_bytes * layers)
                    continue;
                const int next_layer = int(p.copied / job.layer_bytes);
                if (mode != 0 && next_layer > p.computed + 1)
                    continue;
                candidates[i].bytes_left = job.layer_bytes - p.copied % job.layer_bytes;
                candidates[i].need_ms = job.arrival_ms + job.target_ms
                    - (layers - next_layer) * layer_ms;
            }
            copying = select_transfer(candidates, requests, bandwidth, mode == 2);
            if (copying >= 0) {
                const auto &job = jobs[copying];
                const size_t offset = state[copying].copied;
                submitted = std::min(chunk_bytes, candidates[copying].bytes_left);
                CUDA(cudaMemcpyAsync(reinterpret_cast<char *>(job.device) + offset,
                                     reinterpret_cast<const char *>(job.host) + offset,
                                     submitted, cudaMemcpyHostToDevice, copy));
                CUDA(cudaEventRecord(copy_done, copy));
            }
        }
    }

    const double makespan = now_ms() - start;
    std::sort(gaps.begin(), gaps.end());
    const double foreground_p95 = gaps[(gaps.size() * 95 + 99) / 100 - 1];
    for (int i = 0; i < requests; ++i) {
        std::vector<float> result(layers * heads * dim);
        CUDA(cudaMemcpy(result.data(), jobs[i].output, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : result) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "non-finite attention output\n");
                std::exit(EXIT_FAILURE);
            }
        }
        if (reference[i].empty())
            reference[i] = result;
        if (std::memcmp(result.data(), reference[i].data(), result.size() * sizeof(float))) {
            std::fprintf(stderr, "attention output mismatch at request %d\n", i);
            std::exit(EXIT_FAILURE);
        }
        const auto &job = jobs[i];
        const char *names[] = {"full", "fifo", "deadline"};
        std::printf("%d,%s,%d,%d,%.3f,%.3f,%.6f,%.6f,%.6f\n", trial, names[mode], i,
                    job.tokens, job.arrival_ms, job.target_ms,
                    state[i].finished_ms - job.arrival_ms, foreground_p95, makespan);
    }
    CUDA(cudaEventDestroy(copy_done));
    CUDA(cudaEventDestroy(compute_done));
    CUDA(cudaEventDestroy(decode_done));
    CUDA(cudaStreamDestroy(copy));
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaStreamDestroy(decode));
}

int main()
{
    std::array<request, requests> jobs{};
    for (int i = 0; i < requests; ++i)
        allocate(jobs[i], i % 2 == 0 ? 4096 : 512, i * 0.35, i % 2 == 0 ? 65 : 15, i);
    request foreground{};
    allocate(foreground, 2048, 0, 0, requests);
    CUDA(cudaMemcpy(foreground.device, foreground.host, layers * foreground.layer_bytes, cudaMemcpyHostToDevice));
    float *query;
    CUDA(cudaMalloc(&query, heads * dim * sizeof(float)));
    CUDA(cudaMemcpy(query, foreground.host, heads * dim * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < layers; ++i)
        launch(foreground, query, i, nullptr);
    CUDA(cudaDeviceSynchronize());
    double start = now_ms();
    for (int i = 0; i < 16; ++i)
        CUDA(cudaMemcpy(jobs[0].device, jobs[0].host, chunk_bytes, cudaMemcpyHostToDevice));
    const double bandwidth = 16 * chunk_bytes / (now_ms() - start);
    start = now_ms();
    for (int i = 0; i < layers; ++i)
        launch(foreground, query, i, nullptr);
    CUDA(cudaDeviceSynchronize());
    const double layer_ms = (now_ms() - start) / layers;
    std::fprintf(stderr, "measured copy %.2f GB/s, attention %.3f ms/layer\n", bandwidth / 1e6, layer_ms);
    std::array<std::vector<float>, requests> reference{};
    std::puts("trial,mode,request,tokens,arrival_ms,target_ms,request_ms,foreground_p95_ms,makespan_ms");
    for (int trial = 0; trial < 9; ++trial)
        for (int order = 0; order < 3; ++order)
            run(jobs, foreground, query, bandwidth, layer_ms, (trial + order) % 3, trial, reference);
    CUDA(cudaFree(query));
    release(foreground);
    for (auto &job : jobs)
        release(job);
}
