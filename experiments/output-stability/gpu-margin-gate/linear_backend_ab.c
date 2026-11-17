#define _POSIX_C_SOURCE 200809L
#include <curl/curl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    CURL *easy;
    char *response;
    size_t length;
    size_t capacity;
    long status;
    long prompt_tokens;
    long completion_tokens;
    double latency_ms;
    uint64_t content_hash;
} Request;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static size_t receive_body(char *data, size_t size, size_t count, void *opaque) {
    Request *r = (Request *)opaque;
    size_t bytes = size * count;
    if (r->length + bytes + 1 > r->capacity) return 0;
    memcpy(r->response + r->length, data, bytes);
    r->length += bytes;
    r->response[r->length] = '\0';
    return bytes;
}

static long json_integer(const char *json, const char *key) {
    char pattern[80];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    return p ? strtol(p + strlen(pattern), NULL, 10) : -1;
}

static uint64_t hash_content(const char *json) {
    const char *p = strstr(json, "\"content\":\"");
    if (!p) return 0;
    p += strlen("\"content\":\"");
    uint64_t h = UINT64_C(14695981039346656037);
    int escaped = 0;
    while (*p) {
        unsigned char c = (unsigned char)*p++;
        if (!escaped && c == '"') break;
        h = (h ^ c) * UINT64_C(1099511628211);
        if (!escaped && c == '\\') escaped = 1;
        else escaped = 0;
    }
    return h;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void destroy_request(Request *r) {
    if (r->easy) curl_easy_cleanup(r->easy);
    free(r->response);
    memset(r, 0, sizeof(*r));
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s URL MODEL TAG REQUESTS\n", argv[0]);
        return 2;
    }
    const char *url = argv[1], *model = argv[2], *tag = argv[3];
    long total = strtol(argv[4], NULL, 10), concurrency = 8;
    if (total < concurrency || total % concurrency != 0 || total > 4096) {
        fprintf(stderr, "REQUESTS must be a multiple of 8 in [8,4096]\n");
        return 2;
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) return 2;
    struct curl_slist *headers = curl_slist_append(NULL, "Content-Type: application/json");
    double *latencies = calloc((size_t)total, sizeof(double));
    if (!headers || !latencies) return 2;
    double all_start = now_seconds();
    long output_sum = 0, prompt_sum = 0, failed = 0;
    uint64_t first_hash = 0;
    long hash_mismatches = 0;
    printf("tag,index,wave,http_status,latency_ms,wave_ms,prompt_tokens,completion_tokens,content_hash\n");

    for (long base = 0; base < total; base += concurrency) {
        CURLM *multi = curl_multi_init();
        Request req[8] = {0};
        char bodies[8][1024];
        double wave_start = now_seconds();
        for (long j = 0; j < concurrency; ++j) {
            snprintf(bodies[j], sizeof(bodies[j]),
                "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain why batching improves GPU inference throughput in three concise sentences.\"}],\"temperature\":0,\"top_p\":1,\"max_tokens\":64}", model);
            req[j].capacity = 65536;
            req[j].response = calloc(req[j].capacity, 1);
            req[j].easy = curl_easy_init();
            curl_easy_setopt(req[j].easy, CURLOPT_URL, url);
            curl_easy_setopt(req[j].easy, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(req[j].easy, CURLOPT_POSTFIELDS, bodies[j]);
            curl_easy_setopt(req[j].easy, CURLOPT_WRITEFUNCTION, receive_body);
            curl_easy_setopt(req[j].easy, CURLOPT_WRITEDATA, &req[j]);
            curl_easy_setopt(req[j].easy, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(req[j].easy, CURLOPT_TIMEOUT, 90L);
            curl_multi_add_handle(multi, req[j].easy);
        }
        int running = 0;
        CURLMcode mc = curl_multi_perform(multi, &running);
        while (mc == CURLM_OK && running) {
            int fds = 0;
            mc = curl_multi_poll(multi, NULL, 0, 1000, &fds);
            if (mc == CURLM_OK) mc = curl_multi_perform(multi, &running);
        }
        double wave_ms = (now_seconds() - wave_start) * 1000.0;
        for (long j = 0; j < concurrency; ++j) {
            curl_easy_getinfo(req[j].easy, CURLINFO_RESPONSE_CODE, &req[j].status);
            double seconds = 0.0;
            curl_easy_getinfo(req[j].easy, CURLINFO_TOTAL_TIME, &seconds);
            req[j].latency_ms = seconds * 1000.0;
            req[j].prompt_tokens = json_integer(req[j].response, "prompt_tokens");
            req[j].completion_tokens = json_integer(req[j].response, "completion_tokens");
            req[j].content_hash = hash_content(req[j].response);
            latencies[base + j] = req[j].latency_ms;
            prompt_sum += req[j].prompt_tokens > 0 ? req[j].prompt_tokens : 0;
            output_sum += req[j].completion_tokens > 0 ? req[j].completion_tokens : 0;
            if (req[j].status != 200 || req[j].completion_tokens < 0) ++failed;
            if (!first_hash) first_hash = req[j].content_hash;
            else if (req[j].content_hash && req[j].content_hash != first_hash) ++hash_mismatches;
            printf("%s,%ld,%ld,%ld,%.3f,%.3f,%ld,%ld,%016llx\n", tag, base + j,
                base / concurrency, req[j].status, req[j].latency_ms, wave_ms,
                req[j].prompt_tokens, req[j].completion_tokens,
                (unsigned long long)req[j].content_hash);
            curl_multi_remove_handle(multi, req[j].easy);
            destroy_request(&req[j]);
        }
        curl_multi_cleanup(multi);
        fflush(stdout);
    }
    double elapsed = now_seconds() - all_start;
    qsort(latencies, (size_t)total, sizeof(double), compare_double);
    double median = (latencies[total / 2 - 1] + latencies[total / 2]) / 2.0;
    long p95_index = (long)ceil(0.95 * (double)total) - 1;
    printf("SUMMARY,%s,requests=%ld,failed=%ld,prompt_tokens=%ld,completion_tokens=%ld,elapsed_s=%.3f,output_tok_s=%.3f,median_latency_ms=%.3f,p95_latency_ms=%.3f,unique_content_hash_mismatches=%ld\n",
        tag, total, failed, prompt_sum, output_sum, elapsed,
        elapsed > 0.0 ? (double)output_sum / elapsed : 0.0, median,
        latencies[p95_index], hash_mismatches);
    free(latencies);
    curl_slist_free_all(headers);
    curl_global_cleanup();
    return failed ? 1 : 0;
}
