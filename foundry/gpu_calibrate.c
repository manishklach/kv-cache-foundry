/* C-only vLLM client for cold-prefill and exact-prefix cache calibration.
 * Requires curl on PATH. It emits one JSONL record per streamed completion. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Prompt token metadata can make one SSE line tens of kilobytes long. */
enum { BODY_CAP = 90000, LINE_CAP = 262144 };
static const unsigned lengths[] = {512, 1024, 2048, 4096};

static double now_ms(void)
{
    struct timespec time_value;
    if (clock_gettime(CLOCK_MONOTONIC, &time_value) != 0)
        return -1.0;
    return (double)time_value.tv_sec * 1000.0 +
           (double)time_value.tv_nsec / 1000000.0;
}

static char *make_body(const char *model, unsigned length, unsigned seed,
                       unsigned final_token, const char *salt)
{
    char *body = calloc(BODY_CAP, 1);
    if (body == NULL)
        return NULL;
    int used = snprintf(body, BODY_CAP,
        "{\"model\":\"%s\",\"prompt\":[", model);
    if (used < 0 || (size_t)used >= BODY_CAP) {
        free(body);
        return NULL;
    }
    for (unsigned i = 0; i < length; ++i) {
        unsigned token = 1000u + ((i * 2654435761u + seed * 2246822519u) % 7000u);
        int written = snprintf(body + used, BODY_CAP - (size_t)used,
                              "%s%u", i == 0 ? "" : ",", token);
        if (written < 0 || (size_t)written >= BODY_CAP - (size_t)used) {
            free(body);
            return NULL;
        }
        used += written;
    }
    int written = snprintf(body + used, BODY_CAP - (size_t)used,
        ",%u],\"cache_salt\":\"%s\",\"temperature\":0,\"max_tokens\":1,"
        "\"stream\":true,\"ignore_eos\":true,\"add_special_tokens\":false,"
        "\"return_token_ids\":true,\"stream_options\":{\"include_usage\":true}}",
        final_token, salt);
    if (written < 0 || (size_t)written >= BODY_CAP - (size_t)used) {
        free(body);
        return NULL;
    }
    return body;
}

static int json_integer(const char *json, const char *field)
{
    char key[96];
    if (snprintf(key, sizeof(key), "\"%s\":", field) >= (int)sizeof(key))
        return -1;
    const char *value = strstr(json, key);
    if (value == NULL)
        return -1;
    value += strlen(key);
    while (*value == ' ')
        ++value;
    if (*value < '0' || *value > '9')
        return 0;
    char *end = NULL;
    long result = strtol(value, &end, 10);
    if (end == value || result > 2147483647L)
        return -1;
    return (int)result;
}

static int request_once(const char *url, const char *api_key,
                        const char *body, double *first_token_ms,
                        double *total_ms, int *prompt_tokens,
                        int *cached_tokens)
{
    int output_pipe[2];
    if (pipe(output_pipe) != 0)
        return 0;
    pid_t child = fork();
    if (child < 0) {
        close(output_pipe[0]);
        close(output_pipe[1]);
        return 0;
    }
    if (child == 0) {
        close(output_pipe[0]);
        (void)dup2(output_pipe[1], STDOUT_FILENO);
        close(output_pipe[1]);
        char authorization[512];
        if (snprintf(authorization, sizeof(authorization),
                     "Authorization: Bearer %s", api_key) >= (int)sizeof(authorization))
            _exit(126);
        execlp("curl", "curl", "--fail", "--silent", "--show-error",
               "--no-buffer", "--max-time", "120", "-H", authorization,
               "-H", "Content-Type: application/json", "--data-binary", body,
               url, (char *)NULL);
        _exit(127);
    }

    close(output_pipe[1]);
    double start = now_ms();
    double first = -1.0;
    char line[LINE_CAP];
    size_t used = 0;
    char usage_json[LINE_CAP] = {0};
    int saw_token = 0;
    char ch;
    while (read(output_pipe[0], &ch, 1) == 1) {
        if (ch != '\n' && used + 1 < sizeof(line)) {
            line[used++] = ch;
            continue;
        }
        line[used] = '\0';
        if (strncmp(line, "data: ", 6) == 0 && strstr(line, "[DONE]") == NULL) {
            const char *json = line + 6;
            const char *ids = strstr(json, "\"token_ids\":[");
            if (ids != NULL && ids[strlen("\"token_ids\":[")] != ']') {
                if (first < 0.0)
                    first = now_ms() - start;
                saw_token = 1;
            }
            if (strstr(json, "\"usage\":{") != NULL)
                (void)snprintf(usage_json, sizeof(usage_json), "%s", json);
            if (strstr(json, "\"error\"") != NULL)
                (void)snprintf(usage_json, sizeof(usage_json), "%s", json);
        }
        used = 0;
    }
    close(output_pipe[0]);
    int status = 0;
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return 0;
    *total_ms = now_ms() - start;
    if (!saw_token || usage_json[0] == '\0') {
        fprintf(stderr, "incomplete response: %s\n", usage_json);
        return 0;
    }
    *prompt_tokens = json_integer(usage_json, "prompt_tokens");
    *cached_tokens = json_integer(usage_json, "cached_tokens");
    if (*prompt_tokens < 0)
        return 0;
    *first_token_ms = first;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s BASE_URL MODEL API_KEY REPEATS\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *end = NULL;
    long repeats = strtol(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0' || repeats < 1 || repeats > 20) {
        fprintf(stderr, "REPEATS must be from 1 to 20\n");
        return EXIT_FAILURE;
    }
    char url[1024];
    int url_size = snprintf(url, sizeof(url), "%s/v1/completions", argv[1]);
    if (url_size < 0 || (size_t)url_size >= sizeof(url))
        return EXIT_FAILURE;

    /* Keep independent program invocations from sharing vLLM's process-local
     * prefix cache when this calibration is rerun against the same server. */
    unsigned long run_id = (unsigned long)time(NULL);

    puts("prefix_tokens,repeat,case,prompt_tokens,cached_tokens,ttft_ms,total_ms");
    for (size_t bucket = 0; bucket < sizeof(lengths) / sizeof(lengths[0]); ++bucket) {
        for (long repeat = 0; repeat < repeats; ++repeat) {
            unsigned length = lengths[bucket];
            unsigned seed = (unsigned)(bucket * 100u + (size_t)repeat);
            char cold_salt[64], warm_salt[64];
            (void)snprintf(cold_salt, sizeof(cold_salt), "foundry-cold-%lu-%u-%ld",
                           run_id, length, repeat);
            (void)snprintf(warm_salt, sizeof(warm_salt), "foundry-warm-%lu-%u-%ld",
                           run_id, length, repeat);
            const char *cases[] = {"cold", "warm"};
            const char *salts[] = {cold_salt, warm_salt};
            for (unsigned which = 0; which < 2; ++which) {
                /* A warm pair starts with the exact same token prefix and salt. */
                unsigned salt_seed = which == 1 ? seed + 50000u : seed;
                char *body = make_body(argv[2], length, salt_seed,
                                       12000u + (unsigned)repeat + which, salts[which]);
                if (body == NULL)
                    return EXIT_FAILURE;
                double ttft, total;
                int prompt, cached;
                int ok = request_once(url, argv[3], body, &ttft, &total,
                                      &prompt, &cached);
                free(body);
                if (!ok) {
                    fprintf(stderr, "request failed at %u tokens, repeat %ld, %s\n",
                            length, repeat, cases[which]);
                    return EXIT_FAILURE;
                }
                if (which == 1 && cached == 0) {
                    /* A standalone warm pass is required before a cache hit
                     * can be measured. Repeat with the identical key/prefix. */
                    char *warm_body = make_body(argv[2], length, salt_seed,
                        13000u + (unsigned)repeat, salts[which]);
                    if (warm_body == NULL)
                        return EXIT_FAILURE;
                    ok = request_once(url, argv[3], warm_body, &ttft, &total,
                                      &prompt, &cached);
                    free(warm_body);
                    if (!ok) {
                        fprintf(stderr, "warm-up failed at %u tokens\n", length);
                        return EXIT_FAILURE;
                    }
                    /* Issue a measured warm request with same cached prefix. */
                    warm_body = make_body(argv[2], length, salt_seed,
                        14000u + (unsigned)repeat, salts[which]);
                    if (warm_body == NULL)
                        return EXIT_FAILURE;
                    ok = request_once(url, argv[3], warm_body, &ttft, &total,
                                      &prompt, &cached);
                    free(warm_body);
                    if (!ok) {
                        fprintf(stderr, "warm request failed at %u tokens\n", length);
                        return EXIT_FAILURE;
                    }
                }
                if ((which == 0 && cached != 0) ||
                    (which == 1 && cached == 0)) {
                    fprintf(stderr,
                        "cache accounting mismatch at %u tokens, repeat %ld, %s: %d cached\n",
                        length, repeat, cases[which], cached);
                    return EXIT_FAILURE;
                }
                printf("%u,%ld,%s,%d,%d,%.3f,%.3f\n", length, repeat,
                       cases[which], prompt, cached, ttft, total);
                fflush(stdout);
            }
        }
    }
    return EXIT_SUCCESS;
}
