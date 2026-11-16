/* Deterministic synthetic replay for deadline-aware cache value.
 * This is a CPU model, not a serving benchmark. */
#include "cost_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    KEY_COUNT = 64,
    REQUESTS_PER_PHASE = 1024,
    PHASES = 4,
    POLICY_LRU = 0,
    POLICY_ORACLE_REUSE_RANK = 1,
    POLICY_SLACK_PRICED_ADMISSION = 2
};
static const uint32_t lengths[] = {512, 1024, 2048, 4096};

typedef struct {
    unsigned key;
    uint32_t length;
    unsigned last_use;
    double priority;
    int resident;
} Entry;

typedef struct {
    Entry entries[KEY_COUNT];
    unsigned capacity;
    unsigned occupied;
    double age;
    unsigned long long missed_tokens;
    unsigned long long deadline_misses;
    double objective_ms;
} Cache;

static const unsigned long long REUSE[KEY_COUNT] = {
    18, 12, 24, 8, 20, 16, 10, 14,
    2, 1, 3, 2, 1, 4, 2, 3, 1, 2, 3, 1, 2, 4, 1, 3,
    2, 1, 4, 2, 3, 1, 2, 4, 1, 3, 2, 1, 4, 2, 3, 1,
    2, 4, 1, 3, 2, 1, 4, 2, 3, 1, 2, 4, 1, 3, 2, 1,
    4, 2, 3, 1, 2, 4, 1, 3
};

static unsigned rng_next(unsigned *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static uint32_t prefix_length(unsigned x)
{
    return lengths[(x >> 8) & 3u];
}

static void evict_one(Cache *cache, unsigned current, int oracle_reuse_rank)
{
    unsigned victim = KEY_COUNT;
    double lowest = HUGE_VAL;
    for (unsigned i = 0; i < KEY_COUNT; ++i) {
        if (!cache->entries[i].resident)
            continue;
        double score;
        if (oracle_reuse_rank) {
            /* Deliberately privileged comparator: fixed future reuse count,
             * decayed by age. This is not a deployable policy. */
            score = cache->entries[i].priority /
                (double)(current - cache->entries[i].last_use + 1u);
        } else {
            /* LRU chooses the least recently used resident entry. */
            score = (double)cache->entries[i].last_use;
        }
        if (score < lowest) {
            lowest = score;
            victim = i;
        }
    }
    if (victim < KEY_COUNT) {
        cache->occupied -= cache->entries[victim].length;
        cache->entries[victim].resident = 0;
    }
}

static void access_cache(Cache *cache, unsigned key, uint32_t length,
                         unsigned request, unsigned reuse_distance,
                         double miss_cost_ms, int policy)
{
    Entry *entry = &cache->entries[key];
    const double slack = (request % 13u == 0u) ? 5.0 : 80.0;
    const double penalty = 3.0;
    double cost = entry->resident ? 0.0 : miss_cost_ms;
    if (!entry->resident) {
        cache->missed_tokens += length;
        if (cost > slack)
            ++cache->deadline_misses;
        cache->objective_ms += cost + penalty * fmax(0.0, cost - slack);
    }

    entry->length = length;
    entry->last_use = request;
    entry->priority = (double)(REUSE[key] + 1u) * (double)length;

    int admit = 1;
    if (policy == POLICY_SLACK_PRICED_ADMISSION && !entry->resident) {
        /* Admission proxy: protect scarce slack for long rebuilds when the
         * request is not predicted to repeat soon. */
        double expected_saved = (double)REUSE[key] / 20.0 * miss_cost_ms;
        double memory_charge = (double)length / (double)cache->capacity * 18.0;
        double slack_value = slack < 10.0 ? 2.0 : 1.0;
        admit = expected_saved * slack_value > memory_charge &&
                REUSE[key] >= 2u && reuse_distance <= 40u;
    }
    if (!admit || length > cache->capacity)
        return;
    while (cache->occupied + length > cache->capacity)
        evict_one(cache, request, policy == POLICY_ORACLE_REUSE_RANK);
    if (!entry->resident) {
        entry->resident = 1;
        cache->occupied += length;
    }
}

static void run(unsigned seed, unsigned capacity, int policy,
                unsigned long long *missed, unsigned long long *deadline,
                double *objective_total)
{
    Cache cache = {0};
    cache.capacity = capacity;
    unsigned state = seed + 1u;
    unsigned tick = 0;
    for (unsigned phase = 0; phase < PHASES; ++phase) {
        for (unsigned i = 0; i < REQUESTS_PER_PHASE; ++i, ++tick) {
            unsigned key;
            uint32_t length;
            unsigned reuse_distance = 1u;
            if (phase % 2u == 0u) {
                if (rng_next(&state) % 100u < 45u) {
                    key = rng_next(&state) % 7u;
                    length = 4096u;
                    reuse_distance = 1u + rng_next(&state) % 3u;
                } else {
                    key = 8u + (i % 56u);
                    length = prefix_length(rng_next(&state));
                    reuse_distance = 24u + rng_next(&state) % 40u;
                }
            } else {
                unsigned group = i / 16u;
                unsigned slot = (i / 2u) % 8u;
                key = 8u + (group * 8u + slot) % 56u;
                length = lengths[(group + slot + seed) % 4u];
                reuse_distance = (i % 16u < 8u) ? 8u : 1u;
            }
            double miss_cost = 2.0 + (double)length / 96.0;
            access_cache(&cache, key, length, tick, reuse_distance,
                         miss_cost, policy);
        }
    }
    *missed = cache.missed_tokens;
    *deadline = cache.deadline_misses;
    *objective_total = cache.objective_ms;
}

int main(int argc, char **argv)
{
    unsigned capacity = 32768u;
    unsigned seeds = 20u;
    if (argc > 1) capacity = (unsigned)strtoul(argv[1], NULL, 10);
    if (argc > 2) seeds = (unsigned)strtoul(argv[2], NULL, 10);
    if (capacity < 4096u || seeds == 0u || seeds > 10000u) {
        fprintf(stderr, "usage: %s [capacity_tokens>=4096] [seeds=1..10000]\n", argv[0]);
        return EXIT_FAILURE;
    }

    puts("policy,capacity_tokens,seeds,mean_missed_tokens,mean_deadline_misses,mean_objective_ms");
    const char *names[] = {
        "lru_like", "oracle_reuse_rank", "slack_priced_admission_lru_eviction"
    };
    for (int policy = 0; policy < 3; ++policy) {
        unsigned long long missed_sum = 0, deadline_sum = 0;
        double objective_sum = 0.0;
        for (unsigned seed = 0; seed < seeds; ++seed) {
            unsigned long long missed, deadline;
            double objective_ms;
            run(seed, capacity, policy, &missed, &deadline, &objective_ms);
            missed_sum += missed;
            deadline_sum += deadline;
            objective_sum += objective_ms;
        }
        printf("%s,%u,%u,%.2f,%.2f,%.4f\n", names[policy], capacity, seeds,
               (double)missed_sum / seeds, (double)deadline_sum / seeds,
               objective_sum / seeds);
    }
    return EXIT_SUCCESS;
}
