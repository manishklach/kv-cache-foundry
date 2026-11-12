#ifndef FOUNDRY_COST_MODEL_H
#define FOUNDRY_COST_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define FOUNDRY_GPU_NAME_CAP 32
#define FOUNDRY_COST_MODEL_CAP 128

typedef struct {
    char gpu[FOUNDRY_GPU_NAME_CAP];
    uint32_t prefix_tokens;
} FoundryCostKey;

typedef struct {
    double reload_ms;
    double recompute_ms;
    double keep_ms;
    uint64_t samples;
    int has_reload;
    int has_recompute;
    int has_keep;
} FoundryCostEstimate;

typedef struct {
    FoundryCostKey key;
    FoundryCostEstimate estimate;
} FoundryCostEntry;

typedef struct {
    FoundryCostEntry entries[FOUNDRY_COST_MODEL_CAP];
    size_t count;
    double decay;
    FoundryCostEstimate prior;
} FoundryCostModel;

/* decay is the previous-value weight; for example 0.95 gives new data 5%. */
int foundry_cost_model_init(FoundryCostModel *model, double decay,
                            const FoundryCostEstimate *prior);

/* NaN means that a measurement for that cost is absent. */
int foundry_cost_model_observe(FoundryCostModel *model,
                               const FoundryCostKey *key,
                               double reload_ms,
                               double recompute_ms,
                               double keep_ms);

/* Exact GPU/prefix, then nearest prefix on the same GPU, then prior. */
int foundry_cost_model_estimate(const FoundryCostModel *model,
                                const FoundryCostKey *key,
                                FoundryCostEstimate *out,
                                int *used_fallback);

#endif
