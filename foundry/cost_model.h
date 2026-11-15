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

typedef enum {
    FOUNDRY_ACTION_LOCAL_HIT = 0,
    FOUNDRY_ACTION_RESTORE = 1,
    FOUNDRY_ACTION_RECOMPUTE = 2
} FoundryAction;

typedef struct {
    double restore_queue_ms;
    double recompute_queue_ms;
    double remaining_slack_ms;
    double lateness_penalty_per_ms;
    uint64_t minimum_samples;
} FoundryActionContext;

typedef struct {
    FoundryAction action;
    double predicted_finish_ms;
    double objective_ms;
    int used_fallback;
    int conservative_fallback;
} FoundryActionDecision;

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

/* Select the cheaper measured path plus queue delay and deadline penalty.
 * If evidence is missing/fallback/too sparse, choose conservative recompute. */
int foundry_choose_reload_or_recompute(const FoundryCostModel *model,
                                       const FoundryCostKey *key,
                                       const FoundryActionContext *context,
                                       FoundryActionDecision *out);

#endif
