#include "cost_model.h"

#include <math.h>
#include <string.h>

static int valid_measurement(double value)
{
    return isfinite(value) && value >= 0.0;
}

static int valid_or_missing(double value)
{
    return isnan(value) || valid_measurement(value);
}

static int valid_key(const FoundryCostKey *key)
{
    return key != NULL && key->gpu[0] != '\0' &&
           memchr(key->gpu, '\0', sizeof(key->gpu)) != NULL &&
           key->prefix_tokens != 0;
}

static int same_key(const FoundryCostKey *left, const FoundryCostKey *right)
{
    return left->prefix_tokens == right->prefix_tokens &&
           strcmp(left->gpu, right->gpu) == 0;
}

static void update_value(double *current, int *has_value, double value,
                         double decay)
{
    if (!valid_measurement(value))
        return;
    *current = *has_value ? decay * *current + (1.0 - decay) * value : value;
    *has_value = 1;
}

int foundry_cost_model_init(FoundryCostModel *model, double decay,
                            const FoundryCostEstimate *prior)
{
    if (model == NULL || !isfinite(decay) || decay < 0.0 || decay >= 1.0)
        return 0;
    memset(model, 0, sizeof(*model));
    model->decay = decay;
    if (prior != NULL)
        model->prior = *prior;
    return 1;
}

int foundry_cost_model_observe(FoundryCostModel *model,
                               const FoundryCostKey *key,
                               double reload_ms,
                               double recompute_ms,
                               double keep_ms)
{
    size_t index;
    int has_observation = valid_measurement(reload_ms) ||
                          valid_measurement(recompute_ms) ||
                          valid_measurement(keep_ms);

    if (model == NULL || !valid_key(key) || !has_observation ||
        !valid_or_missing(reload_ms) || !valid_or_missing(recompute_ms) ||
        !valid_or_missing(keep_ms))
        return 0;

    for (index = 0; index < model->count; ++index) {
        if (same_key(&model->entries[index].key, key))
            break;
    }
    if (index == model->count) {
        if (model->count == FOUNDRY_COST_MODEL_CAP)
            return 0;
        model->entries[index].key = *key;
        ++model->count;
    }

    FoundryCostEstimate *estimate = &model->entries[index].estimate;
    update_value(&estimate->reload_ms, &estimate->has_reload,
                 reload_ms, model->decay);
    update_value(&estimate->recompute_ms, &estimate->has_recompute,
                 recompute_ms, model->decay);
    update_value(&estimate->keep_ms, &estimate->has_keep,
                 keep_ms, model->decay);
    ++estimate->samples;
    return 1;
}

int foundry_cost_model_estimate(const FoundryCostModel *model,
                                const FoundryCostKey *key,
                                FoundryCostEstimate *out,
                                int *used_fallback)
{
    size_t index;
    size_t nearest;
    uint32_t nearest_distance = UINT32_MAX;

    if (model == NULL || !valid_key(key) || out == NULL)
        return 0;
    nearest = model->count;

    for (index = 0; index < model->count; ++index) {
        const FoundryCostKey *candidate = &model->entries[index].key;
        if (strcmp(candidate->gpu, key->gpu) != 0)
            continue;
        if (candidate->prefix_tokens == key->prefix_tokens) {
            *out = model->entries[index].estimate;
            if (used_fallback != NULL)
                *used_fallback = 0;
            return 1;
        }
        uint32_t distance = candidate->prefix_tokens > key->prefix_tokens
                                ? candidate->prefix_tokens - key->prefix_tokens
                                : key->prefix_tokens - candidate->prefix_tokens;
        if (distance < nearest_distance) {
            nearest = index;
            nearest_distance = distance;
        }
    }

    if (nearest < model->count) {
        *out = model->entries[nearest].estimate;
        if (used_fallback != NULL)
            *used_fallback = 1;
        return 1;
    }

    *out = model->prior;
    if (used_fallback != NULL)
        *used_fallback = 1;
    return 1;
}
