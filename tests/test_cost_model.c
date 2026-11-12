#include "../foundry/cost_model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static FoundryCostKey key(const char *gpu, uint32_t prefix_tokens)
{
    FoundryCostKey result = {{0}, prefix_tokens};
    (void)snprintf(result.gpu, sizeof(result.gpu), "%s", gpu);
    return result;
}

int main(void)
{
    FoundryCostModel model;
    FoundryCostEstimate prior = {100.0, 200.0, 50.0, 0, 1, 1, 1};
    FoundryCostEstimate estimate;
    FoundryCostKey h100_1024 = key("H100-SXM", 1024);
    FoundryCostKey h100_2048 = key("H100-SXM", 2048);
    FoundryCostKey h200_1024 = key("H200-SXM", 1024);
    int fallback = -1;

    assert(foundry_cost_model_init(&model, 0.5, &prior));
    assert(!foundry_cost_model_init(&model, 1.0, &prior));
    assert(foundry_cost_model_estimate(&model, &h100_1024, &estimate, &fallback));
    assert(fallback == 1 && estimate.reload_ms == 100.0);

    assert(foundry_cost_model_observe(&model, &h100_1024, 20.0, 40.0, 30.0));
    assert(foundry_cost_model_observe(&model, &h100_1024, 10.0, NAN, 10.0));
    assert(foundry_cost_model_estimate(&model, &h100_1024, &estimate, &fallback));
    assert(fallback == 0 && estimate.samples == 2);
    assert(fabs(estimate.reload_ms - 15.0) < 1e-12);
    assert(fabs(estimate.recompute_ms - 40.0) < 1e-12);
    assert(fabs(estimate.keep_ms - 20.0) < 1e-12);

    assert(foundry_cost_model_estimate(&model, &h100_2048, &estimate, &fallback));
    assert(fallback == 1 && estimate.reload_ms == 15.0);
    assert(foundry_cost_model_estimate(&model, &h200_1024, &estimate, &fallback));
    assert(fallback == 1 && estimate.reload_ms == 100.0);
    assert(!foundry_cost_model_observe(&model, &h100_1024, NAN, NAN, NAN));
    assert(!foundry_cost_model_observe(&model, &h100_1024, -1.0, NAN, NAN));

    puts("cost model tests passed");
    return 0;
}
