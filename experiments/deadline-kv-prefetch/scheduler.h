#include <stddef.h>

struct transfer {
    size_t bytes_left;
    double need_ms;
};

/* Earliest predicted deadline after subtracting the remaining copy time. */
static inline int select_transfer(const struct transfer *jobs, int count,
                                  double bytes_per_ms, int deadline_order)
{
    int best = -1;
    double best_slack = 0;
    for (int i = 0; i < count; ++i) {
        if (jobs[i].bytes_left == 0)
            continue;
        const double slack = jobs[i].need_ms - jobs[i].bytes_left / bytes_per_ms;
        if (best < 0 || (deadline_order && slack < best_slack)) {
            best = i;
            best_slack = slack;
        }
    }
    return best;
}
