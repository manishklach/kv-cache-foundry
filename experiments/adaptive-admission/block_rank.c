#include <stddef.h>
#include <stdlib.h>

struct candidate {
    size_t position;
    double score;
    int cached;
};

static int compare(const void *left, const void *right)
{
    const struct candidate *a = left;
    const struct candidate *b = right;
    if (a->cached != b->cached)
        return a->cached - b->cached;
    if (a->score != b->score)
        return a->score < b->score ? -1 : 1;
    return (a->position > b->position) - (a->position < b->position);
}

void rank_blocks(struct candidate *candidates, size_t count)
{
    qsort(candidates, count, sizeof(*candidates), compare);
}
