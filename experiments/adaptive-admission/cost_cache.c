/* Replay GreedyDual-Size on whole prefixes. Input: one key/size pair per line. */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MIN_PREFIX = 512 };
static const int lengths[] = {512, 1024, 2048, 4096};

struct entry {
    int key;
    int length;
    double priority;
};

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s capacity cost512 cost1024 cost2048 cost4096\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *end;
    errno = 0;
    long capacity = strtol(argv[1], &end, 10);
    if (errno || end == argv[1] || *end || capacity < 4096 || capacity > INT_MAX) {
        fprintf(stderr, "invalid cache capacity\n");
        return EXIT_FAILURE;
    }

    double density[4];
    for (size_t i = 0; i < 4; ++i) {
        const char *argument = argv[i + 2];
        double cost = strtod(argument, &end);
        if (end == argument || *end || !isfinite(cost) || cost < 0) {
            fprintf(stderr, "invalid rebuild cost\n");
            return EXIT_FAILURE;
        }
        density[i] = cost / lengths[i];
    }

    struct entry *entries = calloc((size_t)capacity / MIN_PREFIX, sizeof(*entries));
    if (!entries) {
        perror("allocating cache metadata");
        return EXIT_FAILURE;
    }

    size_t count = 0;
    long occupied = 0;
    double age = 0;
    int key, length, fields;
    while ((fields = scanf("%d%d", &key, &length)) == 2) {
        size_t bucket = 0;
        while (bucket < 4 && lengths[bucket] != length)
            ++bucket;
        if (bucket == 4) {
            fprintf(stderr, "unsupported prefix size: %d\n", length);
            free(entries);
            return EXIT_FAILURE;
        }

        size_t index = 0;
        while (index < count && entries[index].key != key)
            ++index;
        if (index < count) {
            if (entries[index].length != length) {
                fprintf(stderr, "resident key changed size: %d\n", key);
                free(entries);
                return EXIT_FAILURE;
            }
            memmove(entries + index, entries + index + 1,
                    (count - index - 1) * sizeof(*entries));
            entries[count - 1] = (struct entry){key, length, age + density[bucket]};
            puts("0");
            continue;
        }

        while (length > capacity - occupied) {
            size_t victim = 0;
            for (size_t i = 1; i < count; ++i) {
                if (entries[i].priority < entries[victim].priority)
                    victim = i;
            }
            age = entries[victim].priority;
            occupied -= entries[victim].length;
            --count;
            memmove(entries + victim, entries + victim + 1,
                    (count - victim) * sizeof(*entries));
        }
        entries[count++] = (struct entry){key, length, age + density[bucket]};
        occupied += length;
        printf("%d\n", length);
    }

    free(entries);
    if (fields != EOF || ferror(stdin) || fflush(stdout) == EOF) {
        fprintf(stderr, "could not complete cache replay\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
