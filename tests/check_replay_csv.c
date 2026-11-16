#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 3
#define LINE_CAP 256

typedef struct {
    char name[80];
    unsigned capacity;
    unsigned seeds;
    double missed;
    double deadline;
    double objective;
} Row;

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s replay.csv\n", argv[0]);
        return EXIT_FAILURE;
    }
    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
        perror(argv[1]);
        return EXIT_FAILURE;
    }
    char line[LINE_CAP];
    const char *header = "policy,capacity_tokens,seeds,mean_missed_tokens,mean_deadline_misses,mean_objective_ms";
    if (fgets(line, sizeof(line), input) == NULL) {
        fprintf(stderr, "missing CSV header\n");
        fclose(input);
        return EXIT_FAILURE;
    }
    line[strcspn(line, "\r\n")] = '\0';
    if (strcmp(line, header) != 0) {
        fprintf(stderr, "unexpected CSV header\n");
        fclose(input);
        return EXIT_FAILURE;
    }

    Row rows[ROWS] = {0};
    size_t count = 0;
    while (fgets(line, sizeof(line), input) != NULL) {
        if (count == ROWS) {
            fprintf(stderr, "unexpected extra replay row\n");
            fclose(input);
            return EXIT_FAILURE;
        }
        Row *row = &rows[count];
        if (sscanf(line, "%79[^,],%u,%u,%lf,%lf,%lf",
                   row->name, &row->capacity, &row->seeds,
                   &row->missed, &row->deadline, &row->objective) != 6) {
            fprintf(stderr, "malformed replay CSV row %zu\n", count + 2);
            fclose(input);
            return EXIT_FAILURE;
        }
        ++count;
    }
    fclose(input);
    if (count != ROWS || strcmp(rows[0].name, "lru_like") != 0 ||
        strcmp(rows[1].name, "oracle_reuse_rank") != 0 ||
        strcmp(rows[2].name, "slack_priced_admission_lru_eviction") != 0) {
        fprintf(stderr, "expected the three documented replay policies\n");
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < ROWS; ++i) {
        for (size_t j = i + 1; j < ROWS; ++j) {
            if (rows[i].missed == rows[j].missed &&
                rows[i].deadline == rows[j].deadline &&
                rows[i].objective == rows[j].objective) {
                fprintf(stderr, "policies %s and %s have identical metrics\n",
                        rows[i].name, rows[j].name);
                return EXIT_FAILURE;
            }
        }
    }
    puts("replay CSV contains three distinct documented policies");
    return EXIT_SUCCESS;
}
