/*
 * C11 analyzer for output-stability / throughput frontier measurements.
 *
 * Input is tab-separated, with one request per row:
 * mode<TAB>concurrency<TAB>wave<TAB>prompt_id<TAB>repeat<TAB>wave_ms<TAB>
 * ttft_ms<TAB>output_token_ids<TAB>top1_margin<TAB>flagged<TAB>verified
 *
 * output_token_ids is a semicolon-separated integer sequence. Use NA for an
 * unavailable timing or margin. wave_ms is repeated on each row in a wave;
 * the analyzer counts it once when computing aggregate throughput.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LINE_CAP = 65536, FIELD_CAP = 4096, MAX_ROWS = 100000 };

typedef struct {
    char mode[32];
    char prompt_id[128];
    char output_ids[FIELD_CAP];
    int concurrency;
    int wave;
    int repeat;
    double wave_ms;
    double ttft_ms;
    double margin;
    int flagged;
    int verified;
    int has_wave_ms;
    int has_ttft;
    int has_margin;
} Row;

typedef struct {
    char mode[32];
    int concurrency;
    unsigned requests;
    unsigned stable;
    unsigned matched;
    unsigned verified;
    unsigned flagged;
    unsigned flips;
    unsigned flipped_flagged;
    unsigned waves;
    unsigned long long output_tokens;
    double wave_ms_sum;
    double *ttft;
    size_t ttft_count;
    size_t ttft_cap;
    unsigned exact_tie;
    unsigned margin_count;
} Summary;

static int parse_int(const char *text, int *value)
{
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < 0 || parsed > 2147483647L)
        return 0;
    *value = (int)parsed;
    return 1;
}

static int parse_double_or_na(const char *text, double *value, int *present)
{
    char *end = NULL;
    if (strcmp(text, "NA") == 0 || text[0] == '\0') {
        *present = 0;
        *value = 0.0;
        return 1;
    }
    errno = 0;
    *value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(*value))
        return 0;
    *present = 1;
    return 1;
}

static int count_tokens(const char *ids)
{
    int count = 0;
    const char *cursor = ids;
    if (strcmp(ids, "NA") == 0 || ids[0] == '\0')
        return -1;
    while (*cursor != '\0') {
        char *end = NULL;
        long token;
        errno = 0;
        token = strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || token < 0 || token > 2147483647L)
            return -1;
        ++count;
        if (*end == '\0')
            break;
        if (*end != ';' || end[1] == '\0')
            return -1;
        cursor = end + 1;
    }
    return count;
}

static int split_fields(char *line, char *fields[], int expected)
{
    int count = 0;
    char *cursor = line;
    while (count < expected) {
        char *tab = strchr(cursor, '\t');
        if (tab != NULL)
            *tab = '\0';
        fields[count++] = cursor;
        if (tab == NULL)
            break;
        cursor = tab + 1;
    }
    return count;
}

static int parse_row(char *line, Row *row)
{
    char *fields[11];
    int count = split_fields(line, fields, 11);
    int flagged;
    int verified;
    if (count != 11 || strchr(fields[10], '\t') != NULL ||
        strlen(fields[0]) >= sizeof(row->mode) ||
        strlen(fields[3]) >= sizeof(row->prompt_id) ||
        strlen(fields[7]) >= sizeof(row->output_ids))
        return 0;
    memset(row, 0, sizeof(*row));
    (void)strcpy(row->mode, fields[0]);
    (void)strcpy(row->prompt_id, fields[3]);
    (void)strcpy(row->output_ids, fields[7]);
    if (!parse_int(fields[1], &row->concurrency) || row->concurrency < 1 ||
        !parse_int(fields[2], &row->wave) ||
        !parse_int(fields[4], &row->repeat) ||
        !parse_double_or_na(fields[5], &row->wave_ms, &row->has_wave_ms) ||
        !parse_double_or_na(fields[6], &row->ttft_ms, &row->has_ttft) ||
        !parse_double_or_na(fields[8], &row->margin, &row->has_margin) ||
        !parse_int(fields[9], &flagged) || flagged > 1 ||
        !parse_int(fields[10], &verified) || verified > 1 ||
        count_tokens(row->output_ids) < 0)
        return 0;
    row->verified = verified;
    row->flagged = flagged;
    if (row->has_wave_ms && row->wave_ms <= 0.0)
        return 0;
    if (row->has_ttft && row->ttft_ms < 0.0)
        return 0;
    if (row->has_margin && row->margin < 0.0)
        return 0;
    return row->mode[0] != '\0' && row->prompt_id[0] != '\0';
}

static int same_wave(const Row *left, const Row *right)
{
    return left->concurrency == right->concurrency &&
           left->wave == right->wave && left->repeat == right->repeat &&
           strcmp(left->mode, right->mode) == 0;
}

static int same_request(const Row *left, const Row *right)
{
    return left->repeat == right->repeat &&
           strcmp(left->prompt_id, right->prompt_id) == 0;
}

static int append_ttft(Summary *summary, double value)
{
    if (summary->ttft_count == summary->ttft_cap) {
        size_t next_cap = summary->ttft_cap == 0 ? 32 : summary->ttft_cap * 2;
        double *next = realloc(summary->ttft, next_cap * sizeof(*next));
        if (next == NULL)
            return 0;
        summary->ttft = next;
        summary->ttft_cap = next_cap;
    }
    summary->ttft[summary->ttft_count++] = value;
    return 1;
}

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double percentile(const Summary *summary, double fraction)
{
    size_t index;
    if (summary->ttft_count == 0)
        return NAN;
    qsort(summary->ttft, summary->ttft_count, sizeof(*summary->ttft), compare_double);
    index = (size_t)ceil(fraction * (double)summary->ttft_count);
    if (index == 0)
        index = 1;
    if (index > summary->ttft_count)
        index = summary->ttft_count;
    return summary->ttft[index - 1];
}

static Summary *get_summary(Summary summaries[], size_t *count,
                            const Row *row)
{
    size_t i;
    for (i = 0; i < *count; ++i) {
        if (summaries[i].concurrency == row->concurrency &&
            strcmp(summaries[i].mode, row->mode) == 0)
            return &summaries[i];
    }
    if (*count >= 128)
        return NULL;
    Summary *summary = &summaries[(*count)++];
    memset(summary, 0, sizeof(*summary));
    (void)strcpy(summary->mode, row->mode);
    summary->concurrency = row->concurrency;
    return summary;
}

static void print_value(double value)
{
    if (isfinite(value))
        (void)printf("%.3f", value);
    else
        (void)printf("NA");
}

int main(int argc, char **argv)
{
    FILE *input;
    char line[LINE_CAP];
    Row *rows = NULL;
    size_t row_capacity = 0;
    Summary summaries[128];
    size_t row_count = 0;
    size_t summary_count = 0;
    size_t i;
    const char *baseline_mode;
    int malformed = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s measurements.tsv baseline_mode\n", argv[0]);
        return 2;
    }
    baseline_mode = argv[2];
    input = fopen(argv[1], "r");
    if (input == NULL) {
        perror(argv[1]);
        return 2;
    }
    while (fgets(line, sizeof(line), input) != NULL) {
        size_t length = strlen(line);
        if (length == sizeof(line) - 1 && line[length - 1] != '\n') {
            fprintf(stderr, "input line exceeds %d bytes\n", LINE_CAP - 1);
            malformed = 1;
            break;
        }
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = '\0';
        if (length == 0 || line[0] == '#')
            continue;
        if (row_count == MAX_ROWS) {
            fprintf(stderr, "measurement row limit (%d) exceeded\n", MAX_ROWS);
            malformed = 1;
            break;
        }
        if (row_count == row_capacity) {
            size_t next_capacity = row_capacity == 0 ? 64 : row_capacity * 2;
            Row *next_rows;
            if (next_capacity > MAX_ROWS)
                next_capacity = MAX_ROWS;
            next_rows = realloc(rows, next_capacity * sizeof(*next_rows));
            if (next_rows == NULL) {
                fprintf(stderr, "unable to grow measurement rows\n");
                malformed = 1;
                break;
            }
            rows = next_rows;
            row_capacity = next_capacity;
        }
        if (!parse_row(line, &rows[row_count])) {
            fprintf(stderr, "invalid row at input line %zu\n", row_count + 1);
            malformed = 1;
            break;
        }
        ++row_count;
    }
    fclose(input);
    if (malformed || row_count == 0) {
        free(rows);
        return 2;
    }

    for (i = 0; i < row_count; ++i) {
        Summary *summary = get_summary(summaries, &summary_count, &rows[i]);
        int token_count = count_tokens(rows[i].output_ids);
        size_t j;
        int first_in_wave = 1;
        if (summary == NULL) {
            fprintf(stderr, "too many mode/concurrency groups\n");
            malformed = 1;
            break;
        }
        ++summary->requests;
        summary->output_tokens += (unsigned long long)token_count;
        summary->verified += (unsigned)rows[i].verified;
        summary->flagged += (unsigned)rows[i].flagged;
        if (rows[i].has_ttft && !append_ttft(summary, rows[i].ttft_ms)) {
            fprintf(stderr, "unable to grow TTFT sample array\n");
            malformed = 1;
            break;
        }
        if (rows[i].has_margin) {
            ++summary->margin_count;
            if (rows[i].margin == 0.0)
                ++summary->exact_tie;
        }
        for (j = 0; j < i; ++j) {
            if (same_wave(&rows[i], &rows[j])) {
                first_in_wave = 0;
                break;
            }
        }
        if (first_in_wave && rows[i].has_wave_ms) {
            ++summary->waves;
            summary->wave_ms_sum += rows[i].wave_ms;
        }
        for (j = 0; j < row_count; ++j) {
            if (strcmp(rows[j].mode, baseline_mode) == 0 &&
                rows[j].concurrency == 1 && same_request(&rows[i], &rows[j])) {
                ++summary->matched;
                if (strcmp(rows[i].output_ids, rows[j].output_ids) == 0) {
                    ++summary->stable;
                } else {
                    ++summary->flips;
                    if (rows[i].flagged)
                        ++summary->flipped_flagged;
                }
                break;
            }
        }
    }
    if (malformed) {
        for (i = 0; i < summary_count; ++i)
            free(summaries[i].ttft);
        free(rows);
        return 2;
    }

    puts("mode,concurrency,requests,baseline_matches,stable_matches,flip_rate_pct,"
         "flagged_requests,flagged_pct,flips_captured_pct,verified_requests,"
         "verified_pct,output_tokens,waves,throughput_tokens_s,"
         "ttft_p50_ms,ttft_p95_ms,exact_tie_rate_pct,margin_samples");
    for (i = 0; i < summary_count; ++i) {
        Summary *summary = &summaries[i];
        double flip_rate = summary->matched == 0 ? NAN :
            100.0 * (double)(summary->matched - summary->stable) /
            (double)summary->matched;
        double verified_pct = summary->requests == 0 ? NAN :
            100.0 * (double)summary->verified / (double)summary->requests;
        double flagged_pct = summary->requests == 0 ? NAN :
            100.0 * (double)summary->flagged / (double)summary->requests;
        double flips_captured_pct = summary->flips == 0 ? NAN :
            100.0 * (double)summary->flipped_flagged / (double)summary->flips;
        double throughput = summary->wave_ms_sum <= 0.0 ? NAN :
            1000.0 * (double)summary->output_tokens / summary->wave_ms_sum;
        double exact_tie_pct = summary->margin_count == 0 ? NAN :
            100.0 * (double)summary->exact_tie / (double)summary->margin_count;
        printf("%s,%d,%u,%u,%u,", summary->mode, summary->concurrency,
               summary->requests, summary->matched, summary->stable);
        print_value(flip_rate);
        printf(",%u,", summary->flagged);
        print_value(flagged_pct);
        printf(",");
        print_value(flips_captured_pct);
        printf(",%u,", summary->verified);
        print_value(verified_pct);
        printf(",%llu,%u,", summary->output_tokens, summary->waves);
        print_value(throughput);
        printf(",");
        print_value(percentile(summary, 0.50));
        printf(",");
        print_value(percentile(summary, 0.95));
        printf(",");
        print_value(exact_tie_pct);
        printf(",%u\n", summary->margin_count);
    }
    for (i = 0; i < summary_count; ++i)
        free(summaries[i].ttft);
    free(rows);
    return 0;
}
