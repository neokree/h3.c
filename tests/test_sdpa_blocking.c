/* Query-blocked SDPA: is a block's result the same answer, wherever the block
 * boundaries fall, and is it the right answer?
 *
 * Both questions are asked against a double-precision CPU oracle rather than
 * against the whole-sequence encode, because the whole-sequence encode is not
 * a reference: above roughly 9,000 tokens at 56 heads MPSGraph switches
 * scaledDotProductAttention to a kernel whose error is an order of magnitude
 * worse (see docs/720p-blocked-attention.md). Synthesised inputs only, so this
 * never skips.
 *
 * Two shapes. The small one runs in 0.8 s, is what proved block-size
 * invariance, and is what `make test` gets. The second is the shape production
 * actually runs - N = 33,329 at 56 heads, head_dim 128, bf16, non-causal, which
 * is 1280x704 / 124 frames - and sits behind H3_SDPA_TARGET=1 because it
 * allocates about 6 GiB. The whole-sequence encode is deliberately not run at
 * the target: past 2^33 score elements it aborts with "too large for kernel",
 * which is the whole reason blocking exists. The band where it still encodes
 * and is already wrong, N = 9,100 to 12,000, is measured in the doc. */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SEQUENCE = 1000, HEADS = 8, HEAD_DIM = 128, PROBES = 6 };

/* 1280x704 / 124 frames. TARGET_HEAD_STRIDE samples 8 of the 56 heads: the
 * block offset is in whole rows, so it cannot go wrong on one head and right on
 * another, and 8 heads x 64 rows is already 65,536 scored elements. */
enum { TARGET_SEQUENCE = 33329, TARGET_HEADS = 56, TARGET_PROBES = 64,
       TARGET_HEAD_STRIDE = 7 };

static uint64_t rng_state = 0x2545F4914F6CDD1DULL;

static uint16_t next_bf16(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    uint32_t sign = (uint32_t)(rng_state >> 63) << 15;
    uint32_t exponent = (uint32_t)(118 + (rng_state >> 59 & 7)) << 7;
    uint32_t mantissa = (uint32_t)(rng_state >> 40) & 0x7F;
    return (uint16_t)(sign | exponent | mantissa);
}

static float bf16_to_f32(uint16_t value) {
    union { uint32_t bits; float number; } cast;
    cast.bits = (uint32_t)value << 16;
    return cast.number;
}

static int failures = 0;

static void require(int condition, const char *what) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: %s\n", what);
    failures++;
}

static double seconds_since(clock_t start) {
    return (double)(clock() - start) / (double)CLOCKS_PER_SEC;
}

/* One (query row, head) pair of the attention, in double precision. Every
 * query row of an unmasked attention depends on all of K and V but on no other
 * query row, which is what makes a sampled reference exact rather than an
 * approximation. */
static void oracle_row(const uint16_t *query, const uint16_t *key,
                       const uint16_t *value, uint32_t sequence, uint32_t heads,
                       uint32_t row, uint32_t head, double *out,
                       double *scratch) {
    const size_t stride = (size_t)heads * HEAD_DIM;
    const uint16_t *q = query + (size_t)row * stride + (size_t)head * HEAD_DIM;
    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    double top = -INFINITY, sum = 0.0;
    for (uint32_t j = 0; j < sequence; j++) {
        const uint16_t *k = key + (size_t)j * stride + (size_t)head * HEAD_DIM;
        double accumulator = 0.0;
        for (int d = 0; d < HEAD_DIM; d++)
            accumulator += (double)bf16_to_f32(q[d]) * (double)bf16_to_f32(k[d]);
        scratch[j] = accumulator * scale;
        if (scratch[j] > top) top = scratch[j];
    }
    for (uint32_t j = 0; j < sequence; j++) {
        scratch[j] = exp(scratch[j] - top);
        sum += scratch[j];
    }
    for (int d = 0; d < HEAD_DIM; d++) {
        double accumulator = 0.0;
        for (uint32_t j = 0; j < sequence; j++)
            accumulator += scratch[j] * (double)bf16_to_f32(
                value[(size_t)j * stride + (size_t)head * HEAD_DIM + (size_t)d]);
        out[d] = accumulator / sum;
    }
}

/* One attention encode at the given query block size, read back into buffer. */
static int encode(h3_gpu *gpu, uint32_t block, h3_gpu_tensor *query,
                  const h3_gpu_tensor *key, const h3_gpu_tensor *value,
                  uint32_t sequence, uint32_t heads, uint16_t *buffer) {
    char text[32];
    snprintf(text, sizeof(text), "%u", block);
    setenv("H3_SDPA_QUERY_BLOCK", text, 1);
    const size_t count = (size_t)sequence * heads * HEAD_DIM;
    const float scale = (float)(1.0 / sqrt((double)HEAD_DIM));
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, count);
    int ok = output && h3_gpu_begin(gpu) &&
             h3_gpu_sdpa_bf16(gpu, output, query, key, value, sequence, heads,
                              HEAD_DIM, scale) &&
             h3_gpu_submit(gpu);
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: block %u: %s\n",
                block, h3_gpu_error(gpu));
        h3_gpu_tensor_free(output);
        return 0;
    }
    h3_gpu_tensor_read_bf16(output, buffer, count);
    h3_gpu_tensor_free(output);
    return 1;
}

/* The boundaries the sampled rows straddle. Picked so that 128, 256, 384 and
 * 512 each own some of them, and so that 33,280 - where the 49-row tail block
 * of a 256 run starts - is in. */
static const uint32_t boundaries[] = {
    128, 256, 384, 512, 16128, 16384, 32256, 32768, 33280};

static size_t add_row(uint32_t *rows, size_t used, size_t capacity,
                      uint32_t row) {
    if (used >= capacity) return used;
    for (size_t i = 0; i < used; i++) if (rows[i] == row) return used;
    rows[used] = row;
    return used + 1;
}

/* Sixty-four query rows, chosen and not drawn at random.
 *
 * A full float64 reference at this N is not a memory problem to optimise, it is
 * an impossibility: the score matrix alone is 33329^2 * 56 * 8 bytes, 497 GB.
 * Row independence (see oracle_row) means a subset is still exact, so the
 * question is only which subset earns its place.
 *
 * The only arithmetic query blocking adds is one byte offset per block,
 * start * heads * head_dim * item. The one bug it can hide is therefore an
 * off-by-one at a block edge, and the rows that expose it are the last row of a
 * block and the first row of the next - a row shifted by one lands on a
 * neighbouring row's answer, which is a total mismatch, not a rounding one. So
 * the sample is weighted onto boundaries: both rows of each boundary above,
 * plus the first and last two rows of the sequence. The remainder is spread
 * evenly through block interiors as a control, and the two groups are scored
 * separately: boundary rows agreeing worse than interior rows is the finding
 * this test exists to make. Stride 775 is coprime with 33,329, so the fill
 * never stalls on duplicates. */
static size_t sample_rows(uint32_t *rows, size_t capacity) {
    size_t used = 0;
    used = add_row(rows, used, capacity, 0);
    used = add_row(rows, used, capacity, 1);
    used = add_row(rows, used, capacity, (uint32_t)TARGET_SEQUENCE - 2);
    used = add_row(rows, used, capacity, (uint32_t)TARGET_SEQUENCE - 1);
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); i++) {
        used = add_row(rows, used, capacity, boundaries[i] - 1);
        used = add_row(rows, used, capacity, boundaries[i]);
    }
    for (uint32_t row = 97; used < capacity; row += 775)
        used = add_row(rows, used, capacity, row % (uint32_t)TARGET_SEQUENCE);
    return used;
}

/* The production shape. Not run by `make test`: H3_SDPA_TARGET=1 asks for it. */
static void target_case(h3_gpu *gpu) {
    const uint32_t sequence = TARGET_SEQUENCE, heads = TARGET_HEADS;
    const size_t count = (size_t)sequence * heads * HEAD_DIM;
    const uint32_t sizes[] = {128, 256, 384, 512};
    const size_t variants = sizeof(sizes) / sizeof(*sizes);
    clock_t started = clock();

    uint32_t head_list[TARGET_HEADS];
    size_t head_count = 0;
    for (uint32_t head = 0; head < heads; head += TARGET_HEAD_STRIDE)
        head_list[head_count++] = head;
    uint32_t rows[TARGET_PROBES];
    const size_t probes = sample_rows(rows, TARGET_PROBES);

    /* Five host buffers of `count` (q, k, v, baseline, current) and four on the
     * device (q, k, v, one output at a time), plus MPSGraph's own score tensor,
     * block * sequence * heads * 2 bytes - 1.78 GiB at block 512, the largest
     * single allocation of the run. */
    const double slot = (double)(count * sizeof(uint16_t)) /
                        (1024.0 * 1024.0 * 1024.0);
    printf("\ntarget N=%u heads=%u head_dim=%d bf16 non-causal: host %.2f GiB, "
           "device %.2f GiB, plus up to %.2f GiB of scores\n", sequence, heads,
           HEAD_DIM, 5.0 * slot, 4.0 * slot,
           (double)((size_t)sizes[variants - 1] * sequence * heads *
                    sizeof(uint16_t)) / (1024.0 * 1024.0 * 1024.0));

    uint16_t *host = malloc(count * 3 * sizeof(*host));
    uint16_t *baseline = malloc(count * sizeof(*baseline));
    uint16_t *current = malloc(count * sizeof(*current));
    double *scratch = malloc((size_t)sequence * sizeof(*scratch));
    double *reference = malloc(probes * head_count * HEAD_DIM *
                               sizeof(*reference));
    if (!host || !baseline || !current || !scratch || !reference) {
        fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: out of memory at the "
                "target shape\n");
        failures++;
        goto done;
    }
    for (size_t i = 0; i < count * 3; i++) host[i] = next_bf16();
    const uint16_t *q_host = host, *k_host = host + count,
                   *v_host = host + count * 2;

    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, q_host, count);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, k_host, count);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, v_host, count);
    if (!query || !key || !value) {
        fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: %s\n",
                h3_gpu_error(gpu));
        failures++;
        goto done;
    }

    /* The oracle depends on the inputs, not on the block size, so it is built
     * once and every variant is scored against it. */
    clock_t oracle_started = clock();
    for (size_t p = 0; p < probes; p++)
        for (size_t h = 0; h < head_count; h++)
            oracle_row(q_host, k_host, v_host, sequence, heads, rows[p],
                       head_list[h], reference + (p * head_count + h) * HEAD_DIM,
                       scratch);
    printf("float64 oracle: %zu rows x %zu heads in %.1f s\n", probes,
           head_count, seconds_since(oracle_started));

    for (size_t i = 0; i < variants; i++) {
        uint16_t *buffer = i ? current : baseline;
        clock_t encode_started = clock();
        if (!encode(gpu, sizes[i], query, key, value, sequence, heads, buffer)) {
            failures++;
            goto done;
        }
        double encode_seconds = seconds_since(encode_started);

        /* Block-size invariance needs no reference at all, and is the strongest
         * check available at this N. */
        size_t differing = 0;
        if (i) {
            for (size_t e = 0; e < count; e++)
                if (current[e] != baseline[e]) differing++;
            require(differing == 0, "query block size changed the result at the "
                                    "target N");
        }

        double edge_gap = 0.0, edge_norm = 0.0;
        double inner_gap = 0.0, inner_norm = 0.0;
        size_t edge_rows = 0;
        for (size_t p = 0; p < probes; p++) {
            int edge = rows[p] % sizes[i] == 0 ||
                       rows[p] % sizes[i] == sizes[i] - 1;
            edge_rows += (size_t)(edge != 0);
            for (size_t h = 0; h < head_count; h++)
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    double want = reference[(p * head_count + h) * HEAD_DIM + d];
                    size_t index = (size_t)rows[p] * heads * HEAD_DIM +
                                   (size_t)head_list[h] * HEAD_DIM + d;
                    double gap = (double)bf16_to_f32(buffer[index]) - want;
                    if (edge) { edge_gap += gap * gap; edge_norm += want * want; }
                    else { inner_gap += gap * gap; inner_norm += want * want; }
                }
        }
        double overall = sqrt((edge_gap + inner_gap) / (edge_norm + inner_norm));
        double at_edge = edge_norm > 0.0 ? sqrt(edge_gap / edge_norm) : 0.0;
        double inside = inner_norm > 0.0 ? sqrt(inner_gap / inner_norm) : 0.0;

        printf("block %-4u: rel-L2 %.4g  boundary rows %.4g (%zu)  "
               "interior %.4g (%zu)  differing %zu  encode %.1f s\n",
               sizes[i], overall, at_edge, edge_rows, inside,
               probes - edge_rows, differing, encode_seconds);

        /* 2.1e-3 is the bf16 floor; 3.0e-2 is the degraded whole-sequence
         * kernel. Landing near the latter would mean the collapse reaches the
         * blocked path too, which is a finding, not a test failure line. */
        if (overall > 1e-2)
            fprintf(stderr, "\n*** tests/test_sdpa_blocking.c: rel-L2 %.4g at "
                    "N=%u block %u is the degraded-kernel magnitude, not the "
                    "bf16 floor. The accuracy collapse documented above "
                    "N~9,100 reaches the BLOCKED path at this N. ***\n\n",
                    overall, sequence, sizes[i]);
        require(overall < 5e-3, "blocked SDPA exceeds the bf16 error floor at "
                                "the target N");
        /* An off-by-one in the block offset puts a neighbouring row's answer in
         * a boundary row: rel-L2 near 1.4, not a factor of three. The gate is
         * loose on purpose and still catches that. */
        require(edge_norm == 0.0 || inner_norm == 0.0 ||
                at_edge < 3.0 * inside,
                "rows next to a block boundary agree worse than block "
                "interiors: the block offset arithmetic is wrong");
    }
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    printf("target case: %.1f s\n", seconds_since(started));
done:
    free(reference);
    free(scratch);
    free(current);
    free(baseline);
    free(host);
}

int main(void) {
    const size_t count = (size_t)SEQUENCE * HEADS * HEAD_DIM;
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: %s\n", error);
        return 1;
    }
    uint16_t *host = malloc(count * 3 * sizeof(*host));
    if (!host) return 1;
    for (size_t i = 0; i < count * 3; i++) host[i] = next_bf16();
    const uint16_t *q_host = host, *k_host = host + count,
                   *v_host = host + count * 2;

    h3_gpu_tensor *query = h3_gpu_tensor_from_bf16(gpu, q_host, count);
    h3_gpu_tensor *key = h3_gpu_tensor_from_bf16(gpu, k_host, count);
    h3_gpu_tensor *value = h3_gpu_tensor_from_bf16(gpu, v_host, count);

    /* 0 is the whole-sequence encode; the rest divide SEQUENCE unevenly on
     * purpose, so every one of them ends on a short tail block. */
    const uint32_t sizes[] = {0, 128, 256, 384, 512};
    const size_t variants = sizeof(sizes) / sizeof(*sizes);
    uint16_t **results = calloc(variants, sizeof(*results));
    for (size_t i = 0; i < variants; i++) {
        results[i] = malloc(count * sizeof(**results));
        if (!results[i] ||
            !encode(gpu, sizes[i], query, key, value, SEQUENCE, HEADS,
                    results[i]))
            return 1;
    }

    /* Where the block boundaries fall must not change a single bit. */
    for (size_t i = 2; i < variants; i++) {
        size_t differing = 0;
        for (size_t e = 0; e < count; e++)
            if (results[i][e] != results[1][e]) differing++;
        printf("blocked %-4u vs blocked %-4u: %zu of %zu elements differ\n",
               sizes[i], sizes[1], differing, count);
        require(differing == 0, "query block size changed the result");
    }

    /* And every variant, blocked or not, must sit at the bf16 error floor.
     * The oracle depends on the probe, not on the variant, so it is computed
     * once and every variant is scored against it. */
    const uint32_t rows[PROBES] = {0, 1, 127, 128, SEQUENCE / 2, SEQUENCE - 1};
    double *scratch = malloc(SEQUENCE * sizeof(*scratch));
    double reference[HEAD_DIM], numerator[sizeof(sizes) / sizeof(*sizes)] = {0};
    double denominator = 0.0;
    for (uint32_t head = 0; head < HEADS; head += 3)
        for (size_t p = 0; p < PROBES; p++) {
            oracle_row(q_host, k_host, v_host, SEQUENCE, HEADS, rows[p], head,
                       reference, scratch);
            for (int d = 0; d < HEAD_DIM; d++) {
                size_t index = (size_t)rows[p] * HEADS * HEAD_DIM +
                               (size_t)head * HEAD_DIM + (size_t)d;
                denominator += reference[d] * reference[d];
                for (size_t i = 0; i < variants; i++) {
                    double gap = (double)bf16_to_f32(results[i][index]) -
                                 reference[d];
                    numerator[i] += gap * gap;
                }
            }
        }
    for (size_t i = 0; i < variants; i++) {
        double relative = sqrt(numerator[i] / denominator);
        printf("block %-4u vs float64 oracle: rel-L2 %.4g\n", sizes[i],
               relative);
        require(relative < 5e-3, "blocked SDPA exceeds the bf16 error floor");
    }

    free(scratch);
    for (size_t i = 0; i < variants; i++) free(results[i]);
    free(results);
    free(host);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);

    const char *want_target = getenv("H3_SDPA_TARGET");
    if (want_target && *want_target && strcmp(want_target, "0"))
        target_case(gpu);
    else
        puts("skip: target N=33329 shape, set H3_SDPA_TARGET=1 (~6 GiB)");

    h3_gpu_free(gpu);
    if (failures) return 1;
    puts("ok: query-blocked SDPA is block-size invariant and matches float64");
    return 0;
}
