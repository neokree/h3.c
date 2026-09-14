/* Query-blocked SDPA: is a block's result the same answer, wherever the block
 * boundaries fall, and is it the right answer?
 *
 * Both questions are asked against a double-precision CPU oracle rather than
 * against the whole-sequence encode, because the whole-sequence encode is not
 * a reference: above roughly 9,000 tokens at 56 heads MPSGraph switches
 * scaledDotProductAttention to a kernel whose error is an order of magnitude
 * worse (see docs/720p-blocked-attention.md). Synthesised inputs only, so this
 * never skips. */
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEQUENCE = 1000, HEADS = 8, HEAD_DIM = 128, PROBES = 6 };

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

/* One (query row, head) pair of the attention, in double precision. */
static void oracle_row(const uint16_t *query, const uint16_t *key,
                       const uint16_t *value, uint32_t row, uint32_t head,
                       double *out, double *scratch) {
    const size_t stride = (size_t)HEADS * HEAD_DIM;
    const uint16_t *q = query + (size_t)row * stride + (size_t)head * HEAD_DIM;
    const double scale = 1.0 / sqrt((double)HEAD_DIM);
    double top = -INFINITY, sum = 0.0;
    for (uint32_t j = 0; j < SEQUENCE; j++) {
        const uint16_t *k = key + (size_t)j * stride + (size_t)head * HEAD_DIM;
        double accumulator = 0.0;
        for (int d = 0; d < HEAD_DIM; d++)
            accumulator += (double)bf16_to_f32(q[d]) * (double)bf16_to_f32(k[d]);
        scratch[j] = accumulator * scale;
        if (scratch[j] > top) top = scratch[j];
    }
    for (uint32_t j = 0; j < SEQUENCE; j++) {
        scratch[j] = exp(scratch[j] - top);
        sum += scratch[j];
    }
    for (int d = 0; d < HEAD_DIM; d++) {
        double accumulator = 0.0;
        for (uint32_t j = 0; j < SEQUENCE; j++)
            accumulator += scratch[j] * (double)bf16_to_f32(
                value[(size_t)j * stride + (size_t)head * HEAD_DIM + (size_t)d]);
        out[d] = accumulator / sum;
    }
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
    const float scale = (float)(1.0 / sqrt((double)HEAD_DIM));

    /* 0 is the whole-sequence encode; the rest divide SEQUENCE unevenly on
     * purpose, so every one of them ends on a short tail block. */
    const uint32_t sizes[] = {0, 128, 256, 384, 512};
    const size_t variants = sizeof(sizes) / sizeof(*sizes);
    uint16_t **results = calloc(variants, sizeof(*results));
    for (size_t i = 0; i < variants; i++) {
        char text[32];
        snprintf(text, sizeof(text), "%u", sizes[i]);
        setenv("H3_SDPA_QUERY_BLOCK", text, 1);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, count);
        int ok = h3_gpu_begin(gpu) &&
                 h3_gpu_sdpa_bf16(gpu, output, query, key, value, SEQUENCE,
                                  HEADS, HEAD_DIM, scale) &&
                 h3_gpu_submit(gpu);
        if (!ok) {
            fprintf(stderr, "FAIL tests/test_sdpa_blocking.c: block %u: %s\n",
                    sizes[i], h3_gpu_error(gpu));
            return 1;
        }
        results[i] = malloc(count * sizeof(**results));
        h3_gpu_tensor_read_bf16(output, results[i], count);
        h3_gpu_tensor_free(output);
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
            oracle_row(q_host, k_host, v_host, rows[p], head, reference,
                       scratch);
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
    h3_gpu_free(gpu);
    if (failures) return 1;
    puts("ok: query-blocked SDPA is block-size invariant and matches float64");
    return 0;
}
