/* What the LoRA branch's thin GEMMs cost on this GPU, and whether zero-padding
 * the rank to 256 to reach the MPS fast path pays for itself.
 *
 * h3_gpu_linear_bf16 routes to MPS only when rows >= 32 && input_dim >= 256 &&
 * output_dim >= 256 (h3_gpu.m:2517). In y = Wx + s*B(Ax) the A half has
 * output_dim = rank and the B half has input_dim = rank, so at any real rank
 * (16..128) both halves miss that gate and land on the hand-written 16x16 tile
 * kernel h3_linear_bf16. Padding the rank out to 256 with zero rows in A and
 * zero columns in B is arithmetically a no-op and moves both halves onto MPS.
 * Whether that is a win is the question here.
 *
 * The shapes are real: the four projections come from the checkpoint's own
 * tensor headers and the rank from the LoRA file's own A/B pair. The values are
 * random, because the cost of a BF16 GEMM does not depend on them, and reading
 * a 231 MB fc1 out of a 62 GiB checkpoint to multiply it by noise would only
 * buy a slower bench.
 *
 * Which path each measurement actually took is not assumed: h3_gpu_stats
 * already separates mps_linear_dispatches from direct_dispatches, and every row
 * prints the per-iteration delta of both. A row that claims MPS and shows
 * direct=1 is a bug in the bench, not a number to quote.
 *
 * usage: h3_lora_gemm_bench [LORA] [MODEL_ROOT] [BLOCK]
 */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ROWS_ORACLE is what tests/test_lora_oracle.c uses. It is far too small to say
 * anything about the DiT: a 448x576 / 56-frame run is latent 28x36, patched 2x2
 * to 14x18 = 252 tokens per latent frame, and h3_video_latent_t(56) = 17 latent
 * frames, so the video stream alone is 4284 rows before text and audio. Both are
 * measured; only the second is a claim about a real run. */
enum { ROWS_ORACLE = 4, ROWS_REAL = 4284 };

static const char *const PROJECTIONS[] = {
    "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"
};

/* The real rank sits inside this sweep, so the padded case and the rank-256
 * sweep point are literally the same measurement. */
static const uint32_t RANKS[] = {16, 32, 64, 128, 256};

enum { WARMUP = 3, MAX_ITERS = 64 };
static const double BUDGET_MS = 400.0;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/bench_lora_gemm.c: %s\n", message);
    exit(1);
}

static void *checked_malloc(size_t bytes, const char *what) {
    void *block = malloc(bytes);
    if (!block) {
        fprintf(stderr, "FAIL tests/bench_lora_gemm.c: out of memory for %s\n",
                what);
        exit(1);
    }
    return block;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t value = (*state += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

/* Small centred values in BF16. Magnitude is irrelevant to timing; keeping it
 * near 1 only stops the equivalence check below from drowning in overflow. */
static uint16_t random_bf16(uint64_t *state) {
    uint32_t bits = (uint32_t)(next_random(state) >> 40);
    float value = ((float)bits / 8388608.0f - 1.0f) * 0.05f;
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    return (uint16_t)((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}

static h3_gpu_tensor *random_tensor(h3_gpu *gpu, size_t count, uint64_t seed) {
    uint16_t *values = checked_malloc(count * sizeof(*values), "random tensor");
    for (size_t index = 0; index < count; index++)
        values[index] = random_bf16(&seed);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, count);
    free(values);
    if (!tensor) die("cannot upload a tensor");
    return tensor;
}

/* A [rank, input_dim] zero-padded to [pad, input_dim]: the extra rows are zero,
 * so hidden rows beyond rank are zero and contribute nothing to B(Ax). */
static h3_gpu_tensor *padded_a(h3_gpu *gpu, const uint16_t *a, uint32_t rank,
                               uint32_t input_dim, uint32_t pad) {
    size_t count = (size_t)pad * input_dim;
    uint16_t *values = checked_malloc(count * sizeof(*values), "padded A");
    memset(values, 0, count * sizeof(*values));
    memcpy(values, a, (size_t)rank * input_dim * sizeof(*values));
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, count);
    free(values);
    if (!tensor) die("cannot upload padded A");
    return tensor;
}

/* B [output_dim, rank] zero-padded to [output_dim, pad]: row-major, so each row
 * has to be copied and the tail of each row zeroed. */
static h3_gpu_tensor *padded_b(h3_gpu *gpu, const uint16_t *b,
                               uint32_t output_dim, uint32_t rank,
                               uint32_t pad) {
    size_t count = (size_t)output_dim * pad;
    uint16_t *values = checked_malloc(count * sizeof(*values), "padded B");
    memset(values, 0, count * sizeof(*values));
    for (uint32_t row = 0; row < output_dim; row++)
        memcpy(values + (size_t)row * pad, b + (size_t)row * rank,
               rank * sizeof(*values));
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, count);
    free(values);
    if (!tensor) die("cannot upload padded B");
    return tensor;
}

typedef enum { OP_LINEAR, OP_ADD, OP_EMPTY } op_kind;

typedef struct {
    h3_gpu *gpu;
    op_kind kind;
    h3_gpu_tensor *out;
    const h3_gpu_tensor *left;   /* linear: input   add: left  */
    const h3_gpu_tensor *right;  /* linear: weight  add: right */
    uint32_t rows, input_dim, output_dim, elements;
} bench_op;

static int encode(const bench_op *op) {
    switch (op->kind) {
    case OP_LINEAR:
        return h3_gpu_linear_bf16(op->gpu, op->out, op->left, op->right, NULL,
                                  op->rows, op->input_dim, op->output_dim);
    case OP_ADD:
        return h3_gpu_add_bf16(op->gpu, op->out, op->left, op->right,
                               op->elements);
    case OP_EMPTY:
        return 1;
    }
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : (a > b ? 1 : 0);
}

/* One measurement. h3_gpu_submit ends in waitUntilCompleted on every inflight
 * buffer, MPSGraph's child buffers included, so the wall clock across
 * begin/encode/submit is completion time and not encode time. */
static void bench(const char *label, const char *shape, const bench_op *op) {
    h3_gpu_stats before, after;
    double samples[MAX_ITERS];

    for (int iteration = 0; iteration < WARMUP; iteration++)
        if (!h3_gpu_begin(op->gpu) || !encode(op) || !h3_gpu_submit(op->gpu))
            die(h3_gpu_error(op->gpu));

    if (!h3_gpu_get_stats(op->gpu, &before)) die("cannot read GPU stats");
    int taken = 0;
    double total = 0.0;
    while (taken < MAX_ITERS) {
        double started = now_ms();
        if (!h3_gpu_begin(op->gpu) || !encode(op) || !h3_gpu_submit(op->gpu))
            die(h3_gpu_error(op->gpu));
        samples[taken] = now_ms() - started;
        total += samples[taken++];
        if (taken >= 5 && total > BUDGET_MS) break;
    }
    if (!h3_gpu_get_stats(op->gpu, &after)) die("cannot read GPU stats");

    qsort(samples, (size_t)taken, sizeof(*samples), compare_double);
    double median = samples[taken / 2];

    unsigned long long mps = (unsigned long long)
        (after.mps_linear_dispatches - before.mps_linear_dispatches);
    unsigned long long direct = (unsigned long long)
        (after.direct_dispatches - before.direct_dispatches);
    printf("  %-22s %-22s %8.3f ms  n=%2d  mps=%llu direct=%llu\n",
           label, shape, median, taken, mps / (unsigned)taken,
           direct / (unsigned)taken);
}

/* delta = B(Ax), the whole LoRA branch minus the base GEMM and the add. Used
 * once to prove the padding is arithmetically inert before any of its timings
 * are quoted. */
static void run_delta(h3_gpu *gpu, const h3_gpu_tensor *input,
                      const h3_gpu_tensor *a, const h3_gpu_tensor *b,
                      uint32_t rows, uint32_t input_dim, uint32_t rank,
                      uint32_t output_dim, float *out) {
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * rank);
    h3_gpu_tensor *delta = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)rows * output_dim);
    h3_gpu_tensor *delta_f32 = h3_gpu_tensor_new_f32(gpu,
                                                     (size_t)rows * output_dim);
    if (!hidden || !delta || !delta_f32) die("cannot allocate delta scratch");
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_linear_bf16(gpu, hidden, input, a, NULL, rows, input_dim,
                            rank) ||
        !h3_gpu_linear_bf16(gpu, delta, hidden, b, NULL, rows, rank,
                            output_dim) ||
        !h3_gpu_cast_bf16_to_f32(gpu, delta_f32, delta,
                                 rows * output_dim) ||
        !h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));
    if (!h3_gpu_tensor_read_f32(delta_f32, out, (size_t)rows * output_dim))
        die("cannot read delta");
    h3_gpu_tensor_free(delta_f32);
    h3_gpu_tensor_free(delta);
    h3_gpu_tensor_free(hidden);
}

int main(int argc, char **argv) {
    const char *lora_path = argc > 1 ? argv[1] : "loras/turbo.safetensors";
    const char *model_root = argc > 2 ? argv[2] : "MiniMax-H3";
    int block = argc > 3 ? atoi(argv[3]) : 0;

    char error[512];
    char weight_path[1024];
    snprintf(weight_path, sizeof(weight_path), "%s/FL2VA/transformer",
             model_root);
    h3_weight_store *weights = h3_weight_store_open(weight_path, error,
                                                    sizeof(error));
    if (!weights) die(error);
    h3_st_header lora;
    if (!h3_st_read_header(lora_path, &lora, error, sizeof(error))) die(error);
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    /* The real rank, from the LoRA's own A tensor, not from a constant here. */
    uint32_t real_rank = 0;
    {
        char name[256];
        snprintf(name, sizeof(name),
                 "diffusion_model.blocks.%d.attn.qkv_proj.lora_A.weight", block);
        const h3_st_tensor *a = h3_st_find(&lora, name);
        if (!a) {
            snprintf(name, sizeof(name),
                     "blocks.%d.attn.qkv_proj.lora_A.weight", block);
            a = h3_st_find(&lora, name);
        }
        if (!a || a->ndim != 2) die("cannot find a LoRA A tensor for the rank");
        real_rank = (uint32_t)a->shape[0];
    }

    size_t ranks = sizeof(RANKS) / sizeof(*RANKS);
    const uint32_t row_counts[] = {ROWS_ORACLE, ROWS_REAL};

    printf("bench: block %d, real rank %u, MPS gate is "
           "rows>=32 && input_dim>=256 && output_dim>=256\n", block, real_rank);
    printf("shapes from the checkpoint, values random; median of n timed "
           "begin/encode/submit round trips\n\n");

    {
        bench_op empty = {gpu, OP_EMPTY, NULL, NULL, NULL, 0, 0, 0, 0};
        bench("SUBMIT_FLOOR", "empty command buffer", &empty);
        putchar('\n');
    }

    size_t projections = sizeof(PROJECTIONS) / sizeof(*PROJECTIONS);
    for (size_t which = 0; which < projections; which++) {
        const char *projection = PROJECTIONS[which];
        char weight_name[256];
        snprintf(weight_name, sizeof(weight_name), "blocks.%d.%s.weight",
                 block, projection);
        const h3_st_header *shard = NULL;
        const h3_st_tensor *w = h3_weight_find(weights, weight_name, &shard);
        if (!w || !shard || w->ndim != 2)
            die("checkpoint is missing a target tensor");
        uint32_t output_dim = (uint32_t)w->shape[0];
        uint32_t input_dim = (uint32_t)w->shape[1];

        h3_gpu_tensor *weight = random_tensor(gpu,
                                              (size_t)output_dim * input_dim,
                                              0x5eed0000ull + which);

        for (size_t r = 0; r < sizeof(row_counts) / sizeof(*row_counts); r++) {
            uint32_t rows = row_counts[r];
            char shape[64];
            printf("%s [%u,%u]  rows=%u%s\n", projection, output_dim, input_dim,
                   rows, rows == ROWS_ORACLE ? "  (oracle's ROWS)" :
                                               "  (448x576 / 56 frames)");

            h3_gpu_tensor *input = random_tensor(gpu,
                                                 (size_t)rows * input_dim,
                                                 0xa11ce000ull + which + r);
            h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu,
                                                    (size_t)rows * output_dim);
            h3_gpu_tensor *out2 = h3_gpu_tensor_new_bf16(gpu,
                                                    (size_t)rows * output_dim);
            if (!out || !out2) die("cannot allocate output");

            snprintf(shape, sizeof(shape), "[%u,%u]x[%u]", output_dim,
                     input_dim, rows);
            bench_op base = {gpu, OP_LINEAR, out, input, weight, rows,
                             input_dim, output_dim, 0};
            bench("BASE  W.x", shape, &base);

            for (size_t k = 0; k < ranks; k++) {
                uint32_t rank = RANKS[k];
                h3_gpu_tensor *a = random_tensor(gpu,
                                                 (size_t)rank * input_dim,
                                                 0xaaa0000ull + k);
                h3_gpu_tensor *b = random_tensor(gpu,
                                                 (size_t)output_dim * rank,
                                                 0xbbb0000ull + k);
                h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu,
                                                       (size_t)rows * rank);
                if (!hidden) die("cannot allocate hidden");

                char label[64];
                snprintf(label, sizeof(label), "A_HALF rank %-3u%s", rank,
                         rank == real_rank ? "*" : "");
                snprintf(shape, sizeof(shape), "[%u,%u]x[%u]", rank, input_dim,
                         rows);
                bench_op a_half = {gpu, OP_LINEAR, hidden, input, a, rows,
                                   input_dim, rank, 0};
                bench(label, shape, &a_half);

                snprintf(label, sizeof(label), "B_HALF rank %-3u%s", rank,
                         rank == real_rank ? "*" : "");
                snprintf(shape, sizeof(shape), "[%u,%u]x[%u]", output_dim, rank,
                         rows);
                bench_op b_half = {gpu, OP_LINEAR, out2, hidden, b, rows, rank,
                                   output_dim, 0};
                bench(label, shape, &b_half);

                h3_gpu_tensor_free(hidden);
                h3_gpu_tensor_free(b);
                h3_gpu_tensor_free(a);
            }

            snprintf(shape, sizeof(shape), "%u elements",
                     (unsigned)((size_t)rows * output_dim));
            bench_op add = {gpu, OP_ADD, out, out, out2, 0, 0, 0,
                            (uint32_t)((size_t)rows * output_dim)};
            bench("ADD   y+delta", shape, &add);

            h3_gpu_tensor_free(out2);
            h3_gpu_tensor_free(out);
            h3_gpu_tensor_free(input);
            putchar('\n');
        }

        /* The check the timings rest on: at the oracle's row count, the padded
         * branch must compute the same delta as the unpadded one. If it does
         * not, the padded timings are timing a different function. */
        {
            uint32_t rows = ROWS_ORACLE, pad = 256;
            size_t count = (size_t)rows * output_dim;
            uint64_t seed_a = 0xc0ffee00ull + which;
            uint64_t seed_b = 0xdecade00ull + which;
            size_t a_count = (size_t)real_rank * input_dim;
            size_t b_count = (size_t)output_dim * real_rank;
            uint16_t *a_host = checked_malloc(a_count * sizeof(*a_host), "A");
            uint16_t *b_host = checked_malloc(b_count * sizeof(*b_host), "B");
            for (size_t i = 0; i < a_count; i++) a_host[i] = random_bf16(&seed_a);
            for (size_t i = 0; i < b_count; i++) b_host[i] = random_bf16(&seed_b);

            h3_gpu_tensor *input = random_tensor(gpu, (size_t)rows * input_dim,
                                                 0x1dea0000ull + which);
            h3_gpu_tensor *a = h3_gpu_tensor_from_bf16(gpu, a_host, a_count);
            h3_gpu_tensor *b = h3_gpu_tensor_from_bf16(gpu, b_host, b_count);
            h3_gpu_tensor *ap = padded_a(gpu, a_host, real_rank, input_dim, pad);
            h3_gpu_tensor *bp = padded_b(gpu, b_host, output_dim, real_rank, pad);
            if (!a || !b) die("cannot upload the unpadded pair");

            float *plain = checked_malloc(count * sizeof(*plain), "plain delta");
            float *padded = checked_malloc(count * sizeof(*padded), "padded");
            run_delta(gpu, input, a, b, rows, input_dim, real_rank, output_dim,
                      plain);
            run_delta(gpu, input, ap, bp, rows, input_dim, pad, output_dim,
                      padded);

            double worst = 0.0, scale = 0.0;
            for (size_t i = 0; i < count; i++) {
                double d = fabs((double)padded[i] - (double)plain[i]);
                if (d > worst) worst = d;
                if (fabs((double)plain[i]) > scale) scale = fabs((double)plain[i]);
            }
            printf("%s padding check: rank %u vs %u zero-padded, "
                   "max abs diff %.3e (delta max %.3e)\n\n", projection,
                   real_rank, pad, worst, scale);

            free(padded);
            free(plain);
            h3_gpu_tensor_free(bp);
            h3_gpu_tensor_free(ap);
            h3_gpu_tensor_free(b);
            h3_gpu_tensor_free(a);
            h3_gpu_tensor_free(input);
            free(b_host);
            free(a_host);
        }

        h3_gpu_tensor_free(weight);
    }

    h3_gpu_free(gpu);
    h3_st_free_header(&lora);
    h3_weight_store_free(weights);
    return 0;
}
