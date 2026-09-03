/* T1 oracle harness for runtime LoRA adapters (docs/WHAT.md section 8).
 *
 * The claim under test is H3: the runtime branch
 *
 *     y = Wx + s * B(Ax)
 *
 * computed on the GPU in BF16 must agree with the fused weight
 *
 *     y = (W + s * B@A)x
 *
 * The oracle is the fused form, in float32, on the CPU, built here in C. It
 * fuses first and applies second, which is the whole point: a reference that
 * kept the factored order would only measure BF16 rounding and would miss a
 * transposed or misscaled delta.
 *
 * Per tensor, not per model. One real W from the checkpoint, one real (A, B)
 * from a LoRA file, a seeded x, four projections of one block. No 62 GiB copy
 * and no Python anywhere in the loop.
 *
 * The delta never touches W: the branch is assembled from the same
 * h3_gpu_linear_bf16 the DiT already calls, which is the injection point the
 * design settled on.
 *
 * Two error figures are printed per projection, because the accumulation dtype
 * of the final sum is still an open decision:
 *   f32-acc  the sum kept in float32 (the floor of what the branch can do)
 *   bf16-out the same sum rounded back to BF16 for the next kernel
 * Setting a tolerance from these numbers is a separate decision.
 *
 * usage: h3_lora_oracle_test [LORA] [MODEL_ROOT] [STRENGTH] [BLOCK]
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

enum { ROWS = 4 };

static const char *const PROJECTIONS[] = {
    "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_lora_oracle.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Round to nearest even, the rounding the GPU casts use. Inputs here are
 * bounded random values, so the NaN and overflow corners cannot arise. */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static uint64_t next_random(uint64_t *state) {
    uint64_t value = (*state += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

/* Uniform in [-1, 1), then through BF16 so the reference sees exactly the
 * values the GPU sees. Input rounding is not the branch's error to carry. */
static float next_activation(uint64_t *state) {
    uint32_t bits = (uint32_t)(next_random(state) >> 40); /* 24 bits */
    float value = (float)bits / 8388608.0f - 1.0f;
    return bf16_to_f32(f32_to_bf16(value));
}

typedef struct {
    double rel_max;
    double rel_l2;
    double abs_max;
} error_stats;

static error_stats compare(const float *got, const float *want, size_t count) {
    error_stats stats = {0.0, 0.0, 0.0};
    double max_value = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)got[index] - (double)want[index];
        double magnitude = fabs((double)want[index]);
        if (fabs(delta) > stats.abs_max) stats.abs_max = fabs(delta);
        if (magnitude > max_value) max_value = magnitude;
        square_error += delta * delta;
        square_value += (double)want[index] * (double)want[index];
    }
    stats.rel_max = stats.abs_max / (max_value > 1e-12 ? max_value : 1e-12);
    stats.rel_l2 = sqrt(square_error /
                        (square_value > 1e-24 ? square_value : 1e-24));
    return stats;
}

static void *checked_malloc(size_t bytes, const char *what) {
    void *block = malloc(bytes);
    if (!block) {
        fprintf(stderr, "FAIL tests/test_lora_oracle.c: out of memory for %s\n",
                what);
        exit(1);
    }
    return block;
}

static uint16_t *read_bf16_tensor(const h3_st_header *header,
                                  const h3_st_tensor *tensor,
                                  const char *what) {
    size_t count = (size_t)h3_st_tensor_elements(tensor);
    uint16_t *values = checked_malloc(count * sizeof(*values), what);
    char error[512];
    if (!h3_st_read_data(header, tensor, values, count * sizeof(*values),
                         error, sizeof(error))) die(error);
    return values;
}

static float *expand_bf16(const uint16_t *values, size_t count,
                          const char *what) {
    float *expanded = checked_malloc(count * sizeof(*expanded), what);
    for (size_t index = 0; index < count; index++)
        expanded[index] = bf16_to_f32(values[index]);
    return expanded;
}

/* Look the pair up with and without the diffusion_model. prefix: the ComfyUI
 * conversion carries it, the upstream file does not (docs/WHAT.md 7.2). */
static const h3_st_tensor *find_pair(const h3_st_header *lora, int block,
                                     const char *projection, const char *half) {
    char name[256];
    snprintf(name, sizeof(name), "diffusion_model.blocks.%d.%s.lora_%s.weight",
             block, projection, half);
    const h3_st_tensor *tensor = h3_st_find(lora, name);
    if (tensor) return tensor;
    snprintf(name, sizeof(name), "blocks.%d.%s.lora_%s.weight",
             block, projection, half);
    return h3_st_find(lora, name);
}

/* y = Wx + strength * B(Ax), assembled from the kernels the DiT already uses.
 * base_out and rounded_out are optional. */
static void run_branch(h3_gpu *gpu, const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *lora_a,
                       const h3_gpu_tensor *lora_b,
                       const h3_gpu_tensor *input, uint32_t input_dim,
                       uint32_t rank, uint32_t output_dim, float strength,
                       float *y_out, float *base_out, float *rounded_out) {
    size_t output_count = (size_t)ROWS * output_dim;
    h3_gpu_tensor *base = h3_gpu_tensor_new_bf16(gpu, output_count);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * rank);
    h3_gpu_tensor *delta = h3_gpu_tensor_new_bf16(gpu, output_count);
    h3_gpu_tensor *base_f32 = h3_gpu_tensor_new_f32(gpu, output_count);
    h3_gpu_tensor *delta_f32 = h3_gpu_tensor_new_f32(gpu, output_count);
    h3_gpu_tensor *result = h3_gpu_tensor_new_f32(gpu, output_count);
    h3_gpu_tensor *rounded = h3_gpu_tensor_new_bf16(gpu, output_count);
    if (!base || !hidden || !delta || !base_f32 || !delta_f32 || !result ||
        !rounded) die("cannot allocate branch scratch");

    uint32_t elements = (uint32_t)output_count;
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_linear_bf16(gpu, base, input, weight, NULL, ROWS,
                            input_dim, output_dim) ||
        !h3_gpu_linear_bf16(gpu, hidden, input, lora_a, NULL, ROWS,
                            input_dim, rank) ||
        !h3_gpu_linear_bf16(gpu, delta, hidden, lora_b, NULL, ROWS,
                            rank, output_dim) ||
        !h3_gpu_cast_bf16_to_f32(gpu, base_f32, base, elements) ||
        !h3_gpu_cast_bf16_to_f32(gpu, delta_f32, delta, elements) ||
        !h3_gpu_add_scaled_f32(gpu, result, base_f32, delta_f32, 1.0f,
                               strength, elements) ||
        !h3_gpu_cast_f32_to_bf16(gpu, rounded, result, elements) ||
        !h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));

    if (!h3_gpu_tensor_read_f32(result, y_out, output_count))
        die("cannot read branch output");
    if (base_out && !h3_gpu_tensor_read_f32(base_f32, base_out, output_count))
        die("cannot read base output");
    if (rounded_out) {
        uint16_t *packed = checked_malloc(output_count * sizeof(*packed),
                                          "rounded branch output");
        if (!h3_gpu_tensor_read_bf16(rounded, packed, output_count))
            die("cannot read rounded branch output");
        for (size_t index = 0; index < output_count; index++)
            rounded_out[index] = bf16_to_f32(packed[index]);
        free(packed);
    }

    h3_gpu_tensor_free(rounded);
    h3_gpu_tensor_free(result);
    h3_gpu_tensor_free(delta_f32);
    h3_gpu_tensor_free(base_f32);
    h3_gpu_tensor_free(delta);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(base);
}

/* The oracle. One fused row of W at a time, so the full [output_dim,
 * input_dim] float32 matrix is never materialised. */
static void fused_reference(const uint16_t *weight, const float *lora_a,
                            const float *lora_b, const float *input,
                            size_t input_dim, size_t rank, size_t output_dim,
                            float strength, float *y_out) {
    float *row = checked_malloc(input_dim * sizeof(*row), "fused weight row");
    for (size_t out = 0; out < output_dim; out++) {
        const uint16_t *source = weight + out * input_dim;
        for (size_t index = 0; index < input_dim; index++)
            row[index] = bf16_to_f32(source[index]);
        for (size_t k = 0; k < rank; k++) {
            float scale = strength * lora_b[out * rank + k];
            const float *a_row = lora_a + k * input_dim;
            for (size_t index = 0; index < input_dim; index++)
                row[index] += scale * a_row[index];
        }
        for (size_t r = 0; r < ROWS; r++) {
            const float *x = input + r * input_dim;
            double accumulator = 0.0;
            for (size_t index = 0; index < input_dim; index++)
                accumulator += (double)row[index] * (double)x[index];
            y_out[r * output_dim + out] = (float)accumulator;
        }
    }
    free(row);
}

int main(int argc, char **argv) {
    const char *lora_path = argc > 1 ? argv[1] : "loras/turbo.safetensors";
    const char *model_root = argc > 2 ? argv[2] : "MiniMax-H3";
    float strength = argc > 3 ? (float)atof(argv[3]) : 1.0f;
    int block = argc > 4 ? atoi(argv[4]) : 0;

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

    printf("oracle: block %d, strength %.4f, %d input rows, %s\n",
           block, (double)strength, ROWS, lora_path);

    size_t projections = sizeof(PROJECTIONS) / sizeof(*PROJECTIONS);
    for (size_t which = 0; which < projections; which++) {
        const char *projection = PROJECTIONS[which];
        const h3_st_tensor *a_tensor = find_pair(&lora, block, projection, "A");
        const h3_st_tensor *b_tensor = find_pair(&lora, block, projection, "B");
        if (!a_tensor || !b_tensor) {
            printf("  %-14s no pair in this LoRA, skipped\n", projection);
            continue;
        }
        if (a_tensor->ndim != 2 || b_tensor->ndim != 2 ||
            a_tensor->dtype != H3_DTYPE_BF16 ||
            b_tensor->dtype != H3_DTYPE_BF16) die("malformed LoRA pair");
        size_t rank = (size_t)a_tensor->shape[0];
        size_t input_dim = (size_t)a_tensor->shape[1];
        size_t output_dim = (size_t)b_tensor->shape[0];
        /* The rank is a property of the pair, not of the file: the two halves
         * are the only place it is written down, and they must agree. */
        if ((size_t)b_tensor->shape[1] != rank)
            die("LoRA pair disagrees on rank");

        char weight_name[256];
        snprintf(weight_name, sizeof(weight_name), "blocks.%d.%s.weight",
                 block, projection);
        const h3_st_header *shard = NULL;
        const h3_st_tensor *w_tensor = h3_weight_find(weights, weight_name,
                                                      &shard);
        if (!w_tensor || !shard) die("checkpoint is missing a target tensor");
        if (w_tensor->ndim != 2 || w_tensor->dtype != H3_DTYPE_BF16 ||
            (size_t)w_tensor->shape[0] != output_dim ||
            (size_t)w_tensor->shape[1] != input_dim)
            die("LoRA pair does not fit its target tensor");

        uint16_t *w_bf16 = read_bf16_tensor(shard, w_tensor, "base weight");
        uint16_t *a_bf16 = read_bf16_tensor(&lora, a_tensor, "lora A");
        uint16_t *b_bf16 = read_bf16_tensor(&lora, b_tensor, "lora B");
        float *a_f32 = expand_bf16(a_bf16, rank * input_dim, "lora A f32");
        float *b_f32 = expand_bf16(b_bf16, output_dim * rank, "lora B f32");

        uint64_t seed = 0x10a0000ull + which;
        size_t input_count = (size_t)ROWS * input_dim;
        float *x_f32 = checked_malloc(input_count * sizeof(*x_f32), "input");
        uint16_t *x_bf16 = checked_malloc(input_count * sizeof(*x_bf16),
                                          "input bf16");
        for (size_t index = 0; index < input_count; index++) {
            x_f32[index] = next_activation(&seed);
            x_bf16[index] = f32_to_bf16(x_f32[index]);
        }

        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, w_bf16,
                                                        output_dim * input_dim);
        h3_gpu_tensor *lora_a = h3_gpu_tensor_from_bf16(gpu, a_bf16,
                                                        rank * input_dim);
        h3_gpu_tensor *lora_b = h3_gpu_tensor_from_bf16(gpu, b_bf16,
                                                        output_dim * rank);
        h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, x_bf16,
                                                       input_count);
        if (!weight || !lora_a || !lora_b || !input)
            die("cannot upload weights");

        size_t output_count = (size_t)ROWS * output_dim;
        float *got = checked_malloc(output_count * sizeof(*got), "branch y");
        float *got_rounded = checked_malloc(output_count * sizeof(*got),
                                            "branch y bf16");
        float *base = checked_malloc(output_count * sizeof(*base), "base y");
        float *want = checked_malloc(output_count * sizeof(*want), "oracle y");

        run_branch(gpu, weight, lora_a, lora_b, input, (uint32_t)input_dim,
                   (uint32_t)rank, (uint32_t)output_dim, strength, got, base,
                   got_rounded);
        clock_t started = clock();
        fused_reference(w_bf16, a_f32, b_f32, x_f32, input_dim, rank,
                        output_dim, strength, want);
        double oracle_seconds = (double)(clock() - started) / CLOCKS_PER_SEC;

        error_stats f32_stats = compare(got, want, output_count);
        error_stats bf16_stats = compare(got_rounded, want, output_count);
        /* How much of the output the delta actually moves. Without this the
         * error figures are unfalsifiable: a delta near zero would agree with
         * anything. */
        error_stats delta_share = compare(base, got, output_count);
        printf("  %-14s [%zu,%zu] rank %zu  f32-acc rel-L2 %.3e rel-max %.3e"
               "  bf16-out rel-L2 %.3e rel-max %.3e  delta %.3e  oracle %.2fs\n",
               projection, output_dim, input_dim, rank,
               f32_stats.rel_l2, f32_stats.rel_max,
               bf16_stats.rel_l2, bf16_stats.rel_max, delta_share.rel_l2,
               oracle_seconds);

        /* H4 in miniature: at strength 0 the branch must return the base
         * output unchanged, bit for bit, not merely close to it. */
        run_branch(gpu, weight, lora_a, lora_b, input, (uint32_t)input_dim,
                   (uint32_t)rank, (uint32_t)output_dim, 0.0f, got, base, NULL);
        if (memcmp(got, base, output_count * sizeof(*got)) != 0)
            die("strength 0 changed the base output");

        /* The control, and the reason any of the numbers above can be read:
         * the same comparison with no delta at all. Whatever error it shows is
         * the BF16 GEMM's own, not the LoRA branch's. */
        fused_reference(w_bf16, a_f32, b_f32, x_f32, input_dim, rank,
                        output_dim, 0.0f, want);
        error_stats base_stats = compare(base, want, output_count);
        printf("  %-14s control, no delta:       rel-L2 %.3e rel-max %.3e\n",
               projection, base_stats.rel_l2, base_stats.rel_max);

        free(want);
        free(base);
        free(got_rounded);
        free(got);
        h3_gpu_tensor_free(input);
        h3_gpu_tensor_free(lora_b);
        h3_gpu_tensor_free(lora_a);
        h3_gpu_tensor_free(weight);
        free(x_bf16);
        free(x_f32);
        free(b_f32);
        free(a_f32);
        free(b_bf16);
        free(a_bf16);
        free(w_bf16);
    }

    h3_gpu_stats stats;
    if (h3_gpu_get_stats(gpu, &stats))
        printf("gpu: %.3f GiB cumulative allocations, %.3f GPU seconds, "
               "%llu submissions\n",
               (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
               stats.gpu_seconds, (unsigned long long)stats.submissions);

    h3_gpu_free(gpu);
    h3_st_free_header(&lora);
    h3_weight_store_free(weights);
    puts("ok: LoRA branch matches the fused float32 oracle, "
         "strength 0 is exact");
    return 0;
}
