/* T1 oracle harness for runtime LoRA adapters.
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
 * design settled on. The branches are not rebuilt here either - the file is
 * parsed by h3_lora_parse and materialised by h3_lora_site_build, the shipped
 * pair, so a target whose site comes back empty fails instead of quietly being
 * measured against a copy of the branch that only exists in this file.
 *
 * Two error figures are printed per projection:
 *   f32-acc  the sum kept in float32 (the floor of what the branch can do)
 *   bf16-out the same sum rounded back to BF16 for the next kernel
 * The tolerance is asserted on bf16-out, always the larger of the two and the
 * one the shipped h3_gpu_add_bf16 actually produces, together with the
 * significance guard on the delta's own weight.
 *
 * T1b is this same run seen whole: all four projections of one block in one
 * pass, and a projection whose pair is missing is fatal rather than skipped.
 * A per-tensor oracle is blind to a block where one projection is transposed
 * against its neighbours or never gets its branch at all; requiring the four
 * to be present and to agree in the same pass is what sees it.
 *
 * T5 is the same harness with a second LoRA path: two adapters are two
 * sequential branches over the same input, and the oracle fuses both deltas
 * into W. Passing the two real corpus files (rank 64 and rank 16) measures the composition of two ranks rather than one file twice.
 *
 * usage: h3_lora_oracle_test [LORA] [MODEL_ROOT] [STRENGTH] [BLOCK] [LORA2]
 * STRENGTH defaults to the 100 that T1 is defined at: at strength 1 the delta
 * sits under the BF16 noise floor and the significance guard refuses it.
 */

#include "h3_gpu.h"
#include "h3_lora.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ROWS = 4, MAX_SOURCES = 2 };

/* The tolerance applies to the bf16-out figure, always the larger
 * of the two printed. The significance guard is the other half of the same
 * requirement: a delta too weak to move the output agrees with anything, so a
 * LoRA file swapped for a limp one has to fail instead of passing silently.
 *
 * DEVIATION FROM THE ORIGINAL SPEC, on record. It asked for ten times the
 * tolerance, 3e-2, from a four-block sample. Both numbers below come from a
 * sweep of all 200 projections of the corpus turbo file, 50 blocks at strength
 * 100, with both guards switched off so that no block stopped early.
 *
 * Tolerance: rel-L2 runs 2.275e-03 to 2.384e-03, median 2.343e-03. 3e-3 clears
 * the worst projection by 20% and no projection exceeds it.
 *
 * Significance: the guard fires per projection, and the weakest projections are
 * block 29 mlp.fc2 at 6.583e-03, block 8 mlp.fc2 at 6.584e-03 and block 39
 * mlp.fc1 at 6.625e-03. mlp.fc2 is the weak family throughout, median 7.606e-03
 * against 1.731e-02 for attn.qkv_proj. 4.4e-03 sits 33% below the measured
 * minimum and still catches what the guard exists for: at strength 1 the delta
 * falls to around 1e-04, an order of magnitude under this floor.
 *
 * An earlier revision of this comment recorded weakest deltas of 1.326e-02 and
 * set the floor at 1.0e-2. Those are per-block figures, and the guard is per
 * projection: at 1.0e-2 a correct implementation fails on most blocks, block 0
 * included. Anyone re-measuring should keep the two apart. */
static const double T1_TOLERANCE = 3e-3;
static const double T1_MIN_DELTA_SHARE = 4.4e-03;

/* Per-half amplitude of the synthetic AdaLN pair, on top of the 1/sqrt(fan-in)
 * that keeps the two GEMMs in scale. The corpus design leaves the amplitude
 * free (any pair of the right shape), and this value puts the delta near the
 * 4,7% of the output measured on the real pairs at strength 100
 * (docs/lora.md). That regime is the premise of the argument that the fused-A
 * rounding is negligible; an arbitrary amplitude that drowns the base output in
 * delta would measure a case the design never costed. Measured here at
 * strength 100: delta share 4,08e-02, against 1,67e-02 to 4,96e-02 for the four
 * real pairs of the turbo file. */
#define SYNTHETIC_AMPLITUDE 0.1f

/* The amplitude is calibrated at strength 100, but a file whose own pairs are
 * strong has to run at a much lower one: mystic saturates the output at 100 and
 * is measured at 3. The delta scales as strength times the square of the
 * amplitude, so the amplitude carries 1/sqrt(strength) and the synthetic pair
 * lands in the same regime whatever strength the run uses. Without it the
 * synthetic AdaLN delta sinks under the significance floor at low strength and
 * the run fails for the harness's reason instead of the code's. */
static float synthetic_amplitude(float strength) {
    return SYNTHETIC_AMPLITUDE * sqrtf(100.0f / strength);
}

/* `synthetic_rank` is 0 for a target whose pair is read from the LoRA file.
 * The AdaLN target has no pair in either corpus file (the pruned converter
 * dropped all 51 of them), and the corpus design settles that the numbers do
 * not need the 780 MB upstream download: the oracle builds W + s*B@A from a
 * real W and a pair of the right shape, so the pair is synthesised here at the
 * shapes read from the upstream header, A[16, 2688] and B[96768, 16].
 * The target itself is real, and it is the same
 * blocks.N.adaln_proj.linear.weight that h3_dit_schedule.c:267 loads. */
static const struct {
    const char *name;
    size_t synthetic_rank;
} PROJECTIONS[] = {
    {"attn.qkv_proj", 0}, {"attn.out_proj", 0},
    {"mlp.fc1", 0}, {"mlp.fc2", 0},
    {"adaln_proj.linear", 16}
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

/* One half of a real pair, at the precision the file wrote it. The shipped
 * path rounds an F32 pair to bf16 on its way to the GPU; the reference keeps
 * the file's own values, so that rounding is measured rather than cancelled,
 * the way fuse_scale already does for the strength. */
static float *read_pair_f32(const h3_st_header *header,
                            const h3_st_tensor *tensor, float scale,
                            const char *what) {
    size_t count = (size_t)h3_st_tensor_elements(tensor);
    float *values = checked_malloc(count * sizeof(*values), what);
    if (tensor->dtype == H3_DTYPE_F32) {
        char error[512];
        if (!h3_st_read_data(header, tensor, values, count * sizeof(*values),
                             error, sizeof(error))) die(error);
        if (scale != 1.0f) {
            for (size_t index = 0; index < count; index++) {
                values[index] *= scale;
            }
        }
        return values;
    }
    uint16_t *packed = read_bf16_tensor(header, tensor, what);
    for (size_t index = 0; index < count; index++) {
        values[index] = bf16_to_f32(packed[index]) * scale;
    }
    free(packed);
    return values;
}

/* One half of a synthetic pair, scaled so the delta lands in the regime the
 * real pairs occupy instead of drowning the base output: 1/sqrt(fan-in) on
 * each of the two GEMMs, times the strength-corrected amplitude above. The
 * measured delta share is printed with the errors and guarded below, so the
 * choice is visible rather than assumed. */
static uint16_t *synthetic_half(size_t count, size_t fan_in, uint64_t seed,
                                float amplitude, const char *what) {
    uint16_t *values = checked_malloc(count * sizeof(*values), what);
    float scale = amplitude / sqrtf((float)fan_in);
    for (size_t index = 0; index < count; index++)
        values[index] = f32_to_bf16(next_activation(&seed) * scale);
    return values;
}

/* A bf16 copy of A with the scale already inside it, which is where the
 * shipped path puts it (h3_lora.c h3_lora_branch_build). The
 * oracle keeps the unscaled A in float32, so the extra rounding this
 * introduces is measured rather than cancelled. */
static uint16_t *fuse_scale(const uint16_t *values, size_t count, float scale,
                            const char *what) {
    uint16_t *fused = checked_malloc(count * sizeof(*fused), what);
    for (size_t index = 0; index < count; index++)
        fused[index] = f32_to_bf16(bf16_to_f32(values[index]) * scale);
    return fused;
}

/* `scale` carries the file's own alpha/rank into the reference, which is where
 * the shipped materialisation puts it too (h3_lora.c h3_lora_branch_build,
 * H6). It is 1.0 for both corpus files, which have no .alpha. */
static float *expand_bf16(const uint16_t *values, size_t count, float scale,
                          const char *what) {
    float *expanded = checked_malloc(count * sizeof(*expanded), what);
    for (size_t index = 0; index < count; index++)
        expanded[index] = bf16_to_f32(values[index]) * scale;
    return expanded;
}

/* The parsed pair for one target name, or NULL. h3_lora_parse has already
 * stripped the optional diffusion_model. prefix, so a pair name is
 * its target minus ".weight" and a plain compare finds it. */
static const h3_lora_pair *adapter_pair(const h3_lora_adapter *adapter,
                                        const char *name) {
    for (size_t index = 0; index < adapter->pair_count; index++) {
        if (!strcmp(adapter->pairs[index].name, name))
            return &adapter->pairs[index];
    }
    return NULL;
}

/* y = Wx + sum_i Bi(Ai x), assembled from the kernels the DiT already uses.
 * Each `lora_a` arrives with the strength already fused into it, exactly as
 * the shipped materialisation hands it to h3_dit.c's lora_branch, so nothing
 * is scaled here either. Several adapters are several sequential branches
 * over the same input, never a concatenated A: that is the
 * composition T5 asks about. base_out and rounded_out are optional. */
static void run_branch(h3_gpu *gpu, const h3_gpu_tensor *weight,
                       h3_gpu_tensor *const *lora_a,
                       h3_gpu_tensor *const *lora_b,
                       const uint32_t *ranks, size_t branches,
                       const h3_gpu_tensor *input, uint32_t input_dim,
                       uint32_t output_dim,
                       float *y_out, float *base_out, float *rounded_out) {
    size_t output_count = (size_t)ROWS * output_dim;
    h3_gpu_tensor *base = h3_gpu_tensor_new_bf16(gpu, output_count);
    h3_gpu_tensor *base_f32 = h3_gpu_tensor_new_f32(gpu, output_count);
    h3_gpu_tensor *rounded = h3_gpu_tensor_new_bf16(gpu, output_count);
    /* Two f32 accumulators, alternated: h3_gpu_add_scaled_f32 is not declared
     * safe to write into one of its own inputs. */
    h3_gpu_tensor *accumulator[2] = {h3_gpu_tensor_new_f32(gpu, output_count),
                                     h3_gpu_tensor_new_f32(gpu, output_count)};
    h3_gpu_tensor *hidden[MAX_SOURCES] = {NULL};
    h3_gpu_tensor *delta[MAX_SOURCES] = {NULL};
    h3_gpu_tensor *delta_f32[MAX_SOURCES] = {NULL};
    if (!base || !base_f32 || !rounded || !accumulator[0] || !accumulator[1])
        die("cannot allocate branch scratch");
    for (size_t index = 0; index < branches; index++) {
        hidden[index] = h3_gpu_tensor_new_bf16(gpu,
                                               (size_t)ROWS * ranks[index]);
        delta[index] = h3_gpu_tensor_new_bf16(gpu, output_count);
        delta_f32[index] = h3_gpu_tensor_new_f32(gpu, output_count);
        if (!hidden[index] || !delta[index] || !delta_f32[index])
            die("cannot allocate branch scratch");
    }

    uint32_t elements = (uint32_t)output_count;
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_linear_bf16(gpu, base, input, weight, NULL, ROWS,
                            input_dim, output_dim) ||
        !h3_gpu_cast_bf16_to_f32(gpu, base_f32, base, elements))
        die(h3_gpu_error(gpu));
    h3_gpu_tensor *carried = base_f32;
    for (size_t index = 0; index < branches; index++) {
        h3_gpu_tensor *next = accumulator[index & 1];
        if (!h3_gpu_linear_bf16(gpu, hidden[index], input, lora_a[index], NULL,
                                ROWS, input_dim, ranks[index]) ||
            !h3_gpu_linear_bf16(gpu, delta[index], hidden[index],
                                lora_b[index], NULL, ROWS, ranks[index],
                                output_dim) ||
            !h3_gpu_cast_bf16_to_f32(gpu, delta_f32[index], delta[index],
                                     elements) ||
            /* Unscaled sum: the strength already lives in A. With ONE branch
             * `rounded` is bit-identical to what the shipped h3_gpu_add_bf16
             * writes, which adds in f32 and rounds once
             * (h3_shaders.metal:4036). With two, h3_dit.c's lora_branch adds
             * into a BF16 y once per branch and so rounds between them, while
             * this accumulator stays in f32 to the end: for T5 bf16-out is one
             * rounding short of the shipped path, not identical to it. */
            !h3_gpu_add_scaled_f32(gpu, next, carried, delta_f32[index], 1.0f,
                                   1.0f, elements)) die(h3_gpu_error(gpu));
        carried = next;
    }
    if (!h3_gpu_cast_f32_to_bf16(gpu, rounded, carried, elements) ||
        !h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));

    if (!h3_gpu_tensor_read_f32(carried, y_out, output_count))
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

    for (size_t index = 0; index < branches; index++) {
        h3_gpu_tensor_free(delta_f32[index]);
        h3_gpu_tensor_free(delta[index]);
        h3_gpu_tensor_free(hidden[index]);
    }
    h3_gpu_tensor_free(accumulator[1]);
    h3_gpu_tensor_free(accumulator[0]);
    h3_gpu_tensor_free(rounded);
    h3_gpu_tensor_free(base_f32);
    h3_gpu_tensor_free(base);
}

/* The oracle. One fused row of W at a time, so the full [output_dim,
 * input_dim] float32 matrix is never materialised. */
static void fused_reference(const uint16_t *weight,
                            const float *const *lora_a,
                            const float *const *lora_b, const size_t *ranks,
                            size_t pairs, const float *input,
                            size_t input_dim, size_t output_dim,
                            float strength, float *y_out) {
    float *row = checked_malloc(input_dim * sizeof(*row), "fused weight row");
    for (size_t out = 0; out < output_dim; out++) {
        const uint16_t *source = weight + out * input_dim;
        for (size_t index = 0; index < input_dim; index++)
            row[index] = bf16_to_f32(source[index]);
        for (size_t pair = 0; pair < pairs; pair++) {
            size_t rank = ranks[pair];
            for (size_t k = 0; k < rank; k++) {
                float scale = strength * lora_b[pair][out * rank + k];
                const float *a_row = lora_a[pair] + k * input_dim;
                for (size_t index = 0; index < input_dim; index++)
                    row[index] += scale * a_row[index];
            }
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
    const char *lora_paths[MAX_SOURCES];
    lora_paths[0] = argc > 1 ? argv[1] : "loras/turbo.safetensors";
    const char *model_root = argc > 2 ? argv[2] : "MiniMax-H3";
    float strength = argc > 3 ? (float)atof(argv[3]) : 100.0f;
    int block = argc > 4 ? atoi(argv[4]) : 0;
    /* A second LoRA turns the run into T5: two adapters composed, measured
     * against an oracle that fuses both deltas into the same W. */
    size_t sources = 1;
    if (argc > 5) {
        lora_paths[1] = argv[5];
        sources = 2;
    }

    char error[512];
    char weight_path[1024];
    snprintf(weight_path, sizeof(weight_path), "%s/FL2VA/transformer",
             model_root);
    h3_weight_store *weights = h3_weight_store_open(weight_path, error,
                                                    sizeof(error));
    if (!weights) die(error);
    /* The adapters and the active set are the shipped ones, not a local
     * re-read of the file: everything below materialises through
     * h3_lora_site_build, so what is measured is the branch h3_dit.c
     * dispatches and not a copy of it living in this test. */
    h3_lora_adapter *adapters[MAX_SOURCES];
    h3_lora_entry entries[MAX_SOURCES], zero_entries[MAX_SOURCES];
    for (size_t source = 0; source < sources; source++) {
        adapters[source] = h3_lora_parse(lora_paths[source], weight_path,
                                         error, sizeof(error), NULL, NULL);
        if (!adapters[source]) die(error);
        entries[source].adapter = adapters[source];
        entries[source].strength = strength;
        zero_entries[source].adapter = adapters[source];
        zero_entries[source].strength = 0.0f;
    }
    h3_lora_set set = {entries, sources};
    h3_lora_set zero_set = {zero_entries, sources};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    printf("oracle: block %d, strength %.4f, %d input rows, %zu adapter%s\n",
           block, (double)strength, ROWS, sources, sources == 1 ? "" : "s");
    for (size_t source = 0; source < sources; source++)
        printf("  adapter %zu: %s\n", source + 1, lora_paths[source]);

    size_t projections = sizeof(PROJECTIONS) / sizeof(*PROJECTIONS);
    for (size_t which = 0; which < projections; which++) {
        const char *projection = PROJECTIONS[which].name;
        /* The target comes first: its shape is what a pair has to fit, and for
         * a synthetic pair it is where the two dimensions come from. */
        char weight_name[256];
        snprintf(weight_name, sizeof(weight_name), "blocks.%d.%s.weight",
                 block, projection);
        const h3_st_header *shard = NULL;
        const h3_st_tensor *w_tensor = h3_weight_find(weights, weight_name,
                                                      &shard);
        if (!w_tensor || !shard) die("checkpoint is missing a target tensor");
        if (w_tensor->ndim != 2 || w_tensor->dtype != H3_DTYPE_BF16)
            die("target tensor is not a BF16 matrix");
        size_t output_dim = (size_t)w_tensor->shape[0];
        size_t input_dim = (size_t)w_tensor->shape[1];

        /* The pair name the shipped path uses: the target minus ".weight". */
        char pair_name[256];
        snprintf(pair_name, sizeof(pair_name), "blocks.%d.%s", block,
                 projection);

        const h3_lora_pair *pair[MAX_SOURCES];
        size_t missing = 0, first_missing = 0;
        for (size_t source = 0; source < sources; source++) {
            pair[source] = adapter_pair(adapters[source], pair_name);
            if (!pair[source] && !missing++) first_missing = source;
        }
        /* T1b: the projections of a block are measured together or not at
         * all. Skipping one here is exactly the hole a per-tensor oracle
         * leaves - a projection that never gets its branch passes by not being
         * looked at. The AdaLN target is the exception the corpus design settles: no
         * corpus file carries its pair, so it is synthesised at the upstream
         * shapes and measured against the same real W. */
        int synthetic[MAX_SOURCES] = {0};
        size_t real_sources = 0;
        for (size_t source = 0; source < sources; source++) {
            synthetic[source] = !pair[source] &&
                                PROJECTIONS[which].synthetic_rank != 0;
            if (!synthetic[source]) real_sources++;
        }
        /* Adapters may disagree about a target: turbo carries AdaLN pairs and
         * the style files do not, which is exactly the set a real run mixes.
         * Each one is therefore decided on its own, and only a projection with
         * no synthetic fallback at all is fatal. */
        if (missing && !PROJECTIONS[which].synthetic_rank) {
            fprintf(stderr, "FAIL tests/test_lora_oracle.c: %s has no "
                    "pair for %s; a whole-block oracle needs all four "
                    "projections in one pass\n",
                    lora_paths[first_missing], pair_name);
            exit(1);
        }

        size_t rank[MAX_SOURCES];
        float pair_scale[MAX_SOURCES];
        uint16_t *a_bf16[MAX_SOURCES], *b_bf16[MAX_SOURCES];
        for (size_t source = 0; source < sources; source++) {
            if (synthetic[source]) {
                rank[source] = PROJECTIONS[which].synthetic_rank;
                pair_scale[source] = 1.0f;
                uint64_t seed = 0x5a1e0000ull + which * 16 + source;
                float amplitude = synthetic_amplitude(strength);
                a_bf16[source] = synthetic_half(rank[source] * input_dim,
                                                input_dim, seed, amplitude,
                                                "synthetic lora A");
                b_bf16[source] = synthetic_half(output_dim * rank[source],
                                                rank[source], seed + 0x1000ull,
                                                amplitude,
                                                "synthetic lora B");
                printf("  %-17s adapter %zu: synthetic pair, A[%zu,%zu] "
                       "B[%zu,%zu]\n", projection, source + 1, rank[source],
                       input_dim, output_dim, rank[source]);
            } else {
                /* Shapes and rank agreement were already checked by
                 * h3_lora_parse against this very target, which is the check
                 * that ships; re-deriving them here would measure the test. */
                rank[source] = (size_t)pair[source]->rank;
                pair_scale[source] = pair[source]->scale;
                /* No bf16 copy: the GPU side comes from the shipped
                 * h3_lora_site_build below, and the reference reads the file
                 * at its own precision just under here. */
                a_bf16[source] = NULL;
                b_bf16[source] = NULL;
            }
        }

        uint16_t *w_bf16 = read_bf16_tensor(shard, w_tensor, "base weight");
        const float *a_f32[MAX_SOURCES], *b_f32[MAX_SOURCES];
        for (size_t source = 0; source < sources; source++) {
            if (synthetic[source]) {
                a_f32[source] = expand_bf16(a_bf16[source],
                                            rank[source] * input_dim, 1.0f,
                                            "lora A f32");
                b_f32[source] = expand_bf16(b_bf16[source],
                                            output_dim * rank[source],
                                            pair_scale[source], "lora B f32");
            } else {
                a_f32[source] = read_pair_f32(&adapters[source]->header,
                                              &pair[source]->a, 1.0f,
                                              "lora A f32");
                b_f32[source] = read_pair_f32(&adapters[source]->header,
                                              &pair[source]->b,
                                              pair_scale[source],
                                              "lora B f32");
            }
        }

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
        h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, x_bf16,
                                                       input_count);
        if (!weight || !input) die("cannot upload weights");

        /* The branches themselves, materialised by the shipped
         * h3_lora_site_build at this strength and at strength 0. This is the
         * substitution that makes the harness measure what h3_dit.c
         * dispatches: A already carries strength * alpha/rank, B is the file's
         * own, and a site that came back with fewer branches than the set has
         * adapters is the missing graft T1b exists to catch. The synthetic
         * AdaLN pair has no site because it has no file, so its two halves are
         * uploaded here the way h3_lora_branch_build would. */
        h3_lora_site site = {NULL, 0}, zero_site = {NULL, 0};
        h3_gpu_tensor *lora_a[MAX_SOURCES], *lora_a0[MAX_SOURCES];
        h3_gpu_tensor *lora_b[MAX_SOURCES];
        uint32_t ranks32[MAX_SOURCES];
        if (real_sources) {
            if (!h3_lora_site_build(&set, gpu, pair_name, &site, error,
                                    sizeof(error)) ||
                !h3_lora_site_build(&zero_set, gpu, pair_name, &zero_site,
                                    error, sizeof(error))) die(error);
            /* One branch per adapter that carries this pair, and no more: an
             * adapter that is missing it contributes a synthetic branch below
             * instead. A shortfall against that count is the missing graft
             * T1b exists to catch. */
            if (site.count != real_sources || zero_site.count != real_sources)
                die("h3_lora_site_build gave the target fewer branches than "
                    "the active set has adapters for it");
        }
        for (size_t source = 0, branch = 0; source < sources; source++) {
            if (synthetic[source]) {
                uint16_t *a_fused = fuse_scale(a_bf16[source],
                                               rank[source] * input_dim,
                                               strength, "fused lora A");
                uint16_t *a_zero = fuse_scale(a_bf16[source],
                                              rank[source] * input_dim, 0.0f,
                                              "zero lora A");
                lora_a[source] = h3_gpu_tensor_from_bf16(
                    gpu, a_fused, rank[source] * input_dim);
                lora_a0[source] = h3_gpu_tensor_from_bf16(
                    gpu, a_zero, rank[source] * input_dim);
                lora_b[source] = h3_gpu_tensor_from_bf16(
                    gpu, b_bf16[source], output_dim * rank[source]);
                ranks32[source] = (uint32_t)rank[source];
                free(a_zero);
                free(a_fused);
                if (!lora_a[source] || !lora_a0[source] || !lora_b[source])
                    die("cannot upload weights");
            } else {
                lora_a[source] = site.branches[branch].a;
                lora_b[source] = site.branches[branch].b;
                lora_a0[source] = zero_site.branches[branch].a;
                ranks32[source] = site.branches[branch].rank;
                branch++;
                if (ranks32[source] != (uint32_t)rank[source])
                    die("the materialised branch disagrees with its pair's "
                        "rank");
            }
        }

        size_t output_count = (size_t)ROWS * output_dim;
        float *got = checked_malloc(output_count * sizeof(*got), "branch y");
        float *got_rounded = checked_malloc(output_count * sizeof(*got),
                                            "branch y bf16");
        float *base = checked_malloc(output_count * sizeof(*base), "base y");
        float *want = checked_malloc(output_count * sizeof(*want), "oracle y");

        run_branch(gpu, weight, lora_a, lora_b, ranks32, sources, input,
                   (uint32_t)input_dim, (uint32_t)output_dim, got, base,
                   got_rounded);
        clock_t started = clock();
        fused_reference(w_bf16, a_f32, b_f32, rank, sources, x_f32, input_dim,
                        output_dim, strength, want);
        double oracle_seconds = (double)(clock() - started) / CLOCKS_PER_SEC;

        error_stats f32_stats = compare(got, want, output_count);
        error_stats bf16_stats = compare(got_rounded, want, output_count);
        /* How much of the output the delta actually moves. Without this the
         * error figures are unfalsifiable: a delta near zero would agree with
         * anything. */
        error_stats delta_share = compare(base, got, output_count);
        char rank_text[32];
        int used = snprintf(rank_text, sizeof(rank_text), "%zu", rank[0]);
        for (size_t source = 1; source < sources; source++)
            used += snprintf(rank_text + used, sizeof(rank_text) - (size_t)used,
                             "+%zu", rank[source]);
        printf("  %-17s [%zu,%zu] rank %s  f32-acc rel-L2 %.3e rel-max %.3e"
               "  bf16-out rel-L2 %.3e rel-max %.3e  delta %.3e  oracle %.2fs\n",
               projection, output_dim, input_dim, rank_text,
               f32_stats.rel_l2, f32_stats.rel_max,
               bf16_stats.rel_l2, bf16_stats.rel_max, delta_share.rel_l2,
               oracle_seconds);

        /* T1's two verdicts. Both are needed: the tolerance alone passes a
         * branch whose delta never arrives, and the significance guard alone
         * passes a delta that arrives wrong. */
        if (!(bf16_stats.rel_l2 <= T1_TOLERANCE))
            die("branch exceeds the T1 tolerance of 3e-3 on bf16-out");
        if (!(delta_share.rel_l2 >= T1_MIN_DELTA_SHARE))
            die("delta too weak to be significant: raise the strength or use "
                "a stronger pair (T1 significance guard)");

        /* H4 in miniature: with the strength fused to zero the branch must
         * return the base output unchanged, bit for bit, not merely close. */
        run_branch(gpu, weight, lora_a0, lora_b, ranks32, sources, input,
                   (uint32_t)input_dim, (uint32_t)output_dim, got, base, NULL);
        if (memcmp(got, base, output_count * sizeof(*got)) != 0)
            die("strength 0 changed the base output");

        /* The control, and the reason any of the numbers above can be read:
         * the same comparison with no delta at all. Whatever error it shows is
         * the BF16 GEMM's own, not the LoRA branch's. */
        fused_reference(w_bf16, a_f32, b_f32, rank, sources, x_f32, input_dim,
                        output_dim, 0.0f, want);
        error_stats base_stats = compare(base, want, output_count);
        printf("  %-17s control, no delta:       rel-L2 %.3e rel-max %.3e\n",
               projection, base_stats.rel_l2, base_stats.rel_max);

        free(want);
        free(base);
        free(got_rounded);
        free(got);
        for (size_t source = 0; source < sources; source++) {
            /* Only the synthetic branches are ours to free; the rest belong to
             * the two sites, released just below. */
            if (synthetic[source]) {
                h3_gpu_tensor_free(lora_b[source]);
                h3_gpu_tensor_free(lora_a0[source]);
                h3_gpu_tensor_free(lora_a[source]);
            }
            free((void *)b_f32[source]);
            free((void *)a_f32[source]);
            free(b_bf16[source]);
            free(a_bf16[source]);
        }
        h3_lora_site_free(&zero_site);
        h3_lora_site_free(&site);
        h3_gpu_tensor_free(input);
        h3_gpu_tensor_free(weight);
        free(x_bf16);
        free(x_f32);
        free(w_bf16);
    }

    h3_gpu_stats stats;
    if (h3_gpu_get_stats(gpu, &stats))
        printf("gpu: %.3f GiB cumulative allocations, %.3f GPU seconds, "
               "%llu submissions\n",
               (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
               stats.gpu_seconds, (unsigned long long)stats.submissions);

    h3_gpu_free(gpu);
    for (size_t source = 0; source < sources; source++)
        h3_lora_adapter_free(adapters[source]);
    h3_weight_store_free(weights);
    puts("ok: LoRA branch matches the fused float32 oracle, "
         "strength 0 is exact");
    return 0;
}
