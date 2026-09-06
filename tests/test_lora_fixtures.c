/* Synthetic half of the LoRA corpus: T3, T3b, T4, T4b.
 *
 * The subject here is a file's CONTENT - names, shapes, metadata, a broken
 * header - so every file is synthesised in a temporary directory at each run
 * and deleted on exit. Nothing is committed and nothing is downloaded, which
 * is why these checks CANNOT skip: the AdaLN count and the conversion warning
 * are verified exactly on the machine that has no checkpoint.
 *
 * Safetensors is a uint64 length, a JSON header and raw bytes, so the writer
 * below is thirty lines of C rather than a Python script. The pair bytes are
 * zeros: no test in this file reads a number out of an invented file.
 *
 * The checkpoint is synthesised too. h3_lora_parse resolves every pair against
 * the transformer shard headers, so the fixtures need a shard to be
 * resolved against: one tiny file carrying the six target names at toy
 * shapes. Real shapes would need the real 62 GB.
 */

#include "h3_lora.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define FIXTURE_COUNT 12

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

/* ---- the report callback: every line of one parse, in order ---- */

enum { MAX_LINES = 32, LINE_SIZE = 512 };
static char lines[MAX_LINES][LINE_SIZE];
static size_t line_count;

static void capture(const char *line, void *opaque) {
    (void)opaque;
    if (line_count < MAX_LINES) {
        snprintf(lines[line_count], LINE_SIZE, "%s", line);
    }
    line_count++;
}

static void capture_reset(void) { line_count = 0; }

/* A captured line containing `needle`, or NULL. */
static const char *line_with(const char *needle) {
    size_t seen = line_count < MAX_LINES ? line_count : MAX_LINES;
    for (size_t index = 0; index < seen; index++) {
        if (strstr(lines[index], needle)) return lines[index];
    }
    return NULL;
}

static size_t lines_with(const char *needle) {
    size_t seen = line_count < MAX_LINES ? line_count : MAX_LINES;
    size_t found = 0;
    for (size_t index = 0; index < seen; index++) {
        if (strstr(lines[index], needle)) found++;
    }
    return found;
}

/* ---- the safetensors writer ---- */

typedef struct {
    const char *name;
    const char *dtype;      /* "BF16" or "F32" */
    int ndim;
    uint64_t shape[2];
    const void *bytes;      /* NULL fills the tensor with zeros */
} fixture_tensor;

static size_t dtype_bytes(const char *dtype) {
    return strcmp(dtype, "F32") == 0 ? 4 : 2;
}

static uint64_t tensor_bytes(const fixture_tensor *tensor) {
    uint64_t elements = 1;
    for (int index = 0; index < tensor->ndim; index++) {
        elements *= tensor->shape[index];
    }
    return elements * dtype_bytes(tensor->dtype);
}

static void write_fixture(const char *path, const char *metadata,
                          const fixture_tensor *tensors, size_t count) {
    char json[8192];
    int used = 0;
/* Checked after every append, not once at the end: an overflowing snprintf
 * returns the length it wanted, and the next call would then compute
 * sizeof(json) - used with `used` past the end, underflowing the size_t.
 * Not a CHECK: this is the writer keeping itself honest, not one of the
 * assertions the final count reports. */
#define APPEND(...) do { \
    used += snprintf(json + used, sizeof(json) - (size_t)used, __VA_ARGS__); \
    if (used < 0 || (size_t)used >= sizeof(json)) { \
        fprintf(stderr, "FAIL %s:%d: fixture header does not fit\n", \
                __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)
    APPEND("{");
    if (metadata) {
        APPEND("\"__metadata__\":{%s},", metadata);
    }
    uint64_t offset = 0;
    for (size_t index = 0; index < count; index++) {
        const fixture_tensor *tensor = &tensors[index];
        uint64_t bytes = tensor_bytes(tensor);
        APPEND("%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
               index ? "," : "", tensor->name, tensor->dtype);
        for (int dimension = 0; dimension < tensor->ndim; dimension++) {
            APPEND("%s%llu", dimension ? "," : "",
                   (unsigned long long)tensor->shape[dimension]);
        }
        APPEND("],\"data_offsets\":[%llu,%llu]}",
               (unsigned long long)offset,
               (unsigned long long)(offset + bytes));
        offset += bytes;
    }
    APPEND("}");
#undef APPEND

    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    unsigned char prefix[8];
    uint64_t header_size = (uint64_t)used;
    for (unsigned index = 0; index < 8; index++) {
        prefix[index] = (unsigned char)(header_size >> (index * 8));
    }
    CHECK(fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix));
    CHECK(fwrite(json, 1, (size_t)used, file) == (size_t)used);
    for (size_t index = 0; index < count; index++) {
        const fixture_tensor *tensor = &tensors[index];
        size_t bytes = (size_t)tensor_bytes(tensor);
        if (tensor->bytes) {
            CHECK(fwrite(tensor->bytes, 1, bytes, file) == bytes);
        } else {
            static const unsigned char zeros[256] = {0};
            while (bytes) {
                size_t chunk = bytes < sizeof(zeros) ? bytes : sizeof(zeros);
                CHECK(fwrite(zeros, 1, chunk, file) == chunk);
                bytes -= chunk;
            }
        }
    }
    CHECK(fclose(file) == 0);
}

/* ---- the fixtures ---- */

static char root[512];
static char transformer[600];

static void cleanup(void) {
    /* ponytail: the directory is our own mkdtemp, so one rm -rf beats an
     * nftw walk written for a dozen files we created ourselves. */
    char command[600];
    snprintf(command, sizeof(command), "rm -rf %s", root);
    if (system(command) != 0) {
        fprintf(stderr, "warning: %s was left behind\n", root);
    }
}

static void path_in(char *buffer, size_t size, const char *name) {
    snprintf(buffer, size, "%s/%s", root, name);
}

/* The toy checkpoint. Six targets: the four projections plus the two AdaLN
 * precompute targets, at shapes small enough to write. */
enum { QKV_OUT = 24, QKV_IN = 8, OUT_OUT = 8, OUT_IN = 12,
       FC1_OUT = 32, FC1_IN = 8, ADALN_OUT = 48, ADALN_IN = 8 };

static void write_checkpoint(void) {
    static const fixture_tensor targets[] = {
        {"blocks.0.attn.qkv_proj.weight", "BF16", 2, {QKV_OUT, QKV_IN}, NULL},
        {"blocks.0.attn.out_proj.weight", "BF16", 2, {OUT_OUT, OUT_IN}, NULL},
        {"blocks.0.mlp.fc1.weight", "BF16", 2, {FC1_OUT, FC1_IN}, NULL},
        {"blocks.0.mlp.fc2.weight", "BF16", 2, {OUT_OUT, 16}, NULL},
        {"blocks.0.adaln_proj.linear.weight", "BF16", 2,
         {ADALN_OUT, ADALN_IN}, NULL},
        {"final_layer.adaln_proj.linear.weight", "BF16", 2,
         {ADALN_OUT, ADALN_IN}, NULL}
    };
    char shard[700];
    snprintf(transformer, sizeof(transformer), "%s/transformer", root);
    CHECK(mkdir(transformer, 0700) == 0);
    snprintf(shard, sizeof(shard), "%s/model-00001-of-00001.safetensors",
             transformer);
    write_fixture(shard, NULL, targets,
                  sizeof(targets) / sizeof(*targets));
}

/* Fixture 1 and its variants: one qkv pair and one out_proj pair at rank 4. */
#define PAIR_A(name, in) \
    {name ".lora_A.weight", "BF16", 2, {4, in}, NULL}
#define PAIR_B(name, out) \
    {name ".lora_B.weight", "BF16", 2, {out, 4}, NULL}

static const fixture_tensor CONVENTION_A_PREFIXED[] = {
    PAIR_A("diffusion_model.blocks.0.attn.qkv_proj", QKV_IN),
    PAIR_B("diffusion_model.blocks.0.attn.qkv_proj", QKV_OUT),
    PAIR_A("diffusion_model.blocks.0.attn.out_proj", OUT_IN),
    PAIR_B("diffusion_model.blocks.0.attn.out_proj", OUT_OUT)
};

static void write_all(char paths[FIXTURE_COUNT][700]) {
    static const fixture_tensor bare[] = {
        PAIR_A("blocks.0.attn.qkv_proj", QKV_IN),
        PAIR_B("blocks.0.attn.qkv_proj", QKV_OUT),
        PAIR_A("blocks.0.attn.out_proj", OUT_IN),
        PAIR_B("blocks.0.attn.out_proj", OUT_OUT)
    };
    /* alpha 8 over rank 4 is an effective scale of 2.0, which is only visible
     * when alpha is read as alpha/rank instead of ignored (H6). */
    static const float alpha_value = 8.0f;
    static const fixture_tensor hybrid[] = {
        PAIR_A("diffusion_model.blocks.0.attn.qkv_proj", QKV_IN),
        PAIR_B("diffusion_model.blocks.0.attn.qkv_proj", QKV_OUT),
        {"diffusion_model.blocks.0.attn.qkv_proj.alpha", "F32", 1, {1},
         &alpha_value}
    };
    static const fixture_tensor mixed[] = {
        PAIR_A("blocks.0.attn.qkv_proj", QKV_IN),
        PAIR_B("blocks.0.attn.qkv_proj", QKV_OUT),
        {"blocks.0.mlp.fc1.lora_A.weight", "BF16", 2, {8, FC1_IN}, NULL},
        {"blocks.0.mlp.fc1.lora_B.weight", "BF16", 2, {FC1_OUT, 8}, NULL}
    };
    /* Convention B as the corpus writes it: lora_down/lora_up, the target
     * flattened to underscores behind a lora_unet_ prefix, and an .alpha whose
     * ratio is not 1.0 so a scale read off the wrong sibling key cannot pass.
     * mlp_fc1 is the two-dot case, and its rank 8 proves the rank is still read
     * per pair on this convention. */
    static const float alpha_flat = 8.0f;
    static const fixture_tensor convention_b[] = {
        {"lora_unet_blocks_0_attn_qkv_proj.lora_down.weight", "BF16", 2,
         {4, QKV_IN}, NULL},
        {"lora_unet_blocks_0_attn_qkv_proj.lora_up.weight", "BF16", 2,
         {QKV_OUT, 4}, NULL},
        {"lora_unet_blocks_0_attn_qkv_proj.alpha", "F32", 1, {1}, &alpha_flat},
        {"lora_unet_blocks_0_mlp_fc1.lora_down.weight", "BF16", 2,
         {8, FC1_IN}, NULL},
        {"lora_unet_blocks_0_mlp_fc1.lora_up.weight", "BF16", 2,
         {FC1_OUT, 8}, NULL}
    };
    /* The same convention naming a block the checkpoint does not have. A
     * flattened name that resolves to nothing is unapplicable, the error that
     * already exists, and never a name h3 guesses the dots back into. */
    static const fixture_tensor convention_b_no_target[] = {
        {"lora_unet_blocks_51_attn_qkv_proj.lora_down.weight", "BF16", 2,
         {4, QKV_IN}, NULL},
        {"lora_unet_blocks_51_attn_qkv_proj.lora_up.weight", "BF16", 2,
         {QKV_OUT, 4}, NULL}
    };
    /* Two pairs whose names name nothing in the checkpoint, plus one that
     * resolves: the error has to list both orphans, not stop at the first. */
    static const fixture_tensor orphans[] = {
        PAIR_A("blocks.0.attn.qkv_proj", QKV_IN),
        PAIR_B("blocks.0.attn.qkv_proj", QKV_OUT),
        PAIR_A("blocks.51.attn.qkv_proj", QKV_IN),
        PAIR_B("blocks.51.attn.qkv_proj", QKV_OUT),
        PAIR_A("blocks.51.mlp.fc1", FC1_IN),
        PAIR_B("blocks.51.mlp.fc1", FC1_OUT)
    };
    /* A rank 4 and B rank 8: the pair contradicts itself. */
    static const fixture_tensor incoherent[] = {
        {"blocks.0.attn.qkv_proj.lora_A.weight", "BF16", 2, {4, QKV_IN}, NULL},
        {"blocks.0.attn.qkv_proj.lora_B.weight", "BF16", 2, {QKV_OUT, 8}, NULL}
    };
    static const fixture_tensor adaln[] = {
        PAIR_A("blocks.0.adaln_proj.linear", ADALN_IN),
        PAIR_B("blocks.0.adaln_proj.linear", ADALN_OUT),
        PAIR_A("final_layer.adaln_proj.linear", ADALN_IN),
        PAIR_B("final_layer.adaln_proj.linear", ADALN_OUT)
    };

    static const char *const names[FIXTURE_COUNT] = {
        "01-convention-a-prefixed.safetensors",
        "02-convention-a-bare.safetensors",
        "03-hybrid-alpha.safetensors",
        "04-mixed-ranks.safetensors",
        "05-convention-b.safetensors",
        "06-foreign-base-model.safetensors",
        "07-orphan-pairs.safetensors",
        "08-incoherent-pair.safetensors",
        "09-truncated.safetensors",
        "10-converted.safetensors",
        "11-adaln-pairs.safetensors",
        "12-convention-b-no-target.safetensors"
    };
    for (size_t index = 0; index < FIXTURE_COUNT; index++) {
        path_in(paths[index], 700, names[index]);
    }

    size_t minimal = sizeof(CONVENTION_A_PREFIXED) /
                     sizeof(*CONVENTION_A_PREFIXED);
    write_fixture(paths[0], NULL, CONVENTION_A_PREFIXED, minimal);
    write_fixture(paths[1], NULL, bare, sizeof(bare) / sizeof(*bare));
    write_fixture(paths[2], NULL, hybrid, sizeof(hybrid) / sizeof(*hybrid));
    write_fixture(paths[3], NULL, mixed, sizeof(mixed) / sizeof(*mixed));
    write_fixture(paths[4], NULL, convention_b,
                  sizeof(convention_b) / sizeof(*convention_b));
    write_fixture(paths[5], "\"base_model\":\"stabilityai/stable-diffusion-3\"",
                  CONVENTION_A_PREFIXED, minimal);
    write_fixture(paths[6], NULL, orphans, sizeof(orphans) / sizeof(*orphans));
    write_fixture(paths[7], NULL, incoherent,
                  sizeof(incoherent) / sizeof(*incoherent));
    /* Valid header, incomplete data: the last pair's bytes stop early. */
    write_fixture(paths[8], NULL, CONVENTION_A_PREFIXED, minimal);
    struct stat status;
    CHECK(stat(paths[8], &status) == 0 && status.st_size > 8);
    CHECK(truncate(paths[8], status.st_size - 4) == 0);
    write_fixture(paths[9], "\"partial_conversion\":\"true\","
                  "\"removed_pair_count\":\"51\","
                  "\"adaln_keys_removed\":\"true\"",
                  CONVENTION_A_PREFIXED, minimal);
    write_fixture(paths[10], NULL, adaln, sizeof(adaln) / sizeof(*adaln));
    write_fixture(paths[11], NULL, convention_b_no_target,
                  sizeof(convention_b_no_target) /
                  sizeof(*convention_b_no_target));
}

/* ---- the checks ---- */

static h3_lora_adapter *parse(const char *path, char *summary,
                              size_t summary_size) {
    capture_reset();
    return h3_lora_parse(path, transformer, summary, summary_size, capture,
                         NULL);
}

static const h3_lora_pair *find_pair(const h3_lora_adapter *adapter,
                                     const char *name) {
    for (size_t index = 0; index < adapter->pair_count; index++) {
        if (!strcmp(adapter->pairs[index].name, name)) {
            return &adapter->pairs[index];
        }
    }
    return NULL;
}

/* T3: the parser. Convention A with and without the prefix, alpha/rank as the
 * effective scale, mixed ranks in one file, convention B under its flattened
 * names, and a foreign base_model that warns instead of failing. */
static void test_parser(char paths[FIXTURE_COUNT][700]) {
    char summary[512];

    for (int variant = 0; variant < 2; variant++) {
        h3_lora_adapter *adapter = parse(paths[variant], summary,
                                         sizeof(summary));
        CHECK(adapter != NULL);
        CHECK(adapter->pair_count == 2);
        /* The prefix is stripped, so both files name the same two pairs. */
        const h3_lora_pair *qkv = find_pair(adapter, "blocks.0.attn.qkv_proj");
        CHECK(qkv != NULL);
        CHECK(qkv->rank == 4 && qkv->in_dim == QKV_IN &&
              qkv->out_dim == QKV_OUT);
        CHECK(qkv->scale == 1.0f);   /* no .alpha is scale 1.0 by definition */
        CHECK(find_pair(adapter, "blocks.0.attn.out_proj") != NULL);
        CHECK(adapter->adaln_pair_count == 0);
        h3_lora_adapter_free(adapter);
    }

    h3_lora_adapter *hybrid = parse(paths[2], summary, sizeof(summary));
    CHECK(hybrid != NULL);
    CHECK(hybrid->pair_count == 1);
    CHECK(hybrid->pairs[0].scale == 2.0f);   /* alpha 8 over rank 4 (H6) */
    h3_lora_adapter_free(hybrid);

    h3_lora_adapter *mixed = parse(paths[3], summary, sizeof(summary));
    CHECK(mixed != NULL);
    CHECK(mixed->pair_count == 2);
    CHECK(find_pair(mixed, "blocks.0.attn.qkv_proj")->rank == 4);
    CHECK(find_pair(mixed, "blocks.0.mlp.fc1")->rank == 8);
    h3_lora_adapter_free(mixed);

    /* Convention B loads, and every pair comes back under the checkpoint's own
     * dotted spelling: that is what makes a site match one. */
    h3_lora_adapter *flat = parse(paths[4], summary, sizeof(summary));
    CHECK(flat != NULL);
    CHECK(flat->pair_count == 2);
    const h3_lora_pair *flat_qkv = find_pair(flat, "blocks.0.attn.qkv_proj");
    CHECK(flat_qkv != NULL);
    CHECK(flat_qkv->rank == 4 && flat_qkv->in_dim == QKV_IN &&
          flat_qkv->out_dim == QKV_OUT);
    CHECK(flat_qkv->scale == 2.0f);   /* alpha 8 over rank 4, on down/up too */
    CHECK(find_pair(flat, "blocks.0.mlp.fc1") != NULL);
    CHECK(find_pair(flat, "blocks.0.mlp.fc1")->rank == 8);
    CHECK(flat->adaln_pair_count == 0);
    h3_lora_adapter_free(flat);

    /* A flattened name with no target is unapplicable, and the message keeps
     * the underscores the file wrote rather than guessing the dots back in.
     * The optional lora_unet_ prefix is stripped, the way diffusion_model. is
     * on convention A. */
    CHECK(parse(paths[11], summary, sizeof(summary)) == NULL);
    CHECK(strstr(summary, "1 pair cannot be applied") != NULL);
    CHECK(strstr(summary, "1 with no target") != NULL);
    CHECK(line_with("no target: blocks_51_attn_qkv_proj") != NULL);

    /* base_model is informative: a warning, and the load still succeeds. */
    h3_lora_adapter *foreign = parse(paths[5], summary, sizeof(summary));
    CHECK(foreign != NULL);
    capture_reset();
    h3_lora_emit_report(foreign, capture, NULL);
    const char *warning = line_with("base_model=");
    CHECK(warning != NULL);
    CHECK(strncmp(warning, "warning:", 8) == 0);
    CHECK(strstr(warning, "stabilityai/stable-diffusion-3") != NULL);
    h3_lora_adapter_free(foreign);
}

/* T3b: --lora PATH[:STRENGTH] splits on the LAST colon. */
static void test_command_line(char paths[FIXTURE_COUNT][700]) {
    char argument[800];
    const char *tail = NULL;
    h3_lora requested[2];

    /* A bare path keeps every character and defaults to 1.0. */
    snprintf(argument, sizeof(argument), "%s", paths[0]);
    CHECK(h3_lora_parse_argument(argument, &requested[0], &tail));
    CHECK(!strcmp(requested[0].path, paths[0]));
    CHECK(requested[0].strength == 1.0f);

    /* A path with a strength. */
    snprintf(argument, sizeof(argument), "%s:0.75", paths[0]);
    CHECK(h3_lora_parse_argument(argument, &requested[0], &tail));
    CHECK(!strcmp(requested[0].path, paths[0]));
    CHECK(requested[0].strength == 0.75f);

    /* The flag repeated: two independent entries, order preserved. */
    char first[800], second[800];
    snprintf(first, sizeof(first), "%s:0.25", paths[0]);
    snprintf(second, sizeof(second), "%s", paths[1]);
    CHECK(h3_lora_parse_argument(first, &requested[0], &tail));
    CHECK(h3_lora_parse_argument(second, &requested[1], &tail));
    CHECK(!strcmp(requested[0].path, paths[0]) &&
          requested[0].strength == 0.25f);
    CHECK(!strcmp(requested[1].path, paths[1]) &&
          requested[1].strength == 1.0f);

    /* The subtle one: a path that itself contains a colon. Fixture 1 written
     * again under a colon name, so the split is proved by opening the file
     * the split produced and not by reading the rule off the source. */
    char colon_path[700];
    path_in(colon_path, sizeof(colon_path), "12:00 turbo.safetensors");
    write_fixture(colon_path, NULL, CONVENTION_A_PREFIXED,
                  sizeof(CONVENTION_A_PREFIXED) /
                  sizeof(*CONVENTION_A_PREFIXED));
    snprintf(argument, sizeof(argument), "%s:0.5", colon_path);
    CHECK(h3_lora_parse_argument(argument, &requested[0], &tail));
    CHECK(!strcmp(requested[0].path, colon_path));
    CHECK(requested[0].strength == 0.5f);
    char summary[512];
    h3_lora_adapter *adapter = parse(requested[0].path, summary,
                                     sizeof(summary));
    CHECK(adapter != NULL);
    CHECK(adapter->pair_count == 2);
    h3_lora_adapter_free(adapter);

    /* The other half of the same rule: a tail that is not a finite number is
     * an error and never part of the path, so a colon path given without a
     * strength is refused with the tail quoted. */
    snprintf(argument, sizeof(argument), "%s", colon_path);
    CHECK(!h3_lora_parse_argument(argument, &requested[0], &tail));
    CHECK(tail != NULL && !strcmp(tail, "00 turbo.safetensors"));
    CHECK(!strcmp(argument, colon_path));   /* left intact for the message */
    snprintf(argument, sizeof(argument), "%s:0.8x", paths[0]);
    CHECK(!h3_lora_parse_argument(argument, &requested[0], &tail));
    CHECK(!strcmp(tail, "0.8x"));
    snprintf(argument, sizeof(argument), "%s:nan", paths[0]);
    CHECK(!h3_lora_parse_argument(argument, &requested[0], &tail));
}

/* T4: rejection and reporting. All three fatal, with distinct messages. */
static void test_rejection(char paths[FIXTURE_COUNT][700]) {
    char summary[512];

    /* Both orphans listed, not just the first (H5). */
    CHECK(parse(paths[6], summary, sizeof(summary)) == NULL);
    CHECK(strstr(summary, "2 pairs cannot be applied") != NULL);
    CHECK(strstr(summary, "2 with no target") != NULL);
    CHECK(strstr(summary, paths[6]) != NULL);   /* the path as written */
    CHECK(lines_with("no target:") == 2);
    CHECK(line_with("blocks.51.attn.qkv_proj") != NULL);
    CHECK(line_with("blocks.51.mlp.fc1") != NULL);
    CHECK(line_with("rank 4") != NULL);

    /* A pair that contradicts itself. */
    CHECK(parse(paths[7], summary, sizeof(summary)) == NULL);
    CHECK(strstr(summary, "internally inconsistent") != NULL);
    const char *detail = line_with("inconsistent pair:");
    CHECK(detail != NULL);
    CHECK(strstr(detail, "A rank 4 and B rank 8") != NULL);

    /* A truncated file says it is truncated instead of crashing. */
    CHECK(parse(paths[8], summary, sizeof(summary)) == NULL);
    CHECK(strstr(summary, "truncated file") != NULL);
}

/* T4b: the shape of the report. */
static void test_report(char paths[FIXTURE_COUNT][700]) {
    char summary[512];

    h3_lora_adapter *minimal = parse(paths[0], summary, sizeof(summary));
    CHECK(minimal != NULL);
    capture_reset();
    h3_lora_emit_report(minimal, capture, NULL);
    const char *line = line_with("pairs applied");
    CHECK(line != NULL);
    CHECK(strstr(line, "ranks: 4 x2") != NULL);      /* the histogram, not a range */
    CHECK(strstr(line, "AdaLN pairs: 0") != NULL);   /* declared even at zero */
    CHECK(line_with("warning:") == NULL);
    h3_lora_adapter_free(minimal);

    /* The histogram is ordered by descending count, and two ranks in one file
     * come out as two entries. */
    h3_lora_adapter *mixed = parse(paths[3], summary, sizeof(summary));
    CHECK(mixed != NULL);
    capture_reset();
    h3_lora_emit_report(mixed, capture, NULL);
    CHECK(line_with("ranks: 8 x1, 4 x1") != NULL);
    h3_lora_adapter_free(mixed);

    /* AdaLN pairs are counted, final_layer included. */
    h3_lora_adapter *adaln = parse(paths[10], summary, sizeof(summary));
    CHECK(adaln != NULL);
    CHECK(adaln->adaln_pair_count == 2);
    capture_reset();
    h3_lora_emit_report(adaln, capture, NULL);
    CHECK(line_with("AdaLN pairs: 2") != NULL);
    h3_lora_adapter_free(adaln);

    /* A conversion signature adds a warning line quoting the header verbatim. */
    h3_lora_adapter *converted = parse(paths[9], summary, sizeof(summary));
    CHECK(converted != NULL);
    capture_reset();
    h3_lora_emit_report(converted, capture, NULL);
    const char *warning = line_with("removed_pair_count=51");
    CHECK(warning != NULL);
    CHECK(strncmp(warning, "warning:", 8) == 0);
    CHECK(strstr(warning, "partial_conversion=true") != NULL);
    h3_lora_adapter_free(converted);
}

int main(void) {
    snprintf(root, sizeof(root), "/tmp/h3-lora-fixtures-XXXXXX");
    if (!mkdtemp(root)) {
        fprintf(stderr, "FAIL %s: cannot create a temporary directory\n",
                __FILE__);
        return 1;
    }
    atexit(cleanup);

    char paths[FIXTURE_COUNT][700];
    write_checkpoint();
    write_all(paths);

    test_parser(paths);
    test_command_line(paths);
    test_rejection(paths);
    test_report(paths);

    printf("ok: %d checks on %d synthetic LoRA fixtures\n", checks,
           FIXTURE_COUNT);
    return 0;
}
