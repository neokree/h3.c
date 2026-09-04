#include "h3_lora.h"

#include "h3_internal.h"

#include <dirent.h>
#include <errno.h>
#include <mach/mach.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define H3_LORA_A_SUFFIX ".lora_A.weight"
#define H3_LORA_B_SUFFIX ".lora_B.weight"
#define H3_LORA_ALPHA_SUFFIX ".alpha"
#define H3_LORA_BIAS_SUFFIX ".diff_b"
#define H3_LORA_PREFIX "diffusion_model."
#define H3_LORA_WEIGHT_SUFFIX ".weight"

/* SPEC 7.1: the four projections of a block plus the two AdaLN targets. A
 * name is a target only when it also exists in the checkpoint, so these five
 * suffixes cover blocks.N, token_refiner.blocks.N and final_layer alike. */
static const char *const h3_lora_target_suffixes[] = {
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "adaln_proj.linear.weight"
};

/* The pipeline LoRA applies to. Ref2VA is a separate DiT that this build has
 * no adapter corpus for; validating against FL2VA is what SPEC 7.1 measured.
 * ponytail: one pipeline, no selector until a Ref2VA LoRA exists. */
#define H3_LORA_TRANSFORMER_DIR "FL2VA/transformer"

enum {
    H3_LORA_OK = 0,
    H3_LORA_NO_TARGET,
    H3_LORA_SHAPE_MISMATCH,
    H3_LORA_NO_B,
    H3_LORA_RANK_DISAGREE,
    H3_LORA_NOT_MATRIX
};

typedef struct {
    char *name;      /* target name minus ".weight" */
    uint64_t rows;
    uint64_t cols;
    int adaln;
} h3_lora_target;

typedef struct {
    h3_lora_target *items;
    size_t count;
    size_t capacity;
} h3_lora_targets;

typedef struct {
    int status;
    uint64_t rows;   /* resolved target shape, for the mismatch message */
    uint64_t cols;
    const char *name;
} h3_lora_check;

#define H3_LORA_PLURAL(count) ((count) == 1 ? "" : "s")

static void h3_lora_line(h3_report_callback report, void *opaque,
                         const char *format, ...)
    __attribute__((format(printf, 3, 4)));

static void h3_lora_line(h3_report_callback report, void *opaque,
                         const char *format, ...) {
    if (!report) return;
    char line[512];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    report(line, opaque);
}

static int h3_lora_has_suffix(const char *value, const char *suffix) {
    size_t length = strlen(value);
    size_t suffix_length = strlen(suffix);
    return length > suffix_length &&
           !strcmp(value + length - suffix_length, suffix);
}

static const char *h3_lora_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/* ---- target index: shard headers only, no weight bytes ---- */

static int h3_lora_target_append(h3_lora_targets *targets,
                                 const h3_st_tensor *tensor) {
    size_t length = strlen(tensor->name) - strlen(H3_LORA_WEIGHT_SUFFIX);
    if (targets->count == targets->capacity) {
        size_t next = targets->capacity ? targets->capacity * 2 : 128;
        h3_lora_target *items = realloc(targets->items,
                                        next * sizeof(*items));
        if (!items) return 0;
        targets->items = items;
        targets->capacity = next;
    }
    char *name = malloc(length + 1);
    if (!name) return 0;
    memcpy(name, tensor->name, length);
    name[length] = '\0';
    h3_lora_target *target = &targets->items[targets->count++];
    target->name = name;
    target->rows = tensor->shape[0];
    target->cols = tensor->shape[1];
    target->adaln = h3_lora_has_suffix(tensor->name,
                                       "adaln_proj.linear.weight");
    return 1;
}

static int h3_lora_is_target(const char *name) {
    size_t count = sizeof(h3_lora_target_suffixes) /
                   sizeof(*h3_lora_target_suffixes);
    for (size_t index = 0; index < count; index++) {
        if (h3_lora_has_suffix(name, h3_lora_target_suffixes[index])) return 1;
    }
    return 0;
}

static void h3_lora_targets_free(h3_lora_targets *targets) {
    for (size_t index = 0; index < targets->count; index++) {
        free(targets->items[index].name);
    }
    free(targets->items);
    memset(targets, 0, sizeof(*targets));
}

static int h3_lora_targets_build(const char *directory,
                                 h3_lora_targets *targets,
                                 char *error, size_t error_size) {
    memset(targets, 0, sizeof(*targets));
    DIR *stream = opendir(directory);
    if (!stream) {
        snprintf(error, error_size, "%s: cannot read the transformer shards",
                 directory);
        return 0;
    }
    struct dirent *entry;
    int ok = 1;
    while (ok && (entry = readdir(stream)) != NULL) {
        if (entry->d_name[0] == '.' ||
            !h3_lora_has_suffix(entry->d_name, ".safetensors")) continue;
        size_t size = strlen(directory) + strlen(entry->d_name) + 2;
        char *path = malloc(size);
        if (!path) {
            snprintf(error, error_size, "out of memory indexing targets");
            ok = 0;
            break;
        }
        snprintf(path, size, "%s/%s", directory, entry->d_name);
        h3_st_header header;
        ok = h3_st_read_header(path, &header, error, error_size);
        free(path);
        if (!ok) break;
        for (size_t index = 0; index < header.tensor_count; index++) {
            const h3_st_tensor *tensor = &header.tensors[index];
            if (tensor->ndim != 2 || !h3_lora_is_target(tensor->name)) continue;
            if (!h3_lora_target_append(targets, tensor)) {
                snprintf(error, error_size, "out of memory indexing targets");
                ok = 0;
                break;
            }
        }
        h3_st_free_header(&header);
    }
    closedir(stream);
    if (ok && !targets->count) {
        snprintf(error, error_size, "%s: no adaptable weights in the shards",
                 directory);
        ok = 0;
    }
    if (!ok) h3_lora_targets_free(targets);
    return ok;
}

static const h3_lora_target *h3_lora_target_find(
        const h3_lora_targets *targets, const char *name) {
    for (size_t index = 0; index < targets->count; index++) {
        if (!strcmp(targets->items[index].name, name)) {
            return &targets->items[index];
        }
    }
    return NULL;
}

/* ---- pair assembly ---- */

static char *h3_lora_pair_name(const char *key, const char *suffix) {
    const char *start = key;
    size_t prefix_length = strlen(H3_LORA_PREFIX);
    /* The prefix is optional (SPEC 7.2), and stripping it must leave a name. */
    if (!strncmp(start, H3_LORA_PREFIX, prefix_length) &&
        strlen(start + prefix_length) > strlen(suffix)) {
        start += prefix_length;
    }
    size_t length = strlen(start) - strlen(suffix);
    char *name = malloc(length + 1);
    if (!name) return NULL;
    memcpy(name, start, length);
    name[length] = '\0';
    return name;
}

/* Same key with one suffix swapped for another. */
static char *h3_lora_sibling_key(const char *key, const char *suffix,
                                 const char *replacement) {
    size_t stem = strlen(key) - strlen(suffix);
    char *sibling = malloc(stem + strlen(replacement) + 1);
    if (!sibling) return NULL;
    memcpy(sibling, key, stem);
    memcpy(sibling + stem, replacement, strlen(replacement) + 1);
    return sibling;
}

static float h3_lora_half_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (uint32_t)((bits >> 10) & 0x1fu);
    uint32_t mantissa = (uint32_t)(bits & 0x3ffu);
    uint32_t out;
    if (!exponent) {
        if (!mantissa) {
            out = sign;
        } else {
            exponent = 127u - 15u + 1u;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ffu;
            out = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

/* Read the scalar .alpha of one pair. It is a handful of bytes, not a weight:
 * H6 requires the file's own scale, and only the file has it. *found reports
 * the sibling tensor when there is one, so the caller can tell "no .alpha"
 * (scale 1.0 by definition) from "an .alpha h3 cannot read" (H6 violated in
 * silence otherwise). */
static int h3_lora_read_alpha(const h3_st_header *header, const char *key,
                              double *alpha, const h3_st_tensor **found) {
    char *alpha_key = h3_lora_sibling_key(key, H3_LORA_A_SUFFIX,
                                          H3_LORA_ALPHA_SUFFIX);
    if (!alpha_key) return 0;
    const h3_st_tensor *tensor = h3_st_find(header, alpha_key);
    free(alpha_key);
    if (!tensor) return 0;
    *found = tensor;
    if (h3_st_tensor_elements(tensor) != 1) return 0;
    unsigned char bytes[8] = {0};
    size_t size = h3_dtype_size(tensor->dtype);
    if (!size || size > sizeof(bytes)) return 0;
    if (!h3_st_read_data(header, tensor, bytes, size, NULL, 0)) return 0;
    switch (tensor->dtype) {
        case H3_DTYPE_F32: {
            float value;
            memcpy(&value, bytes, sizeof(value));
            *alpha = (double)value;
            return 1;
        }
        case H3_DTYPE_F64: {
            double value;
            memcpy(&value, bytes, sizeof(value));
            *alpha = value;
            return 1;
        }
        case H3_DTYPE_BF16: {
            uint16_t half;
            memcpy(&half, bytes, sizeof(half));
            uint32_t widened = (uint32_t)half << 16;
            float value;
            memcpy(&value, &widened, sizeof(value));
            *alpha = (double)value;
            return 1;
        }
        case H3_DTYPE_F16: {
            uint16_t half;
            memcpy(&half, bytes, sizeof(half));
            *alpha = (double)h3_lora_half_to_float(half);
            return 1;
        }
        case H3_DTYPE_I32: {
            int32_t value;
            memcpy(&value, bytes, sizeof(value));
            *alpha = (double)value;
            return 1;
        }
        case H3_DTYPE_I64: {
            int64_t value;
            memcpy(&value, bytes, sizeof(value));
            *alpha = (double)value;
            return 1;
        }
        default: return 0;
    }
}

/* ---- the command line ---- */

int h3_lora_parse_argument(char *argument, h3_lora *lora, const char **tail) {
    lora->path = argument;
    lora->strength = 1.0f;
    char *colon = strrchr(argument, ':');
    if (!colon) return 1;
    if (tail) *tail = colon + 1;
    char *end = NULL;
    errno = 0;
    float strength = strtof(colon + 1, &end);
    if (errno || !end || end == colon + 1 || *end || !isfinite(strength)) {
        return 0;
    }
    *colon = '\0';
    lora->strength = strength;
    return 1;
}

/* ---- parse ---- */

static void h3_lora_free_pairs(h3_lora_pair *pairs, size_t count) {
    for (size_t index = 0; index < count; index++) free(pairs[index].name);
    free(pairs);
}

h3_lora_adapter *h3_lora_parse(const char *path, const char *transformer_dir,
                               char *summary, size_t summary_size,
                               h3_report_callback report, void *opaque) {
    if (summary && summary_size) summary[0] = '\0';
    h3_st_header header;
    char detail[512];
    if (!h3_st_read_header(path, &header, detail, sizeof(detail))) {
        /* h3_safetensors rejects data offsets past the end of the file, which
         * is exactly what a truncated file looks like from the header. */
        if (strstr(detail, "exceed file")) {
            snprintf(summary, summary_size,
                     "%s: truncated file: tensor data ends past the end of "
                     "the file", path);
        } else if (!strncmp(detail, path, strlen(path))) {
            snprintf(summary, summary_size, "%s", detail);
        } else {
            snprintf(summary, summary_size, "%s: %s", path, detail);
        }
        return NULL;
    }

    /* Convention is a property of the file, so one offender is the proof. */
    const char *bias_key = NULL;
    const char *unknown_key = NULL;
    for (size_t index = 0; index < header.tensor_count; index++) {
        const char *key = header.tensors[index].name;
        if (strstr(key, "lora_up") || strstr(key, "lora_down")) {
            snprintf(summary, summary_size,
                     "%s: lora_up/lora_down naming is not supported", path);
            h3_lora_line(report, opaque,
                         "  (first seen at %s); h3 reads lora_A/lora_B", key);
            h3_st_free_header(&header);
            return NULL;
        }
        if (!bias_key && h3_lora_has_suffix(key, H3_LORA_BIAS_SUFFIX)) {
            bias_key = key;
        }
        if (!unknown_key &&
            !h3_lora_has_suffix(key, H3_LORA_A_SUFFIX) &&
            !h3_lora_has_suffix(key, H3_LORA_B_SUFFIX) &&
            !h3_lora_has_suffix(key, H3_LORA_ALPHA_SUFFIX) &&
            !h3_lora_has_suffix(key, H3_LORA_BIAS_SUFFIX)) {
            unknown_key = key;
        }
    }

    /* SPEC 7.3: a suffix h3 does not read is fatal and named by category, the
     * way llama.cpp throws (src/llama-adapter.cpp:291). Dropping it would load
     * a DoRA-style file as a plain LoRA and report nothing (H5). One offender,
     * because the extra key is a property of the file. */
    if (unknown_key) {
        const char *suffix = strrchr(unknown_key, '.');
        snprintf(summary, summary_size, "%s: %s keys are not supported", path,
                 suffix ? suffix : unknown_key);
        h3_lora_line(report, opaque,
                     "  (first seen at %s); h3 reads lora_A/lora_B pairs with "
                     "an optional .alpha", unknown_key);
        h3_st_free_header(&header);
        return NULL;
    }

    h3_lora_targets targets;
    if (!h3_lora_targets_build(transformer_dir, &targets, detail,
                               sizeof(detail))) {
        snprintf(summary, summary_size, "%s", detail);
        h3_st_free_header(&header);
        return NULL;
    }

    size_t capacity = 0;
    for (size_t index = 0; index < header.tensor_count; index++) {
        if (h3_lora_has_suffix(header.tensors[index].name, H3_LORA_A_SUFFIX)) {
            capacity++;
        }
    }
    /* Every lora_B without a lora_A is one more incoherent pair. */
    size_t orphan_b = 0;
    for (size_t index = 0; index < header.tensor_count; index++) {
        const char *key = header.tensors[index].name;
        if (!h3_lora_has_suffix(key, H3_LORA_B_SUFFIX)) continue;
        char *sibling = h3_lora_sibling_key(key, H3_LORA_B_SUFFIX,
                                            H3_LORA_A_SUFFIX);
        if (!sibling) continue;
        if (!h3_st_find(&header, sibling)) orphan_b++;
        free(sibling);
    }

    h3_lora_pair *pairs = capacity ? calloc(capacity, sizeof(*pairs)) : NULL;
    h3_lora_check *checks = capacity ? calloc(capacity, sizeof(*checks)) : NULL;
    if (capacity && (!pairs || !checks)) {
        free(pairs);
        free(checks);
        h3_lora_targets_free(&targets);
        h3_st_free_header(&header);
        snprintf(summary, summary_size, "%s: out of memory reading pairs",
                 path);
        return NULL;
    }

    size_t pair_count = 0;
    size_t no_target = 0, shape_mismatch = 0, incoherent = orphan_b;
    size_t adaln_pairs = 0;
    uint64_t resident = 0;
    for (size_t index = 0; index < header.tensor_count; index++) {
        const h3_st_tensor *a = &header.tensors[index];
        if (!h3_lora_has_suffix(a->name, H3_LORA_A_SUFFIX)) continue;
        h3_lora_pair *pair = &pairs[pair_count];
        h3_lora_check *check = &checks[pair_count];
        pair_count++;
        pair->name = h3_lora_pair_name(a->name, H3_LORA_A_SUFFIX);
        if (!pair->name) {
            snprintf(summary, summary_size, "%s: out of memory reading pairs",
                     path);
            goto failed;
        }
        check->name = pair->name;
        pair->a = *a;
        char *b_key = h3_lora_sibling_key(a->name, H3_LORA_A_SUFFIX,
                                          H3_LORA_B_SUFFIX);
        const h3_st_tensor *b = b_key ? h3_st_find(&header, b_key) : NULL;
        free(b_key);
        if (!b) {
            check->status = H3_LORA_NO_B;
            incoherent++;
            continue;
        }
        pair->b = *b;
        if (a->ndim != 2 || b->ndim != 2) {
            check->status = H3_LORA_NOT_MATRIX;
            incoherent++;
            continue;
        }
        pair->rank = a->shape[0];
        pair->in_dim = a->shape[1];
        pair->out_dim = b->shape[0];
        if (b->shape[1] != pair->rank) {
            check->status = H3_LORA_RANK_DISAGREE;
            incoherent++;
            continue;
        }
        const h3_lora_target *target = h3_lora_target_find(&targets,
                                                           pair->name);
        if (!target) {
            check->status = H3_LORA_NO_TARGET;
            no_target++;
            continue;
        }
        check->rows = target->rows;
        check->cols = target->cols;
        if (target->rows != pair->out_dim || target->cols != pair->in_dim) {
            check->status = H3_LORA_SHAPE_MISMATCH;
            shape_mismatch++;
            continue;
        }
        pair->adaln = target->adaln;
        if (pair->adaln) adaln_pairs++;
        double alpha = 0.0;
        const h3_st_tensor *alpha_tensor = NULL;
        pair->scale = 1.0f;
        if (pair->rank && h3_lora_read_alpha(&header, a->name, &alpha,
                                             &alpha_tensor)) {
            pair->scale = (float)(alpha / (double)pair->rank);
        } else if (alpha_tensor) {
            /* An unreadable .alpha changes the effective strength, so say so
             * instead of assuming 1.0 in silence (H6). */
            h3_lora_line(report, opaque,
                         "warning: %s: %s is %s, which h3 cannot read as a "
                         "scale; using scale 1.0", path, alpha_tensor->name,
                         h3_dtype_name(alpha_tensor->dtype));
        }
        resident += (a->data_end - a->data_begin) + (b->data_end - b->data_begin);
    }

    if (incoherent) {
        snprintf(summary, summary_size,
                 "%s: %zu pair%s internally inconsistent, load aborted", path,
                 incoherent, incoherent == 1 ? " is" : "s are");
        for (size_t index = 0; index < pair_count; index++) {
            const h3_lora_check *check = &checks[index];
            if (check->status == H3_LORA_NO_B) {
                h3_lora_line(report, opaque,
                             "  inconsistent pair: %s has no lora_B",
                             check->name);
            } else if (check->status == H3_LORA_NOT_MATRIX) {
                h3_lora_line(report, opaque,
                             "  inconsistent pair: %s is not a matrix pair",
                             check->name);
            } else if (check->status == H3_LORA_RANK_DISAGREE) {
                h3_lora_line(report, opaque,
                             "  inconsistent pair: %s has A rank %llu and "
                             "B rank %llu", check->name,
                             (unsigned long long)pairs[index].a.shape[0],
                             (unsigned long long)pairs[index].b.shape[1]);
            }
        }
        for (size_t index = 0; index < header.tensor_count; index++) {
            const char *key = header.tensors[index].name;
            if (!h3_lora_has_suffix(key, H3_LORA_B_SUFFIX)) continue;
            char *sibling = h3_lora_sibling_key(key, H3_LORA_B_SUFFIX,
                                                H3_LORA_A_SUFFIX);
            if (!sibling) continue;
            int missing = h3_st_find(&header, sibling) == NULL;
            free(sibling);
            if (!missing) continue;
            char *name = h3_lora_pair_name(key, H3_LORA_B_SUFFIX);
            h3_lora_line(report, opaque,
                         "  inconsistent pair: %s has no lora_A",
                         name ? name : key);
            free(name);
        }
        goto failed;
    }

    if (no_target || shape_mismatch) {
        size_t total = no_target + shape_mismatch;
        if (no_target && shape_mismatch) {
            snprintf(summary, summary_size,
                     "%s: %zu pairs cannot be applied (%zu with no target, "
                     "%zu with a shape mismatch)", path, total, no_target,
                     shape_mismatch);
        } else if (no_target) {
            snprintf(summary, summary_size,
                     "%s: %zu pair%s cannot be applied (%zu with no target)",
                     path, total, H3_LORA_PLURAL(total), no_target);
        } else {
            snprintf(summary, summary_size,
                     "%s: %zu pair%s cannot be applied (%zu with a shape "
                     "mismatch)", path, total, H3_LORA_PLURAL(total),
                     shape_mismatch);
        }
        h3_lora_line(report, opaque, "%s: %zu unapplicable pair%s, load "
                     "aborted", path, total, H3_LORA_PLURAL(total));
        for (size_t index = 0; index < pair_count; index++) {
            const h3_lora_check *check = &checks[index];
            if (check->status == H3_LORA_NO_TARGET) {
                h3_lora_line(report, opaque, "  no target: %s (rank %llu)",
                             check->name,
                             (unsigned long long)pairs[index].rank);
            } else if (check->status == H3_LORA_SHAPE_MISMATCH) {
                h3_lora_line(report, opaque,
                             "  shape mismatch: %s is [%llu x %llu], target "
                             "is [%llu x %llu]", check->name,
                             (unsigned long long)pairs[index].out_dim,
                             (unsigned long long)pairs[index].in_dim,
                             (unsigned long long)check->rows,
                             (unsigned long long)check->cols);
            }
        }
        goto failed;
    }

    if (bias_key) {
        snprintf(summary, summary_size,
                 "%s: bias adapters are not supported", path);
        h3_lora_line(report, opaque,
                     "  (first seen at %s); h3 applies low-rank pairs only",
                     bias_key);
        goto failed;
    }

    h3_lora_adapter *adapter = calloc(1, sizeof(*adapter));
    char *path_copy = strdup(path);
    if (!adapter || !path_copy) {
        free(adapter);
        free(path_copy);
        snprintf(summary, summary_size, "%s: out of memory loading LoRA",
                 path);
        goto failed;
    }
    adapter->path = path_copy;
    adapter->header = header;
    adapter->pairs = pairs;
    adapter->pair_count = pair_count;
    adapter->adaln_pair_count = adaln_pairs;
    adapter->resident_bytes = resident;
    adapter->size = header.file_size;
    free(checks);
    h3_lora_targets_free(&targets);
    return adapter;

failed:
    h3_lora_free_pairs(pairs, pair_count);
    free(checks);
    h3_lora_targets_free(&targets);
    h3_st_free_header(&header);
    return NULL;
}

void h3_lora_adapter_free(h3_lora_adapter *adapter) {
    if (!adapter) return;
    h3_lora_free_pairs(adapter->pairs, adapter->pair_count);
    h3_st_free_header(&adapter->header);
    free(adapter->path);
    free(adapter);
}

/* ---- the activation report (SPEC 5.1) ---- */

static const char *const h3_lora_signature_keys[] = {
    "partial_conversion", "removed_pair_count", "adaln_keys_removed"
};

void h3_lora_emit_report(const h3_lora_adapter *adapter,
                         h3_report_callback report, void *opaque) {
    if (!adapter || !report) return;

    /* Rank histogram, ordered by descending count (SPEC 5.1). */
    uint64_t ranks[64];
    size_t counts[64];
    size_t distinct = 0;
    size_t unshown = 0;  /* pairs the histogram does not account for */
    for (size_t index = 0; index < adapter->pair_count; index++) {
        uint64_t rank = adapter->pairs[index].rank;
        size_t slot = 0;
        while (slot < distinct && ranks[slot] != rank) slot++;
        if (slot == distinct) {
            if (distinct == sizeof(ranks) / sizeof(*ranks)) {
                unshown++;
                continue;
            }
            ranks[distinct] = rank;
            counts[distinct] = 0;
            distinct++;
        }
        counts[slot]++;
    }
    char histogram[256];
    size_t written = 0;
    histogram[0] = '\0';
    for (size_t emitted = 0; emitted < distinct; emitted++) {
        size_t best = distinct;
        for (size_t slot = 0; slot < distinct; slot++) {
            if (!counts[slot]) continue;
            if (best == distinct || counts[slot] > counts[best] ||
                (counts[slot] == counts[best] && ranks[slot] > ranks[best])) {
                best = slot;
            }
        }
        if (best == distinct) break;
        int needed = snprintf(histogram + written, sizeof(histogram) - written,
                              "%s%llu x%zu", written ? ", " : "",
                              (unsigned long long)ranks[best], counts[best]);
        if (needed <= 0 || (size_t)needed >= sizeof(histogram) - written) {
            /* Out of room: drop the truncated write and stop, so the tail
             * below can say how many pairs the line leaves out. */
            histogram[written] = '\0';
            break;
        }
        written += (size_t)needed;
        counts[best] = 0;
    }
    /* Whatever is still counted was never printed. */
    for (size_t slot = 0; slot < distinct; slot++) unshown += counts[slot];
    char tail[64];
    tail[0] = '\0';
    if (unshown) {
        snprintf(tail, sizeof(tail), ", +%zu pair%s not shown", unshown,
                 H3_LORA_PLURAL(unshown));
    }

    h3_lora_line(report, opaque,
                 "%s: %zu pair%s applied, ranks: %s%s, AdaLN pairs: %zu, "
                 "resident %.1f MiB", adapter->path, adapter->pair_count,
                 H3_LORA_PLURAL(adapter->pair_count),
                 written ? histogram : "none", tail,
                 adapter->adaln_pair_count,
                 (double)adapter->resident_bytes / (1024.0 * 1024.0));

    /* A converted file knows what was taken out of it, so quote it verbatim. */
    char signature[256];
    size_t used = 0;
    signature[0] = '\0';
    size_t key_count = sizeof(h3_lora_signature_keys) /
                       sizeof(*h3_lora_signature_keys);
    for (size_t index = 0; index < key_count; index++) {
        const char *value = h3_st_metadata(&adapter->header,
                                           h3_lora_signature_keys[index]);
        if (!value) continue;
        int needed = snprintf(signature + used, sizeof(signature) - used,
                              "%s%s=%s", used ? ", " : "",
                              h3_lora_signature_keys[index], value);
        if (needed > 0 && (size_t)needed < sizeof(signature) - used) {
            used += (size_t)needed;
        }
    }
    if (used) {
        h3_lora_line(report, opaque,
                     "warning: this file was converted: %s", signature);
    }

    /* base_model is informative and written by the converter: warn, never
     * fail (SPEC 8, T3). */
    const char *base = h3_st_metadata(&adapter->header, "base_model");
    if (!base) {
        base = h3_st_metadata(&adapter->header, "ss_base_model_version");
    }
    if (base) {
        char folded[64];
        size_t length = 0;
        for (const char *at = base; *at && length + 1 < sizeof(folded); at++) {
            if (*at == '-' || *at == '_' || *at == ' ') continue;
            char lower = *at;
            if (lower >= 'A' && lower <= 'Z') lower = (char)(lower + 32);
            folded[length++] = lower;
        }
        folded[length] = '\0';
        if (!strstr(folded, "minimaxh3")) {
            h3_lora_line(report, opaque,
                         "warning: base_model=%s does not name MiniMax-H3",
                         base);
        }
    }
}

/* ---- the h3_ctx cache (SPEC 6ter.5) ---- */

static char *h3_lora_transformer_dir(h3_ctx *ctx) {
    if (!ctx || !ctx->model_dir) return NULL;
    size_t size = strlen(ctx->model_dir) + strlen(H3_LORA_TRANSFORMER_DIR) + 2;
    char *path = malloc(size);
    if (path) {
        snprintf(path, size, "%s/%s", ctx->model_dir,
                 H3_LORA_TRANSFORMER_DIR);
    }
    return path;
}

static void h3_lora_unlink(h3_ctx *ctx, h3_lora_adapter *adapter) {
    h3_lora_adapter **link = &ctx->loras;
    while (*link && *link != adapter) link = &(*link)->next;
    if (*link) *link = adapter->next;
    h3_lora_adapter_free(adapter);
}

static const h3_lora_adapter *h3_lora_acquire(h3_ctx *ctx, const char *path,
                                              h3_report_callback report,
                                              void *opaque) {
    struct stat status;
    if (stat(path, &status) != 0) {
        h3_set_error(ctx, "%s: cannot stat the LoRA file", path);
        return NULL;
    }
    for (h3_lora_adapter *adapter = ctx->loras; adapter;
         adapter = adapter->next) {
        if (strcmp(adapter->path, path)) continue;
        if (adapter->size == (uint64_t)status.st_size &&
            adapter->mtime_seconds == (int64_t)status.st_mtimespec.tv_sec &&
            adapter->mtime_nanoseconds ==
                (int64_t)status.st_mtimespec.tv_nsec) {
            return adapter;
        }
        /* Same path, different file: reload instead of serving the old one. */
        h3_lora_unlink(ctx, adapter);
        break;
    }
    char *transformer_dir = h3_lora_transformer_dir(ctx);
    if (!transformer_dir) {
        h3_set_error(ctx, "out of memory resolving the transformer directory");
        return NULL;
    }
    char summary[512];
    h3_lora_adapter *adapter = h3_lora_parse(path, transformer_dir, summary,
                                             sizeof(summary), report, opaque);
    free(transformer_dir);
    if (!adapter) {
        h3_set_error(ctx, "%s", summary);
        return NULL;
    }
    adapter->size = (uint64_t)status.st_size;
    adapter->mtime_seconds = (int64_t)status.st_mtimespec.tv_sec;
    adapter->mtime_nanoseconds = (int64_t)status.st_mtimespec.tv_nsec;
    adapter->next = ctx->loras;
    ctx->loras = adapter;
    h3_lora_emit_report(adapter, report, opaque);
    return adapter;
}

int h3_lora_preload(h3_ctx *ctx, const char *path,
                    h3_report_callback report, void *opaque) {
    if (!ctx || !path) return 0;
    return h3_lora_acquire(ctx, path, report, opaque) != NULL;
}

void h3_lora_release(h3_ctx *ctx, const char *path) {
    if (!ctx || !path) return;
    for (h3_lora_adapter *adapter = ctx->loras; adapter;
         adapter = adapter->next) {
        if (!strcmp(adapter->path, path)) {
            h3_lora_unlink(ctx, adapter);
            return;
        }
    }
}

void h3_lora_release_all(h3_ctx *ctx) {
    if (!ctx) return;
    h3_lora_adapter *adapter = ctx->loras;
    ctx->loras = NULL;
    while (adapter) {
        h3_lora_adapter *next = adapter->next;
        h3_lora_adapter_free(adapter);
        adapter = next;
    }
}

/* ---- the active set ---- */

int h3_lora_set_build(h3_ctx *ctx, const h3_params *params,
                      h3_lora_set *set) {
    memset(set, 0, sizeof(*set));
    if (!ctx || !params || !params->lora_count) return 1;
    if (!params->loras) {
        h3_set_error(ctx, "lora list is missing");
        return 0;
    }
    set->entries = calloc(params->lora_count, sizeof(*set->entries));
    if (!set->entries) {
        h3_set_error(ctx, "out of memory building the active LoRA set");
        return 0;
    }
    for (size_t index = 0; index < params->lora_count; index++) {
        const h3_lora *request = &params->loras[index];
        if (!request->path || !request->path[0]) {
            h3_set_error(ctx, "lora entry %zu has no path", index + 1);
            goto failed;
        }
        if (!isfinite(request->strength)) {
            h3_set_error(ctx, "lora: strength for %s is not a finite number",
                         h3_lora_basename(request->path));
            goto failed;
        }
        const h3_lora_adapter *adapter = h3_lora_acquire(
            ctx, request->path, params->on_report, params->callback_opaque);
        if (!adapter) goto failed;
        /* Validated, then dropped: this is what makes H4 structural. */
        if (request->strength == 0.0f) {
            h3_lora_line(params->on_report, params->callback_opaque,
                         "%s: strength 0, not applied", request->path);
            continue;
        }
        set->entries[set->count].adapter = adapter;
        set->entries[set->count].strength = request->strength;
        set->count++;
    }
    return 1;

failed:
    h3_lora_set_free(set);
    return 0;
}

void h3_lora_set_free(h3_lora_set *set) {
    if (!set) return;
    free(set->entries);
    memset(set, 0, sizeof(*set));
}

/* ---- the GPU delta branch (SPEC 7bis.4) ---- */

static float h3_lora_bf16_to_float(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, sizeof(value));
    return value;
}

/* Round to nearest even, matching the GPU casts. */
static uint16_t h3_lora_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (((bits >> 23) & 0xffu) == 0xffu) return (uint16_t)(bits >> 16);
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

/* The pair name resolves the site by itself: an AdaLN pair is named
 * "blocks.N.adaln_proj.linear" and can never collide with the four per-block
 * projections, so one builder serves both kinds of target (SPEC 7bis.3). */
static int h3_lora_pair_targets(const h3_lora_pair *pair, const char *name) {
    return !strcmp(pair->name, name);
}

void h3_lora_set_extents(const h3_lora_set *set, uint64_t *max_rank,
                         uint64_t *max_out_dim) {
    *max_rank = 0;
    *max_out_dim = 0;
    if (!set) return;
    for (size_t index = 0; index < set->count; index++) {
        const h3_lora_adapter *adapter = set->entries[index].adapter;
        for (size_t which = 0; which < adapter->pair_count; which++) {
            const h3_lora_pair *pair = &adapter->pairs[which];
            if (pair->adaln) continue;
            if (pair->rank > *max_rank) *max_rank = pair->rank;
            if (pair->out_dim > *max_out_dim) *max_out_dim = pair->out_dim;
        }
    }
}

static int h3_lora_branch_build(const h3_lora_entry *entry,
                                const h3_lora_pair *pair, h3_gpu *gpu,
                                h3_lora_branch *branch,
                                char *error, size_t error_size) {
    size_t a_count = (size_t)pair->rank * pair->in_dim;
    size_t b_count = (size_t)pair->out_dim * pair->rank;
    uint16_t *a = malloc(a_count * sizeof(*a));
    uint16_t *b = malloc(b_count * sizeof(*b));
    int ok = a && b;
    if (!ok) {
        snprintf(error, error_size, "out of memory materialising LoRA %s",
                 pair->name);
    } else {
        ok = h3_st_read_data(&entry->adapter->header, &pair->a, a,
                             a_count * sizeof(*a), error, error_size) &&
             h3_st_read_data(&entry->adapter->header, &pair->b, b,
                             b_count * sizeof(*b), error, error_size);
    }
    if (ok) {
        /* SPEC 7bis.4: strength * alpha/rank is fused here, into this bf16
         * copy of A. Never into the cached adapter, which is keyed by path,
         * size and mtime, and never at dispatch time. */
        float scale = entry->strength * pair->scale;
        for (size_t index = 0; index < a_count; index++) {
            a[index] = h3_lora_float_to_bf16(
                h3_lora_bf16_to_float(a[index]) * scale);
        }
        branch->a = h3_gpu_tensor_from_bf16(gpu, a, a_count);
        branch->b = h3_gpu_tensor_from_bf16(gpu, b, b_count);
        branch->rank = (uint32_t)pair->rank;
        if (!branch->a || !branch->b) {
            snprintf(error, error_size, "cannot upload LoRA %s: %s",
                     pair->name, h3_gpu_error(gpu));
            /* The caller frees the branches it already has; this half-built
             * one is not among them, so it releases itself. */
            h3_gpu_tensor_free(branch->a);
            h3_gpu_tensor_free(branch->b);
            memset(branch, 0, sizeof(*branch));
            ok = 0;
        }
    }
    free(a);
    free(b);
    return ok;
}

int h3_lora_site_build(const h3_lora_set *set, h3_gpu *gpu, const char *name,
                       h3_lora_site *site, char *error, size_t error_size) {
    memset(site, 0, sizeof(*site));
    if (!set || !set->count) return 1;
    unsigned matches = 0;
    for (size_t index = 0; index < set->count; index++) {
        const h3_lora_adapter *adapter = set->entries[index].adapter;
        for (size_t which = 0; which < adapter->pair_count; which++) {
            if (h3_lora_pair_targets(&adapter->pairs[which], name))
                matches++;
        }
    }
    if (!matches) return 1;
    site->branches = calloc(matches, sizeof(*site->branches));
    if (!site->branches) {
        snprintf(error, error_size, "out of memory building LoRA site %s",
                 name);
        return 0;
    }
    for (size_t index = 0; index < set->count; index++) {
        const h3_lora_entry *entry = &set->entries[index];
        for (size_t which = 0; which < entry->adapter->pair_count; which++) {
            const h3_lora_pair *pair = &entry->adapter->pairs[which];
            if (!h3_lora_pair_targets(pair, name)) continue;
            if (!h3_lora_branch_build(entry, pair, gpu,
                                      &site->branches[site->count],
                                      error, error_size)) {
                h3_lora_site_free(site);
                return 0;
            }
            site->count++;
        }
    }
    return 1;
}

void h3_lora_site_free(h3_lora_site *site) {
    if (!site) return;
    for (unsigned index = 0; index < site->count; index++) {
        h3_gpu_tensor_free(site->branches[index].a);
        h3_gpu_tensor_free(site->branches[index].b);
    }
    free(site->branches);
    memset(site, 0, sizeof(*site));
}

/* ---- the G4 memory guardrail (SPEC 10, G4) ----
 *
 * The ceiling is on the whole process and both gates are always active, empty
 * active set included. At the ceiling h3 stops: never a partial active set. */

/* The only fixed number of the whole design. No flag, no preset, no override:
 * it says what must stay standing while h3 holds 22 GB, not what h3 needs. */
#define H3_MEMORY_RESERVE_BYTES (UINT64_C(4) << 30)

/* SPEC 10, "l'aritmetica di supporto": with --ssd-streaming the DiT holds the
 * block norms plus two alternating BF16 matrix slots. One slot is
 * 21504*5376 + 5376*7168 + 28672*5376 + 5376*14336 = 385,351,680 elements. */
#define H3_MEMORY_WEIGHT_SLOT_BYTES (UINT64_C(385351680) * 2)

/* The tiled video VAE decoder: 9.37 GiB measured at 448x576 and 9.55 GiB at
 * 768x1344, 2% more for 4x the pixels, so it is charged as the
 * canvas-independent constant it is (AGENTS.md). It is also the tallest of the
 * serialised stages: the encoder block peaks between 4.7 and 6.0 GB. */
#define H3_MEMORY_VAE_DECODER_BYTES UINT64_C(10254484275) /* 9.55 GiB */

static double h3_memory_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

/* Pages another process holds active or wired are not free, which is exactly
 * the case the static term is blind to. 0 means the query failed. */
static uint64_t h3_memory_system_free(void) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) != KERN_SUCCESS) return 0;
    return ((uint64_t)vm.free_count + (uint64_t)vm.inactive_count +
            (uint64_t)vm.purgeable_count) * (uint64_t)vm_kernel_page_size;
}

/* The same number `footprint -p <pid>` reads. 0 on failure, which lets the run
 * continue rather than stopping it on a broken measurement. */
static uint64_t h3_memory_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS) return 0;
    return (uint64_t)info.phys_footprint;
}

void h3_memory_ceiling_take(uint64_t recommended_working_set,
                            h3_memory_ceiling *ceiling) {
    memset(ceiling, 0, sizeof(*ceiling));
    ceiling->working_set = recommended_working_set;
    ceiling->system_free = h3_memory_system_free();
    ceiling->footprint = h3_memory_footprint();
    uint64_t working_set_term =
        recommended_working_set > H3_MEMORY_RESERVE_BYTES ?
        recommended_working_set - H3_MEMORY_RESERVE_BYTES : 0;
    /* Without a system reading the static term stands alone; it is never the
     * unmeasured term that decides. */
    uint64_t system_free_term = ceiling->system_free ?
        ceiling->system_free + ceiling->footprint : working_set_term;
    ceiling->system_free_bit = system_free_term < working_set_term;
    ceiling->ceiling = ceiling->system_free_bit ?
        system_free_term : working_set_term;
}

/* Both stop strings must name which of the two min() terms bit, because the
 * remedy differs: the Metal working set wants a smaller canvas, system free
 * wants the co-resident process gone. The term is named by the physical thing,
 * never by our project words, and the gates are never numbered: "gate 1" and
 * "gate 2" are our vocabulary too (SPEC 5bis.5). Returns the remedy line. */
static const char *h3_memory_term(const h3_memory_ceiling *ceiling,
                                  char *term, size_t term_size) {
    if (ceiling->system_free_bit) {
        snprintf(term, term_size, "system free %.1f GiB plus our %.1f GiB",
                 h3_memory_gib(ceiling->system_free),
                 h3_memory_gib(ceiling->footprint));
        return "Another process is holding memory: quit it, or lower the "
               "canvas.";
    }
    snprintf(term, term_size,
             "Metal working set %.1f GiB minus the 4 GiB reserve",
             h3_memory_gib(ceiling->working_set));
    return "Lower the canvas or drop an adapter.";
}

int h3_memory_gate_preflight(const h3_memory_ceiling *ceiling,
                             const h3_lora_set *set, int stages_overlap,
                             char *error, size_t error_size) {
    uint64_t adapters = 0;
    for (size_t index = 0; set && index < set->count; index++)
        adapters += set->entries[index].adapter->resident_bytes;
    /* A necessary and NOT sufficient condition: it models no canvas-dependent
     * term at all, and the denoise footprint is where that term lives (8.92 GB
     * at 448x576 against 22.5 GB at 768x1344, for weight slots that do not move
     * a byte between the two). Only gate 2 sees it. What is summed here is
     * exact and canvas-independent: the tiled decoder, the adapters' resident
     * bytes read from the safetensors headers, and, when the stages overlap
     * instead of being serialised (h3.c:1591), the two weight slots. */
    uint64_t needed = H3_MEMORY_VAE_DECODER_BYTES + adapters;
    if (stages_overlap) needed += 2 * H3_MEMORY_WEIGHT_SLOT_BYTES;
    if (needed <= ceiling->ceiling) return 1;
    char term[128];
    const char *remedy = h3_memory_term(ceiling, term, sizeof(term));
    snprintf(error, error_size,
             "not enough memory for this run: %.1f GiB needed, %.1f GiB "
             "available\n    (%s).\n    %s",
             h3_memory_gib(needed), h3_memory_gib(ceiling->ceiling), term,
             remedy);
    return 0;
}

int h3_memory_gate_steady_state(const h3_memory_ceiling *ceiling,
                                char *error, size_t error_size) {
    uint64_t footprint = h3_memory_footprint();
    if (footprint <= ceiling->ceiling) return 1;
    char term[128];
    const char *remedy = h3_memory_term(ceiling, term, sizeof(term));
    snprintf(error, error_size,
             "memory ceiling hit after the first denoiser evaluation:\n"
             "    footprint %.1f GiB, ceiling %.1f GiB (%s).\n    %s",
             h3_memory_gib(footprint), h3_memory_gib(ceiling->ceiling), term,
             remedy);
    return 0;
}
