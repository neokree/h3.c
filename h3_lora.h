/* Runtime LoRA adapters: parsing, target resolution and the active set.
 * Internal to the build; not part of the libh3.a contract (SPEC 6ter.2). */
#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"

/* One lora_A/lora_B pair, resolved against a base weight. The h3_st_tensor
 * copies keep their name pointers inside `header`, so a pair never outlives
 * the adapter and its names are freed with the header, not with the pair.
 *
 * `a` and `b` name a source rather than holding bytes: the weights stay on
 * disk until a site materialises them. That indirection is the declared hook
 * for a future SSD spill, whose trigger is gate 2 of the G4 guardrail and
 * never an invented threshold. The spill itself is a later filling-in. */
typedef struct {
    char *name;          /* pair name: the target minus ".weight" */
    h3_st_tensor a;      /* [rank, in]  */
    h3_st_tensor b;      /* [out, rank] */
    uint64_t rank;
    uint64_t in_dim;
    uint64_t out_dim;
    float scale;         /* alpha/rank from the file, 1.0 without .alpha */
    int adaln;           /* pair targets an adaln_proj precompute */
} h3_lora_pair;

typedef struct h3_lora_adapter {
    char *path;                 /* the path as the caller wrote it */
    h3_st_header header;        /* tensor table; weight bytes stay on disk */
    h3_lora_pair *pairs;
    size_t pair_count;
    size_t adaln_pair_count;
    uint64_t resident_bytes;
    /* Cache identity: path plus size plus mtime (SPEC 6ter.5). */
    uint64_t size;
    int64_t mtime_seconds;
    int64_t mtime_nanoseconds;
    struct h3_lora_adapter *next;
} h3_lora_adapter;

/* One entry of the active set. The adapter belongs to the h3_ctx cache. */
typedef struct {
    const h3_lora_adapter *adapter;
    float strength;
} h3_lora_entry;

typedef struct {
    h3_lora_entry *entries;
    size_t count;
} h3_lora_set;

/* --lora PATH[:STRENGTH] (SPEC 4, G3): split on the LAST colon, because a
 * colon is legal in a macOS file name. A tail that is not a finite number is
 * an error and never a path: falling back to the whole string would report
 * "file not found" for ":0.8x" and point at the wrong thing. `argument` is
 * the caller's buffer, truncated in place at the colon on success and left
 * whole on failure, where *tail is the offending text for the message. */
int h3_lora_parse_argument(char *argument, h3_lora *lora, const char **tail);

/* Parse one LoRA and resolve every pair against the transformer shard
 * headers. Reads no weight bytes beyond a scalar .alpha. On failure returns
 * NULL, writes the one-line summary into `summary` (required, non-NULL) and
 * sends the uncapped detail through `report`. */
h3_lora_adapter *h3_lora_parse(const char *path, const char *transformer_dir,
                               char *summary, size_t summary_size,
                               h3_report_callback report, void *opaque);
void h3_lora_adapter_free(h3_lora_adapter *adapter);

/* The activation report of SPEC 5.1. */
void h3_lora_emit_report(const h3_lora_adapter *adapter,
                         h3_report_callback report, void *opaque);

/* Validate a LoRA against the loaded checkpoint now, reporting once. Returns
 * 0 with the summary on h3_last_error when the file is fatal. */
int h3_lora_preload(h3_ctx *ctx, const char *path,
                    h3_report_callback report, void *opaque);
void h3_lora_release(h3_ctx *ctx, const char *path);
void h3_lora_release_all(h3_ctx *ctx);

/* One materialised branch of the delta path (SPEC 7bis.4). `a` is a bf16 copy
 * of the file's lora_A with `strength * alpha/rank` already multiplied in, so
 * nothing scales at dispatch time and the cached adapter stays untouched. */
typedef struct {
    h3_gpu_tensor *a;   /* [rank, in]  */
    h3_gpu_tensor *b;   /* [out, rank] */
    uint32_t rank;
} h3_lora_branch;

/* Every branch that targets one base weight, in active-set order: N adapters
 * are N sequential branches, never a concatenated A or block-diagonal B. */
typedef struct {
    h3_lora_branch *branches;
    unsigned count;
} h3_lora_site;

/* Largest rank and largest output width over the four per-block projections
 * of the whole set. AdaLN pairs are excluded: they are a one-shot precompute,
 * not a per-step projection (SPEC 7bis.3). Both are 0 for an empty set. */
void h3_lora_set_extents(const h3_lora_set *set, uint64_t *max_rank,
                         uint64_t *max_out_dim);

/* Materialise the branches targeting `name` (a pair name, e.g.
 * "blocks.0.attn.qkv_proj", or an AdaLN precompute target such as
 * "blocks.0.adaln_proj.linear" and "final_layer.adaln_proj.linear"). A site
 * with no branch stays zeroed and costs nothing, which is what keeps the
 * no-LoRA dispatch loop identical (H4). */
int h3_lora_site_build(const h3_lora_set *set, h3_gpu *gpu, const char *name,
                       h3_lora_site *site, char *error, size_t error_size);
void h3_lora_site_free(h3_lora_site *site);

/* Build the active set for one generation: order preserved, never sorted,
 * strength-0 entries validated and then dropped so the dispatch loop with no
 * LoRA stays identical by construction (H4). */
int h3_lora_set_build(h3_ctx *ctx, const h3_params *params, h3_lora_set *set);
void h3_lora_set_free(h3_lora_set *set);

/* ---- the G4 memory guardrail (SPEC 10, G4) ----
 *
 * The ceiling bounds the WHOLE PROCESS, never the adapters, and both gates are
 * always active, empty active set included: that is the declared widening from
 * "adapter cap" to "h3 memory guardrail", and it means one branch fewer.
 *
 * It is taken once per generation and both gates compare against those same
 * numbers. That is not an optimisation: our own footprint is a term of the
 * system-free ceiling, so re-taking it at gate 2 would make that term grow with
 * whatever it is meant to catch and it could never bite. */
typedef struct {
    uint64_t working_set;   /* recommendedMaxWorkingSetSize */
    uint64_t system_free;   /* free + inactive + purgeable, when taken */
    uint64_t footprint;     /* ours, when taken */
    uint64_t ceiling;       /* min(working_set - 4 GiB, system_free + ours) */
    int system_free_bit;    /* the min() took the system-free term */
} h3_memory_ceiling;

/* The 4 GiB reserve is the only fixed number of the whole design: no flag, no
 * preset, no override (SPEC 10, G4). */
void h3_memory_ceiling_take(uint64_t recommended_working_set,
                            h3_memory_ceiling *ceiling);

/* Gate 1, pre-flight, before a byte of the checkpoint is read: only the terms
 * known exactly and independent of the canvas. `stages_overlap` is --preview or
 * a cached DiT, where the weight slots and the video VAE decoder are resident
 * at once instead of serialised. Writes the stop message and returns 0 when the
 * ceiling is already exceeded. */
int h3_memory_gate_preflight(const h3_memory_ceiling *ceiling,
                             const h3_lora_set *set, int stages_overlap,
                             char *error, size_t error_size);

/* Gate 2, after the first denoiser evaluation, where the footprint is at steady
 * state by measurement (alloc=0.000GiB across the loop): the real
 * phys_footprint, the only number that sees the canvas-dependent term. */
int h3_memory_gate_steady_state(const h3_memory_ceiling *ceiling,
                                char *error, size_t error_size);

#endif
