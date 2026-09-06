# Runtime LoRA adapters: how the shipped code works

What the implementation does and why, for a reader who has the code in front of
them. The vocabulary is `CONTEXT.md`; this file is the mechanism and the
measurements. Everything here was measured on the reference machine (Apple M4
Pro, 48 GiB unified, `AGENTS.md` has the rest).

## What LoRA touches

Four projections per DiT block, enumerated by the code in two places:
`h3_dit.c` `LOAD2(...)` for the resident path and `SOURCE(...)` for
`--ssd-streaming`.

| Target | Shape |
|---|---|
| `blocks.N.attn.qkv_proj.weight` | `[21504, 5376]` |
| `blocks.N.attn.out_proj.weight` | `[5376, 7168]` |
| `blocks.N.mlp.fc1.weight` | `[28672, 5376]` |
| `blocks.N.mlp.fc2.weight` | `[5376, 14336]` |

The two token-refiner blocks go through the same `load_block`, so one graft
covers them, but they run once per generation rather than once per evaluation.

Two more targets are AdaLN precomputes, not per-step projections:
`blocks.N.adaln_proj.linear.weight` `[96768, 2688]` and
`final_layer.adaln_proj.linear.weight` `[10752, 2688]`. `final_layer` is a
single pair, not one per block: a complete upstream turbo LoRA therefore has
**51** AdaLN pairs, not 50, and the report says 51 for the same reason a pruned
file's own metadata says `removed_pair_count=51`.

Coverage of the four projections alone is 208 of 535 tensors, 60.5% of the DiT
by bytes. With the AdaLN targets it is 259 of 535, 99.8% (61.60 of 61.73 GiB),
measured by summing the thirteen FL2VA shard headers.

## The file schema

Target name = the key minus the `lora_A`/`lora_B` (or `lora_down`/`lora_up`)
suffix, plus `.weight`, minus an optional prefix.

**The prefix is optional.** The ComfyUI-converted reference file carries
`diffusion_model.`; the upstream turbo LoRA carries nothing; convention-B files
carry `lora_unet_`. A loader that strips unconditionally finds zero pairs on the
second kind. All three forms are accepted.

**Rank is per pair, never per file.** The upstream turbo has rank 64 on the
backbone and rank 16 on its AdaLN pairs. It is read from the shape of `A`
(`[rank, in]`) and `B` (`[out, rank]`); if the two disagree the pair is
internally inconsistent and the load is fatal.

**`.alpha` is the file's own scale**, applied as `alpha/rank` and multiplied by
the user's strength rather than replaced by it. Absent means 1.0. An `.alpha` in
a dtype the loader cannot read produces a warning, never a silent 1.0.

Conventions are covered in `CONTEXT.md`. `.diff_b` bias adapters are refused:
they are a full-rank delta on a bias, a kind of adapter h3 does not implement,
and dropping them would load a DoRA-style file as a plain LoRA and report
nothing.

## The GPU branch

Three dispatches, all on kernels that already existed. No new Metal kernel.

```
hidden = A·x        h3_gpu_linear_bf16
delta  = B·hidden   h3_gpu_linear_bf16
y      = y + delta  h3_gpu_add_bf16
```

The delta is added to the **output** of the base op and never merged into the
weight buffer, because streaming overwrites weight buffers on every evaluation.

**Strength is fused into `A`**, into the bf16 copy made when the active set is
materialised on the GPU, never into the cached adapter (keyed by path, size and
mtime) and never at dispatch time. One copy per adapter lives in memory, not one
per strength. Changing a strength rematerialises `A`: an in-memory recompute
when the parse is still cached, otherwise 591 MiB at 4.28 GiB/s, 0.14 s.

The numerical cost is one extra bf16 rounding on the coefficients of `A`, order
`1e-3` relative, confined to the delta. At strength 100 the delta is 4.7% of the
output, so the contribution to the total is about `1e-4`, fifteen times below
the measured noise floor.

**The intermediate is one shared bf16 buffer**, `rows x max_rank`, allocated at
load and reused by every projection of every block. The denoise loop reports
`alloc=0.000GiB` across its whole length and must keep doing so. Per-adapter or
per-projection buffers buy no parallelism, because dispatches on the same
command queue are ordered anyway; they buy only memory.

**N adapters are N sequential branches**, never a concatenated `A` or a
block-diagonal `B`. `llama.cpp` makes the same choice. The concatenated form
would carry derived state to rebuild on every `!lora add`.

### Measured cost, and why the rank is not padded

`h3_gpu_linear_bf16` routes to MPS only when
`rows >= 32 && input_dim >= 256 && output_dim >= 256`; otherwise it falls to the
hand-written 16x16 tiled kernel. At rank 16..128 **both** halves of the branch
miss the threshold: `A·x` has output equal to the rank, `B·hidden` has input
equal to the rank.

Measured at 4284 rows (the real geometry of a `448x576` / 56-frame run), rank
64, medians in ms:

| projection | base `W·x` | `A·x` | `B·hidden` | add |
|---|---:|---:|---:|---:|
| qkv | 138.03 | 3.11 | 12.28 | 4.74 |
| out | 45.85 | 4.08 | 3.21 | 1.31 |
| fc1 | 181.79 | 3.12 | 16.33 | 6.23 |
| fc2 | 91.48 | 8.08 | 3.20 | 1.31 |
| **total** | **457.14** | **18.38** | **35.01** | **13.60** |

The branch costs 14.7% of base, about 7% of the denoise, about 54 s on the
56-frame run. Each bench row prints which of `mps_linear_dispatches` and
`direct_dispatches` actually moved, so the path is measured rather than assumed.
Every timing is taken in an isolated command buffer with `waitUntilCompleted`;
inside the real DiT these dispatches share a buffer with everything else and the
GPU may overlap them, so **14.7% is an upper bound**. Bench code:
`tests/bench_lora_gemm.c`, target `bench/lora-thin-gemm`.

Padding the rank to 256 with zeros gets both halves into MPS and cuts the branch
to 9.8% of base, roughly 18 s. It is bit-exact (maximum difference `0.000e+00`
on all four projections). It stays out because it costs 4x the adapter memory,
620 MB to 2.48 GB for the reference file, +1.86 GB resident, against the ~2.2 GiB
of headroom the guardrail measures at maximum canvas with `--preview`. Those
18 s would be paid for with the ability to load more than one adapter, which is
the point of the feature.

On record if it is ever reopened: the break-even between the naive kernel and
MPS falls at rank ~36, below rank 32 padding hurts, and the strong case is not
rank 64 but **rank 128**, where not padding costs 3.4x. The add is 3% of base
and fusing it into the second GEMM's epilogue would be worth about as much as
the padding.

## Precomputes and invalidation

Two things are computed once at load and read many times: the token refiner's
output (cached in `dit->refined_text`) and the AdaLN modulation table. LoRA
applies to both, so a change to the active set has to redo both.

**There is no dedicated recompute path.** The active set enters the prepared-DiT
cache key, `h3_prepared_key`. A different key frees and reloads the DiT, and both
precomputes redo themselves as part of an ordinary load. Measured cost of that
reload: **9.011 s**, 28.5 GiB read (30 steps, 56 frames, `448x576`). The AdaLN
precompute is 24.2 of those 28.5 GiB: 50 blocks of `96768 x 2688` in bf16,
496 MiB each.

**Targeted recompute was rejected**, and not on cost. AdaLN is additive
(`time·(W + s·B·A)ᵀ = time·Wᵀ + s·(time·Aᵀ)·Bᵀ`), so the base table could be kept
and the low-rank term added to it for about 600 MB at 30 steps. It is rejected
because removing an adapter would mean subtracting a float delta, and the third
result would not come back bit-identical to the first. A reload replays the first
run's path, so that identity holds by construction.

**The conditioning key is untouched.** LoRA acts on the DiT's token refiner, not
on the Qwen text encoder, so the encoder's 9.596 s are not re-paid on a change.

**Set-entry identity** is `h3_key_file` (path, size, mtime in seconds and
nanoseconds) plus the strength: the file identity the repo already uses for
`--first-frame` and the reference media, not a new one.

**The key is built from the active set**, not from the raw `h3_params` list.
Strength-0 entries are already dropped while the set is built, so a strength
taken to 0 and back is not two changes and needs no dedicated comparison.

**Order-sensitive, never sorted.** The branches are N sums in sequence and float
addition is not associative, so `[A, B]` and `[B, A]` give AdaLN tables that
differ in the last bits. Reordering costs a 9 s reload, is reachable through
`!lora remove` plus `!lora add`, and is rare.

**Lazy, not immediate.** The comparison happens at the top of `h3_generate`, not
inside `!lora add`, so a block of `!lora` commands recomputes once. Validation
stays immediate: `h3_lora_preload` reads only the safetensors header. An
immediate reload would not save the 9 s, only move them earlier.

**`--reuse` and `--core-reuse` need nothing.** `core_reuse` is already in
`h3_prepared_key`, and the cached residual is `hidden - core_input`, formed after
the delta has been summed inside the blocks, so a skipped evaluation carries it
along. A change to the active set is always a cache miss.

## The memory guardrail

**No static cap on adapters.** A fixed number of adapter bytes does not apply to
the condition at run time: the lever that actually moves the peak is the canvas,
and h3's peak is not known before running (the resolution-dependent term sits
outside `h3_gpu_stats`, which reports 2.58 GiB at both `448x576` and
`768x1344`). So it is not predicted, it is measured.

The cap is on the whole process:

```
static cap  = recommendedMaxWorkingSetSize - 4 GiB      /* 32 GiB here */
dynamic cap = system free + current footprint
cap         = min(static, dynamic)
```

`recommendedMaxWorkingSetSize` is 36 GiB on this machine. The **4 GiB reserve is
the only fixed number in the whole design**, justified by what has to stay
standing while h3 holds 22 GB, not by an estimate of our own consumption: no
flag, no preset, no override.

System free is `free_count + inactive_count + purgeable_count` from
`host_statistics64(HOST_VM_INFO64)`, zero dependencies. Pages another process
holds `active` or `wired` do not count as free: the static term is blind to that
case, the dynamic one is not. The stop message names **which term bit**, because
otherwise the user cannot tell whether to close the other process or lower the
canvas.

**Gate 1, pre-flight, before reading any byte.** Compares the cap against the
terms known exactly and independent of the canvas: the two bf16 weight slots
(1.5 GB), the tiled video VAE decoder (9.55 GiB measured, moving 1.7% for 4x the
pixels) and the adapters' resident bytes, known from the safetensors headers at
preload. With `--preview` or a cached DiT the last two are concurrent, so ~11 GB
plus adapters; otherwise the stages are serialised and the decoder is the
maximum.

It is necessary and not sufficient: it does not model the denoise's
canvas-dependent term. The encoder block, once the declared gap in this
arithmetic, is now measured and does not change it: process peak across
tokenizer, video VAE encoder, Qwen vision and Qwen text encoder sits between 4.7
and 6.0 GB over six runs, about 40% below the decoder's 9.55 GiB. Neither canvas
nor prompt length moves it measurably: the spread between two runs of the *same*
configuration (1.1-1.3 GB) covers the whole interval between the four
configurations tried.

**Gate 2, after the first denoiser evaluation.** There the footprint is at steady
state by measurement (`alloc=0.000GiB` across the loop). It samples
`phys_footprint` through `task_info(mach_task_self(), TASK_VM_INFO, ...)`, the
same number `footprint -p` reads and the only one that sees the
resolution-dependent term. Over the cap, it stops. It costs one evaluation, ~45 s
on a 13-minute run, and replaces a prediction with a fact.

**At the cap it stops, never a partial set.** The active set is declarative and
replaced wholesale, so "refuse the Nth" is not defined, and an adapter dropped in
silence is exactly what the design forbids. One-shot fails at gate 1, before the
checkpoint is read. In session the gate refuses **the generation**, not the
`!lora add`.

**What counts in the budget**: the resident bytes of `A` and `B` of the pairs
matched to a target, in the resident dtype. A float32 file counts what it becomes
on the GPU, not what it occupies on disk.

**Both gates are always active**, even with an empty set: they measure the whole
process, so there is one branch less and runs without adapters are protected too,
which are the ones that swap today.

**SSD spill is a later filling-in**, with its trigger fixed: gate 2, never an
invented threshold. The `source` indirection in the adapter struct is the hook.
It would cost ~7-10 GiB of extra reads on the 575 GiB already read, ~2%, against
a measured `unhidden wait 0.006s`. Cheap, but an abort is testable and a spill
under memory pressure is not.

### Supporting arithmetic

Streaming holds only the block norms plus **two** alternating bf16 matrix slots.
One slot is `21504·5376 + 5376·7168 + 28672·5376 + 5376·14336 = 385.4M` elements
≈ 771 MB in bf16, so two slots ≈ **1.5 GB** resident.

A rank-64 adapter at full coverage is 5.96M elements per block, 11.4 MiB across
52 blocks = **591.7 MiB**, plus ~57 MB for 51 rank-16 AdaLN pairs. Verified
against the real file: `loras/turbo.safetensors` measures 591.6 MiB. H3 LoRAs are
3-4x the size of SDXL/Wan ones because hidden is 5376, so **4-5 adapters together
are 2.4-3.0 GB resident**.

Consequence: with a 32 GiB cap and the worst reachable configuration (maximum
canvas with `--preview` or a cached DiT, peak ~32 GB), adapter headroom is
**~2.2 GiB**, three full rank-64 adapters and not five. That combination is
**expected** to trip gate 2, and stopping there is the intended behaviour. At the
recipe's canvas the headroom is ~20 GiB and nothing bites.

## Reporting

Three text outputs, no dashboard.

1. **On activation**: path, rank histogram, applied pair count, AdaLN pair count,
   resident memory. There is no "skipped pairs" section: an unapplicable pair is
   fatal and its list comes out of the error, not the report.

   The rank histogram is a compact single line ordered by descending count,
   `ranks: 64 x208, 16 x51`. Not a range (`16-64` hides the bimodality, which is
   the useful signal: 16 is AdaLN, 64 the backbone) and not a grouping by target
   kind, which would invent a taxonomy to maintain at every new target.

   The AdaLN count is **always** stated, including when it is zero. Zero AdaLN
   pairs is normal in a style LoRA and pathological in a turbo LoRA, and only the
   user knows which they are loading. If the file's header carries a conversion
   signature (`partial_conversion`, `removed_pair_count`, `adaln_keys_removed`) a
   separate line starting with `warning:` quotes those values verbatim rather
   than paraphrasing them: the file knows more than h3 about what was taken out
   of it, and rewording loses the number.

   A `--lora` at **strength 0** still gets a line
   (`stile.safetensors: strength 0, not applied`). The adapter is dropped while
   the active set is built, which is what makes the no-LoRA path identical by
   construction, but this is the only case where an explicitly requested LoRA
   could be absent with nobody saying so. The file is still validated, so a wrong
   path at strength 0 stays an error and not a silence.

2. **`!status`** lists the active LoRAs and their strengths on one line, with
   basenames and not full paths: `LoRA: style.safetensors 0.80,
   turbo.safetensors 1.00`, and `LoRA: none` when the set is empty. The full
   report of point 1 comes out once, on activation. Bare `!lora` repeats the long
   listing on demand.

3. **`--profile` was withdrawn as a requirement.** The repo's profiling API is
   two functions, coarse phase markers. The LoRA branch is interleaved inside the
   denoise phase, several times per block per step: there is no phase boundary to
   mark, and building per-kernel timing would be a profiling project rather than
   a LoRA feature. The cost is measured out of band, with A/B runs.

The library never prints for itself: the summary leaves on `h3_last_error` and
the detail through the report callback, so library output is never mistaken for
model output.

## Tests

The suite is the repo's existing one (`make test`), same style, no new framework.

Tests split by **subject**. When the subject is a file's *content* (names,
shapes, metadata, a broken header) the file is synthesised as a fixture, written
by the test binary into a temporary directory and deleted on exit. When the
subject is the *numbers* (agreement with the float32 reference, byte identity of
a video) real corpus files are needed, and their absence prints `skip:` the way
the other real-weight tests do. No test reads a number produced by an invented
file, and no test waits on a download to check a string.

Because fixtures cannot be absent, the tests that use them never skip. That is
what puts the report checks on a machine with no checkpoint.

Eleven fixtures cover: convention A with and without the prefix, a hybrid with
`.alpha` where `alpha != rank`, mixed ranks in one file, convention B, a
mismatched `base_model`, orphan pairs, an incoherent pair whose `A` and `B` ranks
disagree, a truncated file, a conversion signature, and AdaLN-named pairs.

**The oracle** (`tests/test_lora_oracle.c`) is a float32 CPU reference the GPU
branch is measured against. Numerical agreement is measured, never asserted at a
threshold tighter than the base matrix multiply's own error: the measured floor,
rel-L2 1.66e-03, is bf16's, not the branch's.

## Dead paths on this machine

`h3_gpu_grouped_qkv_linear_rope_bf16` takes its fused NAX path only when
`gpu.tensorOpsEnabled`, and that flag requires the Metal device name to contain
the literal string `"M5"`. The reference machine is an M4 Pro, so the function
always falls back to `h3_gpu_linear_bf16` followed by
`h3_gpu_grouped_qkv_rope_bf16`.

Operational consequence: `dit->qkv` is the **raw** projection, unnormed and
without RoPE, and the qkv LoRA delta goes **between the two calls**. Teaching the
fused NAX kernel to carry LoRA arguments is out of scope: it is M5-only and
unreachable here.
