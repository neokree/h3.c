# Run-to-run nondeterminism: code-reading pass

Branch `research/nondeterminism`, off `main` (`4af02e2`). Read-only analysis,
no GPU run performed. Triggered by the byte differences already recorded in
`feat/720p-blocked-attention`'s `af7ffbe` and `docs/720p-blocked-attention.md`
on that branch: `blk-off`/`blk-off2` (whole-sequence attention, the shipped
`main` code path, `H3_SDPA_QUERY_BLOCK=0`) and `blk-after`/`blk-after2`
(query-blocked attention) each disagree with their own repeat at
640x352/124 frames, and the divergence is present in the *audio* stream too,
which is decoded from the DiT latent and never touches the video VAE. A
determinism pair from small-canvas runs (`outputs/det-d1.mp4` /
`det-d2.mp4`) is byte-identical. So the search is for a mechanism that (a)
sits at or before the DiT's shared video+audio attention, (b) is present on
both the blocked and unblocked attention code paths, and (c) is switched on
by something that scales with canvas/frame count.

## Ranked candidates

### 1. MPSGraph's attention kernel: non-deterministic reduction over the key axis (leading hypothesis)

`h3_gpu_sdpa_graph` (`h3_gpu.m:1474`) builds one
`scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:...` MPSGraph
op per query block; `h3_gpu_sdpa` (`h3_gpu.m:1550`) loops it over query-row
blocks but always keeps the **full key/value sequence** in every block
(`h3_gpu.m:1596-1600` and `1617-1620`: only the query offset moves, `key`/
`value` are always fed at offset 0 for the whole `sequence`). Two computations
inside that one MPSGraph op have a reduction axis equal to `sequence`
(the softmax normalisation over keys, and the second matmul's contraction
over keys when producing the weighted sum against V) — and `sequence`
(`dit->sequence`, `h3_dit.c:401`) is exactly `video_rows + audio_rows + text
rows`, which scales directly with width x height x frames. No other GPU op in
this pipeline has a reduction axis that depends on canvas: the linear/MLP
matmuls (`h3_gpu_linear_bf16`, `h3_gpu_mlp_bf16`) contract over `HIDDEN`/
`INNER`/`FFN`, which are fixed model dimensions, not canvas-dependent
(`h3_gpu.m:2456-2713`).

This single op is also the one place a size-triggered internal algorithm
switch is already *proven* to exist: `docs/720p-blocked-attention.md` on
`feat/720p-blocked-attention` measured the whole-sequence encode's accuracy
against a float64 oracle and found a sharp jump from bf16-floor error
(~2e-3) to 10-15x that (~2.5e-2 to 3e-2) somewhere between N=9,000 and
N=9,100, at the DiT's real head shape (56 heads, head_dim 128). That is not a
smooth degradation, it is Apple's kernel selection changing underneath the
same call. 640x352/124 frames' actual `dit->sequence` sits above that
threshold (the doc states `probe-P1.mp4` was produced above it). A kernel
that starts partitioning a big reduction across more units above a size
threshold is exactly the shape of thing whose accumulation order between
runs is not guaranteed — this is a general, well-known property of GPU
reduction/matmul kernels (split-K style parallel sums, atomics, or per-
threadgroup partials recombined by whichever thread finishes last), and non-
associativity failure in bf16/f32 summation is why it would show up as a
small, "fine-detail" difference rather than garbage.

This also explains the two facts that rule out everything else on this list:

- **Both attention paths are affected.** Query-blocking only shrinks the
  *materialised score tensor* (`heads x queryRows x sequence` instead of
  `heads x sequence x sequence`); it does not touch the width of the key axis,
  so if the reduction over keys is what's nondeterministic, blocking cannot
  fix it — which matches `blk-after`/`blk-after2` still disagreeing with each
  other even though the block-invariance oracle test proved blocking itself
  introduces no error.
- **Audio is affected too.** Video and audio rows are different query rows of
  the *same* joint self-attention over the combined sequence; a reduction-
  order artifact in that shared op touches both.
- **Small canvases are exempt.** A short enough `sequence` reduction plausibly
  fits in a single threadgroup/simdgroup-synchronized pass with a fixed,
  sequential accumulation order — the small-canvas determinism pair is
  consistent with staying under whatever internal size cutoff triggers the
  split.

Falsifiable form: *the GPU attention kernel returns different bits for the
same bf16 inputs on two separate encodes once `sequence` (the key/value axis)
crosses some threshold in the same 9,000-token neighbourhood already found
for the accuracy regression, and is bit-identical below it.*

### 2. Uninitialised memory in the app's own buffers — checked, not ruled in

Two patterns matched the shapes named in the task and were read line by line;
neither is reachable in the flags the repro used (`--steps --reuse --layers
--seed --ssd-streaming`, no `--token-reduction`, no int8):

- **Token-reduction's "dense tail" packing** (`h3_dit.c:1575-1598`,
  `allocate_activations`): when token reduction is on, `attention_output` is
  allocated at full `sequence * HIDDEN` but only `reduced_sequence` rows are
  produced by the reduced attention; `token_baseline_offset = attention_used`
  packs the baseline (skipped) rows into the remaining capacity right after
  it, `dense`, with an explicit `attention_used + baseline_elements <=
  attention_capacity` bound check. This is a real "write partially into a
  buffer sized for the whole tensor" pattern, but it is gated by
  `dit->token_reduction`, set only by `--token-reduction` or `H3_TOKEN_REDUCTION`
  (`h3_dit.c:405-409`; `H3_PARAMS_DEFAULT` in `h3.h:148-153` has it at 0). Not
  active in the repro.
- **Activation aliasing** (`h3_dit.c:1487-1497`): `dit->attention_heads =
  dit->qkv` and `dit->mod_mlp = dit->qkv` when `activation_aliases` is on
  (the default — `!getenv("H3_DISABLE_DIT_ACTIVATION_ALIAS")`). This reuses
  the `qkv` buffer's memory for the post-attention "heads" tensor and for the
  pre-MLP modulation tensor. Read alone this looks like the risky pattern,
  but the reused region is always the QKV projection's own most recent output
  for *this* block, written earlier in the *same* command buffer by the same
  serial GPU queue, so whatever a run reads back was written deterministically
  by that run's own prior kernel, not by OS/driver leftover memory. It does
  not explain a run varying against itself.

**SSD-streaming's double buffer was also checked as a race candidate and
ruled out by code reading.** `h3_gpu_submit` (`h3_gpu.m:913-937`) calls
`waitUntilCompleted` on every in-flight command buffer before returning, and
`command_blocks` — the mechanism that would otherwise let several block's
command buffers queue without an intervening wait — is forced to 0 whenever
`ssd_streaming` is on (`h3_dit.c:2311`). So every block's GPU work is fully
drained before the loop advances (`h3_dit.c:2392-2434`), and only *after*
that wait does the next block's prefetch thread start overwriting the weight
slot the just-finished GPU work read from. The two buffers in
`dit->stream_slots[2]` never have a live GPU reader and an active CPU writer
at once. This mechanism cannot produce the observed nondeterminism.

**Verdict on the serious case:** I cannot fully rule out uninitialised or
racy memory from reading alone, because `scaledDotProductAttentionWithQueryTensor`
is an opaque Apple kernel and its internal buffer handling is not visible to
this codebase. But within the app's own code, every buffer that is written by
less than its full extent and later read in full is gated behind a flag that
was off in every run that showed the divergence, and the one live
producer/consumer race I could find (SSD streaming) is fenced by a blocking
wait before either side of the double buffer changes hands. Nothing in
`h3.c`/`h3_dit.c`/`h3_gpu.m` reads memory it did not itself write, given the
flags in the repro. What would fully rule out the GPU-kernel-internal case is
below.

### 3. Runtime-dependent heuristics that pick a code path — ruled out on this hardware

`H3_VAE_TILE_PIXELS` (`h3_video_vae.c:658-689`) and the NAX "Morton" kernels
(`h3_gpu.m:2721`, `2664-2780`, etc.) are the two heuristics the task named.
Both are deterministic functions of fixed inputs, not of anything that varies
between runs:

- The VAE tile-pixel search is a pure function of `pixel_height`/
  `pixel_width` (the canvas), so it picks the same tile size every run at a
  fixed canvas — and it cannot be the root cause regardless, because the
  divergence shows up in the *audio* stream, which never reaches the video
  VAE.
- Every NAX/Morton kernel is gated by `gpu.tensorOpsEnabled`
  (`h3_gpu.m:2721`, `2790`), which requires the Metal device name to contain
  the literal string `"M5"` (`h3_gpu.m:364-373`, confirmed in `AGENTS.md`).
  This machine is an M4 Pro: the flag is always false, so these kernels never
  dispatch here. Ruled out for this hardware, not just unlikely.

### 4. Non-deterministic or time-seeded RNG outside the documented seed path — ruled out

Grepped for `arc4random`, `rand`/`srand`, `time(NULL)`, `clock()`,
`mach_absolute_time` across every `.c`/`.m` file. The only `arc4random_buf`
call is `random_seed()` in `h3_cli.c:109-112`, used solely by the interactive
`!seed random` REPL command — not reachable when `--seed 42` (or the default
seed 42, `H3_PARAMS_DEFAULT`, `h3.h:150`) is passed on the command line. Noise
generation itself (`h3.c:324-367`, `h3.c:1606-1609`) seeds an explicit
`h3_rng` (`h3_host.c:502`) from `params->seed`, single-threaded, CPU-side.
Ruled out.

## Best hypothesis, stated to be falsified

**MPSGraph's `scaledDotProductAttention` kernel does not guarantee a fixed
floating-point accumulation order for its key-axis reduction (softmax
normalisation and/or the attention-weights x V contraction) once the key/value
sequence length crosses an internal size threshold in the same ~9,000-token
neighbourhood where its accuracy is already known to change (measured in
`docs/720p-blocked-attention.md`). Below that threshold the reduction runs in
one deterministic pass; above it, work is split across more parallel units
whose completion order is not fixed run to run, producing bit-level
differences at the bf16 error floor.** If true: two encodes of the *same*
synthesised bf16 inputs at the DiT's real head shape (56 heads, head_dim 128)
should be bit-identical below ~9,000 keys and should disagree, run to run, at
N well above it (e.g. 12,000 or the already-measured 33,329) — independent of
`H3_SDPA_QUERY_BLOCK`, since blocking only changes the query axis.

## Cheapest experiment (not yet run)

`tests/test_sdpa_blocking.c` on `feat/720p-blocked-attention` already builds
exactly this harness: it synthesises deterministic bf16 Q/K/V from a fixed
seed (`rng_state`, `next_bf16()`), calls `h3_gpu_sdpa_bf16` once per shape via
`encode()`, and reads the result back into a host buffer. It does not yet
compare a shape against *itself* run twice — it only compares against the
float64 oracle and across block sizes within one run. The smallest addition
that answers the falsifiable question above: call `encode()` twice back to
back with identical inputs (same `query`/`key`/`value` tensors, same block
size) at two `sequence` values — one below ~9,000 (e.g. 4,096, already in the
existing sweep and known clean) and one above (12,000, or the target 33,329
behind `H3_SDPA_TARGET=1`) — and `memcmp` the two output buffers.

```sh
# after adding the repeat-and-memcmp check described above
make tests/test_sdpa_blocking && ./tests/test_sdpa_blocking
# to also exercise the production shape (needs ~6 GiB, per the file's own comment):
H3_SDPA_TARGET=1 ./tests/test_sdpa_blocking
```

Expected if the hypothesis holds: bit-identical at N=4,096, `memcmp`
mismatch at N=12,000/33,329. This needs no video generation, no `./h3`
invocation, and no full denoiser loop — one process, sub-second at the small
shape, using the harness already on disk. It is GPU work, though (an MPSGraph
encode + `waitUntilCompleted`), so per the instruction for this pass it is
proposed but **not run**.

If it comes back bit-identical at both sizes, the next-cheapest move is to
add a matching pure-matmul repeat-and-memcmp (skip the softmax, just Q@K^T and
attn@V at the same N, via `h3_gpu_linear_bf16` on suitably shaped inputs) to
isolate which half of SDPA is responsible, and only after that reach for
`MTL_SHADER_VALIDATION=1`/`MTL_DEBUG_LAYER=1` (Metal's own API/shader
validation) on the harness to positively rule the uninitialised-read case in
or out at the driver level — that is the tool that would give a real answer
to "does anything read memory it did not write", since it is opaque to
`h3.c`'s own source.

## Files read

- `h3_gpu.m` — `h3_gpu_sdpa`, `h3_gpu_sdpa_graph`, `h3_gpu_submit`,
  `h3_gpu_continue`, `h3_gpu_begin`, `h3_gpu_linear_mps`, `h3_gpu_linear_bf16`,
  `h3_gpu_mlp_bf16`, `h3_gpu_fc1_swiglu_nax_bf16`, `h3_gpu_mlp_nax_bf16`.
- `h3_dit.c` — `allocate_activations`, `configure_token_reduction`,
  `allocate_stream_slot`, `read_stream_layer`, the SSD-streaming loop in
  `load_dit`'s block-run driver (`h3_dit.c:2309-2434`), `prepare_rope`,
  `prepare_maps`, `prepare_token_reduction_maps`.
- `h3_host.c` — `h3_rng_seed`, `h3_latent_canvas`, `h3_adapt_canvas`.
- `h3_cli.c` — `random_seed`.
- `h3_video_vae.c` — `configured_tile_pixels`, `tile_count_for_extent`.
- `h3.h` — `H3_PARAMS_DEFAULT`.
- `feat/720p-blocked-attention:docs/720p-blocked-attention.md`,
  `feat/720p-blocked-attention:tests/test_sdpa_blocking.c` (read via
  `git show`, not checked out on this branch).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
