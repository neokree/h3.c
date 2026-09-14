# Cross-domain research round: extraction and ranking

Ten agents ran against the question "does any non-ML field have something for
h3's 720p (1280x704, 124-frame) path?": five fields — `ode-integration`,
`multiresolution`, `graphics`, `linear-algebra`, `control-extrapolation` — each
paired with a hostile-check agent briefed to argue against every idea and to
name anything the ML field had already imported. This document keeps only
what the hostile check did not kill, ranks it, and buries the rest with the
argument that killed it, so nobody re-proposes it next round.

## The budget correction that changes the ranking

The round was briefed with an estimated 774–1290 s per denoiser evaluation.
Measured in situ today: **579 s/evaluation**, fixed costs **417 s** (of which
the video VAE decode is **398 s**). Against a 90-minute (5400 s) budget:

```
(5400 - 417) / 579 = 8.61   ->  8 to 9 evaluations fit
```

The round was briefed that only 4 evaluations fit. That was wrong by roughly
2x, and it matters for more than arithmetic: a 4-evaluation run now costs
`4*579 + 417 = 2733 s`, **45 minutes of a 90-minute budget** — there is
**slack of 4 to 5 evaluations' worth of time (~40 minutes)** even while
holding to the turbo LoRA's 4-step distillation. Two consequences for the
ranking below:

- Every idea that trades quality risk to *save one evaluation* (skip the
  last step, shrink to 3 steps) is now buying something the budget no longer
  needs. It was worth 13–21 minutes against a claimed 4-evaluation ceiling;
  it is worth nothing against a ceiling with 40 minutes of headroom.
- Ideas that shave tens of seconds off a fixed-cost phase (the VAE decode)
  are proportionally smaller than they looked: 25–60 s against a 90-minute
  budget with 40 minutes already unspent is noise, not a win. What's left
  worth ranking highly is **quality at the fixed 4-step schedule** and
  **cheap insurance against a silent quality failure**, not raw seconds.

I rescale every field agent's time estimate to 579 s/evaluation below and say
so each time. Several agents also estimated the VAE decode at "38–40% of wall
clock" from a 2-evaluation probe (`--steps 2 --reuse 1`); at the actual
4-evaluation target the decode's own measured 398 s is **~14.6%** of a
2733 s total, not 38–40%. Where a hostile check re-derived a number
independently of the field agent, I give both and say which I believe — in
every case below, the hostile check's number, because it re-derives from
this repo's code and logs rather than from the field agent's model.

---

## 1. Zero-code items — testable today, no diff

**The sigma-grid finding (already known, and already answered).** The
`ode-integration` field agent's headline claim — that `--steps 8 --reuse 3`
places the same 4 evaluations at strictly better σ nodes than
`--steps 4 --reuse 1` (1.000/0.952/0.800/0.632 vs 1.000/0.973/0.923/0.800),
at byte-identical cost — is genuinely zero-code: `H3_REUSE_STEPS` (env var,
`h3_dit.c:2701`) already lets you place evaluations anywhere on any step
count. **But the hostile check killed the underlying idea**, and the reason
matters more than the arithmetic: README:523 already ran this family and
named it a loser ("zero-order held full-grid velocities, linear velocity
extrapolation, and RES" all beaten by the plain grid), README:122-124
records its failure mode (woven texture, weak motion, clipped colors), and
tracing `extrapolate_velocity` (`h3_dit.c:3086-3098`) shows the first three
Euler sub-steps of `--steps 8 --reuse 3` run on a **zero-order-held**
velocity because `previous_evaluated` is still `-1` — the exact losing
candidate named in the same README sentence. See the graveyard for the full
argument. **Don't spend a run on this**; read README:523 instead of
re-deriving it.

What the hostile check itself produced, in the course of killing that idea,
*is* a zero-code-adjacent test worth naming: if a lower terminal σ is wanted,
`h3_shifted_sigma` (`h3_host.c:129`) is driven by the single compile-time
constant `H3_VIDEO_SIGMA_SHIFT` (`h3_host.h:12`, currently `12.0`). Solving
for a 4-node grid ending at σ=0.632 gives shift ≈ 5.15, and every node would
then get a **fresh** evaluation — no zero-order hold, no extrapolant. This is
not a flag, though: it's a `#define`, so it costs a one-line edit and a
rebuild, not a pure zero-code test. I'm flagging it here rather than as a
survivor because nobody has actually asked for a lower terminal σ — the
shipped shift-12 grid is the distiller's own choice, paired with this exact
LoRA — and changing it without a reason is a solution looking for a problem.

**`H3_SDPA_QUERY_BLOCK={512,1024}`** (env var, `h3_gpu.m:1464`) — sweeping the
attention query-block size is zero-code, but don't bother: the hostile check
on the `multiresolution` idea that proposed it shows the claimed saving
(3–26 s over 4 evaluations) is smaller than this repo's own measured
run-to-run noise (10.6% spread on identical invocations, `docs/720p-blocked-
attention.md`) — you cannot measure a signal that small with the only
available instrument. Listed for completeness, not as an action.

**`H3_VAE_TILE_PIXELS=512`** — also zero-code, also don't run it blind: see
graveyard entry for the linear-algebra idea that proposed it. The one
zero-code test in this neighborhood actually worth running is below, in
survivor #5.

**`H3_PROFILE=1`** — already-shipped instrumentation, not a new idea, but
worth naming because two survivors below use it as their confirm/refute
step: it prints the chosen VAE tile grid (`h3_video_vae.c:922,1069`,
`"h3: video VAE tiles WxH at N pixels"`) and per-step reuse/schedule
diagnostics (`h3_dit.c:1304,2773,2779,3034,3193`) with no code change.

---

## 2. Survivors, ranked

Ranked by value delivered (quality-at-fixed-eval-count and quality-gate
protection weighted above raw seconds, per the budget note above) over
implementation cost.

### 1. Extend the SDPA float64 oracle to the target's attention shape

- **Mechanism, 3 sentences:** `tests/test_sdpa_blocking.c` currently checks
  query-blocked attention against a float64 CPU oracle only at
  `SEQUENCE=1000, HEADS=8` — a shape below every one of the three
  MPSGraph accuracy cliffs `docs/720p-blocked-attention.md` documents at
  56 heads. The target run is 56 heads at N=33,329, 2.7x past the largest N
  ever tested (12,000) and on a head count where the cliffs actually occur.
  Add an env-gated shape to the same harness so `make test` stays at 0.8 s
  by default but a target-shape run can check for a fourth, still-undetected
  cliff between N=12,000 and N=33,329.
- **Field / exactness:** linear-algebra (idea 4), reframed by the hostile
  check; exact (it's a measurement, not an approximation).
- **Hostile check's strongest counter, and whether it holds:** the field
  agent framed this as testing Higham's O(n·u) recursive-summation growth
  bound — the hostile check shows that bound is already empirically false
  for this kernel (error is flat 2.06e-3–2.52e-3 across a 2.9x span of N, not
  growing), so the *stated* mechanism is wrong and the experiment as
  specified (one head) would test a kernel the target run never executes.
  **The counter holds** against the field agent's framing, but the hostile
  check itself reframes the goal correctly: not "measure summation error
  growth" but "check whether there's a fourth kernel-selection *cliff*
  (a discontinuity, not a growth rate) between the largest tested N and the
  target N" — and that reframed test is what survives.
- **File / function:** `tests/test_sdpa_blocking.c` (env-gate a
  56-head, N=33,329 shape; default stays 1000/8 so `make test` doesn't grow
  from 0.8 s to ~20-30 s and ~3.3 GB of GPU allocation on every machine).
- **Diff estimate:** ~25-40 lines (one more shape struct, one getenv gate,
  reuse the existing oracle and rel-L2 comparison).
- **Confirm/refute, literal command:**
  `H3_TEST_SDPA_LARGE=1 make test` (name the gate; wire it to the new shape)
  — pass means rel-L2 stays at the existing 2.07e-3–2.61e-3 bf16 floor at
  N=33,329; a jump like the 3.05e-2 seen at N=9,100 for the *unblocked* path
  means the target run has been silently failing the quality gate this
  whole round was conducted under, on every 720p run so far.

### 2. Negative result: higher-order RK, embedded adaptive stepping, and parallel-in-time do not transfer

- **Mechanism, 3 sentences:** at 4 evaluations the terminal step (σ=0.800→0)
  is algebraically the model's raw x̂0 prediction with zero *discretisation*
  error, so no integrator improvement applies there; the other three steps
  cover only 20% of the σ range at h too small for asymptotic order theory to
  say anything useful; and parallel-in-time (Parareal/PFASST/MGRIT) trades
  total compute for latency on idle hardware this single saturated GPU does
  not have.
- **Field / exactness:** ode-integration (idea 3); approximate (it's an
  argument from schedule geometry, not a measurement).
- **Hostile check's strongest counter, and whether it holds:** two real
  corrections, not a kill. (a) "Zero discretisation error" at the terminal
  step is true but was written to imply the terminal jump is *accurate* —
  it isn't; its error is unbounded *model* error in a single 80%-of-path
  jump, which is the single largest error term in the sampler, and saying
  so plainly (not "solved") matters because idea 1 above is built on the
  hope that this jump can be improved. (b) The claimed rectified-flow
  citation ("RK4 underperforms Euler") is an overreach — Karras et al.
  (EDM) show Heun *beats* Euler at NFE≈20+ on non-distilled models, so the
  right claim is regime-bound (NFE=4, distilled), not general. **Both
  corrections hold** and should replace the stronger wording; the
  operational conclusion (don't build any of these three families) survives
  unchanged, and doubly so once the repo's own `h3_res_step` (h3_host.c:605,
  a second-order exponential multistep, unreached from the CLI) and
  README:523/525 are read: the RES experiment already ran and lost.
- **File / function:** none to change; `h3_host.c:605` (`h3_res_step`),
  `h3_host.c:643` (`h3_euler_velocity_step`) — recommendation is to touch
  neither. If `h3_res_step` is dead on the shipped path, deleting it is a
  smaller diff than anything above it.
- **Diff estimate:** 0 lines of production code; ~1 paragraph of
  documentation correcting the two overreaches.
- **Confirm/refute, literal command:** no run needed — this is a
  documentation fix. Verify the dead-code claim with
  `grep -n "h3_dit_denoise_euler_preview\|h3_dit_denoise(" h3.c h3_dit.c`
  (the CLI calls only the former).

### 3. The boring answer, verified locally: no precision reduction buys time on this GPU

- **Mechanism, 3 sentences:** Apple GPUs through M4 execute FP16/BF16
  `simdgroup_matrix` at the same FMA rate as FP32, so the entire
  mixed-precision/iterative-refinement economy (Ozaki splitting, GMRES-IR,
  int8 tensor cores) has no speed premise here — it arrives with M5's
  matrix accelerators, gated off today by `gpu.tensorOpsEnabled` requiring
  the string "M5" in the device name (`h3_gpu.m:364-373`). This closes off
  a whole family of future "just quantize it" proposals with one hardware
  fact instead of a rediscovery each time.
- **Field / exactness:** linear-algebra (idea 2); exact, but see below.
- **Hostile check's strongest counter, and whether it holds:** the
  field agent's supporting number (GEMMs run at 5.81 TFLOPS, "68-72% of
  peak," "residual headroom 1.25-1.35x, ~55-70 s/eval") divides by a 60.8 s
  remainder that includes RoPE, two RMS norms, AdaLN, SwiGLU and 50
  commit-and-wait boundaries — not just the four matmuls. **The counter
  holds**: the true matmul efficiency is *higher* than claimed (less
  headroom there, strengthening the "no win" conclusion) and the true
  uncosted quantity is the *non-matmul* elementwise/submission remainder,
  which nobody has broken out. The hardware claim itself (no 2x from lower
  precision) rests on a third-party GitHub README's implied roofline rather
  than a local measurement — sound inference, but a gate this load-bearing
  should rest on this machine, not someone else's chip.
- **File / function:** `h3_gpu.m:364-373` (the M5 gate, assessed not
  changed); no production change. Add a ~20-line standalone MPSGraph
  micro-benchmark (same matmul shape at fp32/fp16/bf16) to convert the
  borrowed roofline into a local one.
- **Diff estimate:** 0 lines production; ~20 lines for a throwaway
  benchmark (not part of `make test`).
- **Confirm/refute, literal command:** write the micro-benchmark as e.g.
  `tools/bench_precision.m` (matches this repo's existing `.m` scratch-tool
  convention) and run `clang -framework Metal -framework MetalPerformanceShadersGraph tools/bench_precision.m -o /tmp/bench_precision && /tmp/bench_precision`
  — compare GFLOPS at fp32 vs fp16 vs bf16 for a 9169×5376 matmul. Equal
  throughput confirms the finding; a 1.5x+ gap at fp16 refutes it.

### 4. Negative result: temporal reprojection, TAA/checkerboard, variable-rate shading don't map onto the DiT

- **Mechanism, 3 sentences:** every one of these graphics techniques needs a
  per-frame re-entry point (render frame *k*, then frame *k+1* from it) and a
  cheap geometric correspondence oracle (a G-buffer motion vector); the DiT
  emits all 124 frames from one joint pass over a single 33,329-token
  sequence, so neither exists, and the one temporal axis that *does* exist
  inside it — the denoise-step axis — is exactly the TeaCache/PAB/DeepCache
  family this project already evaluated and rejected (`TODO.md:508`).
- **Field / exactness:** graphics (idea 3); exact structural argument, one
  invented number.
- **Hostile check's strongest counter, and whether it holds:** the "2.06x
  decoder redundancy" figure bounding a hypothetical prize does not exist
  anywhere in this repo — the only 2.06 in the tree is an unrelated rel-L2
  tolerance. Recomputed from the actual tile plan (`tile_axis_build`,
  `configured_tile_pixels`) it is **1.63x**, and that whole 1.63x is the
  seam-blending overlap, not idle waste — attacking it is a quality change,
  not a free lunch. **The counter holds completely**; the number must be
  corrected before this is filed, or a future round will read "up to ~200s
  available" and go looking for it.
- **File / function:** none to change. For context: `h3_dit.c:2759-2900`
  (Euler loop, where a reprojection step would have to live — it shouldn't)
  and `h3_video_vae.c` (the decoder's chunk seams, `stitch_tiles`, are the
  one place a temporal-coherence method could actually touch, correctly
  noted as a quality question rather than a structural impossibility).
- **Diff estimate:** 0 lines code; one number fixed in the write-up
  (2.06x → 1.63x).
- **Confirm/refute, literal command:** no run needed —
  `1.31 (spatial: 18 tiles × 256² / (704×1280)) × 1.24 (temporal: 7×22/124)
  = 1.63`, reproducible from the constants at `h3_video_vae.c:30` and the
  tile/chunk counts alone.

### 5. Size the VAE tile decomposition per-axis, inside the range the planner already searches

- **Mechanism, 3 sentences:** `configured_tile_pixels` (`h3_video_vae.c:666`)
  already sweeps square tile sizes 256–320px and picks the one minimizing
  `tiles × pixels²` — at 1280x704 that's 288px, a 3×6 grid, 1.657x the
  output area. Because the axes are forced to one square size, a genuinely
  better rectangle inside the *same already-searched range* (e.g. 288×320
  on a 3×5 grid) is unreachable today; letting each axis pick its own size
  finds it.
- **Field / exactness:** multiresolution (idea 1); approximate (changes
  decoded pixels, needs a seam-quality check).
- **Hostile check's strongest counter, and whether it holds:** three real
  corrections, not a kill. (a) The premise that the constants are
  "inherited" is false — the planner already tunes them; what's untuned is
  squareness and a 320px search cap the env override already lets you raise
  to 512. (b) The specific headline config (256×272, overlap 20px) is
  **unreachable**: overlaps quantize to 16px (`h3_video_vae.c:720-724`) with
  no alignment check, so a 20px overlap silently misaligns the latent grid
  — the claimed 1.159x/122s saving does not exist at any representable
  configuration. (c) Schwarz theory is decorative here (no coarse space, no
  iteration, no finite decay length in a global-attention ViT decoder) —
  strip the citations and what's left is "tune two config numbers and
  measure the seam." Recomputed honestly: **25s free** (best rectangle
  inside the existing 256–320px search, overlap untouched) and **up to
  60s** if the search cap is raised to 512px, *conditional on a footprint
  check* (larger tiles push the decoder's peak — currently 9.37-9.55 GiB —
  higher; AGENTS.md's own numbers say this tracks tile size, not output
  size). **All three corrections hold**; use the hostile check's 25-60s,
  not the field agent's 122s.
- **File / function:** `h3_video_vae.c`: `configured_tile_pixels` (return a
  size per axis), `tile_count_for_extent` / `tile_axis_build` (accept
  overlap as a parameter). `vae_context` already carries independent
  `latent_h`/`latent_w`, so nothing downstream needs to change.
- **Diff estimate:** ~15-35 lines.
- **Confirm/refute, literal command:**
  `H3_PROFILE=1 H3_VAE_TILE_PIXELS=320 ./h3 -d ./MiniMax-H3 ... 2>&1 | grep "video VAE tiles"`
  to confirm the grid before touching code, then decode a single 288×288
  region untiled (the existing non-tiled path when the canvas fits one
  tile) as ground truth and force a 2×2 tiling at overlap 64/48/32/16 over
  the same region, reporting PSNR on the seam band — no 720p run required.

### 6. Overlap VAE-decode tile encoding with GPU execution (frames-in-flight)

- **Mechanism, 3 sentences:** `h3_gpu_submit` (`h3_gpu.m:926`) commits and
  then blocks on `waitUntilCompleted` before returning, and `h3_gpu_begin`
  refuses to start encoding the next tile until the GPU has drained — so of
  126 independent tile decodes in the target's VAE phase, the GPU sits idle
  during every CPU encode. `h3_gpu_continue` (`h3_gpu.m:902`), which commits
  *without* draining, already exists and is already used by the DiT
  (`h3_dit.c:2407,2876`) — this is applying an existing primitive to a
  loop that doesn't use it yet, not new machinery.
- **Field / exactness:** graphics (idea 1); exact (same kernels, same
  order, host-side scheduling only).
- **Hostile check's strongest counter, and whether it holds:** most of the
  claimed 12% prize is a wound the *sibling branch* (query-blocked
  attention) inflicted on the VAE decoder, and the cheaper fix is to stop
  inflicting it: on `main` without query blocking, the phase's exposed CPU
  is 0.36% (`logs/blk-off.log`), not 12% — the 12.28s of encode in
  `blk-after.log` is almost entirely the blocked-attention branch's own
  9-per-attention MPSGraph encodes, which buy 11.7s of GPU and spend 11.9s
  of CPU, netting ~zero without pipelining. **The counter holds** for the
  *sizing* of the prize, but doesn't kill the idea — it reframes it as
  "pipelining is what makes query-blocking worth keeping in the VAE
  decoder at all," worth ~35-45s at the target if query blocking stays
  there, ~8s if it doesn't. The hostile check also flags real unbudgeted
  cost: this breaks an engine-wide invariant (`h3_gpu_begin`'s
  single-in-flight guarantee, enforced for every phase — text encoder, DiT,
  both VAEs), and with two buffers in flight a Metal error in tile *k*
  would surface under tile *k+1*'s label, on the one machine where a memory
  failure at 1280x704 is the expected failure mode.
- **File / function:** `h3_gpu.m`: `h3_gpu_submit`/`h3_gpu_begin` (relax the
  in-flight guard to 2, or just route the VAE tile loop through the
  existing `h3_gpu_continue`); `h3_video_vae.c`: `decoder_decode_chunk` /
  `run_resident_tile` (double-buffer the per-tile scratch tensors).
- **Diff estimate:** ~50-150 lines (smaller if it reuses
  `h3_gpu_continue` rather than touching the shared invariant).
- **Confirm/refute, literal command:**
  `H3_PROFILE=1 ./h3 --profile -d ./MiniMax-H3 --first-frame refs/x.png -p "..." --width 640 --height 352 --frames 124 --steps 4 --reuse 1 -o /tmp/t.mp4 2>&1 | tee /tmp/blk-check.log`
  compared with the same run under `H3_SDPA_QUERY_BLOCK=0` — read the VAE
  phase's `wall=`/`encode=`/`wait=` line in both; confirms whether query
  blocking is even costing anything in the decoder before spending the
  diff on pipelining it.

### 7. Anytime output: emit each finished VAE-decode chunk as it lands

- **Mechanism, 3 sentences:** `h3_video_vae_decoder_decode` already decodes
  7 chunks strictly in order into `final_rgb`, with frames from chunk *k*
  final the moment chunk *k+1*'s cross-fade completes — but `h3.c` fires
  `on_frame` only after the full ~400s decode returns. Adding a per-chunk
  callback hands out the finished prefix roughly ⅐ into the decode instead
  of at the very end.
- **Field / exactness:** graphics (idea 2, prefix-emission half only — see
  below); exact (same bytes, same order, emitted earlier).
- **Hostile check's strongest counter, and whether it holds:** the idea's
  *other* half (a region-of-interest denoise preview, meant to let a bad
  run be killed early) is where "one aborted run recovered per three
  attempts" — the entire claimed value — was said to come from, and it
  fails three ways: the 9.4 GiB preview VAE decoder stays resident for the
  whole denoise regardless of crop size (`h3.c:1575-1759`), the preview
  shows the partially-denoised latent, not an x̂0 prediction, so at 4
  distilled steps the first preview is noise, and `--show` forces
  `finish=1` (`h3_dit.c:2873`), disabling GPU-continue windowing. **The
  counter holds**, and guts the idea's main selling point — what's left is
  only the chunk-prefix emission, which is correct and cheap but whose own
  ceiling is the untaken tail of a ~400s decode that starts around minute
  85 of 90, and the delivery path (`h3_resize_rgb24_high_quality`, one mp4
  mux at the end) means an early prefix reaches a terminal/preview sink,
  **not a salvageable file** — "look at it," not "keep it."
- **File / function:** `h3_video_vae.c`: `h3_video_vae_decoder_decode` (add
  a per-chunk prefix callback); `h3.c:1654-1706` (the decode call site and
  `on_frame` delivery loop). Do **not** build the ROI preview half.
- **Diff estimate:** ~20-30 lines (prefix-emission only; drop the ROI-preview
  half entirely).
- **Confirm/refute, literal command:** no run needed to decide whether it's
  worth building — the question is only "does anyone want to watch a video
  build incrementally in a terminal frame sink." If yes, verify wiring with
  `grep -n "on_frame" h3.c h3_video_vae.c` and confirm the callback fires
  once per chunk on an existing 640x352 run.

### 8. Free Adams-style local-error diagnostic from the cached velocity history

- **Mechanism, 3 sentences:** `h3_dit_denoise_euler_preview` already keeps
  `last_video`/`previous_video` (v_n, v_{n-1}) in RAM whenever `--reuse > 1`;
  computing the Euler local truncation error from their difference costs a
  few milliseconds and, in principle, tells you where node placement is
  wasting accuracy.
- **Field / exactness:** ode-integration (idea 2); exact as a measurement,
  but see below for why it's the weakest survivor.
- **Hostile check's strongest counter, and whether it holds:** two holds.
  (a) The buffers this idea reads **do not exist** at the configuration the
  whole brief is about: `--steps 4 --reuse 1` allocates them only when
  `reuse_interval > 1` (`h3_dit.c:3041-3047`), so at reuse 1 there is no
  history to difference — this needs a code change, not a printf, exactly
  where it matters most. (b) Even where it *can* run, it only sees the σ
  range the evaluations span (1.000→0.800, 20% of the path) — the remaining
  80% is one terminal jump to σ=0, which is where every real quality
  question in this sampler lives. A small `‖v_n − v_{n-1}‖/‖v_n‖` from this
  probe would read as "the trajectory is straight, no integrator can help,"
  when it's actually an artifact of measuring the flattest available fifth
  of the path — **a confidently wrong inference is worse than no
  diagnostic**, which the field agent's own promise ("kills every
  higher-order proposal with one number") walks straight into. **Both
  counters hold**; it survives only as an always-negligible-cost
  instrumentation add, not as the decision procedure it was pitched as.
- **File / function:** `h3_dit.c`, in the `if (evaluate)` branch around
  lines 3068-3082 where the buffers would need to be allocated
  unconditionally (or gated to print only when `reuse > 1`, which drops
  it out of the one config that matters).
- **Diff estimate:** ~25 lines.
- **Confirm/refute, literal command:**
  `H3_PROFILE=1 ./h3 --steps 8 --reuse 3 -d ./MiniMax-H3 ... --width 640 --height 352 --frames 124 -o /tmp/t.mp4`
  (any config with `--reuse > 1`, since it can't run at reuse 1 without the
  code change) — read the printed `‖v_n − v_{n-1}‖/‖v_n‖`, but treat a small
  number as "the tested 20% of the path is straight," not "the sampler is
  integration-limited," per the counter above.

---

## 3. The graveyard

One line each, with the argument that killed it. Diffusion-field names are
given wherever the ML field has already imported the mechanism, so nobody
proposes it again as new.

**ODE-integration**
- *Decouple the integration grid from the evaluation grid (`--steps 8
  --reuse 3` to move the terminal σ from 0.800 to 0.632 at the same 4
  evaluations)* — dies: this exact experiment already ran and lost
  (README:523, "zero-order held full-grid velocities... beaten"), the
  proposed config actually *runs* the losing zero-order-hold candidate for
  its first three sub-steps (`h3_dit.c:3086-3098`, `previous_evaluated ==
  -1`), and the field agent's supporting evidence misread the turbo LoRA's
  measured strength delta by ~167x (quoting the number at strength 100, not
  the production strength 0.6). Already-imported names: diffusion caching
  (DeepCache, FORA, Delta-DiT, TeaCache, AdaCache), the exact polynomial
  form (TaylorSeer), and the terminal-node objective is timestep shift
  (SD3), which h3 already implements and hardcodes at `h3_host.h:12`.

**Multiresolution**
- *Sweep the SDPA query-block size (256→512→1024)* — dies: the claimed
  saving (3-26s over 4 evaluations) is smaller than this repo's own
  measured run-to-run noise (10.6% spread on identical invocations,
  `docs/720p-blocked-attention.md`) — unmeasurable by the proposed method,
  and the direction of the sweep walks toward MPSGraph's own documented
  accuracy cliffs at larger block sizes.
- *Block the key axis too and fuse the softmax (online-softmax kernel)* —
  dies: this is FlashAttention (Dao et al., NeurIPS 2022), named in the
  proposal's own sources — explicitly out of scope for a round asked what
  ML has *not* taken.

**Graphics** — all three ideas survived-with-caveats (see section 2). No
graveyard entries from this field.

**Linear-algebra**
- *Close the SDPA "roofline gap" with a hand-written fused attention
  kernel* — dies: it is FlashAttention (cited by name in its own sources),
  its own arithmetic is self-contradictory (can't simultaneously claim
  5.81 TFLOPS of *useful* arithmetic and 70-90s of an 86s evaluation as
  *traffic*), and it transfers M1/M3-family efficiency numbers to a
  different (M4 Pro, 273 GB/s) chip — the same error TODO.md already struck
  once. `scaledDotProductAttentionWithQueryTensor` is Apple's own documented
  fused primitive; nobody has captured whether it actually degenerates to
  separate passes at this shape before proposing to replace it.
- *Sweep the video VAE decoder tile size to 512px* — dies: the FLOP
  arithmetic is wrong in the proposal's own favor (per-tile attention is
  quadratic, so total decoder FLOPs rise 25.4% at 512px, not flat — net
  gain ~1.12x / ~48s, not the claimed 1.4x / 135s), and the one real
  precedent in this repo (256px→288px tiles) shows throughput *falling*
  1.9x for a 1.3x FLOP increase — extrapolating that trend to 512px predicts
  a ~1000s decode, not a faster one.

**Control-extrapolation** — all five ideas die; this field's honest verdict
is "no."
- *Safeguarded MPE(1) extrapolation of the x̂0 sequence, to skip the 4th
  evaluation* — dies: the shifted-12 schedule makes the sequence's
  differences grow (~1.85x per step) rather than shrink, so the safeguard's
  own geometric-convergence test (γ ∈ (0, 0.9)) fails **by construction of
  the schedule**, on essentially every run — expected saving is zero, and
  forcing it anyway is exactly the banned trade (a speed number with no
  quality argument). Already-imported: extrapolating the x̂0/data-prediction
  history is what DPM-Solver++(2M), UniPC and DEIS already do; using it to
  delete the terminal evaluation is not, and the repo's own `--reuse`
  machinery already keeps the terminal step evaluated for the same reason.
- *Depth-wise extrapolation of the residual stream to skip tail blocks* —
  dies: `--layers 40` already gets the identical 10-block saving via
  existing gate-ranked block selection (`configure_gate_ranked_blocks`,
  `h3_dit.c:1277`) with zero new code and better block selection than a
  DEQ-style analogy the model's own weights don't satisfy (50 blocks have
  distinct weights, not a tied map); it would also silently discard ~50 of
  the turbo LoRA's own rank-64 pairs distributed across the backbone.
- *Anderson acceleration on the sampler* — dies: no fixed point and no
  cheap residual exist in a sequential sampler where evaluating the map
  once *is* the whole cost of a step; the only formulation with a genuine
  fixed point (Picard over the whole trajectory) needs *more* evaluations
  (8-12 vs 4) and B concurrent activation sets that don't fit in 48 GiB at
  this canvas. Already imported: ParaDiGMS (Picard), ParaTAA (Anderson on
  top of Picard) — both explicitly trade total compute for latency on idle
  parallel hardware this single saturated GPU doesn't have.
- *Continuation/homotopy with a Newton corrector and adaptive step size* —
  dies: a corrector costs one full evaluation (25% of the 4-eval budget)
  for accuracy this repo has already measured to buy nothing visible
  (`docs/720p-schedule-ladder.md`, R1 vs R2: 6 evaluations show no
  improvement over 4 on the subject, plus a new background ring artifact
  R1 doesn't have) — even a *free* corrector would buy nothing here.
  Already imported: predictor-corrector samplers (Song et al., score-SDE),
  DPM-Solver's adaptive mode, Restart sampling.
- *Heavy-ball/Nesterov momentum, higher-order multistep* — dies: saves zero
  evaluations by the field agent's own admission, and the schedule-ladder
  measurement above closes the only escape hatch (better integration
  buying a step reduction) by showing better integration buys nothing in
  the 4-6 evaluation range at all. The repo already owns the momentum-class
  solver (`h3_res_step`, `h3_host.c:605`) unreached from the CLI — switching
  to it perturbs 2 of the 4 distilled steps for no measured benefit.
  Already imported by name: GHVB (ICLR 2024).

---

## 4. Per-field verdicts

**ODE-integration.** A real, narrow "yes" that the hostile check narrowed
further. The field's core correction to the project's own premise —
h3's shipped step is already exponential-Euler/ETD1, not naive Euler, so
there is no free order improvement sitting on the table — stands
unchallenged and is the most durable finding of the whole round; it forecloses
an entire class of future "just use a better integrator" proposals. Its one
positive proposal (move the terminal σ via reuse tricks) turned out to be a
documented prior failure re-discovered from theory instead of from the
README, and its diagnostic idea survives only in a much-diminished,
instrumentation-only form. Net: a well-argued **mostly no**, with one
genuinely load-bearing correction to how the project should think about its
own sampler.

**Multiresolution.** A genuine "yes" on one specific, low-stakes target
(the VAE tile decomposition is literally overlapping Schwarz), and honest
"no"s everywhere else for a single, well-stated reason: multigrid-style
coarse-then-correct schemes need the coarse level to discretize the *same
continuous operator*, and a learned DiT has no continuous operator behind
it — a downsampled latent decodes to a different picture, not a blurry
version of the same one, which is exactly why this project already
rejected render-small-upscale. The one survivor's theoretical framing
(Schwarz overlap selection) turned out to be decorative once the hostile
check traced it — no coarse space, no iteration, no finite decay length —
but the underlying "tune the tile geometry properly" chore is real and
cheap. Net: **one small, real win; the exotic half of the field doesn't
apply here and says so precisely.**

**Graphics.** The strongest "partly yes" of the round, but not for the
reason the field expected. Its algorithmic toolkit (temporal reprojection,
TAA, foveation) needs a per-frame re-entry point and a cheap correspondence
oracle the DiT structurally lacks — correctly diagnosed as a hard no. What
transferred instead was *engineering discipline* rather than an algorithm:
the video VAE decoder is hand-written Metal with a synchronous
submit-and-drain pattern that no ML framework would ever hand-roll (CUDA/
PyTorch pipeline by default), so "never let the GPU idle behind CPU work"
is a real, if modest, win once corrected for how much of its measured
prize was actually a wound the concurrent blocked-attention branch
inflicted on the same phase. Net: **a small real win in engineering
hygiene, not in the field's namesake algorithms.**

**Linear-algebra.** The most valuable *negative* result in the round, at
the cost of its most oversold *positive* one. The field's own quantitative
obstructions (subquadratic attention needs a max-logit bound below what
QK-RMSNorm actually produces; FMM's far-field expansion is worse than N
itself at d=128; RoPE leaves no exploitable Toeplitz structure) are precise
and correctly close off the entire "exotic fast attention" branch of
inquiry. Its flagship positive idea — a hand-fused attention kernel to
close a "2-2.7x roofline gap" — turned out to be FlashAttention with a
self-contradictory arithmetic argument underneath, killed on both counts.
What survives is the field's boring instinct applied correctly: verify the
hardware claim locally instead of trusting the model, and extend the
existing correctness oracle to the shape that actually matters instead of
trusting the one that doesn't. Net: **a valuable "no" on speed, plus the
single most quality-critical survivor of the whole round.**

**Control-extrapolation.** A clean, well-argued "no," and worth recording
as a completed result rather than an empty section. Every mechanism in this
field's toolkit — Anderson acceleration, Aitken/MPE extrapolation,
continuation/homotopy, momentum — assumes either a converging fixed-point
iteration with a cheap residual, or slack to spend on better per-step
accuracy. The sampler here has neither: each of the 4 steps costs a full
DiT evaluation with no cheaper residual, and the schedule-ladder measurement
already shows that spending more accuracy (6 evaluations vs 4) buys nothing
visible, closing the one path (accuracy → fewer needed steps) that could
have made any of this pay off. The field's own literature review confirms
every viable variant is already imported (GHVB, ParaDiGMS, ParaTAA, DPM-
Solver's adaptive mode) and rejected here for GPU/memory reasons specific to
this machine. Net: **no exceptions survive; a genuine, useful zero.**
