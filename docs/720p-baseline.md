# 720p baseline measurement

Branch `measure/720p-baseline`. Goal canvas: 1280x704, 124 frames (5.17s at
24fps), native — no `--render-width`/`--render-height` upscaling. This
document reports what was measured on this machine on 2026-09-14. No source
file was touched; only this document, `logs/`, and `outputs/`.

Prompt used for every run below: `prompts/01-breath.txt` (locked-off medium
close-up, single subject, small but clearly visible motion — breath, hair,
earring — with an explicit style guard-rail against photoreal drift). Same
prompt across all runs so any difference in output is attributable to canvas
and schedule, not content.

## Machine state at measurement time

Before any run, on branch `measure/720p-baseline` freshly checked out from
`main` (the `feat/macos-app` untracked files `.vscode/`, `up.json`, `macos/`
were left alone — untracked files carry across a checkout and were not
touched or committed):

- `./h3 --info -d ./MiniMax-H3` ran without rebuilding; binary was not stale.
- `vm_stat` / `memory_pressure`, page size 16384 bytes:
  - Pages free: 910112 → **13.89 GiB** genuinely free.
  - Pages inactive: 1103678 → 16.84 GiB (reclaimable file-backed cache).
  - Pages purgeable: 54868 → 0.84 GiB.
  - Pages active: 469616 (7.69 GiB) + wired 271390 (4.45 GiB) ≈ **12.1 GiB**
    genuinely in use and not reclaimable.
  - Total: 48.0 GiB (`hw.memsize` = 51539607552 bytes), matches spec.
- No process was found holding more than ~2 GiB. The largest was a VS Code
  helper (`Code Helper (Plu...`, PID 27524) whose `top` "MEM" column read
  2182M but whose actual RSS (`ps`) was ~125 MB — the `top` figure is not a
  reliable resident-set number for that process type. Nothing was killed.
  This machine state is **not** the polluted state some old `logs/probe-*`
  files were captured under (see below) — free+reclaimable here is roughly
  31 GiB, not the ~13-28 GiB implied by the old guardrail trips.

## Measurement A — token count N at the target canvas

```
./h3 --profile -d ./MiniMax-H3 -p "$(cat prompts/01-breath.txt)" \
  --width 1280 --height 704 --frames 124 --steps 2 --reuse 1 --layers 50 \
  --ssd-streaming -o outputs/probe-720p-N.mp4 > logs/base-720p-N.log 2>&1
```

(`--steps 1` is rejected — `h3: denoising steps must be in [2, 1000]` — so
this used the smallest legal value, 2, which is still cheap enough to die on
the first attention dispatch as intended.)

Result: aborts on `denoise 0/2`, before `1/2` prints, with the expected MPS
assertion:

```
dimensionLengths        = [ 33329, 33329, 56, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
```

**N = 33329** at 1280x704 / 124 frames. Full log: `logs/base-720p-N.log`.

Cross-check against a second real measured N, `logs/probe-800x448-124f.log`
(800x448, 124 frames): `dimensionLengths = [13928, 13928, 56, ...]` →
**N = 13928**. Both canvases use 124 frames, so these two points alone fix a
line in pixel count at fixed frame count:

```
pixels(800x448)  = 358400,  N = 13928
pixels(1280x704) = 901120,  N = 33329

slope     = (33329 - 13928) / (901120 - 358400) = 19401 / 542720 = 0.035752 tokens/pixel
intercept = 13928 - 0.035752 * 358400 = 1114.3 tokens
```

So at 124 frames, `N(pixels) ≈ 0.035752 * pixels + 1114`. The ~1114-token
intercept is plausibly the fixed text/conditioning token budget that doesn't
depend on canvas size. This line is used below to assign an N to the three
footprint points that also ran at 124 frames (P1, P3, P4); it is **not**
extrapolated across frame counts (see the P2 discrepancy note).

## Old-log audit: which probe-* logs can be trusted

The task brief flagged a suspected reversal between `probe-1024x576-39f.log`
and `probe-1344x768-39f.log`. Checked directly:

- `probe-1344x768-39f.log`, `-56f`, `-73f`, `-90f` all show a clean MPS
  assertion with N = 12790 / 17886 / 22984 / 28080 respectively — internally
  consistent (fits `N ≈ 299.8*frames + 1098` at that canvas, R² essentially
  1 over 4 points) and consistent with the pixel-line above once scaled by
  pixel ratio (predicts 33413 vs the measured 33329 target N, 0.25% off).
  **These four logs are trustworthy.**
- `probe-1024x576-39f.log` contains **no N at all**: it stops mid-run at
  `denoise 1/8`→`2/8` with no crash message and no trailing newline — it was
  killed or interrupted externally, not crashed on an MPS assertion. The
  claimed "reports a larger N" premise does not hold, because the file
  reports no N to compare. It should be treated as an incomplete/unusable
  log, not as evidence of anything about relative N.
- `probe-1024x576-73f.log` **does** carry a real crash: N = 13480 at
  1024x576/73f. That is smaller than 1344x768/73f's N = 22984, which is the
  expected direction (fewer pixels → smaller N) — no contradiction there.
- `probe-704x384-124f.log` recorded a memory-guardrail stop at "footprint
  29.0 GiB, ceiling 28.6 GiB (system free 28.5 GiB plus our 0.0 GiB)" — i.e.
  it computed a low ceiling because only 28.5 GiB was free/showing as system
  free at that moment, well below the ~31 GiB free+reclaimable measured at
  the start of this session. That log is real but was captured on a more
  memory-pressured machine than today's. On today's clean machine, the same
  canvas (P3 below) completed a full evaluation at 26 GiB and did not trip
  any gate.

## Measurement B — footprint scaling

All four points below used `--steps 2 --reuse 1 --layers 50 --ssd-streaming`
(the same prompt as above). `--steps 1` is illegal on this binary, so this
is the cheapest legal schedule; with `--reuse 1` every step is a full
evaluation, so `--steps 2` gives exactly 2 evaluations unless the guardrail
cuts the run short.

**Sampling method**: for each run, `footprint -p <pid>` (formatted / binary
GiB-MiB output) was sampled every ~2s in a loop for the lifetime of the
process, and the maximum observed value is reported below. `peak=` from
h3's own `--profile` output was recorded too, purely to confirm AGENTS.md's
warning that it is useless for this purpose: it stays pinned at 2.1-2.8 GiB
across all four canvases regardless of resolution (it counts only the two
streaming weight slots), while true process footprint moves by 4x over the
same runs. **`peak=` was not used for anything below except this check.**

| Point | Canvas | Pixels | Frames | Evals completed | Denoise wall (s) | s/eval | Max footprint (measured) | N (via pixel-line above) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| P1 | 640x352 | 225280 | 124 | 2 | 153.124 | 76.56 | 19 GiB (±0.5, tool rounds to whole GiB above ~10 GiB) | 9169 |
| P2 | 640x352 | 225280 | 73 | 2 | 98.279 | 49.14 | 8637 MiB = 8.44 GiB (exact, below rounding threshold) | not computed — see note |
| P3 | 704x384 | 270336 | 124 | 2 | 214.375 | 107.19 | 26 GiB (±0.5) | 10779 |
| P4 | 768x416 | 319488 | 124 | **1** (guardrail stopped it) | 304.526 | 304.53 | **34.7 GiB (exact — h3's own guardrail message, not the rounded sampler)** | 12536 |

Evaluation count check: the SSD-stream line confirms this independently.
P1/P2/P3 each read **72.495 GiB total across the run**; P4 read **36.606
GiB**. At ~36.2-36.6 GiB streamed per evaluation (matches AGENTS.md's
measured ~36 GiB/eval), 72.495/36.25 ≈ 2 → P1/P2/P3 each completed **2**
evaluations, confirming `denoise_wall/2` is the correct per-evaluation time
for those three. P4's 36.606 GiB matches **1** evaluation — the guardrail
fired after that first evaluation, before a second one started, so P4's
whole denoise wall (304.526s) *is* its one evaluation, not half of it.

P4's exact guardrail message:

```
h3: memory ceiling hit after the first denoiser evaluation:
    footprint 34.7 GiB, ceiling 28.7 GiB (system free 28.7 GiB plus our 0.0 GiB).
```

This is a real, valid data point per the task brief (the gate reports the
footprint it measured), and it agrees with the independently-sampled max
("35 GB" formatted) to within the sampler's own rounding.

**P2 note**: P2 is at 73 frames, not 124, so the frames=124 pixel-line above
does not apply to it. A second, independent estimate exists from the old
1344x768 crash logs (`N ≈ 299.8*frames + 1098` at that canvas), scaled by
pixel ratio to 640x352: predicts N(640x352,124f) ≈ 8355, about 9% below the
9169 the frames=124 pixel-line gives for the same canvas. That's a real
disagreement between two different extrapolation paths, not just rounding
noise, so **no N is reported for P2** rather than picking one arbitrarily.
P2's footprint and timing numbers are still real and are reported above and
used below for the frame-scaling observation, just not folded into the
N-based fit.

### Fitting the scaling law

Three points (P1, P3, P4) have both a pixel count and, via the frames=124
pixel-line, an N. A 3-parameter quadratic `footprint = c0 + c1*N + c2*N²`
fit exactly through 3 points has zero degrees of freedom — it cannot be
checked against an independent point, and it produced an **unphysical
negative c0 (-3.16 GiB)**, which is a sign that 3 points is not enough to
separate three independent terms cleanly, especially at the ±0.5 GiB
rounding uncertainty P1 and P3 carry. Reported here for completeness, not
because it should be trusted:

```
c0 = -3.16 GiB, c1 = 7.75e-4 GiB/token, c2 = 1.79e-7 GiB/token²
```

A simpler 2-parameter fit forced through the origin (`footprint = d1*pixels
+ d2*pixels²`, i.e. the fixed ~2-3 GiB weight-slot term folded into noise)
was solved from the two extreme points, **P1 and P4**, and then checked
against the held-out middle point, **P3** — this is the one real
out-of-sample check available with 4 measured points:

```
d1*225280 + d2*225280²  = 19.0     (P1)
d1*319488 + d2*319488²  = 34.7     (P4)
→ d1 = 2.6295e-5 GiB/pixel, d2 = 2.5765e-10 GiB/pixel²

predicted P3 = d1*270336 + d2*270336² = 7.108 + 18.828 = 25.94 GiB
measured  P3 = 26 GiB (±0.5)
```

Agreement within 0.06 GiB, well inside P3's own rounding uncertainty. **This
is the fit used for the projections below.** It says: at this frame count,
roughly a quarter of the footprint at P1's canvas is the linear term and
three-quarters is already the quadratic term — the quadratic term is not a
small correction even at 640x352, it already dominates by P4's canvas
(768x416).

Sanity check against the raw physics: the crashing tensor is
`[N, N, 56 heads]` in bf16, i.e. `N² * 56 * 2 bytes`. In GiB-per-token²
terms that's `112 / 1024³ = 1.043e-7`. The fitted quadratic coefficient
against N directly (from the unstable 3-point fit above) was 1.79e-7 —
about 1.7x the raw score-tensor size, consistent with MPSGraph needing at
least one extra same-size buffer alongside it (e.g. pre/post-softmax) rather
than reusing the tensor in place. Given the c0/c1/c2 split isn't trustworthy
on its own (see above), take this as a plausibility check, not a precise
decomposition.

### Projected footprint at 1280x704 / 124 frames (pixels = 901120)

**(a) As the code is today** (attention materializes the full `[N,N,56]`
score tensor): extrapolating the pixel-fit above 2.8x past the largest
measured point (319488 → 901120 px):

```
d1*901120 + d2*901120² = 23.69 + 209.2 = 232.9 GiB
```

This number is not achievable and never will be measured directly — at
N=33329 the encode itself fails outright (Measurement A), well before any
footprint could be reported. It is reported here only to show *how far*
over budget the "as coded today" path is: not a 20-30% overrun of the
28-29 GiB ceiling this machine has shown, but roughly **8x the entire 48
GiB of unified memory**, dominated almost entirely by the quadratic term
(209 of 233 GiB). Treat this as an order-of-magnitude statement, not a
precise one — it is a 2.8x extrapolation of a 2-point-checked-against-1
fit.

**(b) If the attention score matrix were bounded to a constant instead of
N²** (drop the quadratic term, keep only the measured linear-in-pixels
term):

```
d1*901120 = 23.69 GiB
```

This assumes every other cost that scales with resolution is already
captured by the linear term (consistent with AGENTS.md's framing — the
quadratic term is specifically attributed to the attention score matrix,
and DiT weight cost per evaluation is resolution-independent, confirmed by
the flat ~36 GiB/eval SSD-stream reads across every canvas above). Under
that assumption, 23.7 GiB is comfortably under the ~28-29 GiB ceiling this
machine showed on a clean run, and well under the 48 GiB physical total —
but this is still an extrapolation 2.8x past the largest measured point,
not a measurement.

## Timing model (per-evaluation, measured)

| Point | Canvas | Frames | s/evaluation (measured) |
|---|---|---:|---:|
| P2 | 640x352 | 73 | 49.14 |
| P1 | 640x352 | 124 | 76.56 |
| P3 | 704x384 | 124 | 107.19 |
| P4 | 768x416 | 124 | 304.53 (single eval, then guardrail) |

P1 vs P2 (same canvas, frames 124 vs 73, ratio 1.699): time ratio 76.56/49.14
= 1.558 — sub-linear in frames, consistent with AGENTS.md's own caveat that
its frame-linear cost estimate is "a lower-bound estimate, ±15%" since
attention cost is quadratic in tokens and token count itself grows with
frames. P3 vs P1 (same frames, pixels ratio 1.200): time ratio 107.19/76.56
= 1.400 — faster growth than pixels alone, again consistent with the
attention term being present. P4 vs P3 (pixels ratio 1.182): time ratio
304.53/107.19 = 2.841 — a much larger jump than the pixel ratio alone would
suggest, matching the footprint curve's own finding that the quadratic term
is already dominant by this canvas. No timing projection to 1280x704 is
given here: the same "how far past the measured range" caveat as the
footprint projection applies, and the brief asked for measurement, not a
time estimate to design against.

## Measurement C — quality control video

Command:

```
./h3 --profile -d ./MiniMax-H3 -p "$(cat prompts/01-breath.txt)" \
  --width 640 --height 352 --frames 124 --steps 6 --reuse 1 --layers 50 \
  --seed 42 --ssd-streaming -o outputs/control-640x352-124f.mp4 \
  > logs/control-640x352.log 2>&1
```

640x352 is 1.818:1, the same aspect ratio as the 1280x704 target, so this
control shares framing with any future 720p attempt. Ran in background;
see `logs/control-640x352.log` for the full profile trace.

Completed cleanly, no guardrail, wrote `outputs/control-640x352-124f.mp4`
(3.44 MB). `h3 profile: H3 DiT Euler denoise wall=492.169s`, 6 evaluations
(`--steps 6 --reuse 1`) → 82.03 s/eval, in line with P1's 76.56 s/eval at
the same canvas/frame count (P1 used `--layers 50` too; the ~7% gap is
within run-to-run noise). `ffprobe` confirms two streams, `h264` video +
`aac` audio; `ffmpeg -f null -` decodes the whole file with zero errors.

Contact sheet: `outputs/control-contact.png` (frames 0, 41, 82, 123, tiled
4x1).

**What it shows: this control run did not converge to a coherent image.**
All four sampled frames (confirmed on a fifth, frame 62, pulled separately
to rule out an artifact of the `tile` filter) are a dense checkerboard/mosaic
of small colored patches — no visible subject, no recognizable skin tone,
hair, or background region, no continuity between frames beyond the same
noise pattern persisting. This is not one of the specific defects the brief
asked to check for (grid seams, ringing, ghosted limbs, identity drift,
flicker) so much as a wholesale failure to denoise: the whole frame reads as
near-random RGB blocks at roughly the model's patch granularity. There is no
usable "identity" or "style" to compare frame-to-frame because no subject
ever forms.

This is worth flagging plainly rather than smoothing over: the command run
is exactly the one specified in the task brief, with **no `--lora`**. Several
of the pre-existing low-step logs in this repo (`probe-1024x576-39f.log` and
others) that use a similarly small step count (`--steps 8` and below) load
`loras/turbo.safetensors` alongside it — the AGENTS.md guidance to use
`--steps 4..7` under a tight budget may implicitly assume a distillation
LoRA is active, and 6 undistilled Euler steps may simply not be enough to
converge this model. That is an observation about why this result looks the
way it does, not a fix — no source file, flag, or LoRA was added or
changed for this measurement, per the task's instructions to measure only.
**This control video should not yet be used as the visual-integrity
reference it was meant to be**; whoever designs the actual 720p attempt will
need a control run that actually converges before it's useful for that
comparison.
