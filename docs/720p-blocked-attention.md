# Query-blocked DiT self-attention

Branch `feat/720p-blocked-attention`. Implements item 1 of `TODO.md`
(`plan/720p-todo`, commit `b23b49b`): replace the single whole-sequence
`scaledDotProductAttention` encode with a loop over query blocks, so that
1280x704 / 124 frames (N = 33,329) can encode at all.

The code works and is measured below. **The byte-identity gate the task set
could not be met, and could not have been met by any change**, for reasons that
predate this branch. That is the first section, because it changes how every
other number here has to be read.

## The correctness gate is void: this pipeline is not reproducible

The gate was: regenerate `outputs/probe-P1.mp4` (640x352, 124 frames,
`--steps 2 --reuse 1 --layers 50 --seed 42 --ssd-streaming`) and `cmp` it.

| run | attention | bytes | denoise wall | video VAE decode |
|---|---|---:|---:|---:|
| `probe-P1` (`measure/720p-baseline`) | whole | 3,474,430 | 153.124 s | 105.744 s |
| `blk-off` (`H3_SDPA_QUERY_BLOCK=0`) | whole | 3,327,824 | 150.569 s | 104.332 s |
| `blk-off2` (identical invocation) | whole | 3,404,125 | 148.854 s | 104.029 s |
| `blk-after` | blocked, 256 | 483,162 | 148.217 s | 104.462 s |
| `blk-after2` (identical invocation) | blocked, 256 | 479,222 | 163.940 s | 101.578 s |

`blk-off` and `blk-off2` are the **same binary, same flags, same seed, and the
shipped code path** — `H3_SDPA_QUERY_BLOCK=0` restores the single encode
exactly. They are not byte-identical to each other, and neither reproduces
`probe-P1`. The same holds for the two blocked runs.

The variation is in the model, not the container: the **audio** streams differ
between repeats too (`ffmpeg -map 0:a -f md5`), and audio is decoded from the
DiT latent by the audio VAE without ever touching the video VAE.

This is canvas-dependent, not universal. `outputs/det-d1.mp4` and
`outputs/det-d2.mp4` (a determinism pair left in `outputs/` from 2 September,
small canvas, `video VAE tiles 2x1 at 288 pixels`) **are** byte-identical. So
the pipeline is reproducible at small canvases and stops being reproducible
somewhere below 640x352 / 124 frames.

Consequence: a `cmp` against a stored video cannot certify any change to this
code. It was not a weak gate, it was a gate measuring something that does not
hold. The root cause of the nondeterminism is **not diagnosed here** and is not
caused by this branch; it wants its own investigation.

## What replaced the gate: a float64 oracle

`tests/test_sdpa_blocking.c` (runs in 0.8 s, synthesises its own inputs, never
skips) asks the two questions the `cmp` was meant to ask, against a
double-precision CPU reference instead of against a stored file:

1. **Does the block boundary change the answer?** No. At block 128, 256, 384
   and 512, every one of 1,024,000 output elements is bit-identical.
2. **Is the answer right?** rel-L2 2.61e-3 against float64, for every block
   size and for the unblocked path alike — the bf16 error floor.

## The whole-sequence encode was already degraded, and blocking fixes it

Sweeping a harness at the DiT's real head shape (56 heads, head_dim 128, bf16)
against the same float64 oracle:

| N | whole vs float64 | blocked vs float64 | whole vs blocked |
|---:|---:|---:|---|
| 4,096 | 2.35e-3 | 2.35e-3 | bit-identical |
| 8,192 | 2.52e-3 | 2.52e-3 | bit-identical |
| 8,800 | 2.20e-3 | 2.20e-3 | bit-identical |
| 8,900 | 2.09e-3 | 2.09e-3 | 1.7% of elements, rel-L2 5.3e-3 |
| 9,000 | 2.06e-3 | 2.06e-3 | 3.4% of elements, rel-L2 6.5e-3 |
| 9,100 | **3.05e-2** | 2.07e-3 | 6.8% of elements, rel-L2 1.07e-2 |
| 9,169 | **3.08e-2** | 2.10e-3 | 6.8% of elements, rel-L2 1.17e-2 |
| 9,216 | **2.31e-2** | 2.07e-3 | 8.4% of elements, rel-L2 1.02e-2 |
| 12,000 | **2.69e-2** | 2.41e-3 | 44.0% of elements, rel-L2 2.33e-2 |

### Three thresholds, not one

`scaledDotProductAttention` on this device has **three** size thresholds at 56
heads, of which only the last was known. They are independent and they arrive
in this order:

| # | boundary | what happens | status before this work |
|---|---|---|---|
| 1 | N ≈ 8,800 | whole-sequence and blocked results stop agreeing in the last bit. Both stay at the bf16 error floor. | unknown, and harmless |
| 2 | N ≈ 9,100 | whole-sequence error jumps to **10 to 15x the bf16 floor**. Blocked does not move. | **unknown, and not harmless** |
| 3 | N ≈ 12,370-12,398 (2^33 score elements) | `too large for kernel`, the encode aborts | known; the whole reason this task exists |

Threshold 2 is the consequential one. Between it and threshold 3 there is a
band — roughly N = 9,100 to 12,390 — where the shipped code does not fail, does
not warn, and returns an answer an order of magnitude worse than the hardware
can produce. **640x352 / 124 frames sits inside that band.** So does every
canvas between it and the abort.

### Why it was never seen

The feasibility study measured blocked-against-unblocked agreement at
(H=56, N=4,096), (H=56, N=8,192) and (H=8, N=2,048), found them bit-identical,
and concluded blocking was exact. Every one of those N is a power of two, and
every one is below threshold 1 — precisely the region where the whole-sequence
kernel still agrees. The real canvases are not powers of two: 9,169 at
640x352 / 124f, 33,329 at the target.

It was also invisible to the comparison the study used. Blocked-against-
unblocked can only say the two differ; it cannot say which is wrong, and the
natural reading of a difference is that the *new* path introduced it. Only a
third reference that is neither implementation — here a float64 CPU oracle —
assigns the error to the right side. That is the method worth keeping from this
task, independent of query blocking.

The blocked result is block-size invariant and sits at the error floor at every
N. So the divergence is not blocking perturbing the answer — it is the shipped
path degrading, and blocking avoiding the degradation.

`probe-P1.mp4` was produced at 640x352 / 124 frames, above the second
threshold. It is the output of a numerically degraded attention path, which is
a second reason it is not a reference worth matching.

## Cost of blocking at 640x352 / 124 frames

Nil, within run-to-run noise. Denoise 148.2 s and 163.9 s blocked against
153.1 s, 150.6 s and 148.9 s unblocked. Video VAE decode 104.5 s and 101.6 s
blocked against 105.7 s, 104.3 s and 104.0 s unblocked — worth watching because
this change routes the video VAE's own attention through blocking too, and the
decoder is 38-40% of wall clock. It did not cost it anything here.

Every structural counter is unchanged against `base-P1`: 102 submissions, 314
direct, 400 linear, 100 attention, `alloc=0.000GiB`, and 72.495 GiB streamed to
four decimal places. The SDPA counter deliberately still counts one per
attention op rather than per block.

## The accuracy collapse is visible, and it is total

The threshold is not a numerical nicety. At 640x352 / 124 frames, `--steps 2`,
seed 42, **identical in every respect except the attention path**:

| contact sheet | attention | what it shows |
|---|---|---|
| `outputs/blk-640-blocked-contact.png` | blocked | a woman, red-orange hair, bangs, drop earring, necklace, hand on chest, black strap top |
| `outputs/blk-640-whole-contact.png` | whole (shipped) | **checkerboard noise, no subject** |
| `outputs/blk-probeP1-contact.png` | whole (the gate's reference) | **checkerboard noise, no subject** |

So the difference between the two attention paths at this canvas is the
difference between a picture and no picture, and `outputs/probe-P1.mp4` — the
file the correctness gate required this change to reproduce byte-for-byte — is
noise.

This bears directly on `docs/720p-baseline.md`'s Measurement C, which ran
640x352 / 124f at `--steps 6` with no LoRA, got checkerboard noise, and
concluded that six undistilled Euler steps may simply not converge and that a
distillation LoRA is therefore load-bearing. **That conclusion should be
re-examined.** Two evaluations on the blocked path converge to a coherent
subject at the same canvas with no LoRA at all. The checkerboard was the
degraded attention kernel, not the schedule.

## The 1280x704 / 124-frame probe

`--steps 2 --reuse 1 --layers 50 --seed 42 --ssd-streaming`, no LoRA,
`logs/blk-720p-probe.log`. N = 33,329.

**It encoded.** This canvas has never produced a frame on this machine; the
run completed and wrote 124 frames with audio.

| stage | wall |
|---|---:|
| Qwen text encoder | 9.153 s |
| DiT load | 8.808 s |
| Euler denoise, 2 evaluations | 1158.277 s |
| audio VAE decoder | 0.508 s |
| video VAE decoder (6x3 tiles at 288 px, 126 submissions) | 398.306 s |
| profiled fixed cost | 416.775 s |
| whole process | 1586 s |

`BF16 SSD stream 72.495 GiB` confirms exactly two evaluations, so
**579.1 s per denoiser evaluation**.

### Against the predictions

| quantity | predicted | measured |
|---|---|---|
| s per evaluation | 774-1290 s (`TODO.md` 1.3) | **579.1 s** — below the optimistic end |
| max process footprint | ~24 GiB (`docs/720p-baseline.md`), 17-25 GiB (`TODO.md` 1.5) | **11 GB peak, 9,823 MB flat through the denoise** |
| memory guardrail | the open question | **did not fire** |
| video VAE decode | 400-500 s, 2.4x downside risk | 398.3 s — at the optimistic edge, no 2.4x |

The footprint is the surprise. It sat at **9,823 MB unchanged from 61 s to
1,147 s** — the entire denoise — and only rose to 10 and then 11 GB during the
video VAE decode. For comparison `base-P1` measured **19 GiB at 640x352 /
124f**. This canvas has **four times the pixels at roughly half the memory**,
because the quadratic score-tensor term the baseline document attributed the
growth to is exactly what query blocking removes. Memory was never going to be
the wall once this landed; it is not close to being one.

`TODO.md` 1.4 argued the attention figure was optimistic by 10-30% because it
was measured in isolation on an idle GPU for 15 seconds, and ranked thermal
throttling over a long run as the largest unmeasured risk. Over a 26-minute
sustained run the real number came in **below** the isolation estimate, not
above. The isolation harness was pessimistic.

### How many evaluations fit in 90 minutes

579.1 s is **one sample, not a measurement**. Five repeats of the same
configuration at 640x352 spread 148.2 to 163.9 s, about 10%. Carrying that:

| per evaluation | evaluations in 5400 s |
|---|---|
| 521 s (−10%) | 9 |
| 579 s (measured) | 8 |
| 637 s (+10%) | 7 |

**Seven to nine.** The turbo LoRA's 4-15 s/evaluation does not move the band.
The bar the quality ladder had to clear was four, so four fits with roughly
double the budget to spare: a 4-evaluation run lands at **42 to 49 minutes**.

One assumption to retire: the study treated the video VAE decode as 38-40% of
wall clock. At this canvas it is 25.1% of this 2-evaluation run and would be
**14.6% of a 4-evaluation run**, because that share was formed at much smaller
canvases where the denoise itself was cheap. It is no longer the second-largest
term.

## What the nondeterminism is worth, now that it can be seen

`blk-after` and `blk-after2` are not byte-identical, but their contact sheets
are the same image: same subject, pose, hair, earring, necklace and hand. On
the blocked path the run-to-run variation is a fine-detail wobble. On the
whole-sequence path both runs are noise, so they differ wildly — which is most
of why the byte difference looked alarming.

It still needs its own investigation, and it still means every A/B at this
canvas carries an unmeasured per-run spread: absolute verdicts on a single run
stand, but small differences *between* rungs of a ladder do not.

## Files

- `h3_gpu.m` — `h3_gpu_sdpa`, `h3_gpu_sdpa_graph`, `h3_gpu_graph_data`.
- `tests/test_sdpa_blocking.c` — the block-invariance and oracle check.
- `H3_SDPA_QUERY_BLOCK` — block size, default 256 (0.89 GiB of scores at the
  target). 0 restores the single whole-sequence encode.
