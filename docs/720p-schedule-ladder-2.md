# Schedule ladder 2: how many evaluations, on a prompt that actually moves

Branch `feat/720p-blocked-attention`, measured 2026-09-14 at 640x352 / 124
frames (N = 9,169), the same 1.818:1 screening canvas as the first ladder.
This re-runs the quality question because the first ladder's premise was void
and because the cost estimate it was budgeted against was wrong by 40%.

Every run: `--seed 42 --layers 50 --ssd-streaming`, prompt
`prompts/06-hands.txt`, output `outputs/lad2-<run>.mp4`, log
`logs/lad2-<run>.log`, contact sheet `outputs/lad2-<run>-contact.png` at frames
0 / 41 / 82 / 123.

## Why this was redone

Three things changed under the first ladder.

1. **The kernel.** MPSGraph's whole-sequence attention collapses in accuracy
   above N ≈ 9,100 (`docs/720p-blocked-attention.md`). At this exact canvas it
   produced checkerboard noise with no subject. Query-block tiling fixed it.
   Every quality judgement made at this canvas before that fix measured the
   broken kernel.
2. **The cost.** One evaluation at the 1280x704 target measured **579 s**, not
   the 774-1290 s assumed. With ~417 s of fixed cost, 6 evaluations is ~65 min,
   8 is ~84 min, 10 is ~103 min. The budget is not 4 evaluations.
3. **The prompt.** Every previous quality judgement in this project rests on
   `prompts/01-breath.txt`, a locked-off portrait whose only motion is one
   breath. `06-hands.txt` is a magician's riffle shuffle: fast subject motion
   plus the hardest structural subject there is, fingers and joints.

## Step 0: `make real-lora` after the convention-B merge

Green, exit 0, whole target. `make test` covered the merge but `make real-lora`
is a separate target that had not been run since it. It does real GPU work
against the checkpoint, and all five LoRA files on disk are exercised.

- T1 oracle at strength 100 on turbo, combat, anime_v7, and mystic (strength 3,
  block 24); turbo+combat and turbo+anime_v7 stacked.
- Every pair matched the fused float32 oracle at the bf16 error floor
  (f32-acc rel-L2 ≈ 1.66e-03 throughout, indistinguishable from the no-delta
  control) and strength 0 was exact.
- The significance guard behaved: T1 at strength 1 **failed** with
  `delta too weak to be significant`, as designed.
- `lora-identity` and T6 hot-swap passed.

**No regression from the convention-B merge.** Nothing here blocked the
experiment.

## What was run, and what it cost

| run | flags | evaluations | denoise | s/eval | streamed | verdict |
|---|---|---:|---:|---:|---:|---|
| L1 | `--steps 4 --reuse 1` + turbo | 4 | 336.127 s | 84.03 | 144.272 GiB | clean, detail-starved |
| L2 | `--steps 8 --reuse 3` + turbo | 4 | 317.844 s | 79.46 | 144.272 GiB | **broken** |
| M1 | `--steps 6 --reuse 1` + turbo | 6 | 477.083 s | 79.51 | 216.050 GiB | clean |
| M2 | `--steps 8 --reuse 1` + turbo | 8 | 635.545 s | 79.44 | 287.827 GiB | **clean, best** |
| M3 | `--steps 10 --reuse 1` + turbo | 10 | 975.729 s | 97.57 | 359.604 GiB | clean, over-etched |

All five: `h264` video 640x352 / 124 frames plus one `aac` audio stream,
`ffmpeg -f null -` decodes end to end with zero errors.

L3 and L4 (no-LoRA at 4 and 8 evaluations) were cancelled before starting, on
vendor guidance that the undistilled base model needs ~20 steps — see question 1.

### Evaluation counts are confirmed three ways, and the old method was unsound

Streamed bytes per evaluation are 36.07 / 36.01 / 35.98 / 35.96 GiB — constant,
so the streamed total divides out the evaluation count independently of any
progress counter. `attention=` divides by exactly 50 per evaluation in every
run (200/4, 300/6, 400/8, 500/10), and `direct=` by exactly 807. One DiT pass
per evaluation, no doubling.

**L2 is the case that matters.** Its `denoise N/M` counter reads `8/8`, but it
ran **4 evaluations**: the log says `selected reuse schedule has 4 evaluations`
and it streamed 144.272 GiB, byte-for-byte identical to L1's. L1 and L2 cost
exactly the same, which is what L2 was for.

That also retracts a method. `docs/720p-schedule-ladder.md` states evaluation
counts were "read directly off each log's `denoise N/M` line count". That
counter counts **grid steps, not evaluations**. It was right for R1 and R2 only
because both used `--reuse 1`. Use the `selected reuse schedule` line or the
streamed bytes instead.

### One timing anomaly, unexplained

Per-evaluation cost held at 79.4-84.0 s across L1, L2, M1 and M2 — within 6%,
and flat in step count, which confirms per-evaluation cost does not depend on
how many evaluations a schedule uses. **M3 came in at 97.6 s/eval, 23% above
that band.** M3 was the fifth consecutive heavy run in a ~75-minute session, so
thermal drift is the obvious candidate, but L1 was the *first* run and also the
second-slowest, which argues against a clean thermal ramp. A few seconds of
`ffmpeg` contact-sheet work overlapped M3's start; that cannot account for 180 s.

**I cannot attribute this and I am not going to guess.** It is worth knowing
because `TODO.md` 1.4 ranked sustained-load throttling as the largest unmeasured
risk, and `docs/720p-blocked-attention.md` found a 26-minute run came in *below*
the isolation estimate. Both can be true if the drift appears across a long
back-to-back session rather than within one run. If it is real, the 10-evaluation
target projection is worse than linear.

## Contact sheets, as read

**L1 — 4 evaluations.** Real hands, real deck, and the shuffle genuinely
progresses: frames 0 and 123 show the deck closed, frames 41 and 82 show the two
halves interlaced. Not frozen. Finger anatomy is correct — three fingers plus
thumb on the left hand, thumb and two fingers on the right, nails well formed,
distal and proximal joint creases present, proportions right. The deck edge
resolves into individually separated cards. **What is missing is detail.** The
card faces are red and gold smears with no readable structure; skin is waxy and
smooth with no pores or knuckle texture. There is a translucency artifact where
cards overlap fingers — the fingertip shows *through* the card near the top of
frames 41 and 82. No checkerboard, no chromatic ringing, no tile seams.

**L2 — 4 evaluations, 8-step grid. Broken, and not subtly.** A dense regular
diagonal cross-hatch covers the card block in every frame — a repeating lattice,
not card detail. The deck edge is a regular stair-stepped comb rather than
individual cards. Frame 123 carries severe chromatic ringing: fingers outlined
in cyan and magenta, the deck edge bright cyan, and green speckle in the black
background. Anatomy is damaged: the left hand dissolves into a smeared wedge
with no nails and no joints; in frame 123 the fingers render as concentric
ridged ovals. Only the right thumb survives as a recognisable structure. This is
far larger than the run-to-run wobble and there is no ambiguity about it.

**M1 — 6 evaluations.** Clean. Skin texture returns: knuckle creases and pores
are visible where L1 was waxy. Anatomy correct on both hands. The deck edge
resolves ~15 separated cards with visible red back-printing. The card face
becomes a structured blue filigree with real printed detail, but it is not a
readable card — you can see that it is printed, not what it is. Mild
translucency remains at top-left. No checkerboard, ringing, woven texture or
seams.

**M2 — 8 evaluations. The best of the five.** Card faces become *legible*. Frame
41 shows a diamonds card with **eight countable pips in the correct two-column
arrangement plus a proper index corner** (red character with a small pip beneath
it), and the card back at top-left resolves into an ornate blue printed pattern.
Frame 82 shows a court card with a readable mirrored figure, gold robe, white
ruff and a red index mark. Anatomy is clean on both hands, skin is natural
rather than etched, deck edges crisp. A white shirt cuff appears in the
background as actual fabric. No artifacts of any kind.

**M3 — 10 evaluations. Clean, but a lateral move, and the vendor's warning is
visible.** Frame 82 is a nameable **Queen of Diamonds** — red diamond pip at the
index corner, crowned gold figure, black line work — arguably more identifiable
than M2's court card. But frame 41 goes the other way: **six pips instead of
eight, no clear index corner**, and the top-left card is a red and yellow smear
where M2 resolved a clean printed back. The over-sharpening signature is there:
skin creases on the right index finger read as engraved hard lines rather than
texture, nail edges carry hard bright rim highlights, and the court card's robe
interior is a busy gold-and-black tangle that does not resolve into a coherent
figure — over-detailed noise wearing the costume of detail. No checkerboard,
ringing, woven texture or seams.

**M2 and M3 trade wins frame by frame. Neither is consistently better.**

## Setup checks against the vendor guidance

**LoRA strength is 1.0.** `./h3 --help` documents
`--lora PATH[:STRENGTH]  Apply a LoRA adapter (default 1.0, ...)`. Every run
here passed `--lora loras/turbo.safetensors` with no explicit strength, so all
five ran at 1.0. The load report confirms the file was read: `259 pairs applied,
convention: lora_A/lora_B, ranks: 64 x208, 16 x51, AdaLN pairs: 51, resident
743.7 MiB`.

**There is no classifier-free guidance in h3, so CFG 1.0 is satisfied by
construction.** Grepping the whole C/ObjC source for `cfg`, `guidance`,
`classifier`, `uncond` and `negative prompt` returns nothing — the only hit is
the word "unconditionally" inside a comment in `h3_lora.c`. There is no negative
prompt on the CLI and no unconditional branch in the denoiser. The counters
confirm it from the outside: `attention=` is exactly 50 per evaluation (one op
per DiT block) and streamed bytes are exactly one DiT weight-set per evaluation.
CFG would double both. **h3 does not do guidance, and cannot be made to.**

**Scheduler `simple`: same thing, and this is now checked against source rather
than assumed.** ComfyUI's `simple_scheduler` (`comfy/samplers.py`) is:

```python
def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)
```

h3's `h3_shifted_sigma` (`h3_host.c:129`) is:

```c
int base_index = (index * 1000) / steps;
float base = (float)(1000 - base_index) / 1000.0f;
return shift * base / (1.0f + (shift - 1.0f) * base);
```

with `schedule->video[steps] = 0.0f` appended. Both take a 1000-entry sigma
ramp, select entry `1000 - int(x*1000/steps)` by **uniform index subsampling**,
and append a terminal zero. That selection is element-for-element identical.

The only difference is *where the shift is applied*: ComfyUI shifts the model's
sigma table up front (`ModelSamplingSD3`) and the scheduler then subsamples it;
h3 subsamples first and shifts each selected element. The shift is a monotone
elementwise map and subsampling only selects by index, **so the two operations
commute and the resulting grids are identical**. h3's default video grid *is*
ComfyUI `simple` at shift 12.

One genuine difference, and it is not a footnote: **h3 runs two independent
grids**, video at shift 12 and audio at shift 3 (`H3_VIDEO_SIGMA_SHIFT`,
`H3_AUDIO_SIGMA_SHIFT` in `h3_host.h`). A single-modality ComfyUI sampler has no
equivalent to the audio grid. So "scheduler simple, shift 12" describes h3's
video path exactly and says nothing about its audio path.

For reference, the video grids actually used (shift 12, terminal zero omitted):

| steps | sigmas |
|---:|---|
| 4 | 1.000, 0.973, 0.923, 0.800 |
| 6 | 1.000, 0.984, 0.960, 0.923, 0.857, 0.706 |
| 8 | 1.000, 0.988, 0.973, 0.952, 0.923, 0.878, 0.800, 0.632 |
| 10 | 1.000, 0.991, 0.980, 0.966, 0.947, 0.923, 0.889, 0.837, 0.750, 0.571 |

L2 evaluated the 8-step grid at indices 0, 3, 6, 7 — sigma 1.000, 0.952, 0.800,
0.632 — and extrapolated the rest.

## The questions

### 1. Is the turbo LoRA load-bearing, or merely an accelerator?

**Neither claim is settled by this experiment, and the original claim is
retracted anyway — for a different reason than the one that prompted the
retest.**

The claim as written in `TODO.md:32` — the turbo LoRA "is not an optional
accelerator on this plan, it is a load-bearing part of the only schedule that
fits" — rested entirely on one observation: 6 undistilled steps at this canvas
produced checkerboard noise with no subject. **That evidence is void.** The
checkerboard was the degraded attention kernel, which at N = 9,169 produced
checkerboard noise *with* a schedule that works. `docs/720p-blocked-attention.md`
shows two evaluations on the blocked path converging to a coherent subject at
this canvas with no LoRA at all. The premise is gone.

But the conclusion does not flip to "merely an accelerator" either. L3 and L4
were cancelled on vendor guidance that the undistilled base needs ~20 steps, so
4 or 8 undistilled evaluations were never real configurations and would have
measured an undercooked model. At ~579 s per evaluation, 20 undistilled
evaluations is **over three hours** at the target — unaffordable regardless of
quality.

So the honest statement, which is what the doc should carry:

> The turbo LoRA is **not** required to make the model produce an image at low
> step counts — that claim was an artifact of the broken attention kernel and is
> retracted. It **is** required to make a low-step schedule affordable, because
> the undistilled alternative needs roughly 20 evaluations and does not fit any
> budget under discussion. The practical conclusion is unchanged; the reasoning
> behind it was wrong and is now replaced.

Scope note: `docs/720p-baseline.md` does **not exist on this branch** (it lives
at commit `249b843`), so the retraction is recorded here and applies to
`TODO.md:26-33` and to that document's Measurement C wherever it is merged.

### 2. Does the sigma-grid change help, hurt, or do nothing?

**It hurts, decisively, at identical cost.** L1 and L2 streamed the same
144.272 GiB and ran the same 4 evaluations. L1 is clean; L2 has a repetitive
woven cross-hatch over the whole subject, chromatic ringing, green background
speckle, and destroyed hand anatomy.

This reproduces a documented failure exactly. `README:519` lists the candidates
the released base grid beat, names **"linear velocity extrapolation"** among
them, and gives the failure mode as **"damaged motion or a repetitive woven
background"**. Both halves appear here — except the weave is on the subject, not
just the background, which is worse than documented.

**One caveat, and it matters for how far this generalises.** L2 changes two
things at once, not one. It integrates on an 8-point grid *and* it extrapolates
the four skipped velocities instead of evaluating them. The artifact signature —
a repeating woven lattice — matches the extrapolation family README names, not
obviously the sigma placement. So the clean reading is "`--reuse 3` at 8 steps is
harmful on this prompt", and the narrower question "would those four sigmas help
if all four were genuinely evaluated?" is **not answered here** and would need a
code change to ask. Given README already rejected this family on the model
author's own sweep, it is not worth building.

### 3. Does 4 evaluations hold up under real motion?

**Structurally yes, in detail no.** This is the first test of this project's
schedule against real motion, and the result splits.

What holds: the shuffle genuinely progresses across the clip, hands are
anatomically correct with the right finger count and correct joints, the deck
reads as a deck with individually separated cards, and there are no artifacts.
The first ladder's worry — that 4 evaluations would ghost or smear under motion
— **did not happen**. Temporal coherence is fine.

What does not hold: detail. Card faces are unreadable smears and skin is waxy.
At 640x352 that is the difference between "a video of hands shuffling cards" and
"a video of *these* cards". The vendor's framing matches what I see — 4 steps is
fine for "static shots, slow pans, talking-head framing" — and it explains the
first ladder cleanly: `01-breath.txt` is a static portrait with nothing for 4
evaluations to lose.

So the first ladder's recommendation is not *wrong*, it is **out of scope**. It
measured a case where 4 evaluations suffice and generalised it to cases where
they do not.

### 4 and 5. Which configuration, and which of 6 / 8 / 10 is cheapest that holds?

**`--steps 8 --reuse 1 --lora loras/turbo.safetensors` — 8 evaluations.**

```sh
./h3 --profile -d ./MiniMax-H3 --lora loras/turbo.safetensors \
  -p "$(cat prompts/06-hands.txt)" \
  --width 1280 --height 704 --frames 124 --steps 8 --reuse 1 \
  --layers 50 --ssd-streaming -o outputs/720p.mp4 > logs/720p.log 2>&1
```

Reasons, in order:

1. **8 is where fine printed detail becomes correct**, not merely present. M2 is
   the only run that resolved eight countable pips *and* an index corner *and* a
   clean printed card back in the same frame.
2. **10 does not improve on it.** M3 wins frame 82 and loses frame 41. The gain
   is not monotonic, which is the signature of a lateral move, and the
   over-sharpening the vendor warns about is visible in M3's etched skin and
   hard rim highlights.
3. **8 is the top of the vendor's stated useful range**, which independently
   agrees with what the contact sheets show on our own hardware.
4. It fits 84 minutes at the target, inside the original 90-minute budget with
   no need to spend the relaxation.

Fallback if a shot is static (locked-off portrait, slow pan, talking head): 6
evaluations at ~65 min is genuinely clean and loses only card-face-class detail
the shot does not contain. Do not drop to 4 for anything with fast motion or
fine structure.

### 6. Is 6 to 8 worth 19 minutes? Is 8 to 10 worth another 19?

At the target: 6 evaluations ≈ 65 min, 8 ≈ 84 min, 10 ≈ 103 min (579 s per
evaluation plus ~417 s fixed).

**6 to 8: yes, clearly.** Those 19 minutes buy the step from "you can see the
cards are printed" to "you can read the cards". Concretely: eight countable pips
and an index corner where M1 had a blue filigree smear, plus a resolved printed
card back and a shirt cuff that reads as fabric. On a prompt about hands and
cards, that is the difference between the shot working and the shot being an
impression of the shot. It is repeatable across both mid-shuffle frames and it
is far larger than the run-to-run wobble.

**8 to 10: no.** Those 19 minutes buy a trade, not an improvement — one frame
gains a nameable card, another loses two pips and its index corner — plus the
beginning of over-sharpening on skin. I would not pay it, and I would not expect
the owner to see a difference he likes. If he wants to judge it himself,
`outputs/lad2-M2-contact.png` and `outputs/lad2-M3-contact.png` are the two
sheets to put side by side; the honest summary is that they are equivalent.

The one thing that could change this: M3's 97.6 s/eval anomaly. If that is real
sustained-load throttling rather than noise, 10 evaluations at the target is
worse than 103 minutes, which makes the answer more firmly no rather than less.

## What would sharpen this

- **The 8-evaluation run at the actual target**, once. Everything here is at
  1/4 the pixels, and fine-detail conclusions are exactly the ones a resolution
  change can move.
- **Whether M3's 97.6 s/eval is thermal.** One repeat of M2 at the end of a cold
  session versus the end of a hot one would settle it, and it bears on every
  target-canvas projection in `TODO.md`.
- **A second seed on M1 vs M2.** The 6-to-8 gain is large and appears in both
  mid-shuffle frames, so I am confident in it, but it rests on one seed.
