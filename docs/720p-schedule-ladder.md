# Schedule ladder: how few denoiser evaluations does this model need?

Branch `measure/schedule-ladder`, measured 2026-09-14 at a cheap canvas
(640x352, 124 frames, 1.818:1 — the same aspect ratio as the 1280x704
target, so framing carries over) to answer the question the whole 720p
project turns on: with the target's ~550-750s per evaluation, only 6-7
evaluations fit a 90-minute budget. Is the output usable at that count?

Prompt used for every rung: `prompts/01-breath.txt` — locked-off medium
close-up, single subject (orange-haired woman in a ponytail), small
motion (one breath: shoulders rise and settle, one blink, the earring
sways). Same prompt and `--seed 42` on every rung. All runs used
`--layers 50 --ssd-streaming`, no other flags changed except the ones in
the table.

## Scope actually run

Only R1 and R2 were run. **R3 and R4 were not run — deliberately
descoped mid-task.** R1 alone already answers the yes/no question the
brief was built around (see below), and partway through R3 the GPU was
needed by another agent blocked waiting for it, so the ladder was cut
there rather than finishing the full sweep for its own sake:

- **R3** (`--steps 8 --reuse 2`, 5 evaluations) was started, then killed
  before completion once R1+R2 had already answered the question and the
  GPU was needed elsewhere. It costs more than R1 for no expected
  benefit once R1 clears — its only purpose was as a midpoint between R2
  and R4, and the ladder no longer needed a midpoint. No output or log
  from this aborted run is kept.
- **R4** (`--steps 20 --reuse 2`, no LoRA, 11 evaluations) was never
  started. It is the quality-ceiling reference and was always going to be
  skipped if time ran short, per the brief. At 11 evaluations it is
  unaffordable at the 1280x704 target regardless of how good it looks, so
  skipping it costs nothing for the feasibility decision.

This means the ladder is **not complete** as originally specified — it is
two points (4 and 6 evaluations, both with the turbo LoRA), not four. That
is enough to answer the go/no-go question, not enough to characterize the
full curve from noise (R1's predecessor, the undistilled 6-step control in
`docs/720p-baseline.md`, already established the noise end) up to the
undistilled ceiling.

## Results

| rung | flags | evaluations | denoise wall | s/eval | verdict |
|---|---|---:|---:|---:|---|
| R1 | `--steps 4 --reuse 1 --lora loras/turbo.safetensors` | 4 | 344.727s | 86.18 | **clean, usable** |
| R2 | `--steps 6 --reuse 1 --lora loras/turbo.safetensors` | 6 | 512.847s | 85.47 | clean, no visible gain over R1 |
| R3 | `--steps 8 --reuse 2 --lora loras/turbo.safetensors` | 5 | — | — | **not run** (descoped, see above) |
| R4 | `--steps 20 --reuse 2` (no LoRA) | 11 | — | — | **not run** (descoped, see above) |

Evaluation counts are read directly off each log's `denoise N/M` line count
(`0/4`...`4/4` for R1, `0/6`...`6/6` for R2), not inferred. Both logs
confirm via `ffprobe` that the output MP4 carries one `h264` video stream
and one `aac` audio stream, and `ffmpeg -f null -` decodes each file
end-to-end with zero errors.

s/eval for R1 and R2 agree to within 1% (86.18 vs 85.47), consistent with
AGENTS.md's framing that per-evaluation cost at a fixed canvas is stable
regardless of how many evaluations a schedule uses.

## Contact sheets

Both built with:
```
ffmpeg -v error -i outputs/ladder-R<n>.mp4 \
  -vf "select='eq(n\,0)+eq(n\,41)+eq(n\,82)+eq(n\,123)',tile=4x1" \
  -frames:v 1 -y outputs/ladder-R<n>-contact.png
```

**R1** (`outputs/ladder-R1-contact.png`, 4 evaluations): a clean,
coherent illustrated portrait across all four sampled frames. The subject
matches the prompt precisely — orange-red hair in a high ponytail, side
bangs, a single round drop earring, thin necklace with a pendant,
freckles, a black spaghetti-strap top, pink nail art, hand resting on the
chest. Composition and identity are stable frame to frame: same pose,
same framing, same lighting, only the small prompted motion (a very
slight shift in head/eye angle and hair strands consistent with a
breath) differs between frames. No checkerboard or dense noise, no
chromatic ringing, no ghosted or duplicated limbs, no blur or mush.
Fine detail — earring shape, nail art, lace trim at the bottom of frame,
individual hair strands — is intact in every sampled frame.

**R2** (`outputs/ladder-R2-contact.png`, 6 evaluations): equally
coherent and equally stable — same subject, same pose, same identity
held across all four frames, no checkerboard, no ghosting, no blur. The
one difference from R1: a faint concentric ring / vignette pattern is
visible in the flat lavender-grey backdrop of R2 that is not present in
R1's backdrop, which stays flat and clean. That is a minor artifact, and
it runs the wrong direction — it is something R2 has that R1 doesn't,
not an improvement. On the subject itself (face, hair, jewellery,
clothing, hands) **I cannot see a difference that favors R2** — detail
level, sharpness and stability all read the same as R1 to me. Put
plainly: I could tell R1 and R2 apart (the background rings), but the two
extra evaluations bought no visible improvement on the part of the image
the prompt is actually about, and introduced a small new artifact in the
background instead.

## Recommendation

**R1 — `--steps 4 --reuse 1` with `loras/turbo.safetensors`, 4
evaluations — is the cheapest rung tested and it is genuinely acceptable.**
It has no defects worth naming, matches the prompt precisely, and holds
identity across the full clip. R2 is also acceptable but buys nothing over
R1 for 50% more evaluations (6 vs 4) and adds a minor background artifact,
so there is no reason to pay for it once R1 already clears the bar.

## The question this project turns on

**At 6 or fewer denoiser evaluations, with the turbo LoRA, is the output
usable? Yes.** R1 (4 evaluations) is clean and matches the prompt with no
visible defects; R2 (6 evaluations) is equally clean. This directly
contradicts the earlier finding in `docs/720p-baseline.md`, where 6 Euler
steps **without** the turbo LoRA produced pure checkerboard noise — the
LoRA, not the step count, is what makes low-evaluation counts viable on
this model. Basis: two full runs at the target frame count and aspect
ratio, each independently confirmed by evaluation count in the log, by
denoise wall time, by a four-frame contact sheet actually read and
described above, and by `ffprobe`/`ffmpeg -f null -` decode checks — not
by a single sample or by assumption.

## Caveat that qualifies this result: the prompt tested is nearly static

`01-breath.txt` is a locked-off medium close-up whose only prescribed
motion is one breath — a few centimetres of shoulder rise, one blink, a
slight sway of the earring. Across the four sampled frames (0, 41, 82,
123) the subject barely moves at all. **This is the easiest possible case
for temporal coherence at 4 evaluations**: a schedule that struggles with
motion has very little motion here to struggle with. The result above
should be read as "4 evaluations can render a stable, high-detail single
subject with near-zero motion," not as "4 evaluations hold up under real
motion" — those are different claims, and only the first one has been
measured.

**Before treating 4 evaluations as settled for real content, re-run this
same ladder (at minimum R1, ideally R1+R2) against `prompts/04-water.txt`.**
That prompt is the opposite case on this axis — martial-arts strikes,
spinning kicks, a sweeping arm motion, water wrapped around the arms like
whips, ending in a forward strike — genuine fast choreography rather than
a held pose. If 4 evaluations still produce a stable, recognizable subject
with coherent (not ghosted or smeared) motion on that prompt, the result
here generalizes. If it doesn't, the number of evaluations this project
can afford may need another look specifically for motion-heavy shots,
independent of the static-portrait case this ladder actually tested.
