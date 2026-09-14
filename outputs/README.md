# outputs/

Generated video/image artifacts from manual runs of `./h3` and from the
contact-sheet `ffmpeg` command in `AGENTS.md`. This directory is **gitignored**
(see `.gitignore`) and lives only on this machine — nothing here is backed up
by git, so before deleting anything here, cross-check it against `docs/`,
`logs/`, `README.md`, `AGENTS.md`, and `CONTEXT.md` on every branch, not just
`main`. Below is a manifest of every surviving file, grouped by what it is
evidence for, followed by a record of what was removed in the 2026-09-14
cleanup pass and why.

Unless noted otherwise, every run below used `prompts/01-breath.txt`
(locked-off portrait, one subject, one breath) and `--seed 42`.

## 720p native generation (`feat/720p-blocked-attention`)

Query-blocked attention: the fix that lets 1280x704 encode at all, and the
evidence that the shipped whole-sequence attention path silently degrades
above N ≈ 9,100 tokens. Full writeup: `docs/720p-blocked-attention.md`.

```
blk-720p-probe.mp4 — 1280x704, 124f, 2 evals, seed 42, no LoRA, --layers 50
  --steps 2 --reuse 1 --ssd-streaming. The first native 720p generation this
  project ever produced (579.1 s/evaluation, footprint 9,823 MB flat through
  denoise). logs/blk-720p-probe.log, docs/720p-blocked-attention.md
blk-720p-contact.png — contact sheet of blk-720p-probe.mp4 (frames 0/41/82/123).
  docs/720p-blocked-attention.md
blk-off.mp4 / blk-off2.mp4 — 640x352, 124f, 2 evals, seed 42, whole-sequence
  attention (H3_SDPA_QUERY_BLOCK=0, the shipped code path), identical
  invocation run twice. Byte-differ from each other: proof the shipped path is
  nondeterministic at this canvas. logs/blk-off.log, logs/blk-off2.log,
  docs/720p-blocked-attention.md
blk-after.mp4 / blk-after2.mp4 — same canvas and flags as blk-off, but with
  query-blocked attention (block 256). Also byte-differ from each other, but
  the divergence is a fine-detail wobble, not noise (see below) — the same
  pair on the fixed attention path. logs/blk-after.log, logs/blk-after2.log,
  docs/720p-blocked-attention.md
blk-after2-contact.png — contact sheet of blk-after2.mp4.
  docs/720p-blocked-attention.md
blk-640-blocked-contact.png — contact sheet of blk-after.mp4: a coherent
  subject (woman, red-orange hair, drop earring, hand on chest).
blk-640-whole-contact.png — contact sheet of blk-off.mp4: checkerboard noise,
  no subject. Paired with the file above, this is the single most important
  image in the project — it shows the shipped attention path producing noise
  where the blocked path produces a picture, at identical cost.
  docs/720p-blocked-attention.md
blk-probeP1-contact.png — contact sheet of probe-P1.mp4 (below): the former
  baseline/gate reference, also checkerboard noise, confirming the reference
  itself was a product of the degraded kernel, not a usable target.
  docs/720p-blocked-attention.md
```

## Nondeterminism evidence

The proof that this pipeline is (a) reproducible at small canvases and (b) not
reproducible at 640x352+, which sent this project down the query-blocking and
sigma-schedule investigations. See `docs/nondeterminism.md`
(`research/nondeterminism` branch) and `docs/720p-blocked-attention.md`.

```
det-d1.mp4 / det-d2.mp4 — 512x288, 22f, 4 evals, small-canvas determinism pair
  from 2 September. Byte-identical (cmp confirmed). The only proof this
  pipeline is reproducible at all; irreplaceable, the session that produced
  them is over. logs/det-d1.log, logs/det-d2.log,
  docs/nondeterminism.md, docs/720p-blocked-attention.md
probe-P1.mp4 — 640x352, 124f, 2 evals, seed 42, no LoRA, whole-sequence
  attention, --steps 2 --reuse 1 --layers 50 --ssd-streaming. Originally the
  720p-baseline's footprint-scaling point P1 and the correctness gate's
  reference file (the video query-blocking was required to reproduce
  byte-for-byte); later shown to itself be checkerboard noise from the
  degraded whole-sequence kernel. No dedicated log (the P1/P2/P3/P4 sweep in
  docs/720p-baseline.md was sampled externally with `footprint -p <pid>`, not
  logged by h3 itself). docs/720p-baseline.md, docs/720p-blocked-attention.md,
  docs/nondeterminism.md
```

## Quality ladders — how many denoiser evaluations are enough

Two ladders, same 640x352/124f canvas, different prompts, because "how many
evaluations" turned out to depend on how much the content moves.

```
ladder-R1.mp4 / ladder-R1-contact.png — --steps 4 --reuse 1, turbo LoRA,
  4 evals, prompts/01-breath.txt (static portrait). Clean, recommended rung of
  the first ladder. logs/ladder-R1.log, docs/720p-schedule-ladder.md
ladder-R2.mp4 / ladder-R2-contact.png — --steps 6 --reuse 1, turbo LoRA,
  6 evals, same prompt. Equally clean, no visible gain over R1, a minor
  background-ring artifact. logs/ladder-R2.log, docs/720p-schedule-ladder.md
lad2-L1.mp4 / lad2-L1-contact.png — --steps 4 --reuse 1, turbo LoRA, 4 evals,
  prompts/06-hands.txt (fast riffle-shuffle motion, the re-test the first
  ladder asked for). Clean but detail-starved: correct hand anatomy, unreadable
  card faces. logs/lad2-L1.log, docs/720p-schedule-ladder-2.md
lad2-L2.mp4 / lad2-L2-contact.png — --steps 8 --reuse 3, turbo LoRA, same
  4-evaluation cost as L1 but on the 8-step sigma grid with 4 of 8 points
  extrapolated. Broken: woven cross-hatch, chromatic ringing, damaged hand
  anatomy — proof that evaluation count alone does not determine quality.
  logs/lad2-L2.log, docs/720p-schedule-ladder-2.md
lad2-M1.mp4 / lad2-M1-contact.png — --steps 6 --reuse 1, turbo LoRA, 6 evals.
  Clean, skin texture returns, card back becomes a structured (if unreadable)
  filigree. logs/lad2-M1.log, docs/720p-schedule-ladder-2.md
lad2-M2.mp4 / lad2-M2-contact.png — --steps 8 --reuse 1, turbo LoRA, 8 evals.
  Best of the five: card faces become legible (countable pips, index corner).
  The recommended target-canvas configuration.
  logs/lad2-M2.log, docs/720p-schedule-ladder-2.md
lad2-M3.mp4 / lad2-M3-contact.png — --steps 10 --reuse 1, turbo LoRA, 10 evals.
  Clean but a lateral move against M2 (wins one frame, loses another) plus
  early over-sharpening; also the run with an unexplained 23% per-evaluation
  timing anomaly. logs/lad2-M3.log, docs/720p-schedule-ladder-2.md
```

## Baseline / control measurements (`measure/720p-baseline`)

The no-LoRA control that established (wrongly, as the blocked-attention work
later showed) that undistilled low-step schedules produce noise at this
canvas. `docs/720p-baseline.md`.

```
control-640x352-124f.mp4 — 640x352, 124f, 6 evals (--steps 6 --reuse 1
  --layers 50, no LoRA). Did not converge: checkerboard/mosaic noise, no
  subject. Cited as Measurement C in docs/720p-baseline.md.
  NOTE: docs/720p-baseline.md names its log as `logs/control-640x352.log`,
  but that log file does not exist on disk — see "Missing, cited files" below.
control-contact.png — contact sheet of the control video (frames 0/41/82/123),
  the noise result described above. docs/720p-baseline.md
control-frame62.png — frame 62 pulled separately from the control video "to
  rule out an artifact of the tile filter" (docs/720p-baseline.md, Measurement
  C) — not named explicitly in the doc, but it is the file that check
  produced; kept for the same reason as control-contact.png.
```

## Older history

```
(none — every file predating the 720p work was either promoted to a section
above or removed in this pass; see "Deleted in this pass" below.)
```

## Deleted in this pass (2026-09-14)

33 files removed, 62 → 30 in `outputs/` (plus one stray `.DS_Store`, not part
of the 62). None of these were named in any committed document on any branch
(`main`, `docs/retire-spec`, `feat/*`, `measure/*`, `plan/*`, `research/*`),
in `README.md`, `AGENTS.md`, or `CONTEXT.md`.

- `.DS_Store` — macOS Finder metadata, not a generation artifact.
- `ab-A.mp4` — early (2 Sept) A/B smoke test. Numbers, if any, are in
  `logs/ab-A.log`, which is kept; not cited anywhere.
- `bench-A.mp4`, `bench-B.mp4`, `bench-D.mp4`, `bench-E.mp4`, `bench-F.mp4`,
  `bench-G.mp4`, `bench-I25base.mp4`, `bench-L35.mp4`, `bench-L35mem.mp4`,
  `bench-L50.mp4`, `bench-Q480.mp4` (11 files) — `TODO.md` cites their
  numbers via `logs/bench-grid.csv` and individual `logs/bench-*.log` files
  (e.g. `bench-A.log` for the gate-ranked DiT skip line, `bench-H25lora.log`
  for the turbo pair count). Verified the L35/L50 numbers TODO.md quotes from
  `bench-grid.csv` (518.5 s / 753.6 s) match the individual `.log` files
  directly, so deleting the videos loses no measurement — the underlying
  `.log` files are kept. `logs/bench-grid.csv` itself is missing; see below.
- `contact-01.png`, `contact-02.png` — early (2 Sept) contact sheets, no
  matching log, no doc reference.
- `contact-e2e-a1-lora.png`, `contact-e2e-a2-base.png`, `contact-e2e-b-15s.png`
  and `e2e-a1-lora.mp4`, `e2e-a2-base.mp4`, `e2e-b-15s.mp4` — `TODO.md` cites
  `logs/e2e-a1-lora.log` and `logs/e2e-a2-base.log` by name for their
  dispatch-count/wall-time numbers (the turbo-LoRA-branch cost measurement),
  but never the video or image files; those logs are kept.
- `h3-studio.mp4` — the macOS app's hardcoded default output path
  (`macos/Sources/H3Studio/{AppState,Params}.swift`, `feat/macos-app`), a
  smoke-test leftover, not evidence for any written finding.
- `mem-probe.mp4`, `mem-probe-hires.mp4` — the runs behind `AGENTS.md`'s
  448x576/56f and 768x1344/22f memory-footprint numbers, but `AGENTS.md`
  describes the numbers narratively and never names either file; the source
  logs (`logs/mem-probe.log`, `logs/mem-probe-hires.log`) are kept.
- `ponytail-01-breath.mp4`, `ponytail-01-breath-56f.mp4` — early (2 Sept)
  runs, no log, no doc reference.
- `probe-P2.mp4`, `probe-P3.mp4` — judgement call, deleted. These correspond
  to points P2 and P3 in `docs/720p-baseline.md`'s footprint-scaling table
  (mtimes sequential with `probe-P1.mp4`, which is kept), but the doc reports
  their footprint/timing numbers directly in its own table text and never
  cites the video content itself (no contact sheet, no visual claim), unlike
  `probe-P1.mp4` whose *image* became load-bearing evidence in
  `docs/720p-blocked-attention.md`. No dedicated log exists for either (the
  P1-P4 sweep was sampled externally). Deleting them loses no recorded
  measurement.
- `smoke-contact.png`, `smoke-conv-b.mp4` — a LoRA convention-B smoke test
  (`logs/smoke.log`, 6 Sept), superseded by the `make real-lora` run recorded
  in `docs/720p-schedule-ladder-2.md`'s "Step 0"; not cited by filename
  anywhere.
- `water-t2v-mountain.mp4`, `water-t2v-photoreal.mp4`,
  `water-t2v-photoreal-10step.mp4` — early (4 Sept) prompt experiments with
  `prompts/04-water.txt`, not cited by any current document.

### Missing, cited files (found during this pass, not caused by it)

- **`logs/control-640x352.log`** is named explicitly, twice, in
  `docs/720p-baseline.md` (`measure/720p-baseline` branch) as the log for the
  Measurement C control run — but the file does not exist anywhere in
  `logs/`. This predates this cleanup pass (nothing in `outputs/` referenced
  it) and is reported here because it is exactly the kind of gap the task
  brief asked to flag: a document cites a file that is already gone.
- **`logs/bench-grid.csv`**, cited by `TODO.md` (`plan/720p-todo` branch) for
  the L35/L50 comparison, also does not exist on disk. Its numbers were
  cross-checked against the individual `bench-L35.log` / `bench-L50.log`
  files, which do exist and agree, so nothing is actually lost — but the
  consolidated file itself is gone.
