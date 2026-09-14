# 720p plan: how many denoiser evaluations fit in 90 minutes, and what to build

> ## Correction, 2026-09-14: the cost model below is superseded
>
> Sections 1.1, 1.3 and 1.4 were written before the blocked-attention path was
> measured in situ at the target canvas. Their central numbers are wrong and the
> feasibility verdict that follows from them is wrong.
>
> | quantity | this document says | measured at 1280x704 / 124f |
> |---|---|---|
> | per denoiser evaluation | 774 to 1290 s | **579.1 s** |
> | fixed cost | 425 to 525 s | **416.8 s** |
> | evaluations inside 90 min | about 4 | **7 to 9** |
> | a 4-evaluation run | 59 to 95 min, marginal | **42 to 49 min** |
>
> The in-situ penalty argued in 1.4 does not exist: over a 26-minute sustained
> run the attention figure came in *below* the isolation estimate, so the
> isolation harness was pessimistic rather than optimistic. Memory was not the
> wall either, 9,823 MB flat through the denoise against the 17 to 25 GiB
> predicted in 1.5, because query blocking removes the quadratic score tensor.
>
> The measurement is `docs/720p-blocked-attention.md`.
>
> Everything below is kept as written. Section 3, "Rejected", is the part that
> still earns its place: it records what was already ruled out, so a later
> round does not re-propose it.

Adversarial review of the twelve-agent feasibility study (six research axes,
six source-verification agents), scored against the revised objective: **one
1280x704 / 124-frame generation, native, inside 90 minutes, at output the owner
accepts.**

Every number is labelled with where it came from. `measured-here` means this
machine, this week, in `logs/`, in `docs/720p-baseline.md` (commit `249b843`),
or in the study's own scratch harnesses (`exact.m`, `batch2.m`). Everything
else is extrapolation and says so.

---

## 1. Feasibility, as a function of evaluation count

**Feasible if and only if the output is acceptable at 4 denoiser evaluations or
fewer.** At 5 it fits only if the in-situ penalty on the attention measurement
turns out to be near zero. At 6 or more it does not fit, and no technique in
this study changes that.

The quality half of that conditional is being measured separately on
`measure/schedule-ladder`. This section supplies the time half so the two slot
together.

> **Do not use `outputs/control-640x352-124f.mp4` as a quality reference.** It
> ran at `--steps 6 --reuse 1 --layers 50` with **no LoRA** and produced
> checkerboard noise with no subject. The low-step schedule is not reachable
> without `loras/turbo.safetensors`, which resolves to
> `minimax_h3_turbo_4step.safetensors`, a **4-step distillation**. Every
> historical low-step run in `logs/bench-*.log` used it. The turbo LoRA is
> therefore not an optional accelerator on this plan, it is a load-bearing part
> of the only schedule that fits, and its cost is accounted for below.

### 1.1 Cost of one denoiser evaluation at 1280x704 / 124 frames

Token count is **N = 33,329**, measured (`logs/base-720p-N.log`, MPS assertion
`dimensionLengths = [33329, 33329, 56, ...]`). Two agents used other numbers and
both are wrong for this target: the `approximate` axis assumed N ~ 108,000
(3.2x too high, which inflates every FLOP and attention-share figure it reports),
and `apple-platform` used N = 38,274, which is `logs/bench-HF5s.log`, a
15-second run, not this canvas.

**(a) Attention, query-blocked: 550 to 750 s per evaluation.** Measured here at
11.0 to 14.9 s per layer (N = 33,329, 56 heads, head_dim 128, bf16, swept over
all query blocks with a warm-up pass, `batch2.m`), times 50 blocks.
**This number is optimistic in situ. See 1.4.**

**(b) The per-row linear ops: 200 to 265 s per evaluation.** Extrapolated. Back
the attention term out of three measured end-to-end points:

| Point | N | s/eval measured | attention (N^2 from 0.128 s/layer at N=5034) | remainder | per token |
|---|---:|---:|---:|---:|---:|
| 448x576 / 56f (AGENTS.md) | 5,034 | 46.20 | 6.45 | 39.75 | 7.90e-3 |
| 640x352 / 124f (`control-640x352.log`) | 9,169 | 82.03 | 21.2 | 60.8 | 6.63e-3 |
| 704x384 / 124f (`base-P3.log`) | 10,779 | 107.19 | 29.3 | 77.9 | 7.23e-3 |

Linear in N to within 16% across a 2.1x span. At N = 33,329 that is 220 to
265 s. **None of these three runs carried a LoRA**, so this term excludes the
LoRA branch entirely. That gap is closed next.

**(c) The turbo LoRA branch: 4 to 15 s per evaluation.** Measured here, from the
one clean A/B in the logs. `logs/e2e-a1-lora.log` and `logs/e2e-a2-base.log` are
the same canvas, same 8 evaluations, same 408 submissions / 1600 linear / 400
attention dispatches, differing only in two loaded LoRAs (416 pairs):

```
with LoRA : Euler denoise wall = 582.283 s, direct = 11256
no LoRA   : Euler denoise wall = 535.989 s, direct =  1256
delta     = 46.294 s over 8 evaluations = 5.79 s/evaluation for 416 pairs
            (+10,000 dispatches = 1,250/eval = 3 per pair per evaluation)
```

Turbo alone is 259 pairs (`bench-H25lora.log`), so ~3.6 s/evaluation at that
canvas (N ~ 8,000). The branch is two thin matmuls per pair, so its **compute**
is linear in N, giving 15.0 s at N = 33,329; its **dispatch overhead** (~778
dispatches per evaluation for 259 pairs) is constant in N, giving ~3.6 s. The
true figure sits between. **4 to 15 s, i.e. under 2% of an evaluation.** Not
free, and now not assumed free, but not a threat to the budget either.

Two further LoRA costs, both accounted: resident weights +743.7 MiB (turbo
alone), and DiT load +0.45 s with the AdaLN table recomputed inside it. The
"a LoRA active set disables the int8 paths" concern is moot here, because
`--ssd-streaming` already disables all three (`h3_dit.c:1706,1709,1712`).

```
per evaluation = 550..750  attention   (measured here, in isolation; see 1.4)
               + 220..265  linear ops  (extrapolated 3.1x past the last clean point)
               +   4..15   LoRA branch (measured here, scaled two ways)
               = 774..1030 s before the in-situ penalty
```

Attention goes from 14% of an evaluation at N = 5,034 to roughly **72%** at the
target. That is the reframing, and it holds.

### 1.2 Fixed costs, including the one the study missed entirely

Nobody in twelve agents looked at the **video VAE decoder**. It is 38% of total
wall clock at P1 and 40% at the control run. Measured:

| Canvas | Frames | Tiles | Submissions | Decode wall |
|---|---:|---|---:|---:|
| 640x352 | 124 | 3x2 @ 256 px | 42 | 100.3 s (`control-640x352.log`) |
| 640x352 | 124 | 3x2 @ 256 px | 42 | 105.7 s (`base-P1.log`) |
| 704x384 | 124 | 3x2 @ 288 px | 42 | **248.8 s** (`base-P3.log`) |
| 768x1344 | short | 5x3 @ 320 px | 15 | 56.7 to 71.3 s (`bench-*.log`) |

Dispatch count depends only on frames and tile grid, so per-submission cost is
the scaling quantity. At 1280x704 the grid is 5x3 (as at 768x1344) and 124 frames
gives 7 temporal chunks: 105 submissions at the 768x1344 rate of 3.8 to
4.8 s/submission is **400 to 500 s**. Scaling the control's 100.3 s linearly in
pixels gives 401 s independently. Call it **450 s, 7.5 minutes.**

`base-P3` is the warning: 5.93 s/submission at 288 px tiles against 2.2 to 2.5
elsewhere at comparable tile sizes, a 2.4x unexplained spread. On that side of
it the decode is ~1000 s and eats 17 minutes.

```
fixed = 9 s    Qwen text encoder (measured, canvas-independent)
      + 8.5 s  DiT load with turbo LoRA (measured)
      + 0.5 s  audio VAE decode (measured)
      + 400..500 s video VAE decode (projected; 2.4x downside risk)
      + ~5 s   ffmpeg mux
      = 425..525 s  (7 to 9 minutes; worst case ~17 minutes)
```

### 1.3 The table the ladder result slots into

Budget 90 min = 5400 s. Per-evaluation band 774 to 1030 s from 1.1, widened to
**774 to 1290 s** by the in-situ penalty argued in 1.4. Fixed cost 425 to 525 s.

| Evaluations | Schedule | Denoise | Total with fixed | Verdict |
|---:|---|---|---|---|
| 3 | `--steps 3 --reuse 1` | 2322..3870 s | **46..73 min** | fits in every case |
| **4** | `--steps 4 --reuse 1` | 3096..5160 s | **59..95 min** | **fits at the median; marginal at the pessimistic end** |
| 5 | `--steps 5 --reuse 1` or `--steps 8 --reuse 2` | 3870..6450 s | **72..116 min** | fits only near the optimistic end |
| 6 | `--steps 6 --reuse 1` | 4644..7740 s | **85..138 min** | does not fit except in the best case |
| 8 | `--steps 8 --reuse 1` | 6192..10320 s | **110..181 min** | no |
| 11 | `--steps 20 --reuse 2` | 8514..14190 s | **149..245 min** | no, by a factor of 2 to 3 |

**Recommended target schedule, conditional on the ladder:**

```sh
./h3 --profile -d ./MiniMax-H3 --lora loras/turbo.safetensors \
  -p "$(cat prompts/01-breath.txt)" \
  --width 1280 --height 704 --frames 124 --steps 4 --reuse 1 \
  --layers 50 --ssd-streaming -o outputs/720p.mp4 > logs/720p.log 2>&1
```

4 evaluations is also exactly what the turbo LoRA was distilled for
(`minimax_h3_turbo_4step`), which is a convenient alignment rather than a
coincidence. Fallback order if it misses, following AGENTS.md's own priority
list: drop to `--steps 3`, then `--layers 35` (technique 5), never
`--token-reduction`.

### 1.4 Why the attention number is optimistic, and by how much

`batch2.m` measured attention **in isolation**: 1.91 GB of buffers on an
otherwise idle GPU, all 130 query blocks encoded into **one** command buffer,
committed once, for about 15 seconds. A real evaluation is none of those things.
Four differences, ranked by how much they should worry the owner:

1. **Thermals.** A 4-evaluation target run is 60 to 90 minutes of sustained GPU
   load on a laptop-class M4 Pro. The harness ran for 15 s and AGENTS.md's
   "no thermal warning" observations come from runs of 4 to 13 minutes.
   Sustained-clock throttling on Apple silicon over an hour is routinely 5 to
   20%. This is the largest term and it is entirely unmeasured here.
2. **Command-buffer granularity.** Under `--ssd-streaming` h3 submits **per DiT
   block** (`h3_dit.c:2405`), so the attention encode lands inside a command
   buffer that also carries QKV, RoPE, the LoRA branch, SwiGLU and fc1/fc2, with
   50 commit-and-wait boundaries per evaluation instead of the harness's one.
3. **Memory pressure.** The harness ran at 1.91 GB. A target evaluation runs at
   16 to 24 GiB with wired GPU pages, which changes how MPSGraph's temporary
   heap behaves and puts the run inside the range where `base-P4` saw the SSD
   stream collapse from 4.1 to **1.456 GiB/s**.
4. **Streaming contention.** ~36 GiB read per evaluation, ~8.5 s at 4.28 GiB/s
   against a 774 s evaluation. Comfortably hidden. Low risk, listed for
   completeness.

Zero-filled Q/K/V and warm-up handling are both fine: `batch2.m` runs two passes
and reports the second, and matmul plus softmax are data-independent here.

**Judgement: the 550 to 750 s attention band is optimistic by roughly 10 to
30%, so plan against 550 to 975 s.** That is what widens the per-evaluation band
to 774 to 1290 s in 1.3, and it is the difference between "5 evaluations fit"
and "4 evaluations fit". The owner should hear that now rather than discover it
on the first real run, and technique 2 below turns it into a measurement in one
command.

### 1.5 Memory is no longer the wall

`docs/720p-baseline.md` (commit `249b843`) projects the target two ways from
real `footprint -p` samples: **~233 GiB as coded today** (never reachable, the
encode fails first) and **~24 GiB with the attention score tensor bounded to a
constant**, against the 28 to 29 GiB ceiling this machine has shown.

An independent decomposition agrees. Using the fitted score-tensor coefficient
(1.79e-7 GiB/token^2) to split the three measured footprints:

| Point | N | footprint | score term | linear remainder | per token |
|---|---:|---:|---:|---:|---:|
| P1 640x352 | 9,169 | 19 GiB | 15.05 | 3.95 | 4.31e-4 |
| P3 704x384 | 10,779 | 26 GiB | 20.80 | 5.20 | 4.82e-4 |
| P4 768x416 | 12,536 | 34.7 GiB | 28.13 | 6.57 | 5.24e-4 |

At N = 33,329 the linear remainder is 14.3 to 17.4 GiB, plus the blocked score
tile (0.89 GiB at Bq=256, ~1.5 GiB with MPSGraph's second copy) plus 0.73 GiB of
resident turbo LoRA: **17 to 20 GiB**, or up to **25 GiB** if the per-token
coefficient keeps drifting the way it does across P1 to P4. That brackets the
baseline document's 24 GiB from below.

**Memory fits with 3 to 11 GiB of headroom.** Every memory technique in the
study is therefore solving a problem that query blocking already solves, and all
of them are rejected or deferred in section 3. One operational caveat: the
guardrail computes its ceiling from system free at that instant
(`h3_lora.c:961 h3_memory_ceiling_take`), so the target run needs a clean
machine, not one with VS Code and a browser open.

---

## 2. The top five, ranked by movement toward "done in 90 minutes" per unit of cost

The quality gate is not in this list because it is already being measured on
`measure/schedule-ladder`. The threshold it has to clear is set in 1.3:
**acceptable at 4 evaluations or fewer.**

### 1. Query-block tiling of the DiT self-attention

- **verdict:** accept-with-caveats
- **why:** it is the only change that makes the target encode at all, it is the
  smallest diff of anything that does, and its bit-identity was measured on this
  machine rather than argued from a paper.
- **exactness:** exact along the query axis, **measured**: 100.0000% of elements
  bit-identical to the single-encode reference, max abs diff 0.0, rel-L2 0.0, at
  (H=56, N=4096, Bq=256), (H=56, N=4096, Bq=1024), (H=56, N=8192, Bq=512) and
  (H=8, N=2048, Bq=256). I read the harness: `exact.m` genuinely compares a
  whole-sequence encode against per-block encodes at zero-copy byte offsets, on
  identical seeded random inputs.
  **The caveat is real and the study did not state it.** Both sides of that
  comparison feed Q **row-major** through a `transposeTensor`. The shipped code
  feeds Q head-major (`h3_gpu.m:1522`, the `headMajor` flag). So what was proven
  is *blocked row-major == unblocked row-major*. That the layout flip is itself
  bit-neutral against today's output is **unmeasured**, and MPSGraph may select a
  different kernel for it.
- **time_effect_at_target:** 550 to 750 s per evaluation for the attention term
  in isolation, 550 to 975 s in situ (1.4). It makes nothing faster than today.
  It turns a computation that aborts into one that runs.
- **memory_effect_at_target:** score temporary from 124.4 GB (unallocatable;
  `maxBufferLength` is 27.0 GiB) to 0.890 GiB at Bq=256, 0.445 at Bq=128, 1.335
  at Bq=384. Arithmetic independently recomputed, exact to the digit. Q/K/V/output
  unchanged at 1.78 GiB.
- **diff_size:** ~120 lines, `h3_gpu.m` only
- **evidence_grade:** measured-on-this-machine
- **depends_on:** nothing; everything else touching attention depends on it
- **hypothesis:** replacing the single SDPA encode with a loop of per-query-block
  encodes at byte offsets produces byte-identical video on a canvas that works
  today, and makes 1280x704 / 124f encode.
- **measurement:**

```sh
./h3 --profile -d ./MiniMax-H3 -p "$(cat prompts/01-breath.txt)" \
  --width 640 --height 352 --frames 124 --steps 2 --reuse 1 --layers 50 \
  --seed 42 --ssd-streaming -o outputs/blk-after.mp4 > logs/blk-after.log 2>&1
cmp outputs/probe-P1.mp4 outputs/blk-after.mp4 && echo BYTE-IDENTICAL
```

  **Refuted if** `cmp` reports a difference. That means the Q-layout flip, not
  the blocking, perturbed the result, and the change has to be re-argued as
  approximate with a measured quality number instead of landing free.
- **files and functions:**
  - `h3_gpu.m:1522 h3_gpu_sdpa` - replace the single `encodeToCommandBuffer`
    with a loop over query blocks computing a byte offset per block.
  - `h3_gpu.m:1448 h3_gpu_sdpa_graph` - add `Bq` to the cache key (currently
    `dataType:batch:sequence:heads:head_dim:scale:causal:headMajor:outputHeadMajor`)
    and split the single `headMajor` flag into separate query-layout and
    key/value-layout flags, so Q can be row-major while K/V stay head-major.
  - `h3_gpu.m:137 h3_gpu_graph_data` - add an offset parameter and build
    `MPSGraphTensorData` via `MPSNDArray initWithBuffer:offset:descriptor:`,
    which `initWithMTLBuffer:shape:dataType:` has no equivalent of. The pattern
    is production-proven in PyTorch's `aten/src/ATen/native/mps/OperationUtils.mm`.
    The stable-data cache at `h3_gpu.m:139` keys on the tensor object, not the
    offset, so it must be bypassed for offset views.
  - Call sites need no change: `h3_dit.c:2053`, `h3_dit.c:2058`, `h3_dit.c:849`.
  - Two known blockers, both real: the int8 fast path at `h3_dit.c:2053` uses
    `h3_gpu_sdpa_bf16_head_major_output`, whose output block is strided across
    heads and needs either a small scatter kernel or a fallback to row-major
    output; and N = 33,329 is not divisible by Bq, so the 49-row tail needs a
    second shape-keyed graph entry.

### 2. The two-evaluation target probe, run the moment technique 1 lands

- **verdict:** accept
- **why:** one command replaces **every** extrapolation in this document at
  once: attention in situ rather than in isolation, the linear term at
  N = 33,329, the LoRA branch at the target, the video VAE decode at the target,
  and the real footprint. Nothing else on this list has that ratio.
- **exactness:** measurement only
- **time_effect_at_target:** none; converts 774..1290 s from a band into a number
- **memory_effect_at_target:** none; reports the real footprint
- **diff_size:** 0 lines
- **evidence_grade:** would be measured-on-this-machine
- **depends_on:** technique 1
- **hypothesis:** one denoiser evaluation at 1280x704 / 124f with the turbo LoRA
  costs 774 to 1030 s, and the video VAE decode costs 400 to 500 s, so 4
  evaluations plus fixed costs land inside 5400 s.
- **measurement:**

```sh
( while :; do footprint -p $(pgrep -n h3) 2>/dev/null | tail -1; sleep 2; done ) \
  > logs/720p-probe.footprint &
./h3 --profile -d ./MiniMax-H3 --lora loras/turbo.safetensors \
  -p "$(cat prompts/01-breath.txt)" \
  --width 1280 --height 704 --frames 124 --steps 2 --reuse 1 --layers 50 \
  --ssd-streaming -o outputs/720p-probe.mp4 > logs/720p-probe.log 2>&1
kill %1
tr '\r' '\n' < logs/720p-probe.log | grep -E "h3 profile|^h3:"
```

  Read `Euler denoise wall` and halve it for s/evaluation; read
  `video VAE decoder wall` directly; read the `BF16 SSD stream` line to confirm
  2 evaluations actually ran (~72 GiB); read the footprint samples for the real
  peak.
  **Refuted if** s/evaluation exceeds 1290 s, in which case only 3 evaluations
  fit and the ladder has to clear at 3; or if `video VAE decoder wall` exceeds
  700 s, in which case the decode alone is 12+ minutes and the evaluation budget
  shrinks by one.
- **files:** none. Pure measurement.

### 3. Pre-measure the linear plus LoRA cost at N = 33,329, before technique 1

- **verdict:** accept
- **why:** technique 2 is the better measurement but it costs a 120-line change
  first. This one costs a scratch harness and can kill the project before that
  change is written. The 220 to 265 s linear term is stretched 3.1x past the
  largest clean measured point and it is the second-largest term in the budget.
- **exactness:** measurement only
- **time_effect_at_target:** none
- **memory_effect_at_target:** none
- **diff_size:** ~60 lines of scratch Objective-C, zero production lines
- **evidence_grade:** would be measured-on-this-machine
- **depends_on:** nothing; do it first
- **hypothesis:** the per-row linear ops cost 6.6e-3 to 7.9e-3 s per token per
  evaluation and stay linear at N = 33,329, giving 220 to 265 s, and the turbo
  LoRA branch adds 4 to 15 s.
- **measurement:** extend `batch2.m` (already in the session scratchpad) to time
  the four production matmuls at the target row count instead of SDPA. Shapes
  from `h3_dit.c`: QKV `N x 5376 @ 5376 x 21504`, attn-out `N x 7168 @ 7168 x 5376`,
  fc1 `N x 5376 @ 5376 x 28672`, fc2 `N x 14336 @ 14336 x 5376`, bf16, times 50
  blocks. Add 259 rank-64 pairs of `N x 5376 @ 5376 x 64` then `N x 64 @ 64 x out`
  for the LoRA branch. None of these is near any element limit, so this runs
  today with no change to h3.

```sh
cd "$TMPDIR" && clang -fobjc-arc -O2 linear.m -o linear \
  -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
&& ./linear 33329
```

  **Refuted if** the measured per-block linear time times 50 exceeds ~400 s. The
  per-evaluation total is then above 1150 s even before the in-situ penalty,
  only 3 evaluations fit, and the ladder has to clear at 3 or the project stops.
- **files:** scratch only.

### 4. Sweep Bq at the target and pin the optimum

- **verdict:** accept
- **why:** the measured attention spread is 11.0 to 14.9 s/layer, a 35% band.
  Over 4 evaluations that band is **up to 13 minutes of the 90**, which is the
  gap between the 59-minute and the 95-minute cell in 1.3, for no code beyond the
  knob technique 1 already has to introduce.
- **exactness:** exact; Bq only partitions the query axis
- **time_effect_at_target:** 550 to 750 s per evaluation depending on where in
  the band the run lands. Worth up to 200 s/evaluation.
- **memory_effect_at_target:** 0.445 GiB (Bq=128) to 1.335 GiB (Bq=384), all
  irrelevant against a 28 GiB ceiling. Pick purely on speed.
- **diff_size:** 0 additional lines once technique 1 lands
- **evidence_grade:** measured-on-this-machine, but confounded
- **depends_on:** technique 1
- **hypothesis:** the 11.0-to-14.9 s/layer spread is a function of Bq, and
  pinning Bq at its optimum lands the target reliably near 11.0.
- **the confound, which is the point:** the study reported Bq=384 best then 256,
  with 64 and 128 slower, **and separately** reported "bimodal 11.3/14.7 across
  five fresh processes". If the bimodality is process-to-process rather than
  Bq-to-Bq, then a third of the attention budget varies for reasons nobody has
  identified, and every per-evaluation estimate in this document carries an
  unexplained plus-or-minus 15% on its largest term.
- **measurement:** the harness already exists and already prints this.

```sh
cd /private/tmp/claude-502/-Users-fabiobiola-Developer-AI-Tools-h3-c/*/scratchpad \
&& for bq in 128 256 384 512 768 1024; do for rep in 1 2 3; do ./batch2 $bq 33329; done; done
```

  **Refuted if** the within-Bq spread across the three repetitions is as large as
  the between-Bq spread. Then stop tuning Bq, pick 384, and go find the real
  source of the variance (thermals first, per 1.4) before trusting any
  per-evaluation estimate.
- **files:** the block-size constant introduced in `h3_gpu.m:1522`.

### 5. `--layers 35` as the budget fallback

- **verdict:** accept-with-caveats
- **why:** 31% off per-evaluation time, measured on this machine, for zero lines
  of code. At 4 evaluations that is ~20 minutes of the 90, larger than anything
  else here except the evaluation count itself.
- **exactness:** approximate. It is the shipped gate-ranked DiT block skip
  (`h3: gate-ranked DiT skips 4(0.1183) 14(0.1213) ...`, `bench-A.log`).
  **Quality evidence at the target: none. Quality evidence anywhere: none
  measured**, only the README's ordering, which puts `--layers 50` first. It is a
  fallback, not a default, and it must not ship without its own A/B.
- **time_effect_at_target:** 4 evaluations at 534 to 890 s each instead of 774 to
  1290. From `logs/bench-grid.csv`: L35 10 evaluations in 518.500 s against L50
  10 evaluations in 753.585 s, ratio 0.688 against a layer ratio of 0.70, so the
  scaling is essentially exactly linear in layers.
- **memory_effect_at_target:** negligible; skipped blocks free no persistent
  buffers.
- **diff_size:** 0 lines
- **evidence_grade:** measured-on-this-machine for time,
  claimed-no-measurement for quality
- **depends_on:** nothing; stacks with everything
- **hypothesis:** at 4 evaluations, `--layers 35` costs less perceptible quality
  than dropping from 4 evaluations to 3 does.
- **measurement:** three runs at the small canvas, turbo LoRA, same seed, one
  contact sheet each. This is deliberately a different question from the
  `measure/schedule-ladder` work, which varies evaluations at fixed layers.

```sh
for cfg in "4 50" "3 50" "4 35"; do set -- $cfg; \
./h3 --profile -d ./MiniMax-H3 --lora loras/turbo.safetensors \
  -p "$(cat prompts/01-breath.txt)" \
  --width 640 --height 352 --frames 124 --steps $1 --reuse 1 --layers $2 \
  --seed 42 --ssd-streaming -o outputs/ql-s$1-l$2.mp4 \
  > logs/ql-s$1-l$2.log 2>&1; done
for f in outputs/ql-s*.mp4; do ffmpeg -v error -i "$f" \
  -vf "select='eq(n\,0)+eq(n\,41)+eq(n\,82)+eq(n\,123)',tile=4x1" \
  -frames:v 1 -y "${f%.mp4}-contact.png"; done
```

  **Refuted if** the `--layers 35` contact sheet shows drift the `--steps 3` one
  does not. Then the fallback order in 1.3 reverses: cut evaluations before
  layers.
- **files:** none. `main.c:296` already exposes it.

---

## 3. Rejected

One line each, naming the fact that kills it, so the next round does not
re-propose the same things.

### Solving a problem that is already solved (memory)

`docs/720p-baseline.md` projects ~24 GiB with the score tensor bounded, and the
independent decomposition in 1.5 gives 17 to 25 GiB, against a 28 to 29 GiB
ceiling. Everything here is correct engineering aimed at a constraint that query
blocking removes on its own.

- **Row-tiled MLP / chunked feed-forward** (exact, 180 lines, -4.2 GB): rejected
  as premature. For the record before someone re-proposes it: the cited
  `diffusers` PR #8815 was **closed unmerged** with the maintainer writing "these
  techniques are not particularly effective here", and the only public memory
  measurement in the cited issue thread shows chunking *increasing* memory
  (11.6 GB to 12 GB). Revisit only if technique 2 reports footprint above 25 GiB.
- **`MPSCommandBuffer.heapProvider` heap ownership** (70 lines): rejected as
  premature. It pools, it does not cap; returning nil from
  `newHeapWithDescriptor:` makes MPS fall back silently, so it can never force
  anything to fit.
- **Stop reading `recommendedMaxWorkingSetSize` as a process cap**
  (`h3_lora.c:963`, 25 lines): rejected as premature. Apple's Tech Talk 10580
  genuinely supports it, more strongly than the report claimed. It buys +4 to
  +8 GiB of a ceiling we are 3 to 11 GiB under. One of its three citations is an
  iOS/tvOS jetsam document that never mentions the property.
- **`sysctl iogpu.wired_limit_mb`** (0 lines): rejected as premature and half
  wrong. The `maxBufferLength` half is dead: the cited llama.cpp issue #1815
  never mentions `iogpu` or `wired_limit` anywhere in its body or 20 comments,
  and its actual contents show `maxBufferLength` tracking **physical RAM at
  56.25%** (48 GiB x 0.5625 = 27.0 GiB exactly, this machine's figure), not the
  wired limit. Expect 27.0 GiB to stay 27.0 GiB. Against a documented kernel
  panic precedent (mlx-lm #883, IOGPUMemory.cpp:550).
- **Activation recompute / checkpointing**: rejected. Inference only, no backward
  pass, nothing kept alive to recompute.
- **Spilling activations to SSD**: rejected. Buys ~1.4 GB for ~140 GB of extra
  traffic per evaluation.
- **Splitting the graph into smaller encodes**: rejected. Already shipped;
  `h3_gpu_submit` runs per block under `--ssd-streaming` (`h3_dit.c:2405`).

### Whose real cost is time

- **FlashAttention key-axis blocking with online softmax on top of query
  blocking** (600 lines): rejected. It shrinks a score tile query blocking
  already made 0.89 GiB, by reordering the key-axis reduction in bf16, which
  loses the bit-identity that makes technique 1 cheap to land. The research agent
  argued against its own technique here, correctly.
- **MultiDiffusion / DemoFusion latent tiling of the DiT** (1200 lines):
  rejected. Costed at **17% slower** than untiled, because every op in an H3
  block except SDPA is already per-row. Its two headline numbers are also
  misattributed: the 125 s / 47 s / 1005 s figures are ScaleDiff on an A6000, not
  DemoFusion; the FID pairs come from ASGDiffusion (uncited) and mix resolutions
  (DemoFusion at 3072^2 is 64.85, not the 68.06 quoted).
- **FIFO-Diffusion diagonal denoising** (400 lines): rejected. ~37 forward passes
  against 4, i.e. 1,332 GiB of SSD reads against 144, collapsing the streaming
  prefetch margin from 11x to ~1.6x and ending `--ssd-streaming` being free.
- **Windowed temporal co-denoising (Gen-L-Video / MultiDiffusion)** (600 lines):
  rejected. Its own FLOP costing gives a **6%** saving at the window that clears
  the encode limit, from a technique that removes the only mechanism coupling
  distant frames in a single-stream 3D-attention DiT.
- **Step-level caching (TeaCache, PAB, FasterCache, DeepCache)**: rejected. It
  duplicates the shipped `--reuse` and `--core-reuse`, PAB adds 1.55 GiB per
  cached attention per cached block, and at 3 to 5 evaluations there is nothing
  to cache. The cited VBench 68.81 is from MotionCache on MAGI-1, a paper absent
  from the source list.

### Approximating the model output without a measured number that clears the bar

- **Token merging (ToMe, ToMeSD, VidToMe, ToMA)**: rejected on sight. This **is**
  `--token-reduction` (`h3_dit.c:405 configure_token_reduction`,
  `h3_shaders.metal:4060`), which the owner measured as unusable and banned. It
  also fails on its own terms: a 50% merge leaves N ~ 16,700, still above the
  measured encode boundary of ~12,390. Three of its four cited sources are
  off-target (ToMA is image-only on SDXL/Flux; the diffusers page benchmarks
  single-image Stable Diffusion on A100/V100; VidToMe is video *editing*).
- **Radial Attention** (900 lines): rejected. The report claimed "there is no
  published number for 0 warmup steps"; the verifier found Appendix D.1 Table B,
  which publishes exactly that: **PSNR 12.8, SSIM 0.486, LPIPS 0.522 at 0 warmup
  steps**, against 23.6 / 0.823 / 0.146 at the 12-step default. At 4 evaluations
  a 12-step dense warmup is three times the whole run. The report also welded a
  fine-tuned 253-frame sparsity (80.8%) onto a training-free 117-frame speedup
  (1.88x), then projected 2.5 to 3x on an M4 Pro from a 1.88x measured on an H100.
- **Sliding Tile Attention, training-free** (1600 lines): rejected. Its
  training-free mask comes from a per-(layer, head) search that needs the **true
  full-attention output at the target resolution**, which is exactly what this
  machine cannot compute. `latent_t` (30 or 31 at 124 frames) is also not
  divisible by the (4,4,4) tile. Its quality numbers are sound and
  hardware-independent; its speedups are H100 plus ThunderKittens at 58.79% MFU
  and transfer nothing.
- **Local 3D-window block-sparse attention with global text/audio rows (STA/VSA
  pattern)** (700 lines): rejected for this round, revisitable. It is the only
  technique that cuts FLOPs rather than relocating memory, and the
  in-distribution argument was the best in the study, but the verifier found the
  model-card sentence continues past where it was quoted: "The sparse-attention
  implementation is not included in the initial open-source release and will be
  published separately in a future update." The trained window shape is unknown,
  so this is guess-and-measure, and the hybrid global-text/audio plus local-video
  pattern it needs is implemented by neither STA nor VSA and has no published
  numbers anywhere. Revisit only if technique 2 shows attention above ~70% of
  per-evaluation time **after** query blocking, and only with a measured
  PSNR/SSIM A/B at 640x352 against the exact path.
- **Sparse VideoGen head classification** (1200 lines): rejected for this round.
  The only family member whose calibration is computable here (64 dense query
  rows, a 0.77 GiB probe), but every speed number is H100, its temporal-head
  pattern is a strided gather with no Apple-silicon measurement, and it still
  ships with a 12-step dense warmup.
- **int8 attention (SageAttention, ViDiT-Q W8A8)** (1400 lines): rejected. The
  entire speed case rests on NVIDIA INT8 tensor cores the M4 Pro does not have
  (SageAttention2 is INT4 QK plus FP8 PV on Ada/Hopper, not the INT8 QK the
  report described). Weight quantisation is worth nothing under
  `--ssd-streaming`, where resident weights are 2.58 GiB. And the abort is an
  **element-count** limit, not a byte-size limit, so halving element width does
  not clear it.
- **Autoregressive chunk chaining through `--first-frame`** (0 lines): rejected.
  The audio handoff runs through `H3_SEG_REF_AUDIO`, filled only inside the
  ref2va branch (`h3.c:1195-1250`), and Ref2VA is not installed
  (`./h3 --info` shows `Ref2VA DiT 0 files`). Every chunk's audio starts from
  silence: 4 to 6 independent audio takes with a fade-in at each join, inside a
  5-second clip.
- **Hold-and-splice overlap pinning** (150 lines): rejected, and worth reading
  twice. The report said "no measured quality evidence found for the pin in
  isolation". The verifier found it: vllm-omni PR #6779 measured the
  latent-domain pin in isolation at **~35x the clip's median inter-frame
  difference**, a hard cut to an unrelated shot, tried seven conditioning
  variants ("All of them cut at the same magnitude"), and **deleted the code**.
  The 2.2x figure the report attached to it belongs to a different commit with a
  different mechanism.
- **Resolution-extrapolation family (ScaleCrafter, FouriScale, FreeScale, RoPE
  rescaling)** (0 lines): rejected as not applicable. 1280x704 has a short side
  of 704, below H3's native 768; the DiT block contains no convolution for
  ScaleCrafter to dilate; and `h3_frame_grid` (`h3_host.c:231`) normalises RoPE
  positions by `sqrt(latent_h * latent_w)`, so raising resolution *interpolates*
  positions into the trained range instead of extrapolating out of it. There is
  nothing to correct.

### Dead on this hardware

- **Fused M5/NAX kernel paths**: rejected by the brief and by
  `h3_gpu.m:364-373`, which requires the Metal device name to contain the literal
  string "M5". This is an M4 Pro.
- **Hand-written Metal flash-attention kernel (metal-flash-attention / MLX
  steel)** (650 to 900 lines): rejected for now, and it is the one that will keep
  coming back, so here is why in full. It is the only technique that attacks the
  550 to 975 s attention term rather than merely enabling it. But its single
  speed citation, Draw Things' "up to 20% faster", is an **end-to-end FLUX.1
  image-diffusion** figure whose baseline the article never states and which most
  plausibly means Metal FlashAttention 1.0, not MPSGraph SDPA; the article's
  MPSGraph comparison is appendix charts with no absolute figures. Applying an
  end-to-end image-model percentage to a single video attention layer's wall
  clock is not a valid transfer, and the derived "8.8 to 11.9 s/layer" should be
  struck. philipturner's own README, in the same table as the 4400 GINSTRS/s M1
  Max figure that was quoted, has an **M4 row at roughly a third of it**, which
  was not mentioned. The FastMetal existence proof cited alongside it is about
  INT8 quantization-aware distillation of a 5B model and explicitly runs *dense
  fp16 attention*; flash attention appears nowhere in it. Revisit only after
  techniques 1 to 4 have produced a real end-to-end target run and attention is
  still what stands between that run and 90 minutes.

---

## 4. Risks this plan carries

1. **The attention number was measured in isolation, not in situ.** Argued in
   1.4. Optimistic by an estimated 10 to 30%, dominated by thermal throttling
   over a 60-to-90-minute run and by per-block command-buffer granularity.
   Technique 2 is the fix, and until it runs this is the difference between "5
   evaluations fit" and "4 evaluations fit".
2. **The linear term is a 3.1x extrapolation.** `base-P4` (N = 12,536) took
   **2.84x** the time of `base-P3` (N = 10,779) for a 1.16x increase in N.
   `docs/720p-baseline.md` attributes that to the quadratic score-tensor term,
   and the guardrail firing at 34.7 GiB makes memory thrashing the likely cause.
   But if any part of it is a genuine super-linear break in the **non-attention**
   path, the per-evaluation cost is far above 1290 s and nothing fits at any
   evaluation count. P4 is the only measured point above N = 10,779 and it is
   contaminated. Technique 3 attacks this before any code is written.
3. **Bit-identity was measured under a Q layout the shipped code does not use.**
   See technique 1. The `cmp` check is the whole mitigation and it is one run.
4. **The video VAE decode has a 2.4x unexplained spread** at comparable tile
   sizes (5.93 s/submission in `base-P3` against 2.2 to 2.5 elsewhere). At the
   bad end it costs 17 minutes of the 90 instead of 7.5, which is one whole
   evaluation.
5. **The turbo LoRA branch cost at the target is bracketed, not measured.** 4 s
   if dispatch-overhead-dominated, 15 s if compute-dominated. Under 2% of an
   evaluation either way, so this is the smallest risk here, but it is the one
   that was silently absent from every per-evaluation figure until now.
6. **The 4-evaluation quality bar is unvalidated at the target canvas.** The
   `measure/schedule-ladder` work answers it at 640x352, which is a necessary
   condition only: more tokens may need more steps to resolve.
