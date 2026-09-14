# Is h3 querying the turbo LoRA on the sigmas it was distilled for?

The runs that fit the 720p budget depend on `loras/turbo.safetensors`
(`minimax_h3_turbo_4step.safetensors`, from `larryvrh/MiniMax-H3-Turbo-Lora`).
It is a step distillation, so it is only correct on the sigma trajectory it was
trained against. Off that trajectory it does not error, it just gets quietly
worse.

The vendor guidance page for the turbo LoRAs specifies scheduler `simple`,
LoRA strength 1.0, and CFG 1.0. `simple` is ComfyUI vocabulary and this repo
is not ComfyUI: `README:512` says h3 uses "the released shifted video/audio
schedule", built in `h3_serving_schedule_build` (`h3_host.c:150`) at shift 12.

**Verdict: they are the same sigma sequence.** Not approximately, not close
enough. At 4, 8 and 10 steps the two agree to the last float32 bit; at 6 steps
they differ by at most 5e-4, which is a 1/1000 grid-quantisation artifact in
ComfyUI's implementation, not a difference of intent. Stronger than that:
the LoRA author's own published reference sampler computes the sigma grid with
**h3's exact continuous formula**, not ComfyUI's quantised one (section 5). On
this axis h3 is the faithful implementation and `simple` is the approximation.
Nothing needs to change.

The research did, however, turn up a different and more serious question that
the original one was standing in front of. This LoRA family has two lines: the
544p checkpoints are distilled at video shift **12**, and the 768p checkpoints
at video shift **6**. This project renders at 1280x704, which is 768p
territory. h3 hardcoded 12 with no way to express 6; it no longer does, and
section 9b is the twenty-minute A/B that settles which one this file wants.
That is the part worth acting on.

---

## 1. What h3 produces

`h3_serving_schedule_build` (`h3_host.c:150`) is what the CLI actually runs
(`h3.c:1492`). For `N` steps it emits `N+1` sigmas:

```
base_i  = 1 - i/N                              for i = 0..N
sigma_i = shift * base_i / (1 + (shift - 1) * base_i)
sigma_N = 0                                    (forced)
```

with the shifts defaulting to `H3_VIDEO_SIGMA_SHIFT = 12.0` and
`H3_AUDIO_SIGMA_SHIFT = 3.0` (`h3_host.h:12-13`), read through
`h3_video_sigma_shift()` / `h3_audio_sigma_shift()` and overridable by
`H3_VIDEO_SHIFT` / `H3_AUDIO_SHIFT` (section 7). Video and audio get
independent arrays from the same formula at their own shift. The denoiser walks
`sigma[step] -> sigma[step+1]` with plain Euler (`h3_dit.c`,
`h3_dit_denoise_euler_preview`).

The previously reported 4-step sequence **1.000 / 0.973 / 0.923 / 0.800 is
correct.** Recomputed in float32 from the shipped code, it is
`1.00000000, 0.97297299, 0.92307693, 0.80000001, 0.0`.

Note there is a second builder in the same file, `h3_schedule_build`
(`h3_host.c:135`), which quantises onto a 1000-entry grid via
`h3_shifted_sigma`. It is used only by tests and benches. It is, as it
happens, a literal reimplementation of ComfyUI's `simple` (see below).

## 2. What ComfyUI's `simple` produces

`simple_scheduler`, `comfy/samplers.py:645-652`, unchanged since v0.0.1:

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

It does not compute sigmas, it indexes the model's own table.
For a flow model that table is `ModelSamplingDiscreteFlow`
(`comfy/model_sampling.py:289-329`): 1000 ascending entries,
`sigmas[i] = shift*t/(1 + (shift-1)*t)` with `t = (i+1)/1000`. So

```
k_x     = int(x * 1000/N)          # truncated
t_x     = 1 - k_x/1000
sigma_x = shift * t_x / (1 + (shift - 1) * t_x)
```

plus a trailing `0.0`. Same warp, same `N+1` length, same forced zero. The
only structural difference from h3 is that h3 evaluates `1 - i/N` exactly
while ComfyUI snaps it to the nearest 1/1000 below.

## 3. The shift is not a choice either side made

This is the part that closes the question. Shift 12 is not h3's guess and not
ComfyUI's guess, it is shipped in the checkpoint.

`MiniMaxAI/MiniMax-H3`, `scheduler/scheduler_config.json`:

```json
{"_class_name": "MiniMaxH3Scheduler", "_diffusers_version": "0.36.0.dev0", "shift": 12.0}
```

and `audio_scheduler/scheduler_config.json` gives `"shift": 3.0`. Locally, in
`MiniMax-H3/FL2VA/model_index.json`:

```json
"scheduler": null,
"_minimax_h3": {
  "sigma_shift_scales": { "video": 12.0, "audio": 3.0 }
}
```

ComfyUI, `comfy/supported_models.py:962-970`:

```python
class MiniMaxH3(supported_models_base.BASE):
    unet_config = {"image_model": "minimax_h3"}
    sampling_settings = {
        "shift": 12.0,
        "audio_shift": 3.0,
    }
```

h3, `h3_host.h:12-13`: `H3_VIDEO_SIGMA_SHIFT 12.0`, `H3_AUDIO_SIGMA_SHIFT 3.0`.

Three independent places, one pair of numbers, and the checkpoint is the
source. The pipeline ships `"scheduler": null`, meaning those two shift scales
*are* the entire published schedule specification. There is no room for h3 and
ComfyUI to disagree about it, and neither of them is approximating the other.

## 4. Sigma tables side by side, shift 12 (video)

### 6 steps

| i | h3 `h3_serving_schedule_build` | ComfyUI `simple` | diff |
|--:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.983607 | 0.983684 | 7.7e-05 |
| 2 | 0.960000 | 0.960058 | 5.8e-05 |
| 3 | 0.923077 | 0.923077 | 0 |
| 4 | 0.857143 | 0.857510 | 3.7e-04 |
| 5 | 0.705882 | 0.706380 | 5.0e-04 |
| 6 | 0.000000 | 0.000000 | 0 |

### 8 steps

| i | h3 | ComfyUI `simple` | diff |
|--:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.988235 | 0.988235 | 0 |
| 2 | 0.972973 | 0.972973 | 0 |
| 3 | 0.952381 | 0.952381 | 0 |
| 4 | 0.923077 | 0.923077 | 0 |
| 5 | 0.878049 | 0.878049 | 0 |
| 6 | 0.800000 | 0.800000 | 0 |
| 7 | 0.631579 | 0.631579 | 0 |
| 8 | 0.000000 | 0.000000 | 0 |

### 10 steps

| i | h3 | ComfyUI `simple` | diff |
|--:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.990826 | 0.990826 | 0 |
| 2 | 0.979592 | 0.979592 | 0 |
| 3 | 0.965517 | 0.965517 | 0 |
| 4 | 0.947368 | 0.947368 | 0 |
| 5 | 0.923077 | 0.923077 | 0 |
| 6 | 0.888889 | 0.888889 | 0 |
| 7 | 0.837209 | 0.837209 | 0 |
| 8 | 0.750000 | 0.750000 | 0 |
| 9 | 0.571429 | 0.571429 | 0 |
| 10 | 0.000000 | 0.000000 | 0 |

### 4 steps, for the reference check

| i | h3 | ComfyUI `simple` |
|--:|---:|---:|
| 0 | 1.000000 | 1.000000 |
| 1 | 0.972973 | 0.972973 |
| 2 | 0.923077 | 0.923077 |
| 3 | 0.800000 | 0.800000 |
| 4 | 0.000000 | 0.000000 |

**Why 6 is the odd one out.** ComfyUI truncates `int(x*1000/N)`. When `N`
divides 1000 the truncation is exact and the two schedules are bit-identical:
4 (step 250), 8 (125), 10 (100), also 20, 25, 40, 50. 6 does not divide 1000,
so indices land on 166/333/500/666/833 instead of 166.67/333.33/.../833.33.
The resulting error is bounded by one grid cell, 1e-3 in `t`, and h3 is on the
*exact* side of it. If either schedule is the approximation here, it is
ComfyUI's.

### h3 audio schedule, shift 3

Carried on its own grid, same formula:

| steps | sigmas |
|--:|---|
| 4 | 1.000000, 0.900000, 0.750000, 0.500000, 0 |
| 6 | 1.000000, 0.937500, 0.857143, 0.750000, 0.600000, 0.375000, 0 |
| 8 | 1.000000, 0.954545, 0.900000, 0.833333, 0.750000, 0.642857, 0.500000, 0.300000, 0 |
| 10 | 1.000000, 0.964286, 0.923077, 0.875000, 0.818182, 0.750000, 0.666667, 0.562500, 0.428571, 0.250000, 0 |

ComfyUI mechanises audio differently: `ModelSamplingAV` keeps audio on the
*video* schedule and rescales the stream by `audio_scale = shift/audio_shift
= 4.0`. h3 instead builds the shift-3 array directly and corrects the velocity
with `h3_time_shift_slope` (`h3_host.c:118`). Different arithmetic, same 12/3
pair from the checkpoint, and the video trajectory the LoRA sees is unaffected
either way.

## 5. What the LoRA's own metadata says

Read straight off the safetensors header of `loras/turbo.safetensors`
(518 tensors, 57480-byte header):

```json
{
  "dtype": "bfloat16",
  "sampler_steps": "4",
  "application": "W_eff = W + lora_B @ lora_A",
  "format": "pt",
  "base_model": "MiniMax-H3"
}
```

That is the whole of it. **It records the step count and says nothing about
the trajectory:** no shift, no scheduler name, no sigma list, no training
config. There is no `README.md` in the local HF snapshot either, only the
weights blob.

So the metadata confirms one thing and one thing only, but it is the thing
that matters operationally: this file is a **4-step** distillation, and the
project's `--steps 4` matches it. (The official ComfyUI template ships the
*8-step* lightx2v LoRA at `steps 8` instead; the 4-step and 8-step files are
paired with their own step counts, and mixing them is the mistake to avoid.)

Header shape, for identification: 744 MiB, 518 tensors, rank 64 on the
attention and MLP pairs and rank 16 on the 51 AdaLN pairs, no `.alpha` keys,
covering all 50 blocks plus `token_refiner` and `final_layer`, in convention A
naming.

### The author's reference sampler, which is the real answer

The metadata is silent, but the model card is not. `larryvrh/MiniMax-H3-Turbo-Lora`
ships a `generate.py` that hardcodes the schedule:

```python
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)

def timesteps(n, shift=SHIFT_VIDEO):
    """n-step video sigma grid: ts[0]=1 (pure noise) > ... > ts[n]=0."""
    return [shift_sigma(1.0 - i / n, shift) for i in range(n + 1)]
```

That is `h3_serving_schedule_build`, line for line: the same `1 - i/n`, the
same warp, the same shift 12, the same `n+1` points ending at zero. The card
also says "keep the scheduler on `simple`", "keep strength at `1.0`", and
"4 steps is the recommended minimum; 4-8 is the useful range". There is no CFG
anywhere in `generate.py`.

So h3 is not merely compatible with the vendor's `simple`, it matches the
LoRA author's own reference implementation exactly, including the 1/1000
quantisation that `simple` has and neither h3 nor `generate.py` does.

What is **not** published for this file is the distillation itself: no training
config, no yaml or json recording the sigma schedule or timestep sampling,
nothing in 25 commits of history but weight uploads and README edits. A direct
question about the training method (discussion #41, "single-stage or
multi-stage? DMD or other?") went unanswered. The trained trajectory is
therefore inferred from the author's own inference recipe, not read off a
training config. That inference is strong but it is an inference.

## 6. The vendor recipe, item by item

| Vendor guidance | h3 status |
|---|---|
| scheduler `simple` | Match, exactly (sections 1-4) |
| LoRA strength 1.0 | Match, `--lora PATH` defaults to 1.0 |
| CFG 1.0, guidance-free | Match by construction. There is no CFG, no negative branch and no guidance code anywhere in this repo. The Comfy template agrees: it wires `BasicGuider`, not `CFGGuider`. |
| steps 4-8 | Match, the project runs 4 with a 4-step file |
| sampler | Not specified by the vendor page. Sources disagree; see below. |

The official Comfy subgraph (`Comfy-Org/workflow_templates`,
`video_minimax_h3_t2v.json`) resolves to:

```
KSamplerSelect       res_multistep
BasicScheduler       simple, 4, 1.0
BasicGuider          (no CFG)
LoraLoaderModelOnly  minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16, 1.0
```

larryvrh's own shipped workflow is the same shape with
`BasicScheduler ['simple', 6, 1]`, and ModelTC's workflows use
`KSamplerSelect ['euler']` with `BasicScheduler ['simple', N, 1]`. So every
source agrees on `simple`, strength 1.0 and no CFG, and they split on the
solver: Comfy-Org picks `res_multistep`, ModelTC picks `euler`, which is h3's
default. That is a solver difference, not a schedule difference, and section 9c
explains why it is not worth chasing.

## 7. The question underneath: 768p checkpoints are distilled at shift 6

The original question assumed one turbo LoRA. There are two lines, and they do
not share a shift. From `ModelTC/Minimax-H3-Turbo`, which unlike larryvrh's
repo publishes a training-shift column:

| checkpoint | train res | training shift video/audio | distill NFE |
|---|---|---:|---:|
| FL2VA Turbo 4-step v0.1 | 544p | **12 / 3** | 4 |
| FL2VA Turbo 8-step v1.0 | 544p | **12 / 3** | 8 |
| FL2VA Turbo 4-step v1.0 768p | 1344x768 | **6 / 3** | 4 |
| FL2VA Turbo 8-step v1.0 768p | 1344x768 | **6 / 3** | 8 |

Their runtime configs record it machine-readably
(`LightX2V/configs/minimax_h3/dmd/*.json`):

```json
{"infer_steps": 4, "video_flow_shift": 6.0, "audio_flow_shift": 3.0,
 "h3_step_update": "training_euler", "enable_cfg": false,
 "lora_configs": [{"strength": 1.0, "alpha": 128}]}
```

and their ComfyUI workflows carry an explicit `MiniMaxH3SigmaShift` node,
with the instruction to "set `lora_name`, `steps`, `shift_video`, and
`shift_audio` together to match the model table". They treat a shift/checkpoint
mismatch as a known quality regression. Note also `"h3_step_update":
"training_euler"`, which names the stepping rule after training, and
`enable_cfg: false`, matching the guidance-free claim.

Their published trajectory, verbatim, reproduces the numbers in section 4
exactly: for `NFE = 4` at video shift 12 and audio shift 3, video sigma
`[1, 0.9730, 0.9231, 0.8000] -> 0` and audio sigma
`[1, 0.9000, 0.7500, 0.5000] -> 0`. Both match this document's tables.

**Where that leaves this project.** The file in use is larryvrh's, and
larryvrh's own `generate.py` hardcodes shift 12, so shift 12 is the documented
operating point *for this file* and h3 is on it. The honest caveat is that
larryvrh publishes no resolution or provenance, so which line his 4-step file
descends from cannot be read off the header, the metadata, or the card. Two
readings are consistent with the evidence:

- His file is the 544p/shift-12 line. Then h3's shift 12 is correct, and the
  open risk is a **resolution** mismatch at 1280x704, not a schedule one.
- His file is a 768p checkpoint shipped with a stale reference sampler. Then
  every run is off-trajectory at shift 12 instead of 6, and quietly so.

The first is more likely, since the author would have had to write shift 12
into his own sampler by mistake. It is not settled, and the difference is not
small: at 4 steps, shift 12 gives `1.000 / 0.973 / 0.923 / 0.800` while shift 6
gives `1.000 / 0.947 / 0.857 / 0.667`. That is a much larger gap than anything
in section 4.

**h3 can now express shift 6.** It could not when this document was first
written: `H3_VIDEO_SIGMA_SHIFT` was a `#define` with no override anywhere. Two
environment variables now exist, added in the same commit as this section:

| variable | overrides | default |
|---|---|---:|
| `H3_VIDEO_SHIFT` | video sigma shift | 12.0 |
| `H3_AUDIO_SHIFT` | audio sigma shift | 3.0 |

Both are read through `h3_video_sigma_shift()` / `h3_audio_sigma_shift()`
(`h3_host.c`), and **nothing reads the macros directly any more**. That matters
more than it looks: the audio slope correction in the RES path
(`h3_dit.c`, `h3_time_shift_slope`) used to take both constants independently
of the schedule builder, so a half-applied override would have put the video
grid on one shift and the audio velocity correction on another. Both now come
from the same two accessors, so video and audio cannot disagree about which
grid a run is on.

Unset, they change nothing: the default 4-step grid is still
`1.000000, 0.972973, 0.923077, 0.800000, 0` to the last bit, and the existing
schedule assertions in `tests/test_h3.c` (which hardcode 12.0 and 3.0) still
hold. Set to something invalid, the run refuses rather than guessing:

```
$ H3_VIDEO_SHIFT=abc ./h3 ...
h3: H3_VIDEO_SHIFT must be a number in (0, 1000], got "abc"; refusing to run rather than guess a sigma grid
```

Non-numeric, trailing garbage, zero, negative and absurd values are all
rejected, and both schedule builders return failure rather than silently
falling back to 12. Every run announces the shift it is on in its first log
lines, whether or not it is the default, so a log file is readable later
without knowing what the environment was:

```
h3: video sigma shift 12 (default)
h3: audio sigma shift 3 (default)

h3: video sigma shift 6 (OVERRIDDEN, non-default sigma grid)
h3: audio sigma shift 3 (default)
```

One footgun worth naming: `make test` asserts the default grid, so exporting
`H3_VIDEO_SHIFT` in a shell and then running the tests will fail them. That is
the correct behaviour, but it will look mysterious if you have forgotten the
export.

## 8. `--reuse` does not rebuild the sigma grid

This is a surprising property of this codebase and it is written down nowhere
else. It changes how every schedule experiment here should be read, so it gets
its own section.

**`--steps` alone decides the sigma grid. `--reuse` only decides which points
on that grid get a real denoiser evaluation.** The schedule is built from
`params->steps` (`h3.c:1492` calling `h3_serving_schedule_build`), before reuse
is considered at all. Reuse enters later, in `h3_dit_reuse_schedule`, which
picks a subset of those steps to evaluate and extrapolates the velocity at the
rest.

### Worked example: `--steps 20 --reuse 2`

`h3_dit_reuse_schedule` selects step 0, the last step, and every step where
`step % 2 == 0`, giving steps `0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19`: **11
evaluations**. The sigma handed to the model at each is `sigma[step]` from the
**20-step** grid, so it is queried at

```
1.0000, 0.9908, 0.9796, 0.9655, 0.9474, 0.9231, 0.8889, 0.8372, 0.7500,
0.5714, 0.3871
```

It is **not** queried at the 11-step grid, which would have been

```
1.0000, 0.9917, 0.9818, 0.9697, 0.9545, 0.9351, 0.9091, 0.8727, 0.8182,
0.7273, 0.5455
```

Those are different trajectories, and they diverge most where it matters least
forgivingly, at the clean end: the reuse run's last evaluation is at sigma
0.3871 and the 11-step run's is at 0.5455.

Two details that make the shape of this concrete. First, the even-indexed
entries of the 20-step grid are *exactly* the 10-step grid (`1 - 2k/20` is
`1 - k/10`), so `--steps 20 --reuse 2` is the 10-step trajectory with one extra
evaluation bolted on at step 19. Compare it against the 10-step table in
section 4: the first ten values are identical. Second, that bolted-on point is
the only thing separating the two, and it exists because
`h3_dit_reuse_schedule` force-selects the last step regardless of the interval.

The reuse run walks the *dense* grid sparsely; it does not walk a sparse grid.
The `--profile` line `h3: selected reuse schedule has 11 evaluations` reports
the evaluation count, which is the cost, and says nothing about the grid, which
is the trajectory.

### Why it matters here

Two consequences worth carrying into any future experiment on this repo:

1. **Evaluation count is not step count.** "11 evaluations" and "11 steps" are
   different runs that produce different video. Comparing a `--steps 20
   --reuse 2` run against a `--steps 11 --reuse 1` run is not a reuse
   experiment, it is a reuse experiment confounded with a schedule change.
2. **For a step distillation, only `--reuse 1` sits on the trained grid.** The
   turbo LoRA is distilled for 4 steps, so `--steps 4 --reuse 1` is the only
   configuration that queries it at exactly `1.000 / 0.973 / 0.923 / 0.800`.
   `--steps 8 --reuse 2` also costs 5 evaluations but puts the model on the
   8-step grid at every one of them. This is independent of, and additional to,
   the shift question in section 7.

`docs/720p-schedule-ladder.md` already recommends `--steps 4 --reuse 1` on
quality grounds, so the shipping configuration is correct. It is correct for a
second reason that was not known when it was chosen.

Sideways corroboration, offered as an observation and not as proof: the ladder
found that R2 (`--steps 6 --reuse 1`) bought no visible improvement over R1
(`--steps 4`) and added a faint concentric ring in the flat backdrop. 6 steps
is off the 4-step trained grid. That is what off-trajectory distillation use
tends to look like, but a single artifact on a single prompt is not evidence,
and it is not being claimed as such here.

## 9. Cheapest experiments

### 9a. Free, no GPU: confirm section 4

Reproduces both schedules from their published definitions and prints the
maximum divergence:

```sh
python3 - <<'EOF'
import struct
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
warp = lambda b, s: f32(f32(f32(s) * b) / f32(1.0 + f32(f32(s - 1.0) * b)))
h3     = lambda N, s: [warp(f32(f32(1.0) - f32(f32(i)/f32(N))), s) for i in range(N)] + [0.0]
simple = lambda N, s: [warp(f32(f32(1000 - (i*1000)//N)/1000.0), s) for i in range(N)] + [0.0]
for N in (4, 6, 8, 10):
    a, b = h3(N, 12.0), simple(N, 12.0)
    print(N, 'h3    ', ' '.join('%.6f' % v for v in a))
    print(N, 'simple', ' '.join('%.6f' % v for v in b))
    print(N, 'max|d|', '%.2e' % max(abs(x-y) for x, y in zip(a, b)))
EOF
```

Expected: `0.00e+00` at 4, 8 and 10 steps, `4.98e-04` at 6.

### 9b. The one that could actually change output quality: shift 12 vs shift 6

This is the experiment worth paying for. Section 7 leaves it genuinely open and
the sigma gap is large. The code it needed now exists, so it is two runs.

**Why the cheap canvas is a valid place to test this.** The thing in question
is a property of the **LoRA's trained trajectory**, not of the render canvas.
The sigma grid `h3_serving_schedule_build` produces depends only on the step
count and the shift: it contains no width, no height, no frame count, and is
byte-identical at 640x352 and at 1280x704. So if this file was distilled at
shift 6, it is being queried off-trajectory at *every* canvas, and the defect
should be visible at the cheap one. The cheap canvas costs about a fortieth of
the target per evaluation, which is what turns this from a two-hour question
into a twenty-minute one.

The reasoning has one real limit, stated so nobody over-reads the result: it
establishes **which shift this LoRA wants**, not **how good 1280x704 looks**.
A null result here means shift 12 is correct and this question is closed; it
does not license any claim about the target canvas, which has its own
resolution-mismatch question (section 7) that this experiment does not touch.

Run the pair at the ladder canvas from `docs/720p-schedule-ladder.md`
(640x352, 124 frames, the same 1.818:1 aspect ratio as the target), same seed,
nothing changed between them but the shift:

```sh
make -j8 && \
for S in 12 6; do \
  H3_VIDEO_SHIFT=$S ./h3 --profile -d ./MiniMax-H3 \
    -p "$(cat prompts/01-breath.txt)" \
    --lora loras/turbo.safetensors \
    --width 640 --height 352 --frames 124 \
    --steps 6 --reuse 1 --layers 50 --seed 42 --ssd-streaming \
    -o outputs/shift-$S.mp4 > logs/shift-$S.log 2>&1; \
done
```

Then read the contact sheets rather than the videos:

```sh
for S in 12 6; do \
  ffmpeg -v error -i outputs/shift-$S.mp4 \
    -vf "select='eq(n\,0)+eq(n\,41)+eq(n\,82)+eq(n\,123)',tile=4x1" \
    -frames:v 1 -y outputs/shift-$S-contact.png; \
done
```

Confirm from the logs before comparing anything, because a silently ignored
override would make the two runs identical and look like a null result:

```sh
grep "sigma shift" logs/shift-12.log logs/shift-6.log
```

`logs/shift-12.log` must say `video sigma shift 12 (default)` and
`logs/shift-6.log` must say `video sigma shift 6 (OVERRIDDEN, ...)`. Both must
say `audio sigma shift 3 (default)`.

**The grids being compared**, at 6 steps:

| i | shift 12 | shift 6 | delta |
|--:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.983607 | 0.967742 | -0.0159 |
| 2 | 0.960000 | 0.923077 | -0.0369 |
| 3 | 0.923077 | 0.857143 | -0.0659 |
| 4 | 0.857143 | 0.750000 | -0.1071 |
| 5 | 0.705882 | 0.545455 | -0.1604 |
| 6 | 0 | 0 | 0 |

**What to look for.** Compare the two contact sheets on the failure modes the
ladder already established as this model's tells when the schedule is wrong,
in this order:

1. **Structure.** `docs/720p-baseline.md` recorded that a bad low-step schedule
   here does not degrade gracefully, it produces checkerboard noise with no
   subject. Any trace of checkerboard or an unformed subject on one side is the
   whole answer and nothing else needs weighing.
2. **Background artifacts.** The ladder found a faint concentric ring in R2's
   flat lavender-grey backdrop that R1 did not have. That backdrop is the most
   sensitive surface in this prompt. Rings, banding or a woven texture on one
   side and not the other is the next strongest signal.
3. **Subject detail.** Earring shape, nail art, lace trim, individual hair
   strands. The ladder describes these as intact at 4 and 6 evaluations, so
   softening or mush on one side is meaningful.
4. **Identity across frames.** Same face at frame 0 and frame 123.

**Decision rule.** If shift 6 is visibly better on any of the first three,
larryvrh's file is from the 768p line, shift 12 has been wrong all along, and
both this document and `docs/720p-schedule-ladder.md` need revisiting before
the 1280x704 run. If shift 12 wins, or the two are indistinguishable, section
7's first reading holds, shift 12 is correct, and the question is closed.

Cost: two 6-evaluation runs at the cheap canvas, which is exactly the rung R2
already ran. The shift-12 arm reproduces R2 exactly (same canvas, frames,
steps, reuse, layers, seed, prompt), so `outputs/ladder-R2-contact.png` is a
free third data point: if the new shift-12 contact sheet does not match the
stored R2 one, something other than the shift changed and the comparison is
void.

Two caveats on reading the result. `01-breath.txt` is nearly static, which the
ladder already flags as the easiest possible case; if the two sheets are close,
repeat with `prompts/04-water.txt`, where a trajectory mismatch would show in
motion before it shows in a held pose. And a **null result is not proof of
identity**, only evidence that any difference is below what a four-frame
contact sheet at this canvas resolves.

### 9c. If shift 6 wins

`--steps 4` is what actually ships, not 6. Re-run the winning shift at
`--steps 4 --reuse 1` before changing anything in
`docs/720p-schedule-ladder.md`, because that rung is the one the 720p plan
depends on. The 4-step grids differ by a comparable margin:
`1.000000 / 0.972973 / 0.923077 / 0.800000` at shift 12 against
`1.000000 / 0.947368 / 0.857143 / 0.666667` at shift 6.

### 9d. Not worth it yet: the sampler

The vendor pairs `simple` with `res_multistep`; h3 defaults to Euler. h3's RES
solver **is not reachable from the CLI**: `h3_dit_denoise` (`h3_dit.c:2930`) is
called only from `tests/`, while the CLI calls `h3_dit_denoise_euler_preview`
(`h3.c:1612`). There is no flag for it. `README:523-525` already reports that
RES lost an earlier bake-off on the low-budget path, and ModelTC's own ComfyUI
workflows select `euler`, not `res_multistep`, alongside the same `simple`
schedule. Two of the three vendor-adjacent sources therefore agree with h3's
default. Leave it.

---

## Sources

Repo, read at `main`:

- `h3_host.c:150` `h3_serving_schedule_build`, the shipped schedule
- `h3_host.c:135` `h3_schedule_build` and `h3_shifted_sigma`, tests and benches only
- `h3_host.h:12-13` the two shift defaults, `h3_video_sigma_shift()` and
  `h3_audio_sigma_shift()` in `h3_host.c` the accessors that override them
- `h3.c:1492` schedule construction, `h3.c:1612` the Euler call
- `h3_dit.c:2930` `h3_dit_denoise`, the RES path, tests only
- `MiniMax-H3/FL2VA/model_index.json` `sigma_shift_scales`
- `loras/turbo.safetensors` safetensors header

External, read 2026-09-14:

- ComfyUI `comfy/samplers.py:645-652` (`simple_scheduler`),
  `comfy/model_sampling.py:289-329` (`ModelSamplingDiscreteFlow`),
  `comfy/supported_models.py:962-970` (`MiniMaxH3` sampling settings),
  at `f42b24efbeee194513fff465d84ad6913a2698d5`
- `Comfy-Org/workflow_templates`, `templates/video_minimax_h3_t2v.json`
- `huggingface.co/MiniMaxAI/MiniMax-H3`, `scheduler/scheduler_config.json` and
  `audio_scheduler/scheduler_config.json`
- `huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora`, model card, `generate.py`,
  `minimax_h3_t2v_turbo.json`, discussion #41
- `github.com/ModelTC/Minimax-H3-Turbo` README training-shift table
- `github.com/ModelTC/LightX2V`, `configs/minimax_h3/dmd/*.json`
- `minimax3.com/blog/minimax-h3-turbo-steps`
