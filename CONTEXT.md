# Runtime LoRA adapters in h3-metal

This context covers loading LoRA adapters at run time on the `--ssd-streaming`
path, without baking anything into the 62 GB base checkpoint. The design map is
issue [#1](https://github.com/neokree/h3.c/issues/1); the spec is `docs/WHAT.md`.

The glossary is written for a reader who is not a machine-learning engineer.
Terms in the first section are ours and are used with exactly this meaning;
the second section explains the background vocabulary the first one leans on.

## Language

**Adapter**:
One parsed LoRA file plus the strength it is applied at. Lives in the `h3_ctx`
cache keyed by path, size and mtime, so nothing owns it and it cannot outlive
the model.
_Avoid_: LoRA object, adapter handle (there are no handles; the API is declarative)

**Active set**:
The list of adapters in effect for one generation. Built fresh from `h3_params`
each time and replaced wholesale, never mutated in place. Strength-0 entries are
dropped while it is built, which is why strength 0 is byte-identical to no LoRA.
_Avoid_: adapter stack, loaded LoRAs

**Pair**:
The two tensors `lora_A` and `lora_B` that together describe one adapted weight.
The unit of everything: counting, rank, shape checks, error reporting.
_Avoid_: tensor, layer, module

**Target**:
The base-model weight a pair is meant to modify, found by matching the pair's
name. A pair whose name has no target, or whose shape does not fit one, is
**unapplicable** and stops the load.
_Avoid_: destination, matched weight

**Delta**:
The small correction a pair contributes, `strength * alpha/rank * B*A*x`. It is
added to the **output** of an existing linear op, never merged into the weight
buffer, because streaming overwrites weight buffers on every denoiser
evaluation.
_Avoid_: patch, offset, merge

**Branch**:
The extra path in the compute graph that produces the delta, sitting next to the
untouched base op. With no LoRA the branch does not execute, so the no-LoRA path
is byte-identical by construction.
_Avoid_: hook, injection

**Strength**:
The user's per-adapter multiplier from `--lora PATH:STRENGTH`, default 1.0. It
multiplies the file's own `alpha/rank` scale; it does not replace it.
_Avoid_: weight, scale, alpha

**Base weight**:
A tensor of the original checkpoint. Immutable for the whole life of this
feature: nothing in the LoRA path ever writes one.
_Avoid_: model weight, original weight

**Convention**:
The naming scheme a LoRA file uses for its pairs. Convention A (`lora_A` /
`lora_B`, optional `.alpha`) is supported; convention B (`lora_up` /
`lora_down`) is detected and rejected with a clear error.
_Avoid_: format, dialect

**Guardrail**:
The two measured memory gates that stop a run before it thrashes: gate 1
pre-flight from known sizes, gate 2 after the first denoiser evaluation from the
real process footprint. It bounds the whole process, not the adapters.
_Avoid_: cap, budget, limit (there is no static adapter cap)

**Load report**:
The one-time summary printed when an active set is activated: pair count, rank
histogram, AdaLN pair count, and any conversion warning. Leaves the library
through a callback; the library never prints for itself.
_Avoid_: log, output

**Oracle**:
The float32 CPU reference inside the C test binary that the GPU branch is
measured against (`tests/test_lora_oracle.c`). Numerical agreement is measured,
never asserted.
_Avoid_: golden file, ground truth

## Background terms

These are not ours; they are the vocabulary the definitions above assume.

**LoRA** (Low-Rank Adaptation):
A small file that changes a big model's behaviour without changing the model.
Instead of storing a full replacement weight, it stores two thin matrices whose
product is the correction. That is why a 62 GB model can be restyled by a 592 MiB
file.

**Rank**:
The thickness of those two thin matrices: 16, 64, 128. Higher rank means more
capacity and a bigger file. It is read **per pair** from the file, not assumed
uniform: the reference turbo LoRA is rank 64 everywhere except its AdaLN pairs
at rank 16.

**Alpha**:
A scale factor stored in the LoRA file, applied as `alpha/rank`. It is the
file's own opinion about how strong it should be, distinct from the user's
strength.

**DiT** (Diffusion Transformer):
The model that does the actual denoising, 50 near-identical **blocks** run once
per denoiser evaluation. The 62 GB `FL2VA/transformer` is this. LoRA applies
here, not to the text encoder.

**Denoiser evaluation**:
One full pass of the DiT over the latent. The unit that costs time: `--steps`
and `--reuse` together decide how many happen, and every one re-reads ~36 GiB of
weights from SSD.

**Linear op / projection**:
A matrix multiply, the operation LoRA modifies. In h3 it is
`h3_gpu_linear_bf16`. The four projections in a block are the query-key-value
projection, the attention output, and the two MLP layers.

**Token refiner**:
Two small transformer blocks that pre-process the text embedding once, before
the denoiser loop, and cache the result in `dit->refined_text`. LoRA applies
here, so a change to the active set invalidates that cache.

**AdaLN** (Adaptive Layer Normalisation):
How the DiT is told which denoising step it is on. Each block has one large
`adaln_proj` weight (496 MiB) projected against the timestep features **once at
load**, producing a table the denoiser then just reads. LoRA applies to those
weights, so a change to the active set invalidates the table.

**Precompute**:
Work done once at load whose result is read many times during the run. Both the
token refiner and the AdaLN table are precomputes; both cost real seconds, which
is why "when do we redo them" is a design question and not an implementation
detail.

**Safetensors**:
The file format both the checkpoint and LoRA files use: a JSON header of names,
shapes and dtypes, then raw bytes. The header alone is enough to validate a
file, which is why a bad LoRA can be rejected in a second instead of after
62 GiB.

**bf16**:
The 16-bit float the model runs in. Half the memory of float32 and considerably
less precision, which is why the branch's measured error floor (rel-L2 1.66e-03)
is the base matrix multiply's error, not the LoRA branch's.

**rel-L2**:
The error measure used in T1: the length of the difference between two result
vectors, divided by the length of the correct one. A single relative number, so
one threshold covers tensors of different sizes.

**Streaming** (`--ssd-streaming`):
Reading DiT weights from SSD on demand instead of holding them in RAM, into two
alternating slots that are overwritten constantly. Free on this machine
(measured 0.006 s of unhidden wait) and the reason a delta may never be merged
into a weight buffer.
