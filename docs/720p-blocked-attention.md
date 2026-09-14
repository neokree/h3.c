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

Two separate thresholds, both in MPSGraph and neither documented:

- Around N = 8,800 the whole-sequence kernel starts disagreeing with the
  blocked one in the last bit. Both remain at the bf16 error floor: benign.
- By N = 9,100 the whole-sequence kernel's error jumps to **10 to 15 times the
  bf16 floor**. The blocked path does not move. The already-known hard abort
  (`too large for kernel`, measured between N = 12,370 and N = 12,398 at 56
  heads, 2^33 elements) is a third, further threshold.

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

## Still unmeasured

The 1280x704 / 124-frame probe has not been run. Nothing in this document says
anything about seconds per evaluation at the target, real footprint, or how
many denoiser evaluations fit in 90 minutes.

## Files

- `h3_gpu.m` — `h3_gpu_sdpa`, `h3_gpu_sdpa_graph`, `h3_gpu_graph_data`.
- `tests/test_sdpa_blocking.c` — the block-invariance and oracle check.
- `H3_SDPA_QUERY_BLOCK` — block size, default 256 (0.89 GiB of scores at the
  target). 0 restores the single whole-sequence encode.
