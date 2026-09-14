#ifndef H3_RESUME_H
#define H3_RESUME_H

#include <stddef.h>
#include <stdint.h>

/* One interrupted generation, on disk. Written after every denoiser
 * evaluation of the host Euler sampler and overwritten in place: one file, no
 * rotation and no history. It carries the state the sampler cannot rebuild -
 * the two latents, the position in the sigma schedule and the velocity history
 * the reuse machinery extrapolates from - plus the identity string that
 * decides whether a resume is the same run.
 *
 * Everything else (text embedding, AdaLN table, weights, conditions) is a
 * deterministic function of that identity, so it is rebuilt rather than
 * stored. */

#define H3_RESUME_MAGIC "H3RESUME"
#define H3_RESUME_VERSION 1u

typedef struct {
    /* The run's identity, as h3_resume_key builds it. Owned by a state that
     * came out of h3_resume_read, borrowed by one going into
     * h3_resume_write. */
    char *key;
    /* The step the sampler must run next: latents are as they were on entry
     * to it, so step == steps means the denoise is finished. */
    int step;
    int steps;
    /* Reuse bookkeeping. -1 for "none yet", which is also the state at
     * --reuse 1, where no velocity history exists at all. */
    int last_evaluated;
    int previous_evaluated;
    size_t video_elements;
    size_t audio_elements;
    float *video_latent;
    float *audio_latent;
    /* NULL together when last_evaluated < 0, and the previous_* pair is NULL
     * when previous_evaluated < 0. */
    float *last_video;
    float *last_audio;
    float *previous_video;
    float *previous_audio;
} h3_resume_state;

/* Replace the checkpoint at `path`, atomically: the bytes land in
 * `path`.partial, reach the disk, and only then take the name, so an
 * interruption during the write cannot destroy the previous checkpoint.
 * Nothing in `state` is owned or retained. */
int h3_resume_write(const char *path, const h3_resume_state *state,
                    char *error, size_t error_size);

/* Load the checkpoint at `path` into `state`, refusing it unless its identity
 * is `key` and its latents are the expected sizes. A refusal names the first
 * identity field that differs. On success the caller owns `state` and must
 * release it; on failure `state` is zeroed and nothing is allocated. */
int h3_resume_read(const char *path, const char *key,
                   size_t video_elements, size_t audio_elements,
                   h3_resume_state *state, char *error, size_t error_size);

void h3_resume_release(h3_resume_state *state);

/* The identity of one generation: the prepared-DiT key, which already spells
 * out canvas, frames, steps, layers, every quality flag, the prompt and the
 * adapters with their strengths, plus the three things that steer the
 * trajectory without changing the prepared DiT. The caller owns the result. */
char *h3_resume_key(const char *prepared_key, uint64_t seed,
                    int denoise_reuse, const char *reuse_steps);

#endif
