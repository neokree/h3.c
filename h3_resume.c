/* The resume checkpoint: a fixed 64-byte header, the identity string, then
 * the latents and the velocity history as raw float32.
 *
 * Not safetensors, deliberately. Everything here is float32 and every array
 * length is already implied by two element counts, so a JSON header would add
 * a parser and a dependency to describe six arrays of one dtype. The identity
 * string is the only variable-length field, and it is length-prefixed.
 */

#include "h3_resume.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    H3_RESUME_HAS_LAST = 1u,
    H3_RESUME_HAS_PREVIOUS = 2u
};

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t key_bytes;          /* including the terminating NUL */
    int32_t step;
    int32_t steps;
    int32_t last_evaluated;
    int32_t previous_evaluated;
    uint32_t flags;
    uint32_t reserved;           /* keeps the header padding-free */
    uint64_t video_elements;
    uint64_t audio_elements;
    uint64_t payload_bytes;      /* the float arrays only */
} h3_resume_header;

/* No implicit padding: the file layout is the struct layout. */
_Static_assert(sizeof(h3_resume_header) == 64, "resume header is 64 bytes");

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint32_t state_flags(const h3_resume_state *state) {
    uint32_t flags = 0;
    if (state->last_video && state->last_audio) flags |= H3_RESUME_HAS_LAST;
    if (state->previous_video && state->previous_audio)
        flags |= H3_RESUME_HAS_PREVIOUS;
    return flags;
}

static uint64_t payload_bytes(const h3_resume_state *state, uint32_t flags) {
    uint64_t video = (uint64_t)state->video_elements * sizeof(float);
    uint64_t audio = (uint64_t)state->audio_elements * sizeof(float);
    uint64_t total = video + audio;
    if (flags & H3_RESUME_HAS_LAST) total += video + audio;
    if (flags & H3_RESUME_HAS_PREVIOUS) total += video + audio;
    return total;
}

char *h3_resume_key(const char *prepared_key, uint64_t seed,
                    int denoise_reuse, const char *reuse_steps) {
    if (!prepared_key) return NULL;
    if (!reuse_steps || !*reuse_steps) reuse_steps = "-";
    /* The prepared-DiT key is already the run's identity for everything that
     * is baked into the loaded model. These three are not: the seed picks the
     * initial noise, and the two reuse controls pick which steps evaluate. */
    int wanted = snprintf(NULL, 0, "%s|seed=%llu|denoise-reuse=%d"
                          "|reuse-steps=%s", prepared_key,
                          (unsigned long long)seed, denoise_reuse,
                          reuse_steps);
    if (wanted < 0) return NULL;
    char *key = malloc((size_t)wanted + 1);
    if (!key) return NULL;
    snprintf(key, (size_t)wanted + 1, "%s|seed=%llu|denoise-reuse=%d"
             "|reuse-steps=%s", prepared_key, (unsigned long long)seed,
             denoise_reuse, reuse_steps);
    return key;
}

/* One '|'-separated field of an identity string. */
static size_t field_length(const char *text) {
    const char *bar = strchr(text, '|');
    return bar ? (size_t)(bar - text) : strlen(text);
}

/* The name half of a "name=value" field. */
static size_t name_length(const char *field, size_t length) {
    const char *equals = memchr(field, '=', length);
    return equals ? (size_t)(equals - field) : length;
}

/* Name the first field of the checkpoint's identity that the current run does
 * not match. Both strings are the same key format, so a positional walk is
 * enough and the field name comes straight out of the text. */
static void report_difference(const char *checkpoint, const char *current,
                              char *error, size_t error_size) {
    for (;;) {
        size_t stored = field_length(checkpoint);
        size_t wanted = field_length(current);
        if (stored != wanted || memcmp(checkpoint, current, stored)) {
            size_t stored_name = name_length(checkpoint, stored);
            size_t current_name = name_length(current, wanted);
            if (stored > 72) stored = 72;
            if (wanted > 72) wanted = 72;
            /* Same field with a different value is the ordinary case. A
             * different field means one side gained or lost one, which is
             * what adding or dropping an adapter looks like, so both names
             * have to be in the message. */
            if (stored_name == current_name &&
                !memcmp(checkpoint, current, stored_name))
                fail(error, error_size,
                     "resume refused: %.*s differs (the checkpoint has %.*s, "
                     "this run has %.*s); delete the file to start a new run",
                     (int)stored_name, checkpoint, (int)stored, checkpoint,
                     (int)wanted, current);
            else
                fail(error, error_size,
                     "resume refused: the checkpoint has %.*s where this run "
                     "has %.*s; delete the file to start a new run",
                     (int)stored, checkpoint, (int)wanted, current);
            return;
        }
        checkpoint += stored;
        current += wanted;
        if (!*checkpoint || !*current) {
            if (*checkpoint == *current) {
                fail(error, error_size, "resume refused: the checkpoint was "
                     "written by a different run");
                return;
            }
            /* One side carries a field the other does not, which is what
             * adding or dropping an adapter looks like. */
            const char *extra = *checkpoint ? checkpoint + 1 : current + 1;
            size_t length = name_length(extra, field_length(extra));
            fail(error, error_size, "resume refused: %.*s is %s in the "
                 "checkpoint; delete the file to start a new run",
                 (int)length, extra, *checkpoint ? "extra" : "missing");
            return;
        }
        checkpoint++;
        current++;
    }
}

int h3_resume_write(const char *path, const h3_resume_state *state,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !state || !state->key || !state->video_latent ||
        !state->audio_latent) {
        fail(error, error_size, "invalid resume checkpoint arguments");
        return 0;
    }
    size_t key_bytes = strlen(state->key) + 1;
    uint32_t flags = state_flags(state);
    h3_resume_header header = {
        .version = H3_RESUME_VERSION,
        .key_bytes = (uint32_t)key_bytes,
        .step = state->step,
        .steps = state->steps,
        .last_evaluated = state->last_evaluated,
        .previous_evaluated = state->previous_evaluated,
        .flags = flags,
        .reserved = 0,
        .video_elements = state->video_elements,
        .audio_elements = state->audio_elements,
        .payload_bytes = payload_bytes(state, flags)
    };
    memcpy(header.magic, H3_RESUME_MAGIC, sizeof(header.magic));

    /* Write beside the checkpoint and rename over it. An interruption during
     * the write then costs the new checkpoint, never the previous one: the
     * one outcome this feature must not produce is losing the hour it was
     * meant to protect. */
    size_t partial_size = strlen(path) + sizeof(".partial");
    char *partial = malloc(partial_size);
    if (!partial) {
        fail(error, error_size, "out of memory naming the resume checkpoint");
        return 0;
    }
    snprintf(partial, partial_size, "%s.partial", path);
    FILE *file = fopen(partial, "wb");
    if (!file) {
        fail(error, error_size, "cannot open %s for writing", partial);
        free(partial);
        return 0;
    }
    const float *arrays[6] = {
        state->video_latent, state->audio_latent,
        (flags & H3_RESUME_HAS_LAST) ? state->last_video : NULL,
        (flags & H3_RESUME_HAS_LAST) ? state->last_audio : NULL,
        (flags & H3_RESUME_HAS_PREVIOUS) ? state->previous_video : NULL,
        (flags & H3_RESUME_HAS_PREVIOUS) ? state->previous_audio : NULL
    };
    int ok = fwrite(&header, sizeof(header), 1, file) == 1 &&
             fwrite(state->key, 1, key_bytes, file) == key_bytes;
    for (size_t index = 0; ok && index < 6; index++) {
        if (!arrays[index]) continue;
        size_t count = index % 2 ? state->audio_elements
                                 : state->video_elements;
        if (count && fwrite(arrays[index], sizeof(float), count, file) != count)
            ok = 0;
    }
    if (ok) ok = fflush(file) == 0 && fsync(fileno(file)) == 0;
    if (fclose(file) != 0) ok = 0;
    if (ok && rename(partial, path) != 0) ok = 0;
    if (!ok) {
        fail(error, error_size, "cannot write the resume checkpoint %s", path);
        remove(partial);
    }
    free(partial);
    return ok;
}

void h3_resume_release(h3_resume_state *state) {
    if (!state) return;
    free(state->key);
    free(state->video_latent);
    free(state->audio_latent);
    free(state->last_video);
    free(state->last_audio);
    free(state->previous_video);
    free(state->previous_audio);
    memset(state, 0, sizeof(*state));
}

static float *read_array(FILE *file, size_t count) {
    float *values = malloc(count ? count * sizeof(*values) : 1);
    if (!values) return NULL;
    if (count && fread(values, sizeof(*values), count, file) != count) {
        free(values);
        return NULL;
    }
    return values;
}

int h3_resume_read(const char *path, const char *key,
                   size_t video_elements, size_t audio_elements,
                   h3_resume_state *state, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (state) memset(state, 0, sizeof(*state));
    if (!path || !key || !state) {
        fail(error, error_size, "invalid resume checkpoint arguments");
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail(error, error_size, "cannot open the resume checkpoint %s", path);
        return 0;
    }
    h3_resume_header header;
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        memcmp(header.magic, H3_RESUME_MAGIC, sizeof(header.magic))) {
        fail(error, error_size, "%s is not a resume checkpoint", path);
        fclose(file);
        return 0;
    }
    if (header.version != H3_RESUME_VERSION) {
        fail(error, error_size, "resume checkpoint %s is format version %u, "
             "this build writes version %u", path, header.version,
             H3_RESUME_VERSION);
        fclose(file);
        return 0;
    }
    /* The identity first, so a mismatched run is refused before a hundred
     * megabytes of latent are read. */
    if (!header.key_bytes || header.key_bytes > (1u << 24)) {
        fail(error, error_size, "resume checkpoint %s has a malformed "
             "identity", path);
        fclose(file);
        return 0;
    }
    char *stored = malloc(header.key_bytes);
    if (!stored) {
        fail(error, error_size, "out of memory reading the resume checkpoint");
        fclose(file);
        return 0;
    }
    if (fread(stored, 1, header.key_bytes, file) != header.key_bytes ||
        stored[header.key_bytes - 1] != '\0') {
        fail(error, error_size, "resume checkpoint %s is truncated", path);
        free(stored);
        fclose(file);
        return 0;
    }
    if (strcmp(stored, key)) {
        report_difference(stored, key, error, error_size);
        free(stored);
        fclose(file);
        return 0;
    }
    if (header.video_elements != video_elements ||
        header.audio_elements != audio_elements) {
        fail(error, error_size, "resume refused: the checkpoint holds a "
             "%llu/%llu latent, this run wants %zu/%zu",
             (unsigned long long)header.video_elements,
             (unsigned long long)header.audio_elements,
             video_elements, audio_elements);
        free(stored);
        fclose(file);
        return 0;
    }
    if (header.steps < 1 || header.step < 0 || header.step > header.steps ||
        header.last_evaluated >= header.steps ||
        header.previous_evaluated >= header.steps ||
        (header.previous_evaluated >= 0 &&
         header.previous_evaluated >= header.last_evaluated)) {
        fail(error, error_size, "resume checkpoint %s holds an impossible "
             "schedule position (step %d of %d)", path, header.step,
             header.steps);
        free(stored);
        fclose(file);
        return 0;
    }

    state->key = stored;
    state->step = header.step;
    state->steps = header.steps;
    state->last_evaluated = header.last_evaluated;
    state->previous_evaluated = header.previous_evaluated;
    state->video_elements = video_elements;
    state->audio_elements = audio_elements;
    state->video_latent = read_array(file, video_elements);
    state->audio_latent = read_array(file, audio_elements);
    int ok = state->video_latent && state->audio_latent;
    if (ok && (header.flags & H3_RESUME_HAS_LAST)) {
        state->last_video = read_array(file, video_elements);
        state->last_audio = read_array(file, audio_elements);
        ok = state->last_video && state->last_audio;
    }
    if (ok && (header.flags & H3_RESUME_HAS_PREVIOUS)) {
        state->previous_video = read_array(file, video_elements);
        state->previous_audio = read_array(file, audio_elements);
        ok = state->previous_video && state->previous_audio;
    }
    fclose(file);
    if (!ok) {
        fail(error, error_size, "resume checkpoint %s is truncated", path);
        h3_resume_release(state);
        return 0;
    }
    /* A velocity history the sampler would read but the file does not carry
     * is the one corruption that would resume into a different trajectory
     * without looking wrong. */
    if ((header.last_evaluated >= 0) != (state->last_video != NULL) ||
        (header.previous_evaluated >= 0) != (state->previous_video != NULL)) {
        fail(error, error_size, "resume checkpoint %s names a velocity "
             "history it does not carry", path);
        h3_resume_release(state);
        return 0;
    }
    return 1;
}
