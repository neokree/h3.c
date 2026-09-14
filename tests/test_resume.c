/* The resume checkpoint, round-tripped.
 *
 * The subject here is the FILE: does every field of a sampler state survive a
 * write and a read, and is a checkpoint from a different run refused by name.
 * Nothing is downloaded and no GPU is touched, so these checks cannot skip -
 * which matters, because the thing they protect is an hour of denoising that
 * only shows up wrong at the end.
 *
 * What this cannot cover is the sampler itself. The pipeline is not
 * byte-reproducible at these canvases, so a resumed run cannot be compared
 * byte-for-byte against an uninterrupted one; that half is measured by
 * reading the schedule position back out of the checkpoint and by a contact
 * sheet, not asserted here.
 */

#include "h3_resume.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

enum { VIDEO = 97, AUDIO = 13 };

static char directory[] = "/tmp/h3-resume-XXXXXX";
static char checkpoint[256];
static char partial[256];

static void cleanup(void) {
    remove(checkpoint);
    remove(partial);
    rmdir(directory);
}

/* Distinct, exactly representable, and different in every array. */
static void fill(float *values, size_t count, int tag) {
    for (size_t index = 0; index < count; index++)
        values[index] = (float)((int)index * 4 + tag) * 0.25f;
}

static float video_latent[VIDEO], audio_latent[AUDIO];
static float last_video[VIDEO], last_audio[AUDIO];
static float previous_video[VIDEO], previous_audio[AUDIO];

static const char *base_key(void) {
    static char *key;
    if (!key)
        key = h3_resume_key("mode=0|prompt=5:hello|shape=1280x704x124|steps=6"
                            "|layers=50|lora=12:turbo.safetensors:9:1:2"
                            "|strength=0.8", 42, 2, NULL);
    return key;
}

/* The same identity with one field changed, the way a mistyped flag would. */
static char *variant(const char *replace, const char *with) {
    const char *key = base_key();
    const char *at = strstr(key, replace);
    CHECK(at != NULL);
    size_t prefix = (size_t)(at - key);
    size_t size = strlen(key) + strlen(with) + 1;
    char *changed = malloc(size);
    CHECK(changed != NULL);
    snprintf(changed, size, "%.*s%s%s", (int)prefix, key, with,
             at + strlen(replace));
    return changed;
}

static h3_resume_state full_state(void) {
    h3_resume_state state;
    memset(&state, 0, sizeof(state));
    state.key = (char *)base_key();
    state.step = 4;
    state.steps = 6;
    state.last_evaluated = 3;
    state.previous_evaluated = 1;
    state.video_elements = VIDEO;
    state.audio_elements = AUDIO;
    state.video_latent = video_latent;
    state.audio_latent = audio_latent;
    state.last_video = last_video;
    state.last_audio = last_audio;
    state.previous_video = previous_video;
    state.previous_audio = previous_audio;
    return state;
}

static void test_round_trip(void) {
    char error[512];
    h3_resume_state written = full_state();
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));
    CHECK(error[0] == '\0');
    CHECK(access(partial, F_OK) != 0);

    h3_resume_state read;
    CHECK(h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                         sizeof(error)));
    CHECK(!strcmp(read.key, base_key()));
    CHECK(read.step == 4);
    CHECK(read.steps == 6);
    CHECK(read.last_evaluated == 3);
    CHECK(read.previous_evaluated == 1);
    CHECK(read.video_elements == VIDEO);
    CHECK(read.audio_elements == AUDIO);
    CHECK(!memcmp(read.video_latent, video_latent, sizeof(video_latent)));
    CHECK(!memcmp(read.audio_latent, audio_latent, sizeof(audio_latent)));
    CHECK(!memcmp(read.last_video, last_video, sizeof(last_video)));
    CHECK(!memcmp(read.last_audio, last_audio, sizeof(last_audio)));
    CHECK(!memcmp(read.previous_video, previous_video,
                  sizeof(previous_video)));
    CHECK(!memcmp(read.previous_audio, previous_audio,
                  sizeof(previous_audio)));
    h3_resume_release(&read);
    CHECK(read.key == NULL && read.video_latent == NULL);
}

/* At --reuse 1 there is no velocity history at all, and the first evaluation
 * of any run has no previous one. Both must survive as absent, not as zeros:
 * a zeroed history that reads back as present would extrapolate. */
static void test_absent_history(void) {
    char error[512];
    h3_resume_state written = full_state();
    written.step = 1;
    written.last_evaluated = -1;
    written.previous_evaluated = -1;
    written.last_video = written.last_audio = NULL;
    written.previous_video = written.previous_audio = NULL;
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));

    h3_resume_state read;
    CHECK(h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                         sizeof(error)));
    CHECK(read.last_evaluated == -1 && read.previous_evaluated == -1);
    CHECK(read.last_video == NULL && read.last_audio == NULL);
    CHECK(read.previous_video == NULL && read.previous_audio == NULL);
    CHECK(!memcmp(read.video_latent, video_latent, sizeof(video_latent)));
    h3_resume_release(&read);

    /* Only the last velocity, no previous one. */
    written.last_evaluated = 2;
    written.last_video = last_video;
    written.last_audio = last_audio;
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));
    CHECK(h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                         sizeof(error)));
    CHECK(read.last_evaluated == 2 && read.previous_evaluated == -1);
    CHECK(read.last_video && !memcmp(read.last_video, last_video,
                                     sizeof(last_video)));
    CHECK(read.previous_video == NULL);
    h3_resume_release(&read);
}

/* Every refusal, and the field each one has to name. */
static void test_refusal(void) {
    char error[512];
    h3_resume_state written = full_state();
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));

    static const struct { const char *from; const char *to; const char *named; }
    changes[] = {
        {"seed=42", "seed=43", "seed"},
        {"prompt=5:hello", "prompt=5:world", "prompt"},
        {"shape=1280x704x124", "shape=1280x704x107", "shape"},
        {"steps=6", "steps=8", "steps"},
        {"layers=50", "layers=45", "layers"},
        {"denoise-reuse=2", "denoise-reuse=1", "denoise-reuse"},
        {"reuse-steps=-", "reuse-steps=0,3,5", "reuse-steps"},
        {"strength=0.8", "strength=1", "strength"},
        {"lora=12:turbo.safetensors:9:1:2",
         "lora=13:mystic.safetensors:9:1:2", "lora"}
    };
    for (size_t index = 0; index < sizeof(changes) / sizeof(*changes);
         index++) {
        char *changed = variant(changes[index].from, changes[index].to);
        h3_resume_state read;
        CHECK(!h3_resume_read(checkpoint, changed, VIDEO, AUDIO, &read, error,
                              sizeof(error)));
        CHECK(read.key == NULL && read.video_latent == NULL);
        CHECK(strstr(error, changes[index].named) != NULL);
        CHECK(strstr(error, "resume refused") != NULL);
        free(changed);
    }

    /* An adapter added on the command line inserts a field rather than
     * changing one, so the message has to name both sides instead of the
     * field that happened to shift. */
    h3_resume_state read;
    char *added = h3_resume_key("mode=0|prompt=5:hello|shape=1280x704x124"
                                "|steps=6|layers=50"
                                "|lora=12:turbo.safetensors:9:1:2"
                                "|strength=0.8"
                                "|lora=13:mystic.safetensors:9:1:2"
                                "|strength=0.5", 42, 2, NULL);
    CHECK(added != NULL);
    CHECK(!h3_resume_read(checkpoint, added, VIDEO, AUDIO, &read, error,
                          sizeof(error)));
    CHECK(strstr(error, "seed") && strstr(error, "lora"));
    free(added);

    /* A key the checkpoint simply has more of, which is what a future build
     * that adds an identity field would look like to an old checkpoint. */
    CHECK(!h3_resume_read(checkpoint, "mode=0", VIDEO, AUDIO, &read, error,
                          sizeof(error)));
    CHECK(strstr(error, "prompt") && strstr(error, "extra"));

    /* A key that matches but a latent that does not: a canvas the identity
     * somehow let through must still not be memcpy'd into the wrong buffer. */
    CHECK(!h3_resume_read(checkpoint, base_key(), VIDEO + 1, AUDIO, &read,
                          error, sizeof(error)));
    CHECK(strstr(error, "resume refused") != NULL);
    CHECK(!h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO + 1, &read,
                          error, sizeof(error)));
    CHECK(strstr(error, "resume refused") != NULL);
}

static void test_damaged(void) {
    char error[512];
    h3_resume_state read;

    FILE *file = fopen(checkpoint, "wb");
    CHECK(file != NULL);
    CHECK(fwrite("not a checkpoint at all, not even close", 1, 39, file) == 39);
    fclose(file);
    CHECK(!h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                          sizeof(error)));
    CHECK(strstr(error, "not a resume checkpoint") != NULL);

    /* A header and an identity, then nothing: the latents must not be
     * accepted as whatever malloc happened to return. */
    h3_resume_state written = full_state();
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));
    long size = 0;
    file = fopen(checkpoint, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    fclose(file);
    CHECK(size > 64);
    CHECK(truncate(checkpoint, size - 8) == 0);
    CHECK(!h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                          sizeof(error)));
    CHECK(strstr(error, "truncated") != NULL);
}

/* One file, overwritten. The second write must replace the first completely,
 * and must not leave the partial behind. */
static void test_overwrite(void) {
    char error[512];
    h3_resume_state written = full_state();
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));
    fill(video_latent, VIDEO, 77);
    written.step = 5;
    written.last_evaluated = 4;
    written.previous_evaluated = 3;
    CHECK(h3_resume_write(checkpoint, &written, error, sizeof(error)));
    CHECK(access(partial, F_OK) != 0);

    h3_resume_state read;
    CHECK(h3_resume_read(checkpoint, base_key(), VIDEO, AUDIO, &read, error,
                         sizeof(error)));
    CHECK(read.step == 5 && read.last_evaluated == 4);
    CHECK(!memcmp(read.video_latent, video_latent, sizeof(video_latent)));
    h3_resume_release(&read);
}

int main(void) {
    if (!mkdtemp(directory)) {
        fprintf(stderr, "FAIL %s: cannot create a temporary directory\n",
                __FILE__);
        return 1;
    }
    atexit(cleanup);
    snprintf(checkpoint, sizeof(checkpoint), "%s/run.h3resume", directory);
    snprintf(partial, sizeof(partial), "%s.partial", checkpoint);

    fill(video_latent, VIDEO, 1);
    fill(audio_latent, AUDIO, 2);
    fill(last_video, VIDEO, 3);
    fill(last_audio, AUDIO, 4);
    fill(previous_video, VIDEO, 5);
    fill(previous_audio, AUDIO, 6);
    CHECK(base_key() != NULL);

    test_round_trip();
    test_absent_history();
    test_refusal();
    test_damaged();
    test_overwrite();

    printf("ok: %d checks on the resume checkpoint\n", checks);
    return 0;
}
