CC := clang
AR := ar
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c h3_lora.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)
CLI_OBJ := main.o h3_cli.o linenoise.o

.PHONY: all test parity real-parity real-lora lora-identity \
	lora-hotswap clean

all: h3 libh3.a

h3: $(CLI_OBJ) $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

libh3.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_av_mux_test: tests/test_av_mux.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_lora_oracle_test: tests/test_lora_oracle.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_lora_fixture_tests: tests/test_lora_fixtures.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_lora_gemm_bench: tests/bench_lora_gemm.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
	h3_av_mux_test h3_lora_fixture_tests \
	h3_real_video_encoder_test h3_real_qwen_vision_test \
	h3_real_multimodal_text_test h3_real_ref_video_text_test

	./h3_tests
	./h3_lora_fixture_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if command -v ffmpeg >/dev/null 2>&1; then \
		./h3_av_mux_test; \
	else \
		echo "skip: FFmpeg is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

# ---- the real half of the LoRA corpus (SPEC 8bis) ----
#
# These want the 62 GB checkpoint and the two corpus LoRAs, so they carry no
# `test -f` guard, the way `parity` and `real-parity` already fail hard when
# their files are missing. `make test` stays green on a bare machine; this
# target declares it wants the files. See README, "LoRA test corpus".
LORA_TURBO := loras/turbo.safetensors
LORA_COMBAT := loras/combat.safetensors
# The smallest geometry that actually generates: 22 frames, 4 evaluations.
# 5 frames is a legal canvas but not a legal run - the video VAE decoder wants
# one trained 22-frame chunk and refuses below that.
LORA_GEOMETRY := --width 448 --height 576 --frames 22 --steps 4 --reuse 1 \
	--ssd-streaming --seed 4242
LORA_PROMPT := a candle flame bends in a draft, macro shot, the wick crackles

real-lora: h3_lora_oracle_test h3
	./h3_lora_oracle_test $(LORA_TURBO) MiniMax-H3 100 0
	./h3_lora_oracle_test $(LORA_COMBAT) MiniMax-H3 100 0
	./h3_lora_oracle_test $(LORA_TURBO) MiniMax-H3 100 0 $(LORA_COMBAT)
	@# T1's significance guard, from the outside: at a strength where the
	@# delta sinks under the BF16 noise floor the test has to FAIL, or a
	@# weaker LoRA file would quietly turn T1 into a no-op.
	@set -e; \
	out=$$(mktemp /tmp/h3-t1-guard-XXXXXX); \
	if ./h3_lora_oracle_test $(LORA_TURBO) MiniMax-H3 1 0 > $$out 2>&1; then \
		echo "FAIL: T1 passed with a delta under the significance floor"; \
		rm -f $$out; exit 1; \
	elif ! grep -q "delta too weak to be significant" $$out; then \
		echo "FAIL: T1 at strength 1 failed for some other reason:"; \
		cat $$out; rm -f $$out; exit 1; \
	else \
		echo "ok: T1 fails when the delta is too weak to be significant"; \
		rm -f $$out; \
	fi
	$(MAKE) lora-identity
	$(MAKE) lora-hotswap

# T2: strength 0 is byte-identical to no LoRA, same seed and parameters.
lora-identity: h3
	@set -e; \
	dir=$$(mktemp -d /tmp/h3-t2-XXXXXX); \
	echo "T2: two runs into $$dir"; \
	./h3 -d MiniMax-H3 -p "$(LORA_PROMPT)" $(LORA_GEOMETRY) \
		-o $$dir/plain.mp4 > $$dir/plain.log 2>&1; \
	./h3 -d MiniMax-H3 -p "$(LORA_PROMPT)" $(LORA_GEOMETRY) \
		--lora $(LORA_TURBO):0 -o $$dir/zero.mp4 > $$dir/zero.log 2>&1; \
	if cmp $$dir/plain.mp4 $$dir/zero.mp4; then \
		echo "ok: T2 --lora at strength 0 is byte-identical to no LoRA"; \
		rm -rf $$dir; \
	else \
		echo "FAIL: T2 strength 0 changed the video; logs in $$dir"; \
		exit 1; \
	fi

# T6: one session, one seed. Generate, add a LoRA, regenerate, remove it,
# regenerate. Two assertions, because the first alone is green even when the
# delta is never applied: three runs of the base model agree with each other.
lora-hotswap: h3
	@set -e; \
	dir=$$(mktemp -d /tmp/h3-t6-XXXXXX); \
	echo "T6: one session, three generations into $$dir"; \
	printf '!output %s\n%s\n!lora add %s 0.7\n!again\n!lora remove %s\n!again\n!quit\n' \
		$$dir "$(LORA_PROMPT)" $(LORA_TURBO) $(LORA_TURBO) | \
		./h3 -d MiniMax-H3 $(LORA_GEOMETRY) > $$dir/session.log 2>&1; \
	test -f $$dir/video-0003.mp4 || \
		{ echo "FAIL: T6 produced fewer than three videos; see $$dir/session.log"; \
		  exit 1; }; \
	if ! cmp $$dir/video-0001.mp4 $$dir/video-0003.mp4; then \
		echo "FAIL: T6 removing the LoRA did not restore the first video"; \
		exit 1; \
	fi; \
	if cmp -s $$dir/video-0001.mp4 $$dir/video-0002.mp4; then \
		echo "FAIL: T6 the LoRA changed nothing; the delta never landed"; \
		exit 1; \
	fi; \
	echo "ok: T6 hot-swap restores the first video and the LoRA changes it"; \
	rm -rf $$dir

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

# Vendored from Iris. Keep the main project strict without rewriting this small
# terminal editor for conversion diagnostics unrelated to H3.
linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3 h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_av_mux_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_lora_oracle_test h3_lora_gemm_bench h3_lora_fixture_tests \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 \
	libh3.a *.o *.d tests/*.o tests/*.d
