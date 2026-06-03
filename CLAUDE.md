# CLAUDE.md — whisper-mlx-swift

MLX-Swift port of OpenAI Whisper (encoder-decoder ASR). Net-new: no MLX-Swift Whisper
existed (WhisperKit is CoreML, not MLX). Driven by the `mlx-swift` skill. Companion to
the Python `tts-validation` harness — gives on-device circular TTS validation for the
RosettaCast Swift stack.

## Ground rules
- **Port = transpose, not redesign.** Mirror the mlx-examples/whisper Python reference
  (module/file names, call order) for 1:1 parity diffing. Validate numerically against
  Python `mlx-whisper` (same MLX weights) — encoder output + decoder logits.
- Build with `swift build` to compile; **`xcodebuild` for running/tests** (SPM-CLI hits
  "Failed to load the default metallib" — see mlx-swift skill).
- MLXArray is NOT Sendable; keep arrays within one actor/task.

## Target checkpoint
`mlx-community/whisper-large-v3-turbo` (multilingual incl JP/ZH, fast: 4 decoder layers).
config: nMels 128, audio 1280/32L/20H, text 1280/4L/20H, vocab 51866. Conv weights are
already MLX layout (O,K,I) in the mlx-community safetensors → load direct, no transpose.
`attn.key` has NO bias. Encoder pos-emb = sinusoids (computed); decoder pos-emb learned.

## Status
- ✅ Model (Whisper.swift: attention/blocks/encoder/decoder) + dir loader + CLI, **parity-verified**.
  On **CPU** (true fp32): encoder rel=0.000 (bitwise exact), decoder logits rel 9.9e-5, next-token
  argmax matches Python mlx-whisper. Golden: `scripts/capture_parity.py`.
- ▶ Next (harness around the verified model): mel frontend (log-mel 128 via MLXFFT STFT + baked
  filters + manual reflect-pad), Whisper tokenizer (swift-transformers), autoregressive decoding
  loop (sot/lang/task/timestamp tokens, KV cache), HF auto-download, `transcribe()` API.

## Parity / precision (important)
- Run correctness parity on **CPU** (`Device.withDefaultDevice(Device(.cpu))`, the CLI default;
  `--gpu` to opt in). MLX **GPU fp32 matmul is tf32-like** and accumulates over depth — it gave
  encoder rel 0.31 over 32 layers (NOT a bug; same trap as the Zonos port). CPU is bitwise fp32.
- Constants that are NOT checkpoint weights (encoder sinusoids, causal mask) must not be tracked
  parameters → compute them on the fly (else `update()` expects/!misses them). `attn.key` has no bias.

## Build / run
`swift build` · CPU parity (no metallib needed): `whisper-cli <dir> --golden goldens/whisper_parity.safetensors`
- **metallib (GPU runs only):** `swift run` hits "Failed to load the default metallib". Workaround:
  copy a compatible `mlx.metallib` next to the binary (`.build/.../release/`). Proper fix = xcodebuild
  build-phase. CPU runs avoid this entirely. See the `mlx-swift` skill.
