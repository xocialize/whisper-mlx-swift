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
- ✅ Scaffold + model (Whisper.swift: attention/blocks/encoder/decoder) + dir loader + CLI smoke.
- ▶ Next: mel frontend (log-mel 128 via MLXFFT STFT + baked filters), Whisper tokenizer
  (swift-transformers), decoding loop (lang/task tokens, greedy), HF auto-download,
  parity vs Python mlx-whisper, then a transcribe() API for tts-validation parity.

## Build / run
`swift build`  ·  run smoke: `swift run whisper-cli <dir-with-config+weights>`
(if metallib error on run, build/run via xcodebuild against a generated scheme).
