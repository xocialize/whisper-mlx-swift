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

## Status — working end-to-end STT
Every stage is parity-verified vs Python mlx-whisper (on CPU, true fp32):
- Model (encoder+decoder): encoder rel 0.000 (bitwise exact), logits rel 9.9e-5, argmax match.
- Mel frontend (`Audio.swift`, log-mel 128): rel 1.2e-7.
- Greedy decode + KV-cached decode: token-exact. Timestamps (`ApplyTimestampRules` port): token-exact.
- Tokenizer (`Tokenizer.swift`, swift-transformers) + `WhisperPipeline` (`Transcribe.swift`) + audio
  loader (`AudioFile.swift`, AVFoundation) + HF auto-download (`Download.swift`, URLSession-cached).
Goldens via `scripts/capture_*.py`. Transcribes a real wav → correct text + `[start → end]` segments.

## Parity / precision (important)
- Run correctness parity on **CPU** (`Device.withDefaultDevice(Device(.cpu))`, the CLI default;
  `--gpu` to opt in). MLX **GPU fp32 matmul is tf32-like** and accumulates over depth — encoder
  rel 0.31 over 32 layers on GPU (NOT a bug; same trap as the Zonos port). CPU is bitwise fp32.
- Constants that are NOT checkpoint weights (encoder sinusoids, causal mask) must not be tracked
  parameters → compute them on the fly. `attn.key` has no bias. (See the `mlx-swift` skill's porting ref.)

## Build / run
- **CPU (default; no metallib needed):** `swift build -c release` then
  `whisper-cli --model-repo mlx-community/whisper-large-v3-turbo --audio clip.wav [--timestamps] [--language ja]`
  (or `--model-dir <local>`). Parity modes: `--golden`/`--mel-golden`/`--decode-golden`/`--decode-ts-golden`.
- **GPU (`--gpu`):** `swift build` does NOT bundle the MLX metallib (SPM-CLI gap → "Failed to load the
  default metallib"). Build via xcodebuild instead: `scripts/build-gpu.sh` (compiles + bundles the
  metallib), then run `.xcdd/Build/Products/Release/whisper-cli ... --gpu`. NOTE: the CLI entry file must
  NOT be named `main.swift` (top-level-code vs `@main` conflict under xcodebuild) — it's `WhisperCLI.swift`.
  Real Xcode app targets bundle the metallib automatically.
