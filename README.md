# whisper-mlx-swift

**MLX-Swift** port of OpenAI Whisper (encoder-decoder ASR) for on-device Apple Silicon.
Net-new: there was no MLX-Swift Whisper ([WhisperKit](https://github.com/argmaxinc/WhisperKit)
is CoreML, not MLX). Companion to the Python [`tts-validation`](https://github.com/xocialize/tts-validation)
harness — brings circular TTS validation (transcribe synthesized audio, score content)
to the RosettaCast Swift stack.

> **Status:** Working end-to-end STT, every stage parity-verified vs Python `mlx-whisper`
> (CPU, true fp32): model bitwise-exact (encoder rel 0.000, logits rel 9.9e-5), mel rel 1.2e-7,
> greedy/cached/timestamped decode token-exact.
> Target: `mlx-community/whisper-large-v3-turbo` (multilingual incl. JP/ZH).

## Platforms / dependencies

macOS 14+, iOS 17+, visionOS 1+ (Swift tools 5.9). Depends on `mlx-swift` (≥0.21.0),
`swift-transformers` (≥1.1.6, HF tokenizer), and `swift-argument-parser` (≥1.3.0).

## Layout

```
Sources/WhisperMLX/
  Config.swift / Whisper.swift   ModelDimensions; AudioEncoder + TextDecoder (+ KV cache)
                                 (+ encoderHiddenStates for MuseTalk-style audio conditioning)
  Audio.swift / AudioFile.swift  log-mel frontend; AVFoundation wav loader (→16k mono)
  Decoding.swift                 greedy + KV-cached + timestamped decode (ApplyTimestampRules)
  Tokenizer.swift / Transcribe.swift   HF tokenizer; WhisperPipeline (audio→text / segments)
  Load.swift / Download.swift    local-dir load; HF auto-download (URLSession, cached)
Sources/whisper-cli/WhisperCLI.swift   CLI (transcribe + parity modes)
scripts/                         build-gpu.sh; capture_parity.py / capture_musetalk_audio_golden.py
goldens/                         parity fixtures (mel / decode / encoder / musetalk)
```

> There is no SPM test target — parity is exercised through the `whisper-cli` `--*-golden`
> modes (see below), not `swift test`.

## Usage

```bash
swift build -c release
# transcribe (auto-downloads weights + tokenizer; CPU, no metallib needed):
.build/release/whisper-cli --model-repo mlx-community/whisper-large-v3-turbo \
    --audio clip.wav [--language ja] [--timestamps]
```

CLI options: `--model-dir` (local config.json + weights) or `--model-repo` (HF auto-download);
`--audio` (end-to-end transcribe; needs `--model-dir`/`--model-repo`); `--language`; `--timestamps`;
`--gpu`. Parity/golden modes: `--golden`, `--mel-golden`, `--decode-golden`, `--decode-ts-golden`,
`--musetalk-audio-golden`, `--transcribe-golden`.

```swift
let p = try await WhisperPipeline.fromPretrained()       // weights + tokenizer
let text = p.transcribe(audio: wav16kMono, language: "en")
let segs = p.transcribeSegments(audio: wav16kMono)        // [Segment] (start, end, text)
```

**GPU (`--gpu`):** `swift build` doesn't bundle the MLX metallib (SPM-CLI gap). Build via
`scripts/build-gpu.sh` (xcodebuild compiles + bundles it into `.xcdd/Build/Products/Release/`),
then run `.xcdd/Build/Products/Release/whisper-cli ... --gpu`. CPU runs need none.
See CLAUDE.md / the `mlx-swift` skill.

License: MIT (matches Whisper).
