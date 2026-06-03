# whisper-mlx-swift

**MLX-Swift** port of OpenAI Whisper (encoder-decoder ASR) for on-device Apple Silicon.
Net-new: there was no MLX-Swift Whisper ([WhisperKit](https://github.com/argmaxinc/WhisperKit)
is CoreML, not MLX). Companion to the Python [`tts-validation`](https://github.com/xocialize/tts-validation)
harness — brings circular TTS validation (transcribe synthesized audio, score content)
to the RosettaCast Swift stack.

> **Status:** Pre-alpha. Model (encoder + decoder) + weight loader + CLI smoke landed.
> Next: mel frontend, tokenizer (swift-transformers), decoding loop, parity vs Python
> `mlx-whisper`. Target: `mlx-community/whisper-large-v3-turbo` (multilingual incl. JP/ZH).

## Layout

```
Sources/WhisperMLX/
  Config.swift     ModelDimensions (config.json)
  Whisper.swift    AudioEncoder + TextDecoder + blocks + MultiHeadAttention (mirrors mlx-examples)
  Load.swift       fromDirectory(config.json + weights.safetensors)
Sources/whisper-cli/  smoke CLI (load + encoder/decoder shape check)
```

## Build

```bash
swift build                       # compile
swift run whisper-cli <model-dir> # smoke (dir with config.json + weights.safetensors)
```

Run/tests may need `xcodebuild` (SPM-CLI metallib quirk — see CLAUDE.md / mlx-swift skill).

License: MIT (matches Whisper).
