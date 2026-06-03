import Foundation
import MLX

/// End-to-end Whisper pipeline: audio → log-mel → encode → greedy decode → text.
public struct WhisperPipeline {
    public let model: Whisper
    public let tokenizer: WhisperTokenizer

    public init(model: Whisper, tokenizer: WhisperTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Load model from a local dir (config.json + weights.safetensors) + the HF Whisper tokenizer.
    public static func fromDirectory(
        _ dir: URL, tokenizerRepo: String = "openai/whisper-large-v3"
    ) async throws -> WhisperPipeline {
        let tokenizer = try await WhisperTokenizer.from(repo: tokenizerRepo)
        let model = try WhisperLoader.fromDirectory(dir)
        return WhisperPipeline(model: model, tokenizer: tokenizer)
    }

    /// log-mel `(1, nFrames, nMels)` → transcript text.
    public func transcribe(mel: MLXArray, language: String = "en", translate: Bool = false) -> String {
        let initial = tokenizer.initialTokens(language: language, translate: translate)
        let gen = model.greedyDecode(mel: mel, initialTokens: initial, eot: tokenizer.eot)
        return tokenizer.decode(gen).trimmingCharacters(in: .whitespaces)
    }

    /// 1-D 16 kHz waveform → transcript text.
    public func transcribe(audio: MLXArray, language: String = "en", translate: Bool = false) -> String {
        transcribe(mel: WhisperAudio.mel(forAudio: audio), language: language, translate: translate)
    }
}
