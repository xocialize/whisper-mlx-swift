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
        let gen = model.greedyDecodeCached(mel: mel, initialTokens: initial, eot: tokenizer.eot)
        return tokenizer.decode(gen).trimmingCharacters(in: .whitespaces)
    }

    /// 1-D 16 kHz waveform → transcript text.
    public func transcribe(audio: MLXArray, language: String = "en", translate: Bool = false) -> String {
        transcribe(mel: WhisperAudio.mel(forAudio: audio), language: language, translate: translate)
    }

    /// A timestamped segment of speech.
    public struct Segment: Equatable {
        public let start: Double
        public let end: Double
        public let text: String
    }

    /// Transcribe with segment timestamps (uses Whisper's timestamp rules).
    public func transcribeSegments(mel: MLXArray, language: String = "en") -> [Segment] {
        let initial = tokenizer.initialTokens(language: language, timestamps: true)
        let gen = model.greedyDecodeTimestamped(
            mel: mel, initialTokens: initial, eot: tokenizer.eot,
            timestampBegin: tokenizer.timestampBegin, noTimestamps: tokenizer.noTimestamps)

        var segments: [Segment] = []
        var start: Double?
        var text: [Int] = []
        for t in gen {
            if t >= tokenizer.timestampBegin {
                let time = Double(t - tokenizer.timestampBegin) * 0.02
                if let s = start {
                    segments.append(Segment(start: s, end: time, text: tokenizer.decode(text).trimmingCharacters(in: .whitespaces)))
                    start = nil; text = []
                } else {
                    start = time
                }
            } else {
                text.append(t)
            }
        }
        return segments
    }

    public func transcribeSegments(audio: MLXArray, language: String = "en") -> [Segment] {
        transcribeSegments(mel: WhisperAudio.mel(forAudio: audio), language: language)
    }
}
