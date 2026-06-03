import Foundation
import Tokenizers

/// Thin wrapper over the HF Whisper tokenizer (swift-transformers) — special-token IDs
/// for building the SOT sequence, and detokenization of generated ids to text.
public struct WhisperTokenizer {
    let tok: Tokenizer
    public let sot: Int
    public let eot: Int
    public let transcribe: Int
    public let translate: Int
    public let noTimestamps: Int
    public let timestampBegin: Int  // id of <|0.00|>

    public static func from(repo: String = "openai/whisper-large-v3") async throws -> WhisperTokenizer {
        let t = try await AutoTokenizer.from(pretrained: repo)
        // Special-token IDs via the tokenizer, with large-v3 fallbacks (stable, 100-language).
        func id(_ s: String, _ fallback: Int) -> Int { t.convertTokenToId(s) ?? fallback }
        return WhisperTokenizer(
            tok: t,
            sot: id("<|startoftranscript|>", 50258),
            eot: id("<|endoftext|>", 50257),
            transcribe: id("<|transcribe|>", 50360),
            translate: id("<|translate|>", 50359),
            noTimestamps: id("<|notimestamps|>", 50364),
            timestampBegin: id("<|0.00|>", 50365)
        )
    }

    /// `<|en|>` etc. (defaults to English's large-v3 id if missing).
    public func languageToken(_ code: String) -> Int {
        tok.convertTokenToId("<|\(code)|>") ?? 50259
    }

    /// SOT sequence the decoder is primed with. With `timestamps: false` (default), appends
    /// <|notimestamps|>; with `true`, omits it so the timestamp rules can emit timestamp tokens.
    public func initialTokens(language: String, translate doTranslate: Bool = false, timestamps: Bool = false) -> [Int] {
        var seq = [sot, languageToken(language), doTranslate ? translate : transcribe]
        if !timestamps { seq.append(noTimestamps) }
        return seq
    }

    public func decode(_ ids: [Int]) -> String {
        tok.decode(tokens: ids, skipSpecialTokens: true)
    }
}
