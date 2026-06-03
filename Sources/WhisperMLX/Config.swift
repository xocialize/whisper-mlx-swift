import Foundation

/// Whisper model dimensions (from the mlx-community config.json). Mirrors the
/// Python reference `ModelDimensions`.
public struct ModelDimensions: Codable, Sendable {
    public var nMels: Int
    public var nAudioCtx: Int
    public var nAudioState: Int
    public var nAudioHead: Int
    public var nAudioLayer: Int
    public var nVocab: Int
    public var nTextCtx: Int
    public var nTextState: Int
    public var nTextHead: Int
    public var nTextLayer: Int

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
    }
}
