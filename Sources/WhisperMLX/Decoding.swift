import Foundation
import MLX

extension Whisper {
    /// Greedy autoregressive decode. Returns the generated token ids (excluding the initial
    /// SOT sequence), stopping at `eot`. No KV cache yet — recomputes the full prefix each
    /// step (matches the parity golden; O(n²), fine for short clips). Add a cache for speed later.
    public func greedyDecode(mel: MLXArray, initialTokens: [Int], eot: Int, maxNewTokens: Int = 224) -> [Int] {
        let audio = embedAudio(mel)
        eval(audio)
        var tokens = initialTokens
        for _ in 0 ..< maxNewTokens {
            let t = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            let lg = logits(t, audioFeatures: audio)
            let next = argMax(lg[0, lg.dim(1) - 1], axis: -1).item(Int.self)
            if next == eot { break }
            tokens.append(next)
        }
        return Array(tokens.dropFirst(initialTokens.count))
    }

    /// KV-cached greedy decode — O(n): prefill the SOT sequence, then feed one token per step.
    /// Cross-attention K/V (over the 1500 audio frames) are computed once and reused.
    public func greedyDecodeCached(mel: MLXArray, initialTokens: [Int], eot: Int, maxNewTokens: Int = 224) -> [Int] {
        let audio = embedAudio(mel)
        eval(audio)
        let cache = WhisperKVCache(layers: dims.nTextLayer)
        var tokens = initialTokens

        var t = MLXArray(initialTokens.map { Int32($0) }).reshaped([1, initialTokens.count])
        var lg = decoder(t, xa: audio, cache: cache)
        var next = argMax(lg[0, lg.dim(1) - 1], axis: -1).item(Int.self)

        for _ in 0 ..< maxNewTokens {
            if next == eot { break }
            tokens.append(next)
            t = MLXArray([Int32(next)]).reshaped([1, 1])
            lg = decoder(t, xa: audio, cache: cache)
            next = argMax(lg[0, 0], axis: -1).item(Int.self)
        }
        return Array(tokens.dropFirst(initialTokens.count))
    }
}
