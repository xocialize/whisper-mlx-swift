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

    /// Whisper `ApplyTimestampRules` (port of mlx-whisper) → next token id, given the raw
    /// last-position logits and the generated-so-far ids. Done in Swift floats (no MLX indexing).
    static func nextTokenWithTimestampRules(
        _ logits: MLXArray, generated: [Int], timestampBegin: Int, eot: Int, noTimestamps: Int,
        maxInitialTimestampIndex: Int = 50
    ) -> Int {
        let raw = logits.asArray(Float.self)
        let vocab = raw.count
        var lg = raw
        lg[noTimestamps] = -.infinity

        let lastTs = (generated.last ?? -1) >= timestampBegin
        let penultTs = generated.count < 2 || generated[generated.count - 2] >= timestampBegin
        if lastTs {
            if penultTs { for i in timestampBegin ..< vocab { lg[i] = -.infinity } }   // force text
            else { for i in 0 ..< eot { lg[i] = -.infinity } }                          // force timestamp
        }
        if generated.isEmpty {  // first generated token: force an initial timestamp
            for i in 0 ..< timestampBegin { lg[i] = -.infinity }
            let lastAllowed = timestampBegin + maxInitialTimestampIndex
            if lastAllowed + 1 < vocab { for i in (lastAllowed + 1) ..< vocab { lg[i] = -.infinity } }
        }

        // "timestamps more likely than any text token" rule (logprobs from RAW logits).
        let m = raw.max() ?? 0
        let lse = m + Foundation.log(raw.reduce(Float(0)) { $0 + Foundation.exp($1 - m) })
        let ts = raw[timestampBegin...]
        let tm = ts.max() ?? -Float.infinity
        let tsLogprob = (tm + Foundation.log(ts.reduce(Float(0)) { $0 + Foundation.exp($1 - tm) })) - lse
        let maxTextLogprob = (raw[0 ..< timestampBegin].max() ?? -Float.infinity) - lse
        if tsLogprob > maxTextLogprob { for i in 0 ..< timestampBegin { lg[i] = -.infinity } }

        return lg.indices.max(by: { lg[$0] < lg[$1] })!
    }

    /// KV-cached greedy decode WITH timestamps. Returns generated ids (incl. timestamp tokens).
    public func greedyDecodeTimestamped(
        mel: MLXArray, initialTokens: [Int], eot: Int, timestampBegin: Int, noTimestamps: Int,
        maxNewTokens: Int = 224
    ) -> [Int] {
        let audio = embedAudio(mel)
        eval(audio)
        let cache = WhisperKVCache(layers: dims.nTextLayer)
        var generated: [Int] = []

        func step(_ ids: [Int]) -> MLXArray {
            let t = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
            let lg = decoder(t, xa: audio, cache: cache)
            return lg[0, lg.dim(1) - 1]
        }
        var logits = step(initialTokens)
        for _ in 0 ..< maxNewTokens {
            let next = Self.nextTokenWithTimestampRules(
                logits, generated: generated, timestampBegin: timestampBegin, eot: eot, noTimestamps: noTimestamps)
            if next == eot { break }
            generated.append(next)
            logits = step([next])
        }
        return generated
    }
}
