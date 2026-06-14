import Foundation
import MLX
import MLXNN

/// Sinusoidal position embedding (Whisper encoder). Mirrors reference `sinusoids`.
func sinusoids(length: Int, channels: Int, maxTimescale: Float = 10000) -> MLXArray {
    precondition(channels % 2 == 0)
    let half = channels / 2
    let logInc = Foundation.log(maxTimescale) / Float(half - 1)
    let invTimescales = MLX.exp(MLXArray(0 ..< half).asType(.float32) * (-logInc))
    let scaledTime = MLXArray(0 ..< length).asType(.float32).reshaped([length, 1])
        * invTimescales.reshaped([1, half])
    return concatenated([MLX.sin(scaledTime), MLX.cos(scaledTime)], axis: 1)
}

/// Incremental decode cache. One `Entry` per attention: self-attn grows each step;
/// cross-attn is computed once from the (fixed) encoder output and reused.
final class WhisperKVCache {
    final class Entry {
        var k: MLXArray?
        var v: MLXArray?
    }
    let selfAttn: [Entry]
    let crossAttn: [Entry]
    init(layers: Int) {
        selfAttn = (0 ..< layers).map { _ in Entry() }
        crossAttn = (0 ..< layers).map { _ in Entry() }
    }
}

/// Whisper multi-head attention. `key` has no bias (reference detail). Scales q and k
/// by headDim^-0.25 each (Whisper convention).
final class MultiHeadAttention: Module {
    let nHead: Int
    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var out: Linear

    init(_ nState: Int, _ nHead: Int) {
        self.nHead = nHead
        self.query = Linear(nState, nState, bias: true)
        self.key = Linear(nState, nState, bias: false)
        self.value = Linear(nState, nState, bias: true)
        self.out = Linear(nState, nState, bias: true)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, cache: WhisperKVCache.Entry? = nil) -> MLXArray {
        let q = query(x)
        var k: MLXArray
        var v: MLXArray
        if xa == nil {
            // Self-attention: append new K/V to the (growing) cache.
            k = key(x); v = value(x)
            if let ck = cache?.k, let cv = cache?.v {
                k = concatenated([ck, k], axis: 1)
                v = concatenated([cv, v], axis: 1)
            }
            cache?.k = k; cache?.v = v
        } else if let ck = cache?.k, let cv = cache?.v {
            // Cross-attention, already cached (encoder output is fixed).
            k = ck; v = cv
        } else {
            k = key(xa!); v = value(xa!)
            cache?.k = k; cache?.v = v
        }
        return out(qkvAttention(q, k, v, mask: mask))
    }

    private func qkvAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = q.dim(0), L = q.dim(1), D = q.dim(2)
        let Lk = k.dim(1)
        let headDim = D / nHead
        let scale = Foundation.pow(Float(headDim), -0.25)

        let qh = (q.reshaped([B, L, nHead, headDim]).transposed(0, 2, 1, 3)) * scale
        let kh = (k.reshaped([B, Lk, nHead, headDim]).transposed(0, 2, 3, 1)) * scale  // (B,H,hd,Lk)
        let vh = v.reshaped([B, Lk, nHead, headDim]).transposed(0, 2, 1, 3)

        var qk = qh.matmul(kh)  // (B,H,L,Lk)
        if let mask { qk = qk + mask[0 ..< L, 0 ..< L] }
        let w = softmax(qk.asType(.float32), axis: -1).asType(q.dtype)
        let o = w.matmul(vh).transposed(0, 2, 1, 3).reshaped([B, L, D])
        return o
    }
}

final class ResidualAttentionBlock: Module {
    @ModuleInfo var attn: MultiHeadAttention
    @ModuleInfo var attn_ln: LayerNorm
    @ModuleInfo var cross_attn: MultiHeadAttention?
    @ModuleInfo var cross_attn_ln: LayerNorm?
    @ModuleInfo var mlp1: Linear
    @ModuleInfo var mlp2: Linear
    @ModuleInfo var mlp_ln: LayerNorm

    init(_ nState: Int, _ nHead: Int, crossAttention: Bool) {
        self.attn = MultiHeadAttention(nState, nHead)
        self.attn_ln = LayerNorm(dimensions: nState)
        self.cross_attn = crossAttention ? MultiHeadAttention(nState, nHead) : nil
        self.cross_attn_ln = crossAttention ? LayerNorm(dimensions: nState) : nil
        self.mlp1 = Linear(nState, nState * 4, bias: true)
        self.mlp2 = Linear(nState * 4, nState, bias: true)
        self.mlp_ln = LayerNorm(dimensions: nState)
    }

    func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil,
                        selfCache: WhisperKVCache.Entry? = nil, crossCache: WhisperKVCache.Entry? = nil) -> MLXArray {
        var x = x + attn(attn_ln(x), mask: mask, cache: selfCache)
        if let cross_attn, let cross_attn_ln {
            x = x + cross_attn(cross_attn_ln(x), xa: xa, cache: crossCache)
        }
        x = x + mlp2(gelu(mlp1(mlp_ln(x))))
        return x
    }
}

final class AudioEncoder: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var blocks: [ResidualAttentionBlock]
    @ModuleInfo var ln_post: LayerNorm

    init(_ d: ModelDimensions) {
        self.conv1 = Conv1d(inputChannels: d.nMels, outputChannels: d.nAudioState, kernelSize: 3, padding: 1)
        self.conv2 = Conv1d(inputChannels: d.nAudioState, outputChannels: d.nAudioState, kernelSize: 3, stride: 2, padding: 1)
        self.blocks = (0 ..< d.nAudioLayer).map { _ in ResidualAttentionBlock(d.nAudioState, d.nAudioHead, crossAttention: false) }
        self.ln_post = LayerNorm(dimensions: d.nAudioState)
    }

    /// mel: (B, T, nMels) channels-last → audio features (B, nAudioCtx, nAudioState).
    /// Sinusoidal position embedding is a constant (not a checkpoint weight) → computed here.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = gelu(conv1(mel))
        x = gelu(conv2(x))
        x = x + sinusoids(length: x.dim(1), channels: x.dim(2)).asType(x.dtype)
        for b in blocks { x = b(x) }
        return ln_post(x)
    }

    /// Stacked encoder hidden states — MuseTalk's audio conditioning (HF
    /// `output_hidden_states=True`): the input to each block (pre-layer) plus
    /// `ln_post(final)`, stacked on a new axis → (B, nAudioCtx, nAudioLayer+1, nAudioState).
    /// Mirrors the Python `musetalk_mlx.whisper.WhisperEncoder` collection.
    ///
    /// `positionEmbedding`: HF ships a *learned* `embed_positions.weight` that differs from the
    /// recomputed OpenAI sinusoids by ~2.4e-4; pass it to match the oracle exactly. When nil,
    /// falls back to computed sinusoids (the mlx-community STT path, which ships no pos-emb).
    func hiddenStates(_ mel: MLXArray, positionEmbedding: MLXArray? = nil) -> MLXArray {
        var x = gelu(conv1(mel))
        x = gelu(conv2(x))
        let seq = x.dim(1)
        let pe = positionEmbedding.map { $0[0 ..< seq] } ?? sinusoids(length: seq, channels: x.dim(2))
        x = x + pe.asType(x.dtype)
        var states: [MLXArray] = []
        for b in blocks {
            states.append(x)        // input to this block (HF collects pre-layer)
            x = b(x)
        }
        states.append(ln_post(x))   // final = layer_norm(last)
        return stacked(states, axis: 2)
    }
}

final class TextDecoder: Module {
    @ModuleInfo var token_embedding: Embedding
    @ModuleInfo var blocks: [ResidualAttentionBlock]
    @ModuleInfo var ln: LayerNorm
    var positional_embedding: MLXArray  // learned (checkpoint weight)

    init(_ d: ModelDimensions) {
        self.token_embedding = Embedding(embeddingCount: d.nVocab, dimensions: d.nTextState)
        self.blocks = (0 ..< d.nTextLayer).map { _ in ResidualAttentionBlock(d.nTextState, d.nTextHead, crossAttention: true) }
        self.ln = LayerNorm(dimensions: d.nTextState)
        self.positional_embedding = MLXArray.zeros([d.nTextCtx, d.nTextState])
    }

    /// tokens: (B, L) Int32 ; xa: encoder audio features. Returns logits (B, L, nVocab).
    /// With `cache`, only the new `tokens` are processed; positions start at the cache length.
    func callAsFunction(_ tokens: MLXArray, xa: MLXArray, cache: WhisperKVCache? = nil) -> MLXArray {
        let L = tokens.dim(-1)
        let offset = cache?.selfAttn.first?.k?.dim(1) ?? 0
        var x = token_embedding(tokens) + positional_embedding[offset ..< offset + L]
        let mask = Self.causalMask(L)  // (L,L); broadcasts over cached keys for single-step decode
        for (i, b) in blocks.enumerated() {
            x = b(x, xa: xa, mask: mask, selfCache: cache?.selfAttn[i], crossCache: cache?.crossAttn[i])
        }
        x = ln(x)
        return token_embedding.asLinear(x)
    }

    /// Additive causal mask (upper triangle -inf). Constant → computed, not a parameter.
    static func causalMask(_ n: Int) -> MLXArray {
        let r = MLXArray(0 ..< n).reshaped([n, 1])
        let c = MLXArray(0 ..< n).reshaped([1, n])
        return MLX.which(c .<= r, MLXArray(Float(0)), MLXArray(-Float.infinity))
    }
}

public final class Whisper: Module {
    @ModuleInfo var encoder: AudioEncoder
    @ModuleInfo var decoder: TextDecoder
    public let dims: ModelDimensions

    public init(_ dims: ModelDimensions) {
        self.dims = dims
        self.encoder = AudioEncoder(dims)
        self.decoder = TextDecoder(dims)
    }

    public func embedAudio(_ mel: MLXArray) -> MLXArray { encoder(mel) }

    /// Stacked encoder hidden states for MuseTalk-style audio conditioning.
    /// mel: (B, nAudioCtx, nMels) channels-last → (B, nAudioCtx, nAudioLayer+1, nAudioState).
    /// Pass the checkpoint's `embed_positions.weight` to match HF exactly (see `hiddenStates`).
    public func encoderHiddenStates(_ mel: MLXArray, positionEmbedding: MLXArray? = nil) -> MLXArray {
        encoder.hiddenStates(mel, positionEmbedding: positionEmbedding)
    }

    /// Decoder logits given precomputed audio features (for stage-isolated parity).
    public func logits(_ tokens: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        decoder(tokens, xa: audioFeatures)
    }

    public func callAsFunction(_ mel: MLXArray, _ tokens: MLXArray) -> MLXArray {
        decoder(tokens, xa: encoder(mel))
    }
}
