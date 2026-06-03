import Foundation
import MLX
import MLXFFT

/// Whisper log-mel frontend (mirrors mlx-whisper `audio.log_mel_spectrogram`).
/// n_fft=400, hop=160, 128 mels, 16 kHz. The mel filterbank + hann window are baked
/// resources (parity-verified to ~1e-6) — see references porting rule "bake fixed transforms".
public enum WhisperAudio {
    public static let nFFT = 400
    public static let hop = 160
    public static let nSamples = 480_000     // 30 s @ 16 kHz
    public static let nMels = 128

    private struct Filters { let mel: MLXArray; let window: MLXArray }
    private static let filters: Filters = {
        guard let url = Bundle.module.url(forResource: "mel_filters_128", withExtension: "safetensors"),
              let w = try? loadArrays(url: url), let mel = w["mel_filters"], let window = w["window"]
        else { fatalError("missing mel_filters_128.safetensors resource") }
        return Filters(mel: mel, window: window)
    }()

    /// x[lo..<hi] reversed (MLX-Swift has no `[::-1]`; build reversed indices and gather).
    private static func reversedSlice(_ x: MLXArray, _ lo: Int, _ hi: Int) -> MLXArray {
        let idx = MLXArray(Array(lo ..< hi).reversed().map { Int32($0) })
        return take(x, idx, axis: 0)
    }

    /// Pad/trim a 1-D 16 kHz waveform to exactly `nSamples`.
    public static func padOrTrim(_ audio: MLXArray) -> MLXArray {
        let n = audio.dim(0)
        if n > nSamples { return audio[0 ..< nSamples] }
        if n < nSamples { return padded(audio, widths: [IntOrPair((0, nSamples - n))]) }
        return audio
    }

    /// 1-D waveform (16 kHz) → log-mel `(nFrames, nMels)`. (nFrames = len/hop, 3000 for 30 s.)
    public static func logMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        // Reflect-pad by nFFT/2 (mx.pad has no reflect mode — replicate manually).
        let p = nFFT / 2
        let n = audio.dim(0)
        let x = concatenated([reversedSlice(audio, 1, p + 1), audio, reversedSlice(audio, n - p - 1, n - 1)], axis: 0)

        // Frame: shape [t, nFFT], strides [hop, 1]; t = (size - nFFT + hop)/hop.
        let t = (x.dim(0) - nFFT + hop) / hop
        let frames = asStrided(x, [t, nFFT], strides: [hop, 1])

        let freqs = MLXFFT.rfft(frames * filters.window)        // (t, 201) complex
        let magnitudes = abs(freqs[0 ..< (t - 1)]).square()     // drop last frame → (t-1, 201)
        let melSpec = magnitudes.matmul(filters.mel.transposed())  // (t-1, nMels)

        var logSpec = MLX.log(maximum(melSpec, 1e-10)) * 0.4342944819  // log10
        logSpec = maximum(logSpec, logSpec.max() - 8.0)
        return (logSpec + 4.0) / 4.0
    }

    /// Convenience: waveform → encoder-ready mel `(1, nFrames, nMels)` (NLC, padded to 30 s).
    public static func mel(forAudio audio: MLXArray) -> MLXArray {
        logMelSpectrogram(padOrTrim(audio)).expandedDimensions(axis: 0)
    }
}
