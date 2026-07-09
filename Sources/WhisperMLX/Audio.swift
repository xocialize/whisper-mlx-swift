import Foundation
import MLX
import MLXAudioDSP

/// Whisper log-mel frontend (mirrors mlx-whisper `audio.log_mel_spectrogram`).
/// n_fft=400, hop=160, 128 mels, 16 kHz. The mel filterbank + hann window are baked
/// resources (parity-verified to ~1e-6) — see references porting rule "bake fixed transforms".
///
/// The STFT/mel primitives were extracted into MLXAudioDSP (mlx-audio-dsp); this head keeps
/// only Whisper's specifics: 30-s pad/trim, drop-last-frame, and the log10 dynamics. The
/// refactor is bit-identical to the pre-extraction implementation (gated by
/// `WhisperMLXTests.testAudioRefactorBitIdentical`).
public enum WhisperAudio {
    public static let nFFT = 400
    public static let hop = 160
    public static let nSamples = 480_000     // 30 s @ 16 kHz
    public static let nMels = 128

    struct Filters { let mel: MLXArray; let window: MLXArray }
    static let filters: Filters = {
        guard let url = Bundle.module.url(forResource: "mel_filters_128", withExtension: "safetensors"),
              let w = try? loadArrays(url: url), let mel = w["mel_filters"], let window = w["window"]
        else { fatalError("missing mel_filters_128.safetensors resource") }
        return Filters(mel: mel, window: window)
    }()

    /// Pad/trim a 1-D 16 kHz waveform to exactly `nSamples`.
    public static func padOrTrim(_ audio: MLXArray) -> MLXArray {
        let n = audio.dim(0)
        if n > nSamples { return audio[0 ..< nSamples] }
        if n < nSamples { return padded(audio, widths: [IntOrPair((0, nSamples - n))]) }
        return audio
    }

    /// 1-D waveform (16 kHz) → log-mel `(nFrames, nMels)`. (nFrames = len/hop, 3000 for 30 s.)
    public static func logMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        let x = AudioDSP.reflectPadded(audio, pad: nFFT / 2)
        let frames = AudioDSP.framed(x, frameLength: nFFT, hop: hop)
        let t = frames.dim(0)

        let power = AudioDSP.powerSpectrum(frames, window: filters.window)   // (t, 201)
        let magnitudes = power[0 ..< (t - 1)]                                // drop last frame
        let melSpec = AudioDSP.applyMelFilterbank(magnitudes, filters: filters.mel)

        var logSpec = MLX.log(maximum(melSpec, 1e-10)) * 0.4342944819  // log10
        logSpec = maximum(logSpec, logSpec.max() - 8.0)
        return (logSpec + 4.0) / 4.0
    }

    /// Convenience: waveform → encoder-ready mel `(1, nFrames, nMels)` (NLC, padded to 30 s).
    public static func mel(forAudio audio: MLXArray) -> MLXArray {
        logMelSpectrogram(padOrTrim(audio)).expandedDimensions(axis: 0)
    }
}
