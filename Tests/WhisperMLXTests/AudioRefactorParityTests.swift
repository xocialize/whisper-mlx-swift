// AudioRefactorParityTests.swift — gate for the MLXAudioDSP extraction.
//
// `WhisperAudio.logMelSpectrogram` was refactored onto the shared mlx-audio-dsp primitives.
// This test carries an INLINE COPY of the pre-refactor implementation (verbatim ops) and
// asserts the refactored path is BIT-IDENTICAL on real-shaped input, so the extraction can
// never silently change Whisper's front-end. Also cross-checks the leaf's hann generator
// against the baked (parity-verified) window resource.

import XCTest
import MLX
import MLXFFT
import MLXAudioDSP
@testable import WhisperMLX

final class AudioRefactorParityTests: XCTestCase {

    override func invokeTest() { withMLXCPU { super.invokeTest() } }

    /// Verbatim copy of the pre-extraction implementation (do not "improve").
    private func legacyLogMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        func reversedSlice(_ x: MLXArray, _ lo: Int, _ hi: Int) -> MLXArray {
            let idx = MLXArray(Array(lo ..< hi).reversed().map { Int32($0) })
            return take(x, idx, axis: 0)
        }
        let nFFT = 400, hop = 160
        let p = nFFT / 2
        let n = audio.dim(0)
        let x = concatenated(
            [reversedSlice(audio, 1, p + 1), audio, reversedSlice(audio, n - p - 1, n - 1)],
            axis: 0)
        let t = (x.dim(0) - nFFT + hop) / hop
        let frames = asStrided(x, [t, nFFT], strides: [hop, 1])
        let freqs = MLXFFT.rfft(frames * WhisperAudio.filters.window)
        let magnitudes = abs(freqs[0 ..< (t - 1)]).square()
        let melSpec = magnitudes.matmul(WhisperAudio.filters.mel.transposed())
        var logSpec = MLX.log(maximum(melSpec, 1e-10)) * 0.4342944819
        logSpec = maximum(logSpec, logSpec.max() - 8.0)
        return (logSpec + 4.0) / 4.0
    }

    /// Deterministic 2-second test signal (mixed sines + a click; exercises real dynamics).
    private func testSignal(_ n: Int = 32_000) -> MLXArray {
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let time = Double(i) / 16_000.0
            samples[i] = Float(
                0.5 * sin(2 * .pi * 220 * time)
                + 0.3 * sin(2 * .pi * 1_237 * time)
                + 0.1 * sin(2 * .pi * 5_501 * time))
        }
        samples[12_345] = 0.9  // transient
        return MLXArray(samples)
    }

    func testAudioRefactorBitIdentical() {
        let audio = testSignal()
        let legacy = legacyLogMelSpectrogram(audio)
        let refactored = WhisperAudio.logMelSpectrogram(audio)
        eval(legacy, refactored)

        XCTAssertEqual(legacy.shape, refactored.shape)
        let maxAbs = MLX.abs(legacy - refactored).max().item(Float.self)
        XCTAssertEqual(maxAbs, 0, "refactor must be bit-identical; max_abs=\(maxAbs)")
    }

    func testLeafHannMatchesBakedWindow() {
        // The baked window resource is torch.hann_window(400) (periodic).
        let generated = AudioDSP.hannWindow(400, periodic: true)
        let baked = WhisperAudio.filters.window
        eval(generated, baked)
        let maxAbs = MLX.abs(generated - baked.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxAbs, 1e-6)
    }

    func testPadOrTrimUnchanged() {
        let short = MLXArray([Float](repeating: 0.25, count: 1_000))
        XCTAssertEqual(WhisperAudio.padOrTrim(short).dim(0), WhisperAudio.nSamples)
        let long = MLXArray([Float](repeating: 0.25, count: WhisperAudio.nSamples + 5))
        XCTAssertEqual(WhisperAudio.padOrTrim(long).dim(0), WhisperAudio.nSamples)
    }
}
