import AVFoundation
import Foundation
import MLX

public extension WhisperAudio {
    /// Read an audio file (any AVFoundation-supported format) → mono Float32 MLXArray at 16 kHz.
    static func load(_ url: URL, sampleRate: Double = 16000) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: outFormat)
        else { throw NSError(domain: "WhisperAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "audio format setup failed"]) }

        let ratio = sampleRate / file.processingFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(file.length) * ratio) + 4096
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else {
            throw NSError(domain: "WhisperAudio", code: 2)
        }

        var fed = false
        let input: AVAudioConverterInputBlock = { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            guard let inBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
                  (try? file.read(into: inBuf)) != nil else { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }

        var err: NSError?
        converter.convert(to: out, error: &err, withInputFrom: input)
        if let err { throw err }

        let n = Int(out.frameLength)
        let samples = Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: n))
        return MLXArray(samples)
    }
}
