import ArgumentParser
import Foundation
import MLX
import WhisperMLX

@main
struct WhisperCLI: ParsableCommand {
    @Argument(help: "Directory with config.json + weights.safetensors (mlx-community Whisper layout)")
    var modelDir: String

    func run() throws {
        let model = try WhisperLoader.fromDirectory(URL(fileURLWithPath: modelDir))
        print("Loaded Whisper — dims: \(model.dims)")

        // Smoke: random mel (B, 3000, nMels) → encoder → decoder one step.
        let mel = MLXRandom.normal([1, 3000, model.dims.nMels])
        let feats = model.embedAudio(mel)
        eval(feats)
        print("encoder output shape: \(feats.shape)  (expect [1, \(model.dims.nAudioCtx), \(model.dims.nAudioState)])")

        let tokens = MLXArray([50258, 50259, 50360] as [Int32]).reshaped([1, 3])
        let logits = model(mel, tokens)
        eval(logits)
        print("decoder logits shape: \(logits.shape)  (expect [1, 3, \(model.dims.nVocab)])")
    }
}
