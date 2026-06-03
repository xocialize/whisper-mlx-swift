import ArgumentParser
import Foundation
import MLX
import WhisperMLX

@main
struct WhisperCLI: ParsableCommand {
    @Argument(help: "Directory with config.json + weights.safetensors")
    var modelDir: String

    @Option(help: "Parity golden .safetensors (mel/tokens/encoder_out/logits) from Python mlx-whisper")
    var golden: String?

    @Flag(help: "Run on GPU (default CPU = true fp32, no metallib needed)")
    var gpu = false

    func relMax(_ a: MLXArray, _ b: MLXArray) -> Float {
        let num = MLX.abs(a - b).max().item(Float.self)
        let den = MLX.abs(b).max().item(Float.self)
        return num / (den + 1e-9)
    }

    func run() throws {
        try Device.withDefaultDevice(Device(gpu ? .gpu : .cpu)) {
            try runImpl()
        }
    }

    func runImpl() throws {
        let model = try WhisperLoader.fromDirectory(URL(fileURLWithPath: modelDir))
        print("Loaded Whisper — \(model.dims.nAudioLayer) enc / \(model.dims.nTextLayer) dec layers")

        if let golden {
            let g = try loadArrays(url: URL(fileURLWithPath: golden))
            let mel = g["mel"]!, tokens = g["tokens"]!
            let encGold = g["encoder_out"]!, logitsGold = g["logits"]!

            let enc = model.embedAudio(mel)
            eval(enc)
            print(String(format: "encoder   rel=%.3e  shape=%@", relMax(enc, encGold), "\(enc.shape)"))

            // Stage-isolated: feed the golden encoder output to the decoder.
            let logits = model.logits(tokens, audioFeatures: encGold)
            eval(logits)
            print(String(format: "logits    rel=%.3e  shape=%@", relMax(logits, logitsGold), "\(logits.shape)"))

            let L = logits.dim(1)
            let argSwift = argMax(logits[0, L - 1], axis: -1).item(Int.self)
            let argGold = argMax(logitsGold[0, L - 1], axis: -1).item(Int.self)
            print("last-token argmax: swift=\(argSwift) golden=\(argGold)  \(argSwift == argGold ? "✅ match" : "❌")")
        } else {
            let mel = MLXRandom.normal([1, 3000, model.dims.nMels])
            let feats = model.embedAudio(mel)
            eval(feats)
            print("encoder smoke shape: \(feats.shape)")
        }
    }
}
