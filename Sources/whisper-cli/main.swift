import ArgumentParser
import Foundation
import MLX
import WhisperMLX

@main
struct WhisperCLI: ParsableCommand {
    @Option(help: "Directory with config.json + weights.safetensors (for model parity/smoke)")
    var modelDir: String?

    @Option(help: "Model parity golden .safetensors (mel/tokens/encoder_out/logits)")
    var golden: String?

    @Option(help: "Mel-frontend parity golden .safetensors (audio/mel) — no model needed")
    var melGolden: String?

    @Option(help: "Greedy-decode parity golden .safetensors (mel/initial/eot/generated)")
    var decodeGolden: String?

    @Flag(help: "Run on GPU (default CPU = true fp32, no metallib needed)")
    var gpu = false

    func relMax(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self) / (MLX.abs(b).max().item(Float.self) + 1e-9)
    }

    func run() throws {
        try Device.withDefaultDevice(Device(gpu ? .gpu : .cpu)) { try runImpl() }
    }

    func runImpl() throws {
        if let melGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: melGolden))
            let mel = WhisperAudio.logMelSpectrogram(g["audio"]!)
            eval(mel)
            print(String(format: "mel        rel=%.3e  shape=%@  (golden %@)",
                         relMax(mel, g["mel"]!), "\(mel.shape)", "\(g["mel"]!.shape)"))
        }

        guard let modelDir else { return }
        let model = try WhisperLoader.fromDirectory(URL(fileURLWithPath: modelDir))
        print("Loaded Whisper — \(model.dims.nAudioLayer) enc / \(model.dims.nTextLayer) dec layers")

        if let golden {
            let g = try loadArrays(url: URL(fileURLWithPath: golden))
            let enc = model.embedAudio(g["mel"]!)
            eval(enc)
            print(String(format: "encoder    rel=%.3e", relMax(enc, g["encoder_out"]!)))
            let logits = model.logits(g["tokens"]!, audioFeatures: g["encoder_out"]!)
            eval(logits)
            print(String(format: "logits     rel=%.3e", relMax(logits, g["logits"]!)))
            let L = logits.dim(1)
            let aS = argMax(logits[0, L - 1], axis: -1).item(Int.self)
            let aG = argMax(g["logits"]![0, L - 1], axis: -1).item(Int.self)
            print("argmax: swift=\(aS) golden=\(aG)  \(aS == aG ? "✅" : "❌")")
        }

        if let decodeGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: decodeGolden))
            let initial = g["initial"]!.asArray(Int32.self).map { Int($0) }
            let eot = Int(g["eot"]!.item(Int32.self))
            let goldGen = g["generated"]!.asArray(Int32.self).map { Int($0) }
            let gen = model.greedyDecode(mel: g["mel"]!, initialTokens: initial, eot: eot)
            print("decode swift: \(gen)")
            print("decode golden: \(goldGen)")
            print("greedy decode \(gen == goldGen ? "✅ token-exact match" : "❌ MISMATCH")")
        }
    }
}
