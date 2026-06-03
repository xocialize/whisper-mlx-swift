import ArgumentParser
import Foundation
import MLX
import WhisperMLX

@main
struct WhisperCLI: AsyncParsableCommand {
    @Option(help: "Directory with config.json + weights.safetensors (for model parity/smoke)")
    var modelDir: String?

    @Option(help: "HF model repo to auto-download (e.g. mlx-community/whisper-large-v3-turbo)")
    var modelRepo: String?

    @Option(help: "Model parity golden .safetensors (mel/tokens/encoder_out/logits)")
    var golden: String?

    @Option(help: "Mel-frontend parity golden .safetensors (audio/mel) — no model needed")
    var melGolden: String?

    @Option(help: "Greedy-decode parity golden .safetensors (mel/initial/eot/generated)")
    var decodeGolden: String?

    @Option(help: "Timestamped-decode parity golden (mel/initial/eot/timestamp_begin/no_timestamps/generated)")
    var decodeTsGolden: String?

    @Option(help: "Transcribe a golden's mel to text (loads the HF tokenizer); needs --model-dir")
    var transcribeMel: String?

    @Option(help: "Transcribe an audio file (wav/mp3/...) end-to-end; needs --model-dir")
    var audio: String?

    @Option(help: "Language code for transcription")
    var language: String = "en"

    @Flag(help: "Emit segment timestamps when transcribing --audio")
    var timestamps = false

    @Flag(help: "Run on GPU (default CPU = true fp32, no metallib needed)")
    var gpu = false

    func relMax(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self) / (MLX.abs(b).max().item(Float.self) + 1e-9)
    }

    func run() async throws {
        // Network (tokenizer + weight download) is async — resolve before the device scope.
        let needsTokenizer = transcribeMel != nil || audio != nil
        let tokenizer = needsTokenizer ? try await WhisperTokenizer.from() : nil

        var modelURL: URL? = modelDir.map { URL(fileURLWithPath: $0) }
        if modelURL == nil, let repo = modelRepo {
            modelURL = try await WhisperLoader.downloadRepo(repo, files: ["config.json", "weights.safetensors"])
        }

        try Device.withDefaultDevice(Device(gpu ? .gpu : .cpu)) { try runImpl(tokenizer, modelURL) }
    }

    func runImpl(_ tokenizer: WhisperTokenizer?, _ modelURL: URL?) throws {
        if let melGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: melGolden))
            let mel = WhisperAudio.logMelSpectrogram(g["audio"]!)
            eval(mel)
            print(String(format: "mel        rel=%.3e  shape=%@  (golden %@)",
                         relMax(mel, g["mel"]!), "\(mel.shape)", "\(g["mel"]!.shape)"))
        }

        guard let modelURL else { return }
        let model = try WhisperLoader.fromDirectory(modelURL)
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
            let genCached = model.greedyDecodeCached(mel: g["mel"]!, initialTokens: initial, eot: eot)
            print("decode golden:   \(goldGen)")
            print("decode no-cache: \(gen)   \(gen == goldGen ? "✅" : "❌")")
            print("decode cached:   \(genCached)   \(genCached == goldGen ? "✅ token-exact" : "❌ MISMATCH")")
        }

        if let decodeTsGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: decodeTsGolden))
            let initial = g["initial"]!.asArray(Int32.self).map { Int($0) }
            let eot = Int(g["eot"]!.item(Int32.self))
            let tb = Int(g["timestamp_begin"]!.item(Int32.self))
            let nts = Int(g["no_timestamps"]!.item(Int32.self))
            let goldGen = g["generated"]!.asArray(Int32.self).map { Int($0) }
            let gen = model.greedyDecodeTimestamped(mel: g["mel"]!, initialTokens: initial, eot: eot,
                                                    timestampBegin: tb, noTimestamps: nts)
            print("ts decode golden: \(goldGen)")
            print("ts decode swift:  \(gen)   \(gen == goldGen ? "✅ token-exact" : "❌ MISMATCH")")
        }

        if let transcribeMel, let tokenizer {
            let g = try loadArrays(url: URL(fileURLWithPath: transcribeMel))
            let pipeline = WhisperPipeline(model: model, tokenizer: tokenizer)
            let text = pipeline.transcribe(mel: g["mel"]!, language: language)
            print("transcription: \(text.debugDescription)")
        }

        if let audio, let tokenizer {
            let wav = try WhisperAudio.load(URL(fileURLWithPath: audio))
            let pipeline = WhisperPipeline(model: model, tokenizer: tokenizer)
            if timestamps {
                for s in pipeline.transcribeSegments(audio: wav, language: language) {
                    print(String(format: "[%.2f → %.2f] %@", s.start, s.end, s.text))
                }
            } else {
                print("transcription: \(pipeline.transcribe(audio: wav, language: language).debugDescription)")
            }
        }
    }
}
