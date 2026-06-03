import Foundation
import MLX

extension WhisperLoader {
    /// Cache dir for a repo's files: ~/Library/Caches/whisper-mlx-swift/<repo>/
    public static func cacheDir(for repo: String) -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("whisper-mlx-swift", isDirectory: true)
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "__"), isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    /// Download `files` from a public HF repo into the cache (skips already-present files).
    /// Dependency-free (swift-transformers exposes no importable Hub *product*).
    public static func downloadRepo(_ repo: String, files: [String]) async throws -> URL {
        let dir = cacheDir(for: repo)
        for f in files {
            let dest = dir.appendingPathComponent(f)
            if FileManager.default.fileExists(atPath: dest.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(f)?download=true") else {
                throw NSError(domain: "WhisperLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "bad url for \(f)"])
            }
            let (tmp, resp) = try await URLSession.shared.download(from: url)
            guard let http = resp as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
                throw NSError(domain: "WhisperLoader", code: 4, userInfo: [NSLocalizedDescriptionKey: "download failed: \(repo)/\(f)"])
            }
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
        }
        return dir
    }

    /// Download (config.json + weights.safetensors) from an mlx-community repo and load.
    public static func fromPretrained(
        _ repo: String = "mlx-community/whisper-large-v3-turbo", dtype: DType = .float32
    ) async throws -> Whisper {
        let dir = try await downloadRepo(repo, files: ["config.json", "weights.safetensors"])
        return try fromDirectory(dir, dtype: dtype)
    }
}

extension WhisperPipeline {
    /// Fully self-contained load: download model weights + the HF tokenizer.
    public static func fromPretrained(
        modelRepo: String = "mlx-community/whisper-large-v3-turbo",
        tokenizerRepo: String = "openai/whisper-large-v3"
    ) async throws -> WhisperPipeline {
        let tokenizer = try await WhisperTokenizer.from(repo: tokenizerRepo)
        let model = try await WhisperLoader.fromPretrained(modelRepo)
        return WhisperPipeline(model: model, tokenizer: tokenizer)
    }
}
