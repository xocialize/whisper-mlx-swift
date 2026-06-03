import Foundation
import MLX
import MLXNN

public enum WhisperLoader {
    /// Load from a local directory containing `config.json` + `weights.safetensors`
    /// (mlx-community Whisper layout). HF auto-download is added with the tokenizer work.
    public static func fromDirectory(_ dir: URL, dtype: DType = .float32) throws -> Whisper {
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let dims = try JSONDecoder().decode(ModelDimensions.self, from: configData)
        let model = Whisper(dims)

        var weights = try loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        // Keep model weights only (drop non-module entries like `alignment_heads`).
        weights = weights.filter { $0.key.hasPrefix("encoder.") || $0.key.hasPrefix("decoder.") }
            .mapValues { $0.asType(dtype) }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)
        return model
    }
}
