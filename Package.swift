// swift-tools-version: 5.9
//
// MLX-Swift port of OpenAI Whisper (encoder-decoder ASR), mirroring the Python
// reference in mlx-examples/whisper. Module/file names track the reference for
// 1:1 parity diffing. See CLAUDE.md.
//
import PackageDescription

let package = Package(
    name: "WhisperMLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "WhisperMLX", targets: ["WhisperMLX"]),
        .executable(name: "whisper-cli", targets: ["whisper-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        // HF Whisper tokenizer. 1.x to co-resolve with the engine/wan-core ecosystem (E12):
        // the 0.x line (0.1.18..<1.0) is disjoint with the 1.x range everything else pins.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "WhisperMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            resources: [.copy("Resources/mel_filters_128.safetensors")]
        ),
        .executableTarget(
            name: "whisper-cli",
            dependencies: [
                "WhisperMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
