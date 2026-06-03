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
        // HF hub download + Whisper tokenizer.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.18"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "WhisperMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
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
