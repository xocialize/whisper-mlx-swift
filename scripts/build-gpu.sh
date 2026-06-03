#!/usr/bin/env bash
# Build whisper-cli via xcodebuild so the MLX Metal library (mlx.metallib) is compiled
# and bundled — required for GPU runs (`--gpu`). Plain `swift build`/`swift run` does NOT
# produce the metallib (SPM-CLI gap), so GPU runs there crash with
# "Failed to load the default metallib". CPU runs (the default) need none.
#
# Usage:  scripts/build-gpu.sh           # build
#         scripts/build-gpu.sh --run ... # build, then run whisper-cli with the given args
set -euo pipefail
cd "$(dirname "$0")/.."

DD=".xcdd"
xcodebuild -scheme whisper-cli -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath "$DD" -configuration Release build

BIN="$DD/Build/Products/Release/whisper-cli"
echo "binary: $BIN"
find "$DD/Build/Products/Release" -name "*.metallib" | head -1 | grep -q . \
  && echo "metallib bundled ✓ — run with --gpu" || echo "warning: no metallib found"

if [[ "${1:-}" == "--run" ]]; then
  shift
  exec "$BIN" "$@"
fi
