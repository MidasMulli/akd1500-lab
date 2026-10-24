#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?separately supplied runtime source path}"
: "${FLATBUFFERS_INCLUDE:?compatible FlatBuffers include path}"
: "${AKD_RUNTIME_LIBRARY:?previously built compatible runtime archive}"
mkdir -p build
flags=(-std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" -I"$FLATBUFFERS_INCLUDE")
g++ "${flags[@]}" -fsyntax-only src/stream_bench.cpp
g++ "${flags[@]}" src/stream_bench.cpp "$AKD_RUNTIME_LIBRARY" -o build/stream_bench
