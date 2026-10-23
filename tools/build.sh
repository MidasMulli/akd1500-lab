#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Set to your separately licensed runtime source directory}"
: "${FLATBUFFERS_INCLUDE:?Set to the runtime-compatible FlatBuffers include directory}"
mkdir -p build
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" -I"$FLATBUFFERS_INCLUDE" src/learning_bench.cpp "$AKIDA_ENGINE_ROOT"/src/*.cpp -o build/learning_bench
