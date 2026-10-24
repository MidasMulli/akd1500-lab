#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Supply separately licensed runtime source}"
: "${FLATBUFFERS_INCLUDE:?Supply matching FlatBuffers headers}"
: "${AKIDA_RUNTIME_LIBRARY:?Supply matching runtime archive}"
mkdir -p build
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -Icoprocessor -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" -I"$FLATBUFFERS_INCLUDE" coprocessor/window_bench.cpp "$AKIDA_RUNTIME_LIBRARY" -o build/fnp2_window
