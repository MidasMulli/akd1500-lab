#!/bin/sh
set -eu
: "${AKIDA_ENGINE_ROOT:?User supplied SDK runtime headers}"
: "${AKIDA_RUNTIME_LIBRARY:?User supplied matching runtime library}"
: "${FLATBUFFERS_INCLUDE:?User supplied FlatBuffers headers}"
mkdir -p build
g++ -std=c++17 -O3 -mcpu=native -fopenmp -DAKIDA_VERSION='"2.19.3"' -Isrc \
 -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
 -I"$FLATBUFFERS_INCLUDE" coprocessor/extent_bench.cpp "$AKIDA_RUNTIME_LIBRARY" -ldl -o build/extent_spatial
