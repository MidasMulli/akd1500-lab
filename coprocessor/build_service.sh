#!/bin/sh
set -eu
: "${AKIDA_ENGINE_ROOT:?Supplied SDK headers}"
: "${AKIDA_RUNTIME_LIBRARY:?Supplied matching static runtime}"
: "${FLATBUFFERS_INCLUDE:?Supplied FlatBuffers headers}"
mkdir -p build
g++ -std=c++17 -O3 -mcpu=native -fopenmp -DAKIDA_VERSION='"2.19.3"' -Isrc \
 -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
 -I"$FLATBUFFERS_INCLUDE" coprocessor/service_bench.cpp "$AKIDA_RUNTIME_LIBRARY" -ldl -o build/host_service
