#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Supply the separately licensed runtime source}"
: "${FLATBUFFERS_INCLUDE:?Supply compatible FlatBuffers headers}"
: "${AKIDA_RUNTIME_LIBRARY:?Supply the existing matching runtime static library}"
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc \
  -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
  -I"$FLATBUFFERS_INCLUDE" coprocessor/transport.cpp "$AKIDA_RUNTIME_LIBRARY" -o build/pcie_transport
