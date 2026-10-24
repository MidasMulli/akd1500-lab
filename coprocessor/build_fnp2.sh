#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Supply separately licensed runtime source}"
: "${FLATBUFFERS_INCLUDE:?Supply compatible headers}"
: "${AKIDA_RUNTIME_LIBRARY:?Supply matching runtime archive}"
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc \
  -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
  -I"$FLATBUFFERS_INCLUDE" coprocessor/fnp2_bench.cpp "$AKIDA_RUNTIME_LIBRARY" -o build/fnp2_contract
