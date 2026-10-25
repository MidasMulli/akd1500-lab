#!/bin/sh
set -eu
: "${AKIDA_ENGINE_ROOT:?}" "${AKIDA_RUNTIME_LIBRARY:?}" "${FLATBUFFERS_INCLUDE:?}"
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc \
 -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
 -I"$FLATBUFFERS_INCLUDE" admission/isolation_bench.cpp "$AKIDA_RUNTIME_LIBRARY" -o build/isolated_admission
