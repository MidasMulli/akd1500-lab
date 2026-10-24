#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?runtime source path}"
: "${FLATBUFFERS_INCLUDE:?compatible FlatBuffers path}"
: "${AKD_RUNTIME_LIBRARY:?previously built compatible runtime archive}"
flags=(-std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" -I"$FLATBUFFERS_INCLUDE")
g++ "${flags[@]}" -fsyntax-only src/queue_allocation.cpp
g++ "${flags[@]}" src/queue_allocation.cpp "$AKD_RUNTIME_LIBRARY" -o build/queue_allocation
