#!/bin/sh
set -eu
: "${AKIDA_ENGINE_ROOT:?}" "${AKIDA_RUNTIME_LIBRARY:?}" "${FLATBUFFERS_INCLUDE:?}"
# Exact archive used by the qualified 2.19.3 learned-policy actor on this board.
# Never link the older deep-audit archive just because its filename looks suitable.
EXPECTED_RUNTIME_SHA256=fe83636fcaf3ab229660a6927269f6be86f5dbc779b3e441cd749ef85c1afd8e
test "$(sha256sum "$AKIDA_RUNTIME_LIBRARY" | cut -d ' ' -f 1)" = "$EXPECTED_RUNTIME_SHA256"
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc \
 -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
 -I"$FLATBUFFERS_INCLUDE" instruments/explorer/native.cpp "$AKIDA_RUNTIME_LIBRARY" \
 -o build/explorer_native
