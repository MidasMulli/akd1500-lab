#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Supply separately licensed runtime source}"
: "${FLATBUFFERS_INCLUDE:?Supply compatible FlatBuffers headers}"
: "${AKIDA_RUNTIME_LIBRARY:?Supply matching runtime archive}"
case ${1:-contract} in
 contract) source_file=fnp2_bench; target=fnp2_contract;;
 capacity) source_file=fnp2_capacity; target=fnp2_capacity;;
 guard) source_file=fnp2_guard; target=fnp2_guard;;
 boundary) source_file=fnp2_boundary; target=fnp2_boundary;;
 *) exit 2;;
esac
mkdir -p build
g++ -std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -Icoprocessor \
  -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" \
  -I"$FLATBUFFERS_INCLUDE" "coprocessor/$source_file.cpp" "$AKIDA_RUNTIME_LIBRARY" -o "build/$target"
