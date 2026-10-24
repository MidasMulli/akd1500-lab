#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
: "${AKIDA_ENGINE_ROOT:?Set to your separately licensed runtime source directory}"
: "${FLATBUFFERS_INCLUDE:?Set to the runtime-compatible FlatBuffers include directory}"
mkdir -p build
bench=${AKD_LAB_BENCH:-learning_bench}
case "$bench" in learning_bench|readout_bench|offload_bench|bank_bench|image_service|mario_stream|pong_stream|pong_relearn|relay_bench|metal_bench|queue_bench|ablation_bench|coupled_cpu_bench|resident_bench|fixed_bridge|contract_probe|or_bench|primitive_bench|quiet_relay) ;; *) exit 2;; esac
flags=(-std=c++17 -O3 -mcpu=native -DAKIDA_VERSION='"2.19.3"' -Isrc -I"$AKIDA_ENGINE_ROOT/api" -I"$AKIDA_ENGINE_ROOT/inc" -I"$AKIDA_ENGINE_ROOT/src" -I"$FLATBUFFERS_INCLUDE")
benchmark_source="src/$bench.cpp"
if [ "$bench" = image_service ]; then benchmark_source=image_service/sources/native_service.cpp; fi
# Reject own-source errors before compiling the separately supplied runtime.
g++ "${flags[@]}" -fsyntax-only "$benchmark_source"
runtime_key=$(python3 - "$AKIDA_ENGINE_ROOT" "$FLATBUFFERS_INCLUDE" <<'PY'
from pathlib import Path
import hashlib,subprocess,sys
h=hashlib.sha256(b'akd-lab-runtime-O3-cxx17-v1')
h.update(subprocess.check_output(['g++','--version']))
h.update(subprocess.check_output(['g++','-mcpu=native','-dM','-E','-x','c++','/dev/null']))
for root in map(Path,sys.argv[1:]):
    for f in sorted(root.rglob('*')):
        if f.is_file() and f.suffix in ['.h','.hpp','.cpp']:
            h.update(str(f.relative_to(root)).encode());h.update(f.read_bytes())
print(h.hexdigest()[:24])
PY
)
runtime_library="build/runtime-$runtime_key.a"
if [ ! -f "$runtime_library" ]; then
  task_build_dir=$(mktemp -d build/runtime-objects.XXXXXX)
  trap 'rm -rf "$task_build_dir"' EXIT
  for runtime_source in "$AKIDA_ENGINE_ROOT"/src/*.cpp; do
    g++ "${flags[@]}" -c "$runtime_source" -o "$task_build_dir/$(basename "$runtime_source" .cpp).o"
  done
  ar rcs "$runtime_library.tmp" "$task_build_dir"/*.o
  mv "$runtime_library.tmp" "$runtime_library"
fi
g++ "${flags[@]}" "$benchmark_source" "$runtime_library" -o "build/$bench"
