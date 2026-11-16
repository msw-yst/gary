#!/bin/bash
# tools/qbench/build-ref.sh SRC ARCHS OUT [PREBUILT_BIN]: the reference side of QBENCH.
#   SRC   a llama.cpp checkout (upstream; QBENCH used b11870, commit 74bca395) or a castagna-veloce checkout
#         (QBENCH used main at 3483d4627); both carry ggml under SRC/ggml
#   ARCHS HIP targets, e.g. gfx906 or gfx1100
#   OUT   output dir: OUT/build (the ggml build), OUT/lib (its shared libraries), OUT/qb_ggml (the harness)
#   PREBUILT_BIN  optional: an existing build's bin dir with libggml*.so; then only qb_ggml is compiled against it
# Builds only the ggml targets (shared, Release, HIP on, no curl, no tests / examples / tools / server, GGML_NATIVE off),
# then compiles tools/qbench/qb_ggml.cpp (host C++, the ggml public API only) against them.
# Needs ROCm (hipconfig, its clang), cmake, ninja and g++. No network access is needed once SRC is checked out.
# Note: castagna-veloce main at 3483d4627 did not compile for gfx1100 ("broadcasts are not supported on GFX10+" DPP;
# __builtin_amdgcn_sdot2 needs dot2-insts); it compiles cleanly for gfx906.
set -eu
SRC=$(cd "${1:?usage: build-ref.sh SRC ARCHS OUT [PREBUILT_BIN]}" && pwd); ARCHS=${2:?ARCHS}; OUT=${3:?OUT}; PRE=${4:-}
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
J=${J:-$(nproc)}
if [ -z "$PRE" ]; then
  HIPCXX=$(hipconfig -l)/clang HIP_PATH=$(hipconfig -R) cmake -S "$SRC" -B "$OUT/build" -G Ninja -DGGML_HIP=ON "-DGPU_TARGETS=$ARCHS" \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF > "$OUT/cmake.log" 2>&1 || { echo "cmake failed"; tail -30 "$OUT/cmake.log"; exit 7; }
  cmake --build "$OUT/build" -j "$J" --target ggml ggml-base ggml-cpu ggml-hip > "$OUT/build.log" 2>&1 \
    || { echo "build failed"; grep -E "error" "$OUT/build.log" | head -30; exit 7; }
  L=$OUT/build/bin
else
  L=$(cd "$PRE" && pwd)
fi
mkdir -p "$OUT/lib"; cp -P "$L"/libggml*.so* "$OUT/lib/"
g++ -O2 -std=c++17 -I"$SRC/ggml/include" "$D/qb_ggml.cpp" -L"$OUT/lib" -lggml -lggml-base -Wl,-rpath,"$OUT/lib" -o "$OUT/qb_ggml"
echo "build-ref: $OUT/qb_ggml against $OUT/lib ($(ls "$OUT/lib" | tr '\n' ' '))"
