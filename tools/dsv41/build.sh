#!/bin/bash
# tools/dsv41/build.sh: the DeepSeek V4.1 Flash engine (strata-dsv41) for RX 7900 XTX (gfx1100) with Instinct MI50 (gfx906)
# helpers, as built for the results in docs/dsv41 (ROCm 7.14, Release).  Compile only; no GPU is needed.
#
#   SRC=<repo> B=<build dir> J=<jobs> TGTS="<targets>" bash tools/dsv41/build.sh
#
# Every HIP translation unit carries gfx1100 and gfx906 code (STRATA_HIP_HELPER_ALLTU=ON), so any device order works: the
# engine runs on the XTX cards and the MI50s serve as expert helpers (--helper-device).  The configure is the one every
# build in docs/dsv41 used (docs/dsv41/cards/EB-04.md for the all-TU form).
#   SRC    the repository (default: this script's checkout)
#   B      the build directory (default: $SRC/build/dsv41)
#   J      parallel jobs (default 4)
#   TGTS   targets (default strata-dsv41; e.g. "strata-dsv41 ds4_helper_parity_test ds4_iq_test ds4_iq_helper_bench")
#   GGML   a llama.cpp checkout to take ggml from (STRATA_GGML_DIR); empty: CMake fetches the pinned commit
#   RECONF=1  configure again even if $B is configured
#   LIST_OFFLOAD=1  afterwards, list each strata-dsv41 object's offload architectures (the helper TUs must list both)
# Logs: $B/logs/{configure,build}.log.  Exit: the build's status.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}; cd "$SRC" || exit 2
B=${B:-$SRC/build/dsv41}; L=$B/logs
N="nice -n ${NICE:-10}"
GG=${GGML:-}
mkdir -p "$B" "$L"
if [ ! -f "$B/build.ninja" ] || [ "${RECONF:-0}" = 1 ]; then
  echo "=== configure $(date -Is)"
  rm -f "$B/CMakeCache.txt"
  $N cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_HIP_HELPER_ARCHS=gfx906 -DSTRATA_HIP_HELPER_ALLTU=ON \
    ${GG:+-DSTRATA_GGML_DIR=$GG} > "$L/configure.log" 2>&1
  echo "configure rc=$?"
  grep -E "CMake Error|FATAL" "$L/configure.log" | head -20
fi
TGTS=${TGTS:-strata-dsv41}
echo "targets: $TGTS"; t1=$(date +%s)
$N cmake --build "$B" --target $TGTS -j"${J:-4}" -- -k 0 > "$L/build.log" 2>&1; rc=$?
echo "build rc=$rc build=$(( $(date +%s) - t1 ))s"
grep -E "error:|FAILED:" "$L/build.log" | head -40
if [ "${LIST_OFFLOAD:-0}" = 1 ] && [ $rc = 0 ]; then
  echo "=== offload architectures of the strata-dsv41 objects"
  OBJCOPY=${OBJCOPY:-$(command -v llvm-objcopy || echo /opt/rocm/llvm/bin/llvm-objcopy)}
  BUNDLER=${BUNDLER:-$(command -v clang-offload-bundler || echo /opt/rocm/llvm/bin/clang-offload-bundler)}
  for o in $(find "$B/CMakeFiles/strata-dsv41.dir" -name '*.o' | sort); do
    f=$(mktemp); "$OBJCOPY" --dump-section=.hip_fatbin="$f" "$o" /dev/null 2>/dev/null
    [ -s "$f" ] && echo "$(basename "$o"): $("$BUNDLER" --list --type=o --input="$f" 2>/dev/null | grep -o 'gfx[0-9]*' | sort -u | tr '\n' ' ')"
    rm -f "$f"
  done
fi
echo "=== done $(date -Is)"
exit $rc
