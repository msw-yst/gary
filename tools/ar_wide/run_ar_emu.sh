#!/bin/bash
# tools/ar_wide/run_ar_emu.sh: DS4_AR_WIDE CPU emulation (no GPU).  Runs inside a ROCm 7.14 build container
# (clang++, C++20) with the repo at $SRC: extracts the host-path all-reduce kernels verbatim (extract_ar.py), builds
# ar_wide_emu, runs its matrix (kinds x members x sizes x blocks x threads).  CPU-heavy while it runs: nice 19.
# Exit 0 only if every case is bit-identical (wide = narrow = CPU reference).
set -u
SRC=${SRC:-.}; OUT=${OUT:-./ar_wide_out}
mkdir -p $OUT
python3 $SRC/tools/ar_wide/extract_ar.py $SRC/src/dsv41/ds4_kernels.cu $OUT/ar_extract.inc || exit 2
CXX=${CXX:-/opt/rocm/llvm/bin/clang++}
nice -n 19 $CXX -std=c++20 -O1 -ffp-contract=off -pthread -I$OUT -o $OUT/ar_wide_emu $SRC/tools/ar_wide/ar_wide_emu.cpp || exit 3
timeout -k 5 1200 nice -n 19 $OUT/ar_wide_emu ${1:-1}; rc=$?
echo "run_ar_emu rc=$rc"
exit $rc
