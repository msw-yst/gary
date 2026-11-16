#!/bin/bash
# tools/qbench/run-card.sh OUTDIR [PHASE ...]: QBENCH on ONE GPU - one routed DeepSeek V4.1 Flash expert's decode
# (5120 x 2304 x 3, real expert bytes), our kernels against llama.cpp and castagna-veloce on the same bytes and shapes,
# plus (gfx906) the MI50 helper's request round trip.  See tools/qbench/README.md and docs/dsv41/QBENCH-2026-11-16.md.
#
# Pin the card first: HIP_VISIBLE_DEVICES (and/or ROCR_VISIBLE_DEVICES) so that exactly one GPU is visible. Nothing else
# should run on that card. Each cell is its own process; 3 s pause between cells.
#
# Inputs (environment):
#   QB_BIN   dir with ds4_iq_helper_bench and ds4_helper_rt_bench (this tree's build, targets of the same names)
#   QB_DATA  dir with xxs.qbx xs.qbx s.qbx mx.qbx (mkpool.py: 40 real experts each, IQ2_XXS / IQ2_XS / IQ2_S / MXFP4)
#   QB_LCPP  dir with qb_ggml + lib/ (build-ref.sh on upstream llama.cpp)          (phases lcpp, kt; optional)
#   QB_CV    dir with qb_ggml + lib/ (build-ref.sh on castagna-veloce)            (phases cv, kt; optional)
#   QB_SLICES=1  also run our kernels at the tensor-parallel 1/4 slice shapes (IQ: 768 and 512 rows, MXFP4: 576)
#   REPS (default 300)
# Phases (default: ours lcpp cv kt rt sum):
#   ours  roofline; per format --path win B 1,2,4,8 and --path tok B 1, whole expert
#   lcpp  qb_ggml on llama.cpp: expert mode per format, Q2_K/Q3_K (synthetic), the moe1-shaped layer (unfused)
#   cv    qb_ggml on castagna-veloce: the same, plus its fused moe1 (ggml_cuda_moe1_ffn) at 2304 / 768 / 512 rows, and
#         moe1 with GGML_CUDA_MOE1=0 (its own unfused path)
#   kt    rocprofv3 --kernel-trace of every cell above, one process each: GPU-side kernel us per expert (ktsum.py)
#   rt    ds4_helper_rt_bench: --sweep 1 (the round trip by mechanism), then 1 expert plain, without kernels, as an empty
#         request, with 800 us idle before each request, and 2 experts (gfx906 helper cards)
#   sum   OUTDIR/SUMMARY.txt (every QB / QG / KT / rt line); qbsum.py turns QB / QG lines into tables
# Exit: 0 done | 5 an input is missing | 8 NO DATA (an expected "ours" cell is missing)
set -u
O=${1:?usage: run-card.sh OUTDIR [PHASE ...]}; shift
PHASES=${*:-"ours lcpp cv kt rt sum"}
D=$(cd "$(dirname "$0")" && pwd)
B=${QB_BIN:?set QB_BIN}/ds4_iq_helper_bench; RT=$QB_BIN/ds4_helper_rt_bench; Q=${QB_DATA:?set QB_DATA}
REPS=${REPS:-300}
mkdir -p "$O"; O=$(cd "$O" && pwd)
for f in "$B" "$RT"; do [ -x "$f" ] || { echo "missing $f"; exit 5; }; done
for p in xxs xs s mx; do [ -f "$Q/$p.qbx" ] || { echo "missing $Q/$p.qbx"; exit 5; }; done
unset GGML_CUDA_MOE1
cell() { local f=$1; shift; echo "### $(date -u +%T) $*" >> "$f"; timeout -k 10 600 "$@" >> "$f" 2>&1; echo "rc $?" >> "$f"; sleep 3; }
ref() {   # ref NAME DIR: one reference build's cells into $O/NAME
  local nm=$1 R=$2; [ -x "$R/qb_ggml" ] || { echo "$nm: no $R/qb_ggml, skipped"; return 0; }
  mkdir -p "$O/$nm"; ( cd "$O/$nm" && export LD_LIBRARY_PATH=$R/lib
    for p in xxs xs s mx; do cell $p.txt "$R/qb_ggml" --mode expert --pool-file "$Q/$p.qbx" --B 1,2,4,8 --reps 200; done
    cell q2q3.txt "$R/qb_ggml" --mode expert --synth 10 11 --B 1,2,4,8 --reps 200
    if [ "$nm" = cv ]; then
      for m in 2304 768 512; do GGML_CUDA_MOE1_DEBUG=1 cell moe1.txt "$R/qb_ggml" --mode moe1 --ff $m --reps 200; done
      GGML_CUDA_MOE1=0 cell moe1-off.txt "$R/qb_ggml" --mode moe1 --reps 200
    else
      cell moe1.txt "$R/qb_ggml" --mode moe1 --reps 200
    fi ); }
for ph in $PHASES; do
  case $ph in
    ours) mkdir -p "$O/ours"; ( cd "$O/ours"
      rocminfo 2>/dev/null | grep -E "^\s+Name:\s+gfx" | sort | uniq -c > agents.txt
      cell roofline.txt "$B" --tg 16 --groups 1 --reps 50
      for p in xxs xs s mx; do
        cell $p.txt "$B" --pool-file "$Q/$p.qbx" --path win --B 1,2,4,8 --reps $REPS
        cell $p.txt "$B" --pool-file "$Q/$p.qbx" --path tok --B 1 --reps $REPS
        if [ "${QB_SLICES:-0}" = 1 ]; then
          if [ $p = mx ]; then SL="576"; else SL="768 512"; fi
          for fn in $SL; do
            cell slice-$p.txt "$B" --pool-file "$Q/$p.qbx" --slice $fn --path tok --B 1 --reps $REPS
            cell slice-$p.txt "$B" --pool-file "$Q/$p.qbx" --slice $fn --path win --B 1,2,4,8 --reps $REPS
          done
        fi
      done ) ;;
    lcpp) ref lcpp "${QB_LCPP:-}" ;;
    cv) ref cv "${QB_CV:-}" ;;
    kt) command -v rocprofv3 > /dev/null || { echo "kt: no rocprofv3: skipped"; continue; }
      mkdir -p "$O/kt"; ( cd "$O/kt"; : > calls.txt
      W=10; N=100; OC=$(( 2 * W + 2 * N + 1 ))   # ours: warm-up + iso + busy passes + the error rep (ktsum.py keeps the busy pass)
      tr() { local nm=$1 calls=$2; shift 2; echo "$nm $calls" >> calls.txt
        timeout -k 10 600 rocprofv3 --kernel-trace --output-format csv -d "$PWD/kt/$nm" -o kt -- "$@" > kt-$nm.log 2>&1
        echo "$nm rc $?" >> rcs.txt; sleep 2; }
      for p in xxs xs s mx; do
        tr ours-$p-0-tok-1 $OC "$B" --pool-file "$Q/$p.qbx" --path tok --B 1 --reps $N --warm $W
        for b in 1 2 4 8; do tr ours-$p-0-win-$b $OC "$B" --pool-file "$Q/$p.qbx" --path win --B $b --reps $N --warm $W; done
        for nm in lcpp cv; do R=${QB_LCPP:-}; [ $nm = cv ] && R=${QB_CV:-}; [ -x "$R/qb_ggml" ] || continue
          for b in 1 2 4 8; do LD_LIBRARY_PATH=$R/lib tr $nm-$p-0-exp-$b $(( 55 * 40 )) "$R/qb_ggml" --mode expert --pool-file "$Q/$p.qbx" --B $b --reps 50 --warm 5; done
        done
      done
      for nm in lcpp cv; do R=${QB_LCPP:-}; [ $nm = cv ] && R=${QB_CV:-}; [ -x "$R/qb_ggml" ] || continue
        for b in 1 2 4 8; do LD_LIBRARY_PATH=$R/lib tr $nm-q2q3-0-exp-$b $(( 55 * 40 )) "$R/qb_ggml" --mode expert --synth 10 11 --B $b --reps 50 --warm 5; done
      done
      [ -x "${QB_LCPP:-}/qb_ggml" ] && LD_LIBRARY_PATH=$QB_LCPP/lib tr lcpp-moe1-2304-moe1-1 $(( 55 * 40 * 7 )) "$QB_LCPP/qb_ggml" --mode moe1 --reps 50 --warm 5
      if [ -x "${QB_CV:-}/qb_ggml" ]; then
        for m in 2304 768 512; do LD_LIBRARY_PATH=$QB_CV/lib tr cv-moe1-$m-moe1-1 $(( 55 * 40 * 7 )) "$QB_CV/qb_ggml" --mode moe1 --ff $m --reps 50 --warm 5; done
        GGML_CUDA_MOE1=0 LD_LIBRARY_PATH=$QB_CV/lib tr cv-moe1off-2304-moe1-1 $(( 55 * 40 * 7 )) "$QB_CV/qb_ggml" --mode moe1 --reps 50 --warm 5
      fi ) ;;
    rt) mkdir -p "$O/rt"; ( cd "$O/rt"
      cell sweep.txt "$RT" --dev 0 --sweep 1 --experts 1 --T 1
      cell parts.txt "$RT" --dev 0 --experts 1 --T 1 --reps 2000
      cell parts.txt "$RT" --dev 0 --experts 1 --T 1 --reps 2000 --ablate nokern
      cell parts.txt "$RT" --dev 0 --experts 1 --T 1 --reps 2000 --empty 1
      cell parts.txt "$RT" --dev 0 --experts 1 --T 1 --reps 2000 --idle-us 800
      cell parts.txt "$RT" --dev 0 --experts 2 --T 1 --reps 2000 ) ;;
    sum) ;;
    *) echo "unknown phase $ph"; exit 5 ;;
  esac
done
{ grep -h "^device\|^QBENCH device\|^QGGML device\|^roofline\|^G 1 " "$O"/*/*.txt 2>/dev/null | sort -u
  for s in ours cv lcpp; do grep -h "^QB \|^QG " "$O"/$s/*.txt 2>/dev/null; done
  if [ -f "$O/kt/calls.txt" ]; then while read -r nm calls; do
    case $nm in ours-*) python3 -I "$D/ktsum.py" "$O/kt/kt/$nm" "$calls" "$nm" 100 1;; *) python3 -I "$D/ktsum.py" "$O/kt/kt/$nm" "$calls" "$nm";; esac
  done < "$O/kt/calls.txt"; fi
  for f in "$O"/rt/*.txt; do [ -f "$f" ] && { echo "## $(basename "$f")"; grep -v "^$" "$f" | tail -40; }; done; } > "$O/SUMMARY.txt"
miss=0
if [[ " $PHASES " == *" ours "* ]]; then
  for f in xxs xs s mx; do for b in 1 2 4 8; do grep -q "^QB .*path=win B=$b " "$O/ours/$f.txt" 2>/dev/null || miss=1; done; done
fi
[ $miss = 1 ] && { echo "NO DATA: an expected ours cell is missing ($O/ours)"; exit 8; }
echo "qbench done: $O/SUMMARY.txt"
