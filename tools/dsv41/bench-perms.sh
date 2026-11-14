#!/usr/bin/env bash
# tools/dsv41/bench-perms.sh: "just the numbers" for one GPU permutation of strata-dsv41 (DeepSeek V4.1 Flash): which XTX cards
# (tensor parallel = their count) and which MI50 expert helpers.  It prints one markdown table row per run and appends it to
# OUT/table.md.  This is not an A/B test: each number comes from ONE cold, one-shot engine process.  See
# docs/dsv41/benchmarking-gpu-permutations.md for how to read the numbers.
#
#   bash tools/dsv41/bench-perms.sh --exe build/dsv41/strata-dsv41 --pack <pack dir> --ids-file <ids> --xtx 0,1 [--mi50 4,5]
#
# A run is a set of engine processes ("arms"). Each arm loads the model itself:
#   off   a 64-token prompt (the first 64 ids of --ids-file), --n N (256), greedy; the drafter is loaded but off (--dspark off)
#   on    the same with the drafter on (--dspark on --dspark-conf 0.3 --dspark-block 3)
#   p2k   prompt read: the first 2,048 ids, --n 16     p8k  the first 8,192 ids     p32k  the first 32,768 ids (needs --ctx >= 33024)
#   bt    the engine's batch test at B=8 (8 prompts of 1,024 ids from id 8,192 on, --batch-sizes 8, --n N): ms a step
#   The default arms are: off on p2k p8k.
# Ablation: --ablate "NAME NAME=VALUE ..." first runs the base env. Then it does one run per item (NAME: unset it; NAME=VALUE: set
#   it to that value), then one run with every item applied ("none"), then the base env again ("base2"). The gap between base and
#   base2 is the noise floor.
#
# Options
#   --exe FILE        the engine (tools/dsv41/build.sh)            --pack DIR       the model pack
#   --ids-file FILE   comma-separated token ids: at least 8,192 (16,384 with bt, 32,768 with p32k). Any long text run through
#                     tools/dsv41/tok.py will do. Use the same file for every run, so that every run reads the same prompts.
#   --xtx LIST        XTX HIP ordinals, e.g. 0,1 (tensor parallel = the count)   --mi50 LIST   MI50 helper ordinals (default: none)
#   --arms "..."      the arms to run (default "off on p2k p8k")    --ctx N          --max-context (default 8448)
#   --n N             decode tokens (default 256)                    --label TEXT     the row name (default <xtx count>x<mi50 count>)
#   --env "K=V ..."   the engine env (default: the set below)        --add-env "K=V ..."   added to the env
#   --ablate "..."    see above                                      --rocblas-table FILE  STRATA_ROCBLAS_TUNING (cards/FG-04.md)
#   --xargs "..."     more engine arguments (e.g. "--mmap-experts on")
#   --out DIR         the run directory (default ./runs/bench-perms) --tmo S          time limit per process (default 1800)
#   --stop-at C       do not start a process while a used card's junction is >= C (default 95; exit 3)
#   --cool-at C, --cool-to C   if a used card reached cool-at C (default 88) during a process, wait until all are below cool-to
#                     (default 60, at most 20 min)
#   --gap S           idle seconds between ablation runs (default 60)   --dry-run   print the commands, run nothing
# The default env: build 16's recommended set (docs/dsv41/cards/SV-16.md). Every switch is described in docs/dsv41/FEATURES.md. The helper switches act only with --mi50, and the
# tensor-parallel ones only with two or more XTX:
DEF_ENV="DS4_HC_FUSE=2 DS4_ROUTE_FUSE=2 DS4_DECODE_TRIM=7 DS4_ROUTER_BF16=1 DS4_HELPER_OVERLAP=1 DS4_HELPER_BATCH=1 DS4_HELPER_DD=1"
DEF_ENV="$DEF_ENV DS4_HELPER_DD_SKIP=1 DS4_AR_POST=3 DS4_ATTN_MERGE=3 DS4_AHEAD=0 DS4_LP_PUBFUSE=1 DS4_LP_JOINCOMB=1 DS4_LP_FLAGFUSE=1"
DEF_ENV="$DEF_ENV DS4_HW_FOLD=1 DS4_HW_HEADQ8=1 DS4_HW_PUBT=1 DS4_EG_SPLIT=1 DS4_PF_HELPER=8192 DS4_BATCH_PREFILL_SHARE=0.5"
DEF_ENV="$DEF_ENV DS4_EG_AHEAD=1 DS4_TOPK_MB=1 DS4_XK_QO=1 DS4_PF_TILE=1 DS4_TOPK_MB_BATCH=1"
DEF_ENV="$DEF_ENV DS4_XK_QA=1 DS4_IXS_DISPATCH=16384 DS4_PF_HELPER_MAXP=8192 DS4_EC_FIX=1"
# Run it where the GPUs are visible, with no other GPU job running. For the MI50 clock pin (docs/dsv41/cards/CP-03.md), run it
# under tools/dsv41/with-mi50-pin.sh (as root).  Exit: 0 rows written | 2 bad arguments or checks | 3 a card too hot to start
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE=; PACK=; IDS=; XL=; ML=; ARMS="off on p2k p8k"; CTX=8448; N=256; NP=16; LABEL=; ENVS=$DEF_ENV; ADD=; ABL=; TAB=; XARGS=
OUT=$PWD/runs/bench-perms; TMO=1800; STOP_AT=95; COOL_AT=88; COOL_TO=60; GAP=60; DRY=0
CONF=0.3; BLOCK=3; BTE="DS4_BATCH_DECODE=8 DS4_BATCH_DSPARK=0 DS4_HELPER_PLACE=stripe"
die() { echo "bench-perms: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1; shift; continue ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
  esac
  [ $# -ge 2 ] || die "$1 needs a value (--help)"
  case $1 in
    --exe) EXE=$2 ;; --pack) PACK=$2 ;; --ids-file) IDS=$2 ;; --xtx) XL=$2 ;; --mi50) ML=$2 ;; --arms) ARMS=$2 ;;
    --ctx) CTX=$2 ;; --n) N=$2 ;; --label) LABEL=$2 ;; --env) ENVS=$2 ;; --add-env) ADD=$2 ;; --ablate) ABL=$2 ;;
    --rocblas-table) TAB=$2 ;; --xargs) XARGS=$2 ;; --out) OUT=$2 ;; --tmo) TMO=$2 ;; --stop-at) STOP_AT=$2 ;;
    --cool-at) COOL_AT=$2 ;; --cool-to) COOL_TO=$2 ;; --gap) GAP=$2 ;;
    *) die "unknown option $1 (--help)" ;;
  esac; shift 2
done
[ -x "$EXE" ] || die "--exe: '$EXE' is not an executable"
[ -d "$PACK" ] || die "--pack: '$PACK' is not a directory"
[ -s "$IDS" ] || die "--ids-file: '$IDS' is empty or missing"
[[ "$XL" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--xtx: a comma-separated list of HIP ordinals, e.g. 0,1"
[ -z "$ML" ] || [[ "$ML" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--mi50: a comma-separated list of HIP ordinals"
for v in CTX N TMO STOP_AT COOL_AT COOL_TO GAP; do [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a whole number"; done
for a in $ARMS; do case $a in off|on|p2k|p8k|p32k|bt) ;; *) die "--arms: unknown arm $a" ;; esac; done
for kv in $ENVS $ADD; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=.*$ ]] || die "env entry '$kv' is not NAME=value"; done
for kv in $ABL; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*(=.*)?$ ]] || die "--ablate entry '$kv' is not NAME or NAME=value"; done
[ -n "$TAB" ] && { [ -s "$TAB" ] || die "--rocblas-table: '$TAB' is missing"; ENVS="$ENVS STRATA_ROCBLAS_TUNING=$TAB"; }
ENVS="$ENVS $ADD"
NX=$(tr , '\n' <<<"$XL" | wc -l); NM=0; [ -n "$ML" ] && NM=$(tr , '\n' <<<"$ML" | wc -l)
[ -z "$(tr , '\n' <<<"$XL,$ML" | grep . | sort | uniq -d)" ] || die "an ordinal is listed twice (--xtx / --mi50)"
LABEL=${LABEL:-${NX}x${NM}}
# The cards: their kinds come from tools/dsv41/devmap.sh (the HIP order, read from sysfs). Their PCI addresses give the
# temperatures, the power and the links.
DM=$(bash "$HERE/devmap.sh") || die "tools/dsv41/devmap.sh failed (HIP_VISIBLE_DEVICES set? no amdgpu?)"
kind() { awk -v o="$1" '$1 == o {print $2}' <<<"$DM"; }
bdf() { awk -v o="$1" '$1 == o {print $3}' <<<"$DM"; }
linkw() { local q m=99 w; q=$(readlink -f "/sys/bus/pci/devices/$1"); while [ -e "$q/current_link_width" ]; do
    w=$(cat "$q/current_link_width"); [ "$w" -lt $m ] && m=$w; q=$(dirname "$q"); done; [ $m = 99 ] && m="?"; echo $m; }
USED=""; LINKS=""
for o in $(tr , ' ' <<<"$XL"); do [ "$(kind $o)" = XTX ] || die "--xtx: HIP $o is '$(kind $o)', not an XTX (bash tools/dsv41/devmap.sh)"
  USED="$USED $(bdf $o)"; LINKS="$LINKS${LINKS:+ }$o:x$(linkw $(bdf $o))"; done
for o in $(tr , ' ' <<<"$ML"); do [ "$(kind $o)" = MI50 ] || die "--mi50: HIP $o is '$(kind $o)', not an MI50 (bash tools/dsv41/devmap.sh)"
  USED="$USED $(bdf $o)"; done
# the ids: the slices every run reads
NIDS=$(tr -d ' \r\n' < "$IDS" | tr , '\n' | grep -c '^[0-9][0-9]*$')
need=8192; [[ " $ARMS " == *" bt "* ]] && need=16384; [[ " $ARMS " == *" p32k "* ]] && need=32768
[ "$NIDS" -ge $need ] || die "--ids-file has $NIDS ids; these arms need $need"
[[ " $ARMS " == *" p32k "* ]] && [ "$CTX" -lt 33024 ] && die "p32k needs --ctx 33024 or more"
mkdir -p "$OUT/prompts" || die "cannot write $OUT"
slice() { tr -d ' \r\n' < "$IDS" | tr , '\n' | grep '^[0-9][0-9]*$' | sed -n "$(( $1 + 1 )),$(( $1 + $2 ))p" | paste -sd, -; }
slice 0 64 > "$OUT/prompts/p64.ids"; slice 0 2048 > "$OUT/prompts/p2k.ids"; slice 0 8192 > "$OUT/prompts/p8k.ids"
[ "$NIDS" -ge 32768 ] && slice 0 32768 > "$OUT/prompts/p32k.ids"
P8=""; if [ "$NIDS" -ge 16384 ]; then for i in 0 1 2 3 4 5 6 7; do slice $(( 8192 + i * 1024 )) 1024 > "$OUT/prompts/b$i.ids"
  P8="$P8${P8:+,}$OUT/prompts/b$i.ids"; done; fi
DEV="--gpus $XL --tp $NX"; [ -n "$ML" ] && DEV="$DEV --helper-device $ML"
temps() { local b t; for b in $USED; do t=$(cat /sys/bus/pci/devices/$b/hwmon/hwmon*/temp2_input 2>/dev/null | head -1); echo $(( ${t:-0} / 1000 )); done; }
hottest() { temps | sort -n | tail -1; }
watts() { local b w s=0; for b in $USED; do w=$(cat /sys/bus/pci/devices/$b/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
    [ -z "$w" ] && w=$(cat /sys/bus/pci/devices/$b/hwmon/hwmon*/power1_input 2>/dev/null | head -1); s=$(( s + ${w:-0} / 1000000 )); done; echo $s; }
coolto() { local t=0; while [ $t -lt 1200 ] && [ "$(hottest)" -ge "$1" ]; do sleep 15; t=$((t + 15)); done; }
now() { date +%s.%N | cut -c1-14; }
# one ENV ARGS: one engine process in $OUT/<run>/<arm>; inherited DS4_* / STRATA_* variables are dropped, so the arm is exactly
# its cmd.txt.  times.txt: launch, the "prompt:" line seen (polled every 0.5 s), exit.  The time limit stops only this process.
one() { local d=$1 envs=$2 p i=0 seen=0 pk=0 w h; shift 2; mkdir -p "$d"
  echo "env $envs $EXE --pack $PACK --profile-readonly $DEV $*" > "$d/cmd.txt"
  [ $DRY = 1 ] && { cat "$d/cmd.txt"; return 0; }
  h=$(hottest); [ "$h" -ge "$STOP_AT" ] && { echo "bench-perms: a used card is at $h C (>= $STOP_AT): stopping"; exit 3; }
  ( for v in $(compgen -e); do case $v in DS4_*|STRATA_*) unset "$v" ;; esac; done
    exec env $envs "$EXE" --pack "$PACK" --profile-readonly $DEV "$@" < /dev/null > "$d/engine.stdout" 2> "$d/engine.stderr" ) &
  p=$!; echo "t0 $(now)" > "$d/times.txt"
  while kill -0 $p 2>/dev/null; do sleep 0.5; i=$((i + 1))
    [ $seen = 0 ] && grep -q '^prompt: ' "$d/engine.stderr" 2>/dev/null && { echo "tprompt $(now)" >> "$d/times.txt"; seen=1; }
    w=$(watts); [ "$w" -gt $pk ] && pk=$w
    [ $(( i / 2 )) -ge "$TMO" ] && { echo "time limit $TMO s" > "$d/stall.txt"; kill -TERM $p; sleep 20; kill -KILL $p 2>/dev/null; }
  done; wait $p; echo "rc=$?" > "$d/rc.txt"; echo "tend $(now)" >> "$d/times.txt"; echo "$pk" > "$d/peak_w.txt"
  [ $seen = 0 ] && grep -q '^prompt: ' "$d/engine.stderr" && echo "tprompt $(now) (seen after exit)" >> "$d/times.txt"
  h=$(hottest); [ "$h" -ge "$COOL_AT" ] && coolto "$COOL_TO"; true; }
arm() { local r=$1 a=$2 e="$3 DS4_NO_EOS=1" c="--max-context $CTX $XARGS"
  case $a in
    off)  one "$r/off" "$e" $c --dspark off --ids "@$OUT/prompts/p64.ids" --n $N ;;
    on)   one "$r/on" "$e" $c --dspark on --dspark-conf $CONF --dspark-block $BLOCK --ids "@$OUT/prompts/p64.ids" --n $N ;;
    p2k|p8k|p32k) one "$r/$a" "$e" $c --dspark off --ids "@$OUT/prompts/$a.ids" --n $NP ;;
    bt)   one "$r/bt" "$3 $BTE" $c --dspark off --batch-prompts "$P8" --n $N --batch-sizes 8 ;;
  esac; }
# the row: every number is the engine's own stderr line, except load s (times.txt) and peak W (the used cards' hwmon, boards only)
ln() { grep -hE "$2" "$1/engine.stderr" 2>/dev/null | head -1; }
ok() { local d=$1 n=$2 w=""; [ -d "$d" ] || { echo "-"; return; }
  [ "$(cat "$d/rc.txt" 2>/dev/null)" = rc=0 ] || w="$w $(cat "$d/rc.txt" 2>/dev/null)"; [ -f "$d/stall.txt" ] && w="$w time limit"
  [ "$(grep -c 'in VRAM)' "$d/engine.stderr" 2>/dev/null)" = "$NX" ] || w="$w VRAM lines != $NX"
  [ "$(grep -oE '^helper: device [0-9]+ holds' "$d/engine.stderr" 2>/dev/null | sort -u | wc -l)" = "$NM" ] || w="$w helper lines != $NM"
  grep -q "check FAILED" "$d/engine.stderr" 2>/dev/null && w="$w check FAILED"
  [ "$n" = bt ] && { ln "$d" '^batch-test B=8:' | grep -q . || w="$w no B=8 line"; }
  [ "$n" != bt ] && { ln "$d" "^decode: $n tokens" | grep -q . || w="$w no 'decode: $n tokens' line"; }
  echo "${w# }"; }
br() { if [ -z "$2" ]; then echo "$1"; else echo "[$1]"; fi; }
dec() { [ -d "$1" ] || { echo "-"; return; }; br "$(ln "$1" '^decode: [0-9]' | sed -nE 's/.*\(([0-9.]+) tok\/s\).*/\1/p')" "$(ok "$1" $N)"; }
pr() { [ -d "$1" ] || { echo "-"; return; }
  br "$(ln "$1" '^prompt: [0-9]' | sed -nE 's/^prompt: [0-9]+ tokens in ([0-9]+) ms \(([0-9.]+) tok\/s.*/\1 \2/p' | awk '{printf "%.2f s / %s", $1 / 1000, $2}')" "$(ok "$1" $NP)"; }
row() { local r=$1 lab=$2 o=$1/off pl ld notes="" x w b
  pl=$(grep -oE '\([0-9]+ of [0-9]+ in VRAM\)' "$o/engine.stderr" 2>/dev/null | tr -d '()' | uniq -c | awk '{printf "%s%s of %s%s", (n++ ? ", " : ""), $2, $4, ($1 > 1 ? " (x" $1 ")" : "")}')
  [ -n "$pl" ] && pl="XTX $pl"
  x=$(grep -E '^helper: device [0-9]+ holds' "$o/engine.stderr" 2>/dev/null | awk '!s[$3]++ {printf "%s%s", (n++ ? "+" : ""), $5}')
  [ -n "$x" ] && pl="$pl; MI50 $x"
  x=$(ln "$o" '^ds4: [0-9]+ routed experts on the CPU lane' | sed -nE 's/^ds4: ([0-9]+) routed experts on the CPU lane \(([0-9.]+) GB\).*/\1 (\2 GB)/p')
  [ -n "$x" ] && pl="$pl; CPU-lane set $x"
  ld=$(awk '$1 == "t0" {a = $2} $1 == "tprompt" && !b {b = $2} END {if (a && b) print b - a}' "$o/times.txt" 2>/dev/null)
  x=$(ln "$o" '^prompt: [0-9]' | sed -nE 's/^prompt: [0-9]+ tokens in ([0-9]+) ms.*/\1/p')
  [ -n "$ld" ] && [ -n "$x" ] && ld=$(awk -v a="$ld" -v b="$x" 'BEGIN {printf "%.0f", a - b / 1000}') || ld="-"
  x=$(ln "$o" '^decode: [0-9]' | sed -nE 's/^decode: ([0-9]+) tokens.*CPU lane: ([0-9]+) of.*/\2 \1/p' | awk '$2 > 0 {printf "%.1f", $1 / $2}')
  [ -n "$x" ] && notes="host-RAM experts $x a token"
  x=$(ln "$r/on" '^dspark: [0-9]' | sed -nE 's/.*\(([0-9.]+) tokens a round\).*/\1/p'); [ -n "$x" ] && notes="$notes; drafter $x tokens a round"
  w=$(cat "$r"/*/peak_w.txt 2>/dev/null | sort -n | tail -1); [ -n "$w" ] && [ "$w" != 0 ] && notes="$notes; peak $w W (boards)"
  for a in $ARMS; do n=$NP; case $a in off|on) n=$N ;; bt) n=bt ;; esac; x=$(ok "$r/$a" $n); [ -n "$x" ] && [ "$x" != - ] && notes="$notes; $a:$x"; done
  b="-"; [ -d "$r/bt" ] && b=$(br "$(ln "$r/bt" '^batch-test B=8:' | sed -nE 's/.*\(([0-9.]+) ms a step\).*greedy vs solo: ([0-9]+ of [0-9]+) identical.*/\1 (\2 identical)/p')" "$(ok "$r/bt" bt)")
  echo "| $lab | gpus $XL tp $NX; helpers ${ML:-none}; links $LINKS | ${pl:--} | $ld | $(dec "$r/off") | $(dec "$r/on") | $(pr "$r/p2k") | $(pr "$r/p8k") | $b | $(pr "$r/p32k") | ${notes#; } |"; }
HDR="| run | devices | experts in VRAM (off arm) | load s | decode off tok/s | decode on tok/s | 2K s / tok/s | 8K s / tok/s | B8 ms a step | 32K s / tok/s | notes |"
[ -s "$OUT/table.md" ] || { echo "$HDR"; echo "|---|---|---|---|---|---|---|---|---|---|---|"; } >> "$OUT/table.md"
ST=$(date -u +%Y%m%dT%H%M%SZ)
runset() { local lab=$1 envs=$2 r="$OUT/$LABEL-$1-$ST" a
  echo "== $LABEL $lab: $DEV | env: $envs" | tee -a "$OUT/log.txt"
  for a in $ARMS; do arm "$r" $a "$envs"
    [ $DRY = 0 ] && [ $a = "${ARMS%% *}" ] && ! grep -q '^prompt: ' "$r/$a/engine.stderr" 2>/dev/null && { echo "$lab: the first arm did not load; the other arms are skipped" | tee -a "$OUT/log.txt"; break; }
  done
  [ $DRY = 1 ] && return 0
  row "$r" "$LABEL $lab" | tee -a "$OUT/table.md"; }
drop() { local e=$1 kv=$2 nm=${2%%=*}; e=$(tr ' ' '\n' <<<"$e" | grep . | grep -v "^$nm=" | paste -sd' ')
  [[ "$kv" == *=* ]] && e="$e $kv"; echo "$e"; }
echo "$HDR"
if [ -z "$ABL" ]; then runset run "$ENVS"
else
  runset base "$ENVS"; all="$ENVS"
  for kv in $ABL; do [ $DRY = 0 ] && sleep "$GAP"; if [[ "$kv" == *=* ]]; then lab=${kv/=/:}; else lab=-$kv; fi; runset "$lab" "$(drop "$ENVS" "$kv")"; all=$(drop "$all" "$kv"); done
  [ $DRY = 0 ] && sleep "$GAP"; runset none "$all"
  [ $DRY = 0 ] && sleep "$GAP"; runset base2 "$ENVS"
fi
