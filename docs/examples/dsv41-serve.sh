#!/usr/bin/env bash
# docs/examples/dsv41-serve.sh: DeepSeek V4.1 Flash as an OpenAI-compatible server on four RX 7900 XTX (tensor parallel) with
# four Instinct MI50 expert helpers, one request at a time.  An example, not a supported launcher: it is the launcher of build 11
# as served (docs/dsv41/cards/SV-06.md) with the paths made variables.  Later builds add the switches whose status in
# docs/dsv41/FEATURES.md is "build 12" .. "build 16"; pass them in HFLAGS / XENV (build 16's recommended set: SV-16.md).
# DS4_EC_FIX=1 is always set: without it the expert cache and DS4_HELPER_DD sum some expert picks twice (HX-28.md).
#
# Run it where the GPUs are visible (on the host or in a ROCm container with /dev/kfd and /dev/dri), from the repository root:
#   EXE=build/dsv41/strata-dsv41 PACK=<pack dir> bash docs/examples/dsv41-serve.sh
# For the MI50 clock pin of docs/dsv41/cards/CP-03.md, run it under tools/dsv41/with-mi50-pin.sh (as root).
#   EXE     the engine (tools/dsv41/build.sh)            PACK   the model pack (its tokenizer/ inside)
#   GPUS    the XTX ordinals, HELPER the MI50 ordinals    default: from tools/dsv41/devmap.sh (HIP order, read from sysfs)
#   PORT (8085)  HOST (127.0.0.1)  CTX (131072)  DSPARK (on)  CONF (0.3)  BLOCK (3)  SLOT_KEEP (1)  SLOT_GB (16)
#   HFLAGS  helper switches, DS4_HELPER_*=value (default: build 11's exact set, docs/dsv41/cards/HX-17.md and HX-18.md)
#   XENV    further NAME=value switches for the engine     XARGS  further engine arguments
#   TAB     an optional tuned rocBLAS table for the mHC mix (STRATA_ROCBLAS_TUNING, docs/dsv41/cards/FG-04.md)
#   RUNS    where ds.json and the logs go (default ./runs/dsv41-serve)
set -uo pipefail
: "${EXE:?EXE=<path to strata-dsv41>}" "${PACK:?PACK=<model pack directory>}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
if [ -z "${GPUS:-}" ] || [ -z "${HELPER:-}" ]; then
  eval "$(bash "$ROOT/tools/dsv41/devmap.sh" env)" || { echo "devmap failed: set GPUS and HELPER"; exit 2; }
  GPUS=${GPUS:-$XTX}; HELPER=${HELPER:-$MI50}
fi
TP=$(tr , '\n' <<<"$GPUS" | wc -l)
HFLAGS=${HFLAGS-"DS4_HELPER_OVERLAP=1 DS4_HELPER_BATCH=1 DS4_HELPER_DD=1 DS4_HELPER_DD_SKIP=1"}
XENV=${XENV:-}
PORT=${PORT:-8085}; HOST=${HOST:-127.0.0.1}; CTX=${CTX:-131072}
DSPARK=${DSPARK:-on}; CONF=${CONF:-0.3}; BLOCK=${BLOCK:-3}
SLOT_KEEP=${SLOT_KEEP:-1}; SLOT_GB=${SLOT_GB:-16}; SHARE=${DS4_BATCH_PREFILL_SHARE:-0.5}
XARGS=${XARGS:-}; TAB=${TAB:-}
[ -x "$EXE" ] || { echo "EXE $EXE: no such build"; exit 2; }
D=${RUNS:-$PWD/runs/dsv41-serve}; mkdir -p "$D"
S=""; [ "$DSPARK" != none ] && S=", \"--dspark-conf\", \"$CONF\", \"--dspark-block\", \"$BLOCK\""
X=""; for a in $XARGS; do X="$X, \"$a\""; done
E=""; for kv in $HFLAGS $XENV; do case "$kv" in *=*) E="$E \"${kv%%=*}\": \"${kv#*=}\","; ;; *) echo "HFLAGS / XENV: $kv is not NAME=value"; exit 2 ;; esac; done
[ -n "$TAB" ] && E="$E \"STRATA_ROCBLAS_TUNING\": \"$TAB\","
cat > "$D/ds.json" <<EOF
{ "exe": "$EXE", "backend": "hip", "arch": "deepseek41", "model_name": "deepseek-v4.1-flash",
  "tokenizer": "$PACK/tokenizer", "log": "$D/engine.log", "engine_ready_s": 1800,
  "env": { "STRATA_DS4_SLOT_KEEP": "$SLOT_KEEP", "STRATA_DS4_SLOT_GB": "$SLOT_GB", "DS4_BATCH_PREFILL_SHARE": "$SHARE",$E
           "DS4_HC_FUSE": "2", "DS4_ROUTE_FUSE": "2", "DS4_DECODE_TRIM": "7", "DS4_ROUTER_BF16": "1",
           "DS4_EC_FIX": "1" },
  "args": ["--pack", "$PACK", "--max-context", "$CTX", "--eos", "1", "--vision", "off", "--profile-readonly",
           "--gpus", "$GPUS", "--tp", "$TP", "--helper-device", "$HELPER", "--dspark", "$DSPARK"$S$X] }
EOF
python3 -c "import json, sys; json.load(open(sys.argv[1]))" "$D/ds.json" || { echo "ds.json is not JSON"; exit 2; }
cd "$ROOT" || exit 1
python3 -c "import jinja2" 2>/dev/null || { echo "jinja2 missing: pip install -r requirements.txt"; exit 1; }
echo "DeepSeek V4.1 Flash $HOST:$PORT gpus=$GPUS tp=$TP helper=$HELPER dspark=$DSPARK conf=$CONF block=$BLOCK ctx=$CTX hflags='$HFLAGS' xenv='$XENV' exe=$EXE -> $D"
exec python3 -m serve.server --engine strata --config "$D/ds.json" --host "$HOST" --port "$PORT" > "$D/server.log" 2>&1
