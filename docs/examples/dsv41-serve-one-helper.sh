#!/usr/bin/env bash
# docs/examples/dsv41-serve-one-helper.sh: DeepSeek V4.1 Flash as an OpenAI-compatible server on one RX 7900 XTX with one
# Instinct MI50 expert helper (the first layout, docs/dsv41/cards/EB-03.md).  An example, not a supported launcher.
# Run it where the GPUs are visible, from the repository root:
#   EXE=build/dsv41/strata-dsv41 PACK=<pack dir> GPUS=0 HELPER=1 bash docs/examples/dsv41-serve-one-helper.sh
#   PORT (8085)  HOST (127.0.0.1)  CTX (262144)  GPUS (layer hosts)  HELPER (MI50 ordinal; "" = none)  RUNS (./runs/dsv41-one)
# GPUS / HELPER are HIP ordinals: tools/dsv41/devmap.sh prints which is which.
# Settings follow setup.py's DeepSeek config: --eos 1, --dspark auto + --dspark-conf 0.65, --vision off.
set -uo pipefail
: "${EXE:?EXE=<path to strata-dsv41>}" "${PACK:?PACK=<model pack directory>}" "${GPUS:?GPUS=<XTX ordinal(s)>}"
PORT=${PORT:-8085}; HOST=${HOST:-127.0.0.1}; CTX=${CTX:-262144}; HELPER=${HELPER-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
D=${RUNS:-$PWD/runs/dsv41-one}; mkdir -p "$D"
H=""; [ -n "$HELPER" ] && H=", \"--helper-device\", \"$HELPER\""
cat > "$D/ds.json" <<EOF
{ "exe": "$EXE", "backend": "hip", "arch": "deepseek41", "model_name": "deepseek-v4.1-flash",
  "tokenizer": "$PACK/tokenizer", "log": "$D/engine.log", "engine_ready_s": 3600,
  "args": ["--pack", "$PACK", "--max-context", "$CTX", "--eos", "1", "--vision", "off", "--dspark", "auto",
           "--dspark-conf", "0.65", "--tp", "1", "--gpus", "$GPUS"$H] }
EOF
cd "$ROOT" || exit 1
python3 -c "import jinja2" 2>/dev/null || { echo "jinja2 missing: pip install -r requirements.txt"; exit 1; }
echo "DeepSeek V4.1 Flash $HOST:$PORT gpus=$GPUS helper=${HELPER:-none} ctx=$CTX exe=$EXE -> $D"
exec python3 -m serve.server --engine strata --config "$D/ds.json" --host "$HOST" --port "$PORT" > "$D/server.log" 2>&1
