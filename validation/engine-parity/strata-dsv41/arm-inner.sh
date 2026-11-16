#!/bin/bash
# arm-inner.sh ARM - inside the arm's container (one GPU, no network). As run on 2026-11-15/16, adapted to call the generalised
# tools of ../ (ref_llamacpp.py, select_tf.py, tf_filter.py) copied into /pkg; the run itself used their originals.  Exactly one GPU visible.
#   L : llama-server b11870 on 127.0.0.1:18080: -ngl 99 with the routed experts on the CPU (-ot exps=CPU), ctx = the longest prompt +
#       256, 1 slot, 32 threads, then ref_llamacpp.py (the rendered ids; greedy 256 tokens with the
#       top-20 at every position).
#   V / P1 (1 XTX + host-RAM experts, drafting not loaded: --dspark none), phases PHASES (default "smoke tf free"; one piece each
#   is fine, e.g. PHASES="smoke tf" tonight and PHASES=free later):
#       smoke  the longest selected prompt, --n 1 (does the context fit at 1 XTX?) - NO DATA stops the arm
#       tf     the 12 selected prompts (RES/sel12.txt), one engine process each: --ids @(prompt + L's tokens) --n 0 --window 2048
#              --dump-logits FIFO -> tf_filter.py (the distributions over L's continuation)
#       free   one --serve engine, every prompt L answered: GEN 256 temperature=0 spec=0 (greedy) -> sserve.py
#     V = upstream v0.2.9 (no DS4 env: its native config); P1 = build 17's binary with build17.env's BUILD_ENV (the served flags;
#     helper-only flags have nothing to do at 1 XTX) - the same layout as V.
set -u; ARM=$1; RES=/res; O=$RES/$ARM.inner.log; mkdir -p $RES/$ARM
exec > >(tee -a $O) 2>&1
rcw() { echo $1 > $RES/$ARM.inner.rc; exit $1; }
echo "inner $ARM $(date -u +%FT%TZ): devices $(ls /dev/dri | tr '\n' ' ')"
[ "$(ls /dev/dri | grep -c renderD)" = 1 ] || { echo "not exactly one render node visible"; rcw 5; }
V=129280
if [ $ARM = L ]; then
  export LD_LIBRARY_PATH=/lc/bin
  CTX=${LCTX:-81920}
  nice -n 19 /lc/bin/llama-server -m /m/new/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf --host 127.0.0.1 --port 18080 \
    -ngl 99 -ot "exps=CPU" -c $CTX -np 1 -t $NTH -b 2048 -ub 2048 --jinja > $RES/L/server.log 2>&1 &
  SP=$!
  for i in $(seq 1 360); do python3 -I -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=5)" 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  python3 -I -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=5)" 2>/dev/null || { echo "L: server not healthy"; tail -20 $RES/L/server.log; rcw 6; }
  echo "L: server healthy after $(( i * 5 )) s"
  timeout -k 60 ${LTMO:-21600} python3 -I /pkg/ref_llamacpp.py $RES/ids $RES http://127.0.0.1:18080; rc=$?
  kill $SP; wait $SP 2>/dev/null
  python3 -I /pkg/select_tf.py $RES > $RES/sel12.txt || rc=6
  cat $RES/sel12.txt; rcw $rc
fi
# ---- V / P1
ENVX=""; [ $ARM = P1 ] && ENVX="$(. /pkgenv/build17.env; echo $BUILD_ENV)"
COM="--pack /ds/pack-2.25 --profile-readonly --gpus 0 --tp 1 --dspark none --eos 1 --mmap-experts on"   # 2026-11-16: V OOM-killed (NUMA policy) copying 135 GB of experts in at load; map them instead (same weights)
eng() { timeout -k 30 ${TFTMO:-5400} env $ENVX DS4_NO_EOS=1 nice -n 19 /b/strata $COM "$@"; }   # (unattended: no hang)
ctx() { echo $(( ( ($1 + 512) / 4096 + 1) * 4096 )); }
nids() { tr ',' '\n' < $1 | grep -c .; }
PH=" ${PHASES:-smoke tf free} "
lst=$(cut -d' ' -f1 $RES/sel12.txt | grep -E '^[0-9]{2}$')
big=$(for n in $lst; do echo "$(nids $RES/ids/$n.ids) $n"; done | sort -n | tail -1 | cut -d' ' -f2)
bad=0
if [[ "$PH" == *" smoke "* ]]; then
echo "$ARM smoke: prompt $big ($(nids $RES/ids/$big.ids) ids), --n 1"
eng --max-context $(ctx $(nids $RES/ids/$big.ids)) --ids @$RES/ids/$big.ids --n 1 > $RES/$ARM/smoke.out 2> $RES/$ARM/smoke.err; s=$?
grep -E "prompt:|decode:|CPU lane|loaded in|DSpark|error|refus" $RES/$ARM/smoke.err | tail -6
[ $s = 0 ] || { echo "$ARM smoke rc $s: NO DATA (the context may not fit at 1 XTX)"; rcw 6; }
fi
# 2026-11-16 (HOSTLP): the forced continuation is read from $CONT (default: the L arm's own, as before) and
# written as $TFTAG-NN (default tf). HOSTLP: CONT=$RES/hostlp/forced TFTAG=htf PHASES=tf, forced/NN.json = {"tokens": [ids]}.
CONT=${CONT:-$RES/L}; TFTAG=${TFTAG:-tf}
[[ "$PH" == *" tf "* ]] && for n in $lst; do
  np=$(nids $RES/ids/$n.ids); python3 -I -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['tokens'])))" $CONT/$n.json > /tmp/cont
  { cat $RES/ids/$n.ids; printf ','; cat /tmp/cont; } | tr -d '\n' > /tmp/tf.ids
  rm -f /tmp/tf.fifo; mkfifo /tmp/tf.fifo
  python3 -I /pkg/tf_filter.py /tmp/tf.fifo $np $CONT/$n.json $RES/$ARM/$TFTAG-$n.json $V & fp=$!
  t0=$(date +%s)
  eng --max-context $(ctx $(( np + 300 ))) --ids @/tmp/tf.ids --n 0 --window 2048 --dump-logits /tmp/tf.fifo > /dev/null 2> $RES/$ARM/$TFTAG-$n.err; e=$?
  kill -0 $fp 2>/dev/null && { exec 3> /tmp/tf.fifo; exec 3>&-; }   # (an engine that never opened the FIFO: the reader gets EOF -> NO DATA)
  wait $fp; f=$?
  echo "$ARM $TFTAG $n: $np ids + $(tr ',' '\n' < /tmp/cont | grep -c .) forced, engine rc $e, filter rc $f, $(( $(date +%s) - t0 )) s | $(grep -h 'prompt:' $RES/$ARM/$TFTAG-$n.err | tail -1)"
  [ $e = 0 ] && [ $f = 0 ] || bad=1
done
if [[ "$PH" == *" free "* ]]; then
MAXC=$(for f in $RES/ids/*.ids; do nids $f; done | sort -n | tail -1)
STRATA_DS4_REUSE=0 timeout -k 60 ${FREETMO:-14400} python3 -I /pkg/sserve.py $RES $ARM env $ENVX nice -n 19 /b/strata $COM --serve --max-context $(ctx $MAXC) || bad=1
fi
rcw $([ $bad = 0 ] && echo 0 || echo 6)
