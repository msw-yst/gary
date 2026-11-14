#!/usr/bin/env bash
# tools/dsv41/gates/gates.sh - the ordered test gates of the gfx906 helper (docs/dsv41/cards/EB-01.md):
# an RX 7900 XTX (gfx1100) primary with an MI50 (gfx906, wave64) as a helper expert cache, one fat HIP binary.
#
#   bash /strata/gates/gates.sh [GATE ...]        GATE: gate0..gate5 (or 0..5), default all; always run in ascending order,
#                                            stopping at the first FAIL
#
# Runs INSIDE a ROCm container (README.md: Running).  HIP_VISIBLE_DEVICES comes from the caller and must
# make HIP device 0 the XTX and device 1 the MI50; this script checks that from rocminfo before anything else and
# refuses otherwise.  It never sets HSA_OVERRIDE_GFX_VERSION and refuses the global one if it is set.
#
# Logs: /strata/gates/<UTC timestamp>/ (gates.log = this console, summary.txt = one PASS/FAIL line per gate, one log per
# command).  Configuration: environment variables GATE_* (README.md lists them); the defaults are the container layout in README.md.
set -uo pipefail
umask 022

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS="$HERE/gate_tools.py"
Y=${GATE_Y:-/strata}
M=${GATE_M:-/gguf}
BUILD=${GATE_BUILD:-$Y/build/helper}
REFBUILD=${GATE_REF_BUILD:-$Y/build/gfx906}
PROBE=${GATE_PROBE:-$Y/probe/probe_mi50}
SHARD1=${GATE_SHARD1:-$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf}
SHARD2=${GATE_SHARD2:-$M/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf}
G2_LAYERS=${GATE_G2_LAYERS:-0 10 20 30 47}
G2_PAIRS=${GATE_G2_PAIRS:-q4_K/q5_1 iq3_s/iq4_nl}
G4_ARMS=${GATE_G4_ARMS:-a b c d}
G4_TOKENS=${GATE_G4_TOKENS:-512}
G5_PROMPT=${GATE_G5_PROMPT:-8192}
G5_GEN=${GATE_G5_GEN:-256}
G5_REPEATS=${GATE_G5_REPEATS:-3}
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=${GATE_LOG_DIR:-$Y/gates/$TS}

mkdir -p "$LOG" || { echo "cannot create $LOG"; exit 2; }
exec > >(tee -a "$LOG/gates.log") 2>&1
SUMMARY="$LOG/summary.txt"
MAP="$LOG/devices.json"

say()  { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
rule() { printf '\n==================== %s ====================\n' "$*"; }
record() { echo "$*" >> "$SUMMARY"; }
gate_pass() { local g=$1; shift; say "PASS $g: $*"; record "PASS $g ($(( $(date +%s) - GATE_T0 )) s): $*"; }
gate_fail() {
    local g=$1; shift
    say "FAIL $g: $*"
    record "FAIL $g ($(( $(date +%s) - GATE_T0 )) s): $*"
    say "stopping at the first failing gate; logs in $LOG"
    exit 1
}
refuse() { say "REFUSED: $1"; record "REFUSED: $1"; exit "${2:-2}"; }
tail_log() { echo "---- last lines of $1 ----"; tail -n "${2:-25}" "$1"; echo "----"; }
# guard NAME TIMEOUT_S CMD... : CMD with a timeout and the thermal abort, its output in $LOG/NAME.log
guard() { local n=$1 t=$2; shift 2; python3 "$TOOLS" guard --map "$MAP" --timeout "$t" --log "$LOG/$n.log" -- "$@"; }
need_exe() { [ -x "$1" ] || gate_fail "$CUR" "missing executable $1"; }

# ------------------------------------------------------------------------------------------------ gate selection
WANT=()
[ $# -eq 0 ] && set -- all
for g in "$@"; do
    case "${g,,}" in
        all) WANT+=(0 1 2 3 4 5) ;;
        gate[0-5]|g[0-5]|[0-5]) WANT+=("${g: -1}") ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown gate '$g' (gate0..gate5 or all)"; exit 2 ;;
    esac
done
mapfile -t WANT < <(printf '%s\n' "${WANT[@]}" | sort -un)

rule "gfx906 helper gates: ${WANT[*]/#/gate}"
say "log dir $LOG"
say "build $BUILD; reference build $REFBUILD; probe $PROBE"
say "model $SHARD1 (+ $(basename "$SHARD2"))"
say "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"
echo "gates ${WANT[*]} started $TS; HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}" > "$SUMMARY"

# ------------------------------------------------------------------------------------------------ preflight
command -v python3 >/dev/null || refuse "python3 is not in the image (gate_tools.py needs it)"
python3 "$TOOLS" devmap --out "$MAP" || refuse "device map: HIP 0 must be gfx1100 and HIP 1 gfx906 (see above)" 3
XTX_TOK=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hip"][0]["visible_token"])' "$MAP")
MI50_TOK=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hip"][1]["visible_token"])' "$MAP")
say "XTX alone: HIP_VISIBLE_DEVICES=$XTX_TOK; MI50 alone: HIP_VISIBLE_DEVICES=$MI50_TOK"
if ! python3 "$TOOLS" temps --map "$MAP"; then
    [ "${GATE_ALLOW_NO_THERMAL:-0}" = 1 ] || refuse "no readable GPU temperature sensor (/sys/class/kfd topology -> \
/sys/class/drm/renderD*/device/hwmon): the thermal abort cannot work.  GATE_ALLOW_NO_THERMAL=1 runs without it"
    say "WARNING: running without the thermal abort (GATE_ALLOW_NO_THERMAL=1)"
fi
say "thermal abort per card (XTX 102 C, MI50 95 C; GATE_TEMP_ABORT_C overrides), hottest sensor; stall = no new output for the gate window"

# ------------------------------------------------------------------------------------------------ gate0
g0() {
    need_exe "$PROBE"
    guard gate0-probe 300 "$PROBE"
    local rc=$?
    tail_log "$LOG/gate0-probe.log" 40
    local line
    line=$(python3 "$TOOLS" g0 --log "$LOG/gate0-probe.log" --rc "$rc" --map "$MAP")
    echo "$line" | tee "$LOG/gate0-verdict.txt"
    case "$(tail -n1 <<<"$line")" in
        PASS*) gate_pass gate0 "$(tail -n1 <<<"$line" | sed 's/^PASS gate0: //')" ;;
        *) gate_fail gate0 "$(tail -n1 <<<"$line" | sed 's/^FAIL gate0: //')" ;;
    esac
}

# ------------------------------------------------------------------------------------------------ gate1
g1() {
    need_exe "$BUILD/strata-device"
    guard gate1-list-devices 120 "$BUILD/strata-device" --list-devices
    local rc=$?
    cat "$LOG/gate1-list-devices.log"
    local out
    out=$(python3 "$TOOLS" g1 --log "$LOG/gate1-list-devices.log" --rc "$rc")
    echo "$out"
    case "$(tail -n1 <<<"$out")" in
        PASS*) gate_pass gate1 "$(tail -n1 <<<"$out" | sed 's/^PASS gate1: //')" ;;
        *) gate_fail gate1 "$(tail -n1 <<<"$out" | sed 's/^FAIL gate1: //')" ;;
    esac
}

# ------------------------------------------------------------------------------------------------ gate2
# One parity run on the MI50 alone: NAME EXPECT(ok|refuse1) ENV... -- CMD...  ->  G2_FAILS / G2_LINES
g2_run() {
    local name=$1 expect=$2; shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    guard "gate2-$name" 1800 env "${envs[@]}" "$@"
    local rc=$? log="$LOG/gate2-$name.log" verdict
    if [ "$expect" = refuse1 ]; then
        if [ $rc -eq 1 ] && grep -q 'STRATA_EXP_MODE=1 (W64' "$log"; then verdict="ok (refused, rc 1)"
        else verdict="FAIL (rc $rc; the mode-1 refusal message was expected)"; G2_FAILS+=("$name"); fi
    else
        local summ
        summ=$(grep -Eo '(native_expert_parity|native_grouped_parity): [0-9]+ failures' "$log" | tail -n1)
        if [ $rc -eq 0 ] && [ -n "$summ" ] && grep -q ': 0 failures' <<<"$summ"; then verdict="ok ($summ)"
        else verdict="FAIL (rc $rc; ${summ:-no summary line})"; G2_FAILS+=("$name"); fi
    fi
    say "  gate2 $name: $verdict"
    grep -E 'cpu rel .* gpu rel|gpu decode-once|q5_1 min|FAIL' "$log" | grep -v '^##' | sed 's/^/      /' | head -n 20
    G2_LINES+=("$name: $verdict")
}

g2() {
    need_exe "$BUILD/native_expert_parity"
    need_exe "$BUILD/native_grouped_parity"
    [ -r "$SHARD1" ] || gate_fail gate2 "the model shard $SHARD1 is not readable"
    G2_FAILS=(); G2_LINES=()
    local mode
    # shellcheck disable=SC2086
    for mode in 0 7; do
        g2_run "expert-synthetic-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
            "$BUILD/native_expert_parity" --synthetic $G2_PAIRS
        g2_run "expert-shard-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
            "$BUILD/native_expert_parity" "$SHARD1" $G2_LAYERS
        g2_run "grouped-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
            "$BUILD/native_grouped_parity"
    done
    # WP3's extra checks: the one-pass SwiGLU + q8_1 kernel on wave64, and mode 1 refused on the wave32 card
    g2_run "grouped-m7-swiglu-fused" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=7 STRATA_HIP_SWIGLU_FUSED=1 -- \
        "$BUILD/native_grouped_parity"
    g2_run "xtx-mode1-refused" refuse1 HIP_VISIBLE_DEVICES="$XTX_TOK" STRATA_EXP_MODE=1 -- "$BUILD/native_grouped_parity"

    # the reference: the older pure-gfx906 build's tools, if they were built there (informative, not gating)
    local ref_note
    if [ -x "$REFBUILD/native_expert_parity" ]; then
        local REF_FAILS_BEFORE=${#G2_FAILS[@]}
        for mode in 0 7; do
            g2_run "ref-expert-synthetic-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
                "$REFBUILD/native_expert_parity" --synthetic $G2_PAIRS
            g2_run "ref-expert-shard-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
                "$REFBUILD/native_expert_parity" "$SHARD1" $G2_LAYERS
            if [ -x "$REFBUILD/native_grouped_parity" ]; then
                g2_run "ref-grouped-m$mode" ok HIP_VISIBLE_DEVICES="$MI50_TOK" STRATA_EXP_MODE=$mode -- \
                    "$REFBUILD/native_grouped_parity"
            fi
            # the same error figures from both builds? (the shims differ only in integer-exact primitives)
            local w
            for w in expert-synthetic expert-shard; do
                if diff -q <(grep -E 'cpu rel .* gpu rel' "$LOG/gate2-$w-m$mode.log") \
                           <(grep -E 'cpu rel .* gpu rel' "$LOG/gate2-ref-$w-m$mode.log") >/dev/null; then
                    say "  gate2 helper vs reference build, $w mode $mode: identical cpu/gpu error figures"
                else
                    say "  gate2 helper vs reference build, $w mode $mode: the error figures DIFFER (see the two logs)"
                fi
            done
        done
        local REF_FAILS=$(( ${#G2_FAILS[@]} - REF_FAILS_BEFORE ))
        G2_FAILS=("${G2_FAILS[@]:0:$REF_FAILS_BEFORE}")   # the reference is reported, not gating
        ref_note="reference build: $REF_FAILS failing run(s)"
        [ -x "$REFBUILD/native_grouped_parity" ] || ref_note+=", no native_grouped_parity there (skipped)"
    else
        ref_note="reference build: $REFBUILD/native_expert_parity not built (skipped)"
        say "  gate2 $ref_note"
    fi
    if [ ${#G2_FAILS[@]} -gt 0 ]; then
        gate_fail gate2 "failing on the MI50: ${G2_FAILS[*]}; $ref_note"
    fi
    gate_pass gate2 "MI50 alone, STRATA_EXP_MODE 0 and 7: expert parity synthetic ($G2_PAIRS) and shard layers ($G2_LAYERS), \
grouped parity (+ fused SwiGLU), mode 1 refused on the XTX; $ref_note"
}

# ------------------------------------------------------------------------------------------------ gate3
g3() {
    need_exe "$BUILD/cross_device_grouped_parity"
    # shellcheck disable=SC2086
    guard gate3-cross 1200 "$BUILD/cross_device_grouped_parity" 0 1 $G2_PAIRS
    local rc=$?
    # shellcheck disable=SC2086
    guard gate3-cross-strict 1200 "$BUILD/cross_device_grouped_parity" 0 1 $G2_PAIRS --strict
    local rcs=$?
    grep -vE '^##' "$LOG/gate3-cross.log" | head -n 120
    local out
    out=$(python3 "$TOOLS" g3 --log "$LOG/gate3-cross.log" --rc "$rc" --strict-log "$LOG/gate3-cross-strict.log" --strict-rc "$rcs")
    echo "$out"
    case "$(tail -n1 <<<"$out")" in
        PASS*) gate_pass gate3 "$(tail -n1 <<<"$out" | sed 's/^PASS gate3: //')" ;;
        *) gate_fail gate3 "$(tail -n1 <<<"$out" | sed 's/^FAIL gate3: //')" ;;
    esac
}

# ------------------------------------------------------------------------------------------------ engine prerequisites
PREP_DONE=0
prep_engine() {
    [ $PREP_DONE = 1 ] && return 0
    need_exe "$BUILD/strata"
    [ -r "$SHARD1" ] && [ -r "$SHARD2" ] || gate_fail "$CUR" "model shards not readable under $M"
    # the source tree the helper build was configured from (tools/, data/expert-profile.bin)
    SRC=${GATE_SRC:-}
    if [ -z "$SRC" ] && [ -f "$BUILD/CMakeCache.txt" ]; then
        SRC=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")
    fi
    [ -n "$SRC" ] && [ -f "$SRC/tools/strata_tokenizer.py" ] || \
        gate_fail "$CUR" "the Strata source tree was not found (CMAKE_HOME_DIRECTORY of $BUILD/CMakeCache.txt = '${SRC}'); set GATE_SRC"
    PROFILE=${GATE_PROFILE:-$SRC/data/expert-profile.bin}
    [ -f "$PROFILE" ] || gate_fail "$CUR" "no expert profile at $PROFILE (GATE_PROFILE)"
    say "source tree $SRC; expert profile $PROFILE"

    # the native pack (tools/iq_pack.py) cut from exactly this GGUF
    PACK=${GATE_PACK:-}
    if [ -n "$PACK" ]; then
        python3 "$TOOLS" pack-check --src "$SRC" --pack "$PACK" --gguf "$SHARD1" || gate_fail "$CUR" "GATE_PACK=$PACK does not match the model"
    else
        local cand
        while IFS= read -r cand; do
            cand=$(dirname "$cand")
            if python3 "$TOOLS" pack-check --src "$SRC" --pack "$cand" --gguf "$SHARD1"; then PACK=$cand; break; fi
        done < <(find "$Y" -maxdepth 4 \( -path "$Y/containers" -o -path "$Y/build" -o -path "$Y/gates" \) -prune \
                      -o -name native_experts.txt -print 2>/dev/null)
    fi
    if [ -z "$PACK" ]; then
        [ "${GATE_BUILD_PACK:-1}" = 1 ] || gate_fail "$CUR" "no native pack for $SHARD1 under $Y (GATE_PACK, or GATE_BUILD_PACK=1)"
        PACK=${GATE_PACK_OUT:-$Y/packs/iq3_s-gates}
        say "no matching native pack under $Y: building one at $PACK (tools/iq_pack.py, CPU only; log pack-build.log)"
        python3 "$SRC/tools/iq_pack.py" --gguf "$SHARD1" --out "$PACK" > "$LOG/pack-build.log" 2>&1 || {
            tail_log "$LOG/pack-build.log"; gate_fail "$CUR" "iq_pack.py failed (needs numpy; see pack-build.log)"; }
        python3 "$TOOLS" pack-check --src "$SRC" --pack "$PACK" --gguf "$SHARD1" || gate_fail "$CUR" "the new pack does not check"
    fi
    say "native pack $PACK"

    # the MTP drafter runtime (tools/mtp_rt.py output: dense.txt + experts.bin) - optional
    MTP=${GATE_MTP:-}
    if [ "$MTP" = none ]; then
        MTP=""
    elif [ -z "$MTP" ]; then
        local found=()
        while IFS= read -r cand; do
            cand=$(dirname "$cand")
            [ -f "$cand/experts.bin" ] && [ ! -f "$cand/native_experts.txt" ] && found+=("$cand")
        done < <(find "$Y" -maxdepth 4 \( -path "$Y/containers" -o -path "$Y/build" -o -path "$Y/gates" \) -prune \
                      -o -name dense.txt -print 2>/dev/null)
        if [ ${#found[@]} -gt 1 ]; then gate_fail "$CUR" "several MTP runtimes under $Y (${found[*]}): choose one with GATE_MTP (or none)"; fi
        MTP=${found[0]:-}
    fi
    [ -z "$MTP" ] || [ -f "$MTP/dense.txt" ] || gate_fail "$CUR" "GATE_MTP=$MTP has no dense.txt"
    say "MTP drafter: ${MTP:-none (no --mtp: verify windows draft from prompt lookup only)}"

    BASE=( "$BUILD/strata" --serve --pack "$PACK" --native "$SHARD1" --expert-profile "$PROFILE"
           --spec 4 --spec-min-p 0.5 --prompt-cache 0 --conversation-cache-mib 0 )
    [ -n "$MTP" ] && BASE+=( --mtp "$MTP" )
    [ "${GATE_MMAP_EXPERTS:-0}" = 1 ] && BASE+=( --mmap-experts )
    # shellcheck disable=SC2206
    [ -n "${GATE_ENGINE_EXTRA:-}" ] && BASE+=( $GATE_ENGINE_EXTRA )
    PREP_DONE=1
}

make_ids() {   # NAME TEXT N OVERRIDE -> $LOG/NAME.ids
    local out="$LOG/$1.ids"
    if [ -n "$4" ]; then
        [ -f "$4" ] || gate_fail "$CUR" "ids file $4 not found"
        python3 -c 'import sys; v=open(sys.argv[1]).read().replace(","," ").split(); n=int(sys.argv[3]);
assert len(v)>=n, f"{sys.argv[1]}: {len(v)} ids < {n}"; open(sys.argv[2],"w").write(",".join(v[:n])+"\n")' \
            "$4" "$out" "$3" || gate_fail "$CUR" "ids file $4 is too short"
        return
    fi
    python3 "$TOOLS" ids --src "$SRC" --gguf "$SHARD1" --tokenizer-dir "$PACK/tokenizer" --text "$2" --n "$3" --out "$out" \
        || gate_fail "$CUR" "tokenizing $2 failed"
}

# serve_arm PREFIX REQUESTS  (env: ARM_ENV array, args: ARM_ARGS array)
serve_arm() {
    python3 "$TOOLS" serve --map "$MAP" --out "$1" --requests "$2" -- \
        env ${ARM_ENV[@]+"${ARM_ENV[@]}"} "${BASE[@]}" ${ARM_ARGS[@]+"${ARM_ARGS[@]}"}
}

# ------------------------------------------------------------------------------------------------ gate4
g4() {
    prep_engine
    make_ids text-code "$HERE/text-code.txt" "$G4_TOKENS" "${GATE_IDS_CODE:-}"
    printf '[{"name": "code%s", "max_new": 1, "keys": "ckpt=0", "ids_file": "%s"}]\n' "$G4_TOKENS" "$LOG/text-code.ids" \
        > "$LOG/gate4-requests.json"
    local common=( --prefill 512 --short-read $(( G4_TOKENS + 128 )) --max-context $(( G4_TOKENS * 2 + 1024 )) --kv fp16
                   --adapt-every 100000 --pcie-frac 0 --suffix-draft 0 )
    local envc=( STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_PCIE_FRAC_DEFAULT=old STRATA_LOGPOS_TOPK=256 )
    local arm rc failed=() ran=()
    for arm in $G4_ARMS; do
        ARM_ENV=( "${envc[@]}" "STRATA_LOGPOS=$LOG/gate4-$arm.tsv" )
        ARM_ARGS=( --expert-cache auto "${common[@]}" )
        case $arm in
            a) ;;
            b) ARM_ARGS+=( --expert-cache-device1 auto ) ;;
            c) ARM_ARGS+=( --expert-cache-device1 auto ); ARM_ENV+=( STRATA_REMOTE_ZEROCOPY=1 ) ;;   # zero-copy forced on: the default for a wave64 helper is off
            d) ARM_ARGS+=( --expert-cache-device1 auto --remote-expert-opt ) ;;
            *) gate_fail gate4 "unknown arm '$arm' (a b c d)" ;;
        esac
        rm -f "$LOG/gate4-$arm.tsv"
        say "gate4 arm $arm: ${ARM_ENV[*]} ... ${ARM_ARGS[*]}"
        serve_arm "$LOG/gate4-$arm" "$LOG/gate4-requests.json"
        rc=$?
        ran+=("$arm")
        if [ $rc -ne 0 ]; then
            tail_log "$LOG/gate4-$arm.stderr.txt" 30
            failed+=("$arm(rc $rc)")
            [ "$arm" = a ] && gate_fail gate4 "the reference arm a failed (rc $rc)"
        fi
    done
    [ "${ran[0]:-}" = a ] || gate_fail gate4 "GATE_G4_ARMS must start with a (the reference)"
    local runs=()
    for arm in "${ran[@]:1}"; do runs+=("$LOG/gate4-$arm.tsv"); done
    [ ${#runs[@]} -gt 0 ] || gate_fail gate4 "no helper arm in GATE_G4_ARMS"
    python3 "$HERE/compare_logits.py" --ref "$LOG/gate4-a.tsv" "${runs[@]}" --json "$LOG/gate4-compare.json" \
        --min-top1 "${GATE_G4_MIN_TOP1:-95}" --max-median-kl "${GATE_G4_MAX_MEDIAN_KL:-1e-3}" \
        --min-positions $(( G4_TOKENS * 3 / 4 )) | tee "$LOG/gate4-compare.txt"
    local rc_cmp=${PIPESTATUS[0]}
    python3 "$TOOLS" g4 --dir "$LOG" --arms "${ran[@]}" | tee "$LOG/gate4-entries.txt"
    local rc_ent=${PIPESTATUS[0]}
    local tab
    tab=$(awk 'NR>2 {printf "%s%s", sep, $0; sep=" | "}' "$LOG/gate4-compare.txt" | tr -s ' ')
    if [ ${#failed[@]} -gt 0 ] || [ "$rc_cmp" -ne 0 ] || [ "$rc_ent" -ne 0 ]; then
        gate_fail gate4 "arms failed: ${failed[*]:-none}; compare rc $rc_cmp; entries rc $rc_ent | $tab"
    fi
    gate_pass gate4 "$(tail -n1 "$LOG/gate4-entries.txt") | $tab"
}

# ------------------------------------------------------------------------------------------------ gate5
g5() {
    prep_engine
    make_ids text-8k "$HERE/text-8k.txt" "$G5_PROMPT" "${GATE_IDS_8K:-}"
    python3 - "$LOG" "$G5_GEN" "$G5_REPEATS" <<'EOF' || gate_fail gate5 "cannot write the requests"
import json, sys
log, gen, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = open(f"{log}/text-8k.ids").read().strip().split(",")
open(f"{log}/text-8k-warm.ids", "w").write(",".join(ids[:512]) + "\n")
r = [{"name": "warm", "max_new": 32, "keys": "ckpt=0", "ids_file": f"{log}/text-8k-warm.ids"}]
r += [{"name": f"rep{i + 1}", "max_new": gen, "keys": "ckpt=0", "ids_file": f"{log}/text-8k.ids"} for i in range(reps)]
json.dump(r, open(f"{log}/gate5-requests.json", "w"), indent=1)
EOF
    local arm rc
    for arm in a b; do
        # shellcheck disable=SC2206
        ARM_ENV=( STRATA_DECODE_TIMING=1 ${GATE_G5_ENV:-} )
        ARM_ARGS=( --expert-cache ${GATE_G5_CACHE:-auto} --prefill auto --max-context $(( G5_PROMPT + G5_GEN + 1024 )) )   # GATE_G5_CACHE: cap the XTX cache (slots) to emulate a model that does not fit
        [ $arm = b ] && ARM_ARGS+=( --expert-cache-device1 auto )
        say "gate5 arm $arm: ${ARM_ENV[*]} ... ${ARM_ARGS[*]}"
        serve_arm "$LOG/gate5-$arm" "$LOG/gate5-requests.json"
        rc=$?
        if [ $rc -ne 0 ]; then tail_log "$LOG/gate5-$arm.stderr.txt" 30; gate_fail gate5 "arm $arm failed (rc $rc)"; fi
    done
    local out
    out=$(python3 "$TOOLS" g5 --a "$LOG/gate5-a" --b "$LOG/gate5-b" --repeats "$G5_REPEATS" --json "$LOG/gate5-report.json")
    echo "$out" | tee "$LOG/gate5-report.txt"
    case "$(tail -n1 <<<"$out")" in
        PASS*) gate_pass gate5 "$(tail -n1 <<<"$out" | sed 's/^PASS gate5: //')" ;;
        *) gate_fail gate5 "$(tail -n1 <<<"$out" | sed 's/^FAIL gate5: //')" ;;
    esac
}

# ------------------------------------------------------------------------------------------------ run
T_ALL=$(date +%s)
for n in "${WANT[@]}"; do
    CUR="gate$n"
    rule "$CUR"
    python3 "$TOOLS" temps --map "$MAP" || true
    GATE_T0=$(date +%s)
    "g$n"
done
rule "done"
say "all selected gates passed in $(( $(date +%s) - T_ALL )) s; logs in $LOG"
cat "$SUMMARY"
exit 0
