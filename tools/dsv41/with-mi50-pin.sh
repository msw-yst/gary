#!/bin/bash
# tools/dsv41/with-mi50-pin.sh CMD [ARG ...]: run CMD with every Instinct MI50 (gfx906, PCI device 0x66a0) at
# power_dpm_force_performance_level=high, and put them back to auto when CMD ends: on any exit status, and on INT / TERM / HUP.
# This is the MI50 clock pin of docs/dsv41/cards/CP-03.md (and ME-09); power caps, clock tables and everything else are left
# alone.  Idle MI50s go back to auto, so idle power is saved.  Run it as root (it writes amdgpu's sysfs files).
#   - The MI50s come from sysfs (vendor 0x1002, device 0x66a0, amdgpu bound).  PIN_EXPECT="<bdf> <bdf> ..." (optional) makes the
#     wrapper refuse (exit 2, nothing changed, CMD not run) unless the cards found are exactly those.
#   - CMD runs as a background job in its own process group (job control on, so INT is not ignored in it).  A signal to this
#     wrapper goes on to that group; the wrapper waits for the group to end (KILL after PIN_GRACE s, default 180), then restores.
#   - Holders: several wrappers at once (or nested) share the pin; each registers its pid under a lock, and the last one out
#     restores auto (dead holders are dropped, so a wrapper killed with KILL is cleaned up by the next one).
#   - PIN_MODE=high (default) or sclk8: manual, then the top sclk and mclk levels only (docs/dsv41/cards/CP-04.md: measured, not
#     kept).  Restored as: every level in both masks, then auto.
#   - Exit status: CMD's (128 + n if it ended on signal n); 2 if the pin could not be set.  Log and state: PIN_DIR (default beside
#     this script), PIN_LOG (default $PIN_DIR/with-mi50-pin.log).
set -u
# (started as a background job of a non-interactive shell, INT arrives ignored, and bash cannot trap a signal ignored at entry:
#  start again once with INT / TERM / HUP at their defaults, so the traps below always take)
[ -z "${_MI50PIN_EXEC:-}" ] && exec env --default-signal=INT,TERM,HUP _MI50PIN_EXEC=1 bash "$0" "$@"
unset _MI50PIN_EXEC
D=$(cd "$(dirname "$0")" && pwd)
PIN_DIR=${PIN_DIR:-$D}; PIN_EXPECT=${PIN_EXPECT:-}
PIN_LOG=${PIN_LOG:-$PIN_DIR/with-mi50-pin.log}; PIN_GRACE=${PIN_GRACE:-180}
HOLD=$PIN_DIR/.mi50-pin.holders; LOCK=$PIN_DIR/.mi50-pin.lock
[ $# -ge 1 ] || { echo "usage: $0 CMD [ARG ...]" >&2; exit 2; }
log() { local m; m="$(date -u +%FT%TZ) with-mi50-pin[$$]: $*"; echo "$m" >> "$PIN_LOG"; ( echo "$m" >&2 ) 2> /dev/null; }   # (the file first; stderr in a subshell: a closed reader cannot kill the wrapper)
lvls() { local b o=""; for b in $BDFS; do o="$o ${b#0000:}=$(cat /sys/bus/pci/devices/$b/power_dpm_force_performance_level 2>/dev/null || echo ?)"; done; echo "${o# }"; }

# the MI50s, from sysfs
BDFS=""
for d in /sys/bus/pci/devices/*; do
  [ "$(cat $d/vendor 2>/dev/null)" = 0x1002 ] && [ "$(cat $d/device 2>/dev/null)" = 0x66a0 ] || continue
  [ -f $d/power_dpm_force_performance_level ] || { echo "with-mi50-pin: $(basename $d) is an MI50 without amdgpu's dpm level" >&2; exit 2; }
  BDFS="$BDFS $(basename $d)"
done
BDFS=$(echo $BDFS | tr ' ' '\n' | sort | paste -sd' ')
[ -n "$BDFS" ] || { echo "with-mi50-pin: no MI50 (0x66a0) found in sysfs: nothing changed, CMD not run" >&2; exit 2; }
if [ -n "$PIN_EXPECT" ] && [ "$BDFS" != "$(echo $PIN_EXPECT | tr ' ' '\n' | sort | paste -sd' ')" ]; then
  log "REFUSED: sysfs MI50s (0x66a0) are '$BDFS', not PIN_EXPECT '$PIN_EXPECT' - nothing changed, CMD not run"; exit 2
fi

setlvl() {   # $1: high | auto | manual; every card read back
  local v=$1 b bad=0
  for b in $BDFS; do
    echo "$v" > /sys/bus/pci/devices/$b/power_dpm_force_performance_level 2>/dev/null || bad=1
    [ "$(cat /sys/bus/pci/devices/$b/power_dpm_force_performance_level 2>/dev/null)" = "$v" ] || bad=1
  done
  return $bad; }
PIN_MODE=${PIN_MODE:-high}
case "$PIN_MODE" in high|sclk8) ;; *) echo "with-mi50-pin: PIN_MODE=$PIN_MODE: high or sclk8" >&2; exit 2 ;; esac
topidx() { tail -n 1 "$1" | cut -d: -f1; }
setmask() {   # $1: pp_dpm_sclk | pp_dpm_mclk, $2: top | all; every card
  local f=$1 b n m bad=0
  for b in $BDFS; do n=$(topidx /sys/bus/pci/devices/$b/$f); if [ "$2" = top ]; then m=$n; else m=$(seq -s' ' 0 $n); fi
    echo "$m" > /sys/bus/pci/devices/$b/$f 2>/dev/null || bad=1; done
  return $bad; }
pin_on() {
  [ "$PIN_MODE" = high ] && { setlvl high; return; }
  setlvl manual && setmask pp_dpm_sclk top && setmask pp_dpm_mclk top; }
pin_off() {   # (whatever mode set it: if manual, both masks back to every level first)
  local b man=0; for b in $BDFS; do [ "$(cat /sys/bus/pci/devices/$b/power_dpm_force_performance_level)" = manual ] && man=1; done
  [ $man = 1 ] && { setmask pp_dpm_sclk all; setmask pp_dpm_mclk all; }
  setlvl auto; }
alive() { local p; for p in "$@"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"; done; }

# register as a holder; the first one in sets the pin
exec 9> "$LOCK"; flock 9
others=$(alive $(cat "$HOLD" 2>/dev/null) | grep -vx "$$" | paste -sd' ')
echo $others $$ | tr ' ' '\n' | grep -v '^$' > "$HOLD"
RESTORED=0
restore() {
  [ $RESTORED = 1 ] && return; RESTORED=1
  flock 9
  local left; left=$(alive $(cat "$HOLD" 2>/dev/null) | grep -vx "$$" | paste -sd' ')
  echo $left | tr ' ' '\n' | grep -v '^$' > "$HOLD"
  if [ -z "$left" ]; then
    if pin_off; then log "restored auto ($(lvls)); CMD rc ${RC:-?}"
    else log "RESTORE FAILED: $(lvls) - set power_dpm_force_performance_level auto by hand"; fi
  else log "CMD rc ${RC:-?}; pin kept for the other holders ($left): $(lvls)"; fi
  flock -u 9; }
trap restore EXIT
if [ -z "$others" ]; then
  pin_on || { log "could not set $PIN_MODE on every MI50 ($(lvls)): restoring auto, CMD not run"; RC=2; flock -u 9; exit 2; }
  log "pinned $PIN_MODE ($(lvls); sclk now $(for b in $BDFS; do grep '\*' /sys/bus/pci/devices/$b/pp_dpm_sclk | cut -d: -f1; done | paste -sd/)) for: $*"
else
  log "pin already held by $others ($(lvls)); for: $*"
fi
flock -u 9

# CMD in its own process group (job control), signals forwarded to the group
set -m
"$@" 9>&- &
CH=$!
set +m
SIG=""
fwd() { SIG=$1; log "got SIG$1: passing it to CMD's process group $CH, waiting for it to end (KILL after ${PIN_GRACE} s)"
        kill -s "$1" -- -$CH 2>/dev/null || kill -s "$1" $CH 2>/dev/null; }
trap 'fwd INT' INT; trap 'fwd TERM' TERM; trap 'fwd HUP' HUP
RC=0
while :; do
  wait $CH; RC=$?
  kill -0 $CH 2>/dev/null || break   # (a trapped signal ends wait early: wait again for CMD itself)
done
if [ -n "$SIG" ]; then   # (after a signal: the rest of CMD's group too, then KILL)
  t=0; while pgrep -g $CH > /dev/null 2>&1 && [ $t -lt $PIN_GRACE ]; do sleep 1; t=$((t + 1)); done
  if pgrep -g $CH > /dev/null 2>&1; then log "CMD's group $CH still there after ${PIN_GRACE} s: KILL"; kill -KILL -- -$CH 2>/dev/null; sleep 2; fi
fi
exit $RC
