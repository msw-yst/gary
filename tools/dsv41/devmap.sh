#!/bin/bash
# tools/dsv41/devmap.sh: which HIP ordinal is which card, by PCI address and arch - read at run time, never from memory.
#   bash devmap.sh            the table (host or container; read-only sysfs)
#   bash devmap.sh env        shell lines for eval: XTX=0,1,2,3 MI50=4,5,6 XTX_BDF="..." MI50_BDF="..." NXTX=4 NMI50=3 OTHER=
#   bash devmap.sh verify     (in the GPU container) rocminfo's agent order against the sysfs map: rc 0 if they agree, 3 if not
#
# How: the KFD topology (/sys/class/kfd/kfd/topology/nodes/N/properties) lists the GPU agents in the order ROCr - and so HIP,
# with HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES unset - numbers them; node 0 (simd_count 0) is the CPU.  location_id is the
# PCI bus/device/function ((bus << 8) | (dev << 3) | fn), domain the PCI domain.  Kind: device 0x744c and gfx1100 = XTX
# (RX 7900 XTX), 0x66a0 / 0x66a1 and gfx906 = MI50; anything else is OTHER and in neither list (0x744c alone also matches a
# 7900 XT / GRE: the arch and the count are what the caller checks).
# The order is not a plain bus sort (on the test machine the XTX cards came first, then the MI50s) and changes when a card
# is added, so launchers should take their --gpus / --helper-device from here (docs/examples/dsv41-serve.sh does).  A visibility variable in the environment would make the
# KFD order not HIP's: the script refuses then.
set -u
K=/sys/class/kfd/kfd/topology/nodes
if [ -n "${HIP_VISIBLE_DEVICES:-}${ROCR_VISIBLE_DEVICES:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "devmap: HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES is set - the ordinals would not be the KFD order" >&2
  exit 2
fi
[ -d $K ] || { echo "devmap: no $K (no amdgpu KFD here)" >&2; exit 2; }
prop() { awk -v k="$2" '$1 == k {print $2}' "$1/properties"; }
rows=()   # "ordinal kind bdf arch devid temp_path"
i=0
for n in $(ls $K | sort -n); do
  p=$K/$n; s=$(prop $p simd_count); [ "${s:-0}" -gt 0 ] || continue
  loc=$(prop $p location_id); dom=$(prop $p domain); gtv=$(prop $p gfx_target_version)
  bdf=$(printf '%04x:%02x:%02x.%x' "${dom:-0}" $(( loc >> 8 )) $(( (loc >> 3) & 31 )) $(( loc & 7 )))
  dev=$(cat /sys/bus/pci/devices/$bdf/device 2>/dev/null || echo "?")
  arch=$(printf 'gfx%d%d%x' $(( gtv / 10000 )) $(( gtv / 100 % 100 )) $(( gtv % 100 )))
  case "$dev/$arch" in
    0x744c/gfx1100) kind=XTX ;;
    0x66a0/gfx906|0x66a1/gfx906) kind=MI50 ;;
    *) kind=OTHER ;;
  esac
  t=$(ls /sys/bus/pci/devices/$bdf/hwmon/hwmon*/temp2_input 2>/dev/null | head -1)
  rows+=("$i $kind $bdf $arch $dev ${t:--}")
  i=$((i + 1))
done
[ ${#rows[@]} -gt 0 ] || { echo "devmap: no GPU agents in $K" >&2; exit 2; }
get() {   # get KIND o|b: that kind's ordinals (o) or PCI addresses (b), in HIP order, comma-separated
  local kind=$1 f=$2 out="" r
  for r in "${rows[@]}"; do set -- $r
    [ "$2" = "$kind" ] || continue
    if [ "$f" = o ]; then out="$out${out:+,}$1"; else out="$out${out:+,}$3"; fi
  done
  echo "$out"; }
count() { [ -n "$1" ] && tr , '\n' <<<"$1" | wc -l || echo 0; }
case "${1:-table}" in
table)
  printf '%-4s %-5s %-13s %-8s %-7s %s\n' hip kind pci arch device "junction now"
  for r in "${rows[@]}"; do set -- $r
    j=-; [ "$6" != - ] && j="$(( $(cat $6) / 1000 )) C"
    printf '%-4s %-5s %-13s %-8s %-7s %s\n' $1 $2 $3 $4 $5 "$j"; done
  echo "XTX=$(get XTX o) MI50=$(get MI50 o)" ;;
env)
  x=$(get XTX o); m=$(get MI50 o)
  echo "XTX=$x"; echo "MI50=$m"
  echo "XTX_BDF=\"$(get XTX b | tr , ' ')\""; echo "MI50_BDF=\"$(get MI50 b | tr , ' ')\""
  echo "NXTX=$(count "$x")"; echo "NMI50=$(count "$m")"; echo "OTHER=$(get OTHER o)" ;;
verify)   # rocminfo's GPU agents in its order: each one's arch (and its BDFID, where rocminfo prints one) against the sysfs map
  command -v rocminfo >/dev/null || { echo "devmap verify: no rocminfo here (run it in the GPU container)"; exit 2; }
  mapfile -t ag < <(rocminfo 2>/dev/null | awk '/^Agent [0-9]+/ {if (a && t == "GPU") print n, b; a = 1; n = ""; b = ""; t = ""}
                                                  a && $1 == "Name:" && n == "" {n = $2}
                                                  a && $1 == "BDFID:" {b = $2}
                                                  a && /Device Type:/ {t = $3}
                                                  END {if (a && t == "GPU") print n, b}')
  [ ${#ag[@]} = ${#rows[@]} ] || { echo "devmap verify: rocminfo has ${#ag[@]} GPU agents, sysfs ${#rows[@]}: DIFFER - do not run"; exit 3; }
  bad=0
  for k in "${!rows[@]}"; do set -- ${rows[$k]}; o=$1; bdf=$3; arch=$4; read -r an ab <<<"${ag[$k]}"
    want=$(( (16#${bdf:5:2} << 8) | (16#${bdf:8:2} << 3) | ${bdf:11:1} ))
    s=ok; [ "$an" = "$arch" ] || s="arch $an"
    [ -n "${ab:-}" ] && [ "$ab" != "$want" ] && s="$s bdfid $ab != $want"
    [ "$s" = ok ] || bad=1
    echo "hip $o: sysfs $arch $bdf / rocminfo $an ${ab:--}: $s"; done
  if [ $bad = 0 ]; then echo "devmap verify: AGREE"; exit 0; fi
  echo "devmap verify: DIFFER - do not run"; exit 3 ;;
*) sed -n '2,6p' "$0"; exit 2 ;;
esac
