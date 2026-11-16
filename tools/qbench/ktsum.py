#!/usr/bin/env python3
"""QBENCH ktsum.py DIR N_CALLS [LABEL [LAST SKIP]]: GPU-side kernel time per call from a rocprofv3 --kernel-trace csv under DIR.
Sums End-Start of every kernel except the runtime's own (__amd_rocclr*), divides by N_CALLS; also per kernel name.
With LAST / SKIP (ours: the bench's busy pass): kernels in start order, the last SKIP calls' dropped (the error rep), then
only the last LAST calls' kept (kernels a call = total / N_CALLS); the line is then 'busy'."""
import csv
import glob
import sys
from collections import defaultdict

d, n = sys.argv[1], int(sys.argv[2])
lab = sys.argv[3] if len(sys.argv) > 3 else d
last = int(sys.argv[4]) if len(sys.argv) > 4 else 0
skip = int(sys.argv[5]) if len(sys.argv) > 5 else 0
files = glob.glob(d + "/**/*kernel_trace.csv", recursive=True)
if not files:
    print(f"KT {lab} NO DATA (no kernel_trace.csv)")
    sys.exit(8)
ks = []
for f in files:
    for r in csv.DictReader(open(f)):
        name = r.get("Kernel_Name", "")
        if name.startswith("__amd_rocclr"):
            continue
        ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), name.split("(")[0][-60:]))
ks.sort()
kpc = len(ks) / n
if last:
    k = int(round(kpc))
    ks = ks[: len(ks) - skip * k][-last * k:]
    n = last
tot, per, cnt = 0.0, defaultdict(float), defaultdict(int)
for s, e, nm in ks:
    tot += (e - s) / 1000.0
    per[nm] += (e - s) / 1000.0
    cnt[nm] += 1
span = (ks[-1][1] - ks[0][0]) / 1000.0 / n if last else 0.0
print(f"KT {lab} calls={n} gpu_us_per_call={tot / n:.2f} kernels_per_call={len(ks) / n:.2f}" + (f" span_us_per_call={span:.2f} busy" if last else ""))
for k in sorted(per, key=lambda k: -per[k])[:8]:
    print(f"KT   {per[k] / n:8.2f} us/call  x{cnt[k] / n:.2f}  {k}")
