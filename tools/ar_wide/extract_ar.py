#!/usr/bin/env python3
"""Pull the host-path all-reduce kernels and their helpers out of src/dsv41/ds4_kernels.cu, verbatim, for
ar_wide_emu.cpp (a CPU emulation: no GPU).  usage: extract_ar.py ds4_kernels.cu out.inc"""
import re
import sys

src = open(sys.argv[1]).read().split("\n")


def until_close(i):  # lines i .. the next line that is exactly "}"
    j = i
    while src[j] != "}":
        j += 1
    return src[i:j + 1]


def find(pat, start=0):
    for i in range(start, len(src)):
        if re.match(pat, src[i]):
            return i
    raise SystemExit("marker not found: " + pat)


out = []
out += ["template <class T>"] + until_close(find(r"^__device__ __forceinline__ T ld_fresh\("))
out += ["template <class T>"] + until_close(find(r"^__device__ __forceinline__ T ld_wide\("))
out += ["template <class F2, class F1>"] + until_close(find(r"^__device__ __forceinline__ void wide_pairs\("))
out += until_close(find(r"^__device__ __forceinline__ void wait_at_least\("))
a = find(r"^__global__ void __launch_bounds__\(kThreads\) ar_rs_kernel\(") - 1   # (its template <bool W> line)
b = find(r"^// x\[i\] = the members' partials summed in member order", a)
out += src[a:b]
i = find(r"^__global__ void __launch_bounds__\(kThreads\) ar_sum_kernel\(") - 1
out += until_close(i)
txt = "\n".join(out)
if not txt.startswith("template <class T>") or "template <bool W>" not in txt:
    raise SystemExit("extraction did not find the W templates")
# emulator-only: the spin yields (oversubscribed CPU threads); the logic is unchanged
n1 = txt.count("while ((int) (*(volatile const unsigned*) flag - seq) < 0) {")
txt = txt.replace("while ((int) (*(volatile const unsigned*) flag - seq) < 0) {",
                  "while ((int) (*(volatile const unsigned*) flag - seq) < 0) {\n        ar_emu_spin();")
if n1 != 1:
    raise SystemExit("spin transform did not apply (%d)" % n1)
open(sys.argv[2], "w").write("// extracted verbatim from %s by extract_ar.py\n" % sys.argv[1] + txt + "\n")
print("extracted %d lines" % len(txt.split("\n")))
