#!/usr/bin/env python3
"""QBENCH: assemble a pool of real routed-expert blobs into one QBX1 file for ds4_iq_helper_bench --pool-file.

QBX1 layout (little endian): b"QBX1", u32 tg, u32 td, u32 n_embd, u32 n_ff, u32 count, then per expert the raw GGUF bytes of
w1 (gate [n_ff][n_embd]), w3 (up [n_ff][n_embd]), w2 (down [n_embd][n_ff]) one after another, unmodified.
Inputs: three files holding the w1 / w3 / w2 ranges of experts e0 .. e0+count-1 (contiguous in the GGUF: one 3-D tensor
per role), as cut by dd from the shard, and the per-expert byte counts from tensors.txt.
usage: mkpool.py OUT tg td n_embd n_ff count bytes_w1 bytes_w2 W1FILE W3FILE W2FILE
"""
import struct
import sys


def main() -> int:
    out, tg, td, ne, nf, cnt, b13, b2, f1, f3, f2 = sys.argv[1:12]
    tg, td, ne, nf, cnt, b13, b2 = map(int, (tg, td, ne, nf, cnt, b13, b2))
    srcs = [open(f, "rb") for f in (f1, f3, f2)]
    with open(out, "wb") as o:
        o.write(b"QBX1" + struct.pack("<5I", tg, td, ne, nf, cnt))
        for _ in range(cnt):
            for s, n in zip(srcs, (b13, b13, b2)):
                b = s.read(n)
                if len(b) != n:
                    print("short read", file=sys.stderr)
                    return 3
                o.write(b)
    for s in srcs:
        if s.read(1):
            print("trailing bytes", file=sys.stderr)
            return 3
    print(f"{out}: {cnt} experts, {cnt * (2 * b13 + b2)} bytes of weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
