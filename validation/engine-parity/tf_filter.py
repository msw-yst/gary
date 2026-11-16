#!/usr/bin/env python3
"""tf_filter.py - the teacher-forced reader: raw logits records -> per-position scoring file.

    python3 -I tf_filter.py DUMP NPROMPT REF OUT VOCAB [--topk 20] [--dtype f32]

The engine under test was run over (prompt ids + the reference's continuation ids) in ONE pass and dumped one raw
little-endian record of VOCAB logits for EVERY position, in order, into DUMP (a FIFO or a file). Records of the prompt's
first NPROMPT-1 positions are read and dropped; the next C (C = the reference continuation length) are the
distributions over the reference's tokens 0..C-1 (position NPROMPT-1+j predicts continuation token j); the record after
that is counted, not scored.

Per scored position j: logsumexp, the logprob of the reference's token, the argmax and its logprob, the engine's own
top-k, and the engine's logprobs AT THE REFERENCE'S top-k ids (at_L: what KL(ref || engine) needs).

The total record count must be exactly NPROMPT + C, else the file says ok false (NO DATA): a short or long dump means
the engine did not see what we think it saw. REF is any reference file with "tokens" and "pos"[j]["top"]: a llama.cpp
reference (ref_llamacpp.py) or a hosted one (hostlp/map_tokens.py).

Engines that cannot dump logits this way need their own reader that writes the same OUT shape (see strata-dsv41/README.md for the Strata one).
Uses numpy when present (much faster at 100K+ vocabularies), else pure Python.
"""
import argparse
import array
import heapq
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("nprompt", type=int)
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("vocab", type=int)
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--dtype", default="f32", choices=["f32"])
a = ap.parse_args()
try:
    import numpy as np
except ImportError:
    np = None

L = json.load(open(a.ref))
C = len(L["tokens"])
V = a.vocab
REC = 4 * V
buf = bytearray(64 << 20)
out = []
total = 0


def read_exact(f, mv):
    got = 0
    while got < len(mv):
        k = f.readinto(mv[got:])
        if not k:
            break
        got += k
    return got


def score(rec, j):
    tok = L["tokens"][j]
    reft = [int(t[0]) for t in L["pos"][j]["top"]]
    if np is not None:
        x = np.frombuffer(bytes(rec), dtype="<f4").astype(np.float64)
        m = x.max()
        lse = float(m + math.log(np.exp(x - m).sum()))
        am = int(x.argmax())
        top = np.argpartition(-x, a.topk)[:a.topk]
        top = top[np.argsort(-x[top])]
        g = lambda i: float(x[i])  # noqa: E731
        topl = [int(i) for i in top]
    else:
        x = array.array("f")
        x.frombytes(bytes(rec))
        m = max(x)
        lse = m + math.log(math.fsum(math.exp(v - m) for v in x))
        am = max(range(V), key=x.__getitem__)
        topl = heapq.nlargest(a.topk, range(V), key=x.__getitem__)
        g = x.__getitem__
    return {"lse": lse, "lp_tok": g(tok) - lse, "argmax": am, "lp_argmax": g(am) - lse,
            "top": [[i, g(i) - lse] for i in topl], "at_L": [[i, g(i) - lse] for i in reft]}


with open(a.dump, "rb", buffering=0) as f:
    skip = (a.nprompt - 1) * REC
    while skip > 0:   # the prompt's first NPROMPT-1 positions: read, counted, dropped
        k = f.readinto(memoryview(buf)[:min(len(buf), skip)])
        if not k:
            break
        skip -= k
        total += k
    rec = bytearray(REC)
    for j in range(C + 1 if skip == 0 else 0):
        got = read_exact(f, memoryview(rec))
        total += got
        if got < REC or j == C:
            break
        out.append(score(rec, j))
    while True:   # anything after is counted: a longer dump than expected is NO DATA too
        k = f.readinto(buf)
        if not k:
            break
        total += k
ok = total == (a.nprompt + C) * REC and len(out) == C
json.dump({"n_prompt": a.nprompt, "C": C, "records": total / REC, "ok": ok, "pos": out}, open(a.out, "w"))
print("TF %s records %.3f (want %d) scored %d of %d %s" % (a.out, total / REC, a.nprompt + C, len(out), C, "OK" if ok else "NO DATA"), flush=True)
sys.exit(0 if ok else 6)
