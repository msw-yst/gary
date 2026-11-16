#!/usr/bin/env python3
"""select_tf.py RES [--edges 0,8192,20480,40960] [--per-bin 3] - choose the teacher-forced prompts from the REFERENCE
arm's results ONLY, before any arm under test has run (so the choice cannot be steered by the answer).

Bins by prompt length (the reference's n_prompt); inside a bin, round-robin over the majority token class of the
reference continuation (e.g. think / tool / prose), lowest id first; a short bin's places go to the next bin up, then
down. Prints "<id> <n_prompt> <class> bin<k>" lines (engine-parity's sel12.txt); exit 6 if fewer than bins x per-bin.
Reads RES/<ref>/<id>.json (ref_llamacpp.py output: n_prompt, cls).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("res")
ap.add_argument("--ref", default="L")
ap.add_argument("--edges", default="0,8192,20480,40960")
ap.add_argument("--per-bin", type=int, default=3)
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.res, a.ref, "*.json"))):
    j = json.load(open(f))
    c = Counter(j["cls"]).most_common(1)[0][0] if j.get("cls") else "prose"
    rows.append((os.path.basename(f)[:-5], j["n_prompt"], c))
EDGES = [int(x) for x in a.edges.split(",")] + [1 << 40]
NB = len(EDGES) - 1
bins = [[r for r in rows if EDGES[b] <= r[1] < EDGES[b + 1]] for b in range(NB)]


def pick(cands, k):
    by = {}
    for r in cands:
        by.setdefault(r[2], []).append(r)
    out, keys = [], sorted(by)
    while len(out) < k and any(by.values()):
        for c in keys:
            if by[c] and len(out) < k:
                out.append(by[c].pop(0))
    return out


chosen = [pick(b, a.per_bin) for b in bins]
short = sum(a.per_bin - len(c) for c in chosen)
for b in list(range(NB)) + list(range(NB - 1, -1, -1)):
    if short <= 0:
        break
    rest = [r for r in bins[b] if r not in chosen[b]]
    extra = pick(rest, short)
    chosen[b] += extra
    short -= len(extra)
sel = [(r, b) for b in range(NB) for r in chosen[b]]
for r, b in sel:
    print("%s %d %s bin%d" % (r[0], r[1], r[2], b))
sys.exit(0 if len(sel) == NB * a.per_bin else 6)
