#!/usr/bin/env python3
"""hscore.py - HOSTLP scoring: local arms teacher-forced along the hosted model's greedy tokens, against hosted's top-k.

    python3 -I hscore.py --ref RES/hostlp/forced --arm V=RES/V --arm P1=RES/P1 [--tag htf] [--json out.json]
                         [--engine-floor-kl 0.004 --engine-floor-agree 0.97] [--factor 3]

--ref  dir of <id>.json reference files (map_tokens.py output: tokens, pos[].top = [[id, lp]...], pos[].hosted_lp)
--arm  NAME=DIR, repeatable: DIR/<tag>-<id>.json as written by engine-parity/tf_filter.py
       ({ok, pos[]: {argmax, lp_tok, top: [[id, lp]...] (the arm's own top-k), at_L: [[id, lp]...] (the arm's logprobs at
       the reference's top-k ids)}}).

Per position, for each arm X:
  KL(H || X)  over hosted's top-k plus one rest bucket (same estimator as engine-parity/vsum.py; the rest masses are
              floored at 1e-6). An approximation of full-vocabulary KL, not a bound.
  top-1 agreement   X's argmax == hosted's token
  NLL gap           -log p_X(hosted token) - (-hosted's own logprob for it)
  rank              of hosted's token in X's own top-k (k+1 when absent)
Pairwise (every pair of arms, on the same positions): argmax agreement, KL(A || B) over the reference's top-k ids plus
rest (both arms have logprobs there), and mean |lp_tok(A) - lp_tok(B)|. This is the ENGINE FLOOR: the same quant on
two engines.

Missing data is never a pass: a prompt an arm lacks (no file, ok false, wrong length) is listed as NO DATA and that
arm's pooled numbers say how many prompts they cover. An arm missing more than --max-missing prompts is NO DATA.

READING (pre-registered in the original spec; printed when there are >= 2 arms):
  disagreement with hosted d_H = 1 - agreement(X, H); engine disagreement d_E = 1 - agreement(A, B).
  "QUANT COST SHOWN"   every arm's d_H > factor x d_E and the arms agree with each other (pair KL < each arm's KL vs H / factor)
  "ENGINE COST SHOWN"  one arm's d_H > factor x the others' while those sit within d_E of hosted
  "NO MEASURABLE DIFFERENCE"  every arm's d_H <= d_E
  otherwise "UNCLEAR" with the numbers. --engine-floor-* add the VANILLA (engine-vs-reference-engine) floor for context.
"""
import argparse
import glob
import itertools
import json
import math
import os
import statistics as st
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--arm", action="append", required=True, help="NAME=DIR")
ap.add_argument("--tag", default="htf", help="arm file prefix: DIR/<tag>-<id>.json (VANILLA's own pass used tf)")
ap.add_argument("--max-missing", type=int, default=2)
ap.add_argument("--factor", type=float, default=3.0)
ap.add_argument("--engine-floor-kl", type=float)
ap.add_argument("--engine-floor-agree", type=float)
ap.add_argument("--json")
a = ap.parse_args()
ARMS = [x.split("=", 1) for x in a.arm]
FLOOR = 1e-6


def kl_over(top, lp_of):
    """KL(P || Q) over P's listed ids + a rest bucket; lp_of(id) -> Q's logprob or None."""
    s = mp = mq = 0.0
    used = 0
    for i, lp in top:
        q = lp_of(int(i))
        if q is None:
            continue
        p = math.exp(lp)
        s += p * (lp - q)
        mp += p
        mq += math.exp(q)
        used += 1
    rp, rq = max(FLOOR, 1 - mp), max(FLOOR, 1 - mq)
    return s + rp * (math.log(rp) - math.log(rq)), used


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


refs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.ref, "*.json")))}
if not refs:
    sys.exit("no reference files in %s" % a.ref)
K = max(len(p["top"]) for r in refs.values() for p in r["pos"])
tot = {n: dict(kl=[], agree=0, n=0, gap=[], rank=[], prompts=0, missing=[]) for n, _ in ARMS}
pairs = {(x[0], y[0]): dict(kl=[], agree=0, n=0, dlp=[]) for x, y in itertools.combinations(ARMS, 2)}
for pid, H in refs.items():
    X = {}
    row = []
    for name, d in ARMS:
        f = os.path.join(d, "%s-%s.json" % (a.tag, pid))
        j = json.load(open(f)) if os.path.exists(f) else None
        if j is None or not j.get("ok", True) or len(j.get("pos", [])) != len(H["pos"]):
            why = "no file" if j is None else ("ok false" if not j.get("ok", True) else "%d positions, want %d" % (len(j["pos"]), len(H["pos"])))
            tot[name]["missing"].append("%s (%s)" % (pid, why))
            row.append("%s NO DATA (%s)" % (name, why))
            continue
        X[name] = j
        t = tot[name]
        k, ag = [], 0
        for jj, hp in enumerate(H["pos"]):
            xp = j["pos"][jj]
            tok = H["tokens"][jj]
            at = dict((int(i), lp) for i, lp in xp["at_L"])
            k.append(kl_over(hp["top"], at.get)[0])
            ag += xp["argmax"] == tok
            t["gap"].append(-xp["lp_tok"] + hp["hosted_lp"])
            ids = [int(i) for i, _ in xp["top"]]
            t["rank"].append(ids.index(tok) + 1 if tok in ids else K + 1)
        t["kl"] += k
        t["agree"] += ag
        t["n"] += len(k)
        t["prompts"] += 1
        row.append("%s KL %.4f top-1 %d/%d" % (name, st.mean(k), ag, len(k)))
    for (x, y), pr in pairs.items():
        if x in X and y in X:
            for jj, hp in enumerate(H["pos"]):
                A, B = X[x]["pos"][jj], X[y]["pos"][jj]
                pr["agree"] += A["argmax"] == B["argmax"]
                bl = dict((int(i), lp) for i, lp in B["at_L"])
                pr["kl"].append(kl_over(A["at_L"], bl.get)[0])
                pr["dlp"].append(abs(A["lp_tok"] - B["lp_tok"]))
                pr["n"] += 1
    print("prompt %s (%d steps): %s" % (pid, len(H["pos"]), "; ".join(row)))

out = {"arms": {}, "pairs": {}, "prompts": len(refs)}
for name, _ in ARMS:
    t = tot[name]
    if not t["n"] or len(t["missing"]) > a.max_missing:
        print("ARM %s: NO DATA (%d of %d prompts missing: %s)" % (name, len(t["missing"]), len(refs), "; ".join(t["missing"])))
        out["arms"][name] = None
        continue
    r = dict(kl_mean=st.mean(t["kl"]), kl_median=st.median(t["kl"]), kl_p90=pct(t["kl"], 0.9), agree=t["agree"] / t["n"],
             positions=t["n"], prompts=t["prompts"], gap_mean=st.mean(t["gap"]), gap_median=st.median(t["gap"]),
             gap_p90=pct(t["gap"], 0.9), in_top1=sum(x == 1 for x in t["rank"]) / len(t["rank"]),
             in_top5=sum(x <= 5 for x in t["rank"]) / len(t["rank"]), absent=sum(x == K + 1 for x in t["rank"]) / len(t["rank"]),
             missing=t["missing"])
    out["arms"][name] = r
    print("ARM %s vs HOSTED (%d prompts, %d positions%s): KL mean %.4f nats (median %.5f, p90 %.4f); top-1 agreement %.4f (%d/%d); "
          "hosted-token NLL gap mean %.4f (median %.5f, p90 %.4f); hosted token in X's top-1/top-5/absent from top-%d: %.3f/%.3f/%.3f" % (
              name, r["prompts"], r["positions"], "; missing " + ", ".join(t["missing"]) if t["missing"] else "", r["kl_mean"],
              r["kl_median"], r["kl_p90"], r["agree"], t["agree"], t["n"], r["gap_mean"], r["gap_median"], r["gap_p90"], K,
              r["in_top1"], r["in_top5"], r["absent"]))
for (x, y), pr in pairs.items():
    if not pr["n"]:
        print("FLOOR %s vs %s: NO DATA" % (x, y))
        continue
    r = dict(agree=pr["agree"] / pr["n"], kl_mean=st.mean(pr["kl"]), dlp_mean=st.mean(pr["dlp"]), positions=pr["n"])
    out["pairs"]["%s|%s" % (x, y)] = r
    print("FLOOR %s vs %s on the same %d positions: argmax agreement %.4f; KL(%s || %s) over the reference's top-%d + rest %.5f nats; "
          "mean |lp_tok difference| %.5f nats" % (x, y, pr["n"], r["agree"], x, y, K, r["kl_mean"], r["dlp_mean"]))

ok = {n: r for n, r in out["arms"].items() if r}
if len(ok) >= 2 and out["pairs"]:
    dE = max(1 - p["agree"] for p in out["pairs"].values())
    klE = max(p["kl_mean"] for p in out["pairs"].values())
    dH = {n: 1 - r["agree"] for n, r in ok.items()}
    f = a.factor
    if all(v > f * dE for v in dH.values()) and all(klE < r["kl_mean"] / f for r in ok.values()):
        reading = "QUANT COST SHOWN"
    elif any(dH[n] > f * max([dH[m] for m in dH if m != n] + [dE]) for n in dH) and \
            sum(v <= dE for v in dH.values()) == len(dH) - 1:
        reading = "ENGINE COST SHOWN"
    elif all(v <= dE for v in dH.values()):
        reading = "NO MEASURABLE DIFFERENCE"
    else:
        reading = "UNCLEAR"
    extra = ""
    if a.engine_floor_kl is not None:
        extra += "; KL vs hosted / reference-engine floor: " + ", ".join(
            "%s %.1fx" % (n, r["kl_mean"] / a.engine_floor_kl) for n, r in ok.items())
    if a.engine_floor_agree is not None:
        extra += "; reference-engine disagreement %.4f" % (1 - a.engine_floor_agree)
    print("READING: %s (disagreement with hosted %s vs between arms %.4f, factor %.1f; arm-pair KL %.5f%s)" % (
        reading, ", ".join("%s %.4f" % kv for kv in dH.items()), dE, max(dH.values()) / dE if dE else float("inf"), klE, extra))
    out["reading"] = reading
elif len(ok) < len(ARMS):
    print("READING: NO DATA (an arm has no result)")
    out["reading"] = "NO DATA"
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
