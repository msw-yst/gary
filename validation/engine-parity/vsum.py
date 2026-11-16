#!/usr/bin/env python3
"""vsum.py RES [options] - engine-parity (VANILLA) summary and pre-registered verdicts. Reads only.

Reference arm (--ref, default L: llama.cpp at the same quant): its greedy continuation of every prompt and its top-k at
every position (ref_llamacpp.py). Each arm X under test:
  TF   (the prompts of --sel, X teacher-forced along the reference's tokens, tf_filter.py output RES/X/<tag>-<id>.json):
       per position KL(ref || X) over the reference's top-k + a rest bucket
         sum_i p_i (log p_i - log q_i) + r_p (log r_p - log r_q),   r = 1 - the listed mass
       (an approximation of full-vocabulary KL, not a bound); top-1 agreement (X's argmax == the reference's token);
       first divergence (first position whose argmax differs; = C when none: up to there X's own greedy output would
       be identical). Per prompt, per length band, per token class.
  FREE (every prompt the reference answered; X's own greedy continuation RES/X/free-<id>.json {tokens}): first
       divergence from the reference's tokens and the shared prefix's share. Greedy paths split at near-ties, so this
       is report-only.

e_X = mean over prompts of the per-prompt mean KL; a_X = pooled top-1 agreement.

Verdicts (decided before data; edit the floors only in your pre-registration, never after seeing numbers):
  --verdict NAME=CAND,BASE            ENGINE form: PASS if e_CAND - e_BASE <= max(rel x e_BASE, abs) AND
                                      a_CAND - a_BASE >= -agree. Use for "our engine vs the upstream engine, same layout".
  --verdict NAME=CAND,BASE,PLACEBO    DEPLOY form: the floor is the layout placebo's own distance,
                                      F_e = max(|e_CAND - e_PLACEBO|, abs), F_a = max(|a_CAND - a_PLACEBO|, agree).
  NO DATA when either arm has no TF result, or misses more than --max-missing prompts. Never PASS on missing data.
  --drafting D,P2  report the free-running first-divergence medians of two arms (drafting cannot change a forced pass).

    python3 -I vsum.py RES --arms V P1 --verdict "ENGINE=P1,V" > RES/SUMMARY.txt
"""
import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("res")
ap.add_argument("--ref", default="L")
ap.add_argument("--arms", nargs="*")
ap.add_argument("--sel", help="selection file (default RES/sel12.txt)")
ap.add_argument("--tag", default="tf")
ap.add_argument("--verdict", action="append", default=[])
ap.add_argument("--drafting")
ap.add_argument("--rel", type=float, default=0.10)
ap.add_argument("--abs", type=float, default=0.002)
ap.add_argument("--agree", type=float, default=0.005)
ap.add_argument("--max-missing", type=int, default=2)
a = ap.parse_args()
RES = a.res
ARMS = a.arms or sorted(d for d in os.listdir(RES) if os.path.isdir(os.path.join(RES, d)) and d not in (a.ref, "ids", "hostlp"))
SELF = a.sel or os.path.join(RES, "sel12.txt")
sel = [ln.split() for ln in open(SELF)] if os.path.exists(SELF) else []
SEL = [s[0] for s in sel if s and not s[0].startswith("#")]
BIN = {s[0]: s[3] for s in sel if len(s) > 3}
L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, a.ref, "*.json")))}


def kl(lpos, xpos):
    xl = dict((int(i), q) for i, q in xpos["at_L"])
    s = ml = mx = 0.0
    for i, lp in lpos["top"]:
        q = xl.get(int(i))
        if q is None:
            return None
        p = math.exp(lp)
        s += p * (lp - q)
        ml += p
        mx += math.exp(q)
    rl, rx = max(1e-12, 1 - ml), max(1e-12, 1 - mx)
    return s + rl * (math.log(rl) - math.log(rx))


def tf(arm, n):
    f = os.path.join(RES, arm, "%s-%s.json" % (a.tag, n))
    if not os.path.exists(f) or n not in L:
        return None, "missing"
    j = json.load(open(f))
    if not j.get("ok"):
        return None, "NO DATA (records %s)" % j.get("records")
    lt = L[n]
    ks, agree, first, cls = [], 0, None, defaultdict(list)
    for p, (lp, xp) in enumerate(zip(lt["pos"], j["pos"])):
        k = kl(lp, xp)
        if k is None:
            return None, "NO DATA (a reference top-k id without the arm's logprob)"
        ks.append(k)
        hit = xp["argmax"] == lt["tokens"][p]
        agree += hit
        if not hit and first is None:
            first = p
        cls[lt["cls"][p] if p < len(lt.get("cls", [])) else "?"].append((k, hit))
    C = len(ks)
    return {"C": C, "kl": st.mean(ks) if ks else 0.0, "kl_max": max(ks) if ks else 0.0, "agree": agree,
            "first": C if first is None else first,
            "first_div_kl": st.mean(ks[:first]) if first else (st.mean(ks) if first is None and ks else None), "cls": cls}, None


def free(arm, n):
    f = os.path.join(RES, arm, "free-%s.json" % n)
    if not os.path.exists(f) or n not in L:
        return None
    t = json.load(open(f))["tokens"]
    lt = L[n]["tokens"]
    k = 0
    while k < min(len(t), len(lt)) and t[k] == lt[k]:
        k += 1
    return k, len(lt)


E, A, PER, FREE = {}, {}, {}, {}
for arm in ARMS:
    print("== arm %s vs %s" % (arm, a.ref))
    got, miss, alla, alln = {}, [], 0, 0
    bycls, byband = defaultdict(list), defaultdict(list)
    for n in SEL:
        r, e = tf(arm, n)
        if e:
            miss.append("%s %s" % (n, e))
            print("TF %s %s: %s" % (arm, n, e))
            continue
        got[n] = r
        alla += r["agree"]
        alln += r["C"]
        byband[BIN.get(n, "?")].append(r["kl"])
        for c, v in r["cls"].items():
            bycls[c] += v
        print("TF %s %s (%s, %d ids): KL %.5f (max %.4f), top-1 %d/%d, first divergence %d, KL before it %s" % (
            arm, n, BIN.get(n, "?"), L[n].get("n_prompt", -1), r["kl"], r["kl_max"], r["agree"], r["C"], r["first"],
            "%.5f" % r["first_div_kl"] if r["first_div_kl"] is not None else "-"))
    PER[arm] = got
    if len(miss) > a.max_missing or not got:
        E[arm] = A[arm] = None
        print("ARM %s TF: NO DATA (%d of %d prompts missing: %s)" % (arm, len(miss), len(SEL), "; ".join(miss)))
    else:
        E[arm] = st.mean(r["kl"] for r in got.values())
        A[arm] = alla / alln
        print("ARM %s TF: e = %.5f nats (mean of %d prompts), top-1 agreement %.4f (%d/%d)%s" % (
            arm, E[arm], len(got), A[arm], alla, alln, "; missing: " + "; ".join(miss) if miss else ""))
        for b in sorted(byband):
            print("  band %s: KL %.5f over %d prompts" % (b, st.mean(byband[b]), len(byband[b])))
        for c in sorted(bycls):
            v = bycls[c]
            print("  class %s: KL %.5f, top-1 %.4f over %d positions" % (c, st.mean(k for k, _ in v), sum(h for _, h in v) / len(v), len(v)))
    fr = {n: free(arm, n) for n in L}
    fr = {n: v for n, v in fr.items() if v}
    FREE[arm] = fr
    if fr:
        fd = [v[0] for v in fr.values()]
        print("ARM %s FREE: %d prompts; first divergence median %d (min %d), identical to %s in full: %d; shared prefix %.3f of %s's tokens" % (
            arm, len(fr), st.median(fd), min(fd), a.ref, sum(v[0] == v[1] for v in fr.values()), sum(fd) / sum(v[1] for v in fr.values()), a.ref))
    else:
        print("ARM %s FREE: NO DATA" % arm)


def verdict(name, x, y, floor_e, floor_a):
    de, da = E[x] - E[y], A[x] - A[y]
    ok = de <= floor_e and da >= -floor_a
    line = "VERDICT %s: %s (e_%s - e_%s = %+.5f nats vs floor %.5f; top-1 %s - %s = %+.4f vs floor %.4f)" % (
        name, "PASS" if ok else "DIFF", x, y, de, floor_e, x, y, da, floor_a)
    if not ok:
        worse = sorted(((PER[x][n]["kl"] - PER[y][n]["kl"], n) for n in PER[x] if n in PER[y]), reverse=True)[:6]
        line += "; prompts where %s is furthest above %s: %s" % (x, y, ", ".join("%s %+.5f" % (n, d) for d, n in worse))
    return line


for v in a.verdict:
    name, arms = v.split("=", 1)
    arms = arms.split(",")
    need = arms if len(arms) == 2 else arms[:3]
    if any(E.get(z) is None for z in need):
        print("VERDICT %s: NO DATA (arm %s has no TF result)" % (name, " / ".join(z for z in need if E.get(z) is None)))
        continue
    if len(arms) == 2:
        print(verdict(name, arms[0], arms[1], max(a.rel * E[arms[1]], a.abs), a.agree))
    else:
        c, b, p = arms
        print(verdict(name, c, b, max(abs(E[c] - E[p]), a.abs), max(abs(A[c] - A[p]), a.agree)))
if a.drafting:
    d, p = a.drafting.split(",")
    if FREE.get(d) and FREE.get(p):
        print("REPORT drafting floor (%s vs %s free-running first divergence medians): %d vs %d tokens" % (
            d, p, st.median(v[0] for v in FREE[d].values()), st.median(v[0] for v in FREE[p].values())))
    else:
        print("REPORT drafting floor: NO DATA")
