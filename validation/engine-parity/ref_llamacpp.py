#!/usr/bin/env python3
"""ref_llamacpp.py IDS_DIR RES URL - the REFERENCE arm: llama.cpp's llama-server, greedy, with its top-k at every position.

The prompt ids are NOT rendered here. Every arm must read the same ids (IDS_DIR/<id>.ids, comma-separated, made once by
the RENDER step through the serving stack under test), so a template difference can never pass for an engine one.

POST /completion {prompt: ids, n_predict, temperature 0, n_probs K, cache_prompt false} -> RES/<ref>/<id>.json:
  n_prompt, tokens, pos[] = {id, lp, top: [[id, lp]...]}, cls[] (a per-token class, report-only), timings.
No text is written: the per-token pieces are only used in memory to classify tokens.

Class markers are model-specific: --think-open / --think-close / --tool-marker (repeatable). Defaults suit a
<think>...</think> model; for DeepSeek V4.1 add --tool-marker DSML --tool-marker "<｜tool".

    python3 -I ref_llamacpp.py RES/ids RES http://127.0.0.1:18080 [--ref L] [--npred 256] [--nprobs 20] [--maxp 0]

Exit 0 all done | 6 a prompt failed (its line says why; the others still run).
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("ids")
ap.add_argument("res")
ap.add_argument("url")
ap.add_argument("--ref", default="L")
ap.add_argument("--npred", type=int, default=256)
ap.add_argument("--nprobs", type=int, default=20)
ap.add_argument("--maxp", type=int, default=0, help="skip (and list) prompts longer than this; 0 = no cap")
ap.add_argument("--timeout", type=float, default=7200)
ap.add_argument("--think-open", default="<think>")
ap.add_argument("--think-close", default="</think>")
ap.add_argument("--tool-marker", action="append", default=["tool_call"])
a = ap.parse_args()


def post(path, body):
    r = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=a.timeout) as f:
        return json.load(f)


def klass(pieces):
    out, txt, mode = [], "", "prose"
    for p in pieces:
        txt += p
        if mode == "prose" and a.think_open in txt[-len(p) - len(a.think_open):] and a.think_close not in txt:
            mode = "think"
        if mode == "think" and a.think_close in txt:
            mode = "prose"
        if mode != "think" and any(m in p for m in a.tool_marker):
            mode = "tool"
        out.append(mode)
    return out


os.makedirs(os.path.join(a.res, a.ref), exist_ok=True)
bad = 0
files = sorted(glob.glob(os.path.join(a.ids, "*.ids")))
for fp in files:
    n = os.path.basename(fp)[:-4]
    try:
        ids = [int(x) for x in open(fp).read().split(",")]
        if a.maxp and len(ids) > a.maxp:
            print("%s %s SKIP (%d ids > maxp %d)" % (a.ref, n, len(ids), a.maxp), flush=True)
            continue
        t0 = time.time()
        r = post("/completion", {"prompt": ids, "n_predict": a.npred, "temperature": 0, "n_probs": a.nprobs,
                                 "cache_prompt": False, "stream": False})
        cp = r.get("completion_probabilities") or []
        if not cp:
            raise ValueError("no completion_probabilities in the answer")
        pos = [{"id": c["id"], "lp": c["logprob"], "top": [[t["id"], t["logprob"]] for t in c.get("top_logprobs", [])]} for c in cp]
        tim = r.get("timings", {})
        json.dump({"n": n, "n_prompt": len(ids), "tokens": [p["id"] for p in pos], "pos": pos,
                   "cls": klass([c.get("token", "") for c in cp]), "timings": tim},
                  open(os.path.join(a.res, a.ref, n + ".json"), "w"))
        print("%s %s OK %d ids -> %d tokens prompt %.0f ms (%.1f tok/s) decode %.0f ms (%.2f tok/s) wall %.0f s" % (
            a.ref, n, len(ids), len(pos), tim.get("prompt_ms", -1), tim.get("prompt_per_second", -1),
            tim.get("predicted_ms", -1), tim.get("predicted_per_second", -1), time.time() - t0), flush=True)
    except Exception as e:
        bad += 1
        print("%s %s FAIL %s: %s" % (a.ref, n, type(e).__name__, str(e)[:160]), flush=True)
print("%s DONE %d prompts, %d failed" % (a.ref, len(files), bad), flush=True)
sys.exit(6 if bad else 0)
