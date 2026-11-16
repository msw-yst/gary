#!/usr/bin/env python3
"""fetch_hosted.py - HOSTLP step 1: the hosted model's greedy continuation of each selected prompt, with its top-k logprobs.

For every prompt id in --select (first column of each line, e.g. engine-parity's sel12.txt), the request body
--bodies/<id>.json is sent to an OpenAI-compatible /chat/completions endpoint with ONLY these fields changed:
model, temperature 0, max_tokens, logprobs true, top_logprobs K, stream false (plus --set KEY=JSON overrides, e.g. to
match a thinking switch). Everything else in the body stays as recorded, so the hosted side renders the same chat.

The raw response is saved as OUT/H/<id>.json. It holds the hosted model's generated text: treat OUT/H like the prompts
(private). The printed line per prompt is numbers only: steps, finish reason, hosted prompt_tokens vs our rendered id
count (--ids DIR/<id>.ids; a mismatch means a different template or tokenizer: stop and look), completion tokens,
whether reasoning content came back, wall seconds.

The API key, if any, is read from the environment variable named by --key-env and is never printed. Going through a
local metering proxy that injects the key (as the original run did) is the better pattern: then this script never holds
a key at all.

    python3 -I fetch_hosted.py --base-url http://127.0.0.1:PORT/v1 --model MODEL --bodies PROMPTS --select sel12.txt \
        --ids RES/ids --out RES/hostlp [--key-env HOSTED_API_KEY] [--top-logprobs 20] [--max-tokens 256]

Exit 0 all fetched | 6 at least one prompt failed (its line says why; the others still run).
Provider caps seen: one provider rejected top_logprobs > 20 with HTTP 400.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True, help="OpenAI-compatible base URL ending in /v1")
ap.add_argument("--model", required=True)
ap.add_argument("--bodies", required=True, help="dir of <id>.json request bodies (private)")
ap.add_argument("--select", required=True, help="file whose first column lists the prompt ids to fetch")
ap.add_argument("--out", required=True)
ap.add_argument("--ids", help="dir of <id>.ids (comma-separated rendered prompt ids) to compare prompt_tokens with")
ap.add_argument("--key-env", help="name of an env var holding the API key (never printed)")
ap.add_argument("--top-logprobs", type=int, default=20)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--timeout", type=float, default=900)
ap.add_argument("--set", action="append", default=[], help="KEY=JSON body override, repeatable")
a = ap.parse_args()

os.makedirs(os.path.join(a.out, "H"), exist_ok=True)
hdr = {"Content-Type": "application/json"}
if a.key_env:
    k = os.environ.get(a.key_env)
    if not k:
        sys.exit("no key in $%s" % a.key_env)
    hdr["Authorization"] = "Bearer " + k
url = a.base_url.rstrip("/") + "/chat/completions"
extra = {}
for kv in a.set:
    key, val = kv.split("=", 1)
    extra[key] = json.loads(val)


def n_ids(n):
    if not a.ids:
        return None
    p = os.path.join(a.ids, n + ".ids")
    return len(open(p).read().strip().split(",")) if os.path.exists(p) else None


bad = 0
for line in open(a.select):
    w = line.split()
    if not w or w[0].startswith("#"):
        continue
    n = w[0]
    t0 = time.time()
    try:
        body = json.load(open(os.path.join(a.bodies, n + ".json")))
        body.pop("max_completion_tokens", None)
        body.update(model=a.model, temperature=0, max_tokens=a.max_tokens, logprobs=True, top_logprobs=a.top_logprobs, stream=False)
        body.update(extra)
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr)
        raw = urllib.request.urlopen(req, timeout=a.timeout).read()
        d = json.loads(raw)
        open(os.path.join(a.out, "H", n + ".json"), "wb").write(raw)
        c = d["choices"][0]
        lp = (c.get("logprobs") or {}).get("content") or []
        u = d.get("usage") or {}
        ours = n_ids(n)
        pt = u.get("prompt_tokens")
        flag = "" if ours is None or pt == ours else " PROMPT-TOKENS-DIFFER"
        print("H %s OK steps=%d finish=%s prompt_tokens=%s ours=%s completion=%s reasoning=%s %.1fs%s" % (
            n, len(lp), c.get("finish_reason"), pt, ours, u.get("completion_tokens"),
            bool((c.get("message") or {}).get("reasoning_content")), time.time() - t0, flag), flush=True)
        if not lp:
            bad += 1
            print("H %s NO LOGPROBS in the answer: the endpoint does not expose them for this model" % n, flush=True)
    except Exception as e:  # one prompt's failure is its own NO DATA
        bad += 1
        print("H %s FAILED %s: %s" % (n, type(e).__name__, str(e)[:160]), flush=True)
sys.exit(6 if bad else 0)
