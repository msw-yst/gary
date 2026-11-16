#!/usr/bin/env python3
"""map_tokens.py - HOSTLP steps 2+3: hosted tokens -> our token ids by exact bytes, and the forced-reference files.

For each OUT/H/<id>.json (fetch_hosted.py output; OpenAI chat-completions logprobs: content[] with token, bytes,
logprob, top_logprobs[]):
  1. Concatenate the steps' bytes, decode as UTF-8, re-encode with OUR tokenizer (special tokens parsed).
     ACCEPT the prompt only if that gives exactly one id per hosted step AND every id's bytes equal that step's bytes
     (step by step, not just the concatenation). Else EXCLUDED, with the first mismatching step. Never silently kept.
  2. For accepted prompts, map each step's top-k entries to our ids by exact bytes (a byte string with no single id
     of ours is dropped and counted; on a byte collision a normal id beats a control id). If the hosted token itself
     is missing from the mapped top-k it is appended with its logprob, so the reference always contains its own token.

Writes OUT/forced/<id>.ids (the continuation's ids, comma-separated), OUT/forced/<id>.json
  {"tokens": [ids], "pos": [{"top": [[id, hosted_lp], ...], "hosted_lp": lp}], "source": ...}
(the same shape as engine-parity's reference files, so the same teacher-forcing path and tf_filter can force it), and
OUT/map.tsv: id, hosted steps, our ids, OK | EXCLUDED reason. Last line: "MAP: k of n accepted; top-k entries T,
unmapped by bytes U (p %)".

    python3 -I map_tokens.py --out RES/hostlp --tokenizer hf:/path/tokenizer.json [--source-label "provider model"]

Exit 0 when at least one prompt is accepted and nothing failed to parse | 6 otherwise.
If most prompts are EXCLUDED, stop: the hosted tokenizer differs from yours (a byte-level fallback is needed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokadapt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="the HOSTLP dir holding H/")
ap.add_argument("--tokenizer", required=True, help="hf:FILE | module:FILE (see tokadapt.py)")
ap.add_argument("--source-label", default="hosted greedy, top-k logprobs")
a = ap.parse_args()

tok = tokadapt.load(a.tokenizer)
os.makedirs(os.path.join(a.out, "forced"), exist_ok=True)
b2id = {}
for i in range(tok.n_vocab):
    b = tok.token_bytes(i)
    if not b:
        continue
    if b not in b2id or (not tok.is_normal(b2id[b]) and tok.is_normal(i)):
        b2id[b] = i


def step_bytes(s):
    return bytes(s["bytes"]) if s.get("bytes") is not None else s["token"].encode("utf-8")


rows, acc, tot, miss, broken = [], 0, 0, 0, 0
hdir = os.path.join(a.out, "H")
for f in sorted(os.listdir(hdir)):
    if not f.endswith(".json"):
        continue
    n = f[:-5]
    try:
        steps = json.load(open(os.path.join(hdir, f)))["choices"][0]["logprobs"]["content"]
    except Exception as e:
        broken += 1
        rows.append("%s\t-\t-\tEXCLUDED no logprobs content (%s)" % (n, type(e).__name__))
        continue
    sb = [step_bytes(s) for s in steps]
    blob = b"".join(sb)
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as e:
        rows.append("%s\t%d\t-\tEXCLUDED concatenated bytes are not UTF-8 (at byte %d)" % (n, len(steps), e.start))
        continue
    ids = tok.encode(text)
    ib = [tok.token_bytes(i) for i in ids]
    first_bad = next((k for k, (x, y) in enumerate(zip(sb, ib)) if x != y),
                     None if len(ids) == len(steps) else min(len(ids), len(steps)))
    if len(ids) != len(steps) or first_bad is not None:
        rows.append("%s\t%d\t%d\tEXCLUDED ids %d vs steps %d, bytes %s, first step mismatch at %s" % (
            n, len(steps), len(ids), len(ids), len(steps), "same" if b"".join(ib) == blob else "DIFFER", first_bad))
        continue
    pos = []
    for j, s in enumerate(steps):
        top = []
        for t in s.get("top_logprobs") or []:
            tot += 1
            i = b2id.get(step_bytes(t))
            if i is None:
                miss += 1
            else:
                top.append([i, t["logprob"]])
        if all(x[0] != ids[j] for x in top):
            top.append([ids[j], s["logprob"]])
        pos.append({"top": top, "hosted_lp": s["logprob"]})
    open(os.path.join(a.out, "forced", n + ".ids"), "w").write(",".join(map(str, ids)))
    json.dump({"tokens": ids, "pos": pos, "source": "%s (H/%s.json)" % (a.source_label, n)},
              open(os.path.join(a.out, "forced", n + ".json"), "w"))
    acc += 1
    rows.append("%s\t%d\t%d\tOK" % (n, len(steps), len(ids)))
open(os.path.join(a.out, "map.tsv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
print("MAP: %d of %d accepted; top-k entries %d, unmapped by bytes %d (%.3f %%)" % (
    acc, len(rows), tot, miss, 100.0 * miss / max(tot, 1)))
sys.exit(0 if acc and not broken else 6)
