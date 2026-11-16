#!/usr/bin/env python3
"""render.py PROMPTS RES URL GENLOG - the RENDER step (bench-host, CPU only; inside a container, with build 17's server tree running
against fake-engine.py).  NOTHING HERE HAS BEEN RUN.  For each request body PROMPTS/NN.json (index.tsv: file, prompt_tokens, band,
turn_kind, answer_kind, task, trial, turn): POST it to build 17's /v1/chat/completions with max_tokens 1 and stream false (every
other field as recorded), take the GEN line the server sent the engine, and assert FIRST that the answer's
usage.prompt_tokens == the GEN line's id count (the server's own consistency), and |usage.prompt_tokens - recorded prompt_tokens|
<= max(32, 0.5 % of the recorded) (39 of the 40 bodies are reconstructed from trajectories; the tolerance guards only against a gross
reconstruction error - a missing turn, a wrong template; decided 2026-11-15 before any data, prereg.md 2).  Every prompt's delta is
reported; one beyond the tolerance is EXCLUDED (listed, never used), not a failure of the run.
The ids of the passing ones -> RES/ids/NN.ids: the SAME ids every arm (L, V, P1, D) reads.  RES/render.tsv: one row a prompt.
Last line "RENDER: OK <k> of <n>" (k >= 12 and every band has >= 3) | "RENDER: NO DATA ...".  No text is written anywhere."""
import collections, json, os, sys, time, urllib.request

PROMPTS, RES, URL, GENLOG = sys.argv[1:5]
os.makedirs(os.path.join(RES, "ids"), exist_ok=True)
rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(PROMPTS, "index.tsv")) if ln.strip()]
if rows and not rows[0][1].strip().isdigit():
    rows = rows[1:]   # (a header row)
out, ok, refused, band_ok = [], 0, [], collections.Counter()
for row in rows:
    fn, want, band = row[0], int(row[1]), (row[2] if len(row) > 2 else "?")
    n = os.path.splitext(os.path.basename(fn))[0]
    why, got_pt, got_ids = "", None, None
    try:
        body = json.load(open(os.path.join(PROMPTS, fn)))
        body["max_tokens"] = 1
        body.pop("max_completion_tokens", None)
        body["stream"] = False
        before = os.path.getsize(GENLOG) if os.path.exists(GENLOG) else 0
        r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=600) as f:
            ans = json.load(f)
        got_pt = (ans.get("usage") or {}).get("prompt_tokens")
        time.sleep(0.2)
        with open(GENLOG) as g:
            g.seek(before)
            gens = [ln for ln in g.read().splitlines() if ln.startswith("GEN")]
        if len(gens) != 1:
            why = "%d GEN lines for one request" % len(gens)
        else:
            got_ids = [int(x) for x in gens[0].split()[-1].split(",")]
            if not isinstance(got_pt, int) or len(got_ids) != got_pt:
                why = "GEN ids %d != usage.prompt_tokens %s" % (len(got_ids), got_pt)
            elif abs(got_pt - want) > max(32, 0.005 * want):
                why = "EXCLUDED: delta %+d tokens beyond max(32, 0.5 %% of %d)" % (got_pt - want, want)
    except Exception as e:
        why = "%s: %s" % (type(e).__name__, str(e)[:160])
    if why:
        refused.append(n)
        print("RENDER %s %s (recorded %d, rendered %s)" % (n, why if why.startswith("EXCLUDED") else "REFUSED (%s)" % why, want, got_pt), flush=True)
    else:
        ok += 1
        band_ok[band] += 1
        open(os.path.join(RES, "ids", "%s.ids" % n), "w").write(",".join(map(str, got_ids)))
        print("RENDER %s OK %d ids, recorded %d, delta %+d (%s)" % (n, got_pt, want, got_pt - want, band), flush=True)
    out.append("%s\t%d\t%s\t%s\t%s\t%s" % (n, want, got_pt, len(got_ids) if got_ids else "-", why or "OK",
                                            ("%+d" % (got_pt - want)) if isinstance(got_pt, int) else "-"))
open(os.path.join(RES, "render.tsv"), "w").write("\n".join(out) + "\n")
short = [b for b in sorted(set(r[2] for r in rows if len(r) > 2)) if band_ok[b] < 3]
if ok >= 12 and not short:
    print("RENDER: OK %d of %d (excluded / refused: %s)" % (ok, len(rows), " ".join(refused) or "none"))
    sys.exit(0)
print("RENDER: NO DATA (%d of %d passed; bands under 3: %s; refused: %s)" % (ok, len(rows), " ".join(short) or "none", " ".join(refused)))
sys.exit(6)
