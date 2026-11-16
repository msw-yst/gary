#!/usr/bin/env python3
"""sserve.py RES ARM - the free-running pass of a Strata arm (inside the arm's container): one `--serve` engine (its stdin / stdout
are this script's child pipes), every prompt RES/ids/<n>.ids that L answered, `GEN 256 temperature=0 spec=0 <ids>` (greedy, drafting
off), the T lines -> RES/<ARM>/free-<n>.json {tokens, done}.  STRATA_DS4_REUSE=0 is set by the caller, so every request reads its whole
prompt.  NOTHING HERE HAS BEEN RUN.  Exit 0 all answered | 6 a prompt with no DONE (its line says why)."""
import glob, json, os, subprocess, sys, time

RES, ARM, CMD = sys.argv[1], sys.argv[2], sys.argv[3:]
NPRED = int(os.environ.get("NPRED", "256"))
os.makedirs(os.path.join(RES, ARM), exist_ok=True)
p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(RES, ARM, "serve.stderr"), "ab"),
                     text=True, bufsize=1)
for ln in p.stdout:   # INFO lines, then READY
    if ln.startswith("READY"):
        print("%s serve READY %s" % (ARM, ln.strip()), flush=True)
        break
else:
    print("%s serve: no READY (engine rc %s)" % (ARM, p.wait()), flush=True)
    sys.exit(6)
bad = 0
for lj in sorted(glob.glob(os.path.join(RES, "L", "*.json"))):
    n = os.path.basename(lj)[:2]
    ids = open(os.path.join(RES, "ids", n + ".ids")).read().strip()
    t0 = time.time()
    p.stdin.write("GEN %d temperature=0 spec=0 %s\n" % (NPRED, ids))
    p.stdin.flush()
    toks, done = [], None
    for ln in p.stdout:
        if ln.startswith("T "):
            toks.append(int(ln.split()[1]))
        elif ln.startswith("DONE") or ln.startswith("ERR"):
            done = ln.strip()
            break
    if done is None or not done.startswith("DONE"):
        bad += 1
        print("%s free %s FAIL (%s)" % (ARM, n, done), flush=True)
        if done is None:
            break
        continue
    json.dump({"tokens": toks, "done": done}, open(os.path.join(RES, ARM, "free-%s.json" % n), "w"))
    print("%s free %s OK %d tokens, %s, wall %.0f s" % (ARM, n, len(toks), " ".join(done.split()[:5]), time.time() - t0), flush=True)
try:
    p.stdin.write("QUIT\n"); p.stdin.flush()
except BrokenPipeError:
    pass
# 2026-11-16: V answered all 40 prompts but its engine did not exit within 600 s of QUIT, and the
# TimeoutExpired traceback turned complete data into RC 6. A slow exit is not a data failure: stop it and go on.
try:
    rc = p.wait(timeout=600)
except subprocess.TimeoutExpired:
    print("%s free: engine still up 600 s after QUIT - terminating (data already written)" % ARM, flush=True)
    p.terminate()
    try:
        rc = p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill(); rc = p.wait()
    rc = "slow-exit/%s" % rc
print("%s free DONE rc %s, %d failed" % (ARM, rc, bad), flush=True)
sys.exit(6 if bad else 0)
