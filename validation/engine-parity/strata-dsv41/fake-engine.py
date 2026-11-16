#!/usr/bin/env python3
"""fake-engine.py [engine args ...] - the RENDER step's stand-in engine (no GPU, no model): speaks the engine protocol build 17's
server.py expects (`READY <context> stop`; per `GEN <max_new> [keys] <ids>` one `T 1` (EOS) and a DONE line), and appends every GEN
line it is sent to $FAKE_GEN_LOG, so the exact ids build 17's server would hand the engine are recorded.  NOTHING HERE HAS BEEN RUN."""
import os, sys

ctx = 131072
a = sys.argv[1:]
if "--max-context" in a:
    ctx = int(a[a.index("--max-context") + 1])
log = open(os.environ["FAKE_GEN_LOG"], "a")
sys.stdout.write("READY %d stop\n" % ctx)
sys.stdout.flush()
for ln in sys.stdin:
    w = ln.split()
    if not w:
        continue
    if w[0] == "QUIT":
        break
    if w[0] in ("GEN", "GENI"):
        log.write(ln if ln.endswith("\n") else ln + "\n")
        log.flush()
        ids = w[-1].split(",")
        sys.stdout.write("T 1\nDONE 1 %d 0.0 0.0 stop 0 0 0 0 0\n" % len(ids))
        sys.stdout.flush()
