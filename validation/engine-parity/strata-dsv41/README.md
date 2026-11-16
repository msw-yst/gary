# engine-parity/strata-dsv41/: the DeepSeek V4.1 Flash / Strata pieces of the VANILLA and HOSTLP runs

Reference llama.cpp b11870 (74bca395); arms upstream Strata v0.2.9 (V) and this fork's build 17 (P1); weights the
2.25 bpw IQ2_XS pack. The operator scripts that staged the weights, launched the containers and guarded the machine
are not published (they name our paths and machines); what each did is described below.

| File | Role |
|---|---|
| `prereg.md` | the pre-registration, written 2026-11-15 before any data: arms, inputs, metrics, verdict rules (§5), run rules |
| `arm-inner.sh` | one arm inside a container that sees exactly one GPU and no network. L: llama-server, `../ref_llamacpp.py`, `../select_tf.py`. V / P1: phases `smoke` (does the longest prompt fit?), `tf` (prompt + the reference continuation in one pass, logits to a FIFO, `../tf_filter.py`), `free` (`sserve.py`). HOSTLP: `CONT` (forced-continuation dir) and `TFTAG` (output prefix) |
| `sserve.py` | FREE driver for Strata's `--serve` line protocol (`GEN n temperature=0 spec=0 <ids>` → `T` lines, then `DONE`); an engine still up 600 s after QUIT is stopped, not counted as a data failure |
| `render.py`, `fake-engine.py` | RENDER: build 17's own `serve/server.py` against a stand-in engine that logs every GEN line, which gives the exact ids the server hands its engine. Every arm reads those ids |

## What the unpublished operator scripts did
- **Weights:** the 4-shard IQ2_XS pack for V and P1; for L, the same tensors converted to llama.cpp's 6-shard layout by
  the quant author's own converter, unmodified, with the shards checked against the published sha256s.
- **Containers:** one per arm; only that GPU's render node and `/dev/kfd`; `--network none`; mounts `/b` (engine),
  `/ds` (pack), `/res` (results), `/pkg` (these scripts and `../*.py`), `/pkgenv` (P1's environment file), and for L
  `/lc` (llama.cpp) and `/m` (converted shards). Env: `NTH` (CPU threads for llama-server), `PHASES`, `CONT`, `TFTAG`.
- **Guard:** every 10 s, stop the arm on load, GPU junction temperature ≥ 95 °C, low free space or a STOP file.

## Deviations recorded during the run
- V's first smoke was killed for memory (a NUMA policy while 135 GB of experts were copied in at load):
  `--mmap-experts on` was added to the common flags (the same weights, mapped in place).
- V answered everything but exited slowly after QUIT; recorded as complete by data, and `sserve.py` was fixed.
- HOSTLP: L was not forced along the hosted tokens (llama-server has no forced-scoring path); declared before scoring.
