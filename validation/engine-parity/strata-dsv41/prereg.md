# VANILLA pre-registration (written 2026-11-15, BEFORE any VANILLA data)

Question: does our optimized engine give the same quality as the engines it was built on, with the same quant?
Tags: [OBSERVED] read in a file / source / header today; [ESTIMATE] a prediction, not tested; [DECIDED] a rule fixed now.

## 1. Arms, model, layout

- **Model:** ISTA-DASLab GSQ-RCO **IQ2_XS** (2.25 bpw), the exact quant build 17 serves.
  - [OBSERVED] build 17's pack reads `ist-das-2.25bpw` directly.
  - store-host also holds IQ2_XXS. It is NOT used.
- **L (the reference): llama.cpp b11870.**
  - Commit `74bca395`, "model: DSV4.1 flash (#31086)". Built on bench-host (ROCm, gfx1100) from the unmodified commit.
  - It runs the same IQ2_XS, converted to llama.cpp's layout by the README's own converter. Tensor bytes are copied byte for byte, and shards 1-5 are checked against the published sha256s.
  - Shard 6 is the Engram file. [OBSERVED] Its header is byte-identical to serve-host's live file. The full sha256 is checked by prep-engram.
  - Layout: 1 XTX (c3), `-ngl 99 -ot exps=CPU` (routed experts on the CPU).
- **V: upstream Strata v0.2.9 dsv41.**
  - serve-host `build/dsv41`, sha256 `6ff07eac`. Its source is byte-identical to v0.2.9 c62d5e3a.
  - Native config: no DS4 flags. 1 XTX + host-RAM experts. Drafter not loaded (`--dspark none`).
- **P1 (layout placebo for D): build 17's binary** (`d32de2d0`) with build17.env's engine flags.
  - Same 1-XTX + RAM layout as V.
  - So **P1 vs V is our engine vs upstream with everything else equal.**
- **D: build 17 as served** (TP4 + 4 MI50 helpers, build17.env). serve-host only, a later short phase.
- **P2 (drafting placebo):** D with drafting off. Also serve-host.

## 2. Inputs [DECIDED]

- 40 Terminal-Bench request bodies (private).
  - 39 of 40 are reconstructed from trajectories.
  - Bands: 10 each at 2-8K, 8-20K, 20-40K and 40-70K tokens.
- **Rendering.**
  - The ids every arm reads are the ids **build 17's own server.py** hands its engine. It runs against a stand-in engine (render.py + fake-engine.py), so no model is involved.
  - L does not render with its own template: all arms see identical ids.
- **First assertion (decided 2026-11-15, before any data).**
  - The server's `usage.prompt_tokens` must equal the GEN line's id count.
  - |rendered − recorded prompt_tokens| must be ≤ max(32 tokens, 0.5 %).
  - Every prompt's delta is reported. A prompt beyond the tolerance is excluded and listed; the run does not fail.
  - Rationale: VANILLA compares engines on identical inputs, so exact fidelity to this morning's requests isn't needed. The check guards only against gross reconstruction errors (a missing turn, a wrong template).
  - The run needs ≥ 12 passing prompts and ≥ 3 in every band, else RENDER NO DATA.

## 3. What is measured

- **TF (teacher-forced).** 12 prompts, chosen by `select12.py` from L's results only, before any V / P1 / D data exists: 3 per band, spread over the class of L's continuation.
  - L generates greedily (256 new tokens) with its top-20 log-probabilities at every position.
  - Each other arm reads prompt + L's tokens in one CLI process (`--n 0 --window 2048 --dump-logits`). A FIFO filter keeps only the continuation's positions.
  - Per position:
    - KL(L ‖ X) over L's top-20 plus a rest bucket. This is an approximation of full-vocabulary KL: the rest bucket compares L's and X's remaining mass.
    - top-1 agreement: X's argmax equals L's token.
    - first divergence: the first position where X's argmax differs from L's token. Up to there, X's free-running greedy output would be identical.
- **FREE.** All prompts L answered.
  - X's own greedy 256 tokens (`--serve`, temperature 0, drafting off, reuse off), compared with L's tokens: first divergence, and the shared prefix's share.
- **Breakdowns (report-only):** by band, by token class (think / tool-call / prose, from L's own pieces), per prompt.
  - "By layer" is not visible from the outputs: not reported.
- **Missing data is never a number.** A prompt whose dump has the wrong record count, or no file, is listed as NO DATA. An arm missing more than 2 of the 12 has a NO DATA TF verdict.

## 4. Predictions (ESTIMATE)

- e_X = the mean over the 12 prompts of the per-prompt mean KL(L ‖ X), in nats. a_X = the top-1 agreement over all scored positions.
- **e_V and e_P1: 0.003-0.03 nats; a_V and a_P1: 0.97-0.995.** Different kernels, reduction orders and IQ2_XS dot-product paths make small, non-zero differences.
- **|e_P1 − e_V| ≤ 0.10 × e_V.** build 17 adds fused kernels, but each one was checked bit-identical, or near it, against its unfused path (the ISA / REVAL gates). Our engine should sit no further from llama.cpp than upstream Strata does.
- FREE first divergence: median 30-150 tokens for V and P1 vs L. Greedy paths split at near-ties.

## 5. Verdicts [DECIDED before data]

- **ENGINE (bench-host alone): P1 vs V at the same layout.**
  - PASS if e_P1 − e_V ≤ max(0.10 × e_V, 0.002) nats AND a_P1 − a_V ≥ −0.005.
  - Else DIFF, with the prompts listed where P1 sits furthest above V.
- **DEPLOY (after the serve-host phase): D vs V.**
  - The layout floor F_e = max(|e_D − e_P1|, 0.002) and F_a = max(|a_D − a_P1|, 0.005).
  - PASS if e_D − e_V ≤ F_e AND a_D − a_V ≥ −F_a. Else DIFF, with the per-prompt list.
  - Drafting cannot change a teacher-forced pass. Its floor is reported from FREE: D vs P2 first-divergence medians.
- **NO DATA** when either arm of a verdict has no TF result. Never PASS on missing data.
- **Not part of this run:** the optional quant-author eval slice (LCB v6 / GPQA). Ask first.

## 6. Run rules (unattended-safe)

- **Stop rules (the guard, every 10 s):** load1 > 100, c3 junction ≥ 95 °C, the STOP file, the data volume under 10 GB free.
- **Refuse to start** with MemAvailable < 150 GiB, or with another v16-vanilla container running.
- **Device:** only c3's render node is visible.
- **Hang protection:** every engine call has a timeout (TF 90 min a prompt; FREE 4 h; L 6 h).
- **Pieces:** one arm per piece. Prep is CPU/disk only.
- **Old-shard gate (added 14:1xZ, before any arm data):** arms V and P1 read the old 4-shard layout, so they refuse until
  prep-oldsha.sh has matched those shards against a verified sha256 list.
  - The convert step does not wait for that list: a conversion whose 5 shards all match the published sha256s proves the
    source tensor bytes.
  - The new layout is split over two data volumes (shards 2 and 3 on the second), with per-volume headroom ≥ 10 GB.

## 7. Time (ESTIMATE; all prompts assumed near the band centres, ~1.04 M prompt ids in all)

- Prep a: 5-20 min. Render: 5-10 min.
- Convert: 20-45 min, reading 155 GB, writing 155 GB and reading it back.
- **L:**
  - load 3-10 min;
  - prompt reads 1-3 h, at an assumed 100-300 tok/s for 1 XTX with CPU experts (not measured on this box);
  - decode 20-45 min, at 4-8 tok/s.
  - **1.5-4 h.**
- **V and P1, each:**
  - smoke 3-5 min;
  - TF: 12 loads of ~1.5 min plus ~312 K ids at 150-200 tok/s = **50-70 min**;
  - FREE: ~1.04 M ids at 150-200 tok/s plus decode = **1.5-2 h**.
  - **2.5-3.2 h.**
- **Total c3: about 6.5-10 h,** or 3.5-6 h with `PHASES="smoke tf"` (no FREE pass).
- **serve-host D / P2 phase (later):** ~1 h in all.
  - TF 12 at TP4: 12 loads × 87 s + ~312 K ids at ~1,500 tok/s.
  - FREE 40 for D and for P2: ~11 min of reads each plus decode.
