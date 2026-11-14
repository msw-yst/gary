# Anatomy of one decode token: DeepSeek V4.1 Flash on four RX 7900 XTX and four MI50s

Built by YST. Contact: https://yst.dev

This note follows one single-stream decode token on build 11 ([SV-06](cards/SV-06.md)) from the start of the step to
the sampled token. It covers:

- what runs on which card, and in what order;
- where the time goes, as a budget that is meant to add up to the measured step;
- which earlier change moved which part of the token;
- what the remaining candidates are, by area.

It started as a desk analysis of existing runs. The A/B runs it pointed to have since run: their results are in
section 6, and section 2.5 shows how the budget moved with them. Sections 1-2.4 still describe build 11.

**Labels.**
- **MEASURED**: read from a run file or a script-written summary.
- **INFERRED**: derived from measured numbers by the stated arithmetic, or read from the code.
- **ESTIMATE**: a model or an assumption, with the test that would check it.

The cards use REPORTED for numbers taken from notes made at the time. Where a row below quotes one, it says so.

**Configuration.**
- Build 11 on four XTX in tensor parallel (tp4) with four MI50 helpers (h4).
- Single stream, the drafter off, a 64-token prompt, 256 decoded tokens.
- The device timeline comes from 63 recorded tokens × 40 layers ([HX-20](cards/HX-20.md)).
- **The step is 14.37-14.41 ms a token (MEASURED, HX-20).**
- The "65 tok/s" headline (15.35 ms, SV-06) was measured on a 2,048-token prompt. The 0.95 ms between the two has not
  been broken down.

**Roles.**

| card | role in a decode token |
|---|---|
| XTX #1 | tensor-parallel lead: routes, sends the helpers their requests, joins their rows, gathers the vocabulary logits, samples |
| XTX #2, #3, #4 | tensor-parallel members: the same attention, their share of the FFN and of the VRAM experts |
| MI50 #1-#4 | helpers: hold the routed experts that do not fit in XTX memory and compute them when a token picks one |
| host CPU | enqueues the step (about 3.6 ms of the 14.4 ms), handles sampling bookkeeping between tokens; its expert lane holds no experts on this layout |

## 1. The sequence

### 1.1 Start of the token

- The host copies the next token's embedding row and enqueues all 40 layers on every XTX's single in-order stream.
- It stays well ahead of the GPUs: 3.6 ms to enqueue a token against 10.4 ms waiting for the head (MEASURED, HX-20's
  reference run).
- On every XTX the token then starts with two copies, the embedding kernel and an init kernel.

### 1.2 The attention half of a layer (all four XTX, in parallel)

A plain layer runs 17 kernels on each card. The kernel durations below come from a traced run (MEASURED; the tracer
stretches the gaps between kernels, so only durations are used).

| order | work | µs |
|---|---|---|
| 1-2 | hyper-connection post and mix / split / norm (fused) | 2.4 + 12.0 |
| 3-5 | attention input projections, q norm, wq_b | 13.9 + 7.5 + 13.3 |
| 6-9 | RoPE, window ring, KV row | 12.7 (four small kernels) |
| 10-11 | selected-row unpack and sparse attention | 4.2 + 13.5 |
| 12-15 | output projection with its int8 roundings | 31.8 |
| 16-17 | put the partial sum, then the attention all-reduce | 3.8 + the AR |

The kernels add up to 115 µs. The untimed block from one layer's MoE all-reduce to the next attention all-reduce takes
148 µs (MEASURED median). **So about 33 µs a layer, 1.95 µs per kernel boundary, is launch gap** (INFERRED). That fits
HX-20's measured price of 2.63 µs for one added empty kernel: the gap plus the kernel's own minimum duration.

**Layer types differ** (MEASURED medians of the attention block):

| layers | µs | extra over a plain layer, and what it is |
|---|---|---|
| 27 plain layers | 147-151 | none |
| 2 and 8 (compressed KV + indexer) | 239-244 | +91-96: compressor GEMVs, index scoring (27 µs, one latency-bound block), top-k |
| 20 (candidate source) | 264 | +116 |
| 24, 28, 32, 36 (indexer over the candidates) | 215-216 | +67-68 |
| 1 (Engram) | 309 | +161: an Engram GEMV of about 129 µs, run on every XTX |
| 14 (Engram + compressed + indexer) | 401 | +253 |
| 37-39 | 165 | +17: a tap for the drafter, on the lead only |
| 0 | 885 | +737: the end of the previous token lands here (1.4) |

**The attention all-reduce is balanced.** The last card arrives 0.1-5 µs after the others, and in 28 of 40 layers that
card is XTX #4. That fits HW-08's reading of its lower board limits, at a cost of about 0.1 ms a token or less
(INFERRED).

### 1.3 The FFN / MoE half of a layer (the cards diverge here)

The lead (XTX #1). Segment means come from device timestamps; the kernels come from the trace.

| segment | mean µs | kernels |
|---|---|---|
| P: all-reduce release → helper request published | 48.6 | hc post + mix (14), BF16 router GEMV (6.4), routing + request (14), publish (2) |
| G: request published → join | 67.1 | the shared expert and the lead's share of the VRAM routed experts (41 µs of kernels) |
| J: join | 16.5 | waiting for the last MI50 with a pick (12.8), polling the CPU lane (1.4), adding the rows |
| C: join → MoE all-reduce | 17.8 | combine, put |
| MoE all-reduce | 3.0 | the lead arrives last, so it barely waits |

**The members (XTX #2-#4)** run the same router (without the request stores), the shared expert and their share of the
experts, then reach the MoE all-reduce **after 117.5 µs, against the lead's 150.0 µs**. They wait there about 28 µs
(median) a layer, **1.04 ms a token** (MEASURED, HX-20). An extra kernel on a member's FFN costs only 0.47 µs, because
the member has slack.

**The MI50 helpers** wait on the device for their request and wake 3-5 µs after the publish.
- Each runs a window of about 75 µs (70-86) for its one or two rows.
- That is longer than the lead's own 67 µs of work in G, so the helpers arrive about 15-25 µs after the lead reaches the
  join: they are late.
- Skipping MI50s with no pick removes 129 of 156 waits a token ([HX-18](cards/HX-18.md)), which leaves about 27.
- MI50 #1 is the most often last.

**The host's expert lane** holds no experts on this layout. Delaying it by up to 50 µs a layer changes nothing
(MEASURED, HX-20).

### 1.4 The end of the token

1. **Head.** Each XTX computes a quarter of the vocabulary logits (134 µs). The members put their parts in host memory,
   and the lead gathers the three parts (about 122 µs). This is about 276 µs of kernels on the lead in all.
2. **Drafter seeding, even with the drafter off.** After the head, the engine seeds the drafter's rings from this token:
   a 101 µs projection plus three small KV writes, about 173 µs of kernels.
   - The code calls this whenever the model pack has drafter layers. The sampler is queued behind it.
   - With the drafter on this is needed work. With it off it is about 0.2 ms of every token (INFERRED from the code and
     the trace). Every drafter-off number in the cards carries it, on both sides of each comparison.
3. **Sampling** (28 µs), then the host takes the token and starts the next one.

Layer 0's block puts the whole tail at **at least 735 µs** (ESTIMATE). The known kernels plus launches account for
about 550 µs of it. That leaves **about 190 µs of host round trip**, which is a remainder, not a measurement.

### 1.5 The critical path

In a layer where at least one MI50 has a pick, the lead's time is:

    layer = attention (148) + AR (4) + P (49) + max(G = 67, helper window ≈ 80) + J' (~1-3) + C (18) + AR (3)

- **The path runs through every card's attention, then the lead's router and publish, then the slowest MI50's window,
  then the lead's join and combine, then the all-reduce.**
- In those ~27 layers the lead's own experts (G) are off the path. In the ~12 layers without a helper pick, G is on it.
- At the MoE all-reduce the members are never on the path.

Four rules follow for anyone choosing the next change:
- shortening attention (on all four cards), P, J, C, the all-reduces or the token tail pays one for one;
- shortening G pays only in the ~12 layers without a helper pick;
- shortening the helper window pays up to the helpers' lateness;
- shortening member-only work pays nothing.

One plain layer with a helper pick (mean µs; t = 0 at the previous MoE all-reduce; one character = 5 µs):

```
                0         50        100       150       200       250       300
                |---------|---------|---------|---------|---------|---------|--
XTX #1 (lead)   aaaaaaaaaaaaaaaaaaaaaaaaaaaaaRppppppppppggggggggggggggJJJccccM
XTX #2-#4       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaRffffffffffffffffffffffff.......M
MI50 #1-#4                                              whhhhhhhhhhhhhhh^done
```

Legend:
- a: attention (148)
- R: attention all-reduce (4)
- p: post, mix, router, publish (49)
- g: the lead's own experts (67)
- J: join (17)
- c: combine (18)
- M: MoE all-reduce (3)
- f: a member's router and FFN share (~113)
- `.`: the member waits for the lead (~33)
- w: helper wake (3-5)
- h: helper window (~75)

The critical path is a → R → p → w → h → J → c → M.

## 2. The time budget

### 2.1 Coarse budget: measured, and adds up

The lead's device timestamps have no gaps between consecutive stamps, so these rows add up to the traced step exactly.

| row (lead) | µs a layer | ms a token | share | label |
|---|---|---|---|---|
| attention block, lumped (includes the token tail in layer 0) | 193.5 | 7.739 | 55.1 % | MEASURED |
| post + mix + router → publish | 48.6 | 1.945 | 13.9 % | MEASURED |
| the lead's own experts, publish → join | 67.1 | 2.616 | 18.6 % | MEASURED |
| join | 16.6 | 0.646 | 4.6 % | MEASURED |
| combine → MoE all-reduce | 20.1 | 0.804 | 5.7 % | MEASURED |
| attention all-reduce | 4.3 | 0.171 | 1.2 % | MEASURED |
| MoE all-reduce | 3.0 | 0.120 | 0.9 % | MEASURED |
| **traced step** | | **14.041** | | MEASURED |
| untraced step (separate 256-token run) | | 14.37-14.41 | | MEASURED |
| **not accounted for** | | **0.33-0.37 (2.5 %)** | | the two runs differ in length and process |

### 2.2 Fine budget: the lumped rows split

| slice | ms a token | label |
|---|---|---|
| attention, 40 × a plain layer | **5.92** | MEASURED median × 40 |
|   kernels: post / mix 0.58, input projections 1.39, RoPE / ring 0.51, sparse attention 0.71, output projection 1.27, put 0.15 | 4.60 | INFERRED (traced durations) |
|   launch gaps | 1.32 | INFERRED (block − kernels) |
| indexer and compressor layers, extra | **0.67** | MEASURED medians; layer 14's split INFERRED |
| Engram layers, extra | **0.32** | MEASURED (layer 1); layer 14's part INFERRED |
| drafter taps, layers 37-39 | **0.05** | MEASURED medians; attribution INFERRED |
| token tail | **≥ 0.74** | ESTIMATE |
|   head GEMV + logits gather + norm | ~0.28 | INFERRED |
|   drafter seeding (drafter off) | ~0.20 | INFERRED |
|   sampler, next-token embed, launches | ~0.10 | INFERRED |
|   host round trip | ~0.19 | INFERRED as the remainder; **not measured** |
| spread between plain layers, mean vs median | 0.04 | arithmetic |
| post + mix + router → publish | **1.95** | MEASURED (1.46 kernels + 0.49 gaps, INFERRED) |
| the lead's own experts | **2.62** | MEASURED (1.61 kernels + 1.00 gaps, INFERRED); **off the path in ~27 layers** |
| join | **0.65** | MEASURED: MI50 #1 0.20, #2 0.14, #3 0.09, #4 0.08, lane 0.05 |
| combine → all-reduce | **0.80** | MEASURED |
| both all-reduces | **0.29** | MEASURED |
| **total** | **14.04** | |

One character = 0.2 ms:

```
|AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIIIEETTTTPPPPPPPPPPGGGGGGGGGGGGGJJJCCCCR|
 0         2         4         6         8         10        12      14 ms
```

Legend:
- A: attention, plain part (5.92)
- I: indexer (0.67)
- E: Engram (0.32)
- T: token tail (0.74+)
- P: router → publish (1.95)
- G: the lead's own experts (2.62)
- J: join (0.65)
- C: combine (0.80)
- R: all-reduces (0.29)

Seen another way, the lead spends about 10.2 ms executing kernels, about 0.9 ms waiting (join and all-reduces), and about
2.9 ms between kernels and on the host. That is about 2 µs per boundary over its ~1,463 launches (INFERRED).

### 2.3 Checks against independent numbers

| check | numbers | agrees? |
|---|---|---|
| "helpers free" oracle vs the join slice | −0.71 ms (HX-20) vs 0.65 ms | yes |
| members' idle time vs the lead's extra work | 1.04 ms vs join 0.65 + about 0.4 ms of other lead-only work | yes, but about 0.2 ms of the "other" is still not attributed to a kernel |
| per-kernel price | 2.63-2.75 µs (HX-20) vs a 1.95 µs gap + a ~0.7 µs empty kernel | yes |
| "empty all-reduce" oracle | +0.47 ms **slower** (HX-20) | **yes, once explained**: the oracle corrupted the routing and tripled the helper traffic (44 → 130 helper experts a token); the all-reduce itself could save ≤ ~0.24 ms. See HX-20's update |

### 2.4 What does not close yet

- **The coarse budget closes:** exactly against the traced step, and within 2.5 % of the untraced step.
- **The fine budget does not.** About 70 % of the step sits in two lumped rows. Their split comes from another run's
  kernel durations, not from timestamps, and the host's share of the token tail is only a remainder.

The run that closes it is written, but has not run yet:
- per-stage device stamps (the attention stages, the FFN entry, head, sampler) plus host stamps around sampling and the
  next token's launch, with the untraced arm at the same token count as the traced one;
- the same at 1K, 32K, 64K and 110K tokens of context.

Its pre-registered pass line: the named stages cover ≥ 97 % of the step, and the stamp-corrected step lands within 3 %
of the untraced one.

### 2.5 The budget after build 12

Build 12 (SV-09) took the B1 step from 14.38 to 13.27 ms (MEASURED, same run, 3 interleaved reps a stack step). Where
that came from, by slice of section 2.2:

| slice (build 11, ms a token) | change | measured effect at B1 | card |
|---|---|---|---|
| attention launch gaps (1.32) and small kernels | `DS4_ATTN_MERGE`: −120 launches | −0.46 ms (alone) | [FG-07](cards/FG-07.md) |
| both all-reduces (0.29) + hyper-connection post | `DS4_AR_POST=3`: −160 launches | −0.20 ms (alone) | [FG-07](cards/FG-07.md) |
| the two above together | | −0.67 ms alone; −0.70 as the first stack step | [FG-07](cards/FG-07.md), [SV-09](cards/SV-09.md) |
| join (0.65) | `DS4_LP_FLAGFUSE`: the MI50's flag from its combine | −0.20 ms; the lead's join 19.5 → 13.2 µs a layer in a timestamped run | [HX-21](cards/HX-21.md) |
| router → publish (1.95) | `DS4_LP_PUBFUSE`; `DS4_LP_ROUTE`; `DS4_AHEAD=0` | −0.11; +0.01 (dropped); +0.02 (~0, as read in section 5) | [HX-21](cards/HX-21.md) |
| combine → all-reduce (0.80) | `DS4_LP_JOINCOMB` | −0.005 | [HX-21](cards/HX-21.md) |
| lead-path set in the stack | | −0.23 ms over the fusions | [SV-09](cards/SV-09.md) |
| helper windows (inside the join) | `DS4_HW_FOLD`, `HEADQ8`, `PUBT` | −0.08 ms over the above | [HX-22](cards/HX-22.md) |
| MI50 expert maths (inside the join) | `DS4_MK_GU`, `DS4_MK_DN` | **+0.33 ms** (slower; off) | [HX-23](cards/HX-23.md) |
| Engram (0.32) | `DS4_EG_SPLIT`: the GEMV split over the four XTX | −0.12 ms | [SV-09](cards/SV-09.md) |
| token tail: drafter seeding (~0.20, estimated) | `DS4_DSPARK_NOSEED` (drafter-off processes only) | −0.30 ms; not in build 12, which serves with the drafter on | [SV-09](cards/SV-09.md) |
| **step** | build 12 | **14.38 → 13.27 ms (−1.11, −7.7 %)** | [SV-09](cards/SV-09.md) |

- The desk estimate for "all the built decode flags together" (section 5) was 14.4 → about 13.0 ms. The measured stack
  reached 13.38 without the Engram split and 13.27 with it: the fusions landed close to their estimate, the lead-path set
  at about half of it.
- The drafter-seeding slice was confirmed and is a little larger than estimated (0.30 against ~0.20-0.25). Every
  drafter-off number in the cards carries it on both sides of each comparison.
- The stage-stamp run that would close the fine budget (section 2.4) has still not run, so the build-12 budget is the
  table above, not a re-cut timeline.

## 3. Long context, the drafter's verify rounds, and prefill

**Long context (drafter off).**
- In the code, only the indexer layers depend on context: layers 2, 8, 14 and 20 grow with it, and layers 24-36 grow
  until 16K candidates and then stay flat. Sparse attention reads a fixed number of selected and window rows.
- Predicted: **+0.15 to +0.37 ms at 32K, +0.4 to +0.95 ms at 110K** (ESTIMATE).
- The server's own counters under agent traffic fit +0.006 ms per 1K tokens of context per trunk pass, about +0.6 ms at
  100K ([SV-08](cards/SV-08.md) data, INFERRED).
- Above 128K the KV cache (about 920 B a token on each XTX) takes VRAM expert slots. The cost then moves to more MI50
  picks, i.e. the join, rather than to attention. Measured at a 262,144-token context: 7,011 experts in VRAM per XTX
  against 7,059 at 131,072, and decode with the drafter at 77-80 tok/s after a 240K prompt ([QA-09](cards/QA-09.md)).

**The drafter's verify rounds** (the production path, [SV-08](cards/SV-08.md)). A cost model fitted to 50 live requests
predicts 11.43 ms a token against 11.47 measured (INFERRED):

    round ≈ draft 2.2-2.4 ms + trunk pass 15.5 ms + 1.85 verified drafts × 3.11 ms ≈ 23.6 ms for 2.09 tokens

- **The draft call** is about 180 kernels on the lead, and its cost does not depend on the block size.
- **Each extra verify row costs 3.11 ms**, about 20 % of a trunk pass. Most of that is the extra row's routed experts and
  MI50 picks, not attention (INFERRED: two-row windows carry twice the helper wait of one-row windows).
- So in production the verify rows are 24 % of a round, and the trunk pass is the token anatomy above.
- A draft is worth verifying while the chance that it and the drafts before it are all kept stays at or above
  3.11 / 11.47 ≈ 0.27. The live per-position rates clear that at block 3.

**Prefill.** It is not in a decode token, but under agent traffic it is a quarter of the wall time (SV-08).
- A prompt of N ≥ 700 new tokens takes **3.26 s + 0.43 ms × N** (fit over 28 live requests; [PF-10](cards/PF-10.md)).
- The 3.26 s floor is each XTX streaming 21 GB of host-RAM experts a chunk at about 6.2 GB/s. The MI50s, which hold
  those experts, are idle meanwhile.
- XTX #4 is the slow member in prefill ([PF-09](cards/PF-09.md)).

## 4. History: which change moved which part of the token

The prompts and layouts differ between rows, so read the table as "which slice moved", not as a controlled series.

| change | slice it moved | measured effect | card |
|---|---|---|---|
| upstream, 1 XTX + host-RAM experts | baseline: 27,087 of 61,200 picks on the CPU | 21.46-22.67 tok/s (~44 ms) | [EB-03](cards/EB-03.md) |
| + 1 MI50 helper | CPU experts → MI50 | 26.12-26.80 tok/s | [EB-01](cards/EB-01.md), [EB-03](cards/EB-03.md) |
| stall fixes; warm-up at load | decode stalls; first-token tail | 0 stalls in 9 processes; 86-89 → 38 ms [R] | [EB-05](cards/EB-05.md), [EB-06](cards/EB-06.md) |
| tensor parallel over 3 XTX | every slice split three ways; adds two all-reduces a layer | 30.4-30.9 tok/s [R] | [TP-01](cards/TP-01.md) |
| no SDMA; no-SDMA helper window | the helper round trip | −1 ms a token; 30.65 → 33.39 tok/s (1 XTX + MI50) [R] | [HX-04](cards/HX-04.md), [HX-06](cards/HX-06.md) |
| every routed expert on a GPU (3 + 3) | the CPU expert lane removed | 48.66 tok/s, against ~41 projected | [TP-04](cards/TP-04.md) |
| one-phase all-reduce; 128-bit loads | all-reduce rounds and bandwidth | +1.2 % [R]; decode set +3.1 % [R] | [AR-02](cards/AR-02.md), [AR-05](cards/AR-05.md) |
| armed helper windows | helper wake | B1 +10.5 % [R]; unstable at B8, later replaced | [HX-09](cards/HX-09.md) |
| a fourth XTX (tp4) | every slice split four ways; longer helper wait | B1 50.26 → 55.10 tok/s; 2K prefill 10.7 % slower | [TP-05](cards/TP-05.md), [HX-13](cards/HX-13.md) |
| decode fusions (HC, router, trim, BF16 router) | about 240-280 launches a token; router bytes | launches −17 %, +7.6 % (1 rep) [R] | [FG-02](cards/FG-02.md) |
| one attention launch per batch | attention launches at B > 1 | B8 +36.6 %; none at B1 | [FG-05](cards/FG-05.md) |
| build 10 | the above + fixes + tuned prompt GEMM | B1 59.32, B8 163.3 tok/s | [SV-07](cards/SV-07.md) |
| MI50 DPM `high` | helper window (no clock ramp) | +4.5 % (58.31 → 60.95 tok/s) | [CP-03](cards/CP-03.md) |
| overlap the helper's down rows; packed staging | helper return | B1 16.90 → 16.57 ms; the helper-wait counter barely moved | [HX-10](cards/HX-10.md) |
| device-direct helper windows | host issue and join → device | host share 2.68 → 0.09 ms; step −0.63 ms (predicted −1.5 to −2.0) | [HX-17](cards/HX-17.md) |
| skip MI50s with no pick | 129 of 156 waits a token | B1 ~0; B8 −5.6 % | [HX-18](cards/HX-18.md) |
| **build 11** | the four rows above | **65.16 tok/s**; 14.4 ms a token on a short prompt | [SV-06](cards/SV-06.md), [HX-20](cards/HX-20.md) |
| attention launches merged; post step and put inside the all-reduce | 280 launches a token on every card | −0.67 ms | [FG-07](cards/FG-07.md) |
| publish in the routing launch; join + combine; the MI50's flag from its combine | publish, join | −0.27 ms alone; −0.23 in the stack | [HX-21](cards/HX-21.md) |
| one graph launch a helper window; int8 rounding in the head; T>1 publish | helper dispatch | −0.08 ms | [HX-22](cards/HX-22.md) |
| Engram GEMV split over the four XTX | Engram | −0.12 ms | [SV-09](cards/SV-09.md) |
| **build 12** | the four rows above | **13.27 ms a token on a short prompt (−7.7 %)**; B8 −6.8 %; production −4.4 to −5.3 % | [SV-09](cards/SV-09.md) |
| the drafter (confidence 0.3, block 3; tensor-parallel blocks; batched) | tokens per trunk pass | +23 % [R]; +34.8 % on live agent traffic | [DR-01](cards/DR-01.md)-[DR-04](cards/DR-04.md), [SV-08](cards/SV-08.md) |

**The misses.** Most of them have the same shape: the change removed a wait or a cost that was not on the critical path
of section 1.5, so the time moved somewhere else.

| change | slice it aimed at | result | why |
|---|---|---|---|
| device-direct helper windows ([HX-17](cards/HX-17.md)) | the host's issue and join | a third of the predicted gain | the host's share went, and the lead then waited on the device for every MI50 instead |
| skip MI50s with no pick ([HX-18](cards/HX-18.md)) | waits for empty MI50s | ~0 at B1 | the empty MI50s were never the late ones |
| helper overlap ([HX-10](cards/HX-10.md)) | the helper wait | −2 %, with the wait unchanged | the time came off something else |
| no lane wait ([HX-20](cards/HX-20.md)) | the CPU lane's answer | −0.1 ms B1, worse at B8; dropped | the lane was never binding |
| all-reduce overlap ([AR-07](cards/AR-07.md)) | the all-reduce, under the next kernels | slower, and not exact with the drafter | the overlapped work sits in the lead's own experts, which are off the path in helper layers |
| fused post-all-reduce norm ([AR-07](cards/AR-07.md)) | the all-reduce tail | ±0 | it shortens every card equally; the lead is still last |
| peer-to-peer decode all-reduce ([AR-03](cards/AR-03.md), [AR-06](cards/AR-06.md)) | the all-reduce transfer | −4 % / −11 % | the prediction used a stale profile; the transfer is a small slice |
| defer helper experts by one layer ([HX-11](cards/HX-11.md)) | the join | B1 −10.8 %, but perplexity +11.7 % | not exact; the passing variants were slower than off |
| MI50 link pin ([CP-02](cards/CP-02.md)); manual clock pin ([CP-04](cards/CP-04.md)) | helper link / clock | −10 %; loses at B8 | the mode froze the core clock |
| hipGraph probe ([FG-03](cards/FG-03.md)) | launch cost | 0.35-0.88 µs a kernel; track closed | borderline against its 0.5 µs line |
| faster MI50 kernels ([HX-15](cards/HX-15.md), [HX-19](cards/HX-19.md)) | MI50 maths | not built / no gain | the maths is about 60 µs of a 155-179 µs window ([HX-14](cards/HX-14.md)) |
| MI50 kernels with 3 waves a SIMD ([HX-23](cards/HX-23.md)) | MI50 maths in the windows the lead waits for | +0.33 ms (slower) | the old kernels were grid-limited, not register-limited; the register cut serialised the lookups and the 6× grid removed no work from the busiest SIMD [I] |
| post step inside the all-reduce alone ([FG-07](cards/FG-07.md)) | a launch an all-reduce | +0.06 ms | the five-block sum kernel's added tail costs more than the launch |
| per-warp routing ([HX-21](cards/HX-21.md)) | the routing kernel (14 µs traced) | +0.01 ms; release → publish unchanged | the traced kernel time did not carry over untraced |
| lookahead switch off ([HX-21](cards/HX-21.md)) | the lookahead GEMV | +0.02 ms | not launched with the BF16 router, as section 5 read |
| join + combine in one launch ([HX-21](cards/HX-21.md)) | a launch at the join | −0.005 ms | the saved launch sits in front of the wait; the lead just waits sooner |

## 5. What's next, by area

The savings are per token at B1 and short context, against the slice in section 2. The table was written before the
A/B runs; the "state" column now gives each measured result (section 6 has the detail). The flags are in the
release commit unless [FEATURES.md](FEATURES.md) says otherwise, and all default off.

| area (slice) | candidate | state | expected saving | exact? |
|---|---|---|---|---|
| attention launches (gaps 1.32 ms + small kernels) | `DS4_ATTN_MERGE`: five small attention launches become two (−120 a token on every card) | **measured −0.46 ms, kept** ([FG-07](cards/FG-07.md)) | −0.30 to −0.70 ms | yes, checked byte for byte at load |
| all-reduce + hyper-connection post | `DS4_AR_POST`: the post step and the put inside the all-reduce's sum kernel (−160 a token) | **measured −0.20 ms, kept** ([FG-07](cards/FG-07.md)) | −0.40 to −0.70 ms | yes; the cross-card flag protocol only end to end |
| output projection | int8 tail for the first output GEMV (−40) | designed | −0.10 to −0.20 ms | yes |
| every launch-bound slice (~3.2 ms of critical launches) | one persistent kernel a layer half | designed | realistic 1.5-2 ms | only if every reduction keeps its order |
| indexer (0.67 ms, grows with context) | split index scoring over heads | proposed | ~0.16 ms | yes by construction |
| indexer at long context | multi-block top-k | proposed | 0 at 1K; 0.35-0.9 ms at 110K (ESTIMATE) | can be made exact |
| Engram (0.32 ms) | `DS4_EG_SPLIT`: the 129 µs GEMV, replicated on every XTX, split by rows and gathered | **measured −0.12 ms, in build 12** ([SV-09](cards/SV-09.md)) | −0.06 to −0.17 ms | yes, by construction |
| token tail: drafter seeding (~0.2 ms) | `DS4_DSPARK_NOSEED`: skip it when the drafter is off for the process | **measured −0.30 ms** (drafter-off only) ([SV-09](cards/SV-09.md)) | −0.20 to −0.25 ms of the drafter-off step | yes |
| token tail: logits gather (~0.12 ms) | each XTX sends its top-k logits instead of its whole quarter | idea | ~0.08-0.11 ms | yes for greedy and top-k sampling |
| token tail: host round trip (~0.19 ms, unmeasured) | sample and embed on the device | needs the stage run first | up to the measured share | yes |
| router → publish (1.95 ms) | `DS4_LP_ROUTE`: per-warp top-6 with one merge, 3 barriers instead of 13 | **measured +0.01 ms, dropped** ([HX-21](cards/HX-21.md)) | ~−0.35 ms | yes, checked at load |
| router → publish | `DS4_LP_PUBFUSE`: the publish inside the routing launch | **measured −0.11 ms, kept** ([HX-21](cards/HX-21.md)) | ~−0.11 ms | yes |
| router → publish | `DS4_AHEAD=0`: drop the lookahead gate | **measured +0.02 ms: ~0, as read here** ([HX-21](cards/HX-21.md)) | the A/B's prediction is −0.15 to −0.30 ms. **Our reading of the build-11 trace is ~0**: with the BF16 router the paired lookahead kernel is not launched | yes |
| join (0.65 ms) | `DS4_LP_FLAGFUSE`: the MI50's done flag written by its last block, not a separate kernel | **measured −0.20 ms, kept** ([HX-21](cards/HX-21.md)) | −0.10 to −0.30 ms | yes |
| join | the MI50's request handling inside its expert graph (`DS4_HW_FOLD`, with `HEADQ8` and `PUBT`) | **measured −0.08 ms for the three, kept** ([HX-22](cards/HX-22.md)) | another 5-8 µs a helper layer | yes |
| join | faster MI50 expert kernels (`DS4_MK_GU`, `DS4_MK_DN`): 3 waves a SIMD instead of 2 | **measured +0.33 ms (slower), off; cause inferred from the code objects** ([HX-23](cards/HX-23.md)) | −0.05 to −0.15 ms on top of the above | yes |
| join | rebalance experts between the lead and the members | idea | ceiling 1.0 ms (the members' idle time) | depends |
| join | defer helper experts | shelved ([HX-11](cards/HX-11.md)) | measured −10.8 %, quality fails | no |
| combine → all-reduce (0.80 ms) | `DS4_LP_JOINCOMB`: join and combine in one launch | **measured −0.005 ms**; kept in the stack ([HX-21](cards/HX-21.md)) | −0.05 to −0.15 ms | yes |
| the lead's own experts (2.62 ms) | fuse its expert launches | later | **~0 until the MI50 window fits inside them** | yes |
| XTX #4 | equalise its power cap ([HW-08](cards/HW-08.md)) | deferred | decode ≤ 0.1 ms; prefill up to 38 % ([PF-09](cards/PF-09.md)) | n/a |
| verify rows (3.11 ms a row, 24 % of a round) | share experts across a verify window's rows; batch MI50 rows for windows | no design yet | 3.1 → 2.0 ms a row would give about +8 % tok/s (INFERRED from the model) | yes if sums keep their order |
| verify policy | adapt the drafts per round to observed acceptance | **measured: equal to build 12 at matched output length** ([SV-09](cards/SV-09.md), [DR-06](cards/DR-06.md)) | +0 to +3 % | the keep rule is unchanged |
| verify policy | draw drafts from the drafter's own distribution | existing switch | −3 to +8 % | in distribution |
| draft call (2.2 ms, ~180 kernels) | fuse the drafter's block chain | idea | ≤ ~1 ms a round (ESTIMATE) | yes if bodies are reused |
| prefill floor (3.26 s a chunk) | `DS4_PF_HELPER`: let the MI50s compute the streamed experts in prefill (a second look at [PF-06](cards/PF-06.md), now aimed at the per-chunk floor) | **measured: 1K suffix 3.42 → 1.33 s; not bit-identical, validation pending** ([PF-11](cards/PF-11.md)) | a 1-2K suffix from 3.7-4.1 s to 0.8-1.8 s | no: quality-gated |
| prefill stream | lay each member's down parts out contiguously, so the whole stream is DMA | proposed, conditional | floor to 1.5-2.0 s, only if a raw DMA test shows ≥ 15 GB/s a card | yes |

**What all the built decode flags might give together** (ESTIMATE, written before the runs; the measured stack is in
section 2.5: 14.38 → 13.38 ms without the Engram split, 13.27 ms with it):
- the attention and all-reduce fusions: about −0.95 ms;
- the lead-path set without the lookahead switch: about −0.5 ms.

They mostly hit different slices, so together they would take 14.4 ms to about 13.0 ms (about 77 tok/s), with a range of
12.5-13.5 ms. Two pairs overlap and should not be added blindly:
- the joined join-and-combine and the put-in-sum both touch the combine → all-reduce slice;
- two of the fusion candidates are the same change as two lead-path flags.

**What's next, after build 12.** Ranked by the measured slice each one targets:
1. **Verify rows** (3.11 ms a row, about a quarter of a production round; [DR-06](cards/DR-06.md)): no design yet. The
   largest production lever, about +8 % if a row fell to 2.0 ms.
2. **The prefill floor** (3.26 s a chunk, 58 % of prompt time under agent traffic; [PF-10](cards/PF-10.md)): the MI50
   prefill helper cut a 1K-token suffix from 3.42 to 1.33 s ([PF-11](cards/PF-11.md)), but it is not bit-identical;
   its validation comes first.
3. **The join that is left** (~13 µs a layer after FLAGFUSE). An exact re-tile of the MI50 expert kernels tops out at
   about −0.1 ms ([HX-23](cards/HX-23.md)); what would move it is fewer instructions per weight (a format change, which
   moves the expert placement) or one dispatch fewer a window (rounding the hidden activation inside gate/up).
4. **Launch count on the lead** (~1,183 a token after the fusions): the next exact fusions (an int8 tail for the first
   output GEMV; join with combine already done) and a 20-second measurement of graph replay's per-kernel cost on the GPU,
   which decides between graphs and a persistent kernel.
5. **The Engram chain off the critical path** on a second XTX stream (designed): it would also apply to verify windows,
   which the split does not.
6. **The stage-stamp run** that closes the fine budget, and the same at 32K-110K of context.
7. **A code-size check after every build** ([QA-08](cards/QA-08.md)), so expert placement cannot move silently.

## 6. A/B results

All arms were checked for byte identity before timing; every one was IDENTICAL to its reference. B1 is plain decode,
64-token prompt, 256 tokens, drafter off. Predictions are copied unchanged from the pre-registered run scripts.

**Fusion** ([FG-07](cards/FG-07.md); 3 interleaved reps an arm; off = 14.345 ms):

| arm | launches a token (lead), predicted | measured | ms vs off, predicted | measured | identity |
|---|---|---|---|---|---|
| off | 1,463.5 | (build-11 trace) | 0 | 14.345 | IDENTICAL to build 11 |
| `DS4_AR_POST=1` | −80 | — | −0.20 to −0.35 | **+0.060** (disproved: slower) | IDENTICAL |
| `DS4_AR_POST=3` | −160 | — | −0.40 to −0.70 | −0.202 (under) | IDENTICAL |
| `DS4_ATTN_MERGE=1` | −120 | — | −0.30 to −0.70 | −0.461 (held) | IDENTICAL |
| both | 1,183.5 | 1,183.5 [R] | −0.70 to −1.40 | −0.667 (just under) | IDENTICAL |

- Drafter on (confidence 0.4, block 3, greedy): 10.764 → 10.305 ms a token (−4.3 %).
- Per removed launch: ~3.8 µs for the attention merge, ~1.3 µs for the all-reduce fold [I]. The fused sum kernel gives
  part of its saving back.

**Lead path** ([HX-21](cards/HX-21.md); one run an arm; deltas against the mean of two base runs, 14.323 ms):

| arm | B1 ms, predicted by the run script | our expectation (section 5) | B1 measured | B8 measured | identity |
|---|---|---|---|---|---|
| `DS4_AHEAD=0` | −0.15 to −0.30 | ~0 | +0.020 | unusable (drift) | IDENTICAL |
| + `DS4_LP_ROUTE` | −0.35 to −0.60 | −0.20 to −0.45 | +0.014 | unusable | IDENTICAL |
| + `DS4_LP_PUBFUSE` | −0.25 to −0.40 | ~−0.1 | −0.106 | unusable | IDENTICAL |
| `DS4_LP_JOINCOMB` | −0.05 to −0.15 | same | −0.005 | unusable | IDENTICAL |
| `DS4_LP_FLAGFUSE` | −0.10 to −0.30 | same | −0.201 | unusable | IDENTICAL |
| all | −0.55 to −1.00 | about −0.5 | −0.268 | unusable | IDENTICAL |

Slice checks from the timestamp arms (means, µs a layer): release → publish 48.5 at base, **49.0** with AHEAD=0 +
ROUTE (predicted ~33: disproved); join 19.5 → **13.2** with FLAGFUSE (predicted 9-12; the median, 8.8, is inside); the
members' margin at the MoE all-reduce 35.7 → **29.7** with all three timed flags (predicted ~19: disproved). The B8 base
runs drifted from 44.5 to 52.2 ms a step, so no B8 effect is claimed.

Reading [I]: cuts on the lead before or at its join are absorbed by the join wait in the ~27 layers where a helper has a
pick: the lead simply starts waiting sooner. Only the change that made the helpers earlier paid one for one.

**Build 12 stack** ([SV-09](cards/SV-09.md); 3 interleaved reps a step; off = 14.384 ms):

| stack | B1 ms, predicted vs off | measured | B8 ms a step |
|---|---|---|---|
| F (the two fusions) | −0.60 to −0.73 | −0.699 | |
| F + lead path | −0.75 to −1.00 | −0.932 | |
| F + lead path + helper windows | −0.85 to −1.25 | **−1.009** | 41.33 → 38.53 (−6.8 %) |
| + MI50 kernels | −0.90 to −1.40 | −0.680 (+0.33 over the previous step) | |
| + Engram split (build 12) | −0.06 to −0.17 more | −0.122 more: **13.27 ms** | |
| production, drafter on, sampled | −0.35 to −0.80 ms a generated token | −0.65 (long reps) / −0.49 (short rep), at matched output length | |

## 7. Caveats

- One box, one ROCm version, one quantised pack, a short prompt. The timeline is one run of 63 tokens. The untraced step
  is three runs of 256 tokens.
- The fine budget uses a traced run's kernel durations. If the tracer let the clocks drop between kernels, those
  durations are a little long and the inferred gap shares a little short.
- The prompts are private. As with the cards, compare ratios within a run, not absolute numbers across runs.
- The two observations this note made from the code and one trace (drafter seeding with the drafter off, and the
  lookahead switch being inactive with the BF16 router) have both been tested: skipping the seeding saved 0.30 ms
  (SV-09), and `DS4_AHEAD=0` changed nothing (+0.02 ms, HX-21).
- The A/B results in section 6 are at short context. Long-context decode has not been re-measured on build 12.
