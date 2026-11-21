# Anatomy of a Flash-Next decode on Gary

How Gary (this fork of Strata) serves Qwen3.8 Flash-Next on two RX 7900 XTX and four Instinct MI50: what each part does, why
it is built that way, where the time goes, where it waits, and what we still do not know. It covers decode, the timing study
that compared tensor parallel with one card, prompt reading (in short; the long form is [PREFILL.md](PREFILL.md)), the decode
kernels and the CPU pool's wait between layers.

Labels: **MEASURED** = read from a run file or a script-written summary; **ESTIMATE** = our arithmetic from measured numbers;
**INFERRED** = read from the code, not measured. Run data: [data/](data/). Numbers in sections 2 and 3 come from the cycle-6
timing sweep, whose extract is not published; its untimed totals are in [data/2026-11-20-tp2-cycles.md](data/2026-11-20-tp2-cycles.md).

**A warning about stage numbers.** Every per-stage number comes from a timing build (`STRATA_TP_TIMING=2` under tensor
parallel; `STRATA_VERIFY_PROFILE=1` on one card). It changes the graph (no shared-expert side stream, no fusions, stamps copied
to the host after every window) and runs 1.10-1.21x slower than the untimed engine. Read the classes as shares of the time;
the untimed window is given beside each table.

## Contents

1. Model facts that shape everything
2. The life of a request, and what runs where
3. One decode window under tensor parallel
4. tp2 vs one card: the comparative timing study
5. Decode kernels per quant
6. Prompt reading in short
7. The CPU pool's wait between layers
8. What we changed because of this, and what is left

---

## 1. Model facts that shape everything

| Fact | Value |
|---|---|
| Layers | 48. Every 4th one (3, 7, ... 47) is full attention: 12 attention layers (sparse attention with an indexer that selects the top 2,048 cells), 36 GDN layers (gated delta net, linear attention with a recurrent state) |
| Width | n_embd 2,560, vocabulary 248,320 |
| Attention | 24 query heads, 2 KV heads, head_dim 256; tensor parallel 2 gives each card 12 q / 1 KV head |
| GDN | 16 key groups, 48 value heads, state 128; tensor parallel splits the heads and the state |
| Hyper-connections | 4 residual streams (low rank 320). Every layer reads (mixes) them before the mixer and before the FFN, and writes them after |
| MoE | 512 routed experts a layer (24,576 in all), top-10, n_ff 640, plus a shared expert of the same width. One expert = 3 x 2,560 x 640 weights: ~2.46 MB on average in UD-IQ4_XS, 2.97 MB in our Q4_K |
| PLE | per-layer n-gram embeddings: a host table gathered per token and fed in at layer 1 |
| MTP | a drafter head; on one card it roughly doubles one user's speed. Not ported to tensor parallel |

Routing is nearly flat: a card holding 49 % of the experts sees 52 % of the routed entries, and rows in a batch barely share
experts (1.04 rows per distinct expert in a 4-row window) [MEASURED:
[cycle 4](data/2026-11-20-tp2-cycle4-step2-cpu-experts.txt)]. So placement by popularity gives little, and an expert pass is a
weight stream: every routed expert's bytes are read once per window.

## 2. The life of a request, and what runs where

```
client -> server (chat template, tokenizer) -> scheduler (slots, --batch N)
  PROMPT:   one card: chunked batched prompt path (8,192-token chunks), prompt cache;
            tensor parallel: read through the 8-token decode windows on both cards, no prompt cache (section 6)
  DECODE:   1 user  -> the solo window;  2+ users -> one batch window with a row per active slot
            per layer: hc read -> mixer (GDN or attention) -> out-proj -> [all-reduce 1]
                       -> hc read (FFN side) -> router -> plan + doorbell -> shared expert
                       -> routed experts (own VRAM | MI50 helpers | CPU pool) -> combine -> [all-reduce 2] -> hc write
            head: hc mix -> lm head -> argmax / sampler
  COMMIT:   batch windows advance their slots' state inside the graph (self-commit), tokens are streamed out
```

What runs where in a decode window (tensor parallel 2 + 4 MI50, the cycle-6 layout):

| Who | What it does | Why |
|---|---|---|
| Host thread (one) | stages inputs, launches card 0's graph then card 1's, spins on card 0's per-layer doorbell, runs the expert pass for card 0 (claims helper rows, posts them, joins), syncs, samples, emits | card 0 owns every expert not in card 1's VRAM ("ownership by residency"), so only card 0 asks the host. One pass a layer, one thread, deterministic order |
| Helper threads (one per MI50) | replay the helper graph (fetch the inputs from host memory, expert kernels, push the rows to host memory, raise a flag); spin on the flag | concurrent issue cut the host helper pass from 14.1 to 6.8 ms a window at 4 users [MEASURED] |
| Card 0 (XTX A) | half of every mixer, the router and head (replicated), the shared expert, its VRAM experts, the wait for helper rows, combine, its side of 96 all-reduces. Layer-steps whose experts are all on the two XTX are planned on the device and skip the host | the critical path: it is last to the MoE all-reduce in every layer-window [MEASURED] |
| Card 1 (XTX B) | the other half of the mixers, its VRAM experts, router and head again. Its graph is planned on the device and has no host waits except PLE | it never needs the host |
| MI50 x 4 | each holds ~3,500 experts (Q4_K) and runs ~46 small launches a window of ~3 experts at 4 users | CPU expert path 0 of 24,576 |
| CPU pool | idle in decode on this layout | |

The all-reduce is **host-staged**: each card writes its partial into host-mapped rows plus a flag, spins until the peer's
partial is in, and sums in member order, so both cards hold identical bits. Measured at 11.8 us for 20 KiB in the first proof.
A peer-to-peer push all-reduce was measured slower in the DeepSeek work on the same machine, so host staging stays.

## 3. One decode window under tensor parallel

Card 0, ms per window, cycle-6 timing build (1 user = solo graph; 4 users = batch graph, 3.98 rows) [MEASURED]:

| Stage | What it does | 1 user | 4 users |
|---|---|---|---|
| gdn-proj (36 layers) | hc read, qkv/z/a/b projections of this card's heads, conv | 3.74 | 4.81 |
| gdn-rec | the delta-rule state update + norm, per row | 0.31 | 1.13 |
| attn (12 layers) | hc read, q/k/v, indexer, top-2,048 select, sparse attention | 2.34 | 4.26 |
| out-proj | this card's heads, a partial sum | 0.84 | 1.07 |
| all-reduce 1 (mixer) | transfer / **wait** / sum | 0.18 / **0.55** / 0.25 | 0.23 / **2.14** / 0.37 |
| hc1 + router | FFN-side hc read, router GEMV, top-10, plan, doorbell | 3.30 | 3.94 |
| shared | shared expert + q8_1 of the MoE input; computed on both cards, zeroed on the one that does not own the layer | 1.58 | 1.73 |
| wait-plan | copy the host's plan (skipped when planned on the device) | 0.34 | 0.53 |
| vram-exp | grouped expert kernels over this card's VRAM experts | 1.54 | 3.42 |
| wait for helper rows | card 0 spins until the host has the MI50 rows in mapped memory | **2.42** | **2.28** |
| combine | copy helper rows, add VRAM hits, weight, add shared expert and residual | 0.90 | 1.37 |
| all-reduce 2 (MoE) | transfer / wait / sum | 0.18 / 0.06 / 0.25 | 0.23 / 0.09 / 0.39 |
| head | hc mix, lm head (replicated), argmax | ~0.84 | ~1.21 |
| **GPU window, card 0** | | **19.60** | **29.21** |
| untimed wall per window | | 20.1 (49.7 tok/s) | 30.1 (132.9 all-active) |

Who waits for whom:
- **Card 1 waits for card 0** at every MoE all-reduce: 3.1 ms (1 user) and 3.7 ms (4 users) a window, exactly card 0's
  helper-related extra work (helper wait, heavier combine, plan copy). Card 0 is the critical path, and inside it the helper
  lane.
- **Card 0 waits for card 1 at layer 0's first all-reduce**: 0.47 ms (1 user), 1.77 ms (4 users) in that one call, because card
  1's graph is launched only after card 0's launch returns on the same host thread.
- **The host spins** 14.7 ms (1 user) / 21.4 ms (4 users) of each window on card 0's doorbell: no cost on the GPU path, but one
  core at 100 %.

Batch scaling, untimed: window_ms ~ 13.7 + 4.4 x B over B = 2, 4, 8 [ESTIMATE]. About 60 % of a 4-user window is fixed per
window (dense weight streaming plus launch and sync floors). That is why batching is cheap per row, and why the bus scales. The
curve is convex from 4 to 8 rows; section 4 found the cause.

Sync points that matter: the host doorbell spin; GPU spin kernels on host flags (plan, PCIe share, helper rows); the helper round
trip (claim, post, replay, flag, join; the slowest MI50 sets each layer's wait); 96 all-reduces a window; layer-0 launch skew;
the layer-1 PLE wait (+0.13-0.28 ms); end-of-window syncs; graph capture on the first use of each shape.

## 4. tp2 vs one card: the comparative timing study

Arms: **T** = tp2 + 4 MI50; **S** = one XTX + 2 MI50 (37 % of experts on the CPU); **S+MTP** = S with the drafter;
**S1** = one XTX alone. Full tables: [data/2026-11-21-comparative-timing.md](data/2026-11-21-comparative-timing.md).

1. **Tensor parallel barely shortens the dense path.** Card 0's dense time per window is 11.97 ms on T against 12.80 ms on one
   card doing every head: -6 % at 1 user, -12 % at 8 [MEASURED]. A window is a chain of ~2,300 (1 row) to ~3,650 (8 rows) small
   kernels per card; halving the heads halves only the big projections, while hc reads, router, lm head and norms are
   replicated. T then pays 2.4 / 3.5 / 5.3 ms of all-reduce and peer wait on card 0 and leaves card 1 idle 3.0 / 3.8 / 5.1 ms
   (1 / 4 / 8 users). T's real gain is memory: CPU path 0 % against 37 %.
2. **Per-agent decode** at 1 / 2 / 4 / 8 users: T 50.5 / 40.6 / 31.6 / 18.8 tok/s; S 50.5 / 37.9 / 24.6 / 13.5; S+MTP 76.3 /
   49.3 / 26.7 / 13.2; S1 39.7 / 24.9 / 14.7 [MEASURED]. Slots at >= 40 tok/s per agent: T 2, S 1, S+MTP 2, S1 0.
3. **The one-card server's cost is the CPU path, and the CPU path is DRAM-bound.** Its host pass is 5.4 / 14.2 / 25.0 ms per
   window at 1 / 4 / 8 users, in series with the GPU, and it streams 19-35 GB/s of DRAM [MEASURED]. The machine's practical
   ceiling is 127 GB/s. Two S servers at 4 + 4 users use 54 GB/s, and any extra DRAM load slows them at once (an 8-thread triad
   hog costs 15 %).
4. **CPU threads and host RAM run out before DRAM on a third process.** An S server keeps all its pool workers spinning even at
   one user (~42 threads; T ~6). One server process uses 113-120 GB of host RAM; two use 190 GB of 251 [MEASURED].
5. **The 4 -> 8 user convexity is the dense multi-row kernels.** Every Q6_K multi-column matvec at 8 rows runs at 227-271 GB/s
   against 518-612 at 4 rows (2.2-2.4x the time) [MEASURED].
6. **`STRATA_B6_MMVQ_ROWS=1` is exact and saves 2.4 ms (T) / 3.9 ms (S) per 8-row window**, but `STRATA_TSUM=1` switched it
   off. Fixed the same day (B6 under TSUM, card [NX-09](cards/NX-09.md)).
7. **`STRATA_BATCH_SELF_COMMIT=1` on one card** is exact and worth +5 % / +6 % aggregate at 4 / 8 users ([NX-03](cards/NX-03.md)).
8. **Real agent turns behave the same way.** S+MTP on 46-58k-token contexts: 74-75 tok/s alone, 48-52 per agent at 2 agents,
   25-29 at 4; TTFT 3.6-4.5 s when the prefix is reused, but 44 s cold alone and 120-283 s at 2-4 agents, because prompts are
   read one at a time ([PREFILL.md](PREFILL.md)).

The helper lane under T: one MI50 (#3) has the same kernel time as the others but 2.4-2.9x their PCIe transfer time, so its host
wait is 4.8 / 8.0 ms against 2.9-5.1 at 4 / 8 users. The straggler is its link, not its kernels. The shared pool gives that card
less work as a tuning step ([SHARED-POOL.md](SHARED-POOL.md) section 9).

These findings led to the shared MI50 pool: keep one-card engines (no all-reduce, no idle second card) but give them all the
MI50s so nothing runs on the CPU.

## 5. Decode kernels per quant

Experts on every GPU go through one entry point, `native_expert_grouped`; batch 1 is a group with one entry. Measured per
expert on real bytes, cold, against llama.cpp on the same bytes [MEASURED:
[data/2026-11-21-kernel-bandwidth.md](data/2026-11-21-kernel-bandwidth.md)]:

- **Coverage: complete.** Every type in both files (UD-IQ4_XS and our Q4_K) has a GPU kernel on both architectures; none falls
  back to the CPU.
- **The ceiling is the expert's size.** A cold 2.83 MiB read reaches only 282 GB/s on the XTX and 273 on the MI50, against
  920 / 866 for a large read. At batch 1 our XTX expert kernels take 0.5-1.0x llama.cpp's GPU time at 33-50 % of that ceiling;
  at 8 rows they are 1.4-2.1x faster. The Q4_K expert is our fastest at batch 1 (21.3 us).
- **MI50:** Q4_K is our best MI50 path (1.1x llama.cpp's GPU time). The UD file's IQ3_XXS / IQ4_XS gate/up kernels are 1.45-1.8x
  slower, so the Q4_K file is also the better file for the helpers.
- **Dense Q6_K matvecs fall behind with rows**: fine at 1 row, 1.4x llama.cpp at 4 rows, 2.0-2.2x at 8. B6 under TSUM closes
  part of that ([NX-09](cards/NX-09.md)); the rest is the next kernel item.
- `STRATA_XTX_FAST_PERM=1` (IQ4 codebook lookup by `v_perm` on gfx1100) is bitwise identical and speeds IQ4_XS / IQ4_NL by
  3-9 %; it does nothing for the Q4_K file.
- Q5_0 down has no decode-once multi-row kernel; its cost grows 1.8x from 1 to 8 rows. In decode, rows per expert stay near 1,
  so it costs little today.

## 6. Prompt reading in short

On one card a prompt is read in 8,192-token chunks and is compute-bound on the XTX: 57 % of a chunk was the experts' FP16
GEMMs, and the gate/up GEMM ran on a slow default kernel. Tuned hipBLASLt rows made a cold 52k prompt 28 % faster, and are the
exactness reference since 21 November. Prompts serialise by design; an opt-in concurrent read keeps a decoding agent moving
beside someone else's prompt, and lent experts on the GPU make that decode exact. Details: [PREFILL.md](PREFILL.md).

Under tensor parallel there is no batched prompt path yet: every prompt goes through the 8-token windows (1,631 tokens took
19-25 s). That blocks agent use of tp2 today.

## 7. The CPU pool's wait between layers

On a one-card server with a CPU expert path, all pool workers spun at 100 % from the first token at every load, yet were idle
77-87 % of the time at 1 user and 56-60 % at 8: a layer is two short phases (gate/up rows, then down rows; 43 us p50 at 1 user,
260 us at 8), then ~300-850 us of nothing while the GPU runs the dense part [MEASURED:
[data/2026-11-21-pool-spin.md](data/2026-11-21-pool-spin.md)].

- Plain sleeping costs speed: a futex wake of ~25 sleepers starts the last worker 27 us (p50) / 41-146 us (p90) late, and every
  layer's first phase waits for it (-1.5 to -3 %).
- **`STRATA_POOL_WAIT=pred`** (opt-in, Linux): the host predicts. After each layer's last phase it posts an absolute wake time:
  now + the smallest gap this same layer had in the last 4 windows, minus 30 us. Workers nap to it with a timed futex, spin at
  most 200 us past it, then block. Inside a layer the post says "now". A parked worker keeps spinning until the post for its
  phase lands. Result: the last worker starts 0.6 us (p50) / 1.0 us (p90) after the publish, the same as spinning.
- It is exact by construction (the same row ranges, claimed dynamically) and measured so (every text identical). Speed within
  noise (MTP off) or +1-2 % (MTP on). It frees 17-20 cores and about 10-14 W on one server (the cells without a concurrent compile job; [NX-13](cards/NX-13.md)).

## 8. What we changed because of this, and what is left

Changed (each is a card): fast MI50 round trip ([NX-06](cards/NX-06.md)); self-commit on one card ([NX-03](cards/NX-03.md));
B6 under TSUM ([NX-09](cards/NX-09.md)); penalties in batch windows ([NX-08](cards/NX-08.md)); tuned prompt GEMM rows and the
concurrent read ([NX-10](cards/NX-10.md), [NX-11](cards/NX-11.md)); predicted pool wake ([NX-13](cards/NX-13.md)); and the
shared MI50 pool ([NX-14](cards/NX-14.md)).

Left, ranked by expected gain [ESTIMATE]:

| Item | Evidence | Expected gain |
|---|---|---|
| MTP drafter on tensor parallel | one card: +51 % per agent at 1 user, +30 % at 2 | the low-user rows of tp2 |
| 8-row dense matvecs at 4-row efficiency (lm head first) | 8-row Q6_K calls still 300-450 GB/s against 450-600 at 4 rows | ~1-1.5 ms per 8-row window per card |
| Other 8-row kernels: hc read up, router, shared gate/up | 3x their 1-row time at 8 rows | -1 to -2 ms at 8 users (+3-4 %) |
| Fewer, fused small kernels | 2,300-3,650 kernels per card per window | -1 to -2 ms at 1 user (+5-10 %) |
| tp2: launch both graphs before staging, or card 1 first | layer-0 skew 0.5-1.8 ms | +5-7 % on tp2 |
| tp2: shared expert computed once | both cards compute it, one zeroes it | +2-4 % on tp2 |
| tp2: batched prompt path and prompt cache | 1,631 tokens in 19-25 s today | unblocks agent use of tp2 |
| Helper threads nap like the pool | three host threads still spin per server | cores and power, not speed |

Not ranked, and argued against: two half-batch windows interleaved per card. It was estimated at +10-15 % when the CPU expert
pass was ~19 ms a window; with the experts on the MI50s the hideable waits are 2-4 ms, while the fixed ~16 ms of weight streaming
would be paid twice.
