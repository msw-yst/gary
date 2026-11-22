# Prompt reading (prefill) for Flash-Next

Agent work sends long prompts: a coding agent's turn carries a 10-60k-token context, mostly reused from the previous turn, but
the first turn and every cache miss are read cold. This file covers how a one-card Gary server reads a prompt, where the time
goes, why concurrent prompts serialise, and the two cycles of changes made on 21 November 2026.

Run data: [data/2026-11-21-prefill.md](data/2026-11-21-prefill.md). Cards: [NX-10](cards/NX-10.md), [NX-11](cards/NX-11.md),
[NX-12](cards/NX-12.md). Labels as in [README.md](README.md#how-to-read-the-numbers).

Server: one XTX with two MI50 helper caches, MTP drafter on (`STRATA_BATCH_MTP=1`), prompt cache on,
`STRATA_BATCH_SELF_COMMIT=1`, PARALLEL 4, calibrated Q4_K. Prompts are distinct slices of source code; "50k" means 38-52k tokens.

## Summary

- **A prompt is compute-bound on the XTX, and before the change 57 % of it was the experts' FP16 GEMMs.** The expert gate/up GEMM
  shape (N 1,280, K 2,560) had no row in the packaged hipBLASLt table, so it ran on rocBLAS's default FMA kernel at ~2.4x the
  cost per FLOP of the tuned down GEMM [MEASURED].
- **Tuned rows for that GEMM** (hipBLASLt solutions 1145 / 1146 / 1147 by rows per expert) read a cold 52k prompt in 36.3 s
  instead of 50.7 s (-28 %); an 8K chunk takes 5.40 s instead of 7.60 [MEASURED]. Matrix cores sum in a different order, so this
  changed the output bits. The maintainers chose to re-baseline exactness to it; `STRATA_PF_GU_LEGACY=1` restores the old kernel
  and the old bits.
- **Prompts serialise by design**, at three levels: the server admits one prompt read at a time, the engine's main loop reads
  one prompt at a time, and decoding slots run only between that prompt's chunks. One read already fills the XTX, so no lock is
  worth removing; what can change is who waits for whom.
- **`STRATA_BATCH_CONCURRENT_READ=2`** (opt-in) reads a prompt on its own thread beside the decoding slots. A cold ~45k prompt
  arriving while another agent decodes is read in 41.5 s instead of 62.6 s, and the decoding agent never stops for more than
  0.5 s (was 7.8 s), at ~11 tok/s during the read instead of a bursty ~22 [MEASURED, with the tuned rows and lent experts].
- **`STRATA_BATCH_LENT_GPU=1`** (opt-in) makes decode beside a read exact in the test below (4 x 12k prompts at once == alone,
  16/16). Before it, a slot decoding while another prompt was read computed the read's borrowed experts on the CPU, which
  rounds differently, so its text depended on timing [MEASURED]. On the shared pool, where it had not been set, the same
  mechanism made agent turns differ from run to run; `STRATA_BATCH_LENT_EXACT=1` (development branch) turns it on with a
  guarantee and a counter ([NX-23](cards/NX-23.md)).

## How a prompt is read on one card

- `--prefill auto` gives 8,192-token chunks. Each read borrows 1,556 expert-cache slots (4.30 GiB) for its buffers plus a
  384-slot staging ring, and returns them after the read. The expert arena in host RAM is pinned (68 GB).
- Per chunk, every layer: hyper-connection read, the mixer (GDN chunked recurrence, or sparse attention with its indexer),
  router and shared expert, then the routed experts. With 81,920 (token, expert) rows per chunk nearly all 512 experts of every
  layer are routed. About 13 % are in the XTX's own slots; the rest are DMA'd from the pinned arena through the ring (~63 GB
  per chunk at ~8 GB/s, hidden behind compute). The MI50s do not take part in prompt reading.
- K-quant experts take the FP16 path: each expert is dequantised to FP16 and run as two GEMMs (~160 rows x 1,280 x 2,560 and
  ~160 x 2,560 x 640): 1,024 GEMM calls per layer, 49k per chunk.
- The chunk size is part of the arithmetic: changing it (for example `--prefill auto:16384`) changes the bits.

## Where the time goes (before the change)

Per 8,192-token chunk of a 47.9k prompt; GPU timeline 7.6-8.0 s [MEASURED]:

| Class | ms per chunk | share |
|---|---|---|
| experts: FP16 gate/up GEMM (default kernel) | 3,180 | 40 % |
| experts: dequant to FP16 | 635 | 8 % |
| experts: FP16 down GEMM (hipBLASLt 1147) | 663 | 8.4 % |
| experts: waiting for streamed blobs, grouping, gather, combine | 330 | 4.2 % |
| attention, 12 layers | 1,170 -> 1,590 over the prompt | 15-20 % |
| GDN, 36 layers | 880 | 11 % |
| hyper-connection reads and writes | 400 | 5 % |
| PLE | 245 | 3 % |
| router + shared expert | 85 | 1 % |

A 50k prompt read at ~1,010-1,060 tok/s, an 8k one at 950-1,010. A kernel trace put the XTX 78 % busy during a read, the rest
being ~128k short gaps between per-expert kernels.

## Cycle 1: concurrency, and why the fast kernel was not adopted at first

Alternating A / B / C, then D / A / D [MEASURED]:

| Arm | c1 TTFT (52k) | 4 cold prompts at once: last first token | stream beside a read: tok/s / longest stall / read window | exact? |
|---|---|---|---|---|
| A baseline | 50.7 s | 201.3 s | 21.7 / 7.7 s / 62.3 s | reference |
| B four expert streams | 51.2 | 203.3 | 21.0 / 7.8 / 62.9 | yes, but slower: removed |
| D concurrent read 2 | 50.6 | 194.6 | 9.9 / 0.6 / 48.6 | yes |
| T one borrowed hipBLASLt row | 38.2 | 157.3 | 21.2 / 5.8 / 46.7 | **no** |

`GPU_MAX_HW_QUEUES=16` was slower; `STRATA_MMQ_KQ=1` had no effect on this file (its Q5_0 down stays on the FP16 path).

## Cycle 2: the tuned rows become the reference; exact decode beside a read

1. **Tuned rows.** hipBLASLt has 7 algorithms for f16-in / f32-out on gfx1100. Per rows-per-expert M the fastest is 1145
   (M <= 136), 1146 (137-200), 1147 (201+). Half of all experts have M <= 64, where 1145 takes 20-25 us against the old
   kernel's 65-107. The packaged table `tools/hip/gfx1100-hipblaslt-100200.txt` now carries these rows, plus 1145 for the down
   GEMM at M <= 32 (bit-identical to 1147 there).
2. **What changed numerically.** GEMM max |difference| 8-9e-6 of the largest output; relative L2 against an FP64 product 7.9e-6
   (matrix cores) against 9.0e-7 (old kernel); decoded text first diverges at token 21 of 256, with mean top-5 KL 3.0e-3 before
   it. Same precision class (FP16 products, FP32 sums), different order. New reference hashes are in the data file.
3. **Batched FP16 conversion** (`STRATA_PF_DQ_BATCH=G`): exact but 2.7-6.1 % slower; a group's FP16 weights no longer stay in
   the 96 MB Infinity Cache between dequant and GEMM [INFERRED]. Left off.
4. **Lent experts on the GPU** (`STRATA_BATCH_LENT_GPU=1`): a lent expert goes to a VRAM group pointing at its blob's device
   alias in the pinned arena (the hit kernel, in the hit launch) instead of the CPU. 4 x 12k prompts sent at once == alone 16/16
   in four launches (1/4 without it).

| Arm | c1 TTFT | 4 cold: last first token | stream beside a read | runs agree? |
|---|---|---|---|---|
| A cycle-1 baseline | 50.8 s | 202.6 s | 21.8 tok/s / 7.8 s stall / 62.6 s read | no |
| N tuned rows (new default) | 36.3 | 152.1 | 20.6 / 5.6 / 44.5 | - |
| **F** N + concurrent read 2 + lent experts on the GPU | **36.4** | **159.7** | **11.2 / 0.5 / 41.5** | yes, every phase |

## Settings

| Switch | Default | Effect |
|---|---|---|
| packaged tuned rows | on | the gate/up GEMM on hipBLASLt's matrix-core kernels; the exactness reference |
| `STRATA_PF_GU_LEGACY=1` | off | the old gate/up kernel and the old bits |
| `STRATA_BATCH_CONCURRENT_READ=2` | off | read a prompt on its own thread beside the decoding slots, keeping the slot loan |
| `STRATA_BATCH_LENT_GPU=1` | off | lent experts computed on the GPU: decode beside a read stays exact |
| `STRATA_PF_DQ_BATCH=G` | off | batch the FP16 conversions (exact, slower) |

We recommend `STRATA_BATCH_CONCURRENT_READ=2 STRATA_BATCH_LENT_GPU=1` for agent serving on one card; both are opt-in because
they trade the decoding slots' rate during a read for the new prompt's TTFT.

## Open items

- Attention and GDN are now the largest share of a 5.4 s chunk; profile again on the new default.
- The down GEMM above M 32 (solution 1146, not bit-identical, ~0.1 s per chunk).
- A fused Q4_K dequant-in-GEMM to remove the FP16 round trip (~8 %, new kernel, new bits).
- A rare solo near-tie in long streams (a 2,500-token stream alone flipped once in 7 launches): which kernel is not run-to-run
  deterministic is not known.
- Tensor parallel has no batched prompt path and no prompt cache: every prompt goes through the 8-token decode windows.
- The MI50s idle during a read; using them would run gfx906 kernels (different bits). On the shared pool, where the MI50s
  already hold almost every expert a read needs, that is the planned next step (not built; [NX-25](cards/NX-25.md)): each read
  there streams ~64 GiB of experts to the XTX, about 2.2 s whatever the read's length.
