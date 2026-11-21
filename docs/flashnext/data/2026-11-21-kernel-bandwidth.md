# Kernel bandwidth per quant and device (2026-11-21, data extract)

Sanitised extract behind [ANATOMY.md](../ANATOMY.md) section 5 and [QUANT.md](../QUANT.md). One XTX (gfx1100) and one MI50
(gfx906, held at its high DPM state for the run). Kernels of the cycle-6 engine, called through `native_expert_grouped` /
`native_mmvq` on real expert bytes, cold (>= 384 MiB rotation), B = rows of one expert. Reference: upstream llama.cpp (build
b11870) on the same bytes. No prediction was written down before the run.

## Types in the two files (from the GGUF headers and the kernel traces)

- UD-IQ4_XS routed experts: down is **Q5_0 in 33 of 48 layers**, IQ4_NL in 12, Q2_0 in 2 (no IQ4_XS down); gate/up IQ3_XXS in
  28 layers, IQ4_XS in 17, IQ2_S in 3.
- Calibrated Q4_K: gate/up Q4_K, down Q5_0 in every layer (640-column rows cannot use a 256-block type; Q5_0 is the fallback).
- Q4_K gate/up runs `native_gu_kernel<12>` (the per-token kernel) on both GPUs; the decode-once split kernel runs only with
  `STRATA_KQ_SPLIT=1`, which no launcher sets.
- MI50 kernels: IQ3_XXS / IQ4_XS gate/up `native_gu_lds_kernel`, IQ4_NL down `native_down_lds_kernel<20>`, Q5_0 down
  `native_down_kernel<6>`, Q4_K `native_gu_kernel<12>`, plus a separate `quantize_q8_1_kernel` (~4.4 us).

## Achievable read

XTX 920 GB/s for a large read, **282 GB/s for a cold 2.83 MiB block** (one Q4_K expert); MI50 866 / **273** GB/s. The cold
expert-sized read is the ceiling for one expert; percentages below are of it.

## One routed expert: wall us p50 (GB/s, % of the cold expert-sized read); ours / llama.cpp as wall . GPU kernel time

| Expert types (file) | B | XTX ours | XTX / llama.cpp | MI50 ours | MI50 / llama.cpp |
|---|---|---|---|---|---|
| Q4_K + Q5_0 (Q4_K file) | 1 | 21.3 (140, 50 %) | 1.02 . 0.56 | 28.6 (104, 38 %) | 1.32 . 1.12 |
| | 2 | 24.0 (124, 44 %) | 1.12 . 0.65 | 36.2 (82, 30 %) | 1.27 . 1.12 |
| | 4 | 33.6 (88, 31 %) | 1.45 . 0.68 | 51.7 (57, 21 %) | 1.20 . 1.17 |
| | 8 | 45.5 (65, 23 %) | 0.70 . 0.36 | 87.8 (34, 12 %) | 0.71 . 0.62 |
| IQ3_XXS + Q5_0 (UD, 20 layers) | 1 | 24.0 (99, 35 %) | 1.15 . 0.51 | 42.9 (56, 20 %) | 1.73 . 1.45 |
| | 8 | 36.2 (66, 23 %) | 0.59 . 0.26 | 123.7 (19, 7 %) | 0.86 . 0.81 |
| IQ4_XS + Q5_0 (UD, 13 layers) | 1 | 25.7 (111, 40 %) FP | 1.26 . 0.71 | 40.6 (71, 26 %) | 1.92 . 1.58 |
| | 8 | 37.2 (77, 27 %) FP | 0.64 . 0.32 | 115.4 (25, 9 %) | 0.93 . 0.86 |
| IQ3_XXS + IQ4_NL (UD, 8 layers) | 1 | 24.9 (87, 31 %) FP | 1.21 . 0.57 | 46.1 (47, 17 %) | 1.93 . 1.62 |
| | 8 | 30.5 (71, 25 %) FP | 0.48 . 0.24 | 122.6 (18, 7 %) | 0.86 . 0.80 |
| IQ4_XS + IQ4_NL (UD, 4 layers) | 1 | 26.5 (101, 36 %) FP | 1.32 . 0.98 | 44.0 (61, 22 %) | 2.18 . 1.79 |
| | 8 | 31.4 (85, 30 %) FP | 0.52 . 0.29 | 114.4 (23, 9 %) | 0.92 . 0.85 |

FP = with `STRATA_XTX_FAST_PERM=1`, which only changes IQ4_XS (gate/up -7.5 to -9 %) and IQ4_NL (down -3 to -6 %). llama.cpp's
wall is one 128-expert graph / 128, so it pays no per-launch host cost; the GPU-time ratio is the like-for-like comparison. Four
experts in one launch (B 1): XTX 252-347 GB/s, MI50 150-227 GB/s. Error against a float reference: 1.1e-2 relative RMS with
IQ4_NL down, 2.0-4.3e-2 with Q5_0 down (noted, not investigated).

Per type, wall us B 1 -> B 8, XTX | MI50: Q4_K gate/up 15.6 -> 31.4 | 19.7 -> 53.1; IQ3_XXS gate/up 17.0 -> 20.5 | 33.8 -> 91.2;
IQ4_XS gate/up 20.3 (FP 18.8) -> 24.0 (21.8) | 31.5 -> 82.9; Q5_0 down 10.3 -> 18.8 | 14.6 -> 38.4; IQ4_NL down 13.2 (12.7) ->
15.7 (14.7) | 18.1 -> 37.9.

## Dense matvec, XTX, wall us p50 (ours / llama.cpp)

| Matrix | B 1 | B 4 | B 8 | B 8 with STRATA_B6_MMVQ_ROWS=1 |
|---|---|---|---|---|
| Q6_K 2560 -> 5120 (qkv, tp2 half) | 22.4 (1.16) | 30.6 (1.46) | 61.8 (2.11) | 44.4 (1.52) |
| Q6_K 2560 -> 3072 (gate, tp2 half) | 27.3 (1.09) | 27.5 (1.38) | 40.7 (1.96) | 30.4 (1.47) |
| Q6_K 2560 -> 6144 (attention q, half) | 25.4 (0.78) | 35.0 (1.37) | 71.1 (2.02) | 51.5 (1.46) |
| Q6_K 2560 -> 640 (shared gate / up) | 9.1 (1.28) | 11.0 (1.46) | 18.0 (2.04) | 14.1 (1.60) |
| Q6_K 2560 -> 248320 (lm head) | 715 (1.17; 79 % of 920 GB/s) | 1207 (1.85) | 2503 (2.17) | 1716 (1.49) |
| Q8_0 6144 -> 2560 (ssm_out, whole) | 27.0 (1.06) | 29.4 (1.09) | 35.3 (1.18) | not covered by B6 |
| Q8_0 640 -> 2560 (shared down) | 9.8 (1.01) | 11.8 (1.46) | 14.4 (1.60) | not covered |

Caveats: wall numbers are individually launched calls (the engine replays graphs). One MI50 only, burst numbers: in a later
sustained arm the 135 W-capped card ran 15-20 % slower and hit its 90 C trip once, and was not run again. ssm_out was measured
whole, not the tp2 half.
