# Comparative timing: tp2 vs one card (2026-11-21, data extract)

Sanitised extract of the study behind [ANATOMY.md](../ANATOMY.md) section 4 and the per-agent rows of
[AGENT-TURNS.md](../AGENT-TURNS.md). Measurement only: the engine was not changed. The raw per-cell logs, probe series and
kernel-trace CSVs are kept privately.

Engine: the cycle-6 engine, calibrated Q4_K, temperature 0, presence_penalty 0, 8 short prompts (256-384 tokens out) unless
stated. One measurement at a time; every MI50 held at its high DPM state for each server's life. Peak temperatures: XTX 69 C,
MI50 68 C (junction). No resets.

## Arms

| Arm | Devices | PARALLEL | Experts on GPUs (XTX + MI50) | CPU path |
|---|---|---|---|---|
| T | tp2 on XTX A + B, 4 MI50 | 8 (the 16-user level is 8 slots + a queue) | 5,468 + 5,030 + 3,520 x 2 + 3,519 x 2 | 0 % |
| S | XTX A + MI50 #1, #2, `STRATA_NO_MTP=1` | 8 | 4,248 + 5,586 x 2 = 15,420 | 37 % |
| S1 | XTX A alone | 8 | 3,783 | 85 % |
| S+MTP | S with the drafter and `STRATA_BATCH_MTP=1` | 8 | 3,735 + 5,586 x 2 = 14,907 | 39 % |
| R | S+MTP with the prompt cache on | 4 | 4,859 + 5,586 x 2 = 16,031 | 35 % |
| G | two S servers side by side, 10 CPU-pool workers each | 4 + 4 | as the fair round | ~34 % each |

Timing builds (`STRATA_TP_TIMING=2` on T; `STRATA_VERIFY_PROFILE=1` + `STRATA_DECODE_TIMING=1` on one card) run 1.10-1.21x
slower than the untimed engine; the untimed window is given beside every table. Kernel traces (rocprofv3, paired with the ROCm
7.2.1 HSA runtime; with the engine's bundled runtime the tracer crashed the engine at the first decode window) slow the engine
6-16x, so traces are used only for per-kernel durations and counts.

## Per-agent tok/s and TTFT (untimed)

| Users | T per agent | T agg | T TTFT p95 | S per agent | S agg | S+MTP per agent | S+MTP agg | S1 per agent | S1 agg |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 50.5 | 50.5 | 0.39 s | 50.5 | 50.5 | 76.3 | 76.3 | 39.7 | 39.7 |
| 2 | 40.6 * | 74.1 | 0.83 s | 37.9 | 69.8 | 49.3 | 88.4 | 24.9 | 44.7 |
| 4 | 31.6 | 109.8 | 2.2 s | 24.6 | 90.5 | 26.7 | 93.8 | 14.7 | 51.8 |
| 8 | 18.8 | 133.0 | 4.0 s | 13.5 | 100.4 | 13.2 | 94.8 | - | - |
| 16 (8 slots + queue) | 18.0 | 132.6 | 24.8 s | - | - | - | - | - | - |

\* from the fair round of 20 November (same engine and settings).

## Where the time goes, ms per decode window

T = card 0, the critical path. One-card arms regrouped into T's classes.

| Class | T 1u | S 1u | S1 1u | T 4u | S 4u | S1 4u | S+MTP 4u | T 8u | S 8u | S+MTP 8u |
|---|---|---|---|---|---|---|---|---|---|---|
| dense GDN (hc read, qkv/z/ab, conv) | 3.71 | 4.15 | 4.18 | 4.77 | 5.53 | 5.51 | 9.27 | 7.75 | 10.19 | 9.75 |
| GDN recurrence | 0.30 | 0.36 | 0.37 | 1.12 | 1.17 | 1.16 | 1.24 | 2.26 | 2.21 | 1.32 |
| attention, 12 layers (incl. hc read) | 2.29 | 2.47 | 2.47 | 4.22 | 4.61 | 4.59 | 7.32 | 7.56 | 8.42 | 7.74 |
| out-proj | 0.83 | 1.27 | 1.27 | 1.07 | 1.40 | 1.40 | 1.80 | 1.30 | 1.87 | 1.86 |
| router + FFN hc read (+ plan, doorbell) | 3.27 | 3.05 | 3.09 | 3.95 | 3.82 | 3.82 | 5.68 | 5.83 | 5.92 | 5.94 |
| shared expert (+ q8_1 of the MoE input) | 1.57 | 1.50 | 1.52 | 1.73 | 1.64 | 1.64 | 2.43 | 2.61 | 2.53 | 2.51 |
| experts in own VRAM | 1.26 | 1.23 | 1.50 | 3.29 | 3.06 | 2.99 | 4.51 | 6.02 | 5.53 | 4.93 |
| waits A + B (host plan, PCIe flag) + empty PCIe group | 0.53 | 2.51 | 2.77 | 0.53 | 2.77 | 3.00 | 2.82 | 0.52 | 2.88 | 2.80 |
| wait M (helper / CPU rows) | 2.16 | 1.60 | 4.95 | 2.36 | 6.96 | 31.17 | 13.97 | 2.99 | 14.20 | 13.43 |
| copy + combine | 0.93 | 0.79 | 0.79 | 1.39 | 1.33 | 1.33 | 2.07 | 1.97 | 2.09 | 2.14 |
| all-reduce put + sum (96 calls) | 0.84 | - | - | 1.20 | - | - | - | 2.36 | - | - |
| all-reduce peer wait | 1.57 | - | - | 2.30 | - | - | - | 2.95 | - | - |
| head + gaps | 0.84 | 0.84 | 0.84 | 1.18 | 1.19 | 1.19 | 2.24 | 2.36 | 2.38 | 2.36 |
| **GPU window (timing build)** | **20.1** | **19.8** | **23.7** | **29.1** | **33.5** | **57.8** | **53.3** | **46.2** | **58.2** | **54.8** |
| host CPU-expert + helper pass | 3.5 | 5.4 | 9.5 | 6.8 | 14.2 | 38.7 | 23.6 | 10.8 | 25.0 | 23.5 |
| host commit graph + emit | 0.0 | 0.3 | 0.1 | 0.0 | 2.8 | 2.7 | 6.3 | 0.0 | 5.9 | 6.8 |
| **untimed wall per window** | **18.4** | **18.5** | **23.3** | **28.1** | **35.8** | **60.6** | **60.0** | **45.8** | **64.6** | **61.2** |

## Idle and spin per device

| Device / thread | T 1u | T 4u | T 8u | S 1u | S 4u | S 8u |
|---|---|---|---|---|---|---|
| XTX card 0 waiting (ms/window) | 4.3 | 5.2 | 6.5 | 4.1 | 9.7 | 17.1 |
| XTX card 1 idle at the MoE all-reduce (ms/window) | 3.0 | 3.8 | 5.1 | - | - | - |
| MI50 busy, mean | 10-14 % | 14-25 % | 12-20 % | 18-19 % | 21-23 % | 18-20 % |
| CPU threads used by the server | 6.0 | 5.9 | 5.8 | 41.9 | 41.5 | 41.1 |
| DRAM traffic (actual GB/s, mean) | 2.3 | 4.3 | 4.8 | 18.9 | 35.1 | 35.0 |

Helper lane on T (`STRATA_HELPER_TIMING=1`, graph replay off), per helper per window at 4 users: GPU-in 0.70 ms, kernels 2.62,
out 0.68, round trip 111 us a launch (3.9 experts); at 8 users 1.03 / 3.95 / 0.88 ms, 156 us (6.9 experts). MI50 #3 has the
same kernel time but 2.4-2.9x the transfer time (round trip 151 / 226 us), so its host wait is 4.8 / 8.0 ms against 2.9-5.1
for the others.

## Kernels ranked by time (XTX card 0, trace durations; B = rows)

| Kernel | bytes a call | T 1u us (GB/s) | T 4u | T 8u | S 1u | S 4u | S 8u |
|---|---|---|---|---|---|---|---|
| lm head, Q6_K 248,320 x 2560 | 521 MB | 584 (892) | 859 (607) | 1,925 (271) | 585 (891) | 852 (612) | 1,928 (270) |
| GDN qkv, Q6_K 5120 (tp half) / 10240 x 2560 | 10.8 / 21.5 MB | 15.2 (710) | 20.7 (518) | 45.3 (237) | 28.0 (769) | 39.0 (551) | 87.8 (245) |
| GDN gate, Q6_K 3072 / 6144 x 2560 | 6.4 / 12.9 MB | 20.1 (321) | 20.7 (311) | 28.4 (227) | 19.1 (675) | 23.8 (543) | 55.9 (231) |
| attention q, Q6_K 6144 / 12288 x 2560 | 12.9 / 25.8 MB | 18.2 (710) | 24.1 (535) | 54.1 (238) | 32.3 (798) | 44.7 (577) | 101 (254) |
| shared gate / up, Q6_K 640 x 2560 | 1.34 MB | 3.0 (445) | 4.2 (323) | 9.8 (137) | 3.0 (445) | 4.1 (330) | 10.3 (130) |
| hc read down, BF16 10240 x 320 | 6.55 MB | 11.7 (558) | 14.4 (457) | 12.4 (530) | 11.7 | 14.3 | 12.5 |
| hc read up, BF16 320 x 10240 | 6.55 MB | 10.5 (624) | 15.1 (433) | 30.4 (215) | 10.5 | 15.4 | 30.9 |
| router, BF16 512 x 2560 | 2.6 MB | 5.0 (528) | 6.1 (427) | 14.7 (178) | 5.0 | 6.1 | 15.2 |
| experts, Q4_K gate/up | grouped | 6.6 | 15.8 | 23.9 | 6.4 | 15.1 | 22.9 |
| experts, Q5_0 down | grouped | 5.2 | 13.0 | 19.5 | 5.2 | 12.6 | 19.0 |

Kernels per window on card 0: 2,329 / 2,770 / 3,190 (T) and ~2,160 / ~2,950 / ~3,650 (S) at 1 / 4 / 8 users.

## Realistic agent turns (arm R: S+MTP, prompt cache on, thinking on)

Each agent has its own ~46-58k-token context; turn 1 = cold prompt + 4,096 tokens out; turn 2 = turn 1 + ~1.5k new tokens +
16,384 out.

| Agents | turn 1 TTFT (cold) | turn 1 tok/s per agent | turn 2 TTFT (prefix reused) | turn 2 tok/s per agent | MTP accepted |
|---|---|---|---|---|---|
| 1 | 44.1 s | 75.1 | 4.5 s | 74.0 | ~93 % |
| 2 | 50.4 / 124.2 s | 31.2 / 40.7 | 4.0 / 4.0 s | 48.5 / 52.0 | |
| 4 | 46 / 120 / 205 / 283 s | 14.3-22.3 | 3.6-4.2 s | 25.0-29.3 | 79 % |

## DRAM ceiling and headroom (arm G)

A STREAM-style triad reaches **127 GB/s of actual DRAM traffic** (74 % of 8 x DDR4-2666's 170.6 GB/s). The data-fabric DRAM
counters on this Zen 2 kernel count 128 B per event (calibrated against the triad, 7 points within 1 %).

Two S servers at 4 + 4 users while a triad hog runs on otherwise idle CPUs:

| Hog threads | hog GB/s | total DRAM | per-agent tok/s | agg tok/s | vs no hog | CPU-expert pass (ms/window) |
|---|---|---|---|---|---|---|
| 0 | - | 54 | 21.7 | 159.2 | - | 20.3 |
| 8 | 68 | 107 | 18.4 | 135.6 | -15 % | 27.6 |
| 16 | 82 | 116 | 16.8 | 124.2 | -22 % | 32.2 |
| 24 | 82 | 115 | 16.3 | 121.9 | -23 % | 33.0 |
| 32 | 82 | 115 | 16.3 | 119.5 | -25 % | 34.0 |
| 0 (again) | - | 55 | 22.1 | 160.7 | +1 % | 20.0 |

## STRATA_B6_MMVQ_ROWS=1 and STRATA_TSUM

| Run | env on top of cycle 6 | solo sha (256 tok) | batched == solo | 8-user window (ms) | 8-user agg | per agent |
|---|---|---|---|---|---|---|
| T baseline | - | a7ae6623797f17fe | 8/8 | 45.77 | 133.0 | 18.8 |
| T B6 | B6=1 (TSUM=1 still on) | a7ae6623797f17fe | 8/8 | (no change) | 132.8 | 18.7 |
| T TSUM off | TSUM=0 | a7ae6623797f17fe | - | 46.60 | 130.6 | 18.8 |
| T TSUM off + B6 | TSUM=0, B6=1 | a7ae6623797f17fe | 8/8 | 44.18 | 136.4 | 19.2 |
| S baseline | - | a7ae6623797f17fe | 4/4 | 64.63 | 100.4 | 13.5 |
| S B6 | B6=1 (TSUM on) | a7ae6623797f17fe | 8/8 | (no change) | 100.6 | 13.8 |
| S TSUM off | TSUM=0 | a7ae6623797f17fe | - | 65.47 | 99.1 | 13.4 |
| S TSUM off + B6 | TSUM=0, B6=1 | a7ae6623797f17fe | 8/8 | 61.57 | 104.5 | 14.2 |

B6 was skipped whenever `STRATA_TSUM=1`; the decode A/B of the same day gave B6 a TSUM path
([2026-11-21-decode-ab.md](2026-11-21-decode-ab.md)).

Self-commit on one card (`STRATA_BATCH_SELF_COMMIT=1`, arm S): 4-user aggregate 94.9 vs 90.5, 8-user 106.6 vs 100.4; exact.

## Exactness

Every arm's solo text for the 256-token reference prompt is `a7ae6623797f17fe` except S+MTP (`da29bc7d5a0ba433`), and
batched == solo holds in every arm. S+MTP holds 513 fewer experts on the XTX than S (draft head and slot drafters), so a
different set runs on the CPU path, which changes rounding; MTP itself was not separated from that. (On the later engine with
the standard environment both arms give `a7ae6623797f17fe`: [2026-11-21-dp-pool.md](2026-11-21-dp-pool.md) section 6.)
