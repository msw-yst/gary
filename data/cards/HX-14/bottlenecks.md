# dsv41 serving-config bottlenecks, tp4h4 (2026-11-10)

Config: `--gpus 0,1,2,7 --tp 4 --helper-device 3,4,5,6`, deploy env (B10E + DS4_BATCH_PREFILL_SHARE=0.5), expert cache on as deployed,
`--max-context 262144` (B8: 131072, the BATCH profile), flags off. Binary: arov10's build (best10 bits with its flags off; per-device DS4_PROF).
Traces: rocprofv3 `--kernel-trace --hip-runtime-trace`, pairs that differ only in N (prefill: 2048 vs 64 prompt ids); `helperov/trace_cls.py`
(arov10's trace_sum.py plus a window-class split) takes the difference. Raw: `bn-*/`, `BN.txt`.

**Read the traced columns as kernel time, not wall.** The tracer slows the run 2.7x (B1 45.7 ms a token traced against 17.1 untraced), and the
host-memory kernels slow the most (see note 3). Use the untraced lines for wall-time shares.

## Combined table (MEASURED)

| | (1) B1, DSpark off | (2) B1, DSpark on (conf 0.4, block 3) | (3) B8 batched | (4) prefill, 2K prompt |
|---|---|---|---|---|
| untraced speed | 58.35 tok/s, 17.1 ms a token (stage-1 s1, ctx 8192) | **66.91 tok/s**, 14.9 ms a token; 1.58 tokens a round; drafting 2.24 ms a round | 53.3 ms a step, 150 tok/s (separate diagnostic run) | **583.5 tok/s**, 1.71 ms a prompt token |
| kernels a unit, lead / member / MI50 | 1464 / 1358 / 47-54 a token | 1209 / 1124 / 66-84 a token | T8 class: 1894 / 1807 a step | 5.4-5.6 a prompt token per XTX; MI50 0 |
| lead XTX, traced (compute / spin / AR, ms) | 10.5 / 4.5 / 3.4 a token | per token: T1 windows 4.2 / 1.8 / 0.5; T2 verify 4.8 / 3.5 / 0.4; T3 0.4 / 0.7 / 0.0; drafter 1.2 / 0.0 / 0.3 | T8 class 40.6 / 75.7 (inflated) / 12.6 a step | busy 92-94 % of wall on all 4 XTX (no idle) |
| members, traced (compute / AR incl. waiting for the lead) | 10.0 / 7.2-9.3 | (same pattern: AR 3.2-4.3 in T1 and T2) | 29-37 / 87-104 | AR (ar_p2pk) the top kernel |
| helper share (untraced) | lead's "wait lane/helpers" 3.5 of 18.7 ms (19 %); members' FFN AR 4.6 ms = the same wait | 53.3 helper experts a token (vs ~33 at B1 off); lane issue + wait 4.14 ms a token | 278.5 helper picks a step; lane issue + wait **23.3 ms of the 53.3 ms step**; members' FFN AR 11.6 ms | none (prefill chunks use no helper) |
| per-stage DS4_PROF, lead (untraced, ms) | attention 7.9, routing + shared 2.6, gate/up 1.1, **wait helpers 3.5**, down + combine 1.3, attn AR 1.1, FFN AR 1.0 | T1 windows only (the verify windows are not profiled): as (1), wait helpers 4.2 | bt: attention 12.6, **down + combine 12.0 (the helper wait is inside it)**, routing 3.6, attn AR 2.2, FFN AR 2.2 | stage ms over the 2K prompt, CUDA0: **MoE AR 1103 (33 %)**, **attn AR 819 (24 %)**, experts 520 (16 %), streamed batches 285, router 110, compressor + indexer 104, sparse attn 95 |
| top kernels (traced, all agents) | ar_sum 37.1, iq mv 9.7, MI50 mx_combine 9.1, wait_seq 4.5, hc_mix_split_norm 4.2, mx_wait 2.6, q8_mv 2.6, sparse_attn 2.2, gate_up 2.2, route_publish 2.1 | ar_sum 28.3, sparse_attn 5.4, mv<13,4> 4.5, mx_wait 4.2, mv<13,1> 3.3, hc_mix 2.9, wait_seq 1.8 | ar_sum 513, MI50 mx_combine 125, mv 77, mx_wait 76, sparse_attn 69, hc_mix 38, wait_seq 37, MI50 helper_fetch 33 | ar_p2pk 3.19, tp_strided_rows 2.50, pfg bk64 0.35, iq set_kernel 0.35 + 0.19 + 0.11, sparse_attn_wmma 0.24 (ms a prompt token, 4 XTX summed) |
| bound by | the helper chain (lead waits, members wait in AR); then attention | the same, worse in verify windows (T2 spin 3.5 ms a token) | **the helper chain: 44 % of the step** | **all-reduce and the wait for the slowest member**: 57 % of GPU time is AR; member 7 (XTX #4) has the most compute in every stage and the least AR, so the other three wait for it. Not bandwidth- or compute-bound. |

## Notes

1. **The MI50 window is mostly not expert math.** B1, per helper window on the MI50 (traced): g906 gate_up 36 us, down 24 us (portable XXS down;
   g906 XS / S down similar), q8 quant 2 x 6 us, flag 8 us, against **helper_fetch 80-114 us and mx_combine 320-630 us**. Those two are the
   kernels that touch fine-grained host memory (fetch reads plan + x; combine writes the zc out rows). The tracer inflates them most: the
   untraced window span is ~155 us. Even so, the expert kernels are ~60 us of it, not ~110 us an expert. A1 (the grid fix) works on that 60 us.
   ESTIMATE: 20 % of 60 us x ~27 windows = **~0.3 ms a token**, half the research report's 0.6-0.8 ms.
2. **The grid is as the report says.** The g906 gate_up runs 96 x 6 blocks (one live group: 96 blocks, 3 row tiles each, on 60 CUs). The down
   runs 107 x 6. `DS4_G906_LIVE=1` (2 tiles a block, 144 blocks) and `DS4_G906_BPC` already move this without a code change, and the row
   reduction order is unchanged, so the result is bit-identical. The 2026-11-07 study tuned BPC only at LIVE 2.
3. **Tracer inflation:** B1 45.7 ms traced against 17.1 ms untraced; B8 151.5 ms against 53.3 ms a step. Spin and AR kernel times grow with the
   host's slowdown. The compute kernels (attention, mv, the experts) are the trustworthy traced figures.
4. **tp4h4 leaves the MI50s a third empty** (corrected 14:00Z; an earlier version blamed the deploy config). Each MI50 holds 2142-2183
   experts (19.4-19.7 GiB) on tp4h4 in BOTH configs (ctx 8192 `--no-expert-cache` stage-1 s1, and deploy). The 3460 (31.3 GiB) figure is
   tp3h3's (3 helpers, more non-primary experts). On tp4h4 the CPU lane is ~0, so the helpers hold every non-primary expert and
   **~12 GiB a card, ~48 GiB in all, is free.** That changes the capacity side of the format question (A4): the hottest ~2,500 helper
   experts would fit as an exact 4-bit copy, though not all of them. Not acted on: the operator dropped the format bench.
5. **DS4_HELPER_DD (option 1) works with the expert cache** (corrected): the cache swaps in only experts from cpu_hits, which never holds a
   helper's expert, so the helpers' device tables stay right. The refusal was removed (4c4f2c3a). The gates still run `--no-expert-cache`, to match
   the REF arms.
6. **DSpark:** verify windows (T2/T3) carry more helper wait per token than T1 windows (3.5 + 0.7 ms against 1.8). The drafter is ~1.5 ms a token
   on the lead (~10 %). The window classes come from each layer's MoE-combine grid and the length of the run of layers (trunk = 40 layers,
   drafter = 3 blocks).
