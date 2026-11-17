# DeepSeek V4.1 Flash on RX 7900 XTX + MI50: experiment cards

This folder records what we changed in Strata's DeepSeek V4.1 Flash engine to run it on a mix of AMD consumer and
datacenter cards, and what each change did when measured. The long read is [WRITEUP.md](WRITEUP.md); this file is the
index, and [FEATURES.md](FEATURES.md) maps every switch to the stage of a token it acts on. The latest results are in
[UPDATE-2026-11-14.md](UPDATE-2026-11-14.md), [UPDATE-2026-11-15.md](UPDATE-2026-11-15.md), [UPDATE-2026-11-16.md](UPDATE-2026-11-16.md) (release 3: build 17, the expert-kernel benchmarks, validation against upstream Strata and llama.cpp) and [UPDATE-2026-11-17.md](UPDATE-2026-11-17.md) (release 4). Each card covers one change or event and follows the same template: what and why, the prediction (and whether it
was written down before the run), correctness, timing, results, verdict, how to reproduce, and caveats.

Built by YST. Contact: https://yst.dev

## How to read a card

- **Labels.** Every number is labelled MEASURED (read from a raw run file or a script-written summary), REPORTED
  (from a commit message or write-up made at the time, not re-checked against raw output) or INFERRED (our reading).
- **Correctness first.** Exact changes must produce byte-identical output to the flags-off build (decode output, batch
  test, logits or teacher-forced score). Approximate changes get a paired teacher-forced comparison instead. "Not run"
  means not run.
- **Code.** All changes ship as release commits on top of upstream Strata `v0.2.9` (release 4, build 17 plus opt-in changes, is the latest). Cards cite the release commit plus the
  flags (environment variables or command-line options) that turn each change on. Changes that were tried on side
  branches and not kept are marked "not in the fork's release commit".
- **Prompts.** Timing numbers come from a private prompt set that is not published. Ratios between arms of the same run
  compare like with like; absolute numbers should be read as indicative until the public re-runs below are done.
- **Data.** `data/cards/<ID>/` holds sanitised extracts of the run summaries, with `SHA256SUMS`. The originals are kept
  privately; the extracts were produced with a scrubbing tool that is kept private, which also checks every published file.
- **Scope.** All results are for this box, these models and these packs. They may not transfer to other boards, PCIe
  topologies, ROCm versions or quantisations.

## Hardware (public labels)

| label | card | link |
|---|---|---|
| XTX #1, XTX #3 | RX 7900 XTX 24 GB (XTX #3 re-pasted 12 Nov, [HW-10](cards/HW-10.md)) | PCIe Gen4 x16 |
| XTX #2 | RX 7900 XTX 24 GB (a different board from 13 Nov 14:37, [HW-09](cards/HW-09.md)) | PCIe Gen4 x16 |
| XTX #4 | RX 7900 XTX 24 GB (from 9 Nov; second firmware position from 13 Nov, [HW-09](cards/HW-09.md)) | PCIe Gen4 x8 |
| MI50 #1–4 | Instinct MI50 32 GB (gfx906), 135 W cap | PCIe Gen4 x4 each |
| MI50 #5 | Instinct MI50 32 GB (gfx906), bench card only (from 12 Nov; never a helper in a timed run) | PCIe Gen1 x8 on a riser |

Host: one EPYC Zen 2 16-core, DDR4-2666. ROCm 7.14 on Linux. The layout changed over the project (one XTX with host-RAM
experts → three XTX + three MI50 → four + four); each card's Timing section says which layout it ran on. Runs after
13 Nov 13:09 use the changed XTX set ([HW-09](cards/HW-09.md)); absolute numbers across that line are not directly
comparable, ratios within a run are.

## Builds

Builds 10–12 are described in [SV-07](cards/SV-07.md), [SV-06](cards/SV-06.md) and [SV-09](cards/SV-09.md). Since 11 Nov
(each gated for byte identity against its predecessor; all rebuild from git byte for byte, [ME-10](cards/ME-10.md)):

| build | adds | card |
|---|---|---|
| 12pf | the MI50 prompt helper port (`DS4_PF_HELPER`, not bit-identical) | [PF-11](cards/PF-11.md) |
| 13 | Engram ahead, many-workgroup top-k, an MI50 sign-table kernel (off when served) | [SV-10](cards/SV-10.md) |
| 13s | sampled drafting for concurrent users; served with the prompt helper on and memory-mapped experts | [SV-10](cards/SV-10.md), [SV-11](cards/SV-11.md) |
| 14 | the attention output's int8 tail in its finishing kernel | [FG-08](cards/FG-08.md) |
| 15 | MI50 prompt tiles; the batched top-k (identity against build 14 PASS, served benchmark done) | [SV-14](cards/SV-14.md) |
| 16 | the indexer dispatch; fusion items 1 and 3; a prompt-length cap for the MI50 prompt helper; the expert-cache fix, which also ends a double count of expert picks; four upstream correctness fixes (release 2: identity against build 15 PASS, 9 of 9) | [SV-16](cards/SV-16.md) |
| 17 | the batched-path fusion (`DS4_BT_FUSE`) and the cache fix that keeps promotion (`DS4_ECOWN`; replaces `DS4_EC_FIX`) (**the release build, release 3**: identity against build 16 PASS, served A/B done) | [SV-17](cards/SV-17.md), [HX-30](cards/HX-30.md) |
| 18 | build 17 + swap instead of evict (`DS4_ECSWAP`) (**staged, re-validation pending**; not in the published release) | [HX-32](cards/HX-32.md) |

**Build 17 is the build in this release (release 3).** Its recommended environment is build 16's with `DS4_EC_FIX` unset and
`DS4_ECOWN=1 DS4_BT_FUSE=1` added ([SV-17](cards/SV-17.md)); `DS4_ECOWN` refuses to load together with `DS4_EC_FIX=1`.
Build 16's recommended environment, still the one `docs/examples/dsv41-serve.sh` and `tools/dsv41/bench-perms.sh` set, is build 15's served one with
`DS4_ATTN_MERGE=3` (instead of `=2`) plus `DS4_XK_QA=1 DS4_IXS_DISPATCH=16384 DS4_PF_HELPER_MAXP=8192 DS4_EC_FIX=1`;
every new switch is off by default in the code, so set them ([SV-16](cards/SV-16.md)). Build 18 is measured but
**not in the published release**: its code is not in the public commits ([UPDATE-2026-11-15.md](UPDATE-2026-11-15.md)). Measurement-only builds (an untraced
phase timer, a tree-speculation counter, a double-count audit, an Engram trace) are not in this release; the cards that mention them say so.

## Expert-kernel benchmarks and validation (release 3)

- `tools/qbench/` times one routed expert on real expert bytes with our kernels, llama.cpp's and castagna-veloce's, on
  one card: on the MI50 our 2-bit kernels are the fastest of the three, our MXFP4 is 1.5× slower at batch 1, and
  castagna-veloce's fused Q2_K/Q3_K kernel is the fastest expert path measured ([HX-33](cards/HX-33.md)); the MI50
  helper's round trip, 104.6 µs, splits by stage ([HX-34](cards/HX-34.md)); on the XTX llama.cpp is faster at batch 1
  ([FG-19](cards/FG-19.md)). The gfx906 expert kernels are unchanged since release 2.
- `validation/` compares engines at identical weights: build 17 matches upstream Strata (ENGINE PASS, [QA-10](cards/QA-10.md)),
  and the 2.25 bpw weights, not the engine, are the distance to the hosted model ([QA-11](cards/QA-11.md)).

## Release 4 (17 Nov)

- **Code:** build 17 plus `DS4_BF16MV` (a bit-identical BF16 matvec, [FG-20](cards/FG-20.md)), off by default. The release build and its recommended environment are
  still build 17's ([SV-17](cards/SV-17.md)). `DS4_BF16MV=r2u4n` took 2 ms off a served verify step and added 9.8 % at
  4 users on a native MXFP4 pack; the 2.25-bit pack's dense matrices are not BF16, so it is not expected to help there.
- **MI50 P2P:** peer reads of MI50 memory return stale data when the MI50's own SDMA engine wrote it, until the MI50's
  HDP is flushed ([HW-12](cards/HW-12.md)).
- **Results without code in this release:** mixed precision, native experts on top of the 2.25-bit pack
  ([QA-12](cards/QA-12.md)); drafting at four users is slower in a greedy test ([DR-10](cards/DR-10.md));
  castagna-veloce against llama.cpp on four MI50s ([EB-09](cards/EB-09.md)).
- Details: [UPDATE-2026-11-17.md](UPDATE-2026-11-17.md).

## Benchmarking GPU permutations

`tools/dsv41/bench-perms.sh` measures one choice of XTX and MI50 cards with the release build and prints one row of
plain numbers. How to run it, how to read a row, and a 1 / 2 / 3 / 4 XTX × 0–4 MI50 grid from the test machine are in
[benchmarking-gpu-permutations.md](benchmarking-gpu-permutations.md).

## Fixed in this release: expert picks summed twice

- **What it was** ([HX-28](cards/HX-28.md)): with the expert cache on (the default) and device-direct MI50 windows
  (`DS4_HELPER_DD=1`), every prompt read let the cache move up to 8 experts that an MI50 holds into XTX memory, while the
  MI50s' device-direct tables, fixed at load, still listed them. From then on each pick of such an expert was computed on
  the XTX and on the MI50, and both results were added. In a served run, after 53 prompts, 4.93 % of all device-direct
  expert picks had been summed twice (145,399 of 2.95 million), about half of them with an MI50 input left over from an
  earlier token, and the count grew with every prompt (MEASURED).
- **Who was affected:** servers with MI50 helpers, the expert cache on and `DS4_HELPER_DD=1` (the served configuration
  up to and including build 15), once they had read a prompt. Its output was not the model's true output, and it
  depended on how long the server had been up: the same greedy prompts got different answers after 53 prompts (8 of 8 in
  the audit run). The answers stayed fluent, which is why it went unnoticed. The expert-cache leak of
  [HX-26](cards/HX-26.md) (first token +13.9 % over 50 prompts) is the same promotion seen from the prompt side.
- **The fix:** `DS4_EC_FIX=1`, in build 16's recommended environment: the cache never moves an MI50-held expert into
  XTX memory. With it, no pick is summed twice, the leak is gone, and the answers no longer depend on uptime (MEASURED).
  It is off by default in the code; set it whenever the expert cache and `DS4_HELPER_DD` are both on
  (`docs/examples/dsv41-serve.sh` sets it).
- **What the fix costs:** the leaky cache's gain on 8K prompts. Served 8K first token 5.9 s on build 15, 6.7 s on build 16
  (MEASURED, [SV-16](cards/SV-16.md)). Keeping that gain without the bug needs the MI50 tables to follow the cache's
  moves; that is not built.

## Known issues in this release

- **8K prompts read more slowly than on build 15** (5.9 → 6.7 s served), the price of the fix above
  ([SV-16](cards/SV-16.md), [HX-26](cards/HX-26.md)).
- **The MI50 prompt helper is slower than no helper on long served prompts** ([PF-13](cards/PF-13.md)): faster at 2K,
  even at 8K, 19–25 % slower from 16K to 64K (MEASURED). Build 16's prompt-length cap (`DS4_PF_HELPER_MAXP=8192`, in
  the recommended environment) reads longer prompts without the helper and gets the helper-off times (served 32K 18.3 s,
  MEASURED); without the cap, use `DS4_PF_HELPER=0` for long-prompt workloads.
- **Two prompt-read hangs**, cause open ([SV-12](cards/SV-12.md)).

## Headline cards that need a public re-run

| card | what | estimate |
|---|---|---|
| [EB-03](cards/EB-03.md) | first run, 1 XTX ± MI50 | ~20 GPU-min |
| [EB-02](cards/EB-02.md) | llama.cpp control | ~25 GPU-min |
| [TP-05](cards/TP-05.md) | tp3 vs tp4 | ~60 GPU-min |
| [SV-07](cards/SV-07.md) | build 10 grid | ~60 GPU-min |
| [HX-10](cards/HX-10.md) | helper overlap A/B | ~45 GPU-min |
| [HX-17](cards/HX-17.md) | device-direct helper windows | ~50 GPU-min |
| [HX-18](cards/HX-18.md) | DD_SKIP | ~40 GPU-min |
| [CP-03](cards/CP-03.md) | MI50 clock pin | ~15 GPU-min |
| [SV-06](cards/SV-06.md) | build 11 vs build 10 | ~20 GPU-min |
| [FN-03](cards/FN-03.md) | Qwen3.8-Flash-Next arms | ~30 GPU-min |
| [QA-03](cards/QA-03.md) | build 10 quality on a public text | ~30 GPU-min |

About 395 GPU-min in all. Until these run, each of these cards carries the private-prompt-set caveat.

## Status counts

| status | cards |
|---|---|
| kept | 66 |
| dropped | 21 |
| finding | 27 |
| method | 13 |
| not run | 7 |
| shelved | 4 |
| desk | 3 |
| parked | 4 |
| open | 2 |
| superseded | 2 |
| hardware event | 2 |
| incident | 2 |
| fixed in this release | 2 |
| running | 1 |
| served, validation pending | 1 |
| explained | 2 |
| caveat | 1 |
| card pulled | 1 |
| done | 1 |
| corrected | 1 |
| pointer | 1 |
| **total** | **164** |

Counts as of 17 Nov (release 4 adds FG-20, HW-12, QA-12, DR-10, EB-09). Earlier: as of 15 Nov 09:00 UTC. Added 14–15 Nov evening and night (none of their code is in the published release):
HX-29–32, SV-17, FG-18, HW-11, ME-12. Cards added since 11 Nov: EB-08, TP-08–09, HX-24–28, DR-07–09, FG-08–17, CP-05,
PF-12–13, SV-10–16, HW-09–10, ME-10–11; dated addenda on PF-11, QA-04, QA-08, QA-09, DR-04, DR-06, SV-09, HW-05, HW-06,
HW-08. Updated 13–14 Nov with results: SV-11, SV-13, SV-14, FG-12, DR-08, HX-26. Updated 14 Nov morning: HW-06
(addendum replaced), FG-15 and FG-16 (corrected by FG-17), HX-26, PF-13, FG-07, FG-08, FG-12, DR-09, TP-09, SV-14, SV-15
(results of later builds and runs). Updated 14 Nov afternoon for build 16, the release build: new SV-16 and HX-28;
FG-07, FG-08, FG-12, FG-16, PF-13, HX-26 and SV-14 (their changes are now in this release); DR-01 (the drafter is
DeepSeek's own DSpark).

## Card index

"Held" = whether the prediction held. "Correctness" = what the identity or quality evidence shows.

| card | title | status | prediction held? | correctness | headline |
|---|---|---|---|---|---|
| [EB-01](cards/EB-01.md) | gfx906 (MI50) expert helper inside the RDNA3 build | kept | n/a (gates only) | teacher-forced, 511 positions: pass | x16: +5.8 … +77.8 % decode by cache size (MEASURED) |
| [EB-02](cards/EB-02.md) | llama.cpp control: DeepSeek V4-Flash-0731 Q8 on one XTX, then XTX + MI50 layer split | dropped (control) | n/a | not run | tg256 12.35 → 11.49 tok/s (−7 %) (MEASURED) |
| [EB-03](cards/EB-03.md) | First DeepSeek V4.1 Flash run: 1 XTX + host RAM, then + 1 MI50 helper | kept (baseline) | n/a | see QA-01 (pass) | decode 22.67 → 26.80 tok/s; prefill 212 tok/s at 2K (MEASURED) |
| [EB-04](cards/EB-04.md) | Multi-GPU "invalid kernel file" with ROCm 7.14: all-arch build | kept | n/a | diagnostic script | 3 XTX 29.1, + MI50 31.1 tok/s (REPORTED) |
| [EB-05](cards/EB-05.md) | Decode stalls: prefault, host memory as GTT, no heap growth, ksmtuned | kept | yes (plan said stalls should reach 0) | stall counts | 0 stalls in 9 processes with ksmtuned stopped (REPORTED) |
| [EB-06](cards/EB-06.md) | Warm-up at load (helper and decode) | kept | n/a | not run | token-1 tail 86–89 → 38 ms (REPORTED) |
| [EB-07](cards/EB-07.md) | Four correctness fixes ported from a community MIT project | kept | yes | identity: IDENTICAL | reg / logits IDENTICAL (MEASURED) |
| [EB-08](cards/EB-08.md) | Expert files memory-mapped instead of an 83 GB pinned copy (`--mmap-experts on`) | kept (served from build 13s) | prompt yes; load time no | IDENTICAL [M] | prompt within noise at 1K / 32K / 110K; 83 GB less pinned RAM (MEASURED) |
| [EB-09](cards/EB-09.md) | castagna-veloce against mainline llama.cpp on four MI50s, same GGUF | finding | no (decode 4.6 tok/s against 25–55 predicted) | n/a (speed only) | decode CPU-bound, 4.6 vs 4.4 tok/s; prefill pp512 1.40×, pp2048 1.85× castagna (MEASURED) |
| [TP-01](cards/TP-01.md) | Tensor parallel over 3 XTX (`--tp 3`) | kept | n/a | teacher-forced tp1 vs tp3 [R] | decode 30.4–30.9 tok/s (REPORTED) |
| [TP-02](cards/TP-02.md) | tp3 prompt path: three contiguous DMAs per member | kept | n/a | not run | 2K prompt 384 s → 10 s (REPORTED) |
| [TP-03](cards/TP-03.md) | tp3 + one MI50 helper | kept | yes (32.85 vs a 31.9 bar) | teacher-forced: pass [R] | 32.85 tok/s; prompt 203 tok/s (REPORTED) |
| [TP-04](cards/TP-04.md) | New layout: 3 XTX at x16 + 3 MI50, every expert on a GPU | kept | yes, exceeded (48.66 vs ~41 projected) | not run (no identity vs old layout) | 48.66 tok/s [48.48–48.75] (MEASURED); prompt 294.7 tok/s (REPORTED) |
| [TP-05](cards/TP-05.md) | tp4: four XTX, uneven expert split; 2×2 grid tp3/tp4 × 3/4 helpers | kept | yes for decode; prefill got worse | reg IDENTICAL; q4 95.96 % top-1 [R] | decode set +9.9 % (tp4h4 vs tp3h3); 2K prompt +11 % slower (MEASURED) |
| [TP-06](cards/TP-06.md) | Expert parallel over the tp group (`DS4_EP=1`) | not run | — | not run | compile only |
| [TP-07](cards/TP-07.md) | Permutations and multi-worker serving on the early layouts | superseded | n/a | not run | 3 workers on 3 MI50: 92.7 tok/s total (REPORTED) |
| [TP-08](cards/TP-08.md) | Is the four-way split uneven? A dose probe, and a design for uneven cards | dropped (re-split not built; the run's verdict "no data") | probe yes; barrier and prompt doses no | IDENTICAL [M] | a perfect re-split ≤ 1.4 % decode, 0 % on 8K prompts (MEASURED) |
| [TP-09](cards/TP-09.md) | A layer split (pipeline) of the four XTX for more users | dropped (closed: the existing split lost for prompts and decode) | yes (slower everywhere) | n/a | prompts 1.5–5.2 × slower than tp4; decode 20.6 ms vs 14.9 (MEASURED) |
| [HX-01](cards/HX-01.md) | MI50 helper on a Gen4 x4 link vs x16 | kept (x4 is fine) | n/a | teacher-forced: pass [R] | no measurable decode loss on x4 (MEASURED) |
| [HX-02](cards/HX-02.md) | Zero-copy and one-shot switches for the helper | dropped (ZC_OUTPUT rejected) | n/a | not run | ZC_OUTPUT −8 %; others within noise (REPORTED) |
| [HX-03](cards/HX-03.md) | Shared helper pool, shared-memory mailbox, shared RAM store | shelved | n/a | not run (pool gates never run) | 3 engines on one MI50: ~132 tok/s combined (REPORTED) |
| [HX-04](cards/HX-04.md) | gfx906-specific expert kernels for the MI50 | kept (no decode gain) | n/a | parity 20/20 BITWISE [R] | no decode gain; SDMA off removed a slow mode (−1 ms/token) (REPORTED) |
| [HX-05](cards/HX-05.md) | Per-layer helper slot sizing | kept | n/a | parity 20/20 [R] | +17 % experts held (3,467 vs 2,964); solo decode unchanged (REPORTED) |
| [HX-06](cards/HX-06.md) | No-SDMA helper window (kernel fetch + zero-copy out + flag + graph) | kept | n/a | not run as identity [R] | 1 XTX + MI50 30.65 → 33.39 tok/s; 3 helpers 33.3 → 36.4 (REPORTED) |
| [HX-07](cards/HX-07.md) | Stripe placement of helper experts (default in build 8) | kept | n/a | regression IDENTICAL [R] | part of build 8's +22.3 % decode set (REPORTED) |
| [HX-08](cards/HX-08.md) | Lookahead prediction of helper picks | dropped (NO-GO) | no | n/a | recall 60–81 %; whole layers 35–65 % (REPORTED) |
| [HX-09](cards/HX-09.md) | Armed helper windows | kept (opt-in, default off) | partly | identical [R] | decode set +6.5 %, B1 +10.5 %; B8 bimodal (REPORTED) |
| [HX-10](cards/HX-10.md) | Helper round trip: overlap the down rows, packed staging (A, B) | kept | no (smaller than estimated) | IDENTICAL, all pairs [M] | tp4h4 B1 −2.0 % (AB) (MEASURED) |
| [HX-11](cards/HX-11.md) | Defer helper experts by one layer (C) and partial deferral | shelved | timing yes; quality no | full C: FAIL (ppl +11.7 %); MAXW ≤ 0.2: pass [M] | B8 −29 % but not exact; quality gate failed (MEASURED) |
| [HX-12](cards/HX-12.md) | Partial deferral with two windows in flight (2W) | parked | — | "passes, 94.9 % agreement" [R] | −0.7 ms (REPORTED) |
| [HX-13](cards/HX-13.md) | Diagnostic: why the tp4 helper wait is longer (slow card or TP width?) | finding | n/a | n/a | TP width, not XTX #4 (MEASURED) |
| [HX-14](cards/HX-14.md) | Stage-1 breakdown of the single-stream helper chain | finding | n/a | n/a | one helper request ≈ 179 µs, ~60 µs outside the MI50 (MEASURED) |
| [HX-15](cards/HX-15.md) | Faster gfx906 expert kernels (option 2) | shelved | — | not run | ceiling 4–25 % of the window (INFERRED) |
| [HX-16](cards/HX-16.md) | Device-direct helper windows, v1 | superseded | — | batch test DIFFER → stopped [M] | stopped at the identity gate (MEASURED) |
| [HX-17](cards/HX-17.md) | Device-direct helper windows v2 | kept (opt-in; part of build 11) | no (about a third of the predicted gain) | IDENTICAL (4-XTX score not run) [M] | tp4h4 B1 −3.9 %, B8 −4.2 % (MEASURED) |
| [HX-18](cards/HX-18.md) | DD_SKIP: the lead's join waits only for MI50s that have a pick | kept (part of build 11) | partly (B8 yes, B1 no) | byte-identical [M] | tp4h4 B8 −5.6 % vs DD (MEASURED) |
| [HX-19](cards/HX-19.md) | gfx906 kernel grid knobs (LIVE / BPC) | dropped | n/a | BITWISE [M] | nothing beats the default (MEASURED) |
| [HX-20](cards/HX-20.md) | ddexp: the next single-stream bottleneck experiments (lane-wait fix, oracles, timestamps) | finding (fix dropped; empty-AR oracle explained) | yes (non-oracle flags) | every non-oracle flag IDENTICAL [M] | ~2.7 µs per dependent kernel on the lead; members wait ~1.0 ms/token for the lead; oracles −0.71 / −0.69 / +0.47 ms; the +0.47 is 3× helper traffic from corrupted routing (MEASURED) |
| [HX-21](cards/HX-21.md) | Lead-path A/B: taking the lead XTX off the MoE all-reduce's critical path | kept (in part) | no (−0.27 vs −0.55 … −1.00); FLAGFUSE yes | IDENTICAL, all arms [M] | ALL −0.27 ms; FLAGFUSE −0.20; ROUTE / AHEAD=0 ~0 (MEASURED) |
| [HX-22](cards/HX-22.md) | Helper-window levers: one graph launch a window, int8 in the head, T>1 publish | kept (part of build 12) | partly (−0.08 vs −0.10 … −0.25) | IDENTICAL [M] | −0.08 ms over F + LP (MEASURED) |
| [HX-23](cards/HX-23.md) | Rewritten MI50 expert kernels (3 waves a SIMD): exact but slower | dropped (MK off) | no (+0.33 vs −0.05 … −0.15) | IDENTICAL [M] | B1 +0.33 ms (MEASURED); cause: grid-limited, not register-limited; lookups serialised; 6× grid (INFERRED) |
| [HX-24](cards/HX-24.md) | MI50 kernel switches: per-group kernel choice, an LDS sign table, DPP reductions | kept (sign table in the build-13 code, off when served); the others dropped | B1 yes; production no | IDENTICAL [M] | sign table B1 −0.113 ms; production +0.005 (MEASURED) |
| [HX-25](cards/HX-25.md) | MI50 multi-row levers: a fused down kernel and multi-row kernels | dropped | no | bench bit-identical (one arm within tolerance) [M] | fused down +10.2 µs a one-row window; multi-row −0.65 µs (MEASURED) |
| [HX-26](cards/HX-26.md) | Expert prefetch: not viable; a prompt-time cache leak found instead, then confirmed and fixed | fixed in this release (`DS4_EC_FIX=1`, build 16) | prefetch n/a; leak yes, larger than predicted (+13.9 % vs +3 %); fix: leak yes, 8K gain lost | decode unaffected (−0.47 %); fix off: answers change after 53 prompts (6 of 8), fix on 8 of 8 identical [M] | +8 host-streamed experts a prompt; 2K TTFT +13.9 % over 50 prompts; with the fix 0 and −0.9 % (MEASURED) |
| [HX-27](cards/HX-27.md) | More work for the MI50s: they are not idle | desk | — | not run | drafter experts on the MI50s: +0.2 ms a token predicted (INFERRED) |
| [HX-28](cards/HX-28.md) | Expert picks summed twice: the expert cache and device-direct MI50 windows | fixed in this release (`DS4_EC_FIX=1`, build 16) | yes (4.93 % vs ~2 %, inside 0.2–8 %; fix 0) | a correctness bug: answers changed with uptime, 8 of 8; with the fix 0 double counts [M] | 4.93 % of device-direct picks summed twice after 53 prompts; 0 with the fix (MEASURED) |
| [HX-29](cards/HX-29.md) | Why one user got slower on build 16: the expert-cache fix stops the cache's promotion | explained | yes (3 of 3) | n/a (switch on vs off; output differs, as expected) | without the fix one user −4.6 % (2K) / −8.4 % (8K) a token, 8K first token −12.4 %; 0 vs 20 cache moves (MEASURED) |
| [HX-30](cards/HX-30.md) | A cache fix that keeps promotion and counts each pick once (`DS4_ECOWN`) | kept (build 17, release 3) | 4 of 5 (2K decode out) | identical to build 16; 0 of 2.92 M picks counted twice; batch == solo [M] | 8K first token −12.5 %, 8K decode −2.0 %, 32K +2.7 % (MEASURED) |
| [HX-31](cards/HX-31.md) | Settle the cache after the first token (`DS4_SETTLE_AFTER`) | parked | 7 of 9 (32K out, worse; 8K decode out, a token-path effect) | same end placements; 0 counted twice [M] | 32K first token −0.13 s only (MEASURED) |
| [HX-32](cards/HX-32.md) | Swap instead of evict (`DS4_ECSWAP`): a long-running server keeps its prompt speed | kept (build 18, staged, re-validation pending; not in the published release) | 8 of 10 (life cell out, better; 2K decode out, a token-path effect) | 0 twice / 0 dropped / 0 mismatches over 1,592 handovers [M] | 2K first token after 50 prompts −22.7 % (2.67 → 2.06 s) (MEASURED) |
| [HX-33](cards/HX-33.md) | MI50 expert kernels against llama.cpp and castagna-veloce (QBENCH) | finding | 2 of 5 (MXFP4 2× IQ2_XS out; castagna ≥ 1.5× GB/s out; rel_rms ≤ 1e-2 out, narrowly) | ours vs CPU reference rel_max 1.0–1.3 %; references non-finite only [M] | batch 1 GPU µs, ours / castagna / llama.cpp: IQ2_XXS 51.6 / 70.2 / 79.6, MXFP4 80.8 / 54.2 / 58.5; castagna fused Q2_K/Q3_K 29.1 (MEASURED) |
| [HX-34](cards/HX-34.md) | The MI50 helper's request round trip, stage by stage | finding | yes (104.6 µs, inside 90–130) | every data-path variant BITWISE identical [M] | 104.6 µs p50, kernels ~63; graph + kernel fetch + zero-copy + flag 68.9 µs (−34 %) (MEASURED) |
| [DR-01](cards/DR-01.md) | DSpark drafter settings: confidence 0.3, block 3 | kept | n/a | no gate (divergence explained as numerics) [R] | +23 % decode (REPORTED) |
| [DR-02](cards/DR-02.md) | Per-card drafter memory budget | kept | n/a | 1 repeated-token decode in 9 [R] | one text prompt 54.6 → 63.1 tok/s (REPORTED) |
| [DR-03](cards/DR-03.md) | Drafter blocks tensor-parallel over the lead's group | kept | n/a | not run | +30 % trunk experts per XTX; text 54.2 → 63.6 tok/s (REPORTED) |
| [DR-04](cards/DR-04.md) | Drafting inside batched decode steps, with an adaptive rule | kept | yes (verify row ≤ 3.5 ms) | exact (batched = solo) [R] | B1 77.2 / B2 87.6 / B4 99.1 tok/s (REPORTED) |
| [DR-05](cards/DR-05.md) | Smaller drafter; trimmed vocabulary | not run (desk only) | — | not run | +1–5 % projected (INFERRED) |
| [DR-06](cards/DR-06.md) | The drafter's cost model on live traffic, and acceptance by text type | finding | n/a (fitted after) | n/a | model 11.43 vs 11.47 ms a token; verify row 3.11 ms; tool calls 0.69 / thinking 0.55 acceptance (MEASURED / INFERRED); addendum: verify row 2.66–2.70 ms on builds 12–13 (MEASURED) |
| [DR-07](cards/DR-07.md) | Sampled drafting inside batched decode, for concurrent users | kept (build 13s; served) | yes | distribution test PASS; batched greedy = solo with the prompt helper off [M] | 3 users 129.6 → 157.8 tok/s total (× 1.22) (MEASURED) |
| [DR-08](cards/DR-08.md) | Drafter policy revisited: sampled drafts, the adaptive rule, a depth × confidence sweep, a confirm run | dropped ((5, 0.4) not confirmed; policy stays (3, 0.3)); SAMPLE and ADAPT dropped | no (sweep); confirm: every cell inside ±1.5 % and the noise | exact in distribution | confirm: (5, 0.4) −1.58 % against 3–4 % noise, not separable (MEASURED); SAMPLE +13.0 % (REPORTED) |
| [DR-09](cards/DR-09.md) | Tree speculation: a desk model and a counting build | dropped (closed by the counting run) | n/a (counting run: close) | counting build IDENTICAL to build 15 [M] | h2 0.337 sampled, under the 0.35 close line (MEASURED) |
| [DR-10](cards/DR-10.md) | Drafting with four active users (`DS4_BATCH_DSPARK_MAX=4`) | dropped (MAX stays 3; one served A/B pending) | no (× 1.05–1.45 predicted) | greedy batched = solo everywhere [M] | 4 users, 4 over 3: build 17 × 0.806, native × 0.932, x25c × 0.787 (MEASURED) |
| [FG-01](cards/FG-01.md) | How much of a decode token is launch overhead? | finding | n/a | n/a | expected 0–3 % (REPORTED) |
| [FG-02](cards/FG-02.md) | Decode kernel fusions: HC_FUSE, ROUTE_FUSE, DECODE_TRIM, ROUTER_BF16 | kept (via build 10) | partly | exact-path IDENTICAL; ROUTER_BF16 paired quality within noise [M] | launches −17 %; +7.6 % (1 rep, indicative) (REPORTED) |
| [FG-03](cards/FG-03.md) | Stream vs hipGraph launch probe; graph track closed | dropped (track closed) | n/a | n/a | graph saves 0.35–0.88 µs per kernel (REPORTED) |
| [FG-04](cards/FG-04.md) | Tuned rocBLAS for the mHC mix GEMM (hcmix) | kept | partly | bitwise at T=999 only; differs at T=2048 [M via SV-07] | 1.8–4.2× per bucket; 16K prompt stage 5 −28 % (REPORTED) |
| [FG-05](cards/FG-05.md) | One attention launch per layer for all sequences of a batch (mattn) | kept (default) | yes | identical [R] | B8 +36.6 % (160.8 tok/s) (REPORTED) |
| [FG-06](cards/FG-06.md) | Prompt-attention swizzle, skip unpicked experts, Engram lock | not run | — | not run | not run |
| [FG-07](cards/FG-07.md) | Decode fusions on build 11: post step inside the all-reduce (AR_POST), merged attention launches (ATTN_MERGE) | kept (part of build 12) | ATTN_MERGE yes; AR_POST no; both just under | IDENTICAL, all arms [M] | B1 −0.667 ms (−4.6 %); −280 launches a token (MEASURED) |
| [FG-08](cards/FG-08.md) | XTX kernel round: seven candidates on the bench, one kept (`DS4_XK_QO`) | kept (build 14) | yes (B1 and production) | IDENTICAL [M] | B1 −0.154 ms (−1.18 %); production +0.010, not separable (MEASURED) |
| [FG-09](cards/FG-09.md) | The Engram chain one layer ahead on a side stream (`DS4_EG_AHEAD`) | kept (build 13; served) | yes (production); B1 not separable | IDENTICAL [M] | production −0.111 ms a token (−1.1 %) (MEASURED) |
| [FG-10](cards/FG-10.md) | Indexer top-k over many workgroups (`DS4_TOPK_MB`) | kept (build 13; served) | yes | IDENTICAL [M] | 110K −0.393 ms a token; 256K −0.846 (MEASURED) |
| [FG-11](cards/FG-11.md) | The many-workgroup top-k for batched decode (`DS4_TOPK_MB_BATCH`) | kept (build 15) | yes (all three within 7 %) | IDENTICAL; batched = solo [M] | 64K, 4 users −0.887 ms a step (−3.46 %) (MEASURED) |
| [FG-12](cards/FG-12.md) | A split indexer-score kernel, and a length dispatch (`DS4_IXS_DISPATCH`) | kept (dispatch in build 16; plain switch failed its gate) | dispatch: beyond the range (−0.135 vs −0.04 … −0.07 ms) | bench and engine IDENTICAL [M] | engine B1 −0.135 ms a token (five helpers), −0.165 (four helpers) (MEASURED) |
| [FG-13](cards/FG-13.md) | A faster XTX down-projection kernel (`DS4_DN_TPB`) | dropped (no gain) | no (the floor, 0) | IDENTICAL [M] | bench −20 % (REPORTED); engine B1 +0.026 ms (MEASURED) |
| [FG-14](cards/FG-14.md) | A rewritten XTX gate/up kernel | dropped | no | exact [R] | +0.7 … +10.7 µs a call against the engine's real kernel (REPORTED) |
| [FG-15](cards/FG-15.md) | Where a decode step's time goes on build 15: fixed cost and a kernel trace | finding | yes (every band) | batched = solo [M] | step = 11.07 ms + 3.46 ms a user; 88 % of a one-row step is shared (MEASURED); the traced gaps are real (FG-17) |
| [FG-16](cards/FG-16.md) | A decode fusion plan from the trace; launch gaps cap fusion at about 3 ms | desk (items 1 and 3 in build 16, item 2 in build 17) | items 1 and 3 yes | items 1 and 3 IDENTICAL [M] | five items about −0.45 ms a one-row token (3–4 %) (INFERRED); item 1 −0.208, item 3 −0.157 ms (MEASURED, in build 16) |
| [FG-17](cards/FG-17.md) | Decode phase timings without a tracer: real kernel gaps 2.9 µs; the MI50s on the critical path | finding (phase-timer build not in this release) | phases yes; gap and helper share no | flag on = off IDENTICAL [M] | B1 step 12.97 ms: attention 5.61, publish → join 2.67; gap 2.9 µs a boundary; helper last in 98.2 % of layers (MEASURED) |
| [FG-18](cards/FG-18.md) | Where the MI50 helper's time goes in a served step; graph replay and the XTX launch gap | finding (measurement build not in the published release) | 9 of 9 phase rows; gap, chain and lateness out | identical with stamps on [M] | served one user waits for the helpers 9–13 % of a step; graph replay saves 0.4–0.9 µs a kernel (MEASURED) |
| [FG-19](cards/FG-19.md) | XTX expert decode against llama.cpp (QBENCH) | finding | 2 of 5 (±25 % at batch 1 out; MXFP4 30–45 µs out; rel_rms out, narrowly) | ours vs CPU reference rel_max ≤ 1.4 % [M] | IQ2 batch 1 1.45–1.48× llama.cpp's GPU time, batch 4–8 0.51–0.83×; MXFP4 1.4–2.3× (MEASURED) |
| [FG-20](cards/FG-20.md) | The BF16 matvec re-scheduled for bandwidth (`DS4_BF16MV`) | kept (opt-in, release 4) | 4 users yes; verify step no (× 0.956 vs × 0.80–0.93) | bench BITWISE 16 of 16; served greedy check uninformative [M] | native pack: verify step 43.4 → 41.5 ms (−4.4 %), 4 users +9.8 %; kernel 0.51× (MEASURED) |
| [AR-01](cards/AR-01.md) | FP16 all-reduce for prompt chunks (default) | kept | n/a | at the rounding floor (1.63e-3) [R] | kept as default (REPORTED) |
| [AR-02](cards/AR-02.md) | One-phase decode all-reduce (AR_FUSE) | kept (default) | n/a | bitwise SAME [R] | decode +1.2 % (+0.6 % with drafter) (REPORTED) |
| [AR-03](cards/AR-03.md) | Peer-to-peer decode all-reduce | dropped (NO-GO) | no (predicted +20–30 %, got −4 / −11 %) | bit-identical [R] | −4 % / −11 % (REPORTED) |
| [AR-04](cards/AR-04.md) | Prompt all-reduce by peer stores (P2PK), then peer-major order (P2PK2) | kept (opt-in, then default in build 9) | yes | self-test bit-identical at load [R/M] | 16K prompt −19 %, 8K −23 % (REPORTED) |
| [AR-05](cards/AR-05.md) | 128-bit loads in the host all-reduce (AR_WIDE) | kept (default since build 8) | n/a | ids equal, bits identical [R] | 16K prompt +5.6 %, decode set +3.1 % (REPORTED) |
| [AR-06](cards/AR-06.md) | Peer-to-peer decode all-reduce, second attempt (dp2p) | dropped (NO-GO) | no | [R] | NO-GO (REPORTED) |
| [AR-07](cards/AR-07.md) | Decode all-reduce levers on build 10: FUSE_NORM (FN) and OVERLAP (OV) | shelved | no (predicted +3.5–7 % B1) | FN exact; OV DIFFER with drafter [M] | FN ~0; OV slower (MEASURED) |
| [AR-08](cards/AR-08.md) | Smart expert reduction (SER) | parked | — | not run | not run |
| [CP-01](cards/CP-01.md) | XTX power caps, −50 mV undervolt, boot service, fan hold | kept | n/a | n/a | not measured (power sweep never run) |
| [CP-02](cards/CP-02.md) | Pin the MI50 PCIe link speed | dropped (NO-GO) | no | identical [R] | decode set −10.1 %, B8 −8.5 % (REPORTED) |
| [CP-03](cards/CP-03.md) | MI50 DPM `high` (clock pin) | kept (via launch wrapper) | n/a | IDENTICAL [M] | tp4h4 B1 58.31 → 60.95 tok/s (MEASURED) |
| [CP-04](cards/CP-04.md) | DPM `high` vs manual top-level clock pin | dropped (manual pin) | yes (desk model said manual would not help) | IDENTICAL [M] | manual pin loses at B8 (44.25 vs 41.15 ms) (MEASURED) |
| [CP-05](cards/CP-05.md) | MI50 power cap 155 W and 178 W instead of 135 W | dropped (caps stay 135 W) | decode yes (no change); 110K prompt better than predicted | IDENTICAL [M] | 178 W: 32K prompt −2.78 %, 110K −2.39 %; decode no change (MEASURED) |
| [PF-01](cards/PF-01.md) | Prompt streaming (TP_DIRECT) and a 2,048-row attention slab | kept (default) | n/a | [R] | 2K 7.45 s / 16K 37.29 s baseline (REPORTED) |
| [PF-02](cards/PF-02.md) | Matrix-core prompt attention with hi+lo FP16 ("f32" mode) | kept (default) | corrected theory | paired quality pass [R] | 16K TTFT 18.75 → 11.06 s with HIPK (REPORTED) |
| [PF-03](cards/PF-03.md) | Matrix-core expert tiles (HIPK) default; PF_WMMA variants | kept (HIPK=1) | n/a | within noise [R] | 16K TTFT 12.3 s cold / 10.7 s warm (REPORTED) |
| [PF-04](cards/PF-04.md) | Three-way prompt-attention split (ATTN_SPLIT3) | parked (opt-in) | yes (5 % bar) | paired quality pass [R] | 8K / 16K TTFT −11 / −12 % (REPORTED) |
| [PF-05](cards/PF-05.md) | Skip late layers' prompt work for tokens that do not need it (TP prompt skip) | kept (default) | n/a | KL 1.75e-4 [R] | kept as default (REPORTED) |
| [PF-06](cards/PF-06.md) | MI50 prefill helper | dropped (not adopted) | — | parity pass [R] | MI50 2.4–2.7 TFLOPS end to end (REPORTED) |
| [PF-07](cards/PF-07.md) | Pipelined layer-split prefill beside tp3 decode | dropped (NO-GO) | — | not run | NO-GO (REPORTED) |
| [PF-08](cards/PF-08.md) | Router 8 rows at a time; all-reduce slab for prompts | not run | — | not run | not run |
| [PF-09](cards/PF-09.md) | Prefill discriminator: is XTX #4 the slow member? | finding (cause open) | n/a | n/a | tp3 with XTX #4: 353–367 vs 568–592 tok/s at 2K (MEASURED) |
| [PF-10](cards/PF-10.md) | The prompt's fixed cost: ~3.3 s a prompt, 58 % of prompt time under agent traffic | finding (fix measured in PF-11) | n/a | n/a | T = 3.26 s + 0.43 ms × N; 21.2 GB streamed per XTX at 6.17 GB/s (MEASURED / INFERRED) |
| [PF-11](cards/PF-11.md) | MI50 prompt helper (`DS4_PF_HELPER`): faster short prompts, NOT bit-identical | served (from build 13s), validation pending | partly (fixed cost ~1 s vs 0.2-0.5 predicted) | **not bit-identical**; unset IDENTICAL; validation run formally inconclusive; batched ≠ solo [M/R] | 1K suffix 3.42 → 1.33 s, 2K 3.71 → 2.11 s, 16K equal (MEASURED) |
| [PF-12](cards/PF-12.md) | MI50 prompt tiles (`DS4_PF_TILE`): bit-exact | kept (build 15) | yes, exceeded (× 0.847 vs ≤ 0.93) | bench bit-identical; served IDENTICAL [M] | TTFT −15 % at 2K, 8K, 32K and 4 users × 8K (MEASURED) |
| [PF-13](cards/PF-13.md) | The MI50 prompt helper 2K–128K: never slower in one process, slower from 16K served | finding | yes (both runs, and the cap) | n/a (helper not bit-identical by design) | served 32K TTFT on 22.2 s vs off 18.4 s; 2K 2.9 vs 4.6 s; the cap, in build 16: 32K 18.4 s, 64K 35.0 s (MEASURED) |
| [QA-01](cards/QA-01.md) | Helper quality gate: XTX alone vs XTX + MI50, teacher-forced | kept (pass) | yes | pass: top-1 96.94 %, median KL 4.5e-4 [M] | top-1 96.94 %, ppl 3.3889 → 3.3861 (MEASURED) |
| [QA-02](cards/QA-02.md) | The rounding floor: why median KL fails 1e-3 on every TP change | method | n/a | floor ≈ 1.6–1.8e-3 median KL [M] | tp3 vs single-group reference: median KL 1.79e-3, top-1 95.84 % (MEASURED) |
| [QA-03](cards/QA-03.md) | Build 10 quality vs a fresh reference of the same pack | kept (pass on the paired test) | yes (paired); KL line printed GATE FAIL | top-1 96.28 %; paired intervals include 0 [M] | tp4h4 top-1 96.28 %, ppl 3.4014 vs 3.3968 (MEASURED) |
| [QA-04](cards/QA-04.md) | Higher-precision (Q8) reference | not run | — | not run | never scored; two attempts hung: a Q8 pack does not fit in host RAM (REPORTED) |
| [QA-05](cards/QA-05.md) | Task-suite comparison against a hosted reference | running | — | running | running |
| [QA-06](cards/QA-06.md) | One repeated-token decode | open (unexplained) | — | 1 of 9 runs [R] | 1 garbage decode in 9 runs (REPORTED) |
| [QA-07](cards/QA-07.md) | Build 10's "DEGENERATE continuations" flag in the batched-vs-solo check | explained | n/a | batched = solo, 256/256 every sequence [M] | flag is a repetition heuristic on the prompt's own continuation (MEASURED) |
| [QA-08](cards/QA-08.md) | Output depends on which experts fit in VRAM: code size and context length are part of exactness | finding | n/a | fixed binary IDENTICAL to build 11 [M] | 2 MiB less VRAM per XTX → one expert fewer → DIFFER from token 22 (MEASURED) |
| [QA-09](cards/QA-09.md) | Long context: 240K retrieval at three depths, native 1M context, memory per token | finding (fuller check pending) | n/a | retrieval 3 of 3 exact [M] | prefill ~1,590 tok/s at 240K; decode 77–80 tok/s; 920 B a token per XTX (MEASURED / INFERRED); addendum: 512K context +1.17 ms a token over 1K (MEASURED) |
| [QA-10](cards/QA-10.md) | Engine parity with upstream Strata and llama.cpp at the same weights (VANILLA) | kept (method; ENGINE PASS) | yes (pre-registered rule) | KL from llama.cpp 0.00417 (ours) vs 0.00414 (upstream), floor 0.002 [M] | ours − upstream +0.00003 nats a token, top-1 96.50 vs 96.99 % (MEASURED) |
| [QA-11](cards/QA-11.md) | The 2.25-bit model against the hosted model's logprobs (HOSTLP) | finding | yes (QUANT COST SHOWN; native closes ≥ 70 %) | V and P1 alike vs hosted [M] | 2.25 bpw 0.0434–0.0438 nats, 90.3 % top-1; native MXFP4 0.0054, 96.0 % (88 % of the gap closed) (MEASURED / REPORTED) |
| [QA-12](cards/QA-12.md) | Mixed precision: native MXFP4 hot experts on top of the 2.25-bit pack | finding (code not in this release) | quality bar 0.019 missed by 0.002 | no-manifest build = native build, BITWISE at equal placement [M] | x25c KL 0.0438 → 0.0210 (native 0.0054) at 1.18× build 17's time a token (MEASURED) |
| [SV-01](cards/SV-01.md) | OpenAI-compatible launchers | kept | n/a | smoke prompt [R] | — |
| [SV-02](cards/SV-02.md) | Prefix store: prompt state survives a restart | kept (opt-in) | n/a | [R] | 25K-token prompt TTFT 63.1 → 10.4 s after restart (REPORTED) |
| [SV-03](cards/SV-03.md) | Batched decode banks | kept | yes (conditional go) | exact B1–B8 [R] | B4 103 tok/s total with stripe helpers (REPORTED) |
| [SV-04](cards/SV-04.md) | Continuous batching in the server | not run (results not found) | — | not run | deployed with 1 parallel slot |
| [SV-05](cards/SV-05.md) | Tool-call probe and 4-hour agent soak | not run (results not found) | — | not run | — |
| [SV-06](cards/SV-06.md) | Build 11: helper overlap + device-direct + skip + MI50 clock pin | kept (deployed) | n/a | byte-identical to build 10 [M] | 65.16 vs 57.87 tok/s (+12.6 %) (MEASURED) |
| [SV-07](cards/SV-07.md) | Build 10: fixes + tuned GEMM + one-launch attention + fusions, on tp4 | kept (deployed until build 11) | n/a | flags-off IDENTICAL; flags-on DIFFER from ~token 15 (explained); quality QA-03 [M] | tp4h4 decode set 79.81, B1 59.32, B8 163.30 tok/s, 2K prompt 3,476 ms (MEASURED) |
| [SV-08](cards/SV-08.md) | Build 11 under real agent traffic: serving profile from the server's own counters | finding | n/a | n/a | mean context ~55K, 94.8 % cache hits, 11.47 ms per generated token, 59 % draft acceptance (MEASURED) |
| [SV-09](cards/SV-09.md) | Build 12: the exact overnight flags merged and measured as one stack | kept (build 12) | yes for F, F+LP, EG; HW at the low edge; MK no | IDENTICAL to build 11, every arm [M] | B1 14.38 → 13.27 ms (−7.7 %); B8 −6.8 %; production −4.4 … −5.3 % (MEASURED) |
| [SV-10](cards/SV-10.md) | Builds 13 and 13s: four exact features merged, then concurrent serving | kept | identity yes; batched = solo no (prompt helper) | 14 of 14 IDENTICAL; batched 5 of 8 with the helper, 8 of 8 without [M] | no bank tax: +0.1 % engine, +0.9 % server (MEASURED) |
| [SV-11](cards/SV-11.md) | Served benchmark on build 13s: one to eight users, helper on / off, 4 and 8 slots | finding | n/a | n/a | 1 user 124 tok/s at 2K (8.05 ms); 8 users 186 tok/s total (MEASURED) |
| [SV-12](cards/SV-12.md) | Two prompt-read hangs, and a build that names its own stall | open | n/a | diagnostic build host-only [R] | smoke: 0 hangs, 0 false triggers in 16 prompts (MEASURED) |
| [SV-13](cards/SV-13.md) | Like for like: the first four-plus-four build against build 14, same night | finding | A yes (× 1.325 in × 1.20–1.35); B inside, step 0 not met | n/a | decode × 1.325 short, × 1.283 after 2K; anchor reproduced; one XTX × 1.02 (MEASURED) |
| [SV-14](cards/SV-14.md) | Build 15 and the final served benchmark | kept (build 15; the release build is now build 16) | yes (all three) | 9 of 9 IDENTICAL; served IDENTICAL [M] | 1 user 124 tok/s; TTFT 2K 2.2–2.5 s, 32K 23.1 s; 8 users 185 tok/s (MEASURED) |
| [SV-15](cards/SV-15.md) | Prompt reuse across agent turns: already built and serving | finding | warm turn no (2.1 s vs 0.9–1.6 predicted) | reuse IDENTICAL to a fresh read at fixed placement [M] (first run's differences: placement) | 92.1 % of live prompt tokens reused (logs); 96 % served, warm 32K turn 2.1 s (MEASURED) |
| [SV-16](cards/SV-16.md) | Build 16, the release build: five changes on build 15 and four upstream fixes | kept (the release build) | 3 of 5 (one-user decode and 8K first token out) | 8 of 8 single-process + 1 served IDENTICAL to build 15 [M] | 32K cold first token 23.1 → 18.3 s; 8K 5.9 → 6.7 s; several users ±3 % (MEASURED) |
| [SV-17](cards/SV-17.md) | Build 17 = build 16 + the batched-path fusion + the cache fix that keeps promotion | kept (build 17, the release build of release 3) | 7 of 9 (both 8-user cells out, better) | identical to build 16; 0 counted twice [M] | 8 users 180.3 → 202.8 tok/s (+12.5 %), 4 users +5.3 %, 8K first token −12.5 % (MEASURED) |
| [HW-01](cards/HW-01.md) | XTX at 89–98 °C on the first gate runs | caveat (fixed) | n/a | n/a | junction 89–98 °C [M in internal logs] |
| [HW-02](cards/HW-02.md) | A fourth XTX: junction 31 → 92 °C in 2 seconds | card pulled | n/a | n/a | junction-to-edge gap 46 °C within 2 s at 257 W (MEASURED) |
| [HW-03](cards/HW-03.md) | A PSU cable splitter burned; cards moved to direct cables, then new PSUs | hardware event | n/a | n/a | — |
| [HW-04](cards/HW-04.md) | MI50 #2 reflashed with a different vBIOS | done | n/a | n/a | — |
| [HW-05](cards/HW-05.md) | An XTX's SMU stopped responding mid-sweep; the guard that followed | incident (guard added) | n/a | n/a | ~4 h hang until a manual power cycle [M in internal logs] |
| [HW-06](cards/HW-06.md) | The "Gen3" misreading was idle PCIe power saving | corrected | n/a (a question); pin test: no contrast | n/a | every MI50 link reads 2.5 GT/s x4 at idle by design; under decode load at `high` the links held Gen4 100 % of the time and pinning changed nothing (B8 −0.26 ms, noise) (MEASURED) |
| [HW-07](cards/HW-07.md) | One slow peer leg: inter-quadrant fabric, not errors | finding | n/a | n/a | AER clean (REPORTED) |
| [HW-08](cards/HW-08.md) | Four XTX boards, four sets of limits; one hot card trips the watcher on long-prompt prefill | finding | n/a | n/a | XTX #4 lowest cap and top clock; XTX #3 junction 92 °C peak (MEASURED) |
| [HW-09](cards/HW-09.md) | XTX #2's power controller hung and the host lost power; the card was replaced | incident (card replaced) | n/a | n/a | third SMU hang on the same card; replacement board gap ~28 °C (REPORTED) |
| [HW-10](cards/HW-10.md) | 12 Nov hardware work: a fifth MI50 for benches, XTX #3 re-pasted, a riser at x8 | hardware event | no (the x8 prompt estimate) | n/a | XTX #3 69 °C peak after re-pasting; x8 vs x16 ≤ 0.04 ms a token (REPORTED) |
| [HW-11](cards/HW-11.md) | Host memory and CPU while serving: the 123 GB is the expert RAM store; Engram lives in the page cache | finding | n/a | n/a | RAM store 119.8 GB preloaded; Engram 7.4 of 110.6 GB in page cache; serving 4–5 cores, prompt bursts 14–16 (REPORTED) |
| [HW-12](cards/HW-12.md) | MI50 peer reads go stale after the MI50's own SDMA writes, until its HDP is flushed | finding | yes (the HDP readout) | n/a | SDMA-written source CORRUPT; blit, kernel-written or HDP flush WORKS; settle 1 / 20 ms CORRUPT; kernels 6.17.2 and 7.0 (MEASURED) |
| [FN-01](cards/FN-01.md) | Main-engine (Qwen3.8-Flash-Next) helper gates | pointer | — | see EB-01, HX-01 | see EB-01 |
| [FN-02](cards/FN-02.md) | Flash Next lever audit and ranked plan | desk | — | n/a | MTP ranked first (+45–70 % predicted) (REPORTED) |
| [FN-03](cards/FN-03.md) | Qwen3.8-Flash-Next: 1 XTX + 1 MI50 helper + MTP | kept (best measured layout) | yes (+50.6 %, inside +45–70 %) | teacher-forced 100 % / KL 0 (prompt path only); decoded MTP quality untested [M] | 93.31 tok/s (arm e) (MEASURED) |
| [ME-01](cards/ME-01.md) | Frozen multi-prompt decode set and per-step time | method | n/a | n/a | — |
| [ME-02](cards/ME-02.md) | Paired quality against a common reference | method | n/a | n/a | — |
| [ME-03](cards/ME-03.md) | Pre-registered pass and abort lines | method | n/a | n/a | — |
| [ME-04](cards/ME-04.md) | One GPU job at a time; no bulk I/O during timing | method | n/a | n/a | — |
| [ME-05](cards/ME-05.md) | At least 3 interleaved reps for gains under 5 % | method | n/a | n/a | — |
| [ME-06](cards/ME-06.md) | Tracers inflate wall time 2.7× | method | n/a | n/a | B1 45.7 ms traced vs 17.1 untraced (MEASURED) |
| [ME-07](cards/ME-07.md) | Stale baselines: an error table | method | n/a | n/a | 13 corrected claims (REPORTED) |
| [ME-08](cards/ME-08.md) | Byte-identical off paths for every opt-in flag | method | n/a | n/a | — |
| [ME-09](cards/ME-09.md) | Every GPU arm runs with the MI50s at DPM `high` | method | n/a | n/a | — |
| [ME-10](cards/ME-10.md) | Every benchmarked build is in git, and rebuilds byte for byte | method | n/a | n/a | 9 of 9 builds byte-identical (MEASURED) |
| [ME-11](cards/ME-11.md) | Builds pinned to spare cores may run during timing runs | method | n/a | n/a | 8.823 / 8.853 (with build) / 8.867 ms a token (REPORTED) |
| [ME-12](cards/ME-12.md) | An Engram access-trace build that changes nothing, and the SSD's random-read latency | method (trace build not in the published release) | n/a | identical, trace on and off [M] | SSD 4 KiB random: QD1 59 µs p50, QD8 106.8 K IOPS (MEASURED) |

Groups: EB engine bring-up · TP layout and tensor parallel · HX MI50 helpers · DR drafter · FG fusion and graphs ·
AR all-reduce · CP clocks and power · PF prefill · QA quality · SV serving and builds · HW hardware events ·
FN Qwen3.8-Flash-Next · ME method.

## Acknowledgements

Thanks first to Niko1221 and the Strata contributors, whose engine all of this builds on. Licence notices and credits
for ported code are in [NOTICE.md](../../NOTICE.md) in the repository root.
