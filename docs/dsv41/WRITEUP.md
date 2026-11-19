# DeepSeek V4.1 Flash on four RX 7900 XTX and four MI50s: what we tried and what we measured

Built by YST. Contact: https://yst.dev

This is a long account of five days of work on one box, written for engineers who run models on hardware like this.
It covers the target, the hardware, the method, the sequence of changes from 22.7 to 69 tokens per second on DeepSeek
V4.1 Flash, the 93 tokens per second on Qwen3.8-Flash-Next, what did not work, what the hardware did, and how far the
quality checks go. Every claim links to a card in [cards/](cards/), where the evidence, the data extract and the
caveats are. The index is [README.md](README.md); [FEATURES.md](FEATURES.md) maps every engine switch to the
stage of a token it acts on, with its status and measured effect.

Two caveats apply throughout:

- The timing numbers come from a private prompt set. Ratios between arms of the same run compare like with like.
  Absolute numbers are indicative until the public-text re-runs listed in the README are done.
- Everything here was measured on one box, with one ROCm version and two quantised model packs. It may not transfer
  to other boards, PCIe topologies, drivers or quantisations.

*Update, 13 Nov.* The account above was written on 11 Nov. Two more days of work (builds 13, 13s, 14 and 15, the first
served benchmark, and a hardware incident) are added as dated paragraphs in sections 2, 4, 6, 7, 8, 9 and 10; the
original text is left as it was.

*Update, 19 Nov.* Our fork of Strata is now called Gary; the code identifiers are unchanged. The lessons of the whole project, each with the cards behind it, are collected in
[LESSONS.md](LESSONS.md). The work of 18–19 Nov on the native MXFP4 pack (prompt reads 15.2 → 3.3 s at 2K, P2P, the CPU
lane) is in the README's "Since release 4" section; none of its code is in a release.

## 1. The goal

Run DeepSeek V4.1 Flash locally at a usable single-stream speed, using cards that are cheap per gigabyte of memory:
RX 7900 XTX (24 GB, RDNA3) for the main work and Instinct MI50 (32 GB, gfx906, several generations older) as extra
expert memory and compute. The engine is upstream Strata v0.2.9, which already ran DeepSeek V4.1 Flash on one GPU
with the remaining experts in host RAM. The question was how much of the rest of the box could be put to use, and at
what cost in correctness.

## 2. The hardware

| label | card | link |
|---|---|---|
| XTX #1–3 | RX 7900 XTX 24 GB | PCIe Gen4 x16 |
| XTX #4 | RX 7900 XTX 24 GB (added 9 Nov) | PCIe Gen4 x8 |
| MI50 #1–4 | Instinct MI50 32 GB, 135 W cap | PCIe Gen4 x4 each |

One EPYC Zen 2 16-core host with DDR4-2666, ROCm 7.14 on Linux. From 8 Nov the XTX cards ran at their minimum power
cap with a −50 mV undervolt ([CP-01](cards/CP-01.md)); its speed cost was never measured. From 10 Nov 13:50Z the MI50s
ran at DPM `high` ([CP-03](cards/CP-03.md)).

The layout changed three times, and each card says which one it ran on:

1. **6–7 Nov:** one to three XTX, with the routed experts that did not fit in VRAM held in host RAM and computed by the
   CPU (the "CPU lane"), plus one to three MI50 helpers.
2. **8–9 Nov:** three XTX on x16 links and three MI50s, with every routed expert on a GPU ([TP-04](cards/TP-04.md)).
3. **From 9 Nov afternoon:** four XTX and four MI50s ([TP-05](cards/TP-05.md)).

*Update, 13 Nov.* On 12 Nov XTX #3 was re-pasted and a fifth MI50 (MI50 #5) went in as a bench card only; it is never
a helper in a timed run ([HW-10](cards/HW-10.md)). On 13 Nov XTX #2 was replaced by a different board after an
incident, and XTX #4 now runs its board's second firmware position ([HW-09](cards/HW-09.md)). Runs after 13 Nov 13:09
are on that changed set; each card says which set it used.

## 3. The method

The method was set up because early numbers moved a lot for reasons unrelated to the change being tested.
These rules accumulated over the five days; cards from before a rule say so.

- **Predict first, in writing.** Before a run, a design note or the run script's header states the expected effect,
  a pass line and an abort line, committed with a timestamp earlier than the run ([ME-03](cards/ME-03.md)). Where no
  prediction was written, the card says "pre-registered: no". Predictions cite the profile they were computed from,
  after several predictions built on an older build's profile were too optimistic ([ME-07](cards/ME-07.md),
  [AR-03](cards/AR-03.md)).
- **Byte identity as the gate.** Every new behaviour sits behind a flag that defaults off. Each run first checks that
  flags-off output is byte-identical to the previous build, and that an exact change with its flag on is byte-identical
  too: decode output, batched-versus-solo sequences, logits dumps, teacher-forced scores ([ME-08](cards/ME-08.md)). Any
  DIFFER stops the run. That gate stopped device-direct helper windows v1 ([HX-16](cards/HX-16.md)).
- **Paired quality for approximate changes.** Changes that are not exact are scored teacher-forced against a common
  reference, with paired bootstrap intervals for top-1 agreement, KL and perplexity ([ME-02](cards/ME-02.md)).
- **Interleaved reps.** At least three reps, in forward / reverse / forward order, each in a fresh process, whenever
  the expected effect is under 5 % ([ME-05](cards/ME-05.md)). One GPU job at a time, and no builds or bulk copies
  during timing ([ME-04](cards/ME-04.md)). When a background copy did overlap the build-10 grid, the summary tagged
  every affected run ([SV-07](cards/SV-07.md)).
- **The time budget must add up.** A gain has to come out of a measured slice of the step. If the predicted slice
  shrinks but the step does not, the time went somewhere else, and the card says where. Untraced counters are used for
  wall time, because the tracer made a 17.1 ms token take 45.7 ms ([ME-06](cards/ME-06.md)).
- **A fixed workload.** Single-prompt numbers swung by tens of percent with speculative-drafting acceptance
  ([EB-01](cards/EB-01.md)). From 8 Nov, decode was measured on a frozen 8-prompt set and batch tests in milliseconds
  per step ([ME-01](cards/ME-01.md)).

## 4. The journey: 22.7 to 69 tokens per second

The drafter is DSpark, DeepSeek's own drafter shipped with V4.1, which the engine runs beside the model. The numbers below are single-stream decode with the drafter off, unless a row says otherwise. Prompts differ between
the early and late points, so read the sequence as the shape of the progress, not as a controlled series.

| step | what changed | decode tok/s | card |
|---|---|---|---|
| start | 1 XTX, experts the XTX cannot hold in host RAM | 21.46–22.67 [M] | [EB-03](cards/EB-03.md) |
| + 1 MI50 helper | the MI50 holds 2,964 experts and computes them | 26.12–26.80 [M] | [EB-03](cards/EB-03.md) |
| tensor parallel over 3 XTX | heads 0/32/32, FFN 3 × 768 | 30.4–30.9 [R] | [TP-01](cards/TP-01.md) |
| + MI50 helper | | 32.85 [R] | [TP-03](cards/TP-03.md) |
| no-SDMA helper window | the helper request becomes one device-side sequence | 33.39 (1 XTX + MI50) [R] | [HX-06](cards/HX-06.md) |
| new layout | 3 XTX at x16 + 3 MI50, every expert on a GPU | 48.66 [M] | [TP-04](cards/TP-04.md) |
| build 10 on 4 + 4 | tp4, fixes, tuned GEMM, fusions, one-launch batch attention | 57.87 [M] | [SV-06](cards/SV-06.md) |
| + MI50 clock pin | MI50s at DPM `high` | 59.68 [M] | [CP-03](cards/CP-03.md), [SV-06](cards/SV-06.md) |
| build 11 | helper overlap, device-direct windows, skip empty helpers | 65.16 [M] | [SV-06](cards/SV-06.md) |
| build 12 | two launch fusions, lead-path and helper-window changes, Engram split | **69.41** [M] | [SV-09](cards/SV-09.md) |
| build 14 (13 Nov) | Engram ahead, long-context top-k, one more launch fold (builds 13-14) | **71.0** after a 2K prompt; 77.1 after a 64-token prompt [M] | [SV-13](cards/SV-13.md) |

[M] measured, [R] reported at the time. The build-11 comparison is one run per arm; three interleaved reps are part of
the re-run plan. The build-12 figure is one 2,048-token run of the stack without the Engram split, beside 64.70 for the
build-11 binary in the same run. Its timing reps used a 64-token prompt: 14.38 → 13.27 ms a token (69.5 → 75.3 tok/s),
three interleaved reps a step.

With the drafter on, the 8-prompt decode set reached 79.81 tok/s (geomean) on build 10, and batched decode reached
163.3 tok/s in total at eight sequences ([SV-07](cards/SV-07.md)). On build 12, eight sequences reached 208 tok/s in
total (38.5 ms a step, against 41.3 for build 11 flags-off in the same run), and served long prompts with the drafter on
and sampling took 4.4-5.3 % less time per generated token ([SV-09](cards/SV-09.md)). Those are different measurements from the
single-stream column above, and are not comparable with it.

*Update, 13 Nov.* The build-14 row comes from a like-for-like run: the first four-plus-four build and build 14 measured
back to back in one harness on the same night, drafter off. Build 14 decoded 1.32 × faster after a 64-token prompt
(58.2 → 77.1 tok/s), 1.28 × after a 2K prompt (55.3 → 71.0) and read a 2K prompt 1.21 × faster (501 → 607 tok/s)
([SV-13](cards/SV-13.md)). An anchor phase re-ran the exact 9 Nov commands and reproduced the old numbers within 1 %, so
the baseline was measured right. It is the first ratio in this table that compares like with like. On one XTX with the
experts in host RAM (the very first layout), build 14 is about where the first binaries were (× 1.02 with its flags),
because that layout is bound by streaming experts from host RAM [I].

### Where the steps came from

**Getting experts off the CPU.** On one XTX, 27,087 of 61,200 expert picks in a 256-token decode went to the CPU lane
([EB-03](cards/EB-03.md)). The first MI50 took part of that load and gave +20 %. The physical rebuild that put every
routed expert on a GPU gave the largest single step, from the low 30s to 48.66 tok/s. The written projection for it was
about 41 ([TP-04](cards/TP-04.md)), so the prediction was too low.

**Tensor parallel, then a fourth card.** Splitting attention and FFN over three XTX ([TP-01](cards/TP-01.md)) needed a
prompt-path fix that took a 2,048-token prompt from 384 s to 10 s, by moving each member's expert part in three
contiguous DMAs instead of one command per row ([TP-02](cards/TP-02.md)). A fourth XTX on tp4 improved the decode set by
9.9 % and made 2K prefill 10.7 % slower ([TP-05](cards/TP-05.md)). The prefill cost turned out to be that card
specifically: with XTX #4 swapped into a tp3 group, 2K prefill fell from 568–592 to 353–367 tok/s, and the other members
spent the difference waiting for it in the all-reduce ([PF-09](cards/PF-09.md)). Whether that is its x8 link or its
place in the PCIe topology is not yet separated. For the helper wait, the same swap made no difference: there the cause
was TP width, not the card ([HX-13](cards/HX-13.md)).

**Build 10: smaller kernels, fewer launches.** Build 10 merged four correctness fixes ([EB-07](cards/EB-07.md)), a tuned
rocBLAS table for one GEMM ([FG-04](cards/FG-04.md)), one attention launch per layer for all sequences of a batch
([FG-05](cards/FG-05.md)) and four decode fusions ([FG-02](cards/FG-02.md)). Measured against the previous tp4 build,
B8 rose 44 % and B1 7.9 % ([SV-07](cards/SV-07.md)). With its deploy settings, build 10 was not byte-identical to
build 9. Isolation runs traced that to the tuned GEMM table alone, which sums in a different
FP32 order at 2,048 rows. Its bitwise claim had only been checked at 999 rows. So build 10 got a quality comparison
instead of an identity gate ([QA-03](cards/QA-03.md)).

**Build 11: the helper chain.** A profile of build 10 on four XTX and four MI50s showed where the single-stream token
went: the lead XTX waited 3.5 ms of an 18.7 ms token for the helpers, and at eight sequences the helper chain was 23.3 ms
of a 53.3 ms step ([HX-14](cards/HX-14.md)). One helper request took about 179 µs, of which the MI50's expert kernels were
only about 60 µs. Build 11 attacked that chain in three steps: overlap the return of rows
([HX-10](cards/HX-10.md)); let each MI50 wait for its request on the device and join the results on the lead in one
kernel ([HX-17](cards/HX-17.md)); and skip the join for MI50s with no expert that layer ([HX-18](cards/HX-18.md)). With
the MI50 clock pin ([CP-03](cards/CP-03.md)) this went from 57.87 to 65.16 tok/s, with output byte-identical to build 10
([SV-06](cards/SV-06.md)). Section 6 covers why each step gave less than predicted.

**Build 12: fewer launches, and helpers that finish sooner.** The build-11 timeline ([HX-20](cards/HX-20.md)) put about
2.7 µs on each dependent kernel on the lead's path and showed the lead arriving last at every MoE all-reduce. A desk
breakdown of one token ([ANATOMY.md](ANATOMY.md)) then named the slices, and each candidate was built behind its own flag
with a pre-registered prediction. Merging five small attention launches into two and folding the post step into the
all-reduce removed 280 launches a token on every card and saved 0.67 ms ([FG-07](cards/FG-07.md)). Of five lead-path
changes, the one on the MI50 side (the helper's done flag written by its last block) paid; the lead-side ones mostly did
not ([HX-21](cards/HX-21.md)). Fewer launches per helper window gave a little more ([HX-22](cards/HX-22.md)), and
splitting the Engram layers' GEMV over the four XTX saved 0.12 ms ([SV-09](cards/SV-09.md)). Merged into one binary and
gated for byte identity against build 11, the stack took a short-prompt token from 14.38 to 13.27 ms, B8 from 41.3 to
38.5 ms a step, and production decode per token by 4.4-5.3 % at matched output length. Rewritten MI50 expert kernels were
exact but 0.33 ms slower and were left out ([HX-23](cards/HX-23.md)).

**Builds 13 to 15 (12-13 Nov): long context, prompts and several users.** By now a one-row token on its own had
little left that a kernel change could reach, and most served steps are not one-row steps anyway: with the drafter on,
only 5.1 % of production decode steps are ([FG-08](cards/FG-08.md)). So the gains came from elsewhere. Starting the
Engram layers' host-row copy one layer early took 1.1 % off every served token ([FG-09](cards/FG-09.md)). Splitting the
indexer's top-512 pick over many workgroups took 0.39 ms a token off decode at 110K of context and 0.85 ms at 256K
([FG-10](cards/FG-10.md)), and the same change on the batched path took 3.5 % off a 4-user step at 64K
([FG-11](cards/FG-11.md)). Those went into build 13 with an MI50 kernel change that is exact but helps only one-row
windows ([HX-24](cards/HX-24.md)); the merge was gated byte for byte, 14 of 14 ([SV-10](cards/SV-10.md)). Build 13s let
sampled requests draft inside batched steps, worth +22 % total throughput at three users
([DR-07](cards/DR-07.md)), and serving a single user from a 4-slot process turned out to cost nothing
([SV-10](cards/SV-10.md)). Build 14 added one more launch fold on the attention output (exact; −0.15 ms a one-row
token, nothing in serving) ([FG-08](cards/FG-08.md)). Build 15 adds a bit-exact tiled kernel for the MI50s' share of
prompt reading, which took 15 % off the time to first token at every length tested ([PF-12](cards/PF-12.md)).

The first end-to-end served benchmark, on build 13s with sampled requests and the drafter on, read about 124 tok/s for
one user at a 2K context (8.05 ms a token), 146-153 tok/s in total for two to four users, and 186 tok/s in total for
eight users with eight slots ([SV-11](cards/SV-11.md)). Those are served numbers with the drafter and are not
comparable with the single-stream column above. They also show the next limit: prompts are read one at a time, so
with eight users the last one waits 25 s for a 2K prompt.

*Update, 13 Nov.* The same benchmark on build 15, after its identity phase passed (9 of 9
comparisons and one served user IDENTICAL to build 14): one user 124 tok/s at 2K, time to first token 2.2-2.5 s at 2K,
5.9 s at 8K and 23.1 s at 32K (−4 to −15 %), and 185 tok/s in total for eight users; the last of eight 8K users now waits
52 s instead of 61 ([SV-14](cards/SV-14.md)). Two follow-ups found what still limits long prompts. In one process the MI50
prompt helper is never slower than no helper, but behind the server it is slower from 16K up (32K 22.2 s against
18.4 s), so for long served prompts the helper is better switched off ([PF-13](cards/PF-13.md)). And with the expert cache on,
every prompt leaves 8 more experts streaming from host memory, so a long-running server's first token slows by 13.9 %
over 50 prompts; restarting the server resets it. That leak was a known issue of build 15
([HX-26](cards/HX-26.md)). Its cause was found the next morning and build 16 fixes it; the same run showed that, with the leak, the same greedy prompts can get different (still correct) answers
after a server has read many prompts, which the fix also removes ([HX-26](cards/HX-26.md) addendum).

*Update, 14 Nov.* That change of answers had a cause worse than near-tie flips: with device-direct MI50 windows, an
expert the cache had moved into XTX memory was still listed in its MI50's table, so each pick of it was computed on both
and summed twice: 4.93 % of all device-direct picks after 53 prompts, half of them on a stale MI50 input
([HX-28](cards/HX-28.md)). Every earlier build served this way had it. Build 16, the build in this release, fixes it
with `DS4_EC_FIX=1`, and also carries the indexer dispatch, fusion items 1 and 3, the prompt helper's length cap and four
upstream correctness fixes. It gives build 15's tokens in all 9 identity comparisons; served, a cold 32K prompt's first
token went from 23.1 to 18.3 s, an 8K prompt's from 5.9 to 6.7 s (the fix's price), and several users stayed within
about 3 % ([SV-16](cards/SV-16.md)).

## 5. Qwen3.8-Flash-Next: 93 tokens per second

The same helper machinery runs on Strata's main engine. For Qwen3.8-Flash-Next (ISTA-DASLab's 512-expert IQ3_S pack,
every expert resident in VRAM), a desk audit two days earlier had ranked multi-token prediction (MTP) first and written
down +45–70 % at one stream, to roughly 85–100 tok/s ([FN-02](cards/FN-02.md)). Five arms ran interleaved
([FN-03](cards/FN-03.md)):

| arm | layout | tok/s |
|---|---|---|
| a | XTX #1 + MI50 #1 helper | 61.95 |
| b | layer split over XTX #1–3 | 44.23 |
| c | b + MTP | 66.33 |
| d | c + two pipelined windows | 87.83 |
| e | a + MTP | **93.31** |

The prediction held: +50.6 % from a to e. The layer split's abort line fired, because its decode windows were 47 %
longer, so the split is only useful for prompts. Arm e has two reps rather than three.

## 6. What did not work, and what it showed

The misses are carded in full alongside the gains. Several of them share one cause.

**Removing a wait that is not on the critical path moves the time elsewhere.** Device-direct helper windows
([HX-17](cards/HX-17.md)) were predicted to save 1.5–2.0 ms per single-stream token, from the host's per-layer issue and
join. The host's share did fall from 2.7 ms to 0.1 ms. The step fell by only 0.63 ms. The rest of the time reappeared on
the lead XTX, which now waited on the device for every MI50 in every one of 39 layers, including MI50s that had nothing
to do that layer. A sweep that added 0 / 50 / 100 / 200 µs to each helper's answer moved the step by about 1 ms per
50 µs: the lead was directly exposed to the helpers. Skipping the empty helpers ([HX-18](cards/HX-18.md)) removed 83 %
of the per-layer waits at one stream and gave back almost nothing there (B8 improved 5.6 %). The overlap change before
it ([HX-10](cards/HX-10.md)) showed the same pattern: 1.8–3.1 % at one stream, while the helper-wait counter barely
moved. Each step was exact and kept, but each prediction was wrong on size.

The next set of experiments ([HX-20](cards/HX-20.md)) was designed to find what actually sets the single-stream step. Its
predictions and decision rules were committed before the first run. Two predictions held. The CPU lane is not on the
critical path, so the fix aimed at it was dropped. Each extra dependent kernel on the lead costs about 2.7 µs, so the
~1,463 kernels in a token account for roughly a quarter of it. The three "what if this were free" oracles all missed.
Free helpers and free MI50 maths saved about 0.7 ms each, more than predicted. An empty all-reduce made the token
0.47 ms slower. That one was later explained: the oracle left each card with only its own partial sum, so the lead routed
from wrong values and picked experts outside the VRAM-resident set, tripling the helper traffic (44 → 130 helper experts
a token). An oracle that corrupts values also changes routing, and on this engine routing decides which hardware does
the work. A device-timestamp timeline on one clock showed the lead XTX arriving last at every MoE all-reduce, with the
other three waiting about 1 ms a token for it. Those results chose the build-12 candidates.

**Lead-side cuts moved into the helper wait.** The lead-path A/B ([HX-21](cards/HX-21.md)) was predicted at −0.55 to
−1.00 ms and measured −0.27. Of its five changes, only the one on the MI50 side, which makes the helpers' results
arrive earlier, paid its predicted size (−0.20 ms; the lead's join wait fell from 19.5 to 13.2 µs a layer). Turning off
the lookahead GEMV did nothing, as a desk reading of the build-11 code had said and the run script had not. The rewritten
routing kernel did nothing either: the pre-publish segment it targeted did not shrink at all, so the 14 µs it was priced
at came from a traced run and did not carry over. Joining and combining in one launch saved nothing measurable. Our
reading is the HX-17 lesson again: in the layers where a helper has a pick the lead reaches its join before the latest
MI50, and a cut on the lead before or at the join only makes it start waiting sooner.

**Per-launch prices are not uniform.** The attention merge saved about 3.8 µs per removed launch. Folding the
hyper-connection post step into the all-reduce saved about 1.3 µs per launch with both of its parts, and on its own it
made the token 0.06 ms slower, its pre-registered disproof: the fused sum kernel's longer tail cost more than the launch
([FG-07](cards/FG-07.md)).

**Faster kernels that were slower.** MI50 expert kernels rewritten to run three waves a SIMD instead of two, with every
lane's arithmetic unchanged, were predicted to save 0.05-0.15 ms on top of the other changes and cost 0.33 ms instead
([HX-23](cards/HX-23.md)). The output was exact. Reading the code objects of the binary that ran (not timed) explains it:
occupancy rose as designed, but the old kernels were limited by their grid, at about 1.6 waves a SIMD, not by registers.
To fit the smaller register budget the compiler serialised the codebook lookups and cut the loads in flight from 16-24
to 4-8, and the re-tile made the grid six times bigger without taking work off the busiest SIMD. The estimate, +8 to
+12 µs per helper window with a pick, is consistent with the measured +0.33 ms. The plan's first GPU step was a
single-card kernel micro-bench; it was skipped, and the kernels met a GPU for the first time in an engine timing run.
Kernel changes now run their micro-bench first. MK stays off: an exact re-tile tops out at about −0.1 ms.

**An identity failure that was not a bug.** The first merged build-12 binary differed from build 11 from token 22 with
every new flag off. The new MI50 kernels had also been emitted, empty, into the XTX code object, which pushed it over a
2 MiB mapping boundary; every XTX lost 2 MiB of free VRAM, one routed expert fewer fitted, and that expert's work moved
to a helper with a different summation order ([QA-08](cards/QA-08.md)). The fix was a two-line `#if`. On this engine,
code size and context length are part of the exactness contract.

**A production headline that was an artefact.** The build-12 production phase printed −11 % for an adaptive drafting
policy. The sampled serve runs produced different output lengths from rep to rep, all three of that arm's reps were
short, and short outputs run faster per token. At matched length it equals build 12 ([SV-09](cards/SV-09.md)). Only
length-matched figures are reported.

**Stale baselines.** Peer-to-peer decode all-reduce was predicted at +20–30 % and measured −4 % and −11 %
([AR-03](cards/AR-03.md)). The prediction used a profile from a build in which the all-reduce was a larger part of the
token. A second attempt ([AR-06](cards/AR-06.md)) and two later all-reduce levers ([AR-07](cards/AR-07.md)) also missed.
One of those was not exact with the drafter on, which the identity gate caught.

**A faster approximation that failed quality.** Deferring helper experts by one layer ([HX-11](cards/HX-11.md))
cut the eight-stream step by 27 %. Its teacher-forced perplexity rose 11.7 %, and it failed. The partial versions that
passed quality were slower than not deferring at all.

**Pinning the wrong thing.** Forcing the MI50 PCIe link to its top speed ([CP-02](cards/CP-02.md)) cost 10 %, because
the manual DPM mode needed to pin the link also froze the core clock low. Setting DPM to `high` instead
([CP-03](cards/CP-03.md)) gave +4.5 % with identical output. A manual top-level clock pin lost to `high` at eight
streams ([CP-04](cards/CP-04.md)).

**Smaller items.** Lookahead prediction of helper experts fell below its recall bar ([HX-08](cards/HX-08.md)). A
hipGraph probe found 0.35–0.88 µs saved per kernel, borderline against its 0.5 µs line, and the graph track was closed
([FG-03](cards/FG-03.md)). Faster MI50 kernels were not built once the breakdown showed they were a small share of the
helper window ([HX-15](cards/HX-15.md)). Several changes were written but never measured; each has a "not run" card (for
example [TP-06](cards/TP-06.md), [FG-06](cards/FG-06.md), [PF-08](cards/PF-08.md), [SV-04](cards/SV-04.md)).

*Update, 13 Nov.* **Kernel wins that stayed on the bench.** After HX-23 every kernel change went to a single-card bench
first, and that order paid for itself: four of seven XTX candidates and every MI50 multi-row candidate failed there, in
minutes, without engine time ([FG-08](cards/FG-08.md), [HX-25](cards/HX-25.md)). The bench also misled twice. A faster
XTX down-projection kernel beat its own prediction 3.9 × on the bench (−20 %) and then did nothing in the engine, because
that kernel is not on the token's critical path ([FG-13](cards/FG-13.md)). A "faster" gate/up kernel had been compared
with a baseline the compiler built differently from the engine's; against the engine's real kernel it was slower in
every cell ([FG-14](cards/FG-14.md)). Benches now check that the old kernel's machine code matches the engine's. A
split indexer kernel halved the call at short context and lost at long context; dispatched by length, it took
0.135 ms off a one-row token in the engine with identical output, more than predicted; it is in build 16
([FG-12](cards/FG-12.md)).

**A trace that overstated the gaps.** A kernel trace of one-row decode on build 15 put 26 % of the layer time in gaps
between kernels ([FG-15](cards/FG-15.md)). Read kernel by kernel, most of that is the tracer itself, which adds
3.6-3.8 µs at every boundary; the real gap is about 0.1-1.1 µs. Fusing kernels is worth about 3-4 % a token, not a
quarter ([FG-16](cards/FG-16.md)). The same work showed that 88 % of a one-row step (11.07 ms) is a fixed cost every
user shares, and that the host is done launching after 3.7 ms of a 12.6 ms step.

**A first kernel aimed at the wrong bottleneck.** The first tiled kernel for the MI50s' prompt work assumed the work was
bound by re-reading weights and came out 32 % slower; the sets are bound by arithmetic and activation traffic. The
third version, built on what the first two measured, is the one that took 15 % off the time to first token
([PF-12](cards/PF-12.md)).

**Levers that were not there.** A higher MI50 power cap helped long prompts by 2.4-2.8 % and nothing else, so the caps
stay at 135 W ([CP-05](cards/CP-05.md)). The four XTX turned out to be evenly enough loaded that a weighted re-split
could gain at most 1.4 % ([TP-08](cards/TP-08.md)). Drafting from the drafter's own distribution was 13 % slower, and
the adaptive draft rule at its design point did not help either, and the one sweep cell that beat the noise (block 5,
confidence 0.4) did not confirm on four reps ([DR-08](cards/DR-08.md)). Five desk studies (prefix
reuse, tree speculation, expert prefetch, a layer split, more work for the MI50s) found one already built, three not
viable on this layout and one that needs a measurement first ([SV-15](cards/SV-15.md), [DR-09](cards/DR-09.md),
[HX-26](cards/HX-26.md), [TP-09](cards/TP-09.md), [HX-27](cards/HX-27.md)).

## 7. What the hardware did

**An SMU stopped responding.** During a latency sweep, XTX #2's system management unit stopped responding. The OS
shutdown then hung for about four hours until the box was power-cycled by hand ([HW-05](cards/HW-05.md)). The response
was a watchdog that looks for the driver's SMU-failure messages and forces a hardware reboot, a hardware watchdog,
and remote kernel logging. Its dry run passed; its real-reboot path has not fired since.

**Reading idle power saving as a fault.** A test of whether an MI50's round trip changes at Gen3 found every link reading
2.5 GT/s before, during and after retraining. It was first reported as a link stuck at Gen1 ([HW-06](cards/HW-06.md)).
After a full power cycle, all four MI50 links read 2.5 GT/s at idle. `pp_dpm_pcie` lists two levels, 2.5 GT/s and
16 GT/s, and the card's SMU drops the link when idle and raises it under load. Nothing was broken: the method read link
speed on an idle card. The Gen3 question is still unmeasured. A later reading that the links also dropped to Gen1 under
load was itself a reading of idle moments between runs: under decode load at `high` the links hold Gen4, and pinning
them changed nothing ([HW-06](cards/HW-06.md) addendum).

**Power delivery.** A power-cable splitter feeding XTX cards burned. The cards moved to direct PSU cables, and the box
later got new PSUs ([HW-03](cards/HW-03.md)). A fourth XTX's junction went from 31 °C to 92 °C in two seconds at 257 W
while its edge sensor read 46 °C. The card was pulled, and a different board in the same slot ran 900 s at an 82 °C
peak ([HW-02](cards/HW-02.md)). Early gate runs at 89–98 °C were discarded ([HW-01](cards/HW-01.md)).

**Four boards, four sets of limits.** The four XTX cards come from three vendors. Each power cap is 90 % of that board's
own default, so the card with the lowest default and the lowest top clock point runs 2–3 % slower than the rest
([HW-08](cards/HW-08.md)). In tensor parallel the slowest member sets the pace at each all-reduce. Under a sustained agent
workload one card, XTX #3, ran 10–20 °C hotter than the others at the same power, because it is the one XTX whose
thermal paste has not been replaced. Reading a 245,840-token prompt pushed it to 92 °C, and the thermal watcher stopped
the server, as designed. Long-prompt prefill is burstier than decode, and it found the weakest card first. For the
long-context tests the XTX stop line was raised to 95 °C (the MI50 line stays at 80 °C); the same 240K prompts then
peaked at 93 °C on that card and completed. The card is due to be repasted or replaced; until then it is the box's
thermal limit on long prompts. Short-context GPU runs stay well below that: the bisect runs behind
[QA-08](cards/QA-08.md) peaked at 76 °C on an XTX.

**Clocks matter more than links for helpers.** At idle the MI50s sat at 938 MHz core and 350 MHz memory. Each helper
request is short, so the clocks never ramped fully ([CP-03](cards/CP-03.md)). A Gen4 x4 link cost nothing measurable
against x16 for helper traffic ([HX-01](cards/HX-01.md)).

*Update, 13 Nov.* **XTX #3 was its paste.** Re-pasted on 12 Nov, it peaked at 69 °C junction under a ~270 W load, in
line with the other cards, and is no longer the thermal limit on long prompts ([HW-10](cards/HW-10.md)). The same work
left its slot at Gen4 x8 for four and a half hours (a disturbed riser); a before-and-after probe found no measurable
difference in decode or 8K prompts, so the numbers from that window stand.

**A power controller that took the box down.** XTX #2's SMU hung a second time on 12 Nov, and the watchdog rebooted the
host as designed. On 13 Nov it hung a third time with different wording that the watchdog did not match, and seconds
later the host lost power. The card was removed; two replacement boards were burn-tested in its slot and the one with a
~28 °C junction-to-edge gap was kept. XTX #4's firmware switch was moved during the work, and it now runs its second
position with a higher default power limit ([HW-09](cards/HW-09.md)). Why an SMU hang should end in a power loss is not
known.

**A bench card on a shared watcher.** The fifth MI50 sits on a long riser at Gen1 x8 and is used only for kernel
benches. Under sustained benches it reached 81 °C, and the main queue's thermal watcher, which correctly covers every
MI50, stopped two main jobs before the card got its own fan curve ([HW-10](cards/HW-10.md)).

## 8. Quality: what was checked and what was not

- **Exact changes** were checked by byte identity, as above. All of build 11's changes are exact; its
  output is byte-identical to build 10 ([SV-06](cards/SV-06.md)). Build 12's are exact too: with its flags off and on,
  greedy decode, drafter decode and batched decode are byte-identical to build 11 ([SV-09](cards/SV-09.md)).
- **What byte identity depends on.** The output depends on which routed experts fit in XTX VRAM, because an expert
  that moves to an MI50 is summed in a different order. Free VRAM at load is set by, among other things, the code size
  and the context length, so both are part of the identity contract: a build whose code object crossed a 2 MiB boundary
  differed from token 22 with no numeric bug anywhere ([QA-08](cards/QA-08.md)). Identity comparisons are made at the
  same placement, and the runs print it.
- **The MI50 helper itself** passed a teacher-forced check against XTX-alone: 96.94 % top-1 agreement, median KL
  4.5e-4 ([QA-01](cards/QA-01.md)).
- **The "GATE FAIL" lines.** The original quality line required median KL ≤ 1e-3. Any change in summation order (tensor
  parallel, FP16 all-reduce, a tuned GEMM) puts this engine at about 1.6–1.8e-3 against a single-group reference. Three
  XTX in tensor parallel, otherwise unchanged, measure 1.79e-3 ([QA-02](cards/QA-02.md)). Summaries still print
  "GATE FAIL" on that line. We use paired intervals against a common reference instead, and left the printed line as it
  is rather than moving it after the fact.
- **Build 10** passed the paired test: 96.28 % top-1 against a fresh reference, perplexity 3.4014 against 3.3968, every
  paired interval including zero ([QA-03](cards/QA-03.md)). The reference is the same 2.25 bpw pack run as one
  layer-split group, so this shows that build 10 stayed close to the pack. It does not show how close the pack is to
  the full-precision model. That comparison against a Q8 reference has not been scored ([QA-04](cards/QA-04.md)).
- **MTP on Flash Next** matches arm a exactly on the teacher-forced prompt path, but its decoded output differs from
  arm a on 7 of 8 prompts. Decoded quality with MTP is untested ([FN-03](cards/FN-03.md)).
- **Task-level comparison: started, not scored.** An agent task suite is being run against the deployed server, first at
  a 131,072-token context and then at 262,144 and 524,288 ([QA-05](cards/QA-05.md)). No results are published yet. The
  serving side of the first part is in [SV-08](cards/SV-08.md). All the quality evidence scored so far is token-level.
- **Long-context retrieval.** At a 262,144-token context a passphrase placed at 10 %, 50 % and 90 % depth of a
  ~240,000-token prompt was returned exactly in all three cases ([QA-09](cards/QA-09.md)). The prompts read cold at
  ~1,590 tok/s and decode with the drafter ran at 77-80 tok/s after them. The model's native context is 1,048,576 tokens;
  the KV cache costs 920 B a token on each XTX and is paid for in VRAM expert slots (48 fewer per card at 256K than at
  128K), so memory does not limit context on this box. This is a basic retrieval check, not a quality score. The fuller
  check (needles to 512K and 1M, perplexity on a fixed tail at 123K, 254K and 516K of depth, with a pre-registered pass
  line) is pending.
- **Open items.** One decode in nine produced repeated tokens during the drafter work, unexplained
  ([QA-06](cards/QA-06.md)). The drafter settings themselves were never quality-gated ([DR-01](cards/DR-01.md)). A
  "DEGENERATE" flag in the batched-versus-solo checker turned out to be a repetition heuristic. The flagged sequences
  are identical to solo and have the same score there ([QA-07](cards/QA-07.md)).
- *Update, 13 Nov.* **The prompt helper is now served, and its validation is still open.** From build 13s the served
  configuration turns on the MI50 prompt helper for its time-to-first-token gain ([PF-11](cards/PF-11.md)). Its
  validation run is formally inconclusive: the pre-registered Q8 reference could only be partly produced, because a Q8
  pack does not fit in host RAM ([QA-04](cards/QA-04.md)), and a review found that the harness's bootstrap intervals were
  about 1.5 × too narrow. Under concurrency the helper also makes batched greedy output differ from solo greedy output at
  near-ties (5 of 8 identical; 8 of 8 with it off) ([SV-10](cards/SV-10.md)). Comparing decoded text does not settle it:
  with sampling, even an exact repeat diverges within a few words. Only the task-level comparison
  ([QA-05](cards/QA-05.md)) can say whether it matters. Every other change of 12-13 Nov is exact and was gated byte for
  byte, and the prompt tiles of build 15 are bit-identical to the helper's own path ([PF-12](cards/PF-12.md)).
- *Update, 13 Nov.* **Long context.** At a 1M-token setting, 11 of 12 passphrases were found up to 512K; the twelfth and
  the 1M tier were cut off by thermal stops of the old XTX #3. Decode at 512K of context costs 15.8 ms a token against
  14.6 ms at 1K, and reserving a large context costs nothing measurable at short prompts ([QA-09](cards/QA-09.md)).

## 9. How this was built

Most of the engineering was done by AI coding agents. A person directed the work, made the decisions and reviewed the
results. Every number in the cards comes from scripted, repeatable runs whose scripts ship in the repository; the data
extracts are copied from those runs' own summary files (two, marked as such, are hand-transcribed). The same identity
and quality gates applied to every change.

*Update, 13 Nov.* For two days after build 12, builds were made from working trees and recorded only as diffs and
checksums. On 13 Nov every build since build 12 was committed with a tag carrying its benchmarked binary's checksum, and
each tag was rebuilt on another machine and compared: all nine rebuild byte for byte, five of them once the original
version-string build id is supplied ([ME-10](cards/ME-10.md)). Builds are now committed and tagged before they compile.
A measurement also showed that a build pinned to two spare cores at idle priority does not disturb a timing run, so such
builds may now run during timing ([ME-11](cards/ME-11.md)).

## 10. What's next

- **Public re-runs** of the eleven headline cards on a public benchmark text, about 395 GPU-minutes in all (table in
  [README.md](README.md)).
- **The prompt's fixed cost.** Every prompt chunk of ≥ ~700 new tokens costs ~3.3 s before its first token, because
  each XTX streams 21 GB of host-RAM experts at ~6 GB/s; that floor is 58 % of prompt time under agent traffic
  ([PF-10](cards/PF-10.md)). The MI50s hold those experts and sit idle meanwhile. A change that lets them compute them
  in prefill cut a 1K-token suffix from 3.42 to 1.33 s ([PF-11](cards/PF-11.md)). It is the first change here that is not
  bit-identical, so it stays opt-in until a validation against a higher-precision reference, at several depths and on
  task outcomes, is done.
- **Verify rows.** Each extra verify row costs 3.11 ms, about a quarter of a production round, and most of it is the
  row's routed experts and MI50 picks ([DR-06](cards/DR-06.md)). It is the largest production lever left (about +8 % if a
  row fell to 2.0 ms) and has no design yet.
- **The next exact fusions on the lead** and a short measurement of graph replay's per-kernel cost, which decides
  between graphs and a persistent kernel ([ANATOMY.md](ANATOMY.md) §5). The MI50 expert kernels are not on this list: an
  exact re-tile tops out at about −0.1 ms ([HX-23](cards/HX-23.md)).
- **A full single-stream time budget** that sums to the step within about 5 %, at short and long context. The coarse
  budget closes; the fine one still needs the stage-stamp run ([ANATOMY.md](ANATOMY.md) §2.4).
- **A code-size check after every build**, so expert placement cannot move silently ([QA-08](cards/QA-08.md)).
- **The fuller long-context check** ([QA-09](cards/QA-09.md)), then the choice of serving context.
- **A Q8 reference** for the quality comparisons ([QA-04](cards/QA-04.md)) and a task-level comparison
  ([QA-05](cards/QA-05.md)).
- **XTX #4's prefill cost:** move the card to separate link width from topology ([PF-09](cards/PF-09.md)).
- **The Gen3 question,** measured under load this time ([HW-06](cards/HW-06.md)).
- **Decoded-output quality for MTP** ([FN-03](cards/FN-03.md)) and a drafter quality gate
  ([DR-01](cards/DR-01.md)).
- **Continuous batching in the server**, which is written but unmeasured ([SV-04](cards/SV-04.md)).

*Update, 13 Nov.* Done since: the prompt's fixed cost was cut twice (the prompt helper, then bit-exact tiles,
[PF-12](cards/PF-12.md)); concurrent serving was measured ([SV-11](cards/SV-11.md)); the code-size check runs on every
build ([QA-08](cards/QA-08.md)); XTX #3 was re-pasted. Next:

- **Done since (13-14 Nov):** build 15's identity phase and served benchmark ([SV-14](cards/SV-14.md)); the
  like-for-like run, both halves ([SV-13](cards/SV-13.md)); the drafter policy confirm run (not confirmed,
  [DR-08](cards/DR-08.md)); the indexer dispatch's engine A/B ([FG-12](cards/FG-12.md)); the expert-cache leak test
  (confirmed, [HX-26](cards/HX-26.md)).
- **The prompt helper on long served prompts:** use it only up to a prompt length, then find why the served path is
  slow ([PF-13](cards/PF-13.md)); fix the expert-cache leak ([HX-26](cards/HX-26.md)). Both are in build 16 ([SV-16](cards/SV-16.md)).
- **Decode fusions,** the batched attention path first ([FG-16](cards/FG-16.md)); that item passed on 14 Nov (identical output, +6.4 % with 8 served users) and is in test for the next build ([UPDATE-2026-11-14.md](UPDATE-2026-11-14.md)). Untraced timestamps put the real gap
  between kernels at about 2.9 µs, so fewer launches could be worth up to about 3 ms a token, and the MI50 helper chain is
  on the critical path in 98 % of one-row layers ([FG-17](cards/FG-17.md)); two attempts to batch its decode window were exact but slower with two or more live experts, so not shipped. The MI50 links do not drop under decode load
  (closed, [HW-06](cards/HW-06.md) addendum).
- **Prompts for several users:** they are read one at a time; reading them concurrently is the largest open multi-user
  lever ([SV-11](cards/SV-11.md), [TP-09](cards/TP-09.md)).
- **The prompt-read hangs:** the self-reporting build is in use, so the next one names its site ([SV-12](cards/SV-12.md)).
- **The prompt helper's validation**, now that it is served: a usable higher-precision reference and task-level scores
  ([PF-11](cards/PF-11.md), [QA-05](cards/QA-05.md)).
- **Verify rows** (2.66-2.70 ms each, [DR-06](cards/DR-06.md)). Tree speculation is closed: the drafter's second choice
  rescues only about a third of first-draft misses ([DR-09](cards/DR-09.md)). A layer split loses for prompts and
  decode ([TP-09](cards/TP-09.md)).
- **Done on 14 Nov, in build 16 (the build in this release):** fusion items 1 and 3 and the indexer dispatch, which
  hold +0.15 to +0.17 ms each inside build 16 (all five changes −0.44 ms a one-row token, MEASURED, [FG-16](cards/FG-16.md)); the helper prompt-length cap
  ([PF-13](cards/PF-13.md)); the cache-leak fix ([HX-26](cards/HX-26.md)), which also ends a double count of expert
  picks ([HX-28](cards/HX-28.md)); four upstream correctness fixes, output identical ([SV-14](cards/SV-14.md));
  build 16's re-validation ([SV-16](cards/SV-16.md)). Also on 14 Nov: a warm-turn run, reuse 96 % with a 32K turn at 2.1 s, and reuse is exact at fixed placement ([SV-15](cards/SV-15.md)).
- **14 Nov, the rest of the day:** served build 15 vs 16 on the same day, why one user got slower, a fix for that which is
  measured but not in this release, the batched fusion, and the full card grid: [UPDATE-2026-11-14.md](UPDATE-2026-11-14.md).
- **15 Nov:** build 17 (+12.5 % with 8 users) and swap instead of evict (−22.7 % on a long-running server's first
  token), both measured, neither in this release: [UPDATE-2026-11-15.md](UPDATE-2026-11-15.md).

## Acknowledgements

Thanks first to Niko1221 and the Strata contributors, whose engine all of this builds on. Licence notices and credits
for ported code are in [NOTICE.md](NOTICE.md).

Thanks to the MI50 Discord community for discussion and help.
