# Peer-to-peer (P2P) on gfx906 (MI50) and gfx1100 (RX 7900 XTX): what works, what goes stale, and why

This page collects what we learned about GPU peer-to-peer traffic on the test machine: four RX 7900 XTX (gfx1100) and
four Instinct MI50 (gfx906) on one EPYC Zen 2 host, ROCm 7.14. The cards are [HW-12](cards/HW-12.md) (the stale-read
cause) and the older all-reduce cards [AR-03](cards/AR-03.md), [AR-04](cards/AR-04.md) and [AR-06](cards/AR-06.md).

Labels as in the cards: **MEASURED** (from a run file or summary), **REPORTED** (written at the time, not re-checked),
**INFERRED** (our reading), **ESTIMATE** (arithmetic or a forecast). "Community reports"
means other people's results that we have not re-measured.

**Short version for gfx906 users:** writes into an MI50 from any peer, and writes by an MI50 into any peer, were exact on
every pair we tried. **Reads of MI50 memory by a peer return stale data when the MI50's own SDMA copy engine wrote that
memory**, until the MI50's HDP is flushed. ROCm uses SDMA for host → device copies over 16 KiB. Push, do not pull; or
produce the data with a kernel; or flush the owner's HDP before a peer reads.

## 1. How this engine uses P2P today

- **Prompts:** the prompt-chunk all-reduce among the four XTX runs VRAM to VRAM by kernel stores (`DS4_AR_P2PK`, default
  since build 9, [AR-04](cards/AR-04.md)). It needs three or more members; with fewer it keeps the host-mapped path
  (MEASURED, engine log).
- **Decode: no P2P in the release build.** All 80 decode all-reduces a step (2 a layer × 40 layers) exchange partials
  through pinned, host-mapped memory, at about 32 µs each after the last card arrives (MEASURED, kernel trace). The MI50
  helpers' requests and results also go through host memory.
- **Where decode waits:** the lead XTX alone joins the MI50 helpers' and the CPU lane's results. At the MoE all-reduce it
  arrives last 97–98 % of the time, and the other three XTX wait 13–17 ms a step for it (MEASURED, traced, one-user
  native MXFP4 serving). That wait is what the push-only all-reduce and fan-out (section 4) target.
- The push-only decode all-reduce (section 4) is built and tested on an internal branch; it is **not in this release**:
  it was correct but slower than the host path (section 5).

## 2. Platform checks: is P2P allowed and wired?

All of these were checked on the test machine, and the kernel items on a second machine too (MEASURED unless marked):

| check | result |
|---|---|
| Large BAR | amdgpu resizes every card's BAR0 to 32 GiB at load (the BIOS leaves 256 MiB); all VRAM is visible |
| BAR addresses vs the MI50's 44-bit (16 TiB) DMA limit | highest BAR about 4 TiB, far below; `dma_mask_bits = 44` on every card |
| Kernel configuration | `CONFIG_PCI_P2PDMA`, `CONFIG_HSA_AMD_P2P`, `CONFIG_DMABUF_MOVE_NOTIFY` set; in-kernel amdgpu (no DKMS), `amdgpu.pcie_p2p=Y`; kernels 7.0.6 and 6.17.2 |
| Host bridge | the kernel's P2P DMA whitelist accepts any AMD CPU family ≥ 0x17, so EPYC Zen 2 needs no patch (INFERRED from the upstream 7.0 source) |
| IOMMU | `iommu=pt` (identity domains) |
| ACS / topology | no PCIe switch anywhere; the MI50s sit on separate CPU root ports (bifurcated, Gen4 x4 each); three XTX at Gen4 x16 on their own root complexes, one at x8. The kernel classes every pair as "through host bridge" |
| Atomics | the GPUs request AtomicOps, the root ports do not offer them; KFD's peer links are marked non-coherent. So: plain stores and store-release, never peer read-modify-write (INFERRED) |
| KFD peer links | published for all 56 ordered pairs, MI50s included; `hipDeviceCanAccessPeer` yes on 56 of 56 |
| MI50 VBIOS | all cards carry the same image (113-D1640200-043, the Radeon Pro V420 image). Community reports mention P2P trouble with that image; ours does P2P (writes exact, reads exact after an HDP flush) |

**Verdict at the time:** nothing in the platform blocked MI50 P2P; the open risk was "enabled but wrong", which only a
byte-exact probe could settle.

**For other setups:** two recipes circulate in the community. One is for Intel hosts whose PCIe bridge is missing from the
kernel's P2P whitelist. The other is for MI50 BARs placed above 16 TiB (above the card's 44-bit limit), which makes peer
access crash. The fixes there are a whitelist patch and lowering the BIOS "MMIO high base". Behind PCIe switches, the
fix is a module that enlarges the bridge windows for resizable BAR. None of these applied to this machine, and none of
them touches the stale reads below.

## 3. MI50 peer reads go stale after SDMA writes: the HDP

Full card: [HW-12](cards/HW-12.md).

### 3.1 What happens
1. When an MI50's **SDMA copy engine** writes into the MI50's own VRAM, a **later read of that memory by another GPU**
   (XTX or MI50) can return the **old contents**: typically exactly what the buffer held on the previous round.
2. **The MI50 itself sees the new data:** a copy issued on the MI50 from the same buffer is exact.
3. **ROCm picks SDMA by size.** Host → device copies larger than 16 KiB go through SDMA (`GPU_FORCE_BLIT_COPY_SIZE`,
   default 16 KB, in ROCm's runtime source). Copies of 16 KiB and below use a blit kernel. That is why small buffers looked
   fine and 32 KiB and up went stale.
4. **These read exact** (MEASURED): data written by a kernel (a blit is a kernel too), and SDMA-written data after the host
   writes the MI50's HDP flush register (ROCr gives its address as `HSA_AMD_AGENT_INFO_HDP_FLUSH`).
5. **Waiting does not help:** 1 ms and 20 ms between the write and the read leave it stale (MEASURED).

### 3.2 How it was found: five steps, each narrowing it
| step | what | result (MEASURED) |
|---|---|---|
| 1. first byte-exact probe, all ordered pairs (16 Nov) | copy and kernel reads and writes, every rep checked | XTX ↔ XTX 12 of 12 and XTX → MI50 16 of 16 exact. The first MI50 → XTX pair: 64 KiB pulls wrong in 155 of 200 reps; 4 KiB pulls, pushes and kernel stores clean. The run stopped there. No PCIe errors, no resets |
| 2. offline check of the bad words | the first bad word against the seed space | it equals **exactly the previous rep's value** at the same word: stale data, not random corruption |
| 3. map of all 56 pairs (16 Nov) | adds uncached sources, kernel reads, a "previous value" class, and a second column with `HSA_ENABLE_SDMA=0` | every MI50-source pair (16 MI50 → XTX, 12 MI50 → MI50) stale at 64 KiB, by copy or kernel read, cached or uncached source; 4 KiB, 1 MiB and 64 MiB exact. **With SDMA off: exact on every pair** |
| 4. second machine, different board, kernel 6.17.2 (17 Nov) | one XTX + one MI50, both x16, 4–128 KiB | 4–16 KiB exact; bad reps of 200: 32 KiB ~194–199, 48 KiB 177–199, 64 KiB 136–199, 128 KiB 24–52; non-temporal reads no help (161–163 at 128 KiB); pushes exact both ways at every size |
| 5. the decisive run (17 Nov), pre-registered | two pairs (MI50 → MI50, MI50 → XTX), 11 arms, 8–128 KiB, 200 reps | below |

**The decisive run** (bad reps of 200 for copy pull / kernel read / kernel read of an uncached source; pushes were 0 bad in
every arm, so the probe itself is sound; MEASURED):

| arm | what changes | MI50 → MI50 | MI50 → XTX | reading |
|---|---|---|---|---|
| base | — | 8 / 16 KiB exact; 32 KiB 199 / 200 / 199; 64 KiB 199 / 199 / 199; 128 KiB 193 / 200 / 199 | 8 / 16 KiB exact; 32 KiB 198 / 200 / 199; 64 KiB 153 / 200 / 199; 128 KiB 44 / 40 / 46 | reproduces |
| base, repeated at the end | — | stale at 32–128 KiB | stale at 32–128 KiB | stable over the run |
| SDMA at every size (`GPU_FORCE_BLIT_COPY_SIZE=0`) | the source is SDMA-written even at 8 KiB | **8 and 16 KiB go stale too** | same | an SDMA-written source is the trigger |
| blit at every size (`GPU_FORCE_BLIT_COPY_SIZE=1048576`) | the source is blit-written | **exact at every size** | **exact at every size** | a blit-written source is safe |
| kernel-written source | a kernel on the MI50 fills the source | **exact** | **exact** | kernel-produced data is safe to pull |
| HDP flush | the host writes the MI50's HDP flush register after the write, before the read | **exact** | **exact** | an owner-side HDP flush fixes it |
| ROCr's own SDMA HDP flush off (`HSA_ENABLE_SDMA_HDP_FLUSH=0`) | removes the flush ROCr issues *before* each SDMA copy | no change | no change | that flush is not the relevant one |
| settle 1 ms / 20 ms | a wait between write and read | still stale | still stale | not a short race; persistent, cache-like |
| runtime log at 16 / 32 KiB (`AMD_LOG_LEVEL=4`) | which engine wrote the source | blit at ≤ 16 KiB, SDMA above | same | confirms the threshold |

A later check repeated the base, HDP-flush and kernel-written arms on two more MI50s and against all four XTX: base stale
at 32–128 KiB on every pair, **HDP flush and kernel-written source exact on every pair** (MEASURED).

**Shape of the bad data** (MEASURED): whole 64-byte lines (every stale-word count was a multiple of 16 words), mostly in
aligned runs of four lines (256 bytes). At 64 KiB, about 186 words (~750 B) a bad rep for copy pulls and about 660 words
(~2.6 KiB) for kernel reads.

### 3.3 Mechanism (INFERRED)
- A peer's read of MI50 VRAM arrives over PCIe through the MI50's BAR, so it passes the MI50's **HDP** (host data path),
  which has a read cache.
- An SDMA write to VRAM does not refresh lines the HDP already holds, for example lines left by the previous round's peer
  read. The next peer read hits those old lines.
- This fits the rest of the evidence. Kernel-written data reads exact; the end of a kernel presumably flushes or
  invalidates the HDP, and a blit is a kernel. amdgpu itself invalidates the HDP read cache before its own CPU reads
  through the BAR, but ROCm's user-mode queues never take that path. A host write to the HDP flush register also
  invalidates, because amdgpu sets the flush-invalidates bit at init (all from the upstream 7.0 driver source).
- Not tested: reading each round from a different offset, which would show whether the stale lines are address-cached from
  the previous read rather than in flight.

### 3.4 Why a fresh-buffer test misses it
- A test that allocates a buffer, fills it once and reads it once from a peer **passes**: the HDP holds nothing old for
  those lines (INFERRED). Reads of 16 KiB or less, or of data a kernel produced, are exact anyway. Community P2P read
  checks of this kind pass on MI50s.
- To see the bug: re-seed **the same buffer** every round with a unique pattern, through a host → device copy over 16 KiB;
  poison the destination with the complement of the expected value; sync both cards; read from a peer; classify "previous
  round's value" separately.

### 3.5 Repro
1. Allocate a buffer of 32–128 KiB on an MI50 and a destination on a peer.
2. Each round: write a fresh, unique pattern into the MI50 buffer with `hipMemcpy` host → device (over 16 KiB, so SDMA);
   synchronise both devices; have the peer read it (`hipMemcpyPeer` pull or a kernel load through the peer pointer); copy
   the result back and compare on the CPU. Treat round 0 as an uncounted warm-up.
3. Expect previous-round words from 32 KiB up. Then repeat with an HDP flush on the MI50 between write and read, with the
   pattern written by a kernel, or with `GPU_FORCE_BLIT_COPY_SIZE=1048576`; each gives 0 bad rounds.
4. Cost: the decisive run above took about 1.3 minutes of GPU time (MEASURED).

### 3.6 What was ruled out
| candidate | why it is out |
|---|---|
| a test artifact | the source is re-seeded every round, the destination poisoned and guarded, both devices synced before the read, the expected previous value recomputed from the hash; pushes from the same source are exact (MEASURED) |
| reader-side caching | MI50 readers are as stale as XTX readers; an uncached source is as stale; non-temporal loads do not help (MEASURED) |
| BAR placement / the 44-bit limit | highest BAR about 4 TiB, far under 16 TiB (MEASURED) |
| the Linux kernel or the board | reproduces on 6.17.2 on a second board as on 7.0.6 (MEASURED) |
| kernel configuration | the P2P options are set on both (MEASURED) |
| the community BAR / switch / IOMMU fixes | written for PCIe switches, Intel IOMMUs and BARs above 16 TiB, none of which this machine has; none touches read coherence (INFERRED) |
| a time-bounded race | 1 ms and 20 ms waits still stale (MEASURED) |
| ROCr's pre-copy SDMA HDP flush | turning it off changes nothing (MEASURED) |
| the MI50 VBIOS | **not strictly excluded**: every card has the same image, so there is no A/B. Nothing links a VBIOS to line-granular old data, and the HDP flush fixes it with the VBIOS unchanged (INFERRED) |

## 4. The push-only decode all-reduce (`DS4_AR_PUSH`, `DS4_HELPER_FANOUT`)

Not in this release: built and tested on an internal branch, off by default there. The code may follow in a later release.

### 4.1 Design
- **Push only.** Every card writes into the **receiver's own** uncached VRAM mailbox with kernel stores. No card ever reads
  another card's memory, which sidesteps section 3 entirely. Kernel stores into peer VRAM were exact on every pair.
- **`DS4_AR_PUSH=1`:** the two decode all-reduces a layer become a reduce-scatter plus all-gather over the mailboxes, then
  the post step. Each owner sums its quarter in member order with round-to-nearest adds, the same chain as the host path,
  so the bits are identical.
- **`DS4_HELPER_FANOUT=1`:** the lead skips its join and sends its own experts' sum and its shared-expert part as two
  separate rows. Every member rebuilds the join on its own chunk in exactly the join's order (the CPU lane's row first, then
  the needed helpers in order; if nothing was joined, nothing is added, so −0 stays −0). The helper rows are read from host
  memory, where the helpers already write them: no MI50 change and no MI50 reads. It needs `DS4_AR_PUSH`,
  `DS4_HELPER_DD`, `DS4_LP_JOINCOMB` and one tensor-parallel group.
- **Not built:** a second form in which the MI50s push quarter rows straight into each XTX's VRAM, with no host hop.

### 4.2 Robustness rules
- Every flag word is `(checksum << 32) | sequence`, written last, with release ordering after a system fence.
- A reader accepts a slot only on the exact sequence number, a matching checksum and an unchanged re-read. A sequence number
  from the future is a fault and is remembered as poison, so it is never accepted later.
- Sequence numbers start from a random per-process value, so a restart never accepts its previous run's words.
- Waits are bounded: a transport timeout of 50 ms and a fatal bound of 10 s. On a timeout every member takes the host-path
  fallback from shadow rows, which gives the same bits; each fallback is counted. After 64 fallbacks the run stays on the
  host path. A fallback that cannot finish ends the process (exit 75) before the step's result is used: it never hangs and
  never returns a silent wrong value.
- A load-time self-test (bit-exact sums, plus one injected dropped push that must fall back exactly) gates the switch.
- NaN-ness is exact; NaN payloads may differ because compilers commute operands, and are counted separately, never as wrong.

### 4.3 Test campaign
| test | where | result |
|---|---|---|
| CPU model (the same protocol source on host threads): bit-exact incl. −0 and NaN | host, 4 threads | PASS, 0 wrong (REPORTED) |
| CPU model soak | host | 200 k all-reduces through the sequence wrap: 0 wrong of 1.34e9 values, 0 fallbacks (REPORTED) |
| CPU model faults (delayed, stalled, dying, dropped, reordered, corrupt, stale, future) | host | 88 cases PASS (REPORTED) |
| CPU model fan-out / restart; ThreadSanitizer | host | PASS; ThreadSanitizer clean after one race in the test harness itself was fixed (REPORTED) |
| CPU model at 2, 3 and 4 members | host | PASS (REPORTED) |
| CPU model 10 M soak, 4 members | second machine | 0 wrong of 66.84e9 values through the sequence wrap, 0 timeouts, 0 fallbacks (REPORTED) |
| GPU faults, 2 XTX | second machine | 24 of 24 PASS (MEASURED) |
| GPU restart, 2 XTX / 2 XTX + MI50 | second machine | PASS: 48 / 77 stale words rejected (MEASURED) |
| **GPU soak, 2 XTX** | second machine | **10 M all-reduces in 465 s: 0 wrong, 0 timeouts, 0 fallbacks** (MEASURED) |
| **GPU bit-exact against the host path, 2 XTX + MI50** | second machine | **5.05 M values exact** (MEASURED) |
| GPU fan-out, 2 XTX + MI50 | second machine | 5 of 5 PASS; a helper that never answers → fatal on that call only, never a wrong row (MEASURED) |
| GPU bit-exact, 2 XTX | second machine | no comparison: the host reference needs three members (MEASURED) |
| GPU faults, 2 XTX + MI50 | second machine | 34 of 36: two reordered MI50 pushes timed out and fell back, output still exact. Cause: the test converted its fault delays with the wrong card's clock rate. Fixed in the test (the engine converts per card); not re-run on an MI50 yet (MEASURED) |
| CPU model on the release-4 tree, all modes, quick form | host | PASS, 0 failed (MEASURED) |

Two protocol bugs were found by the CPU model before any GPU run, and fixed: a flag carrying a faked future sequence number
was accepted one parity cycle later and gave wrong data (fixed with poison words), and a spurious give-up at sequence 0
across the wrap.

**On the four production XTX** (the gate before the served run): all nine tests PASS (GPU bit-exact, faults, fan-out,
restart and the 10 M soak; CPU model a, c, d, e) (MEASURED).

## 5. Speed

**As built, pushing loses to host staging on this hardware.** Both measurements say so (MEASURED, 17–18 Nov, test machine):

**Per call** (four XTX, model dim 5120, host-timed p50 / p99 µs, 2,000 calls after 200 untimed):

| rows a call | host path | push | push + fan-out |
|---|---|---|---|
| 1 (20 KiB a member) | 137.8 / 201.3 | 167.9 / 247.1 | 193.2 / 213.8 |
| 4 (80 KiB a member) | 125.6 / 197.9 | 185.6 / 234.7 | 234.9 / 263.0 |

Push is 30 µs a call slower at 1 row and 60 µs at 4 rows. Over 80 all-reduces a step that is +2.4 to +4.8 ms, where the
pre-registered gain was −0.2 to −0.7 ms (2.5–8.75 µs a call faster). The fan-out's extra host-row reads add a further 49 µs
a call at 4 rows in this bench (which does not contain the join the fan-out removes).

**Served A/B** (2.25-bit pack, build 17's served settings; build 17, switches off, push, push + fan-out, build 17 again):

| arm | 1 user 2K, ms a token | 1 user 8K, ms a token | verify step, ms | 4 users, tok/s |
|---|---|---|---|---|
| build 17 | 8.45 | 8.93 | 23.53 | 160.6 |
| switches off | 8.45 | 8.95 | 23.55 | 160.2 |
| push | 9.47 | 11.02 | **27.06** | 139.8 |
| push + fan-out | 9.38 | 10.92 | **26.82** | 139.1 |
| build 17 again | 8.42 | 9.36 | 23.63 | 129.4 |

- Push + fan-out against switches off: verify step +3.3 ms, 1 user 2K ×1.11, 8K ×1.22, 4 users ×0.87. Push alone: verify
  step +3.5 ms. Every speed prediction is out, in the wrong direction.
- **Correctness stayed clean:** about 131,000 pushes in each push arm with 0 timeouts, 0 fallbacks, 0 sequence or checksum
  mismatches and 0 fatals; tokens a step unchanged (2.78).
- **Caveat:** the formal verdict is INCONCLUSIVE, because the 4-user baseline drifted (build 17 at 160.6 tok/s at the start,
  129.4 at the end). The one-user cells and the verify step are consistent across both baseline arms (drift 0.4 %), and the
  per-call bench agrees with them.

**Why it loses is open** (section 7). The transport saving it was built for is a few microseconds a call; what it adds
(the reduce-scatter and all-gather phases, flag polling over PCIe and the checks) costs more than that at these message
sizes (INFERRED from the bench's device times: push 86–104 µs on the GPU against the host path's 39–45 µs).

**Raw P2P transport numbers** (MEASURED, the 56-pair map on the test machine; 64 MiB unless stated):

| path | copy push GB/s | copy pull GB/s | staged via host GB/s | kernel store GB/s (coarse / uncached) | flag one-way µs | 4 KiB copy push p50 µs |
|---|---|---|---|---|---|---|
| XTX x16 ↔ XTX x16 | 26.4 | 8.85–8.96 | 13.37 | 26.3 / 26.3 | 4.50–4.55 | 33.7–35.7 |
| to / from the x8 XTX | 13.89 | 8.4–9.5 | 9.1–9.2 | 13.2 / 13.2 | 4.48–4.57 | 33.3–35.3 |
| XTX → MI50 | 6.97–7.04 | 7.12 | 4.71–5.64 | 6.61 / 6.60 | 5.83–6.04 | 33.1–35.3 |
| MI50 → XTX | 7.13–7.14 | 6.46–6.54 | 4.69–5.62 | 6.15–6.23 / 6.66–6.75 | 5.27–5.64 | 17.7–18.2 |
| MI50 → MI50 | 7.13–7.14 | 6.52–6.63 | 3.56 | 6.15–6.22 / 6.68–6.75 | 6.03–6.13 | 17.5–18.2 |

- The MI50-source pulls were exact at 64 MiB, so their bandwidth is valid; at 64 KiB they are not (section 3).
- The ~34 µs p50 of a 4 KiB `hipMemcpyPeerAsync` issued by an XTX is API and queue overhead (INFERRED). For small hand-offs,
  kernel stores plus a 4.5–6 µs flag are the fast path; that is what the push designs use.
- Community reports for MI50 pairs give about 7.1 GB/s with P2P enabled against about 4.1 GB/s without, in line with the
  push and staged numbers above; a Gen4 x4 link tops out around there (INFERRED).

## 6. What did not work
| when | what | outcome |
|---|---|---|
| ≤ 9 Nov | a peer-to-peer decode all-reduce ([AR-03](cards/AR-03.md)), then a second attempt with the prompt all-reduce's design applied to decode windows ([AR-06](cards/AR-06.md)) | NO-GO twice: −4 % and −11 % (REPORTED). The decode all-reduce transfer is about 0.3 ms a token and small messages are latency-bound, so a one-for-one P2P replacement of the transport lost |
| 13 Nov | a prompt-read hang ([SV-12](cards/SV-12.md)) | leading hypothesis (INFERRED): a prompt all-reduce wait that set an error word after many polls but was configured never to give up. A diagnostic build that does give up named the crash; a 16-prompt smoke test had 0 hangs and 0 false triggers (MEASURED) |
| 17 Nov | non-temporal peer reads of MI50 memory | no help: at 128 KiB 161 bad rounds of 200 against 46 for plain kernel reads (MEASURED) |
| 17 Nov | a push-only request inbox for the MI50 helpers | built, never run; estimated 25–40 µs off each helper window (ESTIMATE) |
| — | two tensor-parallel pairs of XTX instead of one group of four | staged, never run; predicted to lose everywhere because decode is not pipelined across groups and the two-member all-reduces stay on the host path (ESTIMATE) |

## 7. Open questions
1. **Why the push all-reduce is slower than the host path** (section 5): the push kernel itself takes about twice the host
   path's GPU time a call. Whether a leaner kernel (fewer phases, a persistent kernel, no per-slot re-read) could win is
   not known. The lead's join, which the fan-out removes, did not show up as a gain either.
2. **The cost of an HDP flush** was not measured; it matters only if SDMA-written MI50 data is ever pulled. A timed flush arm
   at 4 KiB would take a minute or two of GPU time (ESTIMATE).
3. **Why the staleness depends on size** is not explained: 1 MiB and 64 MiB read exact, and at 128 KiB the MI50 → XTX
   bad-round fraction falls (44 of 200) while MI50 → MI50 stays high (193 of 200). The offset-rotation test in 3.3 would
   show whether it is address caching.
4. **The MI50 fault test with the fixed clock conversion** has not been re-run on an MI50.
5. **The drafter's six all-reduces a drafting step** still use an older host path; moving them would save about 0.05–0.08
   ms a step (ESTIMATE). Copying the drafter's state from the lead to the other XTX by P2P instead of via the host could
   save 0.3–0.7 ms a step (ESTIMATE); it is all XTX → XTX stores, safe on today's evidence.
6. **The fan-out's second form** (MI50s pushing straight into each XTX) is designed, not built.
7. **Not checked:** ROCm 7.14's exact default for `GPU_FORCE_BLIT_COPY_SIZE` (the runtime log shows SDMA above 16 KiB), and
   distribution patches over the upstream driver source we read.
8. **Upstream report:** the finding reproduces with a ~20-line probe on two kernels. We have not filed it with ROCm / amdgpu
   yet.
