# Feature map: what each switch does, and where it sits in a decode token

Built by YST. Contact: https://yst.dev

This is a reference for every engine switch and build feature in this fork, kept and dropped alike: what it does, which
stage of a decode token it acts on, on which card, whether it is exact, whether it is on in build 12, and what it
measured. The stages are the ones in [ANATOMY.md](ANATOMY.md); the evidence is in the cards linked in each row.

**How to read the columns.**
- **Where:** the stage of one token (or prefill, drafter, serving) and the device class: lead XTX (XTX #1, engine device
  0), member XTX (#2-#4), MI50 (helpers), host.
- **Exact:** *by construction* (the same arithmetic in the same order, argued from the code); *measured identical*
  (byte-identical outputs in a run, without a by-construction argument); *quality-gated* (changes summation order or
  values, so it needs the paired teacher-forced comparison, ME-02).
- **Status:** *build 12* = on in build 12, the configuration chosen for serving ([SV-09](cards/SV-09.md)), whether by
  default or by its launcher's environment; *off* = available, default off, not in build 12; *dropped* = measured and
  rejected; *never run* = built or designed, no GPU measurement. *Build 13 / 13s / 14 / 15 / 16* (added 13-14 Nov) = first on
  in that build's served (for 16: recommended) environment; each later build keeps everything earlier builds have unless the row says
  otherwise. Build 16 is the build in this release ([SV-16](cards/SV-16.md)); the README's "Builds" table lists what
  each one adds. Build 16's switches are off by default in the code; its recommended environment sets them.
- **Effect:** the card's headline number, with its label. "Not measured" means no card measures it.
- **Code:** every switch in this file is in the fork's release commit (the build-16 source) unless its row says
  "not in the release commit". Each card's Code line says the same.

## 1. One decode token, and the switches that act on each stage

Build 12, single stream, 40 trunk layers; µs are build-11 per-layer means from [ANATOMY.md](ANATOMY.md) §1-2. The
critical path runs down the left-hand column; in a layer where an MI50 has a pick, the lead's own experts (G) are off the
path and the helper window is on it.

```
 host: enqueue the step (embed, 40 layers)           STRATA_DECODE_WARM, DS4_PREFAULT_SPARE, DS4_HOST_GTT, DS4_MALLOC_PIN
   |
   v  ---------------------------- one trunk layer, x40 ----------------------------
 [A] attention, every XTX (~148 µs plain layer)       DS4_HC_FUSE, DS4_DECODE_TRIM, DS4_ATTN_MERGE*, DS4_MULTI_ATTN (B>1),
   |   + indexer layers, Engram layers               DS4_XK_QO (14), DS4_EG_SPLIT*, DS4_EG_AHEAD (13) (Engram,
   |                                                 layers 1 and 14), DS4_TOPK_MB (13), DS4_TOPK_MB_BATCH (15),
   |                                                 DS4_XK_QA (16), DS4_IXS_DISPATCH (16), DS4_ATTN_MERGE=3 (16)
   v
 [R] attention all-reduce, every XTX (~4 µs)          DS4_AR_FUSE, DS4_AR_WIDE, DS4_AR_POST*, (DS4_AR_P2P, DS4_AR_OVERLAP,
   |                                                  DS4_AR_FUSE_NORM: dropped / shelved)
   v
 [P] post + mix + router -> publish (~49 µs)          DS4_ROUTE_FUSE, DS4_ROUTER_BF16, DS4_AHEAD*, DS4_LP_PUBFUSE*,
   |   lead publishes the helper request;            DS4_HW_PUBT* (B>1), (DS4_LP_ROUTE: dropped)
   |   members route without publishing
   +-----------------------------+-----------------------------------------+
   v                             v                                         v
 [G] lead's own experts        members' FFN share                    [W] MI50 helper window (~27 µs empty,
     (~67 µs; off the path       (slack: they wait for the lead)          ~75 µs one expert, ~106 µs two)
     in helper layers)                                                   DS4_HELPER_DD, DS4_HELPER_OVERLAP,
                                                                        DS4_HELPER_BATCH, DS4_LP_FLAGFUSE*,
                                                                        DS4_HW_FOLD*, DS4_HW_HEADQ8*,
                                                                        (DS4_MK_GU / DS4_MK_DN: off, slower;
                                                                        DS4_G906_SGNLDS: in code, off)
   +-----------------------------+-----------------------------------------+
   v
 [J] join on the lead (~17 µs; waits for the latest  DS4_HELPER_DD_SKIP, DS4_LP_JOINCOMB*, (DS4_HELPER_DEFER: shelved)
   |   MI50 with a pick)
   v
 [C] combine -> MoE all-reduce (~18 + 3 µs)           DS4_AR_POST* (post step in the next all-reduce), DS4_AR_FUSE, DS4_AR_WIDE
   v  -------------------------------------------------------------------------------
 [T] token tail: head on every XTX, logits gather,    DS4_DSPARK_NOSEED (drafter-off processes only)
     drafter seeding, sampler (~0.74 ms)
   |
   v  next token (or, with the drafter on: one draft call ~2.2 ms, then one verify pass with 1-3 extra rows,
      2.66-3.11 ms each)                              --dspark, --dspark-conf, --dspark-block, DS4_DRAFT_TP, DS4_BATCH_DSPARK_MAX,
                                                     DS4_BATCH_DSPARK_SAMPLED (13s), (DS4_DSPARK_ADAPT: off), (DS4_G906_NEMAX: off)

 * = new in build 12 (on in its launcher); (13), (13s), (14), (15), (16) = first on in that build
```

Prefill (a prompt chunk) runs the same layers with many rows:

```
 [PF] prompt chunk, every XTX: attention (matrix cores)     STRATA_DS4_HIPK, matrix-core "f32" attention, DS4_ATTN_SLAB,
      -> prompt all-reduce                                  STRATA_DS4_PROMPT_SKIP_TP, DS4_AR_F16, DS4_AR_P2PK,
      -> experts: VRAM experts on each XTX                  STRATA_ROCBLAS_TUNING, DS4_TP_DIRECT, --prefill-chunk
         + host-RAM experts streamed to each XTX (~3 s a chunk, PF-10)
           or, with DS4_PF_HELPER (served from build 13s, NOT bit-identical, validation pending; PF-11),
           computed on the MI50s that hold them, in one-pass tiles with DS4_PF_TILE (build 15; bit-exact; PF-12),
           for prompts up to DS4_PF_HELPER_MAXP tokens (build 16; longer ones take the helper-off path; PF-13)
         the expert cache's moves leave MI50-held experts alone with DS4_EC_FIX (build 16; HX-26, HX-28)
```

Its switches are in section 2.9.

## 2. The switches, by stage

### 2.1 Token start (host)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `STRATA_DECODE_WARM` (not in the release commit) | Runs warm-up helper and decode rounds at load, so the first token does not pay first-use costs. | token start; all devices | by construction (warm-up output discarded) | build 12 (default on) | token-1 tail 86-89 → 38 ms (REPORTED) ([EB-06](cards/EB-06.md)) |
| `DS4_PREFAULT_SPARE`, `DS4_HOST_GTT`, `DS4_MALLOC_PIN` | Pre-fault spare host memory, map host memory as GTT, stop heap growth: three of the four decode-stall fixes (the fourth was stopping `ksmtuned`). | token start / whole step; host | by construction (no arithmetic) | first two: build 12 (default on); `DS4_MALLOC_PIN`: off | 0 stalls in 9 processes (REPORTED) ([EB-05](cards/EB-05.md)) |
| `DS4_HELPER_WARM` | The helper rounds of the warm-up at load (`=0` off). | load; MI50 | by construction | build 12 (default on) | not measured on its own (part of EB-06) |

### 2.2 Attention (every XTX)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_HC_FUSE=2` | Hyper-connection mix, split, norm and int8 rows in one launch. | attention and FFN entry; every XTX | measured identical | build 12 (launcher) | the four fusions together: launches −17 %, +7.6 % (1 rep) (REPORTED) ([FG-02](cards/FG-02.md)) |
| `DS4_DECODE_TRIM=7` | Int8 tails folded into the producing kernels (SwiGLU, q norm) and the latent copies in one kernel. | attention and FFN; every XTX | measured identical | build 12 (launcher) | part of FG-02's −17 % launches ([FG-02](cards/FG-02.md)) |
| `DS4_ATTN_MERGE=1/2` | Five small independent attention kernels (q norm, ring copies, KV finish, q RoPE) run as two launches; `=2` also for multi-row windows. | attention; every XTX | by construction (checked byte for byte at load) | build 12 (`=2`) | −0.46 ms B1 alone (MEASURED) ([FG-07](cards/FG-07.md)) |
| `DS4_ATTN_MERGE=3` | As `=2`, and the sparse-attention unpack step folded into the second merged launch: one launch fewer a layer. | attention; every XTX | measured identical (to `=2`, drafter off and on) | build 16 (code default off) | B1 −0.157 ms a token against `=2`, placebo −0.007 (MEASURED; measured alone, on a pre-16 build); inside build 16, back to `=2` costs +0.15 ms (MEASURED, ablation, one run an arm) ([FG-16](cards/FG-16.md), [FG-07](cards/FG-07.md)) |
| `DS4_MULTI_ATTN` | One attention launch per layer for all sequences of a batch. | attention at B > 1; every XTX | measured identical | build 12 (default on with batched decode) | B8 +36.6 % (REPORTED); none at B1 ([FG-05](cards/FG-05.md)) |
| `DS4_EG_SPLIT=1` | The Engram layers' 108 MB GEMV, replicated on every XTX, is split by rows over the group and gathered by an all-reduce (other rows −0.0). | attention, layers 1 and 14; every XTX | by construction | build 12 | −0.12 ms B1 (MEASURED) ([SV-09](cards/SV-09.md)) |
| `DS4_EG_AHEAD=1` | Starts each Engram layer's chain (host-row copy, dequantise, GEMV) one layer early on a side stream. | attention, layers 1 and 14; every XTX + host | by construction; measured identical | build 13 | production −0.111 ms a token (−1.1 %); B1 not separable (MEASURED) ([FG-09](cards/FG-09.md)) |
| `DS4_XK_QO=1` | The int8 rounding of the attention output is done in the kernel that finishes it: one dependent launch fewer a layer. | attention output; every XTX | measured identical | build 14 | B1 −0.154 ms (−1.18 %); production +0.010, not separable (MEASURED) ([FG-08](cards/FG-08.md)) |
| `DS4_XK_QX`, `DS4_XK_HCMIX`, `DS4_XK_ROUTE`, `DS4_XK_GATE`, `DS4_XK_ATTN=2` (not in the release commit), `DS4_XK_QA` (taken up later: its own row below) | Six other XTX kernel candidates of the same round (expert x from the fused int8 rows; hc_mix loads; router table in LDS; gate loads in flight; a sparse-attention tile; a GEMV int8 epilogue). | attention / router / experts; every XTX | bench exact (QA after a fix) | dropped | four failed their bench lines; QX bench pass, engine not separable; QA bench pass, not taken further then (REPORTED) ([FG-08](cards/FG-08.md)) |
| `DS4_XK_QA=1` | The attention output's first GEMV (wo_a, Q5_K) rounds its outputs to int8 as it stores them, and wo_b reads them: one launch fewer a layer. One-row steps only; needs `DS4_HC_FUSE`. | attention output; every XTX | measured identical (16 of 16 checks) | build 16 (code default off) | B1 −0.208 ms a token, placebo −0.008 (MEASURED; measured alone, on a pre-16 build); inside build 16, off costs +0.16 ms (MEASURED, ablation, one run an arm) ([FG-16](cards/FG-16.md), [FG-08](cards/FG-08.md)) |
| `DS4_TOPK_MB=1` | The indexer's top-512 pick over up to 64 workgroups when there are at least 8,192 scores. | attention, indexer layers; every XTX | measured identical (same selected set) | build 13 | 110K −0.393 ms a token, 256K −0.846, 1K no change (MEASURED) ([FG-10](cards/FG-10.md)) |
| `DS4_TOPK_MB_BATCH=1` | The same top-k on the batched-decode path. | attention, indexer layers, B > 1; every XTX | measured identical; batched = solo | build 15 | 64K, 4 users −0.887 ms a step (−3.46 %); 110K, 2 users −0.760 (MEASURED) ([FG-11](cards/FG-11.md)) |
| `DS4_IXSCORE_SPLIT` (not in the release commit) | Split indexer-score kernel. | attention, indexer layers; every XTX | bench IDENTICAL | dropped (bench gate) | 1K −16.88 µs a call, 112K +10.56 µs (MEASURED) ([FG-12](cards/FG-12.md)) |
| `DS4_IXS_DISPATCH=<nmax>` | The split score kernel only for one-row steps with at most `<nmax>` keys; the old kernel otherwise. | attention, indexer layers; every XTX | old and split kernels' machine code identical to the benched ones (REPORTED); engine output IDENTICAL (MEASURED) | build 16 (`=16384`; code default off) | B1 −0.135 ms a token, placebo +0.004 (MEASURED; a five-helper run) ([FG-12](cards/FG-12.md)); four helpers −0.165 ms (MEASURED; measured alone, on a pre-16 build); inside build 16, off costs +0.17 ms (MEASURED, ablation, one run an arm) |
| `DS4_ATTN_SPLIT` | Split-KV decode attention. | attention; every XTX | — | off | "no measurable decode gain" (code comment; not carded) |
| `DS4_ENGRAM_LOCK` (not in the release commit) | Engram row-cache locking (ported). | attention, Engram layers; host | — | never run | not measured ([FG-06](cards/FG-06.md)) |
| `DS4_ENGRAM_CACHE_MB` | Size of the host cache for Engram table rows. | attention, Engram layers; host | by construction (I/O only) | build 12 (default) | not measured |
| `DS4_BT_FUSE=1` (not in the release commit) | The one-row path's attention fusions carried into the batched attention path (fusion item 2). | attention, batched steps (2+ requests in a step); every XTX | measured identical (batch test B = 1 / 2 / 8 against solo; served greedy at 4 concurrent) | build 17 (code default off) | batched step −6.0 % at B = 1, −5.8 % at B = 8; served 8 users +6.4 % alone (MEASURED) ([FG-16](cards/FG-16.md)); in build 17 with `DS4_ECOWN`, 8 users +12.5 % ([SV-17](cards/SV-17.md)) |

### 2.3 All-reduces (attention and MoE, every XTX)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_AR_FUSE` | One-phase decode all-reduce (put + sum) instead of a two-round reduce-scatter. | both all-reduces; every XTX | bitwise SAME (REPORTED) | build 12 (default on) | decode +1.2 % (REPORTED) ([AR-02](cards/AR-02.md)) |
| `DS4_AR_WIDE` | 128-bit loads in the host-memory all-reduce. | both all-reduces; every XTX | ids equal, bits identical (REPORTED) | build 12 (default on) | 16K prompt +5.6 %, decode set +3.1 % (REPORTED) ([AR-05](cards/AR-05.md)) |
| `DS4_AR_POST=1/3` | Folds the hyper-connection post step (bit 1) and the put of the partial (bit 2) into the all-reduce's sum kernel. | all-reduce + post; every XTX | by construction (checked at load; cross-card flags end to end) | build 12 (`=3`) | `=3` −0.20 ms; `=1` alone +0.06 ms (MEASURED) ([FG-07](cards/FG-07.md)) |
| `DS4_AR_P2P` (not in the release commit) | Peer-to-peer decode all-reduce. | both all-reduces; every XTX | bit-identical (REPORTED) | dropped | −4 % / −11 % (REPORTED): predicted from a stale profile ([AR-03](cards/AR-03.md)) |
| dp2p (branch-only) | Peer-to-peer decode all-reduce, second attempt. | both all-reduces | (REPORTED) | dropped | NO-GO (REPORTED) ([AR-06](cards/AR-06.md)) |
| `DS4_AR_FUSE_NORM` (not in the release commit) | Fuses the norm after the all-reduce into it. | all-reduce tail; every XTX | exact (MEASURED) | dropped (shelved) | ~0: it shortens every card equally, the lead is still last (MEASURED) ([AR-07](cards/AR-07.md)) |
| `DS4_AR_OVERLAP` | Overlaps the all-reduce with the shared expert on a second stream. | MoE all-reduce / lead's experts; every XTX | DIFFER with the drafter (MEASURED) | dropped (shelved) | slower: the overlapped work is off the path in helper layers (MEASURED) ([AR-07](cards/AR-07.md)) |
| `DS4_SER` (not in the release commit) | Smart expert reduction. | MoE; every XTX | quality-gated | never run (parked) | not run ([AR-08](cards/AR-08.md)) |

### 2.4 Router → publish (lead; members route without publishing)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_ROUTE_FUSE=2` | The router stage in two launches instead of four. | router; every XTX | measured identical (checked at load) | build 12 (launcher) | part of FG-02 ([FG-02](cards/FG-02.md)) |
| `DS4_ROUTER_BF16=1` | Keeps the BF16 router weights in BF16: half the bytes read. | router; every XTX | quality-gated (paired quality within noise) | build 12 (launcher) | part of FG-02; 96.10 % top-1 gate passed (REPORTED) ([FG-02](cards/FG-02.md)) |
| `DS4_AHEAD=0` | Drops the lookahead router GEMV paired into each layer's router. With the BF16 router that kernel is not launched anyway. | router; lead | by construction | build 12 (harmless) | +0.02 ms: ~0, as the desk reading said (MEASURED) ([HX-21](cards/HX-21.md)) |
| `DS4_LP_PUBFUSE=1` | Publishes the one-row helper request from inside the routing launch (−1 dependent launch a layer). | publish; lead | by construction (checked at load) | build 12 | −0.11 ms B1 (MEASURED) ([HX-21](cards/HX-21.md)) |
| `DS4_HW_PUBT=1` | The same for windows of 2-16 rows: the routing launch's last block publishes. | publish at B > 1 / verify; lead | by construction (checked at load) | build 12 | not separated; nothing visible at B8 in one run (MEASURED) ([HX-22](cards/HX-22.md)) |
| `DS4_LP_ROUTE=1` | Top-6 picked per warp with one merge: 3 barriers instead of 13. | router; every XTX | by construction (checked at load) | dropped | +0.01 ms; the pre-publish segment did not shrink (MEASURED) ([HX-21](cards/HX-21.md)) |
| `DS4_ROUTER_ROWS=1` (not in the release commit) | Router 8 rows at a time (prompt). | router, prefill | — | never run | not run ([PF-08](cards/PF-08.md)) |

### 2.5 The MI50 helper window

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `--helper-device` | Lists the MI50s that hold and compute the routed experts that do not fit in XTX VRAM. | helper window; MI50 | quality-gated (passed: top-1 96.94 %, median KL 4.5e-4) | build 12 (4 helpers) | first MI50: +20 % decode (MEASURED) ([EB-01](cards/EB-01.md), [EB-03](cards/EB-03.md), [QA-01](cards/QA-01.md)) |
| no-SDMA window (default path; `HSA_ENABLE_SDMA=0` finding) | The helper request as one device-side sequence: kernel fetch, zero-copy out, flag, graph. | helper window; MI50 | not run as identity (REPORTED) | build 12 (default) | 30.65 → 33.39 tok/s on 1 XTX + MI50 (REPORTED) ([HX-04](cards/HX-04.md), [HX-06](cards/HX-06.md)) |
| `DS4_HELPER_PLACE=stripe` | Spreads each layer's helper experts across the MI50s (`layer` puts a layer on one MI50). | placement at load; MI50 | regression IDENTICAL (REPORTED) | build 12 (default) | part of build 8's +22.3 % decode set (REPORTED) ([HX-07](cards/HX-07.md)) |
| per-layer slot sizing (default; `DS4_HELPER_UNIFORM_SLOTS=1` restores uniform) | Sizes helper slots per layer, so more experts fit. | placement at load; MI50 | parity 20/20 (REPORTED) | build 12 (default) | +17 % experts held; solo decode unchanged (REPORTED) ([HX-05](cards/HX-05.md)) |
| `DS4_HELPER_OVERLAP=1` | Overlaps the return of the helper's down rows with the lead's work. | helper window / join; lead + MI50 | IDENTICAL (MEASURED) | build 12 (launcher) | with BATCH: tp4h4 B1 −2.0 % (MEASURED) ([HX-10](cards/HX-10.md)) |
| `DS4_HELPER_BATCH=1` | Packs the activation rows for all helpers into one staging buffer. | helper window; lead + MI50 | IDENTICAL (MEASURED) | build 12 (launcher) | see OVERLAP ([HX-10](cards/HX-10.md)) |
| `DS4_HELPER_DD=1` | Device-direct windows: each MI50 waits for its request on the device, and the lead joins the results in one kernel; the host leaves the per-layer loop. | helper window + join; lead + MI50 | IDENTICAL (MEASURED) | build 12 (launcher) | B1 −3.9 %, B8 −4.2 % (MEASURED); v1 stopped at the identity gate ([HX-17](cards/HX-17.md), [HX-16](cards/HX-16.md)) |
| `DS4_LP_FLAGFUSE=1` | The helper's done flag is written by the last block of its combine, not by a separate kernel after its graph. | helper window; MI50 | by construction | build 12 | −0.20 ms B1; the lead's join 19.5 → 13.2 µs a layer (MEASURED) ([HX-21](cards/HX-21.md)) |
| `DS4_HW_FOLD=1` | The window's head and flag become the first and last nodes of its expert graph: one launch a window. | helper window; MI50 | by construction | build 12 | with HEADQ8 and PUBT: −0.08 ms B1 (MEASURED) ([HX-22](cards/HX-22.md)) |
| `DS4_HW_HEADQ8=1` | The head rounds the activation rows to int8 as it copies them; no separate quantise kernel. | helper window; MI50 | by construction (checked at load) | build 12 | see FOLD ([HX-22](cards/HX-22.md)) |
| `DS4_MK_GU=1`, `DS4_MK_DN=1` (knobs `DS4_MK_PH`, `DS4_MK_AO`, `DS4_MK_GU_TPB`, `DS4_MK_DN_TPB`) | Rewritten MI50 gate/up and down kernels: 3 waves a SIMD instead of 2. | helper window, one-row windows; MI50 | by construction; measured identical | dropped (off) | +0.33 ms B1 (MEASURED): grid-limited, not register-limited; lookups serialised; 6× grid (INFERRED) ([HX-23](cards/HX-23.md)) |
| `DS4_G906_NEMAX=4` | Sends windows of up to 4 rows (verify) to the gfx906-specific kernels instead of the portable ones. | helper window, verify rows; MI50 | by construction | off | production +0.6 %, within the spread (MEASURED) ([SV-09](cards/SV-09.md)) |
| `DS4_G906_SGNLDS=1` | IQ2/IQ3 sign masks read from a 128-entry LDS table. | helper window, one-row windows; MI50 | bench bit-identical; measured identical | off (in the code from build 13, not in the served environment) | B1 −0.113 ms; production +0.005, not separable (MEASURED) ([HX-24](cards/HX-24.md)) |
| `DS4_G906_PERGROUP=1`, `DS4_G906_DPP=1` | gfx906 kernel chosen per row group; DPP / swizzle reductions. | helper window; MI50 | bench bit-identical | dropped | multi-entry groups +8.8 to +51.6 µs; DPP ~0 (REPORTED) ([HX-24](cards/HX-24.md)) |
| `DS4_G906_FUSEDOWN=1` (`_TOK`); multi-row kernel arms (not in the release commit) | A fused int8 + down + combine kernel; kernels that load each weight block once for all rows of a window. | helper window, 2-4-row windows; MI50 | bench bit-identical (one row-loop arm within tolerance only) | dropped | fused down +10.2 µs a one-row window; multi-row −0.65 µs, under its line (MEASURED) ([HX-25](cards/HX-25.md)) |
| `DS4_G906_LIVE`, `DS4_G906_BPC` | gfx906 kernel grid knobs. | helper window; MI50 | BITWISE (MEASURED) | dropped | nothing beats the default (MEASURED) ([HX-19](cards/HX-19.md)) |
| `DS4_HELPER_ARMED=1` (`DS4_HELPER_ARMED_MAXROWS`) | Armed helper windows (pre-armed requests), the predecessor of device-direct windows. | helper window; MI50 | identical (REPORTED) | off | decode set +6.5 %, B1 +10.5 %; B8 bimodal (REPORTED) ([HX-09](cards/HX-09.md)) |
| `STRATA_HELPER_ZC_OUTPUT` (and zero-copy input, one-shot) (not in the release commit) | Early zero-copy and one-shot switches for the helper. | helper window; MI50 | not run | dropped | ZC_OUTPUT −8 %; others within noise (REPORTED) ([HX-02](cards/HX-02.md)) |
| `STRATA_POOL_SERVE`, `STRATA_POOL_CLIENT` (`DS4_POOL_STREAMS`, `DS4_POOL_UNION`) | A helper pool shared by several engines, with a shared-memory mailbox. | helper window; MI50 + host | not run (pool gates never run) | dropped (shelved) | 3 engines on one MI50: ~132 tok/s combined (REPORTED) ([HX-03](cards/HX-03.md)) |
| `DS4_HSTEP_HEAD` (not in the release commit) | A faster MI50 helper head (fewer PCIe reads at the start of each window). | helper window; MI50 | pending | pending: not yet measured | target: the lead's served wait for the helpers, 9–13 % of a one-user step (MEASURED) ([FG-18](cards/FG-18.md)) |

### 2.6 Join and combine (lead)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_HELPER_DD_SKIP=1` | The lead's join waits only for MI50s that have a pick in this layer. | join; lead | byte-identical (MEASURED) | build 12 (launcher) | B8 −5.6 %; B1 ~0 (MEASURED) ([HX-18](cards/HX-18.md)) |
| `DS4_LP_JOINCOMB=1` | The lead's device-direct join and its combine in one launch. | join + combine; lead | by construction | build 12 | −0.005 ms: the saved launch sits in front of the wait (MEASURED) ([HX-21](cards/HX-21.md)) |
| `DS4_HELPER_DEFER=1` (`DS4_HELPER_DEFER_MAXW`) | Joins the helper experts one layer late, taking them off the lead's path. | join; lead | quality-gated: full deferral FAILED (perplexity +11.7 %); MAXW ≤ 0.2 passed | dropped (shelved) | B8 −29 % but not exact; passing variants slower than off (MEASURED) ([HX-11](cards/HX-11.md)) |
| `DS4_HELPER_DEFER_2W` | Partial deferral with two windows in flight. | join; lead | "passes, 94.9 % agreement" (REPORTED) | off (parked) | −0.7 ms (REPORTED) ([HX-12](cards/HX-12.md)) |
| `DS4_DD_NO_LANE_WAIT` (not in the release commit) | The lead stops waiting for the CPU lane's empty answer each layer. | join; lead | IDENTICAL (MEASURED) | dropped | −0.1 ms B1, +3.8 % B8: the lane was never binding (MEASURED) ([HX-20](cards/HX-20.md)) |

### 2.7 Token tail (head, drafter seeding, sampler)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_DSPARK_NOSEED=1` | With `--dspark off`, skips feeding the drafter after every token (it could never draft). | token tail; lead | by construction | off (drafter-off processes only; build 12 serves with the drafter on) | −0.30 ms B1 drafter off (MEASURED) ([SV-09](cards/SV-09.md)) |

### 2.8 Drafter (speculative decoding)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `--dspark on/off/none` | The drafter, DSpark (DeepSeek's own drafter, shipped with V4.1): on drafts; `off` loads it but never drafts (and still feeds it, see NOSEED); `none` does not load it. | drafter round; lead XTX group | greedy: measured identical to plain decode; sampled: exact in distribution | build 12 (on) | drafter worth +34.8 % on live traffic (INFERRED from the cost model) ([DR-06](cards/DR-06.md), [SV-08](cards/SV-08.md)) |
| `--dspark-conf 0.3`, `--dspark-block 3` | The confidence floor for each draft, and the cap on drafts verified a round. The drafter always runs its 5-row window; the block only truncates. | drafter round; lead | as above (policy changes only the count, never the keep rule) | build 12 (0.3 / 3) | +23 % decode (REPORTED) ([DR-01](cards/DR-01.md)); near the optimum on live traffic (INFERRED) ([DR-06](cards/DR-06.md)); block 5 / confidence 0.4 −3.97 % against a 2.44 % placebo: adopt-candidate, confirm pending (MEASURED) ([DR-08](cards/DR-08.md)) |
| per-card drafter memory budget (default) | Places the drafter's memory per card. | load; every XTX | 1 repeated-token decode in 9 (REPORTED) | build 12 (default) | one prompt 54.6 → 63.1 tok/s (REPORTED) ([DR-02](cards/DR-02.md)) |
| `DS4_DRAFT_TP` | The drafter's blocks tensor-parallel over the lead's group. | draft call; every XTX | not run | build 12 (default on) | +30 % trunk experts per XTX; text 54.2 → 63.6 tok/s (REPORTED) ([DR-03](cards/DR-03.md)) |
| `DS4_BATCH_DSPARK_MAX` | Drafting inside batched decode steps, with an adaptive rule; at most this many banks draft in a step. | drafter at B > 1; every XTX | exact, batched = solo (REPORTED) | build 12 (default 2); build 13s (3) | B1 77.2 / B2 87.6 / B4 99.1 tok/s (REPORTED) ([DR-04](cards/DR-04.md)); at 4 users MAX 4 +11 %, MAX 3 +0.4 % (REPORTED) ([DR-07](cards/DR-07.md)) |
| `DS4_BATCH_DSPARK_SAMPLED=1` | Sampled requests draft and verify with the sampling rule inside batched steps (concurrent users). | drafter at B > 1; every XTX | exact in distribution (600-sequence test PASS; a random-stream overlap bug found and fixed first) | build 13s | 3 users 129.6 → 157.8 tok/s total (× 1.22) (MEASURED) ([DR-07](cards/DR-07.md)) |
| `DS4_DSPARK_ADAPT=1` | Sets the drafts a round from an online-calibrated keep rate. | drafter policy; host | exact in distribution | dropped | equal to build 12 at matched output length (MEASURED) ([SV-09](cards/SV-09.md), [DR-06](cards/DR-06.md)); at its design point (0.1, 5) +3.8 %, fewer tokens a round (REPORTED) ([DR-08](cards/DR-08.md)) |
| `DS4_DSPARK_SAMPLE=1` | Draws drafts from the drafter's own distribution instead of its argmax. | draft call; lead | exact in distribution | dropped | +13.0 % ms a generated token; acceptance 0.526 vs 0.583 (REPORTED) ([DR-08](cards/DR-08.md)) |
| `DS4_DRAFT_EXPERTS=helper` | Puts the drafter's routed experts on the MI50s, freeing ~1.8 GB a card of XTX VRAM for trunk experts. | draft call; MI50 + every XTX | changes placement | never run | predicted +0.2 ms a token at one user (slower) (INFERRED) ([HX-27](cards/HX-27.md)) |
| `DS4_DRAFT_KEEP` | A smaller drafter; trimmed vocabulary (study). | draft call | not run | never run | +1-5 % projected (INFERRED) ([DR-05](cards/DR-05.md)) |

### 2.9 Prefill (prompt chunks)

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_TP_DIRECT`, `DS4_ATTN_SLAB` | Prompt streaming of the host-RAM experts to each member, and a 2,048-row attention slab. | prefill; every XTX + host | (REPORTED) | build 12 (default) | 2K 7.45 s / 16K 37.29 s baseline (REPORTED) ([PF-01](cards/PF-01.md)) |
| three contiguous DMAs per member (default) | Each member's expert part moves in three contiguous DMAs instead of one command per row. | prefill; every XTX + host | not run | build 12 (default) | 2K prompt 384 s → 10 s (REPORTED) ([TP-02](cards/TP-02.md)) |
| matrix-core prompt attention, "f32" mode (default) | Prompt attention on the matrix cores with hi+lo FP16. | prefill attention; every XTX | quality-gated (passed) | build 12 (default) | 16K TTFT 18.75 → 11.06 s with HIPK (REPORTED) ([PF-02](cards/PF-02.md)) |
| `STRATA_DS4_HIPK` | Matrix-core expert tiles for prompt chunks. | prefill experts; every XTX | within noise (REPORTED) | build 12 (default 1) | 16K TTFT 12.3 s cold / 10.7 s warm (REPORTED) ([PF-03](cards/PF-03.md)) |
| `DS4_PF_WMMA` (not in the release commit) | WMMA expert-tile variants. | prefill experts; every XTX | (REPORTED) | dropped | ([PF-03](cards/PF-03.md)) |
| `STRATA_DS4_PROMPT_SKIP_TP` (alias `DS4_YOCO_TP`) | Skips late layers' prompt work for tokens that do not need it. | prefill, layers 20-39; every XTX | quality-gated (KL 1.75e-4) | build 12 (default 1) | kept as default (REPORTED) ([PF-05](cards/PF-05.md)) |
| `DS4_AR_F16` | FP16 all-reduce for prompt chunks of ≥ 65,536 elements; decode stays F32. | prompt all-reduce; every XTX | quality-gated (at the rounding floor) | build 12 (auto) | kept as default (REPORTED) ([AR-01](cards/AR-01.md)) |
| `DS4_AR_P2PK` | Prompt all-reduce by peer stores, then peer-major order. | prompt all-reduce; every XTX | self-test bit-identical at load | build 12 (default since build 9) | 16K prompt −19 %, 8K −23 % (REPORTED) ([AR-04](cards/AR-04.md)) |
| `STRATA_ROCBLAS_TUNING=<table>` | A tuned rocBLAS table for the hyper-connection mix GEMM. | prefill (and windows ≥ that size); every XTX | bitwise at 999 rows only; differs at 2,048 (quality-gated via QA-03) | build 12 (launcher) | 1.8-4.2× per bucket; 16K stage −28 % (REPORTED) ([FG-04](cards/FG-04.md)) |
| `DS4_ATTN_SPLIT3` | Three-way prompt-attention split. | prefill attention; every XTX | quality-gated (passed) | off (parked) | 8K / 16K TTFT −11 / −12 % (REPORTED) ([PF-04](cards/PF-04.md)) |
| `DS4_FAST_ATTN` | WMMA sparse attention for prompt chunks. | prefill attention; every XTX | quality-gated: failed the accuracy bar | dropped | 16K prompt 37.4 → 31.4 s on tp3, not adopted (design note; not carded) |
| `DS4_PF_HELPER=<rows>` (`_MIX`, `_CHUNK_MIN`, `_TIME`) | The MI50s compute the experts they hold for prompt chunks up to `<rows>` tokens, instead of the XTX streaming them; `_CHUNK_MIN` (default 128) sends 128-511-token prompts through the chunked path too; `_TIME` adds a helper-side timer. **Not bit-identical**: int8 activations on the MI50 and a different summation order. Decode untouched; unset is byte-identical. | prefill floor; MI50 + every XTX | **quality-gated; validation pending** (first look: no harm at one depth on one text; the validation run formally inconclusive; batched greedy output ≠ solo under concurrency) | build 13s (`=8192`, served; validation still pending) | 1K suffix 3.42 → 1.33 s, 2K 3.71 → 2.11, 4K 4.72 → 3.75, 8K 6.70 → 5.71, 16K equal (MEASURED); build 15 served: faster at 2K (2.9 vs 4.6 s), slower from 16K (32K 22.2 vs 18.4 s) (MEASURED) ([PF-11](cards/PF-11.md), [PF-10](cards/PF-10.md), [SV-10](cards/SV-10.md), [PF-13](cards/PF-13.md)) |
| `DS4_PF_TILE=1` | The helper's prompt expert sets in one-pass tiles (several outputs a lane share each x load; per-type dispatch; the old path for small sets). Needs `DS4_PF_HELPER`. | prefill floor; MI50 | bit-identical to the `DS4_PF_HELPER` path (bench every cell; served greedy IDENTICAL) | build 15 | TTFT −15 % at 2K, 8K, 32K and 4 users × 8K; decode unchanged (MEASURED) ([PF-12](cards/PF-12.md)) |
| `DS4_PF_HELPER_MAXP=<tokens>` | With `DS4_PF_HELPER`: a prompt read of more than `<tokens>` new tokens takes the helper-off path; shorter ones keep the helper. | prefill floor; MI50 + every XTX | measured: above the cap the helper-off tokens, below it the helper-on tokens | build 16 (`=8192`; code default unset: no cap) | served first token 16K −14 %, 32K 22.2 → 18.4 s (−17 %), 64K 43.7 → 35.0 s (−20 %); 2K and 8K unchanged (MEASURED) ([PF-13](cards/PF-13.md)); single process inside build 16, cold 32K 16.93 → 16.08 s (−5 %) with the cap (MEASURED, ablation) |
| `DS4_PF_PREC` (not in the release commit) | F16 operands on the MI50 for the helper's prompt sets. | prefill floor; MI50 | accuracy held on the bench | dropped | 1.73-2.52 × slower on the bench (REPORTED) ([PF-11](cards/PF-11.md)) |
| MI50 prefill helper (earlier) | Earlier MI50 prefill via set requests. | prefill; MI50 | parity pass (REPORTED) | dropped | MI50 2.4-2.7 TFLOPS end to end (REPORTED) ([PF-06](cards/PF-06.md)) |
| pipelined layer-split prefill (study) | Layer-split prefill beside tp3 decode. | prefill | not run | dropped | NO-GO (REPORTED) ([PF-07](cards/PF-07.md)) |
| `DS4_HIP_ATTN_SWIZZLE`, `DS4_SKIP_UNPICKED` (not in the release commit) | Prompt-attention swizzle; skip unpicked experts (ported). | prefill | not run | never run | not run ([FG-06](cards/FG-06.md)) |
| `DS4_AR_SLAB` (not in the release commit) | All-reduce slab for prompts (ported). | prompt all-reduce | not run | never run | not run ([PF-08](cards/PF-08.md)) |
| `--prefill-chunk` | Prompt chunk size (8,192 by default). | prefill | 16,384 is not bitwise (quality-gated) | build 12 (default) | tp3 16K: chunk 16,384 was +1 s (design note; untested at tp4) |

### 2.10 Layout, placement and serving

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `--tp 4`, `--gpus` | Tensor parallel over four XTX; uneven expert split where 9 FF blocks do not divide by 4. | every stage; every XTX | reg IDENTICAL; q4 95.96 % top-1 (REPORTED) | build 12 (tp4) | decode set +9.9 %, 2K prompt +11 % slower (MEASURED) ([TP-05](cards/TP-05.md), [TP-01](cards/TP-01.md)) |
| `DS4_TP_CHECK=1` | Load-time check of each member's part. | load | n/a | off (diagnostic) | — ([TP-05](cards/TP-05.md)) |
| `DS4_TP_UNEVEN`, `DS4_TP_SHARES` | Turn off the uneven split; force block counts per member (test knob). | load; every XTX | changes placement (quality-gated) | build 12 (uneven split on); `TP_SHARES` off (test only) | a deliberate skew is visible in decode; a perfect re-split would gain ≤ 1.4 % (MEASURED) ([TP-08](cards/TP-08.md)) |
| `DS4_EP=1` (not in the release commit) | Expert parallel over the tensor-parallel group. | MoE; every XTX | not run | never run | compile only ([TP-06](cards/TP-06.md)) |
| `--max-context` | The context limit. The KV cache (920 B a token per XTX) is allocated before the experts, so a larger context places fewer experts in VRAM. | load; every XTX | changes placement, so not byte-comparable across contexts ([QA-08](cards/QA-08.md)) | build 12 (262,144, its launcher's default) | 7,059 → 7,011 VRAM experts per XTX, 128K → 256K (MEASURED) ([QA-09](cards/QA-09.md)) |
| `DS4_BATCH_DECODE` | Batched decode banks. | every stage at B > 1 | exact B1-B8 (REPORTED) | off in serving up to build 12; build 13s serves 4 banks (PAR 4) | B4 103 tok/s total (REPORTED) ([SV-03](cards/SV-03.md)); no single-user cost from 4 banks: +0.1 % engine, +0.9 % server (MEASURED) ([SV-10](cards/SV-10.md)) |
| `--mmap-experts on` | Memory-maps the expert files instead of copying the CPU-lane experts into 83 GB of pinned host memory. | load, prefill fallback; host | measured identical | build 13s | prompt time within noise at 1K / 32K / 110K; 83 GB less pinned RAM; load time unchanged (MEASURED) ([EB-08](cards/EB-08.md)) |
| `DS4_EC_FIX=1` | The expert cache never moves an MI50-held expert into XTX VRAM. Without it each prompt read moved up to 8: their places in VRAM went to experts then streamed from host RAM (the leak, HX-26), and with `DS4_HELPER_DD=1` the moved experts were also computed on their MI50 and summed twice (HX-28). | expert cache (prompt reads, and every later step); host | a correctness fix: without it output depends on uptime; with it 8 of 8 answers the same after 53 prompts (MEASURED) | build 16 (code default off; set it whenever the expert cache and `DS4_HELPER_DD` are on) | picks summed twice 4.93 % → 0; 2K first-token drift over 50 prompts +15.8 % → −0.9 %; served 8K first token 5.9 → 6.7 s, the leaky cache's gain given up (MEASURED) ([HX-28](cards/HX-28.md), [HX-26](cards/HX-26.md), [SV-16](cards/SV-16.md)); same-day served A/B, fix on vs off: one user +4.8 % a token at 2K, +9.2 % at 8K, 8K first token +14 %; no decode cost in one process (MEASURED, [UPDATE-2026-11-14.md](UPDATE-2026-11-14.md)) |
| `DS4_ECOWN=1` (not in the release commit) | The expert cache moves hot MI50-held experts into XTX VRAM as before, and each such pick is computed once, by the lead: each request to an MI50 marks the picks the lead holds, and the MI50 skips them. Refuses `DS4_EC_FIX=1`. | expert cache, MI50 helper window, join; lead + MI50 | measured identical to build 16 (switch off; on with the cache off); 0 picks counted twice in 2.92 M with the cache on; batch == solo in a fixed-timing mode | build 17 (code default off) | against build 16, one user: 8K first token −12.5 %, 2K −4.3 %, 8K decode −2.0 %, 2K decode +0.6 %, 32K +2.7 % (MEASURED) ([HX-30](cards/HX-30.md), [HX-29](cards/HX-29.md)) |
| `DS4_ECSWAP=1` (not in the release commit) | With `DS4_ECOWN`: the cold expert a move pushes off an XTX is copied into the moved expert's freed MI50 slot and handed over at a step boundary, instead of falling to the host-RAM path. | expert cache; MI50 + lead | 0 twice, 0 dropped, 0 table mismatches over 1,592 handovers (200-prompt soak); identical with the switch off and with the cache off | build 18, staged, re-validation pending (code default off) | one user's 2K first token after 50 prompts on one server −22.7 % (2.67 → 2.06 s); fresh-server cells unchanged (MEASURED) ([HX-32](cards/HX-32.md)) |
| `DS4_SETTLE_AFTER=1\|2` (not in the release commit) | Runs a prompt's cache settle after its first token: `=1` at once, blocking; `=2` its rounds over the decode ticks. | expert cache, first token; lead | the same end placements as without it; `=1` gives the same tokens | parked (code default off) | 32K first token −0.13 s only; a leftover settle after a short answer can delay the next request up to ~1.6 s (INFERRED) ([HX-31](cards/HX-31.md)) |
| `STRATA_DS4_PREFIX_DIR` (not in the release commit) | Prefix store: prompt state survives a restart. | prefill (skipped); host | (REPORTED) | off (opt-in) | 25K prompt TTFT 63.1 → 10.4 s after restart (REPORTED) ([SV-02](cards/SV-02.md)) |
| `STRATA_DS4_SLOT_KEEP`, `STRATA_DS4_SLOT_GB` (`_CAP`, `_DIR`, `_MIN`, `_DAYS`) | Conversation slots: a request that does not continue the conversation in the model sets it aside and takes back the one it does continue; kept across runs within a folder budget. | prefill (skipped); host RAM + disk | by construction (saved state) | build 12 (launcher: on, 16 GB) | not measured in the cards |
| `DS4_BATCH_PREFILL_SHARE` | A launcher setting (0.5). Corrected 13 Nov: the variable does not exist in the source or the binary, so it has no effect. | serving | — | no effect | none ([SV-12](cards/SV-12.md)) |
| server parallel slots (PAR) | Several conversations served at once, one bank each. | serving | greedy batched = solo except with the prompt helper on ([SV-10](cards/SV-10.md)) | build 13s (PAR 4); 8 measured | 4 slots: 146-153 tok/s total at 2-4 users; 8 slots: 186 tok/s at 8 users, about −4 % for one user (MEASURED) ([SV-11](cards/SV-11.md)); the continuous-batching card [SV-04](cards/SV-04.md) predates this |
| MI50 DPM `high` (host sysfs, launch wrapper) | Pins the MI50s' power state so helper requests do not wait on a clock ramp. | helper window; MI50 | IDENTICAL (MEASURED) | build 12 (wrapper) | B1 58.31 → 60.95 tok/s (MEASURED) ([CP-03](cards/CP-03.md)) |
| MI50 power cap 155 / 178 W (host sysfs) | Raises the MI50s' cap from 135 W. | prefill and helper window; MI50 | IDENTICAL (MEASURED) | dropped (caps stay 135 W) | 178 W: 32K prompt −2.78 %, 110K −2.39 %; decode no change (MEASURED) ([CP-05](cards/CP-05.md)) |
| MI50 PCIe link pin; manual top clock pin (host sysfs) | Pinning the link speed or a manual clock level. | helper window; MI50 | identical | dropped | link pin −10 %; manual pin loses at B8 (REPORTED / MEASURED) ([CP-02](cards/CP-02.md), [CP-04](cards/CP-04.md)) |
| XTX power caps, −50 mV (host sysfs) | Each XTX at 90 % of its board's default power limit with an undervolt. | every stage; every XTX | n/a | build 12 (boot service) | not measured ([CP-01](cards/CP-01.md), [HW-08](cards/HW-08.md)) |

### 2.11 Build options

| option | what it does | exact | status | effect (card) |
|---|---|---|---|---|
| `STRATA_HIP_HELPER_ALLTU=1` | Builds every translation unit for both the RDNA3 and gfx906 targets, so one binary drives both. | n/a | build 12 | fixes multi-GPU "invalid kernel file" on ROCm 7.14 ([EB-04](cards/EB-04.md), [EB-01](cards/EB-01.md)) |
| gfx906 kernels only in the gfx906 device pass | Keeps MI50-only kernels out of the XTX code object; a 2 MiB code-object boundary otherwise costs an expert slot per XTX. | restores identity | build 12 | DIFFER → IDENTICAL (MEASURED) ([QA-08](cards/QA-08.md)) |

### 2.12 Diagnostics and oracles (timing or debugging only; never served)

| switch | what it does | effect (card) |
|---|---|---|
| `DS4_PROF` | Per-stage time profile (stretches wall time; durations only). | [HX-14](cards/HX-14.md), [PF-09](cards/PF-09.md), [ME-06](cards/ME-06.md) |
| `DS4_TSTAMP` (not in the release commit) | Device timestamps on one clock at fixed points of each layer. | [HX-20](cards/HX-20.md), [HX-21](cards/HX-21.md) |
| `DS4_INJECT_EMPTY` (not in the release commit) | Injects empty dependent kernels to price one launch. | 2.75 / 2.63 / 0.47 µs a kernel (MEASURED) ([HX-20](cards/HX-20.md)) |
| `DS4_DD_LANE_DELAY_US`, `DS4_HELPER_DELAY_US` (`DS4_DD_LANE_DELAY_US` not in the release commit) | Delay the CPU lane's or the helpers' answers to test what binds. | lane not binding below 50 µs; ~1 ms per 50 µs of helper delay ([HX-20](cards/HX-20.md), [HX-17](cards/HX-17.md)) |
| oracles (no helper wait, no MI50 maths, empty all-reduce) | Produce wrong outputs; timing only. | −0.71 / −0.69 / +0.47 ms; the last explained by corrupted routing ([HX-20](cards/HX-20.md)) |
| `DS4_HELPER_TRACE`, `DS4_HELPER_TIMELINE`, `DS4_TOKLOG`, `DS4_DUMP_*`, `DS4_LAUNCH_CHECK`, `DS4_SCORE_PREFIX`, `DS4_NO_EOS` | Traces, logs, dumps, a launch-bounds check, teacher-forced scoring after a prefix, and a no-end-of-sequence switch for fixed-length runs. | method only |
| `DS4_PROF_ROWS` (not in the release commit) | Per-stage profile by verify-row count. Its own cost changes with the row count, so it could not attribute the row cost. | no data ([DR-06](cards/DR-06.md) addendum) |
| `DS4_HANG_DIAG=1` (not in the release commit) | Host-only: a stalled prompt all-reduce stops with a named error instead of spinning; per-worker breadcrumbs after 30 s without progress; a deadline on the helper-key wait. GPU code byte-identical. | smoke 0 false triggers in 16 prompts (MEASURED) ([SV-12](cards/SV-12.md)) |
| `DS4_TREE_STATS=1` (not in the release commit) | Measurement only: counts rejections at draft position 1 and how often the trunk's token was the drafter's second choice (h2), for the tree-speculation question. Never served. | built, kernels identical to build 15 (REPORTED); h2 0.337 sampled: tree speculation closed (MEASURED) ([DR-09](cards/DR-09.md)) |

### 2.13 XTX expert kernels (the lead's own experts [G] and the members' FFN share)

Added 13 Nov; placed after the diagnostics so that the section numbers above stay stable.

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_DN_TPB=2/4` (not in the release commit) | Down-projection kernels with 2 or 4 tiles a block, in their own source file. | routed experts; every XTX | measured identical | dropped (kept in the code, default 0) | bench −20 % (REPORTED); engine B1 +0.026 ms, served no change (MEASURED) ([FG-13](cards/FG-13.md)) |
| gate/up rewrite (bench only) | Three variants of the gate/up kernel. | routed experts; every XTX | exact on the bench | dropped | +0.7 to +10.7 µs a call against the engine's real kernel (REPORTED) ([FG-14](cards/FG-14.md)) |

### 2.14 Further switches in the code (not carded)

These switches are in the release commit but no card measures them. Each row is read from the code: what it does and its
default are INFERRED from the code, not measured. Section numbers say where each acts.

| switch | what it does | where | exact | status | effect (card) |
|---|---|---|---|---|---|
| `DS4_AR_BLOCKS=<n>` | Blocks of a large (1 MB and up, prompt-chunk) host all-reduce; default 64. More blocks keep the PCIe link busier. | prefill all-reduce; every XTX | by construction (each element summed in member order whatever the count) | off (default 64) | not measured (INFERRED from code) |
| `DS4_AR_P2PK_ORDER=0/1` | P2PK sends: 1 (default) peer-major, one peer's VRAM at a time; 0 tile-major. | prefill all-reduce; every XTX | by construction (same sums) | default 1 with `DS4_AR_P2PK` | part of AR-04's P2PK2 (REPORTED) ([AR-04](cards/AR-04.md)) |
| `DS4_AR_P2PK_TILE=<KiB>` | P2PK tile size; default 1024 KiB (was 64). | prefill all-reduce; every XTX | by construction | default 1024 with `DS4_AR_P2PK` | not measured in a card (INFERRED from code; the code comment cites a microbenchmark) |
| `DS4_AR_P2PK_BLOCKS=<n>` | Blocks of the P2PK kernel; default 192. | prefill all-reduce; every XTX | by construction | default 192 with `DS4_AR_P2PK` | not measured (INFERRED from code) |
| `DS4_AR_P2PK_SPLIT=w,w,...` | Per-member weights (1..1000, one a member) for the share of the chunk each member reduces; default equal. | prefill all-reduce; every XTX | by construction | off (equal shares) | not measured (INFERRED from code) |
| `DS4_AR_P2PK_SEED=<n>` | Fixes the seed of the load-time P2PK self-test (default: from the clock). | load; every XTX | n/a (test input) | test only | n/a |
| `DS4_AR_P2PK_DRY=1` | Allocates and self-tests the P2PK buffers, then uses the host-mapped path anyway. | load; every XTX | by construction (P2PK not used) | test only | n/a |
| `DS4_BATCH_DSPARK=0` | Turns drafting off inside batched decode steps; default on once batched decode is on and the drafter is loaded. | drafter, B > 1; lead | exact (greedy verify) | default on with `DS4_BATCH_DECODE` | see DR-04 ([DR-04](cards/DR-04.md)) |
| `DS4_DSPARK_ADAPT_THETA=<f>` | With `DS4_DSPARK_ADAPT`: fixes the keep-chance threshold theta (default: row ms / ms a token, learned). | drafter; lead | exact (only how many drafts are verified changes) | off | not measured on its own (INFERRED from code) |
| `DS4_DSPARK_ADAPT_ROW_MS=<f>` | With `DS4_DSPARK_ADAPT`: the cost of one extra verify row; default 3.1 ms. | drafter; lead | exact | off | not measured on its own (INFERRED from code) |
| `DS4_DSPARK_ADAPT_PRIOR=<f>` | With `DS4_DSPARK_ADAPT`: pseudo-counts of the confidence prior; default 16. | drafter; lead | exact | off | not measured on its own (INFERRED from code) |
| `DS4_DSPARK_ADAPT_DECAY=<f>` | With `DS4_DSPARK_ADAPT`: the exponential memory of the learned keep rates (0.5..1); default 0.998. | drafter; lead | exact | off | not measured on its own (INFERRED from code) |
| `DS4_DRAFT_BUDGET_SHARED=1` | The old VRAM charge for the drafter: its bytes come out of every member's budget instead of the lead's alone. | load (placement); every XTX | changes expert placement, so outputs can differ (QA-08) | off | not measured (INFERRED from code) |
| `DS4_DRAFT_COUNTS=<path>` | Drafter expert picks: added to the file at exit (profiling run), or with `DS4_DRAFT_KEEP` the file the kept set is read from. | drafter; host | n/a (profiling) / see `DS4_DRAFT_KEEP` | off | part of DR-05's design ([DR-05](cards/DR-05.md)) |
| `DS4_DRAFT_CAL=<dir>` | Sums of x² of each helper-held drafter expert's gate/up input, for an importance matrix (`tools/dsv41/drafter_requant.py --cal`). | drafter; host | by construction (read only) | off | n/a (calibration data) |
| `DS4_DRAFT_HIST=1` | Prints the drafter-expert pick histogram at exit. | drafter; host | by construction | off (diagnostic) | n/a |
| `DS4_FAST_ATTN_MODE=<mode>` | Arithmetic of the matrix-core prompt attention. Unset: f32 (hi + lo FP16 pairs) where the fast path is on by this fork's rule (gfx1100 in a tensor-parallel group), upstream's f16 elsewhere. | prefill attention; every XTX | quality-gated (f32 mode: [PF-02](cards/PF-02.md)) | default as described | see PF-02 ([PF-02](cards/PF-02.md)) |
| `DS4_FAST_ATTN_LAYERS=a-b,c` | Limits the fast prompt attention to these layers; the others keep the scalar kernel. | prefill attention; every XTX | n/a (diagnostic) | off (all layers) | n/a |
| `DS4_FAST_ATTN_CHECK=1` | Compares each fast attention call with the scalar kernel and prints the errors; syncs and allocates every call. | prefill attention; every XTX | n/a (diagnostic) | off | n/a |
| `DS4_G906_XXS_DOWN=1` | Runs the IQ2_XXS down projection on the gfx906 kernel too; by default it stays on the portable kernel, measured faster there at one group. | helper window; MI50 | bench exact (the gfx906 kernels' parity tests) | off | not measured in a card (INFERRED from code; the code comment cites a kernel trace) |
| `DS4_IQ_G906=0` | Turns off the gfx906 i-quant token kernels (the portable kernels run instead). | helper window; MI50 | parity-tested kernels | default on on gfx906 | see HX-04 ([HX-04](cards/HX-04.md)) |
| `DS4_HELPER_IN=copy2/copy1/kfetch/zc` | How a helper window gets its input rows. HIP default `kfetch` (the helper's kernel fetches them). | helper window; MI50 | by construction (data movement only) | default `kfetch` (HIP) | see HX-06 ([HX-06](cards/HX-06.md)) |
| `DS4_HELPER_OUT=d2h/zc` | How a helper window returns its rows. HIP default `zc` (zero-copy into pinned host rows). | helper window; MI50 | by construction | default `zc` (HIP) | see HX-06 ([HX-06](cards/HX-06.md)) |
| `DS4_HELPER_WAIT=sync/query/flag` | How the lead waits for a helper window. HIP default `flag`. | helper window; lead | by construction | default `flag` (HIP) | see HX-06 ([HX-06](cards/HX-06.md)) |
| `DS4_HELPER_GRAPH=0/1` | Runs each helper window as a captured graph. HIP default 1; needs an input mode other than `copy2`. | helper window; MI50 | by construction | default 1 (HIP) | see HX-06 ([HX-06](cards/HX-06.md)) |
| `DS4_HELPER_WEIGHTS=w,w,...` | Relative shares of the expected picks for each helper in the layer placement. | load (placement); MI50 | changes placement only | off | not measured (INFERRED from code) |
| `DS4_HELPER_ARMED_MS=<ms>` | With `DS4_HELPER_ARMED`: how long an armed window waits for its command; default 50 ms. | helper window; MI50 | by construction | off | see HX-09 ([HX-09](cards/HX-09.md)) |
| `DS4_HELPER_DD_MAXT=<n>` | With `DS4_HELPER_DD`: windows of more than n rows keep the lane's path; default 16 (the window limit). | helper window; lead | by construction | off | not measured on its own (INFERRED from code) |
| `DS4_HELPER_DD_DELAY_US=<us>` | With `DS4_HELPER_DD`: an injected delay in the device-direct window (test). | helper window; MI50 | n/a (test) | test only | n/a |
| `DS4_HELPER_DEFER_MAXT=<n>` | With `DS4_HELPER_DEFER`: windows of more than n rows do not defer; default 16. | helper window; lead | quality-gated (as `DS4_HELPER_DEFER`) | off | see HX-11 ([HX-11](cards/HX-11.md)) |
| `DS4_PIN_LANE=0` | Leaves the CPU lane's expert parts pageable instead of pinned (A/B for the eviction stalls). | whole step; host | by construction | default 1 (pinned) | not measured (INFERRED from code) |
| `DS4_MALLOC_PIN_MB=<MiB>` | With `DS4_MALLOC_PIN`: the heap top pad, 0..1024; default 64. | whole step; host | by construction | off | see EB-05 ([EB-05](cards/EB-05.md)) |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm's own variable. The engine sets it to 0 (host memory as GTT) before the first HIP call unless it is already set or `DS4_HOST_GTT=0`. | load; host | by construction | set to 0 by default | see EB-05 ([EB-05](cards/EB-05.md)) |
| `DS4_WARM_WAIT=0` | Lets load return before the warm-up has finished (default 1: load waits, so the first prompt never does). | load; all devices | by construction | default 1 | not measured (INFERRED from code) |
| `DS4_MIX_TUNED=0` | Ignores `STRATA_ROCBLAS_TUNING` for the mHC mix GEMM. | attention entry and prefill; every XTX | the untuned GEMM ([FG-04](cards/FG-04.md)) | default on when a table is given | see FG-04 ([FG-04](cards/FG-04.md)) |
| `DS4_PF_HELPER_TOL=<f>` | With `DS4_PF_HELPER`: tolerance of the load-time check of each helper's prompt path against the CPU lane (relative L2); default 0.03. | load; MI50 | n/a (a check) | off | n/a |
| `DS4_ROUTER_CHECK=0` | Skips the load-time check that the BF16 router gives the F32 copy's bits; default on. | load; every XTX | n/a (a check) | default on with `DS4_ROUTER_BF16` | see FG-02 ([FG-02](cards/FG-02.md)) |
| `DS4_ROUTER_BF16_PAD=1` | Identity tests: keeps the VRAM the BF16 router saves in an unused allocation, so the expert placement is the F32 copy's. | load (placement); every XTX | makes `DS4_ROUTER_BF16` runs comparable byte for byte | test only | n/a |
| `DS4_TOPK_MB_MIN=<n>` | With `DS4_TOPK_MB`: the fewest scores for the many-workgroup top-k; default 8192. | attention, indexer layers; every XTX | measured identical (as `DS4_TOPK_MB`) | default 8192 | see FG-10 ([FG-10](cards/FG-10.md)) |
| `DS4_TP_ROWDMA=1` | The old per-row 2D DMA for the MXFP4 halves of non-resident experts (A/B against the contiguous copies). | prefill; every XTX | by construction | off | see TP-02 ([TP-02](cards/TP-02.md)) |
| `STRATA_DS4_HELPER=<dev[,dev]>`, `STRATA_DS4_HELPER_SLOTS=<n>` | Environment forms of `--helper-device` and `--helper-slots`, used when the options are not given. | load; MI50 | n/a | off | n/a |
| `STRATA_DS4_SHARED_STORE=<name, dir, path or 1>` | One RAM copy of a GGUF's routed experts shared by several engines on one box. | load; host | by construction (same bytes) | off | see HX-03 ([HX-03](cards/HX-03.md)) |
| `STRATA_DS4_SHARED_STORE_DROP=1` | The last engine to leave removes the shared segment (default: it outlives the engines). | exit; host | n/a | off | n/a |
| `STRATA_DS4_SHARED_STORE_POPULATE=0` | Maps the shared store without building page tables at once (default: built at load). | load; host | by construction | default on | not measured (INFERRED from code) |
| `STRATA_DS4_SHARED_STORE_WAIT=<s>` | How long an engine waits for another one's fill of the shared store; default 1800 s. | load; host | n/a | default 1800 | n/a |
| `STRATA_POOL_CORE=c[,c]` | Pins the n-th shared-pool helper server thread to the n-th core. | serving; host | by construction | off | see HX-03 ([HX-03](cards/HX-03.md)) |
| `STRATA_CPU_SLOTS=<n>`, `STRATA_CPU_SLOT=<i>` | Several engines on one host: each takes the lowest free CPU slot (a lock file in `/dev/shm`) and narrows itself to that slot's cores; `STRATA_CPU_SLOT` picks one. | load; host | by construction | off | not measured (INFERRED from code) |
| `DS4_ROUND_LOG_MS=1` | With `DS4_ROUND_LOG`: each drafting round's line also gets its draft and verify ms and the drafts' confidences. | drafter; host | n/a (log) | off (diagnostic) | n/a |
| `DS4_TOKLOG_X=<x>` | With `DS4_TOKLOG=2`: tokens slower than x times the median are listed with wall-clock times; default 5. | decode; host | n/a (log) | off (diagnostic) | n/a |
| `STRATA_DS4_DIAG=1` | Start-up diagnostics of devices, code objects and memory (the multi-GPU "invalid kernel file" hunt). | load; all devices | n/a (diagnostic) | off | see EB-04 ([EB-04](cards/EB-04.md)) |
| `STRATA_DS4_DIAG_RESERVE_MIB=<n>` | With `STRATA_DS4_DIAG`: leaves n MiB more free on each GPU. | load; all devices | n/a (diagnostic) | off | n/a |
| `STRATA_DS4_DIAG_SERIAL_PREFLIGHT=1` | With `STRATA_DS4_DIAG`: probes the devices one at a time with a sync between. | load; all devices | n/a (diagnostic) | off | n/a |
| `STRATA_DSV41_ARGS`, `STRATA_DSV41_CTX`, `STRATA_DSV41_PREFIX`, `STRATA_DSV41_WINDOW` | Inputs of `tools/dsv41/test_prompt_skip.py` (engine arguments, context, prefix and window lengths; defaults `--tp 1 --threads off`, 6144, 4200, 128). | test harness | n/a | test only | n/a |

## 3. What's next, by stage

The ranked list is in [ANATOMY.md](ANATOMY.md) §5 ("What's next, after build 12"). By stage:

- **Attention:** the next exact fusions (an int8 tail for the first output GEMV), and the Engram chain on a second XTX
  stream, which would also help verify windows.
- **Launches everywhere on the lead:** a short measurement of graph replay's per-kernel cost decides between graphs and a
  persistent kernel.
- **Helper window and join:** an exact re-tile of the MI50 kernels tops out at about −0.1 ms ([HX-23](cards/HX-23.md));
  what would move the join is fewer instructions per weight (a format change, which moves the placement) or one dispatch
  fewer a window. Two attempts at the latter on 14 Nov (one persistent launch, then a hybrid that keeps today's kernels for
  live work) were exact but slower once two or more experts are active in a window: not shipped
  ([UPDATE-2026-11-14.md](UPDATE-2026-11-14.md)).
- **Verify rows (drafter):** 3.11 ms a row, the largest production lever; no design yet ([DR-06](cards/DR-06.md)).
- **Prefill:** `DS4_PF_HELPER` is measured and fast at 1-8K-token suffixes, but not bit-identical; its validation (a Q8 reference, several depths, task outcomes) comes before any default change ([PF-11](cards/PF-11.md)).
- **Every build:** a code-size check so expert placement cannot move silently ([QA-08](cards/QA-08.md)).

**Pending analysis.** Three studies are in progress and will be folded into this table when they report: a re-test of
shelved fixes whose bottleneck may now be gone (the rows marked shelved or parked above), a second look at the MI50
expert kernels (section 2.5), and a study of the XTX-side kernels (sections 2.2-2.4).

**Update, 13 Nov.** The MI50 kernel study reported (HX-24, HX-25: one exact one-row gain, kept in the code; the
multi-row levers are spent) and so did the XTX kernel study (FG-08, FG-13, FG-14: one launch fold kept; bench wins on
the expert kernels did not reach the engine). What moved the served numbers was elsewhere: the long-context indexer
(FG-10, FG-11), the Engram chain (FG-09), the MI50 prompt tiles (PF-12) and concurrency (DR-07). Next, by stage:

- **Indexer:** the length-dispatched score kernel passed its engine A/B (−0.135 ms a one-row token); in build 16 ([FG-12](cards/FG-12.md)).
- **Decode fusions:** the five planned items are worth about 3–4 % a token, the batched attention path first ([FG-15](cards/FG-15.md), [FG-16](cards/FG-16.md)); the batched path's item was measured on 14 Nov: identical output, −6 % a batched step, +6.4 % with 8 served users, not in this release ([FG-16](cards/FG-16.md)); the real gap between kernels is about 2.9 µs, so fewer launches have a ceiling of about 3 ms a token ([FG-17](cards/FG-17.md)).

**Update, 14 Nov (build 16, the build in this release).** `DS4_XK_QA` −0.208 ms and `DS4_ATTN_MERGE=3` −0.157 ms a one-row token, both output-identical ([FG-16](cards/FG-16.md)); `DS4_IXS_DISPATCH` −0.165 ms with four helpers ([FG-12](cards/FG-12.md)); `DS4_PF_HELPER_MAXP=8192`, the helper only up to 8K-token prompts: served 32K 22.2 → 18.4 s ([PF-13](cards/PF-13.md)); `DS4_EC_FIX=1`, the expert-cache fix, which also ends a double count of expert picks ([HX-26](cards/HX-26.md), [HX-28](cards/HX-28.md)). Each was measured alone on a pre-16 build; build 16's re-validation is [SV-16](cards/SV-16.md). Switched off one at a time inside build 16, the three decode changes cost +0.15 to +0.17 ms each, and all five off +0.44 ms a token (12.47 → 12.89 ms, single process, MEASURED); served, one user is still slower than on build 15 because of the cache fix, and the cause is measured ([SV-16](cards/SV-16.md), [UPDATE-2026-11-14.md](UPDATE-2026-11-14.md)). Not in this release: `DS4_PHASE_TIMES=1`, untraced phase stamps ([FG-17](cards/FG-17.md)); `DS4_TREE_STATS`, which closed tree speculation ([DR-09](cards/DR-09.md)).
- **Verify rows:** still the largest production lever (2.66-2.70 ms a row); a profiler whose cost does not depend on the
  row count is needed before a design ([DR-06](cards/DR-06.md) addendum). Tree speculation waits on one counting run
  ([DR-09](cards/DR-09.md)).
- **Drafter policy:** block 5 / confidence 0.4 did not confirm on build 15; the policy stays block 3 / 0.3 ([DR-08](cards/DR-08.md)).
- **Prompts with several users:** prompts are read one at a time, which now limits multi-user latency more than decode
  ([SV-11](cards/SV-11.md)); the prompt-time expert-cache leak, and the double count it caused, are fixed in build 16 by
  `DS4_EC_FIX=1` ([HX-26](cards/HX-26.md), [HX-28](cards/HX-28.md)); served, the prompt helper is slower than none from 16K
  up, which build 16's cap avoids ([PF-13](cards/PF-13.md)).
- **Reliability:** two prompt-read hangs, cause open; the self-reporting build is in use ([SV-12](cards/SV-12.md)).
