# Two XTX workers sharing one MI50 expert pool

Date: 2026-11-21. Model: Qwen3.8 Flash-Next (48 layers x 512 experts = 24,576 routed experts), our calibrated Q4_K file
([QUANT.md](QUANT.md)).
Hardware: EPYC 7642 (48 cores, Zen 2), 256 GB DDR4, 2 x RX 7900 XTX 24 GB (XTX A, XTX B), 4 x Instinct MI50 16 GB
(one of them, "the slow-link MI50", has PCIe transfers that cost 2.4-2.9x the others). Software: ROCm 7.2 / 7.2.1,
Strata v0.2.9 + Gary (this fork).

Numbers are labelled **MEASURED** (read from a run file or a script-written summary), **REPORTED** (from a write-up at
the time, not re-checked against raw output) or **ESTIMATE** / **INFERRED** (our own arithmetic or reading of the code).
Run data: [data/2026-11-21-dp-pool.md](data/2026-11-21-dp-pool.md). Timing model and coverage data:
[data/2026-11-21-comparative-timing.md](data/2026-11-21-comparative-timing.md). Cards: [NX-14](cards/NX-14.md),
[NX-15](cards/NX-15.md), [NX-16](cards/NX-16.md). How to launch it: [README.md](README.md#launch), layout 3.

## 1. Summary

- **The idea.** Run two data-parallel engine processes, one per XTX. Each keeps its own dense weights, KV slots and a
  hot-expert cache on its XTX. Instead of each owning two MI50s and sending the rest of the experts to the CPU, both
  share **one pool of experts on all four MI50s**. One process fills the MI50 caches and exports them as HIP IPC memory
  handles. The other maps the same VRAM read-only. Each process drives the MI50s with its own helper threads, streams
  and graphs, exactly as a single-card server does today.
- **Fit.** With both XTX caching the same hot prefix of the expert ranking, and every other expert on exactly one MI50,
  nothing goes to the CPU at 2, 4 or 8 slots per worker, with or without MTP. The pool needs 19,208-21,305 experts and
  four MI50s hold 22,344 [ESTIMATE].
- **Exact.** Both workers give the single-card reference hash. Batched == solo, loaded == alone and worker 1 == worker 2
  were all 8/8 [MEASURED].
- **Faster.** Per agent, against two separate single-card servers measured in the same session: **1.106x at 4 users**
  (bar 1.10, met), **1.175x at 8 users** (bar 1.20, missed; abort line 1.05, clear), and **1.27x at 16 users** with 8
  slots per worker [MEASURED].
- **One user, with MTP.** One user per worker with the MTP drafter: 83-87 tok/s per user, 1.11-1.17x the baseline; in the
  pool, MTP changes no bit of the output [MEASURED].
- **Cheaper on the host.** Host RAM 124 GiB against 190 GiB, and about 13 busy cores against 32-40 [MEASURED].
- **Two runtime conditions.** Export needs the ROCm 7.2.1 HSA runtime. The importing process must share the exporter's
  network and PID namespace (we use one podman pod with `--share net,pid,ipc`).
- **Next.** Placement tuning: a slower-card weight, a hot set ranked from agent traffic, and slots vs hot-cache size.

## 2. Facts the design rests on

| Fact | Value | Label |
|---|---|---|
| Expert size in the Q4K pack | 2,969,600 B in all 48 layers (Q4_K gate 921,600 + Q4_K up 921,600 + Q5_0 down 1,126,400; the 640-column down falls back to Q5_0 in every layer). Attention and linear-attention layers have identical experts | MEASURED (quantize log, types file) |
| MI50 capacity | 5,586 experts = 15.45 GiB each, with a 512 MiB guard; 4 cards = 22,344 | MEASURED (engine logs) |
| XTX cache, single-card server | XTX A: 5,302 experts at 4 slots, 4,248 at 8. XTX B: 4,840 at 4 (1.27 GiB less free VRAM than XTX A; cause unknown) | MEASURED |
| KV slot cost | 0.73 GiB per `--batch` slot (max context 131,072, int8 KV, 32,768 resident) = 264 experts | MEASURED |
| MTP cost | -443 experts at 4 slots, -513 at 8 | MEASURED |
| Expert profile | `data/expert-profile.bin` is a ranking only, with no counts | INFERRED (tools/make_profile.py) |
| Coverage of the ranking in decode | top 4,248 ranks carry 28-31 % of routed entries; 4,840 carry 30 %; 5,302 carry 36 %; 15,420 carry 78-85 %. On agent turns 4,859 ranks carry 46-52 %. Fit H(f) = 1 - (1 - f)^b gives b = 1.7 (1.6-1.9) for short prompts, ~2.4 at the top for agent turns | MEASURED counts; fit ESTIMATE (`tools/pool_fit.py`) |
| MI50 service time (timing build, graphs off) | normal card 44.9 us per launch + 10.2 us per entry; the slow-link MI50 48.9 + 19.2. Its kernels are as fast; its PCIe in/out costs 2.4-2.9x | MEASURED; fit ESTIMATE |
| Host RAM | one single-card server 113-120 GB; two side by side 190 GB (of 251); expert arena 68 GB pinned per process | MEASURED |
| Per-agent speed before this work (1/2/4/8 users) | tp2 (T): 50.5 / 40.6 / 31.6 / 18.8 tok/s. Single-card server (S): 50.5 / 37.9 / 24.6 / 13.5. Two S servers side by side at 4+4 users: 21.7-23.3 | MEASURED ([comparative timing](data/2026-11-21-comparative-timing.md)) |

The coverage numbers say the shipped ranking predicts this workload only weakly: the top 17 % of ranks carry 28-31 % of
the routed entries. A ranking built from our own traffic should sharpen the hot set (section 9).

Today's single-card server keeps about 17 % of the experts on its XTX, 45 % on its two MI50s and **37 % on the CPU**.
That 37 % of experts is 19-22 % of routed entries and costs 5-25 ms per decode window, in series with the GPU [MEASURED].
The pool exists to remove that CPU share without paying tp2's per-layer split.

## 3. Fit arithmetic

All rows come from `python3 tools/pool_fit.py` (b = 1.7; `--b 1.6` and the agent-turn curve give the range in brackets).
Percentages are shares of one worker's routed entries in decode. "Hot" = the ranks every worker keeps on its own XTX.
Capacities at 2 slots, and XTX B at 8 slots, are extrapolated by 264 experts per slot [ESTIMATE].

### (a) Both XTX cache the same hot set; the pool holds every other rank (chosen)

The pool must hold everything the smaller XTX cache does not, so hot = min(XTX A, XTX B).

| Slots per worker | XTX caches A / B | hot (share of experts) | pool needs (of 22,344) | XTX local | pool | CPU |
|---|---|---|---|---|---|---|
| 2 | 5,830 / 5,368 | 5,368 (21.8 %) | 19,208 (slack 3,136) | 34.2 % (32.6-44.6) | 65.8 % | 0 |
| 4 | 5,302 / 4,840 | 4,840 (19.7 %) | 19,736 (slack 2,608) | 31.1 % (29.6-40.9) | 68.9 % | 0 |
| 8 | 4,248 / 3,784 | 3,784 (15.4 %) | 20,792 (slack 1,552) | 24.7 % (23.5-33.1) | 75.3 % | 0 |
| 4 + MTP | 4,859 / 4,397 | 4,397 (17.9 %) | 20,179 | 28.5 % | 71.5 % | 0 |
| 8 + MTP | 3,735 / 3,271 | 3,271 (13.3 %) | 21,305 (slack 1,039) | 21.6 % | 78.4 % | 0 |

A variant (a') lets XTX A keep its whole, larger cache. Its worker's local share rises from 31.1 to 33.8 % at 4 slots.
The cost is that the two workers no longer share a placement, so the same prompt can give a different text on each
worker. We kept the same hot set as the default.

### (b) Disjoint hot sets on the two XTX, each reaching the other's

The union of the two caches is 10,142 experts at 4 slots. Each worker would compute 28-31 % locally, 28-31 % on the
other XTX and 40.5 % in the pool.

- Fetching weights from the other XTX is too slow. One expert is 2.97 MB. Even at ~25 GB/s a peer read takes 120 us or
  more, against ~10 us per entry on an MI50.
- Computing on the other XTX puts that card on this worker's critical path every layer, while it is busy with its own
  dense work. That is tp2's coupling again.
- It only helps if the pool saturates, and with two workers it does not (section 6). We rejected it for two workers.

### (c) The pool also covers what neither XTX holds; the coldest overflow goes to the CPU

With four MI50s there is never an overflow. Overflow appears only with three MI50s (for example a box with four XTX):

| Slots per worker, 3 MI50 | pool needs (of 16,758) | overflow (coldest) | CPU share of entries |
|---|---|---|---|
| 2 | 19,208 | 2,450 (10.0 % of experts) | 2.0 % |
| 4 | 19,736 | 2,978 (12.1 %) | 2.8 % |
| 8 | 20,792 | 4,034 (16.4 %) | 4.6 % |
| 8 + MTP | 21,305 | 4,547 (18.5 %) | 5.7 % |

The placement puts exactly the coldest ranks on the CPU. That is 50-170 CPU entries per window per worker, against
365-760 for one single-card server today [ESTIMATE].

## 4. Architecture

### Three ways to build it

| | A. One process, two in-process workers | B. Two processes + a pool service that issues the MI50 work | C. Two processes sharing pool VRAM by IPC handles (chosen) |
|---|---|---|---|
| What runs where | one engine, two schedulers, each owning an XTX | each worker is today's single-card engine; a pool process owns the MI50s and computes for both; workers post inputs through shared memory | each worker is today's single-card engine and opens the four MI50s itself; one cache per MI50 is filled once and shared by `hipIpcMemHandle`; each worker runs its own helper threads, streams and graphs on it |
| Request path | today's | one extra cross-process hop | today's, unchanged |
| Engine work | large: the serve loop in `main()` is ~17,000 lines of closures and statics and would have to become a re-entrant worker object; one control channel and one prompt-admission lock serialise the two workers | moderate: a shared-memory transport and a pool binary | small: an expert cache over an imported base, an import path in the helper setup, an export step, a manifest |
| Failure blast radius | one fault stops both | pool death stops both | the exporter's death stops both; the importer's death leaves the exporter serving |
| Main risk | refactor size, re-entrancy bugs | the extra hop | HIP IPC on gfx906 / ROCm 7.2 in a container; two processes' queues on one MI50 |
| 3-4 workers | threads | can merge concurrent requests into one launch | no merging; each worker launches its own |

We chose C, with B as the fallback and as the likely path to four or more workers. The deciding points:

- C keeps the helper request path that cut the host helper pass from 14.1 to 6.8 ms per window
  (`STRATA_HELPER_FAST`, `STRATA_HELPER_GRAPH`, `STRATA_HELPER_THREADS`) exactly as it is.
- Each worker is literally today's single-card binary with a fixed placement, so "a worker equals itself running alone"
  is easy to argue and to test.
- `--shared-expert-arena` (one copy of the expert arena in host RAM for several engines) already existed.
- A is the only option that needs the serve loop re-architected.

### How two workers reach the same MI50

- Per worker and per MI50 there is one helper queue: pinned staging, device scratch and metadata, a non-blocking stream,
  a flag word, a helper-graph cache and an issue thread. With two workers that is eight issue threads; each worker gets
  its own CPU set (`STRATA_HELPER_CPUS`).
- Two processes give two sets of hardware queues on each MI50. The firmware scheduler dispatches work from both. The
  expert kernels are small (3-15 entries), latency-bound and use 7-14 % of HBM bandwidth, so they mostly overlap rather
  than queue [INFERRED].
- Each process captures its own helper graphs. The only thing shared is the read-only expert weights.
- There is no ordering between workers and none is needed: the pool VRAM is written once, before the importer attaches.

## 5. Exactness

Greedy (temperature 0) decode. A configuration is **exact** when its token ids equal the single-card reference for the
same prompts (reference hash of the 256-token prompt `a7ae6623797f17fe`), batched == solo, and repeat runs are identical.
A placement change that alters rounding (XTX vs MI50 vs CPU kernels) re-baselines the reference only deliberately.

Why a pool worker's output should equal that worker running alone [INFERRED from the code]:

- **Static placement.** Which device computes each (layer, expert) is fixed before the workers start, from the ranking
  and the configuration only, never from load (`include/strata/core/pool_plan.hpp`). An entry goes to the same device
  kind whether or not the other worker is busy.
- **Same launch contents.** A worker's launches carry only its own rows, grouped by its own window, so a launch is the
  same alone or beside the other worker. The expert kernels use no float atomics. Batched == solo and repeat identity
  on tp2 already showed that per-row results do not depend on the other rows [MEASURED].
- **Same bytes.** On import, each worker byte-compares sampled pool slots with its own copy of the blobs before serving
  (`STRATA_POOL_VERIFY`, default 16 slots per card).
- **Same hot set on both workers.** Both compute every expert on the same device kind, so the same prompt gives the same
  text on either worker, and a conversation that moves between workers keeps its bits.
- tp2 and the single-card server (with 22 % of entries on the CPU) both give the reference hash, so a pool worker was
  expected to give it too. It did (section 8).

An open shortcut: if one Q4K expert is bitwise identical on gfx1100 (XTX) and gfx906 (MI50), no XTX/MI50 placement
choice can ever change a text. We have not tested that.

## 6. Contention estimate

Per worker window and per MI50, option (a), from the measured service times [ESTIMATE; `tools/pool_fit.py`]:

| Users per worker | window | pool entries per MI50 | busy per worker, normal card | busy, slow-link MI50: equal share -> weight 0.58 | other worker's load on a card |
|---|---|---|---|---|---|
| 2 | ~22 ms | 165 | 3.8 ms | 5.5 -> 4.2 ms | ~0.17-0.19 |
| 4 | ~28 ms | 331 (369 weighted) | 5.5 (5.9) ms | 8.7 -> 6.4 ms | ~0.21 |
| 8 | ~42 ms | 722 (807) | 9.6 (10.4) ms | 16.2 -> 11.3 ms | ~0.23-0.25 |

The service model comes from timing builds with graph replay off, so these busy times are upper bounds.

- **If a card ran one launch at a time (upper bound):** a launch finds the card busy with the other worker with
  probability ~0.21 and waits half a launch on average, ~50-60 us. A layer waits for the slowest of its four cards, so
  P(some card busy) = 1 - 0.79^4 = 0.61, about +40 us per layer. That is at most 1.9 ms per 28 ms window at 4 users
  per worker (7 %), 1.3 ms (6 %) at 2 and 2.5-3 ms (7 %) at 8.
- Only part of the helper pass is visible today, and launches from two processes overlap on the card, so we expected
  0.5-1.5 ms (2-5 %). The IPC smoke test later measured two processes' saturated launches at 1.18x one process's
  throughput, so they do overlap partly [MEASURED].
- **The slow-link MI50.** Its kernels are as fast as the others; its transfers are not. Giving it weight 0.58 (16 % of
  the pool's routed mass: the coldest ~3,200-4,000 experts) equalises its round trip with the others and saves ~60 us
  per layer [ESTIMATE]. The switch exists (`STRATA_POOL_WEIGHTS`); the measured runs below used the plain stripe, and the
  weight is the first tuning step.
- **Expected speed [ESTIMATE].** Remove the single-card server's CPU wait and add a tp2-like visible helper wait: about
  28 ms per window at 4 users per worker (~31 tok/s per agent, against 21.7-23.3 for two single-card servers), ~22 ms at
  2 users (43-45 tok/s), ~42 ms at 8 (~21 tok/s). Section 8 compares these with what we measured.
- MI50 duty rises from 10-25 % busy to roughly 30-50 %, so the MI50 thermal watcher stays on.

## 7. What was built, and the switches

All of it is opt-in: with no `STRATA_POOL_*` variable set, the engine behaves as before.

| Piece | Where |
|---|---|
| Placement and manifest: hot prefix on the XTX, every other rank to exactly one MI50, balanced by routed mass per unit of device weight while room lasts, coldest ranks overflow; deterministic. Manifest written to a temporary file, then renamed | `include/strata/core/pool_plan.hpp`; test target `pool_plan_test` |
| Expert cache over another process's base (no allocation, no fill, never freed) | `open_sized_external` in `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp` |
| Import: `hipIpcOpenMemHandle`, the same slot layout over it, admit the manifest's pairs, byte-check sampled slots, then the usual helper setup | `src/core/remote_experts.cpp` |
| Export after the fill, import before the helper setup, the per-engine split line | `src/program/generate.cpp` |
| IPC smoke test (no engine) | `src/kernels/pool_ipc_smoke.cpp` |
| Fit arithmetic for sections 3 and 6 | `tools/pool_fit.py` |

The design also sketched a separate pool-holder binary and a pool option in the server front. Neither exists: the first
engine is the holder, and the client split load evenly over the two workers.

Switches:

| Switch | Effect |
|---|---|
| `STRATA_POOL_EXPORT=<manifest>` | after filling its MI50 caches, this engine exports them as IPC handles and writes the manifest |
| `STRATA_POOL_IMPORT=<manifest>` | this engine waits (up to 1,800 s) for the manifest, checks the model geometry, maps each MI50 cache and runs its own helpers on it. Every helper card must be in the manifest |
| `STRATA_POOL_VERIFY=N` | slots byte-checked per card on import (default 16) |
| `STRATA_POOL_WEIGHTS="<pci-bus-id>=<w>,..."` | replaces the round-robin stripe with mass-weighted placement (cards not named get weight 1) |
| `STRATA_POOL_COVER_B=<b>` | exponent of the coverage curve used for the routed mass in weighted placement (default 1.7) |
| `--shared-expert-arena FILE` | backs the host expert arena with one shared file, so the second engine finds it "already filled by another engine". Needed in practice (section 8) |

Both workers set the same `--expert-cache` (the hot prefix) and the same MI50 cache size. Each startup prints
`dp pool (exporter|importer): CUDA0 <n>, pool <n>, CPU path <n> of 24576`.

Environment used on both workers in the measured runs (and on the baseline): `STRATA_HELPER_FAST=1`,
`STRATA_HELPER_GRAPH=1`, `STRATA_HELPER_THREADS=1`, `STRATA_HELPER_CPUS` (a different CPU set per worker),
`STRATA_BATCH_SELF_COMMIT=1`, `STRATA_TSUM=1`, `STRATA_B6_MMVQ_ROWS=1`, `STRATA_POOL_WAIT=pred`, `STRATA_NO_MTP=1`, and
the tuned prompt-reading rows. The design also requires the adaptive expert tier to stay off (the pool is written once)
and `STRATA_BATCH_LENT_GPU=1` for exact decode beside a prompt read.

### Two runtime conditions

Found with the IPC smoke test on gfx906 (a holder exports 16 synthetic Q4_K/Q5_0 experts; clients map them, run the
grouped expert kernel over 7 of them and bit-compare against a private copy) [MEASURED]:

| Setup | Result |
|---|---|
| The HSA runtime bundled with the engine build | export refused: `hsa_amd_ipc_memory_create` returns `HSA_STATUS_ERROR_INVALID_ARGUMENT` at every size tried (12 MB - 1.2 GB), also with `HSA_ENABLE_IPC_MODE_LEGACY=1` |
| The ROCm 7.2.1 HSA runtime, alone or under the engine's own `libamdhip64` | export works |
| Exporter and importer in one container | import works; imported == private bytewise, stable after the concurrent phase |
| Separate containers with their own network namespaces | import fails (`invalid device context`, or `invalid device pointer` after a 10.8 s timeout): ROCr 7's DMA-buf IPC passes the file descriptor over a socket in the exporter's network namespace and needs its PID namespace |
| Separate containers sharing host network and PID, or **one podman pod with `--share net,pid,ipc`** | import works, bitwise equal. The pod is what we use |

Concurrency (one container, 7 experts, 1 token, back to back): one client 90.9 us p50 per launch + sync; two clients at
once 153.8 us p50 each (p99 173 us). So two launches take 153.8 us, 1.18x the throughput of one. That is the saturated
case; in service each worker keeps a card only 15-25 % busy.

## 8. Pre-registered bars and results

Baseline for every bar, measured fresh in the same session and alternated with the pool: two separate single-card
servers (XTX A + MI50 #1, #2; XTX B + the slow-link MI50 + MI50 #4), same environment, full CPU expert threads, MTP off.
Load: the standard 8-prompt set, 384 tokens, temperature 0, presence penalty 0, split evenly over the two servers.
Per agent = median over users of streamed chunks per second.

**Bars (written before the runs):** per agent at least **1.10x** the baseline at 4 users and **1.20x** at 8 users;
4 agents (2+2) at 40 tok/s or more; above tp2 at 8 users (21.2 tok/s per agent, REPORTED). **Abort:** below 1.05x at
8 users after one tuning pass; the other worker's load costing a worker more than 15 % (or more than 3 ms of helper wait
per window); or any exactness failure.

### First trial and the stall

- Both workers loaded with CPU path 0 (4,700 experts on each XTX, 19,876 in the pool), each gave the reference hash,
  and worker 1 == worker 2 [MEASURED].
- One trial at 4 users: 38.85 tok/s per agent and 166.2 aggregate, against the baseline's 36.60 / 153.4: **+6.1 % per
  agent, +8.3 % aggregate** (bar not met on that one run) [MEASURED].
- On the next cell the importing worker's first request made no progress for 60 s. The engine watchdog restarted it,
  and the restarted engine hung again at its first request. The kernel log showed KFD queue eviction ("queue evicted")
  whose restore never completed, then hung-task reports for kernel workers blocked in `svm_range_restore_work` /
  `amdgpu_amdkfd_restore_userptr_worker` (6 reports, 122-245 s) [MEASURED]. We stopped the cell; the stop was clean,
  with no reset. At that point each worker had its own private 68 GB pinned arena, with 190 GiB of host RAM in use.

### Retries with one shared arena

We reran with one shared expert arena (`--shared-expert-arena`), so only one 68 GB registration exists, and with a
kernel-log watch that stops the cell on any queue-eviction, hung-task, reset or ring-timeout line.

- **First retry.** Both workers loaded with CPU path 0; the second found the arena "already filled by another engine";
  host RAM 124 GiB instead of 190; reference hash on both [MEASURED]. The cell stopped at the client's 30 s no-progress
  watch. The cause was not IPC: XTX B's memory temperature had reached the thermal guard's pause limit (94 C), and the
  guard paused that worker for 30 s, exactly the watch's threshold. The kernel log had no line from the abort list
  [MEASURED]. We then set the pause limits to 100 C junction / memory (warning from 97 C) and made the no-progress
  watch ignore paused time.
- **Second retry (10 cells, about 58 min of pool cells): the pool holds.** No abort-list line in the kernel log, no stall,
  no reset, no thermal pause [MEASURED]. That fits the first stall being memory-pressure eviction with two private
  arenas rather than an IPC fault [INFERRED].

### Final results (second retry)

Pool: one pod, one shared arena. 4 slots per worker: hot 4,700, MI50 caches 5,000 each. 8 slots per worker: hot 3,700,
MI50 caches 5,300 each. CPU path 0 of 24,576 on both workers in every pool cell. Mean of trials (3 + 3 at 4 slots,
2 + 2 at 8 slots); pool and baseline ranges do not overlap [MEASURED].

**Exactness (8/8 everywhere):** reference `a7ae6623797f17fe` on both workers; batched == solo 4/4 and 8/8 on each
worker, alone and loaded; loaded == alone 8/8 on both; worker 1 == worker 2 8/8 (the baseline gives 2/8, because its
two servers have different placements) [MEASURED].

| Slots / users | Pool per agent | Baseline per agent | Ratio | Pool aggregate | Baseline aggregate | Ratio | Bar |
|---|---|---|---|---|---|---|---|
| 4 slots, 4 users | 39.82 (39.47-40.00) | 35.99 (35.48-36.26) | **1.106** | 166.5 | 152.8 | 1.089 | 1.10: met |
| 4 slots, 8 users | 28.23 (27.72-28.70) | 24.04 (23.71-24.27) | **1.175** | 241.1 | 205.1 | 1.176 | 1.20: missed (abort 1.05: clear) |
| 8 slots, 8 users | 27.73 | 23.62 | 1.174 | 236.5 | 194.3 | 1.217 | - |
| 8 slots, 16 users | 17.08 | 13.43 | **1.272** | 296.1 | 229.1 | 1.292 | - |

| Other measurement | Pool | Baseline | Label |
|---|---|---|---|
| Host RAM, 4 / 8 slots | 124 / 137 GiB | 190 / 202 GiB | MEASURED |
| Busy CPU cores | about 13 | 32-40 | MEASURED |
| MI50 board power | 59-93 W | 45-62 W | MEASURED (the pool now runs the experts the baseline ran on the CPU) |
| Spread between the two workers at 4 users | within 2 % | baseline server 2 is 3-7 % slower than server 1 | MEASURED |

Against the other bars: 4 agents reached 39.82 tok/s on average, a hair under the 40 tok/s bar (best trial 40.00).
At 8 users the pool's 28.2 tok/s is well above tp2's 21.2. Against the estimates in section 6 the pool came in 10-20 %
below prediction at every level: 39.8 vs 43-45 tok/s at 2 users per worker, 28.2 vs ~31 at 4, 17.1 vs ~21 at 8.
We did not measure the cross-worker helper-wait histogram separately; the one-user test below gives the cost of the
other worker's load directly.

### One user per worker, MTP off

One user on worker 1 with worker 2 idle, the same on worker 2, then one user on each at once. Pool: hot 4,700, MI50
caches 5,000, CPU path 0. Baseline: two single-card servers with automatic cache sizes (16,473 / 16,011 experts on the
GPUs, the rest on the CPU). 384 tokens, temperature 0, one trial each [MEASURED].

| 1 user | Pool tok/s | Baseline tok/s | Ratio |
|---|---|---|---|
| on worker 1 (XTX A), the other idle | 54.35 | 51.48 | 1.056 |
| on worker 2 (XTX B), the other idle | 53.59 | 50.31 | 1.065 |
| one on each, per user | 52.26 | 47.48 | 1.101 |
| one on each, aggregate | 104.5 | 95.0 | 1.100 |

- Reference hash on both pool workers; worker 1 == worker 2 3/3 on the speed prompts (baseline 1/3) [MEASURED].
- A user beside the other worker's user loses 3.6 % on the pool and 7.8 % on the baseline, where the two servers' CPU
  expert paths share DRAM bandwidth [MEASURED]. That is far inside the 15 % abort line.
- About 6 busy cores per active pool worker against about 15 per baseline server [MEASURED].

### One user per worker, MTP on

The same workload with the MTP drafter (`STRATA_BATCH_MTP=1`) on every server. Pool: hot 4,300, MI50 caches 5,100 (the
drafter costs ~443 XTX slots), CPU path 0. Baseline: MTP on, automatic caches. Alternating pool / baseline, two trials each
[MEASURED]:

| 1 user, MTP on | Pool tok/s | Baseline tok/s | Ratio |
|---|---|---|---|
| on worker 1 (XTX A), the other idle | 86.52 | 78.21 | 1.106 |
| on worker 2 (XTX B), the other idle | 86.95 | 74.42 | 1.168 |
| one on each, per user | 83.13 | 72.47 | 1.147 |
| one on each, aggregate | 166.3 | 145.0 | 1.147 |

- MTP's own gain at one user on worker 1: pool 1.59x, baseline 1.52x; drafts accepted ~90 % on these prompts. Trials agree
  within 0.6 % [MEASURED].
- Pool exact with MTP: reference hash on both workers; batched == solo, loaded == alone and worker 1 == worker 2 all 8/8; the
  texts equal the MTP-off pool's, so in the pool MTP changes no bit of the output [MEASURED].
- The baseline with MTP also gives the no-MTP reference `a7ae6623797f17fe` in this environment. An earlier one-card MTP arm
  on an older engine and environment gave `da29bc7d5a0ba433`; that is not reproduced and was not re-baselined.

### Side measurement: XTX power cap

On two single-card servers at 4 and 8 users, caps of 402 W (maximum), 303 W (default) and 272 W (the driver's floor)
gave the same decode speed within 0.3 % aggregate (24.4-24.6 tok/s per agent at 8 users). Uncapped, each XTX drew
~15-20 W more on average for the same speed [MEASURED]. Prompt reading, which is compute-bound, was not tested.

## 9. Open questions and next steps

Next: **placement tuning**, one step at a time, A/B against the current pool, each kept only if it gains at least 2 % at
8 users. Steps that change placement re-run the full exactness set, and a changed reference hash is accepted only
deliberately.

1. Weight 0.58 for the slow-link MI50 vs 1.0 (`STRATA_POOL_WEIGHTS`; expected +2-4 % at 8 users) [ESTIMATE].
2. A ranking from served agent-like traffic instead of the shipped profile. We have collected one from worker 1 on 48
   coding-agent prompts (21,376 generated tokens); agent turns already show 46 % local at the same cache size, so we
   expect a local share of 40 % or more and +3-6 % [ESTIMATE].
3. Own hot set per worker (a') vs the same hot set.
4. Planning the expert dispatch on the device, as tp2's first card does, to skip the 2.5-3.0 ms per window of host-plan
   waits [ESTIMATE]. This must be bit-identical to before.
5. Slots per worker vs hot-cache size (264 experts per slot; a smaller resident context buys hot cache).

Open questions:

- Is a Q4K expert bitwise equal on XTX and MI50? If yes, placement can never change a text.
- Why does XTX B have 1.27 GiB less free VRAM than XTX A? It sets the common hot prefix (~460 experts, ~2.7 % of entries).
- Which workload should the hot set be tuned for: short prompts or agent turns?
- MTP on the pool workers at more than one user per worker: the pool fits with MTP at every slot count (section 3), and
  one user per worker is measured (above), but 4-16 users with MTP are not.
- Three or four workers. With 3 XTX + 4 MI50 the pool still fits and host RAM (~157 GB with one arena) fits too
  [ESTIMATE]. With 4 XTX + 3 MI50, per-launch overhead would saturate the cards without merging concurrent requests into
  one launch (design B) [ESTIMATE].
- Whether the first stall was memory-pressure eviction (likely) or an IPC / userptr interaction is not proven; it has
  not recurred with one shared arena.
