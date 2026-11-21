# Qwen3.8 Flash-Next on RX 7900 XTX + MI50

This folder records what we changed in Gary ([github.com/msw-yst/gary](https://github.com/msw-yst/gary), our fork of Strata,
branch `flashnext-amd`) to serve Qwen3.8 Flash-Next to several agents at once on a mix
of AMD consumer and datacenter cards, and what each change did when measured. It sits beside the DeepSeek V4.1 Flash work in
[docs/dsv41](../dsv41/README.md); both ship from this repository.

Built by YST. Contact: https://yst.dev

**Start here:**
- [JOURNEY.md](JOURNEY.md): what we did, in order, and why.
- [ANATOMY.md](ANATOMY.md): where the time goes in a decode window, tp2 against one card, the decode kernels, the CPU pool's wait.
- [SHARED-POOL.md](SHARED-POOL.md): two XTX workers sharing one MI50 expert pool, design and results.
- [PREFILL.md](PREFILL.md): prompt reading.
- [QUANT.md](QUANT.md): the calibrated Q4_K file and how it scores against Q8_0.
- [AGENT-TURNS.md](AGENT-TURNS.md): how fast each agent's stream must be.
- [cards/](cards/): one card per change or finding (index below); [data/](data/): sanitised run extracts.

## What it is

Flash-Next is a mixture-of-experts model: 48 layers (36 gated-delta-net, 12 sparse attention), 512 routed experts a layer
(24,576 in all), top-10 routing, a shared expert, per-layer n-gram embeddings and an MTP drafter. In our 4-bit file its experts
alone are 73 GB. The work here puts all of them in GPU memory on one machine and serves several agents from it.

The fork's Flash-Next changes, all in Strata's generic engine (`strata`), on top of the DeepSeek branch's HIP backend and gfx906
helper build:

- **Tensor parallel over two XTX** (`--tensor-parallel 2`), with MI50 expert helpers.
- **MI50 helpers** on one card or under tensor parallel: up to 5 helper GPUs, a fast helper round trip (kernel fetch / push /
  flag, graph replay, one issue thread per helper).
- **A shared MI50 expert pool** for two one-card workers: one exports its MI50 caches by HIP IPC handle, the other imports them.
- **Batched serving fixes**: penalties in batch windows, batch self-commit, lent experts computed on the GPU during a prompt read.
- **Kernels**: multi-row Q6_K matvecs under TSUM (B6), IQ4 lookup by `v_perm` on gfx1100, router rank.
- **Prompt reading**: tuned hipBLASLt rows for the experts' gate/up GEMM, a concurrent read mode.
- **CPU pool**: a predicted wake between layers instead of spinning.

Every new path is opt-in unless a card or the [Settings](#settings) table says it is the default.

## Hardware

| Label | Card | Notes |
|---|---|---|
| XTX A, XTX B | RX 7900 XTX 24 GB (gfx1100) | XTX A is tensor-parallel card 0 and pool worker 1 |
| MI50 #1-#4 | Instinct MI50 16 GB (gfx906), 135 W cap | MI50 #3's PCIe transfers cost 2.4-2.9x the others' ("the slow-link MI50"); its kernels are as fast |

Host: one EPYC 7642 (48 cores, Zen 2), 256 GB DDR4-2666 (8 channels). Software: Strata v0.2.9 plus this fork; ROCm 7.2 (the
runtime in Strata's ROCm container image); the shared pool needs the ROCm 7.2.1 HSA runtime (below).

## Layouts

| Layout | Cards | What runs where | Best for |
|---|---|---|---|
| 1. One XTX + MI50 helpers | 1 XTX + 2 MI50 (two such servers per machine) | dense layers and the hottest experts on the XTX, ~45 % of experts in the MI50 caches, the rest (~1/3) on the CPU | 1-2 agents per server; long agent prompts (batched prompt path, prompt cache, MTP) |
| 2. Tensor parallel 2 | 2 XTX + 4 MI50 | every layer split over the two XTX; whole experts owned per card; the MI50s hold every other expert; CPU path 0 | short-context work; no MTP and no batched prompt path yet |
| 3. Shared MI50 pool | 2 XTX + 4 MI50, two workers | each worker is layout 1 on its own XTX; both share one pool of experts on all four MI50s; CPU path 0 | the most agents per machine |

## Build

The engine is the generic `strata` binary, built for gfx1100 with gfx906 code in every HIP translation unit (so the MI50s can
serve as helpers). The configure is the DeepSeek branch's ([tools/dsv41/build.sh](../../tools/dsv41/build.sh)); on a Linux
machine with ROCm, CMake 3.24+ and Ninja:

```sh
TGTS="strata strata-device" bash tools/dsv41/build.sh            # build dir: build/dsv41, logs in build/dsv41/logs
# optional, CPU-only checks of the new code:
TGTS="pool_plan_test" bash tools/dsv41/build.sh && build/dsv41/pool_plan_test
```

`GGML=<llama.cpp checkout>` builds offline; otherwise CMake fetches the pinned llama.cpp commit. Extra targets used in the
studies: `tp_proof_bench`, `b6_mmvq_bench`, `pool_wait_stress`, `pool_ipc_smoke`.

To run it in Strata's container, package the build as an engine component and build an image from it, as
[docs/DOCKER.md](../DOCKER.md#build-it-yourself) describes (`packaging/package_engine.py --backend hip ...`, then
`docker/build.sh --flavor hip --engine <package folder>`). The commands below assume that image as `strata-gary:rocm`.

## Launch

These are the settings we ran with, written out as `docker run` commands. They follow [docs/DOCKER.md](../DOCKER.md); card
numbers are HIP ordinals (`strata-device --list-devices`). The engine needs the main XTX at ordinal 0, so list it first in
`HIP_VISIBLE_DEVICES`. The commands themselves have not been run as written: the studies used an equivalent private launcher.

**Model.** `MODEL_PATH` = a folder with our calibrated Q4_K GGUF shards ([QUANT.md](QUANT.md)). We served it through the
Unsloth family entry (`FAMILY=unsloth MODEL=UD-IQ4_XS`), which picks the folder's only `-00001-of-` file, with the shipped pack's
`chat_template.jinja`.

**Environment common to every layout** (each is a card; all are off in the code unless stated):

```sh
COMMON="-e STRATA_HELPER_FAST=1 -e STRATA_HELPER_GRAPH=1 -e STRATA_HELPER_THREADS=1 -e STRATA_REMOTE_ZEROCOPY=0 \
  -e STRATA_BATCH_SELF_COMMIT=1 -e STRATA_TSUM=1 -e STRATA_B6_MMVQ_ROWS=1 -e STRATA_POOL_WAIT=pred \
  -e STRATA_ATTN_LANECELL=1 -e STRATA_ATTN_MERGE_V2=1 -e STRATA_EXPDEC=1 -e STRATA_GR_DOWN_TG=1 -e STRATA_XTX_FAST_PERM=1 \
  -e STRATA_ROUTE_TAIL_SKIP=0 -e STRATA_IQ_MT_MIN=1"
SET="pcie-frac=0 adapt-swaps=0 spec=2 suffix-draft=0 batch-groups=1"
GPU="--device /dev/kfd --device /dev/dri --group-add video --group-add $(getent group render | cut -d: -f3) \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120"
```

`STRATA_HELPER_CPUS=<one CPU per helper>` gives each helper issue thread its own core; on two servers per machine give each
server its own CPUs (`--cpuset-cpus`, and `host-core` / `pool-cores` in `STRATA_SET`).

**Layout 1: one XTX + two MI50 helpers** (one server; run a second on the other XTX and MI50 #3, #4):

```sh
docker run -d --name gary-1 $GPU -p 127.0.0.1:8080:8080 -v gary-data-1:/data -v /models:/models:ro \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=4 \
  -e HIP_VISIBLE_DEVICES=<XTX A>,<MI50 #1>,<MI50 #2> -e GPU_ROLES=0=main,1=helper,2=helper \
  -e STRATA_SET="$SET expert-cache-device1=auto expert-cache-device2=auto" \
  $COMMON -e STRATA_HELPER_CPUS=<c1>,<c2> -e STRATA_BATCH_MTP=1 \
  -e STRATA_BATCH_CONCURRENT_READ=2 -e STRATA_BATCH_LENT_GPU=1 \
  strata-gary:rocm
```

For agents keep the prompt cache on (the default); our speed tables used `prompt-cache=0` because the client repeats prompts.
`STRATA_NO_MTP=1` turns the drafter off.

**Layout 2: tensor parallel 2 + four MI50 helpers:**

```sh
docker run -d --name gary-tp2 $GPU -p 127.0.0.1:8080:8080 -v gary-data-tp:/data -v /models:/models:ro \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=8 -e GPUS=0 \
  -e STRATA_TP_HIP_VISIBLE=<XTX A>,<XTX B>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e STRATA_SET="$SET tensor-parallel=2 expert-cache-device1=auto expert-cache-device2=auto expert-cache-device3=auto expert-cache-device4=auto" \
  $COMMON -e STRATA_HELPER_CPUS=<c1>,<c2>,<c3>,<c4> \
  strata-gary:rocm
```

Under tensor parallel the MTP drafter, the prompt cache, the batched prompt path and vision are off (the engine says so at
start). Batch self-commit is on by default.

**Layout 3: two workers sharing one MI50 pool.** Both workers in one pod that shares the network, PID and IPC namespaces (HIP IPC
on ROCm 7 passes a file descriptor over a socket in the exporter's network namespace). Both need the ROCm 7.2.1
`libhsa-runtime64.so.1` in place of the one bundled with the engine (the bundled runtime refuses IPC export). Take it from a
ROCm 7.2.1 installation (`lib/libhsa-runtime64.so.1` under the ROCm folder); we took ours from llama.cpp's b11870 ROCm container
image, which ships ROCm 7.2.1. Mount it over the engine's copy as below (`<hsa 7.2.1 lib>` = that file, `<engine folder>` = where
the image keeps the engine's libraries). Both use one shared host expert arena in `/dev/shm` (two private 68 GB arenas led to a
driver stall, [SHARED-POOL.md](SHARED-POOL.md) section 8).

```sh
podman pod create --name gary-pool --share net,pid,ipc -p 127.0.0.1:8080-8081:8080-8081
POOLSET="$SET expert-cache-device1=5000 expert-cache-device2=5000 expert-cache-device3=5000 expert-cache-device4=5000 \
  shared-expert-arena=/hostshm/gary-arena.bin"
MOUNTS="-v /dev/shm:/hostshm -v gary-pool:/pool -v /models:/models:ro -v <hsa 7.2.1 lib>:<engine folder>/libhsa-runtime64.so.1:ro"
# worker 1 (XTX A): fills the MI50 caches and exports them
podman run -d --pod gary-pool --name gary-w1 $GPU $MOUNTS -v gary-data-1:/data -e PORT=8080 \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=4 \
  -e HIP_VISIBLE_DEVICES=<XTX A>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e GPU_ROLES=0=main,1=helper,2=helper,3=helper,4=helper -e STRATA_SET="$POOLSET expert-cache=4700" \
  -e STRATA_POOL_EXPORT=/pool/manifest.bin -e STRATA_BATCH_LENT_GPU=1 $COMMON -e STRATA_HELPER_CPUS=<4 CPUs> \
  strata-gary:rocm
# worker 2 (XTX B): waits for the manifest, maps the same MI50 memory
podman run -d --pod gary-pool --name gary-w2 $GPU $MOUNTS -v gary-data-2:/data -e PORT=8081 \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=4 \
  -e HIP_VISIBLE_DEVICES=<XTX B>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e GPU_ROLES=0=main,1=helper,2=helper,3=helper,4=helper -e STRATA_SET="$POOLSET expert-cache=4700" \
  -e STRATA_POOL_IMPORT=/pool/manifest.bin -e STRATA_BATCH_LENT_GPU=1 $COMMON -e STRATA_HELPER_CPUS=<4 other CPUs> \
  strata-gary:rocm
```

Both workers must use the same `expert-cache` (the hot prefix) and MI50 cache size. Each prints
`dp pool (exporter|importer): CUDA0 <n>, pool <n>, CPU path <n> of 24576` at start; check that the CPU path is 0. Sizes we ran,
in experts, for our Q4_K file (every expert 2.97 MB): 4 slots per worker hot 4,700 / MI50 5,000; 8 slots hot 3,700 / MI50 5,300;
with MTP (`STRATA_BATCH_MTP=1`) hot 4,300 / MI50 5,100. Spread clients evenly over ports 8080 and 8081.

With the shipped UD-IQ4_XS these counts do not apply: its experts are smaller and of mixed size (~2.46 MB on average), so the
same memory holds more of them. Layouts 1 and 2 size themselves (`auto`). For layout 3 (not run with that file): start each XTX
once as a layout-1 server with `auto` caches and note the XTX expert count each reports; use the smaller one as `expert-cache`
on both workers, give each MI50 cache a count that fits its card, and confirm `CPU path 0` at start (the hot prefix plus the
four MI50 caches must cover all 24,576 experts).

## Settings

| Switch | Default | Effect | Card |
|---|---|---|---|
| `--tensor-parallel 2` / `STRATA_TP=2` | off | tensor parallel over two XTX | [NX-01](cards/NX-01.md) |
| `STRATA_TP_HIP_VISIBLE=<list>` | unset | the device list the engine sees under tensor parallel (replaces `HIP_VISIBLE_DEVICES`) | NX-01 |
| `expert-cache-device1..4` (`STRATA_SET`) | - | MI50 helper caches (`auto` or a slot count) | [NX-05](cards/NX-05.md) |
| `STRATA_HELPER_FAST` / `_GRAPH` / `_THREADS` | off | fast helper round trip | [NX-06](cards/NX-06.md) |
| `STRATA_HELPER_CPUS=<list>` | unset | one CPU per helper issue thread | NX-06 |
| `STRATA_BATCH_SELF_COMMIT` | on under tp2, off on one card | batch windows advance their slots themselves | [NX-03](cards/NX-03.md) |
| penalties in batch windows | on | batched requests get the same penalties as solo | [NX-08](cards/NX-08.md) |
| `STRATA_B6_MMVQ_ROWS=1` (`STRATA_B6_MMVQ_MIN`, 7) | off | multi-row Q6_K matvecs, also under `STRATA_TSUM=1` | [NX-09](cards/NX-09.md) |
| `STRATA_XTX_FAST_PERM=1` | off | IQ4_NL / IQ4_XS lookup by `v_perm` on gfx1100; bitwise identical | [ANATOMY.md](ANATOMY.md) section 5 |
| `STRATA_ROUTE_RANK=1` | off | router rank kernel | - |
| packaged hipBLASLt rows (`STRATA_PF_GU_LEGACY=1` for the old kernel) | on | the experts' prompt GEMM on matrix cores; the exactness reference | [NX-10](cards/NX-10.md) |
| `STRATA_BATCH_CONCURRENT_READ=2` | off | read a prompt beside the decoding slots | [NX-11](cards/NX-11.md) |
| `STRATA_BATCH_LENT_GPU=1` | off | lent experts on the GPU during a read: decode beside a read stays exact | NX-11 |
| `STRATA_PF_DQ_BATCH=G` | off | batched FP16 conversion (exact, slower) | [NX-12](cards/NX-12.md) |
| `STRATA_POOL_WAIT=spin\|futex\|nap\|pred` | spin | CPU pool wait between layers | [NX-13](cards/NX-13.md) |
| `STRATA_POOL_EXPORT` / `STRATA_POOL_IMPORT=<manifest>` | unset | shared MI50 pool | [NX-14](cards/NX-14.md) |
| `STRATA_DISJOINT_ADAPT=0\|1` | **on whenever a helper cache is set** (changed here; it was opt-in before) | the adaptive tier never promotes an expert a helper GPU already holds; `=0` restores the old behaviour | [DISJOINT_EXPERT_CACHE.md](../DISJOINT_EXPERT_CACHE.md) |
| `STRATA_POOL_VERIFY=N` | 16 | pool slots byte-checked per card on import | NX-14 |
| `STRATA_POOL_WEIGHTS="<pci-bus-id>=<w>,..."`, `STRATA_POOL_COVER_B` (1.7) | unset | weighted pool placement (not used in the measured runs) | [SHARED-POOL.md](SHARED-POOL.md) |
| `STRATA_NO_MTP=1` / `STRATA_BATCH_MTP=1` | - | MTP drafter off / drafter in batch windows | [NX-15](cards/NX-15.md) |
| `STRATA_TP_TIMING=1\|2\|3`, `STRATA_HELPER_TIMING=1` | off | timing builds (they change the graph; 1.10-1.21x slower) | [ANATOMY.md](ANATOMY.md) |

## How to read the numbers

- **Labels.** MEASURED = read from a run file or a script-written summary; REPORTED = from a write-up at the time, not re-checked
  against raw output; ESTIMATE = our arithmetic; INFERRED = our reading of the code.
- **Exactness first.** Greedy decode (temperature 0). A configuration is exact when its token ids equal the single-card
  reference for the same prompt, batched == solo, a loaded server == the same server alone, and repeat runs are identical. Only
  then do its speed numbers count. A change that alters rounding (another kernel, another device for an expert, another
  summation order) re-baselines the reference only by a deliberate decision, recorded on its card. The current reference for the
  256-token prompt with our Q4_K file is `a7ae6623797f17fe` (sha256 prefix of the text); the tuned prompt GEMM rows set new
  references for the long prompts ([NX-10](cards/NX-10.md)).
- **Method.** One GPU measurement at a time; arms alternated within a session (A B A B); bars and abort lines written down
  before the run where the card says so.
- **Speed.** Per agent = median over users of streamed chunks per second; aggregate = total tok/s (wall clock, or while every
  stream is active, as labelled). Short-prompt runs: 8 fixed prompts, 384 tokens out, temperature 0, presence penalty 0.
- **Prompts.** The 8-prompt set is private; the long prompts are slices of source code. Ratios between arms of the same run
  compare like with like; absolute numbers are indicative.
- **Data.** [data/](data/) holds sanitised extracts of the run summaries with `SHA256SUMS`. The originals, per-cell logs and
  traces are kept privately.
- **Scope.** All results are for this machine, this model and these files. They may not transfer to other boards, PCIe
  topologies, ROCm versions or quantisations.

## Results

Per-agent decode tok/s, calibrated Q4_K, short prompts (384 tokens out), temperature 0 [MEASURED]:

| Layout | 1 user | 2 users | 4 users | 8 users | 16 users | Source |
|---|---|---|---|---|---|---|
| tp2 + 4 MI50 (before NX-09) | 50.5 | 40.6 | 31.6 | 18.8 | 18.0 (8 slots + queue) | [comparative timing](data/2026-11-21-comparative-timing.md) |
| one XTX + 2 MI50 (one server, before NX-09) | 50.5 | 37.9 | 24.6 | 13.5 | - | same |
| one XTX + 2 MI50, MTP (before NX-09) | 76.3 | 49.3 | 26.7 | 13.2 | - | same |
| two one-card servers, users split evenly | 51.5 | 47.5 | 36.0 | 24.0 | 13.4 (8 slots each) | [pool runs](data/2026-11-21-dp-pool.md) |
| **two workers + shared MI50 pool** | **54.4** | **52.3** | **39.8** | **28.2** | **17.1** (8 slots each) | same |
| shared pool, MTP | **86.5-87.0** | **83.1** | not measured | not measured | - | same |

- **Runtime confound.** Every pool cell ran an engine build with the ROCm 7.2.1 HSA runtime swapped in (export needs it); every
  baseline cell ran the build with the bundled runtime. The pool / baseline ratios in this table therefore include any difference
  between the two runtimes and builds, not only the pool. A baseline on the same runtime is being measured; these numbers will be
  updated when it is in.
- The pool and its baseline were alternated in one session at 4-16 users (3 + 3 trials at 4 slots per worker, 2 + 2 at 8; ranges
  do not overlap): pool / baseline 1.106x at 4 users, 1.175x at 8, 1.272x at 16. The 1- and 2-user cells are a separate test
  (one trial each without MTP, two with). The two-servers row's 1-user cell is one server alone; its 2-user cell one user on
  each.
- The tp2 and one-server rows come from the comparative timing study, a different session, before B6 under TSUM
  ([NX-09](cards/NX-09.md)); compare them with the pool rows loosely. With NX-09 on (the decode A/B of the same day,
  [data](data/2026-11-21-decode-ab.md)) only cells with 7-8 rows a window change: tp2 at 8 users 19.4 per agent, 169.4 aggregate;
  one server at 8 users 15.3; one server with MTP 27.5 at 4 users and 13.6 at 8. The pool rows were measured with NX-09 on.
- Aggregate at 16 users: pool 296 tok/s, two servers 229. Host RAM: pool 124 GiB, two servers 190 GiB; busy cores ~13 against
  32-40.
- Agent turns with ~50k-token contexts (one server, MTP): 74-75 tok/s alone; prefix reused in 3.6-4.5 s; a cold 52k prompt in
  36 s with the tuned rows ([PREFILL.md](PREFILL.md)).
- Our per-agent target for thinking-on coding agents is 40 tok/s ([AGENT-TURNS.md](AGENT-TURNS.md)): the pool meets it for about
  four agents (39.8, at the line), tp2 and a single server for two.

Exactness: every row above gives the reference text `a7ae6623797f17fe` on its solo prompt except one XTX + 2 MI50 with MTP,
which gave `da29bc7d5a0ba433`: the drafter takes 513 XTX expert slots, so a different set of experts runs on that server's CPU
path, and the CPU rounds differently from the GPUs (MTP itself was not separated from that). The pool has no CPU path and gives `a7ae6623797f17fe` with and without MTP.
Batched == solo on every server.
Two separate one-card servers do not give each other's texts at 384 tokens (different experts on each one's CPU path); the two
pool workers do (8/8).

## Limitations and open items

- **Tensor parallel**: no MTP drafter, no batched prompt path and no prompt cache (every prompt goes through 8-token decode
  windows; 1,631 tokens took 19-25 s), no vision. Not usable for long agent prompts yet.
- **Build from this tree not yet verified**: the published `flashnext-amd` tree has not yet been built end to end from the
  exported tree itself; that build check is pending.
- **Pool vs baseline runtime**: the pool's ratios mix the pool with a runtime / build difference (pool on the ROCm 7.2.1 HSA
  runtime, baseline on the bundled one); a same-runtime baseline is being measured.
- **Shared pool**: needs the ROCm 7.2.1 HSA runtime and one pod with shared network / PID / IPC namespaces; needs one shared host
  expert arena; the exporter's death stops both workers. Placement tuning (slow-link MI50 weight, a hot set from agent traffic)
  and MTP at more than one user per worker are not measured. The cause of the first trial's driver stall is likely
  memory-pressure eviction but not proven.
- **Long prompts serialise**: four cold 50k prompts at once still take ~150-160 s for the last first token.
- **Exactness corners**: with penalties on, tp2's solo text differs from one card's (pre-existing); a rare solo near-tie in long
  streams (one flip in 7 launches of a 2,500-token stream); the earlier one-card MTP reference `da29bc7d5a0ba433` is no longer
  reproduced in today's environment and was not re-baselined.
- **Kernels left**: 8-row Q6_K matvecs (mostly the lm head) at 300-450 GB/s against 450-600 at 4 rows; hc read, router and shared
  expert at 8 rows; ~2,300-3,650 small kernels per window; Q5_0 down has no decode-once multi-row kernel.
- **Quality**: KL against Q8_0 only; task-level quality (agent benchmarks, tool calling) on the optimised engine is not run. The
  Q4_K recipe's calibration text and exact command lines are not yet published ([QUANT.md](QUANT.md)).
- **No public re-run yet**: the short-prompt set is private; a public prompt set and a published launcher example
  (`docs/examples/`) are to come.
- **Hardware**: one machine; one MI50 with a slow PCIe link; host RAM, DRAM bandwidth and CPU threads limit a third server.

## Card index

| card | title | status | correctness | headline |
|---|---|---|---|---|
| [NX-01](cards/NX-01.md) | Tensor parallel 2 on two RX 7900 XTX | kept (opt-in) | 256/256 tokens = one card | exact from cycle 1; 0.72-0.85x one card at first |
| [NX-02](cards/NX-02.md) | Batching under tensor parallel, device-planned dispatch | kept | batched == solo (presence 0) | 1.28-1.38x one card at 2-8 users |
| [NX-03](cards/NX-03.md) | Batch self-commit | kept (default under tp2) | exact | tp2 +3 %; one card +5 / +6 % at 4 / 8 users |
| [NX-04](cards/NX-04.md) | Missed experts over PCIe during the CPU wait | dropped | **not exact** | +4-5 % at 4-8 users, rejected |
| [NX-05](cards/NX-05.md) | MI50s hold every expert the XTX cards do not | kept | exact | +26 % at 4 users, +37 % at 8 |
| [NX-06](cards/NX-06.md) | Fast MI50 helper round trip | kept | 96/96 repeat; exact | +3.4-5.8 %; helper wait 8.5 -> 2.3 ms |
| [NX-07](cards/NX-07.md) | Calibrated Q4_K fits at the same speed | kept | reference `a7ae6623797f17fe` | 50.2 vs 49.4 tok/s at 1 user |
| [NX-08](cards/NX-08.md) | Penalties in batch windows | kept (default) | batched == solo 0/8 -> 8/8 | no cost |
| [NX-09](cards/NX-09.md) | B6 multi-row matvecs under TSUM | kept (opt-in) | bitwise equal | +4.6-6.6 % at 8 users |
| [NX-10](cards/NX-10.md) | Tuned hipBLASLt rows for the prompt GEMM | kept (default; re-baselined) | new reference | cold 52k prompt 50.7 -> 36.3 s |
| [NX-11](cards/NX-11.md) | Concurrent prompt read, lent experts on the GPU | kept (opt-in) | 16/16 == alone | stall 7.8 -> 0.5 s; read 62.6 -> 41.5 s |
| [NX-12](cards/NX-12.md) | Expert streams, batched FP16 conversion | dropped | exact | 1.3-14 % / 2.7-6.1 % slower |
| [NX-13](cards/NX-13.md) | CPU pool sleeps between layers (pred) | kept (opt-in) | every text identical | 17-20 cores freed, speed within noise |
| [NX-14](cards/NX-14.md) | Two XTX workers sharing one MI50 pool | kept (opt-in) | 8/8 on every check | 1.106x / 1.175x / 1.272x per agent at 4 / 8 / 16 users |
| [NX-15](cards/NX-15.md) | Shared pool at one user, with and without MTP | finding | MTP changes no bit | 83-87 tok/s per user with MTP (1.11-1.17x) |
| [NX-16](cards/NX-16.md) | XTX power cap | finding | n/a | 272 W costs nothing in decode |
