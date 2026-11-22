# Qwen3.8 Flash-Next on RX 7900 XTX + MI50

This folder records what we changed in Gary ([github.com/msw-yst/gary](https://github.com/msw-yst/gary), our fork of Strata,
branch `flashnext-amd`) to serve Qwen3.8 Flash-Next to several agents at once on a mix
of AMD consumer and datacenter cards, and what each change did when measured. It sits beside the DeepSeek V4.1 Flash work in
[docs/dsv41](../dsv41/README.md); both ship from this repository.

Built by YST. Contact: https://yst.dev

**Start here:**
- [JOURNEY.md](JOURNEY.md): what we did, in order, and why.
- [ANATOMY.md](ANATOMY.md): where the time goes in a decode window, tp2 against one card, the decode kernels, the CPU pool's wait.
- [SHARED-POOL.md](SHARED-POOL.md): two XTX workers sharing one MI50 expert pool, design and results; section 10 has the
  second day (MTP under load, host memory, how many agents it serves, run-to-run exactness under concurrency).
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
- **A shared MI50 expert pool** for two one-card workers: one exports its MI50 caches by HIP IPC handle, the other imports them;
  its host expert arena on 2 MiB hugetlb pages.
- **Batched serving fixes**: penalties in batch windows, batch self-commit, lent experts computed on the GPU during a prompt read
  (and a guarantee that none goes to the CPU), adaptive MTP (no drafter work above N live slots), and a reproducible-run mode
  (every request in a batch slot, continuations only from checkpoints).
- **Kernels**: multi-row Q6_K matvecs under TSUM (B6), IQ4 lookup by `v_perm` on gfx1100, router rank.
- **Prompt reading**: tuned hipBLASLt rows for the experts' gate/up GEMM, a concurrent read mode.
- **CPU pool**: a predicted wake between layers instead of spinning.

Every new path is opt-in unless a card or the [Settings](#settings) table says it is the default. The hugetlb arena, the
lent-expert guarantee, adaptive MTP and the reproducible-run mode were measured on our development branch and are not yet in
this tree (marked † in [Settings](#settings)).

## Hardware

| Label | Card | Notes |
|---|---|---|
| XTX A, XTX B | RX 7900 XTX 24 GB (gfx1100) | XTX A is tensor-parallel card 0 and pool worker 1 |
| MI50 #1-#4 | Instinct MI50 16 GB (gfx906), 135 W cap | links Gen4 x16, Gen4 x8, Gen3 x8, Gen3 x8. Until 22 November MI50 #3's link trained at Gen1 x8 and its PCIe transfers cost 2.4-2.9x the others' ("the slow-link MI50"); a reseat fixed it ([NX-18](cards/NX-18.md)) |

Host: one EPYC 7642 (48 cores, Zen 2), 256 GB DDR4-2666 (8 channels). Software: Strata v0.2.9 plus this fork; ROCm 7.2 (the
runtime in Strata's ROCm container image); the shared pool needs the ROCm 7.2.1 HSA runtime (below).

## Host setup

What the 22 November results ran on, beyond a stock ROCm host. Each item says what it is for.

- **2 MiB huge pages for the pool's expert arena** ([NX-19](cards/NX-19.md)): `vm.nr_hugepages=35000` (persist it in
  `/etc/sysctl.d`; on a running host drop caches and compact memory first, then read back `HugePages_Total`) and a hugetlbfs
  mount with `pagesize=2M,size=70000M`. 35,000 pages hold one arena of our Q4_K file (34,803 pages). A GPU-registered arena on
  4 KB pages can have its pages moved by the kernel, and each move stops every GPU queue of the process; huge pages are never
  moved. Two private-arena servers need 70,000 instead, and a /dev/shm pool does not fit beside 70,000.
- **PCIe completion timeout of 1-3.5 s on every GPU root port.** Some of our ports defaulted to the 65-210 ms range. We set
  the Device Control 2 timeout field on each GPU's root port, for example `setpci -s <root port> CAP_EXP+0x28.w=000a:000f`
  (value `0xa` = 1-3.5 s; the mask leaves the register's other bits alone). It does not survive a reboot, so apply it at boot.
  It changes how a slow completion is handled, not throughput; every result from 22 November on ran with it, and we have not
  measured its effect on its own.
- **MI50s pinned at DPM `high` while serving** ([NX-21](cards/NX-21.md)): `power_dpm_force_performance_level=high` on each
  MI50 after the workers load, `auto` when they stop. `auto` was 3.4 % slower on the pool at 16 users.
- **XTX power cap 272 W** (the driver's floor): no measurable cost in decode ([NX-16](cards/NX-16.md)).
- **Check every card's PCIe link** (`current_link_speed` / `current_link_width`, or `lspci -vv`) against what its root port
  supports, and **remove AC power before changing or reseating a card**: on some server boards a soft power-off keeps the
  slots powered on standby, and cards changed that way can fail to enumerate or train their links until the board is fully
  drained ([NX-18](cards/NX-18.md)).

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
TGTS="strata strata-device pool_plan_test" bash tools/dsv41/build.sh   # build dir: build/dsv41, logs in build/dsv41/logs
build/dsv41/pool_plan_test                                             # CPU-only check of the pool placement code
```

`GGML=<llama.cpp checkout>` builds offline; otherwise CMake fetches the pinned llama.cpp commit. Extra targets used in the
studies: `tp_proof_bench`, `b6_mmvq_bench`, `pool_wait_stress`, `pool_ipc_smoke`.

**Prompt-reading build option.** The engines we measured were configured with `STRATA_PREFILL_MMQ=ON` (the GGML MMQ prompt
path, which Strata's own HIP packages also enable); `tools/dsv41/build.sh` leaves it at CMake's default, OFF, because the
DeepSeek builds in [docs/dsv41](../dsv41/README.md) did not use it. To build what we measured, set it in the build folder
before building the engine (the script configures only once, so the setting stays):

```sh
TGTS="pool_plan_test" bash tools/dsv41/build.sh                        # configures build/dsv41
cmake -B build/dsv41 -DSTRATA_PREFILL_MMQ=ON
TGTS="strata strata-device" bash tools/dsv41/build.sh
```

The build gate on the published tree used the first form (MMQ off); the MMQ-on sequence has not been run as written.

To run it in Strata's container, package the build as an engine component and build an image from it, as
[docs/DOCKER.md](../DOCKER.md#build-it-yourself) describes (`packaging/package_engine.py --backend hip ...`, then
`docker/build.sh --flavor hip --engine <package folder>`). The commands below assume that image as `strata-gary:rocm`.

## Launch

These are the settings we ran with, written out as `docker run` commands. They follow [docs/DOCKER.md](../DOCKER.md); card
numbers are HIP ordinals of the whole machine (`strata-device --list-devices` with every card visible). `GPUS` names the main
XTX for setup; `STRATA_TP_HIP_VISIBLE` (main XTX first) is the list the engine sees. Do not set `HIP_VISIBLE_DEVICES` or
`GPU_ROLES` with MI50 helpers: the Terminal sets `HIP_VISIBLE_DEVICES` itself and its setup refuses gfx906 cards by number.

The studies ran every layout through an equivalent private launcher; each layout below says whether its command has been run as
written. On hosts where podman's AppArmor profile blocks the container runtime's socket, add `--security-opt apparmor=unconfined`.

**Model.** `MODEL_PATH` = a folder with our calibrated Q4_K GGUF shards ([QUANT.md](QUANT.md)). We served it through the
Unsloth family entry (`FAMILY=unsloth MODEL=UD-IQ4_XS`), which picks the folder's only `-00001-of-` file. Replace the template
the pack exports from the GGUF (it has `{% generation %}` tags, which the server refuses at load) with the Hugging Face
repository's `chat_template.jinja`: copy it to `/data/Strata-data/packs/unsloth-ud-iq4_xs/tokenizer/chat_template.jinja` in the
data volume after setup has made the pack, then start the container again. Setup makes the pack with `tools/iq_pack.py`, which
needs llama.cpp's `gguf-py` (setup points `STRATA_GGUF_PY` at its own llama.cpp copy; when you run `iq_pack.py` yourself, set
`STRATA_GGUF_PY=<llama.cpp checkout>/gguf-py`).

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

**Layout 1: one XTX + two MI50 helpers** (one server; run a second on the other XTX and MI50 #3, #4). Run as written on our
test box (podman, image built from this tree), one chat request answered:

```sh
docker run -d --name gary-1 $GPU -p 127.0.0.1:8080:8080 -v gary-data-1:/data -v /models:/models:ro \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=4 \
  -e GPUS=<XTX A> -e STRATA_TP_HIP_VISIBLE=<XTX A>,<MI50 #1>,<MI50 #2> \
  -e STRATA_SET="$SET expert-cache-device1=auto expert-cache-device2=auto" \
  $COMMON -e STRATA_HELPER_CPUS=<c1>,<c2> -e STRATA_BATCH_MTP=1 \
  -e STRATA_BATCH_CONCURRENT_READ=2 -e STRATA_BATCH_LENT_GPU=1 \
  strata-gary:rocm
```

For agents keep the prompt cache on (the default); our speed tables used `prompt-cache=0` because the client repeats prompts.
`STRATA_NO_MTP=1` turns the drafter off.

**Layout 2: tensor parallel 2 + four MI50 helpers** (corrected, not run as written):

```sh
docker run -d --name gary-tp2 $GPU -p 127.0.0.1:8080:8080 -v gary-data-tp:/data -v /models:/models:ro \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=8 -e GPUS=<XTX A> \
  -e STRATA_TP_HIP_VISIBLE=<XTX A>,<XTX B>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e STRATA_SET="$SET tensor-parallel=2 expert-cache-device1=auto expert-cache-device2=auto expert-cache-device3=auto expert-cache-device4=auto" \
  $COMMON -e STRATA_HELPER_CPUS=<c1>,<c2>,<c3>,<c4> \
  strata-gary:rocm
```

Under tensor parallel the MTP drafter, the prompt cache, the batched prompt path and vision are off (the engine says so at
start). Batch self-commit is on by default.

**Layout 3: two workers sharing one MI50 pool** (corrected, not run as written). Both workers in one pod that shares the network, PID and IPC namespaces (HIP IPC
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
  -e GPUS=<XTX A> -e STRATA_TP_HIP_VISIBLE=<XTX A>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e STRATA_SET="$POOLSET expert-cache=4700" \
  -e STRATA_POOL_EXPORT=/pool/manifest.bin -e STRATA_BATCH_LENT_GPU=1 $COMMON -e STRATA_HELPER_CPUS=<4 CPUs> \
  strata-gary:rocm
# worker 2 (XTX B): waits for the manifest, maps the same MI50 memory
podman run -d --pod gary-pool --name gary-w2 $GPU $MOUNTS -v gary-data-2:/data -e PORT=8081 \
  -e FAMILY=unsloth -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models/flashnext-q4k -e PARALLEL=4 \
  -e GPUS=<XTX B> -e STRATA_TP_HIP_VISIBLE=<XTX B>,<MI50 #1>,<MI50 #2>,<MI50 #3>,<MI50 #4> \
  -e STRATA_SET="$POOLSET expert-cache=4700" \
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

**Layout 3, best-known configuration for agents** (what [NX-22](cards/NX-22.md) measured; it needs the switches marked † in
[Settings](#settings), which are not yet in this tree). Changes against the command above:

- the [host setup](#host-setup) above (huge pages, completion timeout, MI50 pin, link check);
- the arena on hugetlbfs: mount the hugetlbfs folder into the pod instead of `/dev/shm` and point `shared-expert-arena=` at a
  file in it;
- 8 slots per worker (`PARALLEL=8`), `STRATA_BATCH_MTP=1 STRATA_BATCH_MTP_ADAPT=4`, hot `expert-cache=3150`, MI50 caches
  `5400` each;
- `STRATA_BATCH_LENT_EXACT=1` in place of `STRATA_BATCH_LENT_GPU=1` (it implies it);
- the prompt cache on (do not set `prompt-cache=0`), with `conversation-cache-mib=8192 conversation-cache-slots=8` on each worker
  (two workers would otherwise both size their cache from the same free RAM).

For reproducible runs (A/Bs, regression gates) add `STRATA_REUSE_CKPT_ONLY=1` to both workers and run the server with
`STRATA_PARALLEL_SLOTS_ONLY=1` ([NX-24](cards/NX-24.md)); it costs ~5 % of steady multi-user decode.

## Settings

Switches marked † were measured on our development branch and their code is not yet in this tree; until it is, setting them
here does nothing.

| Switch | Default | Effect | Card |
|---|---|---|---|
| `--tensor-parallel 2` / `STRATA_TP=2` | off | tensor parallel over two XTX | [NX-01](cards/NX-01.md) |
| `STRATA_TP_HIP_VISIBLE=<list>` | unset | the device list the engine sees (replaces the `HIP_VISIBLE_DEVICES` the Terminal sets; any layout, not only tensor parallel) | NX-01 |
| `expert-cache-device1..4` (`STRATA_SET`) | - | MI50 helper caches (`auto` or a slot count) | [NX-05](cards/NX-05.md) |
| `STRATA_HELPER_FAST` / `_GRAPH` / `_THREADS` | off | fast helper round trip | [NX-06](cards/NX-06.md) |
| `STRATA_HELPER_CPUS=<list>` | unset | one CPU per helper issue thread | NX-06 |
| `STRATA_BATCH_SELF_COMMIT` | on under tp2, off on one card | batch windows advance their slots themselves | [NX-03](cards/NX-03.md) |
| penalties in batch windows | on | batched requests get the same penalties as solo | [NX-08](cards/NX-08.md) |
| `STRATA_B6_MMVQ_ROWS=1` (`STRATA_B6_MMVQ_MIN`, 7) | off | multi-row Q6_K matvecs, also under `STRATA_TSUM=1` | [NX-09](cards/NX-09.md) |
| `STRATA_XTX_FAST_PERM=1` | off | IQ4_NL / IQ4_XS lookup by `v_perm` on gfx1100; bitwise identical | [ANATOMY.md](ANATOMY.md) section 5 |
| `STRATA_ROUTE_RANK=1` | off | router rank kernel | - |
| packaged hipBLASLt rows (`STRATA_PF_GU_LEGACY=1` for the old kernel) | on | the experts' prompt GEMM on matrix cores; the exactness reference | [NX-10](cards/NX-10.md) |
| `STRATA_BATCH_CONCURRENT_READ=2` | off | read a prompt beside the decoding slots (on the pool: +7-10 % per agent at 4-8 agents, but a client stalled at 16; up to 8 agents only) | [NX-11](cards/NX-11.md), [NX-23](cards/NX-23.md) |
| `STRATA_BATCH_LENT_GPU=1` | off | lent experts on the GPU during a read: decode beside a read measured exact on one card (16/16 == alone); on the pool use `STRATA_BATCH_LENT_EXACT=1` | NX-11 |
| `STRATA_PF_DQ_BATCH=G` | off | batched FP16 conversion (exact, slower) | [NX-12](cards/NX-12.md) |
| `STRATA_POOL_WAIT=spin\|futex\|nap\|pred` | spin | CPU pool wait between layers | [NX-13](cards/NX-13.md) |
| `STRATA_POOL_EXPORT` / `STRATA_POOL_IMPORT=<manifest>` | unset | shared MI50 pool | [NX-14](cards/NX-14.md) |
| `STRATA_DISJOINT_ADAPT=0\|1` | **on whenever a helper cache is set** (changed here; it was opt-in before) | the adaptive tier never promotes an expert a helper GPU already holds; `=0` restores the old behaviour | [DISJOINT_EXPERT_CACHE.md](../DISJOINT_EXPERT_CACHE.md) |
| `STRATA_POOL_VERIFY=N` | 16 | pool slots byte-checked per card on import | NX-14 |
| `STRATA_POOL_WEIGHTS="<pci-bus-id>=<w>,..."`, `STRATA_POOL_COVER_B` (1.7) | unset | weighted pool placement (not used in the measured runs) | [SHARED-POOL.md](SHARED-POOL.md) |
| `STRATA_NO_MTP=1` / `STRATA_BATCH_MTP=1` | - | MTP drafter off / drafter in batch windows | [NX-15](cards/NX-15.md) |
| `STRATA_BATCH_MTP_ROWS=R` † | 8 | the MTP verify window holds up to R rows (2-16); not kept | [NX-17](cards/NX-17.md) |
| `STRATA_BATCH_MTP_OFF_ABOVE=N` † | off | no MTP proposals in a window over more than N live slots (the drafters still catch up); not kept | NX-17 |
| `STRATA_BATCH_MTP_ADAPT=N` † (`STRATA_BATCH_MTP_ADAPT_HEAL_DIV`, 8) | off; **we run 4** on the 8-slot pool | above N live slots no drafter work at all; the drafters' missing K/V rebuilt in one batched pass when N or fewer are live | [NX-20](cards/NX-20.md) |
| `shared-expert-arena=<file on hugetlbfs>` † (`STRATA_SET`) | opt-in by path (a /dev/shm file takes the old code) | the shared host expert arena on 2 MiB huge pages; **our pool default** with `vm.nr_hugepages=35000` | [NX-19](cards/NX-19.md) |
| `power_dpm_force_performance_level` of each MI50 (sysfs) | driver `auto` | **we pin `high`** while serving | [NX-21](cards/NX-21.md) |
| `prompt-cache`, `conversation-cache-mib`, `conversation-cache-slots` (`STRATA_SET`) | cache on; size from free RAM | **on for agents**; on the pool set 8192 MiB / 8 slots per worker | [NX-22](cards/NX-22.md) |
| `STRATA_BATCH_LENT_EXACT=1` † | off; **recommended on the pool** | implies `STRATA_BATCH_LENT_GPU=1`; no lent expert is ever computed on the CPU (no window runs while one lacks a device alias), and the CPU's share is counted | [NX-23](cards/NX-23.md) |
| `STRATA_BATCH_DECODE_SHARE=F` | 0.5 | share of a read's time the decoding slots get between its parts; `0` was a diagnostic | NX-23 |
| `STRATA_REUSE_CKPT_ONLY=1` † | off | continue a conversation only from checkpoints, never from an idle slot's or a parked conversation's live end; reproducible runs only (with the next row) | [NX-24](cards/NX-24.md) |
| `STRATA_PARALLEL_SLOTS_ONLY=1` † (server) | off | with `--batch`, every request decodes in a batch slot, never on the solo path; reproducible runs only (~5 % steady cost) | NX-24 |
| `STRATA_PREFILL_POOL_HELP=1` | does not exist yet | planned MI50 prompt helper | [NX-25](cards/NX-25.md) |
| `STRATA_TP_TIMING=1\|2\|3`, `STRATA_HELPER_TIMING=1` | off | timing builds (they change the graph; 1.10-1.21x slower) | [ANATOMY.md](ANATOMY.md) |

## How to read the numbers

- **Labels.** MEASURED = read from a run file or a script-written summary; REPORTED = from a write-up at the time, not re-checked
  against raw output; ESTIMATE = our arithmetic; INFERRED = our reading of the code.
- **Exactness first.** Greedy decode (temperature 0). In these docs "exact" means what we tested: the token ids of the 256-token
  reference prompt equal the single-card reference, batched == solo, a loaded server == the same server alone, the two pool
  workers give each other's texts, and repeat runs of the short-prompt speed set are identical. Those prompts are short (~30
  tokens for the speed set), so they are read through the decode windows. Only exact configurations' speed numbers count. A
  change that alters rounding (another kernel, another device for an expert, another summation order) re-baselines the
  reference only by a deliberate decision, recorded on its card. The current reference for the 256-token prompt with our Q4_K
  file is `a7ae6623797f17fe` (sha256 prefix of the text); the tuned prompt GEMM rows set new references for the long prompts
  ([NX-10](cards/NX-10.md)).
- **Exactness caveat: long prompts under concurrency.** The checks above do not cover several agents with long prompts at
  once. On the shared pool, two identical agent-turn runs gave identical texts for only 73 of 96 turns at 16 agents (32/48 at
  8): a decode step that landed while another agent's prompt was being read computed some experts on the CPU, which rounds
  differently, and whether a step landed there depended on timing. `STRATA_BATCH_LENT_EXACT=1` removes that cause (0 experts
  on the CPU; 89/96, 47/48). The remaining differences come from a race in how the server re-sends a turn after its
  thinking-budget wrap-up; `STRATA_PARALLEL_SLOTS_ONLY=1` with `STRATA_REUSE_CKPT_ONLY=1` removes it: 100 % identical at 4, 8
  and 16 agents, at ~5 % of steady decode ([NX-23](cards/NX-23.md), [NX-24](cards/NX-24.md)). Without those switches, treat
  agent-turn output under concurrency as not reproducible from run to run. On one-card servers `STRATA_BATCH_LENT_GPU=1` was
  measured exact beside reads ([NX-11](cards/NX-11.md)); the wrap-up race was not tested there.
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
| shared pool, MTP | **86.5-87.0** | **83.1** | 51.0 | 30.6 (4 slots) | 14.9 (8 slots) | [pool, 2nd day](data/2026-11-22-dp-pool.md) section 1 |
| shared pool, 8 slots, MTP off, after the reseat | - | - | 40.8 | 30.5 | 18.4 | same, section 4 |
| **shared pool, 8 slots, adaptive MTP (N = 4), after the reseat** | **88.1-89.0** | **85.3-85.8** | **54.0-54.6** | **31.7-32.4** | **18.2-18.3** | same, section 6 |

- **Runtime control.** Every pool cell ran an engine build with the ROCm 7.2.1 HSA runtime swapped in (export needs it); every
  baseline cell in this table ran the build with the bundled runtime. A control put the baseline on the pool's own build and
  runtime (MTP off, 2 trials per arm): pool vs that baseline 39.96 vs 36.38 tok/s per agent at 4 users (1.098x), 28.64 vs 24.25
  at 8 users (1.181x). The bundled-runtime and 7.2.1 baselines are within 0.985-1.003 of each other, so the runtime does not
  explain the gain ([data](data/2026-11-21-dp-pool.md) section 7). The MTP-on pool row has no such control.
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
- The last three rows ran on 22 November after MI50 #3's link was fixed (5-10 % faster on every arm, [NX-18](cards/NX-18.md))
  with the arena on huge pages; the "shared pool, MTP" row's 4- to 16-user cells ran before the fix. Rows from different
  sessions compare loosely. The adaptive-MTP row gives per-agent means over users (two trials), the other rows medians.
- Our per-agent target for thinking-on coding agents is 40 tok/s with TTFT p95 at most 30 s ([AGENT-TURNS.md](AGENT-TURNS.md)).
  On the best-known pool configuration ([NX-22](cards/NX-22.md)), counting only levels where every agent meets it in both
  trials: **4 agents on the steady short-prompt load** (slowest 49.8-50.7 tok/s) and **2 agents on agent turns** (6-turn
  sessions with 13-32k-token contexts, prompt cache on: slowest 76 tok/s, warm TTFT p95 6.1 s). At 4 agents on agent turns
  the slowest agent gets 32-36 tok/s: each worker reads prompts on its XTX alone, and decode waits during the reads. tp2 and a
  single server meet the target for two agents on short prompts.
- **Keep the prompt cache on for agents.** On the pool at 4 agents it gave 1.8x the per-agent decode of cache off and warm
  TTFT p95 6 s instead of 26 s; with it off, clients stalled at 8-16 agents.

Exactness (as defined [above](#how-to-read-the-numbers): short prompts): every row above gives the reference text
`a7ae6623797f17fe` on its solo prompt except one XTX + 2 MI50 with MTP, which gave `da29bc7d5a0ba433`: the drafter takes 513
XTX expert slots, so a different set of experts runs on that server's CPU path, and the CPU rounds differently from the GPUs
(MTP itself was not separated from that). The pool has no CPU path in normal decode and gives `a7ae6623797f17fe` with and
without MTP, with adaptive MTP and with any window setting; MTP did not change any of its short-prompt texts. Batched == solo
on every server. Two separate one-card servers do not give each other's texts at 384 tokens (different experts on each one's
CPU path); the two pool workers do (8/8). Agent turns with long prompts under concurrency are a different matter: see the
[exactness caveat](#how-to-read-the-numbers).

## Limitations and open items

- **Tensor parallel**: no MTP drafter, no batched prompt path and no prompt cache (every prompt goes through 8-token decode
  windows; 1,631 tokens took 19-25 s), no vision. Not usable for long agent prompts yet.
- **Build and launch from this tree**: the published tree builds (`strata`, `strata-device`, `pool_plan_test` with
  `tools/dsv41/build.sh`, MMQ off) and `pool_plan_test` passes. Layout 1 was run once as written (one chat request); layouts 2
  and 3 have not been run as written, and the MMQ-on build sequence ([Build](#build)) has not been run from this tree.
- **Pool vs baseline runtime**: controlled with MTP off (the runtime does not explain the gain, [Results](#results)); the MTP-on
  pool ratios have no same-runtime control.
- **Code not yet in this tree**: the switches marked † in [Settings](#settings) (adaptive MTP, the hugetlbfs arena, exact lent
  experts, the reproducible-run switches, the MTP window switches) and the Q4_K cases of `mmq_expert_bench` were measured on our
  development branch; the best-known pool configuration needs them.
- **Shared pool**: needs the ROCm 7.2.1 HSA runtime and one pod with shared network / PID / IPC namespaces; needs one shared host
  expert arena (on huge pages, [NX-19](cards/NX-19.md)); the exporter's death stops both workers. Placement tuning (slow-link
  MI50 weight +1.3 %, a hot set from agent traffic +0.3 %, per-worker hot sets -0.6 % per agent at 8 users) missed the 2 % bar
  and was not kept ([SHARED-POOL.md](SHARED-POOL.md) section 8; measured before the slow link was fixed). A GPU-registered
  arena on 4 KB pages under memory pressure is the likely cause of the earlier stalls (the kernel moving its pages stops every
  queue of the process); huge pages remove that for the arena, and a smaller load-time eviction from another registered range
  remains, not identified.
- **Agents per machine**: 4 on short prompts, 2 on agent turns at 40 tok/s ([NX-22](cards/NX-22.md)). Prompt reads run only on
  each worker's XTX and stream ~64 GiB of experts from host RAM per read part, about 2.2 s of every read; reading beside decode
  (`STRATA_BATCH_CONCURRENT_READ=2`) adds only 7-10 % because the read saturates the XTX. **Future work, not built and not a
  claim**: an MI50 prompt helper that computes the experts where the pool already holds them ([NX-25](cards/NX-25.md); ESTIMATE:
  warm reads 3.1-3.65 s -> 0.3-0.6 s, agent-turn headline 2 -> 4); it would change the arithmetic and be quality-gated, not
  called exact. Above 4 agents decode itself is the limit (32 tok/s per agent at 8 with no reads).
- **Run-to-run exactness under concurrency** needs `STRATA_BATCH_LENT_EXACT=1` plus the reproducible-run switches, which cost ~5 %
  of steady decode ([exactness caveat](#how-to-read-the-numbers)).
- **Long prompts serialise**: four cold 50k prompts at once still take ~150-160 s for the last first token.
- **Exactness corners**: with penalties on, tp2's solo text differs from one card's (pre-existing); a rare solo near-tie in long
  streams (one flip in 7 launches of a 2,500-token stream); the earlier one-card MTP reference `da29bc7d5a0ba433` is no longer
  reproduced in today's environment and was not re-baselined.
- **Kernels left**: 8-row Q6_K matvecs (mostly the lm head) at 300-450 GB/s against 450-600 at 4 rows; hc read, router and shared
  expert at 8 rows; ~2,300-3,650 small kernels per window; Q5_0 down has no decode-once multi-row kernel.
- **Quality**: KL against Q8_0 only; task-level quality (agent benchmarks, tool calling) on the optimised engine is not run. The
  Q4_K recipe's command lines and the calibration text's makeup are in [QUANT.md](QUANT.md); the calibration text itself is not
  published.
- **No public re-run yet**: the short-prompt set is private; a public prompt set and a published launcher example
  (`docs/examples/`) are to come.
- **Hardware**: one machine; host RAM, DRAM bandwidth and CPU threads limit a third server; a third XTX on the pool is a desk
  estimate only (~6 agents on short prompts, ~3 on agent turns).

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
| [NX-15](cards/NX-15.md) | Shared pool at one user, with and without MTP | finding | MTP changes no short-prompt bit | 83-87 tok/s per user with MTP (1.11-1.17x) |
| [NX-16](cards/NX-16.md) | XTX power cap | finding | n/a | 272 W costs nothing in decode |
| [NX-17](cards/NX-17.md) | MTP on the pool at 2-8 users per worker; the verify window | finding (window switches not kept) | MTP changes no short-prompt text | 1.28x at 2 per worker, 1.07x at 4, 0.87x at 8 |
| [NX-18](cards/NX-18.md) | Reseating the slow-link MI50 (Gen1 -> Gen3 x8) | finding | n/a | every arm +5-10 % per agent (median ~+7 %) |
| [NX-19](cards/NX-19.md) | Expert arenas on 2 MiB hugetlb pages | kept (pool default, opt-in by path) | identical texts | load-time eviction 9.8-14.9 s -> <= 2.6 s; pool 0.985x / 1.006x |
| [NX-20](cards/NX-20.md) | Adaptive MTP (`STRATA_BATCH_MTP_ADAPT=4`) | kept (opt-in) | identical texts | MTP's speed at <= 4 live slots, MTP off's above; ramp 1.10x |
| [NX-21](cards/NX-21.md) | MI50 DPM high vs auto | finding (keep the pin) | identical texts | auto 0.966x at 16 users |
| [NX-22](cards/NX-22.md) | Agents at 40 tok/s on the pool | finding | not run-to-run identical under concurrency | 4 agents steady, 2 on agent turns; prompt cache on |
| [NX-23](cards/NX-23.md) | Lent experts never on the CPU; concurrent read on the pool | kept (opt-in) | 0 lent experts on the CPU; 89/96 run to run | concurrent read +7-10 % per agent, headline unchanged |
| [NX-24](cards/NX-24.md) | Run-to-run identical agent turns | kept (opt-in, reproducible runs) | **96/96 run to run** | ~5 % steady cost |
| [NX-25](cards/NX-25.md) | MI50 Q4_K prompt-kernel rate; planned prompt helper | finding + plan | helper would be quality-gated, not exact | 7-12 int8-TOPS on one MI50 |
