# Flash-Next on Gary: how we got here

Gary is our fork of Strata. This is the story of putting Qwen3.8 Flash-Next (a mixture-of-experts model with 48 layers of
512 experts, 24,576 routed experts in all) on one machine with 2 x RX 7900 XTX 24 GB, 4 x Instinct MI50 16 GB, an EPYC 7642
and 256 GB of DDR4-2666, over 20 and 21 November 2026. The cards ([cards/](cards/)) hold the evidence; this file holds the
order of events and the reasons.

## The goal and the method

The goal was throughput for several agents at once, not the fastest single stream: aggregate tok/s at 4 and 8 concurrent
agents, with every expert in GPU memory if possible. Later we turned that into a per-agent floor: at least 40 tok/s for each
agent, or agent jobs start timing out ([AGENT-TURNS.md](AGENT-TURNS.md)).

The method, every cycle: code complete, then exactness, then performance, then a full timing sweep (where every millisecond
goes), then optimise the biggest bucket, and repeat. One GPU measurement at a time. Bars written down before the run. Every
result checked for exactness against one card (greedy decode, token ids identical) before its speed counts. Quality testing
after a few optimise cycles, not before.

## Before tensor parallel: the layer split

The engine already ran Flash-Next on one XTX with the MTP drafter at ~80 tok/s for one user. Splitting the layers over two XTX
cards could not make two cards faster than one, and plateaued at ~78-81 tok/s aggregate at 4 users. A router fusion was exact but
gained nothing there; 14 built-in fused-path variants were swept and none passed on both cards. Two independent one-card servers
reached ~160 tok/s for 2 users but had no way past that. A tensor-parallel proof on two XTX passed every test we set (all-reduce
11.8 us; half a GDN layer 0.635x the time; half an attention layer 0.72x; four layers exact), so we built tensor parallel.

## Tensor parallel, six cycles in one day

| Users | Layer split (with MTP) | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---|---|---|---|---|
| 1 | ~80-87 | 46.0 | 46.7 | 47.6 | **49.7** |
| 2 | - | 66.5 | 66.1 | 69.9 | **73.6** |
| 4 | ~78-81 | 81.2 | 82.8 | 104.2 | **110.1** |
| 8 | plateau | 91.9 | 92.4 | 126.7 | **131.2** |

Aggregate tok/s, UD-IQ4_XS, 384 tokens per request, temperature 0, the same prompts throughout [MEASURED:
[data/2026-11-20-tp2-cycles.md](data/2026-11-20-tp2-cycles.md)]. tp2 rows are without the MTP drafter.

- **Cycle 1: it works, but slower** ([NX-01](cards/NX-01.md)). `--tensor-parallel 2` splits the GDN heads and attention
  query heads over the two cards and gives each card whole experts (expert id modulo 2), with two host-staged all-reduces a
  layer. Exact: 256 / 256 tokens identical to one card. 63 % of experts resident across the two cards (26 % on one). But
  27-41 tok/s against 46-48 on one card.
- **Cycle 2: equal to one card at one user.** The worst stall the timing build showed was an artefact of the stage profiler
  (0.4 us without it). The rest of cycle 1's spread came from rare multi-second windows before a hardware fault on the host
  (three resets that afternoon, a CPU machine check); none since. Experts owned by residency (each card asks the host only for
  experts in neither card's VRAM): +1-2 %.
- **Cycle 3: batching** ([NX-02](cards/NX-02.md)). Cards plan their own expert dispatch on the GPU; the host is skipped where no
  CPU work exists. `--batch` works under tensor parallel, which now beats one card at every user count (1.28-1.38x from 2 to 8).
- **Cycle 4: the CPU path is at its limit** ([NX-03](cards/NX-03.md), [NX-04](cards/NX-04.md)). A batch-vs-solo text mismatch
  turned out to be the presence penalty, which batched requests skipped; with the penalty at 0 the logits were bit-identical.
  Batch self-commit: +3 %. The CPU expert pass was memory-bandwidth-bound (~2.1 GB per 4-user window in ~20 ms) and thread
  count made no difference, so the experts had to move to GPUs.
- **Cycle 5: the MI50s hold every expert the XTX cards do not** ([NX-05](cards/NX-05.md)). CPU path 0 of 24,576; exact. +26 %
  at 4 users, +37 % at 8. The new biggest bucket was the MI50 round trip: helpers issued one after another, each MI50 busy only
  5-8 ms of a 38 ms window.
- **Cycle 6: a faster MI50 round trip, and the better quant fits** ([NX-06](cards/NX-06.md), [NX-07](cards/NX-07.md)). A
  repeat test gave 96 / 96 identical texts. Kernel fetch + push + flag, graph replay and one issue thread per helper cut the helper
  wait from 8.5 to 2.3 ms a window at 4 users: +3-6 %. Our calibrated Q4_K (106 GB) fits on the GPUs at the same speed as
  UD-IQ4_XS, so it became the standard file ([QUANT.md](QUANT.md)).

## The fair round: tp2 against two plain servers

The same evening we measured tp2 + 4 MI50 against two independent servers of one XTX + 2 MI50 each, alternating, same quant,
MTP off on both. Two servers won: +30 % aggregate at 2 users, +18 % at 4, +29 % at 8; tp2 matched them only at 1 user. Each
plain server still ran about a third of the experts on the CPU.

## The timing study: why tensor parallel does not pay here

On 21 November a timing study compared tp2 with one card stage by stage ([ANATOMY.md](ANATOMY.md) section 4). Tensor parallel
cut card 0's dense time by only 6-12 %: a window is ~2,300-3,650 small kernels per card, and halving the heads halves only the
big projections. It then paid 2.4-5.3 ms of all-reduce and peer wait, and left the second card idle 3-5 ms a window. Its real
gain was memory: every expert on a GPU. The one-card server's cost was its CPU path, which is DRAM-bound and in series with the
GPU, and two such servers already used most of the host's spare DRAM bandwidth, CPU threads and RAM.

The study also found kernel and serving fixes that apply to both layouts: B6 multi-row matvecs under TSUM
([NX-09](cards/NX-09.md)), self-commit on one card ([NX-03](cards/NX-03.md)) and penalties in batch windows
([NX-08](cards/NX-08.md)). A kernel survey per quant showed every type has a GPU kernel on both architectures, and that the
expert's size, not the kernel, caps a single expert's read ([QUANT.md](QUANT.md), [ANATOMY.md](ANATOMY.md) section 5).

## Agent turns: how fast is fast enough

A re-analysis of earlier agent-benchmark runs set the target: at least 40 tok/s of sustained decode per agent, TTFT at most 30 s
at p95, a per-request timeout of at least 1,800 s ([AGENT-TURNS.md](AGENT-TURNS.md)). At that floor tp2 served two agents and
two plain servers also two. Real agent contexts (46-58k tokens) showed the next problem: a cold prompt took 44 s alone and up to
283 s for the fourth of four agents, because prompts are read one at a time.

## Prompt reading

Two cycles on prompt reading ([PREFILL.md](PREFILL.md)): a tuned hipBLASLt row for the experts' gate/up GEMM made a cold 52k
prompt 28 % faster and became the new exactness reference; an opt-in concurrent read keeps a decoding agent moving beside
someone else's prompt; and lent experts computed on the GPU made that decode exact.

## The CPU pool's spin

One-card servers kept every CPU pool worker spinning at 100 % from the first token. A predicted wake between layers
(`STRATA_POOL_WAIT=pred`) freed 17-20 cores and 10-14 W per server at no cost in speed and no change in output
([NX-13](cards/NX-13.md)).

## The shared MI50 pool

The last step joins the two results: one-card engines (no all-reduce, no idle second card) that share one pool of experts on all
four MI50s, so nothing runs on the CPU ([SHARED-POOL.md](SHARED-POOL.md), [NX-14](cards/NX-14.md)). One engine fills the MI50
caches and exports them by HIP IPC handle; the other maps the same memory and drives the cards with its own helper threads. It
needed two runtime conditions (the ROCm 7.2.1 HSA runtime, one pod sharing the network and PID namespaces) and one shared host
expert arena (the first trial with two private 68 GB arenas stalled in the kernel driver). Then it was exact on every check and
faster per agent than two plain servers: 1.106x at 4 users, 1.175x at 8, 1.27x at 16; 1.06-1.10x at 1 user, and 1.11-1.17x at 1
user with the MTP drafter (83-87 tok/s) [MEASURED].

## Where it stands

| Layout | Per agent at 1 / 2 / 4 / 8 users | Notes |
|---|---|---|
| tp2 + 4 MI50 | 50.5 / 40.6 / 31.6 / 18.8 | every expert on a GPU; no MTP; slow prompts |
| two one-card servers (1 XTX + 2 MI50 each) | 51.5 / 47.5 / 36.0 / 24.0 | about a third of experts on the CPU |
| two workers + shared MI50 pool | 54.4 / 52.3 / 39.8 / 28.2 | every expert on a GPU; 83-87 at 1 user with MTP |

The 2-user figures of the last two rows are one user on each worker or server; their 1- and 2-user cells come from a separate
one-trial test. Rows come from different sessions; ratios within a session are the reliable
comparison ([README.md](README.md#results)).

Next: placement tuning for the pool, MTP on the pool at more than one user, the remaining 8-row kernels, and quality testing on
agent tasks.
