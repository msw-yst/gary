# Lessons: what this work taught us, in plain words

Short points from our work on Gary, our fork of Strata, each backed by the cards that show it. Most come from 8–19 Nov, and the last week on the native MXFP4 pack
taught the most. Numbers here are rounded; the cards have the exact figures and their labels (MEASURED, REPORTED,
INFERRED). The long read is [WRITEUP.md](WRITEUP.md); the index is [README.md](README.md).

## Measuring

1. **Measure the baseline before pricing a fix.** Several "gains" disappeared once the same build was run against itself:
   one-user decode varies by about 5 % between identical arms, and only four-user throughput is steady enough to judge a
   1–2 % change served. [ME-14](cards/ME-14.md), [ME-05](cards/ME-05.md), [HX-36](cards/HX-36.md)
2. **Size the sample from the noise, before the run.** An A-against-A pilot gives each metric's spread; the RA all-reduce
   run used it to pick 24 repeats a cell and got clean intervals. A cell too short to move can still fail a bar, as a
   four-user cell did that never drafted. [ME-14](cards/ME-14.md), [AR-11](cards/AR-11.md), [DR-12](cards/DR-12.md)
3. **Pre-register the gates, and say NO DATA when the data is missing.** A run whose gate could not be checked (wrong
   path taken, drift, a parser that miscounted, a card too hot to finish its soak) reports NO DATA, not a pass or a fail.
   Several runs this week did; each was fixed and re-run rather than read generously.
   [ME-03](cards/ME-03.md), [PF-17](cards/PF-17.md), [PF-18](cards/PF-18.md), [HW-14](cards/HW-14.md), [ME-14](cards/ME-14.md)
4. **Anything running beside a timed run is part of the run.** A per-second monitoring script cost one arm 21 % of its
   four-user throughput and made a new build look 1.36× slower than it was. Measure a sampler's cost first, or don't run it.
   [ME-15](cards/ME-15.md), [FG-24](cards/FG-24.md)
5. **Tracers and counters change what they measure.** A traced step is 1.2–2.7× slower; use traces for shares and order,
   untraced timers for times. [ME-06](cards/ME-06.md), [FG-21](cards/FG-21.md), [FG-22](cards/FG-22.md)
6. **Check the byte counts against the pack you are running.** The BF16 matvec looked like 1 % of peak bandwidth because
   the arithmetic used the other pack's sizes; it was 39 %. [FG-20](cards/FG-20.md)
7. **A kernel bench is a ceiling, not a forecast.** Kernels 20–34 % faster in isolation gave 4 % served, and only on a
   clean run; the first served try could not see it at all. [HX-35](cards/HX-35.md), [HX-36](cards/HX-36.md)
8. **Correct the record when a reading turns out wrong, in the same place.** The prompt's disk reads were first blamed on
   Engram rows, then on decode page-ins; both were wrong and both corrections are in the cards.
   [PF-15](cards/PF-15.md), [FG-23](cards/FG-23.md), [ME-07](cards/ME-07.md)

## Moving data

9. **Fewer bytes moved is not faster: what moves, and from where, matters.** Turning on the MI50 prompt helper cut the
   data sent to each XTX by 55 % and made prompts 1.75× slower, because the MI50s took the hot experts that were already
   in RAM and left the cold ones on disk. Reading each expert once cut `read()` bytes by three quarters and saved nothing:
   they were page-cache hits. [PF-18](cards/PF-18.md), [PF-16](cards/PF-16.md)
10. **Removing work can fragment what is left and kill parallelism.** With the helper on, every expert it held cut the
    host's copy batches, from about 21 experts to about 2. Re-ordering the stream so batches stayed full took a further
    third off the read. [PF-18](cards/PF-18.md), [PF-19](cards/PF-19.md)
11. **Read-ahead for data already in RAM wastes the disk.** The stream asked the kernel to read ahead about 195 GB a
    prompt of file pages for experts the RAM store already held. Skipping it cut storage reads by 30–54 %; what finally
    removed the disk from the prompt was a store that holds exactly what must be streamed.
    [PF-15](cards/PF-15.md), [PF-16](cards/PF-16.md), [PF-19](cards/PF-19.md), [PF-20](cards/PF-20.md)
12. **Don't copy what you can DMA.** Once the disk was gone, the host's copy from RAM into staging buffers was the bound;
    letting the GPUs DMA straight from the store took a 2K prompt from 4.7 to 3.3 s. [PF-20](cards/PF-20.md)
13. **The disk was hiding the RAM speed.** Memory at 2933 instead of 2666 gave 14–16 % more bandwidth and no first-token
    change, because the prompt was waiting on disk and page-cache churn. With the disk out of the path, RAM speed is back
    in play; that A/B is still to run. [HW-13](cards/HW-13.md), [PF-19](cards/PF-19.md)
14. **The MI50s are fast enough for prefill once they are fed properly.** Their prompt work on the native pack took
    0.5–3.3 s a read and was hidden behind the XTXs' own work (the XTXs waited 1–4 ms for them); the slow part was always the host feeding the XTXs.
    [PF-18](cards/PF-18.md), [PF-19](cards/PF-19.md), [PF-20](cards/PF-20.md), [PF-12](cards/PF-12.md)
15. **Small transfers favour the host route; large ones favour P2P.** At decode sizes a write lands in host memory about
    1 µs sooner than in a peer card, and fanning out small writes to three peers gets a fifth of the bandwidth. At prompt
    sizes a direct reduce-scatter plus all-gather is 2–2.6× faster. [AR-10](cards/AR-10.md), [AR-11](cards/AR-11.md)
16. **On these boards, the P2P limit is the receiving GPU's power state, not the board.** An idle receiver caps a pair at
    about 15 GB/s and doubles a hand-off; forcing clocks up fixes the lab numbers and costs over 200 W idle for nothing
    measurable served. [HW-15](cards/HW-15.md), [CP-06](cards/CP-06.md)

## Prompts

17. **A fixed cost a prompt against a cost a token sets a crossover, and content moves it.** The short path pays per
    token, the chunked path a flat ~15 s; for code the crossover is about 1,190 tokens, for filler about 5,460. A cached
    prefix does not move it. [PF-17](cards/PF-17.md), [PF-10](cards/PF-10.md)
18. **A chain of measured fixes beats one big guess.** Native 2K first token: 15.2 s → 4.7 s (store holds the stream, full
    batches) → 3.3 s (DMA from the store), each step one switch with its own gate.
    [PF-15](cards/PF-15.md), [PF-19](cards/PF-19.md), [PF-20](cards/PF-20.md)

## Decode

19. **In lock-step tensor parallelism, find the waits, not the busy kernels.** On native the lead XTX computes about the
    same as its peers and spends the rest waiting, mostly for the CPU lane: a quarter of a step, last to finish on 70 % of
    layers. [FG-21](cards/FG-21.md), [FG-22](cards/FG-22.md)
20. **When a kernel is near the memory ceiling, move the bytes, don't tune the kernel.** The CPU lane runs at 85–90 % of
    host bandwidth; five ways of speeding it up gave at most 1–2 %. The XTX expert kernels were already at bandwidth too.
    [FG-23](cards/FG-23.md), [HX-35](cards/HX-35.md)
21. **The verify row's cost decides whether more drafting pays.** Four drafting users, tree drafts and longer drafts all
    trade extra verify rows for extra tokens; at 4.7 ms a row, trees lose and four-user drafting loses, while a longer
    block for one user wins 11–12 %. [DR-10](cards/DR-10.md), [DR-13](cards/DR-13.md), [DR-12](cards/DR-12.md)

## Correctness and quality

22. **Determinism depends on placement: pin it for every A/B.** Greedy outputs that "did not repeat" were expert placement
    differing between server starts; with the placement frozen and its moves landing in fixed rounds, 13 of 13 repeat.
    Without that, a "bits differ" result says nothing about the switch.
    [ME-13](cards/ME-13.md), [QA-08](cards/QA-08.md), [FG-20](cards/FG-20.md)
23. **A bench that is bit-exact does not prove the served path is.** The direct all-reduce passed every bench and still
    differed served, because a wait gave up silently after 2 s on long prompts; only the served identity check caught it.
    [AR-11](cards/AR-11.md)
24. **Fluent output can hide a correctness bug.** Expert picks summed twice changed answers with server uptime and nobody
    noticed from the text. [HX-28](cards/HX-28.md)
25. **In a quantized model, the small shared parts matter.** Making only the non-expert tensors native closed almost as
    much of the mixed pack's remaining quality gap as a perfect choice of experts (KL −0.0036 against −0.0041), and the
    drafter's confidence collapses whenever they are quantized. [QA-13](cards/QA-13.md), [DR-11](cards/DR-11.md), [QA-12](cards/QA-12.md)
26. **Text similarity is not a quality ruler; tasks are.** Sampled repeats of the same server diverge within a few words;
    only task outcomes say whether a change matters, and a loop guard that detected loops still rescued none.
    [PF-11](cards/PF-11.md), [QA-14](cards/QA-14.md)

## Hardware

27. **Qualify a card before it carries work, with an abort.** A fifth MI50 stopped its first soak at 80 °C; the gate held
    back every run that needed it until a cooling fix, then it passed at 66 °C. [HW-14](cards/HW-14.md),
    [HW-05](cards/HW-05.md)
28. **Check divisibility before planning a layout.** Tensor-parallel 3 died at load: the 129,280-row output head does not
    split three ways. [TP-10](cards/TP-10.md)
