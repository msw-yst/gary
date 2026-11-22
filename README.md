<p align="center">
  <a href="https://yst.dev">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/brand/yst-lockup-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/brand/yst-lockup-light.svg">
      <img src="docs/brand/yst-lockup-light.svg" alt="YST" width="240" height="120">
    </picture>
  </a>
</p>

# Gary

Gary is a fork of [Strata](https://github.com/Niko1221/Strata), made by YST for one job: serving very large
mixture-of-experts models quickly, to many agents at once, from one machine of mixed AMD GPUs. Consumer RX 7900 XTX
cards run the model, and older Instinct MI50s hold and compute the experts that do not fit on them. Strata does the hard,
general work. Gary pushes in one direction, and several of its ideas come from other projects, credited below. Every
result on this page is measured, and each links to the card that records it.

Each model lives on its own branch with its own build instructions. This `main` branch holds only this page and links.
[Upstream Strata's README](UPSTREAM-README.md) is kept unchanged.

## Models

| Model | Branch | Hardware tested | Headline result | Quick start | Weights |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [`dsv41-amd`](https://github.com/msw-yst/gary/tree/dsv41-amd/docs/dsv41) | 4× RX 7900 XTX + 4× Instinct MI50 | one user 8.45 ms a token (about 118 tok/s), 8 users 202.8 tok/s in all, output identical to the build before it | [Quick start](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/examples/dsv41-serve.sh) | [ISTA-DASLab, 2.25 bpw](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) |
| Qwen3.8 Flash-Next | [`flashnext-amd`](https://github.com/msw-yst/gary/tree/flashnext-amd/docs/flashnext) | 2× RX 7900 XTX + 4× Instinct MI50 | 4 agents ≥ 40 tok/s each; 88 tok/s for one user | [Quick start](https://github.com/msw-yst/gary/tree/flashnext-amd/docs/flashnext#launch) | [Hugging Face](https://huggingface.co/yst-dev/Qwen3.8-Flash-Next-Q4_K-imatrix-GGUF) |

The two models were measured on different machines with different MI50 cards; each section gives its own hardware.

## DeepSeek V4.1 Flash

**What:** four RX 7900 XTX cards run the model in tensor parallel, while four Instinct MI50s hold and compute the
experts the XTX cards can't fit, so DeepSeek V4.1 Flash serves several users from one machine.

**Headline results** (the release build, build 17, against the build before it; served, drafter on, same machine, same
day):

- **Eight users: +12.5 % (202.8 tok/s in all); four users: +5.3 % (161.6 tok/s in all)**
  ([SV-17](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/cards/SV-17.md)).
- **One user, 2K context: 8.45 ms a token, about 118 tok/s; first token at 8K 6.73 → 5.89 s (−12.5 %)**
  ([SV-17](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/cards/SV-17.md)).
- **Exact:** output identical to the build before it, and no expert counted twice in 2.92 million picks
  ([SV-17](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/cards/SV-17.md)). An earlier build summed some experts twice; the fix and what it cost are in
  [HX-28](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/cards/HX-28.md).
- **The MI50s matter most:** with one XTX, decode rises from 25.2 tok/s with the experts in host RAM to 51.7 tok/s with
  four MI50s holding them ([GPU permutations](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/benchmarking-gpu-permutations.md)).

Release 4 keeps build 17 as its release build and recommended setup, so these numbers apply to it. It adds an opt-in,
bit-identical BF16 matrix-vector kernel ([FG-20](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/cards/FG-20.md)), off by default, and a write-up of
[P2P on gfx906 and gfx1100](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/p2p-on-gfx906-and-gfx1100.md). Timings use a private prompt set; ratios within a run
compare like with like.

**Hardware tested:** 4× RX 7900 XTX 24 GB + 4× Instinct MI50 32 GB, one 16-core EPYC (Zen 2), DDR4-2666; ROCm 7.14 on
Linux ([hardware table](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/README.md#hardware-public-labels)).

**Quick start:** build with [`tools/dsv41/build.sh`](https://github.com/msw-yst/gary/blob/dsv41-amd/tools/dsv41/build.sh),
serve with [`docs/examples/dsv41-serve.sh`](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/examples/dsv41-serve.sh);
the switches and the recommended environment are in the
[DeepSeek V4.1 README](https://github.com/msw-yst/gary/tree/dsv41-amd/docs/dsv41) on `dsv41-amd`, with the long
write-up and one card per change alongside it.

**Weights:** [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
the 2.25 bpw IQ2_XS set (ISTA-DASLab's GSQ-RCO quantisation).

**Status:** release 4, [`v0.2.9-dsv41.4`](https://github.com/msw-yst/gary/releases/tag/v0.2.9-dsv41.4)
([release notes](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/UPDATE-2026-11-17.md)). Next release (pending). Every result, decision and dead end has a card:
[cards index](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/README.md#card-index).

## Qwen3.8 Flash-Next

**What:** two RX 7900 XTX workers share one pool of the model's 24,576 experts held on four Instinct MI50s, so the
whole 4-bit model lives in GPU memory on one machine and serves several coding agents at once.

**Headline results** (calibrated Q4_K, temperature 0):

- **One user: 88–89 tok/s; four users: 54 tok/s each** with adaptive MTP on the shared pool
  ([NX-20](https://github.com/msw-yst/gary/blob/flashnext-amd/docs/flashnext/cards/NX-20.md)).
- **The shared pool against two separate servers on the same cards:** 1.11× per agent at 4 users, 1.18× at 8 and 1.27× at
  16, using 124 GiB of host RAM against 190 GiB
  ([NX-14](https://github.com/msw-yst/gary/blob/flashnext-amd/docs/flashnext/cards/NX-14.md)).
- **Agents at 40 tok/s or more each:** 4 on a steady load (slowest 50 tok/s); 2 on agent turns with 13–32k-token contexts
  (slowest 76 tok/s, warm first-token p95 6.1 s)
  ([NX-22](https://github.com/msw-yst/gary/blob/flashnext-amd/docs/flashnext/cards/NX-22.md)).
- **Reproducible runs:** with the reproducible-run switches, two identical 16-agent runs gave identical text on 96 of 96
  turns ([NX-24](https://github.com/msw-yst/gary/blob/flashnext-amd/docs/flashnext/cards/NX-24.md)); without them, see the
  exactness caveat in the Flash-Next README.

**Hardware tested:** 2× RX 7900 XTX 24 GB + 4× Instinct MI50 16 GB, one EPYC 7642, 256 GB DDR4; ROCm 7.2.

**Quick start:** build, launch commands for all three layouts, host setup and the full settings table are in the
[Flash-Next README](https://github.com/msw-yst/gary/tree/flashnext-amd/docs/flashnext) on `flashnext-amd`; the journey and
one card per change sit alongside it.

**Weights:** [yst-dev/Qwen3.8-Flash-Next-Q4_K-imatrix-GGUF](https://huggingface.co/yst-dev/Qwen3.8-Flash-Next-Q4_K-imatrix-GGUF)
(our calibrated Q4_K; how it scores against Q8_0 is in QUANT.md).

**Status:** pre-release. Next: MI50s helping with prompt reading (in development; no results claimed).

## How we measure

Each change goes in behind a switch and gets a card: what and why, the prediction written down before the run,
correctness, timing and a verdict. Exact changes must give byte-identical output to the build before them. Changes that
are not exact must pass a stated quality bar. Every number is labelled MEASURED, REPORTED or INFERRED, and we publish dead
ends too. All results are for our machine, these models and these quantisations, and may not transfer to yours. The
cards are in the [DeepSeek cards index](https://github.com/msw-yst/gary/blob/dsv41-amd/docs/dsv41/README.md) and the
[Flash-Next README](https://github.com/msw-yst/gary/tree/flashnext-amd/docs/flashnext).

## Credits and licence

Gary is a fork of **[Strata](https://github.com/Niko1221/Strata)** by **Niko1221 and the Strata contributors** (MIT).
The DeepSeek branch is based on upstream tag `v0.2.9`. Ideas and code also come from:

- **[Project Maya](https://github.com/mw00/project-maya)** (MIT, Peasant Smith (mw00) and contributors; itself built on
  Strata): several decode fusions, correctness fixes, the batched attention launch and the uneven expert split are ported
  from it. The individual pull-request authors (pepuscz, tctiger, mrsirg97) are credited in NOTICE.md and in the source
  files.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT, the ggml authors): the reference engine we measure
  against.
- **[castagna-veloce](https://github.com/benpeterson40/castagna-veloce)** (MIT, a llama.cpp fork by benpeterson40): its
  gfx906 MoE kernels, including the fused expert kernel idea, are our MI50 benchmark reference.
- **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)**: their GSQ-RCO quantisation of DeepSeek V4.1 Flash (2.25 bpw
  IQ2_XS) is the weights releases 3 and 4 serve.
- **DeepSeek's** official chat-encoding reference: our chat template is checked against it, token for token.

Licence: MIT. Upstream's [LICENSE](LICENSE) and copyright line are unchanged. YST's changes carry "Copyright (c) 2026
YST" under the same licence. The full record is in
[NOTICE.md](https://github.com/msw-yst/gary/blob/dsv41-amd/NOTICE.md).

Thanks to Niko1221 and the Strata contributors, whose engine all of this builds on, and to Peasant Smith (mw00) and the
Project Maya contributors.

## Status and contact

Gary is a working fork, not a product and not a replacement for Strata. The DeepSeek branch has tagged releases, and
Flash-Next is pre-release. Gary is maintained by YST (Your Systems Team Limited): https://yst.dev.
