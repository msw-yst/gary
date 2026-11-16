# NOTICE

This repository is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors. It is
based on upstream tag `v0.2.9` and adds YST's work running DeepSeek V4.1 Flash on RX 7900 XTX (gfx1100) cards with
Instinct MI50 (gfx906) cards as expert helpers. The changes, their measurements and the flags that turn them on are
documented in [docs/dsv41](docs/dsv41/README.md).

Maintained by YST. Contact: https://yst.dev

## Licence

Strata is released under the MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors. The upstream
[LICENSE](LICENSE) text and copyright line are unchanged and apply to the whole tree. YST's changes are released under the
same MIT License; their copyright line, "Copyright (c) 2026 YST", is added in `LICENSE` below upstream's.

## Project Maya

Several changes port code or designs from [Project Maya](https://github.com/mw00/project-maya), an MIT-licensed project that
is itself built on Strata. Upstream Strata already ships Project Maya's copyright and permission notice, unchanged, in
[third_party/project-maya/NOTICE.md](third_party/project-maya/NOTICE.md) with its `LICENSE` beside it: Copyright (c) 2026
Peasant Smith (mw00) and the Project Maya contributors. That notice covers the further ports made in this fork:

- four correctness fixes in the DeepSeek engine ([EB-07](docs/dsv41/cards/EB-07.md));
- the decode kernel fusions `DS4_HC_FUSE`, `DS4_ROUTE_FUSE`, `DS4_DECODE_TRIM` and `DS4_ROUTER_BF16`
  ([FG-02](docs/dsv41/cards/FG-02.md));
- one attention launch for all sequences of a batch, `DS4_MULTI_ATTN` ([FG-05](docs/dsv41/cards/FG-05.md));
- the uneven whole-block expert split used by `--tp 4` ([TP-05](docs/dsv41/cards/TP-05.md));
- opt-in levers that were ported but never measured ([FG-06](docs/dsv41/cards/FG-06.md),
  [PF-08](docs/dsv41/cards/PF-08.md)).

The ported pull requests are credited in the source files that carry them, with their authors: pepuscz (the
hyper-connection, route and SwiGLU fusions and the uneven split), tctiger (the multi-part copies) and mrsirg97 (the BF16
router), as in Project Maya's history. Source files carrying ported code say so in a header or comment line.

## Benchmark references (release 3)

The expert-kernel benchmark in `tools/qbench/` and the validation in `validation/` compare this fork with two other MIT
projects, each built unmodified from its own sources. No code from either is included in this repository;
`tools/qbench/qb_ggml.cpp` calls ggml's public API.

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml, by the ggml authors (MIT): the reference engine, at
  b11870 (74bca395).
- [castagna-veloce](https://github.com/benpeterson40/castagna-veloce), by benpeterson40, a fork of llama.cpp (MIT, copyright the ggml authors): gfx906 MoE kernels,
  including its fused Q2_K/Q3_K expert kernel, at `main` 3483d4627.

## Acknowledgements

Thanks first to Niko1221 and the Strata contributors, whose engine all of this builds on, and to Peasant Smith and the
Project Maya contributors.

Thanks to the MI50 Discord community for discussion and help.
