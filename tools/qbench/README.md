# QBENCH: one routed expert's decode, per card and format, against llama.cpp and castagna-veloce

Times one routed DeepSeek V4.1 Flash expert (n_embd 5120, n_ff 2304: gate, up, SwiGLU, down) on real expert bytes,
with this fork's kernels, with upstream llama.cpp's and with castagna-veloce's, on the same bytes and shapes, one GPU
at a time. On an MI50 it also times the helper's request round trip. Results and method:
[docs/dsv41/UPDATE-2026-11-16.md](../../docs/dsv41/UPDATE-2026-11-16.md), cards
[HX-33](../../docs/dsv41/cards/HX-33.md) (MI50), [HX-34](../../docs/dsv41/cards/HX-34.md) (round trip) and
[FG-19](../../docs/dsv41/cards/FG-19.md) (XTX).

Only sources and scripts are here: no binaries, no model files.

| File | What |
|---|---|
| `../../tests/dsv41/iq_helper_bench.cpp` | our side: `ds4_iq_helper_bench --pool-file F.qbx` times the helper's window path (`--path win`, B rows) and the lead's one-token path (`--path tok`), back to back and alone, and checks the output against a CPU fp32 reference (QB lines) |
| `../../tests/dsv41/helper_rt_bench.cpp` | `ds4_helper_rt_bench`: the MI50 helper's request round trip, by stage (`--sweep 1`) |
| `qb_ggml.cpp` | the reference side: the same expert through a ggml build (llama.cpp or castagna-veloce) via ggml's public API; `--mode expert` and `--mode moe1` (a 6 + 1-expert Q2_K/Q3_K layer, castagna's fused pattern) (QG lines) |
| `build-ref.sh` | builds a reference's ggml targets for one HIP arch and compiles `qb_ggml` against them |
| `mkpool.py` | packs 40 real experts of one layer into a QBX1 pool file |
| `run-card.sh` | runs every cell on one pinned GPU, then writes `SUMMARY.txt` |
| `ktsum.py` | GPU-side kernel µs per call from a rocprofv3 `--kernel-trace` csv (KT lines) |
| `qbsum.py` | QB / QG lines → markdown tables |

## 1. Build
```bash
# our two bench targets (ROCm; gfx1100 + gfx906 in every HIP unit, as for the engine)
TGTS="ds4_iq_helper_bench ds4_helper_rt_bench" bash tools/dsv41/build.sh        # -> build/dsv41/
# the references, for the card you will run on (gfx906 or gfx1100)
tools/qbench/build-ref.sh ~/src/llama.cpp gfx906 ~/qb/lcpp906                     # QBENCH: b11870, commit 74bca395
tools/qbench/build-ref.sh ~/src/castagna-veloce gfx906 ~/qb/cv906                # QBENCH: main at 3483d4627
```
castagna-veloce at that commit does not compile for gfx1100 (a DPP broadcast and `sdot2` are gfx9-only), so the XTX
runs have no castagna column.

## 2. Expert pools
A QBX1 file holds one layer's experts 0–39 for one format: a 24-byte header (`QBX1`, u32 gate/up type, down type,
n_embd, n_ff, count), then per expert the raw GGUF bytes of gate, up and down, unmodified.

1. Find the layer's three expert tensors in the GGUF's tensor table (`blk.<L>.ffn_gate_exps.weight`,
   `ffn_up_exps`, `ffn_down_exps`: one 3-D tensor per role, experts contiguous), e.g. with the `gguf` Python package.
2. Cut experts 0–39 of each role into its own file (`dd` with the tensor's data offset and 40 × the per-expert size).
3. `python3 -I tools/qbench/mkpool.py xxs.qbx <tg> <td> 5120 2304 40 <bytes_gate_or_up> <bytes_down> gate.bin up.bin down.bin`
   (`tg` / `td` are ggml type ids; the byte counts are per expert).

QBENCH used IQ2_XXS, IQ2_XS and IQ2_S layers (13, 26 and 25) of the 2.25 bpw IQ2 pack the engine serves, and layer 13
of a native-MXFP4 GGUF of the same model. Name the pools `xxs.qbx`, `xs.qbx`, `s.qbx` and `mx.qbx`.

## 3. Run, one card at a time
```bash
export HIP_VISIBLE_DEVICES=<one card>                 # exactly one GPU visible; nothing else on it
export QB_BIN=build/dsv41 QB_DATA=~/qb/pools QB_LCPP=~/qb/lcpp906 QB_CV=~/qb/cv906
tools/qbench/run-card.sh ~/qb/runs/mi50-1                               # phases: ours lcpp cv kt rt sum
QB_SLICES=1 QB_CV= tools/qbench/run-card.sh ~/qb/runs/xtx-1 ours lcpp kt sum   # an XTX: + the 1/4 slice shapes, no rt
python3 -I tools/qbench/qbsum.py MI50 1024 ~/qb/runs/mi50-1/{ours,lcpp,cv}/*.txt
```
- An MI50 run took about 8 minutes; the kernel-trace phase is the long one. `rt` is for MI50 helper cards.
- Keep the card cool and otherwise idle; the scripts do not watch temperatures.
- **What to compare:** the KT lines (GPU-side kernel µs per expert) are like for like across the three; wall times
  include each harness's own launch pattern. Our cells also print `rel_rms` / `rel_max` against the CPU reference; the
  references are checked only for non-finite outputs.
