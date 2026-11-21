# Which 4-bit file: the calibrated Q4_K

All Flash-Next results from cycle 6 on use our calibrated Q4_K file ("YST-Q4K-imat"). This file says why, how it was made, how
it scores against Q8_0, and what it costs.

Labels as in [README.md](README.md#how-to-read-the-numbers). Card: [NX-07](cards/NX-07.md).

**Download:** the file will be published on Hugging Face: **[PLACEHOLDER: Hugging Face repository link, not yet
published]**. The weights are a derivative of Qwen3.8 Flash-Next and are under the Qwen Community License 1.0; the licence text
ships with the files. Gary's MIT licence covers the code in this repository, not the weights.

## Why not the shipped file

The widely used "UD-IQ4_XS" upload of Flash-Next is about 3.84 bits per weight on average, and its routed experts are mostly not
IQ4_XS: across the 48 layers the expert gate/up matrices are IQ3_XXS in 28 layers, IQ4_XS in 17 and IQ2_S in 3, and the down
matrices are Q5_0 in 33 layers, IQ4_NL in 12 and Q2_0 in 2 [MEASURED: GGUF headers]. We wanted a 4-bit file whose quality we had
measured ourselves.

## The recipe

1. Start from the BF16 weights.
2. Make a Q8_0 GGUF. It serves as the intermediate for calibration and as the quality reference below.
3. Compute an importance matrix (llama.cpp's imatrix tool) on the Q8_0 model over calibration text.
4. Quantise the BF16 weights to Q4_K with that importance matrix.
   - Routed experts: gate and up are Q4_K. The down matrix is 640 columns wide, which no 256-block type can quantise, so it
     falls back to **Q5_0 in every layer**.
   - One expert is 921,600 + 921,600 + 1,126,400 = **2,969,600 bytes** in every layer; all experts are 72.98 GB.
   - The dense projections (GDN, attention, shared expert, lm head) are stored at higher precision; the kernels we measured on
     them are Q6_K and Q8_0. The full tensor-type map is in the GGUF header (`gguf-dump` lists it).
5. Keep the shipped pack's PLE / n-gram shard and MTP drafter file unchanged.
6. Pack it for the engine with `tools/iq_pack.py --compat-bf16`, as setup does for the shipped files. Use the shipped pack's
   `chat_template.jinja`: the template exported in the Q4_K GGUF contains Hugging Face's `{% generation %}` tag, which the
   server's jinja2 refuses.

The whole file is 106 GB with the PLE shard.

Not yet in the published record: the calibration text, its size and chunk count, the exact llama.cpp version and command lines.
They will be added before the file itself is published (see [README.md](README.md#limitations-and-open-items)).

## Quality against Q8_0

Scored against our Q8_0 on text held out from calibration [REPORTED: the scoring logs are not in the published extracts, so the
table has not been re-checked against raw output here]:

| File | Size | Mean KL vs Q8_0 | 99 % KL | Same top token |
|---|---|---|---|---|
| Shipped UD-IQ4_XS | ~85 GB | 0.056 | 0.77 | 93.6 % |
| **Ours, calibrated Q4_K** | 106 GB | **0.032** | 0.50 | **95.2 %** |
| Ours, calibrated IQ4_XS | ~98 GB | 0.037 | 0.51 | 94.8 % |
| Another public IQ4_XS upload (3.84 bpw, the shipped file's size) | ~85 GB | 0.246 | 3.37 | 87.5 % |

- Our files are bigger, which explains part of the gain. We did not build a calibrated file at the shipped size.
- The last row scored far worse than its size suggests; we did not investigate why and set it aside.
- KL against Q8_0 is a proxy. Task-level quality (agent benchmarks, tool calling) for shipped vs calibrated on the optimised
  engine has not been run.

## What it costs

- **Memory.** On tp2 + 4 MI50 every expert still fits on a GPU: XTX cards 5,468 + 5,030 experts, each MI50 ~3,520 (9.7 of
  ~15 GiB); CPU path 0 [MEASURED: [data/2026-11-20-tp2-cycles.md](data/2026-11-20-tp2-cycles.md)].
- **Speed.** None measurable on tp2: 50.2 vs 49.4 tok/s at 1 user, 108.7 vs 110.0 at 4, 132.2 vs 132.4 at 8 (alternating runs,
  within the 3-4 % noise) [MEASURED].
- **Kernels.** The Q4_K expert is our fastest expert at batch 1 on the XTX (21.3 us) and our best path on the MI50 (1.1x
  llama.cpp's GPU time, against 1.45-1.8x for the shipped file's IQ3_XXS / IQ4_XS kernels)
  [MEASURED: [data/2026-11-21-kernel-bandwidth.md](data/2026-11-21-kernel-bandwidth.md)].
- **One-card servers.** Bigger experts mean fewer fit, so a one-card server keeps more on the CPU. The shared MI50 pool removes
  that: with four MI50s nothing goes to the CPU at 2, 4 or 8 slots per worker ([SHARED-POOL.md](SHARED-POOL.md)).

## Publishing the file

We have not published the Q4_K GGUF. Before we do: the Qwen model licence terms for redistributing derived weights need
checking, and the recipe details above need filling in.
