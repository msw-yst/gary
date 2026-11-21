# Which 4-bit file: the calibrated Q4_K

All Flash-Next results from cycle 6 on use our calibrated Q4_K file ("YST-Q4K-imat"). This file says why, how it was made, how
it scores against Q8_0, and what it costs.

Labels as in [README.md](README.md#how-to-read-the-numbers). Card: [NX-07](cards/NX-07.md).

**Download:** [huggingface.co/yst-dev/Qwen3.8-Flash-Next-Q4_K-imatrix-GGUF](https://huggingface.co/yst-dev/Qwen3.8-Flash-Next-Q4_K-imatrix-GGUF)
(four GGUF shards, 106.2 GB, with `SHA256SUMS`). The weights are a derivative of Qwen3.8 Flash-Next and are under the Qwen
Community License 1.0; the licence text ships with the files. Gary's MIT licence covers the code in this repository, not the
weights.

## Why not the shipped file

The widely used "UD-IQ4_XS" upload of Flash-Next is about 3.84 bits per weight on average, and its routed experts are mostly not
IQ4_XS: across the 48 layers the expert gate/up matrices are IQ3_XXS in 28 layers, IQ4_XS in 17 and IQ2_S in 3, and the down
matrices are Q5_0 in 33 layers, IQ4_NL in 12 and Q2_0 in 2 [MEASURED: GGUF headers]. We wanted a 4-bit file whose quality we had
measured ourselves.

## The recipe

Tools: llama.cpp b11870 (commit `74bca3950`, the `full-rocm-b11870` container image) for convert, imatrix and quantize, all run
on the CPU [MEASURED: `--version`, image label]. `<...>` are local paths.

1. **BF16 weights.** `Qwen/Qwen3.8-Flash-Next` at revision `812d96e10a0dd82b69d749e68693698cf6c36daa` (the head of `main` when we
   downloaded it; the chat template embedded in our GGUF matches that revision byte for byte) [MEASURED].
   - We skipped the 31 BF16 shards that hold only the n-gram embedding table (step 3 takes that table from the shipped pack)
     [MEASURED: safetensors index].
2. **Q8_0 GGUF.** It is the intermediate for calibration and the quality reference below. It is made with a small wrapper around
   llama.cpp's converter: the wrapper sets `embedding_length_per_layer_input = 160` and drops the n-gram shard tensors.

   ```sh
   python3 convert_full.py <bf16 dir> --outtype q8_0 --no-mtp --no-tensor-first-split --split-max-size 75G \
     --outfile <inter>/Qwen3.8-Flash-Next-YST-Q8_0.gguf --model-name Qwen3.8-Flash-Next-YST
   ```

   - The converter writes 3 shards. We set their `split.count` to 4 (and `split.tensors.count` to 1224) and add the shipped
     pack's shard 4 as the fourth.
   - The architecture metadata equals the shipped file's key for key [MEASURED].
3. **Importance matrix.** llama.cpp's imatrix tool, run on the Q8_0:

   ```sh
   llama-imatrix -m <inter>/Qwen3.8-Flash-Next-YST-Q8_0-00001-of-00004.gguf -f imatrix-corpus.txt -c 512 -t 48 -ngl 0 \
     -o imatrix-Qwen3.8-Flash-Next-YST-Q8_0.gguf
   ```

   - **Calibration text:** 252,067 bytes in three parts:
     - 72 KB of alternating prose and code excerpts from the llama.cpp repository;
     - the first 90 KB of llama.cpp's `docs/*.md` (README files excluded, alphabetical);
     - the first 90 KB of a Python source corpus (open-source inference-server code).
   - **Size:** 140 chunks of 512 tokens = **71,680 tokens**; 926 tensors recorded [MEASURED: imatrix header].
   - **Not covered:** no chat, multilingual or reasoning text.
4. **Quantise the Q8_0** (a requantise, not from BF16) with that importance matrix and a per-tensor type file:

   ```sh
   llama-quantize --keep-split --allow-requantize --imatrix imatrix-Qwen3.8-Flash-Next-YST-Q8_0.gguf \
     --tensor-type-file types.txt <inter>/Qwen3.8-Flash-Next-YST-Q8_0-00001-of-00004.gguf \
     <out>/Qwen3.8-Flash-Next-YST-Q4K-imat Q4_K_M 48
   ```

   - The type file sets every 2-D tensor:
     - routed experts `q4_k`;
     - every other quantised tensor the type the shipped UD-IQ4_XS gives it (where the shipped file keeps BF16 or F16, the
       Q8_0's own type);
     - the n-gram table `iq4_nl`. So against the shipped file only the experts differ.
   - **Routed experts:** gate and up are Q4_K. The down matrix is 640 columns wide, which no 256-block type can quantise, so it
     falls back to **Q5_0 in every layer** (`48 of 1224 tensor(s) required fallback quantization`).
   - One expert is 921,600 + 921,600 + 1,126,400 = **2,969,600 bytes** in every layer; all experts are 72.98 GB.
   - **Totals:** 1224 tensors = F32 388, Q8_0 476, Q6_K 213, Q4_K 96, Q5_0 48, F16 1, IQ4_XS 1, IQ4_NL 1.
     - `output` Q6_K and `token_embd` IQ4_XS;
     - `attn_qkv` Q6_K (35 layers) / Q8_0 (1);
     - shared-expert down Q8_0;
     - experts 4.83 bpw, the other tensors 7.15 bpw [MEASURED: GGUF headers]. The full map is in the GGUF header
       (`gguf-dump` lists it).
5. Keep the shipped pack's PLE / n-gram shard and MTP drafter file unchanged.
   - **Shard 4.** Our shard 4 is the shipped `UD-IQ4_XS-00004-of-00004.gguf` byte for byte (Unsloth revision `f5d175de`,
     sha256 `5de8b5db...ec36`). The tensor the quantiser re-wrote had the same bytes [MEASURED].
   - **Drafter.** We did not build an MTP drafter (`--no-mtp`); the Hugging Face repository does not include one.
6. Pack it for the engine with `tools/iq_pack.py --compat-bf16`, as setup does for the shipped files. `iq_pack.py` needs
   llama.cpp's `gguf-py`: setup sets `STRATA_GGUF_PY` to its own llama.cpp copy; run by hand, set
   `STRATA_GGUF_PY=<llama.cpp checkout>/gguf-py`. Use the shipped pack's
   `chat_template.jinja`: the template exported in the Q4_K GGUF contains Hugging Face's `{% generation %}` tag, which the
   server's jinja2 refuses.
   - The Hugging Face repository ships that template as `chat_template.jinja`: Qwen's revision `de4b8e4d`, sha256 `c3cf9e34...1041`.

The whole file is 106.2 GB (106,212,324,416 bytes) with the PLE shard. Its shard 1 on Hugging Face has two metadata strings
(`quantize.imatrix.file`, `quantize.imatrix.dataset`) shortened to bare file names; no tensor or other key differs from the
file the results here were measured on.

## Quality against Q8_0

**Method:**
- **Reference:** our Q8_0. Every file below shares the same n-gram table.
- **Text:** 140,001 bytes held out from calibration: bytes 90,001-160,000 of the same docs concatenation and of the same Python
  corpus. 1.7 % of its 64-byte windows also occur in the calibration text.
- **Run:** `llama-perplexity -c 512 --chunks 40` (20,480 tokens) on the CPU. First `--kl-divergence-base` from the Q8_0, then
  `--kl-divergence` for each file. The llama-perplexity build's commit is not recorded.

[MEASURED: scoring logs]

| File | Size | Mean KL vs Q8_0 | 99 % KL | Same top token |
|---|---|---|---|---|
| Shipped UD-IQ4_XS | 93.7 GB | 0.056 | 0.77 | 93.6 % |
| **Ours, calibrated Q4_K** | 106.2 GB | **0.032** | 0.50 | **95.2 %** |
| Ours, calibrated IQ4_XS | 98.7 GB | 0.037 | 0.51 | 94.8 % |
| Another public IQ4_XS upload (3.84 bpw) | 84.9 GB | 0.246 | 3.37 | 87.5 % |

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

The file is published at the link above under the Qwen Community License 1.0.
- Redistributing modified weights is allowed if the copyright and permission notice ship with them.
- Products with more than 100M monthly users or US$20M monthly revenue must display the model name.
- Model-as-a-Service and AI-work-assistant businesses need a separate Qwen licence for commercial use.

The model card credits Qwen for the base model and Unsloth for the n-gram shard and the non-expert type choices.
