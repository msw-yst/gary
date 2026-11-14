# Benchmarking GPU permutations

`tools/dsv41/bench-perms.sh` measures one GPU configuration: which RX 7900 XTX cards run the model (tensor parallel = how many
there are) and which Instinct MI50 cards serve as expert helpers. It prints one table row of plain numbers and appends it to
`runs/bench-perms/table.md`. It is not an A/B test. Every number comes from **one** cold, one-shot engine process, so a small
difference between two rows may be noise.

## Running it

```bash
bash tools/dsv41/devmap.sh                       # HIP ordinal, kind and PCI address of every card
python3 tools/dsv41/tok.py --model <dir with tokenizer.json> --text "$(cat some-text.txt)" > ids.txt   # >= 8,192 ids (~40 KB of text)
bash tools/dsv41/bench-perms.sh --exe build/dsv41/strata-dsv41 --pack <pack dir> --ids-file ids.txt --xtx 0 --mi50 4
bash tools/dsv41/bench-perms.sh --exe build/dsv41/strata-dsv41 --pack <pack dir> --ids-file ids.txt --xtx 0,1,2,3 --mi50 4,5,6,7
```

- Take the ordinals from `devmap.sh`. The HIP order is not a bus sort, and it changes when a card is added.
- The script prints each XTX's narrowest PCIe link on its way to the CPU (`links 0:x16 ...`). With fewer than all your XTX cards,
  choose the x16 ones. A card behind an x8 root port reads prompts more slowly, and in tensor parallel the others wait for it.
- Experts that do not fit in VRAM (the cards' and the helpers') are computed from host RAM (the CPU lane). With one XTX, that is
  ~135 GB of RAM for the experts alone.
- Run each configuration on an idle machine. For the MI50 clock pin of [CP-03](cards/CP-03.md), run under
  `tools/dsv41/with-mi50-pin.sh`.
- Use the same `--ids-file` for every configuration, so that every row reads the same prompts.
- Each run starts a separate engine process per measurement (default `off on p2k p8k`), and each one loads the model. Load time
  dominates: expect a few minutes per configuration.

## Ablation: which switch is worth what

```bash
bash tools/dsv41/bench-perms.sh ... --xtx 0,1,2,3 --mi50 4,5,6,7 --arms "off on bt" \
     --ablate "DS4_XK_QO DS4_TOPK_MB DS4_ATTN_MERGE=1"
```

This gives one row with every switch on (`base`), then one row per item: a bare name is unset, and `NAME=VALUE` is set to that
value. Next comes a row with every item applied (`none`), and finally `base` again (`base2`). **The gap between base and base2 is
the noise floor.** A switch whose row is closer to base than that gap has not shown an effect. To test your own build's switches,
add them with `--add-env "NAME=value ..."` and list them in `--ablate`.

## Reading the row

| column | what it is |
|---|---|
| devices | `--gpus` / `--tp` / `--helper-device` as run, and each XTX's narrowest link |
| experts in VRAM | the engine's `N of T in VRAM` per XTX, the experts each MI50 holds, and the CPU-lane set (the experts kept in host RAM; the helpers serve the ones they hold) |
| load s | launch to the `prompt:` line, minus the prompt's own time. It includes process start-up, ~0.5 s resolution |
| decode off / on tok/s | the engine's `decode:` line after a 64-token prompt, 256 tokens, greedy; drafter off / on (`--dspark-conf 0.3 --dspark-block 3`). Notes give the drafter's tokens a round |
| 2K / 8K / 32K | the engine's `prompt:` line: the time to read a cold 2,048 / 8,192 / 32,768-token prompt in a fresh process. This is not a server's time to first token: a server with a warm process and cached prefixes is faster |
| B8 ms a step | the batch test at 8 sequences: ms per decode step, and how many sequences matched their solo runs token for token |
| notes | host-RAM experts per decoded token (0 = everything ran on the GPUs), peak board power of the used cards (hwmon, not the wall), and why any cell is in [brackets] |

A cell in **[brackets]** came from an arm that did not check out: rc not 0, the time limit, a missing engine line, or a
`check FAILED`. The notes say which. Do not compare it with the other cells.

## Results on the test machine (14 Nov, build 16)

One run per cell, single process, build 16 with its recommended environment (the script's default env), drafter
settings 0.3 / 3, cold prompts. 1, 2 and 3 XTX use x16 cards; 4 XTX includes XTX #4 (x8). The MI50s are the four production helpers,
MI50 #1–4. Host-RAM share = expert picks of the 256 decode tokens computed from host RAM (the CPU lane), drafter off.
All numbers MEASURED; a single cold run each, so differences of a few per cent are not significant.

| XTX × MI50 | decode, drafter off, tok/s | decode, drafter on, tok/s | 2K prompt, s | 8K prompt, s | host-RAM share of picks |
|---|---|---|---|---|---|
| 1 × 0 | 25.2 | 21.8 | 9.9 | 21.5 | 48.6 % |
| 1 × 1 | 35.7 | 41.0 | 9.8 | 22.2 | 21.3 % |
| 1 × 2 | 43.6 | 50.8 | 10.0 | 21.4 | 10.7 % |
| 1 × 3 | 49.0 | 58.5 | 9.5 | 21.4 | 3.0 % |
| 1 × 4 | 51.7 | 59.5 | 9.2 | 20.9 | 0.2 % |
| 2 × 0 | 32.9 | 31.5 | 5.5 | 8.7 | 29.5 % |
| 2 × 1 | 48.3 | 50.4 | 5.7 (re-run 10.5) | 18.0 (re-run 14.6) | 8.6 % |
| 2 × 2 | 55.6 | 68.2 | 5.5 | 8.7 | 3.4 % |
| 2 × 3 | 62.3 | 74.5 | 5.5 | 8.7 | 1.1 % |
| 2 × 4 | 64.4 | 73.0 | 4.7 | 7.9 | 0 |
| 3 × 0 | 44.3 | 44.2 | 3.9 | 6.9 | — |
| 3 × 1 | 52.8 | 59.7 | 3.9 | 6.7 | — |
| 3 × 2 | 63.9 | 73.7 | 3.9 | 6.4 | — |
| 3 × 3 | 71.7 | 89.6 | 3.3 | 6.6 | — |
| 3 × 4 | 72.1 | 92.0 | 1.75 | 6.8 | — |
| 4 × 0 | 53.8 | 55.0 | 4.2 | 6.3 | 8.7 % |
| 4 × 1 | 64.5 | 72.8 | 4.2 | 6.3 | 2.9 % |
| 4 × 2 | 74.7 | 91.7 | 4.2 | 6.3 | 0.6 % |
| 4 × 3 | 80.1 | 102.4 | 1.45 | 5.6 | 0 |
| 4 × 4 | 79.7 | 110.5 | 1.40 | 5.2 | 0 |

- **The MI50s help until no experts are left in host RAM.** Each MI50 holds experts the XTX cards cannot, so it takes
  picks off the CPU lane: with one XTX, decode rises 25.2 → 51.7 tok/s as the host-RAM share falls 48.6 → 0.2 %. Once the
  share reaches 0 (2 × 4, 4 × 3) a further MI50 adds little to drafter-off decode: 4 × 3 80.1, 4 × 4 79.7 tok/s.
- **Short prompts speed up from three MI50s (with four XTX), because the prompt helper's buffers only fit then.** The
  MI50 prompt helper needs about 1.3 GB of buffers on each MI50. With one or two MI50s under four XTX, each MI50 is full of
  experts (about 3,460–3,520 each) and the engine turns the prompt helper off; with three MI50s each holds about 2,870,
  with four about 2,150, the buffers fit, and the 2K prompt drops from 4.2 to 1.45 s. It is a memory limit, not a
  three-helper rule. With three XTX each card holds more experts, so the step comes only at four MI50s (2K 3.3 → 1.75 s).
- **The drafter needs the experts on GPUs:** with no MI50 it loses at one and two XTX (host-RAM bound), and it gains up to
  +39 % at 4 × 4.
- **2 XTX + 1 MI50 reads prompts slowly and erratically.** A re-run gave 10.5 s (2K) and 14.6 s (8K), against 5.7 / 18.0 s
  the first time and 5.5 / 8.7 s at 2 × 0 and 2 × 2. Decode there is normal (re-run 49.1 / 50.3 tok/s). The prompt helper is
  off in all three, so that is not the difference. Cause unknown; we are looking into it.
- **Three XTX sits between two and four.** The fourth MI50 adds little drafter-off decode at 3 XTX (71.7 → 72.1 tok/s);
  the drafter still gains (89.6 → 92.0). Host-RAM shares for the 3-XTX rows were not extracted (—).
- These are one-shot single-process numbers, not a served time to first token or a served decode speed.
