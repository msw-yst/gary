# validation/: does this fork compute what upstream Strata and llama.cpp compute, and how far is the quantised model from the hosted one?

Two pre-registered, teacher-forced comparisons, run on 15–16 Nov 2026 for DeepSeek V4.1 Flash. Results:
[docs/dsv41/UPDATE-2026-11-16.md](../docs/dsv41/UPDATE-2026-11-16.md), cards
[QA-10](../docs/dsv41/cards/QA-10.md) and [QA-11](../docs/dsv41/cards/QA-11.md). The method in full: [METHOD.md](METHOD.md).

| step | directory | question | DeepSeek V4.1 Flash answer |
|---|---|---|---|
| engine parity (VANILLA) | `engine-parity/` | At identical weights and prompt ids, does our engine match upstream Strata, and how far are both from llama.cpp? | **ENGINE PASS.** KL from llama.cpp: upstream v0.2.9 0.00414 nats a token (top-1 96.99 %), this fork's build 17 0.00417 (96.50 %); difference +0.00003 against a 0.002 floor |
| hosted logprobs (HOSTLP) | `hostlp/` | Does our quantised model diverge from the hosted model, token by token? | **QUANT COST SHOWN.** 2.25 bpw IQ2_XS: 0.0434–0.0438 nats, 90.3 % top-1 (both engines alike, ~10× the engine floor); native MXFP4 experts: 0.0054, 96.0 % (88 % of the gap closed), inside the hosted API's own run-to-run noise (0.019) |

## Conventions
- **Scripts print numbers only.** Prompts, provider answers, and the token ids of prompts and continuations are private
  inputs; nothing here writes them anywhere public, and none are published with this release.
- **Missing data is never a pass.** Every summariser has a NO DATA outcome.
- **Python:** run as `python3 -I`. Standard library only, except `tokenizers` for `hf:` tokenizers in `map_tokens.py`
  and numpy (optional, faster) in `tf_filter.py`.

## Files
```
METHOD.md                         the validation sequence, its rules and its pitfalls
engine-parity/                    vsum.py (TF / FREE metrics, verdicts), tf_filter.py (logit FIFO reader),
                                  select_tf.py (prompt choice from the reference only), ref_llamacpp.py (reference arm)
engine-parity/strata-dsv41/       the Strata pieces as used: arm-inner.sh (one arm: smoke, tf, free), sserve.py (FREE
                                  driver for --serve), render.py + fake-engine.py (RENDER: the exact ids the server hands
                                  the engine), prereg.md (the pre-registration, written before any data)
hostlp/                           fetch_hosted.py, map_tokens.py, tokadapt.py, hscore.py, strata_pack_tokenizer.py
```

## Reproducing on your own prompts
1. **RENDER** your request bodies through the serving stack against `fake-engine.py`, so every arm reads the same ids
   (`render.py`).
2. **Reference arm L:** llama-server with the same weights (converted to llama.cpp's layout byte for byte), then
   `engine-parity/ref_llamacpp.py` (greedy 256 tokens + top-20 per position), then `engine-parity/select_tf.py`.
3. **Arms V (upstream Strata) and P1 (this fork):** `engine-parity/strata-dsv41/arm-inner.sh` phases `smoke tf free`
   inside a container that sees one GPU. It expects the engine binary at `/b/strata`, the pack at `/ds/pack-2.25`, the
   results at `/res` and the scripts at `/pkg`; P1's engine flags come from `BUILD_ENV` in `/pkgenv/build17.env`.
4. **Summary:** `python3 -I engine-parity/vsum.py RES --arms V P1 --verdict "ENGINE=P1,V"`.
5. **HOSTLP:** `hostlp/README.md` (fetch, map by bytes, force each arm along the hosted tokens with `CONT` / `TFTAG`,
   score with `hscore.py`).

These tools come from our internal validation kit, generalised so the model can be swapped; the operator scripts that
launched the containers on our machines (paths, guards, schedules) are not published.
