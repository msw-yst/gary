# hostlp/: our quantized model against the hosted model, token by token

The idea: take the hosted model's greedy continuation of real prompts, with its top-k logprobs at every step, as the reference. Teacher-force our engines along those exact tokens, then measure how far our distributions sit from hosted's. Compare that distance with the distance between two of our engines at the same weights (the engine floor).

```
fetch_hosted.py  ->  RES/hostlp/H/<id>.json        raw provider answers (PRIVATE: they contain generated text)
map_tokens.py    ->  RES/hostlp/forced/<id>.json   our ids for hosted's steps + hosted's top-k mapped by bytes
                     RES/hostlp/map.tsv            OK / EXCLUDED per prompt
(force)          ->  RES/<ARM>/htf-<id>.json        the engine-parity teacher-forced pass, reference = forced/
hscore.py        ->  per-arm metrics, arm-pair floor, READING
```

## Run
```bash
# 1. fetch: only model / temperature 0 / max_tokens / logprobs / top_logprobs / stream are changed in the recorded bodies
python3 -I hostlp/fetch_hosted.py --base-url http://127.0.0.1:PORT/v1 --model MODEL \
    --bodies PROMPTS --select RES/sel12.txt --ids RES/ids --out RES/hostlp      # [--key-env NAME] if no key-injecting proxy
# 2. map by bytes (HF tokenizer, or a module adapter: see tokadapt.py)
python3 -I hostlp/map_tokens.py --out RES/hostlp --tokenizer hf:/path/to/tokenizer.json
# 3. force each arm on RES/hostlp/forced (DeepSeek/Strata adapter: CONT=/res/hostlp/forced TFTAG=htf PHASES=tf run-arm.sh V, then P1)
# 4. score
python3 -I hostlp/hscore.py --ref RES/hostlp/forced --arm V=RES/V --arm P1=RES/P1 --tag htf \
    --engine-floor-kl <e_V from vsum> --engine-floor-agree <a_V from vsum> --json RES/hostlp/hscore.json
```

## Checks that must pass before reading the numbers
- `fetch_hosted.py` prints `prompt_tokens=X ours=X` for every prompt. A `PROMPT-TOKENS-DIFFER` means a different template or tokenizer: stop.
- `map_tokens.py` ends `MAP: k of n accepted`. EXCLUDED prompts are listed with the first mismatching step, never silently kept.
- Every arm file has `ok: true`, and as many positions as the reference. `hscore.py` lists anything else as NO DATA, and an arm missing more than 2 prompts is NO DATA as a whole.

## Metrics (hscore.py)
- **KL(H‖X)** over hosted's top-k plus a rest bucket, with masses floored at 1e-6. It is the same estimator as `engine-parity/vsum.py`: an approximation, not a bound.
- **top-1 agreement**, the **hosted-token NLL gap**, and the **rank** of hosted's token in X's own top-k.
- **Floor, per pair of arms on the same positions:**
  - argmax agreement;
  - KL(A‖B) over the reference's top-k + rest (both arms have logprobs there);
  - mean |lp_tok difference|.
- **READING:** QUANT COST SHOWN / ENGINE COST SHOWN / NO MEASURABLE DIFFERENCE / UNCLEAR, with factor 3 by default (pre-registered in the original spec).

## Files
- `fetch_hosted.py`, `map_tokens.py`, `tokadapt.py`, `hscore.py`: generalised.
  - `hscore.py` was checked to reproduce the original's numbers exactly on the DeepSeek data. It adds the arm-pair KL, the NO DATA rule and the READING line.
- `strata_pack_tokenizer.py`: the DeepSeek V4.1 Strata pack tokenizer, built exactly as the server builds it.
- The scripts as run on 2026-11-16 are not published; the generalised ones above reproduce their numbers:
  - `h1.py` fetch through the local metering proxy;
  - `h2.py` (HF tokenizers; not usable, because the pack has no HF tokenizer.json) and `h2b.py` (the server's tokenizer) do the mapping;
  - `h3.py` writes the forced references;
  - `hscore.py`.
- DeepSeek numbers: `../README.md` and `../../docs/dsv41/VALIDATION-2026-11-16.md`.

## Privacy
`H/` holds the provider's generated text, and `forced/*.ids` are that text in tokenized form. Treat both like the prompts. The printed lines and `hscore.py` output are numbers only.
