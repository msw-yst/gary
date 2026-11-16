# engine-parity/: VANILLA, does our engine compute what the reference engines compute, at the same weights?

A pre-registered, teacher-forced comparison of engines at **identical weights and identical prompt ids**:
- **L:** an independent reference engine (llama.cpp);
- **V:** the upstream engine our fork came from;
- **P1:** our engine, on V's layout;
- optionally **D** (our engine as served) and **P2** (D with drafting off).

## Flow
1. **RENDER** (adapter): every request body goes through *your* serving stack against a fake engine that logs the ids it is handed.
   - Assert the server's usage count = the logged ids, and |rendered − recorded| ≤ max(32, 0.5 %).
   - → `RES/ids/<id>.ids`. Every arm reads these ids, and no arm renders on its own.
2. **Reference arm L** (`ref_llamacpp.py`): greedy 256 tokens + the top-20 at every position → `RES/L/<id>.json`.
3. **Select** (`select_tf.py`): 3 prompts per length bin, from L's results only → `RES/sel12.txt`.
4. **Arms V / P1** (adapter):
   - TF: prompt + L's tokens in one pass, logits dumped to a FIFO, then `tf_filter.py` → `RES/<ARM>/tf-<id>.json`.
   - FREE: the engine's own greedy 256 tokens → `RES/<ARM>/free-<id>.json`.
5. **Summary** (`vsum.py`):

```bash
python3 -I engine-parity/vsum.py RES --arms V P1 --verdict "ENGINE=P1,V" > RES/SUMMARY.txt
# serving layout later:  --arms V P1 D P2 --verdict "ENGINE=P1,V" --verdict "DEPLOY=D,V,P1" --drafting D,P2
```

## Pre-registered verdicts (from `strata-dsv41/prereg.md` §5)
- **ENGINE:** PASS if e_P1 − e_V ≤ max(0.10 × e_V, 0.002) nats AND a_P1 − a_V ≥ −0.005. Else DIFF, with the prompts listed where P1 sits furthest above V.
- **DEPLOY:** the floor is the layout placebo's own distance, max(|e_D − e_P1|, 0.002) and max(|a_D − a_P1|, 0.005).
- **NO DATA** when either arm of a verdict has no TF result, or misses more than 2 of the 12 prompts. A wrong record count in a dump is NO DATA for that prompt.

## Files
| File | Generic? | What |
|---|---|---|
| `vsum.py` | yes | TF / FREE metrics, per band and per token class, verdicts (arms, reference and floors are options). Also scores hosted-forced files (`--ref` a forced dir, `--tag htf`) |
| `tf_filter.py` | yes, for engines that dump raw f32 logits per position | FIFO reader. The record count must be exactly prompt + continuation. Uses numpy when available |
| `select_tf.py` | yes | the TF prompt choice, from the reference only |
| `ref_llamacpp.py` | yes (llama-server) | reference arm. Class markers are options |
| `strata-dsv41/` | no | the DeepSeek V4.1 / Strata pieces as used: the arm script, the FREE driver, RENDER, the pre-registration |

## DeepSeek V4.1 result
ENGINE PASS: V 0.00414 nats, top-1 96.99 %; P1 0.00417, 96.50 % (`../README.md`, `../../docs/dsv41/VALIDATION-2026-11-16.md`).
