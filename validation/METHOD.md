# Method: is the local model as good as the hosted one, and if not, why?

> Copied from our validation kit for this release. Only `engine-parity/` and `hostlp/` are published here; the
> references below to `harness/`, `agent-reruns/`, `templates/`, `ADAPTING.md`, `LESSONS.md`, the ledger and
> `../deepseek-v41/` point to parts of the kit that are not published. The DeepSeek numbers are in `README.md`.

This is the validation sequence that settled the DeepSeek V4.1 Flash question on 2026-11-15/16:

> "Local scores 18/27 on Terminal-Bench, hosted 24/27. Is our quant or our engine costing quality?"

It is written to be rerun on another model. What changes per model is listed in `ADAPTING.md`; the numbers from the DeepSeek run are in `../deepseek-v41/`.

**Order:** harness check → engine correctness → quant vs hosted → matched agent reruns. All four are governed by the rules in step 5.

Each step removes one confounder before the next step is run. **Do not skip ahead.** On DeepSeek, the agent-run gap looked like "quality". On inspection it was mostly speed, plus a harness default that punished slow turns. The one real quality signal only appeared once both had been removed.

---

## Step 1. Check the harness first (minutes, no GPU)

- **Question:** does the agent harness give both sides the same limits *in effect*, not just in config?
- **Do this:**
  1. Find the model client's per-call timeout and retry policy for every layer: agent framework, LiteLLM (or similar), and the SDK.
  2. Prove what it does with `harness/tmo_stub.py`: a loopback stub that answers late, and a hit counter.
  3. Set the per-call timeout above the longest agent time limit, and set retries to 0 where they can spend task time. Same setting on both sides.
  4. Check the stored per-trial config, not the command line.
  5. Log per-call cancels.
- **DeepSeek:** harbor → LiteLLM → openai SDK defaulted to **600 s per call with silent 3×3 retries**.
  - A local turn at 37 tok/s hits 600 s at ~22K tokens. Hosted at 175 tok/s would need ~105K.
  - At N=4, 10 local calls were killed and **321K tokens (23 % of local output) discarded**. regex-chess lost 78 of 120 min.
  - Fix: `llm_kwargs = {timeout: 3600*AMULT+300, num_retries: 0, max_retries: 0}` (`harness/README.md`).
  - Stub proof: **9 server hits for one call → 1**.
- **Also equalise or record:**
  - concurrency (N agents sharing one local engine vs independent hosted capacity);
  - temperature;
  - the thinking budget or output cap: local was 32,768 thinking tokens by a config default, hosted truncated the whole output at 32,768;
  - the request body after any proxy.
  - Time spent in the model per output token is the decisive number. Put it beside every score.

## Step 2. Engine correctness at equal quant: VANILLA (hours of one GPU)

- **Question:** at the *same* weights, does our engine compute what the upstream engine and an independent reference engine compute?
- **Arms:**
  - **L (reference):** llama.cpp at the same quant, converted from the same tensor bytes (checked by sha256).
  - **V:** upstream engine, native config.
  - **P1:** our engine with the served flags, on the same single-device layout as V, so P1 vs V isolates our code.
  - **Later, on the serving box:** D, our engine as served (multi-device); P2, D with drafting off, as the drafting placebo.
- **Inputs (identical ids for every arm):**
  1. Real agent request bodies, binned by length (DeepSeek: 40 bodies, 10 each at 2-8K, 8-20K, 20-40K and 40-70K).
  2. RENDER once through *our own serving stack*, against a fake engine that only logs the ids it is handed.
  3. Assert:
     - the server's `usage.prompt_tokens` equals the logged id count;
     - |rendered − recorded| ≤ max(32, 0.5 %) (DeepSeek: **40/40, every delta 0**).
- **Selection:** `engine-parity/select_tf.py` picks the teacher-forced set from **L's results only**, before any arm under test has run: 3 per length bin, round-robin over L's token class.
- **TF, teacher-forced:**
  1. L writes its greedy 256-token continuation with its top-20 at every position (`ref_llamacpp.py`).
  2. Each arm reads prompt + L's tokens in one pass and dumps logits.
  3. `tf_filter.py` keeps the continuation positions. The record count must be exact, or that prompt is NO DATA.
- **Metrics (`vsum.py`):**
  - KL(L ‖ X) over L's top-20 plus one rest bucket (an approximation of full-vocabulary KL, not a bound);
  - top-1 agreement;
  - first divergence.
- **FREE (report-only):** each arm's own greedy 256 tokens. Greedy paths split at near-ties: on DeepSeek the median first divergence was 18-21 tokens even between correct engines.
- **Verdict (pre-registered):**
  - ENGINE PASS if e_P1 − e_V ≤ max(0.10 × e_V, 0.002) nats AND a_P1 − a_V ≥ −0.005.
  - DEPLOY uses the layout placebo's own distance as the floor.
- **DeepSeek result:** V 0.00414 nats, 96.99 %; P1 0.00417, 96.50 %. **ENGINE PASS** (+0.00003 nats vs a 0.002 floor). No length-band trend.
- **Practical traps met:**
  - the upstream arm was OOM-killed copying 135 GB of experts under a NUMA memory policy; memory-mapping them fixed it;
  - an engine that answered everything but exited slowly after QUIT turned complete data into RC 6;
  - a sweep on the same cores made one reference prompt decode 7× slower.

## Step 3. Quant vs hosted: HOSTLP (cents, ~1.5 h of one GPU)

- **Question:** does our *quantized* model diverge from the provider's model, and is it the weights or the engine?
- VANILLA cannot see a quant cost, because every arm in it runs the same quant.
- **Steps:**
  1. **Fetch** (`hostlp/fetch_hosted.py`). Send the same selected request bodies to the provider with only model, `temperature 0`, `max_tokens 256`, `logprobs true` and `top_logprobs 20` changed. Leave thinking as recorded.
     - Check that the provider's `usage.prompt_tokens` equals your rendered count. DeepSeek: **12/12 exact**.
  2. **Map by bytes** (`hostlp/map_tokens.py`). Concatenate the steps' bytes, re-encode with your tokenizer, and accept a prompt only if there is one id per step with byte-identical pieces. Map each step's top-20 to your ids by exact bytes.
     - DeepSeek: **12/12 accepted; 58,000 top-20 entries, 0 unmapped**.
  3. **Force:** run the same teacher-forcing path as VANILLA, with the hosted tokens as the reference. For the DeepSeek adapter that is `CONT=<forced dir> TFTAG=htf PHASES=tf`.
  4. **Score** (`hostlp/hscore.py`). Per arm against hosted:
     - KL over hosted's top-20 + rest;
     - top-1 agreement;
     - hosted-token NLL gap;
     - rank of hosted's token.

     Per pair of arms on the same positions, the **engine floor**.
- **Reading (pre-registered):**
  - **QUANT COST SHOWN** when every arm sits off hosted by > 3× the between-arm disagreement and the arms agree with each other.
  - **ENGINE COST SHOWN** when one arm is off and the others are not.
  - **NO MEASURABLE DIFFERENCE** when all are within the floor.
- **DeepSeek result (2,900 positions):** V vs hosted KL 0.0438, top-1 90.34 %; P1 0.0434, 90.28 %. V vs P1 agreement 97.5 %, pair KL 0.0034.
  - Disagreement with hosted is 9.7 % against 2.5 % between engines (3.9×); KL is ~10.5× the engine-vs-llama.cpp distance.
  - **QUANT COST SHOWN.** The divergence is the 2.25-bit weights, or the provider serving something other than the original weights.
- **Limits:**
  - Chat completions gives logprobs only for *generated* tokens, so hosted's own continuation is the reference. You cannot teacher-force the provider on L's tokens.
  - llama-server has no forced-scoring path, so L was not forced on hosted tokens.
  - The provider capped `top_logprobs` at 20.

## Step 4. Matched agent reruns (hours to a day; a few euros hosted)

- **Question:** on the tasks we lose, is the loss speed, luck or quality?
- **Do this:**
  - Apply the step-1 fix on **both** sides, then run K repeats per side on the tasks lost. DeepSeek: feal-linear ×3, regex-chess ×2, query-optimize ×3.
  - Use N=1 locally unless concurrency is the thing being tested.
  - Read **only** `result.json` numbers (`agent-reruns/rsum.py`): reward, exception type, output tokens, agent minutes. That keeps trajectory text out of the analysis and out of the reports.
  - When the question is *where* the time went, use `agent-reruns/traj_metrics.py` + `gap_table.py`. These take lengths, counts and timestamps from the trajectories, never their text.
- **Interpretation:**
  - Hosted reruns first. Without hosted's own base rate nothing can be read. DeepSeek: hosted model-extraction was 1/4 and query-optimize 2/4, so the morning's "quality" losses were luck.
  - Report tokens-to-solve and tok/s beside each pass rate. A task can be a speed task (regex-chess hosted needs ~630-656K tokens, ~135-150 min at local speed against a 120 min limit) or a search-efficiency task (feal-linear: hosted solved within 221K tokens in 3 of 4 runs, local in 0 of 4).
  - At these counts, a single WIN is a lead, not a result: Fisher one-sided p ≈ 0.07 at n=4 each. Three arms × two tasks gives a ~67 % chance of at least one false WIN.
- **DeepSeek after the fix:** hosted 4/5 vs local 2/5 on the two gap tasks; local query-optimize 0/3 (fast wrong answers, 7-9 min, 18-23K tokens).

## Step 5. The rules that hold it together

1. **Pre-register** the prediction and the verdict rule before any data: thresholds, floors, which prompts. Use the selection-from-reference-only trick.
   - A deviation found after data is written as a deviation, never folded into the rule. Example: HOSTLP's L arm was not forced.
   - A template is in `templates/prereg-template.md`.
2. **A ledger row per result, pass or fail:** date, lead, change, prediction, [OBSERVED] result with its numbers, verdict (PASS / FAIL / NO EFFECT / NO DATA / MECH), [INFERRED] lesson, evidence path.
   - Never delete a row. A superseded verdict gets a pointer.
   - MECH means a harness or build defect, not a finding about the model or engine.
   - The DeepSeek ledger has 381 rows (`LEDGER-INDEX.md`). Failures and refutations are a third of them, and they are what LESSONS.md is built from.
3. **Missing data is never a pass.**
   - Every summariser lists missing or short files as NO DATA, and an arm missing more than 2 of 12 prompts is NO DATA as a whole.
   - Every verdict line can say NO DATA.
   - Record counts are checked exactly (prompt + continuation).
   - An exit code is not data: a completed run can exit non-zero (slow exit), and a zero exit can hide a phase that never engaged. Check the engagement line.
4. **Placebos and floors:**
   - Every comparison needs a noise floor measured the same way: the same quant on two engines, a layout placebo, a drafting-off placebo.
   - KL numbers mean nothing alone. 0.004 nats is "two correct engines" and 0.044 is "different weights".
5. **Numbers only out of private data.**
   - Prompt bodies, token ids of prompts, hosted continuations and trajectories stay where they are.
   - Scripts print ids, counts, logprobs and timings.

---

## Costs and timings observed (DeepSeek V4.1 Flash, IQ2_XS 2.25 bpw, ~155 GB; one RX 7900 XTX + host-RAM experts)

| Item | Observed |
|---|---|
| Harness stub proof | seconds; no GPU |
| RENDER (40 bodies through the serving stack + fake engine, CPU) | minutes; 40/40 delta 0 |
| Convert to llama.cpp layout (155 GB read + write + verify) | estimated 20-45 min |
| Arm L, llama.cpp 40 prompts × 256 tokens (experts on CPU) | 12,457 s ≈ 3.5 h; prompt median 106 tok/s, decode median 9.0 tok/s; health after 125 s |
| Arm V / P1, TF 12 prompts (312K ids, windows of 2048) | 2,561 s / 2,509 s ≈ 42 min each; ~147 tok/s forced reading |
| Arm V / P1, FREE 40 prompts | 4,549 s / 4,311 s ≈ 75 min each |
| HOSTLP fetch, 12 requests (259K input tokens, 2,900 output) | ~24 s wall, **EUR 0.07** |
| HOSTLP force, V and P1 (12 prompts each) | ~42 min each |
| HOSTLP map + score | seconds (CPU) |
| Hosted agent reruns (feal ×3 + regex-chess ×2) | estimated ~EUR 2; up to 60 / 120 min agent limit each (AMULT 2) |
| Local agent reruns at N=1 | regex-chess 57-120 min, feal 60 min (limit), query-optimize 7-9 min per run |
| MIXQ Stage A (offline weight-space error) | ~28.5 min per layer at 32 threads; 77-89 s to stream a 7.2 GB layer slice |
| Total for steps 1-3 on one GPU | about one night (~6.5-10 h, mostly the reference arm and the FREE passes; FREE can be dropped) |
