# How fast must each agent's stream be?

Date: 2026-11-20. Model: Qwen3.8 Flash-Next on Gary (our fork of Strata).

The question: when one machine serves many coding agents at once, how slow can each agent's stream get before agent
jobs start timing out? This is about machine clients finishing their tasks, not about how fast output feels to a person.

**Short answer: at least 40 tok/s of sustained decode per agent, and time to first token (TTFT) of at most 30 s at p95.**
Clients also need a per-request timeout of at least 1,800 s and no automatic retries.

Labels: **MEASURED** = read from run data. **ESTIMATE** = our projection from that data.

## Method

A re-analysis of agent-benchmark runs we already had; no new GPU runs. The runs used 27 Terminal-Bench 2.x tasks with an
agent benchmark harness, thinking on (95-96 % of output text is reasoning), temperature 0.6, and the tasks' time
limits doubled. Each server had one agent, so every speed below is what **one agent alone** saw.

Two Flash-Next quants on Gary, one XTX plus host RAM: Unsloth UD-IQ4_XS (43 trials, 25 passed, 18 hit the task limit)
and IQ3_S (96 trials, 54 passed, 34 hit the limit, 8 server or cancel errors). Many trials are reruns of tasks that had
timed out before, so the pool is weighted towards hard tasks. A first-attempt-only sample (54 trials, one per task per
quant) lands within about 4 points of the full pool on the key projection.

## Per-turn and per-task distributions (MEASURED)

Flash-Next, 139 trials, 2,098 turns:

| Metric | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| Prompt tokens per turn (whole context) | 10,340 | 41,908 | 50,345 | 60,808 | 72,825 |
| New (uncached) prompt tokens per turn | 964 | 3,758 | 4,949 | 7,523 | 49,528 |
| Output tokens per turn (thinking included) | 899 | 8,828 | 20,056 | 49,991 | 96,830 |
| Turns per task | 12 | 28 | 34 | 59 | 62 |
| Model time per turn (s) | 23 | 170 | 371 | 898 | 2,041 |
| Agent wall time per task (s) | 1,294 | 3,600 | 3,600 | 7,200 | 7,200 |

- Model time is 78 % (mean) to 85 % (median) of agent wall time; the rest is the agent waiting for its commands. Model
  speed decides whether a task finishes.
- Output is heavy-tailed: 11 % of turns write more than 8k tokens and 63 turns write 32k or more. Single turns reached
  96,830 tokens although the harness declared a 32,768-token output limit; the server did not enforce it. These
  runaway thinking turns drive the p99.
- For comparison, a dense 27B model on llama.cpp (1,191 turns) had output p95 24,692 and max 38,083 tokens, and model
  time p95 404 s.

Measured single-agent speeds:

| Quant | Decode on long turns (> 8k out), p10 / p50 / p90 | Least-squares fit per turn | Short-output turns (TTFT proxy), p50 / p95 |
|---|---|---|---|
| UD-IQ4_XS | 43.7 / 47.9 / 51.9 tok/s | 5.2 s + new/347 + out/48.9 | 7.3 / 15.1 s |
| IQ3_S | 43.4 / 71.8 / 80.4 tok/s | 4.1 s + new/204 + out/65.6 | 5.3 / 11.0 s |

The prefix cache worked: the median turn prefilled only about 1k new tokens, and warm-cache TTFT was about 5-15 s.
**Even at 45-70 tok/s alone, 37 % of Flash-Next trials hit the doubled task limit** (42 % at ~48 tok/s, 35 % at ~72).
Task limits already bind at today's single-agent speed. Measured against per-request limits, 14.7 % of turns took more
than 120 s, 6.5 % more than 300 s and 2.5 % more than 600 s; 31 of 139 trials had at least one turn over 600 s. Short
per-request timeouts (minutes) cannot carry thinking-on agent work at any speed.

## Projection by per-agent speed (ESTIMATE)

Every measured turn is replayed at decode speed S: turn time = TTFT + output / S. TTFT follows the measured warm-cache
behaviour (4.3 s + new tokens / 234 tok/s; p95 25 s). Each trial's non-model time is kept as measured. "Completers kept"
asks whether the 79 trials that finished inside the limit at measured speed would still finish.

| S (tok/s per agent) | Turn p95 | Turn p99 | 2 x p95 | Task wall p50 / p95 | Completers kept, x2 limits | x1 (stock) limits |
|---|---|---|---|---|---|---|
| 10 | 2,013 s | 5,021 s | 4,026 s | 2,543 / 10,527 s | 46 % | 30 % |
| 15 | 1,345 | 3,355 | 2,689 | 1,790 / 7,224 | 57 % | 38 % |
| 20 | 1,010 | 2,521 | 2,021 | 1,396 / 5,569 | 67 % | 42 % |
| 25 | 810 | 2,021 | 1,619 | 1,205 / 4,576 | 73 % | 49 % |
| 30 | 676 | 1,688 | 1,352 | 1,057 / 3,914 | 80 % | 51 % |
| **40** | **509** | **1,271** | **1,018** | **871 / 3,087** | **90 %** | **61 %** |
| 50 (about measured) | 409 | 1,019 | 817 | 750 / 2,591 | 95 % | 67 % |

- First attempts only (35 completers), x2 limits: 74 % kept at 20 tok/s, 86 % at 30, **94 % at 40**, 97 % at 50.
- Dense 27B (61 completers), x2 limits: 72 % at 20, 87 % at 30, 95 % at 40.
- If the server enforced `max_tokens` = 32,768, the turn p99 at 40 tok/s would fall from 1,271 s to 831 s; p95 would
  not change.
- There is no plateau: every tok/s lost costs completions.

TTFT sensitivity (completers kept, x2 limits):

| TTFT model | TTFT p95 | S = 30 | S = 40 | S = 50 |
|---|---|---|---|---|
| Warm prefix cache, 234 tok/s prefill of new tokens (as measured) | 25 s | 80 % | 90 % | 95 % |
| Warm cache, 500 tok/s prefill | 14 s | 81 % | 91 % | 95 % |
| Cache miss every turn, 1,000 tok/s prefill | 55 s | 78 % | 86 % | 92 % |
| Cache miss every turn, 500 tok/s prefill | 105 s | 75 % | 85 % | 89 % |

## Recommended floor

- **At least 40 tok/s sustained decode per agent, under full load.** It keeps 90-94 % of today's completers inside the
  doubled limits, and turn p95 near 510 s. 30 tok/s loses 1 in 5 completers; 20 loses 1 in 3.
- **TTFT at most 30 s at p95** for normal turns (about 5k new tokens on a warm prefix cache). A cold full-context prompt
  (about 60k tokens) should see its first token within 60-120 s, which needs 500-1,000 tok/s of prefill.
- **Per-request timeout of at least 1,800 s, with no automatic retries.** At 40 tok/s that is more than 3x margin at p95
  and 1.4x at p99. A retry re-runs a 20-minute turn from scratch. A 600 s timeout would need about 73 tok/s per agent.
- **Keep each agent's prefix cache between its turns.** Losing cached prefixes under load costs about 5 points of
  completion at 40 tok/s, more at lower speeds.
- **Enforce `max_tokens` on the server.** It trims the p99, not the p95.

## What this means for the current engine

Per-agent decode on 384-token requests (not agent prompts) [MEASURED,
[comparative timing](data/2026-11-21-comparative-timing.md), [shared pool](data/2026-11-21-dp-pool.md)]:

| Layout | 1 | 2 | 4 | 8 users |
|---|---|---|---|---|
| tp2: 2 XTX tensor parallel + 4 MI50 helpers (before NX-09; 8 users after it: 19.4) | 50.5 | 40.6 | 31.6 | 18.8 |
| Two workers + shared MI50 pool (see [SHARED-POOL.md](SHARED-POOL.md)) | 53.6-54.4 | 52.3 | 39.8 | 28.2 |

The two rows come from different sessions, and the pool row's 1-2 user cells from a separate one-trial test.

So the 40 tok/s floor holds today for about two thinking-on agents on tp2, and about four on the shared pool (39.8, just
at the line). At eight agents both layouts sit in the 20-30 tok/s band, which loses roughly a quarter to a third of
completions.

## Caveats

- **Sample size.** 139 trials over 27 tasks, 79 completers, one harness. The first-attempt check agrees, but the
  percentages carry roughly 5-8 points of noise.
- **Single-agent runs.** Every run had one agent per server. Serving many agents adds queueing and batching effects
  these runs never saw. The floor is the speed each agent must actually receive.
- **Harness.** This harness resends the whole conversation each turn (prompt p95 50k), sends no `max_tokens` and calls
  the model without streaming, so the timeout covers the whole response. Other harnesses compact context differently.
- **Thinking on.** About 95 % of output is reasoning. With thinking off, output per turn would drop several-fold and a
  much lower floor would do; we have no thinking-off data to put a number on it.
- **Two quants, one layout.** Both ran on a single XTX plus host RAM. Turn-time shape (for example TTFT under batching)
  may differ on multi-card layouts.
- **Projection.** Non-model time is treated as fixed. Trials that already timed out are excluded.
- The doubled (x2) and stock (x1) limit columns bracket the time limits a real agent job might have.
