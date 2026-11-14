# gfx906 helper: test gates

Test scripts for the prototype in `docs/dsv41/cards/EB-01.md`. The prototype runs an RX 7900 XTX (gfx1100) as the primary
and an MI50 (gfx906, wave64) as a helper expert cache (`--expert-cache-device1`), from one fat HIP binary. Read the WP1, WP2
and WP3 notes alongside this page. None of these scripts has been run against the hardware yet. They have been checked only
offline, against stub binaries and fake tool output.

| File | Role |
|---|---|
| `gates.sh` | Runs inside the container. Runs gates gate0–gate5 in ascending order and stops at the first FAIL. |
| `gate_tools.py` | Standard-library helpers: the rocminfo device map, the thermal guard, the serve-protocol driver, the tokenizer call, the pack check and the per-gate parsers. |
| `compare_logits.py` | The teacher-forced comparison for gate4: top-1 agreement, median and p99 KL, and perplexity. |
| `text-code.txt` | The fixed ~512-token text for gate4: the first 3.2 KB of `src/core/remote_experts.cpp`. |
| `text-8k.txt` | The ~8K-token prompt for gate5: `src/core/expert_cache.cpp`. It is about 10K tokens, and the first 8,192 are used. |

## Running

`gates.sh` runs inside a ROCm 7.14 container that sees both GPUs (`/dev/kfd` and `/dev/dri`), with host IPC and unlimited
memlock. It expects the build tree and its own files under `/strata` and the model shards under `/gguf`
(`GATE_Y` / `GATE_M` change the two roots):

```sh
rocminfo | grep -E '^\s+(Name|Marketing Name|Node):'     # GPU agents in order -> pick the XTX's and the MI50's index
HIP_VISIBLE_DEVICES=<xtx>,<mi50> bash /strata/gates/gates.sh                 # all gates
HIP_VISIBLE_DEVICES=<xtx>,<mi50> bash /strata/gates/gates.sh gate0 gate1 gate3   # a subset (still run in ascending order)
```

- Pass `HIP_VISIBLE_DEVICES`, every `GATE_*` variable and any per-node `HSA_OVERRIDE_GFX_VERSION_<n>` into the container.
- **Logs** go to `/strata/gates/<UTC timestamp>/`:
  - `gates.log`: the console;
  - `summary.txt`: one PASS or FAIL line per gate, with the gate's duration and its key numbers;
  - one log per command;
  - `devices.json`: the device map.
- **Exit codes:** 0 means every selected gate passed, 1 means a gate failed, 2 means a precondition is missing, and 3 means
  the device map is wrong.

### Preflight: always runs, before any gate

1. **Device map.** `rocminfo` lists the GPU agents in HSA order, and `HIP_VISIBLE_DEVICES` (indices or `GPU-` UUIDs) is
   applied to that list. The script prints the HIP ordinal, arch, wave size, marketing name and KFD node of each GPU. It
   refuses to go on (exit 3) unless HIP 0 is gfx1100 and HIP 1 is gfx906. No HIP process runs before this check.
   `strata-device` cross-checks the map in gate1.
2. **`HSA_OVERRIDE_GFX_VERSION`.** The script refuses to run if the global variable is set, and it never sets one itself.
   A per-node `HSA_OVERRIDE_GFX_VERSION_<n>` is only printed. On ROCm 7.14 with native gfx906 code objects none should be
   needed. If one turns out to be needed, it has to be the per-node form.
3. **Thermal guard.** Every GPU command runs under a monitor that reads the amdgpu hwmon sensors of both GPUs every 5 s. It
   follows the path KFD topology node → `drm_render_minor` → `/sys/class/drm/renderD*/device/hwmon`.
   - The command's process group is killed when any sensor reaches `GATE_TEMP_ABORT_C` (default 85 °C). The gate then fails
     with "thermal abort" (rc 125).
   - A warning is printed at `GATE_TEMP_WARN_C` (default 80 °C).
   - If no sensor can be read, the script refuses to run unless `GATE_ALLOW_NO_THERMAL=1`. The MI50 shuts down near 100 °C
     junction.

## The gates

The runtimes are estimates, not measurements. Engine start-up dominates gate4 and gate5: loading the dense weights, filling the
XTX and MI50 caches, and reading or pinning the expert arena from the GGUF.

### gate0: HIP runtime probe (`/strata/probe/probe_mi50`)

**What it proves.** A process with both GPUs visible starts and runs normally. This holds even though most of the fat
binary's TUs have no gfx906 code. The design calls this risk 3: "the HIP runtime might refuse to start". The probe also
shows that the gfx906 code object of a dual-arch TU (TU-A) runs on the MI50 with `warpSize` 64. Finally, it shows that a
gfx1100-only TU (TU-B) fails cleanly on the MI50 without killing the process.

**Pass criteria.**
- The probe exits normally, with no signal and no timeout.
- The output contains no `hipErrorNoBinaryForGpu`, `Aborted` or `terminate called`.
- It prints `probe done`.
- The TU-A line for the MI50 reports warp size 64.

The TU-B result on the MI50 is recorded but does not gate. The expected outcome is "failed cleanly (expected), the process
went on". If TU-B did not fail, that is flagged as unexpected.

**Parsing.** The probe's source is not in this repository, so the output format is assumed, not known.
- A line that names a device (`device 1`, `dev 1`, `gfx906`, …) sets the device for the TU lines that follow it.
- A TU line is any line that matches `TU-A` or `TU-B`.
- A warp value is read from `warpSize=64`, `warp size 64` or `wave64`.

If the real format differs, set `GATE_G0_TUA_RE` and `GATE_G0_TUB_RE`. If the MI50's TU-A line cannot be found, the gate
fails. It never passes by default.

**Runtime:** under 1 min.

### gate1: `strata-device --list-devices`

**What it proves.** The WP2 arch check classifies the cards correctly: the XTX as a normal primary and the MI50 as
helper-only.

**Pass criteria.**
- Device 0 is `arch gfx1100, …, wave32`, with no `cannot run` or `helper-only` line.
- Device 1 is `arch gfx906, …, wave64`, with the line `helper-only (gfx906, wave64): …` (`src/core/device_main.cpp`).

**Runtime:** under 1 min.

### gate2: kernel parity on the MI50 alone

**What it proves.** The helper TUs' gfx906 code computes the native experts correctly in both policies that apply to the
MI50. Mode 0 is the CUDA layout, the same kernels as the XTX. Mode 7 is the wave64 default (LDS, R2 pairs). gate2 also
checks two parts of WP3's per-device policy: the one-pass SwiGLU kernel, and the refusal of mode 1 on wave32.

`HIP_VISIBLE_DEVICES` is set to the MI50's token only. Each of these runs with `STRATA_EXP_MODE=0` and again with `=7`:

| Run | argv (from `src/kernels/native_expert_parity.cpp` / `native_grouped_parity.cpp`) | Pass criterion |
|---|---|---|
| expert-synthetic | `native_expert_parity --synthetic q4_K/q5_1 iq3_s/iq4_nl` | rc 0 and `native_expert_parity: 0 failures` |
| expert-shard | `native_expert_parity /gguf/…-00001-of-00002.gguf 0 10 20 30 47` (expert 7 of each layer, from the real shards) | the same |
| grouped | `native_grouped_parity` (synthetic only; the tool has no shard mode) | rc 0 and `native_grouped_parity: 0 failures` |

`native_expert_parity` counts a failure in any of these cases:
- the GPU or CPU result is off the float reference by rel L1 ≥ 3e-2;
- the decode-once kernels are not bitwise equal to the per-entry kernels;
- a dequantizer is off by rel ≥ 1e-6, or its fp16 rounding is wrong;
- the Hadamard rows are off by ≥ 1e-6.

`native_grouped_parity` fails whenever the output is not bitwise equal to the v1 launches.

Two more runs come from WP3's command list:
- **Fused SwiGLU on wave64:** `STRATA_EXP_MODE=7 STRATA_HIP_SWIGLU_FUSED=1 native_grouped_parity` on the MI50.
- **Mode 1 refused on wave32:** `STRATA_EXP_MODE=1 native_grouped_parity` on the XTX alone. This must exit 1 with the
  message `STRATA_EXP_MODE=1 (W64 …` and nothing else.

**Reference build.** If `/strata/build/gfx906/native_expert_parity` exists (and `native_grouped_parity`, when present), the
same synthetic and shard runs are repeated with it on the MI50. The script also reports whether the `cpu rel` / `gpu rel`
figure lines are identical between the two builds. These runs are informative and do not gate. If the tools were not built
there, the step is skipped and the summary line says so.

**Runtime:** about 3–8 min, or double that with the reference build. The shard runs read five layers' experts from the
GGUF.

### gate3: cross-device bitwise (`cross_device_grouped_parity 0 1 q4_K/q5_1 iq3_s/iq4_nl`, then the same with `--strict`)

**What it proves.** The design's go/no-go test: the same `native_expert_grouped` code gives the same bits on the XTX and on
the MI50. Expected outcomes from the tool's header and WP3 §4:
- in mode 0 on both cards, the q8_1 of x (1), the gate/up floats (2) and the down projection on identical hq (4) are
  bitwise equal;
- the SwiGLU q8_1 codes (3) differ by at most ±1 in at most 0.1 % of the codes, because `__expf` differs between GCN and
  RDNA3;
- the whole pipeline (5) differs by rel L1 ≤ 1e-4;
- every run matches the CPU float reference to rel L1 < 3e-2.

**Pass criteria.**
- The plain run exits 0. Exit 1 means a device missed the CPU reference. Exit 77 (fewer than two devices) is a fail, not a
  skip.
- Device A is gfx1100 wave32 and device B is gfx906 wave64.
- For every pair, (1), (2) and (4) in the "A mode 0 vs B mode 0" block read `bitwise equal`. This is the design's no-go:
  "not bitwise identical in mode 0 and the cause can't be traced to a shim primitive".

The `--strict` run additionally counts (3) beyond ±1 / 0.1 % and (5) above 1e-4. If only those fail, the gate passes with
a WARN in the summary, because they are statistical by construction (WP3 §5).

**Runtime:** about 1–2 min.

### gate4: end to end, teacher-forced

**What it proves.** With the MI50 computing real decode experts inside the engine, the model's next-token distribution on a
fixed text matches the XTX-only engine. This holds with zero-copy pinned memory on and off (design hazard 5), and with
`--remote-expert-opt`.

| Arm | Change from the shared arguments |
|---|---|
| a | none: the XTX alone, `--expert-cache auto` (the reference) |
| b | `--expert-cache-device1 auto` |
| c | b plus `STRATA_REMOTE_ZEROCOPY=1` (zero-copy forced on; a wave64 helper copies by default) |
| d | b plus `--remote-expert-opt` |

Every arm starts its own engine, with both GPUs visible. The shared arguments are:

```text
strata --serve --pack <native pack> --native /gguf/…-00001-of-00002.gguf --expert-profile <src>/data/expert-profile.bin
       --spec 4 --spec-min-p 0.5 [--mtp <rt>] --prompt-cache 0 --conversation-cache-mib 0
       --expert-cache auto --prefill 512 --short-read 640 --max-context 2048 --kv fp16
       --adapt-every 100000 --pcie-frac 0 --suffix-draft 0
env    STRATA_ROUTE_TAIL_SKIP=0 STRATA_PREFILL_CPU_SHARE=0 STRATA_PCIE_FRAC_DEFAULT=old
       STRATA_LOGPOS=<log>/gate4-<arm>.tsv STRATA_LOGPOS_TOPK=256
```

- The engine gets one request, `GEN 1 ckpt=0 <512 ids>`.
- `--short-read 640` sends all 512 tokens through the verify windows. That is the decode path the helpers serve, and every
  position is scored there (`Verifier::window_logprobs`, `src/core/verify.cpp`).
- The engine finds the PLE table's shard itself, by tensor name, so no `--ple-gguf` is given.
- The fixed settings (no adaptation, no PCIe share, no tail skip, no suffix drafts) keep the expert placement identical
  between the arms. They are the settings `tools/quant_quality.py` uses. Without them, the runs would differ for reasons
  other than the helper. Tail skip in particular depends on how much is cached.

**Evidence that the helper computed.** Strata prints `strata generate: CUDA1: N additional experts, X GiB; results return
through pinned host rows` at start-up. After each request it prints `strata serve: CUDA1: N expert entries, M active layer
launches, … in this request; host … ms staging+launching, … ms waiting` (`src/program/generate.cpp`). The `N` in that
second line is `RemoteExperts::computed()`.

**Pass criteria** (`compare_logits.py`, then `gate_tools.py g4`):
- All arms ran and answered `DONE`.
- Arm a has no helper lines. Arms b, c and d each have more than 0 helper experts at start-up and more than 0 entries
  computed in the request.
- Each of b, c and d against a, over at least 384 of the 511 scored positions, with the same target token at every
  position:
  - top-1 agreement ≥ 95 % (`GATE_G4_MIN_TOP1`);
  - median KL(a ‖ run) ≤ 1e-3 (`GATE_G4_MAX_MEDIAN_KL`);
  - perplexity of the actual text within noise: |mean per-token NLL difference| ≤ 2 paired standard errors, or the
    perplexity ratio within 1 %.
- KL is computed over arm a's top-256 tokens, both distributions renormalized to that set (`tools/quant_quality.py`'s
  method). The p99 and the mean are reported too.
- For scale, the prior cache-on/off study (`bench/results/2026-09-27-cache-parity`: RTX 5070, Q2_0) measured 97.7 %
  top-1 and a median KL of 3.0e-4 on 512 code tokens, between GPU experts and CPU-pool experts.

**Runtime:** about 15–35 min: four engine starts at about 3–8 min each, plus about 20–60 s per 512-token read. A first run
that has to build the native pack adds about 5–20 min of CPU and disk time (see "Prerequisites").

### gate5: decode and prefill speed, (a) against (b)

**What it proves.** The helper pays for itself in single-request decode.

The engines use the shared arguments without the gate4 pinning, with `--expert-cache auto --prefill auto --max-context 9472`.
Arm b adds `--expert-cache-device1 auto`. `STRATA_DECODE_TIMING=1` is set, and `GATE_G5_ENV` adds more variables. Each
engine gets:
1. one warm-up request, `GEN 32 ckpt=0 <first 512 ids>`;
2. three timed requests, `GEN 256 ckpt=0 <8,192 ids>`.

With `--prompt-cache 0` and `ckpt=0`, every timed request reads the whole prompt again.

**Reported per request:**
- decode tok/s (= generated × 1000 / decode ms);
- prefill tok/s (= tokens read × 1000 / prompt ms);
- drafts accepted / offered.

These all come from the `DONE` line. The helper figures come from the `CUDA1` line: entries, entries per token, and ms
waited. The CPU-pool figures come from the `strata decode timing` line: CPU-pool ms per window, GPU-reach wait, and CPU
entries per layer-window. The `prefill helper source` blob count is reported when the engine prints it.

**Pass criteria.**
- Three usable timed requests per arm.
- The helper computed more than 0 entries in b.
- **Abort criterion:** median decode tok/s of b < median of a is a FAIL.

The two arms decode greedily from the same prompt. Because the cache contents differ, they can still emit slightly
different tokens with different draft acceptance. The accepted/offered counts are printed so this can be seen.

**Runtime:** about 15–25 min: two engine starts at about 3–8 min each, plus four requests per arm. The 8K prefill is
roughly 10–30 s per request.

## Where the design's flags do not exist as written (nearest real mechanism)

1. **`--dump-logits` writes nothing with this model.** The flag exists (`generate.cpp`). It writes `int32 n_vocab, int32
   n_rows`, then raw float32 rows, and the header goes out with the first row. But only the per-token loop writes it. A
   native (IQ) pack, which is what the IQ3_S GGUF is, "never reaches the per-token dump site". That loop breaks to the
   verify windows at once (`generate.cpp`, `if (native_pack) { spec_pos = pos; break; }`). The nearest real mechanism is
   `STRATA_LOGPOS` with `STRATA_LOGPOS_TOPK=256` on a serve engine whose prompt is read through the verify windows. That
   is the method of `docs/UNSLOTH_Q4.md`, `docs/QUANTS.md` and `tools/quant_quality.py`. `compare_logits.py` still reads
   the `--dump-logits` format, for a canonical pack.
2. **`--remote-expert-opt` requires `--serve`.** Without it the engine refuses: `strata generate: --remote-expert-opt
   requires --serve`. That is the main reason all gate4 and gate5 arms run as serve engines.
3. **There is no plain "model path" flag.** The engine always loads a pack directory (`--pack`, default `pack/full`). For a
   GGUF it needs a native pack built by `tools/iq_pack.py` plus `--native <shard 1>`. A native pack also requires `--spec T`
   with T ≥ 2 and `--prefill CHUNK`.
4. **The engine takes token ids, not text** (`--tokens`, `--tokens-file`, or the serve protocol's `GEN`). The texts are
   tokenized with `tools/strata_tokenizer.py`, from the pack's `tokenizer/` directory or from the GGUF's metadata.
5. **`native_grouped_parity` has no real-shard mode.** Its only argument is `--bench`. Only `native_expert_parity` runs on
   shard layers.
6. **Zero-copy has no log line.** `STRATA_REMOTE_ZEROCOPY=0` is read in `RemoteExperts::open`
   (`src/core/remote_experts.cpp`), but nothing prints which mode is active. Arm c is identified by its environment only.
   Its effect shows in the helper's host staging ms.
7. **Helpers need `--expert-profile` and a CUDA0 cache.** Otherwise: "remote experts need --expert-profile, a CUDA0 expert
   cache and the expert pool". The shipped `data/expert-profile.bin` is passed.

## Prerequisites gate4/gate5 resolve (printed at the start of gate4)

- **Source tree:** `GATE_SRC`, or else `CMAKE_HOME_DIRECTORY` from `/strata/build/helper/CMakeCache.txt`. It must be visible in
  the container. It provides `tools/` and `data/expert-profile.bin`.
- **Native pack:** `GATE_PACK`, or else any `native_experts.txt` under `/strata` (depth 4, skipping `containers/`, `build/` and
  `gates/`).
  - A pack is used only if `gate_tools.py pack-check` confirms that it was cut from exactly this GGUF: every layer's
    gate/down type, and the absolute offsets of gate, up and down in their shards.
  - If no pack checks out, one is built at `/strata/packs/iq3_s-gates` (`GATE_PACK_OUT`) with
    `python3 tools/iq_pack.py --gguf <shard 1> --out …`. That step is CPU and disk only, writes under `/strata`, and needs numpy.
    `GATE_BUILD_PACK=0` turns it off.
- **MTP drafter:** `GATE_MTP=<dir>`, or `none`, or else a single `dense.txt` + `experts.bin` runtime under `/strata`.
  - If several are found, the script stops and asks for `GATE_MTP`.
  - If none is found, the engines run without `--mtp`, so the windows draft only from prompt lookup. gate4 is unaffected,
    because it reads with a fixed text. gate5's absolute tok/s are then lower than a configured serve, though (a) against (b)
    is still like for like.
- **Token ids:** the tokenizer needs the Python `regex` module. Without it, tokenize elsewhere and pass the files with
  `GATE_IDS_CODE` (at least 512 ids) and `GATE_IDS_8K` (at least 8,192 ids), comma-separated.
- **Expert arena:** by default the experts are copied into pinned RAM (the arena). `GATE_MMAP_EXPERTS=1` adds
  `--mmap-experts`, which maps the GGUF shards in place. Use it if the pinned allocation fails or RAM is short.

## Configuration (`GATE_*`)

| Variable | Default | Meaning |
|---|---|---|
| `GATE_BUILD` / `GATE_REF_BUILD` | `/strata/build/helper` / `/strata/build/gfx906` | the build directories |
| `GATE_PROBE` | `/strata/probe/probe_mi50` | the gate0 probe |
| `GATE_SHARD1` / `GATE_SHARD2` | `/gguf/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf` | the model |
| `GATE_SRC`, `GATE_PACK`, `GATE_PACK_OUT`, `GATE_BUILD_PACK`, `GATE_MTP`, `GATE_PROFILE` | see above | engine prerequisites |
| `GATE_IDS_CODE`, `GATE_IDS_8K` | tokenized at run time | pre-tokenized ids |
| `GATE_G2_LAYERS` / `GATE_G2_PAIRS` | `0 10 20 30 47` / `q4_K/q5_1 iq3_s/iq4_nl` | gate2 and gate3 |
| `GATE_G4_ARMS` | `a b c d` | must start with `a` |
| `GATE_G4_TOKENS`, `GATE_G4_MIN_TOP1`, `GATE_G4_MAX_MEDIAN_KL` | 512, 95, 1e-3 | gate4 |
| `GATE_G5_PROMPT`, `GATE_G5_GEN`, `GATE_G5_REPEATS`, `GATE_G5_ENV` | 8192, 256, 3, — | gate5 |
| `GATE_ENGINE_EXTRA`, `GATE_MMAP_EXPERTS` | —, 0 | extra engine arguments for gate4 and gate5 |
| `GATE_TEMP_ABORT_C`, `GATE_TEMP_WARN_C`, `GATE_ALLOW_NO_THERMAL` | 85, 80, 0 | the thermal guard |
| `GATE_G0_TUA_RE`, `GATE_G0_TUB_RE` | `\bTU[-_ ]?A\b`, `\bTU[-_ ]?B\b` | the gate0 probe parsing |
| `GATE_LOG_DIR` | `/strata/gates/<UTC timestamp>` | the log directory |

## Open points

- **gate0's output format is assumed.** The probe's source is not in this repository.
- **The source tree must be visible in the container.** The build was configured inside the container, so the path in
  `CMakeCache.txt` should resolve there. If it does not, set `GATE_SRC`.
- **Python dependencies in the image are unknown.** `python3` is assumed (ROCm's own `rocm_agent_enumerator` needs it).
  `regex` (the tokenizer) and `numpy` (the pack build) may be missing.
- **The thermal sensor path inside podman is unverified.** It is `/sys/class/kfd/kfd/topology/nodes/<node>/properties` →
  `drm_render_minor` → `/sys/class/drm/renderD<minor>/device/hwmon`. If it is not visible, the gates refuse to run until
  `GATE_ALLOW_NO_THERMAL=1` is set.
- **The 85 °C abort threshold is a conservative choice.** It is not a measured limit for this chassis.
