#!/usr/bin/env python3
"""tools/dsv41/gates/gate_tools.py - the helpers gates.sh calls (gfx906 helper, docs/dsv41/cards/EB-01.md).

Standard library only (no numpy), so it runs in the build image as it is.  Subcommands:

  devmap      rocminfo -> HIP ordinals (HIP_VISIBLE_DEVICES applied), the KFD node and the hwmon sensors of each;
              exit 3 unless HIP device 0 is gfx1100 and device 1 is gfx906; also refuses a global
              HSA_OVERRIDE_GFX_VERSION (only the per-node HSA_OVERRIDE_GFX_VERSION_<node> form is acceptable)
  temps       print the current temperatures of the mapped GPUs
  guard       run one command with a timeout and a thermal abort, its stdout+stderr into a log
  serve       drive one `strata --serve` engine over its stdin protocol (READY / GEN / DONE / QUIT) with the same
              thermal abort; per request the DONE fields and the stderr lines the request printed
  ids         tokenize a text with the repo's tokenizer (tools/strata_tokenizer.py) and keep the first N ids
  pack-check  check that a native pack (tools/iq_pack.py) was cut from exactly this GGUF (types and offsets)
  g0 g1 g3    parse the probe / strata-device / cross_device_grouped_parity output into PASS/FAIL
  g4          the helper-entry checks of the teacher-forced runs
  g5          the decode/prefill speed table and the abort criterion

Exit codes of guard/serve: the child's own code; 124 timeout; 125 thermal abort; 126 the engine ended early.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

WANT = {0: "gfx1100", 1: "gfx906"}   # HIP ordinal -> base arch this prototype requires


# ----------------------------------------------------------------------------------------------- device map
def rocminfo_text() -> str:
    for exe in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, PermissionError):
            continue
        except subprocess.TimeoutExpired:
            raise SystemExit("rocminfo did not finish in 120 s")
        if r.returncode == 0:
            return r.stdout
        raise SystemExit(f"{exe} failed (rc {r.returncode}): {r.stderr.strip()[:400]}")
    raise SystemExit("rocminfo not found (PATH, /opt/rocm/bin)")


def parse_rocminfo(text: str) -> list[dict]:
    """The GPU agents in rocminfo order.  Only the first occurrence of a field inside an agent counts (the ISA and
    pool sub-sections repeat `Name:`)."""
    agents, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*Agent\s+(\d+)\s*$", line)
        if m:
            cur = {"agent": int(m.group(1))}
            agents.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^\s{1,4}([A-Za-z][A-Za-z ]*?):\s*(.*?)\s*$", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2)
        if key in ("Name", "Uuid", "Marketing Name", "Device Type", "Node", "Wavefront Size", "Compute Unit") \
                and key not in cur:
            cur[key] = val
    gpus = []
    for a in agents:
        if a.get("Device Type", "").upper() != "GPU":
            continue
        wave = re.match(r"(\d+)", a.get("Wavefront Size", ""))
        gpus.append({"agent": a["agent"], "arch": a.get("Name", "?").split(":")[0], "uuid": a.get("Uuid", ""),
                     "marketing": a.get("Marketing Name", ""), "node": int(a["Node"]) if a.get("Node", "").isdigit() else None,
                     "wave": int(wave.group(1)) if wave else None})
    for i, g in enumerate(gpus):
        g["hsa_gpu_index"] = i
    return gpus


def sensors_of(node) -> list[dict]:
    """The amdgpu hwmon temperature inputs of one KFD node (topology -> render minor -> /sys/class/drm)."""
    if node is None:
        return []
    props = pathlib.Path(f"/sys/class/kfd/kfd/topology/nodes/{node}/properties")
    minor = None
    try:
        for line in props.read_text().splitlines():
            if line.startswith("drm_render_minor "):
                minor = int(line.split()[1])
    except OSError:
        return []
    if minor is None:
        return []
    out = []
    for hw in sorted(pathlib.Path(f"/sys/class/drm/renderD{minor}/device/hwmon").glob("hwmon*")):
        for inp in sorted(hw.glob("temp*_input")):
            lab = inp.with_name(inp.name.replace("_input", "_label"))
            try:
                label = lab.read_text().strip()
            except OSError:
                label = inp.name
            out.append({"label": label, "path": str(inp)})
    return out


def cmd_devmap(a) -> int:
    text = rocminfo_text()
    pathlib.Path(a.out).with_suffix(".rocminfo.txt").write_text(text)
    gpus = parse_rocminfo(text)
    print(f"rocminfo: {len(gpus)} GPU agent(s)")
    for g in gpus:
        print(f"  HSA GPU #{g['hsa_gpu_index']}: agent {g['agent']}, KFD node {g['node']}, {g['arch']}, "
              f"wave{g['wave']}, {g['marketing']!r}, uuid {g['uuid']}")
    hv = os.environ.get("HIP_VISIBLE_DEVICES")
    tokens = [t.strip() for t in hv.split(",")] if hv not in (None, "") else [str(i) for i in range(len(gpus))]
    print(f"HIP_VISIBLE_DEVICES={hv!r} -> tokens {tokens}")
    for var in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if os.environ.get(var):
            print(f"WARNING: {var}={os.environ[var]!r} is set; this prototype is meant to be pinned with "
                  f"HIP_VISIBLE_DEVICES only (ROCR also filters rocminfo, so the map below may be off)")
    hip = []
    for k, t in enumerate(tokens):
        g = None
        if t.isdigit() and int(t) < len(gpus):
            g = gpus[int(t)]
        elif t.upper().startswith("GPU-"):
            g = next((x for x in gpus if x["uuid"].lower() == t.lower()), None)
        if g is None:
            print(f"FAIL devmap: HIP_VISIBLE_DEVICES token {t!r} matches no rocminfo GPU agent")
            return 3
        hip.append(dict(g, hip_ordinal=k, visible_token=t, sensors=sensors_of(g["node"])))
    print("HIP ordinals as this run sees them:")
    for d in hip:
        sens = ", ".join(s["label"] for s in d["sensors"]) or "NONE"
        print(f"  HIP {d['hip_ordinal']}: {d['arch']} wave{d['wave']} {d['marketing']!r} (token {d['visible_token']}, "
              f"KFD node {d['node']}; sensors: {sens})")
    # HSA_OVERRIDE_GFX_VERSION: never the global one; a per-node one only if it turns out to be needed (it should not
    # be on ROCm 7.14 with a native gfx906 code object)
    overrides = {k: v for k, v in os.environ.items() if k.startswith("HSA_OVERRIDE_GFX_VERSION")}
    if "HSA_OVERRIDE_GFX_VERSION" in overrides:
        print(f"FAIL devmap: the global HSA_OVERRIDE_GFX_VERSION={overrides['HSA_OVERRIDE_GFX_VERSION']!r} is set; it "
              "would re-target every GPU (the XTX too).  Unset it; if the MI50 needs one, use the per-node "
              "HSA_OVERRIDE_GFX_VERSION_<node> form")
        return 3
    for k, v in overrides.items():
        print(f"note: per-node override {k}={v!r} is set (KFD nodes here: "
              f"{', '.join(str(d['node']) for d in hip)})")
    if not overrides:
        print("HSA_OVERRIDE_GFX_VERSION*: none set (as intended)")
    bad = []
    for ordinal, arch in WANT.items():
        if ordinal >= len(hip):
            bad.append(f"HIP device {ordinal} is missing (want {arch})")
        elif hip[ordinal]["arch"] != arch:
            bad.append(f"HIP device {ordinal} is {hip[ordinal]['arch']}, want {arch}")
    if len(hip) > 2:
        print(f"note: {len(hip)} GPUs visible; the gates use HIP 0 (XTX) and 1 (MI50) only")
    json.dump({"hip": hip, "rocm_gpus": gpus, "HIP_VISIBLE_DEVICES": hv, "overrides": overrides},
              open(a.out, "w"), indent=1)
    if bad:
        print("FAIL devmap: " + "; ".join(bad) + ".  Set HIP_VISIBLE_DEVICES=<xtx>,<mi50> (rocminfo GPU order above)")
        return 3
    print(f"devmap ok: HIP 0 = {hip[0]['arch']} (token {hip[0]['visible_token']}), "
          f"HIP 1 = {hip[1]['arch']} (token {hip[1]['visible_token']})")
    return 0


# ----------------------------------------------------------------------------------------------- thermal guard
class Thermal:
    def __init__(self, map_path: str | None):
        self.devs = []
        if map_path and os.path.exists(map_path):
            for d in json.load(open(map_path))["hip"][:2]:
                self.devs.append((f"HIP{d['hip_ordinal']}:{d['arch']}", d["sensors"]))
        # Per-card trips from our own records, not invented: XTX junction trip 102 C (pre-registered in the
        # local-llm-serving research record); MI50 throttles at about 93-95 C and shuts down at 100 C
        # (knowledge/applications/mi50.md), so abort at 95.  GATE_TEMP_ABORT_C overrides both.
        self.limits = {}
        for name, _ in self.devs:
            default = 95.0 if name.endswith("gfx906") else 102.0
            ab = float(os.environ.get("GATE_TEMP_ABORT_C", default))
            self.limits[name] = (ab, float(os.environ.get("GATE_TEMP_WARN_C", ab - 7)))
        self.abort_c = max((v[0] for v in self.limits.values()), default=102.0)
        self.peak: dict[str, float] = {}
        self.warned = False

    def read(self) -> dict[str, float]:
        """Per device the hottest of junction / edge / mem (degrees C); devices without a readable sensor are left out."""
        out = {}
        for name, sens in self.devs:
            vals = []
            for s in sens:
                try:
                    vals.append(int(pathlib.Path(s["path"]).read_text().strip()) / 1000.0)
                except (OSError, ValueError):
                    pass
            if vals:
                out[name] = max(vals)
        return out

    def check(self, log=None) -> str | None:
        """None, or the reason to abort."""
        t = self.read()
        for k, v in t.items():
            self.peak[k] = max(self.peak.get(k, v), v)
        hot = {k: v for k, v in t.items() if v >= self.limits.get(k, (102.0, 95.0))[0]}
        if hot:
            return "thermal abort: " + ", ".join(f"{k} {v:.0f} C (trip {self.limits.get(k, (102.0,))[0]:.0f})"
                                                 for k, v in hot.items())
        warm = {k: v for k, v in t.items() if v >= self.limits.get(k, (102.0, 95.0))[1]}
        if warm and not self.warned:
            self.warned = True
            msg = "thermal warning: " + ", ".join(f"{k} {v:.0f} C" for k, v in warm.items())
            print(msg, flush=True)
            if log:
                log.write(f"## {msg}\n")
        return None

    def peak_text(self) -> str:
        return ", ".join(f"{k} peak {v:.0f} C" for k, v in self.peak.items()) or "no temperature readings"


def kill_group(p: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def norm_rc(rc: int) -> int:
    return 128 - rc if rc < 0 else rc   # a signal -N -> 128 + N, as a shell reports it


def cmd_temps(a) -> int:
    th = Thermal(a.map)
    t = th.read()
    print("temperatures: " + (", ".join(f"{k} {v:.0f} C" for k, v in t.items()) or "no readable sensor"))
    return 0 if t else 1


def cmd_guard(a) -> int:
    cmd = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    th = Thermal(a.map)
    t0 = time.time()
    with open(a.log, "w") as log:
        log.write("## cmd: " + " ".join(cmd) + "\n")
        log.write("## HIP_VISIBLE_DEVICES=" + os.environ.get("HIP_VISIBLE_DEVICES", "<unset>") + "\n")
        log.flush()
        try:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            log.write(f"## cannot start: {e}\n")
            print(f"cannot start {cmd[0]}: {e}")
            return 127
        why, rc = None, None
        last_sz, last_t = -1, time.time()
        while True:
            try:
                rc = p.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            why = th.check(log)
            # a stall, not a guessed duration: --timeout is the window with no new output in the log
            sz = os.path.getsize(a.log)
            if sz != last_sz:
                last_sz, last_t = sz, time.time()
            if why is None and time.time() - last_t > a.timeout:
                why = f"stalled: no new output for {a.timeout:.0f} s"
            if why:
                kill_group(p)
                break
        dt = time.time() - t0
        log.write(f"\n## rc {norm_rc(p.returncode) if p.returncode is not None else 'killed'}; {dt:.1f} s; "
                  f"{th.peak_text()}{'; ' + why if why else ''}\n")
    if why:
        print(f"{why} ({cmd[0]})")
        return 125 if why.startswith("thermal") else 124
    return norm_rc(rc)


# ----------------------------------------------------------------------------------------------- serve driver
DONE_FIELDS = ["generated", "prompt", "prompt_ms", "decode_ms", "finish", "drafts_accepted", "drafts_offered",
               "reused", "hits", "lookups", "ram_blobs", "file_blobs", "file_mb", "read_n", "offloaded"]
RE_HELPER = re.compile(r"strata serve: CUDA(\d+): (\d+) expert entries, (\d+) active layer launches, ([\d.]+) MiB returned "
                       r"\(([\d.]+) MiB with full rows\) in this request; host (\d+) ms staging\+launching, (\d+) ms waiting")
RE_DECT = re.compile(r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window")
RE_DECT_LW = re.compile(r"per layer-window: CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)")
RE_DECT_HOST = re.compile(r"GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) jobs ([\d.]+) CPU ([\d.]+)\]")
RE_PF_HELPER = re.compile(r"prefill helper source: (\d+) blobs / ([\d.]+) MiB")
RE_START_HELPER = re.compile(r"strata generate: CUDA(\d+): (\d+) additional experts, ([\d.]+) GiB")


def parse_done(line: str) -> dict:
    f = line.split()
    d = {"raw": line}
    for i, k in enumerate(DONE_FIELDS):
        if 1 + i < len(f) and "=" not in f[1 + i]:
            v = f[1 + i]
            try:
                d[k] = v if k == "finish" else (float(v) if "." in v else int(v))
            except ValueError:
                d[k] = v
    for tok in f:
        if "=" in tok:
            k, _, v = tok.partition("=")
            d[k] = v
    return d


def parse_req_stderr(text: str) -> dict:
    out = {"helpers": [], "decode_timing": None, "prefill_helper": None, "lines": []}
    for line in text.splitlines():
        m = RE_HELPER.search(line)
        if m:
            out["helpers"].append({"cuda": int(m.group(1)), "entries": int(m.group(2)), "launches": int(m.group(3)),
                                   "returned_mib": float(m.group(4)), "full_mib": float(m.group(5)),
                                   "host_ms": int(m.group(6)), "wait_ms": int(m.group(7))})
            out["lines"].append(line)
        m = RE_DECT.search(line)
        if m:
            dt = {"windows": int(m.group(1)), "avg_T": float(m.group(2)), "tok_per_window": float(m.group(3)),
                  "ms_per_window": float(m.group(4))}
            m2 = RE_DECT_LW.search(line)
            if m2:
                dt.update(cpu_experts_lw=float(m2.group(1)), cpu_entries_lw=float(m2.group(2)),
                          vram_hits_lw=float(m2.group(3)), pcie_lw=float(m2.group(4)))
            m3 = RE_DECT_HOST.search(line)
            if m3:
                dt.update(gpu_reach_wait_ms=float(m3.group(1)), host_ms=float(m3.group(2)), cpu_pool_ms=float(m3.group(6)))
            out["decode_timing"] = dt
            out["lines"].append(line)
        m = RE_PF_HELPER.search(line)
        if m:
            out["prefill_helper"] = {"blobs": int(m.group(1)), "mib": float(m.group(2))}
            out["lines"].append(line)
        if "expert tiers:" in line:
            out["lines"].append(line)
    return out


def cmd_serve(a) -> int:
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    reqs = json.load(open(a.requests))
    pre = pathlib.Path(a.out)
    out_txt, err_path = open(str(pre) + ".stdout.txt", "w"), str(pre) + ".stderr.txt"
    errf = open(err_path, "wb")
    th = Thermal(a.map)
    res = {"argv": argv, "env": {k: v for k, v in os.environ.items() if k.startswith(("STRATA_", "HIP_", "HSA_"))},
           "requests": [], "status": "running"}

    def save():
        json.dump(res, open(str(pre) + ".requests.json", "w"), indent=1)

    print("engine: " + " ".join(argv), flush=True)
    t_start = time.time()
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1,
                         start_new_session=True)
    lines: queue.Queue = queue.Queue()

    def reader():
        for ln in p.stdout:
            lines.put(ln)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def wait_for(prefixes, timeout):
        """The first stdout line starting with one of prefixes; None at EOF; raises on timeout / heat."""
        t0 = time.time()
        while True:
            try:
                ln = lines.get(timeout=5)
            except queue.Empty:
                why = th.check()
                if why:
                    raise RuntimeError(why)
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"stalled: no output for {timeout:.0f} s while waiting for {'/'.join(prefixes)}")
                continue
            t0 = time.time()   # any output resets the stall window
            if ln is None:
                return None
            out_txt.write(ln)
            out_txt.flush()
            if ln.startswith(prefixes):
                return ln.strip()

    def finish(code: int, status: str) -> int:
        res["status"] = status
        res["peak_temps"] = th.peak
        res["wall_s"] = round(time.time() - t_start, 1)
        errf.flush()
        try:
            st = pathlib.Path(err_path).read_text(errors="replace")
            res["startup_helpers"] = [{"cuda": int(m.group(1)), "experts": int(m.group(2)), "gib": float(m.group(3))}
                                      for m in RE_START_HELPER.finditer(st)]
            res["startup_cache_lines"] = [l for l in st.splitlines() if "expert cache" in l or "experts in VRAM" in l][:6]
        except OSError:
            pass
        save()
        print(f"{status} ({res['wall_s']} s; {th.peak_text()})", flush=True)
        return code

    try:
        ready = wait_for(("READY",), a.ready_timeout)
        if ready is None:
            rc = p.wait()
            return finish(126, f"the engine ended before READY (rc {norm_rc(rc)}); see {err_path}")
        res["ready"] = ready
        res["ready_s"] = round(time.time() - t_start, 1)
        print(f"{ready} after {res['ready_s']} s", flush=True)
        for r in reqs:
            ids = r.get("ids")
            if ids is None:
                ids = [int(x) for x in open(r["ids_file"]).read().replace(",", " ").split()]
            n0 = os.path.getsize(err_path)
            keys = (" " + r["keys"]) if r.get("keys") else ""
            t0 = time.time()
            p.stdin.write(f"GEN {r['max_new']}{keys} " + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            done = wait_for(("DONE", "ERR"), a.req_timeout)
            wall = time.time() - t0
            time.sleep(1.0)   # the per-request stderr summary follows DONE
            errf.flush()
            with open(err_path, "rb") as f:
                f.seek(n0)
                chunk = f.read().decode(errors="replace")
            rec = {"name": r["name"], "max_new": r["max_new"], "prompt_tokens": len(ids), "wall_s": round(wall, 2)}
            if done is None:
                rec["error"] = "the engine ended during the request"
                res["requests"].append(rec)
                return finish(126, f"{r['name']}: the engine ended (rc {norm_rc(p.wait())}); see {err_path}")
            if done.startswith("ERR"):
                rec["error"] = done
                res["requests"].append(rec)
                return finish(1, f"{r['name']}: {done}")
            rec["done"] = parse_done(done)
            rec.update(parse_req_stderr(chunk))
            res["requests"].append(rec)
            save()
            d = rec["done"]
            hel = ", ".join(f"CUDA{h['cuda']} {h['entries']} entries" for h in rec["helpers"]) or "no helper line"
            print(f"  {r['name']}: {d.get('generated')} tokens, prompt {d.get('prompt')} ({d.get('read_n')} read) in "
                  f"{d.get('prompt_ms')} ms, decode {d.get('decode_ms')} ms; {hel}", flush=True)
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            kill_group(p)
        return finish(0, "ok")
    except RuntimeError as e:   # heat
        kill_group(p)
        return finish(125, str(e))
    except TimeoutError as e:
        kill_group(p)
        return finish(124, str(e))
    except BrokenPipeError:
        rc = p.wait()
        return finish(126, f"the engine closed its stdin (rc {norm_rc(rc)}); see {err_path}")


# ----------------------------------------------------------------------------------------------- tokenizer, pack
def import_tools(src: str):
    sys.path.insert(0, str(pathlib.Path(src) / "tools"))


def cmd_ids(a) -> int:
    import_tools(a.src)
    try:
        import strata_tokenizer as ST
    except ImportError as e:
        print(f"FAIL ids: cannot import tools/strata_tokenizer.py ({e}); it needs the python 'regex' module.  "
              "Either install it in the image or tokenize elsewhere and pass GATE_IDS_CODE / GATE_IDS_8K")
        return 1
    td = pathlib.Path(a.tokenizer_dir) if a.tokenizer_dir else None
    if td and (td / "vocab.json").exists():
        vocab = json.loads((td / "vocab.json").read_text(encoding="utf-8"))
        toks = [None] * len(vocab)
        for t, i in vocab.items():
            toks[i] = t
        tt = json.loads((td / "token_type.json").read_text()) if (td / "token_type.json").exists() else None
        tok = ST.Tokenizer(toks, (td / "merges.txt").read_text(encoding="utf-8").split("\n"), tt)
        src = str(td)
    else:
        tok = ST.Tokenizer.from_gguf(a.gguf)
        src = a.gguf
    ids = tok.encode(pathlib.Path(a.text).read_text(encoding="utf-8"))
    if len(ids) < a.n:
        print(f"FAIL ids: {a.text} is only {len(ids)} tokens (< {a.n})")
        return 1
    pathlib.Path(a.out).write_text(",".join(map(str, ids[:a.n])) + "\n")
    print(f"ids: {a.text}: {len(ids)} tokens ({src}), kept the first {a.n} -> {a.out}")
    return 0


def shard_paths(shard1: str) -> list[pathlib.Path]:
    p = pathlib.Path(shard1)
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not m:
        return [p]
    n = int(m.group(3))
    return [p.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def cmd_pack_check(a) -> int:
    import_tools(a.src)
    from gguf_reader import GGUFFile
    pack = pathlib.Path(a.pack)
    ne = pack / "native_experts.txt"
    if not ne.exists() or not (pack / "index.txt").exists():
        print(f"pack-check: {pack} has no native_experts.txt / index.txt (not a finished native pack)")
        return 1
    where = {}
    shards = shard_paths(a.gguf)
    for s in shards:
        if not s.exists():
            print(f"pack-check: shard {s} is missing")
            return 1
        g = GGUFFile(s)
        for t in g.tensors:
            where[t.name] = (t.type_id, g.data_start + t.offset, s.name)
    s1 = shards[0].name
    n_lay, bad = 0, []
    for line in ne.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        L, gu, dn = int(f[0]), int(f[1]), int(f[2])
        n_lay += 1
        names = [f"blk.{L}.ffn_{r}_exps.weight" for r in ("gate", "up", "down")]
        if any(n not in where for n in names):
            bad.append(f"layer {L}: tensors missing in the GGUF")
            continue
        if where[names[0]][0] != gu or where[names[2]][0] != dn:
            bad.append(f"layer {L}: pack types {gu}/{dn}, GGUF {where[names[0]][0]}/{where[names[2]][0]}")
            continue
        if len(f) >= 8:
            col = f[8] if len(f) > 8 else ""
            per_role = col.split(",") if "," in col else [col] * 3
            for k, (n, off) in enumerate(zip(names, (int(f[5]), int(f[6]), int(f[7])))):
                sh = per_role[k] or s1
                if where[n][2] != sh or where[n][1] != off:
                    bad.append(f"layer {L} {n.split('.')[2]}: pack {sh}@{off}, GGUF {where[n][2]}@{where[n][1]}")
    gl = 1 + max((int(n.split(".")[1]) for n in where if n.startswith("blk.") and n.endswith("_exps.weight")), default=-1)
    if n_lay != gl:
        bad.append(f"{n_lay} layers in the pack, {gl} in the GGUF")
    if bad:
        print(f"pack-check: {pack} does NOT match {a.gguf}: " + "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
        return 1
    print(f"pack-check: {pack} matches {a.gguf} ({n_lay} layers: types and absolute offsets)")
    return 0


# ----------------------------------------------------------------------------------------------- gate parsers
def hipmap(path):
    return json.load(open(path))["hip"]


def cmd_g0(a) -> int:
    text = open(a.log, errors="replace").read()
    body = "\n".join(l for l in text.splitlines() if not l.startswith("## "))
    rc = a.rc
    mi50 = hipmap(a.map)[1]
    problems, notes = [], []
    if rc >= 128 or rc in (124, 125):
        problems.append(f"the probe did not exit normally (rc {rc})")
    for pat in ("hipErrorNoBinaryForGpu", "Aborted", "core dumped", "terminate called"):
        if pat in body:
            problems.append(f"'{pat}' in its output")
    if not re.search(r"(?i)probe done", body):
        problems.append("no 'probe done' line (the process did not reach its end)")
    # per-device context: a line naming an ordinal or an arch sets the device of the TU lines that follow
    tua_re = re.compile(os.environ.get("GATE_G0_TUA_RE", r"(?i)\bTU[-_ ]?A\b"))
    tub_re = re.compile(os.environ.get("GATE_G0_TUB_RE", r"(?i)\bTU[-_ ]?B\b"))
    dev_re = re.compile(r"(?i)\b(?:dev(?:ice)?|hip|ordinal|gpu)\s*[#:=]?\s*(\d+)\b")
    warp_re = re.compile(r"(?i)warp\s*_?size\s*[:=]?\s*(\d+)|wave(?:front)?[ _]?size\s*[:=]?\s*(\d+)|\bwave(32|64)\b")
    err_re = re.compile(r"(?i)hipError\w*|\berror\b|\bfail(?:ed|s)?\b|invalid|no binary|not supported|no kernel image")
    cur = None
    found = {"A": {}, "B": {}}
    for line in body.splitlines():
        dm = dev_re.search(line)
        dev = int(dm.group(1)) if dm else ("gfx906" if "gfx906" in line else "gfx1100" if "gfx1100" in line else None)
        if dev == "gfx906":
            dev = 1
        elif dev == "gfx1100":
            dev = 0
        is_a, is_b = bool(tua_re.search(line)), bool(tub_re.search(line))
        if not (is_a or is_b):
            if dev is not None:
                cur = dev
            continue
        d = dev if dev is not None else cur
        if d is None:
            continue
        wm = warp_re.search(line)
        warp = next((int(x) for x in wm.groups() if x), None) if wm else None
        found["A" if is_a else "B"].setdefault(d, []).append({"line": line.strip(), "warp": warp,
                                                               "error": bool(err_re.search(line))})
    for tu in ("A", "B"):
        for d, recs in sorted(found[tu].items()):
            for r in recs:
                print(f"  TU-{tu} on HIP {d}: warp {r['warp']}, error {r['error']}: {r['line'][:160]}")
    a_mi50 = found["A"].get(1, [])
    warps = [r["warp"] for r in a_mi50 if r["warp"] is not None]
    if not a_mi50:
        problems.append("no TU-A result line for the MI50 (HIP 1) could be found (GATE_G0_TUA_RE adjusts the pattern)")
    elif any(r["error"] for r in a_mi50) or 64 not in warps:
        problems.append(f"TU-A on the MI50 did not return warpSize 64 (parsed {warps or 'nothing'})")
    b_mi50 = found["B"].get(1, [])
    if not b_mi50:
        tub = "TU-B on the MI50: not reported"
    elif all(r["error"] for r in b_mi50):
        tub = "TU-B on the MI50: failed cleanly (expected), the process went on"
    else:
        tub = "TU-B on the MI50: did NOT report an error (unexpected: gfx1100-only code on gfx906)"
    notes.append(tub)
    a_xtx = [r["warp"] for r in found["A"].get(0, [])]
    notes.append(f"TU-A on the XTX: warp {a_xtx or 'not reported'}")
    if problems:
        print(f"FAIL gate0: " + "; ".join(problems) + f" [{'; '.join(notes)}] (MI50 = {mi50['marketing']!r})")
        return 1
    print(f"PASS gate0: probe done, rc {rc}, TU-A on the MI50 warpSize 64; " + "; ".join(notes))
    return 0


def cmd_g1(a) -> int:
    text = open(a.log, errors="replace").read()
    blocks, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^device (\d+): (.*)$", line)
        if m:
            cur = int(m.group(1))
            blocks[cur] = {"name": m.group(2), "lines": []}
        elif cur is not None and line.startswith("  "):
            blocks[cur]["lines"].append(line.strip())
    for d, b in sorted(blocks.items()):
        print(f"  device {d}: {b['name']} | " + " | ".join(b["lines"]))
    probs = []

    def arch_of(b):
        for l in b["lines"]:
            m = re.match(r"arch (\S+), ([\d.]+) GiB, wave(\d+)", l)
            if m:
                return m.group(1), int(m.group(3))
        return None, None

    if a.rc != 0:
        probs.append(f"strata-device exited {a.rc}")
    if 0 not in blocks or 1 not in blocks:
        probs.append("fewer than two devices listed")
    else:
        a0, w0 = arch_of(blocks[0])
        a1, w1 = arch_of(blocks[1])
        if (a0, w0) != ("gfx1100", 32):
            probs.append(f"device 0 is {a0} wave{w0}, want gfx1100 wave32")
        if any(l.startswith(("cannot run", "helper-only")) for l in blocks[0]["lines"]):
            probs.append("device 0 is not runnable as the primary")
        if (a1, w1) != ("gfx906", 64):
            probs.append(f"device 1 is {a1} wave{w1}, want gfx906 wave64")
        if not any(l.startswith("helper-only (gfx906, wave64)") for l in blocks[1]["lines"]):
            probs.append("device 1 has no 'helper-only (gfx906, wave64)' line"
                         + (" (it says: " + next(l for l in blocks[1]["lines"] if l.startswith("cannot run")) + ")"
                            if any(l.startswith("cannot run") for l in blocks[1]["lines"]) else ""))
    if probs:
        print("FAIL gate1: " + "; ".join(probs))
        return 1
    print("PASS gate1: device 0 gfx1100 wave32 runnable; device 1 gfx906 wave64 listed as helper-only")
    return 0


def cmd_g3(a) -> int:
    text = open(a.log, errors="replace").read()
    stext = open(a.strict_log, errors="replace").read() if a.strict_log else ""
    probs, notes = [], []
    if a.rc == 77:
        probs.append("rc 77: fewer than two devices visible (skipped is not a pass)")
    elif a.rc != 0:
        probs.append(f"rc {a.rc}: a device missed the CPU float reference (rel L1 < 3e-2) or did not finish")
    m = re.search(r"device A: ordinal 0 .*\((\S+), warpSize (\d+)\)", text)
    n = re.search(r"device B: ordinal 1 .*\((\S+), warpSize (\d+)\)", text)
    if not (m and n) or (m.group(1).split(":")[0], m.group(2)) != ("gfx1100", "32") or \
            (n.group(1).split(":")[0], n.group(2)) != ("gfx906", "64"):
        probs.append("device A/B are not gfx1100 wave32 / gfx906 wave64 " +
                     f"({m.group(0) if m else '?'} / {n.group(0) if n else '?'})")
    # per pair: the "A mode 0 vs B mode 0" block - the design's no-go is (1), (2) or (4) not bitwise there
    pairs = re.split(r"\n(?=\S+/\S+: \d+ experts x)", text)
    n_pairs = 0
    for chunk in pairs:
        hm = re.match(r"(\S+/\S+): \d+ experts x", chunk.strip())
        if not hm:
            continue
        n_pairs += 1
        blk = chunk.split("A mode 0 vs B mode 0", 1)
        if len(blk) < 2:
            probs.append(f"{hm.group(1)}: no 'A mode 0 vs B mode 0' block")
            continue
        body = blk[1].split("\n  A ", 1)[0]
        def get(k: str) -> str:   # the indented result line, not the block title "(1), (2), (4) expected bitwise"
            mm = re.search(rf"^\s+\({k}\)[^\n]*", body, re.M)
            return mm.group(0) if mm else ""

        l1, l2, l3, l4, l5 = (get(k) for k in "12345")
        for k, l in (("1", l1), ("2", l2), ("4", l4)):
            if "bitwise equal" not in l:
                probs.append(f"{hm.group(1)} mode 0: ({k}) not bitwise ({l.strip()[:120]}) - the design's no-go "
                             "unless traced to a shim primitive")
        r5 = re.search(r"rel L1 ([\d.e+-]+)", l5)
        notes.append(f"{hm.group(1)}: mode0 (3) {('ok' if '(within' in l3 or 'bitwise equal' in l3 else 'BEYOND +-1/0.1%')}"
                     f", (5) rel {r5.group(1) if r5 else '?'}")
    if n_pairs == 0:
        probs.append("no format pair was reported")
    cpu = re.findall(r"^\s+(A mode 0|B mode 0|A default|B default)\s+gpu rel ([\d.e+-]+)\s+(ok|FAIL)", text, re.M)
    if cpu:
        notes.append("vs CPU: " + ", ".join(f"{w} {e} {s}" for w, e, s in cpu))
    s_note = f"--strict rc {a.strict_rc}"
    sm = re.search(r"(\d+) CPU-reference failures, (\d+) same-mode cross-device differences", stext)
    if sm:
        s_note += f" ({sm.group(1)} CPU-reference failures, {sm.group(2)} same-mode differences)"
    if a.strict_rc not in (0, 1):
        probs.append(f"--strict run rc {a.strict_rc}")
    notes.append(s_note + (" - only (3)/(5) beyond expectation: WARN" if a.strict_rc == 1 and not probs else ""))
    if probs:
        print("FAIL gate3: " + "; ".join(probs) + " | " + "; ".join(notes))
        return 1
    print("PASS gate3: CPU reference ok on both devices, mode 0 (1)(2)(4) bitwise for every pair | " + "; ".join(notes))
    return 0


def load_reqs(prefix: str) -> dict:
    p = pathlib.Path(prefix + ".requests.json")
    return json.load(open(p)) if p.exists() else {"status": "missing", "requests": []}


def cmd_g4(a) -> int:
    probs, notes = [], []
    for arm in a.arms:
        r = load_reqs(f"{a.dir}/gate4-{arm}")
        if r.get("status") != "ok":
            probs.append(f"arm {arm}: {r.get('status')}")
            continue
        sh = r.get("startup_helpers", [])
        ent = sum(h["entries"] for q in r["requests"] for h in q.get("helpers", []) if h["cuda"] == 1)
        if arm == "a":
            if sh or ent:
                probs.append(f"arm a (XTX only) has helper lines ({sh}, {ent} entries): not the baseline")
            notes.append("a: no helper")
            continue
        if not sh or sh[0]["experts"] <= 0:
            probs.append(f"arm {arm}: no 'CUDA1: N additional experts' start-up line (helper cache not filled)")
        if ent <= 0:
            probs.append(f"arm {arm}: the helper computed 0 entries ('strata serve: CUDA1: N expert entries')")
        notes.append(f"{arm}: helper {sh[0]['experts'] if sh else 0} experts / {sh[0]['gib'] if sh else 0} GiB, "
                     f"{ent} entries computed")
    if probs:
        print("FAIL gate4 (helper entries): " + "; ".join(probs) + " | " + "; ".join(notes))
        return 1
    print("helper entries ok: " + "; ".join(notes))
    return 0


def cmd_g5(a) -> int:
    def arm_stats(prefix):
        r = load_reqs(prefix)
        rows = []
        for q in r["requests"]:
            if not q["name"].startswith("rep"):
                continue
            d = q.get("done", {})
            gen, dms, pms, rn = d.get("generated"), d.get("decode_ms"), d.get("prompt_ms"), d.get("read_n")
            if not all(isinstance(x, (int, float)) for x in (gen, dms, pms, rn)) or dms <= 0 or pms <= 0 or gen <= 1:
                continue
            hel = sum(h["entries"] for h in q.get("helpers", []))
            dt = q.get("decode_timing") or {}
            rows.append({"name": q["name"], "decode_tps": 1000.0 * gen / dms, "prefill_tps": 1000.0 * rn / pms,
                         "generated": gen, "read_n": rn, "acc": d.get("drafts_accepted"), "off": d.get("drafts_offered"),
                         "helper_entries": hel, "helper_per_tok": hel / gen,
                         "helper_wait_ms": sum(h["wait_ms"] for h in q.get("helpers", [])),
                         "cpu_entries_lw": dt.get("cpu_entries_lw"), "cpu_pool_ms_w": dt.get("cpu_pool_ms"),
                         "ms_window": dt.get("ms_per_window"), "reach_wait_ms_w": dt.get("gpu_reach_wait_ms"),
                         "pf_helper_blobs": (q.get("prefill_helper") or {}).get("blobs")})
        return r, rows

    out, ok = {}, True
    for arm, pre in (("a", a.a), ("b", a.b)):
        r, rows = arm_stats(pre)
        out[arm] = rows
        print(f"arm {arm} ({'XTX only' if arm == 'a' else 'XTX + MI50 helper'}): status {r.get('status')}, "
              f"ready after {r.get('ready_s')} s")
        for x in rows:
            fmt = lambda v, f: (f % v) if isinstance(v, (int, float)) else "-"
            print(f"  {x['name']}: decode {x['decode_tps']:.2f} tok/s ({x['generated']} tok, drafts {x['acc']}/{x['off']}), "
                  f"prefill {x['prefill_tps']:.0f} tok/s ({x['read_n']} tok); per window {fmt(x['ms_window'], '%.2f')} ms, "
                  f"CPU pool {fmt(x['cpu_pool_ms_w'], '%.2f')} ms, GPU-reach wait {fmt(x['reach_wait_ms_w'], '%.2f')} ms; "
                  f"CPU entries/layer-window {fmt(x['cpu_entries_lw'], '%.2f')}; helper {x['helper_entries']} entries "
                  f"({x['helper_per_tok']:.1f}/token, waited {x['helper_wait_ms']} ms); "
                  f"prefill helper blobs {fmt(x['pf_helper_blobs'], '%d')}")
        if len(rows) < a.repeats:
            print(f"FAIL gate5: arm {arm} has {len(rows)} usable timed requests (< {a.repeats})")
            ok = False
    if not ok:
        return 1
    med = {k: statistics.median([x["decode_tps"] for x in v]) for k, v in out.items()}
    pmed = {k: statistics.median([x["prefill_tps"] for x in v]) for k, v in out.items()}
    hpt = statistics.median([x["helper_per_tok"] for x in out["b"]])
    json.dump({"rows": out, "median_decode": med, "median_prefill": pmed}, open(a.json, "w"), indent=1)
    line = (f"decode median a {med['a']:.2f} / b {med['b']:.2f} tok/s ({100 * (med['b'] / med['a'] - 1):+.1f}%), "
            f"prefill median a {pmed['a']:.0f} / b {pmed['b']:.0f} tok/s, helper {hpt:.1f} entries/token")
    if hpt <= 0:
        print("FAIL gate5: the helper computed no entries in arm b | " + line)
        return 1
    if med["b"] < med["a"]:
        print("FAIL gate5 (ABORT criterion): decode with the helper is slower than without | " + line)
        return 1
    print("PASS gate5: " + line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("devmap"); p.add_argument("--out", required=True)
    p = sub.add_parser("temps"); p.add_argument("--map")
    p = sub.add_parser("guard"); p.add_argument("--map"); p.add_argument("--log", required=True)
    p.add_argument("--timeout", type=float, default=3600); p.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("serve"); p.add_argument("--map"); p.add_argument("--out", required=True)
    p.add_argument("--requests", required=True); p.add_argument("--ready-timeout", type=float, default=2400)
    p.add_argument("--req-timeout", type=float, default=2400); p.add_argument("argv", nargs=argparse.REMAINDER)
    p = sub.add_parser("ids"); p.add_argument("--src", required=True); p.add_argument("--gguf", required=True)
    p.add_argument("--tokenizer-dir"); p.add_argument("--text", required=True); p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("pack-check"); p.add_argument("--src", required=True); p.add_argument("--pack", required=True)
    p.add_argument("--gguf", required=True)
    p = sub.add_parser("g0"); p.add_argument("--log", required=True); p.add_argument("--rc", type=int, required=True)
    p.add_argument("--map", required=True)
    p = sub.add_parser("g1"); p.add_argument("--log", required=True); p.add_argument("--rc", type=int, required=True)
    p = sub.add_parser("g3"); p.add_argument("--log", required=True); p.add_argument("--rc", type=int, required=True)
    p.add_argument("--strict-log"); p.add_argument("--strict-rc", type=int, default=0)
    p = sub.add_parser("g4"); p.add_argument("--dir", required=True); p.add_argument("--arms", nargs="+", required=True)
    p = sub.add_parser("g5"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--json", required=True)
    a = ap.parse_args()
    return {"devmap": cmd_devmap, "temps": cmd_temps, "guard": cmd_guard, "serve": cmd_serve, "ids": cmd_ids,
            "pack-check": cmd_pack_check, "g0": cmd_g0, "g1": cmd_g1, "g3": cmd_g3, "g4": cmd_g4,
            "g5": cmd_g5}[a.sub](a)


if __name__ == "__main__":
    raise SystemExit(main())
