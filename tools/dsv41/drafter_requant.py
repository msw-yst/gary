#!/usr/bin/env python3
"""tools/dsv41/drafter_requant.py - a drafter-variant pack: the DSpark drafter's routed experts (mtp.*.ffn.experts.*, MXFP4
in pack-2.25) re-quantised to a smaller ggml type through ggml's own quantizer (libggml-base, ctypes), everything else of
the source pack unchanged.  Option 2 of the smaller-drafter design (card DR-05).

    PYTHONPATH=<numpy> python3 -I drafter_requant.py --pack $MODELS/deepseek-v4.1-flash-1017/pack-2.25 \\
        --model-dir $MODELS/deepseek-v4.1-flash-1017/ist-das-2.25bpw --type IQ3_XXS \\
        --out $MODELS/deepseek-v4.1-flash-1017/pack-2.25-dIQ3_XXS --lib .../libggml-base.so --jobs 6

- The source pack is only read.  The output directory gets ds4.txt / engram.bin / expert-profile.bin / tokenizer/ copied,
  one data file dspark-experts-<TYPE>.bin (each block's w1, w2, w3 runs of 128 experts, contiguous, 4096-aligned runs), and
  tensors.txt with the drafter expert lines pointing there (a new `file N`, relative to model_dir as the others are).
- There is no higher-precision source: the checkpoint's drafter experts are FP4 E2M1 with an E8M0 scale a 32
  (drafter_pack.py: copied exactly into MXFP4), so this re-encodes the FP4 values themselves.
- --type for all three roles, or --gate-up / --down separately (gate and up must match: the engine's rule).
- --cal DIR: DS4_DRAFT_CAL's L<b>_gu.f64 [128][5120] and L<b>_cnt.u64 [128] -> each expert's gate / up importance (shrunk
  toward the block mean, prior 64 picks, as make_imatrix.py); down has none (ones).  IQ2_XXS / IQ2_XS / IQ1_* need one in
  ggml: without --cal they get ones.
- Reads with O_DIRECT (the running engines' page cache is left alone), workers at SCHED_IDLE / nice 19.
"""
from __future__ import annotations

import argparse
import ctypes
import mmap
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import sys
import time

import numpy as np

TYPES = {"IQ2_XXS": (16, 256, 66), "IQ2_XS": (17, 256, 74), "IQ3_XXS": (18, 256, 98), "IQ3_S": (21, 256, 110),
         "IQ2_S": (22, 256, 82), "IQ4_XS": (23, 256, 136), "Q2_K": (10, 256, 84), "Q3_K": (11, 256, 110),
         "Q4_K": (12, 256, 144)}
NEED_IMATRIX = {16, 17, 19, 29}
ALIGN_IO = 4096
E2M1 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
LINE = re.compile(r"^t (mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight) (\S+) (\d+) (\d+) (\d+) (\d+) (\d+)\s*$")

_lib = None
_fds: dict = {}


def _init(lib_path: str):
    global _lib
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        pass
    try:
        os.nice(19)
    except OSError:
        pass
    _lib = ctypes.CDLL(lib_path)
    _lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_void_p]
    _lib.ggml_quantize_init.argtypes = [ctypes.c_int]


def read_src(path: str, off: int, n: int) -> memoryview:
    a0 = off // ALIGN_IO * ALIGN_IO
    a1 = (off + n + ALIGN_IO - 1) // ALIGN_IO * ALIGN_IO
    if path not in _fds:
        try:
            _fds[path] = (os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0)), True)
        except OSError:
            _fds[path] = (os.open(path, os.O_RDONLY), False)
    fd, direct = _fds[path]
    buf = mmap.mmap(-1, a1 - a0)
    got = os.preadv(fd, [buf], a0)
    if got < off - a0 + n:
        raise IOError("%s: short read at %d" % (path, off))
    if not direct:
        os.posix_fadvise(fd, a0, a1 - a0, os.POSIX_FADV_DONTNEED)
    return memoryview(buf)[off - a0:off - a0 + n]


def mxfp4_to_f32(raw: memoryview) -> np.ndarray:
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 17)
    d = np.ldexp(np.float32(1.0), blk[:, 0].astype(np.int32) - 128).astype(np.float32)
    q = blk[:, 1:]
    v = np.empty((blk.shape[0], 32), dtype=np.float32)
    v[:, :16] = E2M1[q & 15]
    v[:, 16:] = E2M1[q >> 4]
    return (v * d[:, None]).reshape(-1)


def _job(j: dict):
    t0 = time.time()
    raw = read_src(j["src"], j["src_off"], j["src_bytes"])
    x = np.ascontiguousarray(mxfp4_to_f32(raw), dtype=np.float32)
    del raw
    rows, k, q = j["rows"], j["k"], j["q"]
    qrb = k // TYPES[j["qname"]][1] * TYPES[j["qname"]][2]
    dst = np.empty(rows * qrb, dtype=np.uint8)
    im = None
    if j["imat"] is not None:
        im = np.ascontiguousarray(np.frombuffer(j["imat"], dtype=np.float32))
    elif q in NEED_IMATRIX:
        im = np.ones(k, dtype=np.float32)
    _lib.ggml_quantize_init(q)
    got = _lib.ggml_quantize_chunk(q, x.ctypes.data, dst.ctypes.data, 0, rows, k, im.ctypes.data if im is not None else None)
    if got != rows * qrb:
        raise RuntimeError("%s: ggml_quantize_chunk wrote %d bytes, expected %d" % (j["name"], got, rows * qrb))
    # the re-encode's error against the FP4 values (relative RMS), a sanity number for the report
    fd = os.open(j["out"], os.O_WRONLY)
    try:
        os.pwrite(fd, memoryview(dst), j["out_off"])
        os.fdatasync(fd)
        os.posix_fadvise(fd, j["out_off"], len(dst), os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return j["name"], len(dst), time.time() - t0


def load_cal(cal: str, n_blocks: int, n_exp: int, dim: int):
    """{(block, expert): float32 [dim] gate/up importance} from DS4_DRAFT_CAL's sums."""
    out = {}
    for b in range(n_blocks):
        gu = np.fromfile("%s/L%d_gu.f64" % (cal, b), dtype=np.float64).reshape(n_exp, dim)
        cnt = np.fromfile("%s/L%d_cnt.u64" % (cal, b), dtype=np.uint64).astype(np.float64)
        tot = max(cnt.sum(), 1.0)
        mean = gu.sum(0) / tot
        h = (gu + 64.0 * mean) / (cnt[:, None] + 64.0)
        h = np.maximum(h, 1e-8).astype(np.float32)
        for e in range(n_exp):
            out[(b, e)] = h[e].tobytes()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="the source pack directory (read only)")
    ap.add_argument("--model-dir", required=True, help="the host path of the pack's model_dir (ds4.txt's may be a container path)")
    ap.add_argument("--out", required=True, help="the new pack directory (must not exist)")
    ap.add_argument("--type", default="IQ3_XXS", help="all three roles' type")
    ap.add_argument("--gate-up", help="gate and up's type (default --type)")
    ap.add_argument("--down", help="down's type (default --type)")
    ap.add_argument("--lib", required=True, help="libggml-base.so")
    ap.add_argument("--cal", help="DS4_DRAFT_CAL directory (gate / up importance)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=40.0, help="refuse an output larger than this")
    a = ap.parse_args()
    pack, out = pathlib.Path(a.pack), pathlib.Path(a.out)
    gu_t, dn_t = a.gate_up or a.type, a.down or a.type
    for t in (gu_t, dn_t):
        if t not in TYPES:
            raise SystemExit("type %s: one of %s" % (t, ", ".join(TYPES)))
    if out.exists():
        raise SystemExit("%s exists: give a new directory" % out)
    if out.resolve() == pack.resolve():
        raise SystemExit("--out is the source pack")
    lines = (pack / "tensors.txt").read_text().splitlines()
    files: dict[int, str] = {}
    ents = []
    for i, ln in enumerate(lines):
        if ln.startswith("file "):
            _, n, name = ln.split(" ", 2)
            files[int(n)] = name
            continue
        m = LINE.match(ln)
        if m:
            name, b, e, role, dt, fi, off, nb, rows, k = m.groups()
            if dt != "MXFP4":
                raise SystemExit("%s: %s, not MXFP4" % (name, dt))
            ents.append(dict(i=i, name=name, b=int(b), e=int(e), role=role, file=int(fi), off=int(off), bytes=int(nb),
                             rows=int(rows), k=int(k)))
    if not ents:
        raise SystemExit("no drafter expert lines in %s/tensors.txt" % pack)
    nb_ = max(x["b"] for x in ents) + 1
    ne_ = max(x["e"] for x in ents) + 1
    new_file = max(files) + 1
    tname = "dspark-experts-%s.bin" % (gu_t if gu_t == dn_t else gu_t + "-" + dn_t)
    rel = os.path.relpath(out / tname, a.model_dir)
    # the layout: block b, role w1 / w2 / w3: experts 0 .. n-1 contiguous (one stride a role), each run 4096-aligned
    off = 0
    order = sorted(ents, key=lambda x: (x["b"], x["role"], x["e"]))
    for x in order:
        qn = gu_t if x["role"] in ("w1", "w3") else dn_t
        x["qname"] = qn
        x["q"] = TYPES[qn][0]
        x["qbytes"] = x["rows"] * (x["k"] // TYPES[qn][1] * TYPES[qn][2])
        if x["e"] == 0:
            off = (off + ALIGN_IO - 1) // ALIGN_IO * ALIGN_IO
        x["out_off"] = off
        off += x["qbytes"]
    total = off
    src_total = sum(x["bytes"] for x in ents)
    if total > a.max_gb * 1e9:
        raise SystemExit("output %.2f GB > --max-gb %.1f" % (total / 1e9, a.max_gb))
    st = os.statvfs(out.parent)
    if st.f_bavail * st.f_frsize < total + 2e9:
        raise SystemExit("%s: %.1f GB free, need %.1f" % (out.parent, st.f_bavail * st.f_frsize / 1e9, total / 1e9 + 2))
    print("drafter experts: %d (%d blocks x %d) x 3 roles, MXFP4 %.2f GB -> %s / %s %.2f GB (-%.2f GB); file %d %s" %
          (len(ents) // 3, nb_, ne_, src_total / 1e9, gu_t, dn_t, total / 1e9, (src_total - total) / 1e9, new_file, rel),
          flush=True)
    cal = load_cal(a.cal, nb_, ne_, ents[0]["k"] if ents[0]["role"] != "w2" else ents[0]["rows"]) if a.cal else None
    tmp = pathlib.Path(str(out) + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / tname, "wb") as f:
        f.truncate(total)
    jobs = []
    for x in order:
        im = cal.get((x["b"], x["e"])) if cal is not None and x["role"] in ("w1", "w3") else None
        jobs.append(dict(name=x["name"], src=os.path.join(a.model_dir, files[x["file"]]), src_off=x["off"],
                         src_bytes=x["bytes"], rows=x["rows"], k=x["k"], q=x["q"], qname=x["qname"], imat=im,
                         out=str(tmp / tname), out_off=x["out_off"]))
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(a.jobs, initializer=_init, initargs=(a.lib,)) as pool:
        for name, n, dt in pool.imap_unordered(_job, jobs):
            done += 1
            if done % 48 == 0 or done == len(jobs):
                print("  %d / %d tensors, %.0f s" % (done, len(jobs), time.time() - t0), flush=True)
    # the pack's index: the drafter's expert lines to the new file, a file line for it; the rest as it was
    by_line = {x["i"]: x for x in ents}
    new = []
    for i, ln in enumerate(lines):
        if i in by_line:
            x = by_line[i]
            new.append("t %s %s %d %d %d %d %d" % (x["name"], x["qname"], new_file, x["out_off"], x["qbytes"], x["rows"], x["k"]))
        else:
            new.append(ln)
            if ln.startswith("file %d " % max(files)):
                new.append("file %d %s" % (new_file, rel))
    for fn in ("ds4.txt", "engram.bin", "expert-profile.bin"):
        if (pack / fn).exists():
            shutil.copy2(pack / fn, tmp / fn)
    if (pack / "tokenizer").is_dir():
        shutil.copytree(pack / "tokenizer", tmp / "tokenizer")
    (tmp / "tensors.txt").write_text("\n".join(new) + "\n")
    (tmp / "VARIANT.txt").write_text("drafter_requant.py: source %s, drafter experts %s (gate/up) / %s (down), cal %s, %s\n" %
                                     (pack, gu_t, dn_t, a.cal or "none", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    os.rename(tmp, out)
    print("wrote %s in %.0f s" % (out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
