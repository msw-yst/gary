#!/usr/bin/env python3
"""tools/dsv41/gates/compare_logits.py - teacher-forced comparison of engine runs against a reference run (gate gate4).

    python3 compare_logits.py --ref gate4-a.tsv gate4-b.tsv gate4-c.tsv gate4-d.tsv [--json out.json]
        [--min-top1 95] [--max-median-kl 1e-3] [--ppl-se 2] [--ppl-rel 0.01] [--min-positions 400]

Per run against the reference, over the positions both scored with the same target token:
  - same top-1: the share of positions whose argmax agrees;
  - KL(ref || run): median, p99 and mean;
  - perplexity of the actual text under each (exp of the mean negative log-probability of the next token), and the
    paired difference of the per-token NLL (run - ref) with its standard error.
Pass (exit 0) when every run has >= --min-positions positions, top-1 >= --min-top1 %, median KL <= --max-median-kl,
and |NLL difference| <= --ppl-se standard errors OR the perplexity ratio is within --ppl-rel.  Missing positions
are never a pass.

Two input formats, told apart by content:
  * STRATA_LOGPOS rows (text; what gates.sh writes): pos, target id, logp(target), argmax id, logp(argmax),
    argmax==target, logp(extra), logp(target without extra), then STRATA_LOGPOS_TOPK `id:logp` fields (natural log;
    src/core/verify.cpp Verifier::window_logprobs).  KL is taken over the reference's top-K ids, both distributions
    renormalized to that set; an id the run does not list gets the run's K-th log-probability (the method of
    tools/quant_quality.py).  With K = 256 the mass outside the set is negligible for this purpose.
  * a `strata --dump-logits` file (binary): int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32 raw logits,
    one row per position (src/program/generate.cpp, the header is written with the first row).  Needs --ids (the
    token ids the run read: row r predicts ids[r + 1]) and numpy; KL is then over the whole vocabulary.
    NOTE: a native (IQ) pack never reaches the per-token loop that writes this file, so with the IQ3_S GGUF the
    engine writes nothing there; the gates use STRATA_LOGPOS instead.  The reader is kept for a canonical pack.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys


def is_binary(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(4096)
    if len(head) < 8:
        return False
    try:
        head.decode("ascii")
        return False
    except UnicodeDecodeError:
        return True


# ------------------------------------------------------------------------------------------ STRATA_LOGPOS rows
def read_logpos(path: str) -> dict:
    rows, dup = {}, 0
    for line in open(path, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        try:
            pos, tgt, lt, top, ltop = int(f[0]), int(f[1]), float(f[2]), int(f[3]), float(f[4])
            topk = [(int(x), float(y)) for x, y in (z.split(":") for z in f[8:] if ":" in z)]
        except ValueError:
            continue
        dup += pos in rows
        rows[pos] = (tgt, lt, top, ltop, topk)
    if dup:
        print(f"warning: {path}: {dup} positions appear more than once (the last one is used)", file=sys.stderr)
    return rows


def kl_topk(ref_topk, run_topk) -> float:
    if not ref_topk or not run_topk:
        return float("nan")
    d = dict(run_topk)
    floor = min(v for _, v in run_topk)
    pr = [math.exp(v) for _, v in ref_topk]
    pa = [math.exp(d.get(i, floor)) for i, _ in ref_topk]
    sr, sa = sum(pr), sum(pa)
    return sum((x / sr) * math.log((x / sr) / (y / sa)) for x, y in zip(pr, pa) if x > 0)


def compare_logpos(ref: dict, run: dict) -> dict:
    pos = sorted(set(ref) & set(run))
    same, kls, dn, nr, na, mismatch = 0, [], [], 0.0, 0.0, 0
    for q in pos:
        tr, lr, topr, _, kr = ref[q]
        ta, la, topa, _, ka = run[q]
        if tr != ta:
            mismatch += 1
            continue
        same += topr == topa
        kls.append(kl_topk(kr, ka))
        nr -= lr
        na -= la
        dn.append(lr - la)   # NLL(run) - NLL(ref)
    return summarize(len(dn), same, kls, nr, na, dn, mismatch, len(ref), len(run),
                     "top-%d renormalized" % (len(next(iter(ref.values()))[4]) if ref else 0))


# ------------------------------------------------------------------------------------------ --dump-logits files
def read_dump(path: str):
    import numpy as np
    with open(path, "rb") as f:
        n_vocab, n_rows = struct.unpack("<ii", f.read(8))
    a = np.memmap(path, dtype=np.float32, mode="r", offset=8)
    rows = a.size // n_vocab
    if rows != n_rows:
        print(f"warning: {path}: header says {n_rows} rows, the file holds {rows}", file=sys.stderr)
    return a[: rows * n_vocab].reshape(rows, n_vocab)


def compare_dump(ref_path: str, run_path: str, ids: list[int]) -> dict:
    import numpy as np
    A, B = read_dump(ref_path), read_dump(run_path)
    n = min(A.shape[0], B.shape[0], len(ids) - 1)
    same, kls, dn, nr, na = 0, [], [], 0.0, 0.0
    for r0 in range(0, n, 32):
        r1 = min(n, r0 + 32)
        pa = A[r0:r1].astype(np.float64)
        pb = B[r0:r1].astype(np.float64)
        pa -= pa.max(1, keepdims=True); pa -= np.log(np.exp(pa).sum(1, keepdims=True))
        pb -= pb.max(1, keepdims=True); pb -= np.log(np.exp(pb).sum(1, keepdims=True))
        kls += list((np.exp(pa) * (pa - pb)).sum(1))
        same += int((pa.argmax(1) == pb.argmax(1)).sum())
        nxt = np.array(ids[r0 + 1:r1 + 1])
        la, lb = pa[np.arange(len(nxt)), nxt], pb[np.arange(len(nxt)), nxt]
        nr -= float(la.sum()); na -= float(lb.sum())
        dn += list(la - lb)
    return summarize(n, same, [float(x) for x in kls], nr, na, [float(x) for x in dn], 0, A.shape[0], B.shape[0],
                     "full vocabulary")


# ------------------------------------------------------------------------------------------ common
def pct(v: list[float], q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def summarize(n, same, kls, nll_ref, nll_run, dn, mismatch, n_ref, n_run, kl_kind) -> dict:
    kls = [k for k in kls if not math.isnan(k)]
    mean_d = sum(dn) / n if n else float("nan")
    se = (math.sqrt(sum((x - mean_d) ** 2 for x in dn) / (n - 1) / n) if n > 1 else float("nan"))
    return {"positions": n, "ref_positions": n_ref, "run_positions": n_run, "target_mismatch": mismatch,
            "top1_pct": 100.0 * same / n if n else float("nan"),
            "kl_median": pct(kls, 0.5), "kl_p99": pct(kls, 0.99), "kl_mean": sum(kls) / len(kls) if kls else float("nan"),
            "kl_max": max(kls) if kls else float("nan"), "kl_kind": kl_kind,
            "ppl_ref": math.exp(nll_ref / n) if n else float("nan"), "ppl_run": math.exp(nll_run / n) if n else float("nan"),
            "nll_diff": mean_d, "nll_diff_se": se}


def verdict(r: dict, a) -> list[str]:
    bad = []
    if r["positions"] < a.min_positions:
        bad.append(f"only {r['positions']} positions (< {a.min_positions})")
    if r["target_mismatch"]:
        bad.append(f"{r['target_mismatch']} positions with a different target token (not the same text)")
    if not r["top1_pct"] >= a.min_top1:
        bad.append(f"top-1 {r['top1_pct']:.2f}% < {a.min_top1}%")
    if not r["kl_median"] <= a.max_median_kl:
        bad.append(f"median KL {r['kl_median']:.2e} > {a.max_median_kl:.0e}")
    d, se = r["nll_diff"], r["nll_diff_se"]
    ratio = r["ppl_run"] / r["ppl_ref"] if r["ppl_ref"] > 0 else float("nan")
    if not (abs(d) <= a.ppl_se * se or abs(ratio - 1.0) <= a.ppl_rel):
        bad.append(f"perplexity {r['ppl_run']:.4f} vs {r['ppl_ref']:.4f}: NLL diff {d:+.4f} +- {se:.4f} "
                   f"(> {a.ppl_se:g} SE and > {100 * a.ppl_rel:g}%)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ids", help="the token ids file (commas/whitespace); needed for --dump-logits files")
    ap.add_argument("--json")
    ap.add_argument("--min-top1", type=float, default=95.0)
    ap.add_argument("--max-median-kl", type=float, default=1e-3)
    ap.add_argument("--ppl-se", type=float, default=2.0)
    ap.add_argument("--ppl-rel", type=float, default=0.01)
    ap.add_argument("--min-positions", type=int, default=400)
    a = ap.parse_args()
    if not pathlib.Path(a.ref).exists() or pathlib.Path(a.ref).stat().st_size == 0:
        print(f"FAIL: the reference {a.ref} is missing or empty (no rows written)")
        return 1
    binary = is_binary(a.ref)
    ids = [int(x) for x in open(a.ids).read().replace(",", " ").split()] if a.ids else None
    if binary and ids is None:
        raise SystemExit("a --dump-logits reference needs --ids")
    ref = None if binary else read_logpos(a.ref)
    if ref is not None and not ref:
        print(f"FAIL: the reference {a.ref} holds no rows")
        return 1
    print(f"reference: {a.ref}" + ("" if binary else f" ({len(ref)} positions)"))
    print(f"{'run':28s} {'pos':>5s} {'top-1':>8s} {'KL med':>9s} {'KL p99':>9s} {'KL mean':>9s} {'PPL ref':>8s} "
          f"{'PPL run':>8s} {'dNLL +- SE':>18s}  verdict")
    out, failed = {}, False
    for path in a.runs:
        if not pathlib.Path(path).exists() or pathlib.Path(path).stat().st_size == 0:
            r, bad = {"positions": 0}, ["no rows written"]
        else:
            r = compare_dump(a.ref, path, ids) if binary else compare_logpos(ref, read_logpos(path))
            bad = verdict(r, a)
        out[path] = dict(r, problems=bad)
        failed |= bool(bad)
        if r.get("positions"):
            print(f"{pathlib.Path(path).name[:28]:28s} {r['positions']:5d} {r['top1_pct']:7.2f}% {r['kl_median']:9.2e} "
                  f"{r['kl_p99']:9.2e} {r['kl_mean']:9.2e} {r['ppl_ref']:8.4f} {r['ppl_run']:8.4f} "
                  f"{r['nll_diff']:+9.4f} +- {r['nll_diff_se']:.4f}  {'ok' if not bad else 'FAIL: ' + '; '.join(bad)}")
        else:
            print(f"{pathlib.Path(path).name[:28]:28s} FAIL: " + "; ".join(bad))
    if a.json:
        json.dump({"ref": a.ref, "runs": out, "criteria": vars(a)}, open(a.json, "w"), indent=1, default=str)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
