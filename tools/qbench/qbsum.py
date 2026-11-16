#!/usr/bin/env python3
"""QBENCH: QB / QG lines -> markdown tables.  usage: qbsum.py CARD PEAK_GBPS FILE...  (lines from ds4_iq_helper_bench / qb_ggml)"""
import re
import sys

card, peak = sys.argv[1], float(sys.argv[2])
kv = re.compile(r"(\w+)=(\S+)")
ours, ref = [], []
for fn in sys.argv[3:]:
    for ln in open(fn, errors="replace"):
        if ln.startswith("QB "):
            ours.append(dict(kv.findall(ln)))
        elif ln.startswith("QG "):
            d = dict(kv.findall(ln))
            d["src"] = "cv" if "/l-cv" in fn or "cv-" in fn.split("/")[-1] or "/cv/" in fn else "lcpp"
            ref.append(d)

NAMES = {"IQ2_XXS": "IQ2_XXS", "IQ2_XS": "IQ2_XS", "IQ2_S": "IQ2_S", "MXFP4": "MXFP4",
         "iq2_xxs": "IQ2_XXS", "iq2_xs": "IQ2_XS", "iq2_s": "IQ2_S", "mxfp4": "MXFP4", "q2_K": "Q2_K/Q3_K"}


def pct(gbs):
    return f"{gbs:.0f} ({100 * gbs / peak:.0f} %)"


print(f"### {card} - ours (ds4_iq_helper_bench --pool-file; busy = back to back, iso = one call alone)\n")
print("| format | shape | path | B | busy p50 / p90 µs | GB/s busy (% peak) | iso p50 / p90 µs | rel_rms / rel_max |")
print("|---|---|---|---|---|---|---|---|")
for d in ours:
    f = NAMES.get(d["tg"], d["tg"])
    shape = "whole" if d["ff"] == "2304" else f"TP slice {d['ff']}"
    b = float(d["bytes"])
    busy = f"{float(d['busy_p50_us']):.1f} / {float(d['busy_p90_us']):.1f}" if "busy_p50_us" in d else "NO DATA"
    gb = pct(b / (float(d["busy_p50_us"]) * 1e3)) if "busy_p50_us" in d else "NO DATA"
    print(f"| {f} | {shape} | {d['path']} | {d['B']} | {busy} | {gb} | {float(d['p50_us']):.1f} / {float(d['p90_us']):.1f} | "
          f"{float(d['rel_rms']):.2e} / {float(d['rel_max']):.2e} |")
print(f"\n### {card} - references (qb_ggml: graph time / experts)\n")
print("| source | mode | format | M | B | p50 / p90 µs per expert(-equiv.) | GB/s (% peak) |")
print("|---|---|---|---|---|---|---|")
for d in ref:
    f = NAMES.get(d["tg"], d["tg"])
    print(f"| {d['src']} | {d['mode']} | {f} | {d['ff']} | {d['B']} | {float(d['p50_us']):.1f} / {float(d['p90_us']):.1f} | "
          f"{pct(float(d['bytes']) / (float(d['p50_us']) * 1e3))} |")
