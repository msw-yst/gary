"""tools/pool_fit.py - fit arithmetic for the shared MI50 expert pool (data-parallel XTX workers), Flash-Next Q4K.

Desk tool (no GPU, no engine): reproduces the fit tables of the design note
`docs/flashnext/SHARED-POOL.md`.  Every constant below is tagged with where it comes from.

The profile file (`data/expert-profile.bin`) carries a RANKING only, no counts (tools/make_profile.py), so the routed
traffic a set of ranks holds comes from a coverage curve H(f) = 1 - (1 - f)^b over the held fraction f of the 24,576
experts (the same family generate.cpp uses for the layer split), with b fitted to the decode batch lines of the
2026-11-21 comparative-timing and decode A/B engine logs ("routed entries per window: VRAM / PCIe / CPU"; arm letters
as in docs/flashnext/data/2026-11-21-comparative-timing.md):

    held top ranks (of 24,576)   local share measured        b
    3,735 (15.2 %)               25.4-27.4 %                 1.8-2.0
    4,248 (17.3 %)               27.8-31.3 %                 1.7-2.0
    4,840 (19.7 %)               30.3 %                      1.6
    5,302 (21.6 %)               36.4 %                      1.9
    15,420 GPU total (62.7 %)    78.2-85.1 % (CPU 14.9-21.8) 1.6-1.9
    agent turns (arm R): 4,859 -> 46-52 %, 16,031 -> 84-89 %  2.4-3.0 at the top, 1.8 overall

    python3 tools/pool_fit.py            # the tables
    python3 tools/pool_fit.py --b 1.6    # another curve
"""
import argparse
import math

N_EXP = 48 * 512                       # experts [geometry]
EXPERT_B = 921_600 + 921_600 + 1_126_400   # Q4_K gate + Q4_K up + Q5_0 down, every layer [MEASURED: types/quantize log]
GIB = 1 << 30
MI50_EXPERTS = 5586                    # per MI50 with the 512 MiB guard: 15.45 GiB [MEASURED: arm S / G engine logs]
SLOT_GIB = 0.73                        # one --batch slot session on a single card (max-context 131072, kv int8) [MEASURED]
PER_SLOT_EXPERTS = round(SLOT_GIB * GIB / EXPERT_B)   # = 264
# expert cache of a single-card worker at P slots: measured at P = 4 (arm G) and P = 8 (arm S); 264 experts per slot
# otherwise.  Keys are short card ids used in the run logs and printed below: "43" = XTX A, "84" = XTX B (in the output,
# "46" = the slow-link MI50).
XTX = {"43": {4: 5302, 8: 4248}, "84": {4: 4840}}      # [MEASURED: arm G's two engine logs, arm S at 8 users]
MTP_COST = {4: 443, 8: 513}            # draft head + slot drafters [MEASURED: R vs G (P 4), S+MTP vs S (P 8)]
LOAN = 1556                            # prompt-path loan of XTX slots during a read [MEASURED]


def xtx_slots(card: str, p: int, mtp: bool = False) -> int:
    known = XTX[card]
    base_p = 4 if 4 in known else min(known)
    n = known.get(p, known[base_p] + (base_p - p) * PER_SLOT_EXPERTS)
    if mtp:
        n -= MTP_COST.get(p, MTP_COST[4])
    return n


def cover(f: float, b: float) -> float:
    """routed share held by the top fraction f of the ranking"""
    f = min(max(f, 0.0), 1.0)
    return 1.0 - (1.0 - f) ** b


def tail(f_from: float, b: float) -> float:
    """routed share of the ranks after the top fraction f_from"""
    return 1.0 - cover(f_from, b)


def option_a(p: int, b: float, mtp: bool, n_mi50: int = 4, same_hot: bool = True, workers=("43", "84")):
    caps = [xtx_slots(w, p, mtp) for w in workers]
    hot = min(caps)                       # the pool must hold every rank past the smallest worker's hot prefix
    pool_need = N_EXP - hot
    pool_cap = n_mi50 * MI50_EXPERTS
    over = max(0, pool_need - pool_cap)   # option (c): the coldest overflow to the CPU path
    rows = []
    for w, c in zip(workers, caps):
        h = hot if same_hot else c
        local = cover(h / N_EXP, b)
        cpu = tail((N_EXP - over) / N_EXP, b) if over else 0.0
        rows.append((w, c, h, local, 1.0 - local - cpu, cpu))
    return hot, pool_need, pool_cap, over, rows


def option_b(p: int, b: float, mtp: bool):
    c43, c84 = xtx_slots("43", p, mtp), xtx_slots("84", p, mtp)
    union = c43 + c84
    both = cover(union / N_EXP, b)
    s43 = both * c43 / union              # ranks split alternately in proportion to the caches
    s84 = both - s43
    return union, N_EXP - union, both, s43, s84, 1.0 - both


def mi50_load(users: int, pool_share: float, weights=(1.0, 1.0, 1.0, 1.0)):
    """entries per worker-window per MI50 and a busy-time estimate from the per-window helper clocks of a tp2 timing build
    (T, STRATA_HELPER_TIMING=1: busy = 44.9 us a launch + 10.2 us an entry on a normal card, 48.9 + 19.2 on the slow-link MI50)"""
    e_total = users * 480 * pool_share
    wsum = sum(weights)
    out = []
    for i, w in enumerate(weights):
        e = e_total * w / wsum
        slow = i == 3
        a, bb = (48.9, 19.2) if slow else (44.9, 10.24)
        launches = 48 * (1 - math.exp(-e / 48))
        out.append((e, (launches * a + e * bb) / 1000.0, a + bb * e / 48, slow))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=float, default=1.7)
    a = ap.parse_args()
    b = a.b
    print(f"expert {EXPERT_B} B; {PER_SLOT_EXPERTS} experts per KV slot; MI50 {MI50_EXPERTS} each, 4 = {4 * MI50_EXPERTS}; "
          f"curve b = {b}")
    print("\n(a) same hot set on both XTX (pool = every rank past it); (c) = coldest overflow to CPU when the pool is short")
    for n_mi50 in (4, 3):
        for mtp in (False, True):
            for p in (2, 4, 8):
                hot, need, cap, over, rows = option_a(p, b, mtp, n_mi50)
                r = rows[0]
                print(f"  MI50 x{n_mi50} P={p}{' +MTP' if mtp else ''}: XTX caches 43 {rows[0][1]} / 84 {rows[1][1]}, hot {hot} "
                      f"({hot / N_EXP:.1%}); pool needs {need} of {cap} -> overflow {over} ({over / N_EXP:.1%} of experts); "
                      f"per worker: XTX {r[3]:.1%}, pool {r[4]:.1%}, CPU {r[5]:.2%}")
    print("\n(a') own hot set per worker (43 keeps its larger cache), MI50 x4")
    for p in (4, 8):
        _, _, _, _, rows = option_a(p, b, False, 4, same_hot=False)
        print("  P=%d: " % p + "; ".join(f"XTX{w} {c} -> local {lo:.1%} pool {po:.1%}" for w, c, h, lo, po, cp in rows))
    print("\n(b) disjoint hot sets, the other XTX's via peer access")
    for p in (4, 8):
        union, pool_need, both, s43, s84, pool = option_b(p, b, False)
        print(f"  P={p}: union {union} ({union / N_EXP:.1%}), pool needs {pool_need}; worker on 43: own {s43:.1%} peer {s84:.1%}"
              f" pool {pool:.1%}; worker on 84: own {s84:.1%} peer {s43:.1%} pool {pool:.1%}; CPU 0")
    print("\nMI50 load per worker-window (option a, same hot), equal split vs 46:00 weighted 0.58")
    for users in (2, 4, 8):
        p = max(users, 4)
        _, _, _, _, rows = option_a(p, b, False)
        ps = rows[1][4]
        for wts in ((1, 1, 1, 1), (1, 1, 1, 0.58)):
            ld = mi50_load(users, ps, wts)
            print(f"  {users} users/worker, pool {ps:.1%}, weights {wts}: " +
                  "; ".join(f"{'46' if s else 'n '} {e:.0f} entries busy {bz:.2f} ms gpu {rt:.0f} us/launch"
                            for e, bz, rt, s in ld))


if __name__ == "__main__":
    main()
