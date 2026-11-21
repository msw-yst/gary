// include/strata/core/pool_plan.hpp - the shared MI50 expert pool: where each expert lives, and the manifest that
// tells the workers.  Header-only, no GPU: the pool holder and every worker include it, and pool_plan_test checks it.
//
// The layout (design note docs/flashnext/SHARED-POOL.md): N independent XTX workers
// (data-parallel engines, each with its own slots, dense weights and hot-expert cache) share ONE pool of experts on
// the MI50s.  Every worker caches the same hot prefix of the profile ranking in its own XTX VRAM (`hot` ranks); the
// pool holds every rank after it, each expert on exactly ONE MI50; what does not fit the pool overflows to each
// worker's CPU path (the coldest ranks, by construction).
//
// EXACTNESS.  A worker's text must equal that worker running alone.  So the placement is decided once, before any
// worker starts, from the ranking and the configuration only - never from load - and every worker reads the same
// manifest.  An expert is computed by the same device kind for a worker whether or not the other workers are busy.
//
// BALANCE.  A layer waits for the slowest MI50 it used, so the pool is balanced by routed MASS per unit of device speed,
// not by expert count: `weight` is a device's relative speed (on our box one MI50, whose transfers cost 2.4-2.9x, gets
// less; per-window helper clocks of a tp2 timing build, docs/flashnext/SHARED-POOL.md section 2).  The profile carries a ranking only (tools/make_profile.py), so
// the mass of rank r comes from the caller (a measured histogram) or from the coverage curve H(f) = 1 - (1 - f)^b
// (tools/pool_fit.py fits b ~ 1.7 to the 2026-11-21 decode logs).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::pool {

using Pair = std::pair<int32_t, int32_t>;   ///< (layer, expert)

struct Device {
    int64_t capacity = 0;   ///< experts this MI50 can hold (uniform Q4K blobs: free bytes after the guard / blob)
    double weight = 1.0;    ///< relative speed: its share of the pool's routed mass is weight / sum(weights)
};

struct Plan {
    int64_t hot = 0;                          ///< ranks [0, hot) stay in every worker's XTX cache
    std::vector<Pair> hot_pairs;              ///< those ranks, in rank order
    std::vector<std::vector<Pair>> by_device; ///< the pool: per device, in rank order (= slot order)
    std::vector<Pair> overflow;               ///< no device had room: the workers' CPU path (the coldest ranks)
    std::vector<double> mass;                 ///< per device: the routed mass it holds (fraction of all routed entries)
    double hot_mass = 0, overflow_mass = 0;
};

/// The routed mass of rank r of n under the coverage curve H(f) = 1 - (1 - f)^b.  Prefixes sum to H(N / n) exactly.
inline double curve_mass(size_t r, size_t n, double b) {
    if (n == 0) return 0.0;
    const double hi = std::pow(1.0 - (double) r / (double) n, b);
    const double lo = std::pow(1.0 - (double) (r + 1) / (double) n, b);
    return hi - lo;
}

/// `ranked`: every (layer, expert) pair once, hottest first.  `mass`: per rank (same length; it need not sum to 1), or
/// empty for the curve with exponent `b`.  Walks the ranks after `hot` from hot to cold and gives each to the device
/// with the lowest held-mass / weight that still has room (ties: the lower index), so the pool is mass-balanced by
/// speed while room lasts, and the devices that fill first push the cold tail to the others.  Deterministic.
inline Plan make_plan(const std::vector<Pair>& ranked, int64_t hot, const std::vector<Device>& devs,
                      const std::vector<double>& mass = {}, double b = 1.7) {
    Plan p;
    const size_t n = ranked.size();
    if (hot < 0) hot = 0;
    if ((size_t) hot > n) hot = (int64_t) n;
    p.hot = hot;
    p.by_device.assign(devs.size(), {});
    p.mass.assign(devs.size(), 0.0);
    double total = 0;
    std::vector<double> m(n);
    for (size_t r = 0; r < n; ++r) total += (m[r] = mass.size() == n ? mass[r] : curve_mass(r, n, b));
    if (total <= 0) total = 1;
    for (size_t r = 0; r < n; ++r) m[r] /= total;
    p.hot_pairs.assign(ranked.begin(), ranked.begin() + hot);
    for (size_t r = 0; r < (size_t) hot; ++r) p.hot_mass += m[r];
    for (size_t r = (size_t) hot; r < n; ++r) {
        int best = -1;
        double best_load = 0;
        for (size_t d = 0; d < devs.size(); ++d) {
            if ((int64_t) p.by_device[d].size() >= devs[d].capacity || devs[d].weight <= 0) continue;
            const double load = p.mass[d] / devs[d].weight;
            if (best < 0 || load < best_load) { best = (int) d; best_load = load; }
        }
        if (best < 0) { p.overflow.push_back(ranked[r]); p.overflow_mass += m[r]; continue; }
        p.by_device[(size_t) best].push_back(ranked[r]);
        p.mass[(size_t) best] += m[r];
    }
    return p;
}

// ---- the manifest: what the pool holder publishes and every worker reads (one small file, little-endian) ----
//
//   "STPM", u32 version (1), u64 pack_hash, u32 n_layers, u32 n_expert, u64 blob_bytes, u32 hot, u32 n_devices,
//   then per device: char bdf[16] (PCI bus id, NUL padded: device ordinals differ between processes), u64 base_bytes
//   (the cache allocation), u8 ipc[64] (hipIpcMemHandle_t of that allocation), u32 n_pairs, float weight,
//   then the hot prefix (hot x (u16 layer, u16 expert)), then each device's pairs in slot order (same encoding).
// Slot i of a device sits at base + i * blob_bytes (uniform blobs; the Q4K pack's 2,969,600 B is 256-aligned).
// A worker must refuse a manifest whose pack hash, geometry or blob size differs from its own model.
struct ManifestDevice {
    char bdf[16] = {};
    uint64_t base_bytes = 0;
    unsigned char ipc[64] = {};
    float weight = 1.0f;
    std::vector<Pair> pairs;
};

struct Manifest {
    static constexpr uint32_t kVersion = 1;
    uint64_t pack_hash = 0;
    uint32_t n_layers = 0, n_expert = 0;
    uint64_t blob_bytes = 0;
    std::vector<Pair> hot_pairs;
    std::vector<ManifestDevice> devices;
};

namespace detail {
template <class T> inline bool put(std::FILE* f, const T& v) { return std::fwrite(&v, sizeof v, 1, f) == 1; }
template <class T> inline bool get(std::FILE* f, T& v) { return std::fread(&v, sizeof v, 1, f) == 1; }
inline bool put_pairs(std::FILE* f, const std::vector<Pair>& v) {
    for (const auto& pr : v) {
        const uint16_t l = (uint16_t) pr.first, e = (uint16_t) pr.second;
        if (!put(f, l) || !put(f, e)) return false;
    }
    return true;
}
inline bool get_pairs(std::FILE* f, std::vector<Pair>& v, uint32_t n, uint32_t n_layers, uint32_t n_expert) {
    v.clear();
    v.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t l = 0, e = 0;
        if (!get(f, l) || !get(f, e) || l >= n_layers || e >= n_expert) return false;
        v.emplace_back((int32_t) l, (int32_t) e);
    }
    return true;
}
}   // namespace detail

/// Written to `path + ".tmp"`, then renamed: a worker never reads half a manifest.
inline bool write_manifest(const std::string& path, const Manifest& m, std::string& err) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { err = "pool manifest: cannot create " + tmp; return false; }
    using detail::put;
    const uint32_t ver = Manifest::kVersion, hot = (uint32_t) m.hot_pairs.size(), nd = (uint32_t) m.devices.size();
    bool ok = std::fwrite("STPM", 4, 1, f) == 1 && put(f, ver) && put(f, m.pack_hash) && put(f, m.n_layers) &&
              put(f, m.n_expert) && put(f, m.blob_bytes) && put(f, hot) && put(f, nd);
    for (const auto& d : m.devices) {
        const uint32_t np = (uint32_t) d.pairs.size();
        ok = ok && std::fwrite(d.bdf, sizeof d.bdf, 1, f) == 1 && put(f, d.base_bytes) &&
             std::fwrite(d.ipc, sizeof d.ipc, 1, f) == 1 && put(f, np) && put(f, d.weight);
    }
    ok = ok && detail::put_pairs(f, m.hot_pairs);
    for (const auto& d : m.devices) ok = ok && detail::put_pairs(f, d.pairs);
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "pool manifest: cannot write " + path;
        return false;
    }
    return true;
}

inline bool read_manifest(const std::string& path, Manifest& m, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "pool manifest: cannot open " + path; return false; }
    using detail::get;
    char magic[4] = {};
    uint32_t ver = 0, hot = 0, nd = 0;
    bool ok = std::fread(magic, 4, 1, f) == 1 && std::memcmp(magic, "STPM", 4) == 0 && get(f, ver) &&
              ver == Manifest::kVersion && get(f, m.pack_hash) && get(f, m.n_layers) && get(f, m.n_expert) &&
              get(f, m.blob_bytes) && get(f, hot) && get(f, nd) && m.n_layers <= 65536 && m.n_expert <= 65536 &&
              nd <= 16 && (uint64_t) hot <= (uint64_t) m.n_layers * m.n_expert;
    std::vector<uint32_t> np(ok ? nd : 0);
    if (ok) m.devices.assign(nd, {});
    for (uint32_t i = 0; ok && i < nd; ++i) {
        auto& d = m.devices[i];
        ok = std::fread(d.bdf, sizeof d.bdf, 1, f) == 1 && get(f, d.base_bytes) &&
             std::fread(d.ipc, sizeof d.ipc, 1, f) == 1 && get(f, np[i]) && get(f, d.weight) &&
             (uint64_t) np[i] * m.blob_bytes <= d.base_bytes;
        d.bdf[sizeof d.bdf - 1] = '\0';
    }
    ok = ok && detail::get_pairs(f, m.hot_pairs, hot, m.n_layers, m.n_expert);
    for (uint32_t i = 0; ok && i < nd; ++i) ok = detail::get_pairs(f, m.devices[i].pairs, np[i], m.n_layers, m.n_expert);
    std::fclose(f);
    if (!ok) err = "pool manifest: " + path + " is not a version 1 pool manifest or is truncated";
    return ok;
}

/// A worker's view: for (layer, expert), which pool device holds it and in which slot (-1, -1: not in the pool).
/// One flat table, built once from the manifest; the hot prefix is the worker's own XTX, not listed here.
struct Residency {
    uint32_t n_expert = 0;
    std::vector<int8_t> device;
    std::vector<int32_t> slot;
    void build(const Manifest& m) {
        n_expert = m.n_expert;
        const size_t n = (size_t) m.n_layers * m.n_expert;
        device.assign(n, -1);
        slot.assign(n, -1);
        for (size_t d = 0; d < m.devices.size(); ++d)
            for (size_t i = 0; i < m.devices[d].pairs.size(); ++i) {
                const auto& pr = m.devices[d].pairs[i];
                const size_t at = (size_t) pr.first * n_expert + (size_t) pr.second;
                device[at] = (int8_t) d;
                slot[at] = (int32_t) i;
            }
    }
    int device_of(int64_t layer, int64_t expert) const { return device[(size_t) layer * n_expert + (size_t) expert]; }
    int32_t slot_of(int64_t layer, int64_t expert) const { return slot[(size_t) layer * n_expert + (size_t) expert]; }
};

}   // namespace strata::core::pool
