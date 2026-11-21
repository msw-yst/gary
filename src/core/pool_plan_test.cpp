// src/core/pool_plan_test.cpp - the shared MI50 pool's placement and manifest (include/strata/core/pool_plan.hpp).
// Header-only, CPU only.  Not yet built (phase 1 of the pool design is desk-only).
#include "strata/core/pool_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>

using namespace strata::core::pool;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
std::vector<Pair> ranking(int layers, int experts) {   // a synthetic ranking: interleaved across the layers
    std::vector<Pair> r;
    for (int e = 0; e < experts; ++e)
        for (int l = 0; l < layers; ++l) r.emplace_back(l, (e * 7 + l * 13) % experts);
    return r;
}
// every pair exactly once in hot + pool + overflow, no device over capacity
void check_partition(const std::vector<Pair>& ranked, const Plan& p, const std::vector<Device>& devs, const char* what) {
    std::set<Pair> seen;
    size_t count = 0;
    for (const auto& pr : p.hot_pairs) { seen.insert(pr); ++count; }
    for (size_t d = 0; d < p.by_device.size(); ++d) {
        check((int64_t) p.by_device[d].size() <= devs[d].capacity, what);
        for (const auto& pr : p.by_device[d]) { seen.insert(pr); ++count; }
    }
    for (const auto& pr : p.overflow) { seen.insert(pr); ++count; }
    check(count == ranked.size() && seen.size() == ranked.size(), what);
}
}

int main() {
    const auto ranked = ranking(48, 512);
    check(ranked.size() == 24576 && std::set<Pair>(ranked.begin(), ranked.end()).size() == 24576, "synthetic ranking");

    // our box, P = 4 per worker: hot prefix 4,840 (the smaller XTX cache), 4 MI50 x 5,586, the slow-link MI50 at weight 0.58
    const std::vector<Device> four = {{5586, 1.0}, {5586, 0.58}, {5586, 1.0}, {5586, 1.0}};
    const Plan a = make_plan(ranked, 4840, four);
    check_partition(ranked, a, four, "P4: a partition");
    check(a.overflow.empty(), "P4: the pool holds everything past the hot prefix");
    double pool = 0;
    for (double m : a.mass) pool += m;
    check(std::fabs(a.hot_mass + pool - 1.0) < 1e-9, "P4: masses sum to 1");
    check(std::fabs(a.hot_mass - (1.0 - std::pow(1.0 - 4840.0 / 24576.0, 1.7))) < 1e-9, "P4: hot mass = H(f)");
    check(std::fabs(a.mass[1] / pool - 0.58 / 3.58) < 0.01, "P4: the slow-link MI50 holds its weight's share of the pool mass");
    check(a.by_device[1].size() < a.by_device[0].size(), "P4: the slow-link MI50 holds fewer experts");

    // P = 8: hot 3,784, pool needs 20,792 of 22,344 - the normal cards fill, the slow-link MI50 takes the cold rest
    const Plan b8 = make_plan(ranked, 3784, four);
    check_partition(ranked, b8, four, "P8: a partition");
    check(b8.overflow.empty(), "P8: still no overflow");

    // 3 MI50 (a 4-XTX box): 2,978 coldest ranks overflow to the CPU, and they are the last ranks
    const std::vector<Device> three = {{5586, 1.0}, {5586, 1.0}, {5586, 1.0}};
    const Plan c = make_plan(ranked, 4840, three);
    check_partition(ranked, c, three, "3 MI50: a partition");
    check(c.overflow.size() == 24576 - 4840 - 3 * 5586, "3 MI50: overflow count");
    check(c.overflow.front() == ranked[ranked.size() - c.overflow.size()] && c.overflow.back() == ranked.back(),
          "3 MI50: the overflow is the coldest tail");
    check(std::fabs(c.overflow_mass - std::pow((double) c.overflow.size() / 24576.0, 1.7)) < 1e-9, "3 MI50: overflow mass");

    // determinism: the same inputs give the same plan (every worker and the holder must agree)
    const Plan a2 = make_plan(ranked, 4840, four);
    check(a2.by_device == a.by_device && a2.hot_pairs == a.hot_pairs, "deterministic");

    // a measured histogram instead of the curve
    std::vector<double> hist(ranked.size(), 1.0);
    const Plan u = make_plan(ranked, 0, four, hist);
    check(u.by_device[1].size() < u.by_device[0].size() && u.overflow.empty(), "uniform mass: 46 still lighter");

    // manifest round trip
    Manifest m;
    m.pack_hash = 0x1234abcdull;
    m.n_layers = 48;
    m.n_expert = 512;
    m.blob_bytes = 2969600;
    m.hot_pairs = a.hot_pairs;
    for (size_t d = 0; d < a.by_device.size(); ++d) {
        ManifestDevice md;
        std::snprintf(md.bdf, sizeof md.bdf, "0000:%02x:00.0", (unsigned) (0x10 + d));
        md.base_bytes = (uint64_t) four[d].capacity * m.blob_bytes;
        for (int i = 0; i < 64; ++i) md.ipc[i] = (unsigned char) (i + d);
        md.weight = (float) four[d].weight;
        md.pairs = a.by_device[d];
        m.devices.push_back(md);
    }
    char path[] = "/tmp/strata-pool-manifest-XXXXXX";
    const int fd = mkstemp(path);
    check(fd >= 0, "temp file");
    std::string err;
    check(write_manifest(path, m, err), "write manifest");
    Manifest r;
    check(read_manifest(path, r, err), "read manifest");
    check(r.pack_hash == m.pack_hash && r.blob_bytes == m.blob_bytes && r.hot_pairs == m.hot_pairs &&
          r.devices.size() == 4, "manifest header");
    for (size_t d = 0; d < 4; ++d)
        check(r.devices[d].pairs == m.devices[d].pairs && std::memcmp(r.devices[d].ipc, m.devices[d].ipc, 64) == 0 &&
              std::string(r.devices[d].bdf) == m.devices[d].bdf, "manifest device");
    Residency res;
    res.build(r);
    const Pair first = a.by_device[2][0];
    check(res.device_of(first.first, first.second) == 2 && res.slot_of(first.first, first.second) == 0, "residency");
    check(res.device_of(a.hot_pairs[0].first, a.hot_pairs[0].second) == -1, "hot pairs are not the pool's");
    std::remove(path);
    std::printf("pool_plan_test: %d checks passed\n", checks);
    return 0;
}
