// Strata prototype (docs/dsv41/cards/EB-01.md). The helper expert cache's device path - gg::win_experts (q8 of
// x, iq::gate_up of grouped entries, q8 of h, iq::down, the weighted per-token combine), exactly what ds4_helper.cu issues -
// on TWO devices of one fat binary (the wave32 primary and the wave64 helper) with identical synthetic experts and inputs,
// memcmp of the T x n_embd result.  Same source, same shim, same scalar dp4a path: bitwise is the expectation; a
// difference is reported with its max |delta| and relative L1 (the fallback bound is 1e-5, the design's).  The i-quant
// blocks are random legal ones (any bits with a sane F16 scale, as tests/dsv41/iq_expert_test.cpp makes them); their values
// are not checked here - ds4_iq_test --devices A,B does that against ggml and a double reference.
//
//     helper_parity_test [devA devB] [--type 16,17,...] [--T 1,4]      exit 77: fewer than two devices; 1: a device refused or differs
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_kernels.cuh"   // kWin, kPicks

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace iq = gg::iq;
namespace x3 = gg::x3;
using strata::ds4::kPicks;
using strata::ds4::kWin;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

// a matrix of n rows of k values in an i-quant / K-quant type: random bytes, each block's leading F16 scale sane
static std::vector<uint8_t> make_mat(std::mt19937_64& rng, int type, int n, int k) {
    const int bb = iq::block_bytes(type), bv = iq::block_values(type);
    std::vector<uint8_t> m((size_t) n * (k / bv) * bb);
    for (auto& b : m) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (size_t b = 0; b < m.size(); b += (size_t) bb) {
        const uint16_t d = f2h(sd(rng));
        if (type == 29) {   // IQ1_M has no d field: its F16 scale is the top nibbles of the four u16 scales at byte 48
            for (int i = 0; i < 4; ++i) {
                uint16_t sc; std::memcpy(&sc, &m[b + 48 + 2 * i], 2);
                sc = (uint16_t) ((sc & 0x0fffu) | (((d >> (4 * i)) & 0xfu) << 12));
                std::memcpy(&m[b + 48 + 2 * i], &sc, 2);
            }
        } else {
            std::memcpy(&m[b], &d, 2);
        }
    }
    return m;
}

struct Inputs {   // host-side, identical for both devices
    int type, E, FF, NEXP, T, k;
    size_t off[3], blob;
    std::vector<uint8_t> blobs;
    std::vector<float> x;                       // [T][E]
    std::vector<unsigned long long> plan_slot;  // [T][8]: expert index + 1 or 0
    std::vector<float> plan_w;
    std::vector<int> grp_start, grp_ent;
    std::vector<int> grp_exp;                   // group -> expert
};

static Inputs make_inputs(int type, uint64_t seed, int T, int E, int FF) {
    Inputs in;
    in.type = type; in.E = E; in.FF = FF; in.NEXP = 5; in.T = T; in.k = 6;
    std::mt19937_64 rng(seed);
    const size_t rb_g = iq::row_bytes(type, in.E), rb_d = iq::row_bytes(type, in.FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    in.off[0] = 0; in.off[1] = al(in.FF * rb_g); in.off[2] = al(in.FF * rb_g) * 2; in.blob = in.off[2] + al((size_t) in.E * rb_d);
    in.blobs.assign((size_t) in.NEXP * in.blob, 0);
    for (int e = 0; e < in.NEXP; ++e) {
        auto G = make_mat(rng, type, in.FF, in.E), U = make_mat(rng, type, in.FF, in.E), D = make_mat(rng, type, in.E, in.FF);
        std::memcpy(&in.blobs[e * in.blob + in.off[0]], G.data(), G.size());
        std::memcpy(&in.blobs[e * in.blob + in.off[1]], U.data(), U.size());
        std::memcpy(&in.blobs[e * in.blob + in.off[2]], D.data(), D.size());
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    in.x.resize((size_t) in.T * in.E);
    for (auto& v : in.x) v = nd(rng);
    // the plan: each token's k picks, about half of them "the helper's" (an expert), the rest 0 (another owner's)
    in.plan_slot.assign((size_t) in.T * kPicks, 0ull);
    in.plan_w.assign((size_t) in.T * kPicks, 0.0f);
    std::vector<std::pair<int, int>> byexp;
    for (int t = 0; t < in.T; ++t)
        for (int i = 0; i < in.k; ++i) {
            in.plan_w[(size_t) t * kPicks + i] = 0.05f + 0.3f * (float) (rng() % 100) / 100.0f;
            if (rng() % 2) {
                const int e = (int) (rng() % (uint64_t) in.NEXP);
                in.plan_slot[(size_t) t * kPicks + i] = (unsigned long long) (e + 1);
                byexp.push_back({e, t * kPicks + i});
            }
        }
    std::sort(byexp.begin(), byexp.end());
    for (size_t a = 0; a < byexp.size();) {
        const int e = byexp[a].first;
        in.grp_exp.push_back(e);
        in.grp_start.push_back((int) in.grp_ent.size());
        while (a < byexp.size() && byexp[a].first == e) in.grp_ent.push_back(byexp[a++].second);
    }
    in.grp_start.push_back((int) in.grp_ent.size());
    return in;
}

// the helper's issue sequence on one device: the result rows [T][E] on the host, or empty on refusal
static std::vector<float> run_on(int dev, const Inputs& in, std::string& why) {
    CK(hipSetDevice(dev));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    if (!gg::prepare(s)) { why = "gg::prepare refused"; return {}; }
    uint8_t* dblob; CK(hipMalloc(&dblob, in.blobs.size())); CK(hipMemcpy(dblob, in.blobs.data(), in.blobs.size(), hipMemcpyHostToDevice));
    std::vector<unsigned long long> plan(in.plan_slot.size(), 0ull), gp(in.grp_exp.size());
    for (size_t i = 0; i < plan.size(); ++i) plan[i] = in.plan_slot[i] ? (unsigned long long) (dblob + (in.plan_slot[i] - 1) * in.blob) : 0ull;
    for (size_t g = 0; g < gp.size(); ++g) gp[g] = (unsigned long long) (dblob + (size_t) in.grp_exp[g] * in.blob);
    auto up = [](const void* p, size_t n) { void* d; CK(hipMalloc(&d, std::max<size_t>(1, n))); CK(hipMemcpy(d, p, n, hipMemcpyHostToDevice)); return d; };
    const int ng = (int) in.grp_exp.size();
    x3::WinArgs a;
    a.grp_ptr = (const unsigned long long*) up(gp.data(), gp.size() * 8);
    a.grp_start = (const int*) up(in.grp_start.data(), in.grp_start.size() * 4);
    a.grp_ent = (const int*) up(in.grp_ent.data(), std::max<size_t>(1, in.grp_ent.size()) * 4);
    a.n_groups = (const int*) up(&ng, 4);
    a.plan_ptr = (const unsigned long long*) up(plan.data(), plan.size() * 8);
    a.plan_w = (const float*) up(in.plan_w.data(), in.plan_w.size() * 4);
    a.x = (const float*) up(in.x.data(), in.x.size() * 4);
    a.x_stride = (size_t) in.E;
    float *h, *part, *out;
    CK(hipMalloc(&h, (size_t) in.T * kPicks * in.FF * 4)); CK(hipMalloc(&part, (size_t) in.T * kPicks * in.E * 4)); CK(hipMalloc(&out, (size_t) in.T * in.E * 4));
    CK(hipMemset(h, 0, (size_t) in.T * kPicks * in.FF * 4)); CK(hipMemset(part, 0, (size_t) in.T * kPicks * in.E * 4));
    a.h = h; a.part = part; a.out = out; a.out_stride = (size_t) in.E;
    a.lay.fmt = gg::kFmtMXFP4;
    for (int r = 0; r < 3; ++r) { a.lay.K[r] = in.type; a.lay.trellis[r] = in.off[r]; }
    a.T = in.T; a.k = in.k; a.n_embd = in.E; a.n_ff = in.FF; a.limit = 7.0f;
    if (!gg::win_experts(a, s)) { why = "win_experts refused"; return {}; }
    CK(hipStreamSynchronize(s));
    std::vector<float> y((size_t) in.T * in.E);
    CK(hipMemcpy(y.data(), out, y.size() * 4, hipMemcpyDeviceToHost));
    CK(hipDeviceSynchronize());
    return y;
}

int main(int argc, char** argv) {
    int devA = 0, devB = 1;
    std::vector<int> types = {16, 17, 22, 18, 29};   // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_M: the IST-DAS sets' types
    // T = 1 is decode: every group has one entry, the shape the gfx906 token kernels (ds4_iq_g906.inc, ne_max <= 1) take;
    // T = 4 has multi-entry groups and so the portable NE kernels (unless DS4_G906_NEMAX is raised)
    std::vector<int> Ts = {1, 4};
    // geometries (n_embd x n_ff): a small one, and DeepSeek-V4.1-Flash's 5120 x 2304 - the gfx906 token kernels decline the
    // small one (their per-lane sub-block count must be 2..8), so only the real geometry exercises them
    const std::pair<int, int> geoms[] = {{512, 256}, {5120, 2304}};
    int ai = 1;
    if (argc > 2 && argv[1][0] != '-' && argv[2][0] != '-') { devA = std::atoi(argv[1]); devB = std::atoi(argv[2]); ai = 3; }
    for (int i = ai; i < argc; ++i)
        if ((std::string(argv[i]) == "--type" || std::string(argv[i]) == "--T") && i + 1 < argc) {
            auto& v = std::string(argv[i]) == "--type" ? types : Ts;
            v.clear(); for (const char* p = argv[++i]; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
        }
    int n = 0;
    CK(hipGetDeviceCount(&n));
    if (n < 2 || devA >= n || devB >= n) { std::printf("helper_parity_test: needs two devices (have %d): skipped\n", n); return 77; }
    for (int d : {devA, devB}) {
        hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, d));
        std::printf("device %d: %s warpSize %d\n", d, p.gcnArchName, p.warpSize);
    }
    int fails = 0;
    for (int type : types)
      for (const auto& geo : geoms)
      for (int T : Ts) {
        if (!iq::supported(type)) { std::printf("%-7s no device kernel: skipped\n", iq::name(type)); break; }
        const Inputs in = make_inputs(type, 11u + (uint64_t) type + 1000u * (uint64_t) T, T, geo.first, geo.second);
        std::string wa, wb;
        const long long l0 = iq::g906_launches();
        const auto ya = run_on(devA, in, wa), yb = run_on(devB, in, wb);
        const long long g906 = iq::g906_launches() - l0;   // gfx906 token-kernel launches taken (0: portable kernels only)
        if (ya.empty() || yb.empty()) { std::printf("%-7s refused: A %s / B %s FAIL\n", iq::name(type), wa.c_str(), wb.c_str()); ++fails; continue; }
        const bool same = std::memcmp(ya.data(), yb.data(), ya.size() * 4) == 0;
        double md = 0, l1 = 0, mag = 0;
        int finite_bad = 0;
        for (size_t i = 0; i < ya.size(); ++i) {
            if (!std::isfinite(ya[i]) || !std::isfinite(yb[i])) ++finite_bad;
            md = std::max(md, (double) std::fabs(ya[i] - yb[i])); l1 += std::fabs(ya[i] - yb[i]); mag += std::fabs(ya[i]);
        }
        const double rel = l1 / (mag + 1e-30);
        const bool ok = finite_bad == 0 && (same || rel < 1e-5);
        std::printf("%-7s %4dx%-4d win_experts T %d groups %zu entries %zu: %s  max|d| %.3e  rel L1 %.3e  non-finite %d  g906 launches %lld  %s\n",
                    iq::name(type), in.E, in.FF, in.T, in.grp_exp.size(), in.grp_ent.size(), same ? "BITWISE" : "differs", md, rel,
                    finite_bad, g906, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    std::printf("%s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
