// Strata (docs/dsv41/cards/HX-04.md). The helper's expert kernels alone (iq::gate_up / iq::down, as
// gg::win_experts launches them) at decode shapes on one device: DeepSeek-V4.1-Flash geometry (n_embd 5120, n_ff 2304, k 6), a
// pool of random legal expert blobs larger than the L2 (each launch reads cold blobs, as the helper does), T tokens, G helper
// groups a launch (one entry a token each).  Reports us a launch, us an expert and GB/s of expert weights, per kernel.
//   --ab: also runs the portable token kernels (iq::g906_force(0)) on the same inputs and memcmps h and part (BITWISE or not).
//   --gap-us N: the host idles N us between launches (the helper's real duty cycle: one layer's request every ~600 us).
//   iq_helper_bench [--tg 16] [--td 16] [--T 1] [--groups 1,2,3,6] [--reps 300] [--pool 40] [--gap-us 0] [--ab] [--mode -1|0|1]
//   (mi50kern2) --ne N: entries a group (1..T; default T, every group all T tokens - the old shape); --ne 0: mixed, group g
//   1 + g % T entries.  Each entry is a different token (row t * 8 + g, x row t); ne_max passed to the kernels stays T, as
//   gg::win_experts passes it.  --ab-mode M: the A/B compares mode 0 (portable) against mode M (default 1: the gfx906 kernels
//   forced; -1: the engine's own policy with this process's env - DS4_NK / DS4_GS / DS4_MK_* / DS4_G906_NEMAX).  Exit 4 on any
//   A/B difference (was 1).
//   (mi50k) --dump FILE: the fixed rep's h and part (rep 7, cleared buffers) of the LAST mode run (with --ab, the --ab-mode
//   policy) appended raw to FILE, one block a G in --groups order: two processes with different env flags are byte-compared
//   with cmp (HX-24/bench.sh).  The engine's flags are this process's env (DS4_G906_PERGROUP / SGNLDS / DPP / NEMAX).
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_gguf.cuh"

#include "ggml.h"   // (QBENCH) ggml_get_type_traits(type)->to_float: the CPU fp32 reference

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
static void fill_mat(std::mt19937_64& rng, int type, int n, int k, uint8_t* m) {
    const int bb = iq::block_bytes(type), bv = iq::block_values(type);
    const size_t bytes = (size_t) n * (k / bv) * bb;
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t r = rng(); std::memcpy(m + i, &r, std::min<size_t>(8, bytes - i)); }
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    if (type == iq::kIQ1_M) return;   // (no leading d)
    for (size_t b = 0; b < bytes; b += (size_t) bb) { const uint16_t d = f2h(sd(rng)); std::memcpy(m + b, &d, 2); }
}
static std::vector<int> ints(const char* p) {
    std::vector<int> v;
    while (*p) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
// the roofline: n bytes read as uint4 (grid-stride), a sum written so nothing is dropped
__global__ void bw_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x + v.y + v.z + v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int qbench_main(int argc, char** argv);   // (QBENCH, below)

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--pool-file") return qbench_main(argc, argv);
    int tg = iq::kIQ2_XXS, td = -1, T = 1, reps = 300, pool = 40, gap_us = 0, mode = -1, E = 5120, FF = 2304, ne_arg = -1, ab_mode = 1;
    bool ab = false;
    std::string dump;
    std::vector<int> groups = {1, 2, 3, 6};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*) "0"; };
        if (a == "--tg") tg = std::atoi(nx());
        else if (a == "--td") td = std::atoi(nx());
        else if (a == "--T") T = std::atoi(nx());
        else if (a == "--reps") reps = std::atoi(nx());
        else if (a == "--pool") pool = std::atoi(nx());
        else if (a == "--gap-us") gap_us = std::atoi(nx());
        else if (a == "--mode") mode = std::atoi(nx());
        else if (a == "--groups") groups = ints(nx());
        else if (a == "--ab") ab = true;
        else if (a == "--ne") ne_arg = std::atoi(nx());
        else if (a == "--ab-mode") ab_mode = std::atoi(nx());
        else if (a == "--dump") dump = nx();
        else if (a == "--dims") { E = std::atoi(nx()); FF = std::atoi(nx()); }
        else { std::printf("unknown %s\n", a.c_str()); return 2; }
    }
    if (td < 0) td = tg;
    constexpr int kPicks = 8, k = 6;
    if (T < 1 || T > 16) { std::printf("T 1..16\n"); return 2; }
    if (ne_arg < 0) ne_arg = T;
    if (ne_arg > T) { std::printf("--ne 0..T\n"); return 2; }
    if (ab_mode < -1 || ab_mode > 1 || ab_mode == 0) { std::printf("--ab-mode -1 or 1\n"); return 2; }
    hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, 0));
    std::printf("device 0: %s warpSize %d CUs %d  | types gate/up %s down %s  E %d FF %d T %d ne %s\n", p.gcnArchName, p.warpSize,
                p.multiProcessorCount, iq::name(tg), iq::name(td), E, FF, T, ne_arg == 0 ? "mix" : std::to_string(ne_arg).c_str());
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    const size_t rb_g = iq::row_bytes(tg, E), rb_d = iq::row_bytes(td, FF);
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al(FF * rb_g), 2 * al(FF * rb_g)};
    const size_t blob = off[2] + al((size_t) E * rb_d);
    const size_t gu_bytes = 2 * FF * rb_g, dn_bytes = (size_t) E * rb_d;
    std::printf("blob %.2f MiB (gate+up %.2f MiB, down %.2f MiB), pool %d blobs = %.0f MiB\n", blob / 1048576.0, gu_bytes / 1048576.0,
                dn_bytes / 1048576.0, pool, pool * blob / 1048576.0);
    uint8_t* d_pool; CK(hipMalloc(&d_pool, (size_t) pool * blob));
    {
        std::mt19937_64 rng(1234);
        std::vector<uint8_t> h(blob, 0);
        for (int e = 0; e < pool; ++e) {
            fill_mat(rng, tg, FF, E, &h[off[0]]);
            fill_mat(rng, tg, FF, E, &h[off[1]]);
            fill_mat(rng, td, E, FF, &h[off[2]]);
            CK(hipMemcpy(d_pool + (size_t) e * blob, h.data(), blob, hipMemcpyHostToDevice));
        }
    }
    // activations: x as int8 rows (T of them) with a scale a 32; h's int8 rows by entry (t * 8 + i)
    const int nent = 16 * kPicks;
    std::mt19937_64 rng(99);
    std::vector<int8_t> xq((size_t) T * E), hq((size_t) nent * FF);
    std::vector<float> xd((size_t) T * E / 32), hd((size_t) nent * FF / 32);
    for (auto& v : xq) v = (int8_t) ((int) (rng() % 255) - 127);
    for (auto& v : hq) v = (int8_t) ((int) (rng() % 255) - 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (auto& v : xd) v = sc(rng);
    for (auto& v : hd) v = sc(rng);
    auto upv = [](const void* src, size_t n) { void* d; CK(hipMalloc(&d, std::max<size_t>(n, 4))); CK(hipMemcpy(d, src, n, hipMemcpyHostToDevice)); return d; };
    int8_t* d_xq = (int8_t*) upv(xq.data(), xq.size());
    float* d_xd = (float*) upv(xd.data(), xd.size() * 4);
    int8_t* d_hq = (int8_t*) upv(hq.data(), hq.size());
    float* d_hd = (float*) upv(hd.data(), hd.size() * 4);
    float *d_h, *d_part;
    CK(hipMalloc(&d_h, (size_t) nent * FF * 4)); CK(hipMalloc(&d_part, (size_t) nent * E * 4));
    hipEvent_t e0, e1, e2;
    CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1)); CK(hipEventCreate(&e2));
    {   // the device's read roofline on the same pool: one blob's bytes (cold: rotating), and the whole pool
        unsigned* d_o; CK(hipMalloc(&d_o, 4));
        for (const size_t nb : {gu_bytes, (size_t) pool * blob}) {
            for (const int blocks : {240, 480, 960, 1920}) {
                float tot = 0; const int R = 50;
                for (int r = 0; r < R; ++r) {
                    const uint8_t* src = nb >= (size_t) pool * blob ? d_pool : d_pool + (size_t) (r % pool) * blob;
                    CK(hipEventRecord(e0, s));
                    hipLaunchKernelGGL(bw_kernel, dim3(blocks), dim3(256), 0, s, (const uint4*) src, nb / 16, d_o);
                    CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
                    float ms; CK(hipEventElapsedTime(&ms, e0, e1)); tot += ms;
                }
                std::printf("roofline: read %.2f MiB with %d blocks x 256: %.2f us, %.1f GB/s\n", nb / 1048576.0, blocks, tot * 1000 / R, nb / (tot / R * 1e-3) / 1e9);
            }
        }
        CK(hipFree(d_o));
    }
    int fails = 0;
    for (int G : groups) {
        if (G < 1 || G > k) { std::printf("G %d: 1..%d\n", G, k); continue; }
        // per rep: its own plan (groups' blobs rotate through the pool)
        const int slotsz = 16 * kPicks;
        std::vector<unsigned long long> gp((size_t) reps * slotsz, 0ull);
        std::vector<int> gs((size_t) reps * (slotsz + 1), 0), ge((size_t) reps * slotsz, 0), ng(reps, G);
        for (int r = 0; r < reps; ++r) {
            int ne = 0;
            for (int g = 0; g < G; ++g) {
                gp[(size_t) r * slotsz + g] = (unsigned long long) (d_pool + (size_t) ((r * G + g) % pool) * blob);
                gs[(size_t) r * (slotsz + 1) + g] = ne;
                const int nge = ne_arg == 0 ? 1 + g % T : ne_arg;   // (each a different token: rows t * 8 + g, distinct over the groups)
                for (int q = 0; q < nge; ++q) ge[(size_t) r * slotsz + ne++] = ((g + q) % T) * kPicks + g;
            }
            gs[(size_t) r * (slotsz + 1) + G] = ne;
        }
        auto* d_gp = (unsigned long long*) upv(gp.data(), gp.size() * 8);
        auto* d_gs = (int*) upv(gs.data(), gs.size() * 4);
        auto* d_ge = (int*) upv(ge.data(), ge.size() * 4);
        auto* d_ng = (int*) upv(ng.data(), ng.size() * 4);
        auto args = [&](int r) {
            iq::Args a;
            a.grp_ptr = d_gp + (size_t) r * slotsz;
            a.grp_start = d_gs + (size_t) r * (slotsz + 1);
            a.grp_ent = d_ge + (size_t) r * slotsz;
            a.n_groups = d_ng + r;
            for (int q = 0; q < 3; ++q) a.tr[q] = off[q];
            a.n_embd = E; a.n_ff = FF; a.limit = 10.0f;
            a.xq = d_xq; a.xd = d_xd; a.hq = d_hq; a.hd = d_hd; a.h = d_h; a.part = d_part;
            a.tg = tg; a.td = td;
            return a;
        };
        const int lg = T * k;   // the grid's groups, as win_experts sizes it
        auto run = [&](int m, int r, bool timed, double* gu_ms, double* dn_ms) {
            iq::g906_force(m);
            const iq::Args a = args(r);
            if (timed) CK(hipEventRecord(e0, s));
            if (!iq::gate_up(a, lg, T, s)) { std::printf("gate_up refused\n"); std::exit(2); }
            if (timed) CK(hipEventRecord(e1, s));
            if (!iq::down(a, lg, T, s)) { std::printf("down refused\n"); std::exit(2); }
            if (timed) {
                CK(hipEventRecord(e2, s)); CK(hipEventSynchronize(e2));
                float a1, a2; CK(hipEventElapsedTime(&a1, e0, e1)); CK(hipEventElapsedTime(&a2, e1, e2));
                *gu_ms += a1; *dn_ms += a2;
            }
        };
        std::vector<int> modes = {mode};
        if (ab) modes = {0, ab_mode};
        std::vector<float> ref_h, ref_p;
        for (int m : modes) {
            const long long l0 = iq::g906_launches();
            for (int r = 0; r < 5; ++r) run(m, r, false, nullptr, nullptr);   // (warm)
            CK(hipStreamSynchronize(s));
            double gu = 0, dn = 0;
            for (int r = 0; r < reps; ++r) {
                run(m, r, true, &gu, &dn);
                if (gap_us > 0) { const double t = now_us(); while (now_us() - t < gap_us) {} }
            }
            // back-to-back (no events, no gaps): the throughput
            CK(hipStreamSynchronize(s));
            const double t0 = now_us();
            for (int r = 0; r < reps; ++r) run(m, r, false, nullptr, nullptr);
            CK(hipStreamSynchronize(s));
            const double bb = (now_us() - t0) / reps;
            const bool g906 = iq::g906_launches() > l0;
            const double gu_us = gu * 1000.0 / reps, dn_us = dn * 1000.0 / reps;
            std::printf("G %d T %d ne %s mode %2d (%s): gate_up %7.2f us (%6.1f GB/s)  down %7.2f us (%6.1f GB/s)  sum %7.2f us = %6.2f us/expert"
                        "  b2b %7.2f us/launch-pair (%6.1f GB/s)\n",
                        G, T, ne_arg == 0 ? "mix" : std::to_string(ne_arg).c_str(), m, g906 ? "g906" : "portable", gu_us, G * gu_bytes / gu_us / 1e3, dn_us, G * dn_bytes / dn_us / 1e3, gu_us + dn_us,
                        (gu_us + dn_us) / G, bb, G * (gu_bytes + dn_bytes) / bb / 1e3);
            if (ab || !dump.empty()) {   // one fixed rep's outputs, from cleared buffers
                CK(hipMemset(d_h, 0, (size_t) nent * FF * 4)); CK(hipMemset(d_part, 0, (size_t) nent * E * 4));
                run(m, 7, false, nullptr, nullptr);
                CK(hipStreamSynchronize(s));
                std::vector<float> hh((size_t) nent * FF), pp((size_t) nent * E);
                CK(hipMemcpy(hh.data(), d_h, hh.size() * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(pp.data(), d_part, pp.size() * 4, hipMemcpyDeviceToHost));
                if (!dump.empty() && m == modes.back()) {   // (mi50k: this mode's bytes for the cross-process cmp)
                    FILE* df = std::fopen(dump.c_str(), "ab");
                    if (df == nullptr || std::fwrite(hh.data(), 4, hh.size(), df) != hh.size() || std::fwrite(pp.data(), 4, pp.size(), df) != pp.size()) {
                        std::printf("dump: cannot write %s\n", dump.c_str()); std::exit(2); }
                    std::fclose(df);
                    std::printf("  dump G %d T %d: %zu + %zu floats -> %s\n", G, T, hh.size(), pp.size(), dump.c_str());
                }
                if (!ab) {}
                else if (ref_h.empty()) { ref_h = hh; ref_p = pp; }
                else {
                    const bool sh = std::memcmp(hh.data(), ref_h.data(), hh.size() * 4) == 0, sp = std::memcmp(pp.data(), ref_p.data(), pp.size() * 4) == 0;
                    double md = 0, mag = 0;
                    for (size_t i = 0; i < pp.size(); ++i) { md = std::max(md, (double) std::fabs(pp[i] - ref_p[i])); mag = std::max(mag, (double) std::fabs(ref_p[i])); }
                    double mdh = 0;
                    for (size_t i = 0; i < hh.size(); ++i) mdh = std::max(mdh, (double) std::fabs(hh[i] - ref_h[i]));
                    std::printf("  A/B G %d T %d: h %s (max|d| %.3e)  part %s (max|d| %.3e of max %.3e)\n", G, T, sh ? "BITWISE" : "DIFFERS", mdh,
                                sp ? "BITWISE" : "DIFFERS", md, mag);
                    fails += !(sh && sp);
                }
            }
        }
        iq::g906_force(-1);
        CK(hipFree(d_gp)); CK(hipFree(d_gs)); CK(hipFree(d_ge)); CK(hipFree(d_ng));
    }
    std::printf("%s\n", fails ? "A/B FAIL" : "ok");
    return fails ? 4 : 0;
}

// ---------------------------------------------------------------------------------------------------------------------------
// QBENCH (2026-11-16): one routed expert's decode as the engine runs it, on REAL expert bytes, per card and format.
//   --pool-file F.qbx   a QBX1 pool (tools: QBENCH mkpool.py): "QBX1", u32 tg, td, n_embd, n_ff, count, then per expert the
//                       GGUF bytes of w1 (gate [n_ff][n_embd]), w3 (up), w2 (down [n_embd][n_ff]).  tg/td: ggml type ids
//                       (16 IQ2_XXS, 17 IQ2_XS, 22 IQ2_S, 39 MXFP4).  The blobs rotate per rep (cold, as a decode reads them).
//   --slice FN [--slice-off F0]  the tensor-parallel member's shape: gate/up rows [F0, F0+FN), down columns [F0, F0+FN) of
//                       every expert (the lead's tp_range split; 0 = the whole expert)
//   --path tok|win      tok: gg::moe_gate_up + gg::moe_down (the lead's T = 1 token path, k picks, one live);
//                       win: gg::win_experts (the helper's body at every T, the lead's at T > 1): one group, B entries
//   --B 1,2,4,8         rows (tokens) that picked this expert (tok: B must be 1)
//   --reps 300 --warm 30
// Each call is timed whole (q8 x, gate_up + swiglu, q8 h, down, combine) by events: iso (p50_us: one call, then a host sync, the
// device idle between calls) and (r2) busy (busy_p50_us: the calls queued back to back, an event between each pair).  Output: one "QB" line per B
// with p50 / p90 / mean us, the weight bytes and GB/s at p50, and the error of the combined output (plan weight 1) against
// a CPU fp32 reference: weights dequantised by ggml's to_float (own E2M1 decode for MXFP4 if ggml has none), fp32
// activations, double sums, the engine's clamped swiglu (limit 10).
namespace {
namespace qgg = strata::ds4::gguf;
namespace qx3 = strata::dsv41k::exl3;
float qb_e8m0(uint8_t e) { return std::ldexp(1.0f, (int) e - 128); }   // ds4_gguf.cuh: kvalues are the E2M1 values doubled
void qb_dequant(int type, const uint8_t* row, int k, float* w) {
    if (type == qgg::kTypeMXFP4) {
        const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
        if (t != nullptr && t->to_float != nullptr && t->blck_size == 32) { t->to_float(row, w, k); return; }
        static const float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        for (int b = 0; b < k / 32; ++b) {
            const uint8_t* p = row + (size_t) b * 17;
            const float d = qb_e8m0(p[0]);
            for (int j = 0; j < 16; ++j) { w[b * 32 + j] = kv[p[1 + j] & 15] * d; w[b * 32 + j + 16] = kv[p[1 + j] >> 4] * d; }
        }
        return;
    }
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    if (t == nullptr || t->to_float == nullptr) { std::printf("ggml has no to_float for type %d\n", type); std::exit(2); }
    t->to_float(row, w, k);
}
size_t qb_row_bytes(int type, int k) { return qgg::type_row_bytes(type, k); }
int qb_blk(int type) { return qgg::type_block(type); }
}  // namespace

static int qbench_main(int argc, char** argv) {
    std::string pf, path = "win";
    int fn = 0, f0 = 0, reps = 300, warm = 30, K = 6;
    std::vector<int> Bs = {1, 2, 4, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*) "0"; };
        if (a == "--pool-file") pf = nx();
        else if (a == "--slice") fn = std::atoi(nx());
        else if (a == "--slice-off") f0 = std::atoi(nx());
        else if (a == "--path") path = nx();
        else if (a == "--B") Bs = ints(nx());
        else if (a == "--reps") reps = std::atoi(nx());
        else if (a == "--warm") warm = std::atoi(nx());
        else { std::printf("qbench: unknown %s\n", a.c_str()); return 2; }
    }
    FILE* f = std::fopen(pf.c_str(), "rb");
    char mg[4]; uint32_t hd[5];
    if (f == nullptr || std::fread(mg, 1, 4, f) != 4 || std::memcmp(mg, "QBX1", 4) != 0 || std::fread(hd, 4, 5, f) != 5) { std::printf("qbench: bad pool %s\n", pf.c_str()); return 2; }
    const int tg = (int) hd[0], td = (int) hd[1], E = (int) hd[2], FFw = (int) hd[3], cnt = (int) hd[4];
    const size_t rgw = qb_row_bytes(tg, E), rdw = qb_row_bytes(td, FFw);
    if (rgw == 0 || rdw == 0 || !qgg::type_on_device(tg) || !qgg::type_on_device(td)) { std::printf("qbench: types %d/%d not on the device\n", tg, td); return 2; }
    const int FF = fn > 0 ? fn : FFw;
    if (f0 < 0 || f0 + FF > FFw || FF % qb_blk(td) != 0 || f0 % qb_blk(td) != 0) { std::printf("qbench: slice %d+%d of %d not whole blocks\n", f0, FF, FFw); return 2; }
    const size_t rd = qb_row_bytes(td, FF), dcol0 = qb_row_bytes(td, f0);   // a sliced down row: bytes [dcol0, dcol0 + rd) of the whole row
    auto al = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t off[3] = {0, al((size_t) FF * rgw), 2 * al((size_t) FF * rgw)};
    const size_t blob = off[2] + al((size_t) E * rd);
    const size_t wbytes = 2 * (size_t) FF * rgw + (size_t) E * rd;   // the weight bytes one expert's call reads
    std::vector<uint8_t> h(blob * (size_t) cnt, 0), whole(2 * (size_t) FFw * rgw + (size_t) E * rdw);
    for (int e = 0; e < cnt; ++e) {
        if (std::fread(whole.data(), 1, whole.size(), f) != whole.size()) { std::printf("qbench: short pool\n"); return 2; }
        uint8_t* b = &h[(size_t) e * blob];
        std::memcpy(b + off[0], whole.data() + (size_t) f0 * rgw, (size_t) FF * rgw);                         // gate rows
        std::memcpy(b + off[1], whole.data() + (size_t) FFw * rgw + (size_t) f0 * rgw, (size_t) FF * rgw);     // up rows
        const uint8_t* dw = whole.data() + 2 * (size_t) FFw * rgw;
        for (int r = 0; r < E; ++r) std::memcpy(b + off[2] + (size_t) r * rd, dw + (size_t) r * rdw + dcol0, rd);   // down columns
    }
    std::fclose(f);
    hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, 0));
    std::printf("QBENCH device %s CUs %d | %s/%s E %d FF %d (of %d, off %d) | blob %.2f MiB, %zu weight bytes, pool %d | path %s\n", p.gcnArchName,
                p.multiProcessorCount, qgg::type_name(tg), qgg::type_name(td), E, FF, FFw, f0, blob / 1048576.0, wbytes, cnt, path.c_str());
    uint8_t* d_pool; CK(hipMalloc(&d_pool, h.size()));
    CK(hipMemcpy(d_pool, h.data(), h.size(), hipMemcpyHostToDevice));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    if (!qgg::prepare(s)) { std::printf("qbench: prepare failed\n"); return 2; }
    qx3::ExpertLayout lay;
    lay.fmt = qgg::kFmtMXFP4;
    lay.K[0] = lay.K[1] = tg == qgg::kTypeMXFP4 ? 0 : tg; lay.K[2] = td == qgg::kTypeMXFP4 ? 0 : td;
    for (int r = 0; r < 3; ++r) lay.trellis[r] = off[r];
    // the CPU reference's weights: blob 0 (the error rep), dequantised once
    std::vector<float> wg((size_t) FF * E), wu((size_t) FF * E), wdn((size_t) E * FF);
    for (int r = 0; r < FF; ++r) { qb_dequant(tg, &h[off[0] + (size_t) r * rgw], E, &wg[(size_t) r * E]); qb_dequant(tg, &h[off[1] + (size_t) r * rgw], E, &wu[(size_t) r * E]); }
    for (int r = 0; r < E; ++r) qb_dequant(td, &h[off[2] + (size_t) r * rd], FF, &wdn[(size_t) r * FF]);
    constexpr int kP = 8, kMaxB = 16;
    std::mt19937_64 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) kMaxB * E);
    for (auto& v : x) v = nd(rng);
    float *d_x, *d_h, *d_part, *d_out, *d_w; unsigned long long *d_plan, *d_gp; int *d_gs, *d_ge, *d_ng;
    CK(hipMalloc(&d_x, x.size() * 4)); CK(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_h, (size_t) kMaxB * kP * FF * 4)); CK(hipMalloc(&d_part, (size_t) kMaxB * kP * E * 4)); CK(hipMalloc(&d_out, (size_t) kMaxB * E * 4));
    const int nrot = cnt;   // per rep r: blob r % cnt; plans pre-built per blob so a rep is one call
    CK(hipMalloc(&d_plan, (size_t) nrot * kMaxB * kP * 8)); CK(hipMalloc(&d_w, (size_t) kMaxB * kP * 4));
    CK(hipMalloc(&d_gp, (size_t) nrot * kMaxB * kP * 8)); CK(hipMalloc(&d_gs, (size_t) nrot * (kMaxB * kP + 1) * 4));
    CK(hipMalloc(&d_ge, (size_t) nrot * kMaxB * kP * 4)); CK(hipMalloc(&d_ng, (size_t) nrot * 4));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    int rc = 0;
    for (const int B : Bs) {
        if (B < 1 || B > kMaxB || (path == "tok" && B != 1)) { std::printf("qbench: B %d not for path %s\n", B, path.c_str()); rc = 2; continue; }
        {   // plans: token t's pick 0 = this blob (weight 1), picks 1..K-1 off the device; one group of B entries
            std::vector<unsigned long long> plan((size_t) nrot * kMaxB * kP, 0ull), gp((size_t) nrot * kMaxB * kP, 0ull);
            std::vector<int> gs((size_t) nrot * (kMaxB * kP + 1), 0), ge((size_t) nrot * kMaxB * kP, 0), ng(nrot, 1);
            std::vector<float> w((size_t) kMaxB * kP, 0.0f);
            for (int t = 0; t < B; ++t) w[(size_t) t * kP] = 1.0f;
            for (int r = 0; r < nrot; ++r) {
                const unsigned long long bp = (unsigned long long) (d_pool + (size_t) r * blob);
                for (int t = 0; t < B; ++t) { plan[(size_t) r * kMaxB * kP + t * kP] = bp; ge[(size_t) r * kMaxB * kP + t] = t * kP; }
                gp[(size_t) r * kMaxB * kP] = bp;
                gs[(size_t) r * (kMaxB * kP + 1) + 0] = 0; gs[(size_t) r * (kMaxB * kP + 1) + 1] = B;
            }
            CK(hipMemcpy(d_plan, plan.data(), plan.size() * 8, hipMemcpyHostToDevice)); CK(hipMemcpy(d_gp, gp.data(), gp.size() * 8, hipMemcpyHostToDevice));
            CK(hipMemcpy(d_gs, gs.data(), gs.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(d_ge, ge.data(), ge.size() * 4, hipMemcpyHostToDevice));
            CK(hipMemcpy(d_ng, ng.data(), ng.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(d_w, w.data(), w.size() * 4, hipMemcpyHostToDevice));
        }
        auto call = [&](int r) {
            r %= nrot;
            if (path == "tok") {
                qx3::MoeArgs ma;
                ma.plan_ptr = d_plan + (size_t) r * kMaxB * kP; ma.plan_w = d_w; ma.lay = lay; ma.k = K; ma.n_embd = E; ma.n_ff = FF; ma.limit = 10.0f;
                qgg::moe_gate_up(ma, d_x, d_h, s);
                qgg::moe_down(ma, d_h, nullptr, d_out, s);
            } else {
                qx3::WinArgs wa;
                wa.grp_ptr = d_gp + (size_t) r * kMaxB * kP; wa.grp_start = d_gs + (size_t) r * (kMaxB * kP + 1);
                wa.grp_ent = d_ge + (size_t) r * kMaxB * kP; wa.n_groups = d_ng + r;
                wa.plan_ptr = d_plan + (size_t) r * kMaxB * kP; wa.plan_w = d_w;
                wa.x = d_x; wa.x_stride = (size_t) E; wa.h = d_h; wa.part = d_part; wa.out = d_out; wa.out_stride = (size_t) E;
                wa.lay = lay; wa.T = B; wa.k = K; wa.n_embd = E; wa.n_ff = FF; wa.limit = 10.0f;
                if (!qgg::win_experts(wa, s)) { std::printf("qbench: win_experts refused\n"); std::exit(2); }
            }
        };
        const int err0 = qgg::launch_errors();
        for (int r = 0; r < warm; ++r) call(r);
        CK(hipStreamSynchronize(s));
        std::vector<double> us((size_t) reps), bu((size_t) reps);
        for (int r = 0; r < reps; ++r) {   // iso: each call alone, the device idle between calls (host sync each rep)
            CK(hipEventRecord(e0, s)); call(r + 1); CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); us[(size_t) r] = ms * 1000.0;
        }
        {   // (r2) busy: the same calls queued back to back (an event between consecutive calls, one sync at the end) - the
            // device never idles, as in a decode step's stream; call r's time = event r -> event r + 1
            std::vector<hipEvent_t> ev((size_t) reps + 1);
            for (auto& e : ev) CK(hipEventCreate(&e));
            for (int r = 0; r < warm; ++r) call(r);
            CK(hipEventRecord(ev[0], s));
            for (int r = 0; r < reps; ++r) { call(r + 1); CK(hipEventRecord(ev[(size_t) r + 1], s)); }
            CK(hipEventSynchronize(ev[(size_t) reps]));
            for (int r = 0; r < reps; ++r) { float ms; CK(hipEventElapsedTime(&ms, ev[(size_t) r], ev[(size_t) r + 1])); bu[(size_t) r] = ms * 1000.0; }
            for (auto& e : ev) CK(hipEventDestroy(e));
        }
        // the error rep: blob 0, cleared outputs
        CK(hipMemset(d_out, 0, (size_t) kMaxB * E * 4));
        call(0);
        CK(hipStreamSynchronize(s));
        std::vector<float> out((size_t) B * E);
        CK(hipMemcpy(out.data(), d_out, out.size() * 4, hipMemcpyDeviceToHost));
        double maxabs = 0, maxref = 0, se = 0, sr = 0; long nonfin = 0;
        std::vector<double> hh((size_t) FF);
        for (int t = 0; t < B; ++t) {
            const float* xt = &x[(size_t) t * E];
            for (int j = 0; j < FF; ++j) {
                double g = 0, u = 0;
                for (int c = 0; c < E; ++c) { g += (double) wg[(size_t) j * E + c] * xt[c]; u += (double) wu[(size_t) j * E + c] * xt[c]; }
                g = std::min(g, 10.0); u = std::min(std::max(u, -10.0), 10.0);
                hh[(size_t) j] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int r = 0; r < E; ++r) {
                double y = 0;
                for (int j = 0; j < FF; ++j) y += (double) wdn[(size_t) r * FF + j] * (float) hh[(size_t) j];
                const double o = out[(size_t) t * E + r];
                if (!std::isfinite(o)) { ++nonfin; continue; }
                const double d = std::fabs(o - y);
                maxabs = std::max(maxabs, d); maxref = std::max(maxref, std::fabs(y)); se += d * d; sr += y * y;
            }
        }
        auto pct = [&](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min((size_t) reps - 1, (size_t) (reps * q))]; };
        const double p50 = pct(us, 0.5), p90 = pct(us, 0.9), b50 = pct(bu, 0.5), b90 = pct(bu, 0.9);
        double mean = 0, bmean = 0; for (double v : us) mean += v; mean /= reps; for (double v : bu) bmean += v; bmean /= reps;
        const int errs = qgg::launch_errors() - err0;
        std::printf("QB arch=%s tg=%s td=%s ff=%d off=%d path=%s B=%d reps=%d busy_p50_us=%.2f busy_p90_us=%.2f busy_mean_us=%.2f GBps_busy=%.1f "
                    "p50_us=%.2f p90_us=%.2f mean_us=%.2f bytes=%zu GBps_p50=%.1f "
                    "maxabs=%.4e maxref=%.4e rel_max=%.4e rel_rms=%.4e nonfinite=%ld launch_errors=%d\n",
                    p.gcnArchName, qgg::type_name(tg), qgg::type_name(td), FF, f0, path.c_str(), B, reps, b50, b90, bmean, wbytes / (b50 * 1e3),
                    p50, p90, mean, wbytes, wbytes / (p50 * 1e3),
                    maxabs, maxref, maxref > 0 ? maxabs / maxref : 0.0, sr > 0 ? std::sqrt(se / sr) : 0.0, nonfin, errs);
        std::fflush(stdout);
        if (errs != 0 || nonfin != 0) rc = 4;
    }
    return rc;
}
