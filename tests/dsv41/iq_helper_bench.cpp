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

int main(int argc, char** argv) {
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
