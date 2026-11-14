// src/kernels/cross_device_grouped_parity.cpp - prototype (docs/dsv41/cards/EB-01.md, EB-01): the grouped native
// expert path on TWO devices of one fat HIP binary (a wave32 primary, e.g. gfx1100, and a wave64 helper, gfx906) on
// identical synthetic inputs, step by step, against each other and against the CPU float reference of
// native_expert_parity --synthetic.
//
//     build/cross_device_grouped_parity [devA devB] [GU/DOWN ...] [--strict]
//       devices: ordinals (default 0 1); pairs: ggml type names (default q4_K/q5_1 iq3_s/iq4_nl, the UD-Q4_K_XL and the
//       GSQ IQ3_S pack layouts at n_embd 2560 / n_ff 640)
//
// What is compared, per format pair, each device running the same native_expert_grouped entry point the helper expert
// caches call (remote_experts.cpp -> expert_cache.cpp), once in STRATA_EXP_MODE 0 (the CUDA layout, the same kernels on
// both) and once in each device's own default policy (wave32: the card's default, usually 0; wave64: 7):
//   (1) quantize_q8_1_rows of x: bitwise (integer rounding of floats, no transcendental);
//   (2) gate/up floats alone (native_expert_set_mode phase 3): bitwise expected in the same mode - the same dp4a chains,
//       the same float sums in the same order;
//   (3) the q8_1 blocks of h after SwiGLU (phase 1): NOT bitwise across GCN and RDNA3 - silu uses __expf (v_exp_f32), whose
//       approximation differs by architecture - so the codes are counted: differing codes, the largest |delta|, the blocks
//       whose fp16 d / s differ.  Expected: |delta| <= 1 in <= 0.1 % of the codes;
//   (4) the down projection alone (phase 2) on IDENTICAL hq bytes (device A's mode-0 blocks copied to both): bitwise;
//   (5) the whole pipeline (phase 0): byte equality, max |diff|, relative L1 between the devices (expected <= 1e-4), and
//       each device against the CPU float reference (ggml to_float + a double-precision SwiGLU expert), rel L1 < 3e-2 as
//       native_expert_parity requires.
// Different modes (0 on A vs 7 on B) take different kernels (16-lane sub-rows vs the LDS R2 pairs for the IQ formats):
// their outputs are reported, not expected equal.
//
// Exit code: 1 when a device's whole-pipeline output misses the CPU-reference tolerance (or is not finite); with
// --strict also when (1), (2) or (4) are not bitwise in mode 0, (3) exceeds +-1 / 0.1 %, or (5) exceeds 1e-4 between
// the devices in mode 0.  77 (ctest SKIP_RETURN_CODE) with fewer than two devices.  Everything else is reported only.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace k = strata::kernels;

namespace {

constexpr int64_t H = 2560, FF = 640;   // the pack layouts (native_expert_parity's shape)
constexpr int G = 3;                    // experts (one group each: a window's groups are different experts)
constexpr int NT = 3;                   // tokens: every expert sees every token (an entry per (expert, token))
constexpr int CAP = G * NT;             // entries
constexpr size_t Q8_1_BYTES = 36;       // block_q8_1: {half d, half s} then 32 int8 codes

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cross_device_grouped_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}
double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = (std::max)(m, (double) std::fabs((double) a[i] - b[i]));
    return m;
}
bool finite(const std::vector<float>& a) {
    for (const float v : a) if (!std::isfinite(v)) return false;
    return true;
}
template<typename T>
bool bitwise(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
const char* yn(bool same) { return same ? "bitwise equal" : "DIFFER"; }

int type_of(std::string name) {
    for (auto& c : name) c = (char) std::tolower((unsigned char) c);
    static const struct { const char* n; int t; } trellis[] = {{"tq2_t", 144}, {"tqk6", 145}, {"tqk7", 146},
                                                              {"tqp5", 147}, {"tqp6", 148}, {"tqp7", 149}};
    for (const auto& e : trellis) if (name == e.n) return e.t;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n == nullptr) continue;
        std::string m(n);
        for (auto& c : m) c = (char) std::tolower((unsigned char) c);
        if (m == name) return t;
    }
    return -1;
}

// Random weights (rows of different magnitudes) quantized by ggml into the blob layout: native_expert_parity's
// synthetic_blob, unchanged.
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    if (f.tail_off) {
        const bool g4 = f.gu_type == 40, d4 = f.d_type == 40;
        const float tail[4] = {g4 ? 0.75f : 1.f, g4 ? 1.25f : 1.f, d4 ? 0.5f : 1.f, 0.f};
        std::memcpy(blob.data() + f.tail_off, tail, sizeof tail);
    }
    return blob;
}

// The CPU float reference of native_expert_parity's check_blob (a): ggml's own to_float, then a double-precision
// SwiGLU expert per (blob, token).  Entry e = g * NT + t writes row e of the output (ent_dst[e] = e).
std::vector<float> cpu_reference(const cpu::NativeFmt& f, const std::vector<std::vector<uint8_t>>& blobs,
                                 const std::vector<float>& x) {
    const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
    std::vector<float> ref((size_t) CAP * H);
    std::vector<float> Gm((size_t) FF * H), Um((size_t) FF * H), Dm((size_t) H * FF), h((size_t) FF);
    for (int g = 0; g < G; ++g) {
        const std::vector<uint8_t>& blob = blobs[(size_t) g];
        for (int64_t r = 0; r < FF; ++r) {
            tg->to_float(blob.data() + r * f.gu_row, Gm.data() + r * H, H);
            tg->to_float(blob.data() + f.up_off + r * f.gu_row, Um.data() + r * H, H);
        }
        for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, Dm.data() + r * FF, FF);
        if (f.tail_off) {
            float tail[4];
            std::memcpy(tail, blob.data() + f.tail_off, sizeof tail);
            for (auto& v : Gm) v *= tail[0];
            for (auto& v : Um) v *= tail[1];
            for (auto& v : Dm) v *= tail[2];
        }
        for (int t = 0; t < NT; ++t) {
            const float* xt = x.data() + (size_t) t * H;
            for (int64_t r = 0; r < FF; ++r) {
                double gg = 0, uu = 0;
                for (int64_t i = 0; i < H; ++i) { gg += (double) Gm[r * H + i] * xt[i]; uu += (double) Um[r * H + i] * xt[i]; }
                h[(size_t) r] = (float) (gg / (1.0 + std::exp(-gg)) * uu);
            }
            float* out = ref.data() + (size_t) (g * NT + t) * H;
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i) o += (double) Dm[r * FF + i] * h[(size_t) i];
                out[r] = (float) o;
            }
        }
    }
    return ref;
}

struct DevInfo {
    int ordinal = -1;
    std::string name, arch;
    int warp = 0;
};
DevInfo dev_info(int dev) {
    DevInfo d;
    d.ordinal = dev;
    cudaDeviceProp p;
    ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    d.name = p.name;
    d.warp = p.warpSize;
#if defined(STRATA_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    d.arch = p.gcnArchName;
#else
    d.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
#endif
    return d;
}

// One device's run of one format pair in one mode, every phase's output brought back to the host
struct DevRun {
    std::vector<uint8_t> xq;        // (1) the q8_1 blocks of x
    std::vector<float> gate_up;     // (2) gate then up floats, CAP x FF each (phase 3)
    std::vector<uint8_t> hq;        // (3) the q8_1 blocks of h (phase 1)
    std::vector<float> down_ref_hq; // (4) the down projection of the REFERENCE hq (phase 2)
    std::vector<float> out;         // (5) the whole pipeline (phase 0)
};

DevRun run_device(int dev, int mode, const k::NativeExpertLayout& L, const std::vector<std::vector<uint8_t>>& blobs,
                  const std::vector<float>& x, const std::vector<uint8_t>* hq_ref) {
    ck(cudaSetDevice(dev), "cudaSetDevice");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "cudaStreamCreate");
    const size_t slot = (L.bytes + 255) & ~(size_t) 255;
    const size_t xq_bytes = (size_t) NT * (H / 32) * Q8_1_BYTES;
    const size_t scr_bytes = k::native_expert_scratch_bytes(CAP, FF);
    const size_t fa = ((size_t) CAP * FF * sizeof(float) + 255) & ~(size_t) 255;   // native_expert_grouped's scratch plan
    const size_t hq_bytes = (size_t) CAP * (FF / 32) * Q8_1_BYTES;
    uint8_t *dblobs = nullptr, *dxq = nullptr, *dscr = nullptr;
    float *dx = nullptr, *dout = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    ck(cudaMalloc((void**) &dblobs, slot * G), "malloc blobs");
    ck(cudaMalloc((void**) &dx, x.size() * sizeof(float)), "malloc x");
    ck(cudaMalloc((void**) &dxq, xq_bytes), "malloc xq");
    ck(cudaMalloc((void**) &dscr, scr_bytes), "malloc scratch");
    ck(cudaMalloc((void**) &dout, (size_t) CAP * H * sizeof(float)), "malloc out");
    ck(cudaMalloc((void**) &dptr, G * sizeof(unsigned long long)), "malloc ptr");
    ck(cudaMalloc((void**) &dstart, (G + 1) * sizeof(int32_t)), "malloc start");
    ck(cudaMalloc((void**) &dn, sizeof(int32_t)), "malloc n");
    ck(cudaMalloc((void**) &ddst, CAP * sizeof(int32_t)), "malloc dst");
    ck(cudaMalloc((void**) &dtok, CAP * sizeof(int32_t)), "malloc tok");
    std::vector<unsigned long long> ptr(G);
    std::vector<int32_t> start(G + 1), dst(CAP), tok(CAP);
    for (int g = 0; g < G; ++g) {
        ck(cudaMemcpy(dblobs + slot * g, blobs[(size_t) g].data(), blobs[(size_t) g].size(), cudaMemcpyHostToDevice), "blob");
        ptr[(size_t) g] = (unsigned long long) (dblobs + slot * g);
        start[(size_t) g] = g * NT;
        for (int t = 0; t < NT; ++t) { dst[(size_t) (g * NT + t)] = g * NT + t; tok[(size_t) (g * NT + t)] = t; }
    }
    start[G] = CAP;
    const int32_t ng = G;
    ck(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dn, &ng, 4, cudaMemcpyHostToDevice), "n");
    ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");

    DevRun r;
    auto grouped = [&](int phase) {
        k::native_expert_set_mode(mode, phase);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, CAP, dxq, dscr, dout, s);
        k::native_expert_set_mode(-1, 0);
        ck(cudaGetLastError(), "native_expert_grouped launch");
        ck(cudaStreamSynchronize(s), "native_expert_grouped");
    };
    auto clear = [&] {   // (synchronous: the plain cudaMemcpy's below do not wait for stream s)
        ck(cudaMemsetAsync(dscr, 0, scr_bytes, s), "memset scratch");
        ck(cudaMemsetAsync(dout, 0, (size_t) CAP * H * sizeof(float), s), "memset out");
        ck(cudaStreamSynchronize(s), "memset");
    };
    // (1) the q8_1 activations
    k::quantize_q8_1_rows(dx, NT, H, dxq, s);
    ck(cudaStreamSynchronize(s), "quantize_q8_1_rows");
    r.xq.resize(xq_bytes);
    ck(cudaMemcpy(r.xq.data(), dxq, xq_bytes, cudaMemcpyDeviceToHost), "xq");
    // (2) gate/up only: the floats in the scratch's first two thirds
    clear();
    grouped(3);
    r.gate_up.resize((size_t) 2 * CAP * FF);
    ck(cudaMemcpy(r.gate_up.data(), dscr, (size_t) CAP * FF * sizeof(float), cudaMemcpyDeviceToHost), "gate");
    ck(cudaMemcpy(r.gate_up.data() + (size_t) CAP * FF, dscr + fa, (size_t) CAP * FF * sizeof(float), cudaMemcpyDeviceToHost), "up");
    // (3) gate/up + SwiGLU + q8_1: the hq blocks at 3 fa
    clear();
    grouped(1);
    r.hq.resize(hq_bytes);
    ck(cudaMemcpy(r.hq.data(), dscr + 3 * fa, hq_bytes, cudaMemcpyDeviceToHost), "hq");
    // (4) the down projection of the reference hq (device A's mode-0 blocks; its own on device A itself)
    clear();
    const std::vector<uint8_t>& hq_in = hq_ref != nullptr ? *hq_ref : r.hq;
    ck(cudaMemcpy(dscr + 3 * fa, hq_in.data(), hq_bytes, cudaMemcpyHostToDevice), "hq ref");
    grouped(2);
    r.down_ref_hq.resize((size_t) CAP * H);
    ck(cudaMemcpy(r.down_ref_hq.data(), dout, r.down_ref_hq.size() * sizeof(float), cudaMemcpyDeviceToHost), "down");
    // (5) the whole pipeline
    clear();
    grouped(0);
    r.out.resize((size_t) CAP * H);
    ck(cudaMemcpy(r.out.data(), dout, r.out.size() * sizeof(float), cudaMemcpyDeviceToHost), "out");

    cudaFree(dblobs); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout);
    cudaFree(dptr); cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    ck(cudaStreamDestroy(s), "cudaStreamDestroy");
    return r;
}

// (3): the q8_1 blocks of two devices, code by code
struct HqDiff {
    size_t codes = 0, codes_differ = 0, max_delta = 0, blocks = 0, ds_differ = 0;
    double differ_pct() const { return codes ? 100.0 * (double) codes_differ / (double) codes : 0.0; }
    bool within_expectation() const { return max_delta <= 1 && differ_pct() <= 0.1; }
};
HqDiff hq_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    HqDiff d;
    if (a.size() != b.size()) { d.codes_differ = d.max_delta = (size_t) -1; return d; }
    d.blocks = a.size() / Q8_1_BYTES;
    d.codes = d.blocks * 32;
    for (size_t blk = 0; blk < d.blocks; ++blk) {
        const uint8_t* pa = a.data() + blk * Q8_1_BYTES;
        const uint8_t* pb = b.data() + blk * Q8_1_BYTES;
        if (std::memcmp(pa, pb, 4) != 0) ++d.ds_differ;
        for (size_t i = 4; i < Q8_1_BYTES; ++i) {
            const int ca = (int8_t) pa[i], cb = (int8_t) pb[i];
            if (ca != cb) {
                ++d.codes_differ;
                d.max_delta = (std::max)(d.max_delta, (size_t) std::abs(ca - cb));
            }
        }
    }
    return d;
}

struct Totals { int failures = 0, strict_failures = 0; };

// the comparisons of two runs (one device each) in the same or in different modes
void compare_runs(const char* tag, const DevRun& A, const DevRun& B, bool same_mode, Totals& T) {
    std::printf("  %s\n", tag);
    const bool xq_same = bitwise(A.xq, B.xq);
    std::printf("    (1) q8_1 of x:                 %s\n", yn(xq_same));
    const bool gu_same = bitwise(A.gate_up, B.gate_up);
    std::printf("    (2) gate/up floats (phase 3):  %s  max |diff| %.3e  rel L1 %.3e\n", yn(gu_same), max_abs(A.gate_up, B.gate_up),
                rel_l1(A.gate_up, B.gate_up));
    const HqDiff hd = hq_diff(A.hq, B.hq);
    std::printf("    (3) q8_1 of h (phase 1):       %zu of %zu codes differ (%.4f %%), max |delta| %zu, %zu of %zu blocks' d/s differ  %s\n",
                hd.codes_differ, hd.codes, hd.differ_pct(), hd.max_delta, hd.ds_differ, hd.blocks,
                hd.codes_differ == 0 ? "(bitwise equal)" : hd.within_expectation() ? "(within +-1 / 0.1 %)" : "(BEYOND +-1 / 0.1 %)");
    const bool dn_same = bitwise(A.down_ref_hq, B.down_ref_hq);
    std::printf("    (4) down of identical hq (ph 2): %s  max |diff| %.3e  rel L1 %.3e\n", yn(dn_same), max_abs(A.down_ref_hq, B.down_ref_hq),
                rel_l1(A.down_ref_hq, B.down_ref_hq));
    const bool out_same = bitwise(A.out, B.out);
    const double out_rel = rel_l1(A.out, B.out);
    std::printf("    (5) whole pipeline (phase 0):  %s  max |diff| %.3e  rel L1 %.3e %s\n", yn(out_same), max_abs(A.out, B.out), out_rel,
                out_rel <= 1e-4 ? "(<= 1e-4)" : "(> 1e-4)");
    if (same_mode) {
        // the same kernels on both: (1), (2) and (4) are expected bitwise, (3) within +-1 / 0.1 %, (5) within 1e-4
        if (!xq_same || !gu_same || !dn_same || !hd.within_expectation() || !(out_rel <= 1e-4)) ++T.strict_failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int devA = 0, devB = 1;
    bool strict = false;
    std::vector<std::string> pairs;
    {
        std::vector<std::string> args(argv + 1, argv + argc);
        std::vector<std::string> rest;
        for (const auto& a : args) {
            if (a == "--strict") strict = true;
            else if (a == "--help" || a == "-h") {
                std::printf("usage: cross_device_grouped_parity [devA devB] [GU/DOWN ...] [--strict]\n");
                return 2;
            } else rest.push_back(a);
        }
        auto is_int = [](const std::string& s) { return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }); };
        size_t i = 0;
        if (rest.size() >= 2 && is_int(rest[0]) && is_int(rest[1])) { devA = std::atoi(rest[0].c_str()); devB = std::atoi(rest[1].c_str()); i = 2; }
        for (; i < rest.size(); ++i) pairs.push_back(rest[i]);
    }
    if (pairs.empty()) pairs = {"q4_K/q5_1", "iq3_s/iq4_nl"};

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::printf("cross_device_grouped_parity: %d device(s) visible; the test needs two (skipped)\n", count);
        return 77;
    }
    if (devA < 0 || devB < 0 || devA >= count || devB >= count || devA == devB) {
        std::printf("cross_device_grouped_parity: devices %d and %d must be two different ordinals below %d\n", devA, devB, count);
        return 2;
    }
    // STRATA_EXPERT_V2K: the S27 path ignores the bench phases this test relies on; off unless the caller asks for it
    // (native_expert_grouped reads the variable once, so it is set before the first call)
    if (std::getenv("STRATA_EXPERT_V2K") == nullptr) {
#ifdef _WIN32
        _putenv_s("STRATA_EXPERT_V2K", "0");
#else
        setenv("STRATA_EXPERT_V2K", "0", 1);
#endif
    }
    if (const char* m = std::getenv("STRATA_EXP_MODE")) {
        std::printf("note: STRATA_EXP_MODE=%s is set: the \"device default\" runs below take that mode on both devices "
                    "(and exit with a message if the device cannot run it)\n", m);
    }
    const DevInfo A = dev_info(devA), B = dev_info(devB);
    std::printf("device A: ordinal %d  %s  (%s, warpSize %d)\n", A.ordinal, A.name.c_str(), A.arch.c_str(), A.warp);
    std::printf("device B: ordinal %d  %s  (%s, warpSize %d)\n", B.ordinal, B.name.c_str(), B.arch.c_str(), B.warp);
    if (A.warp == B.warp)
        std::printf("note: both devices have warpSize %d - the wave32 vs wave64 question is not exercised by this pair\n", A.warp);
    ggml_cpu_init();   // the fp16 tables to_float uses (native_fmt below does it as well)

    Totals T;
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gu = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
        const int dn = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
        if (gu >= (int) GGML_TYPE_COUNT || dn >= (int) GGML_TYPE_COUNT) {
            std::printf("%s: skipped (this ggml has no trellis type traits)\n", pair.c_str());
            continue;
        }
        cpu::NativeFmt f;
        std::string err;
        if (gu < 0 || dn < 0 || !cpu::native_fmt(gu, dn, H, FF, f, err) || !k::native_expert_supported(gu, dn, H, FF)) {
            std::printf("%s: %s\n", pair.c_str(), err.empty() ? "not a pair the GPU expert kernels take" : err.c_str());
            ++T.failures;
            continue;
        }
        const k::NativeExpertLayout L = k::native_expert_layout(gu, dn, H, FF);
        if (L.bytes != f.bytes || L.up_off != f.up_off || L.down_off != f.down_off) {
            std::printf("%s: GPU layout (%zu bytes) and CPU layout (%zu bytes) disagree\n", pair.c_str(), L.bytes, f.bytes);
            ++T.failures;
            continue;
        }
        std::printf("\n%s/%s: %d experts x %d tokens (%d entries), n_embd %lld, n_ff %lld, blob %zu bytes\n",
                    ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) dn), G, NT, CAP, (long long) H, (long long) FF, f.bytes);
        std::vector<std::vector<uint8_t>> blobs;
        for (int g = 0; g < G; ++g) blobs.push_back(synthetic_blob(f, 101 + g));
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NT * H);
        for (auto& v : x) v = nd(rng);
        const std::vector<float> ref = cpu_reference(f, blobs, x);

        // mode 0 on both (the same kernels), then each device's own default policy
        const DevRun A0 = run_device(devA, 0, L, blobs, x, nullptr);
        const DevRun B0 = run_device(devB, 0, L, blobs, x, &A0.hq);
        const DevRun Ad = run_device(devA, -1, L, blobs, x, &A0.hq);
        const DevRun Bd = run_device(devB, -1, L, blobs, x, &A0.hq);
        compare_runs("A mode 0 vs B mode 0 (identical kernels; (1), (2), (4) expected bitwise)", A0, B0, true, T);
        compare_runs("A default vs B default (wave32: the card's default; wave64: mode 7 - different kernels, reported only)", Ad, Bd, false, T);
        compare_runs("A mode 0 vs B default (reported only)", A0, Bd, false, T);
        compare_runs("A mode 0 vs A default (the same card: bitwise unless its default differs from 0)", A0, Ad, false, T);
        compare_runs("B mode 0 vs B default (the same card: mode 7's kernels vs the CUDA layout's)", B0, Bd, false, T);

        // each device against the CPU float reference: the exit-code criterion (native_expert_parity's 3e-2)
        const struct { const char* who; const DevRun* r; } runs[4] = {{"A mode 0", &A0}, {"B mode 0", &B0}, {"A default", &Ad}, {"B default", &Bd}};
        std::printf("  vs the CPU float reference (rel L1 < 3e-2):\n");
        for (const auto& run : runs) {
            const double e = rel_l1(run.r->out, ref);
            const bool ok = e < 3e-2 && std::isfinite(e) && finite(run.r->out);
            std::printf("    %-10s gpu rel %.3e  %s\n", run.who, e, ok ? "ok" : "FAIL");
            if (!ok) ++T.failures;
        }
    }
    std::printf("\ncross_device_grouped_parity: %d CPU-reference failures, %d same-mode cross-device differences%s\n", T.failures,
                T.strict_failures, strict ? " (--strict: counted as failures)" : " (reported; --strict counts them)");
    return (T.failures || (strict && T.strict_failures)) ? 1 : 0;
}
