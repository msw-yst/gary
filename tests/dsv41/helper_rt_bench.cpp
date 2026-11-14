// Strata (docs/dsv41/cards/HX-06.md). The helper's round trip for ONE request, as the decode lane sees
// it: host issue -> copies in -> gg::win_experts (q8 x, gate_up, q8 h, down, combine) -> result on the host -> host wakes.
// One device, DeepSeek-V4.1-Flash geometry and the pack's per-layer type mix (as helper_pool_bench), T tokens, X helper
// experts a request, a pool of cold blobs per layer, the 40 layers walked in order.  The lane's own work is a busy wait
// of --think-us between issue and wait (0: the whole round trip is exposed).
//
// Mechanism (each one of ds4_helper.cu's DS4_HELPER_* switches):
//   --in   copy2  : plan H2D + x H2D (kMaxDim row stride), two SDMA copies (the engine today)
//          copy1  : plan and x packed in one pinned buffer, one H2D copy
//          kfetch : a kernel copies the packed plan + x from fine-grained pinned host memory to the device (no SDMA)
//          zc     : the expert kernels read plan and x from fine-grained pinned host memory (no copy at all)
//   --out  d2h    : out D2H copy (SDMA)        zc: the combine kernel writes straight into fine-grained pinned host memory
//   --wait sync   : hipEventSynchronize        query: spin on hipEventQuery      flag: a 1-thread kernel writes a sequence
//                   number into pinned host memory after the last step, the host spins on it (no event)
//   --graph 1     : each (layer, T)'s sequence captured once into a hipGraph and replayed (a hipGraphLaunch a request)
//   --sched auto|spin|yield|block : hipSetDeviceFlags before anything
//   --ablate none|noplan|nox|nokern|noout : leave one stage out (timing only)  --empty: only the event (the floor)
//   --trace 1     : timing events between the stages (copy2/copy1 + d2h): GPU-side per-stage means
//   --check 1     : compare every variant's out against copy2+d2h+sync bitwise (same inputs)
//   helper_rt_bench [--dev 0] [--in copy2] [--out d2h] [--wait sync] [--graph 0] [--experts 2] [--T 1] [--think-us 0]
//                   [--reps 2000] [--pool 6] [--sched auto] [--ablate none] [--trace 0] [--sweep 1] [--idle-us 0]
// --idle-us N: the device idles N us before each request (untimed), as between the engine's layers (~800 us).
// --sweep 1 runs the standard variant list on this device and prints one line each (the report's table).
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_kernels.cuh"   // kWin, kPicks, kMaxDim

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace iq = gg::iq;
namespace x3 = gg::x3;
using strata::ds4::kMaxDim;
using strata::ds4::kPicks;
using strata::ds4::kWin;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

constexpr int E = 5120, FF = 2304, K = 6, NL = 40;
static const int kSig[NL][2] = {   // (gate/up, down) types by layer, pack-2.25 (as helper_pool_bench)
    {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 22}, {17, 17},
    {16, 22}, {16, 17}, {17, 17}, {16, 16}, {16, 17}, {16, 16}, {16, 22}, {22, 17}, {16, 17}, {16, 16},
    {16, 16}, {16, 22}, {16, 22}, {17, 18}, {17, 22}, {22, 22}, {17, 17}, {17, 17}, {16, 17}, {16, 17},
    {22, 16}, {17, 16}, {16, 22}, {16, 16}, {16, 17}, {16, 16}, {16, 16}, {16, 16}, {17, 22}, {16, 22}};

struct Plan {   // as ds4_helper.cu's
    unsigned long long plan_ptr[kWin * kPicks];
    float plan_w[kWin * kPicks];
    unsigned long long grp_ptr[kWin * kPicks];
    int grp_start[kWin * kPicks + 1];
    int grp_ent[kWin * kPicks];
    int n_groups;
};
constexpr size_t kPlanB = (sizeof(Plan) + 255) / 256 * 256;   // packed: [Plan | x rows kMaxDim apart]

struct LayerB { x3::ExpertLayout lay; size_t blob = 0; std::vector<uint8_t*> ex; };

__global__ void fetch_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
__global__ void flag_kernel(volatile unsigned long long* f, unsigned long long v) {
    if (threadIdx.x == 0) { __threadfence_system(); *f = v; __threadfence_system(); }
}

struct Cfg {
    std::string in = "copy2", out = "d2h", wait = "sync", ablate = "none";
    bool graph = false, trace = false, empty = false;
    int T = 1, n_ex = 2, think = 0, reps = 2000, idle = 0;   // idle: the device left idle this long before each issue (untimed)
};

struct Rig {
    hipStream_t st;
    hipEvent_t ev, tev[6];
    uint8_t *h_pk = nullptr, *d_pk = nullptr;   // packed [Plan | x], pinned (fine-grained) / device
    Plan* h_plan2 = nullptr;                    // copy2's separate pinned plan (portable, as the engine's)
    float* h_x2 = nullptr;                      // copy2's x source (pinned, as the engine's Req)
    Plan* d_plan2 = nullptr;
    float* d_x2 = nullptr;
    float *d_h, *d_part, *d_out, *h_out_c, *h_out_zc;
    volatile unsigned long long* h_flag;
    unsigned long long seq = 0;
    std::map<long long, hipGraphExec_t> graphs;
};

static void open_rig(Rig& R) {
    CK(hipStreamCreateWithFlags(&R.st, hipStreamNonBlocking));
    CK(hipEventCreateWithFlags(&R.ev, hipEventDisableTiming));
    for (auto& e : R.tev) CK(hipEventCreate(&e));
    if (!gg::prepare(R.st)) { std::printf("prepare failed\n"); std::exit(2); }
    const size_t pk = kPlanB + (size_t) kWin * kMaxDim * 4;
    CK(hipHostMalloc((void**) &R.h_pk, pk, hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped));
    CK(hipMalloc((void**) &R.d_pk, pk));
    CK(hipHostMalloc((void**) &R.h_plan2, sizeof(Plan), hipHostMallocPortable));
    CK(hipHostMalloc((void**) &R.h_x2, (size_t) kWin * kMaxDim * 4, hipHostMallocPortable | hipHostMallocMapped));
    CK(hipMalloc((void**) &R.d_plan2, sizeof(Plan)));
    CK(hipMalloc((void**) &R.d_x2, (size_t) kWin * kMaxDim * 4));
    CK(hipMalloc((void**) &R.d_h, (size_t) kWin * kPicks * FF * 4));
    CK(hipMalloc((void**) &R.d_part, (size_t) kWin * kPicks * E * 4));
    CK(hipMalloc((void**) &R.d_out, (size_t) kWin * E * 4));
    CK(hipHostMalloc((void**) &R.h_out_c, (size_t) kWin * E * 4, hipHostMallocPortable));
    CK(hipHostMalloc((void**) &R.h_out_zc, (size_t) kWin * E * 4, hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped));
    void* f; CK(hipHostMalloc(&f, 64, hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped));
    R.h_flag = (volatile unsigned long long*) f; *R.h_flag = 0;
}

// the host side of a request: the plan into P, x into xdst (T rows kMaxDim apart)
static void make_plan(Plan& P, const LayerB& L, const std::vector<std::pair<int, int>>& picks /* (expert, t*8+i) */, const float* w) {
    std::memset(P.plan_ptr, 0, sizeof P.plan_ptr);
    std::memset(P.plan_w, 0, sizeof P.plan_w);
    std::vector<std::pair<int, int>> by = picks;
    std::sort(by.begin(), by.end());
    for (auto& p : by) { P.plan_ptr[p.second] = (unsigned long long) L.ex[(size_t) p.first]; P.plan_w[p.second] = w[p.second]; }
    int ng = 0, ne = 0;
    for (size_t a = 0; a < by.size();) {
        const int e = by[a].first;
        P.grp_ptr[ng] = (unsigned long long) L.ex[(size_t) e];
        P.grp_start[ng] = ne;
        while (a < by.size() && by[a].first == e) P.grp_ent[ne++] = by[a++].second;
        ++ng;
    }
    P.grp_start[ng] = ne;
    P.n_groups = ng;
}

// issue one request (everything after the host-side plan): returns false on refusal
static void issue(Rig& R, const Cfg& c, const LayerB& L, int li, bool capture_ok = true) {
    const int T = c.T;
    const bool packed = c.in != "copy2";
    Plan* dp = c.in == "copy2" ? R.d_plan2 : c.in == "zc" ? (Plan*) R.h_pk : (Plan*) R.d_pk;
    const float* dx = c.in == "copy2" ? R.d_x2 : c.in == "zc" ? (const float*) (R.h_pk + kPlanB) : (const float*) (R.d_pk + kPlanB);
    float* dout = c.out == "zc" ? R.h_out_zc : R.d_out;
    if (c.empty) { CK(hipEventRecord(R.ev, R.st)); return; }
    if (c.trace) CK(hipEventRecord(R.tev[0], R.st));
    if (c.in == "copy2") {
        if (c.ablate != "noplan") CK(hipMemcpyAsync(R.d_plan2, R.h_plan2, sizeof(Plan), hipMemcpyHostToDevice, R.st));
        if (c.trace) CK(hipEventRecord(R.tev[1], R.st));
        if (c.ablate != "nox") CK(hipMemcpyAsync(R.d_x2, R.h_x2, (size_t) T * kMaxDim * 4, hipMemcpyHostToDevice, R.st));
    } else if (c.in == "copy1") {
        if (c.ablate != "noplan") CK(hipMemcpyAsync(R.d_pk, R.h_pk, kPlanB + (size_t) ((T - 1) * kMaxDim + E) * 4, hipMemcpyHostToDevice, R.st));
        if (c.trace) CK(hipEventRecord(R.tev[1], R.st));
    } else if (c.in == "kfetch") {
        const size_t n16 = (kPlanB + (size_t) ((T - 1) * kMaxDim + E) * 4) / 16;
        if (c.ablate != "noplan") hipLaunchKernelGGL(fetch_kernel, dim3(8), dim3(256), 0, R.st, (const uint4*) R.h_pk, (uint4*) R.d_pk, n16);
        if (c.trace) CK(hipEventRecord(R.tev[1], R.st));
    }
    (void) packed;
    if (c.trace) CK(hipEventRecord(R.tev[2], R.st));
    if (c.ablate != "nokern") {
        x3::WinArgs a;
        a.grp_ptr = dp->grp_ptr; a.grp_start = dp->grp_start; a.grp_ent = dp->grp_ent; a.n_groups = &dp->n_groups;
        a.plan_ptr = dp->plan_ptr; a.plan_w = dp->plan_w;
        a.cpu_part = nullptr; a.cpu_flag = nullptr; a.sh_out = nullptr;
        a.x = dx; a.x_stride = kMaxDim;
        a.h = R.d_h; a.part = R.d_part;
        a.out = dout; a.out_stride = E;
        a.lay = L.lay;
        a.T = T; a.k = K; a.n_embd = E; a.n_ff = FF; a.limit = 10.0f;
        if (!gg::win_experts(a, R.st)) { std::printf("win_experts refused (layer %d)\n", li); std::exit(2); }
    }
    if (c.trace) CK(hipEventRecord(R.tev[3], R.st));
    if (c.out == "d2h" && c.ablate != "noout") CK(hipMemcpyAsync(R.h_out_c, R.d_out, (size_t) T * E * 4, hipMemcpyDeviceToHost, R.st));
    if (c.trace) CK(hipEventRecord(R.tev[4], R.st));
    (void) capture_ok;
}

static void finish_wait(Rig& R, const Cfg& c) {
    if (c.wait == "flag") {
        while (*R.h_flag != R.seq) {}
    } else if (c.wait == "query") {
        hipError_t e;
        while ((e = hipEventQuery(R.ev)) == hipErrorNotReady) {}
        CK(e);
    } else {
        CK(hipEventSynchronize(R.ev));
    }
}

struct Res { double rt_p50, rt_mean, rt_p90, issue_mean, wait_mean, wait_p50; double stage[4]; bool same; };

static Res run(Rig& R, const Cfg& c, std::vector<LayerB>& Ls, std::mt19937_64& rng, const std::vector<float>& xin, std::vector<float>* outs,
               const std::vector<std::vector<int>>& pick_seq) {
    for (auto& kv : R.graphs) CK(hipGraphExecDestroy(kv.second));   // (a graph captures this variant's mechanism only)
    R.graphs.clear();
    std::vector<double> rt, is, wt;
    double stage[4] = {0, 0, 0, 0};
    int nst = 0;
    const int warm = 80;
    for (int r = 0; r < c.reps + warm; ++r) {
        const int li = r % NL;
        const LayerB& L = Ls[(size_t) li];
        const std::vector<int>& ex = pick_seq[(size_t) r % pick_seq.size()];
        std::vector<std::pair<int, int>> picks;
        float w[kWin * kPicks] = {};
        for (int t = 0; t < c.T; ++t)
            for (int j = 0; j < c.n_ex; ++j) { picks.push_back({ex[(size_t) (t + j) % ex.size()], t * kPicks + j}); w[t * kPicks + j] = 0.1f + 0.05f * j; }
        if (c.idle > 0) { const double ti = now_us(); while (now_us() - ti < c.idle) {} }
        double t0 = now_us();
        // host side: the plan and x where this mechanism's copy / kernel reads them
        if (c.in == "copy2") {
            make_plan(*R.h_plan2, L, picks, w);
            for (int t = 0; t < c.T; ++t) std::memcpy(R.h_x2 + (size_t) t * kMaxDim, xin.data() + (size_t) t * E, E * 4);   // (the engine: already in the Req)
        } else {
            make_plan(*(Plan*) R.h_pk, L, picks, w);
            for (int t = 0; t < c.T; ++t) std::memcpy(R.h_pk + kPlanB + (size_t) t * kMaxDim * 4, xin.data() + (size_t) t * E, E * 4);
        }
        if (c.ablate == "noplan") {   // (the plan must still be this layer's: uploaded outside the timed window)
            if (c.in == "copy2") CK(hipMemcpy(R.d_plan2, R.h_plan2, sizeof(Plan), hipMemcpyHostToDevice));
            else CK(hipMemcpy(R.d_pk, R.h_pk, kPlanB + (size_t) E * 4, hipMemcpyHostToDevice));
            t0 = now_us();
        }
        ++R.seq;
        if (c.graph) {
            const long long key = (long long) li * 64 + c.T;
            auto it = R.graphs.find(key);
            if (it == R.graphs.end()) {
                hipGraph_t g;
                CK(hipStreamBeginCapture(R.st, hipStreamCaptureModeRelaxed));
                Cfg cc = c; cc.trace = false;
                issue(R, cc, L, li);
                CK(hipStreamEndCapture(R.st, &g));
                hipGraphExec_t ge;
                CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                CK(hipGraphDestroy(g));
                it = R.graphs.emplace(key, ge).first;
            }
            CK(hipGraphLaunch(it->second, R.st));
        } else {
            issue(R, c, L, li);
        }
        if (c.wait == "flag") hipLaunchKernelGGL(flag_kernel, dim3(1), dim3(64), 0, R.st, R.h_flag, R.seq);
        else CK(hipEventRecord(R.ev, R.st));
        const double t1 = now_us();
        if (c.think > 0) while (now_us() - t1 < c.think) {}
        const double t2 = now_us();
        finish_wait(R, c);
        const double t3 = now_us();
        if (r >= warm) {
            rt.push_back(t3 - t0 - (t2 - t1));
            is.push_back(t1 - t0);
            wt.push_back(t3 - t2);
            if (c.trace && !c.graph) {
                float ms[4];
                for (int s = 0; s < 4; ++s) CK(hipEventElapsedTime(&ms[s], R.tev[s], R.tev[s + 1]));
                for (int s = 0; s < 4; ++s) stage[s] += ms[s] * 1000.0;
                ++nst;
            }
        }
        if (outs != nullptr && r >= warm && r < warm + 80) {
            const float* o = c.out == "zc" ? R.h_out_zc : R.h_out_c;
            outs->insert(outs->end(), o, o + (size_t) c.T * E);
        }
    }
    auto med = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; };
    auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / std::max<size_t>(1, v.size()); };
    Res res{};
    res.rt_p50 = med(rt, 0.5); res.rt_p90 = med(rt, 0.9); res.rt_mean = mean(rt); res.issue_mean = mean(is); res.wait_mean = mean(wt); res.wait_p50 = med(wt, 0.5);
    for (int s = 0; s < 4; ++s) res.stage[s] = nst ? stage[s] / nst : 0;
    return res;
}

int main(int argc, char** argv) {
    Cfg c;
    int dev = 0, pool = 6, sweep = 0;
    std::string sched = "auto";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string("0"); };
        if (a == "--dev") dev = std::stoi(nx());
        else if (a == "--in") c.in = nx();
        else if (a == "--out") c.out = nx();
        else if (a == "--wait") c.wait = nx();
        else if (a == "--graph") c.graph = nx() == "1";
        else if (a == "--experts") c.n_ex = std::stoi(nx());
        else if (a == "--T") c.T = std::stoi(nx());
        else if (a == "--think-us") c.think = std::stoi(nx());
        else if (a == "--idle-us") c.idle = std::stoi(nx());
        else if (a == "--reps") c.reps = std::stoi(nx());
        else if (a == "--pool") pool = std::stoi(nx());
        else if (a == "--sched") sched = nx();
        else if (a == "--ablate") c.ablate = nx();
        else if (a == "--trace") c.trace = nx() == "1";
        else if (a == "--empty") c.empty = nx() == "1";
        else if (a == "--sweep") sweep = std::stoi(nx());
        else { std::printf("unknown %s\n", a.c_str()); return 2; }
    }
    if (sched == "spin") CK(hipSetDeviceFlags(hipDeviceScheduleSpin));
    else if (sched == "yield") CK(hipSetDeviceFlags(hipDeviceScheduleYield));
    else if (sched == "block") CK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));
    CK(hipSetDevice(dev));
    hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, dev));
    std::printf("device %d: %s  sched %s  T %d  experts/request %d  think %d us  HSA_ENABLE_SDMA=%s\n", dev, p.gcnArchName, sched.c_str(), c.T,
                c.n_ex, c.think, std::getenv("HSA_ENABLE_SDMA") ? std::getenv("HSA_ENABLE_SDMA") : "(unset)");
    std::vector<LayerB> Ls(NL);
    std::mt19937_64 rng(5);
    size_t total = 0;
    for (int l = 0; l < NL; ++l) {
        const int tg = kSig[l][0], td = kSig[l][1];
        const size_t rg = iq::row_bytes(tg, E), rd = iq::row_bytes(td, FF);
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        LayerB& L = Ls[(size_t) l];
        L.lay.fmt = gg::kFmtMXFP4;
        L.lay.K[0] = L.lay.K[1] = tg; L.lay.K[2] = td;
        L.lay.trellis[0] = 0; L.lay.trellis[1] = al(FF * rg); L.lay.trellis[2] = 2 * al(FF * rg);
        L.blob = L.lay.trellis[2] + al((size_t) E * rd);
        std::vector<uint8_t> h(L.blob);
        for (int e = 0; e < pool; ++e) {
            for (size_t i = 0; i < h.size(); i += 8) { const uint64_t r = rng(); std::memcpy(&h[i], &r, std::min<size_t>(8, h.size() - i)); }
            std::uniform_real_distribution<float> sd(0.002f, 0.02f);
            for (int r = 0; r < 3; ++r) {
                const int ty = L.lay.K[r], bb = iq::block_bytes(ty);
                const size_t n = r < 2 ? FF * rg : (size_t) E * rd;
                for (size_t b = 0; b < n; b += (size_t) bb) { const uint16_t d = f2h(sd(rng)); std::memcpy(&h[L.lay.trellis[r] + b], &d, 2); }
            }
            uint8_t* d; CK(hipMalloc(&d, L.blob)); CK(hipMemcpy(d, h.data(), L.blob, hipMemcpyHostToDevice));
            L.ex.push_back(d);
            total += L.blob;
        }
    }
    std::printf("%d layers x %d experts = %.2f GiB of blobs\n", NL, pool, total / 1073741824.0);
    std::vector<float> xin((size_t) kWin * E);
    { std::normal_distribution<float> nd(0.f, 1.f); for (float& v : xin) v = nd(rng); }
    std::vector<std::vector<int>> pick_seq(997);
    for (auto& v : pick_seq) {   // distinct experts a request, from the layer's pool
        std::vector<int> all(pool); for (int e = 0; e < pool; ++e) all[e] = e;
        std::shuffle(all.begin(), all.end(), rng);
        v.assign(all.begin(), all.begin() + std::min(pool, K));
    }
    Rig R; open_rig(R);
    auto line = [&](const char* name, const Cfg& cc, const std::vector<float>* ref, std::vector<float>* got) {
        std::vector<float> o;
        Res r = run(R, cc, Ls, rng, xin, got ? got : (ref ? &o : nullptr), pick_seq);
        const std::vector<float>& g = got ? *got : o;
        std::string cmp = "";
        if (ref && !cc.empty && cc.ablate == "none") cmp = (g.size() == ref->size() && std::memcmp(g.data(), ref->data(), g.size() * 4) == 0) ? "  BITWISE" : "  DIFFERS";
        std::printf("%-28s in %-6s out %-3s wait %-5s graph %d ablate %-6s : round trip p50 %6.1f mean %6.1f p90 %6.1f us | issue %5.1f | wait mean %6.1f p50 %6.1f",
                    name, cc.in.c_str(), cc.out.c_str(), cc.wait.c_str(), (int) cc.graph, cc.ablate.c_str(), r.rt_p50, r.rt_mean, r.rt_p90, r.issue_mean,
                    r.wait_mean, r.wait_p50);
        if (cc.trace) std::printf(" | GPU: in-a %.1f in-b %.1f kernels %.1f out %.1f", r.stage[0], r.stage[1], r.stage[2], r.stage[3]);
        std::printf("%s\n", cmp.c_str());
        std::fflush(stdout);
    };
    if (!sweep) { line("single", c, nullptr, nullptr); return 0; }
    // the standard list: the decomposition (ablations, trace) then the reductions; every variant checked against the base
    std::vector<float> ref;
    Cfg b = c; b.in = "copy2"; b.out = "d2h"; b.wait = "sync"; b.graph = false; b.ablate = "none"; b.trace = false; b.empty = false;
    line("base (engine today)", b, nullptr, &ref);
    { Cfg x = b; x.trace = true; line("base + timing events", x, &ref, nullptr); }
    { Cfg x = b; x.empty = true; line("event only (floor)", x, nullptr, nullptr); }
    { Cfg x = b; x.ablate = "noplan"; line("ablate plan H2D", x, nullptr, nullptr); }
    { Cfg x = b; x.ablate = "nox"; line("ablate x H2D", x, nullptr, nullptr); }
    { Cfg x = b; x.ablate = "noout"; line("ablate out D2H", x, nullptr, nullptr); }
    { Cfg x = b; x.ablate = "nokern"; line("ablate kernels", x, nullptr, nullptr); }
    { Cfg x = b; x.in = "kfetch"; x.out = "zc"; x.ablate = "nokern"; line("no SDMA, no kernels", x, nullptr, nullptr); }
    { Cfg x = b; x.wait = "query"; line("(a) query spin", x, &ref, nullptr); }
    { Cfg x = b; x.wait = "flag"; line("(a) flag spin", x, &ref, nullptr); }
    { Cfg x = b; x.in = "copy1"; line("(b) one packed H2D", x, &ref, nullptr); }
    { Cfg x = b; x.in = "copy1"; x.trace = true; line("(b) one packed H2D + ev", x, &ref, nullptr); }
    { Cfg x = b; x.in = "kfetch"; line("(b) kernel fetch", x, &ref, nullptr); }
    { Cfg x = b; x.in = "zc"; line("(b) zero-copy in", x, &ref, nullptr); }
    { Cfg x = b; x.out = "zc"; line("(c) zero-copy out", x, &ref, nullptr); }
    { Cfg x = b; x.graph = true; line("(d) graph", x, &ref, nullptr); }
    { Cfg x = b; x.in = "kfetch"; x.out = "zc"; line("(e) no SDMA: kfetch+zc", x, &ref, nullptr); }
    { Cfg x = b; x.in = "kfetch"; x.out = "zc"; x.wait = "query"; line("(e) kfetch+zc+query", x, &ref, nullptr); }
    { Cfg x = b; x.in = "kfetch"; x.out = "zc"; x.wait = "flag"; line("(e) kfetch+zc+flag", x, &ref, nullptr); }
    { Cfg x = b; x.in = "zc"; x.out = "zc"; x.wait = "flag"; line("(e) zc+zc+flag", x, &ref, nullptr); }
    { Cfg x = b; x.in = "kfetch"; x.out = "zc"; x.wait = "flag"; x.graph = true; line("(e) kfetch+zc+flag+graph", x, &ref, nullptr); }
    { Cfg x = b; x.in = "copy1"; x.out = "zc"; x.wait = "query"; line("copy1+zc+query", x, &ref, nullptr); }
    return 0;
}
