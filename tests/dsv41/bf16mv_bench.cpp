// tests/dsv41/bf16mv_bench.cpp - (DS4_BF16MV) ds4_bf16mv.cuh's BF16 matvec kernels alone on ONE GPU: bf16_mv_kernel (the
// engine's default, "old") against bf16_mv2_kernel's variants, at the native MXFP4 pack's real shapes on the TP4 lead (member 0),
// for verify batches B = 1..8.  No model: weights and activations are generated on the device (hash of the index, fixed seed).
//
//   ds4_bf16mv_bench roof                         the card's plain streaming read (GB/s) - "% of achievable"
//   ds4_bf16mv_bench ab   [--B 1,2,..] [--vars a,b] bit for bit: every launch of one decode step (and the edge shapes) by old,
//                                                 then by each variant; "AB <var> B<b> <set> => BITWISE | DIFFER ..."
//   ds4_bf16mv_bench tm   [--B ..] [--vars ..] [--reps N]   per shape class: us a launch, GB/s ("TM ...")
//   ds4_bf16mv_bench step [--B ..] [--vars ..] [--reps N]   one decode step's 256 BF16 launches back to back ("STEP ...")
//
// The step (ds4_model.cu's decode, the native pack's tensor list and metadata; member 0 of TP4: heads 16 of 64, o-groups 2 of 8, the shared
// expert's split128 share 640 / 512 by layer, the head's 32,320 columns), in launch groups as gguf::mv receives them:
//   every layer:     attn_in [wq_a 1280x5120, wkv 512x5120 (+ compressor wkv 512x5120 on 2/8/14/20, + wgate on 2/8/14)]
//                    wq_b 8192x1280 | wo_a 2 x 1024x4096 | wo_b 5120 x 2048 (of 8192) | shared w1+w3 2 x {640|512}x5120 |
//                    shared w2 5120 x {640|512} (of 2304)
//   index layers:    indexer wq_b 4096x1280 (2/8/14/20/24/28/32/36)
//   engram layers:   engram wkv 25600x6144, whole (1, 14; T > 1 is not split by DS4_EG_SPLIT)
//   then the head:   32320x5120
// Each layer has its own weights (no reuse across the step: the card's 96 MB last-level cache cannot hold a step's ~4.8 GB).
// Exit: 0 ok | 1 a DIFFER (ab) | 2 usage / device error.
#include "ds4_gguf.cuh"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ds4::gguf {
namespace {
// (ds4_gguf.cu's three names, verbatim - ds4_bf16mv.cuh needs them declared)
constexpr int kWarps = 8, kThreads = kWarps * 32;
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
#include "ds4_bf16mv.cuh"
}  // namespace
}  // namespace strata::ds4::gguf

using namespace strata::ds4::gguf;

#define CK(x)                                                                                             \
    do {                                                                                                  \
        hipError_t e_ = (x);                                                                              \
        if (e_ != hipSuccess) {                                                                           \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            std::exit(2);                                                                                 \
        }                                                                                                 \
    } while (0)

namespace {

__device__ __forceinline__ uint32_t mix(uint64_t v) {
    v ^= v >> 33; v *= 0xff51afd7ed558ccdull; v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ull; v ^= v >> 33;
    return (uint32_t) v;
}
// BF16 weights: sign random, exponent 2^-14 .. 2^1 (most 2^-9 .. 2^-4, model-like), 1/64 +-0, 1/4096 subnormal, mantissa random
__global__ void fill_w(uint16_t* w, size_t n, uint64_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t h = mix(seed * 0x9e3779b97f4a7c15ull + i), g = mix(i * 0x632be59bd9b4e5ull + seed);
        const uint32_t sign = (h >> 31) << 15, man = h & 0x7f;
        uint32_t e;
        if ((g & 63) == 0 || (g & 4095) == 1) e = 0;   // (+-0, or a subnormal)
        else e = 127 - 4 - ((g >> 6) % 6) - (((g >> 12) & 7) == 0 ? ((g >> 15) % 6) : 0) + (((g >> 18) & 31) == 0 ? 5 : 0);
        const uint32_t m = ((g & 63) == 0) ? 0 : man;
        w[i] = (uint16_t) (sign | (e << 7) | m);
    }
}
// F32 activations: N(0, 1)-ish with a few large outliers (DeepSeek's hidden states carry massive activations)
__global__ void fill_x(float* x, size_t n, uint64_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t a = mix(seed ^ (i * 0x9e3779b97f4a7c15ull)), b = mix(seed + i * 0x2545f4914f6cdd1dull);
        float u = ((a >> 8) + 0.5f) * (1.0f / 16777216.0f), v = ((b >> 8) + 0.5f) * (1.0f / 16777216.0f);
        float z = (u + v - 1.0f) * 2.4f;
        if ((a & 1023) == 7) z *= 300.0f;
        if ((b & 255) == 3) z = -z * 1e-3f;
        x[i] = z;
    }
}
__global__ void roof_kernel(const uint4* p, size_t n, unsigned* out) {
    uint32_t a = 0;
    const size_t st = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * st < n; i += 4 * st) {
        const uint4 q0 = p[i], q1 = p[i + st], q2 = p[i + 2 * st], q3 = p[i + 3 * st];
        a ^= q0.x ^ q0.y ^ q0.z ^ q0.w ^ q1.x ^ q1.y ^ q1.z ^ q1.w ^ q2.x ^ q2.y ^ q2.z ^ q2.w ^ q3.x ^ q3.y ^ q3.z ^ q3.w;
    }
    for (; i < n; i += st) a ^= p[i].x ^ p[i].w;
    if (a == 0x12345678u) out[0] = a;
}

// ---- the launch, as gguf::mv's mv_set / mv_launch (f = 0, VEC) does it; var -1 = bf16_mv_kernel
template <int NC>
void launch_nc(int var, const BfJobs& J, int nj, int max_n, hipStream_t s) {
    if (var < 0) {
        const dim3 grid((unsigned) ((max_n + kWarps - 1) / kWarps), (unsigned) nj);
        bf16_mv_kernel<NC, true><<<grid, kThreads, 0, s>>>(J);
        return;
    }
    const Bf16mvVar& v = kBf16mvVars[var];
    const dim3 grid((unsigned) ((max_n + kWarps * v.R - 1) / (kWarps * v.R)), (unsigned) nj);
    switch (var) {
        case 0: bf16_mv2_kernel<NC, 1, 2, false><<<grid, kThreads, 0, s>>>(J); break;
        case 1: bf16_mv2_kernel<NC, 1, 4, false><<<grid, kThreads, 0, s>>>(J); break;
        case 2: bf16_mv2_kernel<NC, 1, 8, false><<<grid, kThreads, 0, s>>>(J); break;
        case 3: bf16_mv2_kernel<NC, 2, 2, false><<<grid, kThreads, 0, s>>>(J); break;
        case 4: bf16_mv2_kernel<NC, 2, 4, false><<<grid, kThreads, 0, s>>>(J); break;
        case 5: bf16_mv2_kernel<NC, 4, 2, false><<<grid, kThreads, 0, s>>>(J); break;
        case 6: bf16_mv2_kernel<NC, 1, 4, true><<<grid, kThreads, 0, s>>>(J); break;
        default: bf16_mv2_kernel<NC, 2, 4, true><<<grid, kThreads, 0, s>>>(J); break;
    }
}
static_assert(kBf16mvNVars == 8, "launch_nc lists kBf16mvVars");
void launch(int var, const BfJobs& J, int nj, int max_n, int max_c, hipStream_t s) {
    if (max_c <= 1) launch_nc<1>(var, J, nj, max_n, s);
    else if (max_c <= 2) launch_nc<2>(var, J, nj, max_n, s);
    else if (max_c <= 4) launch_nc<4>(var, J, nj, max_n, s);
    else if (max_c <= 8) launch_nc<8>(var, J, nj, max_n, s);
    else launch_nc<16>(var, J, nj, max_n, s);
}

struct WMat {   // a weight matrix view: n rows of k values, ld values apart
    uint16_t* w;
    int n, k;
    size_t ld;
};
struct Job {
    WMat m;
    int ncol;          // 0: the launch's B
    float alpha;
    bool bias;
    size_t xpad;       // activation stride = k + xpad
};
struct Launch {
    std::string cls;   // shape class (tm groups by it)
    std::vector<Job> jobs;
    size_t bytes() const {
        size_t b = 0;
        for (const Job& j : jobs) b += (size_t) j.m.n * j.m.k * 2;
        return b;
    }
};

std::vector<void*> g_allocs;
uint16_t* dev_w(size_t vals, uint64_t seed) {
    void* p = nullptr;
    CK(hipMalloc(&p, vals * 2 + 256));
    g_allocs.push_back(p);
    fill_w<<<4096, 256>>>((uint16_t*) p, vals, seed);
    CK(hipGetLastError());
    return (uint16_t*) p;
}
WMat mat(int n, int k, size_t ld, uint64_t seed) { return WMat{dev_w((size_t) n * ld, seed), n, k, ld}; }

// one decode step's BF16 launches on the TP4 lead (see the header), plus extra copies of the small classes for tm
struct Model {
    std::vector<Launch> step;            // in engine order
    std::vector<Launch> tm_extra;        // more copies of a class whose 40 layers sit inside the 96 MB cache
    std::vector<Launch> edge;            // edge shapes (ab only)
};
Model build_model() {
    Model M;
    uint64_t sd = 1;
    auto is = [](int l, std::initializer_list<int> s) { return std::find(s.begin(), s.end(), l) != s.end(); };
    for (int l = 0; l < 40; ++l) {
        const int ff = (l % 4 == 0 || l % 4 == 3) ? 640 : 512;   // split128(2304, 4, 0, rot l % 4): member 0's share
        Launch a{"attn_in", {}};
        a.jobs.push_back({mat(1280, 5120, 5120, sd++), 0, 1.0f, false, 0});
        a.jobs.push_back({mat(512, 5120, 5120, sd++), 0, 1.0f, false, 0});
        if (is(l, {2, 8, 14, 20})) a.jobs.push_back({mat(512, 5120, 5120, sd++), 0, 1.0f, false, 0});
        if (is(l, {2, 8, 14})) a.jobs.push_back({mat(512, 5120, 5120, sd++), 0, 1.0f, false, 0});
        if (a.jobs.size() > 2) a.cls = "attn_in_c";
        M.step.push_back(a);
        if (is(l, {2, 8, 14, 20, 24, 28, 32, 36})) M.step.push_back({"idx_wq_b", {{mat(4096, 1280, 1280, sd++), 0, 1.0f, false, 0}}});
        M.step.push_back({"wq_b", {{mat(8192, 1280, 1280, sd++), 0, 1.0f, false, 0}}});
        M.step.push_back({"wo_a", {{mat(1024, 4096, 4096, sd++), 0, 1.0f, false, 0}, {mat(1024, 4096, 4096, sd++), 0, 1.0f, false, 0}}});
        M.step.push_back({"wo_b", {{mat(5120, 2048, 8192, sd++), 0, 1.0f, false, 0}}});
        if (is(l, {1, 14})) M.step.push_back({"engram", {{mat(25600, 6144, 6144, sd++), 0, 1.0f, false, 0}}});
        M.step.push_back({ff == 640 ? "sh13_640" : "sh13_512",
                          {{mat(ff, 5120, 5120, sd++), 0, 1.0f, false, 0}, {mat(ff, 5120, 5120, sd++), 0, 1.0f, false, 0}}});
        M.step.push_back({ff == 640 ? "sh2_640" : "sh2_512", {{mat(5120, ff, 2304, sd++), 0, 1.0f, false, 0}}});
    }
    M.step.push_back({"head", {{mat(32320, 5120, 5120, sd++), 0, 1.0f, false, 0}}});
    for (int c = 0; c < 24; ++c) M.tm_extra.push_back({"idx_wq_b", {{mat(4096, 1280, 1280, sd++), 0, 1.0f, false, 0}}});
    // edge shapes: k tails (8, 264, 2056 - not a multiple of 256), rows not a multiple of 8 R, a single row, alpha / bias,
    // padded activation strides, jobs of different ncol and n in one launch, 16 columns
    M.edge.push_back({"e_k8", {{mat(37, 8, 8, sd++), 0, 1.0f, false, 0}}});
    M.edge.push_back({"e_k264", {{mat(61, 264, 264, sd++), 0, 0.5f, true, 8}}});
    M.edge.push_back({"e_k2056", {{mat(1, 2056, 2064, sd++), 0, 1.0f, true, 0}, {mat(77, 2056, 2056, sd++), 1, -1.5f, false, 64}}});
    M.edge.push_back({"e_mixed", {{mat(5120, 640, 2304, sd++), 1, 1.0f, false, 0}, {mat(129, 5120, 5120, sd++), 0, 0.125f, true, 0},
                                  {mat(3, 1280, 1280, sd++), 2, 1.0f, false, 0}}});
    M.edge.push_back({"e_8jobs", {}});
    for (int j = 0; j < 8; ++j) M.edge.back().jobs.push_back({mat(100 + 13 * j, 512 + 256 * j, 512 + 256 * j, sd++), 0, 1.0f, j % 3 == 0, 0});
    CK(hipDeviceSynchronize());
    return M;
}

// device buffers for activations (one per (k, xpad), 16 columns) and outputs (one region per launch-job)
struct Bufs {
    std::vector<std::pair<std::pair<int, size_t>, float*>> xs;
    float* bias = nullptr;   // 32768 values
    const float* x_for(int k, size_t xpad) {
        for (auto& p : xs)
            if (p.first.first == k && p.first.second == xpad) return p.second;
        void* q = nullptr;
        const size_t n = (size_t) 16 * (k + xpad);
        CK(hipMalloc(&q, n * 4 + 256));
        g_allocs.push_back(q);
        fill_x<<<512, 256>>>((float*) q, n, 1000003ull * k + xpad);
        CK(hipGetLastError());
        xs.push_back({{k, xpad}, (float*) q});
        return (const float*) q;
    }
};

struct Prepared {
    BfJobs J;
    int nj, max_n, max_c;
    size_t out_off[8], out_n[8];   // per job: its outputs' offset in the y arena and count (ncol x n)
};
// the launch's BfJobs for B columns into the y arena from `at` (floats); returns the arena used
size_t prep_launch(const Launch& L, int B, Bufs& bf, float* yarena, size_t at, Prepared& P) {
    std::memset(&P.J, 0, sizeof(P.J));
    P.nj = (int) L.jobs.size();
    P.max_n = 0;
    P.max_c = 1;
    for (int j = 0; j < P.nj; ++j) {
        const Job& q = L.jobs[(size_t) j];
        BfJob& b = P.J.j[j];
        const int nc = q.ncol > 0 ? std::min(q.ncol, B) : B;
        b.w = q.m.w;
        b.ld = q.m.ld;
        b.x = bf.x_for(q.m.k, q.xpad);
        b.xcs = (size_t) q.m.k + q.xpad;
        b.y = yarena + at;
        b.ycs = (size_t) q.m.n;
        b.bias = q.bias ? bf.bias : nullptr;
        b.alpha = q.alpha;
        b.k = q.m.k;
        b.n = q.m.n;
        b.ncol = nc;
        P.out_off[j] = at;
        P.out_n[j] = (size_t) nc * q.m.n;
        at += ((P.out_n[j] + 63) / 64) * 64;
        P.max_n = std::max(P.max_n, q.m.n);
        P.max_c = std::max(P.max_c, nc);
    }
    return at;
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}
std::vector<int> parse_vars(const char* s) {   // names or "all"; "old" = -1
    std::vector<int> v;
    if (std::strcmp(s, "all") == 0) {
        for (int i = 0; i < kBf16mvNVars; ++i) v.push_back(i);
        return v;
    }
    std::string t(s);
    size_t a = 0;
    while (a <= t.size()) {
        size_t b = t.find(',', a);
        if (b == std::string::npos) b = t.size();
        const std::string nm = t.substr(a, b - a);
        if (nm == "old") v.push_back(-1);
        else {
            const int i = bf16mv_parse(nm.c_str());
            if (i < 0) {
                std::fprintf(stderr, "unknown variant %s\n", nm.c_str());
                std::exit(2);
            }
            v.push_back(i);
        }
        a = b + 1;
    }
    return v;
}
const char* vname(int v) { return v < 0 ? "old" : kBf16mvVars[v].name; }
double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}
double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[std::min(v.size() - 1, (size_t) (p * (double) v.size()))];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s roof|ab|tm|step [--B 1,2,3,4,5,8] [--vars all|old,r1u4,..] [--reps N]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    std::vector<int> Bs = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> vars = parse_vars("all");
    int reps = 20;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--B")) Bs = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--vars")) vars = parse_vars(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    hipDeviceProp_t pr;
    int dev = 0;
    CK(hipGetDevice(&dev));
    CK(hipGetDeviceProperties(&pr, dev));
    std::printf("DEV %d %s %s CUs %d mem %.1f GiB L2 %d B\n", dev, pr.name, pr.gcnArchName, pr.multiProcessorCount,
                pr.totalGlobalMem / 1073741824.0, pr.l2CacheSize);
    hipStream_t s;
    CK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));

    if (mode == "roof") {
        const size_t n = (size_t) 2 << 30;   // 2 GiB
        void* p = nullptr;
        unsigned* o = nullptr;
        CK(hipMalloc(&p, n));
        CK(hipMalloc(&o, 4));
        CK(hipMemset(p, 1, n));
        std::vector<double> t;
        for (int r = 0; r < std::max(reps, 5) + 2; ++r) {
            CK(hipEventRecord(e0, s));
            roof_kernel<<<pr.multiProcessorCount * 32, 256, 0, s>>>((const uint4*) p, n / 16, o);
            CK(hipEventRecord(e1, s));
            CK(hipEventSynchronize(e1));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, e0, e1));
            if (r >= 2) t.push_back(ms);
        }
        std::printf("ROOF read 2 GiB: median %.3f ms = %.1f GB/s (best %.1f)\n", med(t), n / med(t) / 1e6, n / pct(t, 0) / 1e6);
        return 0;
    }

    Model M = build_model();
    Bufs bf;
    {
        void* q = nullptr;
        CK(hipMalloc(&q, 32768 * 4));
        g_allocs.push_back(q);
        fill_x<<<64, 256>>>((float*) q, 32768, 77);
        bf.bias = (float*) q;
    }
    size_t step_bytes = 0;
    for (const Launch& L : M.step) step_bytes += L.bytes();
    std::printf("MODEL step %zu launches, %.1f MB of BF16 weights a step (TP4 member 0, native pack); edge sets %zu\n", M.step.size(),
                step_bytes / 1e6, M.edge.size());
    // the y arena: the step's outputs at B = 16 plus the edge sets
    size_t ycap = 0;
    for (const Launch& L : M.step)
        for (const Job& j : L.jobs) ycap += (((size_t) 16 * j.m.n + 63) / 64) * 64;
    for (const Launch& L : M.edge)
        for (const Job& j : L.jobs) ycap += (((size_t) 16 * j.m.n + 63) / 64) * 64;
    for (const Launch& L : M.tm_extra)
        for (const Job& j : L.jobs) ycap += (((size_t) 16 * j.m.n + 63) / 64) * 64;
    float* yA = nullptr;
    float* yB = nullptr;
    CK(hipMalloc(&yA, ycap * 4));
    CK(hipMalloc(&yB, ycap * 4));
    CK(hipDeviceSynchronize());

    if (mode == "ab") {
        int bad = 0;
        std::vector<float> ha(ycap), hb(ycap);
        for (int B : Bs) {
            for (int set = 0; set < 2; ++set) {   // 0: the step, 1: the edge shapes
                const std::vector<Launch>& LS = set == 0 ? M.step : M.edge;
                std::vector<Prepared> P(LS.size());
                size_t at = 0;
                for (size_t i = 0; i < LS.size(); ++i) at = prep_launch(LS[i], B, bf, yA, at, P[i]);
                // the reference: old, into yA (arena prefilled with a NaN pattern so an unwritten output shows)
                CK(hipMemsetAsync(yA, 0xff, at * 4, s));
                for (size_t i = 0; i < LS.size(); ++i) launch(-1, P[i].J, P[i].nj, P[i].max_n, P[i].max_c, s);
                CK(hipStreamSynchronize(s));
                CK(hipGetLastError());
                CK(hipMemcpy(ha.data(), yA, at * 4, hipMemcpyDeviceToHost));
                for (int v : vars) {
                    std::vector<Prepared> Q(LS.size());
                    size_t at2 = 0;
                    for (size_t i = 0; i < LS.size(); ++i) at2 = prep_launch(LS[i], B, bf, yB, at2, Q[i]);
                    CK(hipMemsetAsync(yB, 0xff, at2 * 4, s));
                    for (size_t i = 0; i < LS.size(); ++i) launch(v, Q[i].J, Q[i].nj, Q[i].max_n, Q[i].max_c, s);
                    CK(hipStreamSynchronize(s));
                    CK(hipGetLastError());
                    CK(hipMemcpy(hb.data(), yB, at2 * 4, hipMemcpyDeviceToHost));
                    size_t nout = 0, ndiff = 0, unwritten = 0;
                    char first[256] = "";
                    for (size_t i = 0; i < LS.size(); ++i)
                        for (int j = 0; j < P[i].nj; ++j) {
                            const uint32_t* a = (const uint32_t*) (ha.data() + P[i].out_off[j]);
                            const uint32_t* b = (const uint32_t*) (hb.data() + Q[i].out_off[j]);
                            for (size_t e = 0; e < P[i].out_n[j]; ++e) {
                                ++nout;
                                if (a[e] == 0xffffffffu) ++unwritten;
                                if (a[e] != b[e]) {
                                    if (ndiff == 0)
                                        std::snprintf(first, sizeof(first), "launch %zu (%s) job %d col %zu row %zu: old 0x%08x new 0x%08x",
                                                      i, LS[i].cls.c_str(), j, e / LS[i].jobs[(size_t) j].m.n,
                                                      e % LS[i].jobs[(size_t) j].m.n, a[e], b[e]);
                                    ++ndiff;
                                }
                            }
                        }
                    const char* sn = set == 0 ? "step" : "edge";
                    if (ndiff == 0 && unwritten == 0)
                        std::printf("AB %s B%d %s => BITWISE (%zu outputs, %zu launches)\n", vname(v), B, sn, nout, LS.size());
                    else {
                        std::printf("AB %s B%d %s => DIFFER %zu of %zu outputs (unwritten by old: %zu) first: %s\n", vname(v), B,
                                    sn, ndiff, nout, unwritten, first);
                        bad = 1;
                    }
                    std::fflush(stdout);
                }
            }
        }
        std::printf("AB DONE %s\n", bad ? "DIFFER" : "ALL BITWISE");
        return bad;
    }

    if (mode == "tm") {   // per class: all of its launches in the step (+ extras) back to back, reps times; us a launch
        std::vector<std::string> classes;
        for (const Launch& L : M.step)
            if (std::find(classes.begin(), classes.end(), L.cls) == classes.end()) classes.push_back(L.cls);
        for (int B : Bs)
            for (const std::string& c : classes) {
                std::vector<const Launch*> LS;
                for (const Launch& L : M.step)
                    if (L.cls == c) LS.push_back(&L);
                for (const Launch& L : M.tm_extra)
                    if (L.cls == c) LS.push_back(&L);
                std::vector<Prepared> P(LS.size());
                size_t at = 0, bytes = 0;
                for (size_t i = 0; i < LS.size(); ++i) {
                    at = prep_launch(*LS[i], B, bf, yA, at, P[i]);
                    bytes += LS[i]->bytes();
                }
                for (int v : vars) {
                    std::vector<double> t;
                    for (int r = 0; r < reps + 2; ++r) {
                        CK(hipEventRecord(e0, s));
                        for (size_t i = 0; i < LS.size(); ++i) launch(v, P[i].J, P[i].nj, P[i].max_n, P[i].max_c, s);
                        CK(hipEventRecord(e1, s));
                        CK(hipEventSynchronize(e1));
                        float ms = 0;
                        CK(hipEventElapsedTime(&ms, e0, e1));
                        if (r >= 2) t.push_back(ms * 1000.0 / (double) LS.size());
                    }
                    CK(hipGetLastError());
                    const double us = med(t), per = (double) bytes / (double) LS.size();
                    std::printf("TM %s B%d %s launches %zu MB/launch %.2f us %.1f p10 %.1f p90 %.1f GB/s %.1f\n", vname(v), B,
                                c.c_str(), LS.size(), per / 1e6, us, pct(t, 0.1), pct(t, 0.9), per / us / 1e3);
                    std::fflush(stdout);
                }
            }
        return 0;
    }

    if (mode == "step") {   // the step's launches in engine order, reps times, variants interleaved rep by rep
        for (int B : Bs) {
            std::vector<Prepared> P(M.step.size());
            size_t at = 0;
            for (size_t i = 0; i < M.step.size(); ++i) at = prep_launch(M.step[i], B, bf, yA, at, P[i]);
            std::vector<std::vector<double>> t(vars.size());
            for (int r = 0; r < reps + 2; ++r)
                for (size_t vi = 0; vi < vars.size(); ++vi) {
                    const size_t v2 = (vi + (size_t) r) % vars.size();   // rotate which variant goes first
                    CK(hipEventRecord(e0, s));
                    for (size_t i = 0; i < M.step.size(); ++i) launch(vars[v2], P[i].J, P[i].nj, P[i].max_n, P[i].max_c, s);
                    CK(hipEventRecord(e1, s));
                    CK(hipEventSynchronize(e1));
                    float ms = 0;
                    CK(hipEventElapsedTime(&ms, e0, e1));
                    if (r >= 2) t[v2].push_back(ms);
                }
            CK(hipGetLastError());
            for (size_t vi = 0; vi < vars.size(); ++vi)
                std::printf("STEP %s B%d launches %zu ms %.3f p10 %.3f p90 %.3f GB/s %.1f\n", vname(vars[vi]), B, M.step.size(),
                            med(t[vi]), pct(t[vi], 0.1), pct(t[vi], 0.9), step_bytes / med(t[vi]) / 1e6);
            std::fflush(stdout);
        }
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
