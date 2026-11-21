// src/kernels/tp_proof_bench.cpp - experiment 0 of the Flash-Next tensor-parallel study: does splitting a layer over
// cards shrink its time enough, does an in-graph all-reduce cost little enough, and how far does a split layer's
// output move from the one-card layer.  A measurement tool only: it changes nothing in the engine, it calls the
// engine's own kernels (the verify window's calls, src/core/verify.cpp record_window, as a gfx1100 runs them with the
// --native kernel set), and every result line starts with "TPP " for grep.
//
//   tp_proof_bench split-time --gguf <first shard> --layers 0,3 [--device 0] [--T 1,3,8] [--iters 200] [--ctx 4096]
//   tp_proof_bench ar-time --devices 0,1 [--bytes 10240,20480,81920] [--iters 2000]
//   tp_proof_bench exact --gguf <first shard> --layers 0,1,2,3 --devices 0,1 [--T 3] [--ctx 4096]
//
// split-time (one device): per layer and window size T, the layer's decode-path GPU work in one captured graph, three
// ways - FULL (all heads, all routed experts on this card), HALF (2 cards: half the GDN key groups with their value
// heads, or half the query heads with one KV head, plus the routed experts with id % 2 == card, plus the shared expert
// on card layer % 2) and QUARTER (the same for 4 cards; a KV head serves two cards).  HALF / QUARTER are timed for
// every card and the slowest card is the layer's time (the critical path).  Each graph is replayed --iters times; the
// median is reported.  The routing is the router's own on the bench's random inputs, taken once and kept (the plan is
// static inside the graph; the router GEMV and top-10 still run in it).
//
// ar-time (two devices): the host-staged all-reduce of the DeepSeek engine (src/dsv41/ds4_kernels.cu allreduce in
// the tensor-parallel build: xfer_put into host-mapped rows + a flag, ar_sum_kernel summing in member order) ported so it runs
// INSIDE a captured graph: the sequence number lives in device memory (a graph replays fixed kernel arguments), every
// wait is bounded (5 s, then "TPP AR TIMEOUT"), and a host watchdog ends the process if no all-reduce completes for 5 s.
//
// exact (two devices): consecutive real layers at T rows, one device FULL against the 2-device split (head-group split,
// expert-parallel experts, the all-reduce above after the mixer and after the MoE).  0,1,2,3 takes the first attention
// layer (3) too: each card's 12 query heads over its one KV head, the out-projection's partial summed by the all-reduce.
//
// Attention split (the DeepSeek engine's method, src/dsv41/ds4_model.cu: a member's head range
// [ah0, ah0 + ahn), its q / output-projection slices of those heads, the attention kernel run with the member's head
// count, the output projection's partial all-reduced): card c of tp holds query heads [c * NH/tp, ...) as its heads
// 0.., the KV head they read (kv0, a copy on both cards of a tp4 pair) as its KV head 0, and calls
// qsa_decode_attn_batch with n_head / n_head_kv = its own counts (12 / 1 tp2, 6 / 1 tp4; qsa_decode_attn_geometry_ok).  The sums run in another order, so
// the outputs are not expected to be bit-identical; the report says whether they are and how far apart they are.
//
// Not reproduced (the same in every variant, so they do not change the comparisons): the token embedding and the PLE
// block of layer 1 (the 27 GiB PLE table lives in host RAM), CPU / PCIe expert shares (every routed expert is in VRAM
// here), KV streaming (an FP16 KV cache, every page resident), and the T = 1 per-token router / combine calls (T = 1
// uses the multi-row kernels with one row).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <vector_types.h>   // the host syntax check (metal_compat): float4 / make_float4
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace K = strata::kernels;

#define CK(x)                                                                                                         \
    do {                                                                                                              \
        cudaError_t e_ = (x);                                                                                         \
        if (e_ != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "tp_proof_bench: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__,    \
                         __LINE__);                                                                                   \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "tp_proof_bench: %s\n", m.c_str());
    std::exit(2);
}

constexpr float EPS = 1e-6f;   // verify.cpp's EPS (the GDN, hyper-connection and QSA norms)
constexpr int MT = 8;          // the most rows a window here holds (T = 1, 3, 8)

// =====================================================================================================================
// device code: the all-reduce (a port of the DeepSeek engine's host-staged one) and the dummy kernels around it
// =====================================================================================================================
#if defined(__HIPCC__) || defined(__CUDACC__)
#define TPP_LAUNCH(kern, grid, block, stream, ...) kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#else   // the host syntax check: the kernels are plain functions there
#define TPP_LAUNCH(kern, grid, block, stream, ...) ((void) (grid), (void) (block), (void) (stream), kern(__VA_ARGS__))
#endif

constexpr int kArThreads = 256;

__device__ __forceinline__ unsigned long long tpp_now_ns() {
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx906__)
    return wall_clock64() * 40ull;   // gfx906: a 25 MHz wall clock (verify_kernels.cu strata_now_ns)
#else
    return wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter
#endif
#elif defined(__CUDA_ARCH__)
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return 0ull;
#endif
}

__device__ __forceinline__ void tpp_pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(100);
#endif
}

// The put (ds4 xfer_put_kernel): this member's partial x -> its host-mapped row of the call's slot, a system fence, and
// the last block to finish raises the member's flag of the slot to the call's sequence number.  The sequence number is
// *ctr + 1 (device memory, advanced by the sum kernel) instead of an argument, so a captured graph replays correctly.
__global__ void __launch_bounds__(kArThreads)
    tpp_ar_put_kernel(const float* x, int n4, float* rows, unsigned long long stride, unsigned* flags, int mi, int slots,
                      const unsigned* ctr, unsigned* cnt, unsigned long long* t_start, unsigned stamp_cap) {
    const unsigned seq = *(const volatile unsigned*) ctr + 1u;
    const int slot = (int) (seq % (unsigned) slots);
    if (t_start != nullptr && blockIdx.x == 0 && threadIdx.x == 0) t_start[seq % stamp_cap] = tpp_now_ns();
    float4* dst = (float4*) (rows + ((size_t) mi * slots + slot) * stride);
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) dst[q] = ((const float4*) x)[q];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *(volatile unsigned*) cnt = 0u;
        __threadfence_system();
        *(volatile unsigned*) (flags + ((size_t) mi * slots + slot) * 32) = seq;
        __threadfence_system();
    }
}

// The sum (ds4 ar_sum_kernel): wait until every other member's flag of the slot reached the sequence number (bounded:
// past timeout_ns, or once another member gave up, *err = 1 and the sums are skipped), then x = the members' partials
// summed in member order (this member's from x, the others' from their host rows with volatile loads), the same bits on
// every member.  The last block advances *ctr and publishes the progress word the host watchdog reads.
__global__ void __launch_bounds__(kArThreads)
    tpp_ar_sum_kernel(float* x, int n4, const float* rows, unsigned long long stride, const unsigned* flags, int nm, int mi,
                      int slots, unsigned* ctr, unsigned* cnt, unsigned* err, unsigned long long timeout_ns,
                      unsigned* progress, unsigned long long* t_end, unsigned stamp_cap) {
    const unsigned seq = *(const volatile unsigned*) ctr + 1u;
    const int slot = (int) (seq % (unsigned) slots);
    __shared__ int bad;
    if (threadIdx.x == 0) {
        bad = 0;
        const unsigned long long t0 = tpp_now_ns();
        for (int m = 0; m < nm && !bad; ++m) {
            if (m == mi) continue;
            const volatile unsigned* f = flags + ((size_t) m * slots + slot) * 32;
            while ((int) (*f - seq) < 0) {
                if (*(const volatile unsigned*) err != 0u) { bad = 1; break; }
                if (tpp_now_ns() - t0 > timeout_ns) {
                    *(volatile unsigned*) err = 1u;
                    __threadfence_system();
                    bad = 1;
                    break;
                }
                tpp_pause();
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    if (!bad) {
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int m = 0; m < nm; ++m) {
                float4 p;
                if (m == mi) p = ((const float4*) x)[q];
                else {
                    const volatile float* r = rows + ((size_t) m * slots + slot) * stride + (size_t) q * 4;
                    p = make_float4(r[0], r[1], r[2], r[3]);
                }
                if (m == 0) v = p;
                else {
                    v.x = v.x + p.x;
                    v.y = v.y + p.y;
                    v.z = v.z + p.z;
                    v.w = v.w + p.w;
                }
            }
            ((float4*) x)[q] = v;
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *(volatile unsigned*) cnt = 0u;
        if (t_end != nullptr) t_end[seq % stamp_cap] = tpp_now_ns();
        *(volatile unsigned*) ctr = seq;
        __threadfence_system();
        *(volatile unsigned*) progress = seq;
        __threadfence_system();
    }
}

// the ar-time graph's work before and after the all-reduce: produce this member's partial, consume the sum
__global__ void tpp_produce_kernel(float* x, const float* src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = src[i];
}
__global__ void tpp_consume_kernel(const float* x, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = x[i] * 0.5f;
}

// =====================================================================================================================
// the all-reduce group (host side)
// =====================================================================================================================
struct ArGroup {
    int nm = 0, slots = 4;
    size_t stride = 0;              // floats a row
    float* rows_h = nullptr;        // host-mapped [nm][slots][stride]
    unsigned* flags_h = nullptr;    // host-mapped [nm][slots][32]
    unsigned* err_h = nullptr;      // host-mapped abort word (shared)
    unsigned* progress_h = nullptr; // host-mapped [nm][32]
    struct Member {
        int dev = 0;
        float* rows = nullptr;
        unsigned *flags = nullptr, *err = nullptr, *progress = nullptr;
        unsigned *ctr = nullptr, *cnt_put = nullptr, *cnt_sum = nullptr;   // device memory
    };
    std::vector<Member> m;
    unsigned long long timeout_ns = 5000000000ull;
};

void ar_open(ArGroup& g, const std::vector<int>& devs, size_t stride_floats) {
    g.nm = (int) devs.size();
    g.stride = (stride_floats + 63) / 64 * 64;
    CK(cudaSetDevice(devs[0]));
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    CK(cudaHostAlloc((void**) &g.rows_h, (size_t) g.nm * g.slots * g.stride * 4 + 64, fl));
    CK(cudaHostAlloc((void**) &g.flags_h, (size_t) g.nm * g.slots * 32 * 4 + 64, fl));
    CK(cudaHostAlloc((void**) &g.err_h, 64, fl));
    CK(cudaHostAlloc((void**) &g.progress_h, (size_t) g.nm * 32 * 4 + 64, fl));
    std::memset(g.rows_h, 0, (size_t) g.nm * g.slots * g.stride * 4);
    std::memset(g.flags_h, 0, (size_t) g.nm * g.slots * 32 * 4);
    std::memset(g.err_h, 0, 64);
    std::memset(g.progress_h, 0, (size_t) g.nm * 32 * 4);
    g.m.resize(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        ArGroup::Member& M = g.m[i];
        M.dev = devs[i];
        CK(cudaSetDevice(M.dev));
        CK(cudaHostGetDevicePointer((void**) &M.rows, g.rows_h, 0));
        CK(cudaHostGetDevicePointer((void**) &M.flags, g.flags_h, 0));
        CK(cudaHostGetDevicePointer((void**) &M.err, g.err_h, 0));
        CK(cudaHostGetDevicePointer((void**) &M.progress, g.progress_h, 0));
        M.progress += i * 32;
        CK(cudaMalloc((void**) &M.ctr, 256));
        CK(cudaMemset(M.ctr, 0, 256));
        M.cnt_put = M.ctr + 16;
        M.cnt_sum = M.ctr + 32;
    }
}

// every member idle: sequence numbers, flags and the abort word back to zero
void ar_reset(ArGroup& g) {
    for (auto& M : g.m) {
        CK(cudaSetDevice(M.dev));
        CK(cudaDeviceSynchronize());
        CK(cudaMemset(M.ctr, 0, 256));
        CK(cudaDeviceSynchronize());
    }
    std::memset(g.flags_h, 0, (size_t) g.nm * g.slots * 32 * 4);
    std::memset(g.progress_h, 0, (size_t) g.nm * 32 * 4);
    *(volatile unsigned*) g.err_h = 0u;
}

// one member's all-reduce of x (n floats, n % 4 == 0) on stream s - capturable
void ar_call(const ArGroup& g, int mi, float* x, int n, cudaStream_t s, unsigned long long* t_start = nullptr,
             unsigned long long* t_end = nullptr, unsigned stamp_cap = 1) {
    if (n % 4 != 0 || (size_t) n > g.stride) die("ar_call: the payload must be a multiple of 4 floats and fit a row");
    const ArGroup::Member& M = g.m[(size_t) mi];
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, (n4 + kArThreads - 1) / kArThreads));
    TPP_LAUNCH(tpp_ar_put_kernel, blocks, kArThreads, s, x, n4, M.rows, (unsigned long long) g.stride, M.flags, mi,
               g.slots, M.ctr, M.cnt_put, t_start, stamp_cap);
    TPP_LAUNCH(tpp_ar_sum_kernel, blocks, kArThreads, s, x, n4, M.rows, (unsigned long long) g.stride, M.flags, g.nm,
               mi, g.slots, M.ctr, M.cnt_sum, M.err, g.timeout_ns, M.progress, t_end, stamp_cap);
    CK(cudaGetLastError());
}

// The host watchdog: while armed, ends the process with "TPP AR TIMEOUT" when no member's progress word moved for
// timeout_s (a hung kernel that never reaches its bounded wait, or a graph that never ran).
struct Watchdog {
    std::atomic<bool> armed{false}, stop{false};
    std::thread th;
    void start(const ArGroup& g, double timeout_s, const char* what) {
        armed = true;
        th = std::thread([this, &g, timeout_s, what] {
            std::vector<unsigned> last((size_t) g.nm, ~0u);
            auto t_move = std::chrono::steady_clock::now();
            while (!stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                bool moved = false;
                for (int i = 0; i < g.nm; ++i) {
                    const unsigned p = *(const volatile unsigned*) (g.progress_h + i * 32);
                    if (p != last[(size_t) i]) { last[(size_t) i] = p; moved = true; }
                }
                const auto now = std::chrono::steady_clock::now();
                if (moved) t_move = now;
                else if (armed.load() && std::chrono::duration<double>(now - t_move).count() > timeout_s) {
                    std::printf("TPP AR TIMEOUT %s: no all-reduce completed for %.1f s (progress", what, timeout_s);
                    for (int i = 0; i < g.nm; ++i) std::printf(" dev%d=%u", g.m[(size_t) i].dev, last[(size_t) i]);
                    std::printf(" err=%u)\n", *(const volatile unsigned*) g.err_h);
                    std::fflush(stdout);
                    std::_Exit(3);
                }
            }
        });
    }
    void finish() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

void check_ar_err(const ArGroup& g, const char* what) {
    if (*(const volatile unsigned*) g.err_h != 0u) {
        std::printf("TPP AR TIMEOUT %s: a member's bounded wait gave up (%.1f s)\n", what, (double) g.timeout_ns * 1e-9);
        std::fflush(stdout);
        std::_Exit(3);
    }
}

// =====================================================================================================================
// small host helpers
// =====================================================================================================================
std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> v;
    size_t at = 0;
    while (at < s.size()) {
        size_t e = s.find(',', at);
        if (e == std::string::npos) e = s.size();
        if (e > at) v.push_back(std::atoi(s.substr(at, e - at).c_str()));
        at = e + 1;
    }
    return v;
}

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& def = "") const {
        const auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    bool has(const std::string& k) const { return kv.count(k) != 0; }
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = std::min(v.size() - 1, (size_t) std::ceil(p * (double) v.size()) - (p > 0 ? 1 : 0));
    return v[i];
}

template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc((void**) &p, n * sizeof(T) + 256));
    CK(cudaMemset(p, 0, n * sizeof(T) + 256));
    return p;
}
template <class T> T* dup(const std::vector<T>& h) {
    T* p = dalloc<T>(h.size());
    if (!h.empty()) CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
std::vector<uint16_t> to_bf16(const std::vector<float>& f) {
    std::vector<uint16_t> o(f.size());
    for (size_t i = 0; i < f.size(); ++i) o[i] = K::bf16_from_f32(f[i]);
    return o;
}
std::vector<float> randn(size_t n, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// =====================================================================================================================
// the model file
// =====================================================================================================================
struct Tensor {
    const strata::TensorInfo* ti = nullptr;
    const uint8_t* p = nullptr;
    int type = -1;
    int64_t ne0 = 1, ne1 = 1, ne2 = 1;
    size_t row_bytes = 0;   // bytes of one ne0 row
    int blk_elems = 1, blk_bytes = 4;
};

struct Geo {
    int64_t N = 0, HC = 0, HCLR = 0, NH = 0, NKV = 0, HD = 0, HK = 0, HV = 0, S = 0, C = 0, ZV = 0, NE = 0, K = 0,
            NFF = 0, IQ = 0, ID = 0, interval = 0, n_layers = 0;
    bool attn(int64_t l) const { return l % interval == interval - 1; }
};

struct Model {
    std::unique_ptr<strata::GgufModel> g;
    Geo geo;

    Tensor t(const std::string& name) const {
        size_t sh = 0;
        const strata::TensorInfo* ti = g->find(name, &sh);
        if (ti == nullptr) die("the model has no tensor " + name);
        Tensor r;
        r.ti = ti;
        r.p = g->shard(sh).tensor_data(*ti);
        r.type = (int) ti->type;
        if (!strata::block_geometry(ti->type, r.blk_elems, r.blk_bytes))
            die(name + ": unknown ggml type " + std::to_string(ti->type));
        r.ne0 = ti->shape.size() > 0 ? (int64_t) ti->shape[0] : 1;
        r.ne1 = ti->shape.size() > 1 ? (int64_t) ti->shape[1] : 1;
        r.ne2 = ti->shape.size() > 2 ? (int64_t) ti->shape[2] : 1;
        if (r.ne0 % r.blk_elems != 0) die(name + ": a row is not a whole number of blocks");
        r.row_bytes = (size_t) (r.ne0 / r.blk_elems) * (size_t) r.blk_bytes;
        return r;
    }
    int64_t meta_u(const std::string& key) const {
        const strata::MetaValue* v = g->meta().get(key);
        if (v == nullptr || !v->is_num()) die("the model has no numeric key " + key);
        return (int64_t) v->num();
    }
};

void open_model(Model& M, const std::string& first_shard) {
    try {
        M.g = std::make_unique<strata::GgufModel>(strata::GgufModel::open(first_shard));
    } catch (const std::exception& e) {
        die(std::string("cannot open the model: ") + e.what());
    }
    const strata::MetaValue* arch = M.g->meta().get("general.architecture");
    const std::string a = arch ? arch->s : std::string("?");
    Geo& G = M.geo;
    G.N = M.meta_u(a + ".embedding_length");
    G.NH = M.meta_u(a + ".attention.head_count");
    G.NKV = M.meta_u(a + ".attention.head_count_kv");
    G.HD = M.meta_u(a + ".attention.key_length");
    G.HK = M.meta_u(a + ".ssm.group_count");
    G.HV = M.meta_u(a + ".ssm.time_step_rank");
    G.S = M.meta_u(a + ".ssm.state_size");
    G.ZV = M.meta_u(a + ".ssm.inner_size");
    G.NE = M.meta_u(a + ".expert_count");
    G.K = M.meta_u(a + ".expert_used_count");
    G.NFF = M.meta_u(a + ".expert_feed_forward_length");
    G.IQ = M.meta_u(a + ".attention.indexer.head_count");
    G.ID = M.meta_u(a + ".attention.indexer.key_length");
    G.interval = M.meta_u(a + ".full_attention_interval");
    G.n_layers = M.meta_u(a + ".block_count");
    G.C = 2 * G.HK * G.S + G.ZV;
    const Tensor hn = M.t("blk.0.hc_attn_norm.weight"), hd = M.t("blk.0.hc_attn_down.weight");
    G.HC = hn.ne0 / G.N;
    G.HCLR = hd.ne1;
    const K::QsaShapes rs = K::qsa_real_shapes();
    if (G.NH != rs.n_head || G.NKV != rs.n_head_kv || G.HD != rs.head_dim || G.IQ != rs.idx_n_head || G.ID != rs.idx_dim)
        die("the attention geometry is not the one the QSA kernels are built for");
    if (G.K != 10 || G.NE != 512) die("the native router here is the 512-expert top-10 one");
    if (G.HV % G.HK != 0 || G.HK % 4 != 0 || G.NH % 4 != 0) die("the head counts do not split over 4 cards");
    if (!K::fused_gr_supported(G.N, G.HC, G.HCLR)) die("fused_gr does not support this hyper-connection geometry");
    std::fprintf(stderr, "# model %s: %s, %lld layers, N %lld, hc %lld x %lld, GDN %lld k-groups / %lld v-heads, attention "
                         "%lld q / %lld kv heads every %lld, %lld experts top-%lld\n",
                 a.c_str(), first_shard.c_str(), (long long) G.n_layers, (long long) G.N, (long long) G.HC,
                 (long long) G.HCLR, (long long) G.HK, (long long) G.HV, (long long) G.NH, (long long) G.NKV,
                 (long long) G.interval, (long long) G.NE, (long long) G.K);
}

// a whole small tensor as f32 (F32, F16, BF16 and Q8_0 sources)
std::vector<float> to_f32(const Tensor& t) {
    const size_t n = (size_t) (t.ne0 * t.ne1 * t.ne2);
    std::vector<float> o(n);
    switch (t.type) {
        case 0: std::memcpy(o.data(), t.p, n * 4); break;
        case 1:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, t.p + i * 2, 2);
                o[i] = K::f32_from_f16(h);
            }
            break;
        case 30:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, t.p + i * 2, 2);
                o[i] = K::f32_from_bf16(h);
            }
            break;
        case 8:
            for (size_t b = 0; b < n / 32; ++b) {
                const uint8_t* blk = t.p + b * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d = K::f32_from_f16(dh);
                for (int j = 0; j < 32; ++j) o[b * 32 + j] = d * (float) (int8_t) blk[2 + j];
            }
            break;
        default: die(std::string("to_f32: type ") + t.ti->type_name() + " of " + t.ti->name + " is not handled");
    }
    return o;
}

// rows [r0, r1) ranges of a 2-D quantized (or plain) tensor, concatenated in order
std::vector<uint8_t> slice_rows(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& rr) {
    std::vector<uint8_t> o;
    for (const auto& r : rr) {
        if (r.first < 0 || r.second > t.ne1 || r.first > r.second) die(t.ti->name + ": a row range is out of bounds");
        o.insert(o.end(), t.p + (size_t) r.first * t.row_bytes, t.p + (size_t) r.second * t.row_bytes);
    }
    return o;
}
// columns [c0, c1) ranges of every row (each range whole blocks), concatenated per row: the new rows
std::vector<uint8_t> slice_cols(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& cr, int64_t& new_ne0) {
    new_ne0 = 0;
    for (const auto& c : cr) {
        if (c.first % t.blk_elems || c.second % t.blk_elems)
            die(t.ti->name + ": a column split is not on a block boundary of " + t.ti->type_name());
        new_ne0 += c.second - c.first;
    }
    std::vector<uint8_t> o;
    o.reserve((size_t) t.ne1 * (size_t) (new_ne0 / t.blk_elems) * t.blk_bytes);
    for (int64_t r = 0; r < t.ne1; ++r)
        for (const auto& c : cr) {
            const uint8_t* row = t.p + (size_t) r * t.row_bytes;
            o.insert(o.end(), row + (size_t) (c.first / t.blk_elems) * t.blk_bytes,
                     row + (size_t) (c.second / t.blk_elems) * t.blk_bytes);
        }
    return o;
}
std::vector<std::pair<int64_t, int64_t>> merge(std::vector<std::pair<int64_t, int64_t>> v) {
    std::vector<std::pair<int64_t, int64_t>> o;
    for (const auto& r : v) {
        if (!o.empty() && o.back().second == r.first) o.back().second = r.second;
        else o.push_back(r);
    }
    return o;
}
void need_native(const Tensor& t) {
    if (!K::native_mmvq_supported(t.type))
        die(t.ti->name + ": native_mmvq has no kernel for " + std::string(t.ti->type_name()));
}

// =====================================================================================================================
// the weights of one layer on one device: the replicated part, and the part of one (tp, card) variant
// =====================================================================================================================
struct HcW {
    float* norm = nullptr;
    uint16_t *down = nullptr, *up = nullptr, *inject = nullptr;
};

struct CommonW {   // replicated on every card of a split
    HcW hc[2];
    uint16_t* router = nullptr;        // bf16 [NE][N]
    uint16_t* sh_gate_inp = nullptr;   // bf16 [N]
    int shg_t = -1, shu_t = -1, shd_t = -1;
    void *shg = nullptr, *shu = nullptr, *shd = nullptr;
    uint8_t* experts = nullptr;        // [NE] native blobs (gate rows | up rows | down rows)
    K::NativeExpertLayout L;
    // attention layers: the indexer (replicated so every card selects the same cells)
    uint16_t *ik = nullptr, *iq = nullptr;
    float *ikn = nullptr, *iqn = nullptr;
};

struct Plan {   // a static grouped-expert plan (device), the layout native_expert_grouped reads
    unsigned long long* grp_ptr = nullptr;
    int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    int groups = 0, entries = 0;
};

struct Part {   // one (tp, card) variant of one layer
    int tp = 1, card = 0;
    bool attn = false, has_shared = true;
    // GDN
    int HK = 0, HV = 0, C = 0, ZV = 0;
    int qkv_t = -1, g_t = -1, out_t = -1;
    void *qkv = nullptr, *gw = nullptr, *out = nullptr;
    uint16_t *alpha = nullptr, *beta = nullptr;
    float *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *norm = nullptr, *state = nullptr;
    // attention
    int NH = 0, NKV = 0;
    int q_t = -1, k_t = -1, v_t = -1, o_t = -1;
    void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    float *qn = nullptr, *kn = nullptr;
    uint16_t *kpool = nullptr, *vpool = nullptr;
    int32_t* ptab = nullptr;
    float *itail = nullptr, *idead = nullptr, *ipooled = nullptr, *tail_snap = nullptr;
    int32_t* iblockpos = nullptr;
    int64_t max_cells = 0, max_blocks = 0, cap = 0, attn_floats = 0;
    K::QsaShapes s;
    int32_t* pos = nullptr;   // [MT * NH | MT * NKV | MT * IQ] rope positions
    float *scores = nullptr, *attn_scratch = nullptr;
    int32_t* sel = nullptr;
    // experts
    Plan plan;
    float* parts = nullptr;   // [MT][K][N], rows of other cards' experts stay zero
    // the engine's on-device plan kernel (resident_plan, E-6) with this card's residency (id % tp == card): its cost is
    // in every variant's graph; its output goes to scratch (the experts read the static plan above, the same groups for
    // the one-card variant; resident_plan cannot plan a card that holds only some of a window's experts)
    int32_t *res = nullptr, *rplan = nullptr;
    uint32_t* rskip = nullptr;
};

// per-device scratch (one window's buffers, shared by the variants run one after another)
struct Bufs {
    float *R = nullptr, *R0 = nullptr, *bo_zero = nullptr, *inj_in = nullptr, *inj = nullptr, *inj2 = nullptr;
    float *mixed = nullptr, *xn = nullptr, *lo = nullptr, *rs = nullptr;
    void *xq = nullptr, *xil = nullptr, *nat_xq = nullptr, *exp_scratch = nullptr;
    float *qkv = nullptr, *hb = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *y = nullptr;
    float *bo = nullptr, *bo_out = nullptr;
    float *qfull = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *idx_raw = nullptr, *qidx = nullptr;
    float *attn = nullptr, *attn32 = nullptr;
    int32_t* step = nullptr;
    float *logits = nullptr, *w = nullptr;
    int32_t* ids = nullptr;
    float *sh_gate = nullptr, *sh_up = nullptr, *sh_g = nullptr, *shared = nullptr, *shared_zero = nullptr;
    uint16_t* sh_bf16 = nullptr;
};

void make_bufs(Bufs& b, const Geo& G, int64_t ctx) {
    const int64_t RN = MT * G.HC * G.N;
    b.R = dalloc<float>((size_t) RN);
    b.R0 = dalloc<float>((size_t) RN);
    b.bo_zero = dalloc<float>((size_t) (MT * G.N));
    b.inj_in = dalloc<float>((size_t) (MT * G.HC));
    b.inj = dalloc<float>((size_t) (MT * G.HC));
    b.inj2 = dalloc<float>((size_t) (MT * G.HC));
    b.mixed = dalloc<float>((size_t) (MT * G.N));
    b.xn = dalloc<float>((size_t) RN);
    b.lo = dalloc<float>((size_t) (MT * G.HCLR));
    b.rs = dalloc<float>((size_t) (MT * G.HC));
    const int max_in = (int) std::max({G.N, G.ZV, G.NH * G.HD, G.NFF});
    b.xq = dalloc<uint8_t>(K::native_q8_1_bytes(max_in, MT));
    b.xil = dalloc<uint8_t>(K::native_q8_1_il_bytes(max_in, MT));
    b.nat_xq = dalloc<uint8_t>((size_t) (MT * (G.N / 32) * 36));
    b.exp_scratch = dalloc<uint8_t>(K::native_expert_scratch_bytes(MT * G.K, G.NFF));
    b.qkv = dalloc<float>((size_t) (MT * G.C));
    b.hb = dalloc<float>((size_t) (MT * G.C));
    b.gate = dalloc<float>((size_t) (MT * G.HV));
    b.beta = dalloc<float>((size_t) (MT * G.HV));
    b.z = dalloc<float>((size_t) (MT * G.ZV));
    b.y = dalloc<float>((size_t) (MT * G.ZV));
    b.bo = dalloc<float>((size_t) (MT * G.N));
    b.bo_out = dalloc<float>((size_t) (MT * G.N));
    b.qfull = dalloc<float>((size_t) (MT * G.NH * 2 * G.HD));
    b.qcur = dalloc<float>((size_t) (MT * G.NH * G.HD));
    b.kcur = dalloc<float>((size_t) (MT * G.NKV * G.HD));
    b.vcur = dalloc<float>((size_t) (MT * G.NKV * G.HD));
    b.idx_raw = dalloc<float>((size_t) (MT * G.ID));
    b.qidx = dalloc<float>((size_t) (MT * G.IQ * G.ID));
    b.attn = dalloc<float>((size_t) (MT * G.NH * G.HD));
    b.attn32 = dalloc<float>((size_t) (MT * G.NH * G.HD));
    b.logits = dalloc<float>((size_t) (MT * G.NE));
    b.w = dalloc<float>((size_t) (MT * G.K));
    b.ids = dalloc<int32_t>((size_t) (MT * G.K));
    b.sh_gate = dalloc<float>((size_t) (MT * G.NFF));
    b.sh_up = dalloc<float>((size_t) (MT * G.NFF));
    b.sh_g = dalloc<float>((size_t) MT);
    b.shared = dalloc<float>((size_t) (MT * G.N));
    b.shared_zero = dalloc<float>((size_t) (MT * G.N));
    b.sh_bf16 = dalloc<uint16_t>((size_t) (MT * G.N));
    // the window's steps: row t at position ctx + t (qsa.hpp's step state)
    const K::QsaShapes s = K::qsa_real_shapes();
    std::vector<int32_t> st((size_t) MT * K::kStepCount);
    for (int t = 0; t < MT; ++t) K::qsa_step_fill(&st[(size_t) t * K::kStepCount], ctx + t, s);
    b.step = dup(st);
}

void set_inputs(Bufs& b, const Geo& G, uint32_t seed) {
    const std::vector<float> R = randn((size_t) (MT * G.HC * G.N), 1.0f, seed);
    CK(cudaMemcpy(b.R0, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.R, R.data(), R.size() * 4, cudaMemcpyHostToDevice));
    const std::vector<float> inj = randn((size_t) (MT * G.HC), 0.5f, seed + 1);
    CK(cudaMemcpy(b.inj_in, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice));
}

HcW load_hc(const Model& M, int64_t l, const char* which) {
    const std::string p = "blk." + std::to_string(l) + "." + which;
    HcW h;
    h.norm = dup(to_f32(M.t(p + "norm.weight")));
    h.down = dup(to_bf16(to_f32(M.t(p + "down.weight"))));
    h.up = dup(to_bf16(to_f32(M.t(p + "up.weight"))));
    h.inject = dup(to_bf16(to_f32(M.t(p + "inject.weight"))));
    return h;
}

void* dup_bytes(const std::vector<uint8_t>& v) { return (void*) dup(v); }

void load_common(CommonW& c, const Model& M, int64_t l, bool experts) {
    const Geo& G = M.geo;
    const std::string p = "blk." + std::to_string(l) + ".";
    c.hc[0] = load_hc(M, l, "hc_attn_");
    c.hc[1] = load_hc(M, l, "hc_ffn_");
    c.router = dup(to_bf16(to_f32(M.t(p + "ffn_gate_inp.weight"))));
    c.sh_gate_inp = dup(to_bf16(to_f32(M.t(p + "ffn_gate_inp_shexp.weight"))));
    const Tensor sg = M.t(p + "ffn_gate_shexp.weight"), su = M.t(p + "ffn_up_shexp.weight"),
                 sd = M.t(p + "ffn_down_shexp.weight");
    need_native(sg);
    need_native(su);
    need_native(sd);
    c.shg_t = sg.type;
    c.shu_t = su.type;
    c.shd_t = sd.type;
    c.shg = dup_bytes(slice_rows(sg, {{0, sg.ne1}}));
    c.shu = dup_bytes(slice_rows(su, {{0, su.ne1}}));
    c.shd = dup_bytes(slice_rows(sd, {{0, sd.ne1}}));
    if (G.attn(l)) {
        c.ik = dup(to_bf16(to_f32(M.t(p + "indexer.k_proj.weight"))));
        c.iq = dup(to_bf16(to_f32(M.t(p + "indexer.q_proj.weight"))));
        c.ikn = dup(to_f32(M.t(p + "indexer.k_norm.weight")));
        c.iqn = dup(to_f32(M.t(p + "indexer.q_norm.weight")));
    }
    if (!experts) return;
    const Tensor eg = M.t(p + "ffn_gate_exps.weight"), eu = M.t(p + "ffn_up_exps.weight"),
                 ed = M.t(p + "ffn_down_exps.weight");
    if (eg.type != eu.type) die(p + ": gate and up experts of different types");
    if (!K::native_expert_supported(eg.type, ed.type, G.N, G.NFF))
        die(p + ": native_expert_grouped has no kernels for " + eg.ti->type_name() + " / " + ed.ti->type_name());
    c.L = K::native_expert_layout(eg.type, ed.type, G.N, G.NFF);
    if (c.L.tail_off != 0 || c.L.gu_row != eg.row_bytes || c.L.d_row != ed.row_bytes)
        die(p + ": the expert blob layout is not [gate | up | down] GGUF rows");
    const size_t gu_bytes = (size_t) G.NFF * eg.row_bytes, d_bytes = (size_t) G.N * ed.row_bytes;
    c.experts = dalloc<uint8_t>((size_t) G.NE * c.L.bytes);
    std::vector<uint8_t> blob(c.L.bytes);
    for (int64_t e = 0; e < G.NE; ++e) {
        std::memcpy(blob.data(), eg.p + (size_t) e * gu_bytes, gu_bytes);
        std::memcpy(blob.data() + c.L.up_off, eu.p + (size_t) e * gu_bytes, gu_bytes);
        std::memcpy(blob.data() + c.L.down_off, ed.p + (size_t) e * d_bytes, d_bytes);
        CK(cudaMemcpy(c.experts + (size_t) e * c.L.bytes, blob.data(), c.L.bytes, cudaMemcpyHostToDevice));
    }
}

// The full layer's random recurrent state / KV cache / indexer state (the same values for every variant, which slices them)
struct HostState {
    std::vector<float> gdn_state, conv_hist;              // [S][HV][S], [C][3]
    std::vector<uint16_t> kpool, vpool;                   // fp16 [page][NKV][ps][HD]
    std::vector<float> itail, idead, ipooled;
    int64_t max_cells = 0, n_pages = 0;
};

HostState make_state(const Geo& G, int64_t l, int64_t ctx) {
    HostState h;
    if (!G.attn(l)) {
        h.gdn_state = randn((size_t) (G.S * G.HV * G.S), 0.02f, 1000u + (uint32_t) l);
        h.conv_hist = randn((size_t) (G.C * 3), 0.5f, 2000u + (uint32_t) l);
        return h;
    }
    const K::QsaShapes s = K::qsa_real_shapes();
    h.max_cells = (ctx + 64 + s.page_size - 1) / s.page_size * s.page_size;
    h.n_pages = h.max_cells / s.page_size;
    const size_t cells = (size_t) (h.n_pages * G.NKV * s.page_size * G.HD);
    const std::vector<float> kf = randn(cells, 0.5f, 3000u + (uint32_t) l), vf = randn(cells, 0.5f, 4000u + (uint32_t) l);
    h.kpool.resize(cells);
    h.vpool.resize(cells);
    for (size_t i = 0; i < cells; ++i) {
        h.kpool[i] = K::f16_from_f32(kf[i]);
        h.vpool[i] = K::f16_from_f32(vf[i]);
    }
    h.itail = randn((size_t) ((s.idx_block - 1) * G.ID), 0.5f, 5000u + (uint32_t) l);
    h.idead = randn((size_t) G.ID, 0.5f, 6000u + (uint32_t) l);
    h.ipooled = randn((size_t) ((h.max_cells / s.idx_block + 2) * G.ID), 0.5f, 7000u + (uint32_t) l);
    return h;
}

// The (tp, card) variant of layer l: its slices of the mixer, its slice of the state, its expert filter.
void load_part(Part& P, const Model& M, int64_t l, int tp, int card, const HostState& hs) {
    const Geo& G = M.geo;
    const std::string p = "blk." + std::to_string(l) + ".";
    P.tp = tp;
    P.card = card;
    P.attn = G.attn(l);
    P.has_shared = (l % tp) == card;
    P.parts = dalloc<float>((size_t) (MT * G.K * G.N));
    P.plan.grp_ptr = dalloc<unsigned long long>((size_t) (MT * G.K));
    P.plan.grp_start = dalloc<int32_t>((size_t) (MT * G.K + 1));
    P.plan.n_groups = dalloc<int32_t>(4);
    P.plan.ent_dst = dalloc<int32_t>((size_t) (MT * G.K));
    P.plan.ent_tok = dalloc<int32_t>((size_t) (MT * G.K));
    {
        std::vector<int32_t> res((size_t) G.NE);
        for (int64_t e = 0; e < G.NE; ++e) res[(size_t) e] = (e % tp == card) ? (int32_t) e : -1;
        P.res = dup(res);
        P.rplan = dalloc<int32_t>((size_t) (8 * MT * G.K + 64));
        P.rskip = dalloc<uint32_t>(4);
    }
    if (!P.attn) {
        // key groups [card * HK/tp, ...) and their value heads; value head i of the slice is original head
        // kset[i % Gk] + HK * (i / Gk), so the kernels' modulo pairing (v-head i reads k-head i % h_k) still holds
        const int64_t Gk = G.HK / tp, S = G.S;
        P.HK = (int) Gk;
        P.HV = (int) (G.HV / tp);
        std::vector<int64_t> kset, vlist;
        for (int64_t j = 0; j < Gk; ++j) kset.push_back(card * Gk + j);
        for (int64_t i = 0; i < P.HV; ++i) vlist.push_back(kset[(size_t) (i % Gk)] + G.HK * (i / Gk));
        P.ZV = (int) (P.HV * S);
        P.C = (int) (2 * Gk * S + P.ZV);
        std::vector<int64_t> chans;   // q | k | v channels, in the slice's order
        for (int64_t j : kset)
            for (int64_t d = 0; d < S; ++d) chans.push_back(j * S + d);
        for (int64_t j : kset)
            for (int64_t d = 0; d < S; ++d) chans.push_back(G.HK * S + j * S + d);
        for (int64_t h : vlist)
            for (int64_t d = 0; d < S; ++d) chans.push_back(2 * G.HK * S + h * S + d);
        std::vector<std::pair<int64_t, int64_t>> crow, zrow, ocol;
        for (int64_t c : chans) crow.push_back({c, c + 1});
        for (int64_t h : vlist) zrow.push_back({h * S, (h + 1) * S});
        for (int64_t h : vlist) ocol.push_back({h * S, (h + 1) * S});
        const Tensor tq = M.t(p + "attn_qkv.weight"), tg = M.t(p + "attn_gate.weight"), to = M.t(p + "ssm_out.weight");
        need_native(tq);
        need_native(tg);
        need_native(to);
        P.qkv_t = tq.type;
        P.g_t = tg.type;
        P.out_t = to.type;
        P.qkv = dup_bytes(slice_rows(tq, merge(crow)));
        P.gw = dup_bytes(slice_rows(tg, merge(zrow)));
        int64_t ne0 = 0;
        P.out = dup_bytes(slice_cols(to, merge(ocol), ne0));
        if (ne0 != P.ZV) die("ssm_out slice width");
        const std::vector<float> al = to_f32(M.t(p + "ssm_alpha.weight")), be = to_f32(M.t(p + "ssm_beta.weight")),
                                 dt = to_f32(M.t(p + "ssm_dt.bias")), sa = to_f32(M.t(p + "ssm_a")),
                                 cw = to_f32(M.t(p + "ssm_conv1d.weight"));
        std::vector<float> al2, be2, dt2, sa2, cw2;
        for (int64_t h : vlist) {
            al2.insert(al2.end(), al.begin() + h * G.N, al.begin() + (h + 1) * G.N);
            be2.insert(be2.end(), be.begin() + h * G.N, be.begin() + (h + 1) * G.N);
            dt2.push_back(dt[(size_t) h]);
            sa2.push_back(sa[(size_t) h]);
        }
        const int64_t taps = (int64_t) cw.size() / G.C;
        for (int64_t c : chans) cw2.insert(cw2.end(), cw.begin() + c * taps, cw.begin() + (c + 1) * taps);
        P.alpha = dup(to_bf16(al2));
        P.beta = dup(to_bf16(be2));
        P.dt = dup(dt2);
        P.ssm_a = dup(sa2);
        P.conv = dup(cw2);
        P.norm = dup(to_f32(M.t(p + "ssm_norm.weight")));
        // the state [S][HV][S] and the conv history [C][3], sliced the same way
        std::vector<float> st((size_t) (S * P.HV * S + P.C * 3));
        for (int64_t r = 0; r < S; ++r)
            for (int64_t i = 0; i < P.HV; ++i)
                std::memcpy(&st[(size_t) ((r * P.HV + i) * S)], &hs.gdn_state[(size_t) ((r * G.HV + vlist[(size_t) i]) * S)],
                            (size_t) S * 4);
        float* hist = st.data() + S * P.HV * S;
        for (size_t c = 0; c < chans.size(); ++c)
            for (int j = 0; j < 3; ++j) hist[c * 3 + j] = hs.conv_hist[(size_t) chans[c] * 3 + j];
        P.state = dup(st);
        return;
    }
    // attention: query heads [card * NH/tp, ...); KV heads [card * NKV/tp, ...), or the one KV head those queries read
    P.NH = (int) (G.NH / tp);
    P.NKV = (int) std::max<int64_t>(1, G.NKV / tp);
    const int64_t h0 = card * P.NH, kv0 = tp <= G.NKV ? card * P.NKV : (h0 / (G.NH / G.NKV));
    const int64_t HD = G.HD;
    const Tensor tq = M.t(p + "attn_q.weight"), tk = M.t(p + "attn_k.weight"), tv = M.t(p + "attn_v.weight"),
                 to = M.t(p + "attn_output.weight");
    need_native(tq);
    need_native(tk);
    need_native(tv);
    need_native(to);
    P.q_t = tq.type;
    P.k_t = tk.type;
    P.v_t = tv.type;
    P.o_t = to.type;
    P.q = dup_bytes(slice_rows(tq, {{h0 * 2 * HD, (h0 + P.NH) * 2 * HD}}));   // per head [q | gate] rows
    P.k = dup_bytes(slice_rows(tk, {{kv0 * HD, (kv0 + P.NKV) * HD}}));
    P.v = dup_bytes(slice_rows(tv, {{kv0 * HD, (kv0 + P.NKV) * HD}}));
    int64_t ne0 = 0;
    P.o = dup_bytes(slice_cols(to, {{h0 * HD, (h0 + P.NH) * HD}}, ne0));
    P.qn = dup(to_f32(M.t(p + "attn_q_norm.weight")));
    P.kn = dup(to_f32(M.t(p + "attn_k_norm.weight")));
    P.s = K::qsa_real_shapes();
    P.s.n_head = P.NH;
    P.s.n_head_kv = P.NKV;
    if (!K::qsa_decode_attn_geometry_ok(P.s))
        die("layer " + std::to_string(l) + " tp " + std::to_string(tp) + ": qsa_decode_attn does not run " +
            std::to_string(P.NH) + " q / " + std::to_string(P.NKV) + " kv heads a card");
    P.max_cells = hs.max_cells;
    P.max_blocks = hs.max_cells / P.s.idx_block + 2;
    P.cap = K::qsa_selection_width(K::kTopkMaxCells, P.s);
    P.attn_floats = (int64_t) K::qsa_decode_attn_scratch_floats(P.cap, P.s);
    // the KV pools: the slice's KV heads of every page
    const int64_t ps = P.s.page_size;
    std::vector<uint16_t> kp((size_t) (hs.n_pages * P.NKV * ps * HD)), vp(kp.size());
    for (int64_t pg = 0; pg < hs.n_pages; ++pg)
        for (int64_t h = 0; h < P.NKV; ++h) {
            const size_t src = (size_t) (((pg * G.NKV + kv0 + h) * ps) * HD), dst = (size_t) (((pg * P.NKV + h) * ps) * HD);
            std::memcpy(&kp[dst], &hs.kpool[src], (size_t) (ps * HD) * 2);
            std::memcpy(&vp[dst], &hs.vpool[src], (size_t) (ps * HD) * 2);
        }
    P.kpool = dup(kp);
    P.vpool = dup(vp);
    std::vector<int32_t> pt((size_t) hs.n_pages);
    for (int64_t i = 0; i < hs.n_pages; ++i) pt[(size_t) i] = (int32_t) i;
    P.ptab = dup(pt);
    P.itail = dup(hs.itail);
    P.idead = dup(hs.idead);
    P.ipooled = dup(hs.ipooled);
    P.iblockpos = dalloc<int32_t>(4);
    P.tail_snap = dalloc<float>(hs.itail.size());
    P.scores = dalloc<float>((size_t) (MT * P.max_blocks));
    P.sel = dalloc<int32_t>((size_t) (MT * P.cap));
    P.attn_scratch = dalloc<float>((size_t) (MT * P.attn_floats));
}

void set_positions(Part& P, const Geo& G, int64_t ctx) {
    if (!P.attn) return;
    std::vector<int32_t> pos((size_t) (MT * (P.NH + P.NKV + G.IQ)));
    for (int t = 0; t < MT; ++t) {
        for (int h = 0; h < P.NH; ++h) pos[(size_t) (t * P.NH + h)] = (int32_t) (ctx + t);
        for (int h = 0; h < P.NKV; ++h) pos[(size_t) (MT * P.NH + t * P.NKV + h)] = (int32_t) (ctx + t);
        for (int h = 0; h < G.IQ; ++h) pos[(size_t) (MT * (P.NH + P.NKV) + t * G.IQ + h)] = (int32_t) (ctx + t);
    }
    P.pos = dup(pos);
}

// the variant's expert plan from the window's routing: its entries (id % tp == card), distinct experts in routing
// order, each expert's entries ascending (the host pool's / resident_plan's order)
void build_plan(Part& P, const CommonW& c, const Geo& G, const std::vector<int32_t>& ids, int n, cudaStream_t s) {
    const int cap = n * (int) G.K;
    std::vector<int> order;
    std::map<int, std::vector<int>> ent;
    for (int e = 0; e < cap; ++e) {
        const int id = ids[(size_t) e];
        if (id < 0 || id >= G.NE) die("the router returned an expert id out of range");
        if (id % P.tp != P.card) continue;
        if (!ent.count(id)) order.push_back(id);
        ent[id].push_back(e);
    }
    std::vector<unsigned long long> ptr((size_t) cap, 0);
    std::vector<int32_t> start((size_t) cap + 1, 0), dst((size_t) cap, 0), tok((size_t) cap, 0);
    int at = 0;
    for (size_t gi = 0; gi < order.size(); ++gi) {
        ptr[gi] = (unsigned long long) (uintptr_t) (c.experts + (size_t) order[gi] * c.L.bytes);
        start[gi] = at;
        for (int e : ent[order[gi]]) {
            dst[(size_t) at] = e;
            tok[(size_t) at] = e / (int) G.K;
            ++at;
        }
    }
    for (size_t gi = order.size(); gi <= (size_t) cap; ++gi) start[gi] = at;
    const int32_t ng[4] = {(int32_t) order.size(), 0, 0, 0};
    CK(cudaMemcpyAsync(P.plan.grp_ptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(P.plan.grp_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(P.plan.ent_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(P.plan.ent_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(P.plan.n_groups, ng, sizeof ng, cudaMemcpyHostToDevice, s));
    CK(cudaMemsetAsync(P.parts, 0, (size_t) (MT * G.K * G.N) * 4, s));   // other cards' rows: zeros in the combine
    CK(cudaStreamSynchronize(s));
    P.plan.groups = (int) order.size();
    P.plan.entries = at;
}

// =====================================================================================================================
// one layer's window work, recorded on a stream: the verify path's calls (record_window) for one group of n rows
// =====================================================================================================================
struct Quant {   // verify.cpp's quant_in / mm pair: a projection input's q8_1 blocks, then the native GEMV (F4 il form)
    Bufs& b;
    int n;
    cudaStream_t s;
    bool il_ready = false;
    void in(const float* x, int n_in) {
        K::native_quantize_q8_1(x, b.xq, n_in, n, s);
        il_ready = false;
    }
    void mm(int type, const void* w, float* out, int n_in, int n_out) {
        if (K::native_mmvq_il_supported(type, n, n_out)) {
            if (!il_ready && n >= 2) {
                K::native_q8_1_interleave(b.xq, b.xil, n_in, n, s);
                il_ready = true;
            }
            K::native_mmvq_il(type, w, b.xq, b.xil, out, n_in, n_out, n, s);
        } else {
            K::native_mmvq(type, w, b.xq, out, n_in, n_out, n, s);
        }
    }
};

void hc_read(const HcW& h, const Geo& G, Bufs& b, int n, bool apply, const float* bo_prev, const float* inj_prev,
             float* inj_out, cudaStream_t s) {
    K::FusedGrArgs fa[K::kFusedGrWideT];
    for (int t = 0; t < n; ++t) {
        K::FusedGrArgs& a = fa[t];
        a.R = b.R + (size_t) t * G.HC * G.N;
        a.R_out = b.R + (size_t) t * G.HC * G.N;
        a.apply = apply;
        a.bo_prev = bo_prev + (size_t) t * G.N;
        a.inj_prev = inj_prev + (size_t) t * G.HC;
        a.w_norm = h.norm;
        a.w_down = h.down;
        a.w_up = h.up;
        a.w_inject = h.inject;
        a.eps = EPS;
        a.lo = b.lo + (size_t) t * G.HCLR;
        a.rs = b.rs + (size_t) t * G.HC;
        a.inject_out = inj_out + (size_t) t * G.HC;
        a.mixed = b.mixed + (size_t) t * G.N;
    }
    K::fused_gr_read_multi(fa, n, b.xn, s);
}

// the attention half: the hyper-connection read, then the GDN or QSA mixer of this variant -> b.bo (this card's partial)
void record_mixer(const CommonW& c, Part& P, const Geo& G, Bufs& b, int n, bool apply, const float* bo_prev,
                  const float* inj_prev, cudaStream_t s) {
    hc_read(c.hc[0], G, b, n, apply, bo_prev, inj_prev, b.inj, s);
    float* xm = b.mixed;
    Quant Q{b, n, s};
    const int N = (int) G.N;
    if (!P.attn) {
        float* conv_hist = P.state + (size_t) (G.S * P.HV * G.S);
        Q.in(xm, N);
        Q.mm(P.qkv_t, P.qkv, b.qkv, N, P.C);
        K::gdn_conv_l2_multi(conv_hist, b.qkv, P.conv, b.hb, P.C, 2 * P.HK, EPS, n, s, 0, false);
        K::gdn_ab_multi(xm, P.alpha, P.beta, P.dt, P.ssm_a, b.gate, b.beta, N, P.HV, n, s);
        Q.mm(P.g_t, P.gw, b.z, N, P.ZV);
        K::gdn_step_norm_multi(P.state, b.hb, P.C, b.gate, b.beta, b.z, P.norm, EPS, b.y, P.HK, P.HV, n, nullptr, s, 0,
                               nullptr);
        Q.in(b.y, P.ZV);
        Q.mm(P.out_t, P.out, b.bo, P.ZV, N);
        return;
    }
    // ---- QSA (verify.cpp's in-line order: idx key, k/v + norm-rope, KV append, idx snapshot + append, q chain, idx
    // query, select, attention, gate, out-projection)
    const int HD = (int) G.HD, NH = P.NH, NKV = P.NKV, IQ = (int) G.IQ, ID = (int) G.ID;
    const K::QsaShapes& sh = P.s;
    const bool qb = n > 1 && K::native_qsa_enabled() && K::native_rope_enabled();
    const bool fuse_nr = K::native_qsa_enabled() && K::native_rope_enabled() && K::native_norm_rope_usable(HD, (int) sh.n_rot);
    const int32_t* pos_q = P.pos;
    const int32_t* pos_k = P.pos + MT * NH;
    const int32_t* pos_i = P.pos + MT * (NH + NKV);
    auto norm_rope_on = [&](float* data, const float* norm, int rows, int cols, const int32_t* pos) {
        K::native_qsa_rms_norm_weighted(data, norm, data, cols, rows, EPS, s);
        K::native_rope_apply(data, data, rows, cols, (int) sh.n_rot, K::rope_scaling(), pos, s);
    };
    auto norm_rope = [&](float* data, const float* norm, int rows, int cols, const int32_t* pos) {
        if (fuse_nr && K::native_norm_rope_usable(cols, (int) sh.n_rot)) {
            K::native_qsa_rms_norm_rope(data, cols, norm, data, rows, cols, (int) sh.n_rot, EPS, K::rope_scaling(), pos, s);
            return;
        }
        norm_rope_on(data, norm, rows, cols, pos);
    };
    Q.in(xm, N);
    if (qb) K::bf16_gemv_fp32_mmvf_multi(xm, N, c.ik, b.idx_raw, ID, N, ID, n, s);
    else
        for (int t = 0; t < n; ++t) K::bf16_gemv_fp32_mmvf(xm + (size_t) t * N, c.ik, b.idx_raw + (size_t) t * ID, N, ID, s);
    Q.mm(P.k_t, P.k, b.kcur, N, NKV * HD);
    Q.mm(P.v_t, P.v, b.vcur, N, NKV * HD);
    if (qb) norm_rope(b.kcur, P.kn, n * NKV, HD, pos_k);
    else
        for (int t = 0; t < n; ++t) norm_rope(b.kcur + (size_t) t * NKV * HD, P.kn, NKV, HD, pos_q + (size_t) t * NH);
    K::copy_from_mapped(P.tail_snap, P.itail, (int64_t) ((sh.idx_block - 1) * ID), s);
    for (int t = 0; t < n; ++t)   // the FP16 cache: one append per row (verify.cpp's kv_append_step loop)
        K::kv_append_step(P.kpool, P.vpool, P.ptab, b.step + (size_t) t * K::kStepCount, b.kcur + (size_t) t * NKV * HD,
                          b.vcur + (size_t) t * NKV * HD, sh, s, nullptr);
    const K::QsaIndexerBuffers ib{P.itail, P.idead, P.ipooled, P.iblockpos};
    K::native_qsa_indexer_append_steps(b.idx_raw, b.step + K::kStepPos, K::kStepCount, n, 0, c.ikn, EPS, ib, sh,
                                       P.max_cells, K::rope_scaling(), s);
    // the query chain
    Q.mm(P.q_t, P.q, b.qfull, N, NH * 2 * HD);
    if (qb) {
        if (fuse_nr)
            K::native_qsa_rms_norm_rope(b.qfull, 2 * HD, P.qn, b.qcur, n * NH, HD, (int) sh.n_rot, EPS, K::rope_scaling(),
                                        pos_q, s);
        else {
            K::copy_rows_strided(b.qcur, b.qfull, (int64_t) n * NH, HD, 2 * HD, s);
            norm_rope(b.qcur, P.qn, n * NH, HD, pos_q);
        }
    } else {
        for (int t = 0; t < n; ++t) {
            float* qc = b.qcur + (size_t) t * NH * HD;
            if (fuse_nr)
                K::native_qsa_rms_norm_rope(b.qfull + (size_t) t * NH * 2 * HD, 2 * HD, P.qn, qc, NH, HD, (int) sh.n_rot,
                                            EPS, K::rope_scaling(), pos_q + (size_t) t * NH, s);
            else {
                K::copy_rows_strided(qc, b.qfull + (size_t) t * NH * 2 * HD, NH, HD, 2 * HD, s);
                norm_rope(qc, P.qn, NH, HD, pos_q + (size_t) t * NH);
            }
        }
    }
    if (qb) {
        K::bf16_gemv_fp32_mmvf_multi(xm, N, c.iq, b.qidx, IQ * ID, N, IQ * ID, n, s);
        norm_rope(b.qidx, c.iqn, n * IQ, ID, pos_i);
    } else
        for (int t = 0; t < n; ++t) {
            float* qx = b.qidx + (size_t) t * IQ * ID;
            K::bf16_gemv_fp32_mmvf(xm + (size_t) t * N, c.iq, qx, N, IQ * ID, s);
            norm_rope(qx, c.iqn, IQ, ID, pos_q + (size_t) t * NH);
        }
    K::qsa_block_scores(P.ipooled, P.idead, b.qidx, b.step, n, P.max_blocks, sh, P.scores, s);
    K::qsa_block_topk(P.scores, b.step, n, P.max_blocks, P.cap, sh, P.sel, s);
    K::QsaAttnPools pools;
    pools.k_pool = P.kpool;
    pools.v_pool = P.vpool;
    pools.page_table = P.ptab;
    K::qsa_decode_attn_batch(b.qcur, pools, P.sel, b.step, P.cap, sh, P.attn_scratch, b.attn, n, s);
    if (qb) K::native_qsa_gate_apply(b.attn, b.qfull, b.attn32, n * NH, HD, s);
    else
        for (int t = 0; t < n; ++t)
            K::native_qsa_gate_apply(b.attn + (size_t) t * NH * HD, b.qfull + (size_t) t * NH * 2 * HD,
                                     b.attn32 + (size_t) t * NH * HD, NH, HD, s);
    Q.in(b.attn32, NH * HD);
    Q.mm(P.o_t, P.o, b.bo, NH * HD, N);
}

// the FFN half up to the experts: the hyper-connection read (folding the mixer's write), the router, the shared expert
// (on its card), the experts' q8_1 input
void record_ffn_pre(const CommonW& c, const Part& P, const Geo& G, Bufs& b, int n, cudaStream_t s) {
    hc_read(c.hc[1], G, b, n, true, b.bo, b.inj, b.inj2, s);
    const float* xm = b.mixed;
    const int N = (int) G.N;
    K::bf16_gemv_fp32_mmvf_multi(xm, N, c.router, b.logits, G.NE, N, G.NE, n, s);
    K::native_router_top10_multi(b.logits, b.ids, b.w, n, s);
    K::resident_plan(b.ids, n * (int) G.K, (int) G.K, P.res, (int) G.NE, c.experts, nullptr, (long long) c.L.bytes, P.rplan,
                     (long long) (MT * G.K), P.rskip, 1u, s);
    if (P.has_shared) {
        K::NativeSharedWeights nsw;
        nsw.gate_type = c.shg_t;
        nsw.gate_data = c.shg;
        nsw.up_type = c.shu_t;
        nsw.up_data = c.shu;
        nsw.down_type = c.shd_t;
        nsw.down_data = c.shd;
        nsw.q8_1 = b.xq;
        K::shared_expert_multi(n, xm, b.sh_bf16, nsw, c.sh_gate_inp, b.sh_gate, b.sh_up, b.sh_g, b.shared, G.N, G.NFF, s,
                               nullptr, 0);
    }
    K::quantize_q8_1_rows(xm, n, G.N, b.nat_xq, s);
}

// the experts of this card's plan and the combine -> b.bo_out (this card's partial: its experts, plus the shared
// expert on its card; the other cards' rows are zeros, the shared row zeros off its card)
void record_ffn_post(const CommonW& c, Part& P, const Geo& G, Bufs& b, int n, cudaStream_t s) {
    const int64_t cap = (int64_t) n * G.K;
    K::native_expert_grouped(c.L, P.plan.grp_ptr, P.plan.grp_start, P.plan.n_groups, P.plan.ent_dst, P.plan.ent_tok, cap,
                             cap, b.nat_xq, b.exp_scratch, P.parts, s);
    K::native_moe_combine_multi(P.parts, b.w, P.has_shared ? b.shared : b.shared_zero, b.bo_out, G.N, G.K, n, s);
}

void set_engine_defaults() {
    // the --native kernel set generate.cpp switches on for a GGUF model (process-wide), and the gfx1100 IQ4 lookup flag
    // (xtx_fast_perm_startup: once per process, every visible device, before any capture)
    K::xtx_fast_perm_startup();
    K::gr_set_native_mmvf(true);
    K::shared_expert_set_native_bf16(true);
    K::native_moe_combine_set_enabled(true);
    K::native_gdn_set_enabled(true);
    K::native_router_set_enabled(true);
    K::native_qsa_set_enabled(true);
    K::native_qsa_indexer_set_enabled(true);
    K::native_rope_set_enabled(true);
}

double time_graph(cudaStream_t s, const std::function<void()>& rec, int iters, const std::string& what) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    try {
        rec();
    } catch (const std::exception& e) {
        cudaStreamEndCapture(s, &g);
        die(what + ": " + e.what());
    }
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < 10; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    std::vector<double> us;
    us.reserve((size_t) iters);
    for (int r = 0; r < iters; ++r) {
        CK(cudaEventRecord(e0, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        us.push_back(1e3 * ms);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return median(us);
}

std::vector<int32_t> route_once(const CommonW& c, Part& full, const Geo& G, Bufs& b, int n, cudaStream_t s) {
    CK(cudaMemcpyAsync(b.R, b.R0, (size_t) (MT * G.HC * G.N) * 4, cudaMemcpyDeviceToDevice, s));
    try {
        record_mixer(c, full, G, b, n, true, b.bo_zero, b.inj_in, s);
        record_ffn_pre(c, full, G, b, n, s);
    } catch (const std::exception& e) {
        die(std::string("the routing run: ") + e.what());
    }
    CK(cudaStreamSynchronize(s));
    std::vector<int32_t> ids((size_t) (n * G.K));
    CK(cudaMemcpy(ids.data(), b.ids, ids.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpyAsync(b.R, b.R0, (size_t) (MT * G.HC * G.N) * 4, cudaMemcpyDeviceToDevice, s));
    CK(cudaStreamSynchronize(s));
    return ids;
}

// =====================================================================================================================
// split-time
// =====================================================================================================================
int cmd_split_time(const Args& a) {
    if (!a.has("gguf") || !a.has("layers")) die("split-time needs --gguf <first shard> --layers L1,L2,...");
    const int dev = std::atoi(a.get("device", "0").c_str());
    const std::vector<int> layers = parse_ints(a.get("layers"));
    const std::vector<int> Ts = parse_ints(a.get("T", "1,3,8"));
    const int iters = std::atoi(a.get("iters", "200").c_str());
    const int64_t ctx = std::atoll(a.get("ctx", "4096").c_str());
    for (int T : Ts)
        if (T < 1 || T > MT) die("T must be 1.." + std::to_string(MT));
    Model M;
    open_model(M, a.get("gguf"));
    const Geo& G = M.geo;
    set_engine_defaults();
    CK(cudaSetDevice(dev));
    K::fused_gr_check();   // once per card (Verifier::init does): picks the hyper-connection read's variant
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Bufs b;
    make_bufs(b, G, ctx);
    set_inputs(b, G, 11u);
    double worst_half3 = -1.0;
    int worst_layer = -1;
    double wsum_full3 = 0, wsum_half3 = 0, wsum_quarter3 = 0;
    bool have3 = false;
    for (int l : layers) {
        if (l < 0 || l >= G.n_layers) die("layer " + std::to_string(l) + " is out of range");
        const bool attn = G.attn(l);
        std::fprintf(stderr, "# layer %d (%s): loading\n", l, attn ? "attn" : "gdn");
        CommonW c;
        load_common(c, M, l, true);
        const HostState hs = make_state(G, l, ctx);
        std::vector<std::unique_ptr<Part>> parts;   // [0] full, [1..2] tp2, [3..6] tp4
        const int cfg[7][2] = {{1, 0}, {2, 0}, {2, 1}, {4, 0}, {4, 1}, {4, 2}, {4, 3}};
        for (const auto& cf : cfg) {
            parts.push_back(std::make_unique<Part>());
            load_part(*parts.back(), M, l, cf[0], cf[1], hs);
            set_positions(*parts.back(), G, ctx);
        }
        for (int T : Ts) {
            const std::vector<int32_t> ids = route_once(c, *parts[0], G, b, T, s);
            double t_var[7] = {};
            for (size_t pi = 0; pi < parts.size(); ++pi) {
                Part& P = *parts[pi];
                build_plan(P, c, G, ids, T, s);
                const std::string what = "layer " + std::to_string(l) + " T " + std::to_string(T) + " tp " +
                                         std::to_string(P.tp) + " card " + std::to_string(P.card);
                t_var[pi] = time_graph(s, [&] {
                    record_mixer(c, P, G, b, T, true, b.bo_zero, b.inj_in, s);
                    record_ffn_pre(c, P, G, b, T, s);
                    record_ffn_post(c, P, G, b, T, s);
                }, iters, what);
                std::printf("TPP split-card layer=%d kind=%s T=%d tp=%d card=%d us=%.2f shared=%d expert_groups=%d "
                            "expert_entries=%d\n",
                            l, attn ? "attn" : "gdn", T, P.tp, P.card, t_var[pi], P.has_shared ? 1 : 0, P.plan.groups,
                            P.plan.entries);
            }
            const double full = t_var[0], half = std::max(t_var[1], t_var[2]),
                         quarter = std::max({t_var[3], t_var[4], t_var[5], t_var[6]});
            std::printf("TPP split layer=%d kind=%s T=%d full_us=%.2f half_us=%.2f quarter_us=%.2f half_ratio=%.3f "
                        "quarter_ratio=%.3f\n",
                        l, attn ? "attn" : "gdn", T, full, half, quarter, half / full, quarter / full);
            std::fflush(stdout);
            if (T == 3) {
                have3 = true;
                if (half / full > worst_half3) { worst_half3 = half / full; worst_layer = l; }
                const double wgt = attn ? 1.0 : 3.0;   // 36 GDN : 12 attention layers
                wsum_full3 += wgt * full;
                wsum_half3 += wgt * half;
                wsum_quarter3 += wgt * quarter;
            }
        }
        // (the device memory of this layer is kept until exit: a measurement run, two layers)
    }
    if (!have3) {
        std::printf("TPP VERDICT split NONE (T=3 was not measured)\n");
        return 0;
    }
    std::printf("TPP split-weighted T=3 gdn:attn=3:1 half_ratio=%.3f quarter_ratio=%.3f (layers measured, weighted as 36 "
                "GDN + 12 attention)\n",
                wsum_half3 / wsum_full3, wsum_quarter3 / wsum_full3);
    std::printf("TPP VERDICT split %s (worst half_ratio at T=3 = %.3f, layer %d; FAIL above 0.75)\n",
                worst_half3 > 0.75 ? "FAIL" : "PASS", worst_half3, worst_layer);
    return 0;
}

// =====================================================================================================================
// ar-time
// =====================================================================================================================
int cmd_ar_time(const Args& a) {
    const std::vector<int> devs = parse_ints(a.get("devices", "0,1"));
    const std::vector<int> bytes = parse_ints(a.get("bytes", "10240,20480,81920"));
    const int iters = std::atoi(a.get("iters", "2000").c_str());
    const int warm = 50;
    if (devs.size() != 2) die("ar-time takes two devices (--devices 0,1)");
    int max_floats = 0;
    for (int bb : bytes) {
        if (bb <= 0 || bb % 16 != 0) die("--bytes: each size must be a positive multiple of 16");
        max_floats = std::max(max_floats, bb / 4);
    }
    ArGroup g;
    ar_open(g, devs, (size_t) max_floats);
    const unsigned stamp_cap = (unsigned) (iters + warm + 16);
    struct DevRun {
        cudaStream_t s = nullptr;
        float *x = nullptr, *src = nullptr, *out = nullptr;
        unsigned long long *t0 = nullptr, *t1 = nullptr;
        std::vector<float> src_h;
    };
    std::vector<DevRun> dr(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        CK(cudaSetDevice(devs[i]));
        CK(cudaStreamCreateWithFlags(&dr[i].s, cudaStreamNonBlocking));
        dr[i].x = dalloc<float>((size_t) max_floats);
        dr[i].out = dalloc<float>((size_t) max_floats);
        dr[i].src_h.resize((size_t) max_floats);
        for (int k = 0; k < max_floats; ++k) dr[i].src_h[(size_t) k] = (float) ((k * 7 + (int) i * 13) % 101) / 64.0f;
        dr[i].src = dup(dr[i].src_h);
        dr[i].t0 = dalloc<unsigned long long>(stamp_cap);
        dr[i].t1 = dalloc<unsigned long long>(stamp_cap);
    }
    bool any20480 = false;
    double med20480 = 0.0;
    for (int nb : bytes) {
        const int n = nb / 4;
        ar_reset(g);
        std::vector<cudaGraphExec_t> ge(devs.size());
        for (size_t i = 0; i < devs.size(); ++i) {
            CK(cudaSetDevice(devs[i]));
            cudaGraph_t gr;
            CK(cudaStreamBeginCapture(dr[i].s, cudaStreamCaptureModeThreadLocal));
            TPP_LAUNCH(tpp_produce_kernel, (n + 255) / 256, 256, dr[i].s, dr[i].x, dr[i].src, n);
            ar_call(g, (int) i, dr[i].x, n, dr[i].s, dr[i].t0, dr[i].t1, stamp_cap);
            TPP_LAUNCH(tpp_consume_kernel, (n + 255) / 256, 256, dr[i].s, dr[i].x, dr[i].out, n);
            CK(cudaStreamEndCapture(dr[i].s, &gr));
            CK(cudaGraphInstantiate(&ge[i], gr, nullptr, nullptr, 0));
            CK(cudaGraphDestroy(gr));
        }
        Watchdog wd;
        wd.start(g, 5.0, "ar-time");
        std::atomic<int> ready{0};
        std::vector<double> loop_us(devs.size(), 0.0);
        std::vector<std::thread> th;
        for (size_t i = 0; i < devs.size(); ++i)
            th.emplace_back([&, i] {
                CK(cudaSetDevice(devs[i]));
                ready.fetch_add(1);
                while (ready.load() < (int) devs.size()) std::this_thread::yield();
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < warm + iters; ++r) CK(cudaGraphLaunch(ge[i], dr[i].s));
                CK(cudaStreamSynchronize(dr[i].s));
                loop_us[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
                             (double) (warm + iters);
            });
        for (auto& t : th) t.join();
        wd.finish();
        check_ar_err(g, "ar-time");
        // the result: every member holds src0 + src1 in member order, the same bits
        bool ok = true;
        std::vector<std::vector<unsigned long long>> t0h(devs.size()), t1h(devs.size());
        for (size_t i = 0; i < devs.size(); ++i) {
            CK(cudaSetDevice(devs[i]));
            std::vector<float> xh((size_t) n);
            CK(cudaMemcpy(xh.data(), dr[i].x, (size_t) n * 4, cudaMemcpyDeviceToHost));
            for (int k = 0; k < n && ok; ++k) {
                const float want = dr[0].src_h[(size_t) k] + dr[1].src_h[(size_t) k];
                if (std::memcmp(&xh[(size_t) k], &want, 4) != 0) ok = false;
            }
            t0h[i].resize(stamp_cap);
            t1h[i].resize(stamp_cap);
            CK(cudaMemcpy(t0h[i].data(), dr[i].t0, (size_t) stamp_cap * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(t1h[i].data(), dr[i].t1, (size_t) stamp_cap * 8, cudaMemcpyDeviceToHost));
            CK(cudaGraphExecDestroy(ge[i]));
        }
        std::vector<double> worst, best, per_dev[2];
        for (int r = warm + 1; r <= warm + iters; ++r) {
            const unsigned k = (unsigned) r % stamp_cap;
            double mx = 0, mn = 1e30;
            for (size_t i = 0; i < devs.size(); ++i) {
                const double us = (double) (t1h[i][k] - t0h[i][k]) * 1e-3;
                per_dev[i].push_back(us);
                mx = std::max(mx, us);
                mn = std::min(mn, us);
            }
            worst.push_back(mx);
            best.push_back(mn);
        }
        const double med = median(worst);
        std::printf("TPP ar bytes=%d median_us=%.2f p99_us=%.2f\n", nb, med, pct(worst, 0.99));
        std::printf("TPP ar-detail bytes=%d iters=%d sum_check=%s slower_member_median_us=%.2f "
                    "faster_member_median_us=%.2f dev%d_median_us=%.2f dev%d_median_us=%.2f host_us_per_graph=%.2f/%.2f\n",
                    nb, iters, ok ? "ok" : "MISMATCH", med, median(best), devs[0], median(per_dev[0]), devs[1],
                    median(per_dev[1]), loop_us[0], loop_us[1]);
        std::fflush(stdout);
        if (nb == 20480) {
            any20480 = true;
            med20480 = med;
        }
    }
    if (!any20480) std::printf("TPP VERDICT ar NONE (20480 bytes was not measured)\n");
    else
        std::printf("TPP VERDICT ar %s (median at 20480 bytes = %.2f us; FAIL above 40)\n", med20480 > 40.0 ? "FAIL" : "PASS",
                    med20480);
    return 0;
}

// =====================================================================================================================
// exact
// =====================================================================================================================
struct DevModel {   // one device's layers for exact: the replicated weights and its variant(s) of each layer
    int dev = 0;
    cudaStream_t s = nullptr;
    Bufs b;
    std::vector<CommonW> common;
    std::vector<std::unique_ptr<Part>> parts;
};

// one device's pass over the layers: mixer, [all-reduce], FFN up to the router, the plan from the routing, experts +
// combine, [all-reduce]; then the last layer's write into R.  Returns R (T rows) and the routing of every layer.
std::vector<float> run_layers(DevModel& D, const Geo& G, const std::vector<int>& layers, int T, const ArGroup* ar, int mi,
                              std::vector<std::vector<int32_t>>& routing) {
    Bufs& b = D.b;
    cudaStream_t s = D.s;
    CK(cudaMemcpyAsync(b.R, b.R0, (size_t) (MT * G.HC * G.N) * 4, cudaMemcpyDeviceToDevice, s));
    routing.clear();
    const float* bo_prev = b.bo_zero;
    const float* inj_prev = b.inj_in;
    try {
        for (size_t li = 0; li < layers.size(); ++li) {
            const CommonW& c = D.common[li];
            Part& P = *D.parts[li];
            // the first layer of the chain reads R as it is (no pending write); later ones fold the previous layer's
            // FFN write (bo_out, inj2) into their read, as verify.cpp's pending read does
            record_mixer(c, P, G, b, T, li > 0, bo_prev, inj_prev, s);
            if (ar) ar_call(*ar, mi, b.bo, T * (int) G.N, s);
            record_ffn_pre(c, P, G, b, T, s);
            CK(cudaStreamSynchronize(s));
            if (ar) check_ar_err(*ar, "exact");
            std::vector<int32_t> ids((size_t) (T * G.K));
            CK(cudaMemcpy(ids.data(), b.ids, ids.size() * 4, cudaMemcpyDeviceToHost));
            routing.push_back(ids);
            build_plan(P, c, G, ids, T, s);
            record_ffn_post(c, P, G, b, T, s);
            if (ar) ar_call(*ar, mi, b.bo_out, T * (int) G.N, s);
            CK(cudaStreamSynchronize(s));
            if (ar) check_ar_err(*ar, "exact");
            bo_prev = b.bo_out;
            inj_prev = b.inj2;
        }
        K::GrShapes gs{G.N, G.HC, G.HCLR};
        K::gr_write_multi(b.R, b.bo_out, b.inj2, gs, b.R, T, s);
    } catch (const std::exception& e) {
        die(std::string("exact: ") + e.what());
    }
    CK(cudaStreamSynchronize(s));
    std::vector<float> R((size_t) (T * G.HC * G.N));
    CK(cudaMemcpy(R.data(), b.R, R.size() * 4, cudaMemcpyDeviceToHost));
    return R;
}

int cmd_exact(const Args& a) {
    if (!a.has("gguf") || !a.has("layers")) die("exact needs --gguf <first shard> --layers 0,1,2,3 --devices 0,1");
    const std::vector<int> layers = parse_ints(a.get("layers"));
    const std::vector<int> devs = parse_ints(a.get("devices", "0,1"));
    const int T = std::atoi(a.get("T", "3").c_str());
    const int64_t ctx = std::atoll(a.get("ctx", "4096").c_str());
    if (devs.size() != 2) die("exact takes two devices (--devices 0,1)");
    if (T < 1 || T > MT) die("T must be 1.." + std::to_string(MT));
    for (size_t i = 1; i < layers.size(); ++i)
        if (layers[i] != layers[i - 1] + 1) die("exact: the layers must be consecutive");
    Model M;
    open_model(M, a.get("gguf"));
    const Geo& G = M.geo;
    // device 0 holds the one-card model and member 0 of the split (the same replicated weights); device 1 member 1
    DevModel D[3];   // [0] full on devs[0], [1] member 0 on devs[0], [2] member 1 on devs[1]
    set_engine_defaults();
    for (int k = 0; k < 3; ++k) {
        D[k].dev = devs[k == 2 ? 1 : 0];
        CK(cudaSetDevice(D[k].dev));
        if (k != 1) K::fused_gr_check();
        CK(cudaStreamCreateWithFlags(&D[k].s, cudaStreamNonBlocking));
        make_bufs(D[k].b, G, ctx);
        set_inputs(D[k].b, G, 21u);   // the same fixed inputs everywhere
    }
    for (int l : layers) {
        if (l < 0 || l >= G.n_layers) die("layer " + std::to_string(l) + " is out of range");
        std::fprintf(stderr, "# layer %d (%s): loading\n", l, G.attn(l) ? "attn" : "gdn");
        const HostState hs = make_state(G, l, ctx);
        for (int k = 0; k < 3; ++k) {
            CK(cudaSetDevice(D[k].dev));
            if (k == 1) D[1].common.push_back(D[0].common.back());   // the same device: share the replicated weights
            else {
                D[k].common.emplace_back();
                load_common(D[k].common.back(), M, l, true);
            }
            D[k].parts.push_back(std::make_unique<Part>());
            load_part(*D[k].parts.back(), M, l, k == 0 ? 1 : 2, k == 0 ? 0 : k - 1, hs);
            set_positions(*D[k].parts.back(), G, ctx);
        }
    }
    // one card
    std::vector<std::vector<int32_t>> route_full, route_m[2];
    CK(cudaSetDevice(D[0].dev));
    const std::vector<float> full = run_layers(D[0], G, layers, T, nullptr, 0, route_full);
    // two cards, one host thread each, the all-reduce between them
    ArGroup g;
    ar_open(g, devs, (size_t) (MT * G.N));
    std::vector<float> out[2];
    Watchdog wd;
    wd.start(g, 5.0 + 120.0, "exact");   // (the host builds a plan between all-reduces; the device waits stay at 5 s)
    std::vector<std::thread> th;
    for (int m = 0; m < 2; ++m)
        th.emplace_back([&, m] {
            CK(cudaSetDevice(D[1 + m].dev));
            out[m] = run_layers(D[1 + m], G, layers, T, &g, m, route_m[m]);
        });
    for (auto& t : th) t.join();
    wd.finish();
    check_ar_err(g, "exact");
    // compare
    double maxabs = 0, maxref = 0, maxrel_el = 0;
    bool bitident = true;
    for (size_t i = 0; i < full.size(); ++i) {
        const double d = std::fabs((double) full[i] - (double) out[0][i]);
        maxabs = std::max(maxabs, d);
        maxref = std::max(maxref, std::fabs((double) full[i]));
        if (std::memcmp(&full[i], &out[0][i], 4) != 0) bitident = false;
    }
    for (size_t i = 0; i < full.size(); ++i) {   // elementwise, over the elements not near zero
        const double ref = std::fabs((double) full[i]);
        if (ref > 1e-3 * maxref) maxrel_el = std::max(maxrel_el, std::fabs((double) full[i] - (double) out[0][i]) / ref);
    }
    const bool cards_agree = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
    bool route_cards = route_m[0] == route_m[1], route_full_same = route_full == route_m[0];
    int nonfinite = 0;
    for (float v : full) nonfinite += std::isfinite(v) ? 0 : 1;
    for (float v : out[0]) nonfinite += std::isfinite(v) ? 0 : 1;
    int n_attn = 0;
    for (int l : layers) n_attn += G.attn(l) ? 1 : 0;
    std::printf("TPP exact layers=%d-%d attn_layers=%d bitident=%s maxabs=%.6g maxrel=%.6g\n", layers.front(),
                layers.back(), n_attn, bitident ? "y" : "n", maxabs, maxref > 0 ? maxabs / maxref : 0.0);
    std::printf("TPP exact-detail T=%d compared=R_after_last_layer(%zu floats) max_abs_ref=%.6g maxrel=maxabs/max|ref| "
                "maxrel_elementwise(|ref|>1e-3*max)=%.6g split_cards_bitident=%s router_picks_same_across_cards=%s "
                "router_picks_same_as_full=%s nonfinite=%d\n",
                T, full.size(), maxref, maxrel_el, cards_agree ? "y" : "n", route_cards ? "y" : "n",
                route_full_same ? "y" : "n", nonfinite);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  tp_proof_bench split-time --gguf <first shard> --layers L1,L2 [--device 0] [--T 1,3,8] "
                     "[--iters 200] [--ctx 4096]\n"
                     "  tp_proof_bench ar-time --devices 0,1 [--bytes 10240,20480,81920] [--iters 2000]\n"
                     "  tp_proof_bench exact --gguf <first shard> --layers 0,1,2,3 --devices 0,1 [--T 3] [--ctx 4096]\n");
        return 2;
    }
    const std::string cmd = argv[1];
    Args a;
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) die("unexpected argument " + k);
        k = k.substr(2);
        if (i + 1 >= argc) die("--" + k + " needs a value");
        a.kv[k] = argv[++i];
    }
    if (cmd == "split-time") return cmd_split_time(a);
    if (cmd == "ar-time") return cmd_ar_time(a);
    if (cmd == "exact") return cmd_exact(a);
    die("unknown subcommand " + cmd);
}
