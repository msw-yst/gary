// Strata prototype (docs/dsv41/cards/EB-01.md, phase 1). src/dsv41/ds4_helper.cu - the helper expert cache's host
// side (ds4_helper.hpp).  This TU is compiled for the primary's architecture(s) and, with STRATA_HIP_HELPER_ARCHS, for
// the helper's (gfx906) as well; it launches no kernel of its own: the expert work is gg::win_experts (ds4_gguf.cu,
// which the same CMake rule compiles for both), the same entry point the primary's window path takes, so a helper
// expert gives the bits a primary slot would (ds4_iq.cuh: "a window's entry gets the token path's bits").
#include "ds4_helper.hpp"
#include "ds4_os.hpp"   // malloc_pin_thread (the pool server's arena)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <numeric>
#include <thread>
#include <map>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define DS4_POOL_PAUSE() _mm_pause()
#else
#define DS4_POOL_PAUSE() std::this_thread::yield()
#endif

namespace strata::ds4::helper {

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// a request's plan for the helper, host-written, one copy to the device: the window's picks it owns as win_experts
// takes them (plan_ptr[t][i] the blob or 0, plan_w = Req::w, one group an expert with its entries t * 8 + i)
struct Plan {
    unsigned long long plan_ptr[kWin * kPicks];
    float plan_w[kWin * kPicks];
    unsigned long long grp_ptr[kWin * kPicks];
    int grp_start[kWin * kPicks + 1];
    int grp_ent[kWin * kPicks];
    int n_groups;
};

// (docs/dsv41/cards/HX-06.md): the round trip's mechanism, one switch a stage (read once at open)
//   DS4_HELPER_IN   copy2 (plan H2D + x H2D: the old path) | copy1 (plan + x packed, one H2D) |
//                   kfetch (HIP default: a kernel pulls the packed plan + x from fine-grained pinned host memory: no SDMA) |
//                   zc (the expert kernels read plan + x in fine-grained pinned host memory)
//   DS4_HELPER_OUT  d2h (a D2H copy: the old path) | zc (HIP default: combine writes into fine-grained pinned host memory)
//   DS4_HELPER_WAIT sync (cudaEventSynchronize: the old path) | query (spin on cudaEventQuery) | flag (HIP default: a 1-thread
//                   kernel writes the window's sequence number into pinned host memory after the last step; the host spins)
//   DS4_HELPER_GRAPH=1 (HIP default; 0 off) each (layer, T)'s sequence captured once into a hipGraph and replayed (not with copy2)
//   Non-HIP builds keep the old path (copy2 / d2h / sync / no graph).
//   DS4_HELPER_TRACE=1  timing events between the stages and host timestamps: a per-window breakdown on the end-of-run line
constexpr size_t kPlanB = (sizeof(Plan) + 255) / 256 * 256;   // packed staging: [Plan | x rows kMaxDim apart]

__global__ void helper_fetch_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
// DS4_PF_HELPER: a prompt chunk's x rows for this helper - row j of the device copy is row tok[j] of the shared
// fine-grained staging (only the tokens with a pick here cross the link).  grid (n4 / 256 rounded up, nt)
__global__ void pf_gather_kernel(const float4* __restrict__ hx, const int* __restrict__ tok, int n4, float4* __restrict__ dx) {
    const int j = blockIdx.y;
    const size_t src = (size_t) tok[j] * (size_t) n4, dst = (size_t) j * (size_t) n4;
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) dx[dst + q] = hx[src + q];
}
// token j's weighted sum of its picks' down rows, in pick order (pos[j][i] the row, -1 not here), from 0 - into the
// fine-grained out row j.  grid (n / 256 rounded up, nt)
__global__ void pf_combine_kernel(const float* __restrict__ y, const int* __restrict__ pos, const float* __restrict__ w, int k, int n,
                                  float* __restrict__ out) {
    const int j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float a = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) j * kPicks + i];
        if (r >= 0) a = __fmaf_rn(w[(size_t) j * kPicks + i], y[(size_t) r * n + c], a);
    }
    out[(size_t) j * n + c] = a;
}

__global__ void helper_flag_kernel(volatile unsigned long long* f, unsigned long long v) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        *f = v;
        __threadfence_system();
    }
}

// DS4_HELPER_BATCH=1 (docs/dsv41/cards/HX-10.md): kfetch from two sources - the helper's own staging's plan (np16 uint4s) and
// the x rows every helper of the request reads from one shared staging (nx16) - into the same [Plan | x] device layout.
// go (an armed window's body, may be null): as helper_fetch_go_kernel - nothing copied and an empty plan when *go is 0.
__global__ void helper_fetch2_kernel(const uint4* __restrict__ plan, size_t np16, const uint4* __restrict__ x, size_t nx16,
                                     uint4* __restrict__ dst, const int* go) {
    if (go != nullptr && *(const volatile int*) go == 0) {
        if (blockIdx.x != 0) return;
        Plan* p = reinterpret_cast<Plan*>(dst);
        for (int i = threadIdx.x; i < kWin * kPicks; i += blockDim.x) p->plan_ptr[i] = 0ull;
        if (threadIdx.x == 0) p->n_groups = 0;
        return;
    }
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < np16 + nx16; i += stride)
        dst[i] = i < np16 ? plan[i] : x[i - np16];
}

#if defined(STRATA_USE_HIP)
// DS4_HELPER_ARMED (docs/dsv41/cards/HX-08.md, docs/dsv41/cards/HX-09.md): an armed window's head. One thread spins on
// the lane's command word (fine-grained host memory) until it names this window (cmd >> 1 == want; bit 0: go) or the
// wall-clock bound passes. Then *go (device) = 1 only for an accepted window, and *ack (host) = want * 4 + status
// (1 go, 2 cancelled, 3 timed out), fenced, so the lane can tell a window that computed from one that did not.
__global__ void helper_await_kernel(const volatile unsigned long long* cmd, unsigned long long want, int* go,
                                    volatile unsigned long long* ack, unsigned long long ticks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned long long t0 = (unsigned long long) wall_clock64();
    int stat = 3;
    for (;;) {
        const unsigned long long c = *cmd;
        if ((c >> 1) == want) {
            stat = (c & 1ull) ? 1 : 2;
            break;
        }
        if ((unsigned long long) wall_clock64() - t0 > ticks) break;
        __builtin_amdgcn_s_sleep(1);
    }
    *(volatile int*) go = stat == 1 ? 1 : 0;
    __threadfence_system();
    *ack = want * 4ull + (unsigned long long) stat;
    __threadfence_system();
}
// kfetch's copy when *go, else an empty plan (no group, no expert pointer) in the device staging, so the window's kernels
// run nothing (their grids are fixed; blocks past n_groups return at once) and the staging's old plan is not replayed
__global__ void helper_fetch_go_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16, const int* go) {
    if (*(const volatile int*) go) {
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
        return;
    }
    if (blockIdx.x != 0) return;
    Plan* p = reinterpret_cast<Plan*>(dst);
    for (int i = threadIdx.x; i < kWin * kPicks; i += blockDim.x) p->plan_ptr[i] = 0ull;
    if (threadIdx.x == 0) p->n_groups = 0;
}

// DS4_HELPER_DD (docs/dsv41/cards/HX-17.md): a device-direct window's head.  Every block's thread 0 waits
// for the ring slot to carry `want` (or the cancel word to name it); then the blocks copy the T x rows (n_embd floats, kMaxDim
// apart in the request) into the device staging (xr apart, as the body reads them), and block 0 builds the plan exactly as
// begin_raw does - plan_ptr / plan_w by t * kPicks + i (0: not this helper's), one group an expert in (expert, t * kPicks + i)
// order - and writes has[t] (host).  Cancelled: an empty plan and has = 0.  Launch with blockDim 256 (>= kWin * kPicks).
struct DdHead {
    const Req* req = nullptr;
    unsigned want = 0;
    const volatile unsigned long long* cancel = nullptr;
    const unsigned long long* own = nullptr;   // [n_exp]: the expert's blob on this helper, 0 not taken
    int n_exp = 0, k = 0, T = 0, n_embd = 0;
    int copy_x = 1;                            // 0: the helper holds nothing of the layer (no body runs)
    size_t xr = 0;
    uint8_t* dst = nullptr;                    // device: [Plan | x rows]
    int* has = nullptr;                        // host (fine-grained) [kWin]
    unsigned long long delay = 0;              // wall-clock ticks after the request (DS4_HELPER_DD_DELAY_US: the sweep)
    int skip = 0;                              // DS4_HELPER_DD_SKIP: the lead may not wait for this window, so the
                                               // helper can fall behind the ring; a slot already carrying a later request
                                               // means the lead went past this one without it (it had no pick here): cancel
    unsigned long long* want_out = nullptr;    // DS4_LP_FLAGFUSE (device, may be null): want, for the body's combine to
                                               // write as the window's flag (helper_flag_kernel's value)
    // helperwin (docs/dsv41/cards/HX-22.md; HW bits of helper_dd_head_kernel<HW>, 0 = the kernel as it was):
    //  bit 1 DS4_HW_FOLD - the head is the first node of the window's (layer, T) graph, so its arguments cannot name the
    //        request: want = ring[cnt[0] % kFoldRing] (the lane's host ring, written before the graph's launch), req =
    //        req_base + want % kRing; the last block to read them counts the window (cnt[0] + 1, cnt[1] its block count)
    //  bit 2 DS4_HW_HEADQ8 - the x rows the head copies are also rounded to int8 here (q8_quant_kernel's arithmetic: one
    //        32-value block per 8 lanes of float4s) into the body's xq / xd, and the body skips its q8_quant of x
    const unsigned* fold_ring = nullptr;       // host (fine-grained) [kFoldRing]
    unsigned* fold_cnt = nullptr;              // device [2]
    const Req* req_base = nullptr;
    int8_t* q8 = nullptr;                      // device: the stream's xq (row t at t * n_embd)
    float* q8d = nullptr;                      //         and xd (row t at t * n_embd / 32)
};
constexpr int kFoldRing = 256;   // (DS4_HW_FOLD: windows in flight on one helper are a handful; the lane's count wraps here)

// DS4_HW_HEADQ8: q8_quant_kernel's rounding of one 32-value block held as 8 lanes' float4s (lanes 8 g .. 8 g + 7 of a
// wave, all active together): the block's max |x| (fmaxf over the same 32 values: the same value in any order - fmaxf
// drops a NaN against a number, as the shuffle tree does), its scale max / 127, each value v / scale rounded to nearest even
__device__ __forceinline__ void hw_q8_quad(const float4 v, int8_t* __restrict__ q4, float* __restrict__ d, bool first) {
    float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am = fmaxf(am, __shfl_xor(am, o));
    const float sc = am / 127.0f;
    char4 c;
    c.x = (signed char) (int8_t) (am > 0.0f ? __float2int_rn(v.x / sc) : 0);
    c.y = (signed char) (int8_t) (am > 0.0f ? __float2int_rn(v.y / sc) : 0);
    c.z = (signed char) (int8_t) (am > 0.0f ? __float2int_rn(v.z / sc) : 0);
    c.w = (signed char) (int8_t) (am > 0.0f ? __float2int_rn(v.w / sc) : 0);
    *reinterpret_cast<char4*>(q4) = c;
    if (first) *d = sc;
}
// (the load-time check's twin of the head's loop: rows of k values ldx apart -> q [r][k], d [r][k / 32]; grid (any, rows))
__global__ void __launch_bounds__(256) hw_q8_test_kernel(const float* __restrict__ x, size_t ldx, int k, int8_t* q, float* d) {
    const int r = blockIdx.y, n4 = k / 4;
    const float4* xs = reinterpret_cast<const float4*>(x + (size_t) r * ldx);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
        hw_q8_quad(xs[i], q + (size_t) r * k + (size_t) i * 4, d + (size_t) r * (k / 32) + i / 8, (i & 7) == 0);
}
// (DS4_HW_FOLD without DS4_LP_FLAGFUSE: helper_flag_kernel's store as the graph's last node, its value the head's)
__global__ void helper_flag_dev_kernel(volatile unsigned long long* f, const unsigned long long* v) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        *f = *(const volatile unsigned long long*) v;
        __threadfence_system();
    }
}

template <int HW>
__global__ void __launch_bounds__(256) helper_dd_head_kernel(const DdHead a) {
    __shared__ int go;
    __shared__ int key[kWin * kPicks];
    __shared__ int ord[kWin * kPicks];
    __shared__ int tok[kWin];
    unsigned want = a.want;
    const Req* req = a.req;
    if constexpr ((HW & 1) != 0) {   // (DS4_HW_FOLD: this window's request from the lane's ring)
        __shared__ unsigned want_s;
        if (threadIdx.x == 0) {
            const unsigned c = *(const volatile unsigned*) &a.fold_cnt[0];
            want_s = *(const volatile unsigned*) &a.fold_ring[c % (unsigned) kFoldRing];
            __threadfence();
            if (atomicAdd(&a.fold_cnt[1], 1u) == gridDim.x - 1u) {   // (every block has read c: the next window's)
                a.fold_cnt[1] = 0u;
                __threadfence();
                a.fold_cnt[0] = c + 1u;
            }
        }
        __syncthreads();
        want = want_s;
        req = a.req_base + (want % (unsigned) kRing);
    }
    if (a.want_out != nullptr && blockIdx.x == 0 && threadIdx.x == 0) *a.want_out = (unsigned long long) want;
    if (threadIdx.x == 0) {
        for (;;) {
            const unsigned rs = *(const volatile unsigned*) &req->seq;
            if (rs == want) {
                go = 1;
                break;
            }
            if (*a.cancel == (unsigned long long) want || (a.skip && (int) (rs - want) > 0)) {
                go = 0;
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
        if (go && a.delay != 0 && blockIdx.x == 0) {
            const unsigned long long t0 = (unsigned long long) wall_clock64();
            while ((unsigned long long) wall_clock64() - t0 < a.delay) __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
    }
    __syncthreads();
    const int j = threadIdx.x;
    Plan* P = reinterpret_cast<Plan*>(a.dst);
    // the tokens with a pick of this helper's (every block works it out: only their x rows cross the bus)
    if (j < kWin) tok[j] = 0;
    __syncthreads();
    if (go && j < kWin * kPicks) {
        const int t = j / kPicks, i = j % kPicks;
        if (t < a.T && i < a.k) {
            const int e = *(const volatile int*) &req->ids[t][i];
            if (e >= 0 && e < a.n_exp && a.own[e] != 0ull) tok[t] = 1;
        }
    }
    __syncthreads();
    if (go && a.copy_x) {   // the x rows: every block, float4s
        const int n4 = a.n_embd / 4;
        float4* xd = reinterpret_cast<float4*>(a.dst + kPlanB);
        for (int t = 0; t < a.T; ++t) {
            if (!tok[t]) continue;   // (a token without a pick here: its row is never read - plan_ptr 0 for all its entries)
            const float4* xs = reinterpret_cast<const float4*>(&req->x[t][0]);
            if constexpr ((HW & 2) != 0) {   // (DS4_HW_HEADQ8: the same copy, and each block of 32 rounded as it passes)
                for (int q = blockIdx.x * blockDim.x + j; q < n4; q += gridDim.x * blockDim.x) {
                    const float4 v = xs[q];
                    xd[(size_t) t * (a.xr / 4) + q] = v;
                    hw_q8_quad(v, a.q8 + (size_t) t * a.n_embd + (size_t) q * 4, a.q8d + (size_t) t * (a.n_embd / 32) + q / 8, (q & 7) == 0);
                }
            } else {
                for (int q = blockIdx.x * blockDim.x + j; q < n4; q += gridDim.x * blockDim.x)
                    xd[(size_t) t * (a.xr / 4) + q] = xs[q];
            }
        }
    }
    if (blockIdx.x != 0) return;
    int kj = 0x7fffffff;
    __syncthreads();
    if (j < kWin) tok[j] = 0;
    __syncthreads();
    if (j < kWin * kPicks) {
        P->plan_ptr[j] = 0ull;
        P->plan_w[j] = 0.0f;
        const int t = j / kPicks, i = j % kPicks;
        if (go && t < a.T && i < a.k) {
            const int e = *(const volatile int*) &req->ids[t][i];
            if (e >= 0 && e < a.n_exp) {
                const unsigned long long p = a.own[e];
                if (p != 0ull) {
                    P->plan_ptr[j] = p;
                    P->plan_w[j] = *(const volatile float*) &req->w[t][i];
                    kj = e * (kWin * kPicks) + j;
                    tok[t] = 1;
                }
            }
        }
        key[j] = kj;
    }
    __syncthreads();
    if (j < kWin * kPicks && kj != 0x7fffffff) {   // rank sort (keys distinct)
        int r = 0;
        for (int q = 0; q < kWin * kPicks; ++q) r += key[q] < kj;
        ord[r] = j;
    }
    __syncthreads();
    if (j == 0) {
        int n = 0;
        for (int q = 0; q < kWin * kPicks; ++q) n += key[q] != 0x7fffffff;
        int ng = 0, prev = -1;
        for (int r = 0; r < n; ++r) {
            const int jj = ord[r], e = key[jj] / (kWin * kPicks);
            if (e != prev) {
                P->grp_ptr[ng] = a.own[e];
                P->grp_start[ng] = r;
                ++ng;
                prev = e;
            }
            P->grp_ent[r] = jj;
        }
        P->grp_start[ng] = n;
        P->n_groups = ng;
    }
    if (j < kWin) a.has[j] = j < a.T ? tok[j] : 0;
    __threadfence_system();
}
#endif

int env_choice(const char* name, std::initializer_list<const char*> opts, int def = 0) {   // index of the value in opts, def when unset / unknown
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return def;
    int i = 0;
    for (const char* o : opts) {
        if (std::strcmp(v, o) == 0) return i;
        ++i;
    }
    std::fprintf(stderr, "ds4 helper: %s=%s not known: the default\n", name, v);
    return def;
}

struct DevGuard {   // the caller's device back on return (the lane thread runs on the primary's context)
    int prev = -1;
    DevGuard() { cudaGetDevice(&prev); }
    ~DevGuard() {
        if (prev >= 0) cudaSetDevice(prev);
    }
};

bool arch_in_helper_list(const char* arch) {
#if defined(STRATA_HIP_HELPER_ARCHS)
    // "gfx906" or "gfx906,gfx90a": a prefix match on the device's gcnArchName ("gfx906:sramecc+:xnack-")
    const char* p = STRATA_HIP_HELPER_ARCHS;
    while (*p) {
        const char* q = p;
        while (*q && *q != ',') ++q;
        const size_t n = (size_t) (q - p);
        if (n > 0 && std::strncmp(arch, p, n) == 0 && (arch[n] == '\0' || arch[n] == ':')) return true;
        p = *q ? q + 1 : q;
    }
#else
    (void) arch;
#endif
    return false;
}

// the experts no primary slot holds, hottest first: (layer, expert) by profile share (prof[li * prof_stride + e] over its
// layer's sum; a layer without counts ranks by index, after the profiled ones)
// (share: when given, each candidate's share of its layer's routing, in the same order)
std::vector<std::pair<int, int>> rank_candidates(const std::vector<LayerGeo>& Ls, const uint32_t* prof, size_t prof_stride,
                                                 const Helper::OnPrimary& on_primary, std::vector<double>* share = nullptr) {
    struct Cand {
        int li, e;
        double share;
        int prio;
    };
    std::vector<Cand> cands;
    for (const LayerGeo& l : Ls) {
        if (!l.eligible) continue;
        const uint32_t* p = prof != nullptr ? prof + (size_t) l.li * prof_stride : nullptr;
        double sum = 0;
        if (p != nullptr)
            for (int e = 0; e < l.n_exp; ++e) sum += p[e];
        for (int e = 0; e < l.n_exp; ++e) {
            if (on_primary(l.li, e)) continue;
            // no counts: a small share by index so the layer still fills in order after the profiled ones
            const double share = p != nullptr && sum > 0 ? p[e] / sum : 1e-9 / (1.0 + e);
            cands.push_back({l.li, e, share, l.prio});
        }
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.prio != b.prio ? a.prio > b.prio : a.share > b.share; });
    std::vector<std::pair<int, int>> out;
    out.reserve(cands.size());
    for (const Cand& c : cands) out.push_back({c.li, c.e});
    if (share != nullptr) {
        share->clear();
        share->reserve(cands.size());
        for (const Cand& c : cands) share->push_back(c.share);
    }
    return out;
}


// ---- the shared pool (docs/dsv41/cards/HX-03.md): a dsv41 engine serves its helpers to other dsv41 engines, or is their client ----
// The same environment, mailbox and plan.bin mechanism as the main engine's (src/core/remote_experts.cpp), other magics:
// a main-engine ("YSPP") client cannot attach to a dsv41 server nor the reverse, and plan.bin names the model.
constexpr uint32_t kDsPlanMagic = 0x34445359;   // "YSD4"
constexpr uint32_t kDsMbMagic = 0x4d445359;     // "YSDM"
constexpr uint32_t kMainPlanMagic = 0x59535050; // "YSPP" (the main engine's plan.bin)
constexpr uint32_t kDsPoolVer = 1;
constexpr int kMaxPairs = kWin * kPicks;

/// plan.bin: the model's identity and what the serving helper holds (count pairs of int32 layer, expert follow).
struct DsPlan {
    uint32_t magic, ver;
    uint64_t model_id;
    uint32_t n_embd, n_ff;
    int32_t n_layers, n_exp;
    int64_t count;
    int32_t server_pid, pad;
};

/// One mailbox per client helper, one request outstanding (a layer's helper picks; layers are sequential).
/// File: DsMb | x[kWin][kMaxDim] | out[kWin][kMaxDim].  A pick j is token tok[j]'s i-th pick, expert[j].
struct alignas(64) DsMb {
    uint32_t magic, ver;
    uint64_t model_id;
    int32_t client_pid, pad0;
    alignas(64) std::atomic<uint64_t> req_seq;
    alignas(64) std::atomic<uint64_t> resp_seq;
    alignas(64) int32_t layer, T, n_pick, status;   // status: 0 ok, 1 bad request, 2 not owned, 3 GPU error
    uint8_t has[kWin];                              // (answer) token t's out row is valid
    int32_t tok[kMaxPairs];
    int32_t slot[kMaxPairs];
    int32_t expert[kMaxPairs];
    float w[kWin][kPicks];
};
static_assert(std::atomic<uint64_t>::is_always_lock_free, "the pool mailbox needs lock-free 64-bit atomics");
size_t mb_bytes() { return sizeof(DsMb) + sizeof(float) * (size_t) kMaxDim * (size_t) (2 * kWin); }
float* mb_x(DsMb* m) { return reinterpret_cast<float*>(m + 1); }
float* mb_out(DsMb* m) { return mb_x(m) + (size_t) kWin * kMaxDim; }

uint64_t pool_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// The model's identity: the geometry and the expert blob layout of every layer (FNV-1a).
uint64_t model_identity(const std::vector<LayerGeo>& Ls) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 1099511628211ull;
        }
    };
    mix(0x44535634ull);   // "DSV4"
    for (const LayerGeo& l : Ls) {
        mix((uint64_t) l.li);
        mix(l.eligible ? 1 : 0);
        mix((uint64_t) l.n_exp);
        mix((uint64_t) l.k_exp);
        mix((uint64_t) l.blob);
        mix((uint64_t) l.n_embd);
        mix((uint64_t) l.n_ff);
        for (int r = 0; r < 3; ++r) mix((uint64_t) (uint32_t) l.lay.K[r]);
    }
    return h;
}

/// "dir[,dir...]": the n-th helper takes the n-th directory (as the main engine's pool env).
std::string pool_dir_for(const char* list, size_t index) {
    std::string s = list;
    for (size_t i = 0; i < index; ++i) {
        const size_t c = s.find(',');
        if (c == std::string::npos) return {};
        s.erase(0, c + 1);
    }
    return s.substr(0, s.find(','));
}
const char* env_nonempty(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] ? v : nullptr;
}
}  // namespace

const char* pool_client_dirs() { return env_nonempty("STRATA_POOL_CLIENT"); }

std::string accept_device(int ordinal, bool as_helper) {
#if defined(STRATA_USE_HIP)
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) return "device " + std::to_string(ordinal) + ": no properties";
    const bool wave64 = prop.warpSize == 64;
    if (!wave64) return "";
    const bool carried = arch_in_helper_list(prop.gcnArchName);
    if (!as_helper)
        return std::string("device ") + std::to_string(ordinal) + " (" + prop.gcnArchName +
               ") is a wave64 GPU: it can only be the dsv41 helper expert cache (--helper-device), never a layer host (--gpus)";
    if (!carried)
        return std::string("device ") + std::to_string(ordinal) + " (" + prop.gcnArchName +
               ") has no code in this binary (built without STRATA_HIP_HELPER_ARCHS for it)";
    return "";
#else
    (void) ordinal;
    (void) as_helper;
    return "";
#endif
}

struct Helper::Impl {
    Config cfg;
    int dev = -1;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    // the experts: one arena, slot s at arena + slot_off[s].  Per-layer sizes by default (slot cost = the layer's blob,
    // 256-byte aligned); `uniform` (DS4_HELPER_UNIFORM_SLOTS=1): every slot the largest layer's blob.
    uint8_t* arena = nullptr;
    bool owns_arena = true;     // false: a view (make_view) - the arena is the source helper's
    float* h_xs = nullptr;      // pinned staging of a request's x (serve(), pool passes)
    bool uniform = false;
    std::vector<size_t> slot_off;   // [slot] byte offset in the arena
    size_t arena_bytes = 0;
    int64_t n_slots = 0;
    // pool pass state (views only): the plans of up to kWin sub-batches, one per layer type in the pass
    Plan* h_plans = nullptr;     // pinned [kWin]
    Plan* d_plans = nullptr;     // [kWin]
    // a pass's items and its sub-batch tables in fixed arrays (a pass takes at most kWin requests, each at most
    // kMaxPairs picks): the pool server's passes allocate nothing (they were a vector a request, six more a pass)
    struct PassItem {
        Helper::PoolReq* rq = nullptr;
        int row0 = 0;                 // first row (device x / out row) of the request
        bool tok_has[kWin] = {};
        int n_picks = 0;
        std::pair<int32_t, int> picks[kMaxPairs];   // (slot, t * kPicks + i), the valid ones
    };
    std::vector<PassItem> pass_items;   // (reserved kWin by the first pass)
    bool pass_pending = false;
    int pass_rows = 0, pass_stride = 0;
    std::vector<LayerGeo> layers;              // by li (empty entries for layers never asked about)
    std::vector<std::vector<int32_t>> of;      // [li][e] slot or -1
    std::vector<int> n_held;                   // [li]
    // scratch (sized at fill for the largest layer): x rows kMaxDim apart, h / part by entry, out by token, the plan
    float *d_x = nullptr, *d_h = nullptr, *d_part = nullptr, *d_out = nullptr;
    Plan* d_plan = nullptr;
    Plan* h_plan = nullptr;      // pinned
    float* h_out = nullptr;      // pinned [kWin][kMaxDim]
    int n_embd_max = 0, n_ff_max = 0;
    // the request in flight (begin -> finish)
    bool pending = false, failed = false;
    int cur_T = 0, cur_n_embd = 0;
    bool tok_has[kWin] = {};
    std::vector<std::pair<int, int>> byexp_;   // begin_raw's (expert, pick) list, kept (this lane's thread only)
    Stats st_;
    std::string what;   // (describe)
    // round-trip switches (DS4_HELPER_IN / _OUT / _WAIT / _GRAPH / _TRACE, see the top of the file)
    int in_mode = 0, out_mode = 0, wait_mode = 0;   // in: copy2 copy1 kfetch zc; out: d2h zc; wait: sync query flag
    bool use_graph = false, trace = false;
    uint8_t* h_pk = nullptr;        // pinned fine-grained [Plan | x rows]  (in_mode != 0)
    uint8_t* d_pk = nullptr;        // its device copy (copy1 / kfetch)
    uint8_t* h_pk_dev = nullptr;    // h_pk as the device sees it (zc / kfetch)
    float* h_outz = nullptr;        // pinned fine-grained out (out_mode zc)
    float* h_outz_dev = nullptr;
    volatile unsigned long long* h_flag = nullptr;   // (wait_mode flag) bumped by the GPU once a window is done
    unsigned long long* h_flag_dev = nullptr;
    unsigned long long flag_seq = 0;
    // DS4_HELPER_BATCH=1 (docs/dsv41/cards/HX-10.md): pack_x - the staging's x rows n_embd apart (not kMaxDim), so kfetch moves
    // T x n_embd floats and no gaps; xs_dev - the x rows of HelperSet's one shared staging (written once a request for every
    // helper), read by this helper's fetch instead of its own copy (null: its own); xs_staged - this begin's x already there
    bool pack_x = false;
    const uint8_t* xs_dev = nullptr;
    uint8_t* xs_host = nullptr;   // (xs_dev's host side)
    bool xs_staged = false;
#if defined(STRATA_USE_HIP)
    std::map<long long, hipGraphExec_t> graphs;   // (li * 64 + T) -> the captured sequence
    std::map<long long, bool> graph_bad;          // capture refused: issued directly
#endif
    cudaEvent_t tev[4] = {nullptr, nullptr, nullptr, nullptr};   // (trace) start, inputs in, kernels done, out done
    double tb0 = 0, tb1 = 0;                                     // (trace) this window's begin start / end
    struct Trace {
        int64_t n = 0, n_wait = 0;
        double issue = 0, overlap = 0, wait = 0, rt = 0, g_in = 0, g_k = 0, g_out = 0, g_span = 0, outside = 0;
    } tr;
    // DS4_HELPER_ARMED=1 (HIP, kfetch + zc + flag only): the next request's window launched before it comes - the head
    // (helper_await_kernel) waits on the device for the lane's command, so the request pays no launch, no wake and no
    // graph launch on the host. At most one armed window at a time (the lane arms the next layer after it answers).
    bool armed_on = false;
    volatile unsigned long long* h_cmd = nullptr;   // fine-grained host: the lane's command (seq * 2 + go)
    unsigned long long* h_cmd_dev = nullptr;
    volatile unsigned long long* h_ack = nullptr;   // fine-grained host: the head's answer (seq * 4 + status)
    unsigned long long* h_ack_dev = nullptr;
    int* d_go = nullptr;                            // device: the head's decision for the window's fetch
    unsigned long long arm_ticks = 0;               // the head's bound in wall-clock ticks (DS4_HELPER_ARMED_MS, 50)
    int wall_khz = 0;                               // (the wall clock's rate, for the timeout line)
    int arm_max_rows = 4;                           // windows of more rows are not armed (DS4_HELPER_ARMED_MAXROWS, 4: B = 8 was
                                                    // bimodal armed, +7 / -7..-12 %, docs/dsv41/cards/HX-09.md)
    unsigned arm_seq = 0;                           // the armed window's request (0: none armed)
    int arm_li = -1, arm_T = 0;
    bool cur_armed = false;                         // the window in flight (begin -> finish) is an accepted armed one
    struct ArmStats {
        int64_t armed = 0, used = 0, cancelled = 0, timeouts = 0, fallbacks = 0;
        // (docs/dsv41/cards/HX-10.md): the same by window size (T 1, 2, 3-4, 5-8, 9-16), the lane's arm() time, and for the
        // used windows the time from arm() to the command (a head that times out saw no command within its bound)
        int64_t by_armed[5] = {}, by_used[5] = {}, by_cancelled[5] = {}, by_timeouts[5] = {};
        double arm_ms[5] = {}, lead_ms[5] = {}, lead_max_ms = 0;
    } as_;
    static int tb(int T) { return T <= 1 ? 0 : T == 2 ? 1 : T <= 4 ? 2 : T <= 8 ? 3 : 4; }
    double arm_t = 0;   // (armed) host time the armed window was launched
    // DS4_HELPER_DD=1 (docs/dsv41/cards/HX-17.md): the persistent device-direct window
    struct DdHostBlk {
        unsigned long long seq;      // the flag kernel's: the last request answered
        unsigned long long cancel;   // the host's: a request whose armed head should stop
        int has[kWin];
    };
    bool dd = false;
    DdHostBlk* h_dd = nullptr;        // fine-grained host
    DdHostBlk* h_dd_dev = nullptr;
    const Req* dd_req_dev = nullptr;  // the request ring as this device sees it
    unsigned long long* d_own = nullptr;   // device [li][n_exp_max]: the blob, 0 not taken
    int dd_nexp = 0;
    unsigned long long dd_delay = 0;  // DS4_HELPER_DD_DELAY_US in wall-clock ticks
    bool dd_skip = false;             // DS4_HELPER_DD_SKIP=1: the lead skips this helper where it has no pick (see dd_owns)
    std::vector<uint8_t> dd_has;      // [li * dd_nexp + e]: d_own's entry is not 0 (the host's copy, for the lead's owner table)
    unsigned dd_seq = 0;              // the armed window's request (0: none)
    int dd_li = -1, dd_T = 0;
    int64_t dd_n = 0, dd_fail = 0;
    // DS4_LP_FLAGFUSE=1 (docs/dsv41/cards/HX-21.md; with DD; default off; exact by construction): a body window's flag is
    // written by its combine's last block (gg::win_combine_flag, captured in the body graph) instead of helper_flag_kernel's
    // launch after the graph; the head leaves the request's seq in d_ddw for it.  Windows without a body keep the flag kernel
    bool dd_ff = false;
    unsigned long long* d_ddw = nullptr;   // device: the window's seq (the head's)
    unsigned* d_ddc = nullptr;             // device: the combine's block count (0 between launches)
    // helperwin (docs/dsv41/cards/HX-22.md; each with DD, default off, exact by construction):
    // DS4_HW_FOLD=1 - a body window is ONE graph launch: the head (helper_dd_head_kernel<1|..>, its request from h_fold), the
    //   body, and the flag (helper_flag_dev_kernel on d_ddw, or DS4_LP_FLAGFUSE's combine): no head launch and no flag launch
    //   beside the graph.  Windows without a body keep the head + flag launches.
    // DS4_HW_HEADQ8=1 - the head rounds the x rows it copies into the body's int8 scratch (q8_quant_kernel's arithmetic,
    //   checked at load against it), and the body launches no q8_quant of x: one kernel fewer a window
    bool hw_fold = false, hw_q8 = false;
    unsigned* h_fold = nullptr;           // host (fine-grained) [kFoldRing]: the want of the n-th folded window at n % kFoldRing
    unsigned* h_fold_dev = nullptr;
    unsigned* d_fold = nullptr;           // device [2]: folded windows the heads have taken, the head's block count
    unsigned fold_n = 0;                  // the lane's count of folded windows launched (d_fold[0] once they have run)
    int8_t* q8_x = nullptr;               // DS4_HW_HEADQ8: the stream's scratch (gg::q8_scratch)
    float* q8_xd = nullptr;
    int64_t fold_win = 0;
    // DS4_PF_HELPER (pf_helper_rows): a prompt chunk's picks of this helper's experts.  A layer's held experts are one
    // strided run (load() laid them out layer by layer, expert order): expert e is local index le[e] at base + le * stride.
    struct PfLay {
        bool ok = false;
        size_t base = 0, stride = 0;
        int n = 0;
        std::vector<int> le;   // [n_exp]: local index, -1 not held
    };
    std::vector<PfLay> pfl;
    bool pf_ok = false;
    cudaStream_t pf_st = nullptr;
    cudaEvent_t pf_ev = nullptr;
    int pf_maxt = 0;
    int64_t pf_maxrows = 0, pf_maxb = 0;
    float* pf_dx = nullptr;         // device [max_tok][n_embd]: the chunk's x rows this helper needs
    uint16_t* pf_dh = nullptr;      // device [maxrows][n_ff]: gate/up's F16 swiglu
    float* pf_dy = nullptr;         // device [maxrows][n_embd]: the down rows
    int* pf_dint = nullptr;         // device plan: tok [max_tok] | rowtok [maxrows] | pos [max_tok * kPicks] | w [max_tok * kPicks] | bounds [maxb]
    int* pf_hint = nullptr;         // its pinned host side
    float* pf_hout = nullptr;       // fine-grained host [max_tok][n_embd]: token j's weighted sum
    float* pf_hout_dev = nullptr;
    int* pf_hmap = nullptr;         // fine-grained host [max_tok]: token t -> its out row, -1
    const float* pf_hx_dev = nullptr;   // HelperSet's x staging as this device sees it
    int pf_nt = 0;
    bool pf_pending = false;
    std::vector<int> pf_cnt, pf_q;   // (pf_issue's scratch)
    struct PfStats {
        int64_t layers = 0, tokens = 0, rows = 0, rounds = 0, sets = 0;
        double issue_ms = 0;
        int64_t timed = 0;                                   // (pf3 DS4_PF_HELPER_TIME=1: the helper stream's own times)
        double copy_ms = 0, gather_ms = 0, sets_ms = 0, combine_ms = 0;
        int64_t tile_layers = 0, tile_rounds = 0;            // (DS4_PF_TILE=1: layers in one pass / through the rounds)
    } pf_s;
    cudaEvent_t pf_tev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};   // (pf3: copies | gather | sets | combine ends, start)
    bool pf_timing = false, pf_timed_pending = false;
    bool pf_tile = false, pf_tile_said = false;   // (PF-12 DS4_PF_TILE=1: one pass a layer; the engaged line said)
    cudaEvent_t pf_t0 = nullptr;

    ~Impl() {
        if (dev < 0) return;
        DevGuard g;
        cudaSetDevice(dev);
#if defined(STRATA_USE_HIP)
        for (auto& kv : graphs) hipGraphExecDestroy(kv.second);
        if (h_pk) hipHostFree(h_pk);
        if (h_outz) hipHostFree(h_outz);
        if (h_flag) hipHostFree((void*) h_flag);
        if (h_cmd) {   // (an armed window still waiting is cancelled and drained first)
            if (arm_seq != 0) *h_cmd = (unsigned long long) arm_seq * 2ull;
            cudaStreamSynchronize(st);
            hipHostFree((void*) h_cmd);
        }
        if (h_ack) hipHostFree((void*) h_ack);
        if (d_go) cudaFree(d_go);
        if (h_dd) {   // (DS4_HELPER_DD: an armed head still waiting is cancelled and drained first)
            if (dd_seq != 0) *(volatile unsigned long long*) &h_dd->cancel = (unsigned long long) dd_seq;
            cudaStreamSynchronize(st);
            hipHostFree(h_dd);
        }
        if (d_own) cudaFree(d_own);
        if (d_ddw) cudaFree(d_ddw);
        if (d_ddc) cudaFree(d_ddc);
        if (h_fold) (void) hipHostFree(h_fold);   // (DS4_HW_FOLD; the stream was drained above)
        if (d_fold) (void) cudaFree(d_fold);
        if (pf_st) cudaStreamSynchronize(pf_st);   // (DS4_PF_HELPER)
        if (pf_hout) hipHostFree(pf_hout);
        if (pf_hmap) hipHostFree(pf_hmap);
#endif
        if (pf_dx) cudaFree(pf_dx);
        if (pf_dh) cudaFree(pf_dh);
        if (pf_dy) cudaFree(pf_dy);
        if (pf_dint) cudaFree(pf_dint);
        if (pf_hint) cudaFreeHost(pf_hint);
        if (pf_ev) cudaEventDestroy(pf_ev);
        for (cudaEvent_t& e : pf_tev)
            if (e) cudaEventDestroy(e);
        if (pf_st) cudaStreamDestroy(pf_st);
        if (d_pk) cudaFree(d_pk);
        for (cudaEvent_t e : tev)
            if (e) cudaEventDestroy(e);
        if (ev) cudaEventDestroy(ev);
        if (st) cudaStreamDestroy(st);
        if (arena && owns_arena) cudaFree(arena);
        if (h_xs) cudaFreeHost(h_xs);
        if (d_x) cudaFree(d_x);
        if (d_h) cudaFree(d_h);
        if (d_part) cudaFree(d_part);
        if (d_out) cudaFree(d_out);
        if (d_plan) cudaFree(d_plan);
        if (d_plans) cudaFree(d_plans);
        if (h_plans) cudaFreeHost(h_plans);
        if (h_plan) cudaFreeHost(h_plan);
        if (h_out) cudaFreeHost(h_out);
    }

    bool open(const Config& c, std::string& err) {
        cfg = c;
        if (c.device < 0) return true;
        const std::string why = accept_device(c.device, true);
        if (!why.empty()) {
            err = "helper: " + why;
            return false;
        }
        DevGuard g;
        if (cudaSetDevice(c.device) != cudaSuccess) {
            err = "helper: cudaSetDevice(" + std::to_string(c.device) + ") failed";
            return false;
        }
        if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            err = "helper: stream / event on device " + std::to_string(c.device);
            return false;
        }
        if (!gg::prepare(st)) {   // the MXFP4 / i-quant kernels' per-stream scratch on this device
            err = "helper: gg::prepare on device " + std::to_string(c.device);
            return false;
        }
        if (cudaHostAlloc((void**) &h_plan, sizeof(Plan), cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_out, (size_t) kWin * kMaxDim * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
            err = "helper: pinned buffers";
            return false;
        }
        std::memset(h_plan, 0, sizeof(Plan));
        if (cudaMalloc((void**) &d_plan, sizeof(Plan)) != cudaSuccess) {
            err = "helper: plan buffer";
            return false;
        }
        if (!open_rt(err)) return false;
        dev = c.device;
        return true;
    }

    // the round-trip switches and their buffers (on the helper's device, current)
    bool open_rt(std::string& err) {
        // HIP default (docs/dsv41/cards/HX-06.md): no SDMA on the window - kfetch, zc out, flag, graph.
        // The pre-2026-11-07 path: DS4_HELPER_IN=copy2 DS4_HELPER_OUT=d2h DS4_HELPER_WAIT=sync DS4_HELPER_GRAPH=0.
#if defined(STRATA_USE_HIP)
        const int d_in = 2, d_out = 1, d_wait = 2, d_graph = 1;
#else
        const int d_in = 0, d_out = 0, d_wait = 0, d_graph = 0;
#endif
        in_mode = env_choice("DS4_HELPER_IN", {"copy2", "copy1", "kfetch", "zc"}, d_in);
        out_mode = env_choice("DS4_HELPER_OUT", {"d2h", "zc"}, d_out);
        wait_mode = env_choice("DS4_HELPER_WAIT", {"sync", "query", "flag"}, d_wait);
        use_graph = env_choice("DS4_HELPER_GRAPH", {"0", "1"}, d_graph) == 1;
        trace = env_choice("DS4_HELPER_TRACE", {"0", "1"}) == 1;
        pack_x = env_choice("DS4_HELPER_BATCH", {"0", "1"}) == 1 && in_mode != 0;   // (copy2 reads the request's rows as they lie)
#if defined(STRATA_USE_HIP)
        const unsigned fg = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;   // fine-grained: no stale GPU-cached lines
        const size_t pk_b = kPlanB + (size_t) kWin * kMaxDim * sizeof(float);
        if (in_mode != 0) {
            if (hipHostMalloc((void**) &h_pk, pk_b, fg) != hipSuccess || hipHostGetDevicePointer((void**) &h_pk_dev, h_pk, 0) != hipSuccess ||
                (in_mode != 3 && cudaMalloc((void**) &d_pk, pk_b) != cudaSuccess)) {
                err = "helper: DS4_HELPER_IN staging";
                return false;
            }
            std::memset(h_pk, 0, pk_b);
        }
        if (out_mode == 1 && (hipHostMalloc((void**) &h_outz, (size_t) kWin * kMaxDim * sizeof(float), fg) != hipSuccess ||
                              hipHostGetDevicePointer((void**) &h_outz_dev, h_outz, 0) != hipSuccess)) {
            err = "helper: DS4_HELPER_OUT=zc buffer";
            return false;
        }
        if (wait_mode == 2) {
            void* f = nullptr;
            if (hipHostMalloc(&f, 64, fg) != hipSuccess || hipHostGetDevicePointer((void**) &h_flag_dev, f, 0) != hipSuccess) {
                err = "helper: DS4_HELPER_WAIT=flag word";
                return false;
            }
            h_flag = (volatile unsigned long long*) f;
            *h_flag = 0;
            flag_seq = 0;
        }
        if (use_graph && in_mode == 0) {   // (copy2's x source moves from window to window: nothing to capture)
            if (std::getenv("DS4_HELPER_GRAPH") != nullptr)
                std::fprintf(stderr, "ds4 helper: DS4_HELPER_GRAPH=1 needs DS4_HELPER_IN copy1 / kfetch / zc (the x source moves): off\n");
            use_graph = false;
        }
        // armed windows - opt-in (DS4_HELPER_ARMED=1); off by default again after the cap test's timeouts inside one
        // solo stall (docs/dsv41/cards/HX-09.md) until the timeout measurement (section 9) settles it
        if (env_choice("DS4_HELPER_ARMED", {"0", "1"}) == 1) {
            if (in_mode != 2 || out_mode != 1 || wait_mode != 2) {
                if (env_nonempty("DS4_HELPER_ARMED") != nullptr)   // (said only when asked for: the other modes turn it off quietly)
                    std::fprintf(stderr, "ds4 helper: DS4_HELPER_ARMED=1 needs DS4_HELPER_IN=kfetch, OUT=zc, WAIT=flag (the HIP defaults): off\n");
            } else {
                void* c = nullptr;
                void* a = nullptr;
                int khz = 0;
                if (hipHostMalloc(&c, 64, fg) != hipSuccess || hipHostGetDevicePointer((void**) &h_cmd_dev, c, 0) != hipSuccess ||
                    hipHostMalloc(&a, 64, fg) != hipSuccess || hipHostGetDevicePointer((void**) &h_ack_dev, a, 0) != hipSuccess ||
                    cudaMalloc((void**) &d_go, 64) != cudaSuccess ||
                    hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, cfg.device) != hipSuccess || khz <= 0) {
                    err = "helper: DS4_HELPER_ARMED buffers / wall clock";
                    return false;
                }
                h_cmd = (volatile unsigned long long*) c;
                h_ack = (volatile unsigned long long*) a;
                *h_cmd = 0;
                *h_ack = 0;
                const char* ms = std::getenv("DS4_HELPER_ARMED_MS");
                const double bound_ms = ms != nullptr && std::atof(ms) > 0 ? std::atof(ms) : 50.0;
                arm_ticks = (unsigned long long) (bound_ms * (double) khz);
                wall_khz = khz;
                armed_on = true;
                if (const char* mr = env_nonempty("DS4_HELPER_ARMED_MAXROWS")) arm_max_rows = std::max(1, std::min(kWin, std::atoi(mr)));
                std::fprintf(stderr, "ds4 helper: device %d armed windows on (DS4_HELPER_ARMED; windows of up to %d rows; head bound %.0f ms, wall clock %d kHz)\n",
                             cfg.device, arm_max_rows, bound_ms, khz);
            }
        }
#else
        if (in_mode != 0 || out_mode != 0 || wait_mode == 2 || use_graph) {
            std::fprintf(stderr, "ds4 helper: DS4_HELPER_IN / OUT / WAIT=flag / GRAPH are HIP-only here: the defaults\n");
            in_mode = out_mode = 0;
            wait_mode = wait_mode == 2 ? 0 : wait_mode;
            use_graph = false;
        }
#endif
        if (trace)
            for (cudaEvent_t& e : tev)
                if (cudaEventCreate(&e) != cudaSuccess) {
                    err = "helper: DS4_HELPER_TRACE events";
                    return false;
                }
        if (in_mode || out_mode || wait_mode || use_graph || trace) {
            static const char* in_n[] = {"copy2", "copy1", "kfetch", "zc"};
            static const char* out_n[] = {"d2h", "zc"};
            static const char* wait_n[] = {"sync", "query", "flag"};
            std::fprintf(stderr, "ds4 helper: device %d round trip: in %s, out %s, wait %s, graph %d, trace %d%s\n", cfg.device, in_n[in_mode],
                         out_n[out_mode], wait_n[wait_mode], (int) use_graph, (int) trace,
                         pack_x ? ", x rows packed (DS4_HELPER_BATCH)" : "");
        }
        return true;
    }

    // the window's work is done on the device (the mode's wait); false: a device error
    bool wait_done() {
        if (wait_mode == 2) {
            while (*h_flag < flag_seq) DS4_POOL_PAUSE();
            return true;
        }
        if (wait_mode == 1) {
            cudaError_t e;
            while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) {}
            if (e != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            return true;
        }
        return cudaEventSynchronize(ev) == cudaSuccess;
    }

    size_t blob_max = 0, blob_min = 0, room = 0;
    int64_t capacity = 0;     // slots this device can take at most (Config::slots, or what fits at the smallest slot)

    static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
    // bytes a slot of layer li takes in the arena
    size_t cost(int li) const { return uniform ? blob_max : align_up(layers[(size_t) li].blob, 256); }
    bool take(int li, size_t& used_bytes, int64_t& used_slots) const {
        if (dev < 0 || li < 0 || li >= (int) layers.size() || !layers[(size_t) li].eligible) return false;
        if (cfg.slots > 0) {   // a given count: the bytes are checked in load() (an error there, as before)
            if (used_slots >= cfg.slots) return false;
            used_bytes += cost(li);
            ++used_slots;
            return true;
        }
        const size_t c = cost(li);
        if (used_bytes + c > room) return false;
        used_bytes += c;
        ++used_slots;
        return true;
    }

    // the layer table, the scratch, and the capacity (the first half of fill)
    bool prepare(const std::vector<LayerGeo>& Ls, std::string& err) {
        if (dev < 0) return true;
        DevGuard g;
        cudaSetDevice(dev);
        int li_max = -1;
        for (const LayerGeo& l : Ls) li_max = std::max(li_max, l.li);
        layers.assign((size_t) li_max + 1, LayerGeo{});
        of.assign((size_t) li_max + 1, {});
        n_held.assign((size_t) li_max + 1, 0);
        blob_max = 0;
        blob_min = 0;
        {
            const char* u = std::getenv("DS4_HELPER_UNIFORM_SLOTS");
            uniform = u != nullptr && u[0] != '\0' && std::strcmp(u, "0") != 0;
        }
        for (const LayerGeo& l : Ls) {
            layers[(size_t) l.li] = l;
            of[(size_t) l.li].assign((size_t) l.n_exp, -1);
            if (!l.eligible) continue;
            blob_max = std::max(blob_max, l.blob);
            blob_min = blob_min == 0 ? l.blob : std::min(blob_min, l.blob);
            n_embd_max = std::max(n_embd_max, l.n_embd);
            n_ff_max = std::max(n_ff_max, l.n_ff);
        }
        if (blob_max == 0) {
            err = "helper: no eligible layer (GGUF experts, trunk, one-member groups)";
            return false;
        }
        if (n_embd_max > kMaxDim) {
            err = "helper: n_embd " + std::to_string(n_embd_max) + " > kMaxDim";
            return false;
        }
        // the scratch
        const size_t x_b = (size_t) kWin * kMaxDim * sizeof(float), h_b = (size_t) kWin * kPicks * n_ff_max * sizeof(float),
                     p_b = (size_t) kWin * kPicks * n_embd_max * sizeof(float), o_b = (size_t) kWin * n_embd_max * sizeof(float);
        if (cudaMalloc((void**) &d_x, x_b) != cudaSuccess || cudaMalloc((void**) &d_h, h_b) != cudaSuccess ||
            cudaMalloc((void**) &d_part, p_b) != cudaSuccess || cudaMalloc((void**) &d_out, o_b) != cudaSuccess) {
            err = "helper: scratch on device " + std::to_string(dev);
            return false;
        }
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            err = "helper: cudaMemGetInfo";
            return false;
        }
        const size_t reserve = (size_t) std::max<int64_t>(0, cfg.reserve_mib) << 20;
        room = free_b > reserve ? free_b - reserve : 0;
        capacity = cfg.slots > 0 ? cfg.slots : (int64_t) (room / (uniform ? blob_max : align_up(blob_min, 256)));
        return true;
    }

    // the arena and the fill with the given (li, e) pairs, slot i the i-th (the second half of fill)
    bool load(const std::vector<std::pair<int, int>>& assigned_in, const Reader& read, std::string& err) {
        if (dev < 0) return true;
        // DS4_PF_HELPER (set at all): the same experts, laid out layer by layer in expert order, so each layer's are one
        // strided run (pf_open); the decode path finds every expert through slot_off, so its windows read the same blobs
        std::vector<std::pair<int, int>> sorted_;
        if (pf_helper_rows() > 0) {
            sorted_ = assigned_in;
            std::sort(sorted_.begin(), sorted_.end());
        }
        const std::vector<std::pair<int, int>>& assigned = pf_helper_rows() > 0 ? sorted_ : assigned_in;
        DevGuard g;
        cudaSetDevice(dev);
        const int64_t slots = (int64_t) assigned.size();
        if (slots <= 0) {
            err = "helper: no room for an expert on device " + std::to_string(dev);
            return false;
        }
        // slot offsets: assigned order (rank order), each slot its layer's size (or the uniform one)
        slot_off.assign((size_t) slots, 0);
        size_t total = 0;
        std::map<size_t, int64_t> classes;   // slot cost -> experts
        for (int64_t i = 0; i < slots; ++i) {
            const int li = assigned[(size_t) i].first;
            if (li < 0 || li >= (int) layers.size() || !layers[(size_t) li].eligible) {
                err = "helper: load: layer " + std::to_string(li) + " is not eligible";
                return false;
            }
            slot_off[(size_t) i] = total;
            total += cost(li);
            ++classes[cost(li)];
        }
        if (cfg.slots > 0 && total > room) {
            err = "helper: " + std::to_string(cfg.slots) + " slots (" + std::to_string(total >> 20) + " MiB; largest blob " +
                  std::to_string(blob_max >> 20) + " MiB) do not fit " + std::to_string(room >> 20) + " MiB free on device " +
                  std::to_string(dev);
            return false;
        }
        arena_bytes = total;
        if (cudaMalloc((void**) &arena, total) != cudaSuccess) {
            err = "helper: cudaMalloc " + std::to_string(total >> 20) + " MiB on device " + std::to_string(dev);
            return false;
        }
        n_slots = slots;
        // the fill: a few reader threads, each a pinned staging blob, synchronous copies (the device is idle)
        std::atomic<int64_t> next{0};
        std::atomic<bool> fail{false};
        std::mutex mu;
        std::string ferr;
        const double t0 = now_ms();
        const int nth = 4;
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&] {
                cudaSetDevice(dev);
                uint8_t* stage = nullptr;
                if (cudaHostAlloc((void**) &stage, blob_max, cudaHostAllocPortable) != cudaSuccess) {
                    std::lock_guard<std::mutex> lk(mu);
                    ferr = "helper: staging blob";
                    fail.store(true);
                    return;
                }
                for (;;) {
                    const int64_t i = next.fetch_add(1);
                    if (i >= slots || fail.load()) break;
                    const std::pair<int, int>& c = assigned[(size_t) i];
                    const LayerGeo& l = layers[(size_t) c.first];
                    bool ok = read(c.first, c.second, stage);
                    if (ok) ok = cudaMemcpy(arena + slot_off[(size_t) i], stage, l.blob, cudaMemcpyHostToDevice) == cudaSuccess;
                    if (!ok) {
                        std::lock_guard<std::mutex> lk(mu);
                        ferr = "helper: loading layer " + std::to_string(c.first) + " expert " + std::to_string(c.second) + " failed";
                        fail.store(true);
                        break;
                    }
                }
                cudaFreeHost(stage);
            });
        for (auto& t : th) t.join();
        if (fail.load()) {
            err = ferr;
            return false;
        }
        for (int64_t i = 0; i < slots; ++i) {
            const std::pair<int, int>& c = assigned[(size_t) i];
            of[(size_t) c.first][(size_t) c.second] = (int32_t) i;
            ++n_held[(size_t) c.first];
        }
        cudaDeviceSynchronize();
        st_.slots = slots;
        st_.bytes = total;
        int covered = 0;
        for (int n : n_held) covered += n > 0;
        std::string cls;   // "9.25 MiB x 1200, 10.81 MiB x 300"
        for (const auto& kv : classes) {
            char cb[64];
            std::snprintf(cb, sizeof cb, "%s%.2f MiB x %lld", cls.empty() ? "" : ", ", kv.first / 1048576.0, (long long) kv.second);
            cls += cb;
        }
        char buf[768];
        if (uniform)
            std::snprintf(buf, sizeof buf, "helper: device %d holds %lld experts (%.2f GiB, uniform slots of %.2f MiB; DS4_HELPER_UNIFORM_SLOTS) over %d layers, filled in %.1f s\n",
                          dev, (long long) slots, st_.bytes / 1073741824.0, blob_max / 1048576.0, covered, (now_ms() - t0) / 1000.0);
        else
            std::snprintf(buf, sizeof buf, "helper: device %d holds %lld experts (%.2f GiB, per-layer slots [%s]; uniform %.2f MiB slots would hold %lld) over %d layers, filled in %.1f s\n",
                          dev, (long long) slots, st_.bytes / 1073741824.0, cls.c_str(), blob_max / 1048576.0,
                          (long long) (room / blob_max), covered, (now_ms() - t0) / 1000.0);
        what = buf;
        if (cfg.verbose) std::fprintf(stderr, "%s", buf);
        return true;
    }

    // the device sequence of one window (the plan / x already where in_mode reads them): inputs in, win_experts, out back.
    // Capturable (no host work that depends on the window beyond what the staging holds).
    // (go: an armed window's body - kfetch only if the head accepted it, else an empty plan; kfetch mode only)
    // (ddb: DS4_HELPER_DD's body - the head already wrote the plan and x into d_pk: no fetch)
    bool issue_seq(const LayerGeo& l, int T, const int* go = nullptr, bool ddb = false) {
        // (DS4_HELPER_BATCH: packed rows, T x n_embd floats; else rows kMaxDim apart, the last one n_embd long)
        const size_t x_used = pack_x ? (size_t) T * l.n_embd * sizeof(float) : (size_t) ((T - 1) * kMaxDim + l.n_embd) * sizeof(float);
        Plan* dp = d_plan;
        const float* dx = d_x;
        bool ok = true;
        if (ddb) {
        } else if (in_mode == 0) {
            ok = cudaMemcpyAsync(d_plan, h_plan, sizeof(Plan), cudaMemcpyHostToDevice, st) == cudaSuccess;
            ok = ok && cudaMemcpyAsync(d_x, cur_x, (size_t) T * kMaxDim * sizeof(float), cudaMemcpyHostToDevice, st) == cudaSuccess;
        } else if (in_mode == 1) {
            ok = cudaMemcpyAsync(d_pk, h_pk, kPlanB + x_used, cudaMemcpyHostToDevice, st) == cudaSuccess;
        } else if (in_mode == 2 && xs_dev != nullptr) {   // (DS4_HELPER_BATCH, shared x)
            helper_fetch2_kernel<<<8, 256, 0, st>>>(reinterpret_cast<const uint4*>(h_pk_dev), kPlanB / 16, reinterpret_cast<const uint4*>(xs_dev),
                                                    (x_used + 15) / 16, reinterpret_cast<uint4*>(d_pk), go);
            ok = cudaGetLastError() == cudaSuccess;
        } else if (in_mode == 2) {
#if defined(STRATA_USE_HIP)
            if (go != nullptr)
                helper_fetch_go_kernel<<<8, 256, 0, st>>>(reinterpret_cast<const uint4*>(h_pk_dev), reinterpret_cast<uint4*>(d_pk), (kPlanB + x_used + 15) / 16, go);
            else
#endif
            helper_fetch_kernel<<<8, 256, 0, st>>>(reinterpret_cast<const uint4*>(h_pk_dev), reinterpret_cast<uint4*>(d_pk), (kPlanB + x_used + 15) / 16);
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ddb || in_mode == 1 || in_mode == 2) {
            dp = reinterpret_cast<Plan*>(d_pk);
            dx = reinterpret_cast<const float*>(d_pk + kPlanB);
        } else if (in_mode == 3) {
            dp = reinterpret_cast<Plan*>(h_pk_dev);
            dx = reinterpret_cast<const float*>(xs_dev != nullptr ? xs_dev : h_pk_dev + kPlanB);
        }
        if (trace) ok = ok && cudaEventRecord(tev[1], st) == cudaSuccess;
        // the experts, as the primary's window path runs them (gate/up, down, the weighted per-token sum into out)
        x3::WinArgs a;
        a.grp_ptr = dp->grp_ptr;
        a.grp_start = dp->grp_start;
        a.grp_ent = dp->grp_ent;
        a.n_groups = &dp->n_groups;
        a.plan_ptr = dp->plan_ptr;
        a.plan_w = dp->plan_w;
        a.cpu_part = nullptr;
        a.cpu_flag = nullptr;
        a.x = dx;
        a.x_stride = pack_x ? (size_t) l.n_embd : (size_t) kMaxDim;   // (DS4_HELPER_BATCH: packed; the same values either way)
        a.h = d_h;
        a.part = d_part;
        a.sh_out = nullptr;
        a.out = out_mode == 1 ? h_outz_dev : d_out;
        a.out_stride = (size_t) l.n_embd;
        a.lay = l.lay;
        a.T = T;
        a.k = l.k_exp;
        a.n_embd = l.n_embd;
        a.n_ff = l.n_ff;
        a.limit = l.limit;
        a.skip_empty = ddb && dd_skip;   // (DS4_HELPER_DD_SKIP: the head's has[t] = 0 rows are never read)
        a.x_q8_given = ddb && hw_q8;     // (DS4_HW_HEADQ8: the head rounded the rows it copied; the others are never read)
        if (ddb && dd_ff) {   // (DS4_LP_FLAGFUSE: the rows, then the combine that also writes the window's flag)
            a.no_combine = true;
            ok = ok && gg::win_experts(a, st);
            if (ok) gg::win_combine_flag(a, &h_dd_dev->seq, d_ddw, d_ddc, st);
            ok = ok && cudaGetLastError() == cudaSuccess;
        } else
        ok = ok && gg::win_experts(a, st);
        if (trace) ok = ok && cudaEventRecord(tev[2], st) == cudaSuccess;
        if (out_mode == 0)
            ok = ok && cudaMemcpyAsync(h_out, d_out, (size_t) T * l.n_embd * sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess;
        if (trace) ok = ok && cudaEventRecord(tev[3], st) == cudaSuccess;
        return ok;
    }
    const float* cur_x = nullptr;   // (in_mode copy2) this window's x, read by the DMA

    unsigned req_seq = 0;   // (armed) the request begin() serves, for begin_raw's armed check; 0 for serve() / pool passes
    int cur_li = -1;        // (armed) the window in flight's layer (finish's fallback after a timed-out head)
    bool begin(int li, const Req& r, int T, const Pick* picks, int n, bool x_staged = false) {
        req_seq = r.seq;
        xs_staged = x_staged && xs_dev != nullptr;   // (DS4_HELPER_BATCH: HelperSet wrote the rows into the shared staging)
        const bool ok = begin_raw(li, &r.x[0][0], r.w, T, picks, n);
        xs_staged = false;
        req_seq = 0;
        return ok;
    }
    // DS4_HELPER_BATCH: this helper's fetch reads the x rows from `host` (HelperSet's shared fine-grained staging, packed
    // as this helper packs them) from now on.  Only before the first window (the graphs capture the source); kfetch / zc only.
    bool share_x(uint8_t* host) {
#if defined(STRATA_USE_HIP)
        if (dev < 0 || host == nullptr || !pack_x || (in_mode != 2 && in_mode != 3) || !graphs.empty()) return false;
        DevGuard g;
        if (cudaSetDevice(dev) != cudaSuccess) return false;
        uint8_t* d = nullptr;
        if (hipHostGetDevicePointer((void**) &d, host, 0) != hipSuccess || d == nullptr) {
            cudaGetLastError();
            return false;
        }
        xs_dev = d;
        xs_host = host;
        return true;
#else
        (void) host;
        return false;
#endif
    }

    // x: T rows kMaxDim apart (any host memory the DMA can read: pinned), w[t][i] the weight of token t's pick i
    bool begin_raw(int li, const float* x, const float (*w)[kPicks], int T, const Pick* picks, int n) {
        failed = false;
        pending = false;
        if (dev < 0 || n <= 0 || li < 0 || li >= (int) layers.size()) return false;
        const LayerGeo& l = layers[(size_t) li];
        if (!l.eligible || T <= 0 || T > kWin) return false;
        const double t0 = now_ms();
        DevGuard g;
        if (cudaSetDevice(dev) != cudaSuccess) {
            failed = true;
            return false;
        }
        if (trace) tb0 = t0;
        // the plan: plan_ptr / plan_w by (t, i); one group an expert, its entries in (t, i) order
        // (DS4_HELPER_IN != copy2: written into the packed staging, x rows after it)
        Plan& P = in_mode != 0 ? *reinterpret_cast<Plan*>(h_pk) : *h_plan;
        std::memset(P.plan_ptr, 0, sizeof P.plan_ptr);
        std::memset(P.plan_w, 0, sizeof P.plan_w);
        for (int t = 0; t < kWin; ++t) tok_has[t] = false;
        std::vector<std::pair<int, int>>& byexp = byexp_;   // (expert, t * kPicks + i); kept, no allocation a request
        byexp.clear();
        if (byexp.capacity() < (size_t) std::max(n, kMaxPairs)) byexp.reserve((size_t) std::max(n, kMaxPairs));
        for (int j = 0; j < n; ++j) {
            const Pick& p = picks[j];
            if (p.t < 0 || p.t >= T || p.i < 0 || p.i >= kPicks || p.e < 0 || p.e >= l.n_exp) continue;
            const int32_t s = of[(size_t) li][(size_t) p.e];
            if (s < 0) continue;
            P.plan_ptr[p.t * kPicks + p.i] = (unsigned long long) (arena + slot_off[(size_t) s]);
            P.plan_w[p.t * kPicks + p.i] = w[p.t][p.i];
            tok_has[p.t] = true;
            byexp.push_back({p.e, p.t * kPicks + p.i});
        }
        if (byexp.empty()) return false;
        std::sort(byexp.begin(), byexp.end());
        int ng = 0, ne = 0;
        for (size_t a = 0; a < byexp.size();) {
            const int e = byexp[a].first;
            P.grp_ptr[ng] = (unsigned long long) (arena + slot_off[(size_t) of[(size_t) li][(size_t) e]]);
            P.grp_start[ng] = ne;
            while (a < byexp.size() && byexp[a].first == e) P.grp_ent[ne++] = byexp[a++].second;
            ++ng;
        }
        P.grp_start[ng] = ne;
        P.n_groups = ng;
        cur_x = x;
        // the x rows into the packed staging (kMaxDim apart, n_embd each: what the kernels read; DS4_HELPER_BATCH: n_embd apart,
        // or nothing when HelperSet already wrote them into the shared staging this helper's fetch reads)
        if (in_mode != 0 && !xs_staged) {   // (a sharing helper not given the rows staged writes them where its fetch reads)
            const size_t xr = pack_x ? (size_t) l.n_embd : (size_t) kMaxDim;
            uint8_t* xd = xs_host != nullptr ? xs_host : h_pk + kPlanB;
            for (int t = 0; t < T; ++t)
                std::memcpy(xd + (size_t) t * xr * sizeof(float), x + (size_t) t * kMaxDim, (size_t) l.n_embd * sizeof(float));
        }
#if defined(STRATA_USE_HIP)
        if (arm_seq != 0) {   // DS4_HELPER_ARMED
            if (req_seq != 0 && req_seq == arm_seq && li == arm_li && T == arm_T) {
                // the armed window takes this request: the staging holds its plan and x (the same bytes as above), the
                // command releases the head - no launch, no graph launch, no wake
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *h_cmd = (unsigned long long) arm_seq * 2ull + 1ull;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                cur_arm_seq = arm_seq;
                arm_seq = 0;
                cur_armed = true;
                ++as_.used;
                ++as_.by_used[tb(T)];
                cur_lead_ms = now_ms() - arm_t;   // (arm() to the command: a head bound shorter than this times out)
                as_.lead_ms[tb(T)] += cur_lead_ms;
                as_.lead_max_ms = std::max(as_.lead_max_ms, cur_lead_ms);
                pending = true;
                cur_T = T;
                cur_n_embd = l.n_embd;
                cur_li = li;
                st_.experts += (int64_t) byexp.size();
                ++st_.windows;
                st_.begin_ms += now_ms() - t0;
                return true;
            }
            arm_drop();   // another request: the armed window runs nothing, and this one is issued behind it as before
        }
#endif
        cur_li = li;
        if (!launch_staged(l, li, T)) {
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 3) std::fprintf(stderr, "ds4 helper: layer %d: issue failed (%s); its picks fall back to the CPU lane\n", li, cudaGetErrorString(cudaGetLastError()));
            failed = true;
            return false;
        }
        pending = true;
        cur_T = T;
        cur_n_embd = l.n_embd;
        st_.experts += (int64_t) byexp.size();
        ++st_.windows;
        st_.begin_ms += now_ms() - t0;
        if (trace) tb1 = now_ms();
        return true;
    }

    // the staged window's launch (begin_raw's, unchanged): its (layer, T) graph or the direct sequence, then the flag / event
    // (the current device the helper's)
    bool launch_staged(const LayerGeo& l, int li, int T) {
        bool ok = true;
#if defined(STRATA_USE_HIP)
        const long long gkey = (long long) li * 64 + T;
        if (use_graph && !graph_bad[gkey]) {
            auto it = graphs.find(gkey);
            if (it == graphs.end()) {   // capture this (layer, T)'s sequence once (the warm-up does T = 1 for every layer)
                hipGraph_t gr = nullptr;
                hipGraphExec_t ge = nullptr;
                bool cap = hipStreamBeginCapture(st, hipStreamCaptureModeRelaxed) == hipSuccess;
                const bool tr0 = trace;
                trace = false;   // (the stage events stay outside the graph)
                const bool iss = cap && issue_seq(l, T);
                trace = tr0;
                cap = hipStreamEndCapture(st, &gr) == hipSuccess && iss;
                cap = cap && hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0) == hipSuccess;
                if (gr) hipGraphDestroy(gr);
                if (!cap) {
                    cudaGetLastError();
                    graph_bad[gkey] = true;
                    static std::atomic<int> said{0};
                    if (said.fetch_add(1) < 3) std::fprintf(stderr, "ds4 helper: layer %d T %d: graph capture refused: issued directly\n", li, T);
                } else {
                    it = graphs.emplace(gkey, ge).first;
                }
            }
            if (it != graphs.end()) {
                if (trace) ok = cudaEventRecord(tev[0], st) == cudaSuccess;
                ok = ok && hipGraphLaunch(it->second, st) == hipSuccess;
                if (trace) ok = ok && cudaEventRecord(tev[1], st) == cudaSuccess && cudaEventRecord(tev[2], st) == cudaSuccess &&
                                cudaEventRecord(tev[3], st) == cudaSuccess;
                goto issued;
            }
        }
#endif
        if (trace) ok = cudaEventRecord(tev[0], st) == cudaSuccess;
        ok = ok && issue_seq(l, T);
#if defined(STRATA_USE_HIP)
    issued:
#endif
        if (wait_mode == 2) {
            helper_flag_kernel<<<1, 64, 0, st>>>(h_flag_dev, flag_seq + 1);
            ok = ok && cudaGetLastError() == cudaSuccess;
            if (ok) ++flag_seq;
        } else {
            ok = ok && cudaEventRecord(ev, st) == cudaSuccess;
        }
        return ok;
    }

#if defined(STRATA_USE_HIP)
    unsigned cur_arm_seq = 0;   // the accepted armed window's request (finish checks its head's answer)
    double cur_lead_ms = 0;     // its arm() -> command time
    // DS4_HELPER_ARMED: launch the window for request `seq` (layer li, T rows) now, before it comes: the head waits for
    // the lane's command, the body is kfetch-if-go + the experts (its own (layer, T) graph), then the flag. On the lane's
    // thread, between requests (the lane answers first, then arms). false: not armed (off, nothing held, or a failure).
    bool arm(int li, int T, unsigned seq) {
        if (!armed_on || dev < 0 || pending || arm_seq != 0 || seq == 0) return false;
        if (li < 0 || li >= (int) layers.size() || T <= 0 || T > kWin || T > arm_max_rows) return false;
        const LayerGeo& l = layers[(size_t) li];
        if (!l.eligible || n_held[(size_t) li] == 0) return false;
        DevGuard g;
        if (cudaSetDevice(dev) != cudaSuccess) return false;
        const double ta = now_ms();
        const bool tr0 = trace;
        trace = false;   // (no stage events in an armed window)
        bool ok = true;
        const long long gkey = (1ll << 40) | ((long long) li * 64 + T);
        // (docs/dsv41/cards/HX-10.md): the armed body of (layer, T) is captured and instantiated BEFORE the head is launched.
        // It was after: a capture / instantiate that waits on the stream (or the device) would then wait on a head that waits
        // for this thread's command - a stall bounded only by the head's 50 ms, ending in a "timed out" window.
        hipGraphExec_t body_ge = nullptr;
        if (use_graph && !graph_bad[gkey]) {
            auto it = graphs.find(gkey);
            if (it == graphs.end()) {   // the armed body of (layer, T), captured once
                hipGraph_t gr = nullptr;
                hipGraphExec_t ge = nullptr;
                bool cap = hipStreamBeginCapture(st, hipStreamCaptureModeRelaxed) == hipSuccess;
                const bool iss = cap && issue_seq(l, T, d_go);
                cap = hipStreamEndCapture(st, &gr) == hipSuccess && iss;
                cap = cap && hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0) == hipSuccess;
                if (gr) hipGraphDestroy(gr);
                if (!cap) {
                    cudaGetLastError();
                    graph_bad[gkey] = true;
                } else {
                    it = graphs.emplace(gkey, ge).first;
                }
            }
            if (it != graphs.end()) body_ge = it->second;
        }
        helper_await_kernel<<<1, 64, 0, st>>>(h_cmd_dev, (unsigned long long) seq, d_go, h_ack_dev, arm_ticks);
        ok = cudaGetLastError() == cudaSuccess;
        if (ok && body_ge != nullptr) ok = hipGraphLaunch(body_ge, st) == hipSuccess;
        else if (ok) ok = issue_seq(l, T, d_go);
        trace = tr0;
        if (ok) {
            helper_flag_kernel<<<1, 64, 0, st>>>(h_flag_dev, flag_seq + 1);
            ok = cudaGetLastError() == cudaSuccess;
            if (ok) ++flag_seq;
        }
        if (!ok) {   // armed windows off for good: the head (if launched) cancelled, the stream drained
            std::fprintf(stderr, "ds4 helper: device %d: arming layer %d failed (%s): armed windows off\n", dev, li,
                         cudaGetErrorString(cudaGetLastError()));
            *h_cmd = (unsigned long long) seq * 2ull;
            cudaStreamSynchronize(st);
            flag_seq = std::max(flag_seq, (unsigned long long) *h_flag);
            armed_on = false;
            return false;
        }
        arm_seq = seq;
        arm_li = li;
        arm_T = T;
        ++as_.armed;
        ++as_.by_armed[tb(T)];
        arm_t = now_ms();
        as_.arm_ms[tb(T)] += arm_t - ta;
        return true;
    }
#endif
    // DS4_HELPER_ARMED: the armed window (if any) told to run nothing - it ends at once, its flag still bumped
    void arm_drop() {
#if defined(STRATA_USE_HIP)
        if (arm_seq == 0) return;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *h_cmd = (unsigned long long) arm_seq * 2ull;
        arm_seq = 0;
        ++as_.cancelled;
        ++as_.by_cancelled[tb(arm_T)];
#endif
    }

    // DS4_HELPER_DD: the device table (the experts take() leaves this helper), the host block, the ring's device view
    bool dd_open(const Req* reqs, const Helper::Take& take, std::string& err) {
#if defined(STRATA_USE_HIP)
        if (dev < 0) return true;
        if (in_mode != 2 || out_mode != 1 || wait_mode != 2 || !use_graph) {
            err = "helper: DS4_HELPER_DD needs DS4_HELPER_IN=kfetch, OUT=zc, WAIT=flag, GRAPH=1 (the HIP defaults)";
            return false;
        }
        DevGuard g;
        cudaSetDevice(dev);
        const unsigned fg = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;
        int khz = 0;
        if (hipHostMalloc((void**) &h_dd, sizeof(DdHostBlk), fg) != hipSuccess ||
            hipHostGetDevicePointer((void**) &h_dd_dev, h_dd, 0) != hipSuccess ||
            hipHostGetDevicePointer((void**) &dd_req_dev, const_cast<Req*>(reqs), 0) != hipSuccess ||
            hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, cfg.device) != hipSuccess || khz <= 0) {
            cudaGetLastError();
            err = "helper: DS4_HELPER_DD host block / ring mapping / wall clock on device " + std::to_string(dev);
            return false;
        }
        std::memset(h_dd, 0, sizeof(DdHostBlk));
        dd_nexp = 0;
        for (const LayerGeo& l : layers) dd_nexp = std::max(dd_nexp, l.n_exp);
        std::vector<unsigned long long> tab(layers.size() * (size_t) std::max(1, dd_nexp), 0ull);
        int64_t n = 0;
        for (size_t li = 0; li < layers.size(); ++li)
            for (int e = 0; e < layers[li].n_exp && li < of.size(); ++e) {
                const int32_t s = of[li][(size_t) e];
                if (s < 0 || !take((int) li, e)) continue;
                tab[li * (size_t) dd_nexp + (size_t) e] = (unsigned long long) (arena + slot_off[(size_t) s]);
                ++n;
            }
        if (cudaMalloc((void**) &d_own, tab.size() * 8) != cudaSuccess ||
            cudaMemcpy(d_own, tab.data(), tab.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "helper: DS4_HELPER_DD owner table on device " + std::to_string(dev);
            return false;
        }
        dd_has.assign(tab.size(), 0);
        for (size_t q = 0; q < tab.size(); ++q) dd_has[q] = tab[q] != 0ull ? 1 : 0;
        const char* du = std::getenv("DS4_HELPER_DD_DELAY_US");
        dd_delay = du != nullptr && std::atof(du) > 0 ? (unsigned long long) (std::atof(du) / 1000.0 * khz) : 0ull;
        dd_skip = std::getenv("DS4_HELPER_DD_SKIP") != nullptr && std::atoi(std::getenv("DS4_HELPER_DD_SKIP")) == 1;
        dd_ff = std::getenv("DS4_LP_FLAGFUSE") != nullptr && std::atoi(std::getenv("DS4_LP_FLAGFUSE")) == 1;
        if (dd_ff && (cudaMalloc((void**) &d_ddw, sizeof(unsigned long long)) != cudaSuccess ||
                      cudaMalloc((void**) &d_ddc, sizeof(unsigned)) != cudaSuccess ||
                      cudaMemset(d_ddw, 0, sizeof(unsigned long long)) != cudaSuccess ||
                      cudaMemset(d_ddc, 0, sizeof(unsigned)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)) {
            cudaGetLastError();
            err = "helper: DS4_LP_FLAGFUSE device words on device " + std::to_string(dev);
            return false;
        }
        if (dd_ff)
            std::fprintf(stderr, "ds4 helper: device %d DS4_LP_FLAGFUSE=1: a body window's flag written by its combine's last block\n", dev);
        if (!hw_open(fg, err)) return false;   // helperwin
        dd = true;
        std::fprintf(stderr, "ds4 helper: device %d device-direct windows on (DS4_HELPER_DD; %lld experts in its table%s%s)\n", dev, (long long) n,
                     dd_delay ? ", DS4_HELPER_DD_DELAY_US set" : "", dd_skip ? ", DS4_HELPER_DD_SKIP: empty rows not written" : "");
        return true;
#else
        (void) reqs; (void) take;
        err = "helper: DS4_HELPER_DD is HIP-only";
        return false;
#endif
    }
#if defined(STRATA_USE_HIP)
    // helperwin (docs/dsv41/cards/HX-22.md): DS4_HW_FOLD / DS4_HW_HEADQ8, read in dd_open (this helper's device current;
    // after the fill, so the helper's arena - its expert placement - is already fixed)
    bool hw_open(unsigned fg, std::string& err) {
        auto on = [](const char* n) {
            const char* v = std::getenv(n);
            return v != nullptr && std::atoi(v) == 1;
        };
        hw_fold = on("DS4_HW_FOLD");
        hw_q8 = on("DS4_HW_HEADQ8");
        if (hw_q8) {
            bool fits = true;
            for (const LayerGeo& l : layers)
                if (l.eligible && (l.n_embd % 32 != 0 || l.n_embd > kMaxDim)) fits = false;
            if (!fits || !gg::q8_scratch(st, &q8_x, &q8_xd)) {
                std::fprintf(stderr, "ds4 helper: device %d DS4_HW_HEADQ8=1: off (n_embd not a multiple of 32, or no int8 scratch)\n", dev);
                hw_q8 = false;
            } else if (!hw_q8_check(err)) {
                return false;
            }
        }
        if (hw_fold) {
            if (hipHostMalloc((void**) &h_fold, kFoldRing * sizeof(unsigned), fg) != hipSuccess ||
                hipHostGetDevicePointer((void**) &h_fold_dev, h_fold, 0) != hipSuccess ||
                cudaMalloc((void**) &d_fold, 2 * sizeof(unsigned)) != cudaSuccess ||
                cudaMemset(d_fold, 0, 2 * sizeof(unsigned)) != cudaSuccess ||
                (d_ddw == nullptr && cudaMalloc((void**) &d_ddw, sizeof(unsigned long long)) != cudaSuccess) ||
                cudaMemset(d_ddw, 0, sizeof(unsigned long long)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                (void) cudaGetLastError();
                err = "helper: DS4_HW_FOLD ring / counter on device " + std::to_string(dev);
                return false;
            }
            std::memset(h_fold, 0, kFoldRing * sizeof(unsigned));
            fold_n = 0;
        }
        if (hw_fold || hw_q8)
            std::fprintf(stderr, "ds4 helper: device %d helperwin:%s%s\n", dev,
                         hw_fold ? " DS4_HW_FOLD=1 (a body window's head, body and flag in one graph launch)" : "",
                         hw_q8 ? " DS4_HW_HEADQ8=1 (the head rounds x to int8; the body launches no q8_quant of x)" : "");
        return true;
    }
    // DS4_HW_HEADQ8's load check (before any window): the head's rounding (hw_q8_test_kernel: hw_q8_quad over float4s, the
    // head's lane mapping) against win_experts' q8_quant_kernel on kWin rows of the layers' n_embd - random, zeros, NaNs,
    // infinities, denormals, huge, all equal, signed zeros, one value a block, scales 1e-30 .. 1e30 - every output byte
    // (0xA5 first) equal, or the load fails
    bool hw_q8_check(std::string& err) {
        int k = 0;
        for (const LayerGeo& l : layers)
            if (l.eligible) k = std::max(k, l.n_embd);
        const int R = kWin;
        const size_t nq = (size_t) R * k, nd = (size_t) R * (k / 32), nb = nq + nd * 4;
        float* x = nullptr;
        uint8_t *o = nullptr, *back = nullptr;
        auto fail_at = [&](const std::string& why) {
            err = "helper: DS4_HW_HEADQ8 check on device " + std::to_string(dev) + ": " + why;
            std::fprintf(stderr, "ds4 helper: device %d DS4_HW_HEADQ8 check FAILED: %s\n", dev, why.c_str());
            if (x) (void) cudaFree(x);
            if (o) (void) cudaFree(o);
            if (back) (void) cudaFreeHost(back);
            return false;
        };
        if (cudaMalloc((void**) &x, nq * 4) != cudaSuccess || cudaMalloc((void**) &o, 2 * nb) != cudaSuccess ||
            cudaMallocHost((void**) &back, 2 * nb) != cudaSuccess) {
            (void) cudaGetLastError();
            return fail_at("its buffers");
        }
        std::vector<float> h(nq);
        uint32_t s = 0x9e3779b9u;
        auto rnd = [&]() {
            s = s * 1664525u + 1013904223u;
            return ((float) (s >> 8) / 16777216.0f - 0.5f) * 8.0f;
        };
        for (int r = 0; r < R; ++r)
            for (int i = 0; i < k; ++i) {
                float v = rnd();
                switch (r) {
                case 1: v = 0.0f; break;
                case 2: if (i % 29 == 3) v = std::nanf(""); break;
                case 3: if (i % 31 == 7) v = (i & 1) ? INFINITY : -INFINITY; break;
                case 4: v *= 1e-39f; break;
                case 5: v *= 4e37f; break;
                case 6: v = 0.75f; break;
                case 7: v = (i & 1) ? -0.0f : 0.0f; if (i % 64 == 9) v = -1.5f; break;
                case 8: v = i % 32 == 17 ? v : 0.0f; break;
                case 9: if (i % 32 < 31) v = std::nanf(""); break;   // (blocks of 31 NaNs and one number)
                default: v *= std::pow(10.0f, ((float) r - 12.5f) * 12.0f); break;   // (rows 10 .. 15: 1e-30 .. 1e30)
                }
                h[(size_t) r * k + i] = v;
            }
        bool ok = cudaMemcpyAsync(x, h.data(), nq * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
                  cudaMemsetAsync(o, 0xA5, 2 * nb, st) == cudaSuccess;
        if (ok) {
            gg::q8_rows(x, (size_t) k, R, k, (int8_t*) o, (float*) (o + nq), st);
            hw_q8_test_kernel<<<dim3(8u, (unsigned) R), 256, 0, st>>>(x, (size_t) k, k, (int8_t*) (o + nb), (float*) (o + nb + nq));
            ok = cudaGetLastError() == cudaSuccess && cudaMemcpyAsync(back, o, 2 * nb, cudaMemcpyDeviceToHost, st) == cudaSuccess &&
                 cudaStreamSynchronize(st) == cudaSuccess;
        }
        if (!ok) {
            (void) cudaGetLastError();
            return fail_at("a launch refused or a device error");
        }
        for (size_t b = 0; b < nb; ++b)
            if (back[b] != back[nb + b])
                return fail_at("byte " + std::to_string(b) + (b < nq ? " (int8 row " + std::to_string(b / (size_t) k) + ")" : " (scales)"));
        (void) cudaFree(x);
        (void) cudaFree(o);
        (void) cudaFreeHost(back);
        std::fprintf(stderr, "ds4 helper: device %d DS4_HW_HEADQ8 check: %d rows of %d (random / zero / NaN / inf / denormal / huge / equal / "
                             "signed zero / sparse / scaled): the head's int8 rounding q8_quant_kernel's bit for bit\n", dev, R, k);
        return true;
    }
    // launch the window of request seq (layer li, T rows): [body captured first] head, body, flag (the view's seq word)
    // (DS4_HW_FOLD: a body window is one graph - head, body, flag - captured once a (layer, T); its request from the ring)
    bool dd_arm(int li, int T, unsigned seq) {
        if (!dd || dev < 0 || pending || seq == 0 || li < 0 || li >= (int) layers.size() || T <= 0 || T > kWin) return false;
        const LayerGeo& l = layers[(size_t) li];
        const bool body = l.eligible && n_held[(size_t) li] > 0;
        const bool fold = body && hw_fold;
        DevGuard g;
        if (cudaSetDevice(dev) != cudaSuccess) return false;
        const bool tr0 = trace;
        trace = false;
        bool ok = true;
        hipGraphExec_t ge = nullptr;
        auto head_args = [&](bool folded) {   // (the head's arguments; folded: the request named by the ring, not here)
            DdHead a;
            a.req = dd_req_dev + (seq % kRing);
            a.want = seq;
            a.cancel = &h_dd_dev->cancel;
            a.own = d_own + (size_t) li * (size_t) dd_nexp;
            a.n_exp = l.n_exp;
            a.k = l.k_exp;
            a.T = T;
            a.n_embd = l.n_embd;
            a.copy_x = body ? 1 : 0;
            a.xr = pack_x ? (size_t) l.n_embd : (size_t) kMaxDim;
            a.dst = d_pk;
            a.has = h_dd_dev->has;
            a.delay = dd_delay;
            a.skip = dd_skip ? 1 : 0;
            a.want_out = (dd_ff || folded) && body ? d_ddw : nullptr;   // (DS4_LP_FLAGFUSE / DS4_HW_FOLD: the flag's value)
            if (folded) {
                a.req = nullptr;
                a.want = 0;
                a.fold_ring = h_fold_dev;
                a.fold_cnt = d_fold;
                a.req_base = dd_req_dev;
            }
            if (hw_q8 && body) {
                a.q8 = q8_x;
                a.q8d = q8_xd;
            }
            return a;
        };
        auto launch_head = [&](const DdHead& a, int grid) {   // (helper_dd_head_kernel<HW> as the flags set it; 0: as it was)
            const int hw = (a.fold_ring != nullptr ? 1 : 0) | (a.q8 != nullptr ? 2 : 0);
            if (hw == 0) helper_dd_head_kernel<0><<<grid, 256, 0, st>>>(a);
            else if (hw == 1) helper_dd_head_kernel<1><<<grid, 256, 0, st>>>(a);
            else if (hw == 2) helper_dd_head_kernel<2><<<grid, 256, 0, st>>>(a);
            else helper_dd_head_kernel<3><<<grid, 256, 0, st>>>(a);
            return cudaGetLastError() == cudaSuccess;
        };
        if (body) {
            const long long gkey = ((fold ? 3ll : 2ll) << 40) | ((long long) li * 64 + T);
            auto it = graphs.find(gkey);
            if (it == graphs.end()) {   // (before the head: a capture never waits behind a waiting head)
                hipGraph_t gr = nullptr;
                bool cap = hipStreamBeginCapture(st, hipStreamCaptureModeRelaxed) == hipSuccess;
                bool iss = cap;
                if (iss && fold) iss = launch_head(head_args(true), 8);   // (DS4_HW_FOLD: the head, the graph's first node)
                iss = iss && issue_seq(l, T, nullptr, true);
                if (iss && fold && !dd_ff) {   // (and the flag its last, the value the head left in d_ddw)
                    helper_flag_dev_kernel<<<1, 64, 0, st>>>(&h_dd_dev->seq, d_ddw);
                    iss = cudaGetLastError() == cudaSuccess;
                }
                cap = hipStreamEndCapture(st, &gr) == hipSuccess && iss;
                cap = cap && hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0) == hipSuccess;
                if (gr) hipGraphDestroy(gr);
                if (!cap) {
                    cudaGetLastError();
                    ge = nullptr;
                    ok = false;
                } else {
                    graphs.emplace(gkey, ge);
                }
            } else {
                ge = it->second;
            }
        }
        if (ok && fold) {   // (DS4_HW_FOLD: the request for the fold_n-th folded window, then the one launch)
            h_fold[fold_n % (unsigned) kFoldRing] = seq;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ok = hipGraphLaunch(ge, st) == hipSuccess;
            if (ok) {
                ++fold_n;
                ++fold_win;
            }
        } else {
            if (ok) ok = launch_head(head_args(false), body ? 8 : 1);
            if (ok && ge != nullptr) ok = hipGraphLaunch(ge, st) == hipSuccess;
            if (ok && !(dd_ff && ge != nullptr)) {   // (DS4_LP_FLAGFUSE: a body's flag is its combine's)
                helper_flag_kernel<<<1, 64, 0, st>>>(&h_dd_dev->seq, (unsigned long long) seq);
                ok = cudaGetLastError() == cudaSuccess;
            }
        }
        trace = tr0;
        if (!ok) {   // (a head launched without its flag would hang the lead: cancel it, drain, answer on the host)
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 5)
                std::fprintf(stderr, "ds4 helper: device %d: DS4_HELPER_DD arming layer %d T %d failed (%s): the host answers it\n", dev, li, T,
                             cudaGetErrorString(cudaGetLastError()));
            *(volatile unsigned long long*) &h_dd->cancel = (unsigned long long) seq;
            cudaStreamSynchronize(st);
            if (hw_fold && d_fold != nullptr) {   // (DS4_HW_FOLD: the lane's count back to the heads' own)
                unsigned c = fold_n;
                if (cudaMemcpy(&c, d_fold, sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess) fold_n = c;
                (void) cudaGetLastError();
            }
            ++dd_fail;
            dd_seq = 0;
            return false;
        }
        dd_seq = seq;
        dd_li = li;
        dd_T = T;
        ++dd_n;
        return true;
    }
#endif
    void dd_answer_host(unsigned seq, int T) {
        if (h_dd == nullptr) return;
        if (dd_skip && dev >= 0) {   // (DS4_HELPER_DD_SKIP: a window the lead went past may still be in flight - its flag
            DevGuard g;               //  must not land after this answer; with the skip off the lead has already waited for it)
            cudaSetDevice(dev);
            cudaStreamSynchronize(st);
        }
        for (int t = 0; t < kWin; ++t) h_dd->has[t] = 0;
        (void) T;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *(volatile unsigned long long*) &h_dd->seq = (unsigned long long) seq;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    // ---- DS4_PF_HELPER: a prompt chunk's picks of this helper's experts (HelperSet::pf_issue deals them) ----
    // The buffers for chunks of up to max_tok rows, and each layer's strided run (load() laid the experts out layer by
    // layer when DS4_PF_HELPER was set; a layer whose slots are not one run is left out: pf_local -1, it streams as before)
    bool pf_open(int max_tok, const float* x_host, std::string& err) {
#if !defined(STRATA_USE_HIP)
        (void) max_tok; (void) x_host;
        err = "a non-HIP build";
        return false;
#else
        if (dev < 0 || arena == nullptr || max_tok <= 0 || x_host == nullptr) {
            err = "no filled helper";
            return false;
        }
        DevGuard g;
        cudaSetDevice(dev);
        pfl.assign(layers.size(), PfLay{});
        int kmax = 0, nexp_max = 0, runs = 0;
        for (size_t li = 0; li < layers.size(); ++li) {
            const LayerGeo& l = layers[li];
            if (!l.eligible || n_held[li] <= 0) continue;
            kmax = std::max(kmax, l.k_exp);
            nexp_max = std::max(nexp_max, l.n_exp);
            PfLay& P = pfl[li];
            int32_t smin = INT32_MAX, smax = -1;
            for (int e = 0; e < l.n_exp; ++e) {
                const int32_t s = of[li][(size_t) e];
                if (s < 0) continue;
                smin = std::min(smin, s);
                smax = std::max(smax, s);
            }
            P.n = n_held[li];
            P.stride = cost((int) li);
            bool ok = smax >= smin && smax - smin + 1 == P.n;
            P.le.assign((size_t) l.n_exp, -1);
            for (int e = 0; e < l.n_exp && ok; ++e) {
                const int32_t s = of[li][(size_t) e];
                if (s < 0) continue;
                if (slot_off[(size_t) s] != slot_off[(size_t) smin] + (size_t) (s - smin) * P.stride) ok = false;
                P.le[(size_t) e] = s - smin;
            }
            if (!ok) P.le.assign((size_t) l.n_exp, -1);
            P.base = ok ? slot_off[(size_t) smin] : 0;
            P.ok = ok;
            runs += ok ? 1 : 0;
        }
        if (runs == 0) {
            err = "no layer's experts lie in one strided run on device " + std::to_string(dev) + " (DS4_PF_HELPER must be set at load)";
            return false;
        }
        pf_maxt = max_tok;
        pf_maxrows = (int64_t) max_tok * kmax;
        pf_maxb = (int64_t) ((max_tok + 15) / 16) * (nexp_max + 1);
        const size_t n_int = (size_t) max_tok + (size_t) pf_maxrows + (size_t) 2 * max_tok * kPicks + (size_t) pf_maxb;
        const unsigned fg = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;
        if (cudaStreamCreateWithFlags(&pf_st, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&pf_ev, cudaEventDisableTiming) != cudaSuccess || !gg::prepare(pf_st) ||
            cudaMalloc((void**) &pf_dx, (size_t) max_tok * n_embd_max * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &pf_dh, (size_t) pf_maxrows * n_ff_max * sizeof(uint16_t)) != cudaSuccess ||
            cudaMalloc((void**) &pf_dy, (size_t) pf_maxrows * n_embd_max * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &pf_dint, n_int * sizeof(int)) != cudaSuccess ||
            cudaHostAlloc((void**) &pf_hint, n_int * sizeof(int), cudaHostAllocPortable) != cudaSuccess ||
            hipHostMalloc((void**) &pf_hout, (size_t) max_tok * n_embd_max * sizeof(float), fg) != hipSuccess ||
            hipHostGetDevicePointer((void**) &pf_hout_dev, pf_hout, 0) != hipSuccess ||
            hipHostMalloc((void**) &pf_hmap, (size_t) max_tok * sizeof(int), fg) != hipSuccess ||
            hipHostGetDevicePointer((void**) &pf_hx_dev, const_cast<float*>(x_host), 0) != hipSuccess) {
            cudaGetLastError();
            err = "buffers on device " + std::to_string(dev) + " (" + std::to_string(((size_t) pf_maxrows * (n_ff_max * 2 + n_embd_max * 4) +
                  (size_t) max_tok * n_embd_max * 4) >> 20) + " MiB device memory)";
            return false;
        }
        for (int t = 0; t < max_tok; ++t) pf_hmap[t] = -1;
        // pf3 DS4_PF_HELPER_TIME=1: timing events around the stream's four parts (read back in pf_wait; nothing else changes)
        pf_timing = std::getenv("DS4_PF_HELPER_TIME") != nullptr && std::atoi(std::getenv("DS4_PF_HELPER_TIME")) != 0;
        for (int i = 0; i < 5 && pf_timing; ++i)
            if (cudaEventCreate(&pf_tev[i]) != cudaSuccess) pf_timing = false;
        // PF-12: DS4_PF_TILE=1 - a layer's prompt sets in one pass when gg::pf_tile_choose takes it (default 0: the rounds)
        pf_tile = std::getenv("DS4_PF_TILE") != nullptr && std::atoi(std::getenv("DS4_PF_TILE")) == 1;
        if (pf_tile) {   // (the one-pass kernels exist only in the gfx906 code objects)
            cudaDeviceProp pp{};
            if (cudaGetDeviceProperties(&pp, dev) != cudaSuccess || std::strstr(pp.gcnArchName, "gfx906") == nullptr) {
                std::fprintf(stderr, "ds4 helper: device %d DS4_PF_TILE=1 ignored (not a gfx906 device: the one-pass kernels are gfx906-only)\n", dev);
                pf_tile = false;
            }
        }
        if (pf_tile)
            std::fprintf(stderr, "ds4 helper: device %d DS4_PF_TILE=1: the prompt sets in one pass a layer (prep-pftile v3 dispatch: "
                                 "the rounds below 189 rows, else C2L / C4 by type and size; bit-identical by construction)\n", dev);
        pf_ok = true;
        return true;
#endif
    }
    int pf_local(int li, int e) const {
        if (!pf_ok || li < 0 || li >= (int) pfl.size() || !pfl[(size_t) li].ok || e < 0 || e >= (int) pfl[(size_t) li].le.size()) return -1;
        return pfl[(size_t) li].le[(size_t) e];
    }
    // the picks (t, i, e) of T rows (t ascending, i ascending within a row: the caller's order), weights w[t * kPicks + i]
    bool pf_issue(int li, int T, const Pick* picks, int n, const float* w, std::string& err) {
        pf_pending = false;
        pf_nt = 0;
        pf_timed_pending = false;
#if !defined(STRATA_USE_HIP)
        (void) li; (void) T; (void) picks; (void) n; (void) w;
        err = "a non-HIP build";
        return false;
#else
        const double t0 = now_ms();
        if (!pf_ok || li < 0 || li >= (int) pfl.size() || !pfl[(size_t) li].ok || T <= 0 || T > pf_maxt) {
            err = "layer " + std::to_string(li) + " / " + std::to_string(T) + " rows not set up on device " + std::to_string(dev);
            return false;
        }
        if (n <= 0) return true;
        const PfLay& P = pfl[(size_t) li];
        const LayerGeo& l = layers[(size_t) li];
        const int D = l.n_embd, FF = l.n_ff, NL = P.n;
        if (n > pf_maxrows) {
            err = "too many rows";
            return false;
        }
        int* tok = pf_hint;
        int* rowtok = tok + pf_maxt;
        int* pos = rowtok + pf_maxrows;
        float* wv = reinterpret_cast<float*>(pos + (size_t) pf_maxt * kPicks);
        int* bnd = reinterpret_cast<int*>(wv + (size_t) pf_maxt * kPicks);
        // the tokens with a pick here, in token order: out row j is token tok[j] (map[t] = j)
        for (int t = 0; t < T; ++t) pf_hmap[t] = -1;
        int nt = 0;
        for (int j = 0; j < n; ++j) {
            const Pick& p = picks[j];
            if (pf_hmap[p.t] < 0) {
                pf_hmap[p.t] = nt;
                tok[nt++] = p.t;
            }
        }
        // rows: round r takes each expert's picks 16 r .. 16 r + 15 (in pick order); within a round, experts in local order
        pf_cnt.assign((size_t) NL, 0);
        int maxc = 0;
        for (int j = 0; j < n; ++j) {
            const int le = P.le[(size_t) picks[j].e];
            if (le < 0) {
                err = "expert " + std::to_string(picks[j].e) + " of layer " + std::to_string(li) + " is not held on device " + std::to_string(dev);
                return false;
            }
            maxc = std::max(maxc, ++pf_cnt[(size_t) le]);
        }
        int tg_arm = 1, td_arm = 1;   // (DS4_PF_TILE: the one-pass arms when the dispatch takes the layer)
        const bool tile = pf_tile && gg::pf_tile_choose(l.lay, D, FF, n, &tg_arm, &td_arm);
        if (pf_tile) ++(tile ? pf_s.tile_layers : pf_s.tile_rounds);
        if (tile && !pf_tile_said) {
            pf_tile_said = true;
            std::fprintf(stderr, "ds4 helper: device %d DS4_PF_TILE engaged: layer %d, %d rows in one pass (gate / up arm %d, down arm %d)\n",
                         dev, li, n, tg_arm, td_arm);
        }
        const int rounds = tile ? 1 : (maxc + 15) / 16;
        struct Round { int r0, lo, hi, maxr, b0; };
        std::vector<Round> rs((size_t) rounds);
        // per (round, expert) row offset: pf_q[r * NL + le]
        pf_q.assign((size_t) rounds * NL, 0);
        int rows = 0, nb = 0;
        for (int r = 0; r < rounds; ++r) {
            Round& R = rs[(size_t) r];
            R.r0 = rows;
            R.lo = NL;
            R.hi = 0;
            R.maxr = 0;
            for (int le = 0; le < NL; ++le) {
                const int c = tile ? pf_cnt[(size_t) le] : std::min(16, std::max(0, pf_cnt[(size_t) le] - 16 * r));
                if (c > 0) {
                    R.lo = std::min(R.lo, le);
                    R.hi = std::max(R.hi, le + 1);
                    R.maxr = std::max(R.maxr, c);
                }
                pf_q[(size_t) r * NL + le] = rows - R.r0;   // (local to the round)
                rows += c;
            }
            // the round's bounds over its live range [lo, hi): local rows from 0
            R.b0 = nb;
            for (int le = R.lo; le <= R.hi; ++le)
                bnd[nb++] = le < R.hi ? pf_q[(size_t) r * NL + le] - pf_q[(size_t) r * NL + R.lo] : rows - R.r0 - pf_q[(size_t) r * NL + R.lo];
        }
        // each pick's row; its token's pos / w
        for (int j = 0; j < nt * kPicks; ++j) {
            pos[j] = -1;
            wv[j] = 0.0f;
        }
        std::fill(pf_cnt.begin(), pf_cnt.end(), 0);
        for (int j = 0; j < n; ++j) {
            const Pick& p = picks[j];
            const int le = P.le[(size_t) p.e], q = pf_cnt[(size_t) le]++, r = tile ? 0 : q / 16;
            const int row = rs[(size_t) r].r0 + pf_q[(size_t) r * NL + le] + (tile ? q : q % 16);
            const int jj = pf_hmap[p.t];
            rowtok[row] = jj;
            pos[(size_t) jj * kPicks + p.i] = row;
            wv[(size_t) jj * kPicks + p.i] = w[(size_t) p.t * kPicks + p.i];
        }
        DevGuard g;
        cudaSetDevice(dev);
        int* dtok = pf_dint;
        int* drowtok = dtok + pf_maxt;
        int* dpos = drowtok + pf_maxrows;
        float* dw = reinterpret_cast<float*>(dpos + (size_t) pf_maxt * kPicks);
        int* dbnd = reinterpret_cast<int*>(dw + (size_t) pf_maxt * kPicks);
        cudaEvent_t tev0 = nullptr;   // (pf3 DS4_PF_HELPER_TIME: the start mark)
        if (pf_timing && cudaEventRecord(pf_tev[4], pf_st) == cudaSuccess) tev0 = pf_tev[4];
        bool ok = cudaMemcpyAsync(dtok, tok, (size_t) nt * 4, cudaMemcpyHostToDevice, pf_st) == cudaSuccess &&
                  cudaMemcpyAsync(drowtok, rowtok, (size_t) rows * 4, cudaMemcpyHostToDevice, pf_st) == cudaSuccess &&
                  cudaMemcpyAsync(dpos, pos, (size_t) nt * kPicks * 4, cudaMemcpyHostToDevice, pf_st) == cudaSuccess &&
                  cudaMemcpyAsync(dw, wv, (size_t) nt * kPicks * 4, cudaMemcpyHostToDevice, pf_st) == cudaSuccess &&
                  cudaMemcpyAsync(dbnd, bnd, (size_t) nb * 4, cudaMemcpyHostToDevice, pf_st) == cudaSuccess;
        if (ok && tev0 != nullptr) cudaEventRecord(pf_tev[0], pf_st);
        if (ok) {
            const int n4 = D / 4;
            pf_gather_kernel<<<dim3((unsigned) ((n4 + 255) / 256), (unsigned) nt), 256, 0, pf_st>>>(
                reinterpret_cast<const float4*>(pf_hx_dev), dtok, n4, reinterpret_cast<float4*>(pf_dx));
            ok = cudaGetLastError() == cudaSuccess;
            if (ok && tev0 != nullptr) cudaEventRecord(pf_tev[1], pf_st);
        }
        for (int r = 0; r < rounds && ok; ++r) {   // the prompt sets' kernels, every expert of a round <= 16 rows: the light ones
            const Round& R = rs[(size_t) r];
            if (R.hi <= R.lo) continue;
            x3::SetArgs sa;
            sa.wbase = arena + P.base + (size_t) R.lo * P.stride;
            sa.stride = P.stride;
            sa.lay = l.lay;
            sa.n_exp = R.hi - R.lo;
            sa.bounds = dbnd + R.b0;
            sa.row0 = 0;
            sa.max_rows = R.maxr;
            sa.n_embd = D;
            sa.n_ff = FF;
            sa.limit = l.limit;
            sa.x16 = nullptr;
            bool tok = true;   // (DS4_PF_TILE: the one-pass sets; tile false: the two calls below, as before)
            if (tile) {
                tok = gg::pf_tile_set(sa, true, pf_dx, D, drowtok + R.r0, nullptr, pf_dh + (size_t) R.r0 * FF, FF, nullptr, 0,
                                      rows - R.r0, tg_arm, pf_st) &&
                      gg::pf_tile_set(sa, false, nullptr, 0, nullptr, pf_dh + (size_t) R.r0 * FF, nullptr, FF,
                                      pf_dy + (size_t) R.r0 * D, D, rows - R.r0, td_arm, pf_st);
                if (!tok) err = "DS4_PF_TILE: a one-pass set was refused";
            } else {
            gg::set_gate_up(sa, pf_dx, D, drowtok + R.r0, pf_dh + (size_t) R.r0 * FF, FF, pf_st);
            gg::set_down(sa, pf_dh + (size_t) R.r0 * FF, FF, pf_dy + (size_t) R.r0 * D, D, pf_st);
            }
            ok = tok && cudaGetLastError() == cudaSuccess;
            ++pf_s.sets;
        }
        if (ok && tev0 != nullptr) cudaEventRecord(pf_tev[2], pf_st);
        if (ok) {
            pf_combine_kernel<<<dim3((unsigned) ((D + 255) / 256), (unsigned) nt), 256, 0, pf_st>>>(pf_dy, dpos, dw, l.k_exp, D, pf_hout_dev);
            if (tev0 != nullptr) cudaEventRecord(pf_tev[3], pf_st);
            ok = cudaGetLastError() == cudaSuccess && cudaEventRecord(pf_ev, pf_st) == cudaSuccess;
        }
        pf_t0 = tev0;
        pf_timed_pending = ok && tev0 != nullptr;
        if (!ok) {
            err = std::string("issue on device ") + std::to_string(dev) + ": " + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        pf_nt = nt;
        pf_pending = true;
        ++pf_s.layers;
        pf_s.tokens += nt;
        pf_s.rows += rows;
        pf_s.rounds += rounds;
        pf_s.issue_ms += now_ms() - t0;
        return true;
#endif
    }
    bool pf_wait() {
        if (!pf_pending) return true;
        pf_pending = false;
        DevGuard g;
        cudaSetDevice(dev);
        const bool ok = cudaEventSynchronize(pf_ev) == cudaSuccess;
        if (ok && pf_timed_pending) {   // (pf3 DS4_PF_HELPER_TIME: every mark is before pf_ev on the stream, so all are done)
            pf_timed_pending = false;
            float a = 0, b = 0, c = 0, d = 0;
            if (cudaEventElapsedTime(&a, pf_t0, pf_tev[0]) == cudaSuccess && cudaEventElapsedTime(&b, pf_tev[0], pf_tev[1]) == cudaSuccess &&
                cudaEventElapsedTime(&c, pf_tev[1], pf_tev[2]) == cudaSuccess && cudaEventElapsedTime(&d, pf_tev[2], pf_tev[3]) == cudaSuccess) {
                pf_s.copy_ms += a;
                pf_s.gather_ms += b;
                pf_s.sets_ms += c;
                pf_s.combine_ms += d;
                ++pf_s.timed;
            }
        }
        return ok;
    }


    // A second lane on the same experts: its own stream, event and scratch, the arena and the tables shared read-only
    // (the experts are static after fill).  For the pool server thread, which answers other engines while the owner's
    // lane thread keeps using the first.  The source must outlive the view.
    bool make_view_of(const Impl& src, std::string& err) {
        cfg = src.cfg;
        if (src.dev < 0 || src.arena == nullptr) {
            err = "helper: nothing to share (not filled)";
            return false;
        }
        Config c = src.cfg;
        c.device = src.dev;
        if (!open(c, err)) return false;
        DevGuard g;
        cudaSetDevice(dev);
        layers = src.layers;
        of = src.of;
        n_held = src.n_held;
        arena = src.arena;
        owns_arena = false;
        slot_off = src.slot_off;
        uniform = src.uniform;
        arena_bytes = src.arena_bytes;
        blob_max = src.blob_max;
        n_slots = src.n_slots;
        n_embd_max = src.n_embd_max;
        n_ff_max = src.n_ff_max;
        const size_t x_b = (size_t) kWin * kMaxDim * sizeof(float), h_b = (size_t) kWin * kPicks * n_ff_max * sizeof(float),
                     p_b = (size_t) kWin * kPicks * n_embd_max * sizeof(float), o_b = (size_t) kWin * n_embd_max * sizeof(float);
        if (cudaMalloc((void**) &d_x, x_b) != cudaSuccess || cudaMalloc((void**) &d_h, h_b) != cudaSuccess ||
            cudaMalloc((void**) &d_part, p_b) != cudaSuccess || cudaMalloc((void**) &d_out, o_b) != cudaSuccess ||
            cudaHostAlloc((void**) &h_xs, x_b, cudaHostAllocPortable) != cudaSuccess ||
            cudaMalloc((void**) &d_plans, sizeof(Plan) * kWin) != cudaSuccess ||
            cudaHostAlloc((void**) &h_plans, sizeof(Plan) * kWin, cudaHostAllocPortable) != cudaSuccess) {
            err = "helper: scratch of the pool lane on device " + std::to_string(dev);
            return false;
        }
        std::memset(h_plans, 0, sizeof(Plan) * kWin);
        what = src.what;
        return true;
    }

    // Synchronous: the picks (all owned) of layer li for T tokens whose x rows are `x` (stride kMaxDim, any host
    // memory) computed, then out[t * kMaxDim ..) (n_embd floats) written for every token with a pick, has[t] = 1.
    bool serve(int li, const float* x, const float (*w)[kPicks], int T, const Pick* picks, int n, float* out, uint8_t* has) {
        const LayerGeo* lg = li >= 0 && li < (int) layers.size() ? &layers[(size_t) li] : nullptr;
        if (lg == nullptr || !lg->eligible || T <= 0 || T > kWin) return false;
        std::memcpy(h_xs, x, (size_t) T * kMaxDim * sizeof(float));
        if (!begin_raw(li, h_xs, w, T, picks, n)) return false;
        pending = false;
        {
            DevGuard g;
            cudaSetDevice(dev);
            if (wait_mode == 2) {
                if (!wait_done()) return false;
            } else {
                cudaError_t e;
                while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) {}
                if (e != cudaSuccess) {
                    cudaGetLastError();
                    return false;
                }
            }
        }
        const float* res = out_mode == 1 ? h_outz : h_out;
        const int D = cur_n_embd;
        for (int t = 0; t < kWin; ++t) has[t] = 0;
        for (int t = 0; t < T && t < cur_T; ++t) {
            if (!tok_has[t]) continue;
            std::memcpy(out + (size_t) t * kMaxDim, res + (size_t) t * D, (size_t) D * sizeof(float));
            has[t] = 1;
        }
        return true;
    }

    // ---- pool passes (views): union batching of several clients' requests, asynchronous ----
    // Requests whose layers share a geometry and expert layout (a "type") form one sub-batch: their tokens are rows of one
    // window, an expert two of them pick is one group (one weight read).  Sub-batches run one after the other on the
    // stream (they reuse h / part / the quantisation scratch; each has its own plan and out rows).  Per-row math is the
    // window path's, so a token's answer does not depend on who shares the pass.
    static bool same_type(const LayerGeo& a, const LayerGeo& b) {
        if (a.n_embd != b.n_embd || a.n_ff != b.n_ff || a.k_exp != b.k_exp || a.limit != b.limit) return false;
        const x3::ExpertLayout &x = a.lay, &y = b.lay;
        for (int r = 0; r < 3; ++r)
            if (x.K[r] != y.K[r] || x.trellis[r] != y.trellis[r] || x.suh[r] != y.suh[r] || x.svh[r] != y.svh[r]) return false;
        return x.cb == y.cb && x.fmt == y.fmt;
    }

    bool pass_begin(Helper::PoolReq* rq, int n) {
        pass_pending = false;
        pass_items.clear();
        if (pass_items.capacity() < (size_t) kWin) pass_items.reserve(kWin);
        if (dev < 0 || h_plans == nullptr || n <= 0) return false;
        // validate, resolve the picks to slots, give rows
        int rows = 0;
        for (int a = 0; a < n; ++a) {
            Helper::PoolReq& q = rq[a];
            q.ok = false;
            if (q.li < 0 || q.li >= (int) layers.size() || !layers[(size_t) q.li].eligible || q.T <= 0 || q.T > kWin || rows + q.T > kWin) continue;
            if (pass_items.size() >= (size_t) kWin) continue;   // (rows <= kWin and T >= 1 already keep it so)
            const LayerGeo& l = layers[(size_t) q.li];
            pass_items.emplace_back();
            PassItem& it = pass_items.back();
            it.rq = &q;
            for (int j = 0; j < q.n && it.n_picks < kMaxPairs; ++j) {
                const Pick& p = q.picks[j];
                if (p.t < 0 || p.t >= q.T || p.i < 0 || p.i >= kPicks || p.e < 0 || p.e >= l.n_exp) continue;
                const int32_t s = of[(size_t) q.li][(size_t) p.e];
                if (s < 0) continue;
                it.tok_has[p.t] = true;
                it.picks[it.n_picks++] = {s, p.t * kPicks + p.i};
            }
            if (it.n_picks == 0) {
                pass_items.pop_back();
                continue;
            }
            rows += q.T;
            q.ok = true;
        }
        if (pass_items.empty()) return false;
        // sub-batches by type, in order of first appearance; rows laid out by sub-batch
        const size_t n_items = pass_items.size();
        int grp_of[kWin], grp_rep[kWin];   // representative item of each type
        int ng_sub = 0;
        for (size_t a = 0; a < n_items; ++a) {
            grp_of[a] = -1;
            const LayerGeo& l = layers[(size_t) pass_items[a].rq->li];
            for (int g = 0; g < ng_sub && grp_of[a] < 0; ++g)
                if (same_type(l, layers[(size_t) pass_items[(size_t) grp_rep[g]].rq->li])) grp_of[a] = g;
            if (grp_of[a] < 0) {
                grp_of[a] = ng_sub;
                grp_rep[ng_sub++] = (int) a;
            }
        }
        int g_row0[kWin] = {}, g_rows[kWin] = {};
        {
            int cnt[kWin] = {};
            for (size_t a = 0; a < n_items; ++a) cnt[grp_of[a]] += pass_items[a].rq->T;
            int r = 0;
            for (int g = 0; g < ng_sub; ++g) {
                g_row0[g] = r;
                g_rows[g] = cnt[g];
                r += cnt[g];
            }
            int fill[kWin];
            for (int g = 0; g < ng_sub; ++g) fill[g] = g_row0[g];
            for (size_t a = 0; a < n_items; ++a) {
                pass_items[a].row0 = fill[grp_of[a]];
                fill[grp_of[a]] += pass_items[a].rq->T;
            }
        }
        DevGuard g;
        if (cudaSetDevice(dev) != cudaSuccess) {
            for (auto& it : pass_items) it.rq->ok = false;
            return false;
        }
        pass_stride = n_embd_max;
        pass_rows = rows;
        // x rows and the plans
        for (auto& it : pass_items) {
            const int D = layers[(size_t) it.rq->li].n_embd;
            for (int t = 0; t < it.rq->T; ++t)
                std::memcpy(h_xs + (size_t) (it.row0 + t) * kMaxDim, it.rq->x + (size_t) t * kMaxDim, (size_t) D * sizeof(float));
        }
        for (int s = 0; s < ng_sub; ++s) {
            Plan& P = h_plans[s];
            std::memset(P.plan_ptr, 0, sizeof P.plan_ptr);
            std::memset(P.plan_w, 0, sizeof P.plan_w);
            std::pair<int32_t, int> byslot[kMaxPairs];   // (slot, local row * kPicks + i); a sub-batch's rows <= kWin
            int nbs = 0;
            for (size_t a = 0; a < n_items; ++a) {
                if (grp_of[a] != s) continue;
                const PassItem& it = pass_items[a];
                const int lr = it.row0 - g_row0[s];
                for (int q = 0; q < it.n_picks; ++q) {
                    const auto& pk = it.picks[q];
                    const int t = pk.second / kPicks, i = pk.second % kPicks;
                    const int e = lr * kPicks + pk.second;   // (lr + t) * kPicks + i
                    P.plan_ptr[e] = (unsigned long long) (arena + slot_off[(size_t) pk.first]);
                    P.plan_w[e] = it.rq->w[t][i];
                    if (nbs < kMaxPairs) byslot[nbs++] = {pk.first, e};
                }
            }
            std::sort(byslot, byslot + nbs);
            int ngr = 0, ne = 0;
            for (int a = 0; a < nbs;) {
                const int32_t sl = byslot[a].first;
                P.grp_ptr[ngr] = (unsigned long long) (arena + slot_off[(size_t) sl]);
                P.grp_start[ngr] = ne;
                while (a < nbs && byslot[a].first == sl) P.grp_ent[ne++] = byslot[a++].second;
                ++ngr;
            }
            P.grp_start[ngr] = ne;
            P.n_groups = ngr;
        }
        bool ok = cudaMemcpyAsync(d_plans, h_plans, sizeof(Plan) * (size_t) ng_sub, cudaMemcpyHostToDevice, st) == cudaSuccess;
        ok = ok && cudaMemcpyAsync(d_x, h_xs, (size_t) rows * kMaxDim * sizeof(float), cudaMemcpyHostToDevice, st) == cudaSuccess;
        for (int s = 0; s < ng_sub && ok; ++s) {
            const LayerGeo& l = layers[(size_t) pass_items[(size_t) grp_rep[(size_t) s]].rq->li];
            Plan* dp = d_plans + s;
            x3::WinArgs a;
            a.grp_ptr = dp->grp_ptr;
            a.grp_start = dp->grp_start;
            a.grp_ent = dp->grp_ent;
            a.n_groups = &dp->n_groups;
            a.plan_ptr = dp->plan_ptr;
            a.plan_w = dp->plan_w;
            a.cpu_part = nullptr;
            a.cpu_flag = nullptr;
            a.x = d_x + (size_t) g_row0[(size_t) s] * kMaxDim;
            a.x_stride = (size_t) kMaxDim;
            a.h = d_h;
            a.part = d_part;
            a.sh_out = nullptr;
            a.out = d_out + (size_t) g_row0[(size_t) s] * (size_t) pass_stride;
            a.out_stride = (size_t) pass_stride;
            a.lay = l.lay;
            a.T = g_rows[(size_t) s];
            a.k = l.k_exp;
            a.n_embd = l.n_embd;
            a.n_ff = l.n_ff;
            a.limit = l.limit;
            ok = gg::win_experts(a, st);
        }
        ok = ok && cudaMemcpyAsync(h_out, d_out, (size_t) rows * (size_t) pass_stride * sizeof(float), cudaMemcpyDeviceToHost, st) == cudaSuccess;
        ok = ok && cudaEventRecord(ev, st) == cudaSuccess;
        if (!ok) {
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 3) std::fprintf(stderr, "ds4 helper: pool pass: issue failed (%s)\n", cudaGetErrorString(cudaGetLastError()));
            for (auto& it : pass_items) it.rq->ok = false;
            pass_items.clear();
            // (anything issued before the failure still runs on the stream; the next pass queues behind it)
            return false;
        }
        pass_pending = true;
        return true;
    }

    int pass_poll() {
        if (!pass_pending) return 1;
        DevGuard g;
        cudaSetDevice(dev);
        const cudaError_t e = cudaEventQuery(ev);
        if (e == cudaErrorNotReady) return 0;
        if (e != cudaSuccess) {
            cudaGetLastError();
            for (auto& it : pass_items) it.rq->ok = false;
            pass_items.clear();
            pass_pending = false;
            return -1;
        }
        return 1;
    }

    void pass_finish() {
        if (!pass_pending) return;
        pass_pending = false;
        for (auto& it : pass_items) {
            Helper::PoolReq& q = *it.rq;
            const int D = layers[(size_t) q.li].n_embd;
            for (int t = 0; t < kWin; ++t) q.has[t] = 0;
            for (int t = 0; t < q.T; ++t) {
                if (!it.tok_has[t]) continue;
                std::memcpy(q.out + (size_t) t * kMaxDim, h_out + (size_t) (it.row0 + t) * (size_t) pass_stride, (size_t) D * sizeof(float));
                q.has[t] = 1;
            }
        }
        pass_items.clear();
    }

    // (trace) one window's host times and the GPU's stage times (the events are complete: the window's work is)
    void trace_window(double tf0, double tf1) {
        float ms[3] = {0, 0, 0}, span = 0;
        bool ok = true;
        for (int s = 0; s < 3 && ok; ++s) ok = cudaEventElapsedTime(&ms[s], tev[s], tev[s + 1]) == cudaSuccess;
        ok = ok && cudaEventElapsedTime(&span, tev[0], tev[3]) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            return;
        }
        ++tr.n;
        tr.issue += (tb1 - tb0) * 1e3;
        tr.overlap += (tf0 - tb1) * 1e3;
        tr.wait += (tf1 - tf0) * 1e3;
        tr.rt += (tf1 - tb0) * 1e3;
        tr.g_in += ms[0] * 1e3;
        tr.g_k += ms[1] * 1e3;
        tr.g_out += ms[2] * 1e3;
        tr.g_span += span * 1e3;
        if ((tf1 - tf0) * 1e3 > 5.0) {   // the host was waiting when the work finished: what lies outside the GPU's span
            ++tr.n_wait;
            tr.outside += (tf1 - tb0) * 1e3 - span * 1e3;
        }
    }
    std::string trace_line() const {
        if (!trace || tr.n == 0) return std::string();
        const double n = (double) tr.n;
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "; trace (DS4_HELPER_TRACE, %lld windows, us each): host issue %.1f, lane overlap %.1f, wait %.1f, issue start -> wake %.1f; "
                      "GPU inputs %.1f, kernels %.1f, out %.1f, span %.1f; outside the GPU span (launch + wake, %lld waited windows) %.1f",
                      (long long) tr.n, tr.issue / n, tr.overlap / n, tr.wait / n, tr.rt / n, tr.g_in / n, tr.g_k / n, tr.g_out / n, tr.g_span / n,
                      (long long) tr.n_wait, tr.n_wait ? tr.outside / (double) tr.n_wait : 0.0);
        return buf;
    }

    void finish(Resp& rp, int T) {
        if (!pending) return;
        pending = false;
        const double t0 = now_ms();
        wait_done();
#if defined(STRATA_USE_HIP)
        if (cur_armed) {   // DS4_HELPER_ARMED: did the head take the command, or time out before it came?
            cur_armed = false;
            const unsigned long long a = *h_ack;
            if ((a >> 2) != (unsigned long long) cur_arm_seq || (a & 3ull) != 1ull) {
                // the window ran nothing: issue it now from the staging (begin_raw left it the request's plan and x)
                ++as_.timeouts;
                ++as_.fallbacks;
                ++as_.by_timeouts[tb(cur_T)];
                {   // (docs/dsv41/cards/HX-10.md): which side was late.  A command later than the bound: the request came late
                    // (a stall upstream of the lane - the XTX or the host - the head did its job).  A command inside the bound
                    // with the head still timing out: the head missed it (the armed path's own fault).
                    static std::atomic<int> said{0};
                    const double bound = (double) arm_ticks / std::max(1.0, (double) wall_khz);
                    if (said.fetch_add(1) < 20)
                        std::fprintf(stderr, "ds4 helper: device %d: armed window timed out: layer %d, T %d, seq %u, ack %llu/%llu; arm -> command %.2f ms "
                                     "(bound %.0f ms): %s\n", dev, cur_li, cur_T, cur_arm_seq, (unsigned long long) (a >> 2), (unsigned long long) (a & 3ull),
                                     cur_lead_ms, bound, cur_lead_ms >= bound ? "the request came late (a stall before the lane)" : "THE HEAD MISSED A TIMELY COMMAND");
                }
                DevGuard g;
                cudaSetDevice(dev);
                if (cur_li >= 0 && launch_staged(layers[(size_t) cur_li], cur_li, cur_T)) wait_done();
                else std::fprintf(stderr, "ds4 helper: device %d: the re-issue after a timed-out armed window failed\n", dev);
            }
        } else
#endif
        if (trace) trace_window(t0, now_ms());
        const float* res = out_mode == 1 ? h_outz : h_out;
        const int D = cur_n_embd;
        for (int t = 0; t < T && t < cur_T; ++t) {
            if (!tok_has[t]) continue;
            const float* y = res + (size_t) t * D;
            float* o = rp.part[t];
            if (rp.cpu[t]) {
                for (int e = 0; e < D; ++e) o[e] += y[e];
            } else {
                std::memcpy(o, y, (size_t) D * sizeof(float));
                rp.cpu[t] = 1;
            }
            ++st_.tokens;
        }
        st_.wait_ms += now_ms() - t0;
    }
};

Helper::Helper() : p_(new Impl) {}
Helper::~Helper() = default;
bool Helper::enabled() const { return p_->dev >= 0; }
int Helper::device() const { return p_->dev; }
bool Helper::open(const Config& c, std::string& err) { return p_->open(c, err); }
bool Helper::fill(const std::vector<LayerGeo>& layers, const uint32_t* prof, size_t prof_stride, const OnPrimary& on_primary,
                  const Reader& read, std::string& err) {
    if (p_->dev < 0) return true;
    if (!p_->prepare(layers, err)) return false;
    const std::vector<std::pair<int, int>> cands = rank_candidates(layers, prof, prof_stride, on_primary);
    // rank order; the first that does not fit ends the list
    size_t used = 0;
    int64_t cnt = 0;
    size_t n = 0;
    while (n < cands.size() && p_->take(cands[n].first, used, cnt)) ++n;
    return p_->load(std::vector<std::pair<int, int>>(cands.begin(), cands.begin() + (std::ptrdiff_t) n), read, err);
}
bool Helper::prepare(const std::vector<LayerGeo>& layers, std::string& err) { return p_->prepare(layers, err); }
int64_t Helper::capacity() const { return p_->capacity; }
bool Helper::load(const std::vector<std::pair<int, int>>& assigned, const Reader& read, std::string& err) {
    return p_->load(assigned, read, err);
}
std::unique_ptr<Helper> Helper::make_view(std::string& err) const {
    auto v = std::make_unique<Helper>();
    if (!v->p_->make_view_of(*p_, err)) return nullptr;
    return v;
}
bool Helper::serve(int li, const float* x, const float (*w)[kPicks], int T, const Pick* picks, int n, float* out, uint8_t* has) {
    return p_->serve(li, x, w, T, picks, n, out, has);
}
const std::vector<LayerGeo>& Helper::layer_table() const { return p_->layers; }
bool Helper::take(int li, size_t& used_bytes, int64_t& used_slots) const { return p_->take(li, used_bytes, used_slots); }
bool Helper::pass_begin(PoolReq* rq, int n) { return p_->pass_begin(rq, n); }
bool Helper::pass_busy() const { return p_->pass_pending; }
int Helper::pass_poll() { return p_->pass_poll(); }
void Helper::pass_finish() { p_->pass_finish(); }
bool Helper::owns(int li, int e) const {
    return li >= 0 && li < (int) p_->of.size() && e >= 0 && e < (int) p_->of[(size_t) li].size() && p_->of[(size_t) li][(size_t) e] >= 0;
}
int Helper::held(int li) const { return li >= 0 && li < (int) p_->n_held.size() ? p_->n_held[(size_t) li] : 0; }
bool Helper::begin(int li, const Req& r, int T, const Pick* picks, int n) { return p_->begin(li, r, T, picks, n); }
bool Helper::begin_staged(int li, const Req& r, int T, const Pick* picks, int n) { return p_->begin(li, r, T, picks, n, true); }
bool Helper::share_x(uint8_t* host) { return p_->share_x(host); }
bool Helper::x_shared() const { return p_->xs_dev != nullptr; }
void Helper::finish(Resp& rp, int T) { p_->finish(rp, T); }
bool Helper::fallback() const { return p_->failed; }
// DS4_HELPER_ARMED (the lane's thread)
bool Helper::armed_on() const { return p_->armed_on; }
bool Helper::arm(int li, int T, unsigned seq) {
#if defined(STRATA_USE_HIP)
    return p_->arm(li, T, seq);
#else
    (void) li; (void) T; (void) seq;
    return false;
#endif
}
void Helper::arm_drop() { p_->arm_drop(); }
// DS4_HELPER_DD (the lane's thread)
bool Helper::dd_open(const Req* reqs, const Take& take, std::string& err) { return p_->dd_open(reqs, take, err); }
bool Helper::dd_on() const { return p_->dd; }
Helper::DdView Helper::dd_view() const {
    DdView v;
    if (p_->h_dd == nullptr) return v;
    v.seq = &p_->h_dd->seq;
    v.has = p_->h_dd->has;
    v.out = p_->h_outz;
    return v;
}
bool Helper::dd_arm(int li, int T, unsigned seq) {
#if defined(STRATA_USE_HIP)
    return p_->dd_arm(li, T, seq);
#else
    (void) li; (void) T; (void) seq;
    return false;
#endif
}
bool Helper::dd_armed(unsigned seq, int li, int T) const { return p_->dd_seq == seq && p_->dd_li == li && p_->dd_T == T && seq != 0; }
void Helper::dd_answer_host(unsigned seq, int T) { p_->dd_answer_host(seq, T); }
bool Helper::dd_owns(int li, int e) const {   // DS4_HELPER_DD_SKIP
    const auto& q = p_->dd_has;
    if (!p_->dd || li < 0 || e < 0 || e >= p_->dd_nexp) return false;
    const size_t i = (size_t) li * (size_t) p_->dd_nexp + (size_t) e;
    return i < q.size() && q[i] != 0;
}
// DS4_PF_HELPER
bool Helper::pf_open(int max_tok, const float* x_host, std::string& err) { return p_->pf_open(max_tok, x_host, err); }
bool Helper::pf_ready() const { return p_->pf_ok; }
int Helper::pf_local(int li, int e) const { return p_->pf_local(li, e); }
bool Helper::pf_issue(int li, int T, const Pick* picks, int n, const float* w, std::string& err) {
    return p_->pf_issue(li, T, picks, n, w, err);
}
int Helper::pf_tokens() const { return p_->pf_nt; }
bool Helper::pf_wait() { return p_->pf_wait(); }
const float* Helper::pf_out() const { return p_->pf_hout; }
const int* Helper::pf_map() const { return p_->pf_hmap; }
std::string Helper::pf_describe() const {
    const auto& s = p_->pf_s;
    char b[256];
    std::snprintf(b, sizeof b, "device %d: %lld layers, %lld tokens, %lld rows (%.1f a layer), %lld rounds, %lld sets, host %.0f ms issuing",
                  p_->dev, (long long) s.layers, (long long) s.tokens, (long long) s.rows, s.layers ? (double) s.rows / (double) s.layers : 0.0,
                  (long long) s.rounds, (long long) s.sets, s.issue_ms);
    std::string r = b;
    if (s.timed > 0) {   // (pf3 DS4_PF_HELPER_TIME=1)
        char t[200];
        std::snprintf(t, sizeof t, ", stream ms (%lld layers timed): copies %.0f, gather %.0f, sets %.0f, combine %.0f", (long long) s.timed,
                      s.copy_ms, s.gather_ms, s.sets_ms, s.combine_ms);
        r += t;
    }
    if (p_->pf_tile) {   // (DS4_PF_TILE=1)
        char t[160];
        std::snprintf(t, sizeof t, ", DS4_PF_TILE: %lld layers in one pass, %lld through the rounds", (long long) s.tile_layers,
                      (long long) s.tile_rounds);
        r += t;
    }
    return r;
}
const Stats& Helper::stats() const { return p_->st_; }
void Helper::reset_stats() {
    Stats z;
    z.slots = p_->st_.slots;
    z.bytes = p_->st_.bytes;
    p_->st_ = z;
    p_->tr = Impl::Trace{};
}
std::string Helper::describe() const {
    if (p_->dev < 0) return std::string();
    if (p_->st_.windows == 0) return p_->what;
    // the run so far, appended to the fill line (before the line's newline)
    std::string w = p_->what;
    while (!w.empty() && w.back() == '\n') w.pop_back();
    char buf[256];
    std::snprintf(buf, sizeof buf, "; served %lld experts in %lld windows (%lld tokens), host %.0f ms issuing + %.0f ms waiting\n",
                  (long long) p_->st_.experts, (long long) p_->st_.windows, (long long) p_->st_.tokens, p_->st_.begin_ms, p_->st_.wait_ms);
    std::string b = buf;
    const std::string tl = p_->trace_line();
    if (!tl.empty()) b = b.substr(0, b.size() - 1) + tl + "\n";
    if (p_->dd) {   // DS4_HELPER_DD
        char db[256];
        if (p_->hw_fold || p_->hw_q8)   // (helperwin: the folded windows; the counts are the lane's)
            std::snprintf(db, sizeof db, "; device-direct windows %lld (arming failed %lld; helperwin: %lld folded%s)\n", (long long) p_->dd_n,
                          (long long) p_->dd_fail, (long long) p_->fold_win, p_->hw_q8 ? ", x rounded by the head" : "");
        else
        std::snprintf(db, sizeof db, "; device-direct windows %lld (arming failed %lld)\n", (long long) p_->dd_n, (long long) p_->dd_fail);
        b = b.substr(0, b.size() - 1) + db;
    }
    if (p_->armed_on || p_->as_.armed > 0) {   // DS4_HELPER_ARMED: the armed windows' fate
        char ab[200];
        std::snprintf(ab, sizeof ab, "; armed %lld (used %lld, cancelled %lld, timed out %lld)\n", (long long) p_->as_.armed,
                      (long long) p_->as_.used, (long long) p_->as_.cancelled, (long long) p_->as_.timeouts);
        b = b.substr(0, b.size() - 1) + ab;
        // by window size, with the lane's arm() time and the arm -> command lead (docs/dsv41/cards/HX-10.md)
        static const char* tn[5] = {"1", "2", "3-4", "5-8", "9-16"};
        std::string by;
        for (int k = 0; k < 5; ++k) {
            const auto& q = p_->as_;
            if (q.by_armed[k] == 0) continue;
            char e[200];
            std::snprintf(e, sizeof e, " T%s: armed %lld used %lld cancelled %lld timed out %lld, arm %.1f us, lead %.1f us;", tn[k],
                          (long long) q.by_armed[k], (long long) q.by_used[k], (long long) q.by_cancelled[k], (long long) q.by_timeouts[k],
                          1e3 * q.arm_ms[k] / (double) q.by_armed[k], q.by_used[k] ? 1e3 * q.lead_ms[k] / (double) q.by_used[k] : 0.0);
            by += e;
        }
        if (!by.empty()) {
            char e[80];
            std::snprintf(e, sizeof e, " lead max %.2f ms", p_->as_.lead_max_ms);
            b = b.substr(0, b.size() - 1) + "; armed by T:" + by + e + "\n";
        }
    }
    return w + b;
}

// ---- HelperSet: local helpers, pool servers, pool clients ----

struct HelperSet::Remote {
    std::string dir, path;
    int fd = -1;
    DsMb* mb = nullptr;
    uint64_t seq = 0;
    int n_exp = 0, n_embd = 0;
    std::vector<uint8_t> holds;   // [li * n_exp + e]
    int64_t resident = 0;
    int server_pid = 0;
    bool dead = false, pending = false;
    std::vector<Pick> cur;        // the request in flight
    int cur_T = 0;
    Stats st;
    bool owns(int li, int e) const {
        return !dead && li >= 0 && e >= 0 && e < n_exp && (size_t) li * (size_t) n_exp + (size_t) e < holds.size() &&
               holds[(size_t) li * (size_t) n_exp + (size_t) e] != 0;
    }
    ~Remote() {
        if (mb) munmap(static_cast<void*>(mb), mb_bytes());
        if (fd >= 0) {
            unlink(path.c_str());   // the mailbox goes; closing drops the lifetime lock
            ::close(fd);
        }
    }
};

struct HelperSet::Server {
    std::string dir;
    size_t index = 0;
    Helper* src = nullptr;
    std::vector<std::unique_ptr<Helper>> views;   // one lane (stream, event, scratch) each: DS4_POOL_STREAMS
    int max_batch = kWin;                          // requests a pass takes at most (DS4_POOL_UNION=0: 1)
    std::thread th;
    std::atomic<bool> stop{false};
    uint64_t served = 0, served_picks = 0, attached = 0, reaped = 0, refused = 0;
    uint64_t passes = 0, pass_reqs = 0, pass_max = 0, bad_reqs = 0;
    uint64_t model_id = 0;
    ~Server() {
        stop.store(true);
        if (th.joinable()) th.join();
    }
};

HelperSet::HelperSet() = default;
HelperSet::~HelperSet() {
    stop_servers();
    hd_.clear();   // (the deferred lanes are views: before their sources)
#if defined(STRATA_USE_HIP)
    if (xs_host_ != nullptr) {   // (the helpers gone first - views before their sources - so no window still reads it)
        srv_.clear();
        for (auto& h : hs_) h.reset();
        hipHostFree(xs_host_);
    }
#endif
}
bool HelperSet::enabled() const { return !hs_.empty() || !rem_.empty(); }
size_t HelperSet::size() const { return client_ ? rem_.size() : hs_.size(); }
Helper& HelperSet::at(size_t i) { return *hs_[i]; }
void HelperSet::reset_stats() {
    for (auto& h : hs_) h->reset_stats();
    for (auto& r : rem_) r->st = Stats();
}
const Helper& HelperSet::at(size_t i) const { return *hs_[i]; }

bool HelperSet::open(const std::vector<int>& devices, const Config& base, std::string& err) {
    for (size_t i = 0; i < devices.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (devices[i] == devices[j]) {
                err = "helper: device " + std::to_string(devices[i]) + " listed twice";
                return false;
            }
    if (const char* dirs = pool_client_dirs()) {
        // a pool client: no device is touched here; the n-th helper is the n-th directory's server
        client_ = true;
        for (size_t i = 0; i < devices.size(); ++i) {
            auto r = std::make_unique<Remote>();
            r->dir = pool_dir_for(dirs, i);
            if (r->dir.empty()) {
                err = "STRATA_POOL_CLIENT: no directory for helper " + std::to_string(i + 1) +
                      " (give one per --helper-device entry, comma-separated)";
                return false;
            }
            rem_.push_back(std::move(r));
        }
        return true;
    }
    for (int d : devices) {
        Config c = base;
        c.device = d;
        hs_.push_back(std::make_unique<Helper>());
        if (!hs_.back()->open(c, err)) return false;
    }
    issued_.assign(hs_.size(), 0);
#if defined(STRATA_USE_HIP)
    // DS4_HELPER_BATCH=1: one fine-grained staging of a request's x rows for every helper (each fetch reads it), so the lane
    // copies the rows once a request instead of once a helper.  A helper that cannot share keeps its own copy.
    if (env_choice("DS4_HELPER_BATCH", {"0", "1"}) == 1 && !hs_.empty()) {
        void* m = nullptr;
        const unsigned fg = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;
        if (hipHostMalloc(&m, (size_t) kWin * kMaxDim * sizeof(float), fg) == hipSuccess && m != nullptr) {
            std::memset(m, 0, (size_t) kWin * kMaxDim * sizeof(float));
            xs_host_ = static_cast<uint8_t*>(m);
            int shared = 0;
            for (auto& h : hs_) shared += h->share_x(xs_host_) ? 1 : 0;
            std::fprintf(stderr, "ds4 helper: DS4_HELPER_BATCH: x rows packed, one shared staging for %d of %zu helpers\n", shared, hs_.size());
            if (shared == 0) {
                hipHostFree(m);
                xs_host_ = nullptr;
            }
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "ds4 helper: DS4_HELPER_BATCH: no shared staging (hipHostMalloc): each helper keeps its own\n");
        }
    }
#endif
    if (const char* serve = env_nonempty("STRATA_POOL_SERVE"))   // a plan left by an earlier server is not this engine's
        for (size_t i = 0; i < hs_.size(); ++i) {
            const std::string dir = pool_dir_for(serve, i);
            if (!dir.empty()) {
                mkdir(dir.c_str(), 0700);
                unlink((dir + "/plan.bin").c_str());
            }
        }
    return true;
}

bool HelperSet::fill(const std::vector<LayerGeo>& layers, const uint32_t* prof, size_t prof_stride,
                     const Helper::OnPrimary& on_primary, const Helper::Reader& read, std::string& err) {
    if (client_) return fill_client(layers, on_primary, err);
    if (hs_.empty()) return true;
    if (hs_.size() == 1) {
        if (!hs_[0]->fill(layers, prof, prof_stride, on_primary, read, err)) return false;   // exactly as one helper alone
        return start_servers(layers, err);
    }
    for (auto& h : hs_)
        if (!h->prepare(layers, err)) return false;
    std::vector<double> share;
    const std::vector<std::pair<int, int>> cands = rank_candidates(layers, prof, prof_stride, on_primary, &share);
    const size_t H = hs_.size();
    // DS4_HELPER_PLACE: layer - every layer's helper-held experts on ONE helper, so a request goes to one helper and the
    // lane pays one round trip a layer; stripe - rank r to helper r mod n (every layer on all of them; a request of a layer
    // usually goes to each).  build 8: unset, stripe_default decides (stripe on tp3 + 2+ helpers, else layer)
    bool by_layer = !stripe_default;
    if (const char* pl = env_nonempty("DS4_HELPER_PLACE")) {
        if (std::strcmp(pl, "stripe") == 0) by_layer = false;
        else if (std::strcmp(pl, "layer") == 0) by_layer = true;   // (build 8: needed now that the default can be stripe)
        else {
            err = std::string("DS4_HELPER_PLACE=") + pl + ": layer or stripe";
            return false;
        }
    }
    std::vector<double> wgt(H, 0.0);   // the helpers' relative shares of the expected picks (layer placement)
    bool wgt_env = false;
    if (const char* ws = env_nonempty("DS4_HELPER_WEIGHTS")) {
        std::vector<double> v;
        const char* q = ws;
        while (*q) {
            char* e = nullptr;
            const double x = std::strtod(q, &e);
            if (e == q || !(x > 0.0) || !std::isfinite(x)) break;
            v.push_back(x);
            q = *e == ',' ? e + 1 : e;
            if (*e != ',' && *e != '\0') break;
        }
        if (v.size() != H || *q != '\0') {
            err = std::string("DS4_HELPER_WEIGHTS=") + ws + ": give one positive number per helper (" + std::to_string(H) + "), comma-separated";
            return false;
        }
        wgt = v;
        wgt_env = true;
    } else {
        for (size_t k = 0; k < H; ++k) wgt[k] = (double) std::max<int64_t>(1, hs_[k]->capacity());   // slot capacity
    }
    // a candidate's expected picks a token: its share of its layer's routing x the layer's picks a token
    auto picks_of = [&](size_t c) { return share[c] * (double) layers[(size_t) cands[c].first].k_exp; };
    std::vector<std::vector<std::pair<int, int>>> mine(H);
    // the stripe (also, for layer placement, the set the helpers hold between them: the ranked prefix that fits)
    size_t n_held = 0;
    {
        size_t next = 0;
        std::vector<size_t> used(H, 0);
        std::vector<int64_t> cnt(H, 0);
        std::vector<uint8_t> full(H, 0);   // a helper the next candidate did not fit is done (rank order kept)
        for (const auto& c : cands) {
            int target = -1;
            for (size_t i = 0; i < H; ++i) {
                const size_t k = (next + i) % H;
                if (full[k]) continue;
                if (hs_[k]->take(c.first, used[k], cnt[k])) {
                    target = (int) k;
                    break;
                }
                full[k] = 1;
            }
            if (target < 0) break;   // every helper is full
            if (!by_layer) mine[(size_t) target].push_back(c);
            ++n_held;
            next = ((size_t) target + 1) % H;
        }
    }
    std::vector<int> owner(layers.size(), -1);   // layer placement: the helper of each layer
    if (by_layer) {
        // each layer's load: the expected picks a token of its experts in the held set (they are the layer's hottest)
        std::vector<double> load(layers.size(), 0.0);
        std::vector<std::vector<size_t>> of_layer(layers.size());   // a layer's held candidates, rank order
        std::vector<uint8_t> has_cand(layers.size(), 0);
        for (size_t c = 0; c < cands.size(); ++c) {
            const size_t li = (size_t) cands[c].first;
            has_cand[li] = 1;
            if (c < n_held) {
                load[li] += picks_of(c);
                of_layer[li].push_back(c);
            }
        }
        // heaviest layer first (ties: lower index), each to the helper whose load / weight stays lowest among those its
        // held experts fit (none fits whole: the one that takes most of them; the rest go to the CPU lane)
        std::vector<int> order;
        for (size_t li = 0; li < layers.size(); ++li)
            if (has_cand[li]) order.push_back((int) li);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return load[(size_t) a] > load[(size_t) b]; });
        std::vector<size_t> used(H, 0);
        std::vector<int64_t> cnt(H, 0);
        std::vector<double> got(H, 0.0);
        for (int li : order) {
            const std::vector<size_t>& mem = of_layer[(size_t) li];
            int best = -1;
            size_t best_fit = 0;
            double best_cost = 0.0;
            for (size_t k = 0; k < H; ++k) {
                size_t u = used[k];
                int64_t n = cnt[k];
                size_t fit = 0;
                while (fit < mem.size() && hs_[k]->take(li, u, n)) ++fit;
                const double cost = (got[k] + load[(size_t) li]) / wgt[k];
                const bool whole = fit == mem.size();
                const bool best_whole = best >= 0 && best_fit == mem.size();
                bool better;
                if (best < 0) better = true;
                else if (whole != best_whole) better = whole;
                else if (!whole && fit != best_fit) better = fit > best_fit;
                else better = cost < best_cost;
                if (better) {
                    best = (int) k;
                    best_fit = fit;
                    best_cost = cost;
                }
            }
            owner[(size_t) li] = best;
            const size_t k = (size_t) best;
            for (size_t j = 0; j < best_fit; ++j) {
                hs_[k]->take(li, used[k], cnt[k]);
                got[k] += picks_of(mem[j]);
            }
        }
        // each helper's list: its layers' candidates in rank order, the first that does not fit ends it (any room left
        // takes its layers' next ranks; what does not fit stays on the CPU lane)
        for (size_t k = 0; k < H; ++k) {
            size_t u = 0;
            int64_t n = 0;
            for (const auto& c : cands) {
                if (owner[(size_t) c.first] != (int) k) continue;
                if (!hs_[k]->take(c.first, u, n)) break;
                mine[k].push_back(c);
            }
        }
    }
    {   // the placement line: each helper's layers and the expected picks a token of what it holds
        std::map<std::pair<int, int>, double> pk;
        for (size_t c = 0; c < cands.size(); ++c) pk[cands[c]] = picks_of(c);
        double tot_w = 0.0;
        for (double w : wgt) tot_w += w;
        std::string line = std::string("helper: placement ") + (by_layer ? "layer" : "stripe") +
                           (by_layer ? " (DS4_HELPER_PLACE=stripe: rank r on helper r mod n)" : " (DS4_HELPER_PLACE=layer: a layer on one helper)");
        double all = 0.0;
        for (size_t k = 0; k < H; ++k) {
            std::vector<uint8_t> lay(layers.size(), 0);
            double e = 0.0;
            for (const auto& c : mine[k]) {
                lay[(size_t) c.first] = 1;
                e += pk[c];
            }
            all += e;
            char buf[200];
            std::snprintf(buf, sizeof buf, "%s device %d: %d layers, %zu experts, %.1f expected picks a token", k ? ";" : ":",
                          hs_[k]->device(), (int) std::count(lay.begin(), lay.end(), (uint8_t) 1), mine[k].size(), e);
            line += buf;
            if (by_layer) {
                std::snprintf(buf, sizeof buf, " (weight %.2f)", wgt[k] / tot_w);
                line += buf;
            }
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "; %.1f in all%s\n", all,
                      by_layer ? (wgt_env ? "; weights from DS4_HELPER_WEIGHTS" : "; weights by slot capacity (DS4_HELPER_WEIGHTS=a,b,...)") : "");
        std::fprintf(stderr, "%s%s", line.c_str(), buf);
    }
    // the helpers read their experts one after the other (each already reads with several threads)
    for (size_t k = 0; k < hs_.size(); ++k) {
        if (!hs_[k]->load(mine[k], read, err)) return false;
    }
    return start_servers(layers, err);
}

// ---- DS4_PF_HELPER ----
int pf_helper_rows() {
    static const int v = [] {
        const char* e = std::getenv("DS4_PF_HELPER");
        return e != nullptr && e[0] != '\0' ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
}
bool HelperSet::pf_open(int max_tok, std::string& err) {
    pf_on_ = false;
#if !defined(STRATA_USE_HIP)
    (void) max_tok;
    err = "a non-HIP build";
    return false;
#else
    if (client_ || hs_.empty()) {
        err = client_ ? "a pool client" : "no helper";
        return false;
    }
    if (pf_helper_rows() <= 0) {   // (the layout load() makes only when the variable is set at load)
        err = "DS4_PF_HELPER is not set";
        return false;
    }
    int n_embd = 0;
    for (const auto& h : hs_)
        for (const LayerGeo& l : h->layer_table())
            if (l.eligible) n_embd = std::max(n_embd, l.n_embd);
    if (pf_x_ == nullptr) {
        void* m = nullptr;
        const unsigned fg = hipHostMallocCoherent | hipHostMallocPortable | hipHostMallocMapped;
        if (hipHostMalloc(&m, (size_t) max_tok * n_embd * sizeof(float), fg) != hipSuccess || m == nullptr) {
            cudaGetLastError();
            err = "the shared x staging (" + std::to_string(((size_t) max_tok * n_embd * 4) >> 20) + " MiB fine-grained host memory)";
            return false;
        }
        pf_x_ = static_cast<float*>(m);   // (freed with the process: the helpers' kernels may read it until they are gone)
    }
    for (size_t k = 0; k < hs_.size(); ++k) {
        std::string e2;
        if (!hs_[k]->pf_open(max_tok, pf_x_, e2)) {
            err = "helper " + std::to_string(k) + ": " + e2;
            return false;
        }
    }
    pf_by_.assign(hs_.size(), {});
    pf_on_ = true;
    return true;
#endif
}
int HelperSet::pf_owner(int li, int e) const {
    if (!pf_on_) return -1;
    for (size_t k = 0; k < hs_.size(); ++k) {
        if (!hs_[k]->owns(li, e)) continue;
        return hs_[k]->pf_local(li, e) >= 0 ? (int) k : -1;   // (the first owner, as begin() deals a pick; its run or nothing)
    }
    return -1;
}
bool HelperSet::pf_issue(int li, int T, int k, const int* ids, const float* w, const std::function<bool(int)>& host, std::string& err) {
    if (!pf_on_) {
        err = "not open";
        return false;
    }
    for (auto& v : pf_by_) v.clear();
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < k; ++i) {
            const int e = ids[(size_t) t * kPicks + i];
            if (e < 0 || !host(e)) continue;
            const int h = pf_owner(li, e);
            if (h < 0) {
                err = "layer " + std::to_string(li) + " expert " + std::to_string(e) + ": no helper takes it";
                return false;
            }
            Pick p;
            p.t = t;
            p.i = i;
            p.e = e;
            pf_by_[(size_t) h].push_back(p);
        }
    for (size_t h = 0; h < hs_.size(); ++h)   // (every helper's: a helper with no pick resets its count, nothing to wait for)
        if (!hs_[h]->pf_issue(li, T, pf_by_[h].data(), (int) pf_by_[h].size(), w, err)) return false;
    return true;
}
std::string HelperSet::pf_describe() const {
    std::string s;
    for (size_t k = 0; k < hs_.size(); ++k) s += (k ? "; " : "") + hs_[k]->pf_describe();
    return s;
}

bool HelperSet::owns(int li, int e) const {
    for (const auto& r : rem_)
        if (r->owns(li, e)) return true;
    for (const auto& h : hs_)
        if (h->owns(li, e)) return true;
    return false;
}

bool HelperSet::begin(int li, const Req& r, int T, const Pick* picks, int n, std::vector<Pick>& rejected) {
    bool any = false;
    // the per-target lists kept from request to request, each with room for a window's picks (kMaxPairs)
    std::vector<std::vector<Pick>>& by = by_;
    const size_t n_targets = client_ ? rem_.size() : hs_.size();
    if (by.size() != n_targets) {
        by.resize(n_targets);
        for (auto& v : by) v.reserve((size_t) kMaxPairs);
        for (auto& rm : rem_) rm->cur.reserve((size_t) kMaxPairs);
    }
    for (auto& v : by) v.clear();
    if (client_) {
        for (int j = 0; j < n; ++j) {
            size_t k = 0;
            while (k < rem_.size() && !rem_[k]->owns(li, picks[j].e)) ++k;
            if (k < rem_.size()) by[k].push_back(picks[j]);
            else rejected.push_back(picks[j]);
        }
        for (size_t k = 0; k < rem_.size(); ++k) {
            Remote& m = *rem_[k];
            m.pending = false;
            if (by[k].empty()) continue;
            if (by[k].size() > (size_t) kMaxPairs || T <= 0 || T > kWin) {
                rejected.insert(rejected.end(), by[k].begin(), by[k].end());
                continue;
            }
            const double t0 = now_ms();
            DsMb& b = *m.mb;
            b.layer = li;
            b.T = T;
            b.n_pick = (int32_t) by[k].size();
            for (size_t j = 0; j < by[k].size(); ++j) {
                b.tok[j] = by[k][j].t;
                b.slot[j] = by[k][j].i;
                b.expert[j] = by[k][j].e;
            }
            std::memcpy(b.w, r.w, sizeof b.w);
            for (int t = 0; t < T; ++t) std::memcpy(mb_x(&b) + (size_t) t * kMaxDim, r.x[t], (size_t) m.n_embd * sizeof(float));
            b.req_seq.store(++m.seq, std::memory_order_release);
            m.cur.assign(by[k].begin(), by[k].end());   // (a copy into its kept room: by[k] keeps its own)
            m.cur_T = T;
            m.pending = true;
            m.st.begin_ms += now_ms() - t0;
            any = true;
        }
        return any;
    }
    // DS4_HELPER_BATCH: the request's x rows written once (packed) into the staging every sharing helper's fetch reads
    const bool xs = xs_host_ != nullptr && n > 0 && T >= 1 && T <= kWin && li >= 0 && li < (int) hs_[0]->layer_table().size();
    if (xs) {
        const size_t D = (size_t) hs_[0]->layer_table()[(size_t) li].n_embd;
        for (int t = 0; t < T; ++t) std::memcpy(xs_host_ + (size_t) t * D * sizeof(float), r.x[t], D * sizeof(float));
    }
    const auto issue = [&](size_t k, const Pick* p, int m) {
        return xs && hs_[k]->x_shared() ? hs_[k]->begin_staged(li, r, T, p, m) : hs_[k]->begin(li, r, T, p, m);
    };
    if (hs_.size() == 1) {
        issued_[0] = issue(0, picks, n);
        if (!issued_[0]) rejected.insert(rejected.end(), picks, picks + n);
        return issued_[0] != 0;
    }
    for (int j = 0; j < n; ++j) {
        size_t k = 0;
        while (k < hs_.size() && !hs_[k]->owns(li, picks[j].e)) ++k;
        if (k < hs_.size()) by[k].push_back(picks[j]);
        else rejected.push_back(picks[j]);
    }
    for (size_t k = 0; k < hs_.size(); ++k) {
        issued_[k] = 0;
        if (by[k].empty()) continue;
        issued_[k] = issue(k, by[k].data(), (int) by[k].size());
        if (issued_[k]) any = true;
        else rejected.insert(rejected.end(), by[k].begin(), by[k].end());
    }
    return any;
}

void HelperSet::finish(Resp& rp, int T, std::vector<Pick>& failed) {
    if (client_) {
        for (auto& rm : rem_) {
            Remote& m = *rm;
            if (!m.pending) continue;
            m.pending = false;
            DsMb& b = *m.mb;
            const double t0 = now_ms();
            const uint64_t deadline = pool_now_ns() + 5'000'000'000ull;   // the main engine's 5 s
            uint32_t spins = 0;
            bool timeout = false;
            while (b.resp_seq.load(std::memory_order_acquire) != m.seq) {
                DS4_POOL_PAUSE();
                if ((++spins & 0xffff) == 0 && pool_now_ns() > deadline) {
                    timeout = true;
                    break;
                }
            }
            m.st.wait_ms += now_ms() - t0;
            if (timeout || b.status != 0) {
                std::fprintf(stderr,
                             "ds4 pool: the serving engine of %s %s layer %d; this engine now computes those experts on the CPU lane\n",
                             m.dir.c_str(), timeout ? "did not answer within 5 s" : ("refused (status " + std::to_string(b.status) + ")").c_str(),
                             (int) b.layer);
                m.dead = true;
                failed.insert(failed.end(), m.cur.begin(), m.cur.end());
                continue;
            }
            const int D = m.n_embd;
            for (int t = 0; t < T && t < m.cur_T; ++t) {
                if (!b.has[t]) continue;
                const float* y = mb_out(&b) + (size_t) t * kMaxDim;
                float* o = rp.part[t];
                if (rp.cpu[t]) {
                    for (int e = 0; e < D; ++e) o[e] += y[e];
                } else {
                    std::memcpy(o, y, (size_t) D * sizeof(float));
                    rp.cpu[t] = 1;
                }
                ++m.st.tokens;
            }
            m.st.experts += (int64_t) m.cur.size();
            ++m.st.windows;
        }
        return;
    }
    (void) failed;
    for (size_t k = 0; k < hs_.size(); ++k)
        if (issued_[k]) {
            hs_[k]->finish(rp, T);
            issued_[k] = 0;
        }
}

// DS4_HELPER_DEFER two windows in flight: a view a local helper (docs/dsv41/cards/HX-12.md)
bool HelperSet::open_deferred(std::string& err) {
    if (client_ || hs_.empty()) return false;
    std::vector<std::unique_ptr<Helper>> v;
    for (auto& h : hs_) {
        std::unique_ptr<Helper> x = h->make_view(err);
        if (!x) return false;   // (v's views go with it; the caller keeps the one-lane path)
        v.push_back(std::move(x));
    }
    hd_ = std::move(v);
    issued_d_.assign(hd_.size(), 0);
    return true;
}
bool HelperSet::begin_deferred(int li, const Req& r, int T, const Pick* picks, int n, std::vector<Pick>& rejected) {
    if (!deferred_ready()) {
        rejected.insert(rejected.end(), picks, picks + n);
        return false;
    }
    std::vector<std::vector<Pick>>& by = by_;   // (begin()'s lists, free again: begin() copied its picks into its plans)
    if (by.size() != hs_.size()) {
        by.resize(hs_.size());
        for (auto& b : by) b.reserve((size_t) kMaxPairs);
    }
    for (auto& b : by) b.clear();
    for (int j = 0; j < n; ++j) {
        size_t k = 0;
        while (k < hs_.size() && !hs_[k]->owns(li, picks[j].e)) ++k;
        if (k < hs_.size()) by[k].push_back(picks[j]);
        else rejected.push_back(picks[j]);
    }
    bool any = false;
    for (size_t k = 0; k < hd_.size(); ++k) {
        issued_d_[k] = 0;
        if (by[k].empty()) continue;
        issued_d_[k] = hd_[k]->begin(li, r, T, by[k].data(), (int) by[k].size());
        if (issued_d_[k]) any = true;
        else rejected.insert(rejected.end(), by[k].begin(), by[k].end());
    }
    return any;
}
void HelperSet::finish_deferred(Resp& rp, int T) {
    for (size_t k = 0; k < hd_.size(); ++k)
        if (issued_d_[k]) {
            hd_[k]->finish(rp, T);
            issued_d_[k] = 0;
        }
}

// DS4_HELPER_DD: each pick to the first local helper that owns it (begin()'s rule), never one the primary holds
bool HelperSet::dd_open(const Req* reqs, const Helper::OnPrimary& on_primary, std::string& err) {
    if (client_ || hs_.empty() || reqs == nullptr) {
        err = "helper: DS4_HELPER_DD needs local helpers (not a pool client)";
        return false;
    }
    for (size_t k = 0; k < hs_.size(); ++k) {
        const auto take = [&, k](int li, int e) {
            if (on_primary(li, e)) return false;
            for (size_t q = 0; q < k; ++q)
                if (hs_[q]->owns(li, e)) return false;
            return true;
        };
        if (!hs_[k]->dd_open(reqs, take, err)) return false;
    }
    dd_on_ = true;
    return true;
}
std::vector<int8_t> HelperSet::dd_owner_table(int n_layers, int n_exp) const {   // DS4_HELPER_DD_SKIP
    std::vector<int8_t> t((size_t) std::max(0, n_layers) * (size_t) std::max(0, n_exp), (int8_t) -1);
    for (int li = 0; li < n_layers; ++li)
        for (int e = 0; e < n_exp; ++e)
            for (size_t k = 0; k < hs_.size() && k < 127; ++k)
                if (hs_[k]->dd_owns(li, e)) {   // (take() gives each expert to one helper at most: the first is the only)
                    t[(size_t) li * (size_t) n_exp + (size_t) e] = (int8_t) k;
                    break;
                }
    return t;
}
std::vector<Helper::DdView> HelperSet::dd_views() const {
    std::vector<Helper::DdView> v;
    for (const auto& h : hs_) v.push_back(h->dd_view());
    return v;
}
bool HelperSet::dd_arm(int li, int T, unsigned seq) {
    bool all = true;
    for (auto& h : hs_) all = h->dd_arm(li, T, seq) && all;
    dd_armed_n += all ? 1 : 0;
    return all;
}
int HelperSet::dd_check(unsigned seq, int li, int T, const std::vector<Pick>& picks, std::vector<Pick>& cpu) {
    int n = 0;
    for (size_t k = 0; k < hs_.size(); ++k) {
        if (hs_[k]->dd_armed(seq, li, T)) continue;
        for (const Pick& p : picks) {   // (begin()'s rule: the first owner takes it)
            size_t q = 0;
            while (q < hs_.size() && !hs_[q]->owns(li, p.e)) ++q;
            if (q == k) cpu.push_back(p);
        }
        hs_[k]->dd_answer_host(seq, T);
        ++n;
    }
    dd_host_n += n;
    return n;
}

// DS4_HELPER_ARMED: every local helper's window for request `seq` (layer li, T rows) armed now (not for a pool client)
bool HelperSet::armed_on() const {
    if (client_) return false;
    for (const auto& h : hs_)
        if (h->armed_on()) return true;
    return false;
}
bool HelperSet::arm(int li, int T, unsigned seq) {
    if (client_) return false;
    bool any = false;
    for (auto& h : hs_) any = h->arm(li, T, seq) || any;
    return any;
}
void HelperSet::arm_drop() {
    for (auto& h : hs_) h->arm_drop();
}

std::string HelperSet::describe() const {
    std::string s;
    for (const auto& h : hs_) s += h->describe();
    for (const auto& r : rem_) {
        char buf[320];
        std::snprintf(buf, sizeof buf, "helper: pool client of %s (server pid %d) routes %lld experts to the shared helper%s", r->dir.c_str(),
                      r->server_pid, (long long) r->resident, r->dead ? "; DROPPED (the server stopped answering)" : "");
        s += buf;
        if (r->st.windows > 0) {
            std::snprintf(buf, sizeof buf, "; served %lld experts in %lld windows (%lld tokens), host %.0f ms issuing + %.0f ms waiting",
                          (long long) r->st.experts, (long long) r->st.windows, (long long) r->st.tokens, r->st.begin_ms, r->st.wait_ms);
            s += buf;
        }
        s += "\n";
    }
    for (const auto& sv : srv_) {
        if (sv->served == 0 && sv->attached == 0) continue;
        char buf[320];
        std::snprintf(buf, sizeof buf, "helper: pool server %s: %llu requests (%llu experts) answered in %llu passes (max %llu a pass; %zu streams), %llu clients attached, %llu reaped, %llu refused\n",
                      sv->dir.c_str(), (unsigned long long) sv->served, (unsigned long long) sv->served_picks,
                      (unsigned long long) sv->passes, (unsigned long long) sv->pass_max, sv->views.size(),
                      (unsigned long long) sv->attached, (unsigned long long) sv->reaped, (unsigned long long) sv->refused);
        s += buf;
    }
    return s;
}

bool HelperSet::fill_client(const std::vector<LayerGeo>& layers, const Helper::OnPrimary& on_primary, std::string& err) {
    const uint64_t id = model_identity(layers);
    int n_exp = 0, n_embd = 0, n_ff = 0;
    for (const LayerGeo& l : layers) {
        n_exp = std::max(n_exp, l.n_exp);
        if (l.eligible) {
            n_embd = std::max(n_embd, l.n_embd);
            n_ff = std::max(n_ff, l.n_ff);
        }
    }
    if (n_embd <= 0 || n_embd > kMaxDim) {
        err = "STRATA_POOL_CLIENT: no eligible layer (GGUF experts, trunk, one-member groups)";
        return false;
    }
    std::vector<uint8_t> claimed((size_t) layers.size() * (size_t) n_exp, 0);
    for (auto& rp : rem_) {
        Remote& m = *rp;
        const std::string plan_path = m.dir + "/plan.bin";
        // the serving engine publishes plan.bin once its helper is filled; wait for it (up to 15 min)
        int fd = -1;
        for (int i = 0; i < 9000 && (fd = ::open(plan_path.c_str(), O_RDONLY)) < 0; ++i) {
            if (i % 50 == 0) std::fprintf(stderr, "ds4 pool: waiting for %s (the serving engine fills its helper first)\n", plan_path.c_str());
            usleep(100000);
        }
        if (fd < 0) {
            err = "STRATA_POOL_CLIENT: no " + plan_path;
            return false;
        }
        DsPlan plan{};
        std::vector<int32_t> pairs;
        const bool got = ::read(fd, &plan, sizeof plan) == (ssize_t) sizeof plan;
        if (got && plan.magic == kMainPlanMagic) {
            ::close(fd);
            err = "STRATA_POOL_CLIENT: " + plan_path + " belongs to a Qwen / main-engine pool server (plan magic YSPP), not a "
                  "DeepSeek-V4.1 (strata-dsv41) one; refusing to attach";
            return false;
        }
        bool ok = got && plan.magic == kDsPlanMagic && plan.ver == kDsPoolVer && plan.count > 0 &&
                  plan.count <= (int64_t) layers.size() * n_exp;
        if (ok) {
            pairs.resize((size_t) plan.count * 2);
            ok = ::read(fd, pairs.data(), pairs.size() * sizeof(int32_t)) == (ssize_t) (pairs.size() * sizeof(int32_t));
        }
        ::close(fd);
        if (!ok) {
            err = "STRATA_POOL_CLIENT: " + plan_path + " is not a valid strata-dsv41 pool plan";
            return false;
        }
        if (plan.model_id != id || plan.n_embd != (uint32_t) n_embd || plan.n_ff != (uint32_t) n_ff || plan.n_layers != (int32_t) layers.size() ||
            plan.n_exp != n_exp) {
            err = "STRATA_POOL_CLIENT: the serving engine runs another model or pack (plan.bin identity differs: layers / experts / "
                  "geometry / expert blob layout)";
            return false;
        }
        m.n_exp = n_exp;
        m.n_embd = n_embd;
        m.server_pid = plan.server_pid;
        m.holds.assign((size_t) layers.size() * (size_t) n_exp, 0);
        m.resident = 0;
        for (int64_t i = 0; i < plan.count; ++i) {
            const int32_t l = pairs[(size_t) i * 2], e = pairs[(size_t) i * 2 + 1];
            if (l < 0 || l >= (int32_t) layers.size() || e < 0 || e >= n_exp || !layers[(size_t) l].eligible || e >= layers[(size_t) l].n_exp) continue;
            const size_t ix = (size_t) l * (size_t) n_exp + (size_t) e;
            if (claimed[ix] || on_primary(l, e)) continue;   // another helper of this process has it / a primary slot does
            m.holds[ix] = 1;
            claimed[ix] = 1;
            ++m.resident;
        }
        // the mailbox: created under a temporary name, renamed whole (the server never sees it half made); the
        // exclusive flock lives as long as this process (the server's reaper drops the mailbox when it can take it)
        m.path = m.dir + "/mb-" + std::to_string(getpid()) + "-" + std::to_string(&rp - &rem_[0]);
        const std::string tmp = m.path + ".tmp";
        m.fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        const bool made = m.fd >= 0 && ftruncate(m.fd, (off_t) mb_bytes()) == 0 && flock(m.fd, LOCK_EX) == 0;
        void* map = made ? mmap(nullptr, mb_bytes(), PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0) : MAP_FAILED;
        if (map == MAP_FAILED) {
            err = "STRATA_POOL_CLIENT: cannot create mailbox " + tmp + ": " + std::strerror(errno);
            if (m.fd >= 0) {
                unlink(tmp.c_str());
                ::close(m.fd);
                m.fd = -1;
            }
            return false;
        }
        m.mb = static_cast<DsMb*>(map);
        std::memset(static_cast<void*>(m.mb), 0, sizeof(DsMb));
        m.mb->ver = kDsPoolVer;
        m.mb->model_id = id;
        m.mb->client_pid = getpid();
        std::atomic_thread_fence(std::memory_order_release);
        m.mb->magic = kDsMbMagic;
        if (rename(tmp.c_str(), m.path.c_str()) != 0) {
            err = "STRATA_POOL_CLIENT: rename " + tmp + ": " + std::strerror(errno);
            unlink(tmp.c_str());
            return false;   // (~Remote unmaps and closes; m.path does not exist, its unlink is harmless)
        }
    }
    return true;
}

bool HelperSet::start_servers(const std::vector<LayerGeo>& layers, std::string& err) {
    const char* serve = env_nonempty("STRATA_POOL_SERVE");
    if (!serve) return true;
    const uint64_t id = model_identity(layers);
    int n_exp = 0, n_embd = 0, n_ff = 0;
    for (const LayerGeo& l : layers) {
        n_exp = std::max(n_exp, l.n_exp);
        if (l.eligible) {
            n_embd = std::max(n_embd, l.n_embd);
            n_ff = std::max(n_ff, l.n_ff);
        }
    }
    for (size_t k = 0; k < hs_.size(); ++k) {
        const std::string dir = pool_dir_for(serve, k);
        if (dir.empty()) continue;   // fewer directories than helpers: the later helpers stay private to this engine
        auto sv = std::make_unique<Server>();
        sv->dir = dir;
        sv->index = k;
        sv->src = hs_[k].get();
        sv->model_id = id;
        int n_streams = 2;
        if (const char* e = env_nonempty("DS4_POOL_STREAMS")) n_streams = std::atoi(e);
        n_streams = std::max(1, std::min(4, n_streams));
        if (const char* e = env_nonempty("DS4_POOL_UNION"))
            if (std::atoi(e) == 0) sv->max_batch = 1;
        for (int v = 0; v < n_streams; ++v) {
            auto view = hs_[k]->make_view(err);
            if (!view) {
                err = "STRATA_POOL_SERVE: " + err;
                return false;
            }
            sv->views.push_back(std::move(view));
        }
        std::vector<int32_t> pairs;
        for (int li = 0; li < (int) layers.size(); ++li)
            for (int e = 0; e < layers[(size_t) li].n_exp; ++e)
                if (hs_[k]->owns(li, e)) {
                    pairs.push_back(li);
                    pairs.push_back(e);
                }
        DsPlan plan{kDsPlanMagic, kDsPoolVer, id, (uint32_t) n_embd, (uint32_t) n_ff, (int32_t) layers.size(), n_exp,
                    (int64_t) (pairs.size() / 2), (int32_t) getpid(), 0};
        mkdir(dir.c_str(), 0700);
        const std::string path = dir + "/plan.bin", tmp = path + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        const bool wrote = f && std::fwrite(&plan, sizeof plan, 1, f) == 1 &&
                           std::fwrite(pairs.data(), sizeof(int32_t), pairs.size(), f) == pairs.size();
        if (f) std::fclose(f);
        if (!wrote || rename(tmp.c_str(), path.c_str()) != 0) {
            err = "STRATA_POOL_SERVE: cannot write " + path;
            return false;
        }
        Server* sp = sv.get();
        sv->th = std::thread([sp, n_embd] {
            if (const char* cores = env_nonempty("STRATA_POOL_CORE")) {   // "c[,c...]": the n-th server on the n-th core
#if defined(__linux__)
                const std::string core = pool_dir_for(cores, sp->index);
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(core.empty() ? std::atoi(cores) : std::atoi(core.c_str()), &set);
                sched_setaffinity(0, sizeof set, &set);
#endif
            }
            struct Client {
                int fd;
                DsMb* mb;
                uint64_t seen_seq = 0, seen_ns = 0;   // the request seq last seen pending and when (arrival order)
                bool inflight = false;                // in a lane's pass: not reaped, not taken again
            };
            // a lane: one view (stream) with the pass it is running
            struct Lane {
                Helper* h = nullptr;
                bool busy = false;
                int n = 0;
                Helper::PoolReq rq[kWin];
                Client* cl[kWin] = {};
                uint64_t seq[kWin] = {};
                int32_t np[kWin] = {};
                std::vector<Pick> pk;   // [kWin][kMaxPairs]
            };
            std::map<std::string, Client> clients;
            std::vector<Lane> lanes(sp->views.size());
            for (size_t v = 0; v < lanes.size(); ++v) {
                lanes[v].h = sp->views[v].get();
                lanes[v].pk.resize((size_t) kWin * kMaxPairs);
            }
            auto publish = [](Client* c, uint64_t sq, int32_t status) {
                c->mb->status = status;
                c->mb->resp_seq.store(sq, std::memory_order_release);
                c->inflight = false;
            };
            uint64_t last_scan = 0;
            std::vector<Client*> pend;
            pend.reserve(64);
            // the directory held open and rewound for each scan (opendir allocated its 32 KB buffer every 100 ms, freed
            // again: a thread-arena heap grown and trimmed by mprotect / madvise on this engine's mm, steady state)
            DIR* dd = nullptr;
            while (!sp->stop.load(std::memory_order_relaxed)) {
                const uint64_t now = pool_now_ns();
                if (now - last_scan > 100'000'000ull) {   // every 100 ms: attach new mailboxes, reap dead clients
                    last_scan = now;
                    os::malloc_pin_thread();   // this thread's arena pre-grown once the decode is pinned
                    if (dd == nullptr) dd = opendir(sp->dir.c_str());
                    else rewinddir(dd);
                    if (dd != nullptr) {
                        while (dirent* de = readdir(dd)) {
                            const std::string name = de->d_name;
                            if (name.rfind("mb-", 0) != 0 || name.find(".tmp") != std::string::npos || clients.count(name)) continue;
                            const int fd = ::open((sp->dir + "/" + name).c_str(), O_RDWR);
                            if (fd < 0) continue;
                            struct stat sb{};
                            void* v = (fstat(fd, &sb) == 0 && (size_t) sb.st_size >= mb_bytes())
                                          ? mmap(nullptr, mb_bytes(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
                            if (v == MAP_FAILED) {
                                ::close(fd);
                                continue;
                            }
                            auto* mb = static_cast<DsMb*>(v);
                            if (mb->magic != kDsMbMagic || mb->ver != kDsPoolVer || mb->model_id != sp->model_id) {
                                munmap(v, mb_bytes());
                                ::close(fd);
                                ++sp->refused;
                                continue;
                            }
                            clients[name] = Client{fd, mb};
                            ++sp->attached;
                            std::fprintf(stderr, "ds4 pool: attached %s (pid %d)\n", name.c_str(), mb->client_pid);
                        }
                    }
                    for (auto it = clients.begin(); it != clients.end();) {
                        if (!it->second.inflight && flock(it->second.fd, LOCK_EX | LOCK_NB) == 0) {   // the client's lifetime lock is gone
                            std::fprintf(stderr, "ds4 pool: reaped %s (pid %d)\n", it->first.c_str(), it->second.mb->client_pid);
                            munmap(it->second.mb, mb_bytes());
                            ::close(it->second.fd);
                            unlink((sp->dir + "/" + it->first).c_str());
                            it = clients.erase(it);
                            ++sp->reaped;
                        } else {
                            ++it;
                        }
                    }
                }
                bool progress = false;
                // 1. passes that finished: publish their answers
                for (Lane& ln : lanes) {
                    if (!ln.busy) continue;
                    const int r = ln.h->pass_poll();
                    if (r == 0) continue;
                    progress = true;
                    if (r > 0) ln.h->pass_finish();
                    for (int j = 0; j < ln.n; ++j) {
                        const bool ok = r > 0 && ln.rq[j].ok;
                        if (ok) {
                            ++sp->served;
                            sp->served_picks += (uint64_t) ln.np[j];
                        }
                        publish(ln.cl[j], ln.seq[j], ok ? 0 : 3);
                    }
                    ++sp->passes;
                    sp->pass_reqs += (uint64_t) ln.n;
                    sp->pass_max = std::max<uint64_t>(sp->pass_max, (uint64_t) ln.n);
                    ln.busy = false;
                }
                // 2. the pending requests, in arrival order (the order this loop first saw each one)
                bool idle_lane = false;
                for (const Lane& ln : lanes) idle_lane = idle_lane || !ln.busy;
                if (idle_lane) {
                    pend.clear();
                    for (auto& [name, c] : clients) {
                        if (c.inflight) continue;
                        const uint64_t sq = c.mb->req_seq.load(std::memory_order_acquire);
                        if (sq == c.mb->resp_seq.load(std::memory_order_relaxed)) continue;
                        if (c.seen_seq != sq) {
                            c.seen_seq = sq;
                            c.seen_ns = now;
                        }
                        pend.push_back(&c);
                    }
                    // (stable, by arrival: an insertion sort of the few pending - std::stable_sort took a temporary buffer
                    // from the heap on every pass of this loop)
                    for (size_t a = 1; a < pend.size(); ++a)
                        for (size_t b = a; b > 0 && pend[b]->seen_ns < pend[b - 1]->seen_ns; --b) std::swap(pend[b], pend[b - 1]);
                    // 3. every idle lane takes what is pending (up to a window's rows), oldest first
                    size_t next = 0;
                    for (Lane& ln : lanes) {
                        if (ln.busy) continue;
                        ln.n = 0;
                        int rows = 0;
                        while (next < pend.size() && ln.n < sp->max_batch) {
                            Client& c = *pend[next];
                            DsMb& m = *c.mb;
                            const uint64_t sq = m.req_seq.load(std::memory_order_acquire);
                            const int32_t layer = m.layer, T = m.T, np = m.n_pick;
                            int32_t status = 0;
                            if (layer < 0 || T < 1 || T > kWin || np < 1 || np > kMaxPairs) status = 1;
                            if (status == 0 && rows + T > kWin) break;   // the next pass (arrival order kept)
                            Pick* pk = ln.pk.data() + (size_t) ln.n * kMaxPairs;
                            for (int32_t j = 0; j < np && status == 0; ++j) {
                                const int32_t t = m.tok[j], i = m.slot[j], e = m.expert[j];
                                if (t < 0 || t >= T || i < 0 || i >= kPicks || e < 0) { status = 1; break; }
                                if (!sp->src->owns(layer, e)) { status = 2; break; }
                                pk[j] = {t, i, e};
                            }
                            ++next;
                            progress = true;
                            if (status != 0) {
                                ++sp->bad_reqs;
                                publish(&c, sq, status);
                                continue;
                            }
                            Helper::PoolReq& q = ln.rq[ln.n];
                            q = Helper::PoolReq{};
                            q.li = layer;
                            q.x = mb_x(&m);
                            q.w = m.w;
                            q.T = T;
                            q.picks = pk;
                            q.n = np;
                            q.out = mb_out(&m);
                            q.has = m.has;
                            ln.cl[ln.n] = &c;
                            ln.seq[ln.n] = sq;
                            ln.np[ln.n] = np;
                            c.inflight = true;
                            rows += T;
                            ++ln.n;
                        }
                        if (ln.n == 0) continue;
                        if (ln.h->pass_begin(ln.rq, ln.n)) {
                            ln.busy = true;
                        } else {
                            for (int j = 0; j < ln.n; ++j) publish(ln.cl[j], ln.seq[j], 3);
                            ln.n = 0;
                        }
                    }
                }
                if (!progress) DS4_POOL_PAUSE();
            }
            // (a pass still in flight at stop: the lanes' events are waited by the views' destructors' device sync)
            for (Lane& ln : lanes)
                while (ln.busy && ln.h->pass_poll() == 0) {}
            for (auto& [name, c] : clients) {
                munmap(c.mb, mb_bytes());
                ::close(c.fd);
            }
            if (dd != nullptr) closedir(dd);
            (void) n_embd;
        });
        std::fprintf(stderr, "helper: pool: device %d serves %lld experts to other engines through %s (%zu streams, up to %d requests a pass)\n",
                     hs_[k]->device(), (long long) plan.count, dir.c_str(), sv->views.size(), sv->max_batch);
        srv_.push_back(std::move(sv));
    }
    return true;
}

void HelperSet::stop_servers() {
    for (auto& sv : srv_) {
        sv->stop.store(true);
        if (sv->th.joinable()) sv->th.join();
        unlink((sv->dir + "/plan.bin").c_str());
        std::fprintf(stderr, "helper: pool server %s stopped: %llu requests (%llu experts) answered in %llu passes (mean %.2f, max %llu a pass; %zu streams), %llu bad, %llu clients attached, %llu reaped, %llu refused\n",
                     sv->dir.c_str(), (unsigned long long) sv->served, (unsigned long long) sv->served_picks,
                     (unsigned long long) sv->passes, sv->passes ? (double) sv->pass_reqs / (double) sv->passes : 0.0,
                     (unsigned long long) sv->pass_max, sv->views.size(), (unsigned long long) sv->bad_reqs,
                     (unsigned long long) sv->attached, (unsigned long long) sv->reaped, (unsigned long long) sv->refused);
    }
    srv_.clear();   // views before their sources (hs_)
}

}  // namespace strata::ds4::helper
