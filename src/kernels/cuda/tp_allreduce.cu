// src/kernels/cuda/tp_allreduce.cu - see include/strata/kernels/tp_allreduce.hpp.  The kernels are tp_proof_bench's
// (src/kernels/tp_proof_bench.cpp, measured there) without the timing stamps.
#include "strata/kernels/tp_allreduce.hpp"

#include <cuda_runtime.h>
#if !defined(__HIPCC__) && !defined(__CUDACC__)
#include <vector_types.h>   // the host syntax check (metal_compat): float4 / make_float4
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {

namespace {

#if defined(__HIPCC__) || defined(__CUDACC__)
#define TP_LAUNCH(kern, grid, block, stream, ...) kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#else   // the host syntax check: the kernels are plain functions there
#define TP_LAUNCH(kern, grid, block, stream, ...) ((void) (grid), (void) (block), (void) (stream), kern(__VA_ARGS__))
#endif

constexpr int kArThreads = 256;
constexpr int kSlots = 4;

__device__ __forceinline__ unsigned long long tp_now_ns() {
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

__device__ __forceinline__ void tp_pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(100);
#endif
}

// The put: this member's partial x -> its host-mapped row of the call's slot, a system fence, and the last block to
// finish raises the member's flag of the slot to the call's sequence number (*ctr + 1, device memory).
__global__ void __launch_bounds__(kArThreads)
    tp_ar_put_kernel(const float* x, int n4, float* rows, unsigned long long stride, unsigned* flags, int mi, int slots,
                     const unsigned* ctr, unsigned* cnt, unsigned long long* st) {
    if (st != nullptr && blockIdx.x == 0 && threadIdx.x == 0) st[0] = tp_now_ns();   // STRATA_TP_TIMING: the put starts
    const unsigned seq = *(const volatile unsigned*) ctr + 1u;
    const int slot = (int) (seq % (unsigned) slots);
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

// The sum: wait (bounded) until every other member's flag of the slot reached the sequence number, then x = the
// members' partials summed in member order (this member's from x, the others' from their host rows), the same bits on
// every member.  The last block advances *ctr and publishes the progress word.
__global__ void __launch_bounds__(kArThreads)
    tp_ar_sum_kernel(float* x, int n4, const float* rows, unsigned long long stride, const unsigned* flags, int nm, int mi,
                     int slots, unsigned* ctr, unsigned* cnt, unsigned* err, unsigned long long timeout_ns,
                     unsigned* progress, unsigned long long* st) {
    const unsigned seq = *(const volatile unsigned*) ctr + 1u;
    const int slot = (int) (seq % (unsigned) slots);
    __shared__ int bad;
    if (threadIdx.x == 0) {
        bad = 0;
        const unsigned long long t0 = tp_now_ns();
        if (st != nullptr && blockIdx.x == 0) st[1] = t0;   // this card's partial is out: it waits for the others from here
        for (int m = 0; m < nm && !bad; ++m) {
            if (m == mi) continue;
            const volatile unsigned* f = flags + ((size_t) m * slots + slot) * 32;
            while ((int) (*f - seq) < 0) {
                if (*(const volatile unsigned*) err != 0u) { bad = 1; break; }
                if (tp_now_ns() - t0 > timeout_ns) {
                    *(volatile unsigned*) err = 1u;
                    __threadfence_system();
                    bad = 1;
                    break;
                }
                tp_pause();
            }
        }
        __threadfence_system();
        if (st != nullptr && blockIdx.x == 0) st[2] = tp_now_ns();   // every other card's partial is in
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
        if (st != nullptr) st[3] = tp_now_ns();   // the sum is done (the last block)
        *(volatile unsigned*) ctr = seq;
        __threadfence_system();
        *(volatile unsigned*) progress = seq;
        __threadfence_system();
    }
}

// ---- cycle 3: the card's own plan on the device (tp_resident_plan below).  One thread per entry (<= kTpPlanMax).
constexpr int kTpPlanMax = 256;
__global__ void __launch_bounds__(kTpPlanMax)
    tp_plan_kernel(const int32_t* ids, int n, int k, const int32_t* res, const int32_t* peer_res, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* pl,
                   long long capx, uint32_t* skip, uint32_t ring, uint32_t* plan_err) {
    __shared__ int32_t s_id[kTpPlanMax];
    __shared__ int s_first[kTpPlanMax], s_size[kTpPlanMax];
    __shared__ int s_bad;
    const int i = threadIdx.x;
    if (i == 0) s_bad = 0;
    __syncthreads();
    int32_t e = -1;
    bool keep = false;
    if (i < n) {
        e = ids[i];
        if (e < 0 || e >= n_expert) s_bad = 1;
        else if (res[e] >= 0) keep = true;
        else if (peer_res != nullptr && peer_res[e] < 0) s_bad = 1;   // card 0: in neither card's VRAM - the CPU's
    }
    s_id[i] = keep ? e : -1;
    __syncthreads();
    if (s_bad) {
        if (i == 0) {
            if (skip != nullptr) *skip = 0u;
            else {   // (the plan of a card that owns only its VRAM experts never gets here)
                pl[0] = 0; pl[1] = 0; pl[2] = 0;
                if (plan_err != nullptr) { *(volatile uint32_t*) plan_err = 1u; __threadfence_system(); }
            }
        }
        return;
    }
    int first = -1, rank = 0, size = 0;
    if (keep) {
        first = i;
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same ? 1 : 0;
        }
    }
    s_first[i] = first;
    s_size[i] = size;
    __syncthreads();
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    if (keep) {
        // group = the first occurrences (of kept entries) before `first`; its start = their sizes' sum
        int group = 0, gstart = 0;
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) {
            const int32_t slot = res[e];
            ptr[group] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[group] = gstart;
        }
        dst[gstart + rank] = i;
        tok[gstart + rank] = i / k;
    }
    if (i == 0) {
        int groups = 0, kept = 0;
        for (int j = 0; j < n; ++j) {
            groups += s_first[j] == j ? 1 : 0;
            kept += s_id[j] >= 0 ? 1 : 0;
        }
        start[groups] = kept;
        start2[0] = kept;
        counts[0] = groups;
        counts[1] = kept;
        counts[2] = 0;
    }
    __threadfence();
    __syncthreads();
    if (i == 0 && skip != nullptr) *skip = ring;
}

// The doorbell of a card whose layer may be device-planned: planned (*skip == ring) -> no rows cross, the step's mark
// = ring, then the ring; else doorbell_publish_res (the ids, the activations when an id is not in this card's VRAM).
__global__ void tp_doorbell_kernel(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int n, int k,
                                   float* x_out, int32_t* ids_out, uint32_t* seq, const uint32_t* skip, uint32_t ring,
                                   uint32_t* mark) {
    __shared__ int s_any;
    if (*(const volatile uint32_t*) skip == ring) {
        if (threadIdx.x == 0) {
            *(volatile uint32_t*) mark = ring;
            __threadfence_system();
            *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
            __threadfence_system();
        }
        return;
    }
    if (threadIdx.x == 0) s_any = 0;
    __syncthreads();
    if ((int) threadIdx.x < k) {
        const int32_t id = ids[threadIdx.x];
        ids_out[threadIdx.x] = id;
        if (id < 0 || id >= n_expert || d_res[id] < 0) s_any = 1;
    }
    __syncthreads();
    if (s_any)
        for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
        __threadfence_system();
    }
}

// wait_copy_i32_from_mapped unless the device planned the step
__global__ void tp_wait_copy_plan_kernel(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int n,
                                         const uint32_t* skip) {
    __shared__ int s_skip;
    if (threadIdx.x == 0) {
        s_skip = *(const volatile uint32_t*) skip == value ? 1 : 0;
        if (!s_skip) {
            while (*(const volatile uint32_t*) flag < value) tp_pause();
            __threadfence_system();
        }
    }
    __syncthreads();
    if (s_skip) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = ((const volatile int32_t*) src)[i];
}

// copy_rows_from_mapped_after unless the device planned the step (then every row is zero: no CPU rows)
__global__ void tp_copy_rows_kernel(float4* dst, const float4* src, long long row4, const int32_t* hit_rows,
                                    const int32_t* count, const uint32_t* flag, uint32_t value, const uint32_t* skip) {
    const int row = blockIdx.x;
    __shared__ int s_zero;
    if (threadIdx.x == 0) {
        int z = *(const volatile uint32_t*) skip == value ? 1 : 0;
        if (!z) {
            while (*(const volatile uint32_t*) flag < value) tp_pause();
            __threadfence_system();
            const int c = *(const volatile int32_t*) count;
            for (int i = 0; i < c; ++i) z |= hit_rows[i] == row ? 1 : 0;
        }
        s_zero = z;
    }
    __syncthreads();
    float4* d = dst + (long long) row * row4;
    if (s_zero) {
        for (long long i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const float4* sr = src + (long long) row * row4;   // (mapped rows written before the flag: plain loads after it)
        for (long long i = threadIdx.x; i < row4; i += blockDim.x) d[i] = sr[i];
    }
}

}  // namespace

struct TpArGroup {
    int nm = 0, slots = kSlots;
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
    // 60 s (the proof's 5 s is too short here: a layer's CPU experts are served by the host between the two cards'
    // all-reduces, and a cold expert read from the model file can hold one card that long; the host's own per-layer
    // bound is 20 s).  STRATA_TP_AR_TIMEOUT_MS overrides.
    unsigned long long timeout_ns = 60000000000ull;
};

TpArGroup* tp_ar_open(const std::vector<int>& devs, size_t max_floats, std::string& err) {
    auto* g = new TpArGroup();
    auto fail = [&](const char* what, cudaError_t e) -> TpArGroup* {
        err = std::string("tensor parallel all-reduce: ") + what + ": " + cudaGetErrorString(e);
        (void) cudaGetLastError();
        return nullptr;   // (a failed start: the process ends; nothing is freed)
    };
    g->nm = (int) devs.size();
    g->stride = (max_floats + 63) / 64 * 64;
    if (const char* t = std::getenv("STRATA_TP_AR_TIMEOUT_MS"); t != nullptr && std::atoll(t) > 0)
        g->timeout_ns = (unsigned long long) std::atoll(t) * 1000000ull;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaError_t e = cudaSetDevice(devs[0]);
    if (e != cudaSuccess) return fail("set device", e);
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    const size_t rows_b = (size_t) g->nm * g->slots * g->stride * 4 + 64, flags_b = (size_t) g->nm * g->slots * 32 * 4 + 64,
                 prog_b = (size_t) g->nm * 32 * 4 + 64;
    if ((e = cudaHostAlloc((void**) &g->rows_h, rows_b, fl)) != cudaSuccess) return fail("rows", e);
    if ((e = cudaHostAlloc((void**) &g->flags_h, flags_b, fl)) != cudaSuccess) return fail("flags", e);
    if ((e = cudaHostAlloc((void**) &g->err_h, 64, fl)) != cudaSuccess) return fail("error word", e);
    if ((e = cudaHostAlloc((void**) &g->progress_h, prog_b, fl)) != cudaSuccess) return fail("progress", e);
    std::memset(g->rows_h, 0, rows_b);
    std::memset(g->flags_h, 0, flags_b);
    std::memset(g->err_h, 0, 64);
    std::memset(g->progress_h, 0, prog_b);
    g->m.resize(devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        TpArGroup::Member& M = g->m[i];
        M.dev = devs[i];
        if ((e = cudaSetDevice(M.dev)) != cudaSuccess) return fail("set device", e);
        if ((e = cudaHostGetDevicePointer((void**) &M.rows, g->rows_h, 0)) != cudaSuccess ||
            (e = cudaHostGetDevicePointer((void**) &M.flags, g->flags_h, 0)) != cudaSuccess ||
            (e = cudaHostGetDevicePointer((void**) &M.err, g->err_h, 0)) != cudaSuccess ||
            (e = cudaHostGetDevicePointer((void**) &M.progress, g->progress_h, 0)) != cudaSuccess)
            return fail("device pointers", e);
        M.progress += i * 32;
        if ((e = cudaMalloc((void**) &M.ctr, 256)) != cudaSuccess) return fail("counters", e);
        if ((e = cudaMemset(M.ctr, 0, 256)) != cudaSuccess) return fail("counters", e);
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) return fail("counters", e);
        M.cnt_put = M.ctr + 16;
        M.cnt_sum = M.ctr + 32;
    }
    cudaSetDevice(cur);
    return g;
}

void tp_ar_call(TpArGroup* g, int mi, float* x, int n, void* stream, unsigned long long* stamps) {
    if (g == nullptr || mi < 0 || mi >= g->nm) throw std::runtime_error("tp_ar_call: no such member");
    if (n % 4 != 0 || (size_t) n > g->stride) throw std::runtime_error("tp_ar_call: the payload must be a multiple of 4 floats and fit a row");
    const TpArGroup::Member& M = g->m[(size_t) mi];
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, (n4 + kArThreads - 1) / kArThreads));
    cudaStream_t s = (cudaStream_t) stream;
    TP_LAUNCH(tp_ar_put_kernel, blocks, kArThreads, s, x, n4, M.rows, (unsigned long long) g->stride, M.flags, mi, g->slots,
                                                   M.ctr, M.cnt_put, stamps);
    TP_LAUNCH(tp_ar_sum_kernel, blocks, kArThreads, s, x, n4, M.rows, (unsigned long long) g->stride, M.flags, g->nm, mi,
                                                   g->slots, M.ctr, M.cnt_sum, M.err, g->timeout_ns, M.progress, stamps);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("tp_ar_call: ") + cudaGetErrorString(e));
}

bool tp_ar_failed(const TpArGroup* g) { return g != nullptr && *(const volatile unsigned*) g->err_h != 0u; }
unsigned tp_ar_progress(const TpArGroup* g, int mi) {
    return g == nullptr ? 0u : *(const volatile unsigned*) (g->progress_h + (size_t) mi * 32);
}
int tp_ar_members(const TpArGroup* g) { return g == nullptr ? 0 : g->nm; }

void tp_resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, const int32_t* peer_res_layer,
                      int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                      int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    if (n_entries < 1 || n_entries > kTpPlanMax) throw std::runtime_error("tp_resident_plan: 1.." + std::to_string(kTpPlanMax) + " entries");
    TP_LAUNCH(tp_plan_kernel, 1, kTpPlanMax, (cudaStream_t) stream, ids, n_entries, k, res_layer, peer_res_layer, n_expert,
              cache_base, slot_off, blob, plan, capx, skip, ring, plan_err);
}
void tp_doorbell(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k, float* x_out,
                 int32_t* ids_out, uint32_t* d_seq, const uint32_t* skip, uint32_t ring, uint32_t* d_mark, void* stream) {
    if (k > 1024) throw std::runtime_error("tp_doorbell: k too large");
    TP_LAUNCH(tp_doorbell_kernel, 1, 1024, (cudaStream_t) stream, x, ids, d_res, n_expert, (int) n, (int) k, x_out, ids_out,
              d_seq, skip, ring, d_mark);
}
void tp_wait_copy_plan(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n, const uint32_t* skip,
                       void* stream) {
    TP_LAUNCH(tp_wait_copy_plan_kernel, 1, 128, (cudaStream_t) stream, flag, value, dst, src, (int) n, skip);
}
void tp_copy_rows(const uint32_t* flag, uint32_t value, float* dst, const float* src, int64_t rows, int64_t width,
                  const int32_t* hit_rows, const int32_t* count, const uint32_t* skip, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0)
        throw std::runtime_error("tp_copy_rows: width must be a multiple of 4 and both pointers 16-byte aligned");
    TP_LAUNCH(tp_copy_rows_kernel, (unsigned) rows, 128, (cudaStream_t) stream, (float4*) dst, (const float4*) src,
              (long long) (width / 4), hit_rows, count, flag, value, skip);
}

}  // namespace strata::kernels
