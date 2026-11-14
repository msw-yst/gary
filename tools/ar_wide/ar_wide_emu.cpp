// tools/ar_wide/ar_wide_emu.cpp - DS4_AR_WIDE CPU emulation (no GPU).  ar_rs_kernel / ar_rs16_kernel / ar_sum_kernel
// <W> extracted verbatim from src/dsv41/ds4_kernels.cu (extract_ar.py), run on CPU threads: every block x thread of every
// member a std::thread, a std::barrier a __syncthreads, plain memory for the host rows.  Each call runs on the same
// partials narrow (ld_fresh), then wide (ld_wide; DS4_AR_WIDE=1), and for ar_rs16_kernel also W 2 (wide loads + 16-byte
// stores; DS4_AR_WIDE=2), each with the next seq; every member's x is compared bit for bit between them, across members,
// and with a CPU reference (member-order __fadd_rn; F16: to_h4(sat4) of every partial and
// of the sum).  Odd chunk edges, 1-thread blocks and sizes that do not split evenly are in the matrix (wide_pairs' ends).
// Members run unsynchronised for the reduce-scatter kernels (their flags order them, as on the GPUs); the one-phase
// ar_sum_kernel path puts its row and flag first (xfer_put's job), with a barrier between calls (its slots' reuse is the
// engine's window ack, not modelled here).  Build / run: run_ar_emu.sh.  Exit 0: every case bit-identical.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define STRATA_USE_HIP 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(x)
struct dim3 {
    unsigned x = 0, y = 0, z = 0;
};
thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
thread_local std::barrier<>* tl_bar = nullptr;
thread_local std::mt19937* tl_rng = nullptr;
static void jitter() {
    if (tl_rng == nullptr) return;
    const unsigned r = (*tl_rng)() % 64u;
    if (r == 0) std::this_thread::sleep_for(std::chrono::microseconds((*tl_rng)() % 100u));
    else if (r < 8) std::this_thread::yield();
}
static void __syncthreads() {
    jitter();
    tl_bar->arrive_and_wait();
}
static void __threadfence_system() { std::atomic_thread_fence(std::memory_order_seq_cst); }
static void ar_emu_spin() { std::this_thread::yield(); }
struct float4 {
    float x, y, z, w;
};
struct float2 {
    float x, y;
};
struct uint2 {
    unsigned x, y;
};
struct uint4 {
    unsigned x, y, z, w;
};
static float4 make_float4(float a, float b, float c, float d) { return {a, b, c, d}; }
static uint2 make_uint2(unsigned a, unsigned b) { return {a, b}; }
static uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return {a, b, c, d}; }
struct __half2 {
    _Float16 x, y;
};
static __half2 __floats2half2_rn(float a, float b) { return {(_Float16) a, (_Float16) b}; }
static float2 __half22float2(__half2 h) { return {(float) h.x, (float) h.y}; }
static float __fadd_rn(float a, float b) {
    volatile float r = a + b;
    return r;
}
static int min(int a, int b) { return a < b ? a : b; }
static int max(int a, int b) { return a > b ? a : b; }
constexpr int kThreads = 256;
constexpr int kArBlocks = 512;

#include "ar_extract.inc"

template <class F>
static void launch(int grid, int block, unsigned seed, F&& f) {
    std::vector<std::thread> ts;
    std::vector<std::unique_ptr<std::barrier<>>> bars;
    for (int b = 0; b < grid; ++b) bars.emplace_back(new std::barrier<>(block));
    for (int b = 0; b < grid; ++b)
        for (int t = 0; t < block; ++t)
            ts.emplace_back([&, b, t] {
                std::mt19937 rng(seed * 7919u + (unsigned) (b * 131 + t));
                tl_rng = &rng;
                threadIdx.x = (unsigned) t;
                blockIdx.x = (unsigned) b;
                blockDim.x = (unsigned) block;
                gridDim.x = (unsigned) grid;
                tl_bar = bars[(size_t) b].get();
                f();
            });
    for (auto& t : ts) t.join();
}

static float* falloc(size_t n) {
    void* p = std::aligned_alloc(4096, ((n * 4 + 4095) / 4096) * 4096);
    std::memset(p, 0, ((n * 4 + 4095) / 4096) * 4096);
    return (float*) p;
}

// kind 0 ar_rs_kernel (F32), 1 ar_rs16_kernel (F16), 2 ar_sum_kernel (F32 one-phase)
static bool run_case(int kind, int nm, int n4, int grid, int block, unsigned seed, int calls) {
    const int slots = 2;
    const size_t stride = ((size_t) n4 * 4 + 7) / 8 * 8;
    float* rows = falloc((size_t) nm * slots * stride);
    float* sums = falloc((size_t) slots * stride);
    unsigned* bflags = (unsigned*) falloc((size_t) 2 * nm * slots * kArBlocks * 8);
    unsigned* flags = (unsigned*) falloc((size_t) nm * slots * 32);
    std::vector<std::vector<float>> part((size_t) nm, std::vector<float>((size_t) n4 * 4));
    const int nw = kind == 1 ? 3 : 2;
    std::vector<std::vector<float>> out[3];
    for (auto& o : out) o.assign((size_t) nm, {});
    std::barrier<> mbar(nm);
    bool ok = true;
    std::mt19937 g(seed);
    for (int c = 0; c < calls && ok; ++c) {
        for (int m = 0; m < nm; ++m)
            for (auto& v : part[(size_t) m]) {   // full mantissas, |v| 2^-30 .. 2^17 (FP16 zeros, subnormals, saturation)
                const float mant = (float) ((int) (g() % 2000001u) - 1000000) * 1e-6f;
                v = std::ldexp(mant, (int) (g() % 48u) - 30);
            }
        std::vector<std::thread> mt;
        for (int m = 0; m < nm; ++m)
            mt.emplace_back([&, m] {
                for (int w = 0; w < nw; ++w) {
                    const unsigned seq = (unsigned) (nw * c + w + 1);
                    const int slot = (int) (seq % (unsigned) slots);
                    std::vector<float> x = part[(size_t) m];
                    if (kind == 2) {
                        mbar.arrive_and_wait();   // (a call's slots free on every member: see the header)
                        std::memcpy(rows + ((size_t) m * slots + slot) * stride, x.data(), (size_t) n4 * 16);
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        *(volatile unsigned*) (flags + ((size_t) m * slots + slot) * 32) = seq;
                    }
                    launch(grid, block, seed + (unsigned) (m * 17 + w * 5 + c * 101), [&] {
                        if (kind == 0) {
                            if (w) ar_rs_kernel<true>(x.data(), n4, rows, sums, stride, bflags, nm, m, slot, slots, seq);
                            else ar_rs_kernel<false>(x.data(), n4, rows, sums, stride, bflags, nm, m, slot, slots, seq);
                        } else if (kind == 1) {
                            if (w == 2) ar_rs16_kernel<2>(x.data(), n4, rows, sums, stride, bflags, nm, m, slot, slots, seq);
                            else if (w == 1) ar_rs16_kernel<1>(x.data(), n4, rows, sums, stride, bflags, nm, m, slot, slots, seq);
                            else ar_rs16_kernel<0>(x.data(), n4, rows, sums, stride, bflags, nm, m, slot, slots, seq);
                        } else {
                            if (w) ar_sum_kernel<true>(x.data(), n4, rows, stride, flags, nm, m, slot, slots, seq);
                            else ar_sum_kernel<false>(x.data(), n4, rows, stride, flags, nm, m, slot, slots, seq);
                        }
                    });
                    out[w][(size_t) m] = x;
                    if (kind == 2) mbar.arrive_and_wait();
                }
            });
        for (auto& t : mt) t.join();
        // the CPU reference
        std::vector<float> ref((size_t) n4 * 4);
        for (int q = 0; q < n4; ++q) {
            float4 v{};
            for (int m = 0; m < nm; ++m) {
                const float* pp = part[(size_t) m].data() + (size_t) q * 4;
                float4 p = make_float4(pp[0], pp[1], pp[2], pp[3]);
                if (kind == 1) p = from_h4(to_h4(sat4(p)));
                if (m == 0) v = p;
                else v = make_float4(__fadd_rn(v.x, p.x), __fadd_rn(v.y, p.y), __fadd_rn(v.z, p.z), __fadd_rn(v.w, p.w));
            }
            if (kind == 1) v = from_h4(to_h4(sat4(v)));
            std::memcpy(ref.data() + (size_t) q * 4, &v, 16);
        }
        for (int m = 0; m < nm && ok; ++m)
            for (int w = 0; w < nw; ++w)
                if (std::memcmp(out[w][(size_t) m].data(), ref.data(), (size_t) n4 * 16) != 0) {
                    std::printf("FAIL kind %d nm %d n4 %d grid %d block %d seed %u call %d member %d: W %d != CPU reference\n", kind,
                                nm, n4, grid, block, seed, c, m, w);
                    ok = false;
                }
    }
    std::free(rows);
    std::free(sums);
    std::free(bflags);
    std::free(flags);
    return ok;
}

int main(int argc, char** argv) {
    const unsigned seed0 = argc > 1 ? (unsigned) std::atoi(argv[1]) : 1u;
    int cases = 0, bad = 0;
    unsigned seed = seed0;
    for (int kind : {0, 1, 2})
        for (int nm : {2, 3, 4})
            for (int n4 : {1, 2, 3, 5, 17, 64, 257, 1000})
                for (int grid : {1, 3, 5})
                    for (int block : {1, 4, 5, 8}) {
                        if (kind < 2 && nm < 3) continue;   // (the engine runs the reduce-scatter for three members or more)
                        // (the reduce-scatter kernels need a block of nm threads or more: thread m waits for member m's flag;
                        // the engine launches kThreads)
                        if (kind < 2 && block < nm) continue;
                        ++cases;
                        if (!run_case(kind, nm, n4, grid, block, seed++, 3)) ++bad;
                    }
    std::printf("ar_wide_emu: %d cases x 3 calls (W 0 / 1, and 2 for ar_rs16_kernel, each), %d failed: %s\n", cases, bad,
                bad ? "FAIL" : "all bit-identical (every W = the CPU reference, every member)");
    return bad ? 1 : 0;
}
