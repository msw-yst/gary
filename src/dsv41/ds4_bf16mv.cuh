// src/dsv41/ds4_bf16mv.cuh - (DS4_BF16MV) the BF16 matvec's job layout and kernels, included by ds4_gguf.cu inside its
// strata::ds4::gguf anonymous namespace (and by tests/dsv41/bf16mv_bench.cpp, which defines the same three names first).
// Needs, already declared: kWarps / kThreads (8 warps of 32 lanes), warp_sum (the xor butterfly 16, 8, 4, 2, 1) and
// x3::kMaxMvJobs.
//
// BfJob / BfJobs / bf16_mv_kernel: moved here VERBATIM from ds4_gguf.cu (build 17); the default path, unchanged.
//
// bf16_mv2_kernel<NC, R, U, NT> (DS4_BF16MV=<variant>, default off): the same arithmetic re-scheduled for bandwidth.
//   Why: bf16_mv_kernel's loop (gfx1100 ISA, build 17) issues ONE 16-byte weight load a lane, waits for it (s_waitcnt vmcnt
//   right after the issue), does its 8 FMAs a column, and only then issues the next - one 512-byte request a wave in flight,
//   a DRAM round trip per 256 values of the row.  A row is a warp, so the bytes in flight on the card are (rows resident) x
//   512 B: the lead's 1,024-1,792-row jobs (wo_a, wq_a + wkv, the shared expert) cannot cover the latency, and short rows
//   (k 512-1,280) spend a large part of their time outside the loop.  It also re-reads the job's x / xcs from the kernel
//   arguments every step (s_load in the loop).
//   What: (1) U weight loads a lane issued before the first FMA (U x 512 B a row in flight), the remainder (< U steps) also
//   issued all at once; (2) R rows a warp, each activation load feeding R rows (x traffic / R); (3) the job's fields read
//   once; (4) NT: the weights as non-temporal loads (streamed once a step; keeps the activations in cache) - HIP only.
//   Bit-identical BY CONSTRUCTION: every output (row, column) is the same chain of fmaf in the same order as
//   bf16_mv_kernel's VEC path - lane l takes elements l * 8 + 256 t, t ascending, the 8 of a step in order 0..7, each
//   fmaf(w, x, acc) - then the same warp_sum butterfly and the same alpha * v + bias expression (contracted the same way:
//   v_fmac_f32 in both, checked in the ISA).  Only the issue order of LOADS changes, and which warp computes a row.
//   The grid: (ceil(max rows / (8 R)), jobs).  Only the VEC case (rows and activations 16-byte aligned) takes it; any other
//   job set stays on bf16_mv_kernel.
#pragma once
struct BfJob {
    const void* w;
    size_t ld;          // values a row (BF16), bytes a row (MXFP4)
    const float* x;
    size_t xcs;
    float* y;
    size_t ycs;
    const float* bias;
    float alpha;
    int k, n, ncol;
    const int8_t* xq;   // (Q8_0: the activations rounded to int8, q8_quant's)
    const float* xd;
};
struct BfJobs {
    BfJob j[x3::kMaxMvJobs];
};

// grid (max rows / 8, jobs): a warp a row, each of its activations' dot in F32 - a lane 8 BF16 weights (16 bytes) and
// 8 activations a step (VEC), or one of each (rows or activations not 16-byte aligned)
template <int NC, bool VEC>
__global__ void __launch_bounds__(kThreads) bf16_mv_kernel(const BfJobs J) {
    const BfJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= b.n) return;
    const uint16_t* w = (const uint16_t*) b.w + (size_t) row * b.ld;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if (VEC) {
        for (int i = lane * 8; i < b.k; i += 256) {
            const uint4 q = *(const uint4*) (w + i);
            const float wf[8] = {__uint_as_float(q.x << 16), __uint_as_float(q.x & 0xffff0000u),
                                 __uint_as_float(q.y << 16), __uint_as_float(q.y & 0xffff0000u),
                                 __uint_as_float(q.z << 16), __uint_as_float(q.z & 0xffff0000u),
                                 __uint_as_float(q.w << 16), __uint_as_float(q.w & 0xffff0000u)};
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) {
                    const float4* xp = (const float4*) (b.x + c * b.xcs + i);
                    const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                    float a = acc[c];
                    a = fmaf(wf[0], x0.x, a);
                    a = fmaf(wf[1], x0.y, a);
                    a = fmaf(wf[2], x0.z, a);
                    a = fmaf(wf[3], x0.w, a);
                    a = fmaf(wf[4], x1.x, a);
                    a = fmaf(wf[5], x1.y, a);
                    a = fmaf(wf[6], x1.z, a);
                    a = fmaf(wf[7], x1.w, a);
                    acc[c] = a;
                }
        }
    } else {
        for (int i = lane; i < b.k; i += 32) {
            const float wf = __uint_as_float((uint32_t) w[i] << 16);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < b.ncol) acc[c] = fmaf(wf, b.x[c * b.xcs + i], acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < b.ncol) {
            const float v = warp_sum(acc[c]);
            if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
        }
}

// ---------------------------------------------------------------- (DS4_BF16MV) bf16_mv2_kernel, DS4_BF16MV
#if !defined(STRATA_METAL)
// the variants DS4_BF16MV names (bf16mv_bench's too): rows a warp R, weight steps in flight a lane U, non-temporal NT
struct Bf16mvVar {
    const char* name;
    int R, U;
    bool NT;
};
constexpr Bf16mvVar kBf16mvVars[] = {
    {"r1u2", 1, 2, false}, {"r1u4", 1, 4, false}, {"r1u8", 1, 8, false}, {"r2u2", 2, 2, false},
    {"r2u4", 2, 4, false}, {"r4u2", 4, 2, false}, {"r1u4n", 1, 4, true}, {"r2u4n", 2, 4, true},
};
constexpr int kBf16mvNVars = (int) (sizeof(kBf16mvVars) / sizeof(kBf16mvVars[0]));
constexpr int kBf16mvDefault = 1;   // DS4_BF16MV=1: r1u4 (bf16_mv_kernel's grid, U = 4)
// a DS4_BF16MV value -> a variant index; -1: off ("" / "0"); -2: not a known name
inline int bf16mv_parse(const char* e) {
    if (e == nullptr || e[0] == '\0' || (e[0] == '0' && e[1] == '\0')) return -1;
    if (e[0] == '1' && e[1] == '\0') return kBf16mvDefault;
    for (int i = 0; i < kBf16mvNVars; ++i) {
        const char* a = kBf16mvVars[i].name;
        const char* b = e;
        while (*a != '\0' && *a == *b) ++a, ++b;
        if (*a == '\0' && *b == '\0') return i;
    }
    return -2;
}

struct BfQ {   // 8 BF16 weights (16 bytes)
    uint32_t x, y, z, w;
};
template <bool NT>
__device__ __forceinline__ BfQ bf_ld16(const uint16_t* p) {
#if defined(STRATA_USE_HIP)
    typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));
    u32x4_t v;
    if constexpr (NT) v = __builtin_nontemporal_load((const u32x4_t*) p);
    else v = *(const u32x4_t*) p;
    return BfQ{v.x, v.y, v.z, v.w};
#else
    const uint4 v = *(const uint4*) p;
    return BfQ{v.x, v.y, v.z, v.w};
#endif
}
// one 256-value step at i (this lane's 8): bf16_mv_kernel's VEC body for R rows - per column c the activations' two float4
// loads, then per row the 8 fmaf in element order into acc[r][c]
template <int NC, int R>
__device__ __forceinline__ void bf_step(const BfQ (&q)[R], const float* x, size_t xcs, int ncol, int i,
                                        float (&acc)[R][NC]) {
    float wf[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wf[r][0] = __uint_as_float(q[r].x << 16);
        wf[r][1] = __uint_as_float(q[r].x & 0xffff0000u);
        wf[r][2] = __uint_as_float(q[r].y << 16);
        wf[r][3] = __uint_as_float(q[r].y & 0xffff0000u);
        wf[r][4] = __uint_as_float(q[r].z << 16);
        wf[r][5] = __uint_as_float(q[r].z & 0xffff0000u);
        wf[r][6] = __uint_as_float(q[r].w << 16);
        wf[r][7] = __uint_as_float(q[r].w & 0xffff0000u);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (c < ncol) {
            const float4* xp = (const float4*) (x + c * xcs + i);
            const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = acc[r][c];
                a = fmaf(wf[r][0], x0.x, a);
                a = fmaf(wf[r][1], x0.y, a);
                a = fmaf(wf[r][2], x0.z, a);
                a = fmaf(wf[r][3], x0.w, a);
                a = fmaf(wf[r][4], x1.x, a);
                a = fmaf(wf[r][5], x1.y, a);
                a = fmaf(wf[r][6], x1.z, a);
                a = fmaf(wf[r][7], x1.w, a);
                acc[r][c] = a;
            }
        }
}
// grid (ceil(max rows / (kWarps R)), jobs): warp w of block b takes rows (b kWarps + w) R .. + R - 1 (those < n)
template <int NC, int R, int U, bool NT>
__global__ void __launch_bounds__(kThreads) bf16_mv2_kernel(const BfJobs J) {
    const BfJob& b = J.j[blockIdx.y];
    const int n = b.n;
    const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= n) return;
    const int k = b.k, ncol = b.ncol;
    const size_t ld = b.ld, xcs = b.xcs;
    const float* const x = b.x;
    const uint16_t* w[R];
#pragma unroll
    for (int r = 0; r < R; ++r) w[r] = (const uint16_t*) b.w + (size_t) (row0 + r < n ? row0 + r : row0) * ld;
    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    int i = lane * 8;
    for (; i + (U - 1) * 256 < k; i += U * 256) {   // U whole steps: every load issued, then the FMAs in step order
        BfQ q[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int r = 0; r < R; ++r) q[u][r] = bf_ld16<NT>(w[r] + i + u * 256);
#pragma unroll
        for (int u = 0; u < U; ++u) bf_step<NC, R>(q[u], x, xcs, ncol, i + u * 256, acc);
    }
    if (U > 1 && i < k) {   // the last 1 .. U - 1 steps of this lane: issued together too
        BfQ q[U > 1 ? U - 1 : 1][R] = {};
#pragma unroll
        for (int u = 0; u < U - 1; ++u)
            if (i + u * 256 < k)
#pragma unroll
                for (int r = 0; r < R; ++r) q[u][r] = bf_ld16<NT>(w[r] + i + u * 256);
#pragma unroll
        for (int u = 0; u < U - 1; ++u)
            if (i + u * 256 < k) bf_step<NC, R>(q[u], x, xcs, ncol, i + u * 256, acc);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= n) break;   // (warp-uniform)
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < ncol) {
                const float v = warp_sum(acc[r][c]);
                if (lane == 0) b.y[c * b.ycs + row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
            }
    }
}
#endif  // !STRATA_METAL
