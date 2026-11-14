// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_kernels.cu - DeepSeek-V4.1's device kernels (see ds4_kernels.cuh).  No fast math here: the rotary
// angles, the softmaxes and the roundings follow the reference's F32 arithmetic.  A window's token is blockIdx.y.
#include "ds4_launch.h"   // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_kernels.cuh"
#if defined(STRATA_METAL)
#include <ds4_metal_host.h>
#endif
#include "ds4_numerics.cuh"
#include "ds4_q6k.cuh"
#include "ds4_attn_split.cuh"
#include "ds4_wmma.cuh"
#include "ds4_experts_rule.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <mutex>
#include <cstdio>

namespace strata::ds4 {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ __forceinline__ uint16_t f2bf16(float f) {   // round to nearest even (the values here are exact anyway)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) (u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (a multiple of 32); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 2^ceil(log2(x)) for a positive normal x (the reference's fast_round_scale)
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

// a load of memory another device or the host writes, never from a stale cache line (CUDA's __ldcv; HIP: volatile
// 32-bit loads)
template <class T>
__device__ __forceinline__ T ld_fresh(const T* p) {
#if defined(STRATA_USE_HIP)
    T v;
    const volatile uint32_t* s = (const volatile uint32_t*) p;
    uint32_t* d = (uint32_t*) &v;
#pragma unroll
    for (int i = 0; i < (int) (sizeof(T) / 4); ++i) d[i] = s[i];
    return v;
#else
    return __ldcv(p);
#endif
}
// DS4_AR_WIDE=1: ld_fresh as one volatile load of the whole 8 or 16 bytes (gfx11: one global_load_b64 / b128 glc dlc,
// against 2 / 4 dword loads; tools/p2pk_bw on the test machine, section d: pinned host memory read at 26-28 GB/s with 128-bit loads,
// 7.0 GB/s with ld_fresh).  The same freshness as ld_fresh (volatile: past L0 / GL1 every time); like ld_fresh it is not
// one atomic access - the flags order it.  p must be sizeof(T)-aligned
template <class T>
__device__ __forceinline__ T ld_wide(const T* p) {
    static_assert(sizeof(T) == 8 || sizeof(T) == 16, "ld_wide: 8 or 16 bytes");
#if defined(STRATA_USE_HIP)
    T v;
    if constexpr (sizeof(T) == 16) {
        typedef unsigned lw4 __attribute__((ext_vector_type(4)));
        const lw4 w = *(const volatile lw4*) p;
        __builtin_memcpy(&v, &w, 16);
    } else {
        typedef unsigned lw2 __attribute__((ext_vector_type(2)));
        const lw2 w = *(const volatile lw2*) p;
        __builtin_memcpy(&v, &w, 8);
    }
    return v;
#else
    return __ldcv(p);
#endif
}
// DS4_AR_WIDE: the block's threads over the 8-byte indexes [a, b) two at a time - f2(q) for every even q with q and q + 1
// in [a, b) (16 aligned bytes), f1(q) for an odd first and an odd last index alone
template <class F2, class F1>
__device__ __forceinline__ void wide_pairs(int a, int b, F2&& f2, F1&& f1) {
    if (a >= b) return;
    int s = a, e = b;
    if (s & 1) {
        if (threadIdx.x == 0) f1(s);
        ++s;
    }
    if (e > s && (e & 1)) {
        if (threadIdx.x == (blockDim.x > 1 ? 1u : 0u)) f1(e - 1);
        --e;
    }
    for (int q = s + 2 * (int) threadIdx.x; q < e; q += 2 * (int) blockDim.x) f2(q);
}

__device__ __forceinline__ float fp8_rt(float v) { return e4m3_round(v); }   // E4M3, nearest even (input clamped)

__device__ __forceinline__ float e2m1_rt(float v) {   // E2M1 magnitudes 0 .5 1 1.5 2 3 4 6, nearest, ties to even
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

// ---------------------------------------------------------------- the caches' packed rows (kv_row_bytes)
// the E2M1 code of a value e2m1_rt returned (bit 3 the sign, -0 kept; magnitudes 0 .5 1 1.5 2 3 4 6 as 0 .. 7)
__device__ __forceinline__ uint32_t e2m1_code(float q) {
    const float a = fabsf(q);
    const uint32_t m = a <= 1.5f ? (uint32_t) (a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((__float_as_uint(q) >> 31) << 3);
}
// the E4M3 code of v / 2^k for a v whose quotient is an E4M3 value (|v / 2^k| <= 448, at most 4 significant bits, a
// multiple of 2^-9; v 0 or a normal F32): integer arithmetic, exact, -0 kept
__device__ __forceinline__ uint32_t e4m3_code(float v, int k = 0) {
    const uint32_t b = __float_as_uint(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0u) return sg;
    const int e = (int) (a >> 23) - 127 - k;   // the quotient's binade
    if (e >= -6) return sg | ((uint32_t) (e + 7) << 3) | ((a >> 20) & 7u);
    return sg | (((a & 0x7fffffu) | 0x800000u) >> (14 - e));   // a subnormal: m 2^-9
}
// an E2M1 code's value times 2^-14, by the conversion of an FP16 (the code's sign, exponent and mantissa bit are an
// FP16's at bits 15 and 11..9: exact, subnormals too) - the readers' decode: (value 2^-14) (scale 2^14) is value * scale
// exactly, both products exact
__device__ __forceinline__ float e2m1_val14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}
// a UE8M0 scale byte's value (the byte the F32 exponent: 2^(b - 127); 0 -> 0, an unwritten row's)
__device__ __forceinline__ float ue8m0_val(uint32_t b) { return __uint_as_float(b << 23); }

// ---------------------------------------------------------------- embed
__global__ void embed_kernel(const uint16_t* __restrict__ rows, float* __restrict__ h, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    const float v = bf16f(rows[(size_t) t * dim + e]);
    for (int s = 0; s < hc; ++s) h[((size_t) t * hc + s) * dim + e] = v;
}

// ---------------------------------------------------------------- finish
constexpr int kFinMax = 32;   // values a thread holds: len <= 32 * 256

// kPack: the rounded rows packed to f.pdst (f.quant's cache format) instead of written to x or f.dst
// kQ8 (DS4_DECODE_TRIM bit 2): the values written to x also rounded to int8 into f.q8 / f.q8d (finish checks the shape)
template <bool kPack, bool kQ8 = false>
__global__ void __launch_bounds__(kThreads) finish_kernel(float* x, int len, long long stride, Finish f) {
    float* v = x + (long long) blockIdx.x * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = f.pos + (int) (blockIdx.x / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0 && !kPack) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    if (kPack) {   // the same roundings, written as their codes and each group's scale (the readers' q * s: r exactly)
        const int g = f.quant == 2 ? 16 : 32;
        uint8_t* pr = f.pdst + (long long) blockIdx.x * f.pdst_stride;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            float q = 0.0f, s = 1.0f;
            if (e < len) {
                if (f.quant == 1) {
                    s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                    q = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f));
                } else if (f.quant == 2) {
                    s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                    q = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f));
                } else {
                    s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                    q = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f));
                }
                r[j] = q * s;
            }
            if (f.quant == 1) {
                if (e < len) pr[e] = (uint8_t) e4m3_code(q);
            } else {
                const uint32_t c = e2m1_code(q);
                const uint32_t hi = __shfl_down_sync(0xffffffffu, c, 1);   // (value e + 1: the next lane's)
                if (e < len && (e & 1) == 0) pr[e >> 1] = (uint8_t) (c | (hi << 4));
            }
            if (e < len && e % g == 0)
                pr[(f.quant == 1 ? len : len / 2) + e / g] =
                    (uint8_t) (f.quant == 2 ? e4m3_code(s) : __float_as_uint(s) >> 23);
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + (long long) blockIdx.x * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else if (!kPack) v[e] = r[j];
    }
    if (kQ8 && !kPack) {   // (DS4_DECODE_TRIM bit 2, Maya #512: the stored values' int8 rounding, q8_quant's ops)
        int8_t* q = f.q8 + (long long) blockIdx.x * len;   // (a warp's lanes are a 32-block: e = lane + 32 w + 256 j)
        float* qd = f.q8d + (long long) blockIdx.x * (len / 32);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float am = e < len ? fabsf(r[j]) : 0.0f;   // (len % 32 == 0: a warp all in or all past len)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            if (e >= len) continue;
            const float sc = am / 127.0f;
            q[e] = (int8_t) (am > 0.0f ? __float2int_rn(r[j] / sc) : 0);
            if ((threadIdx.x & 31) == 0) qd[e / 32] = sc;
        }
    }
}

// ---------------------------------------------------------------- hyper-connections
__global__ void __launch_bounds__(kThreads) hc_mix_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                          int hcdim, float eps, float* __restrict__ mixes) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
}

// a warp a token: lane j * hc + k holds comb[j][k] (hc <= 4: lanes 0..15; wider streams fall back to lane 0 alone).  The body is
// a device function (hc_mix_kernel's last block runs it too); mx is the token's 32 mixes
__device__ __forceinline__ void hc_split_body(const float* mx, int t, int lane, const float* __restrict__ scale,
                                              const float* __restrict__ base, int hc, int iters, float eps, float* pre,
                                              float* post, float* comb) {
    pre += (size_t) t * 8;
    post += (size_t) t * 8;
    comb += (size_t) t * 64;
    if (lane < hc) {
        pre[lane] = 1.0f / (1.0f + expf(-(mx[lane] * scale[0] + base[lane]))) + eps;
        post[lane] = 2.0f / (1.0f + expf(-(mx[hc + lane] * scale[1] + base[hc + lane])));
    }
    if (hc == 4) {
        // rows are the 4-lane groups (j = lane / 4), columns the lanes 4 apart (k = lane % 4)
        const bool on = lane < 16;
        float c = on ? mx[8 + lane] * scale[2] + base[8 + lane] : -FLT_MAX;
        auto row_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto col_sum = [&](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float m = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        c = on ? expf(c - m) : 0.0f;
        c = c / row_sum(c) + eps;                       // softmax over each row, + eps
        c = on ? c : 0.0f;
        c = c / (col_sum(c) + eps);                     // each destination column over its sources
        for (int it = 1; it < iters; ++it) {
            c = c / (row_sum(c) + eps);
            c = c / (col_sum(c) + eps);
        }
        if (on) comb[lane] = c;
        return;
    }
    if (lane != 0) return;
    float c[8][8];
    for (int j = 0; j < hc; ++j) {
        float m = -FLT_MAX;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = mx[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            m = fmaxf(m, c[j][k]);
        }
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = expf(c[j][k] - m);
            s += c[j][k];
        }
        for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / s + eps;
    }
    for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) s += c[j][k];
        for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
    }
    for (int it = 1; it < iters; ++it) {
        for (int j = 0; j < hc; ++j) {
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += c[j][k];
            for (int k = 0; k < hc; ++k) c[j][k] /= s + eps;
        }
        for (int k = 0; k < hc; ++k) {
            float s = 0.0f;
            for (int j = 0; j < hc; ++j) s += c[j][k];
            for (int j = 0; j < hc; ++j) c[j][k] /= s + eps;
        }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = c[j][k];
}
__global__ void hc_split_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                const float* __restrict__ base, int hc, int iters, float eps, float* pre, float* post,
                                float* comb) {
    const int t = blockIdx.x;
    hc_split_body(mixes + (size_t) t * 32, t, threadIdx.x, scale, base, hc, iters, eps, pre, post, comb);
}

// hc_mix_kernel with hc_split_kernel behind it: the block that finishes the token's last mix (an atomic count, reset by it) runs
// the split on warp 0 - the same arithmetic as the two launches, one launch less and no round trip through the launch gap
__global__ void __launch_bounds__(kThreads) hc_mix_split_kernel(const float* __restrict__ h, const float* __restrict__ fn,
                                                                int hcdim, float eps, float* mixes, unsigned* cnt,
                                                                const float* __restrict__ scale,
                                                                const float* __restrict__ base, int hc, int iters,
                                                                float hc_eps, float* pre, float* post, float* comb) {
    const int m = blockIdx.x, t = blockIdx.y;
    const float* w = fn + (size_t) m * hcdim;
    const float* x = h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        const float4 wv = *(const float4*) (w + k);
        acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    acc = block_sum(acc);
    ss = block_sum(ss);
    __shared__ unsigned last;
    __shared__ float smx[32];
    if (threadIdx.x == 0) {
        mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + eps);
        __threadfence();
        last = atomicAdd(cnt + t, 1u) == gridDim.x - 1 ? 1u : 0u;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < 32) {
        const int n = gridDim.x;   // the mixes of the token (n_mix <= 32); read past the L1
        smx[threadIdx.x] = threadIdx.x < n ? *(const volatile float*) (mixes + (size_t) t * 32 + threadIdx.x) : 0.0f;
        __syncwarp();
        hc_split_body(smx, t, threadIdx.x, scale, base, hc, iters, hc_eps, pre, post, comb);
    }
    if (threadIdx.x == 0) cnt[t] = 0u;
}

__global__ void __launch_bounds__(1024) hc_pre_norm_kernel(const float* __restrict__ h, const float* __restrict__ mix,
                                                           const float* __restrict__ w, int dim, int hc, float eps,
                                                           float* __restrict__ x) {
    const int t = blockIdx.x;
    h += (size_t) t * hc * dim;
    mix += (size_t) t * 8;
    x += (size_t) t * dim;
    float r[8];
    float ss = 0.0f;
    int n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
        float a = 0.0f;
        for (int s = 0; s < hc; ++s) a += mix[s] * h[(size_t) s * dim + e];
        r[n] = a;
        ss += a * a;
    }
    const float inv = rsqrtf(block_sum(ss) / (float) dim + eps);
    n = 0;
    for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) x[e] = w != nullptr ? r[n] * inv * w[e] : r[n];
}

// DS4_HC_FUSE (after Project Maya #511, hc_split_pre_norm_kernel by @pepuscz; MIT): hc_mix_split_kernel and
// hc_pre_norm_kernel as the blocks of one launch of 1024 threads (ds4_kernels.cuh HcFuse).  Blocks x < n_mix: the mix's
// code on threads 0 .. 255 (kThreads, its stride), threads 256 .. 1023 adding +0 to its two block sums - block_sum's
// second level then reads 32 warps' sums where hc_mix_split's 256 threads leave lanes 8 .. 31 at +0: the same values in
// the same order, the same bits; the last block of a token runs the split (+ pre_copy).  Block x == n_mix: the norm,
// hc_pre_norm_kernel's code as it is, then (q set) each warp rounds the 32 values of x it stored as gguf's
// q8_quant_kernel does (warp w holds x[1024 n + 32 w + lane]: whole blocks, every lane of a warp in or past dim together)
__global__ void __launch_bounds__(1024) hc_mix_split_norm_kernel(const HcFuse a) {
    const int t = blockIdx.y;
    if ((int) blockIdx.x == a.n_mix) {   // the norm: hc_pre_norm_kernel's block t
        const int dim = a.dim, hc = a.hc;
        const float* h = a.h + (size_t) t * hc * dim;
        const float* mix = a.mix + (size_t) t * 8;
        float* x = a.x + (size_t) t * dim;
        int8_t* q = a.q != nullptr ? a.q + (size_t) t * dim : nullptr;
        float* qd = a.qd != nullptr ? a.qd + (size_t) t * (dim / 32) : nullptr;
        const float* w = a.w;
        float r[8];
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
            float v = 0.0f;
            for (int s = 0; s < hc; ++s) v += mix[s] * h[(size_t) s * dim + e];
            r[n] = v;
            ss += v * v;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) dim + a.eps);
        n = 0;
        for (int e = threadIdx.x; e < dim; e += blockDim.x, ++n) {
            const float v = w != nullptr ? r[n] * inv * w[e] : r[n];
            x[e] = v;
            if (q == nullptr) continue;
            float am = fabsf(v);   // (q8_quant_kernel's ops on the value stored, from the warp's own registers)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            const float sc = am / 127.0f;
            q[e] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
            if ((threadIdx.x & 31) == 0) qd[e / 32] = sc;
        }
        return;
    }
    // the mix: hc_mix_split_kernel's block (m, t)
    const int m = blockIdx.x, hcdim = a.hc * a.dim;
    const float* w = a.fn + (size_t) m * hcdim;
    const float* x = a.h + (size_t) t * hcdim;
    float acc = 0.0f, ss = 0.0f;
    if (threadIdx.x < kThreads)
        for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
            const float4 hv = *(const float4*) (x + k);
            const float4 wv = *(const float4*) (w + k);
            acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
            ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
        }
    acc = block_sum(acc);
    ss = block_sum(ss);
    __shared__ unsigned last;
    __shared__ float smx[32];
    if (threadIdx.x == 0) {
        a.mixes[(size_t) t * 32 + m] = acc * rsqrtf(ss / (float) hcdim + a.eps);
        __threadfence();
        last = atomicAdd(a.cnt + t, 1u) == (unsigned) a.n_mix - 1u ? 1u : 0u;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < 32) {
        const int n = a.n_mix;   // the mixes of the token (n_mix <= 32); read past the L1
        smx[threadIdx.x] = (int) threadIdx.x < n ? *(const volatile float*) (a.mixes + (size_t) t * 32 + threadIdx.x) : 0.0f;
        __syncwarp();
        hc_split_body(smx, t, threadIdx.x, a.scale, a.base, a.hc, a.iters, a.hc_eps, a.pre, a.post, a.comb);
        // (the layer's copy of its 8 pre slots, each lane its own slot: the k < hc it just wrote, read back as the float
        //  it stored; the rest, which no lane writes here, as pre's own bits)
        const size_t o = (size_t) t * 8 + threadIdx.x;
        if (a.pre_copy != nullptr && threadIdx.x < 8) {
            if ((int) threadIdx.x < a.hc) a.pre_copy[o] = a.pre[o];
            else ((uint32_t*) a.pre_copy)[o] = ((const uint32_t*) a.pre)[o];
        }
    }
    if (threadIdx.x == 0) a.cnt[t] = 0u;
}

__global__ void hc_post_kernel(const float* __restrict__ y, const float* __restrict__ res,
                               const float* __restrict__ post, const float* __restrict__ comb, int dim, int hc,
                               float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= dim * hc) return;
    const int k = i / dim, e = i - k * dim;
    const float* r = res + (size_t) t * hc * dim;
    float a = post[(size_t) t * 8 + k] * y[(size_t) t * dim + e];
    for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
    out[(size_t) t * hc * dim + i] = a;
}

// ---------------------------------------------------------------- gemv over F16 weights, a warp a row
__global__ void __launch_bounds__(kThreads) gemv_f16_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float alpha, float* __restrict__ y) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    const float* xv = x + (size_t) t * k;
    float acc = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            acc += f.x * xv[c + 2 * i] + f.y * xv[c + 2 * i + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * n + row] = alpha * acc;
}
// ... for T activations (rows ldx apart, 16-byte aligned; outputs ldy apart), kRowsT of them a block (grid y): each
// weight row read once for the block's activations, theirs in float4s - each activation's sum gemv_f16_kernel's (the
// same products in the same order), so its bits too.  (A block for all of them, 16 at most, was slower than a launch
// a token: its accumulators and guards kept the loop from unrolling - 66 against 29 us at 5 tokens on an RTX 5060 Ti,
// this 9)
constexpr int kRowsT = 2;
__global__ void __launch_bounds__(kThreads) gemv_f16_rows_kernel(const uint16_t* __restrict__ W,
                                                                 const float* __restrict__ x, size_t ldx, int n, int k,
                                                                 int T, float alpha, float* __restrict__ y, size_t ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.y * kRowsT, nt = min(kRowsT, T - t0);
    if (row >= n) return;
    const uint16_t* w = W + (size_t) row * k;
    float acc[kRowsT];
#pragma unroll
    for (int t = 0; t < kRowsT; ++t) acc[t] = 0.0f;
    for (int c = lane * 8; c < k; c += 32 * 8) {
        const uint4 q = *(const uint4*) (w + c);
        const __half2* hp = (const __half2*) &q;
        float xv[kRowsT][8];
#pragma unroll
        for (int t = 0; t < kRowsT; ++t)
            if (t < nt) {
                const float* xp = x + (size_t) (t0 + t) * ldx + c;
                const float4 a = *(const float4*) xp, b = *(const float4*) (xp + 4);
                xv[t][0] = a.x, xv[t][1] = a.y, xv[t][2] = a.z, xv[t][3] = a.w;
                xv[t][4] = b.x, xv[t][5] = b.y, xv[t][6] = b.z, xv[t][7] = b.w;
            }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
#pragma unroll
            for (int t = 0; t < kRowsT; ++t)
                if (t < nt) acc[t] += f.x * xv[t][2 * i] + f.y * xv[t][2 * i + 1];
        }
    }
#pragma unroll
    for (int t = 0; t < kRowsT; ++t)
        if (t < nt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) y[(size_t) (t0 + t) * ldy + row] = alpha * a;
        }
}

// ---------------------------------------------------------------- gemv over F32 weights, a warp a row (the router)
__global__ void __launch_bounds__(kThreads) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            int n, int k, float* __restrict__ y, int ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const float4* w = (const float4*) (W + (size_t) row * k);
    const float4* xv = (const float4*) (x + (size_t) t * k);
    float acc = 0.0f;
    for (int c = lane; c < k / 4; c += 32) {
        const float4 a = w[c], b = xv[c];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}
// (after Project Maya #511 / #745, MIT): float4 c of a router row, and a row's sum as gemv_f32_kernel makes it - a
// lane's float4s c = lane + 32 j, fmaf in that order, then the warp's sum.  On HIP kGateLd of a lane's float4s are loaded
// before any is summed (the loads in flight together), the sums in the same order: the same bits (Maya #745: the router
// 52 -> 21 us on gfx1151).  gemv_f32_kernel itself stays as it was
__device__ __forceinline__ float4 gate_ld(const float* __restrict__ w, int c) { return ((const float4*) w)[c]; }
// (DS4_ROUTER_BF16, Maya #745) the pack's BF16 router row: float4 c is its 4 values widened, exactly (8 aligned bytes)
__device__ __forceinline__ float4 gate_ld(const uint16_t* __restrict__ w, int c) {
    const uint2 a = ((const uint2*) w)[c];
    return make_float4(bf16f((uint16_t) (a.x & 0xffffu)), bf16f((uint16_t) (a.x >> 16)), bf16f((uint16_t) (a.y & 0xffffu)),
                       bf16f((uint16_t) (a.y >> 16)));
}
#if defined(STRATA_USE_HIP)
constexpr int kGateLd = 4;
#else
constexpr int kGateLd = 1;
#endif
template <typename TW>
__device__ __forceinline__ float gate_row(const TW* __restrict__ w, const float4* __restrict__ xv, int k4, int lane) {
    float acc = 0.0f;
    int c = lane;
    for (; c + 32 * (kGateLd - 1) < k4; c += 32 * kGateLd) {
        float4 a[kGateLd];
#pragma unroll
        for (int u = 0; u < kGateLd; ++u) a[u] = gate_ld(w, c + 32 * u);
#pragma unroll
        for (int u = 0; u < kGateLd; ++u) {
            const float4 b = xv[c + 32 * u];
            acc = fmaf(a[u].x, b.x, acc);
            acc = fmaf(a[u].y, b.y, acc);
            acc = fmaf(a[u].z, b.z, acc);
            acc = fmaf(a[u].w, b.w, acc);
        }
    }
    for (; c < k4; c += 32) {
        const float4 a = gate_ld(w, c), b = xv[c];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
    }
    return warp_sum(acc);
}
// DS4_ROUTE_FUSE (after Project Maya #511's gemv_f32_pair_kernel by @pepuscz): two gemv_f32_kernel launches in one
// grid - its first cdiv(n0, 8) blocks are W0's rows into y0, the rest W1's into y1; grid y the T rows of x (k apart), y0's
// and y1's rows ldy apart, as gemv_f32_kernel's - each row its code, so its bits.  route_fuse_check compares the two
template <typename TW>
__global__ void __launch_bounds__(kThreads) gemv_f32_pair_kernel(const TW* __restrict__ W0, int n0, float* __restrict__ y0,
                                                                 const TW* __restrict__ W1, int n1, float* __restrict__ y1,
                                                                 const float* __restrict__ x, int k, int ldy) {
    const int b0 = (n0 + kThreads / 32 - 1) / (kThreads / 32);
    const bool second = (int) blockIdx.x >= b0;
    const TW* W = second ? W1 : W0;
    const int n = second ? n1 : n0;
    float* y = second ? y1 : y0;
    const int row = ((int) blockIdx.x - (second ? b0 : 0)) * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const float acc = gate_row(W + (size_t) row * k, (const float4*) (x + (size_t) t * k), k / 4, lane);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}
// DS4_ROUTER_BF16 (after Project Maya #745 by mrsirg97-rgb): gemv_f32_kernel on a BF16 router - each row its code on the
// widened values (gate_row), so an F32 copy's bits; router_check (ds4_model.cu) compares them at load
template <typename TW>
__global__ void __launch_bounds__(kThreads) gemv_gate_kernel(const TW* __restrict__ W, const float* __restrict__ x, int n,
                                                             int k, float* __restrict__ y, int ldy) {
    const int row = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= n) return;
    const float acc = gate_row(W + (size_t) row * k, (const float4*) (x + (size_t) t * k), k / 4, lane);
    if (lane == 0) y[(size_t) t * ldy + row] = acc;
}

// ---------------------------------------------------------------- the router
// upstream fix C1 (upstream f6eed469): a pick that never came out - every score left is NaN (the gate's input went
// non-finite), nothing beat -FLT_MAX, and the index is still the 0x7fffffff it started as; the next line then wrote
// bs[0x7fffffff] (shared memory) and the kernels after it read vram[] / sc[] / the helpers' tables at that index.  The
// lowest expert not picked yet instead: a valid index, so nothing reads outside its tables; the weights stay NaN as before.
// Only taken when ix >= n, which a finite score never leaves (a finite score >= -FLT_MAX wins or ties at a lower index),
// so the finite path's picks, weights and stores are the same values; it pays one integer compare a pick in thread 0.
__device__ __forceinline__ int route_fallback_pick(const int* pick, int r, int n) {
    int ix = 0;
    for (bool again = true; again;) {   // (the picks are few and unsorted: start over when one of them is the candidate)
        again = false;
        for (int q = 0; q < r && !again; ++q)
            if (pick[q] == ix) ++ix, again = true;
    }
    return ix < n ? ix : 0;
}
__global__ void __launch_bounds__(512) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    int n, int k, float route_scale, int* idx, float* w,
                                                    const float* __restrict__ bias_vl, const VlBits vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl.w[t >> 5] >> (t & 31)) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            if (ix >= n) ix = route_fallback_pick(pick, r, n);   // (upstream fix C1: a NaN gate input)
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        for (int r = 0; r < k; ++r) {
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = sc[pick[r]] / (s + 1e-20f) * route_scale;
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float lim,
                              float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}
// DS4_DECODE_TRIM bit 1 (after Project Maya #512's swiglu_q8_kernel by @pepuscz): swiglu_kernel's value, then
// q8_quant's ops on it - a warp's lanes are a 32-block of the flat array (blocks of kThreads from 0, n % 32 == 0: a warp
// all in or all past n).  The old kernel stays as it is: trim_q8_check (ds4_model.cu) compares the two
__global__ void swiglu_q8_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float lim,
                                 float* __restrict__ h, int8_t* __restrict__ q, float* __restrict__ qd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (lim > 0.0f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    const float v = gv / (1.0f + expf(-gv)) * uv;
    h[i] = v;
    float am = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[i] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if ((threadIdx.x & 31) == 0) qd[i / 32] = sc;
}
// DS4_DECODE_TRIM bit 4 (after Project Maya #512's copy_regions_kernel): region blockIdx.y's 4-byte words, the grid's x
// blocks striding over them
__global__ void copy_regions_kernel(const CopyRegions r) {
    const int k = blockIdx.y;
    uint32_t* d = (uint32_t*) r.dst[k];
    const uint32_t* sr = (const uint32_t*) r.src[k];
    const size_t n = r.bytes[k] / 4;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) d[i] = sr[i];
}

// ---------------------------------------------------------------- the compressor (ratio > 1), a window in order
__global__ void __launch_bounds__(512) compress_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                                       float* kv_state, float* score_state, int ratio, int p0, int T,
                                                       int d, const float* __restrict__ w, float eps,
                                                       float* __restrict__ latent) {
    int g = 0;
    for (int t = 0; t < T; ++t) {
        const int slot = (p0 + t) % ratio;
        const bool pool = (p0 + t + 1) % ratio == 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x) {
            kv_state[(size_t) slot * d + e] = kv[(size_t) t * d + e];
            score_state[(size_t) slot * d + e] = score[(size_t) t * d + e];
        }
        __syncthreads();
        if (!pool) continue;
        // (kv * softmax(score, over the slots)).sum: each slot's probability first, then the sum, as torch does
        float val[2] = {0.0f, 0.0f};   // d <= 2 * blockDim
        float ss = 0.0f;
        int n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) {
            float m = -FLT_MAX;
            for (int s = 0; s < ratio; ++s) m = fmaxf(m, score_state[(size_t) s * d + e]);
            float den = 0.0f;
            for (int s = 0; s < ratio; ++s) den += expf(score_state[(size_t) s * d + e] - m);
            float acc = 0.0f;
            for (int s = 0; s < ratio; ++s)
                acc += kv_state[(size_t) s * d + e] * (expf(score_state[(size_t) s * d + e] - m) / den);
            val[n] = acc;
            ss += acc * acc;
        }
        const float inv = rsqrtf(block_sum(ss) / (float) d + eps);
        n = 0;
        for (int e = threadIdx.x; e < d; e += blockDim.x, ++n) latent[(size_t) g * d + e] = val[n] * inv * w[e];
        ++g;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- per-query sizes
__device__ __forceinline__ void rows_at(const Rows& r, int t, int& n, int& k) {
    if (r.gen == 0) {
        n = r.n[t];
        k = r.k[t];
        return;
    }
    const int nn = (int) ((r.p0 + t + 1) / r.R);
    const int cb = min(r.ccap, (nn + r.blk - 1) / r.blk);
    switch (r.gen) {
    case 1: n = nn, k = min(r.kcap, nn); break;
    case 2: n = nn, k = cb; break;
    case 3: n = cb * r.blk, k = min(r.kcap, nn); break;
    default: n = (nn + r.blk - 1) / r.blk, k = r.kcap; break;
    }
}

// ---------------------------------------------------------------- the indexer
// a thread a position: its key row (KF: kKvFp4p packed, or BF16) against every head's query (shared memory), relu,
// weighted sum
template <int HD, int KF>
__global__ void __launch_bounds__(kThreads) index_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                               const uint8_t* __restrict__ kc, int heads, int n,
                                                               const int* __restrict__ cand, int cand_stride, int blk,
                                                               Rows lim, float* __restrict__ out,
                                                               long long out_stride) {
    extern __shared__ float qs[];   // heads * HD, then heads weights
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    int p = i;
    if (cand != nullptr) {
        if (i >= lk * blk) {
            o[i] = -INFINITY;
            return;
        }
        p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    if (p >= ln) {
        o[i] = -INFINITY;
        return;
    }
    const uint8_t* kr = kc + (size_t) p * kv_row_bytes(KF, HD);
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.0f;
#pragma unroll 2
    for (int c = 0; c < HD; c += 8) {
        float kv[8];
        if (KF == kKvFp4p) {   // 8 codes in 4 bytes, the group's power-of-two scale: the values the finish rounded to
            const uint32_t cw = *(const uint32_t*) (kr + c / 2);
            const float ks = ue8m0_val(kr[HD / 2 + c / 32]);
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;   // (any 2^k: exact)
        } else {
            const uint4 raw = *(const uint4*) (kr + c * 2);
            const uint16_t* b = (const uint16_t*) &raw;
#pragma unroll
            for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
        }
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h >= heads) break;
            const float* qh = qs + h * HD + c;
#pragma unroll
            for (int x = 0; x < 8; ++x) acc[h] += qh[x] * kv[x];
        }
    }
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 32; ++h) {
        if (h >= heads) break;
        s += fmaxf(acc[h], 0.0f) * qs[heads * HD + h];
    }
    o[i] = s;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// a block a row: the k-th largest key by four 8-bit radix passes, then everything above it and enough of its ties
__global__ void __launch_bounds__(1024) topk_kernel(const float* __restrict__ vals, long long stride, Rows r,
                                                    const int* __restrict__ cand, int cand_stride, int blk,
                                                    int* __restrict__ out, int out_stride) {
    __shared__ unsigned hist[256];
    __shared__ uint32_t prefix_s, mask_s;
    __shared__ int remaining_s;
    __shared__ int cnt_gt, cnt_eq;
    const int t = blockIdx.x;
    int n, k;
    rows_at(r, t, n, k);
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (n <= 0) return;
    if (k >= n) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    if (threadIdx.x == 0) {
        prefix_s = 0u;
        mask_s = 0u;
        remaining_s = k;
        cnt_gt = 0;
        cnt_eq = 0;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        const uint32_t prefix = prefix_s, mask = mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t u = ord_key(vals[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int rem = remaining_s;
            int cum = 0;
            for (int d = 255; d >= 0; --d) {
                const int h = (int) hist[d];
                if (cum + h >= rem) {
                    prefix_s = prefix | ((uint32_t) d << shift);
                    mask_s = mask | (0xffu << shift);
                    remaining_s = rem - cum;
                    break;
                }
                cum += h;
            }
        }
        __syncthreads();
    }
    const uint32_t thr = prefix_s;
    const int remaining = remaining_s;
    const int n_gt = k - remaining;
    // the picks in index order (a scan a pass of 1024 values), so the attention sums them in the same order every run
    // and the ties kept are the lowest positions
    __shared__ int wgt[32], weq[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const uint32_t u = i < n ? ord_key(vals[i]) : 0u;
        const bool gt = i < n && u > thr, eq = i < n && u == thr;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wgt[wid] = __popc(bg);
            weq[wid] = __popc(be);
        }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                const int a = __shfl_up_sync(0xffffffffu, gs, o), b = __shfl_up_sync(0xffffffffu, es, o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            __syncwarp();
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        __syncthreads();
    }
}

// ---------------------------------------------------------------- (DS4_TOPK_MB) the multi-block top-k
// topk_kernel walks a row with one workgroup (one WGP of 48): four 8-bit radix passes over n, then an ordered scan, so at
// 110K tokens (n ~ 28K scores) it is ~n / 1024 serial steps a pass.  Here a row is split into G <= kTopkMbMaxG chunks, one
// workgroup each, over five launches: h1 / h2 / h3 the k-th largest key by three histogram levels (key bits 31..20,
// 19..8 of the keys in level 1's bin, 7..0 of the keys in level 2's bin), c each chunk's count of keys above it / equal
// to it, write the picks.  The selection rule is topk_kernel's: thr = the k-th largest ord_key (any radix split finds
// the same one), remaining = k - #(keys > thr); out[0, n_gt) = the keys above thr in index order, out[n_gt + e] = the
// e-th key equal to thr in index order for e < remaining (ties: the lowest positions) - so out[t][0, k) is the same
// bits.  Integer adds only, so every count is exact in any order.
// Row scratch (kTopkMbRow unsigned): [0, 4096) level 1, [4096, 8192) level 2, [8192, 8448) level 3, [8448, 8576) each
// workgroup's {above, equal}, [8576, 8582) the state (b1, rem1, b2, rem2, thr, remaining).  The histograms are zero on
// entry (dalloc zeroes) and topk_mb_write zeroes them after the row's last reader (topk_mb_c), stream-ordered
constexpr int kTmbT = 256, kTmbH2 = 4096, kTmbH3 = 8192, kTmbGc = 8448, kTmbSt = 8576, kTmbZero = 8448;
static_assert(kTmbSt + 8 <= kTopkMbRow && kTmbGc + 2 * kTopkMbMaxG <= kTmbSt, "DS4_TOPK_MB row scratch");

// the bin of h[0, nb) holding the rem-th largest key and how many of that bin's keys are still wanted: the first d from
// the top with h[d] + the bins above >= rem - topk_kernel's thread-0 loop, split over the threads (nb / kTmbT bins each,
// top first) and a scan of their sums.  Exactly one thread has excl < rem <= incl; every thread returns its answer
__device__ void tmb_pick(const unsigned* __restrict__ h, int nb, int rem, int& bin, int& rem_out) {
    __shared__ int sum_s[kTmbT];
    __shared__ int pick_s[2];
    const int q = nb / kTmbT, tid = threadIdx.x, top = nb - 1 - tid * q;
    if (tid == 0) {
        pick_s[0] = 0;
        pick_s[1] = rem;
    }
    int s = 0;
    for (int j = 0; j < q; ++j) s += (int) h[top - j];
    sum_s[tid] = s;
    __syncthreads();
    for (int o = 1; o < kTmbT; o <<= 1) {   // inclusive scan, the top bins' thread first
        const int a = tid >= o ? sum_s[tid - o] : 0;
        __syncthreads();
        sum_s[tid] += a;
        __syncthreads();
    }
    const int incl = sum_s[tid], excl = incl - s;
    if (excl < rem && incl >= rem) {
        int cum = excl;
        for (int j = 0; j < q; ++j) {
            const int hv = (int) h[top - j];
            if (cum + hv >= rem) {
                pick_s[0] = top - j;
                pick_s[1] = rem - cum;
                break;
            }
            cum += hv;
        }
    }
    __syncthreads();
    bin = pick_s[0];
    rem_out = pick_s[1];
}

// the chunk of workgroup blockIdx.x: [i0, i1); false: nothing to do here (an empty row, k >= n, or past the row's end)
__device__ __forceinline__ bool tmb_chunk(const Rows& r, int chunk, int& n, int& k, int& i0, int& i1) {
    rows_at(r, blockIdx.y, n, k);
    i0 = blockIdx.x * chunk;
    i1 = min(n, i0 + chunk);
    return n > 0 && k > 0 && k < n && i0 < n;   // (k <= 0: topk_kernel writes nothing either)
}

__global__ void __launch_bounds__(kTmbT) topk_mb_h1(const float* __restrict__ vals, long long stride, Rows r, int chunk,
                                                   unsigned* __restrict__ mb) {
    __shared__ unsigned h[4096];
    int n, k, i0, i1;
    if (!tmb_chunk(r, chunk, n, k, i0, i1)) return;
    vals += (long long) blockIdx.y * stride;
    unsigned* g = mb + (size_t) blockIdx.y * kTopkMbRow;
    for (int i = threadIdx.x; i < 4096; i += kTmbT) h[i] = 0u;
    __syncthreads();
    for (int i = i0 + threadIdx.x; i < i1; i += kTmbT) atomicAdd(&h[ord_key(vals[i]) >> 20], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += kTmbT)
        if (h[i] != 0u) atomicAdd(&g[i], h[i]);
}

__global__ void __launch_bounds__(kTmbT) topk_mb_h2(const float* __restrict__ vals, long long stride, Rows r, int chunk,
                                                   unsigned* __restrict__ mb) {
    __shared__ unsigned h[4096];
    int n, k, i0, i1;
    if (!tmb_chunk(r, chunk, n, k, i0, i1)) return;
    vals += (long long) blockIdx.y * stride;
    unsigned* g = mb + (size_t) blockIdx.y * kTopkMbRow;
    int b1, rem1;
    tmb_pick(g, 4096, k, b1, rem1);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g[kTmbSt + 0] = (unsigned) b1;
        g[kTmbSt + 1] = (unsigned) rem1;
    }
    for (int i = threadIdx.x; i < 4096; i += kTmbT) h[i] = 0u;
    __syncthreads();
    for (int i = i0 + threadIdx.x; i < i1; i += kTmbT) {
        const uint32_t u = ord_key(vals[i]);
        if ((u >> 20) == (uint32_t) b1) atomicAdd(&h[(u >> 8) & 0xfffu], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 4096; i += kTmbT)
        if (h[i] != 0u) atomicAdd(&g[kTmbH2 + i], h[i]);
}

__global__ void __launch_bounds__(kTmbT) topk_mb_h3(const float* __restrict__ vals, long long stride, Rows r, int chunk,
                                                   unsigned* __restrict__ mb) {
    __shared__ unsigned h[256];
    int n, k, i0, i1;
    if (!tmb_chunk(r, chunk, n, k, i0, i1)) return;
    vals += (long long) blockIdx.y * stride;
    unsigned* g = mb + (size_t) blockIdx.y * kTopkMbRow;
    const uint32_t b1 = g[kTmbSt + 0];
    int b2, rem2;
    tmb_pick(g + kTmbH2, 4096, (int) g[kTmbSt + 1], b2, rem2);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g[kTmbSt + 2] = (unsigned) b2;
        g[kTmbSt + 3] = (unsigned) rem2;
    }
    const uint32_t p24 = (b1 << 12) | (uint32_t) b2;
    for (int i = threadIdx.x; i < 256; i += kTmbT) h[i] = 0u;
    __syncthreads();
    for (int i = i0 + threadIdx.x; i < i1; i += kTmbT) {
        const uint32_t u = ord_key(vals[i]);
        if ((u >> 8) == p24) atomicAdd(&h[u & 0xffu], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += kTmbT)
        if (h[i] != 0u) atomicAdd(&g[kTmbH3 + i], h[i]);
}

__global__ void __launch_bounds__(kTmbT) topk_mb_c(const float* __restrict__ vals, long long stride, Rows r, int chunk,
                                                  unsigned* __restrict__ mb) {
    __shared__ int c_gt, c_eq;
    int n, k, i0, i1;
    if (!tmb_chunk(r, chunk, n, k, i0, i1)) return;
    vals += (long long) blockIdx.y * stride;
    unsigned* g = mb + (size_t) blockIdx.y * kTopkMbRow;
    int b3, rem3;
    tmb_pick(g + kTmbH3, 256, (int) g[kTmbSt + 3], b3, rem3);
    const uint32_t thr = (g[kTmbSt + 0] << 20) | (g[kTmbSt + 2] << 8) | (uint32_t) b3;
    if (threadIdx.x == 0) {
        c_gt = 0;
        c_eq = 0;
        if (blockIdx.x == 0) {
            g[kTmbSt + 4] = thr;
            g[kTmbSt + 5] = (unsigned) rem3;
        }
    }
    __syncthreads();
    int a = 0, e = 0;
    for (int i = i0 + threadIdx.x; i < i1; i += kTmbT) {
        const uint32_t u = ord_key(vals[i]);
        a += u > thr;
        e += u == thr;
    }
    if (a != 0) atomicAdd(&c_gt, a);
    if (e != 0) atomicAdd(&c_eq, e);
    __syncthreads();
    if (threadIdx.x == 0) {
        g[kTmbGc + 2 * blockIdx.x] = (unsigned) c_gt;
        g[kTmbGc + 2 * blockIdx.x + 1] = (unsigned) c_eq;
    }
}

// topk_kernel's ordered scan over this workgroup's chunk, its counts starting at the chunks before it (so a pick's slot
// is the one topk_kernel gives it); then the row's histograms zeroed for the next call
__global__ void __launch_bounds__(kTmbT) topk_mb_write(const float* __restrict__ vals, long long stride, Rows r,
                                                      const int* __restrict__ cand, int cand_stride, int blk,
                                                      int* __restrict__ out, int out_stride, int chunk,
                                                      unsigned* __restrict__ mb) {
    __shared__ int cnt_gt, cnt_eq;
    __shared__ int wgt[32], weq[32];
    const int t = blockIdx.y;
    int n, k;
    rows_at(r, t, n, k);
    const int i0 = blockIdx.x * chunk, i1 = min(n, i0 + chunk);
    if (n <= 0 || k <= 0 || i0 >= n) return;
    vals += (long long) t * stride;
    out += (size_t) t * out_stride;
    const int* cd = cand != nullptr ? cand + (size_t) t * cand_stride : nullptr;
    if (k >= n) {   // (topk_kernel's take-all; no histogram was touched)
        for (int i = i0 + threadIdx.x; i < i1; i += kTmbT) out[i] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        return;
    }
    unsigned* g = mb + (size_t) t * kTopkMbRow;
    const uint32_t thr = g[kTmbSt + 4];
    const int remaining = (int) g[kTmbSt + 5], n_gt = k - remaining;
    if (threadIdx.x == 0) {
        int a = 0, e = 0;
        for (int x = 0; x < (int) blockIdx.x; ++x) {
            a += (int) g[kTmbGc + 2 * x];
            e += (int) g[kTmbGc + 2 * x + 1];
        }
        cnt_gt = a;
        cnt_eq = e;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = kTmbT >> 5;
    const unsigned below = (1u << lane) - 1u;
    for (int b0 = i0; b0 < i1; b0 += kTmbT) {
        const int i = b0 + threadIdx.x;
        const uint32_t u = i < i1 ? ord_key(vals[i]) : 0u;
        const bool gt = i < i1 && u > thr, eq = i < i1 && u == thr;
        const unsigned bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wgt[wid] = __popc(bg);
            weq[wid] = __popc(be);
        }
        __syncthreads();
        if (wid == 0) {
            const int g0 = lane < nw ? wgt[lane] : 0, e0 = lane < nw ? weq[lane] : 0;
            int gs = g0, es = e0;
            for (int o = 1; o < 32; o <<= 1) {
                const int a = __shfl_up_sync(0xffffffffu, gs, o), b = __shfl_up_sync(0xffffffffu, es, o);
                if (lane >= o) {
                    gs += a;
                    es += b;
                }
            }
            if (lane < nw) {
                wgt[lane] = cnt_gt + gs - g0;
                weq[lane] = cnt_eq + es - e0;
            }
            __syncwarp();
            if (lane == 31) {
                cnt_gt += gs;
                cnt_eq += es;
            }
        }
        __syncthreads();
        int slot = -1;
        if (gt) slot = wgt[wid] + __popc(bg & below);
        else if (eq) {
            const int e = weq[wid] + __popc(be & below);
            if (e < remaining) slot = n_gt + e;
        }
        if (slot >= 0) out[slot] = cd != nullptr ? cd[i / blk] * blk + i % blk : i;
        __syncthreads();
    }
    const int na = (n + chunk - 1) / chunk;   // (the row's workgroups past h1..c)
    for (int j = blockIdx.x * kTmbT + threadIdx.x; j < kTmbZero; j += na * kTmbT) g[j] = 0u;
}

__global__ void block_max_kernel(const float* __restrict__ score, long long stride, Rows r, int blk,
                                 float* __restrict__ bm, long long bm_stride) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    int n, k_;
    rows_at(r, t, n, k_);
    const int nb = (n + blk - 1) / blk;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int j = 0; j < blk; ++j) {
        const int p = b * blk + j;
        if (p < n) m = fmaxf(m, score[(long long) t * stride + p]);
    }
    if (b == (n - 1) / blk) m = INFINITY;   // the newest position's block is pinned in
    bm[(long long) t * bm_stride + b] = m;
}

// ---------------------------------------------------------------- sparse attention, a block a (head, query)
constexpr int kAttTile = 64;

// kContig: query t's selected rows are ckv's rows t * sel_stride .. (the scratch of rows decoded once a query) - read
// without sel, a warp's rows of a tile loaded kPre at a time before their dots (0: as the BF16 kernel); the
// arithmetic the same
template <bool kContig, int kPre = 0>
__global__ void __launch_bounds__(kThreads) sparse_attn_kernel(const float* __restrict__ q,
                                                               const uint16_t* __restrict__ win,
                                                               const uint16_t* __restrict__ ckv,
                                                               const int* __restrict__ sel, int sel_stride,
                                                               AttnRows ar, const float* __restrict__ sink, int heads,
                                                               int d, float scale, float* __restrict__ out) {
    // d == 512: a thread owns 2 output values, a lane 16 of a row's dot
    const int h = blockIdx.x, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);   // the window buffer row of max(0, pa - win + 1)
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    __shared__ float qs[512];
    __shared__ float sc[kAttTile];
    __shared__ const uint16_t* rows[kAttTile];
    __shared__ float m_run_s, tile_scale_s, den_s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) qs[i] = q[((size_t) t * heads + h) * d + i];
    if (threadIdx.x == 0) {
        m_run_s = -1e30f;
        den_s = 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float o0 = 0.0f, o1 = 0.0f;
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? wbase + (size_t) j * d
                                          : ckv + (kContig ? (size_t) t * sel_stride + (j - n_win) : (size_t) sl[j - n_win]) * d;
        }
        __syncthreads();
        if (kContig && kPre > 0) {   // the warp's rows of the tile (wp, wp + 8, ..), kPre loads at a time, then dots
#pragma unroll
            for (int k0 = 0; k0 < kAttTile / 8; k0 += (kPre > 0 ? kPre : 1)) {
                uint4 ra[kPre > 0 ? kPre : 1], rb[kPre > 0 ? kPre : 1];
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        ra[k] = *(const uint4*) (rows[jj] + lane * 16);
                        rb[k] = *(const uint4*) (rows[jj] + lane * 16 + 8);
                    }
                }
#pragma unroll
                for (int k = 0; k < kPre; ++k) {
                    const int jj = wp + 8 * (k0 + k);
                    if (jj < nt) {
                        const uint16_t* pa = (const uint16_t*) &ra[k];
                        const uint16_t* pb = (const uint16_t*) &rb[k];
                        float s = 0.0f;
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                        for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                        s = warp_sum(s);
                        if (lane == 0) sc[jj] = s * scale;
                    }
                }
            }
        } else
            for (int jj = wp; jj < nt; jj += kThreads / 32) {
                const uint16_t* rr = rows[jj];
                const uint4 a = *(const uint4*) (rr + lane * 16);
                const uint4 b = *(const uint4*) (rr + lane * 16 + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qs[lane * 16 + 8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[jj] = s * scale;
            }
        __syncthreads();
        if (wp == 0) {
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, sc[j]);
            m = warp_max(m);
            const float m_old = m_run_s;
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(sc[j] - m_new);
                sc[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s = expf(m_old - m_new);
                den_s = den_s * tile_scale_s + s;
                m_run_s = m_new;
            }
        }
        __syncthreads();
        const float cs = tile_scale_s;
        o0 *= cs;
        o1 *= cs;
        const int c = threadIdx.x * 2;
        if (kContig && kPre > 0) {   // (eight rows' loads at once)
#pragma unroll 8
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        } else
            for (int j = 0; j < nt; ++j) {
                const uint32_t pr = *(const uint32_t*) (rows[j] + c);
                o0 += sc[j] * bf16f((uint16_t) (pr & 0xffffu));
                o1 += sc[j] * bf16f((uint16_t) (pr >> 16));
            }
        __syncthreads();
    }
    const float den = den_s + expf(sink[h] - m_run_s);
    const int c = threadIdx.x * 2;
    float* ov = out + ((size_t) t * heads + h) * d;
    ov[c] = o0 / den;
    ov[c + 1] = o1 / den;
}

// ---------------------------------------------------------------- sparse attention over the packed compressed rows
// G heads of one query a block (8 / G warps a head): each pass stages 16 of a tile's rows in shared memory as BF16
// (the window's rows as they are; the packed rows decoded - exact in BF16), once for all G heads, through one 16-byte
// load a thread.  Each head's arithmetic is sparse_attn_kernel's, operation for operation: its tiles of 64 rows, its
// dots by a warp's lanes (16 values each, in order) and the same butterfly, its softmax by one warp the same way, each
// output value's weighted sum by the same thread in row order - the same bits for any G.
constexpr int kStage = 16;   // rows a pass stages (a thread: 32 values of one)

// values part * 32 .. + 32 of a packed compressed row (kKvFp4, 512 values) as BF16 into dst[0 .. 4) (exact: the
// values are E2M1 times an E4M3 scale)
__device__ __forceinline__ void unpack32(const uint8_t* rr, int part, uint4* dst) {
    const uint4 cw = *(const uint4*) (rr + part * 16);   // 32 codes, two scale groups
    const float k0 = e4m3_value(rr[256 + part * 2]) * 16384.0f, k1 = e4m3_value(rr[256 + part * 2 + 1]) * 16384.0f;
    const uint32_t wv[4] = {cw.x, cw.y, cw.z, cw.w};
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {   // values 2i, 2i + 1 (the low and the high nibble of byte i)
        const uint32_t b = wv[i >> 2] >> (8 * (i & 3));
        const float ks = i < 8 ? k0 : k1;
        const float lo = e2m1_val14(b) * ks, hi = e2m1_val14(b >> 4) * ks;   // (exact: their BF16 bits)
        o[i] = (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
    }
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    dst[2] = make_uint4(o[8], o[9], o[10], o[11]);
    dst[3] = make_uint4(o[12], o[13], o[14], o[15]);
}

// a few queries' attention (a decode step, a short window): the rows each query selects decoded once to BF16 rows -
// out[t * cap + i] = ckv[sel[t][i]] for i < the query's n_sel - for the BF16 kernel reading them in place (once a
// query, not once a head)
__global__ void __launch_bounds__(kThreads) unpack_sel_kernel(const uint8_t* __restrict__ ckv,
                                                              const int* __restrict__ sel, int sel_stride, AttnRows ar,
                                                              int cap, uint16_t* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * kStage + (int) (threadIdx.x >> 4), part = threadIdx.x & 15;
    int n_sel;
    if (ar.gen) {
        const long long pa = ar.p0 + ar.t0 + t;
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_sel = ar.nsel[t];
    }
    if (i >= n_sel) return;
    const size_t r = (size_t) t * cap + i;
    unpack32(ckv + (size_t) sel[(size_t) t * sel_stride + i] * kv_row_bytes(kKvFp4, 512), part, (uint4*) (out + r * 512 + part * 32));
}

template <int G>
__global__ void __launch_bounds__(kThreads) sparse_attn_packed_kernel(const float* __restrict__ q,
                                                                      const uint16_t* __restrict__ win,
                                                                      const uint8_t* __restrict__ ckv,
                                                                      const int* __restrict__ sel, int sel_stride,
                                                                      AttnRows ar, const float* __restrict__ sink,
                                                                      int heads, float scale, float* __restrict__ out) {
    constexpr int d = 512, W = 8 / G;   // (kThreads / 32 warps)
    const int hb = blockIdx.x * G, t = blockIdx.y;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * d;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, d);
    __shared__ __align__(16) uint16_t tile[kStage * d];
    __shared__ float sc[G][kAttTile];
    __shared__ const uint8_t* rows[kAttTile];
    __shared__ float m_run_s[G], tile_scale_s[G], den_s[G];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, gw = wp / W, wr = wp % W;
    float qr[16];   // this lane's 16 values of its warp's head
#pragma unroll
    for (int x = 0; x < 16; ++x) qr[x] = q[((size_t) t * heads + hb + gw) * d + lane * 16 + x];
    if (threadIdx.x < G) {
        m_run_s[threadIdx.x] = -1e30f;
        den_s[threadIdx.x] = 0.0f;
    }
    float o0[G], o1[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o0[g] = o1[g] = 0.0f;
    const int c = threadIdx.x * 2;
    const int sr = threadIdx.x >> 4, part = threadIdx.x & 15;   // the staging: a thread's row and 32 values
    // rows [s0, s0 + n) of the tile at t0 into tile[] as BF16
    auto stage = [&](int t0, int s0, int n) {
        if (sr >= n) return;
        const int j = s0 + sr;
        uint4* dst = (uint4*) (tile + sr * d + part * 32);
        if (t0 + j < n_win) {   // a window row: BF16 already
            const uint4* src = (const uint4*) (rows[j] + part * 64);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        }
        unpack32(rows[j], part, dst);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kAttTile) {
        const int nt = min(kAttTile, total - t0);
        if (threadIdx.x < nt) {
            const int j = t0 + threadIdx.x;
            rows[threadIdx.x] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * d) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the dots
            const int n = min(kStage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
#if !defined(STRATA_USE_HIP)   // (HIP leaves a loop with shuffles and a remainder rolled anyway)
#pragma unroll 2
#endif
            for (int r = wr; r < n; r += W) {   // (two rows' chains at once)
                const uint16_t* vr = tile + r * d + lane * 16;
                const uint4 a = *(const uint4*) vr;
                const uint4 b = *(const uint4*) (vr + 8);
                const uint16_t* pa = (const uint16_t*) &a;
                const uint16_t* pb = (const uint16_t*) &b;
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[x] * bf16f(pa[x]);
#pragma unroll
                for (int x = 0; x < 8; ++x) s += qr[8 + x] * bf16f(pb[x]);
                s = warp_sum(s);
                if (lane == 0) sc[gw][s0 + r] = s * scale;
            }
            __syncthreads();
        }
        if (wp < G) {   // head wp's softmax over the tile
            float* scg = sc[wp];
            float m = -INFINITY;
            for (int j = lane; j < nt; j += 32) m = fmaxf(m, scg[j]);
            m = warp_max(m);
            const float m_old = m_run_s[wp];
            const float m_new = fmaxf(m_old, m);
            float s = 0.0f;
            for (int j = lane; j < nt; j += 32) {
                const float p = expf(scg[j] - m_new);
                scg[j] = p;
                s += p;
            }
            s = warp_sum(s);
            __syncwarp();
            if (lane == 0) {
                tile_scale_s[wp] = expf(m_old - m_new);
                den_s[wp] = den_s[wp] * tile_scale_s[wp] + s;
                m_run_s[wp] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float cs = tile_scale_s[g];
            o0[g] *= cs;
            o1[g] *= cs;
        }
        for (int s0 = 0; s0 < nt; s0 += kStage) {   // the weighted sums
            const int n = min(kStage, nt - s0);
            stage(t0, s0, n);
            __syncthreads();
            for (int r = 0; r < n; ++r) {
                const uint32_t pr = *(const uint32_t*) (tile + r * d + c);
                const float v0 = bf16f((uint16_t) (pr & 0xffffu)), v1 = bf16f((uint16_t) (pr >> 16));
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    o0[g] += sc[g][s0 + r] * v0;
                    o1[g] += sc[g][s0 + r] * v1;
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float den = den_s[g] + expf(sink[hb + g] - m_run_s[g]);
        float* ov = out + ((size_t) t * heads + hb + g) * d;
        ov[c] = o0[g] / den;
        ov[c + 1] = o1[g] / den;
    }
}

// DS4_FAST_ATTN=1: the chunked prompt's packed-KV attention on the matrix cores (gfx11/gfx12 WMMA; the R9700's gfx12).
// The matrix's rows are 16 heads of ONE query, so a block stages a 16-key tile once for all of them: Q (16 heads x 512
// dims) is the A operand of the scores' product, the key tile is B, and the same tile is B of the values' product (the
// heads' softmaxes and their 512-dim outputs are its rows).  The QK and PV sums are the matrix cores' order, not
// sparse_attn_packed_kernel's scalar lanes' - hence opt-in (DS4_FAST_ATTN).
// (HIP only: the CUDA build keeps the scalar kernels, so this and the dispatch below compile out there.)
//
// precision variants (docs/dsv41/cards/PF-02.md; DS4_FAST_ATTN_MODE picks one, the default "f16" is the original arithmetic).
// What the original rounds: the K/V rows are BF16 on the FP8 (window) / FP4 x E4M3 (compressed) grid, so their FP16
// copies are exact (in FP16's range); the products accumulate in F32 (wmma_f32_16x16x16_f16), the scale and the online
// softmax are F32.  The two roundings the scalar kernel does not make are Q (F32 -> FP16, 11 bits) and P (the F32
// softmax weights -> FP16 before P.V, while the denominator sums the unrounded weights).
//   QM: 0 Q as FP16; 1 Q and K as BF16 (the BF16 WMMA); 2 Q as an FP16 pair hi + lo (two products, ~22 bits); 3 the
//       scores as scalar F32 dots (no matrix cores for QK).
//   PM: 0 P as FP16; 1 P as an FP16 pair hi + lo (two products); 2 P.V as scalar F32 sums (no matrix cores for PV).
//   QSC: Q scaled per head by a power of two (its max under 2^15) before the cast, the score divided back (exact).
#if defined(STRATA_USE_HIP)
constexpr int kFaHeads = 16;   // heads a tile (the matrix's rows)
constexpr int kFaKeys = 16;    // keys a tile
constexpr int kFaD = 512;      // the head dim
constexpr int kFaLd = 520;     // halves a tile row (512 + 8: the pad keeps the 16 rows off one bank)
template <int QM, int PM, bool QSC>
__global__ void __launch_bounds__(kThreads) sparse_attn_wmma_kernel(const float* __restrict__ q,
                                                                    const uint16_t* __restrict__ win,
                                                                    const uint8_t* __restrict__ ckv,
                                                                    const int* __restrict__ sel, int sel_stride,
                                                                    AttnRows ar, const float* __restrict__ sink,
                                                                    int heads, float scale, float* __restrict__ out) {
    constexpr bool kQf32 = QM == 3;
    constexpr int kQh = QM == 2 ? 2 : 1;   // FP16 planes of Q (hi, lo)
    constexpr int kPh = PM == 1 ? 2 : 1;   // FP16 planes of P
    __shared__ __align__(16) uint16_t qs[kQf32 ? 8 : kQh * kFaHeads * kFaLd];
    __shared__ __align__(16) float qf[kQf32 ? kFaHeads * kFaD : 4];
    __shared__ __align__(16) uint16_t ks[kFaKeys * kFaLd];   // FP16 rows (QM 1: BF16 rows)
    __shared__ float sc[kFaHeads][kFaKeys];
    __shared__ float part[kQf32 ? 1 : 8][kFaHeads][kFaKeys];   // (the 8 warps' 64-dim slices of the scores)
    __shared__ uint16_t ps[kPh][kFaHeads][kFaKeys];
    __shared__ const uint8_t* rows[kFaKeys];
    __shared__ float m_run[kFaHeads], den[kFaHeads], ts[kFaHeads], qmul[kFaHeads];
    const int tid = (int) threadIdx.x, wp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int t = blockIdx.y, hb = blockIdx.x * kFaHeads;
    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t], n_sel = ar.nsel[t], w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * kFaD;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t crow = kv_row_bytes(kKvFp4, kFaD);
    if constexpr (QSC) {   // a head's max |q| (16 threads a head) -> a power of two that takes it under 2^15
        const int h = tid >> 4, p = tid & 15;
        const float* qh = q + ((size_t) t * heads + hb + h) * kFaD;
        float a = 0.0f;
        for (int c = p; c < kFaD; c += 16) a = fmaxf(a, fabsf(qh[c]));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        if (p == 0) {
            int e = 0;
            (void) frexpf(a, &e);   // a < 2^e
            qmul[h] = (a > 0.0f && a < INFINITY) ? ldexpf(1.0f, min(max(15 - e, -100), 100)) : 1.0f;
        }
        __syncthreads();
    }
    // the block's 16 heads' queries (the matrix's A operand), and their running softmax state
    for (int i = tid; i < kFaHeads * kFaD; i += kThreads) {
        const int h = i / kFaD, c = i % kFaD;
        float v = q[((size_t) t * heads + hb + h) * kFaD + c];
        if constexpr (QSC) v *= qmul[h];
        if constexpr (QM == 3) {
            qf[h * kFaD + c] = v;
        } else if constexpr (QM == 1) {
            qs[h * kFaLd + c] = f2bf16(v);
        } else {
            const _Float16 vh = (_Float16) v;
            qs[h * kFaLd + c] = __builtin_bit_cast(uint16_t, vh);
            if constexpr (QM == 2) qs[kFaHeads * kFaLd + h * kFaLd + c] = __builtin_bit_cast(uint16_t, (_Float16) (v - (float) vh));
        }
    }
    if (tid < kFaHeads) {
        m_run[tid] = -1e30f;
        den[tid] = 0.0f;
    }
    f8v o[PM == 2 ? 1 : 4];   // the matrix-core P.V's accumulators (8 of a 16 x 16 tile, 4 tiles: this warp's 64 dims)
    float o2[PM == 2 ? kFaHeads : 1][2];   // PM 2: dims 2 tid, 2 tid + 1 of the 16 heads
#pragma unroll
    for (int f = 0; f < (PM == 2 ? 1 : 4); ++f)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[f][i] = 0.0f;
#pragma unroll
    for (int h = 0; h < (PM == 2 ? kFaHeads : 1); ++h) o2[h][0] = o2[h][1] = 0.0f;
    __syncthreads();
    // the tile's n rows into ks (BF16 rows -> FP16; QM 1: as BF16).  Rows n .. 15 (a short last tile, or a short first one,
    // whose rows were never written) are zeros: their P is 0, and 0 x a stale Inf / NaN half would be NaN (fix).
    auto stage = [&](int t0, int n) {
        const int sr = tid >> 4, part = tid & 15;
        uint16_t* dst = ks + sr * kFaLd + part * 32;
        if (sr >= n) {
#pragma unroll
            for (int i = 0; i < 32; ++i) dst[i] = 0;
            return;
        }
        const uint8_t* row = rows[sr];
        uint16_t tmp[32];
        if (t0 + sr < n_win) {
            const uint4* src = (const uint4*) (row + part * 64);
            const uint4 a = src[0], b = src[1], c = src[2], d = src[3];
            ((uint4*) tmp)[0] = a;
            ((uint4*) tmp)[1] = b;
            ((uint4*) tmp)[2] = c;
            ((uint4*) tmp)[3] = d;
        } else {
            unpack32(row, part, (uint4*) tmp);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            if constexpr (QM == 1) dst[i] = tmp[i];
            else dst[i] = __builtin_bit_cast(uint16_t, (_Float16) bf16f(tmp[i]));
        }
    };
    // a ks value as F32 (FP16 or BF16 rows)
    auto kv_at = [&](int r, int c) -> float {
        if constexpr (QM == 1) return bf16f(ks[r * kFaLd + c]);
        else return (float) __builtin_bit_cast(_Float16, ks[r * kFaLd + c]);
    };
    const int total = n_win + n_sel;
    for (int t0 = 0; t0 < total; t0 += kFaKeys) {
        const int nt = min(kFaKeys, total - t0);
        if (tid < nt) {
            const int j = t0 + tid;
            rows[tid] = j < n_win ? (const uint8_t*) (wbase + (size_t) j * kFaD) : ckv + (size_t) sl[j - n_win] * crow;
        }
        __syncthreads();
        stage(t0, nt);
        __syncthreads();
        if constexpr (QM == 3) {   // the scores as scalar F32 dots: a thread a (head, key)
            const int h = tid >> 4, k = tid & 15;
            const float* qh = qf + h * kFaD;
            float a = 0.0f;
            for (int c = 0; c < kFaD; ++c) a += qh[c] * kv_at(k, c);
            if constexpr (QSC) sc[h][k] = t0 + k < total ? a * (scale / qmul[h]) : -INFINITY;
            else sc[h][k] = t0 + k < total ? a * scale : -INFINITY;
        } else {   // the scores in 8 slices of 64 dims, a warp a slice, summed in shared
            f8v acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int k0 = wp * 64;
            for (int kk = k0; kk < k0 + 64; kk += 16) {
                const uint32_t* qa = (const uint32_t*) (qs + l16 * kFaLd + kk);
                const uint32_t* kb = (const uint32_t*) (ks + l16 * kFaLd + kk);
                if constexpr (QM == 1) {
                    s16v a, b;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a[2 * i] = (short) (qa[i] & 0xffffu);
                        a[2 * i + 1] = (short) (qa[i] >> 16);
                        b[2 * i] = (short) (kb[i] & 0xffffu);
                        b[2 * i + 1] = (short) (kb[i] >> 16);
                    }
                    acc = set_wmma_bf16(a, b, acc);
                } else {
                    h16v a, b;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {   // (two halves a load)
                        a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] & 0xffffu));
                        a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (qa[i] >> 16));
                        b[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] & 0xffffu));
                        b[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (kb[i] >> 16));
                    }
                    acc = set_wmma(a, b, acc);
                    if constexpr (QM == 2) {   // + Q's low half against the same keys
                        const uint32_t* ql = (const uint32_t*) (qs + kFaHeads * kFaLd + l16 * kFaLd + kk);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (ql[i] & 0xffffu));
                            a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (ql[i] >> 16));
                        }
                        acc = set_wmma(a, b, acc);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) part[wp][set_wmma_row(i, hi)][l16] = acc[i];
            __syncthreads();
            const int h = tid >> 4, k = tid & 15;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) a += part[w][h][k];
            if constexpr (QSC) sc[h][k] = t0 + k < total ? a * (scale / qmul[h]) : -INFINITY;
            else sc[h][k] = t0 + k < total ? a * scale : -INFINITY;
        }
        __syncthreads();
        {   // the online softmax: a warp per 2 heads, a lane a key (16 lanes a head)
            const int h = 2 * wp + (lane >> 4), k = lane & 15;
            float m = sc[h][k];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const float mo = m_run[h], mn = fmaxf(mo, m);
            const float p = expf(sc[h][k] - mn);
            if constexpr (PM == 2) {
                sc[h][k] = p;   // (this lane's own entry: read above)
            } else {
                const _Float16 ph = (_Float16) p;
                ps[0][h][k] = __builtin_bit_cast(uint16_t, ph);
                if constexpr (PM == 1) ps[kPh - 1][h][k] = __builtin_bit_cast(uint16_t, (_Float16) (p - (float) ph));
            }
            float s = p;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (k == 0) {
                ts[h] = expf(mo - mn);
                den[h] = den[h] * ts[h] + s;
                m_run[h] = mn;
            }
        }
        __syncthreads();
        if constexpr (PM == 2) {   // the values as scalar F32 sums (sparse_attn_packed_kernel's order a tile)
            const int c = tid * 2;
#pragma unroll
            for (int h = 0; h < kFaHeads; ++h) {
                o2[h][0] *= ts[h];
                o2[h][1] *= ts[h];
            }
            for (int r = 0; r < nt; ++r) {
                const float v0 = kv_at(r, c), v1 = kv_at(r, c + 1);
#pragma unroll
                for (int h = 0; h < kFaHeads; ++h) {
                    o2[h][0] += sc[h][r] * v0;
                    o2[h][1] += sc[h][r] * v1;
                }
            }
        } else {
            const int dim0 = wp * 64;
#pragma unroll
            for (int f = 0; f < 4; ++f)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[f][i] *= ts[set_wmma_row(i, hi)];
#pragma unroll
            for (int f = 0; f < 4; ++f) {   // the values: P (16 heads x 16 keys) x V (16 keys x 64 dims this warp)
                h16v a, b;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int dm = dim0 + f * 16 + l16;
                    if constexpr (QM == 1) {
                        b[2 * i] = (_Float16) bf16f(ks[(2 * i) * kFaLd + dm]);
                        b[2 * i + 1] = (_Float16) bf16f(ks[(2 * i + 1) * kFaLd + dm]);
                    } else {
                        b[2 * i] = __builtin_bit_cast(_Float16, ks[(2 * i) * kFaLd + dm]);
                        b[2 * i + 1] = __builtin_bit_cast(_Float16, ks[(2 * i + 1) * kFaLd + dm]);
                    }
                }
#pragma unroll
                for (int pl = 0; pl < kPh; ++pl) {   // (PM 1: the low plane of P against the same V)
                    const uint32_t* pp = (const uint32_t*) ps[pl][l16];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] & 0xffffu));
                        a[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (pp[i] >> 16));
                    }
                    o[f] = set_wmma(a, b, o[f]);
                }
            }
        }
        __syncthreads();
    }
    if constexpr (PM == 2) {
        const int c = tid * 2;
#pragma unroll
        for (int h = 0; h < kFaHeads; ++h) {
            const float dv = den[h] + expf(sink[hb + h] - m_run[h]);
            float* ov = out + ((size_t) t * heads + hb + h) * kFaD;
            ov[c] = o2[h][0] / dv;
            ov[c + 1] = o2[h][1] / dv;
        }
    } else {
#pragma unroll
        for (int f = 0; f < 4; ++f)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int h = set_wmma_row(i, hi), dm = wp * 64 + f * 16 + l16;
                const float dv = den[h] + expf(sink[hb + h] - m_run[h]);
                out[((size_t) t * heads + hb + h) * kFaD + dm] = o[f][i] / dv;
            }
    }
}
#endif   // STRATA_USE_HIP

// the ring's rows are kKvFp8: E4M3 codes, a UE8M0 scale per 32 (decoded: exactly the BF16 values they were made from)
__global__ void ring_to_ext_kernel(const uint8_t* __restrict__ ring, uint16_t* __restrict__ ext, long long p0, int n,
                                   int d) {
    const int r = blockIdx.x;
    const long long p = p0 - n + r;
    const uint8_t* row = ring + (size_t) (p >= 0 ? p % n : 0) * kv_row_bytes(kKvFp8, d);
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        ext[(size_t) r * d + e] = p >= 0 ? f2bf16(e4m3_value(row[e]) * ue8m0_val(row[d + e / 32])) : (uint16_t) 0;
}

// a window row's BF16 values (FP8 roundings: each an E4M3 value times its group's power of two) as E4M3 codes at the
// smallest power-of-two scale 2^k their group fits under (max |v| <= 448 * 2^k = 1.75 * 2^(8 + k)): never above the
// rounding's own scale, so each v / 2^k is that E4M3 value times a power of two >= 1, at most 448 - an E4M3 value again
__global__ void __launch_bounds__(kThreads) ext_to_ring_kernel(const uint16_t* __restrict__ ext,
                                                               uint8_t* __restrict__ ring, long long p0, int n, int T,
                                                               int d, uint8_t* __restrict__ backup) {
    const int t = blockIdx.x;
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    const size_t rb = kv_row_bytes(kKvFp8, d);
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    if (backup != nullptr) {
        for (size_t i = threadIdx.x; i < rb; i += blockDim.x) backup[(size_t) t * rb + i] = row[i];
        __syncthreads();
    }
    const uint16_t* src = ext + (size_t) (n + t) * d;
    for (int e0 = 0; e0 < d; e0 += blockDim.x) {   // (d a multiple of 32: a warp's 32 values one group)
        const int e = e0 + (int) threadIdx.x;
        const float v = e < d ? bf16f(src[e]) : 0.0f;
        uint32_t a = __float_as_uint(v) & 0x7fffffffu;   // (the magnitudes' order as bits)
        for (int o = 16; o > 0; o >>= 1) a = max(a, (uint32_t) __shfl_xor_sync(0xffffffffu, a, o));
        const int k = a == 0u ? 0 : (int) (a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u ? 1 : 0);
        if (e >= d) continue;
        row[e] = (uint8_t) e4m3_code(v, k);
        if ((e & 31) == 0) row[d + e / 32] = (uint8_t) (k + 127);
    }
}

__global__ void ring_restore_kernel(uint8_t* __restrict__ ring, const uint8_t* __restrict__ backup, long long p0, int n,
                                    int keep, size_t rb) {
    const int t = keep + blockIdx.x;
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) row[i] = backup[(size_t) t * rb + i];
}
__global__ void ring_restore_batch_kernel(const RingBatch b, long long p0, int n, int keep, size_t rb) {
    const int t = keep + blockIdx.x, li = blockIdx.y;
    uint8_t* row = b.ring[li] + (size_t) ((p0 + t) % n) * rb;
    const uint8_t* src = b.bak[li] + (size_t) t * rb;
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) row[i] = src[i];
}

// ---------------------------------------------------------------- DS4_MULTI_ATTN (docs/dsv41/cards/FG-05.md)
// Ported from Project Maya #639 (MIT; third_party/project-maya/NOTICE.md).  Each kernel here is a copy of the single-run
// kernel above it names, operation for operation; only which row / run / cache a block reads and writes differs.  The
// originals are left as they were, so the flag-off path's machine code is unchanged.

// finish_kernel<false> over the rows of several sequences: vector v is row v / vpp's, at that row's rope position,
// written (f.dst) to that row's dst row
__global__ void __launch_bounds__(kThreads) finish_rows_kernel(float* x, int len, long long stride, Finish f,
                                                               FinishRows fr) {
    const int vi = (int) blockIdx.x, row = vi / f.vpp;
    float* v = x + (long long) vi * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = fr.pos[row];
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + ((long long) fr.dst[row] * f.vpp + vi % f.vpp) * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else v[e] = r[j];
    }
}

// ring_to_ext_kernel, block (r, k): run k's window row r into its block of the shared buffer
__global__ void ring_to_ext_runs_kernel(const RingRuns rr, uint16_t* __restrict__ ext, int n, int d) {
    const int r = blockIdx.x, k = blockIdx.y;
    const long long p = rr.p0[k] - n + r;
    const uint8_t* row = rr.ring[k] + (size_t) (p >= 0 ? p % n : 0) * kv_row_bytes(kKvFp8, d);
    uint16_t* dst = ext + ((size_t) rr.e0[k] + r) * d;
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        dst[e] = p >= 0 ? f2bf16(e4m3_value(row[e]) * ue8m0_val(row[d + e / 32])) : (uint16_t) 0;
}

// ext_to_ring_kernel, block (t, k): run k's row t (past its rows: nothing), from its block of the shared buffer, its
// replaced ring row to its own backups
__global__ void __launch_bounds__(kThreads) ext_to_ring_runs_kernel(const RingRuns rr, const uint16_t* __restrict__ ext,
                                                                    int n, int d) {
    const int t = blockIdx.x, k = blockIdx.y, T = rr.T[k];
    if (t >= T) return;
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    const size_t rb = kv_row_bytes(kKvFp8, d);
    uint8_t* row = rr.ring[k] + (size_t) ((rr.p0[k] + t) % n) * rb;
    uint8_t* backup = rr.bak[k];
    if (backup != nullptr) {
        for (size_t i = threadIdx.x; i < rb; i += blockDim.x) backup[(size_t) t * rb + i] = row[i];
        __syncthreads();
    }
    const uint16_t* src = ext + ((size_t) rr.e0[k] + n + t) * d;
    for (int e0 = 0; e0 < d; e0 += blockDim.x) {   // (d a multiple of 32: a warp's 32 values one group)
        const int e = e0 + (int) threadIdx.x;
        const float v = e < d ? bf16f(src[e]) : 0.0f;
        uint32_t a = __float_as_uint(v) & 0x7fffffffu;   // (the magnitudes' order as bits)
        for (int o = 16; o > 0; o >>= 1) a = max(a, (uint32_t) __shfl_xor_sync(0xffffffffu, a, o));
        const int k2 = a == 0u ? 0 : (int) (a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u ? 1 : 0);
        if (e >= d) continue;
        row[e] = (uint8_t) e4m3_code(v, k2);
        if ((e & 31) == 0) row[d + e / 32] = (uint8_t) (k2 + 127);
    }
}

// unpack_sel_kernel (a.gen 0) for rows of several sequences: row t's selection from its own cache c.ckv[t]
__global__ void __launch_bounds__(kThreads) unpack_sel_rows_kernel(const AttnCkv c, const int* __restrict__ sel,
                                                                   int sel_stride, AttnRows ar, int cap,
                                                                   uint16_t* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * kStage + (int) (threadIdx.x >> 4), part = threadIdx.x & 15;
    if (i >= ar.nsel[t]) return;
    const size_t r = (size_t) t * cap + i;
    unpack32((const uint8_t*) c.ckv[t] + (size_t) sel[(size_t) t * sel_stride + i] * kv_row_bytes(kKvFp4, 512), part,
             (uint4*) (out + r * 512 + part * 32));
}

__global__ void hc_mean_kernel(const float* __restrict__ h, int hc, int dim, float* __restrict__ out,
                               long long out_stride) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int s = 0; s < hc; ++s) a += h[((size_t) t * hc + s) * dim + e];
    out[(long long) t * out_stride + e] = a / (float) hc;
}

__global__ void iota_rows_kernel(int* out, int n, int stride) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[(size_t) t * stride + i] = i;
}

// a warp a vocabulary row: E[*tok] (shared memory) . H[v]
__global__ void __launch_bounds__(kThreads) markov_kernel(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                          const uint16_t* __restrict__ H, const int* __restrict__ tok,
                                                          int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// the same bias with the head quantised to Q8_0 blocks of 32 (34 B: an F16 d and 32 int8): half the bytes read for the same sums.
// A lane's 8 values sit inside one block (8 divides 32), so a lane reads one d and 8 codes; the sums are the same terms, in the
// block's own order, with the quantisation's error the only difference.
__global__ void __launch_bounds__(kThreads) markov_kernel_q8(float* __restrict__ logits, const uint16_t* __restrict__ E,
                                                            const uint8_t* __restrict__ Hq, const int* __restrict__ tok,
                                                            int V, int R) {
    extern __shared__ float er[];
    const uint16_t* erow = E + (size_t) (*tok) * R;
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const uint8_t* hr = Hq + (size_t) v * (size_t) (R / 32) * 34;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint8_t* blk = hr + (size_t) (c >> 5) * 34;
        const float d = __half2float(*(const __half*) blk);
        const int8_t* q = (const int8_t*) (blk + 2 + (c & 31));
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += (float) q[i] * er[c + i];
        a += d * t;
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}

// a member's rows [v0, v0 + vn) of the bias (markov_kernel's sums, the prev token's E row already on the device, BF16)
__global__ void __launch_bounds__(kThreads) markov_share_kernel(float* __restrict__ logits,
                                                                const uint16_t* __restrict__ H,
                                                                const uint16_t* __restrict__ erow, int vn, int R) {
    extern __shared__ float er[];
    for (int i = threadIdx.x; i < R; i += blockDim.x) er[i] = bf16f(erow[i]);
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (v >= vn) return;
    const uint16_t* hr = H + (size_t) v * R;
    float a = 0.0f;
    for (int c = lane * 8; c < R; c += 32 * 8) {
        const uint4 q = *(const uint4*) (hr + c);
        const __half2* hp = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(hp[i]);
            a += f.x * er[c + 2 * i] + f.y * er[c + 2 * i + 1];
        }
    }
    a = warp_sum(a);
    if (lane == 0) logits[v] += a;
}
// the share's argmax (the greedy sampler's rule: the larger value, on a tie the lower index) -> slot {value, v0 + i,
// seq}, the seq last
__global__ void __launch_bounds__(1024) argmax_share_kernel(const float* __restrict__ row, int vn, int v0,
                                                           unsigned* slot, unsigned seq) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float bv = __int_as_float(0xff800000);
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < vn; i += blockDim.x) {
        const float x = row[i];
        if (x > bv) bv = x, bi = i;   // (ascending: the lowest index of a tie stays)
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((threadIdx.x & 31) == 0) sv[threadIdx.x >> 5] = bv, si[threadIdx.x >> 5] = bi;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) bv = sv[w], bi = si[w];
        slot[0] = __float_as_uint(bv);
        slot[1] = (unsigned) (v0 + (bi == 0x7fffffff ? 0 : bi));
        __threadfence_system();
        *(volatile unsigned*) (slot + 2) = seq;
    }
}
// every member's share argmax of this step (slots nm apart) -> the global one into *tok, and its E row (host-mapped
// BF16 [V][R]) into erow for the next step's bias
__global__ void markov_pick_kernel(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E,
                                   uint16_t* erow, int R) {
    __shared__ int pick;
    if ((int) threadIdx.x < nm)
        while ((int) (*(volatile const unsigned*) (slots + (size_t) threadIdx.x * 4 + 2) - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(32);
#endif
        }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = __int_as_float(0xff800000);
        int bi = 0x7fffffff;
        for (int m = 0; m < nm; ++m) {   // (members in vocabulary order: a tie keeps the lower index)
            const float v = __uint_as_float(*(volatile const unsigned*) (slots + (size_t) m * 4));
            const int i = (int) *(volatile const unsigned*) (slots + (size_t) m * 4 + 1);
            if (v > bv || (v == bv && i < bi)) bv = v, bi = i;
        }
        pick = bi;
        *tok = bi;
    }
    __syncthreads();
#if defined(STRATA_USE_HIP)   // (E is fixed once loaded: a plain load is as fresh)
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = E[(size_t) pick * R + i];
#else
    for (int i = threadIdx.x; i < R; i += blockDim.x) erow[i] = __ldcv(E + (size_t) pick * R + i);
#endif
}

__global__ void __launch_bounds__(1024) confidence_kernel(const float* __restrict__ x, const uint16_t* __restrict__ E,
                                                          const int* __restrict__ tok, const float* __restrict__ w,
                                                          int dim, int R, float* __restrict__ out) {
    const int t = blockIdx.x;
    const float* xt = x + (size_t) t * dim;
    const uint16_t* er = E + (size_t) tok[t] * R;
    float a = 0.0f;
    for (int i = threadIdx.x; i < dim + R; i += blockDim.x) a += w[i] * (i < dim ? xt[i] : bf16f(er[i - dim]));
    a = block_sum(a);
    if (threadIdx.x == 0) out[t] = 1.0f / (1.0f + expf(-a));
}

// ---------------------------------------------------------------- engram
// ggml's IQ4_NL: a block of 32 values is 18 bytes, an fp16 d then 16 packed nibble bytes; the low nibble of qs[j] is
// value j and its high nibble value j + 16, each an index into this codebook (ggml-common.h's kvalues_iq4nl,
// dequantize_row_iq4_nl).  A row is hd / 32 of these blocks, the d of every one inside it - no separate scales.
__device__ __constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,   38,   53,   69,   89,   113};
__device__ __forceinline__ float iq4nl_value(const uint8_t* b, int e) {
    const uint8_t q = b[2 + (e & 15)];
    const int v = (e & 16) ? (q >> 4) : (q & 15);
    return __half2float(__ushort_as_half(*(const uint16_t*) b)) * (float) kIq4nl[v];
}

__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ scales, int hd,
                                      float* __restrict__ out, int fmt) {
    const int r = blockIdx.x;
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) r * hd + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) r * hd + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const float hv = e4m3_value(rows[(size_t) r * hd + e]);
        const uint8_t sb = scales[(size_t) r * (hd / 32) + e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

__global__ void __launch_bounds__(1024) engram_gate_kernel(float* __restrict__ h, const float* __restrict__ kv,
                                                           const float* __restrict__ w, int hc, int dim, float eps) {
    const int k = blockIdx.x, t = blockIdx.y;
    float* hk = h + ((size_t) t * hc + k) * dim;
    const float* kvt = kv + (size_t) t * (hc + 1) * dim;
    const float* key = kvt + (size_t) k * dim;
    const float* val = kvt + (size_t) hc * dim;
    const float* wk = w + (size_t) k * dim;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int e = threadIdx.x; e < dim; e += blockDim.x) {
        const float hv = hk[e], kv_ = key[e];
        sh += hv * hv;
        sk += kv_ * kv_;
        sd += hv * wk[e] * kv_;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    const float rstd = rsqrtf(sh / (float) dim + eps) * rsqrtf(sk / (float) dim + eps);
    const float dot = sd * rstd * rsqrtf((float) dim);
    const float sq = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float gate = 1.0f / (1.0f + expf(-sq));
    for (int e = threadIdx.x; e < dim; e += blockDim.x) hk[e] = hk[e] + gate * val[e];
}

__global__ void gather_kernel(const float* __restrict__ x, Gather g, int cols, float* __restrict__ out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) out[(size_t) r * cols + c] = x[(size_t) g.src[r] * cols + c];
}

__global__ void combine_kernel(const float* __restrict__ ed, const float* __restrict__ cpu,
                               const float* __restrict__ sh, Combine cm, int dim, float* __restrict__ out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = 0.0f;
    for (int j = 0; j < cm.n[t]; ++j) a += cm.w[t][j] * ed[(size_t) cm.row[t][j] * dim + e];
    if (cpu != nullptr && cm.cpu[t]) a += cpu[(size_t) t * dim + e];
    if (sh != nullptr) a += sh[(size_t) t * dim + e];
    out[(size_t) t * dim + e] = a;
}

__global__ void __launch_bounds__(512) route_publish_kernel(const float* __restrict__ logits,
                                                            const float* __restrict__ bias, int n, int k,
                                                            float route_scale, const unsigned long long* __restrict__ vram,
                                                            const float* __restrict__ x, int dim, int* idx, float* w,
                                                            unsigned long long* plan_ptr, Req* req,
                                                            const float* __restrict__ bias_vl, unsigned vl) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    __shared__ int on_cpu;
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            if (ix >= n) ix = route_fallback_pick(pick, r, n);   // (upstream fix C1: a NaN gate input)
            pick[r] = ix;
            bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        int cpu = 0;
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            cpu |= vram[pick[r]] == 0ull;
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
        on_cpu = cpu;
    }
    // the token's FFN input to the host only where the CPU lane reads it (a pick of its on the CPU lane): each row
    // 20 KB over PCIe, where most windows have none
    __syncthreads();
    if (req != nullptr && on_cpu)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) req->x[t][i] = x[(size_t) t * dim + i];
}

// a later layer's m likely picks (route_kernel's choice from its logits) into req->ahead[t][off ..)
__global__ void __launch_bounds__(512) route_ahead_kernel(const float* __restrict__ logits,
                                                          const float* __restrict__ bias, int n, int m, Req* req,
                                                          int off) {
    __shared__ float bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        bs[i] = sqrtf(sp) + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < m; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            req->ahead[t][off + r] = ix < n ? ix : -1;
            if (ix < n) bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
}

__global__ void publish_kernel(Req* req, int T, unsigned seq, int na0, int na1) {
    __threadfence_system();
    req->na[0] = na0;
    req->na[1] = na1;
    req->T = T;
    __threadfence_system();
    *(volatile unsigned*) &req->seq = seq;
}

// DS4_ROUTE_FUSE (after Project Maya #511's route_pair_kernel by @pepuscz): route_publish_kernel and
// route_ahead_kernel as the two blocks of one launch, row t = blockIdx.y - block 0 the first's code for row t (the
// request's ids / w / x, idx / w / plan_ptr), block 1 the second's (req->ahead[t][off ..)): disjoint writes, read the
// same way.  publish_kernel follows in a launch of its own, as route_publish launches it.  The old kernels stay as they
// are: route_fuse_check (ds4_model.cu) compares the two
__global__ void __launch_bounds__(512) route_pair_kernel(const RoutePair a) {
    if (blockIdx.x == 0) {
        __shared__ float sc[1024], bs[1024];
        __shared__ float bv[32];
        __shared__ int bi[32];
        __shared__ int pick[16];
        __shared__ int on_cpu;
        const int t = blockIdx.y, n = a.n, k = a.k;
        const float* logits = a.logits + (size_t) t * 1024;
        const float* bias = a.bias;
        if (a.bias_vl != nullptr && ((a.vl >> t) & 1u)) bias = a.bias_vl;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float l = logits[i];
            const float sp = l > 20.0f ? l : log1pf(expf(l));
            sc[i] = sqrtf(sp);
            bs[i] = sc[i] + bias[i];
        }
        __syncthreads();
        const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
        for (int r = 0; r < k; ++r) {
            float best = -FLT_MAX;
            int bix = 0x7fffffff;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
                if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
            }
            if (lane == 0) bv[wp] = best, bi[wp] = bix;
            __syncthreads();
            if (threadIdx.x == 0) {
                float b = -FLT_MAX;
                int ix = 0x7fffffff;
                for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                    if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
                if (ix >= n) ix = route_fallback_pick(pick, r, n);   // (upstream fix C1: a NaN gate input)
                pick[r] = ix;
                bs[ix] = -FLT_MAX;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (int r = 0; r < k; ++r) s += sc[pick[r]];
            int cpu = 0;
            for (int r = 0; r < k; ++r) {
                const float wv = sc[pick[r]] / (s + 1e-20f) * a.route_scale;
                a.idx[(size_t) t * kPicks + r] = pick[r];
                a.w[(size_t) t * kPicks + r] = wv;
                a.plan_ptr[(size_t) t * kPicks + r] = a.vram[pick[r]];
                cpu |= a.vram[pick[r]] == 0ull;
                if (a.req != nullptr) {
                    a.req->ids[t][r] = pick[r];
                    a.req->w[t][r] = wv;
                }
            }
            on_cpu = cpu;
        }
        __syncthreads();
        if (a.req != nullptr && on_cpu)
            for (int i = threadIdx.x; i < a.dim; i += blockDim.x) a.req->x[t][i] = a.x[(size_t) t * a.dim + i];
        return;
    }
    __shared__ float bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int t = blockIdx.y, n = a.a_n, m = a.m, off = a.off;
    const float* logits = a.a_logits + (size_t) t * 1024;
    const float* bias = a.a_bias;
    Req* req = a.req;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        bs[i] = sqrtf(sp) + bias[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int r = 0; r < m; ++r) {
        float best = -FLT_MAX;
        int bix = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
            if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
        }
        if (lane == 0) bv[wp] = best, bi[wp] = bix;
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = -FLT_MAX;
            int ix = 0x7fffffff;
            for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
            req->ahead[t][off + r] = ix < n ? ix : -1;
            if (ix < n) bs[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
}

// DS4_LP_ROUTE / DS4_LP_PUBFUSE (docs/dsv41/cards/HX-21.md; each default off; exact by construction, checked at load by
// lp_route_check): route_publish_kernel's work, with
//  FAST - the k best of the n scores found per warp (each warp its own k best, its lanes' 2 values each, k shuffle rounds),
//         then merged by warp 0 (k shuffle rounds over the warps' lists): 3 barriers instead of 2 k + 1.  The scores'
//         order is a total order ((score, index): higher score, then lower index - the old rounds' rule), so whenever the
//         k picks all score above -FLT_MAX and no score is NaN these are the old rounds' picks in the old rounds' order.
//         Otherwise (a NaN, or fewer than k scores above -FLT_MAX: the old rounds then pick by index among -FLT_MAX
//         ties, already picked ones included) the old rounds run here on the same scores.  The weights, the plan and the
//         request's stores are the old code's, unchanged.
//  PUB  - (a window of one row: grid 1) publish_kernel's stores (na, T, then seq after a system fence) at the end of this
//         launch, after every thread's own system fence and a barrier: the same bytes, one launch fewer on the lead.
#if !defined(STRATA_METAL)
template <bool FAST, bool PUB>
__global__ void __launch_bounds__(512) route_publish_lp_kernel(const float* __restrict__ logits,
                                                               const float* __restrict__ bias, int n, int k,
                                                               float route_scale, const unsigned long long* __restrict__ vram,
                                                               const float* __restrict__ x, int dim, int* idx, float* w,
                                                               unsigned long long* plan_ptr, Req* req,
                                                               const float* __restrict__ bias_vl, unsigned vl, unsigned seq,
                                                               int na0, int na1, unsigned* pub_cnt) {
    __shared__ float sc[1024], bs[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int pick[16];
    __shared__ int on_cpu;
    __shared__ float cv[16 * 16];   // FAST: warp q's k best, cv / ci [q * k + r]
    __shared__ int ci[16 * 16];
    __shared__ int slow;
    const int t = blockIdx.x;
    logits += (size_t) t * 1024;
    if (bias_vl != nullptr && ((vl >> t) & 1u)) bias = bias_vl;
    int nan_here = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float l = logits[i];
        const float sp = l > 20.0f ? l : log1pf(expf(l));
        sc[i] = sqrtf(sp);
        bs[i] = sc[i] + bias[i];
        nan_here |= bs[i] != bs[i];
    }
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    bool rounds = true;
    if (FAST) {
        const int any_nan = __syncthreads_or(nan_here);   // (also the barrier after the scores)
        const int i0 = threadIdx.x, i1 = threadIdx.x + 512;
        const float e0 = i0 < n ? bs[i0] : -FLT_MAX, e1 = i1 < n ? bs[i1] : -FLT_MAX;
        bool u0 = i0 >= n, u1 = i1 >= n;   // (taken, or not a score)
        for (int r = 0; r < k; ++r) {
            float best = -FLT_MAX;
            int bix = 0x7fffffff;
            if (!u0 && (e0 > best || (e0 == best && i0 < bix))) best = e0, bix = i0;
            if (!u1 && (e1 > best || (e1 == best && i1 < bix))) best = e1, bix = i1;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
                if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
            }
            if (lane == 0) cv[wp * k + r] = best, ci[wp * k + r] = bix;
            if (bix == i0) u0 = true;
            else if (bix == i1) u1 = true;
        }
        __syncthreads();
        if (wp == 0) {
            const int nc = (int) (blockDim.x >> 5) * k;   // (<= 16 x 16: 8 a lane)
            float m[8];
            int mi[8];
            bool mu[8];
            for (int q = 0; q < 8; ++q) {
                const int c = lane + 32 * q;
                m[q] = c < nc ? cv[c] : -FLT_MAX;
                mi[q] = c < nc ? ci[c] : 0x7fffffff;
                mu[q] = c >= nc || mi[q] == 0x7fffffff;
            }
            int bad = any_nan;
            for (int r = 0; r < k; ++r) {
                float best = -FLT_MAX;
                int bix = 0x7fffffff;
                for (int q = 0; q < 8; ++q)
                    if (!mu[q] && (m[q] > best || (m[q] == best && mi[q] < bix))) best = m[q], bix = mi[q];
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
                    if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
                }
                for (int q = 0; q < 8; ++q)
                    if (!mu[q] && mi[q] == bix) mu[q] = true;
                if (lane == 0) pick[r] = bix;
                bad |= !(best > -FLT_MAX) || bix < 0 || bix >= n;
            }
            if (lane == 0) slow = bad;
        }
        __syncthreads();
        rounds = slow != 0;
    } else {
        __syncthreads();
    }
    if (rounds) {   // (the old rounds: route_publish_kernel's code)
        for (int r = 0; r < k; ++r) {
            float best = -FLT_MAX;
            int bix = 0x7fffffff;
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                if (bs[i] > best || (bs[i] == best && i < bix)) best = bs[i], bix = i;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bix, o);
                if (ov > best || (ov == best && oi < bix)) best = ov, bix = oi;
            }
            if (lane == 0) bv[wp] = best, bi[wp] = bix;
            __syncthreads();
            if (threadIdx.x == 0) {
                float b = -FLT_MAX;
                int ix = 0x7fffffff;
                for (int q = 0; q < (int) (blockDim.x >> 5); ++q)
                    if (bv[q] > b || (bv[q] == b && bi[q] < ix)) b = bv[q], ix = bi[q];
                if (ix >= n) ix = route_fallback_pick(pick, r, n);   // (upstream fix C1: a NaN gate input)
                pick[r] = ix;
                bs[ix] = -FLT_MAX;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += sc[pick[r]];
        int cpu = 0;
        for (int r = 0; r < k; ++r) {
            const float wv = sc[pick[r]] / (s + 1e-20f) * route_scale;
            idx[(size_t) t * kPicks + r] = pick[r];
            w[(size_t) t * kPicks + r] = wv;
            plan_ptr[(size_t) t * kPicks + r] = vram[pick[r]];
            cpu |= vram[pick[r]] == 0ull;
            if (req != nullptr) {
                req->ids[t][r] = pick[r];
                req->w[t][r] = wv;
            }
        }
        on_cpu = cpu;
    }
    __syncthreads();
    if (req != nullptr && on_cpu)
        for (int i = threadIdx.x; i < dim; i += blockDim.x) req->x[t][i] = x[(size_t) t * dim + i];
    if (PUB && req != nullptr) {   // (publish_kernel's stores, once every thread's are out)
        __threadfence_system();
        __syncthreads();
        // (helperwin DS4_HW_PUBT: a window of T > 1 rows - every block's stores fenced to the system before it counts
        //  itself in; the last block to count sets the word back to 0 and publishes, after all T rows' stores)
        bool last = true;
        if (gridDim.x > 1 && threadIdx.x == 0) {
            last = atomicAdd(pub_cnt, 1u) == gridDim.x - 1u;
            if (last) *pub_cnt = 0u;
        }
        if (threadIdx.x == 0 && last) {
            __threadfence_system();
            req->na[0] = na0;
            req->na[1] = na1;
            req->T = (int) gridDim.x;
            __threadfence_system();
            *(volatile unsigned*) &req->seq = seq;
        }
    }
}
#endif

__global__ void wait_seq_kernel(const unsigned* seq_ptr, unsigned seq) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) seq_ptr != seq) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

__global__ void init_pre_kernel(float* pre, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * 8) pre[i] = (i % 8) == 0 ? 1.0f : 0.0f;
}


// the hand-offs through host-mapped memory: a grid-stride copy of the segments (16 bytes a thread), then (put, flag
// given) the last block to finish sets the flag after a system fence; a get first waits for the flag
__device__ __forceinline__ void xfer_copy(const Xfer& x, bool from_host) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    for (size_t q = (size_t) blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (size_t) gridDim.x * blockDim.x) {
        size_t r = q;
        int i = 0;
        while (r >= x.bytes[i] / 16) r -= x.bytes[i++] / 16;
        const uint4* src = (const uint4*) x.src[i] + r;
        ((uint4*) x.dst[i])[r] = from_host ? ld_fresh(src) : *src;
    }
}

__global__ void __launch_bounds__(kThreads) xfer_put_kernel(const Xfer x, unsigned* flag, unsigned seq, unsigned* cnt) {
    xfer_copy(x, false);
    if (flag == nullptr) return;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(cnt, 1u) == gridDim.x - 1) {
        *cnt = 0u;
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}

__device__ __forceinline__ void wait_at_least(const unsigned* flag, unsigned seq) {
    while ((int) (*(volatile const unsigned*) flag - seq) < 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
}

__global__ void __launch_bounds__(kThreads) xfer_get_kernel(const Xfer x, const unsigned* flag, unsigned seq) {
    if (threadIdx.x == 0) wait_at_least(flag, seq);
    __syncthreads();
    xfer_copy(x, true);
}

__global__ void wait_flags_kernel(const unsigned* flags, int n, int stride, unsigned seq) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < n; ++i) wait_at_least(flags + (size_t) i * stride, seq);
    __threadfence_system();
}
__global__ void set_flag_kernel(unsigned* flag, unsigned v) {
    __threadfence_system();
    *(volatile unsigned*) flag = v;
}

// three members or more: a reduce-scatter and an all-gather in one launch, a block a contiguous range of the row cut
// in nm chunks.  The block puts its range of this member's partial out and flags it; once every member's flag of the
// range is up it sums chunk mi of the range in member order (ar_sum_kernel's sums, bit for bit), puts that out and
// flags it; once every member's sum flag is up it reads the other chunks' sums.  Each member reads about two rows over
// PCIe whatever nm is, where every member reading every other row grows with nm
template <bool W>   // (DS4_AR_WIDE: W, the host rows and sums read with ld_wide)
__global__ void __launch_bounds__(kThreads) ar_rs_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                        unsigned* bflags, int nm, int mi, int slot, int slots,
                                                        unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (float4*) (rows + ((size_t) m * slots + slot) * stride); };
    float4* sum = (float4*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        float4* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = x4[q];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? x4[q] : W ? ld_wide(row(m) + q) : ld_fresh(row(m) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        sum[q] = v;
        x4[q] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = W ? ld_wide(sum + q) : ld_fresh(sum + q);
}

// ar_rs_kernel with the rows and the sums in FP16 (DS4_AR_F16): half the bytes over PCIe.  Every member's partial is
// rounded to FP16 (the owner's too), summed in F32 in member order, the sum rounded to FP16 - and every member, the
// owner too, takes that rounded sum, so the members' copies of the stream stay the same
__device__ __forceinline__ uint2 to_h4(float4 v) {
    const __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
    return make_uint2(*(const unsigned*) &a, *(const unsigned*) &b);
}
__device__ __forceinline__ float4 from_h4(uint2 u) {
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 sat4(float4 v) {   // (FP16's range: a partial past it saturates instead of inf)
    const float m = 65504.0f;
    return make_float4(fminf(fmaxf(v.x, -m), m), fminf(fmaxf(v.y, -m), m), fminf(fmaxf(v.z, -m), m), fminf(fmaxf(v.w, -m), m));
}
template <int W>   // (DS4_AR_WIDE: W 1, the host rows and sums read 16 bytes - two FP16 quads - a load: wide_pairs; W 2, also
                    // this member's row and sums stored 16 bytes a store)
__global__ void __launch_bounds__(kThreads) ar_rs16_kernel(float* x, int n4, float* rows, float* sums, size_t stride,
                                                          unsigned* bflags, int nm, int mi, int slot, int slots,
                                                          unsigned seq) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per), cs = (hi - lo + nm - 1) / nm;
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto sflag = [&](int m) {
        return bflags + ((((size_t) nm + m) * slots + slot) * kArBlocks + blockIdx.x) * 8;
    };
    auto row = [&](int m) { return (uint2*) (rows + ((size_t) m * slots + slot) * stride); };   // (FP16: half a row)
    uint2* sum = (uint2*) (sums + (size_t) slot * stride);
    if (lo < hi) {
        uint2* mine = row(mi);
        if constexpr (W == 2)
            wide_pairs(lo, hi, [&](int q) {
                const uint2 u0 = to_h4(sat4(x4[q])), u1 = to_h4(sat4(x4[q + 1]));
                *(uint4*) (mine + q) = make_uint4(u0.x, u0.y, u1.x, u1.y);
            }, [&](int q) { mine[q] = to_h4(sat4(x4[q])); });
        else
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = to_h4(sat4(x4[q]));
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    const int c0 = min(hi, lo + mi * cs), c1 = min(hi, c0 + cs);
    if constexpr (W) {
        // the same per-element chain as below (member order, __fadd_rn, to_h4(sat4) of every partial and of the sum), two
        // elements a thread, the peers' two quads one 16-byte load
        auto add = [](float4& v, const float4 p, int m) {
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        };
        wide_pairs(c0, c1, [&](int q) {
            float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), v1 = v0;
            for (int m = 0; m < nm; ++m) {
                uint2 u0, u1;
                if (m == mi) u0 = to_h4(sat4(x4[q])), u1 = to_h4(sat4(x4[q + 1]));
                else {
                    const uint4 w = ld_wide((const uint4*) (row(m) + q));
                    u0 = make_uint2(w.x, w.y), u1 = make_uint2(w.z, w.w);
                }
                add(v0, from_h4(u0), m);
                add(v1, from_h4(u1), m);
            }
            const uint2 s0 = to_h4(sat4(v0)), s1 = to_h4(sat4(v1));
            if constexpr (W == 2) *(uint4*) (sum + q) = make_uint4(s0.x, s0.y, s1.x, s1.y);
            else {
                sum[q] = s0;
                sum[q + 1] = s1;
            }
            x4[q] = from_h4(s0);
            x4[q + 1] = from_h4(s1);
        }, [&](int q) {
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int m = 0; m < nm; ++m) add(v, from_h4(m == mi ? to_h4(sat4(x4[q])) : ld_wide(row(m) + q)), m);
            const uint2 s16 = to_h4(sat4(v));
            sum[q] = s16;
            x4[q] = from_h4(s16);
        });
    } else
    for (int q = c0 + threadIdx.x; q < c1; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const uint2 u = m == mi ? to_h4(sat4(x4[q])) : ld_fresh(row(m) + q);
            const float4 p = from_h4(u);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        const uint2 s16 = to_h4(sat4(v));
        sum[q] = s16;
        x4[q] = from_h4(s16);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) sflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(sflag(threadIdx.x), seq);
    __syncthreads();
    if constexpr (W) {
        auto get2 = [&](int q) {
            const uint4 w = ld_wide((const uint4*) (sum + q));
            x4[q] = from_h4(make_uint2(w.x, w.y));
            x4[q + 1] = from_h4(make_uint2(w.z, w.w));
        };
        auto get1 = [&](int q) { x4[q] = from_h4(ld_wide(sum + q)); };
        wide_pairs(lo, c0, get2, get1);
        wide_pairs(c1, hi, get2, get1);
    } else
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x)
        if (q < c0 || q >= c1) x4[q] = from_h4(ld_fresh(sum + q));
}

// x[i] = the members' partials summed in member order (this member's from x, the others' from their host rows)
template <bool W>   // (DS4_AR_WIDE: W, the peers' host rows read with ld_wide)
__global__ void __launch_bounds__(kThreads) ar_sum_kernel(float* x, int n4, const float* peers, size_t stride,
                                                         const unsigned* flags, int nm, int mi, int slot, int slots,
                                                         unsigned seq) {
    if (threadIdx.x == 0)
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    __syncthreads();
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? ((const float4*) x)[q]
                                     : W ? ld_wide((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q)
                                         : ld_fresh((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        ((float4*) x)[q] = v;
    }
}

#if defined(STRATA_USE_HIP)
// DS4_AR_P2PK: a prompt chunk's all-reduce as a tiled reduce-scatter + all-gather of kernel stores into the peers'
// VRAM (hipDeviceMallocUncached receive slots, the same placement of work on every member).  The row is cut in nm chunks
// of 16-byte units (chunk k, summed by member k: units [cut[k], cut[k + 1])), every chunk in tiles of tu units; tile t of
// every chunk belongs to block t % gridDim on every member, and a tile has its own flags, so a slow pair (XTX #1's
// links) holds up only the tiles that cross it while the others go on.  Member o keeps, a parity (seq & 1), one slot a
// sender s: first s's partial of o's chunk (the reduce-scatter), then, once o has read that, s's summed chunk (the
// all-gather) - the same VRAM as one buffer, with a "free" flag a tile between the two uses.  Block b on member mi:
//   1. every tile t of b, every peer p: packs tile t of chunk p of its partial into p's slot mi, a system fence, rs[mi][t]
//      raised on p (no waits: a member's sends never depend on anything)
//   2a. every tile t of its own chunk: waits rs[p][t] (every peer, own memory), sums in member order (this member's partial
//      from x, the others' from the slots: ar_rs_kernel's / ar_rs16_kernel's __fadd_rn chain, the same bits), the sum into
//      x; then free[mi][t] raised on every peer (its slot on me is read) - for every tile up to the longest chunk's
//   2b. every tile t of its own chunk: waits free[p][t] (p has read my partial out of my slot on p), stores the sum into
//      p's slot mi, a system fence, ag[mi][t] raised on p
//   3. every tile t of every peer's chunk: waits ag[p][t], copies the sums into x
// The waits only ever reach back to an earlier step of a peer (1 <- nothing, 2a <- 1, 2b <- 2a, 3 <- 2b), so no cycle,
// whatever the block count.  Parity reuse: call seq writes the slots seq - 2 used; a member gets to call seq only past
// call seq - 1, whose step 2a waited for every peer's step 1 of seq - 1, so every peer had finished seq - 2 (stream
// order) - every chunk is non-empty (p2pk_cuts) for exactly that reason.  Flags hold the seq (monotone; >= seq).
// Loads of the slots: volatile 128-bit (glc dlc: past L0 / GL1; the memory is uncached, so no L2) after an acquire
// fence; stores into the peers: 128-bit non-temporal.  F16: a unit is 8 halves (two float4 of x), every partial (this
// member's too) rounded as ar_rs16_kernel does (to_h4(sat4)), the F32 sum rounded the same, and x takes that rounded
// sum on every member; F32: a unit is a float4, the bits as they are
typedef unsigned pk_u4 __attribute__((ext_vector_type(4)));
struct ArPk {
    pk_u4* buf[ArGroup::kPkMax];
    unsigned* flags[ArGroup::kPkMax];
    long long cut[ArGroup::kPkMax + 1];
};
// err[0] the seq given up on (or stuck past the bound), err[1] (1 + kind) << 28 | peer << 24 | tile, err[2] the value
__device__ __forceinline__ void wait_pk(const unsigned* flag, unsigned seq, unsigned* err, bool hard, unsigned tag) {
    for (unsigned w = 0; (int) (*(volatile const unsigned*) flag - seq) < 0; ++w)
        if (w == (1u << 24)) {
            err[1] = tag;
            err[2] = *(volatile const unsigned*) flag;
            __threadfence_system();
            *(volatile unsigned*) err = seq;
            if (hard) return;
        }
}
template <bool F16>
__device__ __forceinline__ pk_u4 pk_pack(const float4* x4, long long u) {
    pk_u4 r;
    if constexpr (F16) {
        const uint2 a = to_h4(sat4(x4[2 * u])), b = to_h4(sat4(x4[2 * u + 1]));
        r.x = a.x;
        r.y = a.y;
        r.z = b.x;
        r.w = b.y;
    } else {
        const float4 v = x4[u];
        r.x = __float_as_uint(v.x);
        r.y = __float_as_uint(v.y);
        r.z = __float_as_uint(v.z);
        r.w = __float_as_uint(v.w);
    }
    return r;
}
template <bool F16>
__device__ __forceinline__ void pk_vals(pk_u4 w, float* f) {
    if constexpr (F16) {
        const float4 a = from_h4(make_uint2(w.x, w.y)), b = from_h4(make_uint2(w.z, w.w));
        f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
    } else {
        f[0] = __uint_as_float(w.x), f[1] = __uint_as_float(w.y), f[2] = __uint_as_float(w.z), f[3] = __uint_as_float(w.w);
    }
}
template <bool F16>
__device__ __forceinline__ void pk_put(float4* x4, long long u, pk_u4 w) {
    float f[8];
    pk_vals<F16>(w, f);
    if constexpr (F16) {
        x4[2 * u] = make_float4(f[0], f[1], f[2], f[3]);
        x4[2 * u + 1] = make_float4(f[4], f[5], f[6], f[7]);
    } else x4[u] = make_float4(f[0], f[1], f[2], f[3]);
}
template <bool F16>
__device__ __forceinline__ pk_u4 pk_round(const float* v) {   // (the sum as every member takes it)
    pk_u4 r;
    if constexpr (F16) {
        const uint2 a = to_h4(sat4(make_float4(v[0], v[1], v[2], v[3]))), b = to_h4(sat4(make_float4(v[4], v[5], v[6], v[7])));
        r.x = a.x;
        r.y = a.y;
        r.z = b.x;
        r.w = b.y;
    } else {
        r.x = __float_as_uint(v[0]);
        r.y = __float_as_uint(v[1]);
        r.z = __float_as_uint(v[2]);
        r.w = __float_as_uint(v[3]);
    }
    return r;
}
__device__ __forceinline__ long long pk_min(long long a, long long b) { return a < b ? a : b; }
__device__ __forceinline__ pk_u4 pk_ld(const pk_u4* p) { return *(const volatile pk_u4*) p; }
__device__ __forceinline__ void pk_st(pk_u4* p, pk_u4 v) { __builtin_nontemporal_store(v, p); }

// DS4_AR_P2PK_ORDER (PM, default 1): the order of the sends of steps 1 and 2b.  PM 1, peer-major: every tile to peer
// mi + 1 first, then every tile to mi + 2, ... - so at a time a card stores into one peer's VRAM.  PM 0, tile-major (as first
// built): tile t to every peer, then tile t + grid.  the test machine (3x 7900 XTX, tools/p2pk_bw): one card storing into two peers'
// uncached VRAM at once moves 2-4 GB/s against 20-27 GB/s into one (fan-out vs ring), whatever the fences, flags, store
// width or block count - that, not the fences or the loads, made P2PK slow.  The flags, slots and waits are the same in both
// orders (step 2b of a tile to p waits only for p's free flag, the one that guards p's slot), so the same sums, bit for bit
template <bool F16, int PM = 1>
__global__ void __launch_bounds__(kThreads) ar_p2pk_kernel(float* x, const ArPk P, int nm, int mi, unsigned seq,
                                                          long long cap, int maxt, int tu, unsigned* err, int hard) {
    constexpr int K = F16 ? 8 : 4;
    float4* x4 = (float4*) x;
    const int par = (int) (seq & 1u), b0 = (int) blockIdx.x, nb = (int) gridDim.x, th = (int) threadIdx.x;
    const int nth = (int) blockDim.x;
    auto nt = [&](int k) { return (int) ((P.cut[k + 1] - P.cut[k] + tu - 1) / tu); };
    int ntm = 0;
    for (int k = 0; k < nm; ++k) ntm = max(ntm, nt(k));
    auto slot = [&](int owner, int sender) {   // (owner's slot of sender, this parity: indexed by unit - chunk start)
        return P.buf[owner] + (long long) (par * (nm - 1) + (sender < owner ? sender : sender - 1)) * cap;
    };
    auto flag = [&](int owner, int kind, int sender, int t) {
        return P.flags[owner] + ((((size_t) par * 3 + kind) * nm + sender) * maxt + t) * 8;
    };
    const bool peer = th < nm && th != mi;
    const bool hd = hard != 0;
    auto tag = [&](int kind, int t) { return (unsigned) ((kind + 1) << 28 | th << 24 | (t & 0xffffff)); };
    // 1. the reduce-scatter's sends
    if constexpr (PM == 1) {
        for (int j = 1; j < nm; ++j) {
            const int p = (mi + j) % nm;
            pk_u4* d = slot(p, mi);
            const long long c0 = P.cut[p];
            for (int t = b0; t < ntm; t += nb) {
                const long long lo = P.cut[p] + (long long) t * tu, hi = pk_min(P.cut[p + 1], lo + tu);
                for (long long u = lo + th; u < hi; u += nth) pk_st(d + (u - c0), pk_pack<F16>(x4, u));
                __threadfence_system();
                __syncthreads();
                if (th == 0) *(volatile unsigned*) flag(p, 0, mi, t) = seq;
            }
        }
    } else
    for (int t = b0; t < ntm; t += nb) {
        for (int j = 1; j < nm; ++j) {
            const int p = (mi + j) % nm;
            const long long lo = P.cut[p] + (long long) t * tu, hi = pk_min(P.cut[p + 1], lo + tu);
            pk_u4* d = slot(p, mi);
            const long long c0 = P.cut[p];
            for (long long u = lo + th; u < hi; u += nth) pk_st(d + (u - c0), pk_pack<F16>(x4, u));
        }
        __threadfence_system();
        __syncthreads();
        if (peer) *(volatile unsigned*) flag(th, 0, mi, t) = seq;
    }
    // 2a. the sums of this member's chunk
    const long long m0 = P.cut[mi], m1 = P.cut[mi + 1];
    for (int t = b0; t < ntm; t += nb) {
        const long long lo = m0 + (long long) t * tu, hi = pk_min(m1, lo + tu);
        if (lo < hi) {
            if (peer) wait_pk(flag(mi, 0, th, t), seq, err, hd, tag(0, t));
            __syncthreads();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            for (long long u = lo + th; u < hi; u += nth) {
                float v[K];
                for (int m = 0; m < nm; ++m) {
                    const pk_u4 w = m == mi ? pk_pack<F16>(x4, u) : pk_ld(slot(mi, m) + (u - m0));
                    float f[K];
                    pk_vals<F16>(w, f);
                    if (m == 0) {
#pragma unroll
                        for (int i = 0; i < K; ++i) v[i] = f[i];
                    } else {
#pragma unroll
                        for (int i = 0; i < K; ++i) v[i] = __fadd_rn(v[i], f[i]);
                    }
                }
                pk_put<F16>(x4, u, pk_round<F16>(v));
            }
        }
        __syncthreads();   // (every load of the slots returned: its value is in x)
        if (peer) *(volatile unsigned*) flag(th, 1, mi, t) = seq;
    }
    // 2b. the all-gather's sends
    if constexpr (PM == 1) {
        for (int j = 1; j < nm; ++j) {
            const int p = (mi + j) % nm;
            pk_u4* d = slot(p, mi);
            for (int t = b0; t < nt(mi); t += nb) {
                const long long lo = m0 + (long long) t * tu, hi = pk_min(m1, lo + tu);
                if (th == p) wait_pk(flag(mi, 1, p, t), seq, err, hd, tag(1, t));
                __syncthreads();
                for (long long u = lo + th; u < hi; u += nth) pk_st(d + (u - m0), pk_pack<F16>(x4, u));   // (x: the rounded sum)
                __threadfence_system();
                __syncthreads();
                if (th == 0) *(volatile unsigned*) flag(p, 2, mi, t) = seq;
            }
        }
    } else
    for (int t = b0; t < nt(mi); t += nb) {
        const long long lo = m0 + (long long) t * tu, hi = pk_min(m1, lo + tu);
        if (peer) wait_pk(flag(mi, 1, th, t), seq, err, hd, tag(1, t));
        __syncthreads();
        for (int j = 1; j < nm; ++j) {
            const int p = (mi + j) % nm;
            pk_u4* d = slot(p, mi);
            for (long long u = lo + th; u < hi; u += nth) pk_st(d + (u - m0), pk_pack<F16>(x4, u));   // (x: the rounded sum)
        }
        __threadfence_system();
        __syncthreads();
        if (peer) *(volatile unsigned*) flag(th, 2, mi, t) = seq;
    }
    // 3. the other chunks' sums
    for (int t = b0; t < ntm; t += nb) {
        if (peer && t < nt(th)) wait_pk(flag(mi, 2, th, t), seq, err, hd, tag(2, t));
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        for (int j = 1; j < nm; ++j) {
            const int p = (mi + j) % nm;
            const long long lo = P.cut[p] + (long long) t * tu, hi = pk_min(P.cut[p + 1], lo + tu);
            const pk_u4* s = slot(mi, p);
            const long long c0 = P.cut[p];
            for (long long u = lo + th; u < hi; u += nth) pk_put<F16>(x4, u, pk_ld(s + (u - c0)));
        }
    }
}

__device__ __forceinline__ uint32_t pk_hash(uint64_t i, uint32_t seed, int m) {
    uint32_t h = (uint32_t) i * 2654435761u ^ (uint32_t) (i >> 32) * 40503u ^ (seed + (uint32_t) m * 0x9e3779b9u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
// (|v| from about 2^-44 to 2^17: FP16's subnormals and zeros at one end, its saturation at the other)
__global__ void p2pk_fill_kernel(float* x, float* y, long long n, uint32_t seed, int m) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const uint32_t h = pk_hash((uint64_t) i, seed, m), h2 = pk_hash((uint64_t) i, seed ^ 0x5bd1e995u, m);
        const float v = ldexpf((float) (int) (h % 2000003u) - 1000001.0f, (int) (h2 % 42u) - 44);
        x[i] = v;
        y[i] = v;
    }
}
__global__ void p2pk_cmp_kernel(const float* a, const float* b, long long n, unsigned* out) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        if (__float_as_uint(a[i]) != __float_as_uint(b[i])) {
            atomicAdd(out, 1u);
            atomicMin(out + 1, (unsigned) i);
        }
}
#endif


// a thread a (token, value): every stream's value read before any is written
__global__ void hc_post_inplace_kernel(const float* __restrict__ y, float* h, const float* __restrict__ post,
                                       const float* __restrict__ comb, int dim, int hc) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float* r = h + (size_t) t * hc * dim;
    float rv[8];
    for (int j = 0; j < hc; ++j) rv[j] = r[(size_t) j * dim + e];
    const float yv = y[(size_t) t * dim + e];
    for (int k = 0; k < hc; ++k) {
        float a = post[(size_t) t * 8 + k] * yv;
        for (int j = 0; j < hc; ++j) a += comb[(size_t) t * 64 + j * hc + k] * rv[j];
        r[(size_t) k * dim + e] = a;
    }
}

__global__ void __launch_bounds__(kThreads) hc_mix_scale_kernel(const float* __restrict__ h, int hcdim, int n_mix,
                                                                float eps, float* mixes) {
    const int t = blockIdx.x;
    const float* x = h + (size_t) t * hcdim;
    float ss = 0.0f;
    for (int k = threadIdx.x * 4; k < hcdim; k += kThreads * 4) {
        const float4 hv = *(const float4*) (x + k);
        ss += hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w;
    }
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) hcdim + eps);
    if (threadIdx.x < n_mix) mixes[(size_t) t * 32 + threadIdx.x] *= inv;
}

__global__ void engram_dequant_tok_kernel(const uint8_t* __restrict__ src, size_t per, int cols, int hd,
                                          float* __restrict__ out, int fmt) {
    const int r = blockIdx.x, t = r / cols, c = r - t * cols;
    const uint8_t* rows = src + (size_t) t * per + (size_t) c * hd;
    const uint8_t* scales = src + (size_t) t * per + (size_t) cols * hd + (size_t) c * (hd / 32);
    if (fmt == 1) {   // (ggml Q6_K superblocks at the row's start)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = q6k_value(rows + (size_t) (e / 256) * kQ6kBytes, e % 256);
        return;
    }
    if (fmt == 2) {   // (ggml IQ4_NL blocks at the row's start, their scales inside)
        for (int e = threadIdx.x; e < hd; e += blockDim.x)
            out[(size_t) r * hd + e] = iq4nl_value(rows + (size_t) (e / 32) * 18, e % 32);
        return;
    }
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
        const float hv = e4m3_value(rows[e]);
        const uint8_t sb = scales[e / 32];
        const float s = sb == 0xffu ? __int_as_float(0x7fc00000) : (sb == 0 ? 5.877471754111438e-39f
                                                                            : __uint_as_float((uint32_t) sb << 23));
        out[(size_t) r * hd + e] = hv * s;
    }
}

// a block a virtual slot: its picks counted (sort pass 1), or (fill) placed in token order by a block-wide scan
template <bool FILL>
__global__ void __launch_bounds__(kThreads) moe_sort_kernel(const int* __restrict__ idx, int T, int k,
                                                            const int* __restrict__ vs, int* bounds, int* row_tok,
                                                            int* pos) {
    const int slot = blockIdx.x;
    __shared__ int warp_n[kThreads / 32];
    __shared__ int base_s;
    if (threadIdx.x == 0) base_s = FILL ? bounds[slot] : 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int n = T * k;
    for (int q0 = 0; q0 < n; q0 += kThreads) {
        const int q = q0 + threadIdx.x;
        const int t = q / k, i = q - t * k;
        const bool hit = q < n && vs[idx[(size_t) t * kPicks + i]] == slot;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) warp_n[wp] = __popc(m);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wp) before += warp_n[w];
            total += warp_n[w];
        }
        if (FILL && hit) {
            const int r = base_s + before + __popc(m & ((1u << lane) - 1u));
            row_tok[r] = t;
            pos[(size_t) t * kPicks + i] = r;
        }
        __syncthreads();
        if (threadIdx.x == 0) base_s += total;
        __syncthreads();
    }
    if (!FILL && threadIdx.x == 0) bounds[slot + 1] = base_s;   // (the counts; the scan makes them bounds)
}

__global__ void bounds_scan_kernel(int* bounds, int n_vs) {   // one thread: bounds[0] = 0, bounds[s + 1] += bounds[s]
    if (threadIdx.x != 0) return;
    bounds[0] = 0;
    for (int s = 0; s < n_vs; ++s) bounds[s + 1] += bounds[s];
}

__global__ void moe_combine_kernel(float* acc, const float* __restrict__ y, const int* __restrict__ pos,
                                   const float* __restrict__ w, int k, int r0, int r1, int dim) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= dim) return;
    float a = acc[(size_t) t * dim + e];
    bool any = false;
    for (int i = 0; i < k; ++i) {
        const int r = pos[(size_t) t * kPicks + i];
        if (r < r0 || r >= r1) continue;
        a = __fmaf_rn(w[(size_t) t * kPicks + i], y[(size_t) (r - r0) * dim + e], a);
        any = true;
    }
    if (any) acc[(size_t) t * dim + e] = a;
}

// ---------------------------------------------------------------- DS4_AR_POST (see allreduce_post)
// hc_post_kernel's per-element code for the four elements 4 q .. 4 q + 3 of the summed row (y = v), every stream k: the
// same statements in the same order (a = post * y, then a += comb * r for j = 0 .. hc - 1), so the same roundings
__device__ __forceinline__ void hc_post_quad(const float4 v, int q, const HcPost& hp) {
    const int dim = hp.dim, hc = hp.hc;
    const int i0 = q * 4, t = i0 / dim, e0 = i0 - t * dim;   // (dim % 4 == 0: the four are one row's)
    const float* r = hp.res + (size_t) t * hc * dim;
    const float yv[4] = {v.x, v.y, v.z, v.w};
    for (int k = 0; k < hc; ++k) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int e = e0 + c;
            float a = hp.post[(size_t) t * 8 + k] * yv[c];
            for (int j = 0; j < hc; ++j) a += hp.comb[(size_t) t * 64 + j * hc + k] * r[(size_t) j * dim + e];
            hp.out[(size_t) t * hc * dim + (size_t) k * dim + e] = a;
        }
    }
}

// ar_sum_kernel with hc_post on the sums (bit 1)
template <bool W>
__global__ void __launch_bounds__(kThreads) ar_sum_post_kernel(float* x, int n4, const float* peers, size_t stride,
                                                              const unsigned* flags, int nm, int mi, int slot, int slots,
                                                              unsigned seq, const HcPost hp) {
    if (threadIdx.x == 0)
        for (int m = 0; m < nm; ++m)
            if (m != mi) wait_at_least(flags + ((size_t) m * slots + slot) * 32, seq);
    __syncthreads();
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? ((const float4*) x)[q]
                                     : W ? ld_wide((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q)
                                         : ld_fresh((const float4*) (peers + ((size_t) m * slots + slot) * stride) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        ((float4*) x)[q] = v;
        hc_post_quad(v, q, hp);
    }
}

// bit 2 (and 1 with post): the put, the block's own flag, the peers' flags of the same block, the sum of the block's range
// in member order (ar_sum_kernel's chain), hc_post on it (post).  The block ranges and the flags' places are
// ar_rs_kernel's first round's (rflag), so every member cuts the row the same way (the same grid: the same n)
template <bool W>
__global__ void __launch_bounds__(kThreads) ar_put_sum_post_kernel(float* x, int n4, float* rows, size_t stride,
                                                                  unsigned* bflags, int nm, int mi, int slot, int slots,
                                                                  unsigned seq, const HcPost hp, int post) {
    float4* x4 = (float4*) x;
    const int per = (n4 + (int) gridDim.x - 1) / (int) gridDim.x;
    const int lo = blockIdx.x * per, hi = min(n4, lo + per);
    auto rflag = [&](int m) { return bflags + (((size_t) m * slots + slot) * kArBlocks + blockIdx.x) * 8; };
    auto row = [&](int m) { return (float4*) (rows + ((size_t) m * slots + slot) * stride); };
    if (lo < hi) {
        float4* mine = row(mi);
        for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) mine[q] = x4[q];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile unsigned*) rflag(mi) = seq;
    if ((int) threadIdx.x < nm && (int) threadIdx.x != mi) wait_at_least(rflag(threadIdx.x), seq);
    __syncthreads();
    for (int q = lo + threadIdx.x; q < hi; q += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int m = 0; m < nm; ++m) {
            const float4 p = m == mi ? x4[q] : W ? ld_wide(row(m) + q) : ld_fresh(row(m) + q);
            if (m == 0) v = p;
            else {
                v.x = __fadd_rn(v.x, p.x);
                v.y = __fadd_rn(v.y, p.y);
                v.z = __fadd_rn(v.z, p.z);
                v.w = __fadd_rn(v.w, p.w);
            }
        }
        x4[q] = v;
        if (post) hc_post_quad(v, q, hp);
    }
}

// ---------------------------------------------------------------- DS4_ATTN_MERGE (see finish_multi)
// finish_kernel's body for row `row` (its blockIdx.x): the same code, blockIdx.x replaced by row
template <bool kQ8>
__device__ __forceinline__ void finish_row(float* x, int len, long long stride, const Finish& f, int row) {
    constexpr bool kPack = false;
    float* v = x + (long long) row * stride;
    float r[kFinMax];
    const int nj = (len + kThreads - 1) / kThreads;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        r[j] = e < len ? v[e] : 0.0f;
        ss += r[j] * r[j];
    }
    if (f.w != nullptr) {
        const float tot = block_sum(ss);
        const float inv = rsqrtf(tot / (float) len + f.eps);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            if (e < len) r[j] = r[j] * inv * f.w[e];
        }
    }
    if (f.rd > 0) {
        const int r0 = len - f.rd;
        const int pos = f.pos + (int) (row / f.vpp) * f.pos_step;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            const float other = __shfl_xor_sync(0xffffffffu, r[j], 1);
            if (e >= r0 && e < len) {
                const int i = (e - r0) >> 1;
                const float ang = (float) pos * f.freqs[i];
                const float c = cosf(ang);
                const float sn = f.inverse ? -sinf(ang) : sinf(ang);
                if (((e - r0) & 1) == 0) r[j] = r[j] * c - other * sn;   // re = a c - b s
                else r[j] = other * sn + r[j] * c;                          // im = a s + b c
            }
        }
    }
    if (f.quant != 0 && !kPack) {
        const int g = f.quant == 2 ? 16 : 32;
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float a = e < len ? fabsf(r[j]) : 0.0f;
            for (int o = g / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            if (e >= len) continue;
            if (f.quant == 1) {
                const float s = pow2_ceil(fmaxf(a, 1e-4f) * (1.0f / 448.0f));
                r[j] = fp8_rt(fminf(fmaxf(r[j] / s, -448.0f), 448.0f)) * s;
            } else if (f.quant == 2) {
                const float s = fp8_rt(fmaxf(a, 6.0f * 0.001953125f) / 6.0f);
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            } else {
                const float s = pow2_ceil(fmaxf(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
                r[j] = e2m1_rt(fminf(fmaxf(r[j] / s, -6.0f), 6.0f)) * s;
            }
        }
    }
    uint16_t* d = f.dst != nullptr ? f.dst + (long long) row * f.dst_stride : nullptr;
#pragma unroll
    for (int j = 0; j < kFinMax; ++j) {
        if (j >= nj) break;
        const int e = threadIdx.x + j * kThreads;
        if (e >= len) continue;
        if (d != nullptr) d[e] = f2bf16(r[j]);
        else if (!kPack) v[e] = r[j];
    }
    if (kQ8 && !kPack) {
        int8_t* q = f.q8 + (long long) row * len;
        float* qd = f.q8d + (long long) row * (len / 32);
#pragma unroll
        for (int j = 0; j < kFinMax; ++j) {
            if (j >= nj) break;
            const int e = threadIdx.x + j * kThreads;
            float am = e < len ? fabsf(r[j]) : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            if (e >= len) continue;
            const float sc = am / 127.0f;
            q[e] = (int8_t) (am > 0.0f ? __float2int_rn(r[j] / sc) : 0);
            if ((threadIdx.x & 31) == 0) qd[e / 32] = sc;
        }
    }
}
// ring_to_ext_kernel's body for row r (its blockIdx.x)
__device__ __forceinline__ void ring_to_ext_row(const uint8_t* __restrict__ ring, uint16_t* __restrict__ ext, long long p0,
                                                int n, int d, int r) {
    const long long p = p0 - n + r;
    const uint8_t* row = ring + (size_t) (p >= 0 ? p % n : 0) * kv_row_bytes(kKvFp8, d);
    for (int e = threadIdx.x; e < d; e += blockDim.x)
        ext[(size_t) r * d + e] = p >= 0 ? f2bf16(e4m3_value(row[e]) * ue8m0_val(row[d + e / 32])) : (uint16_t) 0;
}
// ext_to_ring_kernel's body for row t (its blockIdx.x)
__device__ __forceinline__ void ext_to_ring_row(const uint16_t* __restrict__ ext, uint8_t* __restrict__ ring, long long p0,
                                                int n, int T, int d, uint8_t* __restrict__ backup, int t) {
    if (T - t > n) return;   // overwritten by a later row of the window anyway
    const size_t rb = kv_row_bytes(kKvFp8, d);
    uint8_t* row = ring + (size_t) ((p0 + t) % n) * rb;
    if (backup != nullptr) {
        for (size_t i = threadIdx.x; i < rb; i += blockDim.x) backup[(size_t) t * rb + i] = row[i];
        __syncthreads();
    }
    const uint16_t* src = ext + (size_t) (n + t) * d;
    for (int e0 = 0; e0 < d; e0 += blockDim.x) {   // (d a multiple of 32: a warp's 32 values one group)
        const int e = e0 + (int) threadIdx.x;
        const float v = e < d ? bf16f(src[e]) : 0.0f;
        uint32_t a = __float_as_uint(v) & 0x7fffffffu;   // (the magnitudes' order as bits)
        for (int o = 16; o > 0; o >>= 1) a = max(a, (uint32_t) __shfl_xor_sync(0xffffffffu, a, o));
        const int k = a == 0u ? 0 : (int) (a >> 23) - 135 + ((a & 0x7fffffu) > 0x600000u ? 1 : 0);
        if (e >= d) continue;
        row[e] = (uint8_t) e4m3_code(v, k);
        if ((e & 31) == 0) row[d + e / 32] = (uint8_t) (k + 127);
    }
}
struct FinMulti {
    FinOp op[kFinOps];
    int rows[kFinOps];
    int n;
};
__device__ __forceinline__ void fin_op_run(const FinOp& o, int b) {
    if (o.kind == 0) {
        if (o.f.q8 != nullptr) finish_row<true>(o.x, o.len, o.stride, o.f, b);
        else finish_row<false>(o.x, o.len, o.stride, o.f, b);
    } else if (o.kind == 1) {
        ring_to_ext_row(o.ring_r, o.ext_w, o.p0, o.n, o.d, b);
    } else {
        ext_to_ring_row(o.ext_r, o.ring_w, o.p0, o.n, o.T, o.d, o.backup, b);
    }
}
// block b: op i's row b - (the rows of ops 0 .. i-1); the op a block-uniform branch (each op's ops are its kernel's, with
// its own __syncthreads / shuffles inside the block)
__global__ void __launch_bounds__(kThreads) fin_multi_kernel(const FinMulti m) {
    int b = blockIdx.x;
    if (b < m.rows[0]) {
        fin_op_run(m.op[0], b);
        return;
    }
    b -= m.rows[0];
    if (m.n > 1 && b < m.rows[1]) {
        fin_op_run(m.op[1], b);
        return;
    }
    b -= m.n > 1 ? m.rows[1] : 0;
    if (m.n > 2 && b < m.rows[2]) fin_op_run(m.op[2], b);
}
#if defined(STRATA_USE_HIP) && !defined(STRATA_METAL) && (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__))
// DS4_ATTN_MERGE=3 (see finish_multi_sel): unpack_sel_kernel's body for its block (bx, t) - the same code, blockIdx.x /
// blockIdx.y replaced by bx / t and its arguments read from u
__device__ __forceinline__ void unpack_sel_blk(const SelOp& u, int bx, int t) {
    const int i = bx * kStage + (int) (threadIdx.x >> 4), part = threadIdx.x & 15;
    int n_sel;
    if (u.a.gen) {
        const long long pa = u.a.p0 + u.a.t0 + t;
        n_sel = u.a.R > 0 ? (int) min((long long) u.a.kcap, (pa + 1) / u.a.R) : 0;
    } else {
        n_sel = u.a.nsel[t];
    }
    if (i >= n_sel) return;
    const size_t r = (size_t) t * u.cap + i;
    unpack32(u.ckv + (size_t) u.sel[(size_t) t * u.sel_stride + i] * kv_row_bytes(kKvFp4, 512), part,
             (uint4*) (u.out + r * 512 + part * 32));
}
// fin_multi_kernel's blocks for m's ops (its code), then nx * u.T blocks of u's unpack (block b: x b % nx, query b / nx)
__global__ void __launch_bounds__(kThreads) fin_multi_sel_kernel(const FinMulti m, const SelOp u, int nx) {
    int b = blockIdx.x;
    if (b < m.rows[0]) {
        fin_op_run(m.op[0], b);
        return;
    }
    b -= m.rows[0];
    if (m.n > 1) {
        if (b < m.rows[1]) {
            fin_op_run(m.op[1], b);
            return;
        }
        b -= m.rows[1];
    }
    if (m.n > 2) {
        if (b < m.rows[2]) {
            fin_op_run(m.op[2], b);
            return;
        }
        b -= m.rows[2];
    }
    unpack_sel_blk(u, b % nx, b / nx);
}
#endif

int cdiv(int a, int b) { return (a + b - 1) / b; }

// DS4_XK_QA (HIP, gfx1100 only: instantiated from xk_mv_q5k_q8's guarded launch): ds4_iq's mv_kernel<kQ5_K, 1> - its
// per-row code copied as it is (decode, sub_dot with the MIN term, the lane's sub-blocks kb = lane + 32 i, warp_sum, the
// alpha / bias store), the same grid (8 rows a block, a block a 8-row tile, blockIdx.y the job) - then, by the last of the
// four blocks of each 32-row group to finish (a device counter a group, left zero), the group's stored values rounded to
// int8 with q8_quant's ops: wo_b's input rounded where wo_a makes it (the q8_quant_multi launch gone).  The counters are
// one set a device: one such launch in flight a device (the decode's stream).
#if defined(STRATA_USE_HIP)
namespace xkq {
__device__ __forceinline__ uint32_t ld4(const uint8_t* p) { return *(const uint32_t*) p; }
__device__ __forceinline__ float h2f(const uint8_t* p) { return __half2float(__ushort_as_half(*(const uint16_t*) p)); }
__device__ __forceinline__ void q4k_scale(const uint8_t* q, int j, int& d, int& m) {   // (ds4_iq.cu's, ggml's get_scale_min_k4)
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
struct Sub {   // (ds4_iq.cu's)
    int q[8];
    float d, m0, m1;
    int s0, s1;
};
// (ds4_iq.cu's decode<kQ5_K>: decode_b's Q5_K branch on block kb / 8, sub-block kb % 8)
__device__ __forceinline__ __attribute__((unused)) void decode_q5k(const uint8_t* __restrict__ row, int kb, Sub& s) {
    const uint8_t* b = row + (size_t) (kb >> 3) * 176;
    const int ib = kb & 7;
    int sc, m;
    q4k_scale(b + 4, ib, sc, m);
    const uint8_t* qs = b + 48 + 32 * (ib >> 1);
    const uint8_t* qh = b + 16;
    const int sh = 4 * (ib & 1);
#pragma unroll
    for (int t = 0; t < 8; ++t)
        s.q[t] = (int) (((ld4(qs + 4 * t) >> sh) & 0x0f0f0f0fu) | (((ld4(qh + 4 * t) >> ib) & 0x01010101u) << 4));
    s.s0 = s.s1 = sc;
    s.d = h2f(b);
    s.m0 = s.m1 = h2f(b + 2) * (float) m;
}
// (ds4_iq.cu's sub_dot<kQ5_K>: Ty<kQ5_K>::MIN = 1)
__device__ __forceinline__ __attribute__((unused)) float sub_dot_q5k(const Sub& s, const int4& xa, const int4& xb) {
    int lo = __dp4a(s.q[0], xa.x, 0);
    lo = __dp4a(s.q[1], xa.y, lo);
    lo = __dp4a(s.q[2], xa.z, lo);
    lo = __dp4a(s.q[3], xa.w, lo);
    int hi = __dp4a(s.q[4], xb.x, 0);
    hi = __dp4a(s.q[5], xb.y, hi);
    hi = __dp4a(s.q[6], xb.z, hi);
    hi = __dp4a(s.q[7], xb.w, hi);
    int sl = __dp4a(0x01010101, xa.x, 0);
    sl = __dp4a(0x01010101, xa.y, sl);
    sl = __dp4a(0x01010101, xa.z, sl);
    sl = __dp4a(0x01010101, xa.w, sl);
    int sh = __dp4a(0x01010101, xb.x, 0);
    sh = __dp4a(0x01010101, xb.y, sh);
    sh = __dp4a(0x01010101, xb.z, sh);
    sh = __dp4a(0x01010101, xb.w, sh);
    // (xk-20261111T150445Z: on HIP __fmul_rn / __fadd_rn are plain * / +, and -ffp-contract=fast fused the min term the other way
    //  round from ds4_iq's mv_kernel<kQ5_K, 1> - fma(m, sh, rn(m sl)) against its fma(m, sl, rn(m sh)): y off by a few ulp.
    //  Contraction is off here and both of the old kernel's fusions are written out, as its ISA has them (v_mul m sh, v_fmac
    //  + m sl; v_fma_mix d ints - that): the same roundings, the same bits.)
    {   // (the pragma must open a compound statement)
#pragma clang fp contract(off)
        const float mt = __builtin_fmaf(s.m0, (float) sl, s.m1 * (float) sh);
        return __builtin_fmaf(s.d, (float) (s.s0 * lo + s.s1 * hi), -mt);
    }
}
constexpr int kMaxQaJobs = 4, kQaCnt = 256;
struct QaJobs {
    XkQaJob j[kMaxQaJobs];
    int gmax;   // 32-row groups a job at most (the counters: job y's at y gmax)
};
template <int N>
__device__ unsigned qa_cnt[N];   // (zero at load; each launch leaves its counters zero)
}  // namespace xkq
template <int NW>
__global__ void __launch_bounds__(NW * 32) mv_q5k_q8_xk_kernel(const xkq::QaJobs J) {
    __shared__ int last;
    const XkQaJob& b = J.j[blockIdx.y];
    const int row = blockIdx.x * NW + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row < b.n) {   // (mv_kernel<kQ5_K, 1>'s row)
        const uint8_t* w = b.w + (size_t) row * b.ld;
        const int nsb = b.k / 32;
        float acc = 0.0f;
        for (int kb = lane; kb < nsb; kb += 32) {
            xkq::Sub s;
            xkq::decode_q5k(w, kb, s);
            const int4* xp = (const int4*) (b.xq + (size_t) kb * 32);
            acc = fmaf(b.xd[kb], xkq::sub_dot_q5k(s, xp[0], xp[1]), acc);
        }
        const float v = warp_sum(acc);
        if (lane == 0) b.y[row] = b.alpha * v + (b.bias != nullptr ? b.bias[row] : 0.0f);
    }
    __threadfence();
    __syncthreads();
    const int grp = (blockIdx.x * NW) / 32;
    unsigned* cnt = &xkq::qa_cnt<xkq::kQaCnt>[blockIdx.y * J.gmax + grp];
    if (threadIdx.x == 0) last = atomicAdd(cnt, 1u) == (unsigned) (32 / NW - 1);
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const int r = grp * 32 + lane;   // (n % 32 == 0: the group's rows all in)
    const float v = __uint_as_float(__hip_atomic_load((const unsigned*) (b.y + r), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    float am = fabsf(v);   // (q8_quant_kernel's ops on the stored value)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    b.q[r] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) {
        b.qd[grp] = sc;
        atomicExch(cnt, 0u);
    }
}
#endif

}  // namespace

namespace {

// ---- the split-KV attention's scratch: a device allocation a (device, stream), made on first use and kept
constexpr int kSplitG = 4, kSplitR = 40, kSplitCap = 24, kSplitMaxT = 5;
bool split_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("DS4_ATTN_SPLIT");
        return e != nullptr && std::atoi(e) != 0;   // opt-in: no gain measured on 4x R9700 (21.8 vs 21.9 tok/s)
    }();
    return on;
}
// DS4_FAST_ATTN: the chunked prompt's packed-KV attention on the matrix cores (sparse_attn_wmma_kernel).  The
// QK/PV sums are the matrix cores' order, so the outputs differ from the scalar kernel's in the last places (the owner
// approved the trade for the prefill; the decode path stays on the scalar kernels).
// 0.2.7: ON by default on gfx1200 / gfx1201 (RDNA4: the R9700, RX 9000), where it was measured and gated (KL 0.0003 against the
// 0.006 gate, ids identical); gfx11 and gfx1151 (Strix Halo) are untested and keep the scalar kernels.  DS4_FAST_ATTN=0 is the
// off switch (exactly the 0.2.6 arithmetic), DS4_FAST_ATTN=1 turns it on elsewhere.  Decided per device (the first call there).
// The Marlin stream measured it ON on 4x R9700 (gfx1201), 2.25 bpw, 8K code prompts: 1180 -> 1354 ppts (+14.7%), the same ids,
// KL 0.10833 against 0.10925 for the scalar path.  gfx1151 on Windows was the untested part; measured on Strix Halo 2026-10-28
// (32K prompt, 2.00 bpw): 101.6 -> 70.8 s (+43.6%), the sparse-attention stage 38.3 -> 7.0 s, the same greedy first token
// (27408,), so it joins the default there and stays opt-in on every other gfx11 part.
#if defined(STRATA_USE_HIP)
bool device_is_gfx12() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                         (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#if defined(_WIN32)
bool device_is_gfx1151win() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0;
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
}
#endif
#endif
// gfx1100 (RX 7900 XTX) and whether the engine runs tensor-parallel groups (set_fast_attn_tp, from the model's setup)
std::atomic<int> g_fast_attn_tp{0};
bool device_is_gfx1100() {
#if defined(STRATA_USE_HIP)
    static std::atomic<int> cache[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load();
    if (c == 0) {
        cudaDeviceProp prop{};
        const bool yes = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && std::strncmp(prop.gcnArchName, "gfx1100", 7) == 0 &&
                         (prop.gcnArchName[7] == 0 || prop.gcnArchName[7] == ':');
        c = yes ? 2 : 1;
        cache[dev].store(c);
    }
    return c == 2;
#else
    return false;
#endif
}
bool fast_attn_enabled() {
    static const int env = [] {
        const char* e = std::getenv("DS4_FAST_ATTN");
        return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
    }();
#if defined(STRATA_USE_HIP)
#if defined(_WIN32)
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), device_is_gfx1151win(), strata::ds4::defaults_026(),
                                          g_fast_attn_tp.load() != 0 && device_is_gfx1100());
#else
    return strata::ds4::fast_attn_default(env, device_is_gfx12(), false, strata::ds4::defaults_026(),
                                          g_fast_attn_tp.load() != 0 && device_is_gfx1100());
#endif
#else
    return strata::ds4::fast_attn_default(env, false, false, strata::ds4::defaults_026());
#endif
}
#if defined(STRATA_USE_HIP)
// (docs/dsv41/cards/PF-02.md): DS4_FAST_ATTN_MODE - which precision variant of sparse_attn_wmma_kernel the fast path runs (a
// name or its index below; unset: f32 on gfx1100 tp groups, f16 (the original) on the upstream default-on arches; f16 failed the
// chunked-path quality bar on gfx1100, top-1 -0.72 [-1.32, -0.14]).  Only read when the fast path is on (DS4_FAST_ATTN).
struct FaMode {
    const char* name;
    int qm, pm;
    bool qsc;
};
constexpr FaMode kFaModes[] = {
    {"f16", 0, 0, false},     // the original: Q, P as FP16
    {"f32", 2, 1, true},      // (a) Q, P as FP16 hi + lo pairs (~22-bit operands), F32 accumulate + softmax, Q per-head scaled
    {"bf16", 1, 0, false},    // (b) Q, K as BF16 (the BF16 WMMA), P as FP16
    {"qscale", 0, 0, true},   // (c) Q per-head power-of-two scaled before the FP16 cast
    {"pv32", 0, 2, false},    // (d) WMMA scores, scalar F32 P.V
    {"qk32", 3, 0, false},    // (d) scalar F32 scores, WMMA P.V
    {"qx2", 2, 0, true},      // Q as a hi + lo pair only (isolates Q's rounding)
    {"px2", 0, 1, false},     // P as a hi + lo pair only (isolates P's rounding)
    {"exact", 3, 2, false},   // both scalar F32 (the 16-head block, no matrix cores: a reference for the test)
};
constexpr int kFaModeCount = (int) (sizeof(kFaModes) / sizeof(kFaModes[0]));
constexpr int kFaModeDefault = 1;   // f32
int fa_mode_parse(const char* e) {
    if (e == nullptr || *e == 0) return kFaModeDefault;
    for (int i = 0; i < kFaModeCount; ++i)
        if (std::strcmp(e, kFaModes[i].name) == 0) return i;
    char* endp = nullptr;
    const long v = std::strtol(e, &endp, 10);
    if (endp != e && *endp == 0 && v >= 0 && v < kFaModeCount) return (int) v;
    std::fprintf(stderr, "ds4: DS4_FAST_ATTN_MODE=%s unknown - f32\n", e);
    return kFaModeDefault;
}
int fast_attn_mode() {
    // DS4_FAST_ATTN_MODE set: that mode everywhere.  Unset: f32 only where the fast path's default turned it on - gfx1100
    // in a tensor-parallel group (the arithmetic measured there, docs/dsv41/cards/PF-02.md); everywhere else (gfx12, gfx1151 on
    // Windows: the upstream default-on rule) the original f16 arithmetic those defaults were measured with.
    static const int env_m = [] {
        const char* e = std::getenv("DS4_FAST_ATTN_MODE");
        if (e == nullptr || *e == 0) return -1;
        const int r = fa_mode_parse(e);
        std::fprintf(stderr, "ds4: fast attention mode %s (QM %d, PM %d, QSC %d)\n", kFaModes[r].name, kFaModes[r].qm,
                     kFaModes[r].pm, (int) kFaModes[r].qsc);
        return r;
    }();
    if (env_m >= 0) return env_m;
    return g_fast_attn_tp.load() != 0 && device_is_gfx1100() ? kFaModeDefault : 0;   // (0: f16)
}
// DS4_FAST_ATTN_LAYERS=a-b,c,...: the layers (the model's layer index) the fast path runs on; unset: every layer.  The
// others keep the scalar kernel.  Layer -1 (a caller that does not say): on.
constexpr int kFaMaxLayers = 1024;
bool fast_attn_layer_on(int layer) {
    static const std::vector<bool>* mask = []() -> const std::vector<bool>* {
        const char* e = std::getenv("DS4_FAST_ATTN_LAYERS");
        if (e == nullptr || *e == 0) return nullptr;
        auto* m = new std::vector<bool>(kFaMaxLayers, false);
        const char* p = e;
        int n = 0;
        while (*p) {
            char* qe = nullptr;
            const long a = std::strtol(p, &qe, 10);
            if (qe == p) break;
            long b = a;
            p = qe;
            if (*p == '-') {
                b = std::strtol(p + 1, &qe, 10);
                if (qe == p + 1) break;
                p = qe;
            }
            for (long i = std::max(0L, a); i <= b && i < kFaMaxLayers; ++i) {
                if (!(*m)[(size_t) i]) ++n;
                (*m)[(size_t) i] = true;
            }
            if (*p == ',') ++p;
            else if (*p) break;
        }
        if (*p) std::fprintf(stderr, "ds4: DS4_FAST_ATTN_LAYERS=%s: parse stopped at \"%s\"\n", e, p);
        std::fprintf(stderr, "ds4: fast attention on %d layers (DS4_FAST_ATTN_LAYERS=%s)\n", n, e);
        return m;
    }();
    if (mask == nullptr || layer < 0) return true;
    return layer < kFaMaxLayers && (*mask)[(size_t) layer];
}
template <int QM, int PM, bool QSC>
void launch_fa(const float* q, const uint16_t* win, const uint8_t* ckv, const int* sel, int sel_stride, const AttnRows& a,
               const float* sink, int heads, int T, float scale, float* out, cudaStream_t s) {
    strata_ds4_launch(sparse_attn_wmma_kernel<QM, PM, QSC>, dim3(heads / kFaHeads, T), kThreads, 0, s, q, win, ckv, sel,
                      sel_stride, a, sink, heads, scale, out);
}
bool launch_fast_attn(int mode, const float* q, const uint16_t* win, const uint8_t* ckv, const int* sel, int sel_stride,
                      const AttnRows& a, const float* sink, int heads, int T, float scale, float* out, cudaStream_t s) {
    switch (mode) {
        case 0: launch_fa<0, 0, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 1: launch_fa<2, 1, true>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 2: launch_fa<1, 0, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 3: launch_fa<0, 0, true>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 4: launch_fa<0, 2, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 5: launch_fa<3, 0, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 6: launch_fa<2, 0, true>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 7: launch_fa<0, 1, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        case 8: launch_fa<3, 2, false>(q, win, ckv, sel, sel_stride, a, sink, heads, T, scale, out, s); return true;
        default: return false;
    }
}
static_assert(kFaModeCount == 9, "launch_fast_attn's cases follow kFaModes");
#endif
attnsplit::AttnSplitScratch split_scratch(cudaStream_t s, int heads) {
    static std::mutex mu;
    static std::map<std::pair<int, cudaStream_t>, attnsplit::AttnSplitScratch> m;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(mu);
    auto it = m.find({dev, s});
    if (it != m.end()) return it->second;
    attnsplit::AttnSplitScratch sc;
    sc.nsplit = kSplitCap;
    sc.max_t = kSplitMaxT;
    const size_t np = (size_t) kSplitMaxT * heads * kSplitCap;
    if (cudaMalloc((void**) &sc.part, np * 512 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &sc.ml, np * 2 * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        if (sc.part != nullptr) cudaFree(sc.part);
        sc = attnsplit::AttnSplitScratch{};
    }
    m[{dev, s}] = sc;
    return sc;
}
}  // namespace

#if defined(STRATA_USE_HIP)
// STRATA_DS4_DIAG - hipFuncGetAttributes of kernels of this TU (the one CUDA0 launches from first) on the current device
int diag_probe(int i, const char** name) {
    hipFuncAttributes a{};
    switch (i) {
        case 0: *name = "kernels embed_kernel"; return (int) hipFuncGetAttributes(&a, (const void*) embed_kernel);
        case 1: *name = "kernels hc_post_kernel"; return (int) hipFuncGetAttributes(&a, (const void*) hc_post_kernel);
        default: return -1;
    }
}
#endif

void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl,
                   unsigned vl, int na0, int na1) {
#if defined(STRATA_METAL)
    // selection only: the prefetch schedule (req) is published by the CUDA kernel, and is turned on separately here
    if (strata::dsv41::metal_host::route_publish(logits, bias, n, k, T, route_scale, vram, idx, w, plan_ptr,
            vl != 0 ? bias_vl : nullptr, vl)) {
        // no prefetcher on the Mac path, so publish_kernel has no consumer: the selection is the whole job here
        (void) req; (void) seq; (void) na0; (void) na1; (void) x; (void) dim;
        return;
    }
#endif
    strata_ds4_launch(route_publish_kernel, T, 512, 0, s, logits, bias, n, k, route_scale, vram, x, dim, idx, w, plan_ptr, req,
                                           vl != 0 ? bias_vl : nullptr, vl);
    if (req != nullptr) strata_ds4_launch(publish_kernel, 1, 1, 0, s, req, T, seq, na0, na1);
}
void route_ahead(const float* logits, const float* bias, int n, int m, int T, Req* req, int off, cudaStream_t s) {
#if defined(STRATA_METAL)
    // no prefetch schedule on the Mac path yet: the router publishes no requests, so there is nothing to look ahead for
    (void) logits; (void) bias; (void) n; (void) m; (void) T; (void) req; (void) off; (void) s;
    return;
#endif
    if (req != nullptr && m > 0 && T > 0 && n <= 1024 && off + m <= kAhead)
        strata_ds4_launch(route_ahead_kernel, T, 512, 0, s, logits, bias, n, m, req, off);
}
bool route_pair(const RoutePair& a, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) a; (void) s;
    return false;
#else
    if (a.req == nullptr || a.logits == nullptr || a.bias == nullptr || a.vram == nullptr || a.x == nullptr ||
        a.idx == nullptr || a.w == nullptr || a.plan_ptr == nullptr || a.a_logits == nullptr || a.a_bias == nullptr ||
        a.n < 1 || a.n > 1024 || a.k < 1 || a.k > kPicks || a.a_n < 1 || a.a_n > 1024 || a.m < 1 || a.off < 0 ||
        a.off + a.m > kAhead || a.dim < 1 || a.dim > kMaxDim || a.T < 1 || a.T > kWin)
        return false;
    strata_ds4_launch(route_pair_kernel, dim3(2u, (unsigned) a.T), 512, 0, s, a);
    if (cudaGetLastError() != cudaSuccess) return false;   // (a routing that did not go in is never published)
    strata_ds4_launch(publish_kernel, 1, 1, 0, s, a.req, a.T, a.seq, a.na0, a.na1);
    return cudaGetLastError() == cudaSuccess;
#endif
}
// xtxkern: the process's DS4_XK_* bits (set once by the model at load, before any launch; read at every launch)
static std::atomic<int> g_xk_bits{0};
void xk_set(int bits) { g_xk_bits.store(bits, std::memory_order_relaxed); }
int xk_bits() { return g_xk_bits.load(std::memory_order_relaxed); }
bool xk_mv_q5k_q8(const XkQaJob* jobs, int n, cudaStream_t s) {
#if defined(STRATA_USE_HIP) && (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__))   // (DS4_XK_QA: gfx1100 only)
    if (!(xk_bits() & kXkQa) || n < 1 || n > xkq::kMaxQaJobs) return false;
    xkq::QaJobs J{};
    int max_n = 0;
    for (int i = 0; i < n; ++i) {
        const XkQaJob& b = jobs[i];
        if (b.w == nullptr || b.xq == nullptr || b.xd == nullptr || b.y == nullptr || b.q == nullptr || b.qd == nullptr ||
            b.k < 256 || b.k % 256 != 0 || b.n < 32 || b.n % 32 != 0)
            return false;
        J.j[i] = b;
        max_n = std::max(max_n, b.n);
    }
    J.gmax = max_n / 32;
    if (J.gmax * n > xkq::kQaCnt) return false;
    constexpr int NW = kThreads / 32;   // (ds4_iq's kWarps: 8 rows a block)
    strata_ds4_launch(mv_q5k_q8_xk_kernel<NW>, dim3((unsigned) (max_n / NW), (unsigned) n), NW * 32, 0, s, J);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) jobs;
    (void) n;
    (void) s;
    return false;
#endif
}
void route_publish_lp(bool fast, bool pub, const float* logits, const float* bias, int n, int k, int T, float route_scale,
                      const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                      unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl,
                      unsigned vl, int na0, int na1, unsigned* pub_cnt) {
#if defined(STRATA_METAL)
    (void) fast; (void) pub; (void) pub_cnt;
    route_publish(logits, bias, n, k, T, route_scale, vram, x, dim, idx, w, plan_ptr, req, seq, s, bias_vl, vl, na0, na1);
#else
    pub = pub && req != nullptr && (T == 1 || pub_cnt != nullptr);   // (DS4_HW_PUBT: T > 1 with its counter)
    if ((!fast && !pub) || n < 1 || n > 1024 || k < 1 || k > 16) {   // (the old launches)
        route_publish(logits, bias, n, k, T, route_scale, vram, x, dim, idx, w, plan_ptr, req, seq, s, bias_vl, vl, na0, na1);
        return;
    }
    const float* bvl = vl != 0 ? bias_vl : nullptr;
    if (fast && pub)
        strata_ds4_launch((route_publish_lp_kernel<true, true>), T, 512, 0, s, logits, bias, n, k, route_scale, vram, x, dim, idx, w,
                          plan_ptr, req, bvl, vl, seq, na0, na1, pub_cnt);
    else if (fast)
        strata_ds4_launch((route_publish_lp_kernel<true, false>), T, 512, 0, s, logits, bias, n, k, route_scale, vram, x, dim, idx, w,
                          plan_ptr, req, bvl, vl, seq, na0, na1, pub_cnt);
    else
        strata_ds4_launch((route_publish_lp_kernel<false, true>), T, 512, 0, s, logits, bias, n, k, route_scale, vram, x, dim, idx, w,
                          plan_ptr, req, bvl, vl, seq, na0, na1, pub_cnt);
    if (req != nullptr && !pub) strata_ds4_launch(publish_kernel, 1, 1, 0, s, req, T, seq, na0, na1);
#endif
}
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s) {
#if defined(STRATA_METAL)
    // nothing is published, so nothing is waited for: the CUDA path spins until the host's prefetcher acknowledges
    (void) seq_ptr; (void) seq; (void) s;
    return;
#endif
    strata_ds4_launch(wait_seq_kernel, 1, 32, 0, s, seq_ptr, seq);
}

// the RAM store's experts for the device (a decode token, window of one): a pick the VRAM table left at 0 whose expert the store holds
// (zc_tab[expert] = the slot's device address over PCIe, 0: not) is copied to staging slot i with wide coalesced loads (the PCIe
// rate; the expert kernels' own small loads over PCIe ran at a tenth of it), then the plan points at the copy
__global__ void zc_copy_kernel(const unsigned long long* __restrict__ plan, const int* __restrict__ idx,
                               const unsigned long long* __restrict__ zc_tab, uint8_t* __restrict__ stage, size_t stride, size_t bytes) {
    const int i = blockIdx.y;
    if (plan[i] != 0) return;
    const unsigned long long p = zc_tab[idx[i]];
    if (p == 0) return;
    const uint4* src = (const uint4*) p;
    uint4* dst = (uint4*) (stage + (size_t) i * stride);
    const size_t n = bytes / 16;
    for (size_t o = (size_t) blockIdx.x * blockDim.x + threadIdx.x; o < n; o += (size_t) gridDim.x * blockDim.x) dst[o] = src[o];
}
__global__ void zc_fix_kernel(unsigned long long* __restrict__ plan, const int* __restrict__ idx, const unsigned long long* __restrict__ zc_tab,
                              int k, uint8_t* __restrict__ stage, size_t stride) {
    const int i = threadIdx.x;
    if (i >= k || plan[i] != 0) return;
    if (zc_tab[idx[i]] != 0) plan[i] = (unsigned long long) (stage + (size_t) i * stride);
}
void zc_stage(unsigned long long* plan, const int* idx, const unsigned long long* zc_tab, int k, uint8_t* stage, size_t stride, size_t bytes,
              cudaStream_t s) {
    strata_ds4_launch(zc_copy_kernel, dim3(64, k), 256, 0, s, plan, idx, zc_tab, stage, stride, bytes);
    strata_ds4_launch(zc_fix_kernel, 1, 32, 0, s, plan, idx, zc_tab, k, stage, stride);
}

void init_pre(float* pre, int T, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_INITPRE")) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::init_pre(pre, T, s)) return;   // the ported MSL kernel
#endif
    strata_ds4_launch(init_pre_kernel, cdiv(T * 8, 128), 128, 0, s, pre, T);
}

void embed(const uint16_t* rows, int T, float* h, int dim, int hc, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed_addr(rows, T, h, dim, hc)) return;
#endif
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::embed(rows, T, h, dim, hc, s)) return;
#endif
    strata_ds4_launch(embed_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, rows, h, dim, hc);
}

void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s) {
    if (len > kFinMax * kThreads) {
        std::fprintf(stderr, "ds4 finish: a vector of %d values is more than %d\n", len, kFinMax * kThreads);
        return;
    }
    if (n_vec <= 0) return;
    if (f.q8 != nullptr) {   // (DS4_DECODE_TRIM bit 2: the rows also rounded to int8)
#if !defined(STRATA_METAL)
        if (f.q8d == nullptr || len % 32 != 0 || stride != len || f.dst != nullptr || f.pdst != nullptr) {
            std::fprintf(stderr, "ds4 finish: int8 rows of %d values (stride %lld) not covered\n", len, stride);
            return;
        }
        strata_ds4_launch(finish_kernel<false, true>, n_vec, kThreads, 0, s, x, len, stride, f);
#endif
        return;
    }
    if (f.pdst != nullptr) {
        if (f.quant < kKvFp8 || f.quant > kKvFp4p || len % 32 != 0) {
            std::fprintf(stderr, "ds4 finish: packed rows of quant %d, %d values not covered\n", f.quant, len);
            return;
        }
#if defined(STRATA_METAL)
        const auto& mp = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mp)) return;   // finish_kernel<true> (the KV pack)
#endif
        strata_ds4_launch(finish_kernel<true>, n_vec, kThreads, 0, s, x, len, stride, f);
    } else {
#if defined(STRATA_METAL)
        const auto& mf = *reinterpret_cast<const strata::dsv41::metal_host::FinishIn*>(&f);
        if (strata::dsv41::metal_host::finish(x, n_vec, len, stride, mf)) return;   // the ported MSL kernel
#endif
        strata_ds4_launch(finish_kernel<false>, n_vec, kThreads, 0, s, x, len, stride, f);
    }
}

void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix(h, fn, hcdim, n_mix, T, eps, mixes)) return;
#endif
    strata_ds4_launch(hc_mix_kernel, dim3(n_mix, T), kThreads, 0, s, h, fn, hcdim, eps, mixes);
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, unsigned* cnt,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre, float* post,
                  float* comb, cudaStream_t s) {
    strata_ds4_launch(hc_mix_split_kernel, dim3(n_mix, T), kThreads, 0, s, h, fn, hcdim, eps, mixes, cnt, scale, base, hc, iters, hc_eps, pre,
                                                            post, comb);
}
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb)) return;
#endif
    strata_ds4_launch(hc_split_kernel, T, 32, 0, s, mixes, scale, base, hc, iters, hc_eps, pre, post, comb);
}
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre, float* post,
                  float* comb, cudaStream_t s) {
#if defined(STRATA_METAL)
    // one dispatch: the mix and the sinkhorn split of its result (bit-identical to the two, see ds4_init.metal)
    if (strata::dsv41::metal_host::hc_mix_split(h, fn, hcdim, n_mix, T, eps, mixes, scale, base, hc, iters, hc_eps,
                                                pre, post, comb))
        return;
#endif
    hc_mix(h, fn, hcdim, n_mix, T, eps, mixes, s);
    hc_split(mixes, scale, base, hc, iters, hc_eps, T, pre, post, comb, s);
}

void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_pre_norm(h, mix, w, dim, hc, T, eps, x)) return;
#endif
    strata_ds4_launch(hc_pre_norm_kernel, T, 1024, 0, s, h, mix, w, dim, hc, eps, x);
}
bool hc_mix_split_norm(const HcFuse& a, int T, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) a; (void) T; (void) s;
    return false;   // (no Metal kernel: the caller keeps the old launches)
#else
    if (a.n_mix < 1 || a.n_mix > 32 || a.hc < 1 || a.hc > 8 || a.dim < 32 || a.dim % 32 != 0 || a.dim > 8 * 1024 || T < 1 ||
        T > kWin || a.h == nullptr || a.fn == nullptr || a.mixes == nullptr || a.cnt == nullptr || a.scale == nullptr ||
        a.base == nullptr || a.pre == nullptr || a.post == nullptr || a.comb == nullptr || a.mix == nullptr ||
        a.x == nullptr || (a.q != nullptr && a.qd == nullptr))
        return false;
    strata_ds4_launch(hc_mix_split_norm_kernel, dim3((unsigned) a.n_mix + 1u, (unsigned) T), 1024, 0, s, a);
    return true;
#endif
}
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post_v(y, res, post, comb, dim, hc, T, out)) return;
#endif
    strata_ds4_launch(hc_post_kernel, dim3(cdiv(dim * hc, kThreads), T), kThreads, 0, s, y, res, post, comb, dim, hc, out);
}

void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_post(y, h, post, comb, dim, hc, T)) return;
#endif
    strata_ds4_launch(hc_post_inplace_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, y, h, post, comb, dim, hc);
}
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::hc_mix_scale(h, hcdim, n_mix, T, eps, mixes)) return;
#endif
    strata_ds4_launch(hc_mix_scale_kernel, T, kThreads, 0, s, h, hcdim, n_mix, eps, mixes);
}
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s, int fmt) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::engram_dequant_tok(src, per, cols, T, hd, out, fmt)) return;
#endif
    if (T > 0) strata_ds4_launch(engram_dequant_tok_kernel, T * cols, kThreads, 0, s, src, per, cols, hd, out, fmt);
}
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::moe_sort(idx, T, k, vs, n_vs, bounds, row_tok, pos)) return;
#endif
    strata_ds4_launch(moe_sort_kernel<false>, n_vs, kThreads, 0, s, idx, T, k, vs, bounds, row_tok, pos);
    strata_ds4_launch(bounds_scan_kernel, 1, 32, 0, s, bounds, n_vs);
    strata_ds4_launch(moe_sort_kernel<true>, n_vs, kThreads, 0, s, idx, T, k, vs, bounds, row_tok, pos);
}
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s) {
    if (r1 > r0 && T > 0) {
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::moe_combine(acc, y, pos, w, T, k, r0, r1, dim)) return;
#endif
        strata_ds4_launch(moe_combine_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, acc, y, pos, w, k, r0, r1, dim);
    }
}

void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s) {
    strata_ds4_launch(gemv_f16_kernel, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W, x, n, k, alpha, y);
}
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy,
                   cudaStream_t s) {
#if defined(STRATA_METAL)
    // gemv_f16_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gemv_f16_rows(W, x, ldx, n, k, T, alpha, y, ldy)) return;
#endif
    if (T > 0)
        strata_ds4_launch(gemv_f16_rows_kernel, dim3(cdiv(n, kThreads / 32), cdiv(T, kRowsT)), kThreads, 0, s, W, x, ldx, n, k, T, alpha,
                                                                                               y, ldy);
}

void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (T > 0 && ::strata::dsv41::metal_host::gemv_f32(W, x, n, k, T, y, ldy)) return;
#endif
    if (T > 0) strata_ds4_launch(gemv_f32_kernel, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W, x, n, k, y, ldy);
}
template <typename TW>
static bool gemv_f32_pair_t(const TW* W0, int n0, float* y0, const TW* W1, int n1, float* y1, const float* x, int k, int T,
                            int ldy, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) W0; (void) n0; (void) y0; (void) W1; (void) n1; (void) y1; (void) x; (void) k; (void) T; (void) ldy; (void) s;
    return false;
#else
    if (W0 == nullptr || W1 == nullptr || n0 < 1 || n0 > 1024 || n1 < 1 || n1 > 1024 || k < 4 || k % 4 != 0 || T < 1 ||
        T > kWin || ldy < std::max(n0, n1))
        return false;
    const int b0 = cdiv(n0, kThreads / 32), b1 = cdiv(n1, kThreads / 32);
    strata_ds4_launch(gemv_f32_pair_kernel<TW>, dim3((unsigned) (b0 + b1), (unsigned) T), kThreads, 0, s, W0, n0, y0, W1, n1,
                      y1, x, k, ldy);
    return cudaGetLastError() == cudaSuccess;
#endif
}
bool gemv_f32_pair(const float* W0, int n0, float* y0, const float* W1, int n1, float* y1, const float* x, int k, int T,
                   int ldy, cudaStream_t s) {
    return gemv_f32_pair_t(W0, n0, y0, W1, n1, y1, x, k, T, ldy, s);
}
void gemv_f32(GateW W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s) {
    if (W.h == nullptr) {   // (F32: the old kernel, as before)
        gemv_f32(W.f, x, n, k, T, y, ldy, s);
        return;
    }
#if !defined(STRATA_METAL)
    if (T > 0 && n > 0)
        strata_ds4_launch(gemv_gate_kernel<uint16_t>, dim3(cdiv(n, kThreads / 32), T), kThreads, 0, s, W.h, x, n, k, y, ldy);
#endif
}
bool gemv_f32_pair(GateW W0, int n0, float* y0, GateW W1, int n1, float* y1, const float* x, int k, int T, int ldy,
                   cudaStream_t s) {
    if ((W0.h != nullptr) == (W1.h != nullptr))
        return W0.h != nullptr ? gemv_f32_pair_t(W0.h, n0, y0, W1.h, n1, y1, x, k, T, ldy, s)
                               : gemv_f32_pair_t(W0.f, n0, y0, W1.f, n1, y1, x, k, T, ldy, s);
    // (one router BF16, the other F32: each its own launch - each row's code, so its bits, as the pair's)
    if (W0 == nullptr || W1 == nullptr || n0 < 1 || n0 > 1024 || n1 < 1 || n1 > 1024 || k < 4 || k % 4 != 0 || T < 1 ||
        T > kWin || ldy < std::max(n0, n1))
        return false;
    gemv_f32(W0, x, n0, k, T, y0, ldy, s);
    gemv_f32(W1, x, n1, k, T, y1, ldy, s);
    return cudaGetLastError() == cudaSuccess;
}

void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl, const VlBits* vl) {
    static const VlBits none = {};
#if defined(STRATA_METAL)
    // route_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::route(logits, bias, n, k, T, route_scale, idx, w, bias_vl, vl)) return;
#endif
    strata_ds4_launch(route_kernel, T, 512, 0, s, logits, bias, n, k, route_scale, idx, w, vl != nullptr ? bias_vl : nullptr,
                                   vl != nullptr ? *vl : none);
}

void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s) {
#if defined(STRATA_METAL)
    // declared here rather than in the header: the header carries the other entry points, this one is only used here
    namespace mh = strata::dsv41::metal_host;
    if (n > 0 && mh::swiglu(g, u, n, lim, h)) return;
#endif
    if (n > 0) strata_ds4_launch(swiglu_kernel, cdiv(n, kThreads), kThreads, 0, s, g, u, n, lim, h);
}
bool swiglu_q8(const float* g, const float* u, int n, float lim, float* h, int8_t* q, float* qd, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) g; (void) u; (void) n; (void) lim; (void) h; (void) q; (void) qd; (void) s;
    return false;
#else
    if (n < 0 || n % 32 != 0 || q == nullptr || qd == nullptr) return false;
    if (n > 0) strata_ds4_launch(swiglu_q8_kernel, cdiv(n, kThreads), kThreads, 0, s, g, u, n, lim, h, q, qd);
    return true;
#endif
}
bool copy_regions(const CopyRegions& r, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) r; (void) s;
    return false;
#else
    if (r.n < 1 || r.n > 4) return false;
    size_t most = 0;
    for (int k = 0; k < r.n; ++k) {
        if (r.bytes[k] % 4 != 0 || (uintptr_t) r.dst[k] % 4 != 0 || (uintptr_t) r.src[k] % 4 != 0) return false;
        most = std::max(most, r.bytes[k] / 4);
    }
    const unsigned bx = (unsigned) std::max<size_t>(1, std::min<size_t>(64, (most + kThreads - 1) / kThreads));
    strata_ds4_launch(copy_regions_kernel, dim3(bx, (unsigned) r.n), kThreads, 0, s, r);
    return true;
#endif
}

void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s) {
#if defined(STRATA_METAL)
    // compress_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::compress_win(kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent)) return;
#endif
    strata_ds4_launch(compress_kernel, 1, 512, 0, s, kv, score, kv_state, score_state, ratio, p0, T, d, w, eps, latent);
}

// FG-12 (DS4_IXS_DISPATCH): defined at the end of this file
namespace {
bool ixs_split(const float* q, const float* w, const void* k, int heads, int T, int n, const int* cand, int cand_stride,
               int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s);
}  // namespace

void index_score(const float* q, const float* w, const void* k, int heads, int hd, int T, int n, const int* cand,
                 int cand_stride, int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s,
                 int kfmt) {
    if (hd != 128 || heads > 32 || (kfmt != kKvFp4p && kfmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 index_score: %d heads of %d (keys %d) not covered\n", heads, hd, kfmt);
        return;
    }
    if (n <= 0) return;
#if defined(STRATA_METAL)
    // index_score_kernel<128, kfmt> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::index_score(q, w, (const uint8_t*) k, heads, hd, T, n, cand, cand_stride, blk, &lim,
                                               out, out_stride, kfmt))
        return;
#endif
    if (kfmt == kKvFp4p && ixs_split(q, w, k, heads, T, n, cand, cand_stride, blk, lim, out, out_stride, s))
        return;   // (DS4_IXS_DISPATCH: T 1, n <= its value, gfx1100; unset / 0: false, nothing launched)
    const size_t smem = (size_t) (heads * hd + heads) * sizeof(float);
    const dim3 grid(cdiv(n, kThreads), T);
    if (kfmt == kKvFp4p)
        strata_ds4_launch(index_score_kernel<128, kKvFp4p>, grid, kThreads, smem, s, q, w, (const uint8_t*) k, heads, n, cand,
                                                                       cand_stride, blk, lim, out, out_stride);
    else
        strata_ds4_launch(index_score_kernel<128, kKvBf16>, grid, kThreads, smem, s, q, w, (const uint8_t*) k, heads, n, cand,
                                                                       cand_stride, blk, lim, out, out_stride);
}

void topk_mb(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
             int* out, int out_stride, unsigned* mb, cudaStream_t s) {
    if (r.gen != 0 || mb == nullptr || T < 1 || T > kWin) {   // (device-made sizes: the one-workgroup kernel)
        strata_ds4_launch(topk_kernel, T, 1024, 0, s, vals, stride, r, cand, cand_stride, blk, out, out_stride);
        return;
    }
    int nmax = 1;
    for (int t = 0; t < T; ++t) nmax = r.n[t] > nmax ? r.n[t] : nmax;
    // ~2048 keys a workgroup (8 a thread), at most kTopkMbMaxG workgroups a row; chunks a multiple of kTmbT
    int G = cdiv(nmax, 2048);
    if (G > kTopkMbMaxG) G = kTopkMbMaxG;
    const int chunk = cdiv(cdiv(nmax, G), kTmbT) * kTmbT;
    G = cdiv(nmax, chunk);
    const dim3 grid(G, T);
    strata_ds4_launch(topk_mb_h1, grid, kTmbT, 0, s, vals, stride, r, chunk, mb);
    strata_ds4_launch(topk_mb_h2, grid, kTmbT, 0, s, vals, stride, r, chunk, mb);
    strata_ds4_launch(topk_mb_h3, grid, kTmbT, 0, s, vals, stride, r, chunk, mb);
    strata_ds4_launch(topk_mb_c, grid, kTmbT, 0, s, vals, stride, r, chunk, mb);
    strata_ds4_launch(topk_mb_write, grid, kTmbT, 0, s, vals, stride, r, cand, cand_stride, blk, out, out_stride, chunk, mb);
}

void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s, unsigned* mb, int mb_rows, int mb_min) {
#if defined(STRATA_METAL)
    // topk_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::topk(vals, stride, T, &r, cand, cand_stride, blk, out, out_stride)) return;
#endif
    if (mb != nullptr && r.gen == 0 && T >= 1 && T <= mb_rows && T <= kWin) {   // (DS4_TOPK_MB=1)
        int nmax = 0;
        for (int t = 0; t < T; ++t) nmax = r.n[t] > nmax ? r.n[t] : nmax;
        if (nmax >= mb_min) {
            static std::atomic<int> said{0};
            if (said.exchange(1) == 0)
                std::fprintf(stderr, "ds4: DS4_TOPK_MB engaged: a top-k over %d scores (%d rows) took the multi-block path "
                                     "(n >= %d)\n", nmax, T, mb_min);
            topk_mb(vals, stride, T, r, cand, cand_stride, blk, out, out_stride, mb, s);
            return;
        }
    }
    strata_ds4_launch(topk_kernel, T, 1024, 0, s, vals, stride, r, cand, cand_stride, blk, out, out_stride);
}

void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s, unsigned* mb, int mb_rows, int mb_min) {
    int nb_max = 1;
    Rows rb = r;
    if (r.gen != 0) {   // (a chunk: the widest query's blocks, made on the device per query)
        rb.gen = 4;
        rb.blk = blk;
        nb_max = cdiv((int) ((r.p0 + T) / r.R), blk);
    } else
        for (int t = 0; t < T; ++t) {
            rb.n[t] = cdiv(r.n[t], blk);
            rb.k[t] = r.k[t];
            nb_max = rb.n[t] > nb_max ? rb.n[t] : nb_max;
        }
#if defined(STRATA_METAL)
    // block_max_kernel on the Metal dispatch; the launch below is the fallback.  The selection of the best blocks (topk) follows either way:
    // returning here skipped it, the candidate list stayed unwritten and every context past 16K tokens (4,096 compressed rows) read garbage.
    if (!strata::dsv41::metal_host::block_max(score, stride, T, &r, blk, bm, bm_stride, nb_max))
#endif
    strata_ds4_launch(block_max_kernel, dim3(cdiv(nb_max, kThreads), T), kThreads, 0, s, score, stride, r, blk, bm, bm_stride);
    topk(bm, bm_stride, T, rb, nullptr, 0, 1, out, out_stride, s, mb, mb_rows, mb_min);
}

// the multiprocessors of the current device (cached a device)
static int sm_count() {
    static std::atomic<int> cached[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    int v = cached[dev].load(std::memory_order_relaxed);
    if (v <= 0) {
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || v <= 0) v = 1;
        cached[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}

namespace {
// packed rows: as many heads a block as still leaves two blocks a multiprocessor (a decoded row serves them all)
void launch_packed(const float* q, const uint16_t* win, const uint8_t* c, const int* sel, int sel_stride, const AttnRows& a,
                   const float* sink, int heads, int T, float scale, float* out, cudaStream_t s, int heads_per_block) {
    int G = heads_per_block;
    if (G != 1 && G != 2 && G != 4 && G != 8) {
        G = 8;
        while (G > 1 && (heads % G != 0 || (heads / G) * T < 2 * sm_count())) G /= 2;
    }
    if (heads % G != 0) G = 1;
    const dim3 grid(heads / G, T);
    if (G == 8) strata_ds4_launch(sparse_attn_packed_kernel<8>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else if (G == 4) strata_ds4_launch(sparse_attn_packed_kernel<4>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else if (G == 2) strata_ds4_launch(sparse_attn_packed_kernel<2>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
    else strata_ds4_launch(sparse_attn_packed_kernel<1>, grid, kThreads, 0, s, q, win, c, sel, sel_stride, a, sink, heads, scale, out);
}

#if defined(STRATA_USE_HIP)
// DS4_FAST_ATTN_CHECK=1 (docs/dsv41/cards/PF-02.md): every fast-path call also runs the scalar kernel into a scratch and prints
// a line - the layer, the chunk, the error of the fast output against the scalar one (max / mean absolute, the worst row's
// relative L2, the rows past 1e-2 / 1e-1), the non-finite outputs of each, and Q's FP16 range (max |q|, values past FP16's
// max, nonzero values under its smallest normal).  A diagnostic: it syncs the stream and allocates a call.
bool fast_attn_check() {
    static const bool on = [] {
        const char* e = std::getenv("DS4_FAST_ATTN_CHECK");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}
struct FaStats {
    unsigned nf_fast, nf_ref, rows_2, rows_1, q_over, q_sub;
    unsigned max_d, max_rel, max_q, max_ref;   // (non-negative floats' bits: atomicMax orders them)
    float sum_d, sum_ref;
};
__global__ void __launch_bounds__(kThreads) fa_check_kernel(const float* __restrict__ fast, const float* __restrict__ ref,
                                                            const float* __restrict__ q, FaStats* st) {
    __shared__ float rd[kThreads / 32][8];
    const size_t base = (size_t) blockIdx.x * 512;
    float d2 = 0.0f, r2 = 0.0f, md = 0.0f, sd = 0.0f, sr = 0.0f, mr = 0.0f, qm = 0.0f;
    float nfa = 0.0f, nfb = 0.0f, qo = 0.0f, qsub = 0.0f;
    for (int e = (int) threadIdx.x; e < 512; e += kThreads) {
        const float x = fast[base + e], y = ref[base + e], qv = fabsf(q[base + e]);
        const bool fx = isfinite(x), fy = isfinite(y);
        nfa += fx ? 0.0f : 1.0f;
        nfb += fy ? 0.0f : 1.0f;
        if (fx && fy) {
            const float d = fabsf(x - y);
            d2 += d * d;
            r2 += y * y;
            md = fmaxf(md, d);
            sd += d;
            sr += fabsf(y);
            mr = fmaxf(mr, fabsf(y));
        }
        if (!(qv <= 65504.0f)) qo += 1.0f;
        else if (qv > 0.0f && qv < 6.103515625e-05f) qsub += 1.0f;
        if (isfinite(qv)) qm = fmaxf(qm, qv);
    }
    float v[11] = {d2, r2, md, sd, sr, mr, qm, nfa, nfb, qo, qsub};
#pragma unroll
    for (int i = 0; i < 11; ++i) {   // sums 0, 1, 3, 4, 7 .. 10; maxima 2, 5, 6
        const bool mx = i == 2 || i == 5 || i == 6;
        for (int o = 16; o > 0; o >>= 1) {
            const float w = __shfl_xor_sync(0xffffffffu, v[i], o);
            v[i] = mx ? fmaxf(v[i], w) : v[i] + w;
        }
    }
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int pass = 0; pass < 2; ++pass) {   // (eight values a pass through rd)
        const int i0 = pass * 8, n = pass == 0 ? 8 : 3;
        if (lane == 0)
            for (int i = 0; i < n; ++i) rd[wp][i] = v[i0 + i];
        __syncthreads();
        if (threadIdx.x == 0)
            for (int i = 0; i < n; ++i) {
                const bool mx = i0 + i == 2 || i0 + i == 5 || i0 + i == 6;
                float a = rd[0][i];
                for (int w = 1; w < kThreads / 32; ++w) a = mx ? fmaxf(a, rd[w][i]) : a + rd[w][i];
                v[i0 + i] = a;
            }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    const float rel = v[1] > 0.0f ? sqrtf(v[0] / v[1]) : (v[0] > 0.0f ? INFINITY : 0.0f);
    if (rel > 1e-2f) atomicAdd(&st->rows_2, 1u);
    if (rel > 1e-1f) atomicAdd(&st->rows_1, 1u);
    atomicAdd(&st->nf_fast, (unsigned) v[7]);
    atomicAdd(&st->nf_ref, (unsigned) v[8]);
    atomicAdd(&st->q_over, (unsigned) v[9]);
    atomicAdd(&st->q_sub, (unsigned) v[10]);
    atomicMax(&st->max_d, __float_as_uint(v[2]));
    atomicMax(&st->max_rel, __float_as_uint(fminf(rel, 3.0e38f)));
    atomicMax(&st->max_q, __float_as_uint(v[6]));
    atomicMax(&st->max_ref, __float_as_uint(v[5]));
    atomicAdd(&st->sum_d, v[3]);
    atomicAdd(&st->sum_ref, v[4]);
}
void fa_check_report(const float* fast, const float* ref, const float* q, int rows, FaStats* st, cudaStream_t s, int layer,
                     long long p0, int t0, int T, const char* mode) {
    cudaMemsetAsync(st, 0, sizeof(FaStats), s);
    strata_ds4_launch(fa_check_kernel, rows, kThreads, 0, s, fast, ref, q, st);
    FaStats h{};
    cudaMemcpyAsync(&h, st, sizeof(FaStats), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    auto f = [](unsigned b) {
        float x;
        std::memcpy(&x, &b, 4);
        return x;
    };
    int dev = 0;
    cudaGetDevice(&dev);
    const double n = (double) rows * 512;
    std::fprintf(stderr,
                 "ds4 fa-check dev %d L %d p0 %lld t0 %d T %d mode %s: max|d| %.3e mean|d| %.3e (mean|ref| %.3e max|ref| %.3e) "
                 "worst-row-rel %.3e rows>1e-2 %u rows>1e-1 %u of %d nonfinite fast %u ref %u | max|q| %.3e q>f16max %u "
                 "q<f16min %u\n",
                 dev, layer, p0, t0, T, mode, f(h.max_d), h.sum_d / n, h.sum_ref / n, f(h.max_ref), f(h.max_rel), h.rows_2,
                 h.rows_1, rows, h.nf_fast, h.nf_ref, f(h.max_q), h.q_over, h.q_sub);
}
#endif
}  // namespace

void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out, cudaStream_t s,
                 int ckv_fmt, int heads_per_block, const AttnScratch* scr, int layer, bool sel_done) {
    (void) layer;
    if (d != 512 || (ckv_fmt != kKvFp4 && ckv_fmt != kKvBf16)) {
        std::fprintf(stderr, "ds4 sparse_attn: head dim %d (rows %d) not covered\n", d, ckv_fmt);
        return;
    }
#if defined(STRATA_METAL)
    // sparse_attn_packed_kernel<G> / sparse_attn_kernel<false> on the Metal dispatch; the launches below are the fallback
    if (strata::dsv41::metal_host::sparse_attn(q, win, ckv, sel, sel_stride, &a, sink, heads, d, T, scale, out, ckv_fmt))
        return;
#endif
    if (ckv_fmt == kKvBf16) {
        strata_ds4_launch(sparse_attn_kernel<false>, dim3(heads, T), kThreads, 0, s, q, win, (const uint16_t*) ckv, sel, sel_stride, a,
                                                                       sink, heads, d, scale, out);
        return;
    }
    // a decode step / short verify window of packed rows: the split-KV kernel (Strata's CUDA lab, ds4_attn_split.cuh; 2.9x at
    // one query on an A4000).  The same inputs; the merge sums the chunks in another order, so the bits differ in the last
    // place (4e-7 against a double reference).  DS4_ATTN_SPLIT=1 turns it on (off by default: no measurable decode gain on the R9700 box, the decode is bound elsewhere).
    if (ckv_fmt == kKvFp4 && a.gen == 0 && T >= 1 && T <= kSplitMaxT && heads % kSplitG == 0 && split_enabled()) {
        int total_max = 0;
        for (int t = 0; t < T; ++t) total_max = std::max(total_max, a.nwin[t] + a.nsel[t]);
        if (total_max > 0 && total_max <= kSplitCap * kSplitR) {
            const attnsplit::AttnSplitScratch sc = split_scratch(s, heads);
            if (sc.part != nullptr) {
                attnsplit::sparse_attn_split<kSplitG, kSplitR>(q, win, (const uint8_t*) ckv, kv_row_bytes(kKvFp4, d), sel,
                                                               sel_stride, a, sink, heads, T, scale, out, sc, total_max, s);
                return;
            }
        }
    }
    // a few queries (a decode step, a short window): their selected rows decoded once a query into the scratch, then
    // the BF16 rows' kernel itself (heads_per_block -1: this way at any T)
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (few && scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap) {
        if (!sel_done) {   // (DS4_ATTN_MERGE=3: else the caller's finish_multi_sel ran this very unpack, same arguments)
#if defined(STRATA_METAL)
        if (strata::dsv41::metal_host::unpack_sel(ckv, sel, sel_stride, a.gen, a.p0, a.t0, a.R, a.kcap, a.nsel,
                scr->cap, scr->rows, T, a.gen ? 1 : (T < kWin ? T : kWin), kv_row_bytes(kKvFp4, 512))) return;
#endif
        strata_ds4_launch(unpack_sel_kernel, dim3(cdiv(sel_stride, kStage), T), kThreads, 0, s, (const uint8_t*) ckv, sel, sel_stride, a,
                                                                               scr->cap, scr->rows);
        }
#if !defined(STRATA_USE_HIP)
        if (T == 1)   // (V100: the loads ahead pay for the decode's kernel at one query, not at more)
            strata_ds4_launch(sparse_attn_kernel<true, 8>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap, a, sink,
                                                                             heads, d, scale, out);
        else
#endif
            strata_ds4_launch(sparse_attn_kernel<true, 0>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap, a, sink,
                                                                             heads, d, scale, out);
        return;
    }
    // DS4_FAST_ATTN=1: the chunked prompt's matrix-core path (16 heads a block, a 16-key tile).  The decode and the
    // verify windows keep the scalar kernels: one query's heads are not the matrix's rows there.
#if defined(STRATA_USE_HIP)
    // DS4_FAST_ATTN_MODE picks the precision variant, DS4_FAST_ATTN_LAYERS the layers (docs/dsv41/cards/PF-02.md)
    if (fast_attn_enabled() && a.gen != 0 && T >= kFaKeys && heads % kFaHeads == 0 && fast_attn_layer_on(layer)) {
        const int mode = fast_attn_mode();
        launch_fast_attn(mode, q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, out, s);
        if (fast_attn_check()) {   // DS4_FAST_ATTN_CHECK=1: the scalar kernel on the same inputs, the difference a line a call
            float* ref = nullptr;
            FaStats* st = nullptr;
            if (cudaMalloc((void**) &ref, (size_t) T * heads * 512 * sizeof(float)) == cudaSuccess &&
                cudaMalloc((void**) &st, sizeof(FaStats)) == cudaSuccess) {
                launch_packed(q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, ref, s, heads_per_block);
                fa_check_report(out, ref, q, T * heads, st, s, layer, a.p0, a.t0, T, kFaModes[mode].name);
            } else {
                cudaGetLastError();
                std::fprintf(stderr, "ds4 fa-check: no scratch (%d rows)\n", T * heads);
            }
            if (ref != nullptr) cudaFree(ref);
            if (st != nullptr) cudaFree(st);
        }
        return;
    }
#endif
    launch_packed(q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, out, s, heads_per_block);
}

#if defined(STRATA_USE_HIP)
// test hook (tests/dsv41/fattn_test.cu): sparse_attn's chunked-prompt path with the kernel forced - mode -1 the scalar
// packed kernel (the G sparse_attn picks), 0 .. the DS4_FAST_ATTN_MODE variants; false if the shape does not take it.
bool sparse_attn_forced(int mode, const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                        const AttnRows& a, const float* sink, int heads, int T, float scale, float* out, cudaStream_t s) {
    if (mode < 0) {
        launch_packed(q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, out, s, 0);
        return true;
    }
    if (a.gen == 0 || T < kFaKeys || heads % kFaHeads != 0) return false;
    return launch_fast_attn(mode, q, win, (const uint8_t*) ckv, sel, sel_stride, a, sink, heads, T, scale, out, s);
}
const char* fast_attn_mode_name(int mode) { return mode >= 0 && mode < kFaModeCount ? kFaModes[mode].name : nullptr; }
#endif
void set_fast_attn_tp(bool tp_groups) { g_fast_attn_tp.store(tp_groups ? 1 : 0); }

#if defined(STRATA_METAL)
void tp_strided_rows(uint8_t* dst, size_t dp, const uint8_t* src, size_t sp, size_t width, int rows, cudaStream_t s) {
    for (int r = 0; r < rows; ++r)   // (no kernel of this on the Metal dispatch: unified memory, copies a row)
        (void) cudaMemcpyAsync(dst + (size_t) r * dp, src + (size_t) r * sp, width, cudaMemcpyDefault, s);
}
#else
template <typename W>
__global__ void tp_strided_rows_kernel(uint8_t* __restrict__ dst, size_t dp, const uint8_t* __restrict__ src, size_t sp, int wpr,
                                   long long n) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / wpr, w = i % wpr;
        reinterpret_cast<W*>(dst + (size_t) r * dp)[w] = reinterpret_cast<const W*>(src + (size_t) r * sp)[w];
    }
}
void tp_strided_rows(uint8_t* dst, size_t dp, const uint8_t* src, size_t sp, size_t width, int rows, cudaStream_t s) {
    if (rows <= 0 || width == 0) return;
    const uintptr_t m = (uintptr_t) dst | (uintptr_t) src | (uintptr_t) dp | (uintptr_t) sp | (uintptr_t) width;
    auto go = [&](auto w) {
        using W = decltype(w);
        const int wpr = (int) (width / sizeof(W));
        const long long n = (long long) wpr * rows;
        const unsigned blocks = (unsigned) std::min<long long>(1024, (n + 255) / 256);
        strata_ds4_launch(tp_strided_rows_kernel<W>, blocks, 256, 0, s, dst, dp, src, sp, wpr, n);
    };
    if ((m & 15) == 0) go(uint4{});
    else if ((m & 7) == 0) go(uint2{});
    else if ((m & 3) == 0) go(uint32_t{});
    else go(uint8_t{});
}
#endif

void ring_to_ext(const uint8_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ring_to_ext(ring, ext, (long long) p0, n, d, kv_row_bytes(kKvFp8, d))) return;
#endif
    strata_ds4_launch(ring_to_ext_kernel, n, kThreads, 0, s, ring, ext, (long long) p0, n, d);
}
void ext_to_ring(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s, uint8_t* backup) {
    if (d % 32 != 0) {
        std::fprintf(stderr, "ds4 ext_to_ring: row of %d not covered\n", d);
        return;
    }
#if defined(STRATA_METAL)
    if (strata::dsv41::metal_host::ext_to_ring(ext, ring, (long long) p0, n, T, d, kv_row_bytes(kKvFp8, d), backup)) return;
#endif
    strata_ds4_launch(ext_to_ring_kernel, T, kThreads, 0, s, ext, ring, (long long) p0, n, T, d, backup);
}
void ring_restore(uint8_t* ring, const uint8_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    {   // (row copies on the context queue: the drafter's rollback is a few rows)
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int t = keep; t < T; ++t) cudaMemcpyAsync(ring + (size_t) ((p0 + t) % n) * rb, backup + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep)
        strata_ds4_launch(ring_restore_kernel, T - keep, kThreads, 0, s, ring, backup, (long long) p0, n, keep, kv_row_bytes(kKvFp8, d));
}
void ring_restore_batch(const RingBatch& b, int64_t p0, int n, int keep, int T, int d, cudaStream_t s) {
#if defined(STRATA_METAL)
    {
        const size_t rb = kv_row_bytes(kKvFp8, d);
        for (int li = 0; li < b.n; ++li)
            for (int t = keep; t < T; ++t)
                cudaMemcpyAsync(b.ring[li] + (size_t) ((p0 + t) % n) * rb, b.bak[li] + (size_t) t * rb, rb, cudaMemcpyDeviceToDevice, s);
        return;
    }
#endif
    if (T > keep && b.n > 0)
        strata_ds4_launch(ring_restore_batch_kernel, dim3(T - keep, b.n), kThreads, 0, s, b, (long long) p0, n, keep,
                                                                             kv_row_bytes(kKvFp8, d));
}

// ---------------------------------------------------------------- DS4_MULTI_ATTN (docs/dsv41/cards/FG-05.md; Maya #639)
bool sparse_attn_rows_ok(int T, int max_run, int sel_stride, const AttnScratch* scr, bool mixed) {
#if defined(STRATA_METAL)
    (void) T, (void) max_run, (void) sel_stride, (void) scr, (void) mixed;
    return false;   // (the Metal dispatch has its own sparse_attn kernels: the per-run path)
#else
#if defined(STRATA_USE_HIP)
    (void) mixed;   // (HIP: sparse_attn_kernel<true, 0> at any T)
#else
    if (mixed) return false;   // (CUDA: <true, 8> for a run of one, <true, 0> for more)
#endif
    // a run of n rows takes sparse_attn's few-queries path when n <= kAttnScratchT and the split kernel is off; the
    // shared launch needs every row's scratch rows
    if (split_enabled() || T < 1 || T > kWin || max_run < 1 || max_run > kAttnScratchT) return false;
    return scr != nullptr && scr->rows != nullptr && T <= scr->max_t && sel_stride <= scr->cap;
#endif
}
void sparse_attn_rows(const float* q, const uint16_t* win, const AttnCkv& c, const int* sel, int sel_stride,
                      const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                      cudaStream_t s, bool one, const AttnScratch* scr) {
    if (d != 512 || a.gen != 0 || T < 1 || T > kWin || scr == nullptr || scr->rows == nullptr || T > scr->max_t ||
        sel_stride > scr->cap) {
        std::fprintf(stderr, "ds4 sparse_attn_rows: %d rows (head dim %d, gen %d) not covered\n", T, d, a.gen);
        return;
    }
    strata_ds4_launch(unpack_sel_rows_kernel, dim3(cdiv(sel_stride, kStage), T), kThreads, 0, s, c, sel, sel_stride, a,
                      scr->cap, scr->rows);
#if !defined(STRATA_USE_HIP)
    if (one)   // (sparse_attn's T = 1 kernel on these builds: every run here is of one row)
        strata_ds4_launch(sparse_attn_kernel<true, 8>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap,
                          a, sink, heads, d, scale, out);
    else
#else
    (void) one;
#endif
        strata_ds4_launch(sparse_attn_kernel<true, 0>, dim3(heads, T), kThreads, 0, s, q, win, scr->rows, nullptr, scr->cap,
                          a, sink, heads, d, scale, out);
}
void ring_to_ext_runs(const RingRuns& r, uint16_t* ext, int n, int d, cudaStream_t s) {
    if (r.runs > 0) strata_ds4_launch(ring_to_ext_runs_kernel, dim3(n, r.runs), kThreads, 0, s, r, ext, n, d);
}
void ext_to_ring_runs(const RingRuns& r, const uint16_t* ext, int n, int d, cudaStream_t s) {
    if (d % 32 != 0) {
        std::fprintf(stderr, "ds4 ext_to_ring_runs: row of %d not covered\n", d);
        return;
    }
    int tmax = 0;
    for (int k = 0; k < r.runs; ++k) tmax = std::max(tmax, r.T[k]);
    if (tmax > 0) strata_ds4_launch(ext_to_ring_runs_kernel, dim3(tmax, r.runs), kThreads, 0, s, r, ext, n, d);
}
bool finish_rows(float* x, int n_vec, int len, long long stride, const Finish& f, const FinishRows& r, cudaStream_t s) {
    if (len > kFinMax * kThreads || f.pdst != nullptr || f.vpp < 1 || n_vec > kWin * f.vpp) {
        std::fprintf(stderr, "ds4 finish_rows: %d vectors of %d values (%d a row) not covered\n", n_vec, len, f.vpp);
        return false;
    }
    if (n_vec > 0) strata_ds4_launch(finish_rows_kernel, n_vec, kThreads, 0, s, x, len, stride, f, r);
    return true;
}

void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s) {
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SKIP_HCMEAN")) return;
#endif
#if defined(STRATA_METAL)
    // hc_mean_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::hc_mean(h, T, hc, dim, out, out_stride)) return;
#endif
    strata_ds4_launch(hc_mean_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, h, hc, dim, out, out_stride);
}
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s) {
#if defined(STRATA_METAL)
    // iota_rows_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::iota_rows(out, T, n, stride)) return;
#endif
    strata_ds4_launch(iota_rows_kernel, T, 64, 0, s, out, n, stride);
}
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s) {
#if defined(STRATA_METAL)
    // markov_kernel on the Metal dispatch (the drafter's Markov bias); the launch below is the fallback
    if (strata::dsv41::metal_host::markov_add(logits, E, H, tok, V, R)) return;
#endif
    strata_ds4_launch(markov_kernel, cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, E, H, tok, V, R);
}
void markov_add_q8(float* logits, const uint16_t* E, const uint8_t* Hq, const int* tok, int V, int R, cudaStream_t s) {
    strata_ds4_launch(markov_kernel_q8, cdiv(V, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, E, Hq, tok, V, R);
}
void markov_share(float* logits, const uint16_t* H, const uint16_t* erow, int vn, int R, cudaStream_t s) {
    strata_ds4_launch(markov_share_kernel, cdiv(vn, kThreads / 32), kThreads, (size_t) R * sizeof(float), s, logits, H, erow, vn, R);
}
void argmax_share(const float* row, int vn, int v0, unsigned* slot, unsigned seq, cudaStream_t s) {
    strata_ds4_launch(argmax_share_kernel, 1, 1024, 0, s, row, vn, v0, slot, seq);
}
void markov_pick(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E, uint16_t* erow, int R,
                 cudaStream_t s) {
    strata_ds4_launch(markov_pick_kernel, 1, 256, 0, s, slots, nm, seq, tok, E, erow, R);
}
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s) {
#if defined(STRATA_METAL)
    // confidence_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::confidence(x, E, tok, w, T, dim, R, out)) return;
#endif
    strata_ds4_launch(confidence_kernel, T, 1024, 0, s, x, E, tok, w, dim, R, out);
}

void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s, int fmt) {
#if defined(STRATA_METAL)
    // engram_dequant_kernel on the Metal dispatch (fmt 2, this model's IQ4_NL tables); the launch below is the fallback
    if (strata::dsv41::metal_host::engram_dequant(rows, scales, n, hd, out, fmt)) return;
#endif
    strata_ds4_launch(engram_dequant_kernel, n, kThreads, 0, s, rows, scales, hd, out, fmt);
}

void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s) {
#if defined(STRATA_METAL)
    // engram_gate_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::engram_gate(h, kv, w, hc, dim, T, eps)) return;
#endif
    strata_ds4_launch(engram_gate_kernel, dim3(hc, T), 1024, 0, s, h, kv, w, hc, dim, eps);
}

void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s) {
#if defined(STRATA_METAL)
    // gather_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::gather_rows(x, &g, n, cols, out)) return;
#endif
    if (n > 0) strata_ds4_launch(gather_kernel, dim3(cdiv(cols, kThreads), n), kThreads, 0, s, x, g, cols, out);
}

void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s) {
#if defined(STRATA_METAL)
    // combine_kernel on the Metal dispatch; the launch below is the fallback
    if (strata::dsv41::metal_host::combine(ed, cpu, sh, &c, T, dim, out)) return;
#endif
    strata_ds4_launch(combine_kernel, dim3(cdiv(dim, kThreads), T), kThreads, 0, s, ed, cpu, sh, c, dim, out);
}

void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    strata_ds4_launch(xfer_put_kernel, blocks, kThreads, 0, s, x, flag, seq, cnt);
    // (another GPU, or the host, waits for this flag: on Windows the put must not sit in the queue - #303: two RTX 3080s
    //  hung at the first hand-off, the receiver spinning on a flag whose writer had never been submitted)
    win_submit(s);
}
void win_submit(cudaStream_t s) {
#if defined(_WIN32)
    static const bool on = [] {
        const char* v = std::getenv("DS4_WIN_SUBMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (on) {
        (void) cudaStreamQuery(s);
        (void) cudaGetLastError();
    }
#else
    (void) s;
#endif
}
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s) {
    size_t total = 0;
    for (int i = 0; i < x.n; ++i) total += x.bytes[i] / 16;
    const int blocks = (int) std::max<size_t>(1, std::min<size_t>(64, (total + kThreads - 1) / kThreads));
    strata_ds4_launch(xfer_get_kernel, blocks, kThreads, 0, s, x, flag, seq);
}
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s) {
    if (n > 0) strata_ds4_launch(wait_flags_kernel, 1, 32, 0, s, flags, n, stride, seq);
}
void set_flag(unsigned* flag, unsigned v, cudaStream_t s) {
    strata_ds4_launch(set_flag_kernel, 1, 1, 0, s, flag, v);
    win_submit(s);   // (the writer on another GPU waits for this acknowledgement before its next put - #384)
}
int p2pk_tile_units() {
    static const int v = [] {
        const char* e = std::getenv("DS4_AR_P2PK_TILE");
        // (default 1024 KiB, was 64: tools/p2pk_bw on the test machine, the 10240-row F16 call alone on 3 cards: 64 KiB 62-65 ms,
        // 256 KiB 25-30, 1024 KiB 14-21 - bigger tiles, fewer switches between the two peers' VRAM)
        return std::max(1, std::min(1024, e != nullptr ? std::atoi(e) : 1024)) * 64;   // (KiB -> 16-byte units)
    }();
    return v;
}
int p2pk_order() {   // DS4_AR_P2PK_ORDER: 1 (default) peer-major sends, 0 tile-major (ar_p2pk_kernel)
    static const int v = [] {
        const char* e = std::getenv("DS4_AR_P2PK_ORDER");
        return e != nullptr && std::atoi(e) == 0 ? 0 : 1;
    }();
    return v;
}
int p2pk_blocks() {
    static const int v = [] {
        const char* e = std::getenv("DS4_AR_P2PK_BLOCKS");
        return std::max(1, std::min(1024, e != nullptr ? std::atoi(e) : 192));
    }();
    return v;
}
long long p2pk_units(int n, bool f16) {
    const long long n4 = n / 4;   // (the host path's n4: a tail of n % 4 is not summed there either)
    if (f16) return n4 % 2 == 0 ? n4 / 2 : 0;
    return n4;
}
#if defined(STRATA_USE_HIP)
// DS4_AR_P2PK: the chunks of a call of nu units, member order, by the weights pk_w (integer arithmetic: the same cuts
// on every member).  false (the call takes the host path) unless every chunk has a unit - the parity reuse needs every
// member to wait on every peer each call (ar_p2pk_kernel) - and none passes a slot
static bool p2pk_cuts(const ArGroup& a, long long nu, long long* cut) {
    long long w = 0, acc = 0;
    for (int m = 0; m < a.nm; ++m) w += a.pk_w[m];
    if (w <= 0) return false;
    cut[0] = 0;
    for (int m = 0; m < a.nm; ++m) {
        acc += a.pk_w[m];
        cut[m + 1] = nu * acc / w;
        if (cut[m + 1] - cut[m] < 1 || cut[m + 1] - cut[m] > a.pk_cap) return false;
    }
    return true;
}
static bool p2pk_launch(float* x, int n, const ArGroup& a, int mi, unsigned seq, bool f16, int hard, cudaStream_t s) {
    const long long nu = p2pk_units(n, f16);
    const int tu = p2pk_tile_units();
    if (nu <= 0 || a.nm < 3 || a.nm > ArGroup::kPkMax || tu != a.pk_tu) return false;
    ArPk P;
    if (!p2pk_cuts(a, nu, P.cut)) return false;
    for (int m = 0; m < ArGroup::kPkMax; ++m) {
        P.buf[m] = (pk_u4*) a.pk_buf[m];
        P.flags[m] = a.pk_flags[m];
    }
    for (int m = a.nm + 1; m <= ArGroup::kPkMax; ++m) P.cut[m] = P.cut[a.nm];
    long long ntm = 0;
    for (int k = 0; k < a.nm; ++k) ntm = std::max(ntm, (P.cut[k + 1] - P.cut[k] + tu - 1) / tu);
    if (ntm > a.pk_maxt) return false;
    const int blocks = (int) std::min<long long>(p2pk_blocks(), ntm);
    if (p2pk_order() == 1) {
        if (f16)
            strata_ds4_launch(ar_p2pk_kernel<true, 1>, blocks, kThreads, 0, s, x, P, a.nm, mi, seq, a.pk_cap, a.pk_maxt, tu, a.pk_err[mi], hard);
        else
            strata_ds4_launch(ar_p2pk_kernel<false, 1>, blocks, kThreads, 0, s, x, P, a.nm, mi, seq, a.pk_cap, a.pk_maxt, tu, a.pk_err[mi], hard);
    } else if (f16)
        strata_ds4_launch(ar_p2pk_kernel<true, 0>, blocks, kThreads, 0, s, x, P, a.nm, mi, seq, a.pk_cap, a.pk_maxt, tu, a.pk_err[mi], hard);
    else
        strata_ds4_launch(ar_p2pk_kernel<false, 0>, blocks, kThreads, 0, s, x, P, a.nm, mi, seq, a.pk_cap, a.pk_maxt, tu, a.pk_err[mi], hard);
    win_submit(s);
    return true;
}
#endif
bool allreduce_p2pk_test(float* x, int n, const ArGroup& a, int mi, unsigned seq, bool f16, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    return a.pk_cap > 0 && p2pk_launch(x, n, a, mi, seq, f16, 1, s);
#else
    (void) x, (void) n, (void) a, (void) mi, (void) seq, (void) f16, (void) s;
    return false;
#endif
}
void allreduce_rs_ref(float* x, int n, const ArGroup& a, int mi, unsigned seq, bool f16, cudaStream_t s) {
    const int slot = (int) (seq % (unsigned) a.slots);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));   // (the sums' bits do not depend on the count)
    if (f16)
        strata_ds4_launch(ar_rs16_kernel<0>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
    else
        strata_ds4_launch(ar_rs_kernel<false>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
}
void p2pk_fill(float* x, float* y, size_t n, unsigned seed, int m, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    strata_ds4_launch(p2pk_fill_kernel, 1024, kThreads, 0, s, x, y, (long long) n, (uint32_t) seed, m);
#else
    (void) x, (void) y, (void) n, (void) seed, (void) m, (void) s;
#endif
}
void p2pk_cmp(const float* a, const float* b, size_t n, unsigned* out, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    strata_ds4_launch(p2pk_cmp_kernel, 1024, kThreads, 0, s, a, b, (long long) n, out);
#else
    (void) a, (void) b, (void) n, (void) out, (void) s;
#endif
}
// DS4_AR_WIDE (default 1 since build 8: the test machine 16K F16 prompt +5.6 %, decode set +3.1 %, ids the same - hostwide/
// docs/dsv41/cards/AR-05.md; 0: ld_fresh as before): 1 - the host-path all-reduce kernels read the host rows / sums 16 bytes a
// load (ld_wide); 2 - also ar_rs16_kernel's stores of this member's row and sums 16 bytes a store (the F32 kernels store
// float4 already; measured no faster than 1).  Only where every row / sum base is 16-byte aligned (hmap: page-aligned allocations; stride a
// multiple of 4 floats); otherwise 0, ld_fresh as before.  Next candidate, not here: xfer_copy's ld_fresh (xfer_get)
static int ar_wide(const ArGroup& a) {
    static const int lv = [] {
        const char* e = std::getenv("DS4_AR_WIDE");
        return e == nullptr || *e == '\0' ? 1 : std::max(0, std::min(2, std::atoi(e)));
    }();
    const bool al = a.stride % 4 == 0 && ((uintptr_t) a.data % 16) == 0 && (a.sums == nullptr || ((uintptr_t) a.sums % 16) == 0);
    return al ? lv : 0;
}
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    // DS4_AR_P2PK=1 (a prompt chunk's group only: ArGroup::p2pk is set on no other): the peer-VRAM reduce-scatter +
    // all-gather in the precision the host path below would take for this n (DS4_AR_FUSE / DS4_AR_F16 read the same way),
    // so the same bits; a call it cannot take (p2pk_cuts) goes on below
    if (a.p2pk == 1) {
        static const bool pk_fuse = [] {
            const char* e = std::getenv("DS4_AR_FUSE");
            return e == nullptr || std::atoi(e) != 0;
        }();
        static const int pk_f16_env = [] {
            const char* e = std::getenv("DS4_AR_F16");
            return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
        }();
        const int n4 = n / 4;
        const bool f16 = !(pk_fuse && n4 < (1 << 16)) && (pk_f16_env < 0 ? n4 >= (1 << 16) : pk_f16_env == 1);
        if (p2pk_launch(x, n, a, mi, seq, f16, 0, s)) return;
    }
#endif
    const int slot = (int) (seq % (unsigned) a.slots);
    const int wide = ar_wide(a);
    // DS4_AR_FUSE=1: a small all-reduce (a decode / verify window's, n4 < 65536) of three or more members takes the
    // one-phase path below (each member puts its whole partial out, then sums every member's in member order) instead of the
    // reduce-scatter's two flag rounds (put + sum + put the sum + read the sums): one cross-GPU wait an all-reduce, not two.
    // The bytes it reads grow with nm, but a window's are a few KB a member.  The same sums, bit for bit (both add the members'
    // partials in member order with __fadd_rn).  On by default (the test machine tp3 + 3 MI50: decode +1.2 %, +0.6 % with the drafter, ids
    // bitwise the same; docs/dsv41/FEATURES.md section 2.9); DS4_AR_FUSE=0: the reduce-scatter for every size, as before.
    static const bool ar_fuse = [] {
        const char* e = std::getenv("DS4_AR_FUSE");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (a.nm > 2 && a.sums != nullptr && a.bflags != nullptr && !(ar_fuse && n / 4 < (1 << 16))) {   // (the reduce-scatter: ar_rs_kernel)
        // DS4_AR_F16: 1 every reduce-scatter in FP16, 0 none.  unset: the large ones only (a prompt chunk's, n4 >= 65536,
        // 1 MB and up) - on the test machine's tp3 the 16K prompt 37.6 -> 31.5 s, its chunked-path teacher-forced score within noise of
        // F32 (docs/dsv41/FEATURES.md section 2.9); the decode windows' stay F32 (no gain measured there on 4x R9700, 21.8 vs
        // 21.9 tok/s)
        static const int f16_env = [] {
            const char* e = std::getenv("DS4_AR_F16");
            return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
        }();
        const int n4 = n / 4;
        const bool f16 = f16_env < 0 ? n4 >= (1 << 16) : f16_env == 1;
        // DS4_AR_BLOCKS (default 64, the old count): the blocks of a large all-reduce (a prompt chunk's, 1 MB and up).  Each
        // block's PCIe reads are uncached loads of the peers' host rows, so a few blocks leave the link mostly idle: on the test machine's
        // tp3 a chunk's 128 MB all-reduce took ~61 ms (docs/dsv41/FEATURES.md section 2.9).  The sums are the same bits whatever
        // the count (each element summed over the members in member order).
        static const int big_blocks = [] {
            const char* e = std::getenv("DS4_AR_BLOCKS");
            return std::max(1, std::min(kArBlocks, e != nullptr ? std::atoi(e) : 64));
        }();
        const int cap = n4 >= (1 << 16) ? big_blocks : 64;
        const int blocks = std::max(1, std::min(cap, cdiv(n4, kThreads)));
        if (f16 && wide == 2)
            strata_ds4_launch(ar_rs16_kernel<2>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else if (f16 && wide == 1)
            strata_ds4_launch(ar_rs16_kernel<1>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else if (f16)
            strata_ds4_launch(ar_rs16_kernel<0>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else if (wide)
            strata_ds4_launch(ar_rs_kernel<true>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        else
            strata_ds4_launch(ar_rs_kernel<false>, blocks, kThreads, 0, s, x, n4, a.data, a.sums, a.stride, a.bflags, a.nm, mi, slot, a.slots, seq);
        return;
    }
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    if (wide)
        strata_ds4_launch(ar_sum_kernel<true>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots, seq);
    else
        strata_ds4_launch(ar_sum_kernel<false>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots, seq);
}

// ---- DS4_AR_POST (see ds4_kernels.cuh).  Only the calls allreduce() would give its one-phase host path (xfer_put +
// ar_sum: not P2PK, not the reduce-scatter), so the sums are that path's bits; the put's rows are the same rows
static bool hc_post_ok(const HcPost* hp, int n) {
    return hp != nullptr && hp->res != nullptr && hp->post != nullptr && hp->comb != nullptr && hp->out != nullptr &&
           hp->dim > 0 && hp->dim % 4 == 0 && hp->hc >= 1 && hp->hc <= 8 && n % hp->dim == 0 && n / hp->dim <= kWin;
}
bool allreduce_post(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, const HcPost* hp, int mode,
                    cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) x, (void) n, (void) a, (void) mi, (void) seq, (void) cnt, (void) hp, (void) mode, (void) s;
    return false;
#else
    if (mode <= 0 || mode > 3 || a.nm < 2 || n <= 0 || n % 4 != 0 || a.data == nullptr || a.flags == nullptr) return false;
    if (a.p2pk == 1) return false;   // (allreduce's first path)
    static const bool ar_fuse = [] {   // (allreduce's DS4_AR_FUSE, read the same way)
        const char* e = std::getenv("DS4_AR_FUSE");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (a.nm > 2 && a.sums != nullptr && a.bflags != nullptr && !(ar_fuse && n / 4 < (1 << 16))) return false;   // (the rs path)
    const bool post = (mode & 1) != 0, put = (mode & 2) != 0;
    if (post && !hc_post_ok(hp, n)) return false;
    const int slot = (int) (seq % (unsigned) a.slots);
    const int wide = ar_wide(a);
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    const HcPost h = post ? *hp : HcPost{};
    if (put) {
        if (a.nm < 3 || a.bflags == nullptr || blocks > kArBlocks || a.nm > kThreads) return false;
        if (wide)
            strata_ds4_launch(ar_put_sum_post_kernel<true>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.bflags, a.nm, mi, slot,
                              a.slots, seq, h, post ? 1 : 0);
        else
            strata_ds4_launch(ar_put_sum_post_kernel<false>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.bflags, a.nm, mi, slot,
                              a.slots, seq, h, post ? 1 : 0);
        return true;
    }
    // bit 1 alone: allreduce()'s put as it is, then the sum with hc_post
    if (cnt == nullptr) return false;
    Xfer xp;
    xp.n = 1;
    xp.src[0] = x;
    xp.dst[0] = a.data + ((size_t) mi * a.slots + slot) * a.stride;
    xp.bytes[0] = (size_t) n * 4;
    xfer_put(xp, a.flags + ((size_t) mi * a.slots + slot) * 32, seq, cnt, s);
    if (wide)
        strata_ds4_launch(ar_sum_post_kernel<true>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots,
                          seq, h);
    else
        strata_ds4_launch(ar_sum_post_kernel<false>, blocks, kThreads, 0, s, x, n4, a.data, a.stride, a.flags, a.nm, mi, slot, a.slots,
                          seq, h);
    return true;
#endif
}
bool allreduce_post_selftest(float* x, int n, float* rows, unsigned* flags, const HcPost& hp, int mode, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) x, (void) n, (void) rows, (void) flags, (void) hp, (void) mode, (void) s;
    return false;
#else
    if (mode <= 0 || mode > 3 || n <= 0 || n % 4 != 0 || rows == nullptr || flags == nullptr || !hc_post_ok(&hp, n)) return false;
    const int n4 = n / 4;
    const int blocks = std::max(1, std::min(64, cdiv(n4, kThreads)));
    // one member, slot 0 of 1, seq 1: no waits; the flags [1][1][kArBlocks][8] (bit 2) or [1][1][32] (bit 1)
    if (mode & 2)
        strata_ds4_launch(ar_put_sum_post_kernel<false>, blocks, kThreads, 0, s, x, n4, rows, (size_t) n, flags, 1, 0, 0, 1, 1u, hp,
                          (mode & 1) ? 1 : 0);
    else
        strata_ds4_launch(ar_sum_post_kernel<false>, blocks, kThreads, 0, s, x, n4, (const float*) rows, (size_t) n,
                          (const unsigned*) flags, 1, 0, 0, 1, 1u, hp);
    return true;
#endif
}

// ---- DS4_ATTN_MERGE (see ds4_kernels.cuh): each op checked as its own wrapper checks it, then one launch
bool finish_multi(const FinOp* ops, int nops, cudaStream_t s) {
#if defined(STRATA_METAL)
    (void) ops, (void) nops, (void) s;
    return false;
#else
    if (nops < 1 || nops > kFinOps) return false;
    FinMulti m;
    m.n = nops;
    long long total = 0;
    for (int i = 0; i < kFinOps; ++i) m.rows[i] = 0;
    for (int i = 0; i < nops; ++i) {
        const FinOp& o = ops[i];
        int rows = 0;
        if (o.kind == 0) {   // finish(): the plain path (f.pdst null) or the int8 one (f.q8: its shape rules)
            if (o.x == nullptr || o.n_vec <= 0 || o.len <= 0 || o.len > kFinMax * kThreads || o.f.pdst != nullptr) return false;
            if (o.f.q8 != nullptr && (o.f.q8d == nullptr || o.len % 32 != 0 || o.stride != o.len || o.f.dst != nullptr)) return false;
            rows = o.n_vec;
        } else if (o.kind == 1) {   // ring_to_ext(): n rows
            if (o.ring_r == nullptr || o.ext_w == nullptr || o.n <= 0) return false;
            rows = o.n;
        } else if (o.kind == 2) {   // ext_to_ring(): T rows, d % 32 == 0
            if (o.ext_r == nullptr || o.ring_w == nullptr || o.n <= 0 || o.T <= 0 || o.d % 32 != 0) return false;
            rows = o.T;
        } else
            return false;
        m.op[i] = o;
        m.rows[i] = rows;
        total += rows;
    }
    if (total <= 0 || total > (1 << 20)) return false;
    strata_ds4_launch(fin_multi_kernel, (int) total, kThreads, 0, s, m);
    return true;
#endif
}

// ---- DS4_ATTN_MERGE=3 (see ds4_kernels.cuh).  sparse_attn_sel_op is sparse_attn's own path choice up to its few-queries
// branch (d 512 and kKvFp4 rows; Metal's dispatch and the kKvBf16 kernel are not it; the split-KV branch only with
// DS4_ATTN_SPLIT on; then few && the scratch fits), restricted to HIP gfx1100 devices, where fin_multi_sel_kernel exists.
#if defined(STRATA_USE_HIP) && !defined(STRATA_METAL) && (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__))
bool sparse_attn_sel_op(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int d, int T, int ckv_fmt,
                        int heads_per_block, const AttnScratch* scr, SelOp& u) {
    if (d != 512 || ckv_fmt != kKvFp4 || T < 1 || split_enabled()) return false;
    const bool few = heads_per_block == -1 || (heads_per_block == 0 && T <= kAttnScratchT);
    if (!few || scr == nullptr || scr->rows == nullptr || T > scr->max_t || sel_stride > scr->cap || sel_stride < 1) return false;
    if (!device_is_gfx1100()) return false;
    u.ckv = (const uint8_t*) ckv;
    u.sel = sel;
    u.sel_stride = sel_stride;
    u.cap = scr->cap;
    u.T = T;
    u.out = scr->rows;
    u.a = a;
    return true;
}
bool finish_multi_sel(const FinOp* ops, int nops, const SelOp& u, cudaStream_t s) {
    if (nops < 1 || nops > kFinOps || u.T < 1 || u.T > kWin || u.sel_stride < 1 || u.out == nullptr || !device_is_gfx1100())
        return false;
    FinMulti m;   // (finish_multi's checks and layout, unchanged)
    m.n = nops;
    long long total = 0;
    for (int i = 0; i < kFinOps; ++i) m.rows[i] = 0;
    for (int i = 0; i < nops; ++i) {
        const FinOp& o = ops[i];
        int rows = 0;
        if (o.kind == 0) {
            if (o.x == nullptr || o.n_vec <= 0 || o.len <= 0 || o.len > kFinMax * kThreads || o.f.pdst != nullptr) return false;
            if (o.f.q8 != nullptr && (o.f.q8d == nullptr || o.len % 32 != 0 || o.stride != o.len || o.f.dst != nullptr)) return false;
            rows = o.n_vec;
        } else if (o.kind == 1) {
            if (o.ring_r == nullptr || o.ext_w == nullptr || o.n <= 0) return false;
            rows = o.n;
        } else if (o.kind == 2) {
            if (o.ext_r == nullptr || o.ring_w == nullptr || o.n <= 0 || o.T <= 0 || o.d % 32 != 0) return false;
            rows = o.T;
        } else
            return false;
        m.op[i] = o;
        m.rows[i] = rows;
        total += rows;
    }
    const int nx = cdiv(u.sel_stride, kStage);   // (unpack_sel_kernel's grid: (nx, T))
    total += (long long) nx * u.T;
    if (total <= 0 || total > (1 << 20)) return false;
    strata_ds4_launch(fin_multi_sel_kernel, (int) total, kThreads, 0, s, m, u, nx);
    return true;
}
bool unpack_sel_ref(const SelOp& u, cudaStream_t s) {
    if (u.T < 1 || u.sel_stride < 1 || u.out == nullptr) return false;
    strata_ds4_launch(unpack_sel_kernel, dim3(cdiv(u.sel_stride, kStage), u.T), kThreads, 0, s, u.ckv, u.sel, u.sel_stride, u.a,
                      u.cap, u.out);
    return true;
}
#else
bool sparse_attn_sel_op(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int d, int T, int ckv_fmt,
                        int heads_per_block, const AttnScratch* scr, SelOp& u) {
    (void) ckv, (void) sel, (void) sel_stride, (void) a, (void) d, (void) T, (void) ckv_fmt, (void) heads_per_block, (void) scr;
    (void) u;
    return false;
}
bool finish_multi_sel(const FinOp* ops, int nops, const SelOp& u, cudaStream_t s) {
    (void) ops, (void) nops, (void) u, (void) s;
    return false;
}
bool unpack_sel_ref(const SelOp& u, cudaStream_t s) {
    (void) u, (void) s;
    return false;
}
#endif

// ---- FG-12 (DS4_IXS_DISPATCH=<nmax>, 2026-11-13): the indexer's scores by length.  The c3 bench (ixs_bench,
// c3b-20261113T162047Z, one XTX, exact in all 88 runs) has index_score_split_kernel (L11B's DS4_IXSCORE_SPLIT, bench only until
// now) faster than index_score_kernel<128, kKvFp4p> only at T 1 and n <= 16384 (heads 32, us a call: n 1K -21.8, 16K -9.1,
// 24K -0.1, slower from 32K; heads 24: 1K -15.6, 16K -2.7, 24K +5.8; T 3 slower already at 16K).  DS4_IXS_DISPATCH=N (N > 0):
// index_score takes the split kernel when T == 1, n <= N, the keys are kKvFp4p and the device is gfx1100; everything else -
// and DS4_IXS_DISPATCH unset / empty / 0 - is index_score_kernel exactly as before (ixs_split returns false first thing).
// The kernel is the bench's text byte for byte (ixs_bench.cu sha256 38e90228, the kernel unchanged from
// build 15/L11B 65422c04): the same operations on the same values in the same order as index_score_kernel, so the same bits.
// Compiled for the host and gfx1100 passes only: no gfx906 code (the MI50 helpers never run the indexer; ixs_split checks the
// device).  The launch geometry is the bench's index_score_split, unchanged.
namespace {
#if defined(STRATA_USE_HIP) && !defined(STRATA_METAL) && (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1100__))
// build 15 L11B (DS4_IXSCORE_SPLIT; bench only): index_score_kernel's work, 32 positions a tile (a lane a position), the
// heads split over the block's 8 waves (wave v: heads 4v .. 4v + 3).  Per head the same 128-value dot in the same order as
// index_score_kernel (c ascending in steps of 8, x ascending inside: acc += q * k), the same key decode; relu into an LDS stage;
// then wave 0's lane i sums position i's heads h = 0 .. heads-1 in index_score_kernel's order (s += relu_h * w_h): the same
// operations on the same values in the same order, so the same bits (checked byte for byte here; contraction is the compiler's
// in both kernels, so the bench's comparison is the proof, not the source).  kt tiles a block.
template <int HD, int KF>
__global__ void __launch_bounds__(256) index_score_split_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                                const uint8_t* __restrict__ kc, int heads, int n,
                                                                const int* __restrict__ cand, int cand_stride, int blk,
                                                                Rows lim, float* __restrict__ out, long long out_stride,
                                                                int kt) {
    extern __shared__ float qs[];   // heads * HD queries, heads weights, then the relu stage [32][33]
    float* stg = qs + heads * HD + heads;
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < heads * HD; i += blockDim.x) qs[i] = q[(size_t) t * heads * HD + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * HD + i] = w[(size_t) t * 64 + i];
    __syncthreads();
    float* o = out + (long long) t * out_stride;
    int ln, lk;
    rows_at(lim, t, ln, lk);
    const int wv = threadIdx.x >> 5, lane = threadIdx.x & 31, h0 = wv * 4;
    for (int tt = 0; tt < kt; ++tt) {
        const int i = (blockIdx.x * kt + tt) * 32 + lane;
        const bool live = i < n;
        bool neg = false;
        int p = i;
        if (live && cand != nullptr) {
            if (i >= lk * blk) neg = true;
            else p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
        }
        if (live && !neg && p >= ln) neg = true;
        if (live && !neg && h0 < heads) {
            const uint8_t* kr = kc + (size_t) p * kv_row_bytes(KF, HD);
            float acc[4];
#pragma unroll
            for (int hh = 0; hh < 4; ++hh) acc[hh] = 0.0f;
#pragma unroll 2
            for (int c = 0; c < HD; c += 8) {
                float kv[8];
                if (KF == kKvFp4p) {   // (index_score_kernel's decode, unchanged)
                    const uint32_t cw = *(const uint32_t*) (kr + c / 2);
                    const float ks = ue8m0_val(kr[HD / 2 + c / 32]);
#pragma unroll
                    for (int x = 0; x < 8; ++x) kv[x] = e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;
                } else {
                    const uint4 raw = *(const uint4*) (kr + c * 2);
                    const uint16_t* b = (const uint16_t*) &raw;
#pragma unroll
                    for (int x = 0; x < 8; ++x) kv[x] = bf16f(b[x]);
                }
#pragma unroll
                for (int hh = 0; hh < 4; ++hh) {
                    if (h0 + hh >= heads) break;
                    const float* qh = qs + (h0 + hh) * HD + c;
#pragma unroll
                    for (int x = 0; x < 8; ++x) acc[hh] += qh[x] * kv[x];
                }
            }
#pragma unroll
            for (int hh = 0; hh < 4; ++hh)
                if (h0 + hh < heads) stg[lane * 33 + h0 + hh] = fmaxf(acc[hh], 0.0f);
        }
        __syncthreads();
        if (wv == 0 && live) {
            if (neg) {
                o[i] = -INFINITY;
            } else {
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < 32; ++h) {   // (index_score_kernel's last loop, its order)
                    if (h >= heads) break;
                    s += stg[lane * 33 + h] * qs[heads * HD + h];
                }
                o[i] = s;
            }
        }
        __syncthreads();
    }
}

int ixs_dispatch_nmax() {
    static const int v = [] {
        const char* e = std::getenv("DS4_IXS_DISPATCH");
        if (e == nullptr || *e == 0) return 0;
        const int n = std::atoi(e);
        if (n <= 0) {
            if (n < 0)
                std::fprintf(stderr, "ds4 kernels: DS4_IXS_DISPATCH=%s ignored (0 or a positive n): the indexer keeps index_score_kernel\n", e);
            return 0;
        }
        std::fprintf(stderr, "ds4 kernels: DS4_IXS_DISPATCH=%d: the indexer's scores through index_score_split_kernel on gfx1100 when "
                             "T == 1 and n <= %d (prep-ixsdisp; index_score_kernel otherwise; bit-identical by construction)\n", n, n);
        return n;
    }();
    return v;
}
std::atomic<bool> ixs_said[64];
bool ixs_split(const float* q, const float* w, const void* k, int heads, int T, int n, const int* cand, int cand_stride,
               int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s) {
    const int nmax = ixs_dispatch_nmax();
    if (nmax == 0) return false;
    if (T != 1 || n <= 0 || n > nmax || heads < 1 || heads > 32 || !device_is_gfx1100()) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && !ixs_said[dev].exchange(true))
        std::fprintf(stderr, "ds4 kernels: device %d DS4_IXS_DISPATCH engaged: index_score_split_kernel (first launch: heads %d, n %d, "
                             "%s)\n", dev, heads, n, cand != nullptr ? "candidates" : "no candidates");
    const int kt = std::max(1, std::min(8, n / (32 * 192)));   // (>= ~192 blocks when n allows; 8 tiles = 256 positions a block)
    const size_t smem = (size_t) (heads * 128 + heads + 32 * 33) * sizeof(float);
    const dim3 grid(cdiv(n, 32 * kt), T);
    strata_ds4_launch(index_score_split_kernel<128, kKvFp4p>, grid, 256, smem, s, q, w, (const uint8_t*) k, heads, n, cand,
                      cand_stride, blk, lim, out, out_stride, kt);
    return true;
}
#else
bool ixs_split(const float* q, const float* w, const void* k, int heads, int T, int n, const int* cand, int cand_stride,
               int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s) {
    (void) q, (void) w, (void) k, (void) heads, (void) T, (void) n, (void) cand, (void) cand_stride, (void) blk, (void) lim;
    (void) out, (void) out_stride, (void) s;
    return false;
}
#endif   // (FG-12: the split kernel on the host and gfx1100 passes only)
}  // namespace

}  // namespace strata::ds4
