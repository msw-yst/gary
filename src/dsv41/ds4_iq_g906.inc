// Strata (docs/dsv41/cards/HX-04.md). The token / window experts' gate_up and down on gfx906 (the MI50
// helper: wave64, 60 CUs, HBM2) for groups of one entry (a decode window, T = 1).  Included by ds4_iq.cu inside its anonymous
// namespace (it uses Ty, decode_b<>, load_tab<>, sub_dot<>, lanes_sum<>, swiglu, group, ksigns8).
//
// The same bits as gate_up_kernel / down_kernel (measured: ds4_iq_helper_bench --ab, every type and shape BITWISE): a row's
// 32 / RW lanes take the same sub-blocks in the same order (lane l: ib = l & 7, blocks l >> 3, + kStep, ...), each the same
// fmaf(dx, sub_dot, acc) on the same integer codes, and the same xor tree over those lanes (lanes_sum<RW>: offsets below 32,
// so a wave64 is two such groups, as the portable kernel already runs there).  What differs is the work's layout and the
// instructions that make the same integers:
//   - the weights are read through the global address space (the portable kernel's are FLAT loads: a generic pointer read from
//     memory, and on gfx9 a FLAT load counts on lgkmcnt too, so every codebook (LDS) wait also drains the weight loads);
//   - a lane's activation sub-blocks are loaded once into registers for all the group's rows (the portable kernel re-reads 36
//     bytes of activations for every 8-10 bytes of weights), and its j loop has no branch, so all its loads issue up front;
//   - the signs by one v_perm a word (sgnp) and the integer scales by 24-bit multiplies (sub_dot9): sgn4 / sub_dot's 32-bit
//     multiplies are quarter rate on gfx906;
//   - the grid is sized to the device (DS4_G906_BPC blocks a CU over DS4_G906_LIVE live groups, the tiles spread evenly).
// The kernels are VALU / issue bound on gfx906 (about 100 VALU instructions per 32-weight sub-block), not memory bound: see the
// results file for what was tried (more rows a lane, activations in LDS, a sign table in LDS, forced occupancy: all slower).
// Off on every other device (the kernels' bodies are empty there: the XTX code objects carry no new code path that runs);
// DS4_IQ_G906=0 turns them off on gfx906 too.
#pragma once
#if defined(STRATA_USE_HIP)

namespace g906 {
inline std::atomic<int>& force_mode() { static std::atomic<int> f{-1}; return f; }
inline std::atomic<long long>& launches() { static std::atomic<long long> n{0}; return n; }

inline bool ok_device() {
    static std::atomic<int> arch[16];   // 0 unknown, 1 gfx906, 2 other
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    int v = arch[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        cudaDeviceProp p{};
        v = cudaGetDeviceProperties(&p, dev) == cudaSuccess && std::strncmp(p.gcnArchName, "gfx906", 6) == 0 &&
                    (p.gcnArchName[6] == '\0' || p.gcnArchName[6] == ':') ? 1 : 2;
        arch[dev].store(v, std::memory_order_relaxed);
    }
    if (v != 1) return false;
    const int f = force_mode().load(std::memory_order_relaxed);
    if (f >= 0) return f == 1;
    static const bool env_off = [] { const char* e = std::getenv("DS4_IQ_G906"); return e != nullptr && e[0] == '0'; }();
    return !env_off;
}
inline int env_int(const char* n, int d) {
    const char* e = std::getenv(n);
    return e != nullptr && *e ? std::atoi(e) : d;
}

template <int T>
constexpr bool kTypes = kQK<T> == 256 && Fast<T>::WORDS == 0 && T != kQ5_K && T != kQ6_K;

// the weights through the global address space (see tok_kernel: generic pointers become FLAT loads)
using gptr = const __attribute__((address_space(1))) uint8_t*;
__device__ __forceinline__ uint32_t gld2(gptr p) {
    const __attribute__((address_space(1))) uint16_t* q = (const __attribute__((address_space(1))) uint16_t*) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ float gh2f(gptr p) { return __half2float(__ushort_as_half(*(const __attribute__((address_space(1))) uint16_t*) p)); }
// sgn4 (ds4_iq.cu) by one byte permute: the 4 grid bytes g with the signs of bits 0..3 of n applied.  ng = every byte of g
// negated (all grid bytes are 1..62: ~g + 1 a byte never carries); v_perm picks byte i of g (selector i) or of ng (i + 4) as
// bit i says, the selector built by one 24-bit multiply that spreads the 4 bits to bits 2, 10, 18, 26.  sgn4's 32-bit
// multiplies (quarter rate on gfx906) are gone; the bytes are sgn4's, so the integer dots and the bits are the same.
__device__ __forceinline__ int sgnp(uint32_t g, uint32_t n) {
    const uint32_t ng = (g ^ 0xffffffffu) + 0x01010101u;
    const uint32_t sel = (__umul24(n & 15u, 0x810204u) & 0x04040404u) | 0x03020100u;
    return (int) __builtin_amdgcn_perm(ng, g, sel);
}
// a group of 8 grid bytes (gx: values 0..3, gy: 4..7) with its signs: NB 7 a ksigns8 index (parity sign), NB 8 the 8 bits
template <int NB>
__device__ __forceinline__ void sgn8(uint32_t v, uint32_t gx, uint32_t gy, int& q0, int& q1) {
    const uint32_t sg = NB == 7 ? ksigns8(v) : v;
    q0 = sgnp(gx, sg);
    q1 = sgnp(gy, sg >> 4);
}
template <int T>
__device__ __forceinline__ void decode_g(gptr b, int ib, const uint32_t* tab, Sub& s) {
    if constexpr (T == kIQ2_XXS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[(a0 >> (8 * l)) & 255u];
            sgn8<7>((a1 >> (7 * l)) & 127u, g.x, g.y, s.q[2 * l], s.q[2 * l + 1]);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
        const uint32_t v[4] = {a0 & 0xffffu, a0 >> 16, a1 & 0xffffu, a1 >> 16};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[v[l] & 511u];
            sgn8<7>(v[l] >> 9, g.x, g.y, s.q[2 * l], s.q[2 * l + 1]);
        }
        const uint8_t sc = b[66 + ib];
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const gptr qs = b + 2;
        const uint32_t i4 = gld2(qs + 4 * ib), sg4 = gld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            sgn8<8>((sg4 >> (8 * l)) & 255u, g.x, g.y, s.q[2 * l], s.q[2 * l + 1]);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ3_XXS) {
        const gptr qs = b + 2;
        const uint32_t w0 = gld2(qs + 8 * ib), w1 = gld2(qs + 8 * ib + 4), aux = gld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            sgn8<7>((aux >> (7 * l)) & 127u, tab[(w >> sh) & 255u], tab[(w >> (sh + 8)) & 255u], s.q[2 * l], s.q[2 * l + 1]);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = gh2f(b) * 0.25f;
    } else {
        decode_b<T>((const uint8_t*) b, ib, tab, s);
    }
}
// sub_dot<T> with 24-bit multiplies for the integer scales (v_mul_i32_i24, full rate; sub_dot's int multiply is v_mul_lo_u32,
// quarter rate): |s| <= 63 and a 16-value sum |lo|, |hi| <= 16 * 127 * 127 < 2^23 for every type here, so the products and
// the sum are sub_dot's integers exactly, and the F32 operations are sub_dot's in its order
template <int T>
__device__ __forceinline__ float sub_dot9(const Sub& s, const int4& xa, const int4& xb) {
    if constexpr (Ty<T>::MIN) {
        return sub_dot<T>(s, xa, xb);
    } else {
        int lo = __dp4a(s.q[0], xa.x, 0);
        lo = __dp4a(s.q[1], xa.y, lo);
        lo = __dp4a(s.q[2], xa.z, lo);
        lo = __dp4a(s.q[3], xa.w, lo);
        int hi = __dp4a(s.q[4], xb.x, 0);
        hi = __dp4a(s.q[5], xb.y, hi);
        hi = __dp4a(s.q[6], xb.z, hi);
        hi = __dp4a(s.q[7], xb.w, hi);
        return __fmul_rn(s.d, (float) (__mul24(s.s0, lo) + __mul24(s.s1, hi)));
    }
}

// grid (blocks a group, groups): block b of a live group loops over the row tiles b, b + gridDim.x, ...; a tile is 8 half-waves
// x RW rows.  GU: gate and up rows (RW 1), h = the clamped swiglu; else down rows (RW 2) -> part.  NJ: a lane's sub-blocks a row.
template <int T, bool GU, int NJ>
__global__ void __launch_bounds__(256) tok_kernel(const Args a, int groups) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
    (void) a;
    (void) groups;
#else
    constexpr int RW = GU ? kGuRW : kDnRW, LPR = 32 / RW, kStep = LPR / 8;
    static_assert(LPR % 8 == 0, "a lane keeps one sub-block position");
    __shared__ __align__(16) uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, (int) blockIdx.y, blob, m0, m1)) return;   // (uniform)
    const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
    const int nsb = K / 32, nblk = nsb / 8;
    const uint32_t rb = (uint32_t) row_bytes_t<T>(K);
    const int lane32 = (int) threadIdx.x & 31, hw = (int) threadIdx.x >> 5;
    const int sub = lane32 / LPR, lane = lane32 % LPR;
    const int ib0 = lane & 7, blk0 = lane >> 3;
    // the lane's activations of an entry, in registers for all the rows of the group
    int4 xa[NJ], xb[NJ];
    float dx[NJ];
    int e = 0;
    auto load_x = [&](int mm) {
        e = a.grp_ptr != nullptr ? a.grp_ent[mm] : mm;
        const int8_t* xq = GU ? a.xq + (size_t) (a.grp_ptr != nullptr ? e >> 3 : 0) * a.n_embd : a.hq + (size_t) e * a.n_ff;
        const float* xd = GU ? a.xd + (size_t) (a.grp_ptr != nullptr ? e >> 3 : 0) * nsb : a.hd + (size_t) e * nsb;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {   // (a lane past the row's end - n_ff 2304: 9 blocks over 2 - reads its last block, unused)
            const int kb = min(blk0 + kStep * j, nblk - 1) * 8 + ib0;
            const int4* xp = (const int4*) (xq + (size_t) kb * 32);
            xa[j] = xp[0];
            xb[j] = xp[1];
            dx[j] = xd[kb];
        }
    };
    int mm = m0;
    load_x(mm);
    load_tab<T>(tab);
    __syncthreads();
    // The weights through the global address space: the blob's address comes from memory (grp_ptr), so through a generic
    // pointer the compiler emits FLAT loads, which count on both vmcnt and lgkmcnt - every wait for a codebook (LDS) lookup
    // then also waits for every weight load in flight.  A uniform 64-bit base and 32-bit offsets (an expert is < 4 GiB) also
    // save the 64-bit address adds of every load.
    const gptr w0 = (gptr) blob + a.tr[GU ? 0 : 2];
    const gptr w1 = (gptr) blob + a.tr[1];
    constexpr int kRowsTile = 8 * RW;
    const int ntile = N / kRowsTile;
    for (;;) {   // the group's entries, one at a time (a decode window's group has one; more re-read the weights)
        for (int tile = (int) blockIdx.x; tile < ntile; tile += (int) gridDim.x) {
            const int row = tile * kRowsTile + hw * RW + sub;
            const uint32_t roff = (uint32_t) row * rb;
            float ag = 0.0f, au = 0.0f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                // no branch on the lane's last blocks (a branch per j keeps the compiler from issuing every j's loads up
                // front): a lane past the row's end decodes its last block again and keeps its sum (a select, not an fmaf with 0)
                const int blk = blk0 + kStep * j;
                const bool live = blk < nblk;
                const uint32_t off = roff + (uint32_t) (live ? blk : nblk - 1) * (uint32_t) Ty<T>::B;
                Sub sw;
                decode_g<T>(w0 + off, ib0, tab, sw);
                const float v0 = fmaf(dx[j], sub_dot9<T>(sw, xa[j], xb[j]), ag);
                ag = live ? v0 : ag;
                if constexpr (GU) {
                    Sub su;
                    decode_g<T>(w1 + off, ib0, tab, su);
                    const float v1 = fmaf(dx[j], sub_dot9<T>(su, xa[j], xb[j]), au);
                    au = live ? v1 : au;
                }
            }
            if constexpr (GU) {
                const float gv = lanes_sum<RW>(ag), uv = lanes_sum<RW>(au);
                if (lane == 0) a.h[(size_t) e * a.n_ff + row] = swiglu(gv, uv, a.limit);
            } else {
                const float v = lanes_sum<RW>(ag);
                if (lane == 0) a.part[(size_t) e * a.n_embd + row] = v;
            }
        }
        if (++mm >= m1) break;
        load_x(mm);
    }
#endif
}

template <int T, bool GU, int NJ>
void run(const Args& a, int groups, int live_hint, cudaStream_t s) {
    constexpr int RW = GU ? kGuRW : kDnRW;
    const int N = GU ? a.n_ff : a.n_embd;
    const int ntile = N / (8 * RW);
    // blocks a group: about bpc blocks a CU over the live groups (live_hint: the most groups that are live), the tiles spread
    // evenly over them (288 tiles at a target of 240 blocks: 144 blocks of 2, not 48 of 2 and 192 of 1)
    static const int cus = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp p{};
        return cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.multiProcessorCount > 0 ? p.multiProcessorCount : 60;
    }();
    static const int bpc = env_int("DS4_G906_BPC", 4);
    const int target = std::max(1, (cus * bpc + live_hint - 1) / std::max(1, live_hint));
    const int each = (ntile + target - 1) / target;
    const int per = std::max(1, (ntile + each - 1) / each);
    const dim3 grid((unsigned) per, (unsigned) groups);
    strata_ds4_launch((tok_kernel<T, GU, NJ>), grid, 256, 0, s, a, groups);
    launches().fetch_add(1, std::memory_order_relaxed);
}

// false: not taken (the portable kernels run)
template <int T, bool GU>
bool launch_t(const Args& a, int groups, cudaStream_t s) {
    if constexpr (!kTypes<T>) {
        return false;
    } else {
        constexpr int RW = GU ? kGuRW : kDnRW, kStep = 32 / RW / 8;
        const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
        if (K % 256 != 0 || N % (8 * RW) != 0) return false;
        const int nblk = K / 256, nj = (nblk + kStep - 1) / kStep;
        // the live groups a launch can have: T * k of the grid, but a decode window's helper share is a few
        const int live = std::max(1, std::min(groups, env_int("DS4_G906_LIVE", 2)));
        switch (nj) {   // the lane's sub-blocks: activations in registers (n_embd 5120 -> 5, n_ff 2304 -> 5; 4096 / 2048 -> 4)
            case 2: run<T, GU, 2>(a, groups, live, s); return true;
            case 3: run<T, GU, 3>(a, groups, live, s); return true;
            case 4: run<T, GU, 4>(a, groups, live, s); return true;
            case 5: run<T, GU, 5>(a, groups, live, s); return true;
            case 6: run<T, GU, 6>(a, groups, live, s); return true;
            case 8: run<T, GU, 8>(a, groups, live, s); return true;
            default: return false;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------- mi50kern
// (docs/dsv41/cards/HX-23.md) DS4_MK_GU=1 / DS4_MK_DN=1, default off: the same windows' gate_up / down, re-laid for occupancy.
// tok_kernel above runs at 114 VGPRs (2 waves a SIMD) and, at one live group, 96 blocks of 3 tiles: 1.6 waves a SIMD, every
// LDS / memory wait exposed, 36 CUs with 6 tiles and 24 with 3.  Its VALU work is ~100 instructions a 32-weight sub-block
// (isa.sh counts: gate_up IQ2_XXS 977 VALU + 16 quarter-rate a tile-lane of 10 sub-blocks; down IQ2_XS 504 a tile-lane of 5).
//   - mk_gu_kernel: the gate rows and the up rows on different half-waves (tok_kernel has each lane do both): about half the
//     live registers and twice the waves; the up half-wave's sum goes to the gate half-wave through LDS, swiglu there.
//   - PH 2: a lane's j loop in two phases (a scheduling barrier between), its activations loaded per step from global memory
//     (not hoisted for the whole block): fewer live VGPRs.  PH 1: one phase.
//   - AO 1: sgnp's (mul24 & 0x04040404) | 0x03020100 as one v_and_or_b32 (gfx9's VOP3 takes no literal, so the compiler
//     emits and + or with literals; here the mask sits in a VGPR and the or-constant in an SGPR: one constant-bus read).
//   - the grid: DS4_MK_GU_TPB / DS4_MK_DN_TPB tiles a block (default 1: the device balances single-tile blocks).
// Exactness, by construction: a row's 32 / RW lanes take the same sub-blocks (ib0 = lane & 7, blk0 = lane >> 3, + kStep j) in the
// same j order, each the same fmaf(dx, sub_dot9<T>(s, xa, xb), acc) / live-select on the same integer codes (v_and_or gives the
// selector's same bits, so v_perm the same bytes), then the same lanes_sum<RW> xor tree; swiglu(gv, uv, limit) on the same two
// floats (the up sum crosses LDS as its 32 bits); the entries of a group in the same order (the first always, as tok_kernel).
// So h and part are tok_kernel's bit for bit.  Test: ds4_iq_helper_bench --ab (memcmp against the portable kernels) with the
// env set, then the engine's identity runs.
namespace mk {
inline bool env_on(const char* n) {
    const char* e = std::getenv(n);
    return e != nullptr && e[0] == '1';
}
inline int ph() { static const int v = env_int("DS4_MK_PH", 2) == 1 ? 1 : 2; return v; }   // phases of a lane's j loop
inline bool ao() { static const bool v = env_int("DS4_MK_AO", 1) != 0; return v; }
inline int tpb(bool gu) {
    static const int g = std::max(1, env_int("DS4_MK_GU_TPB", 1)), d = std::max(1, env_int("DS4_MK_DN_TPB", 1));
    return gu ? g : d;
}
inline std::atomic<long long>& launches() { static std::atomic<long long> n{0}; return n; }
// the load line (once a process, at the first gfx906 window that asks) and the launch count at exit: the A/B's engagement check
inline bool announce(const char* name, bool on) {
    if (on) {
        std::fprintf(stderr, "%s=1 (mi50kern): PH %d AO %d TPB gate_up %d down %d\n", name, ph(), ao() ? 1 : 0, tpb(true), tpb(false));
        static const bool reg = [] {
            std::atexit([] { std::fprintf(stderr, "mi50kern: %lld mk launches\n", launches().load()); });
            return true;
        }();
        (void) reg;
    }
    return on;
}
inline bool gu_on() { static const bool v = announce("DS4_MK_GU", env_on("DS4_MK_GU")); return v; }
inline bool dn_on() { static const bool v = announce("DS4_MK_DN", env_on("DS4_MK_DN")); return v; }

template <int T>
constexpr bool kTypes = T == kIQ2_XXS || T == kIQ2_XS || T == kIQ2_S || T == kIQ3_XXS;

typedef int v4i __attribute__((ext_vector_type(4)));   // (a builtin vector: loads through the global address space; HIP's int4 class cannot)
using gi4 = const __attribute__((address_space(1))) v4i*;
using gf = const __attribute__((address_space(1))) float*;
__device__ __forceinline__ int4 i4of(const v4i v) { return make_int4(v.x, v.y, v.z, v.w); }

#if defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ uint32_t and_or(uint32_t x, uint32_t m, uint32_t c) {   // (x & m) | c, m in a VGPR, c an SGPR
    uint32_t r;
    asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(r) : "v"(x), "v"(m), "s"(c));
    return r;
}
#else
__device__ __forceinline__ uint32_t and_or(uint32_t x, uint32_t m, uint32_t c) { return (x & m) | c; }
#endif
// sgnp, with the selector's and + or in one instruction when AO (mvec: 0x04040404 kept in a VGPR by the caller)
template <bool AO>
__device__ __forceinline__ int sgnq(uint32_t g, uint32_t n, uint32_t mvec) {
    if constexpr (!AO) {
        (void) mvec;
        return sgnp(g, n);
    } else {
        const uint32_t ng = (g ^ 0xffffffffu) + 0x01010101u;
        const uint32_t sel = and_or(__umul24(n & 15u, 0x810204u), mvec, 0x03020100u);
        return (int) __builtin_amdgcn_perm(ng, g, sel);
    }
}
template <int NB, bool AO>
__device__ __forceinline__ void sgn8q(uint32_t v, uint32_t gx, uint32_t gy, int& q0, int& q1, uint32_t mvec) {
    const uint32_t sg = NB == 7 ? ksigns8(v) : v;
    q0 = sgnq<AO>(gx, sg, mvec);
    q1 = sgnq<AO>(gy, sg >> 4, mvec);
}
// decode_g with sgn8q: the same loads, indices and bytes
template <int T, bool AO>
__device__ __forceinline__ void decode_q(gptr b, int ib, const uint32_t* tab, Sub& s, uint32_t mvec) {
    if constexpr (T == kIQ2_XXS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[(a0 >> (8 * l)) & 255u];
            sgn8q<7, AO>((a1 >> (7 * l)) & 127u, g.x, g.y, s.q[2 * l], s.q[2 * l + 1], mvec);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
        const uint32_t v[4] = {a0 & 0xffffu, a0 >> 16, a1 & 0xffffu, a1 >> 16};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[v[l] & 511u];
            sgn8q<7, AO>(v[l] >> 9, g.x, g.y, s.q[2 * l], s.q[2 * l + 1], mvec);
        }
        const uint8_t sc = b[66 + ib];
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_S) {
        const gptr qs = b + 2;
        const uint32_t i4 = gld2(qs + 4 * ib), sg4 = gld2(qs + 32 + 4 * ib), h = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[((i4 >> (8 * l)) & 255u) | ((h << (8 - 2 * l)) & 0x300u)];
            sgn8q<8, AO>((sg4 >> (8 * l)) & 255u, g.x, g.y, s.q[2 * l], s.q[2 * l + 1], mvec);
        }
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = gh2f(b) * 0.125f;
    } else {   // kIQ3_XXS
        const gptr qs = b + 2;
        const uint32_t w0 = gld2(qs + 8 * ib), w1 = gld2(qs + 8 * ib + 4), aux = gld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            sgn8q<7, AO>((aux >> (7 * l)) & 127u, tab[(w >> sh) & 255u], tab[(w >> (sh + 8)) & 255u], s.q[2 * l], s.q[2 * l + 1], mvec);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = gh2f(b) * 0.25f;
    }
}

// One matrix's row (RW rows a half-wave) over a tile: the lane's j chain exactly as tok_kernel's, its activations (int8 row xq,
// scales xd of the entry) loaded per step through the global address space, then lanes_sum<RW>.
template <int T, int RW, int NJ, int PH, bool AO>
__device__ __forceinline__ float row_sum(gptr w, uint32_t roff, int nblk, int ib0, int blk0, const int8_t* xq, const float* xd,
                                         const uint32_t* tab, uint32_t mvec) {
    constexpr int kStep = 32 / RW / 8;
    constexpr int JA = PH == 1 ? NJ : (NJ + 1) / 2;   // the first phase's steps
    float acc = 0.0f;
    auto step = [&](int j) {
        const int blk = blk0 + kStep * j;
        const bool live = blk < nblk;
        const int kb = min(blk, nblk - 1) * 8 + ib0;   // (tok_kernel's load_x index)
        const gi4 xp = (gi4) (xq + (size_t) kb * 32);
        const int4 xa = i4of(xp[0]), xb = i4of(xp[1]);
        const float dx = ((gf) xd)[kb];
        const uint32_t off = roff + (uint32_t) (live ? blk : nblk - 1) * (uint32_t) Ty<T>::B;
        Sub sw;
        decode_q<T, AO>(w + off, ib0, tab, sw, mvec);
        const float v0 = fmaf(dx, sub_dot9<T>(sw, xa, xb), acc);
        acc = live ? v0 : acc;
    };
#pragma unroll
    for (int j = 0; j < JA; ++j) step(j);
    if constexpr (PH == 2) {
        // (the second phase's loads are not hoisted above the first's maths.  isa.sh: 80-82 VGPRs, 3 waves a SIMD, for
        // IQ2_XXS / XS / IQ3_XXS; a barrier after every step measured 160-199 VGPRs at compile time: not kept)
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = JA; j < NJ; ++j) step(j);
    }
    return lanes_sum<RW>(acc);
}

// grid (blocks a group, groups); a block's 8 half-waves: 0-3 the gate rows tile * 4 + 0..3, 4-7 the up rows of the same 4.
template <int T, int NJ, int PH, bool AO>
__global__ void __launch_bounds__(256) mk_gu_kernel(const Args a, int groups) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
    (void) a;
    (void) groups;
#else
    (void) groups;
    __shared__ __align__(16) uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ float s_up[4];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, (int) blockIdx.y, blob, m0, m1)) return;   // (uniform)
    const int K = a.n_embd, N = a.n_ff;
    const int nsb = K / 32, nblk = nsb / 8;
    const uint32_t rb = (uint32_t) row_bytes_t<T>(K);
    const int lane = (int) threadIdx.x & 31, hw = (int) threadIdx.x >> 5;
    const bool up = hw >= 4;
    const int r4 = hw & 3;
    const int ib0 = lane & 7, blk0 = lane >> 3;
    uint32_t mvec = 0x04040404u;
    asm volatile("" : "+v"(mvec));   // (kept in a VGPR for and_or)
    load_tab<T>(tab);
    __syncthreads();
    const gptr w = (gptr) blob + (up ? a.tr[1] : a.tr[0]);
    const int ntile = N / 4;
    int mm = m0;
    do {   // (the group's entries in order, the first always: tok_kernel's loop; a decode window's group has one)
        const int e = a.grp_ptr != nullptr ? a.grp_ent[mm] : mm;
        const int xr = a.grp_ptr != nullptr ? e >> 3 : 0;
        const int8_t* xq = a.xq + (size_t) xr * a.n_embd;
        const float* xd = a.xd + (size_t) xr * nsb;
        for (int tile = (int) blockIdx.x; tile < ntile; tile += (int) gridDim.x) {
            const int row = tile * 4 + r4;
            const float v = row_sum<T, 1, NJ, PH, AO>(w, (uint32_t) row * rb, nblk, ib0, blk0, xq, xd, tab, mvec);
            if (up && lane == 0) s_up[r4] = v;
            __syncthreads();
            if (!up && lane == 0) a.h[(size_t) e * a.n_ff + row] = swiglu(v, s_up[r4], a.limit);
            __syncthreads();   // (s_up is the next tile's)
        }
    } while (++mm < m1);
#endif
}

// down: tok_kernel<T, false, NJ>'s rows (RW 2: a half-wave two rows of 16 lanes), against the int8 h row of each entry
template <int T, int NJ, int PH, bool AO>
__global__ void __launch_bounds__(256) mk_dn_kernel(const Args a, int groups) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
    (void) a;
    (void) groups;
#else
    (void) groups;
    constexpr int RW = kDnRW, LPR = 32 / RW;
    __shared__ __align__(16) uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, (int) blockIdx.y, blob, m0, m1)) return;
    const int K = a.n_ff, N = a.n_embd;
    const int nsb = K / 32, nblk = nsb / 8;
    const uint32_t rb = (uint32_t) row_bytes_t<T>(K);
    const int lane32 = (int) threadIdx.x & 31, hw = (int) threadIdx.x >> 5;
    const int sub = lane32 / LPR, lane = lane32 % LPR;
    const int ib0 = lane & 7, blk0 = lane >> 3;
    uint32_t mvec = 0x04040404u;
    asm volatile("" : "+v"(mvec));
    load_tab<T>(tab);
    __syncthreads();
    const gptr w = (gptr) blob + a.tr[2];
    constexpr int kRowsTile = 8 * RW;
    const int ntile = N / kRowsTile;
    int mm = m0;
    do {
        const int e = a.grp_ptr != nullptr ? a.grp_ent[mm] : mm;
        const int8_t* hq = a.hq + (size_t) e * a.n_ff;
        const float* hd = a.hd + (size_t) e * nsb;
        for (int tile = (int) blockIdx.x; tile < ntile; tile += (int) gridDim.x) {
            const int row = tile * kRowsTile + hw * RW + sub;
            const float v = row_sum<T, RW, NJ, PH, AO>(w, (uint32_t) row * rb, nblk, ib0, blk0, hq, hd, tab, mvec);
            if (lane == 0) a.part[(size_t) e * a.n_embd + row] = v;
        }
    } while (++mm < m1);
#endif
}

template <int T, bool GU, int PH, bool AO>
void run_mk(const Args& a, int groups, cudaStream_t s) {
    const int ntile = GU ? a.n_ff / 4 : a.n_embd / (8 * kDnRW);
    const int t = tpb(GU);
    const dim3 grid((unsigned) ((ntile + t - 1) / t), (unsigned) groups);
    // (bisect: the mk kernels only in the gfx906 code object.  Named in a gfx1100 device pass, the 32 instantiations were
    //  emitted there as empty kernels: ds4_iq.cu's gfx1100 code object's loaded span went 12570001 -> 12608913 bytes, over
    //  6 x 2 MiB, and the loader maps code objects in 2 MiB pieces - every XTX had 2 MiB less free at place_experts, one
    //  expert fewer in VRAM, so the default path was no longer build 11's byte for byte.  The launch only ever runs on a gfx906
    //  device: gate_up / down return before it elsewhere - ok_device())
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
    if constexpr (GU)
        strata_ds4_launch((mk_gu_kernel<T, 5, PH, AO>), grid, 256, 0, s, a, groups);
    else
        strata_ds4_launch((mk_dn_kernel<T, 5, PH, AO>), grid, 256, 0, s, a, groups);
#else
    (void) grid;
    (void) s;
#endif
    launches().fetch_add(1, std::memory_order_relaxed);
    g906::launches().fetch_add(1, std::memory_order_relaxed);   // (the bench's "g906" label)
}
// false: not taken (tok_kernel / the portable kernel runs).  Only the five-step shapes, as tok_kernel's NJ 5: n_embd 5120 for
// gate_up (20 blocks over 4 lane positions), n_ff 2304 for down (9 over 2).
template <int T, bool GU>
bool launch_mk(const Args& a, int groups, cudaStream_t s) {
    if constexpr (!kTypes<T>) {
        return false;
    } else {
        const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
        constexpr int kStep = GU ? 4 : 2;
        if (K % 256 != 0 || N % (GU ? 4 : 8 * kDnRW) != 0 || (K / 256 + kStep - 1) / kStep != 5) return false;
        const bool o = ao();
        if (ph() == 1) {
            if (o) run_mk<T, GU, 1, true>(a, groups, s); else run_mk<T, GU, 1, false>(a, groups, s);
        } else {
            if (o) run_mk<T, GU, 2, true>(a, groups, s); else run_mk<T, GU, 2, false>(a, groups, s);
        }
        return true;
    }
}
template <bool GU>
bool take(int type, const Args& a, int groups, cudaStream_t s) {
    switch (type) {
        case kIQ2_XXS: return launch_mk<kIQ2_XXS, GU>(a, groups, s);
        case kIQ2_XS: return launch_mk<kIQ2_XS, GU>(a, groups, s);
        case kIQ2_S: return launch_mk<kIQ2_S, GU>(a, groups, s);
        case kIQ3_XXS: return launch_mk<kIQ3_XXS, GU>(a, groups, s);
        default: return false;
    }
}
}  // namespace mk

// ---------------------------------------------------------------------------------------------------------------- mi50k
// (pre-registered before the runs) Three gfx906-only changes, each behind its own env flag, default off.  Off, nothing
// below is launched and nothing above is changed: tok_kernel, gate_up_kernel / down_kernel and lanes_sum are untouched (their
// bodies are COPIED here, not refactored, so the gfx1100 code object and today's gfx906 kernels stay as they are).
//   DS4_G906_PERGROUP=1  a T 2-4 window (ne_max > DS4_G906_NEMAX, ne_max <= 4) runs ONE launch whose kernel picks the path per
//                        group (blockIdx.y; m1 - m0 is the same for the whole block, so the branch is uniform): a group of one
//                        entry runs tok_kernel's body (the gfx906 fixes) on the first `per` blocks (tok_kernel's grid for the
//                        group; the others leave at once), a group of 0 or 2+ entries runs gate_up_kernel / down_kernel<T, 4>'s
//                        body (one decode of a sub-block for all the group's entries) on block x = its tile, as today.
//   DS4_G906_SGNLDS=1    IQ2_XXS / IQ2_XS / IQ3_XXS (the ksigns8 types): the 8 sign bytes of a 7-bit sign index from a
//                        128 x uint2 LDS table of 0x00 / 0xFF byte masks, applied as (g ^ m) + (m & 0x01010101), in place of
//                        ksigns8's popc + sgnp's mul24 / and / or / perm per word.  (IQ2_S's 8 raw sign bits: unchanged.)
//   DS4_G906_DPP=1       lanes_sum<RW> by ds_swizzle (xor 16, xor 4: bit mode, groups of 32 lanes) and DPP (row_ror 8 = xor 8
//                        in a 16-lane row; quad_perm [2,3,0,1] = xor 2, [1,0,3,2] = xor 1) in place of __shfl_xor's ds_bpermute.
// Exactness (each flag on its own and together; the bench's --ab memcmp and the engine's identity runs check it):
//   PERGROUP: every row is computed by the same body as before on the same inputs - a one-entry group by tok_kernel's (proven
//     BITWISE against the portable kernels, k2b; KNE4 = the same body on T 2-4 groups), a 2+ entry group by the portable NE 4
//     body (today's T 2-4 path, verbatim).  Which block computes a tile does not enter a row's arithmetic.  Token ids follow.
//   SGNLDS: sgnp gives byte i = bit i of the sign word ? (~g_i + 1) & 255 : g_i.  The table's mask byte m_i = bit i ? 0xFF : 0,
//     so (g_i ^ m_i) + (m_i & 1) is the same byte; every grid byte is 1..62, so no byte add carries into the next.  The sign
//     word is ksigns8(index) in both.  Same int8 codes -> the same dp4a integers -> the same floats.
//   DPP: each step is v += v' with v' = the value of lane (l ^ o) for EVERY lane (all five operations are exact xor permutes
//     within the lanes they cover; no lane reads outside its row / 32-lane group), in lanes_sum's order (o = 16 / RW .. 1),
//     the same F32 add on the same two operands.  All lanes are active at every lanes_sum (no divergent branch around them).
// The kernels exist only in the gfx906 code object (and the host's launch stubs): in a gfx1100 device pass this whole section
// is compiled out (bisect's 2 MiB trap: an XTX code object must not grow), as run_mk's launch is.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
namespace mi50k {
template <int T>
constexpr bool kTypes = T == kIQ2_XXS || T == kIQ2_XS || T == kIQ2_S || T == kIQ3_XXS;   // (the pack's expert types)
template <int T>
constexpr bool kSgn = T == kIQ2_XXS || T == kIQ2_XS || T == kIQ3_XXS;                   // (signs by a 7-bit ksigns8 index)

// The sign table: idea and layout after ggml-cuda moe-vec.cu (moe_coal_iq's sg[] / moe_iq_dot), used under the MIT License:
//   Copyright (c) 2023-2026 The ggml authors; Copyright (c) 2026 Ben Peterson.  Permission is hereby granted, free of charge,
//   to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the
//   Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
//   sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject
//   to the following conditions: The above copyright notice and this permission notice shall be included in all copies or
//   substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
//   EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
//   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.
// sgt[i] = the byte masks of ksigns8(i): .x bytes 0..3 = sign bits 0..3, .y = bits 4..7 (0xFF negative, 0x00 positive)
__device__ __forceinline__ void sgn_table(uint2* sgt) {
    if (threadIdx.x < 128) {
        const uint32_t s = ksigns8((uint32_t) threadIdx.x);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((s >> b) & 1u) ? (0xFFu << (8 * b)) : 0u;
            hi |= ((s >> (b + 4)) & 1u) ? (0xFFu << (8 * b)) : 0u;
        }
        sgt[threadIdx.x] = make_uint2(lo, hi);
    }
}
// sgn8<7>(v, gx, gy) by the table: v the 7-bit sign index (0..127 at every call site)
__device__ __forceinline__ void sgn8t(uint32_t v, uint32_t gx, uint32_t gy, int& q0, int& q1, const uint2* sgt) {
    const uint2 m = sgt[v];
    q0 = (int) ((gx ^ m.x) + (m.x & 0x01010101u));
    q1 = (int) ((gy ^ m.y) + (m.y & 0x01010101u));
}
// decode_g with the table for the ksigns8 types (the same loads, indices and bytes); everything else is decode_g
template <int T, bool SG>
__device__ __forceinline__ void decode_gs(gptr b, int ib, const uint32_t* tab, const uint2* sgt, Sub& s) {
    if constexpr (!SG || !kSgn<T>) {
        (void) sgt;
        decode_g<T>(b, ib, tab, s);
    } else if constexpr (T == kIQ2_XXS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[(a0 >> (8 * l)) & 255u];
            sgn8t((a1 >> (7 * l)) & 127u, g.x, g.y, s.q[2 * l], s.q[2 * l + 1], sgt);
        }
        s.s0 = s.s1 = 2 * (int) (a1 >> 28) + 1;
        s.d = gh2f(b) * 0.125f;
    } else if constexpr (T == kIQ2_XS) {
        const gptr q = b + 2 + 8 * ib;
        const uint32_t a0 = gld2(q), a1 = gld2(q + 4);
        const uint32_t v[4] = {a0 & 0xffffu, a0 >> 16, a1 & 0xffffu, a1 >> 16};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = ((const uint2*) tab)[v[l] & 511u];
            sgn8t(v[l] >> 9, g.x, g.y, s.q[2 * l], s.q[2 * l + 1], sgt);
        }
        const uint8_t sc = b[66 + ib];
        s.s0 = 2 * (int) (sc & 15) + 1;
        s.s1 = 2 * (int) (sc >> 4) + 1;
        s.d = gh2f(b) * 0.125f;
    } else {   // kIQ3_XXS
        const gptr qs = b + 2;
        const uint32_t w0 = gld2(qs + 8 * ib), w1 = gld2(qs + 8 * ib + 4), aux = gld2(qs + 64 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t w = l < 2 ? w0 : w1, sh = 16 * (l & 1);
            sgn8t((aux >> (7 * l)) & 127u, tab[(w >> sh) & 255u], tab[(w >> (sh + 8)) & 255u], s.q[2 * l], s.q[2 * l + 1], sgt);
        }
        s.s0 = s.s1 = 2 * (int) (aux >> 28) + 1;
        s.d = gh2f(b) * 0.25f;
    }
}

// lane l ^ O's value (every lane; O a power of two below 32).  The host pass never runs it (its __shfl_xor only parses).
template <int O>
__device__ __forceinline__ float xlane(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int i = __float_as_int(v);
    if constexpr (O == 16)
        return __int_as_float(__builtin_amdgcn_ds_swizzle(i, 0x401F));             // bit mode: and 0x1f, or 0, xor 16
    else if constexpr (O == 8)
        return __int_as_float(__builtin_amdgcn_mov_dpp(i, 0x128, 0xF, 0xF, true));  // row_ror:8 (+8 == -8 mod 16)
    else if constexpr (O == 4)
        return __int_as_float(__builtin_amdgcn_ds_swizzle(i, 0x101F));             // bit mode: xor 4
    else if constexpr (O == 2)
        return __int_as_float(__builtin_amdgcn_mov_dpp(i, 0x4E, 0xF, 0xF, true));   // quad_perm [2,3,0,1]
    else {
        static_assert(O == 1, "xor 16, 8, 4, 2, 1");
        return __int_as_float(__builtin_amdgcn_mov_dpp(i, 0xB1, 0xF, 0xF, true));   // quad_perm [1,0,3,2]
    }
#else
    return __shfl_xor_sync(0xffffffffu, v, O);
#endif
}
// lanes_sum<RW> (ds4_iq.cu) step for step: o = 16 / RW, ..., 1, v += lane (l ^ o)'s v
template <int RW, bool DP>
__device__ __forceinline__ float lsum(float v) {
    if constexpr (!DP) {
        return lanes_sum<RW>(v);
    } else {
        static_assert(RW == 1 || RW == 2, "16 / RW lanes");
        if constexpr (RW == 1) v += xlane<16>(v);
        v += xlane<8>(v);
        v += xlane<4>(v);
        v += xlane<2>(v);
        v += xlane<1>(v);
        return v;
    }
}

// tok_kernel's body (a copy, lines above; V bit 0 = SGNLDS, bit 1 = DPP): tiles blockIdx.x, + tstride, ...; tab / sgt in LDS
template <int T, bool GU, int NJ, int V>
__device__ __forceinline__ void tok_body(const Args& a, const uint8_t* blob, int m0, int m1, int tstride, uint32_t* tab, uint2* sgt) {
    constexpr bool SG = (V & 1) != 0, DP = (V & 2) != 0;
    constexpr int RW = GU ? kGuRW : kDnRW, LPR = 32 / RW, kStep = LPR / 8;
    static_assert(LPR % 8 == 0, "a lane keeps one sub-block position");
    const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
    const int nsb = K / 32, nblk = nsb / 8;
    const uint32_t rb = (uint32_t) row_bytes_t<T>(K);
    const int lane32 = (int) threadIdx.x & 31, hw = (int) threadIdx.x >> 5;
    const int sub = lane32 / LPR, lane = lane32 % LPR;
    const int ib0 = lane & 7, blk0 = lane >> 3;
    int4 xa[NJ], xb[NJ];
    float dx[NJ];
    int e = 0;
    auto load_x = [&](int mm) {
        e = a.grp_ptr != nullptr ? a.grp_ent[mm] : mm;
        const int8_t* xq = GU ? a.xq + (size_t) (a.grp_ptr != nullptr ? e >> 3 : 0) * a.n_embd : a.hq + (size_t) e * a.n_ff;
        const float* xd = GU ? a.xd + (size_t) (a.grp_ptr != nullptr ? e >> 3 : 0) * nsb : a.hd + (size_t) e * nsb;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int kb = min(blk0 + kStep * j, nblk - 1) * 8 + ib0;
            const int4* xp = (const int4*) (xq + (size_t) kb * 32);
            xa[j] = xp[0];
            xb[j] = xp[1];
            dx[j] = xd[kb];
        }
    };
    int mm = m0;
    load_x(mm);
    load_tab<T>(tab);
    if constexpr (SG && kSgn<T>) sgn_table(sgt);
    __syncthreads();
    const gptr w0 = (gptr) blob + a.tr[GU ? 0 : 2];
    const gptr w1 = (gptr) blob + a.tr[1];
    constexpr int kRowsTile = 8 * RW;
    const int ntile = N / kRowsTile;
    for (;;) {
        for (int tile = (int) blockIdx.x; tile < ntile; tile += tstride) {
            const int row = tile * kRowsTile + hw * RW + sub;
            const uint32_t roff = (uint32_t) row * rb;
            float ag = 0.0f, au = 0.0f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int blk = blk0 + kStep * j;
                const bool live = blk < nblk;
                const uint32_t off = roff + (uint32_t) (live ? blk : nblk - 1) * (uint32_t) Ty<T>::B;
                Sub sw;
                decode_gs<T, SG>(w0 + off, ib0, tab, sgt, sw);
                const float v0 = fmaf(dx[j], sub_dot9<T>(sw, xa[j], xb[j]), ag);
                ag = live ? v0 : ag;
                if constexpr (GU) {
                    Sub su;
                    decode_gs<T, SG>(w1 + off, ib0, tab, sgt, su);
                    const float v1 = fmaf(dx[j], sub_dot9<T>(su, xa[j], xb[j]), au);
                    au = live ? v1 : au;
                }
            }
            if constexpr (GU) {
                const float gv = lsum<RW, DP>(ag), uv = lsum<RW, DP>(au);
                if (lane == 0) a.h[(size_t) e * a.n_ff + row] = swiglu(gv, uv, a.limit);
            } else {
                const float v = lsum<RW, DP>(ag);
                if (lane == 0) a.part[(size_t) e * a.n_embd + row] = v;
            }
        }
        if (++mm >= m1) break;
        load_x(mm);
    }
}

// gate_up_kernel<T, 4>'s body for one (group, tile) item, HIP's release grid (a copy, ds4_iq.cu; DP: lsum by DPP)
template <int T, bool DP>
__device__ __forceinline__ void gu_port_body(const Args& a, const uint8_t* blob, int m0, int m1, int tile, uint32_t* tab, int* es, int* xr) {
    constexpr int NE = 4, RW = kGuRW;
    constexpr int LPR = 32 / RW;
    constexpr int kBPS = kQK<T> == 32 ? 1 : 8, kStep = LPR / kBPS;
    static_assert(LPR % kBPS == 0, "a lane keeps one sub-block position");
    const int lane32 = threadIdx.x & 31;
    const size_t rb = row_bytes_t<T>(a.n_embd);
    const int nsb = a.n_embd / 32, nblk = nsb / kBPS;
    load_tab_fast<T>(tab);
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) {
            const int e = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
            es[threadIdx.x] = e;
            xr[threadIdx.x] = a.grp_ptr != nullptr ? e >> 3 : 0;
        }
        __syncthreads();
        {
            const int row = (tile * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
            const int ib0 = kBPS == 8 ? lane & 7 : 0, blk0 = kBPS == 8 ? lane >> 3 : lane;
            const uint8_t* g = blob + a.tr[0] + (size_t) row * rb;
            const uint8_t* u = blob + a.tr[1] + (size_t) row * rb;
            float ag[NE], au[NE];
#pragma unroll
            for (int i = 0; i < NE; ++i) ag[i] = au[i] = 0.0f;
            for (int blk = blk0; blk < nblk; blk += kStep) {
                const int kb = blk * kBPS + ib0;
                Sub sg, su;
                decode_fast_b<T>(g + (size_t) blk * Ty<T>::B, ib0, tab, sg);
                decode_fast_b<T>(u + (size_t) blk * Ty<T>::B, ib0, tab, su);
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        const int4* xp = (const int4*) (a.xq + (size_t) xr[i] * a.n_embd + (size_t) kb * 32);
                        const int4 xa = xp[0], xb = xp[1];
                        const float dx = a.xd[(size_t) xr[i] * nsb + kb];
                        ag[i] = fmaf(dx, sub_dot<T>(sg, xa, xb), ag[i]);
                        au[i] = fmaf(dx, sub_dot<T>(su, xa, xb), au[i]);
                    }
            }
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const float gv = lsum<RW, DP>(ag[i]), uv = lsum<RW, DP>(au[i]);
                    if (lane == 0) a.h[(size_t) es[i] * a.n_ff + row] = swiglu(gv, uv, a.limit);
                }
        }
    }
}
// down_kernel<T, 4>'s body for one (group, tile) item (a copy, ds4_iq.cu)
template <int T, bool DP>
__device__ __forceinline__ void dn_port_body(const Args& a, const uint8_t* blob, int m0, int m1, int tile, uint32_t* tab, int* es) {
    constexpr int NE = 4, RW = kDnRW;
    constexpr int LPR = 32 / RW;
    constexpr int kBPS = kQK<T> == 32 ? 1 : 8, kStep = LPR / kBPS;
    static_assert(LPR % kBPS == 0, "a lane keeps one sub-block position");
    const int lane32 = threadIdx.x & 31;
    const int nsb = a.n_ff / 32, nblk = nsb / kBPS;
    load_tab_fast<T>(tab);
    for (int mm = m0; mm < m1; mm += NE) {
        const int ne = min(NE, m1 - mm);
        __syncthreads();
        if ((int) threadIdx.x < ne) es[threadIdx.x] = a.grp_ptr != nullptr ? a.grp_ent[mm + threadIdx.x] : mm + threadIdx.x;
        __syncthreads();
        {
            const int row = (tile * kWarps + (threadIdx.x >> 5)) * RW + lane32 / LPR, lane = lane32 % LPR;
            const int ib0 = kBPS == 8 ? lane & 7 : 0, blk0 = kBPS == 8 ? lane >> 3 : lane;
            const uint8_t* d = blob + a.tr[2] + (size_t) row * row_bytes_t<T>(a.n_ff);
            float ad[NE];
#pragma unroll
            for (int i = 0; i < NE; ++i) ad[i] = 0.0f;
            for (int blk = blk0; blk < nblk; blk += kStep) {
                const int kb = blk * kBPS + ib0;
                Sub sd;
                decode_fast_b<T>(d + (size_t) blk * Ty<T>::B, ib0, tab, sd);
#pragma unroll
                for (int i = 0; i < NE; ++i)
                    if (i < ne) {
                        const int4* hp = (const int4*) (a.hq + (size_t) es[i] * a.n_ff + (size_t) kb * 32);
                        const int4 ha = hp[0], hb = hp[1];
                        ad[i] = fmaf(a.hd[(size_t) es[i] * nsb + kb], sub_dot<T>(sd, ha, hb), ad[i]);
                    }
            }
#pragma unroll
            for (int i = 0; i < NE; ++i)
                if (i < ne) {
                    const float v = lsum<RW, DP>(ad[i]);
                    if (lane == 0) a.part[(size_t) es[i] * a.n_embd + row] = v;
                }
        }
    }
}

// T 1 (SGNLDS / DPP): tok_kernel with the variant body; grid (per, groups) as tok_kernel's
template <int T, bool GU, int NJ, int V>
__global__ void __launch_bounds__(256) tok_kernel_v(const Args a, int groups) {
    (void) groups;
    __shared__ __align__(16) uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ __align__(16) uint2 sgt[(V & 1) && kSgn<T> ? 128 : 1];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, (int) blockIdx.y, blob, m0, m1)) return;   // (uniform)
    tok_body<T, GU, NJ, V>(a, blob, m0, m1, (int) gridDim.x, tab, sgt);
}
// T 2-4 (PERGROUP): grid (ntile, groups) = the portable kernels' grid; per = tok_kernel's blocks a group
template <int T, bool GU, int NJ, int V>
__global__ void __launch_bounds__(256) pg_kernel(const Args a, int groups, int per) {
    (void) groups;
    __shared__ __align__(16) uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ __align__(16) uint2 sgt[(V & 1) && kSgn<T> ? 128 : 1];
    __shared__ int es[4], xr[4];
    const uint8_t* blob;
    int m0, m1;
    if (!group(a, (int) blockIdx.y, blob, m0, m1)) return;   // (uniform)
    if (m1 - m0 == 1) {                                       // (uniform: the group's entry count)
        if ((int) blockIdx.x >= per) return;
        tok_body<T, GU, NJ, V>(a, blob, m0, m1, per, tab, sgt);
    } else if constexpr (GU) {
        gu_port_body<T, (V & 2) != 0>(a, blob, m0, m1, (int) blockIdx.x, tab, es, xr);
    } else {
        (void) xr;
        dn_port_body<T, (V & 2) != 0>(a, blob, m0, m1, (int) blockIdx.x, tab, es);
    }
}

// ---- host
inline std::atomic<long long>& n_pg() { static std::atomic<long long> n{0}; return n; }
inline std::atomic<long long>& n_sg() { static std::atomic<long long> n{0}; return n; }
inline std::atomic<long long>& n_dp() { static std::atomic<long long> n{0}; return n; }
struct Flags {
    bool pg, sg, dp;
};
// the flags, read once a process; the load lines (once, at the first gfx906 window that asks) and the launch counts at exit
// are the A/B's engagement check
inline const Flags& flags() {
    static const Flags f = [] {
        auto on = [](const char* n) { const char* e = std::getenv(n); return e != nullptr && e[0] == '1'; };
        const Flags r{on("DS4_G906_PERGROUP"), on("DS4_G906_SGNLDS"), on("DS4_G906_DPP")};
        if (r.pg)
            std::fprintf(stderr, "DS4_G906_PERGROUP=1 (mi50k): a T 2-4 window's groups of one entry on tok_kernel's body, "
                                 "of 2-4 on the portable NE 4 body, one launch (LIVE %d BPC %d)\n",
                         env_int("DS4_G906_LIVE", 2), env_int("DS4_G906_BPC", 4));
        if (r.sg) std::fprintf(stderr, "DS4_G906_SGNLDS=1 (mi50k): IQ2_XXS / IQ2_XS / IQ3_XXS signs from a 128 x uint2 LDS mask table\n");
        if (r.dp) std::fprintf(stderr, "DS4_G906_DPP=1 (mi50k): lanes_sum by ds_swizzle / DPP (xor 16 8 4 2 1)\n");
        if (r.pg || r.sg || r.dp)
            std::atexit([] {
                std::fprintf(stderr, "mi50k: %lld pergroup launches, %lld sgnlds launches, %lld dpp launches\n", n_pg().load(),
                             n_sg().load(), n_dp().load());
            });
        return r;
    }();
    return f;
}
template <int T>
inline int variant() {   // bit 0 SGNLDS (the ksigns8 types only), bit 1 DPP
    const Flags& f = flags();
    return (f.sg && kSgn<T> ? 1 : 0) | (f.dp ? 2 : 0);
}
inline void counted(int v, bool pg) {
    if (pg) n_pg().fetch_add(1, std::memory_order_relaxed);
    if (v & 1) n_sg().fetch_add(1, std::memory_order_relaxed);
    if (v & 2) n_dp().fetch_add(1, std::memory_order_relaxed);
    g906::launches().fetch_add(1, std::memory_order_relaxed);   // (the bench's "g906" label)
}
// tok_kernel's blocks a group (a copy of run()'s sizing: about BPC blocks a CU over the live groups, the tiles spread evenly)
inline int per_blocks(int ntile, int live_hint) {
    static const int cus = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp p{};
        return cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.multiProcessorCount > 0 ? p.multiProcessorCount : 60;
    }();
    static const int bpc = env_int("DS4_G906_BPC", 4);
    const int target = std::max(1, (cus * bpc + live_hint - 1) / std::max(1, live_hint));
    const int each = (ntile + target - 1) / target;
    return std::max(1, (ntile + each - 1) / each);
}
// the shapes these kernels take: the pack's types, a lane's five sub-block steps (tok_kernel's NJ 5: n_embd 5120 for gate_up,
// n_ff 2304 for down); anything else: not taken (today's kernels run)
template <int T, bool GU>
inline bool shape_ok(const Args& a) {
    constexpr int RW = GU ? kGuRW : kDnRW, kStep = 32 / RW / 8;
    const int K = GU ? a.n_embd : a.n_ff, N = GU ? a.n_ff : a.n_embd;
    return K % 256 == 0 && N % (8 * RW) == 0 && (K / 256 + kStep - 1) / kStep == 5;
}
template <int T, bool GU>
bool v_launch(const Args& a, int groups, cudaStream_t s) {
    if constexpr (!kTypes<T>) {
        return false;
    } else {
        const int v = variant<T>();
        if (v == 0 || !shape_ok<T, GU>(a)) return false;
        const int N = GU ? a.n_ff : a.n_embd;
        const int live = std::max(1, std::min(groups, env_int("DS4_G906_LIVE", 2)));
        const dim3 grid((unsigned) per_blocks(N / (8 * (GU ? kGuRW : kDnRW)), live), (unsigned) groups);
        if constexpr (kSgn<T>) {
            switch (v) {
                case 1: strata_ds4_launch((tok_kernel_v<T, GU, 5, 1>), grid, 256, 0, s, a, groups); break;
                case 2: strata_ds4_launch((tok_kernel_v<T, GU, 5, 2>), grid, 256, 0, s, a, groups); break;
                default: strata_ds4_launch((tok_kernel_v<T, GU, 5, 3>), grid, 256, 0, s, a, groups); break;
            }
        } else {   // (IQ2_S: no sign table, so v is 2 here)
            strata_ds4_launch((tok_kernel_v<T, GU, 5, 2>), grid, 256, 0, s, a, groups);
        }
        counted(v, false);
        return true;
    }
}
template <int T, bool GU>
bool pg_launch(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if constexpr (!kTypes<T>) {
        return false;
    } else {
        if (ne_max > 4 || !shape_ok<T, GU>(a)) return false;
        if constexpr (!GU) {   // (g906::down's rule: IQ2_XXS down stays on the portable kernel unless DS4_G906_XXS_DOWN=1 / forced)
            static const bool xxs_dn = env_int("DS4_G906_XXS_DOWN", 0) != 0;
            if (T == kIQ2_XXS && !xxs_dn && force_mode().load(std::memory_order_relaxed) != 1) return false;
        }
        const int v = variant<T>();
        const int ntile = (GU ? a.n_ff : a.n_embd) / (8 * (GU ? kGuRW : kDnRW));
        const int live = std::max(1, std::min(groups, env_int("DS4_G906_LIVE", 2)));
        const int per = per_blocks(ntile, live);
        const dim3 grid((unsigned) ntile, (unsigned) groups);
        if constexpr (kSgn<T>) {
            switch (v) {
                case 0: strata_ds4_launch((pg_kernel<T, GU, 5, 0>), grid, 256, 0, s, a, groups, per); break;
                case 1: strata_ds4_launch((pg_kernel<T, GU, 5, 1>), grid, 256, 0, s, a, groups, per); break;
                case 2: strata_ds4_launch((pg_kernel<T, GU, 5, 2>), grid, 256, 0, s, a, groups, per); break;
                default: strata_ds4_launch((pg_kernel<T, GU, 5, 3>), grid, 256, 0, s, a, groups, per); break;
            }
        } else {   // (IQ2_S: v is 0 or 2)
            if (v == 0) strata_ds4_launch((pg_kernel<T, GU, 5, 0>), grid, 256, 0, s, a, groups, per);
            else strata_ds4_launch((pg_kernel<T, GU, 5, 2>), grid, 256, 0, s, a, groups, per);
        }
        counted(v, true);
        return true;
    }
}
// g906::gate_up / down's hooks.  false: not taken (the code after the hook runs, as today)
template <bool GU>
bool pg_try(const Args& a, int groups, int ne_max, cudaStream_t s) {
    const int type = GU ? a.tg : a.td;
    if (!flags().pg || is_tq(type) || !ok_device()) return false;
    switch (type) {
        case kIQ2_XXS: return pg_launch<kIQ2_XXS, GU>(a, groups, ne_max, s);
        case kIQ2_XS: return pg_launch<kIQ2_XS, GU>(a, groups, ne_max, s);
        case kIQ2_S: return pg_launch<kIQ2_S, GU>(a, groups, ne_max, s);
        case kIQ3_XXS: return pg_launch<kIQ3_XXS, GU>(a, groups, ne_max, s);
        default: return false;
    }
}
template <bool GU>
bool v_try(const Args& a, int groups, cudaStream_t s) {
    const Flags& f = flags();
    if (!f.sg && !f.dp) return false;
    switch (GU ? a.tg : a.td) {
        case kIQ2_XXS: return v_launch<kIQ2_XXS, GU>(a, groups, s);
        case kIQ2_XS: return v_launch<kIQ2_XS, GU>(a, groups, s);
        case kIQ2_S: return v_launch<kIQ2_S, GU>(a, groups, s);
        case kIQ3_XXS: return v_launch<kIQ3_XXS, GU>(a, groups, s);
        default: return false;
    }
}
}  // namespace mi50k
#else
namespace mi50k {   // (a gfx1100 device pass: no kernels; the host functions are never emitted here)
template <bool GU>
inline bool pg_try(const Args&, int, int, cudaStream_t) { return false; }
template <bool GU>
inline bool v_try(const Args&, int, cudaStream_t) { return false; }
}  // namespace mi50k
#endif

// a group's most entries these kernels take (each further entry re-reads the group's weights; the portable kernels read them once
// for up to 16 entries): DS4_G906_NEMAX, default 1 (decode windows)
inline int ne_cap() {
    static const int n = env_int("DS4_G906_NEMAX", 1);
    return n;
}
inline bool gate_up(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (ne_max > ne_cap() && mi50k::pg_try<true>(a, groups, ne_max, s)) return true;   // (mi50k DS4_G906_PERGROUP)
    if (ne_max > ne_cap() || is_tq(a.tg) || !ok_device()) return false;
    if (mk::gu_on() && mk::take<true>(a.tg, a, groups, s)) return true;   // (mi50kern DS4_MK_GU)
    if (mi50k::v_try<true>(a, groups, s)) return true;   // (mi50k DS4_G906_SGNLDS / DS4_G906_DPP)
#define G9_(T) return launch_t<T, true>(a, groups, s)
    switch (a.tg) {
        case kIQ2_XXS: G9_(kIQ2_XXS);
        case kIQ2_XS: G9_(kIQ2_XS);
        case kIQ2_S: G9_(kIQ2_S);
        case kIQ3_XXS: G9_(kIQ3_XXS);
        case kIQ3_S: G9_(kIQ3_S);
        case kIQ4_XS: G9_(kIQ4_XS);
        case kQ4_K: G9_(kQ4_K);
        case kQ2_K: G9_(kQ2_K);
        case kQ3_K: G9_(kQ3_K);
        default: return false;
    }
#undef G9_
}
inline bool down(const Args& a, int groups, int ne_max, cudaStream_t s) {
    if (ne_max > ne_cap() && mi50k::pg_try<false>(a, groups, ne_max, s)) return true;   // (mi50k DS4_G906_PERGROUP)
    if (ne_max > ne_cap() || is_tq(a.td) || !ok_device()) return false;
    if (mk::dn_on() && mk::take<false>(a.td, a, groups, s)) return true;   // (mi50kern DS4_MK_DN: IQ2_XXS too)
    // IQ2_XXS down: the portable kernel's one block every 16 rows was measured faster on the device at one group (kernel trace,
    // 17.0 against 18.3 us; every other type and gate_up the other way): it keeps it unless DS4_G906_XXS_DOWN=1
    static const bool xxs_dn = env_int("DS4_G906_XXS_DOWN", 0) != 0;
    if (a.td == kIQ2_XXS && !xxs_dn && force_mode().load(std::memory_order_relaxed) != 1) return false;
    if (mi50k::v_try<false>(a, groups, s)) return true;   // (mi50k DS4_G906_SGNLDS / DS4_G906_DPP)
#define G9_(T) return launch_t<T, false>(a, groups, s)
    switch (a.td) {
        case kIQ2_XXS: G9_(kIQ2_XXS);
        case kIQ2_XS: G9_(kIQ2_XS);
        case kIQ2_S: G9_(kIQ2_S);
        case kIQ3_XXS: G9_(kIQ3_XXS);
        case kIQ3_S: G9_(kIQ3_S);
        case kIQ4_XS: G9_(kIQ4_XS);
        case kQ4_K: G9_(kQ4_K);
        case kQ2_K: G9_(kQ2_K);
        case kQ3_K: G9_(kQ3_K);
        default: return false;
    }
#undef G9_
}
}  // namespace g906

#else
namespace g906 {
inline std::atomic<int>& force_mode() { static std::atomic<int> f{-1}; return f; }
inline std::atomic<long long>& launches() { static std::atomic<long long> n{0}; return n; }
inline bool ok_device() { return false; }
inline bool gate_up(const Args&, int, int, cudaStream_t) { return false; }
inline bool down(const Args&, int, int, cudaStream_t) { return false; }
}  // namespace g906
#endif
