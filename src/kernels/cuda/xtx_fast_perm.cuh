// src/kernels/cuda/xtx_fast_perm.cuh - STRATA_XTX_FAST_PERM=1: the IQ4_NL / IQ4_XS 16-entry codebook lookup on gfx1100
// (RX 7900 XTX) by v_perm_b32 directly, the gfx906 body of iq_kernels.cu get_int_from_table_16 (llama.cpp's HIP branch;
// the DeepSeek line's c18a7159b did the same for its MXFP4 table).
//
// Today's gfx1100 lookup is the CUDA form: six __byte_perm, each through hip_compat/intrinsics.hpp byte_perm, which
// spreads the 4-bit nibble selector to byte selectors (and/shift/or) before its v_perm_b32.  The form below feeds
// v_perm_b32 byte selectors directly: x & 0x07070707 indexes each 8-byte table half, then bit 3 of each nibble picks
// the half.  Both give result byte k = table[(q4 >> 8k) & 15] (.x) and table[(q4 >> (8k + 4)) & 15] (.y), so the dp4a
// operands, the integer sums and every float are unchanged (exhaustive CPU check, all 2^32 words, 0 mismatches).
//
// The flag is a per-TU __constant__ int (one copy per device; iq_kernels.cu and native_mmvq.cu each have one), 0 unless
// xtx_fast_perm_startup (iq_kernels.cu, declared in strata/kernels/iq_kernels.hpp) wrote 1 into both on that device.
// The engine calls it once at process start (generate.cpp main before the family switch, q36_main.cpp, the expert
// bench), before any context work or graph capture: every gfx1100 device is set with a synchronous
// cudaMemcpyToSymbol on a thread with no stream capturing, then cudaDeviceSynchronize.  (04fcd5673 set it lazily
// from the IQ4 launchers, skipping a capturing stream; in serve mode those launchers only ever run inside the session's
// graph captures, so it was skipped every time, silently.)  OFF never writes it.  Only the gfx1100 device pass reads
// it; every other arch (gfx906 included) and CUDA compile today's code.  The flag can only change speed: both forms
// are bitwise the same.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace strata::kernels::xtx_perm {

// The raw STRATA_XTX_FAST_PERM value, or nullptr when unset
inline const char* env_value() { return std::getenv("STRATA_XTX_FAST_PERM"); }

// STRATA_XTX_FAST_PERM=1 (exactly "1"); anything else, or unset: off
inline bool requested() {
    static const bool on = [] {
        const char* v = env_value();
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
// llama.cpp's HIP lookup (vecdotq.cuh), as get_int_from_table_16's gfx906 branch: 4 + 2 v_perm_b32 per 8 values
__device__ __forceinline__ int2 table16_perm(uint32_t q4, const int8_t* table) {
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = q4, q_odd = q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
}
#endif

}  // namespace strata::kernels::xtx_perm
