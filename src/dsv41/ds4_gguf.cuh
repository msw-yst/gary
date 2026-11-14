// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_gguf.cuh - a GGUF's weights on the device for maya-ds4 (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs,
// tools/ds4/pack_gguf.py): the dense matrices in BF16 and the routed experts in MXFP4.  They stand in for
// strata::dsv41k::exl3's matrices and experts (an x3::Mat with fmt kFmtBF16, an x3::ExpertLayout with fmt kFmtMXFP4)
// with the same arguments and the same results' layout, so the engine's paths (decode, verify windows, chunked prompts,
// tensor parallel) take either format.
//
// MXFP4 is ggml's block_mxfp4: 32 values in 17 bytes, an E8M0 scale e then 16 bytes of E2M1 codes - value j of the
// block in byte j's low nibble (j < 16) or byte j - 16's high nibble, worth kvalues[code] * 2^(e - 128) with kvalues
// {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12} (the E2M1 values doubled).  A matrix is [out][in / 32] blocks
// (row r = output r), as the GGUF stores it, so an expert's blob is its gate, up and down matrices one after another.
// Every product is an F32 sum of exactly dequantised weights times F32 activations, and a window's entry gets the bits
// the token path gives it (the same kernel, the same order).
#pragma once

#include "strata/dsv41/exl3.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4::gguf {

namespace x3 = strata::dsv41k::exl3;

constexpr int kFmtBF16 = 1;    // x3::Mat::fmt
constexpr int kFmtMXFP4 = 2;   // x3::ExpertLayout::fmt (and an expert's matrix as an x3::Mat)
constexpr int kFmtQ8_0 = 3;    // x3::Mat::fmt: ggml's Q8_0, an FP16 scale and 32 int8 in 34 bytes (UD-Q4_K_XL's dense)
constexpr int kFmtGG = 4;      // x3::Mat::fmt: ggml's low-bit block type m.K (ds4_iq.cuh's: IQ2_XXS .. Q4_K), [n][k / 256]
constexpr int kMxBlock = 32, kMxBytes = 17, kQ8Bytes = 34;
constexpr int kTypeMXFP4 = 39;   // ggml's type id

/// Bytes of an MXFP4 row of k values (k % 32 == 0), and of a matrix of n such rows.
__host__ __device__ inline size_t mx_row_bytes(int64_t k) { return (size_t) (k / kMxBlock) * kMxBytes; }
__host__ __device__ inline size_t mx_bytes(int64_t n, int64_t k) { return (size_t) n * mx_row_bytes(k); }
__host__ __device__ inline size_t q8_row_bytes(int64_t k) { return (size_t) (k / 32) * kQ8Bytes; }

/// A GGUF expert layer (x3::ExpertLayout::fmt kFmtMXFP4: block-quantized experts read in place) keeps each role's ggml
/// type in lay.K[r] (r: gate, up, down; 0 = MXFP4): MXFP4 takes these kernels, the low-bit types ds4_iq.cuh's (gate
/// and up of one type).  Roles of one layer may differ (down of another type); MXFP4 and a low-bit type do not mix.
inline int expert_type(const x3::ExpertLayout& l, int r) { return l.K[r] == 0 ? kTypeMXFP4 : l.K[r]; }
inline bool mx_experts(const x3::ExpertLayout& l) {
    return expert_type(l, 0) == kTypeMXFP4 && expert_type(l, 1) == kTypeMXFP4 && expert_type(l, 2) == kTypeMXFP4;
}
/// Bytes of a row of k values of an expert type (0: not one the device kernels take, or k not whole blocks).
size_t type_row_bytes(int type, int64_t k);
/// Values a block of an expert type (32: MXFP4, 256: the low-bit ones; 0: neither).
int type_block(int type);
/// The type's name (ggml's), "?" when unknown.
const char* type_name(int type);
/// Whether the device kernels take this type (MXFP4 is its own path; the i-quants and K-quants ds4_iq.cu instantiates).
/// A false here with a layout known to type_row_bytes() means the CPU lane reads it and it must not enter the VRAM tier.
bool type_on_device(int type);
/// Whether the *chunk* path's expert tile GEMM (ds4_mm.metal on Metal) takes this type.  IQ1_M has a matvec kernel on
/// Metal but no tile kernel there, so a prompt in chunks would stream it to the GPU and stop; false here means the
/// prompts must run in windows instead (ds4_model.cu switches, and says so).
bool type_on_chunk(int type);

/// The per-stream scratch of the MXFP4 experts on the current device (once per stream, as x3::prepare).
bool prepare(cudaStream_t s);

/// x3::mv's contract for GGUF matrices (F32 activations): each job's m->fmt kFmtBF16 (raw = BF16 [n][ld], ld values a
/// row), kFmtMXFP4 (raw = MXFP4 [n][k / 32] blocks, ld bytes a row; 0: mx_row_bytes(k)) - an expert's matrix in VRAM -
/// or kFmtQ8_0 (raw = Q8_0 [n][k / 32] blocks, ld bytes a row; 0: q8_row_bytes(k)).
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s);
/// (DS4_HC_FUSE / DS4_DECODE_TRIM, after Project Maya #511's PreQ8) an F32 input x of k values a row, already rounded
/// to int8 as mv rounds a Q8_0 or low-bit (kFmtGG) job's input (q [rows][k], d [rows][k / 32]: q8_quant's bytes, see
/// hc_mix_split_norm / swiglu_q8 / Finish::q8): such a job whose xf is x, its k this k and its activations these rows (k
/// apart when more than one) reads them instead of rounding x again; every other job (another input, format, k or
/// rows) rounds its own as before.  With every rounding given, the call launches no rounding at all.
struct PreQ8 {
    const float* x = nullptr;
    const int8_t* q = nullptr;
    const float* d = nullptr;
    int k = 0;
    int rows = 1;
    // (xtxkern DS4_XK_QO) nslice > 1: nslice inputs slice values apart (x + i slice, q + i slice, d + i slice / 32; one
    // row each, slice % 32 == 0) - wo_a's groups reading the heads' one rounded vector
    int nslice = 1;
    size_t slice = 0;
};
bool mv(const x3::MvJob* jobs, int n, cudaStream_t s, const PreQ8* pre);
/// xtxkern DS4_XK_QA: a Q5_K (kFmtGG) matrix's rows as bytes - w its first row, ld bytes a row; false: another format
bool q5k_rows(const x3::Mat& m, const uint8_t** w, size_t* ld);
/// The int8 rounding mv gives a Q8_0 / low-bit job's rows of k F32 values (k apart): q8_quant's, through the batched
/// launch as mv makes it (one item) - what the load-time checks compare a fused rounding against.
void q8_round(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s);
/// The columns [c0, c0 + nc) of such a matrix: its rows c0 .. c0 + nc - 1, a contiguous range.
x3::Mat view(const x3::Mat& m, int c0, int nc);
/// x3::reconstruct_rows' layout from a BF16 or Q8_0 matrix: dst[r][c] = W[c][r] (n rows of k values), F16 or BF16
/// bits (a Q8_0 weight rounded to them).
void rows(const x3::Mat& m, uint16_t* dst, bool bf16, cudaStream_t s);

/// x3::moe_gate_up / moe_down / win_experts / set_gate_up / set_down for a GGUF's experts (lay.fmt == kFmtMXFP4; the
/// roles' types in lay.K, see expert_type).
void moe_gate_up(const x3::MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const x3::MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
/// DS4_HELPER_OVERLAP (docs/dsv41/cards/HX-10.md): moe_down in its two halves, the same launches in the same order on the
/// same stream (so the same bits) - moe_down_part: h's int8 rounding and the down rows into the stream's scratch part;
/// moe_combine: the per-token combine of that part with the CPU lane's / helpers' part and the shared expert's.  The caller
/// may put the wait for the CPU lane's answer between them (nothing in moe_down_part reads cpu_part).
void moe_down_part(const x3::MoeArgs& a, const float* h, cudaStream_t s);
void moe_combine(const x3::MoeArgs& a, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const x3::WinArgs& a, cudaStream_t s);
/// DS4_HELPER_DD: win_experts' combine alone (after a win_experts with a.no_combine): the same launch it would have made
/// (a.part the rows win_experts wrote; a.cpu_part / cpu_stride / cpu_flag as given here; a.wait_ptr ignored).
void win_combine(const x3::WinArgs& a, cudaStream_t s);
/// DS4_LP_JOINCOMB (docs/dsv41/cards/HX-21.md): ds4_model.cu's dd_join_k arguments (DdJoin and its launch's), for the
/// join and the combine after it in one launch on the lead: moe_join_combine (a decoded row: moe_combine's sum over the
/// stream's scratch part, after moe_down_part) and win_join_combine (a window: win_combine's, after win_experts with
/// no_combine; a.cpu_part / cpu_flag unused).  The joined row stands where the combine read dd_join_k's part; part / flag
/// (may be null) are written as dd_join_k writes them.  Bit for bit dd_join_k + the combine.  false: nothing launched.
struct DdJoinArgs {
    const unsigned long long* seq[8] = {};
    const int* has[8] = {};
    const float* out[8] = {};
    int nh = 0;
    const int8_t* own = nullptr;   // (null: every helper waited for)
    int nexp = 0;
    unsigned long long* cnt = nullptr;
    const unsigned* rseq = nullptr;   // the lane's answer word, and the request
    unsigned want = 0;
    const float* lane_part = nullptr;   // the lane's rows (lane_stride apart) and flags
    size_t lane_stride = 0;
    const int* lane_flag = nullptr;
    const int* ridx = nullptr;   // the request's routing on this device ([t][8])
    int il = 0, kexp = 0;
    float* part = nullptr;   // dd_join_k's outputs
    int* flag = nullptr;
};
bool moe_join_combine(const x3::MoeArgs& a, const DdJoinArgs& j, const float* sh_out, float* out, cudaStream_t s);
bool win_join_combine(const x3::WinArgs& a, const DdJoinArgs& j, cudaStream_t s);
/// DS4_LP_FLAGFUSE (a helper's DS4_HELPER_DD body): win_combine, then *flag = *val (helper_flag_kernel's store, between
/// system fences) by the launch's last block, once every block's rows are out; cnt: a device word, 0 between launches.
void win_combine_flag(const x3::WinArgs& a, unsigned long long* flag, const unsigned long long* val, unsigned* cnt,
                      cudaStream_t s);
/// helperwin DS4_HW_HEADQ8 (docs/dsv41/cards/HX-22.md): the stream's int8 x rows and scales (the scratch win_experts
/// rounds x into: row r at xq + r k, its scales at xd + r k / 32), created as prepare() does; false: none.  And win_experts'
/// own rounding of x (q8_quant_kernel, rows ldx apart) as a launch of its own - what the head's rounding is checked against.
bool q8_scratch(cudaStream_t s, int8_t** xq, float** xd);
void q8_rows(const float* x, size_t ldx, int rows, int k, int8_t* q, float* d, cudaStream_t s);
void set_gate_up(const x3::SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s);
void set_down(const x3::SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s);

/// Launch failures of these kernels since the process started (each is also printed).
int launch_errors();

/// PF-12 (DS4_PF_TILE): whether a prompt-helper layer of `rows` rows (every expert's) runs in one pass
/// (ds4_iq's pft3_choose; GGUF low-bit experts of the four pack types only, gate type == up type), and its arms.
bool pf_tile_choose(const x3::ExpertLayout& lay, int n_embd, int n_ff, int rows, int* agu, int* adn);
/// One set of such a layer in one pass (gu: x / row_idx -> h16; else h16in -> y), every expert's rows contiguous.
bool pf_tile_set(const x3::SetArgs& sa, bool gu, const float* x, int ldx, const int* row_idx, const uint16_t* h16in,
                 uint16_t* h16, int ldh, float* y, int ldy, int nrows, int arm, cudaStream_t s);

}  // namespace strata::ds4::gguf
