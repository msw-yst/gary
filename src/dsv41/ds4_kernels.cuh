// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_kernels.cuh - DeepSeek-V4.1's own device kernels for a window of T tokens (1 <= T <= kWin) at positions
// p0 .. p0 + T - 1 (the EXL3 matrices go through strata::dsv41k::exl3).  The math is DeepSeek's inference/model.py
// (MIT); tools/ds4/ref.py is the oracle the engine is checked against.  Every pointer is device memory, every activation
// F32, rows of a window back to back; the KV caches keep the reference's FP8 / FP4 roundings packed as DeepSeek stores
// them (codes and scales, kv_row_bytes), the window attention's working rows BF16 (those values are exact in BF16).
// What differs from query to query (its window, how much of the compressed KV it sees, how many rows it selects) is
// known on the host and passed by value.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::ds4 {

constexpr int kWin = 16;      // tokens a window takes at most (the EXL3 GEMV's activations per job)
constexpr int kPicks = 8;     // routed experts a token can take at most

/// The caches' rows: DeepSeek's own roundings kept as codes and scales - lossless, the values are on those grids already
/// (the numbers are Finish::quant's; a finish with that quant and a pdst writes the format), a row of d values in
/// kv_row_bytes(fmt, d) bytes, padded to 16 (the hand-offs copy 16-byte units):
///   kKvFp8   the window ring: d E4M3 codes, then d / 32 power-of-two scales (UE8M0: the F32 exponent byte)
///   kKvFp4   the compressed KV: d / 2 bytes of E2M1 codes (value 2i in the low nibble), then d / 16 E4M3 scales
///   kKvFp4p  the index keys: d / 2 bytes of E2M1 codes, then d / 32 UE8M0 scales
/// kKvBf16: plain BF16 rows (a DSpark block's own rows, and the reference the tests check the packed rows against).
constexpr int kKvBf16 = 0, kKvFp8 = 1, kKvFp4 = 2, kKvFp4p = 3;
__host__ __device__ constexpr size_t kv_row_bytes(int fmt, int d) {
    return fmt == kKvBf16  ? (size_t) d * 2
           : fmt == kKvFp8 ? ((size_t) d + (size_t) d / 32 + 15) / 16 * 16
           : fmt == kKvFp4 ? ((size_t) d / 2 + (size_t) d / 16 + 15) / 16 * 16
                           : ((size_t) d / 2 + (size_t) d / 32 + 15) / 16 * 16;
}

/// h[t][s][e] = BF16 rows[t][e] for each of the hc streams.
void embed(const uint16_t* rows_bf16, int T, float* h, int dim, int hc, cudaStream_t s);

/// The per-vector finish: n_vec vectors of len values at x + v * stride, each in order
///   norm  (w != null): x = x * rsqrt(mean(x^2) + eps) * w
///   rope  (rd > 0): the last rd values as adjacent complex pairs rotated by pos_v * freqs[i] (inverse: the conjugate),
///         pos_v = pos + (v / vpp) * pos_step
///   quant 0 none, 1 FP8 E4M3 with a power-of-two scale per 32, 2 FP4 E2M1 with an E4M3 scale per 16, 3 FP4 with a
///         power-of-two scale per 32 (the reference kernels' act_quant / fp4_act_quant, inplace)
/// and written back to x, or as BF16 to dst + v * dst_stride when dst is given, or packed (quant > 0: the cache format of
/// that number, kv_row_bytes) to pdst + v * pdst_stride bytes when pdst is given (x then left as it was).
struct Finish {
    const float* w = nullptr;
    float eps = 1e-20f;
    int rd = 0, pos = 0, inverse = 0, vpp = 1, pos_step = 1;
    const float* freqs = nullptr;   // rd / 2 values
    int quant = 0;
    uint16_t* dst = nullptr;
    long long dst_stride = 0;
    uint8_t* pdst = nullptr;
    long long pdst_stride = 0;
    // (DS4_DECODE_TRIM bit 2, after Project Maya #512) non-null: the values written to x also rounded to int8 as gguf mv
    // rounds a Q8_0 / low-bit job's input (q8_quant's ops on the stored value) - q8 [n_vec][len], q8d [n_vec][len / 32];
    // len % 32 == 0, stride len, no dst / pdst (finish refuses otherwise: nothing launched, a line printed)
    int8_t* q8 = nullptr;
    float* q8d = nullptr;
};
void finish(float* x, int n_vec, int len, long long stride, const Finish& f, cudaStream_t s);

/// Hyper-Connections, row t of each [T][..] array: mixes[t][m] = (h_flat . fn[m]) * rsqrt(mean(h_flat^2) + eps), then
/// pre / post / comb as hc_split_sinkhorn (comb[t][j * hc + k], j the source stream, k the destination).  Strides: mixes
/// 32, pre / post 8, comb 64 floats a row.
void hc_mix(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s);
/// hc_mix then hc_split in one launch (cnt: T zeroed unsigned counters, left zero; n_mix <= 32, T <= kWin): the last block of a token
/// runs the split - the same arithmetic as the two launches
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes, unsigned* cnt,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre, float* post,
                  float* comb, cudaStream_t s);
void hc_split(const float* mixes, const float* scale, const float* base, int hc, int iters, float hc_eps, int T,
              float* pre, float* post, float* comb, cudaStream_t s);
/// the same two in one dispatch (the Metal kernel ds4_hc_mix_split); the CUDA path calls hc_mix then hc_split.
void hc_mix_split(const float* h, const float* fn, int hcdim, int n_mix, int T, float eps, float* mixes,
                  const float* scale, const float* base, int hc, int iters, float hc_eps, float* pre,
                  float* post, float* comb, cudaStream_t s);
/// x[t][e] = sum_s mix[t][s] h[t][s][e], then x = rmsnorm(x) * w (w null: no norm).
void hc_pre_norm(const float* h, const float* mix, const float* w, int dim, int hc, int T, float eps, float* x,
                 cudaStream_t s);
/// (DS4_HC_FUSE, after Project Maya #511's hc_split_pre_norm) hc_mix_split and hc_pre_norm in one launch - they share
/// nothing: the norm reads the lagged mix (`mix`, the split before this one's), the split writes this one's pre.  Grid
/// (n_mix + 1, T) of 1024 threads: blocks 0 .. n_mix - 1 are hc_mix_split_kernel's, their first 256 threads running its
/// code (the other threads add +0 into its block sums, as its own block_sum pads its 8 warps to 32: the same bits), the
/// last of them running the split; block n_mix is hc_pre_norm_kernel's (its 1024 threads, its code).  Extra, where
/// set: pre_copy - the 8 pre slots copied there too as the layer's end copy did them (the k < hc just written, the rest
/// pre's own bits); q / qd - x's int8 rounding, gguf mv's for a Q8_0 or low-bit job's input (q [T][dim], qd [T][dim/32]).
/// The old kernels stay as they are: hc_fuse_check (ds4_model.cu) compares the two paths byte for byte at load.
struct HcFuse {
    const float* h = nullptr;     // [T][hc][dim]
    const float* fn = nullptr;    // hc_mix_split's weights, n_mix rows of hc * dim
    int n_mix = 0;
    float eps = 0.0f;             // hc_mix's and hc_pre_norm's (the model's rms eps)
    float* mixes = nullptr;       // [T][32]
    unsigned* cnt = nullptr;      // T zeroed counters (left zero)
    const float *scale = nullptr, *base = nullptr;
    int hc = 4, iters = 1;
    float hc_eps = 0.0f;
    float *pre = nullptr, *post = nullptr, *comb = nullptr, *pre_copy = nullptr;
    const float* mix = nullptr;   // the norm's lagged mix, [T][8]; not pre, not pre_copy
    const float* w = nullptr;     // the norm's weights (null: none)
    int dim = 0;
    float* x = nullptr;           // [T][dim]
    int8_t* q = nullptr;
    float* qd = nullptr;
};
/// false, nothing launched: n_mix not 1 .. 32, hc not 1 .. 8, dim not a multiple of 32 up to 8192, T not 1 .. kWin, a
/// buffer missing, q without qd.
bool hc_mix_split_norm(const HcFuse& a, int T, cudaStream_t s);
/// xtxkern (docs/dsv41/cards/FG-08.md; FG-16 takes DS4_XK_QA alone of its kernel flags; default off, bit-identical by
/// construction, gfx1100 layer devices only): the process's bits, set once at load before any launch.  kXkQa - wo_a's GEMV
/// rounds its outputs for wo_b (xk_mv_q5k_q8).  The kernel is instantiated only in the gfx1100 device pass: the other code
/// objects are unchanged.
constexpr int kXkQa = 128;
void xk_set(int bits);
int xk_bits();
/// xtxkern DS4_XK_QA: Q5_K GEMV jobs (one activation: xq int8 [k], xd [k / 32] - q8_quant's) as ds4_iq's mv computes them
/// (the same per-row code, grid and stores: y[r] = alpha (w_r . x) + bias[r]), each output also rounded to int8 into q [n] /
/// qd [n / 32] with q8_quant's ops on the stored values.  false, nothing launched: kXkQa unset, not a gfx1100 HIP build,
/// n not 1 .. 4, a job's k not a multiple of 256 or rows not a multiple of 32, a buffer missing.
struct XkQaJob {
    const uint8_t* w = nullptr;   // row 0 of the Q5_K blocks, ld bytes a row
    size_t ld = 0;
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    int k = 0, n = 0;
    float* y = nullptr;
    const float* bias = nullptr;
    float alpha = 1.0f;
    int8_t* q = nullptr;
    float* qd = nullptr;
};
bool xk_mv_q5k_q8(const XkQaJob* jobs, int n, cudaStream_t s);
/// out[t][k][e] = post[t][k] y[t][e] + sum_j comb[t][j][k] res[t][j][e]  (out must not alias res).
void hc_post(const float* y, const float* res, const float* post, const float* comb, int dim, int hc, int T,
             float* out, cudaStream_t s);
/// hc_post with out = res (h), the same arithmetic: a chunk keeps one copy of its streams.
void hc_post_inplace(const float* y, float* h, const float* post, const float* comb, int dim, int hc, int T,
                     cudaStream_t s);
/// A chunk's hc_mix: the dots of each row with the n_mix weights already in mixes (a GEMM's, rows 32 apart), times
/// rsqrt(mean(h_row^2) + eps) - hc_mix's value up to the dots' summation order.
void hc_mix_scale(const float* h, int hcdim, int n_mix, int T, float eps, float* mixes, cudaStream_t s);

/// y[t][r] = alpha * sum_k W[r][k] x[t][k], W F16 bits [n][k] (the indexer's weights_proj).
void gemv_f16(const uint16_t* W, const float* x, int n, int k, int T, float alpha, float* y, cudaStream_t s);
/// The same for T activations x[t] = x + t ldx into y + t ldy, each weight read once for all of them; each output is
/// gemv_f16's bit for bit.
void gemv_f16_rows(const uint16_t* W, const float* x, size_t ldx, int n, int k, int T, float alpha, float* y, size_t ldy,
                   cudaStream_t s);
/// y[t ldy + r] = sum_k W[r][k] x[t][k] in F32 (W [n][k], k % 128 == 0): the router's logits, as DeepSeek's
/// inference computes them (linear(x.float(), weight.float())) - a warp a row, the same summing order for any T, so
/// the token, window and chunk paths get the same bits.
void gemv_f32(const float* W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s);
/// (DS4_ROUTE_FUSE, after Project Maya #511's gemv_f32_pair) gemv_f32 of two matrices on the same x in one launch:
/// y0 = x W0 (n0 rows), y1 = x W1 (n1 rows), for T rows of x (k apart) into rows ldy apart - each row gemv_f32's code (a
/// warp a row, a lane's float4s c = lane + 32 j in order, fmaf, the warp's sum), so its bits.  false, nothing launched:
/// n0 or n1 not 1 .. 1024, k not a positive multiple of 4, T not 1 .. kWin, ldy < max(n0, n1); false: the launch did not
/// go in.
bool gemv_f32_pair(const float* W0, int n0, float* y0, const float* W1, int n1, float* y1, const float* x, int k, int T,
                   int ldy, cudaStream_t s);
/// (DS4_ROUTER_BF16, after Project Maya #745) the router's weights on the device: F32 (f), or the pack's BF16 rows as
/// they are (h: half the bytes to read; each value widened to F32 exactly in the kernel, the same products summed in the
/// same order - the bits of an F32 copy of them).  One is set (none: no router).
struct GateW {
    const float* f = nullptr;
    const uint16_t* h = nullptr;
    bool operator==(std::nullptr_t) const { return f == nullptr && h == nullptr; }
    bool operator!=(std::nullptr_t) const { return !(*this == nullptr); }
};
/// gemv_f32 / gemv_f32_pair on such weights: F32 ones as above (the same kernels), BF16 ones by the same code on the
/// widened values.  A pair of one F32 and one BF16 router: each its own launch.
void gemv_f32(GateW W, const float* x, int n, int k, int T, float* y, int ldy, cudaStream_t s);
bool gemv_f32_pair(GateW W0, int n0, float* y0, GateW W1, int n1, float* y1, const float* x, int k, int T, int ldy,
                   cudaStream_t s);

/// The router, row t: scores = sqrt(softplus(logits[t])) (n experts, rows 1024 apart), the top `k` of scores + bias;
/// weights from the unbiased scores, normalized to sum 1 (+1e-20) and times route_scale -> idx[t][i], w[t][i] (rows
/// kPicks apart).  Rows of image tokens (bit t of vl) take bias_vl instead (training's noaux_tc_for_vl).
constexpr int kVlRows = 8192;
struct VlBits {
    uint32_t w[kVlRows / 32];
};
void route(const float* logits, const float* bias, int n, int k, int T, float route_scale, int* idx, float* w,
           cudaStream_t s, const float* bias_vl = nullptr, const VlBits* vl = nullptr);

/// h = silu(min(g, lim)) * clamp(u, -lim, lim)  (lim <= 0: no clamp); n values.
void swiglu(const float* g, const float* u, int n, float lim, float* h, cudaStream_t s);
/// (DS4_DECODE_TRIM bit 1, after Project Maya #512) swiglu, its stored values also rounded to int8 as gguf mv rounds a
/// Q8_0 / low-bit job's input: q [n], qd [n / 32] - 32-blocks of the flat array (rows of k values, k % 32 == 0, are rows
/// of q k apart).  false, nothing launched: n % 32 != 0, q or qd null.
bool swiglu_q8(const float* g, const float* u, int n, float lim, float* h, int8_t* q, float* qd, cudaStream_t s);
/// (DS4_DECODE_TRIM bit 4, after Project Maya #512) up to 4 copies of 4-byte multiples between 4-aligned device ranges
/// in one kernel (false, nothing launched: a range not so, or n not 1 .. 4).
struct CopyRegions {
    void* dst[4] = {nullptr, nullptr, nullptr, nullptr};
    const void* src[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t bytes[4] = {0, 0, 0, 0};
    int n = 0;
};
bool copy_regions(const CopyRegions& r, cudaStream_t s);

/// The ratio-r compressor over a window: token t (position p0 + t) takes slot (p0 + t) % r of the state; each token that
/// completes a group writes the group's pooled latent (softmax over the slots, per value, then rmsnorm(w)) to the next
/// row of latent.
void compress_win(const float* kv, const float* score, float* kv_state, float* score_state, int ratio, int p0, int T,
                  int d, const float* w, float eps, float* latent, cudaStream_t s);

/// Per-query sizes, by value - or (gen > 0, a chunk of any length) made on the device from the query's position: query
/// t at position p0 + t has nn = (p0 + t + 1) / R compressed positions and cb = min(ccap, ceil(nn / blk)) candidate
/// blocks;  gen 1: n = nn, k = min(kcap, nn)  (the indexer's scores and their top-k)
///          gen 2: n = nn, k = cb             (index_score over the candidates)
///          gen 3: n = cb * blk, k = min(kcap, nn)   (the top-k of the candidates' scores)
///          gen 4: n = ceil(nn / blk), k = kcap      (the candidate blocks' top-k)
struct Rows {
    int n[kWin];
    int k[kWin];
    int gen = 0, R = 1, kcap = 0, ccap = 0, blk = 1;
    long long p0 = 0;
};

/// The indexer's scores, row t: out[t][i] = sum_h relu(q[t][h] . k[p]) w[t][h] for i < n (the widest query's), p = i, or
/// with candidates p = cand[t][i / blk] * blk + i % blk for i < lim.k[t] * blk; positions >= lim.n[t] (or past the
/// query's candidates) -> -inf.  q rows heads * hd apart, w rows 64, cand rows cand_stride, out rows out_stride; the keys
/// k rows of format kfmt (the cache's kKvFp4p, or kKvBf16).
void index_score(const float* q, const float* w, const void* k, int heads, int hd, int T, int n, const int* cand,
                 int cand_stride, int blk, const Rows& lim, float* out, long long out_stride, cudaStream_t s,
                 int kfmt = kKvFp4p);
/// The top r.k[t] of vals[t][0, r.n[t]) (row t at vals + t * stride): their positions (unordered; with candidates
/// cand[t][i / blk] * blk + i % blk) into out[t][0, k).  r.k[t] >= r.n[t] takes them all.
/// (DS4_TOPK_MB) mb != nullptr, r.gen == 0, T <= mb_rows and the widest row's n >= mb_min: topk_mb instead (the
/// same out bits); mb nullptr (the default): the one-workgroup kernel, as before.
void topk(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
          int* out, int out_stride, cudaStream_t s, unsigned* mb = nullptr, int mb_rows = 0, int mb_min = 0);
/// (DS4_TOPK_MB) the multi-block top-k, r.gen == 0 rows only: up to kTopkMbMaxG workgroups a row, the k-th largest
/// key by three histogram levels (key bits 31..20, 19..8, 7..0), then every key above it and the lowest-position keys
/// equal to it, written in index order - the one-workgroup kernel's out[t][0, k) bit for bit (its rule: the same k-th
/// key, the ties kept are the lowest positions, the keys above it first in index order, then the ties in index order).
/// mb: T rows of kTopkMbRow unsigned (zero on entry; left zero on exit), on the stream's device, used by one stream.
constexpr int kTopkMbRow = 8704, kTopkMbMaxG = 64;
void topk_mb(const float* vals, long long stride, int T, const Rows& r, const int* cand, int cand_stride, int blk,
             int* out, int out_stride, unsigned* mb, cudaStream_t s);
/// Level one of the candidate pre-filter, row t: block b's score = the max over its blk positions of the r.n[t] the
/// query sees; the block of its newest position is pinned; the top r.k[t] blocks -> out[t] (bm: scratch rows bm_stride).
/// (mb, mb_rows, mb_min: passed to its topk)
void cand_blocks(const float* score, long long stride, int T, const Rows& r, int blk, float* bm, long long bm_stride,
                 int* out, int out_stride, cudaStream_t s, unsigned* mb = nullptr, int mb_rows = 0, int mb_min = 0);

/// Sparse attention, query t: heads x d values of q[t] against win rows win0[t] .. win0[t] + nwin[t] - 1 of the window
/// buffer (d BF16 each) and the nsel[t] compressed rows ckv[sel[t][i]] (format ckv_fmt: the cache's kKvFp4, or kKvBf16);
/// K = V, a learned sink logit per head.  The packed rows, the same bits either way: up to kAttnScratchT queries (with
/// a scratch, heads_per_block 0; or -1, any T) decoded once a query into scr's BF16 rows for the BF16 rows' kernel;
/// else a kernel that decodes each staged row once for heads_per_block heads of a query (1, 2, 4, 8; 0: as many as
/// keep two blocks a multiprocessor).
struct AttnScratch {
    uint16_t* rows = nullptr;   // [max_t * cap][512] BF16
    int max_t = 0, cap = 0;     // queries, rows a query (at least the sel_stride)
};
constexpr int kAttnScratchT = 7;   // (the staged kernel wins from 8 queries: V100, Sun)
struct AttnRows {
    int win0[kWin], nwin[kWin], nsel[kWin];
    // gen: query t is a chunk's row t0 + t at position p0 + t0 + t (the window buffer: the ring's win rows before p0,
    // then the chunk's), selecting min(kcap, (its position + 1) / R) compressed rows when R > 0
    int gen = 0, t0 = 0, win = 0, R = 0, kcap = 0;
    long long p0 = 0;
};
void sparse_attn(const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                 const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out, cudaStream_t s,
                 int ckv_fmt = kKvFp4, int heads_per_block = 0, const AttnScratch* scr = nullptr, int layer = -1,
                 bool sel_done = false);   // (DS4_ATTN_MERGE=3: true - finish_multi_sel already ran this call's unpack)
/// (docs/dsv41/cards/PF-02.md; HIP builds): the test hook of tests/dsv41/fattn_test.cu - the chunked prompt's packed-KV
/// attention (a.gen != 0, kKvFp4 rows, d 512) with the kernel forced: mode -1 the scalar sparse_attn_packed_kernel, 0 .. the
/// sparse_attn_wmma_kernel variants of DS4_FAST_ATTN_MODE (fast_attn_mode_name(mode): its name, null past the last).
/// False: the shape does not take the fast kernel (T < 16, heads not a multiple of 16).
bool sparse_attn_forced(int mode, const float* q, const uint16_t* win, const void* ckv, const int* sel, int sel_stride,
                        const AttnRows& a, const float* sink, int heads, int T, float scale, float* out, cudaStream_t s);
const char* fast_attn_mode_name(int mode);
/// the engine runs tensor-parallel groups (a group of more than one member) - DS4_FAST_ATTN's default is on for gfx1100
/// there (mode f32; docs/dsv41/cards/PF-02.md).  Set once at setup, before the first prompt; DS4_FAST_ATTN=0 / 1 still wins.
void set_fast_attn_tp(bool tp_groups);
/// The window buffer: ext[r] = ring[(p0 - n + r) % n] for the n rows before p0 (zeros before position 0); afterwards
/// ring[(p0 + t) % n] = ext[n + t] for t < T.  The ring's rows kKvFp8 (each group of 32 at the smallest power-of-two
/// scale its BF16 values fit under: exact for values on the FP8 grid, the window's), the window buffer's BF16 (p0 = n:
/// the ring's rows as they lie).
void ring_to_ext(const uint8_t* ring, uint16_t* ext, int64_t p0, int n, int d, cudaStream_t s);
/// rows strided copy by a kernel, `rows` rows of `width` bytes from src (pitch sp: device memory, or pinned host
/// memory's mapped address - the TP prompt stream's down parts, DS4_TP_DIRECT) to dst (pitch dp).  One launch; the widest
/// word all of the addresses, pitches and the width allow.
void tp_strided_rows(uint8_t* dst, size_t dp, const uint8_t* src, size_t sp, size_t width, int rows, cudaStream_t s);
void ext_to_ring(const uint16_t* ext, uint8_t* ring, int64_t p0, int n, int T, int d, cudaStream_t s,
                 uint8_t* backup = nullptr);
/// A speculative window's rollback: ring[(p0 + t) % n] = backup[t] for keep <= t < T (the rows ext_to_ring replaced).
void ring_restore(uint8_t* ring, const uint8_t* backup, int64_t p0, int n, int keep, int T, int d, cudaStream_t s);
/// The same for up to kMaxRingBatch layers' rings at once (one launch).
constexpr int kMaxRingBatch = 64;
struct RingBatch {
    uint8_t* ring[kMaxRingBatch] = {};
    const uint8_t* bak[kMaxRingBatch] = {};
    int n = 0;
};
void ring_restore_batch(const RingBatch& b, int64_t p0, int n, int keep, int T, int d, cudaStream_t s);

/// DS4_MULTI_ATTN (docs/dsv41/cards/FG-05.md; ported from Project Maya #639, MIT - third_party/project-maya/NOTICE.md): a
/// batch step's attention for the rows of several sequences in one launch each, where attn_batch launches it once a
/// sequence.  A run is one sequence's contiguous rows; each run's window buffer is its own block of one shared buffer
/// (ds4_multi_attn.hpp).  Every kernel below is the single-run kernel's code with only its row / run indexing changed,
/// so each row's bits are the single-run launch's.  HIP / CUDA only (Metal: not built, the caller keeps the per-run path).
///
/// The few-queries path of sparse_attn (kKvFp4, a.gen 0) for T rows of several sequences: row t's compressed rows from
/// its own cache ckv[t] (null with nsel 0), its window rows a.win0 / a.nwin of `win`, its selection sel[t], its scratch
/// rows scr's t-th - unpack_sel_kernel then sparse_attn_kernel<true, 0> (non-HIP `one`: <true, 8>, the T = 1 kernel).
struct AttnCkv {
    const void* ckv[kWin] = {};
};
/// whether sparse_attn takes its few-queries path for every run of a step - runs of at most kAttnScratchT rows,
/// DS4_ATTN_SPLIT off - and sparse_attn_rows can take all T rows in one launch (T <= scr->max_t, sel_stride <=
/// scr->cap).  mixed: the runs are neither all of one row nor all of several (non-HIP builds pick the T = 1 kernel by
/// run, so they keep the per-run path then)
bool sparse_attn_rows_ok(int T, int max_run, int sel_stride, const AttnScratch* scr, bool mixed);
void sparse_attn_rows(const float* q, const uint16_t* win, const AttnCkv& c, const int* sel, int sel_stride,
                      const AttnRows& a, const float* sink, int heads, int d, int T, float scale, float* out,
                      cudaStream_t s, bool one, const AttnScratch* scr);
/// ring_to_ext / ext_to_ring for every run at once: run k has T[k] rows from position p0[k], its ring ring[k], its
/// window buffer rows e0[k] .. e0[k] + n + T[k] - 1 of `ext`, its backups bak[k] (or null)
struct RingRuns {
    uint8_t* ring[kWin] = {};
    uint8_t* bak[kWin] = {};
    long long p0[kWin] = {};
    int e0[kWin] = {}, T[kWin] = {};
    int runs = 0;
};
void ring_to_ext_runs(const RingRuns& r, uint16_t* ext, int n, int d, cudaStream_t s);
void ext_to_ring_runs(const RingRuns& r, const uint16_t* ext, int n, int d, cudaStream_t s);
/// finish over the rows of several sequences: vector v is row v / f.vpp's, which takes the rope position pos[row]
/// (f.pos and f.pos_step not read) and, with f.dst, writes to f.dst + (dst[row] f.vpp + v % f.vpp) f.dst_stride.  No
/// pdst (false, nothing launched; also more than kWin rows)
struct FinishRows {
    int pos[kWin] = {};
    int dst[kWin] = {};
};
bool finish_rows(float* x, int n_vec, int len, long long stride, const Finish& f, const FinishRows& r, cudaStream_t s);

/// Engram.  rows: n rows of hd values -> out F32 [n * hd].  fmt, the pack's row layout: 0 FP8 E4M3 values with E8M0
/// scales (scales: n x (hd / 32)); 1 ggml Q6_K superblocks (hd / 256 of 210 bytes at the start of each row's hd bytes;
/// no scales), decoded as ggml's dequantize_row_q6_K; 2 ggml IQ4_NL blocks (hd / 32 of 18 bytes, an fp16 d and 16
/// nibble bytes at the start of each row's hd bytes; no scales), decoded as ggml's dequantize_row_iq4_nl.
void engram_dequant(const uint8_t* rows, const uint8_t* scales, int n, int hd, float* out, cudaStream_t s, int fmt = 0);
/// The same over the host's per-token layout (a chunk): token t's cols rows, then their scales, `per` bytes a token;
/// T tokens -> out F32 [T * cols][hd].
void engram_dequant_tok(const uint8_t* src, size_t per, int cols, int T, int hd, float* out, cudaStream_t s,
                        int fmt = 0);
/// h[t][k] += gate[t][k] * value[t] with the gate from the normalized dot of h[t][k] * w[k] * key[t][k]
/// (kv[t] = [key hc x dim | value dim]).
void engram_gate(float* h, const float* kv, const float* w, int hc, int dim, int T, float eps, cudaStream_t s);

/// out[r] = x[src[r]] for r < n (rows of cols floats).
struct Gather {
    int src[kWin * kPicks];
};
void gather_rows(const float* x, const Gather& g, int n, int cols, float* out, cudaStream_t s);
/// The MoE's sum, token t: out[t] = sum_j w[t][j] ed[row[t][j]] (its VRAM experts, in pick order) + cpu[t] (when its
/// cpu flag is set) + sh[t].
struct Combine {
    int n[kWin];
    int row[kWin][kPicks];
    float w[kWin][kPicks];
    int cpu[kWin];
};
void combine(const float* ed, const float* cpu, const float* sh, const Combine& c, int T, int dim, float* out,
             cudaStream_t s);

/// The mailboxes of a small window (decode, a draft's verification, the drafter's block): host-mapped memory, a ring of
/// kRing.  The device publishes a layer's routing and its FFN inputs for the window's T tokens (seq written last, after
/// a system fence); the host answers with the CPU lane's weighted part of each token that took a CPU-lane expert.
constexpr int kRing = 64;
constexpr int kMaxDim = 8192;
constexpr int kAhead1 = 16, kAhead = kAhead1 + kPicks;   // a token's guesses for the next layer, then the one after
struct Req {
    unsigned seq;
    int T;
    int na[2];   // lookahead: ahead[t][0 .. na[0]) the next layer's likely picks, ahead[t][kAhead1 ..) +na[1] the one after's
    int ids[kWin][kPicks];
    float w[kWin][kPicks];
    int ahead[kWin][kAhead];
    float x[kWin][kMaxDim];
};
struct Resp {
    unsigned seq;
    int pad[3];
    int cpu[kWin];   // 1: part[t] is the CPU lane's weighted sum for token t
    float part[kWin][kMaxDim];
};
/// route() for the T tokens of a window, plus: plan_ptr[t][i] = vram[idx[t][i]] (0: not on the device), each token's x
/// (dim floats) and picks into the request slot, then T and seq (req null: no request, the plan alone).
void route_publish(const float* logits, const float* bias, int n, int k, int T, float route_scale,
                   const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                   unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl = nullptr,
                   unsigned vl = 0, int na0 = 0, int na1 = 0);
/// DS4_LP_ROUTE / DS4_LP_PUBFUSE (docs/dsv41/cards/HX-21.md): route_publish with (fast) the k best found per warp and
/// merged by one warp - the old rounds' picks in their order whenever the k best scores are above -FLT_MAX and none is NaN,
/// else the old rounds in the same launch - and (pub, a request of one row) the publish's stores at the end of the routing's
/// launch instead of publish_kernel's own.  The same bytes everywhere; both false (or n / k out of range): route_publish.
void route_publish_lp(bool fast, bool pub, const float* logits, const float* bias, int n, int k, int T, float route_scale,
                      const unsigned long long* vram, const float* x, int dim, int* idx, float* w,
                      unsigned long long* plan_ptr, Req* req, unsigned seq, cudaStream_t s, const float* bias_vl = nullptr,
                      unsigned vl = 0, int na0 = 0, int na1 = 0, unsigned* pub_cnt = nullptr);
/// (helperwin DS4_HW_PUBT: pub_cnt, a device word that is 0 between launches - given, PUB publishes a window of T > 1
/// rows too, from the routing launch's last block to finish; null: T > 1 keeps publish_kernel's launch)
/// Lookahead: the m picks a later layer will likely make (its gate's logits on this layer's FFN input, the routing's
/// score + bias, the m best of n) into req->ahead[t][off ..) for the T tokens - before route_publish, which publishes them.
void route_ahead(const float* logits, const float* bias, int n, int m, int T, Req* req, int off, cudaStream_t s);
/// (DS4_ROUTE_FUSE, after Project Maya #511's route_pair) route_publish's routing and route_ahead's guesses for the
/// next layer in one launch of two blocks a row (block 0 route_publish_kernel's code: idx / w / plan_ptr and the
/// request's ids / w / x; block 1 route_ahead_kernel's: req->ahead[t][off ..) - disjoint writes, the same reads), then
/// publish_kernel as route_publish launches it (na0 / na1, T, seq).  The caller passes bias_vl only with vl != 0, as
/// route_publish does.  false, nothing launched: no request, n or a_n not 1 .. 1024, k not 1 .. kPicks, m < 1 or off + m
/// past kAhead, dim not 1 .. kMaxDim, T not 1 .. kWin; false, nothing published: the routing's launch did not go in;
/// false: the publish's launch did not go in.
struct RoutePair {
    const float *logits = nullptr, *bias = nullptr;   // route_publish's
    int n = 0, k = 0;
    float route_scale = 1.0f;
    const unsigned long long* vram = nullptr;
    const float* x = nullptr;
    int dim = 0;
    int* idx = nullptr;
    float* w = nullptr;
    unsigned long long* plan_ptr = nullptr;
    Req* req = nullptr;
    const float* bias_vl = nullptr;
    unsigned vl = 0;
    unsigned seq = 0;
    int na0 = 0, na1 = 0;
    const float *a_logits = nullptr, *a_bias = nullptr;   // route_ahead's
    int a_n = 0, m = 0, off = 0;
    int T = 1;
};
bool route_pair(const RoutePair& a, cudaStream_t s);
/// The RAM store's picks for the device (a decode token's k plan pointers): those the VRAM table left at 0 whose expert zc_tab has (the
/// slot's device address) are copied into stage slot i (stride apart, `bytes` each) and the plan then points there.
void zc_stage(unsigned long long* plan, const int* idx, const unsigned long long* zc_tab, int k, uint8_t* stage, size_t stride, size_t bytes,
              cudaStream_t s);
/// Waits (one thread, on the device) until *seq_ptr == seq.
void wait_seq(const unsigned* seq_ptr, unsigned seq, cudaStream_t s);
/// A chunk's routed experts as sorted rows: the picks (t, i < k) of T tokens (idx rows kPicks apart) by the virtual
/// slot vs[expert] (n_vs of them: the VRAM slots, then the experts streamed in), each slot's rows in token order:
/// bounds[n_vs + 1], row_tok[r], pos[t][i] (the pick's sorted row, rows kPicks apart).
void moe_sort(const int* idx, int T, int k, const int* vs, int n_vs, int* bounds, int* row_tok, int* pos,
              cudaStream_t s);
/// acc[t] += w[t][i] * y[pos[t][i] - r0] for each pick of t (in pick order) whose sorted row is in [r0, r1).
void moe_combine(float* acc, const float* y, const int* pos, const float* w, int T, int k, int r0, int r1, int dim,
                 cudaStream_t s);
/// rows[t][0] = 1, the rest 0 (the first layer's pre-mix: stream 0).
void init_pre(float* pre, int T, cudaStream_t s);

/// out[t][e] = mean over the hc streams of h[t][s][e] (the DSpark drafter's taps), rows out_stride floats apart.
void hc_mean(const float* h, int T, int hc, int dim, float* out, long long out_stride, cudaStream_t s);
/// out[t][i] = i for i < n, rows `stride` apart (the drafter's block rows as every query's selection).
void iota_rows(int* out, int T, int n, int stride, cudaStream_t s);
/// The DSpark markov head: logits[v] += E[*tok] . H[v] for v < V (E BF16 [V][R], H F16 [V][R]).
void markov_add(float* logits, const uint16_t* E, const uint16_t* H, const int* tok, int V, int R, cudaStream_t s);
/// The same bias with the head quantised to Q8_0 blocks of 32 (34 B a block: F16 d and 32 int8, R a multiple of 32).
void markov_add_q8(float* logits, const uint16_t* E, const uint8_t* Hq, const int* tok, int V, int R, cudaStream_t s);
/// The markov bias over a head group: a member's rows [v0, v0 + vn) of it added to its share of a logits row (erow:
/// the previous token's E row, BF16, on the device) - markov_add's sums; the share's argmax (the greedy sampler's rule)
/// into a host-mapped slot {value, index, seq}; every member's slot of the step -> the global argmax into *tok and its
/// E row (E host-mapped) into erow.
void markov_share(float* logits, const uint16_t* H, const uint16_t* erow, int vn, int R, cudaStream_t s);
void argmax_share(const float* row, int vn, int v0, unsigned* slot, unsigned seq, cudaStream_t s);
void markov_pick(const unsigned* slots, int nm, unsigned seq, int* tok, const uint16_t* E, uint16_t* erow, int R,
                 cudaStream_t s);
/// The DSpark confidence: out[t] = sigmoid(w . [x[t] | E[tok[t]]]) (x rows dim floats, w F32 [dim + R]).
void confidence(const float* x, const uint16_t* E, const int* tok, const float* w, int T, int dim, int R, float* out,
                cudaStream_t s);

/// The pipeline's hand-offs and tensor parallelism, through host-mapped memory (these cards copy peer to peer only
/// through the driver's staging).  xfer_put copies up to 4 segments (16-byte multiples, 16-byte aligned) device ->
/// host-mapped, then (flag given) sets *flag = seq after a system fence (cnt: a device counter of the stream, 0
/// between uses); xfer_get waits on the device until *flag >= seq, then copies host-mapped -> device.
struct Xfer {
    const void* src[4] = {nullptr, nullptr, nullptr, nullptr};
    void* dst[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t bytes[4] = {0, 0, 0, 0};
    int n = 0;
};
void xfer_put(const Xfer& x, unsigned* flag, unsigned seq, unsigned* cnt, cudaStream_t s);
// Windows (WDDM, CUDA and HIP alike) holds launches back until the host waits or queries the stream: the work queued on
// `s` submitted now (cudaStreamQuery, which does not wait).  Elsewhere nothing.  DS4_WIN_SUBMIT=0: off (A/B tests)
void win_submit(cudaStream_t s);
void xfer_get(const Xfer& x, const unsigned* flag, unsigned seq, cudaStream_t s);
/// Waits (one thread) until flags[i * stride] >= seq for i < n (a pipelined chunk's readers have taken a buffer);
/// set_flag: *flag = v after a system fence (a reader's acknowledgement).
void wait_flags(const unsigned* flags, int n, int stride, unsigned seq, cudaStream_t s);
void set_flag(unsigned* flag, unsigned v, cudaStream_t s);
/// A tensor-parallel group's all-reduce: host-mapped rows data[member][slot] (stride floats), flags[member][slot]
/// (32 apart).  x (n floats, n % 4 == 0: this member's partial) -> the sum of the nm members' partials in member order,
/// the same bits on every member; seq counts the group's all-reduces (the same on each member), slot = seq % slots.
struct ArGroup {
    float* data = nullptr;
    unsigned* flags = nullptr;
    size_t stride = 0;
    int nm = 1, slots = 4;
    // three members or more: a reduce-scatter + all-gather (each member sums its chunk of every block's range, in
    // member order, then reads the others' sums) - [slots][stride] sums, [2][nm][slots][kArBlocks][8] flags
    float* sums = nullptr;
    unsigned* bflags = nullptr;
    // DS4_AR_P2PK=1 (a prompt chunk's group, three members or more, peer access between every pair): the all-reduce
    // as a reduce-scatter + all-gather of kernel stores straight into the peers' VRAM (ar_p2pk_kernel), no host memory.
    // On member m, all hipDeviceMallocUncached: pk_buf[m] [2 parities][nm - 1 senders][pk_cap] 16-byte units (a sender's
    // slot: its partial of m's chunk, then its own chunk's sum), pk_flags[m] [2][3 kinds][nm senders][pk_maxt tiles][8],
    // pk_err[m] the error words of a bounded wait.  pk_w: the chunk weights (member order; DS4_AR_P2PK_SPLIT), pk_tu the
    // tile in units.  p2pk 0: the host-mapped path; 1: on.  The decode windows' group (ArGroup ar) never has it
    static constexpr int kPkMax = 8;
    int p2pk = 0;
    void* pk_buf[kPkMax] = {};
    unsigned* pk_flags[kPkMax] = {};
    unsigned* pk_err[kPkMax] = {};
    long long pk_cap = 0;
    int pk_maxt = 0, pk_tu = 0;
    int pk_w[kPkMax] = {};
};
constexpr int kArBlocks = 512;   // (the flags' room; the blocks a launch takes: ar_blocks(), DS4_AR_BLOCKS)
void allreduce(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, cudaStream_t s);
/// DS4_AR_P2PK: the tile in 16-byte units (DS4_AR_P2PK_TILE KiB, default 64) and the blocks a launch takes at most
/// (DS4_AR_P2PK_BLOCKS, default 192) - process-wide, so the same on every member
int p2pk_tile_units();
int p2pk_blocks();
int p2pk_order();
/// DS4_AR_P2PK: the units (16 bytes: 4 floats F32, 8 halves F16) a call of n floats takes, 0 if p2pk cannot take it
long long p2pk_units(int n, bool f16);
/// DS4_AR_P2PK load-time self-test: one ar_p2pk_kernel all-reduce in the given precision whose waits give up (the
/// error word set) instead of spinning on; false if the call does not fit the group's buffers (nothing launched)
bool allreduce_p2pk_test(float* x, int n, const ArGroup& a, int mi, unsigned seq, bool f16, cudaStream_t s);
/// the host-mapped reduce-scatter (ar_rs_kernel / ar_rs16_kernel) in a forced precision - the self-test's reference
void allreduce_rs_ref(float* x, int n, const ArGroup& a, int mi, unsigned seq, bool f16, cudaStream_t s);
/// x[i] = y[i] = a random float of member m (seed; full mantissas, exponents past FP16's range at both ends)
void p2pk_fill(float* x, float* y, size_t n, unsigned seed, int m, cudaStream_t s);
/// out[0] += the i with a's bits != b's, out[1] = min(out[1], the first such i)  (out set to {0, ~0} beforehand)
void p2pk_cmp(const float* a, const float* b, size_t n, unsigned* out, cudaStream_t s);

// ---- DS4_AR_POST (docs/dsv41/cards/FG-07.md; default off).  A decode / verify window's all-reduce on the one-phase
// host path (DS4_AR_FUSE's xfer_put + ar_sum) with the launches after it folded in:
//   bit 1: hc_post (out = post * y + sum_j comb[j][k] h[j], hc_post_kernel's per-element code, on the summed value the
//          thread holds: the same bits) runs in the sum kernel - one launch fewer an all-reduce;
//   bit 2: the put too (each block puts its range of this member's partial and raises its own flag, as ar_rs_kernel's
//          first round does, then sums its range over the members in member order - ar_sum_kernel's chain, the same
//          bits) - one launch fewer again.  Three members or more (the per-block flags of the reduce-scatter's rows).
// false: the call is not one it takes (another path, a shape it does not cover) - nothing launched, the caller runs
// allreduce() (and hc_post) as before.
struct HcPost {
    const float* res = nullptr;    // h  [T][hc][dim]
    const float* post = nullptr;   // [T][8]
    const float* comb = nullptr;   // [T][64]
    float* out = nullptr;          // h2 [T][hc][dim]
    int dim = 0, hc = 0;
};
bool allreduce_post(float* x, int n, const ArGroup& a, int mi, unsigned seq, unsigned* cnt, const HcPost* hp, int mode,
                    cudaStream_t s);
/// The load-time check's run of allreduce_post's kernels on one device alone (nm 1, rows / flags scratch of the caller's,
/// device memory): x is its own sum, so it compares the folded hc_post (and the put's copy into rows) with hc_post's
/// launch.  false: not launched.
bool allreduce_post_selftest(float* x, int n, float* rows, unsigned* flags, const HcPost& hp, int mode, cudaStream_t s);

// ---- DS4_ATTN_MERGE (docs/dsv41/cards/FG-07.md; default off).  Independent launches of a decode window's attention
// as one launch, each block running its op's kernel body for its row (finish_kernel<false, kQ8>, ring_to_ext_kernel,
// ext_to_ring_kernel - the same per-element code): the same bits, fewer launches.  The ops of one call must not read
// what another writes.  false: an op finish() / ring_to_ext() / ext_to_ring() would refuse - nothing launched.
struct FinOp {
    int kind = 0;   // 0 finish (q8: f.q8 set), 1 ring_to_ext, 2 ext_to_ring
    // finish
    float* x = nullptr;
    int n_vec = 0, len = 0;
    long long stride = 0;
    Finish f;
    // ring_to_ext (ring -> ext) / ext_to_ring (ext -> ring, backup)
    const uint8_t* ring_r = nullptr;
    uint8_t* ring_w = nullptr;
    uint16_t* ext_w = nullptr;
    const uint16_t* ext_r = nullptr;
    uint8_t* backup = nullptr;
    long long p0 = 0;
    int n = 0, T = 0, d = 0;
};
constexpr int kFinOps = 3;
bool finish_multi(const FinOp* ops, int nops, cudaStream_t s);
// ---- DS4_ATTN_MERGE=3 (FG-16 item 3; default off: =2 and below never call these).  sparse_attn's few-queries
// unpack - unpack_sel_kernel: each query's selected compressed rows decoded to BF16 rows of the scratch - as one more kind of
// block in the second merged launch (q's RoPE + the KV row into the ring): its blocks run unpack_sel_kernel's body for their
// (x, y) block, the same code, so the same bits (the values are exact products anyway).  It reads ckv and sel and writes only
// the scratch rows, so it is independent of that launch's ops; the caller folds it only where nothing between that launch
// and sparse_attn writes ckv's selected rows, sel or the scratch.  HIP, gfx1100 code objects only (no gfx906 code).
struct SelOp {
    const uint8_t* ckv = nullptr;
    const int* sel = nullptr;
    int sel_stride = 0, cap = 0, T = 0;
    uint16_t* out = nullptr;
    AttnRows a;
};
/// whether sparse_attn(q, win, ckv, sel, sel_stride, a, sink, heads, d, T, scale, out, s, ckv_fmt, heads_per_block, scr)
/// launches unpack_sel_kernel (its few-queries path: kKvFp4 rows, d 512, DS4_ATTN_SPLIT off, a scratch that fits) on this
/// device (gfx1100); true: u holds that launch's arguments
bool sparse_attn_sel_op(const void* ckv, const int* sel, int sel_stride, const AttnRows& a, int d, int T, int ckv_fmt,
                        int heads_per_block, const AttnScratch* scr, SelOp& u);
/// finish_multi(ops, nops, s)'s launch with u's unpack blocks after the ops' blocks (each checked as finish_multi checks it);
/// false: nothing launched
bool finish_multi_sel(const FinOp* ops, int nops, const SelOp& u, cudaStream_t s);
/// DS4_ATTN_MERGE=3's load check reference: unpack_sel_kernel launched exactly as sparse_attn launches it (false: not launched)
bool unpack_sel_ref(const SelOp& u, cudaStream_t s);

}  // namespace strata::ds4
