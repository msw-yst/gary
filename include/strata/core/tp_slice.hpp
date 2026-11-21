// include/strata/core/tp_slice.hpp - --tensor-parallel N: which rows / columns of each dense weight a card holds.
//
// The split is tp_proof_bench's (src/kernels/tp_proof_bench.cpp load_part, measured exact on 2 XTX):
//
//   GDN layers: card c holds key groups [c * HK/N, (c+1) * HK/N) and the value heads that read them.  Value head i of
//     the card's slice is original head kset[i % Gk] + HK * (i / Gk), so the kernels' modulo pairing (value head i reads
//     key head i % h_k) still holds on the card.  attn_qkv rows (q | k | v channels of those heads), attn_gate rows,
//     ssm_alpha / ssm_beta rows, ssm_dt / ssm_a entries and ssm_conv1d rows follow the heads; ssm_out is split by its
//     input COLUMNS (the card's value heads), so its output is a partial sum (all-reduced by the window).
//   Attention layers: card c holds query heads [c * NH/N, ...) (attn_q rows: per head [q | gate]) and the KV heads they
//     read (attn_k / attn_v rows); attn_output is split by its input columns (a partial sum, all-reduced).
//   Everything else (norms, hyper-connections, the indexer, the router, the shared expert, embeddings, head) is whole on
//   every card.  The routed experts are whole too and split by owner: expert e lives on card e % N.
//
// The geometry a card runs is the model's with its head counts divided (`tp_geometry`), so the verify window's kernels
// see a smaller model and need no other change.
#pragma once

#include "strata/core/geometry.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct TpSlice {
    int tp = 1, card = 0;
    // the WHOLE model's shape (the slices are computed from it)
    int64_t HK = 0, HV = 0, S = 0, NH = 0, NKV = 0, HD = 0, n_layers = 0;
    bool on() const { return tp > 1; }

    using Ranges = std::vector<std::pair<int64_t, int64_t>>;
    /// The key groups / value heads (slice order) / conv channels (q | k | v, slice order) of a GDN layer on this card.
    std::vector<int64_t> gdn_kset() const;
    std::vector<int64_t> gdn_vlist() const;
    std::vector<int64_t> gdn_chans() const;
    /// The attention heads of this card: query heads [h0, h0 + nh), KV heads [kv0, kv0 + nkv).
    int64_t attn_h0() const { return card * (NH / tp); }
    int64_t attn_nh() const { return NH / tp; }
    int64_t attn_nkv() const { return NKV >= tp ? NKV / tp : 1; }
    int64_t attn_kv0() const { return NKV >= tp ? card * attn_nkv() : attn_h0() / (NH / NKV); }

    enum class Axis { None, Rows, Cols };
    /// How the tensor `name` (a `blk.<l>.` dense weight, GGUF naming) is split on this card: by rows (the outer axis,
    /// ne1; for a 1-D tensor its elements) or by columns (ne0), and the ranges this card keeps, in order (merged).
    /// Axis::None: whole on every card.
    Axis plan(const std::string& name, Ranges& out) const;
};

/// The process-wide setting (--tensor-parallel / STRATA_TP): set once at startup, before any load.  The card is
/// changed per device as each card's weights load (`set_card`).
TpSlice& tp_slice();

/// The geometry card `s.card` runs: the head counts and the widths derived from them divided by s.tp.
ModelGeometry tp_geometry(const ModelGeometry& g, const TpSlice& s);

/// Host bytes of rows `r` (row = `row_bytes` contiguous bytes) of `p`, concatenated.
std::vector<uint8_t> tp_take_rows(const uint8_t* p, size_t row_bytes, const TpSlice::Ranges& r);
/// Host bytes of columns `c` of every one of `ne1` rows (each range whole blocks of `blk_elems` / `blk_bytes`).
std::vector<uint8_t> tp_take_cols(const uint8_t* p, int64_t ne1, size_t row_bytes, int blk_elems, int blk_bytes,
                                  const TpSlice::Ranges& c, std::string& err);

class WeightTable;
/// The arena's (canonical) rows of the split tensors that are not native projections (ssm_alpha / ssm_beta / ssm_dt /
/// ssm_a / ssm_conv1d) replaced, on the CURRENT device, by copies of this card's rows; the table points at them.
/// Returns the bytes allocated (the whole rows stay in the arena: they are small).
bool tp_slice_arena(WeightTable& wt, const TpSlice& s, uint64_t& bytes, std::string& err);

}  // namespace strata::core
