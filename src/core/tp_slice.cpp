// src/core/tp_slice.cpp - see include/strata/core/tp_slice.hpp.
#include "strata/core/tp_slice.hpp"

#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {
TpSlice::Ranges merged(const TpSlice::Ranges& v) {
    TpSlice::Ranges o;
    for (const auto& r : v) {
        if (!o.empty() && o.back().second == r.first) o.back().second = r.second;
        else o.push_back(r);
    }
    return o;
}
bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}
}  // namespace

TpSlice& tp_slice() {
    static TpSlice s;
    return s;
}

std::vector<int64_t> TpSlice::gdn_kset() const {
    std::vector<int64_t> k;
    const int64_t Gk = HK / tp;
    for (int64_t j = 0; j < Gk; ++j) k.push_back(card * Gk + j);
    return k;
}

std::vector<int64_t> TpSlice::gdn_vlist() const {
    const std::vector<int64_t> k = gdn_kset();
    const int64_t Gk = (int64_t) k.size(), nv = HV / tp;
    std::vector<int64_t> v;
    for (int64_t i = 0; i < nv; ++i) v.push_back(k[(size_t) (i % Gk)] + HK * (i / Gk));
    return v;
}

std::vector<int64_t> TpSlice::gdn_chans() const {
    std::vector<int64_t> c;
    for (int64_t j : gdn_kset())
        for (int64_t d = 0; d < S; ++d) c.push_back(j * S + d);
    for (int64_t j : gdn_kset())
        for (int64_t d = 0; d < S; ++d) c.push_back(HK * S + j * S + d);
    for (int64_t h : gdn_vlist())
        for (int64_t d = 0; d < S; ++d) c.push_back(2 * HK * S + h * S + d);
    return c;
}

TpSlice::Axis TpSlice::plan(const std::string& name, Ranges& out) const {
    out.clear();
    if (!on() || name.rfind("blk.", 0) != 0) return Axis::None;
    // the MTP block (blk.<n_layers>, the drafter's attention layer) is not split: it runs whole on its card
    if (std::strtol(name.c_str() + 4, nullptr, 10) >= n_layers) return Axis::None;
    Ranges r;
    auto heads = [&](int64_t width) {
        for (int64_t h : gdn_vlist()) r.push_back({h * width, (h + 1) * width});
    };
    Axis a = Axis::None;
    if (ends_with(name, ".attn_qkv.weight") || ends_with(name, ".ssm_conv1d.weight")) {
        for (int64_t c : gdn_chans()) r.push_back({c, c + 1});
        a = Axis::Rows;
    } else if (ends_with(name, ".attn_gate.weight")) {
        heads(S);
        a = Axis::Rows;
    } else if (ends_with(name, ".ssm_out.weight")) {
        heads(S);
        a = Axis::Cols;
    } else if (ends_with(name, ".ssm_alpha.weight") || ends_with(name, ".ssm_beta.weight") ||
               ends_with(name, ".ssm_dt.bias") || ends_with(name, ".ssm_a")) {
        heads(1);
        a = Axis::Rows;
    } else if (ends_with(name, ".attn_q.weight")) {
        r.push_back({attn_h0() * 2 * HD, (attn_h0() + attn_nh()) * 2 * HD});
        a = Axis::Rows;
    } else if (ends_with(name, ".attn_k.weight") || ends_with(name, ".attn_v.weight")) {
        r.push_back({attn_kv0() * HD, (attn_kv0() + attn_nkv()) * HD});
        a = Axis::Rows;
    } else if (ends_with(name, ".attn_output.weight")) {
        r.push_back({attn_h0() * HD, (attn_h0() + attn_nh()) * HD});
        a = Axis::Cols;
    }
    if (a != Axis::None) out = merged(r);
    return a;
}

ModelGeometry tp_geometry(const ModelGeometry& g, const TpSlice& s) {
    if (!s.on()) return g;
    ModelGeometry t = g;
    t.ssm_k_heads = g.ssm_k_heads / s.tp;
    t.ssm_v_heads = g.ssm_v_heads / s.tp;
    t.ssm_value_dim = g.ssm_state_size * t.ssm_v_heads;
    t.ssm_conv_channels = 2 * g.ssm_state_size * t.ssm_k_heads + t.ssm_value_dim;
    t.n_head = s.attn_nh();
    t.n_head_kv = s.attn_nkv();
    return t;
}

std::vector<uint8_t> tp_take_rows(const uint8_t* p, size_t row_bytes, const TpSlice::Ranges& r) {
    std::vector<uint8_t> o;
    for (const auto& x : r) o.insert(o.end(), p + (size_t) x.first * row_bytes, p + (size_t) x.second * row_bytes);
    return o;
}

std::vector<uint8_t> tp_take_cols(const uint8_t* p, int64_t ne1, size_t row_bytes, int blk_elems, int blk_bytes,
                                  const TpSlice::Ranges& c, std::string& err) {
    int64_t w = 0;
    for (const auto& x : c) {
        if (x.first % blk_elems || x.second % blk_elems) {
            err = "a column split is not on a block boundary";
            return {};
        }
        w += x.second - x.first;
    }
    std::vector<uint8_t> o;
    o.reserve((size_t) ne1 * (size_t) (w / blk_elems) * (size_t) blk_bytes);
    for (int64_t r = 0; r < ne1; ++r) {
        const uint8_t* row = p + (size_t) r * row_bytes;
        for (const auto& x : c)
            o.insert(o.end(), row + (size_t) (x.first / blk_elems) * blk_bytes, row + (size_t) (x.second / blk_elems) * blk_bytes);
    }
    return o;
}

bool tp_slice_arena(WeightTable& wt, const TpSlice& s, uint64_t& bytes, std::string& err) {
    bytes = 0;
    if (!s.on()) return true;
    std::vector<std::string> names;
    for (const auto& [name, ref] : wt.all()) {
        TpSlice::Ranges r;
        if (s.plan(name, r) == TpSlice::Axis::Rows && ref.data != nullptr && ref.native_data == nullptr) names.push_back(name);
    }
    for (const std::string& name : names) {
        WeightRef* w = wt.find_mut(name);
        TpSlice::Ranges r;
        s.plan(name, r);
        const int64_t rows = w->ne1 > 1 ? w->ne1 : w->ne0;
        const int64_t per_row = w->ne1 > 1 ? w->ne0 : 1;
        if (w->elements <= 0 || w->bytes % (uint64_t) w->elements != 0 || w->elements != rows * per_row) {
            err = "tensor parallel: " + name + " is not a plain 1-D or 2-D tensor in the arena";
            return false;
        }
        const uint64_t esz = w->bytes / (uint64_t) w->elements, row_bytes = esz * (uint64_t) per_row;
        int64_t kept = 0;
        for (const auto& x : r) {
            if (x.first < 0 || x.second > rows) { err = "tensor parallel: a row range of " + name + " is out of bounds"; return false; }
            kept += x.second - x.first;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) kept * row_bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "tensor parallel: no device memory for the slice of " + name;
            return false;
        }
        uint64_t at = 0;
        for (const auto& x : r) {
            const uint64_t nb = (uint64_t) (x.second - x.first) * row_bytes;
            if (cudaMemcpy((uint8_t*) p + at, (const uint8_t*) w->data + (uint64_t) x.first * row_bytes, nb,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                (void) cudaGetLastError();
                err = "tensor parallel: copying the slice of " + name + " failed";
                return false;
            }
            at += nb;
        }
        w->data = p;   // (the slices live as long as the process, as the arena does)
        w->bytes = at;
        w->elements = kept * per_row;
        if (w->ne1 > 1) w->ne1 = kept;
        else w->ne0 = kept;
        bytes += at;
    }
    return true;
}

}  // namespace strata::core
