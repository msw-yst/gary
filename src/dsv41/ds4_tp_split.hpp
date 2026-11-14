// Strata (TP-05). How a tensor-parallel group's members split an FFN: whole blocks, as even as they go.
//
// Part p of np of `whole` (a multiple of blk) in blk-blocks: whole / blk % np parts take a block more - those from part rot on,
// cyclically - so a group's layers, each with its own rot (layer % np), even out what each member holds.  split128 (EXL3's FFN
// and the head's columns) is splitb in 128-blocks.  A GGUF pack's 256-value experts (n_ff 2304 = 9 blocks) split this way over
// 4 members as 3/2/2/2, the 3 going round the members layer by layer.
//
// sh (the test knob DS4_TP_SHARES): part i takes sh[(i - rot) mod np] blocks instead - any split, going round the same way.
//
// After Project Maya's splitb (github.com/mw00/project-maya e3f2475, #408, by pepuscz; MIT,
// third_party/project-maya/NOTICE.md).
#pragma once

#include <vector>

namespace strata::ds4 {

inline void splitb(int whole, int blk, int np, int p, int& off, int& n, int rot = 0, const std::vector<int>* sh = nullptr) {
    const int b = whole / blk, q = b / np, r = b % np;
    auto blocks = [&](int i) {
        const int k = ((i - rot) % np + np) % np;
        return sh != nullptr ? (*sh)[(size_t) k] : q + (k < r ? 1 : 0);
    };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = blk * at;
    n = blk * blocks(p);
}

}  // namespace strata::ds4
