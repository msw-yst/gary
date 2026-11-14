// Strata (TP-05). The tensor-parallel FFN split (src/dsv41/ds4_tp_split.hpp), CPU only:
//   g++ -std=c++17 -O2 -I src/dsv41 tests/dsv41/tp_split_test.cpp -o tp_split_test && ./tp_split_test
// - splitb in 128-blocks is the old split128, value for value (EXL3 FFNs and the head keep their columns);
// - n_ff 2304 in 256-blocks over 1, 3 or 9 members is the old even split (m n_ff / nm, n_ff / nm) at every rotation - the
//   ranges tp3 had;
// - over 4: 3/2/2/2 blocks, the 3 going round the members layer by layer, every member 10 x 768 + 30 x 512 over 40 layers;
// - every split tiles [0, n_ff) in member order with whole blocks; DS4_TP_SHARES' shares go round the same way.
#include "ds4_tp_split.hpp"

#include <cstdio>
#include <vector>

using strata::ds4::splitb;

static int fails = 0;
#define CHECK(c)                                                     \
    do {                                                             \
        if (!(c)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                 \
        }                                                            \
    } while (0)

// build 9's split128, verbatim
static void split128_old(int whole, int np, int p, int& off, int& n, int rot = 0) {
    const int b = whole / 128, q = b / np, r = b % np;
    auto blocks = [&](int i) { return q + (((i - rot) % np + np) % np < r ? 1 : 0); };
    int at = 0;
    for (int i = 0; i < p; ++i) at += blocks(i);
    off = 128 * at;
    n = 128 * blocks(p);
}

static bool tiles(int whole, int blk, int np, int rot, const std::vector<int>* sh) {
    int at = 0;
    for (int p = 0; p < np; ++p) {
        int off = -1, n = -1;
        splitb(whole, blk, np, p, off, n, rot, sh);
        if (off != at || n <= 0 || n % blk != 0) return false;
        at += n;
    }
    return at == whole;
}

int main() {
    int checked = 0;
    for (int whole = 128; whole <= 8192; whole += 128)
        for (int np = 1; np <= 8 && np <= whole / 128; ++np)
            for (int rot = 0; rot < np; ++rot)
                for (int p = 0; p < np; ++p) {
                    int a0 = 0, an = 0, b0 = 0, bn = 0;
                    split128_old(whole, np, p, a0, an, rot);
                    splitb(whole, 128, np, p, b0, bn, rot);
                    CHECK(a0 == b0 && an == bn);
                    ++checked;
                }
    const int ff = 2304, blk = 256;
    for (int np : {1, 3, 9})
        for (int rot = 0; rot < np; ++rot)
            for (int m = 0; m < np; ++m) {
                int off = 0, n = 0;
                splitb(ff, blk, np, m, off, n, rot);
                CHECK(n == ff / np && off == m * (ff / np));
            }
    // tp4 on 40 layers, rot = layer % 4
    std::vector<long> tot(4, 0);
    for (int il = 0; il < 40; ++il) {
        CHECK(tiles(ff, blk, 4, il % 4, nullptr));
        int wide = 0;
        for (int m = 0; m < 4; ++m) {
            int off = 0, n = 0;
            splitb(ff, blk, 4, m, off, n, il % 4);
            CHECK(n == 768 || n == 512);
            if (n == 768) {
                ++wide;
                CHECK(m == il % 4);   // (the odd block on member rot)
            }
            tot[(size_t) m] += n;
        }
        CHECK(wide == 1);
    }
    for (int m = 0; m < 4; ++m) CHECK(tot[(size_t) m] == 10L * 768 + 30L * 512);
    for (int np = 2; np <= 9; ++np)
        for (int rot = 0; rot < np; ++rot) CHECK(tiles(ff, blk, np, rot, nullptr));
    // DS4_TP_SHARES: 4,3,2 over 3 members (an uneven split at a width that divides)
    const std::vector<int> sh = {4, 3, 2};
    for (int rot = 0; rot < 3; ++rot) {
        CHECK(tiles(ff, blk, 3, rot, &sh));
        for (int m = 0; m < 3; ++m) {
            int off = 0, n = 0;
            splitb(ff, blk, 3, m, off, n, rot, &sh);
            CHECK(n == blk * sh[(size_t) (((m - rot) % 3 + 3) % 3)]);
        }
    }
    std::printf("tp_split_test: %d split128 cases, the 2304 / 256 splits: %s\n", checked, fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
