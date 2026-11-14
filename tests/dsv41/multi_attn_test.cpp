// DS4_MULTI_ATTN (docs/dsv41/cards/FG-05.md): a batch window's attention for all its sequences' rows in shared launches
// (attn_batch_multi's calls) against the per-run calls attn_batch makes without the flag, bit for bit on random rings,
// compressed caches, selections and queries: q's rotation, the window buffers, the rings' new rows and their backups,
// and the attention's output rotated back.  Decode steps (one row a sequence, B = 1 .. 16), verify windows (a few rows a
// sequence), positions near 0 (window rows before position 0) and past the window, a layer without compressed rows.
// After Project Maya's tests/ds4_multi_attn_test.cu (#639, MIT - third_party/project-maya/NOTICE.md), written for our
// kernels' API.  --time: both ways' launches timed at decode shapes (B = 2 / 4 / 8, one row each).  One GPU, no model.
//
//   ds4_multi_attn_test [--heads 32] [--time] [--reps 200] [--seed 1]
//
// rc 0: every case identical; rc 3: a case differs (its line says where); rc 2: a CUDA / HIP error.
#include "../../src/dsv41/ds4_kernels.cuh"
#include "../../src/dsv41/ds4_multi_attn.hpp"
#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds4;

namespace {

constexpr int HD = 512, WIN = 128, RD = 64, TS = 512, NCK = 2048;

#define CKT(x)                                                                                      \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                           \
        }                                                                                           \
    } while (0)

template <class T>
T* dev(size_t n) {
    T* p = nullptr;
    CKT(cudaMalloc((void**) &p, n * sizeof(T) + 256));
    CKT(cudaMemset(p, 0, n * sizeof(T) + 256));
    return p;
}
template <class T>
void put(T* d, const std::vector<T>& h) {
    CKT(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T>
std::vector<T> get(const T* d, size_t n) {
    std::vector<T> h(n);
    CKT(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

const size_t RB8 = kv_row_bytes(kKvFp8, HD), RB4 = kv_row_bytes(kKvFp4, HD);

// a sequence's state: its window ring (kKvFp8 rows: E4M3 codes, UE8M0 scales a group of 32) and its compressed cache
// (kKvFp4 rows: E2M1 codes, E4M3 scales a group of 16) - valid codes, moderate scales
struct Seq {
    uint8_t* ring = nullptr;
    uint8_t* ckv = nullptr;
};
void fill_ring(std::vector<uint8_t>& r, std::mt19937& rng) {
    r.assign((size_t) WIN * RB8, 0);
    for (int row = 0; row < WIN; ++row) {
        uint8_t* p = r.data() + (size_t) row * RB8;
        for (int e = 0; e < HD; ++e) p[e] = (uint8_t) ((rng() % 0x7f) | (rng() & 1 ? 0x80 : 0));
        for (int gi = 0; gi < HD / 32; ++gi) p[HD + gi] = (uint8_t) (118 + rng() % 6);
    }
}
void fill_ckv(std::vector<uint8_t>& c, std::mt19937& rng) {
    c.assign((size_t) NCK * RB4, 0);
    for (int row = 0; row < NCK; ++row) {
        uint8_t* p = c.data() + (size_t) row * RB4;
        for (int i = 0; i < HD / 2; ++i) p[i] = (uint8_t) rng();
        for (int gi = 0; gi < HD / 16; ++gi) p[HD / 2 + gi] = (uint8_t) (0x28 + rng() % 16);
    }
}

struct Case {
    std::string name;
    std::vector<int> n;          // a run's rows
    std::vector<long long> p0;   // its first row's position
    int ratio = 4;               // 0: a layer without compressed rows
    bool bak = false;            // a verify window: the replaced ring rows backed up
};

struct Bufs {
    int heads = 32;
    float *q0, *q, *kvr, *o, *knorm, *freqs, *sink;
    int* sel;
    uint16_t *ext, *mext;
    AttnScratch scr7, scr16;
    std::vector<Seq> seq_a, seq_b;   // the per-run path's copies, the shared path's
    std::vector<uint8_t*> bak_a, bak_b;
};

// the per-run path (attn_batch without the flag): q rotated a row at a time, then a run at a time its ring into the
// window buffer, its KV rows' finish, the ring's new rows, its sparse attention and the inverse rotation
void per_run(Bufs& b, const Case& c, const MaPlan& p, const int* nsel, cudaStream_t s) {
    const int H = b.heads, HQ = H * HD;
    for (int t = 0; t < p.T; ++t) {
        Finish fr;
        fr.rd = RD;
        fr.pos = p.pos[t];
        fr.vpp = H;
        fr.freqs = b.freqs;
        finish(b.q + (size_t) t * HQ, H, HD, HD, fr, s);
    }
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        Seq& sq = b.seq_a[(size_t) k];
        ring_to_ext(sq.ring, b.ext, u.p0, WIN, HD, s);
        Finish fk;
        fk.w = b.knorm;
        fk.eps = 1e-6f;
        fk.rd = RD;
        fk.pos = (int) u.p0;
        fk.freqs = b.freqs;
        fk.quant = 1;
        fk.dst = b.ext + (size_t) WIN * HD;
        fk.dst_stride = HD;
        finish(b.kvr + (size_t) u.r0 * HD, u.n, HD, HD, fk, s);
        ext_to_ring(b.ext, sq.ring, u.p0, WIN, u.n, HD, s, c.bak ? b.bak_a[(size_t) k] : nullptr);
        AttnRows ar;
        for (int t = 0; t < u.n; ++t) {
            const int first = (int) std::max<long long>(WIN - u.p0, t + 1);
            ar.win0[t] = first;
            ar.nwin[t] = WIN + t - first + 1;
            ar.nsel[t] = nsel[u.r0 + t];
        }
        sparse_attn(b.q + (size_t) u.r0 * HQ, b.ext, c.ratio > 0 ? sq.ckv : nullptr, b.sel + (size_t) u.r0 * TS, TS, ar, b.sink,
                    H, HD, u.n, 1.0f / std::sqrt((float) HD), b.o + (size_t) u.r0 * HQ, s, kKvFp4, 0, &b.scr7);
        Finish fo;
        fo.rd = RD;
        fo.pos = (int) u.p0;
        fo.vpp = H;
        fo.freqs = b.freqs;
        fo.inverse = 1;
        fo.quant = 1;
        finish(b.o + (size_t) u.r0 * HQ, u.n * H, HD, HD, fo, s);
    }
}

// the shared path (attn_batch_multi): the same, each step one launch for all the runs
bool shared(Bufs& b, const Case& c, const MaPlan& p, const int* nsel, cudaStream_t s) {
    const int H = b.heads;
    Finish fr;
    fr.rd = RD;
    fr.vpp = H;
    fr.freqs = b.freqs;
    FinishRows fq;
    for (int t = 0; t < p.T; ++t) fq.pos[t] = p.pos[t];
    if (!finish_rows(b.q, p.T * H, HD, HD, fr, fq, s)) return false;
    RingRuns rr;
    rr.runs = p.nrun;
    bool one = true;
    for (int k = 0; k < p.nrun; ++k) {
        const MaRun& u = p.run[k];
        rr.ring[k] = b.seq_b[(size_t) k].ring;
        rr.bak[k] = c.bak ? b.bak_b[(size_t) k] : nullptr;
        rr.p0[k] = u.p0;
        rr.e0[k] = u.e0;
        rr.T[k] = u.n;
        one = one && u.n == 1;
    }
    ring_to_ext_runs(rr, b.mext, WIN, HD, s);
    Finish fk;
    fk.w = b.knorm;
    fk.eps = 1e-6f;
    fk.rd = RD;
    fk.freqs = b.freqs;
    fk.quant = 1;
    fk.dst = b.mext;
    fk.dst_stride = HD;
    FinishRows fkr;
    for (int t = 0; t < p.T; ++t) {
        fkr.pos[t] = p.pos[t];
        fkr.dst[t] = p.dst[t];
    }
    if (!finish_rows(b.kvr, p.T, HD, HD, fk, fkr, s)) return false;
    ext_to_ring_runs(rr, b.mext, WIN, HD, s);
    AttnRows ar;
    AttnCkv ck;
    for (int t = 0; t < p.T; ++t) {
        ar.win0[t] = p.win0[t];
        ar.nwin[t] = p.nwin[t];
        ar.nsel[t] = nsel[t];
        ck.ckv[t] = c.ratio > 0 ? b.seq_b[(size_t) p.row_run[t]].ckv : nullptr;
    }
    sparse_attn_rows(b.q, b.mext, ck, b.sel, TS, ar, b.sink, H, HD, p.T, 1.0f / std::sqrt((float) HD), b.o, s, one, &b.scr16);
    Finish fo;
    fo.rd = RD;
    fo.vpp = H;
    fo.freqs = b.freqs;
    fo.inverse = 1;
    fo.quant = 1;
    FinishRows fp;
    for (int t = 0; t < p.T; ++t) fp.pos[t] = p.pos[t];
    return finish_rows(b.o, p.T * H, HD, HD, fo, fp, s);
}

bool make_plan(const Case& c, MaPlan& p, int* nsel, int64_t* rpos) {
    int rseq[kWin];
    int T = 0;
    for (size_t k = 0; k < c.n.size(); ++k)
        for (int i = 0; i < c.n[k]; ++i, ++T) {
            rseq[T] = (int) k;
            rpos[T] = c.p0[k] + i;
        }
    if (!ma_plan(rseq, rpos, T, WIN, false, p)) return false;
    for (int t = 0; t < T; ++t) {
        const long long avail = c.ratio > 0 ? std::min<long long>(NCK, (rpos[t] + 1) / c.ratio) : 0;
        nsel[t] = (int) std::min<long long>(TS, avail);
    }
    return true;
}

int run_case(Bufs& b, const Case& c, std::mt19937& rng, cudaStream_t s) {
    const int H = b.heads, HQ = H * HD;
    MaPlan p;
    int nsel[kWin] = {};
    int64_t rpos[kWin] = {};
    if (!make_plan(c, p, nsel, rpos)) {
        std::printf("%-28s PLAN REFUSED\n", c.name.c_str());
        return 3;
    }
    const int T = p.T;
    int max_run = 0;
    bool any1 = false, anyn = false;
    for (int k = 0; k < p.nrun; ++k) {
        max_run = std::max(max_run, p.run[k].n);
        (p.run[k].n == 1 ? any1 : anyn) = true;
    }
    if (!sparse_attn_rows_ok(T, max_run, TS, &b.scr16, any1 && anyn)) {
        std::printf("%-28s not the shared path's shape (sparse_attn_rows_ok false): skipped\n", c.name.c_str());
        return 0;
    }
    // inputs: q, the KV rows, each row's selection (its own sequence's compressed rows), each sequence's ring and cache
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hq((size_t) T * HQ), hk((size_t) T * HD);
    for (float& v : hq) v = nd(rng);
    for (float& v : hk) v = nd(rng) * 3.0f;
    std::vector<int> hs((size_t) T * TS, 0);
    for (int t = 0; t < T; ++t) {
        const long long avail = c.ratio > 0 ? std::min<long long>(NCK, (rpos[t] + 1) / c.ratio) : 0;
        for (int i = 0; i < nsel[t]; ++i) hs[(size_t) t * TS + i] = (int) (rng() % (unsigned) avail);
    }
    put(b.q0, hq);
    put(b.kvr, hk);
    put(b.sel, hs);
    for (int k = 0; k < p.nrun; ++k) {
        std::vector<uint8_t> r, ck;
        fill_ring(r, rng);
        fill_ckv(ck, rng);
        put(b.seq_a[(size_t) k].ring, r);
        put(b.seq_b[(size_t) k].ring, r);
        put(b.seq_a[(size_t) k].ckv, ck);
        put(b.seq_b[(size_t) k].ckv, ck);
        CKT(cudaMemset(b.bak_a[(size_t) k], 0x5a, (size_t) kWin * RB8));
        CKT(cudaMemset(b.bak_b[(size_t) k], 0x5a, (size_t) kWin * RB8));
    }
    // the per-run path
    CKT(cudaMemcpyAsync(b.q, b.q0, (size_t) T * HQ * 4, cudaMemcpyDeviceToDevice, s));
    CKT(cudaMemsetAsync(b.o, 0, (size_t) T * HQ * 4, s));
    per_run(b, c, p, nsel, s);
    CKT(cudaStreamSynchronize(s));
    CKT(cudaGetLastError());
    const std::vector<float> qa = get(b.q, (size_t) T * HQ), oa = get(b.o, (size_t) T * HQ);
    // the shared path
    CKT(cudaMemcpyAsync(b.q, b.q0, (size_t) T * HQ * 4, cudaMemcpyDeviceToDevice, s));
    CKT(cudaMemsetAsync(b.o, 0, (size_t) T * HQ * 4, s));
    if (!shared(b, c, p, nsel, s)) {
        std::printf("%-28s the shared launches refused\n", c.name.c_str());
        return 3;
    }
    CKT(cudaStreamSynchronize(s));
    CKT(cudaGetLastError());
    const std::vector<float> qb = get(b.q, (size_t) T * HQ), ob = get(b.o, (size_t) T * HQ);
    int bad = 0;
    std::string where;
    auto cmp = [&](const void* x, const void* y, size_t bytes, const char* what) {
        if (std::memcmp(x, y, bytes) != 0) {
            ++bad;
            where += std::string(" ") + what;
        }
    };
    cmp(qa.data(), qb.data(), qa.size() * 4, "q");
    cmp(oa.data(), ob.data(), oa.size() * 4, "out");
    size_t nonfinite = 0;
    for (float v : oa) nonfinite += std::isfinite(v) ? 0 : 1;
    for (int k = 0; k < p.nrun; ++k) {
        const std::vector<uint8_t> ra = get(b.seq_a[(size_t) k].ring, (size_t) WIN * RB8),
                                   rb = get(b.seq_b[(size_t) k].ring, (size_t) WIN * RB8);
        const std::vector<uint8_t> ba = get(b.bak_a[(size_t) k], (size_t) kWin * RB8),
                                   bb = get(b.bak_b[(size_t) k], (size_t) kWin * RB8);
        char w[32];
        std::snprintf(w, sizeof w, "ring%d", k);
        cmp(ra.data(), rb.data(), ra.size(), w);
        std::snprintf(w, sizeof w, "backup%d", k);
        cmp(ba.data(), bb.data(), ba.size(), w);
    }
    std::printf("%-28s T %2d, %2d runs, ratio %3d%s: %s%s (non-finite outputs %zu)\n", c.name.c_str(), T, p.nrun, c.ratio,
                c.bak ? ", backups" : "", bad == 0 ? "IDENTICAL" : "DIFFER:", where.c_str(), nonfinite);
    return bad == 0 ? 0 : 3;
}

// the per-run and the shared launches at a decode step's shapes (B rows, one a sequence, at position p0), us a layer
void time_case(Bufs& b, int B, long long p0, int reps, std::mt19937& rng, cudaStream_t s) {
    Case c;
    c.name = "time B=" + std::to_string(B);
    c.n.assign((size_t) B, 1);
    c.p0.assign((size_t) B, p0);
    run_case(b, c, rng, s);   // (fills the inputs, and checks them)
    MaPlan p;
    int nsel[kWin] = {};
    int64_t rpos[kWin] = {};
    make_plan(c, p, nsel, rpos);
    cudaEvent_t e0, e1;
    CKT(cudaEventCreate(&e0));
    CKT(cudaEventCreate(&e1));
    float ms[2] = {};
    for (int way = 0; way < 2; ++way) {
        for (int w = 0; w < 5; ++w) {
            if (way == 0) per_run(b, c, p, nsel, s);
            else shared(b, c, p, nsel, s);
        }
        CKT(cudaEventRecord(e0, s));
        for (int r = 0; r < reps; ++r) {
            if (way == 0) per_run(b, c, p, nsel, s);
            else shared(b, c, p, nsel, s);
        }
        CKT(cudaEventRecord(e1, s));
        CKT(cudaEventSynchronize(e1));
        CKT(cudaEventElapsedTime(&ms[way], e0, e1));
    }
    // (launches: per run, B q rotations + 6 a run; shared, 7)
    std::printf("time B=%d p0 %lld: per run %.1f us a layer (%d launches), shared %.1f us a layer (7 launches): %.2fx\n", B,
                p0, 1000.0 * ms[0] / reps, 7 * B, 1000.0 * ms[1] / reps, ms[0] / std::max(1e-6f, ms[1]));
    CKT(cudaEventDestroy(e0));
    CKT(cudaEventDestroy(e1));
}

}  // namespace

int main(int argc, char** argv) {
    int heads = 32, reps = 200;
    unsigned seed = 1;
    bool timing = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--heads" && i + 1 < argc) heads = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (unsigned) std::atoi(argv[++i]);
        else if (a == "--time") timing = true;
        else {
            std::fprintf(stderr, "usage: ds4_multi_attn_test [--heads 32] [--time] [--reps 200] [--seed 1]\n");
            return 1;
        }
    }
    cudaStream_t s;
    CKT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Bufs b;
    b.heads = heads;
    const size_t HQ = (size_t) heads * HD;
    b.q0 = dev<float>(kWin * HQ);
    b.q = dev<float>(kWin * HQ);
    b.o = dev<float>(kWin * HQ);
    b.kvr = dev<float>((size_t) kWin * HD);
    b.sel = dev<int>((size_t) kWin * TS);
    b.ext = dev<uint16_t>(((size_t) WIN + kWin) * HD);
    b.mext = dev<uint16_t>((size_t) ma_rows_max(WIN) * HD);
    b.scr7.max_t = kAttnScratchT;
    b.scr7.cap = TS;
    b.scr7.rows = dev<uint16_t>((size_t) kAttnScratchT * TS * HD);
    b.scr16.max_t = kWin;
    b.scr16.cap = TS;
    b.scr16.rows = dev<uint16_t>((size_t) kWin * TS * HD);
    std::mt19937 rng(seed);
    {
        std::vector<float> kn(HD), fq(RD / 2), sk((size_t) heads);
        for (float& v : kn) v = 0.5f + (float) (rng() % 1000) / 1000.0f;
        for (int i = 0; i < RD / 2; ++i) fq[(size_t) i] = std::pow(10000.0f, -2.0f * i / RD);
        for (float& v : sk) v = (float) ((int) (rng() % 200) - 100) / 50.0f;
        b.knorm = dev<float>(HD);
        b.freqs = dev<float>(64);
        b.sink = dev<float>((size_t) heads);
        put(b.knorm, kn);
        put(b.freqs, fq);
        put(b.sink, sk);
    }
    for (int k = 0; k < kWin; ++k) {
        b.seq_a.push_back({dev<uint8_t>((size_t) WIN * RB8), dev<uint8_t>((size_t) NCK * RB4)});
        b.seq_b.push_back({dev<uint8_t>((size_t) WIN * RB8), dev<uint8_t>((size_t) NCK * RB4)});
        b.bak_a.push_back(dev<uint8_t>((size_t) kWin * RB8));
        b.bak_b.push_back(dev<uint8_t>((size_t) kWin * RB8));
    }
    std::vector<Case> cases = {
        {"decode B=1", {1}, {3000}},
        {"decode B=2", {1, 1}, {2048, 700}},
        {"decode B=4 near 0", {1, 1, 1, 1}, {0, 5, 127, 128}},
        {"decode B=8", std::vector<int>(8, 1), {1100, 1500, 2047, 3000, 4000, 129, 900, 7000}},
        {"decode B=16", std::vector<int>(16, 1), std::vector<long long>(16, 5000)},
        {"decode B=8 no compressed", std::vector<int>(8, 1), {1100, 1500, 2047, 3000, 4000, 129, 900, 7000}, 0},
        {"verify 3x3 backups", {3, 3, 3}, {10, 200, 4096}, 4, true},
        {"verify 2+4+1 backups", {2, 4, 1}, {125, 1000, 6000}, 4, true},
        {"verify 7+1 backups", {7, 1}, {120, 7}, 4, true},
        {"verify 4x4", {4, 4, 4, 4}, {3000, 3001, 64, 7000}, 4, false},
        {"ratio 128 B=8", std::vector<int>(8, 1), {1100, 1500, 2047, 3000, 4000, 129, 900, 7000}, 128},
    };
    int rc = 0;
    for (const Case& c : cases) rc = std::max(rc, run_case(b, c, rng, s));
    for (int it = 0; it < 40; ++it) {   // random mixes of runs and positions
        Case c;
        c.name = "random " + std::to_string(it);
        int T = 0;
        while (T < kWin) {
            const int n = 1 + (int) (rng() % (rng() % 3 == 0 ? 7 : 2));
            if (T + n > kWin) break;
            c.n.push_back(n);
            c.p0.push_back(rng() % 4 == 0 ? (long long) (rng() % 140) : (long long) (rng() % 8000));
            T += n;
            if (rng() % 4 == 0) break;
        }
        c.ratio = rng() % 5 == 0 ? 0 : 4;
        c.bak = rng() % 2 == 0;
        rc = std::max(rc, run_case(b, c, rng, s));
    }
    if (timing)
        for (int B : {2, 4, 8}) time_case(b, B, 2048, reps, rng, s);
    std::printf("ds4_multi_attn_test: %s\n", rc == 0 ? "every case IDENTICAL" : "a case DIFFERS");
    return rc;
}
