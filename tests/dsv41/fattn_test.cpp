// (docs/dsv41/cards/PF-02.md): the chunked prompt's packed-KV attention - the scalar sparse_attn_packed_kernel (the engine's
// default) against each DS4_FAST_ATTN_MODE variant of sparse_attn_wmma_kernel on identical inputs, and both against a
// double-precision reference on a sample of rows.  Inputs: the window rows on the FP8 grid and the compressed rows as
// FP4 x E4M3 codes (the engine's own roundings), Q, sinks; synthetic at chosen magnitudes, or Q from a DS4_DUMP_S "q"
// record of a real run (--q-dump, its layer --q-layer; T, heads and p0 then come from the record).  Needs one GPU, no model.
//
//   ds4_fattn_test [--rows 2048,8192,16384] [--heads 64] [--p0 0] [--ratio 4] [--q-rms 1] [--kv-rms 1] [--sink 0]
//                  [--q-big X --q-big-frac F] [--kv-big X --kv-big-frac F] [--modes f16,f32,...|all] [--reps 3]
//                  [--ref-rows 24] [--seed 1] [--q-dump FILE --q-layer L [--q-p0 P]]
//
// A line a (rows, mode): vs the scalar kernel - max / mean |d|, the worst row's relative L2, rows past 1e-2 / 1e-1, the
// non-finite outputs; vs the double reference (the sampled rows) - the worst row's relative L2 of this kernel; and the
// time a call (mean of --reps after a warm-up).  The scalar kernel's own line (mode "scalar") gives its reference error and time.
#include "../../src/dsv41/ds4_kernels.cuh"
#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::ds4;

namespace {

constexpr int kD = 512, kWinRows = 128, kTopk = 512;

#define CKT(x)                                                                                   \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

float e4m3_dec(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
// the positive E4M3 values (codes 0 .. 0x7e) for rounding to nearest
std::vector<float> e4m3_pos() {
    std::vector<float> v;
    for (int b = 0; b <= 0x7e; ++b) v.push_back(e4m3_dec((uint8_t) b));
    return v;
}
uint8_t e4m3_enc(float x, const std::vector<float>& pos) {   // round to nearest (ties: the even code), saturating at 448
    const float a = std::min(std::fabs(x), 448.0f);
    auto it = std::lower_bound(pos.begin(), pos.end(), a);
    int i = (int) (it - pos.begin());
    if (i >= (int) pos.size()) i = (int) pos.size() - 1;
    if (i > 0 && (a - pos[(size_t) i - 1] < pos[(size_t) i] - a || (a - pos[(size_t) i - 1] == pos[(size_t) i] - a && ((i - 1) & 1) == 0))) --i;
    return (uint8_t) (i | (x < 0 ? 0x80 : 0));
}
const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
uint8_t e2m1_enc(float x) {   // nearest, ties to the even code
    const float a = std::min(std::fabs(x), 6.0f);
    int best = 0;
    for (int i = 1; i < 8; ++i) {
        const float d = std::fabs(a - kE2m1[i]), db = std::fabs(a - kE2m1[best]);
        if (d < db || (d == db && (i & 1) == 0)) best = i;
    }
    return (uint8_t) (best | (x < 0 && best != 0 ? 8 : 0));
}
uint16_t f2bf(float f) {   // (exact for the grid values used here)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float pow2_ceil(float x) { return std::ldexp(1.0f, (int) std::ceil(std::log2(x))); }

struct Opt {
    std::vector<int> rows{2048, 8192, 16384};
    int heads = 64, ratio = 4, reps = 3, ref_rows = 24;
    long long p0 = 0;
    float q_rms = 1.0f, kv_rms = 1.0f, sink = 0.0f, q_big = 1.0f, q_big_frac = 0.0f, kv_big = 1.0f, kv_big_frac = 0.0f;
    unsigned seed = 1;
    std::vector<int> modes;
    std::string q_dump;
    int q_layer = -1;
    long long q_p0 = -1;
};

std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        v.push_back((int) std::strtol(p, &e, 10));
        if (e == p) break;
        p = *e == ',' ? e + 1 : e;
    }
    return v;
}

// a DS4_DUMP_S record: name[16], int64 {layer, p0, T, bytes}, then the bytes
bool read_q_dump(const std::string& path, int layer, long long want_p0, std::vector<float>& q, long long& p0, int& T) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char nm[16];
    int64_t h[4];
    bool ok = false;
    while (std::fread(nm, 1, 16, f) == 16 && std::fread(h, 8, 4, f) == 4) {
        const bool hit = std::strncmp(nm, "q", 16) == 0 && (layer < 0 || h[0] == layer) && (want_p0 < 0 || h[1] == want_p0);
        if (hit) {
            q.resize((size_t) h[3] / 4);
            ok = std::fread(q.data(), 1, (size_t) h[3], f) == (size_t) h[3];
            p0 = h[1];
            T = (int) h[2];
            break;
        }
        std::fseek(f, (long) h[3], SEEK_CUR);
    }
    std::fclose(f);
    return ok;
}

struct Inputs {
    int T = 0, heads = 0, sel_stride = kTopk;
    long long p0 = 0;
    AttnRows ar;
    std::vector<float> q, sink;
    std::vector<uint16_t> ext;   // (kWinRows + T) BF16 rows
    std::vector<uint8_t> ckv;    // clen kKvFp4 rows
    std::vector<int> sel;        // T x kTopk
    std::vector<float> ckv_f;    // the compressed rows decoded (the reference's)
};

void make_inputs(const Opt& o, int T, Inputs& in, std::mt19937& rng) {
    const std::vector<float> pos = e4m3_pos();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    in.T = T;
    in.p0 = o.p0;
    if (in.q.empty()) {
        in.q.resize((size_t) T * in.heads * kD);
        for (auto& x : in.q) {
            x = nd(rng) * o.q_rms;
            if (o.q_big_frac > 0 && ud(rng) < o.q_big_frac) x *= o.q_big;
        }
    }
    in.sink.assign((size_t) in.heads, o.sink);
    // the window buffer: kWinRows ring rows before p0 (zeros before position 0), then the chunk's rows; FP8-grid values
    // (Finish::quant 1: E4M3 at a power-of-two scale a group of 32)
    in.ext.assign((size_t) (kWinRows + T) * kD, 0);
    std::vector<float> row(kD);
    for (int r = 0; r < kWinRows + T; ++r) {
        if (in.p0 - kWinRows + r < 0) continue;
        for (int e = 0; e < kD; ++e) {
            row[(size_t) e] = nd(rng) * o.kv_rms;
            if (o.kv_big_frac > 0 && ud(rng) < o.kv_big_frac) row[(size_t) e] *= o.kv_big;
        }
        for (int g = 0; g < kD; g += 32) {
            float a = 0.0f;
            for (int e = g; e < g + 32; ++e) a = std::max(a, std::fabs(row[(size_t) e]));
            const float s = pow2_ceil(std::max(a, 1e-4f) / 448.0f);
            for (int e = g; e < g + 32; ++e)
                in.ext[(size_t) r * kD + e] = f2bf(e4m3_dec(e4m3_enc(row[(size_t) e] / s, pos)) * s);
        }
    }
    // the compressed rows: E2M1 codes, an E4M3 scale a group of 16 (Finish::quant 2)
    const int R = o.ratio;
    const int clen = R > 0 ? (int) ((in.p0 + T) / R) : 0;
    const size_t crow = kv_row_bytes(kKvFp4, kD);
    in.ckv.assign((size_t) std::max(clen, 1) * crow, 0);
    in.ckv_f.assign((size_t) std::max(clen, 1) * kD, 0.0f);
    for (int r = 0; r < clen; ++r) {
        uint8_t* rr = in.ckv.data() + (size_t) r * crow;
        for (int e = 0; e < kD; ++e) {
            row[(size_t) e] = nd(rng) * o.kv_rms;
            if (o.kv_big_frac > 0 && ud(rng) < o.kv_big_frac) row[(size_t) e] *= o.kv_big;
        }
        for (int g = 0; g < kD / 16; ++g) {
            float a = 0.0f;
            for (int e = g * 16; e < g * 16 + 16; ++e) a = std::max(a, std::fabs(row[(size_t) e]));
            const uint8_t sb = e4m3_enc(std::max(a, 6.0f * 0.001953125f) / 6.0f, pos);
            const float s = e4m3_dec(sb);
            rr[kD / 2 + g] = sb;
            for (int e = g * 16; e < g * 16 + 16; ++e) {
                const uint8_t c = e2m1_enc(row[(size_t) e] / s);
                rr[e / 2] |= (uint8_t) ((e & 1) ? c << 4 : c);
                in.ckv_f[(size_t) r * kD + e] = (c & 8 ? -1.0f : 1.0f) * kE2m1[c & 7] * s;
            }
        }
    }
    // each query's selection: min(kTopk, (pos + 1) / R) distinct compressed rows (the indexer's top-k: a random subset here)
    in.sel.assign((size_t) T * kTopk, 0);
    std::vector<int> pool;
    for (int t = 0; t < T; ++t) {
        const long long pa = in.p0 + t;
        const int avail = R > 0 ? (int) ((pa + 1) / R) : 0, n = std::min(kTopk, avail);
        if ((int) pool.size() < avail) {
            const int old = (int) pool.size();
            pool.resize((size_t) avail);
            std::iota(pool.begin() + old, pool.end(), old);
        }
        for (int i = 0; i < n; ++i) {   // a partial shuffle of [0, avail)
            std::uniform_int_distribution<int> pick(i, avail - 1);
            std::swap(pool[(size_t) i], pool[(size_t) pick(rng)]);
            in.sel[(size_t) t * kTopk + i] = pool[(size_t) i];
        }
    }
    in.ar = AttnRows{};
    in.ar.gen = 1;
    in.ar.p0 = in.p0;
    in.ar.t0 = 0;
    in.ar.win = kWinRows;
    in.ar.R = R;
    in.ar.kcap = kTopk;
}

// the double reference of query t, head h (the kernels' row arithmetic: window rows, then the selected ones, with the sink)
void ref_row(const Inputs& in, int t, int h, float scale, std::vector<double>& out) {
    const long long ta = t, pa = in.p0 + ta;
    const long long first = std::max((long long) kWinRows - in.p0, ta + 1);
    const int w0 = (int) first, n_win = (int) (kWinRows + ta - first + 1);
    const int n_sel = in.ar.R > 0 ? (int) std::min((long long) kTopk, (pa + 1) / in.ar.R) : 0;
    const float* qv = in.q.data() + ((size_t) t * in.heads + h) * kD;
    std::vector<double> s((size_t) (n_win + n_sel));
    auto key = [&](int j, int e) -> double {
        return j < n_win ? (double) bf2f(in.ext[(size_t) (w0 + j) * kD + e])
                         : (double) in.ckv_f[(size_t) in.sel[(size_t) t * kTopk + (j - n_win)] * kD + e];
    };
    double m = -1e300;
    for (int j = 0; j < n_win + n_sel; ++j) {
        double a = 0.0;
        for (int e = 0; e < kD; ++e) a += (double) qv[e] * key(j, e);
        s[(size_t) j] = a * scale;
        m = std::max(m, s[(size_t) j]);
    }
    double den = std::exp((double) in.sink[(size_t) h] - m);
    out.assign(kD, 0.0);
    for (int j = 0; j < n_win + n_sel; ++j) {
        const double p = std::exp(s[(size_t) j] - m);
        den += p;
        for (int e = 0; e < kD; ++e) out[(size_t) e] += p * key(j, e);
    }
    for (auto& x : out) x /= den;
}

// a against b (rows of 512): sums and maxima on the device
struct Cmp {
    unsigned nf_a, nf_b, rows_2, rows_1, max_d, max_rel;
    float sum_d, sum_b;
};
__global__ void cmp_kernel(const float* __restrict__ a, const float* __restrict__ b, Cmp* st) {
    __shared__ float sh[5][256];
    const size_t base = (size_t) blockIdx.x * kD;
    float d2 = 0, b2 = 0, md = 0, sd = 0, sb = 0, nf = 0, nfb = 0;
    for (int e = (int) threadIdx.x; e < kD; e += 256) {
        const float x = a[base + e], y = b[base + e];
        if (!std::isfinite(x)) nf += 1;
        if (!std::isfinite(y)) nfb += 1;
        if (std::isfinite(x) && std::isfinite(y)) {
            const float d = std::fabs(x - y);
            d2 += d * d;
            b2 += y * y;
            md = std::fmax(md, d);
            sd += d;
            sb += std::fabs(y);
        }
    }
    const int i = (int) threadIdx.x;
    sh[0][i] = d2;
    sh[1][i] = b2;
    sh[2][i] = md;
    sh[3][i] = nf;
    sh[4][i] = nfb;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (i < w) {
            sh[0][i] += sh[0][i + w];
            sh[1][i] += sh[1][i + w];
            sh[2][i] = std::fmax(sh[2][i], sh[2][i + w]);
            sh[3][i] += sh[3][i + w];
            sh[4][i] += sh[4][i + w];
        }
        __syncthreads();
    }
    atomicAdd(&st->sum_d, sd);
    atomicAdd(&st->sum_b, sb);
    if (i != 0) return;
    const float rel = sh[1][0] > 0 ? std::sqrt(sh[0][0] / sh[1][0]) : (sh[0][0] > 0 ? 3e38f : 0.0f);
    if (rel > 1e-2f) atomicAdd(&st->rows_2, 1u);
    if (rel > 1e-1f) atomicAdd(&st->rows_1, 1u);
    const unsigned nfa = (unsigned) sh[3][0], nfbb = (unsigned) sh[4][0];
    atomicAdd(&st->nf_a, nfa);
    atomicAdd(&st->nf_b, nfbb);
    atomicMax(&st->max_d, __float_as_uint(sh[2][0]));
    atomicMax(&st->max_rel, __float_as_uint(std::fmin(rel, 3e38f)));
}
float ubits(unsigned b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    Opt o;
    std::vector<std::string> mode_names;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--rows") o.rows = parse_ints(nx());
        else if (a == "--heads") o.heads = std::atoi(nx());
        else if (a == "--p0") o.p0 = std::atoll(nx());
        else if (a == "--ratio") o.ratio = std::atoi(nx());
        else if (a == "--q-rms") o.q_rms = (float) std::atof(nx());
        else if (a == "--kv-rms") o.kv_rms = (float) std::atof(nx());
        else if (a == "--sink") o.sink = (float) std::atof(nx());
        else if (a == "--q-big") o.q_big = (float) std::atof(nx());
        else if (a == "--q-big-frac") o.q_big_frac = (float) std::atof(nx());
        else if (a == "--kv-big") o.kv_big = (float) std::atof(nx());
        else if (a == "--kv-big-frac") o.kv_big_frac = (float) std::atof(nx());
        else if (a == "--reps") o.reps = std::atoi(nx());
        else if (a == "--ref-rows") o.ref_rows = std::atoi(nx());
        else if (a == "--seed") o.seed = (unsigned) std::atoi(nx());
        else if (a == "--q-dump") o.q_dump = nx();
        else if (a == "--q-layer") o.q_layer = std::atoi(nx());
        else if (a == "--q-p0") o.q_p0 = std::atoll(nx());
        else if (a == "--modes") {
            std::string s = nx();
            for (size_t p = 0; p <= s.size();) {
                const size_t c = s.find(',', p);
                mode_names.push_back(s.substr(p, c == std::string::npos ? std::string::npos : c - p));
                if (c == std::string::npos) break;
                p = c + 1;
            }
        } else {
            std::fprintf(stderr, "unknown %s (see the head of tests/dsv41/fattn_test.cpp)\n", a.c_str());
            return 1;
        }
    }
    for (int m = 0; fast_attn_mode_name(m) != nullptr; ++m) {
        const bool all = mode_names.empty() || std::find(mode_names.begin(), mode_names.end(), "all") != mode_names.end();
        if (all || std::find(mode_names.begin(), mode_names.end(), fast_attn_mode_name(m)) != mode_names.end()) o.modes.push_back(m);
    }
    std::vector<float> dq;
    long long dp0 = 0;
    int dT = 0;
    if (!o.q_dump.empty()) {
        if (!read_q_dump(o.q_dump, o.q_layer, o.q_p0, dq, dp0, dT) || dT <= 0) {
            std::fprintf(stderr, "no q record (layer %d) in %s\n", o.q_layer, o.q_dump.c_str());
            return 1;
        }
        o.heads = (int) (dq.size() / ((size_t) dT * kD));
        o.p0 = dp0;
        o.rows = {dT};
        std::printf("q from %s: layer %d p0 %lld T %d heads %d\n", o.q_dump.c_str(), o.q_layer, dp0, dT, o.heads);
    }
    cudaDeviceProp prop{};
    CKT(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s (%s); heads %d ratio %d p0 %lld q-rms %g kv-rms %g sink %g q-big %g@%g kv-big %g@%g seed %u\n",
                prop.name, prop.gcnArchName, o.heads, o.ratio, o.p0, o.q_rms, o.kv_rms, o.sink, o.q_big, o.q_big_frac, o.kv_big,
                o.kv_big_frac, o.seed);
    const float scale = 1.0f / std::sqrt((float) kD);
    cudaStream_t s;
    CKT(cudaStreamCreate(&s));
    int fails = 0;
    for (int T : o.rows) {
        std::mt19937 rng(o.seed + (unsigned) T);
        Inputs in;
        in.heads = o.heads;
        if (!dq.empty()) in.q = dq;
        make_inputs(o, T, in, rng);
        // the inputs' FP16 range: Q values past FP16's max (Inf in an FP16 operand), nonzero under its smallest normal
        size_t q_over = 0, q_sub = 0;
        float q_max = 0;
        for (float x : in.q) {
            const float a = std::fabs(x);
            if (!(a <= 65504.0f)) ++q_over;
            else if (a > 0 && a < 6.103515625e-05f) ++q_sub;
            if (std::isfinite(a)) q_max = std::max(q_max, a);
        }
        std::printf("== rows %d: max|q| %.3e q>f16max %zu q<f16min %zu of %zu\n", T, q_max, q_over, q_sub, in.q.size());
        const size_t nq = in.q.size();
        float *d_q, *d_sink, *d_ref, *d_out;
        uint16_t* d_ext;
        uint8_t* d_ckv;
        int* d_sel;
        Cmp* d_cmp;
        CKT(cudaMalloc((void**) &d_q, nq * 4));
        CKT(cudaMalloc((void**) &d_ref, nq * 4));
        CKT(cudaMalloc((void**) &d_out, nq * 4));
        CKT(cudaMalloc((void**) &d_sink, in.sink.size() * 4));
        CKT(cudaMalloc((void**) &d_ext, in.ext.size() * 2));
        CKT(cudaMalloc((void**) &d_ckv, in.ckv.size()));
        CKT(cudaMalloc((void**) &d_sel, in.sel.size() * 4));
        CKT(cudaMalloc((void**) &d_cmp, sizeof(Cmp)));
        CKT(cudaMemcpy(d_q, in.q.data(), nq * 4, cudaMemcpyHostToDevice));
        CKT(cudaMemcpy(d_sink, in.sink.data(), in.sink.size() * 4, cudaMemcpyHostToDevice));
        CKT(cudaMemcpy(d_ext, in.ext.data(), in.ext.size() * 2, cudaMemcpyHostToDevice));
        CKT(cudaMemcpy(d_ckv, in.ckv.data(), in.ckv.size(), cudaMemcpyHostToDevice));
        CKT(cudaMemcpy(d_sel, in.sel.data(), in.sel.size() * 4, cudaMemcpyHostToDevice));
        // the sampled rows' double reference (the first and the last 16-row block's edges included)
        std::vector<int> rt;
        for (int t : {0, 1, 14, 15, 16, T - 1}) if (t >= 0 && t < T) rt.push_back(t);
        std::uniform_int_distribution<int> pt(0, T - 1);
        while ((int) rt.size() < o.ref_rows) rt.push_back(pt(rng));
        std::vector<std::vector<double>> refs;   // [rt][head][512]
        for (int t : rt)
            for (int h = 0; h < o.heads; ++h) {
                refs.emplace_back();
                ref_row(in, t, h, scale, refs.back());
            }
        auto vs_double = [&](const float* dev) {   // the worst sampled row's relative L2 against the double reference
            std::vector<float> row((size_t) o.heads * kD);
            double worst = 0;
            size_t k = 0;
            for (int t : rt) {
                CKT(cudaMemcpy(row.data(), dev + (size_t) t * o.heads * kD, row.size() * 4, cudaMemcpyDeviceToHost));
                for (int h = 0; h < o.heads; ++h, ++k) {
                    double d2 = 0, r2 = 0;
                    for (int e = 0; e < kD; ++e) {
                        const double d = (double) row[(size_t) h * kD + e] - refs[k][(size_t) e];
                        d2 += std::isfinite(d) ? d * d : 1e300;
                        r2 += refs[k][(size_t) e] * refs[k][(size_t) e];
                    }
                    worst = std::max(worst, r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2));
                }
            }
            return worst;
        };
        auto timed = [&](int mode, float* dst) -> double {
            if (!sparse_attn_forced(mode, d_q, d_ext, d_ckv, d_sel, kTopk, in.ar, d_sink, o.heads, T, scale, dst, s)) return -1;
            CKT(cudaStreamSynchronize(s));
            cudaEvent_t e0, e1;
            CKT(cudaEventCreate(&e0));
            CKT(cudaEventCreate(&e1));
            CKT(cudaEventRecord(e0, s));
            for (int r = 0; r < o.reps; ++r)
                sparse_attn_forced(mode, d_q, d_ext, d_ckv, d_sel, kTopk, in.ar, d_sink, o.heads, T, scale, dst, s);
            CKT(cudaEventRecord(e1, s));
            CKT(cudaEventSynchronize(e1));
            float ms = 0;
            CKT(cudaEventElapsedTime(&ms, e0, e1));
            CKT(cudaGetLastError());
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            return ms / std::max(1, o.reps);
        };
        const double t_ref = timed(-1, d_ref);
        std::printf("%-7s rows %5d: %8.3f ms/call  vs-double worst-row-rel %.3e\n", "scalar", T, t_ref, vs_double(d_ref));
        for (int m : o.modes) {
            const double ms = timed(m, d_out);
            if (ms < 0) {
                std::printf("%-7s rows %5d: not taken (heads %d)\n", fast_attn_mode_name(m), T, o.heads);
                continue;
            }
            CKT(cudaMemset(d_cmp, 0, sizeof(Cmp)));
            cmp_kernel<<<(unsigned) (nq / kD), 256, 0, s>>>(d_out, d_ref, d_cmp);
            Cmp c{};
            CKT(cudaMemcpyAsync(&c, d_cmp, sizeof(Cmp), cudaMemcpyDeviceToHost, s));
            CKT(cudaStreamSynchronize(s));
            const double nv = (double) nq;
            std::printf("%-7s rows %5d: %8.3f ms/call (x%.2f)  vs-scalar max|d| %.3e mean|d| %.3e (mean|ref| %.3e) worst-row-rel %.3e "
                        "rows>1e-2 %u rows>1e-1 %u nonfinite %u (scalar %u)  vs-double worst-row-rel %.3e\n",
                        fast_attn_mode_name(m), T, ms, t_ref / ms, ubits(c.max_d), c.sum_d / nv, c.sum_b / nv, ubits(c.max_rel),
                        c.rows_2, c.rows_1, c.nf_a, c.nf_b, vs_double(d_out));
            if (c.nf_a > c.nf_b) ++fails;
        }
        cudaFree(d_q);
        cudaFree(d_ref);
        cudaFree(d_out);
        cudaFree(d_sink);
        cudaFree(d_ext);
        cudaFree(d_ckv);
        cudaFree(d_sel);
        cudaFree(d_cmp);
    }
    std::printf(fails ? "FAIL: %d (rows, mode) with non-finite outputs the scalar kernel does not have\n" : "done\n", fails);
    return fails ? 1 : 0;
}
