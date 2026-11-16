// QBENCH (2026-11-16): reference timings of a routed expert's decode through a ggml build (upstream llama.cpp b11870 or
// castagna-veloce), on the same expert bytes and shapes as ds4_iq_helper_bench --pool-file.  Host code only: it links the
// tree's libggml / libggml-base and runs on its first GPU device (pin the card with ROCR_VISIBLE_DEVICES).
//
//   --mode expert : one routed expert on B rows, as llama.cpp's build_moe_ffn runs it - MUL_MAT_ID gate, MUL_MAT_ID up,
//                   GLU swiglu_clamp (limit 10), MUL_MAT_ID down - with every row picking this expert (n_used 1).
//                   NX experts (default 40) in one 3-D tensor per role; the graph runs one sub-graph per expert (each reads
//                   a different expert: cold, as in the engine bench).  us per expert = graph time / NX.
//                   Weights: --pool-file F.qbx (real bytes: IQ2_XXS / IQ2_XS / IQ2_S / MXFP4) or --synth TG TD (legal
//                   random blocks, e.g. Q2_K Q3_K = 10 11).  --ff FN takes the first FN rows of gate/up and columns of down
//                   (the tensor-parallel slice shape).
//   --mode moe1   : castagna's fused pattern (batch 1): n_used 6 routed experts (MUL_MAT_ID x3 + GLU) + llama.cpp's
//                   weighted reduction (MUL by the router weights, views, adds) + a shared expert (MUL_MAT x3 + GLU) + ADD,
//                   Q2_K gate/up, Q3_K down (synthetic), NL layers (default 40) of distinct weights in one graph.
//                   us per expert-equivalent = graph time / NL / 7 (6 routed + the shared one, all M rows).  On castagna's
//                   build this is ggml_cuda_moe1_ffn (GGML_CUDA_MOE1=0 turns it off); elsewhere the unfused ops.
//   --B 1,2,4,8 --reps 200 --warm 10 --nx 40 --nl 40
// Output: one "QG" line per B: p50 / p90 / mean us per expert (or expert-equivalent), weight bytes per expert, GB/s at p50.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<int> ints(const char* p) {
    std::vector<int> v;
    while (*p) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
static uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}
// legal random blocks: random bytes, then the F16 scale fields set to sane values (no inf / nan)
static void synth_rows(std::mt19937_64& rng, ggml_type t, size_t bytes, uint8_t* p) {
    for (size_t i = 0; i < bytes; i += 8) { const uint64_t r = rng(); std::memcpy(p + i, &r, std::min<size_t>(8, bytes - i)); }
    const size_t bb = ggml_type_size(t);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (size_t b = 0; b + bb <= bytes; b += bb) {
        uint8_t* q = p + b;
        uint16_t d = f2h(sd(rng)), m = f2h(sd(rng));
        switch (t) {
            case GGML_TYPE_Q2_K: std::memcpy(q + 80, &d, 2); std::memcpy(q + 82, &m, 2); break;   // scales[16] qs[64] d dmin
            case GGML_TYPE_Q3_K: std::memcpy(q + 108, &d, 2); break;                             // hmask[32] qs[64] scales[12] d
            case GGML_TYPE_MXFP4: q[0] = (uint8_t) (120 + rng() % 8); break;                   // e8m0
            default: std::memcpy(q, &d, 2); break;                                               // the i-quants: d first
        }
    }
}
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::string mode = "expert", pf;
    int tgi = -1, tdi = -1, ffs = 0, reps = 200, warm = 10, NX = 40, NL = 40, E = 5120, FFw = 2304;
    std::vector<int> Bs = {1, 2, 4, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*) "0"; };
        if (a == "--mode") mode = nx();
        else if (a == "--pool-file") pf = nx();
        else if (a == "--synth") { tgi = std::atoi(nx()); tdi = std::atoi(nx()); }
        else if (a == "--ff") ffs = std::atoi(nx());
        else if (a == "--B") Bs = ints(nx());
        else if (a == "--reps") reps = std::atoi(nx());
        else if (a == "--warm") warm = std::atoi(nx());
        else if (a == "--nx") NX = std::atoi(nx());
        else if (a == "--nl") NL = std::atoi(nx());
        else { std::printf("unknown %s\n", a.c_str()); return 2; }
    }
    std::vector<uint8_t> pool;   // qbx: per expert w1 w3 w2
    if (!pf.empty()) {
        FILE* f = std::fopen(pf.c_str(), "rb");
        char mg[4]; uint32_t hd[5];
        if (!f || std::fread(mg, 1, 4, f) != 4 || std::memcmp(mg, "QBX1", 4) || std::fread(hd, 4, 5, f) != 5) { std::printf("bad pool\n"); return 2; }
        tgi = (int) hd[0]; tdi = (int) hd[1]; E = (int) hd[2]; FFw = (int) hd[3]; NX = std::min(NX, (int) hd[4]);
        const size_t per = 2 * ggml_row_size((ggml_type) tgi, E) * FFw + ggml_row_size((ggml_type) tdi, FFw) * E;
        pool.resize(per * NX);
        if (std::fread(pool.data(), 1, pool.size(), f) != pool.size()) { std::printf("short pool\n"); return 2; }
        std::fclose(f);
    }
    if (mode == "moe1") { tgi = GGML_TYPE_Q2_K; tdi = GGML_TYPE_Q3_K; Bs = {1}; }
    if (tgi < 0) { std::printf("need --pool-file or --synth\n"); return 2; }
    const ggml_type tg = (ggml_type) tgi, td = (ggml_type) tdi;
    const int FF = ffs > 0 ? ffs : FFw;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { std::printf("no GPU device\n"); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    std::printf("QGGML device %s (%s) | mode %s %s/%s E %d FF %d (of %d) NX %d NL %d\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                mode.c_str(), ggml_type_name(tg), ggml_type_name(td), E, FF, FFw, NX, NL);
    const size_t rg = ggml_row_size(tg, E), rd = ggml_row_size(td, FF), rdw = ggml_row_size(td, FFw);
    const size_t ebytes = 2 * rg * FF + rd * E;   // one expert's weight bytes at this shape
    std::mt19937_64 rng(77);
    int rc = 0;
    for (const int B : Bs) {
        const int nsub = mode == "moe1" ? NL : NX;
        ggml_init_params wp = {ggml_tensor_overhead() * (size_t) (16 * nsub + 64), nullptr, true};
        ggml_context* wc = ggml_init(wp);
        std::vector<ggml_tensor*> Wg, Wu, Wd, Sg, Su, Sd;
        ggml_tensor *x, *ids = nullptr, *rw = nullptr;
        std::vector<ggml_tensor*> idsv;
        const int nused = mode == "moe1" ? 6 : 1;
        if (mode == "moe1") {
            for (int l = 0; l < NL; ++l) {
                Wg.push_back(ggml_new_tensor_3d(wc, tg, E, FF, nused)); Wu.push_back(ggml_new_tensor_3d(wc, tg, E, FF, nused));
                Wd.push_back(ggml_new_tensor_3d(wc, td, FF, E, nused));
                Sg.push_back(ggml_new_tensor_2d(wc, tg, E, FF)); Su.push_back(ggml_new_tensor_2d(wc, tg, E, FF)); Sd.push_back(ggml_new_tensor_2d(wc, td, FF, E));
            }
            x = ggml_new_tensor_2d(wc, GGML_TYPE_F32, E, 1);
            ids = ggml_new_tensor_2d(wc, GGML_TYPE_I32, nused, 1);
            rw = ggml_new_tensor_3d(wc, GGML_TYPE_F32, 1, nused, 1);
        } else {
            Wg.push_back(ggml_new_tensor_3d(wc, tg, E, FF, NX)); Wu.push_back(ggml_new_tensor_3d(wc, tg, E, FF, NX));
            Wd.push_back(ggml_new_tensor_3d(wc, td, FF, E, NX));
            x = ggml_new_tensor_3d(wc, GGML_TYPE_F32, E, 1, B);
            for (int e = 0; e < NX; ++e) idsv.push_back(ggml_new_tensor_2d(wc, GGML_TYPE_I32, 1, B));
        }
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wc, be);
        if (!wb) { std::printf("weights do not fit\n"); return 2; }
        {   // weights
            if (mode == "moe1") {
                for (int l = 0; l < NL; ++l)
                    for (ggml_tensor* t : {Wg[l], Wu[l], Wd[l], Sg[l], Su[l], Sd[l]}) {
                        std::vector<uint8_t> h(ggml_nbytes(t)); synth_rows(rng, t->type, h.size(), h.data()); ggml_backend_tensor_set(t, h.data(), 0, h.size());
                    }
                std::vector<int32_t> id(nused); for (int i = 0; i < nused; ++i) id[i] = i;
                ggml_backend_tensor_set(ids, id.data(), 0, id.size() * 4);
                std::vector<float> w(nused, 1.0f / nused); ggml_backend_tensor_set(rw, w.data(), 0, w.size() * 4);
            } else {
                std::vector<uint8_t> hg(rg * FF), hu(rg * FF), hd(rd * E);
                const size_t per = 2 * rg * FFw + rdw * E;
                for (int e = 0; e < NX; ++e) {
                    if (!pool.empty()) {
                        const uint8_t* b = &pool[(size_t) e * per];
                        std::memcpy(hg.data(), b, rg * FF);                     // gate rows 0..FF-1
                        std::memcpy(hu.data(), b + rg * FFw, rg * FF);          // up rows
                        for (int r = 0; r < E; ++r) std::memcpy(&hd[(size_t) r * rd], b + 2 * rg * FFw + (size_t) r * rdw, rd);   // down columns 0..FF-1
                    } else {
                        synth_rows(rng, tg, hg.size(), hg.data()); synth_rows(rng, tg, hu.size(), hu.data()); synth_rows(rng, td, hd.size(), hd.data());
                    }
                    ggml_backend_tensor_set(Wg[0], hg.data(), (size_t) e * hg.size(), hg.size());
                    ggml_backend_tensor_set(Wu[0], hu.data(), (size_t) e * hu.size(), hu.size());
                    ggml_backend_tensor_set(Wd[0], hd.data(), (size_t) e * hd.size(), hd.size());
                    std::vector<int32_t> id(B, e); ggml_backend_tensor_set(idsv[e], id.data(), 0, id.size() * 4);
                }
            }
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> xv(ggml_nelements(x)); for (auto& v : xv) v = nd(rng);
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
        }
        // the graph
        ggml_init_params gp = {ggml_tensor_overhead() * (size_t) (64 * nsub + 256) + ggml_graph_overhead_custom(16384, false), nullptr, true};
        ggml_context* gc = ggml_init(gp);
        ggml_cgraph* gf = ggml_new_graph_custom(gc, 16384, false);
        if (mode == "moe1") {
            for (int l = 0; l < NL; ++l) {   // llama.cpp build_moe_ffn's shapes (n_tokens 1), then its shared expert and add
                ggml_tensor* cur = ggml_reshape_3d(gc, x, E, 1, 1);
                ggml_tensor* g = ggml_mul_mat_id(gc, Wg[l], cur, ids);
                ggml_tensor* u = ggml_mul_mat_id(gc, Wu[l], cur, ids);
                ggml_tensor* hgl = ggml_swiglu_clamp(gc, g, u, 10.0f);
                ggml_tensor* ex = ggml_mul_mat_id(gc, Wd[l], hgl, ids);       // [E, nused, 1]
                ex = ggml_mul(gc, ex, rw);
                ggml_tensor* views[8];
                for (int i = 0; i < nused; ++i) views[i] = ggml_view_2d(gc, ex, E, 1, ex->nb[2], i * ex->nb[1]);
                ggml_tensor* mo = views[0];
                for (int i = 1; i < nused; ++i) mo = ggml_add(gc, mo, views[i]);
                ggml_tensor* sg = ggml_mul_mat(gc, Sg[l], x);
                ggml_tensor* su = ggml_mul_mat(gc, Su[l], x);
                ggml_tensor* sh = ggml_swiglu_clamp(gc, sg, su, 10.0f);
                ggml_tensor* sd = ggml_mul_mat(gc, Sd[l], sh);
                ggml_build_forward_expand(gf, ggml_add(gc, mo, sd));
            }
        } else {
            for (int e = 0; e < NX; ++e) {
                ggml_tensor* g = ggml_mul_mat_id(gc, Wg[0], x, idsv[e]);
                ggml_tensor* u = ggml_mul_mat_id(gc, Wu[0], x, idsv[e]);
                ggml_tensor* h = ggml_swiglu_clamp(gc, g, u, 10.0f);
                ggml_build_forward_expand(gf, ggml_mul_mat_id(gc, Wd[0], h, idsv[e]));
            }
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { std::printf("graph alloc failed\n"); return 2; }
        for (int r = 0; r < warm; ++r) { if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { std::printf("compute failed\n"); return 2; } }
        ggml_backend_synchronize(be);
        std::vector<double> us((size_t) reps);
        const double per_unit = mode == "moe1" ? (double) NL * 7 : (double) NX;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_us();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            us[(size_t) r] = (now_us() - t0) / per_unit;
        }
        std::vector<double> so = us; std::sort(so.begin(), so.end());
        const double p50 = so[(size_t) (reps * 0.5)], p90 = so[std::min((size_t) reps - 1, (size_t) (reps * 0.9))];
        double mean = 0; for (double v : us) mean += v; mean /= reps;
        // nonfinite check on one output
        ggml_tensor* last = ggml_graph_node(gf, -1);
        std::vector<float> o(ggml_nelements(last)); ggml_backend_tensor_get(last, o.data(), 0, o.size() * 4);
        long nf = 0; double mx = 0; for (float v : o) { if (!std::isfinite(v)) ++nf; else mx = std::max(mx, (double) std::fabs(v)); }
        std::printf("QG dev=%s mode=%s tg=%s td=%s ff=%d B=%d reps=%d p50_us=%.2f p90_us=%.2f mean_us=%.2f bytes=%zu GBps_p50=%.1f out_maxabs=%.3e nonfinite=%ld nodes=%d\n",
                    ggml_backend_dev_description(dev), mode.c_str(), ggml_type_name(tg), ggml_type_name(td), FF, B, reps, p50, p90, mean, ebytes,
                    ebytes / (p50 * 1e3), mx, nf, ggml_graph_n_nodes(gf));
        std::fflush(stdout);
        if (nf) rc = 4;
        ggml_gallocr_free(ga); ggml_free(gc); ggml_backend_buffer_free(wb); ggml_free(wc);
    }
    ggml_backend_free(be);
    return rc;
}
