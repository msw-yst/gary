// Strata (docs/dsv41/cards/HX-04.md, section "several clients"). One helper GPU serving K concurrent
// clients (worker engines) on one device: synthetic per-layer requests (T = 1 token, E helper experts each, the DeepSeek-V4.1
// pack's per-layer type mix), each client walking the 40 layers with a think time between its requests (its own layer work),
// and a server that answers them through gg::win_experts exactly as the helper does (H2D of x + plan, the expert kernels, D2H,
// event).  Server modes:
//   serial  - one server thread, one stream, one request a pass (the HX-03 server today)
//   union   - one server thread, one stream, a work-conserving drain: every pending request in one pass (one H2D, one
//             win_experts per type signature present - the rows of one signature are one window - one D2H, one wait)
//   streams - S server threads, each its own stream and scratch, each serial (passes overlap on the device)
// Reports requests/s, the client-seen latency (submit -> answer: queue wait + service), passes and their mean batch.
//   helper_pool_bench [--mode serial|union|streams] [--clients 1,2,4,8] [--experts 1] [--think-us 650] [--secs 3] [--streams 2]
#include "../../src/dsv41/ds4_gguf.cuh"
#include "../../src/dsv41/ds4_iq.cuh"
#include "../../src/dsv41/ds4_kernels.cuh"   // kWin, kPicks

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace gg = strata::ds4::gguf;
namespace iq = gg::iq;
namespace x3 = gg::x3;
using strata::ds4::kPicks;
using strata::ds4::kWin;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint16_t f2h(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffffu;
    uint32_t hm = m >> 13; const uint32_t rem = m & 0x1fffu;
    uint32_t v = ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;
    return (uint16_t) (s | v);
}

constexpr int E = 5120, FF = 2304, K = 6, NL = 40;
// the pack's (gate/up, down) types by layer (pack-2.25 tensors.txt: w1 = w3's type, w2's type)
static const int kSig[NL][2] = {
    {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 16}, {16, 22}, {17, 17},
    {16, 22}, {16, 17}, {17, 17}, {16, 16}, {16, 17}, {16, 16}, {16, 22}, {22, 17}, {16, 17}, {16, 16},
    {16, 16}, {16, 22}, {16, 22}, {17, 18}, {17, 22}, {22, 22}, {17, 17}, {17, 17}, {16, 17}, {16, 17},
    {22, 16}, {17, 16}, {16, 22}, {16, 16}, {16, 17}, {16, 16}, {16, 16}, {16, 16}, {17, 22}, {16, 22}};

struct LayerB {
    x3::ExpertLayout lay;
    size_t blob = 0;
    std::vector<uint8_t*> ex;   // device blobs
};

// a request (one client's mailbox): written by the client, served by a server
struct alignas(64) Slot {
    std::atomic<uint64_t> req{0}, resp{0};
    std::atomic<int> claim{0};   // (streams mode: a server took it)
    int layer = 0, n = 0;
    int ex[kPicks];
    float w[kPicks];
    float x[E];
    float out[E];
    double t_submit = 0;
};

struct Plan {   // one window's plan on the device, as ds4_helper.cu's
    unsigned long long plan_ptr[kWin * kPicks];
    float plan_w[kWin * kPicks];
    unsigned long long grp_ptr[kWin * kPicks];
    int grp_start[kWin * kPicks + 1];
    int grp_ent[kWin * kPicks];
    int n_groups;
};

struct Lane {   // a server's stream and scratch
    hipStream_t st;
    hipEvent_t ev;
    uint8_t* h_stage;   // pinned: [Plan x signatures][x rows]
    uint8_t* d_stage;
    float *d_h, *d_part, *d_out, *h_out;
    void open() {
        CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
        CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        if (!gg::prepare(st)) { std::printf("prepare failed\n"); std::exit(2); }
        const size_t sb = sizeof(Plan) * kWin + (size_t) kWin * E * 4;
        CK(hipHostMalloc((void**) &h_stage, sb, 0));
        CK(hipMalloc((void**) &d_stage, sb));
        CK(hipMalloc((void**) &d_h, (size_t) kWin * kPicks * FF * 4));
        CK(hipMalloc((void**) &d_part, (size_t) kWin * kPicks * E * 4));
        CK(hipMalloc((void**) &d_out, (size_t) kWin * E * 4));
        CK(hipHostMalloc((void**) &h_out, (size_t) kWin * E * 4, 0));
    }
};

// one pass over the given requests (at most kWin): rows grouped by type signature, each signature one window
static void pass(Lane& L, const std::vector<LayerB>& Ls, Slot** rq, int nr) {
    int sig_of[kWin], row_of[kWin], nsig = 0, sig_rows[kWin], sig_key[kWin], sig_layer[kWin];
    for (int i = 0; i < nr; ++i) {
        const int key = kSig[rq[i]->layer][0] * 256 + kSig[rq[i]->layer][1];
        int s = 0;
        while (s < nsig && sig_key[s] != key) ++s;
        if (s == nsig) { sig_key[nsig] = key; sig_rows[nsig] = 0; sig_layer[nsig] = rq[i]->layer; ++nsig; }
        sig_of[i] = s;
        row_of[i] = sig_rows[s]++;
    }
    int row0[kWin + 1];
    row0[0] = 0;
    for (int s = 0; s < nsig; ++s) row0[s + 1] = row0[s] + sig_rows[s];
    Plan* hp = (Plan*) L.h_stage;
    float* hx = (float*) (L.h_stage + sizeof(Plan) * (size_t) nsig);   // x rows right after the plans used
    for (int s = 0; s < nsig; ++s) { std::memset(hp[s].plan_ptr, 0, sizeof hp[s].plan_ptr); std::memset(hp[s].plan_w, 0, sizeof hp[s].plan_w); }
    std::vector<std::pair<unsigned long long, int>> by[kWin];
    for (int i = 0; i < nr; ++i) {
        const Slot& q = *rq[i];
        const int s = sig_of[i], t = row_of[i];
        std::memcpy(hx + (size_t) (row0[s] + t) * E, q.x, E * 4);
        for (int j = 0; j < q.n; ++j) {
            const unsigned long long p = (unsigned long long) Ls[q.layer].ex[q.ex[j]];
            hp[s].plan_ptr[t * kPicks + j] = p;
            hp[s].plan_w[t * kPicks + j] = q.w[j];
            by[s].push_back({p, t * kPicks + j});
        }
    }
    for (int s = 0; s < nsig; ++s) {
        std::sort(by[s].begin(), by[s].end());
        int ng = 0, ne = 0;
        for (size_t a = 0; a < by[s].size();) {
            const unsigned long long p = by[s][a].first;
            hp[s].grp_ptr[ng] = p;
            hp[s].grp_start[ng] = ne;
            while (a < by[s].size() && by[s][a].first == p) hp[s].grp_ent[ne++] = by[s][a++].second;
            ++ng;
        }
        hp[s].grp_start[ng] = ne;
        hp[s].n_groups = ng;
    }
    const size_t bytes = sizeof(Plan) * (size_t) nsig + (size_t) nr * E * 4;
    CK(hipMemcpyAsync(L.d_stage, L.h_stage, bytes, hipMemcpyHostToDevice, L.st));
    Plan* dp = (Plan*) L.d_stage;
    const float* dx = (const float*) (L.d_stage + sizeof(Plan) * (size_t) nsig);
    for (int s = 0; s < nsig; ++s) {
        x3::WinArgs a;
        a.grp_ptr = dp[s].grp_ptr; a.grp_start = dp[s].grp_start; a.grp_ent = dp[s].grp_ent; a.n_groups = &dp[s].n_groups;
        a.plan_ptr = dp[s].plan_ptr; a.plan_w = dp[s].plan_w;
        a.x = dx + (size_t) row0[s] * E; a.x_stride = E;
        a.h = L.d_h; a.part = L.d_part;
        a.out = L.d_out + (size_t) row0[s] * E; a.out_stride = E;
        a.lay = Ls[sig_layer[s]].lay;
        a.T = sig_rows[s]; a.k = K; a.n_embd = E; a.n_ff = FF; a.limit = 10.0f;
        if (!gg::win_experts(a, L.st)) { std::printf("win_experts refused\n"); std::exit(2); }
    }
    CK(hipMemcpyAsync(L.h_out, L.d_out, (size_t) nr * E * 4, hipMemcpyDeviceToHost, L.st));
    CK(hipEventRecord(L.ev, L.st));
    hipError_t e;
    while ((e = hipEventQuery(L.ev)) == hipErrorNotReady) {}
    CK(e);
    for (int i = 0; i < nr; ++i) std::memcpy(rq[i]->out, L.h_out + (size_t) (row0[sig_of[i]] + row_of[i]) * E, E * 4);
}

int main(int argc, char** argv) {
    std::string mode = "serial";
    std::vector<int> clients = {1, 2, 4, 8};
    int n_ex = 1, think = 650, nstreams = 2, per_layer = 3;
    double secs = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : (char*) "0"; };
        if (a == "--mode") mode = nx();
        else if (a == "--clients") { clients.clear(); for (const char* p = nx(); *p;) { clients.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; } }
        else if (a == "--experts") n_ex = std::atoi(nx());
        else if (a == "--think-us") think = std::atoi(nx());
        else if (a == "--secs") secs = std::atof(nx());
        else if (a == "--streams") nstreams = std::atoi(nx());
        else if (a == "--per-layer") per_layer = std::atoi(nx());
        else { std::printf("unknown %s\n", a.c_str()); return 2; }
    }
    if (n_ex < 1 || n_ex > K) { std::printf("--experts 1..%d\n", K); return 2; }
    hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, 0));
    std::printf("device 0: %s  mode %s  experts/request %d  think %d us  g906 kernels %s\n", p.gcnArchName, mode.c_str(), n_ex, think,
                iq::g906_force(-1) ? "on" : "off");
    std::vector<LayerB> Ls(NL);
    std::mt19937_64 rng(5);
    size_t total = 0;
    for (int l = 0; l < NL; ++l) {
        const int tg = kSig[l][0], td = kSig[l][1];
        const size_t rg = iq::row_bytes(tg, E), rd = iq::row_bytes(td, FF);
        auto al = [](size_t n) { return (n + 255) / 256 * 256; };
        LayerB& L = Ls[l];
        L.lay.fmt = gg::kFmtMXFP4;
        L.lay.K[0] = L.lay.K[1] = tg; L.lay.K[2] = td;
        L.lay.trellis[0] = 0; L.lay.trellis[1] = al(FF * rg); L.lay.trellis[2] = 2 * al(FF * rg);
        L.blob = L.lay.trellis[2] + al((size_t) E * rd);
        std::vector<uint8_t> h(L.blob);
        for (int e = 0; e < per_layer; ++e) {
            for (size_t i = 0; i < h.size(); i += 8) { const uint64_t r = rng(); std::memcpy(&h[i], &r, std::min<size_t>(8, h.size() - i)); }
            std::uniform_real_distribution<float> sd(0.002f, 0.02f);
            for (int r = 0; r < 3; ++r) {
                const int ty = L.lay.K[r], bb = iq::block_bytes(ty);
                const size_t n = r < 2 ? FF * rg : (size_t) E * rd;
                for (size_t b = 0; b < n; b += (size_t) bb) { const uint16_t d = f2h(sd(rng)); std::memcpy(&h[L.lay.trellis[r] + b], &d, 2); }
            }
            uint8_t* d; CK(hipMalloc(&d, L.blob)); CK(hipMemcpy(d, h.data(), L.blob, hipMemcpyHostToDevice));
            L.ex.push_back(d);
            total += L.blob;
        }
    }
    std::printf("%d layers x %d experts = %.2f GiB of blobs\n", NL, per_layer, total / 1073741824.0);
    const int nlanes = mode == "streams" ? nstreams : 1;
    std::vector<Lane> lanes(nlanes);
    for (auto& L : lanes) L.open();
    {   // warm every layer's kernels on every lane
        Slot* w = new Slot;
        for (int l = 0; l < NL; ++l) {
            w->layer = l; w->n = 1; w->ex[0] = 0; w->w[0] = 0.5f; std::memset(w->x, 0, sizeof w->x);
            for (auto& L : lanes) { Slot* q[1] = {w}; pass(L, Ls, q, 1); }
        }
        delete w;
    }
    for (int nc : clients) {
        if (nc < 1 || nc > kWin) { std::printf("clients 1..%d\n", kWin); continue; }
        std::vector<Slot*> slots(nc);
        for (auto& s : slots) {
            s = new Slot;
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float& v : s->x) v = nd(rng);
        }
        std::atomic<bool> stop{false};
        std::atomic<long long> passes{0}, batched{0};
        std::vector<std::vector<double>> lat(nc);
        std::vector<double> svc;
        std::vector<int> svc_b;
        std::mutex svc_mu;
        std::vector<std::thread> srv;
        for (int li = 0; li < nlanes; ++li)
            srv.emplace_back([&, li] {
                Lane& L = lanes[li];
                Slot* rq[kWin];
                int rr = 0;
                std::vector<double> my_svc;
                std::vector<int> my_b;
                while (!stop.load(std::memory_order_relaxed)) {
                    int nr = 0;
                    if (mode == "union") {
                        for (int c = 0; c < nc && nr < kWin; ++c) {
                            Slot* s = slots[c];
                            if (s->req.load(std::memory_order_acquire) != s->resp.load(std::memory_order_relaxed)) rq[nr++] = s;
                        }
                    } else {   // one request: round robin from the last served (streams: claim it)
                        for (int k = 0; k < nc && nr == 0; ++k) {
                            Slot* s = slots[(rr + k) % nc];
                            const uint64_t q = s->req.load(std::memory_order_acquire);
                            if (q == s->resp.load(std::memory_order_relaxed)) continue;
                            if (mode == "streams") { int z = 0; if (!s->claim.compare_exchange_strong(z, 1)) continue; }
                            rq[nr++] = s;
                            rr = (rr + k + 1) % nc;
                        }
                    }
                    if (nr == 0) continue;
                    const double t0 = now_us();
                    pass(L, Ls, rq, nr);
                    my_svc.push_back(now_us() - t0);
                    my_b.push_back(nr);
                    for (int i = 0; i < nr; ++i) {
                        const uint64_t q = rq[i]->req.load(std::memory_order_relaxed);
                        rq[i]->claim.store(0, std::memory_order_relaxed);
                        rq[i]->resp.store(q, std::memory_order_release);
                    }
                    ++passes;
                    batched += nr;
                }
                std::lock_guard<std::mutex> lk(svc_mu);
                svc.insert(svc.end(), my_svc.begin(), my_svc.end());
                svc_b.insert(svc_b.end(), my_b.begin(), my_b.end());
            });
        std::vector<std::thread> cl;
        const double t_start = now_us();
        for (int c = 0; c < nc; ++c)
            cl.emplace_back([&, c] {
                Slot& s = *slots[c];
                std::mt19937_64 r(100 + c);
                int layer = (int) (r() % NL);
                uint64_t seq = 0;
                while (now_us() - t_start < secs * 1e6) {
                    s.layer = layer;
                    s.n = n_ex;
                    for (int j = 0; j < n_ex; ++j) { s.ex[j] = (int) ((r() % (uint64_t) per_layer + j) % per_layer); s.w[j] = 0.1f + 0.1f * j; }
                    s.t_submit = now_us();
                    s.req.store(++seq, std::memory_order_release);
                    while (s.resp.load(std::memory_order_acquire) != seq) {}
                    lat[c].push_back(now_us() - s.t_submit);
                    const double t = now_us();
                    while (now_us() - t < think) {}
                    layer = (layer + 1) % NL;
                }
            });
        for (auto& t : cl) t.join();
        const double wall = (now_us() - t_start) / 1e6;
        stop.store(true);
        for (auto& t : srv) t.join();
        std::vector<double> all;
        for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
        std::sort(all.begin(), all.end());
        double mean = 0;
        for (double v : all) mean += v;
        mean /= (double) std::max<size_t>(1, all.size());
        auto pct = [&](double q) { return all.empty() ? 0.0 : all[std::min(all.size() - 1, (size_t) (q * (double) all.size()))]; };
        double sb[kWin + 1] = {};
        int nb[kWin + 1] = {};
        double busy = 0;
        for (size_t i = 0; i < svc.size(); ++i) { sb[svc_b[i]] += svc[i]; ++nb[svc_b[i]]; busy += svc[i]; }
        std::printf("clients %2d: %7.0f req/s  latency mean %6.1f us p50 %6.1f p99 %6.1f  passes %lld batch %.2f  busy %3.0f%%  service by batch:",
                    nc, (double) all.size() / wall, mean, pct(0.5), pct(0.99), (long long) passes.load(),
                    (double) batched.load() / (double) std::max<long long>(1, passes.load()), 100.0 * busy / (wall * 1e6 * nlanes));
        for (int b = 1; b <= kWin; ++b)
            if (nb[b] > 0) std::printf(" %d:%.0fus(n%d)", b, sb[b] / nb[b], nb[b]);
        std::printf("\n");
        for (auto* s : slots) delete s;
    }
    return 0;
}
