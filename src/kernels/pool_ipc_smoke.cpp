// src/kernels/pool_ipc_smoke.cpp - the shared MI50 pool's go/no-go test (design note docs/flashnext/SHARED-POOL.md,
// section 7): can one process own an expert cache on an MI50 and other processes compute from it through a
// HIP IPC memory handle, bit for bit, at the same time?
//
//   pool_ipc_smoke holder <dev> <handle-file> [G] [seconds]
//       allocates G synthetic Q4_K/Q5_0 expert blobs (the Q4K pack's types, n_embd 2560, n_ff 640; seeds 1000+g) on device
//       <dev>, exports the allocation (hipIpcGetMemHandle) to <handle-file> (written to .tmp, then renamed), then sleeps until
//       <handle-file>.stop exists or <seconds> (default 600) pass, and frees.
//   pool_ipc_smoke client <dev> <handle-file> <label> [G] [NT] [iters] [start-epoch-ms]
//       opens the handle, runs native_expert_grouped over the G imported experts x NT tokens (every expert sees every token;
//       G*NT entries) and over a PRIVATE copy of the same blobs (made from the same seeds), and prints:
//         RESULT <label> equal=<imported == private, bytewise> sha=<fnv of the output> finite=<0|1>
//         LAT <label> n=<iters> p50_us=.. p90_us=.. p99_us=.. t0_ms=.. t1_ms=..   (one launch + one stream sync per
//         iteration: the helper lane's pattern), after waiting until start-epoch-ms (so two clients start together).
// Overlap is judged by the driver: each client's latency alone vs both at once, and whether their [t0, t1] overlap.
// Exit: 0 ok; 1 a mismatch or a non-finite output; 2 a HIP error; 77 not a HIP build.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cuda_runtime.h>
#if defined(STRATA_USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define POOL_IPC_HIP 1
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

namespace cpu = strata::kernels::cpu;
namespace k = strata::kernels;

namespace {
constexpr int64_t H = 2560, FF = 640;
constexpr size_t Q8_1_BYTES = 36;
constexpr int GU = 12, DN = 6;   // GGML_TYPE_Q4_K, GGML_TYPE_Q5_0: the Q4K pack's experts in every layer

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "pool_ipc_smoke: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

// cross_device_grouped_parity's synthetic_blob (random rows of different magnitudes, quantized by ggml)
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    return blob;
}

cpu::NativeFmt fmt_or_die() {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GU, DN, H, FF, f, err)) { std::fprintf(stderr, "pool_ipc_smoke: format: %s\n", err.c_str()); std::exit(2); }
    return f;
}

size_t slot_bytes(const cpu::NativeFmt& f) { return (f.bytes + 255) & ~(size_t) 255; }

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

bool exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }

uint64_t fnv(const void* p, size_t n) {
    const auto* b = (const uint8_t*) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

#if defined(POOL_IPC_HIP)
int holder(int dev, const std::string& file, int G, int seconds) {
    const cpu::NativeFmt f = fmt_or_die();
    const size_t slot = slot_bytes(f);
    ck(cudaSetDevice(dev), "set device");
    void* base = nullptr;
    ck(cudaMalloc(&base, slot * (size_t) G), "malloc pool");
    for (int g = 0; g < G; ++g) {
        const auto blob = synthetic_blob(f, 1000 + g);
        ck(cudaMemcpy((uint8_t*) base + slot * (size_t) g, blob.data(), blob.size(), cudaMemcpyHostToDevice), "fill");
    }
    ck(cudaDeviceSynchronize(), "sync");
    hipIpcMemHandle_t h;
    if (hipIpcGetMemHandle(&h, base) != hipSuccess) { std::fprintf(stderr, "pool_ipc_smoke: hipIpcGetMemHandle failed\n"); return 2; }
    const std::string tmp = file + ".tmp";
    std::FILE* o = std::fopen(tmp.c_str(), "wb");
    if (!o || std::fwrite(&h, sizeof h, 1, o) != 1 || std::fwrite(&G, sizeof G, 1, o) != 1 || std::fclose(o) != 0 ||
        std::rename(tmp.c_str(), file.c_str()) != 0) { std::fprintf(stderr, "pool_ipc_smoke: cannot write %s\n", file.c_str()); return 2; }
    std::printf("HOLDER dev=%d G=%d bytes=%zu handle=%s\n", dev, G, slot * (size_t) G, file.c_str());
    std::fflush(stdout);
    const int64_t t_end = now_ms() + (int64_t) seconds * 1000;
    while (!exists(file + ".stop") && now_ms() < t_end) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ck(cudaFree(base), "free");
    std::remove(file.c_str());
    std::printf("HOLDER done\n");
    return 0;
}

int client(int dev, const std::string& file, const std::string& label, int G, int NT, int iters, int64_t start_ms) {
    const cpu::NativeFmt f = fmt_or_die();
    const size_t slot = slot_bytes(f);
    hipIpcMemHandle_t h;
    int hg = 0;
    for (int i = 0; i < 600 && !exists(file); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::FILE* in = std::fopen(file.c_str(), "rb");
    if (!in || std::fread(&h, sizeof h, 1, in) != 1 || std::fread(&hg, sizeof hg, 1, in) != 1) {
        std::fprintf(stderr, "pool_ipc_smoke: cannot read %s\n", file.c_str());
        return 2;
    }
    std::fclose(in);
    if (G > hg) G = hg;
    ck(cudaSetDevice(dev), "set device");
    void* shared = nullptr;
    const hipError_t oe = hipIpcOpenMemHandle(&shared, h, hipIpcMemLazyEnablePeerAccess);
    if (oe != hipSuccess) { std::fprintf(stderr, "pool_ipc_smoke: hipIpcOpenMemHandle: %s\n", hipGetErrorString(oe)); return 2; }
    // the private copy: the same seeds, so the same bytes (checked)
    uint8_t* priv = nullptr;
    ck(cudaMalloc((void**) &priv, slot * (size_t) G), "malloc private");
    std::vector<uint8_t> back(f.bytes);
    bool bytes_equal = true;
    for (int g = 0; g < G; ++g) {
        const auto blob = synthetic_blob(f, 1000 + g);
        ck(cudaMemcpy(priv + slot * (size_t) g, blob.data(), blob.size(), cudaMemcpyHostToDevice), "fill private");
        ck(cudaMemcpy(back.data(), (uint8_t*) shared + slot * (size_t) g, f.bytes, cudaMemcpyDeviceToHost), "read shared");
        bytes_equal = bytes_equal && std::memcmp(back.data(), blob.data(), f.bytes) == 0;
    }
    const k::NativeExpertLayout L = k::native_expert_layout(GU, DN, H, FF);
    const int CAP = G * NT;
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::vector<float> x((size_t) NT * H);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    float *dx = nullptr, *dout = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    ck(cudaMalloc((void**) &dx, x.size() * 4), "x");
    ck(cudaMalloc((void**) &dxq, (size_t) NT * (H / 32) * Q8_1_BYTES), "xq");
    ck(cudaMalloc((void**) &dscr, k::native_expert_scratch_bytes(CAP, FF)), "scratch");
    ck(cudaMalloc((void**) &dout, (size_t) CAP * H * 4), "out");
    ck(cudaMalloc((void**) &dptr, (size_t) 2 * G * 8), "ptr");
    ck(cudaMalloc((void**) &dstart, (G + 1) * 4), "start");
    ck(cudaMalloc((void**) &dn, 4), "n");
    ck(cudaMalloc((void**) &ddst, CAP * 4), "dst");
    ck(cudaMalloc((void**) &dtok, CAP * 4), "tok");
    std::vector<unsigned long long> ptr((size_t) 2 * G);
    std::vector<int32_t> start(G + 1), dst(CAP), tok(CAP);
    for (int g = 0; g < G; ++g) {
        ptr[(size_t) g] = (unsigned long long) ((uint8_t*) shared + slot * (size_t) g);
        ptr[(size_t) (G + g)] = (unsigned long long) (priv + slot * (size_t) g);
        start[(size_t) g] = g * NT;
        for (int t = 0; t < NT; ++t) { dst[(size_t) (g * NT + t)] = g * NT + t; tok[(size_t) (g * NT + t)] = t; }
    }
    start[(size_t) G] = CAP;
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x up");
    ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr up");
    ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start up");
    ck(cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice), "n up");
    ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst up");
    ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok up");
    k::quantize_q8_1_rows(dx, NT, H, dxq, s);
    auto run = [&](const unsigned long long* p, std::vector<float>& o) {
        k::native_expert_grouped(L, p, dstart, dn, ddst, dtok, G, CAP, dxq, dscr, dout, s);
        ck(cudaGetLastError(), "grouped");
        ck(cudaStreamSynchronize(s), "grouped sync");
        o.resize((size_t) CAP * H);
        ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost), "out down");
    };
    std::vector<float> o_shared, o_priv;
    run(dptr, o_shared);
    run(dptr + G, o_priv);
    bool fin = true;
    for (float v : o_shared) fin = fin && std::isfinite(v);
    const bool equal = bytes_equal && std::memcmp(o_shared.data(), o_priv.data(), o_shared.size() * 4) == 0;
    // latency: one launch + one sync per iteration, as the helper lane does; started together with the other client
    while (start_ms > 0 && now_ms() < start_ms) std::this_thread::sleep_for(std::chrono::microseconds(200));
    std::vector<double> lat((size_t) iters);
    const int64_t t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        const auto a = std::chrono::steady_clock::now();
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, CAP, dxq, dscr, dout, s);
        ck(cudaStreamSynchronize(s), "lat sync");
        lat[(size_t) i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
    }
    const int64_t t1 = now_ms();
    std::vector<float> o_after;
    run(dptr, o_after);   // the same bits after the concurrent phase
    const bool stable = std::memcmp(o_after.data(), o_shared.data(), o_after.size() * 4) == 0;
    std::sort(lat.begin(), lat.end());
    auto q = [&](double p) { return lat[(size_t) std::min<double>((double) iters - 1, p * (double) iters)]; };
    std::printf("RESULT %s bytes_equal=%d equal=%d stable=%d sha=%016llx finite=%d G=%d NT=%d\n", label.c_str(), (int) bytes_equal,
                (int) equal, (int) stable, (unsigned long long) fnv(o_shared.data(), o_shared.size() * 4), (int) fin, G, NT);
    std::printf("LAT %s n=%d p50_us=%.1f p90_us=%.1f p99_us=%.1f t0_ms=%lld t1_ms=%lld\n", label.c_str(), iters, q(0.5), q(0.9),
                q(0.99), (long long) t0, (long long) t1);
    hipIpcCloseMemHandle(shared);
    return equal && stable && fin ? 0 : 1;
}
#endif
}  // namespace

int main(int argc, char** argv) {
#if !defined(POOL_IPC_HIP)
    (void) argc; (void) argv;
    std::fprintf(stderr, "pool_ipc_smoke: HIP builds only\n");
    return 77;
#else
    if (argc < 4) {
        std::fprintf(stderr, "usage: pool_ipc_smoke holder <dev> <file> [G] [seconds] | client <dev> <file> <label> [G] [NT] [iters] [start-ms]\n");
        return 2;
    }
    ggml_cpu_init();
    const std::string mode = argv[1];
    const int dev = std::atoi(argv[2]);
    if (mode == "holder") return holder(dev, argv[3], argc > 4 ? std::atoi(argv[4]) : 16, argc > 5 ? std::atoi(argv[5]) : 600);
    if (mode == "client" && argc >= 5)
        return client(dev, argv[3], argv[4], argc > 5 ? std::atoi(argv[5]) : 7, argc > 6 ? std::atoi(argv[6]) : 1,
                      argc > 7 ? std::atoi(argv[7]) : 2000, argc > 8 ? std::atoll(argv[8]) : 0);
    std::fprintf(stderr, "pool_ipc_smoke: unknown mode\n");
    return 2;
#endif
}
