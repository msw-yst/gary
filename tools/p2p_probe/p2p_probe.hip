// tools/p2p_probe/p2p_probe.hip - can the GPUs on this box reach each other directly, and what does a small hand-off cost?
//   hipcc -O2 --offload-arch=gfx1100 --offload-arch=gfx906 -o p2p_probe p2p_probe.hip && ./p2p_probe [bytes=16384] [iters=2000] [uncached]
// 1. hipDeviceCanAccessPeer for every ordered pair; enable it where possible.
// 2. Per pair, a small copy (default 16 KB, about one decode hand-off) three ways, timed end to end on the host, median:
//    (a) hipMemcpyPeerAsync src->dst (direct, if peer access is on; else the runtime stages it)
//    (b) staged: D2H into pinned host memory, sync, H2D to the other device, sync   (what the engine does today)
//    (c) remote flag: a kernel on src writes the payload + a flag (fine-grained, system-scope atomics) into dst's memory (peer pointer); a kernel on dst spins on the
//        flag and acks into src's memory; the round trip, halved, is the one-way GPU->GPU latency with no CPU in the loop.
// 3. Large-copy bandwidth per pair (64 MB, peer copy), to show the link, not the latency.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("  %s: %s\n", #x, hipGetErrorString(e_)); } } while (0)
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

__global__ void send_k(volatile unsigned* dst_flag, unsigned* dst_buf, const unsigned* src_buf, int n, unsigned seq,
                       volatile unsigned* ack, int iters, unsigned* ack_timeout) {
    __shared__ int stop;
    if (threadIdx.x == 0) stop = 0;
    __syncthreads();
    for (int it = 0; it < iters; ++it) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst_buf[i] = src_buf[i] + it;
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store((unsigned*) dst_flag, seq + it, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            for (long long w = 0; __hip_atomic_load((unsigned*) ack, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq + it; ++w) if (w > (1ll << 26)) { ack_timeout[0] = 1; stop = 1; break; }
        }
        __syncthreads();
        if (stop) return;   // one lost ack ends the run (each further iteration would spin its whole bound again)
    }
}
__global__ void recv_k(volatile unsigned* flag, volatile unsigned* ack_remote, int iters, unsigned seq) {
    if (threadIdx.x != 0) return;
    for (int it = 0; it < iters; ++it) {
        for (long long w = 0; __hip_atomic_load((unsigned*) flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq + it; ++w)
            if (w > (1ll << 26)) return;   // bounded: a write that never lands
        __hip_atomic_store((unsigned*) ack_remote, seq + it, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // print as it goes (a hang must not swallow the output)
    const size_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 16384;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    const bool uncached = argc > 3 && std::string(argv[3]) == "uncached";   // the flags hipDeviceMallocUncached, not fine-grained
    int n = 0;
    CK(hipGetDeviceCount(&n));
    std::printf("devices: %d, payload %zu bytes, %d iterations, flags %s\n", n, bytes, iters, uncached ? "uncached" : "fine-grained");
    for (int d = 0; d < n; ++d) {
        hipDeviceProp_t p{};
        CK(hipGetDeviceProperties(&p, d));
        std::printf("  HIP %d: %s %s pci %02x:%02x.0 large-bar %d\n", d, p.name, p.gcnArchName, p.pciBusID, p.pciDeviceID,
                    p.isLargeBar);
    }
    std::printf("\ncan-access-peer matrix (row = from, col = to):\n     ");
    for (int j = 0; j < n; ++j) std::printf(" %3d", j);
    std::printf("\n");
    std::vector<std::vector<int>> can(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        std::printf("  %2d ", i);
        for (int j = 0; j < n; ++j) {
            if (i != j) CK(hipDeviceCanAccessPeer(&can[i][j], i, j));
            std::printf(" %3s", i == j ? "-" : can[i][j] ? "yes" : "no");
        }
        std::printf("\n");
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j && can[i][j]) {
                CK(hipSetDevice(i));
                hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                    std::printf("  enable %d->%d: %s\n", i, j, hipGetErrorString(e));
            }
    std::vector<void*> buf(n), big(n);
    std::vector<unsigned*> flag(n);
    std::vector<hipStream_t> st(n);
    void* host = nullptr;
    CK(hipHostMalloc(&host, bytes, hipHostMallocDefault));
    const size_t bigb = 64ull << 20;
    for (int d = 0; d < n; ++d) {
        CK(hipSetDevice(d));
        CK(hipMalloc(&buf[d], bytes));
        CK(hipMalloc(&big[d], bigb));
        // the flags fine-grained (coherent while kernels run, across devices); coarse-grained hipMalloc memory is not
        CK(hipExtMallocWithFlags((void**) &flag[d], 1024, uncached ? hipDeviceMallocUncached : hipDeviceMallocFinegrained));   // [0] flag, [32..63] ack, [64] timeout (unsigned offsets)
        CK(hipMemset(flag[d], 0, 1024));
        CK(hipStreamCreateWithFlags(&st[d], hipStreamNonBlocking));
    }
    for (int d = 0; d < n; ++d) { CK(hipSetDevice(d)); CK(hipDeviceSynchronize()); }
    // (d) the same flag round trip through pinned host-mapped memory (what the engine's --threads hand-off does): payload and
    // flag written by src into host memory, dst spins on the host flag and acks into host memory
    unsigned *hflag = nullptr, *hbuf = nullptr;
    CK(hipHostMalloc((void**) &hflag, 1024, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    CK(hipHostMalloc((void**) &hbuf, bytes, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    std::printf("\nsmall hand-off, %zu bytes, median one-way microseconds:\n", bytes);
    std::printf("  %-8s %10s %10s %15s %12s   %-14s %15s %15s\n", "pair", "peer copy", "staged", "peer flag 1-way", "64MB GB/s", "peer copy data", "host flag 1-way", "peer pull 1-way");
    unsigned seq = 1000;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            std::vector<double> a, b;
            for (int k = 0; k < 200; ++k) {
                CK(hipSetDevice(i));
                auto t0 = clk::now();
                CK(hipMemcpyPeerAsync(buf[j], j, buf[i], i, bytes, st[i]));
                CK(hipStreamSynchronize(st[i]));
                a.push_back(us(t0, clk::now()));
                t0 = clk::now();
                CK(hipMemcpyAsync(host, buf[i], bytes, hipMemcpyDeviceToHost, st[i]));
                CK(hipStreamSynchronize(st[i]));
                CK(hipSetDevice(j));
                CK(hipMemcpyAsync(buf[j], host, bytes, hipMemcpyHostToDevice, st[j]));
                CK(hipStreamSynchronize(st[j]));
                b.push_back(us(t0, clk::now()));
            }
            // does a peer copy actually move the bytes? (src/core/peer_experts.cpp: "success having moved nothing" on some pairs)
            std::string ok = "?";
            {
                std::vector<unsigned> pat(bytes / 4), got(bytes / 4, 0);
                for (size_t q = 0; q < pat.size(); ++q) pat[q] = 0x9e3779b9u * (unsigned) (q + 1) + (unsigned) (i * 16 + j);
                CK(hipSetDevice(j)); CK(hipMemset(buf[j], 0, bytes)); CK(hipDeviceSynchronize());
                CK(hipSetDevice(i)); CK(hipMemcpy(buf[i], pat.data(), bytes, hipMemcpyHostToDevice));
                CK(hipMemcpyPeerAsync(buf[j], j, buf[i], i, bytes, st[i])); CK(hipStreamSynchronize(st[i]));
                CK(hipSetDevice(j)); CK(hipMemcpy(got.data(), buf[j], bytes, hipMemcpyDeviceToHost));
                ok = pat == got ? "ok" : "BAD";
            }
            std::string flag1 = "no peer";
            if (can[i][j] && can[j][i]) {
                CK(hipSetDevice(j));
                CK(hipMemset(flag[j], 0, 4));
                CK(hipDeviceSynchronize());
                CK(hipSetDevice(i));
                CK(hipMemset(flag[i] + 32, 0, 128));
                CK(hipDeviceSynchronize());
                CK(hipSetDevice(j));
                hipLaunchKernelGGL(recv_k, dim3(1), dim3(64), 0, st[j], flag[j], flag[i] + 32, iters, seq);
                CK(hipSetDevice(i));
                auto t0 = clk::now();
                hipLaunchKernelGGL(send_k, dim3(1), dim3(256), 0, st[i], flag[j], (unsigned*) buf[j], (const unsigned*) buf[i],
                                   (int) (bytes / 4), seq, flag[i] + 32, iters, flag[i] + 64);
                CK(hipStreamSynchronize(st[i]));
                CK(hipSetDevice(j));
                CK(hipStreamSynchronize(st[j]));
                unsigned to = 0, fj = 0, ai = 0;
                CK(hipSetDevice(i));
                CK(hipMemcpy(&to, flag[i] + 64, 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(&ai, flag[i] + 32, 4, hipMemcpyDeviceToHost));
                CK(hipSetDevice(j));
                CK(hipMemcpy(&fj, flag[j], 4, hipMemcpyDeviceToHost));
                CK(hipSetDevice(i));
                if (to) std::printf("  (%d -> %d peer flag: dst flag holds %u (sent %u), src ack holds %u)\n", i, j, fj, seq, ai);
                char t[32];
                // the payload the last iteration wrote into dst: src + iters - 1 (buf[i] holds the pattern from the check above)
                std::vector<unsigned> src(bytes / 4), dst(bytes / 4);
                CK(hipMemcpy(src.data(), buf[i], bytes, hipMemcpyDeviceToHost));
                CK(hipSetDevice(j)); CK(hipMemcpy(dst.data(), buf[j], bytes, hipMemcpyDeviceToHost));
                bool fok = true;
                for (size_t q = 0; q < src.size(); ++q) fok = fok && dst[q] == src[q] + (unsigned) (iters - 1);
                std::snprintf(t, sizeof t, to ? "TIMEOUT" : "%.1f%s", us(t0, clk::now()) / iters / 2.0, fok ? "" : " BAD");
                flag1 = t;
                seq += iters + 10;
            }
            // (e) pull: each side writes its flag into its OWN memory and polls the other's through the peer mapping
            std::string pl1 = "no peer";
            if (can[i][j] && can[j][i]) {
                CK(hipSetDevice(i)); CK(hipMemset(flag[i], 0, 1024)); CK(hipDeviceSynchronize());
                CK(hipSetDevice(j)); CK(hipMemset(flag[j], 0, 1024)); CK(hipDeviceSynchronize());
                hipLaunchKernelGGL(recv_k, dim3(1), dim3(64), 0, st[j], flag[i], flag[j] + 32, iters, seq);
                CK(hipSetDevice(i));
                auto t0 = clk::now();
                hipLaunchKernelGGL(send_k, dim3(1), dim3(256), 0, st[i], flag[i], (unsigned*) buf[j], (const unsigned*) buf[i],
                                   (int) (bytes / 4), seq, flag[j] + 32, iters, flag[i] + 64);
                CK(hipStreamSynchronize(st[i]));
                CK(hipSetDevice(j)); CK(hipStreamSynchronize(st[j]));
                unsigned to = 0;
                CK(hipSetDevice(i)); CK(hipMemcpy(&to, flag[i] + 64, 4, hipMemcpyDeviceToHost));
                char t[32];
                std::snprintf(t, sizeof t, to ? "TIMEOUT" : "%.1f", us(t0, clk::now()) / iters / 2.0);
                pl1 = t;
                seq += iters + 10;
            }
            std::string hf1;
            {
                std::memset(hflag, 0, 1024);
                CK(hipSetDevice(j));
                hipLaunchKernelGGL(recv_k, dim3(1), dim3(64), 0, st[j], hflag, hflag + 32, iters, seq);
                CK(hipSetDevice(i));
                auto t0 = clk::now();
                hipLaunchKernelGGL(send_k, dim3(1), dim3(256), 0, st[i], hflag, hbuf, (const unsigned*) buf[i], (int) (bytes / 4), seq,
                                   hflag + 32, iters, hflag + 64);
                CK(hipStreamSynchronize(st[i]));
                CK(hipSetDevice(j));
                CK(hipStreamSynchronize(st[j]));
                char t[32];
                std::snprintf(t, sizeof t, hflag[64] ? "TIMEOUT" : "%.1f", us(t0, clk::now()) / iters / 2.0);
                hf1 = t;
                seq += iters + 10;
            }
            CK(hipSetDevice(i));
            CK(hipMemcpyPeerAsync(big[j], j, big[i], i, bigb, st[i]));
            CK(hipStreamSynchronize(st[i]));
            auto t0 = clk::now();
            for (int k = 0; k < 5; ++k) CK(hipMemcpyPeerAsync(big[j], j, big[i], i, bigb, st[i]));
            CK(hipStreamSynchronize(st[i]));
            const double gbs = 5.0 * bigb / (us(t0, clk::now()) * 1e3);
            std::fflush(stdout);
            std::printf("  %d -> %d   %10.1f %10.1f %15s %12.1f   data %-9s %15s %15s\n", i, j, med(a), med(b), flag1.c_str(), gbs, ok.c_str(), hf1.c_str(), pl1.c_str());
        }
    return 0;
}
