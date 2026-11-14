// strata_ds4_launch(kernel, grid, block, shared, stream, args...): the engine's one spelling of a kernel launch.  On CUDA it is
// kernel<<<grid, block, shared, stream>>>(args...); on Metal metal/include/metal_compat/cuda_launch.h maps the same call onto
// a dispatch.  (Every file that launches kernels includes this header.)
#pragma once
#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)) && !defined(STRATA_METAL)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template <class... KA, class... CA>
inline void strata_ds4_launch_at(const char* file, int line, void (*kernel)(KA...), dim3 grid, dim3 block, size_t shared,
                                 cudaStream_t stream, CA&&... args) {
    kernel<<<grid, block, shared, stream>>>(static_cast<KA>(args)...);
    static const bool check = std::getenv("DS4_LAUNCH_CHECK") != nullptr;   // report the launch that failed (not a later call)
    if (check) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s (grid %u,%u,%u block %u,%u,%u shared %zu)\n", file, line,
                         cudaGetErrorString(e), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
    }
}
#define strata_ds4_launch(...) strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// diagnostics (STRATA_DS4_DIAG=1; HIP builds; docs/dsv41/cards/EB-04.md): the engine's launch
// error reports name the device, arch, raw error code, free VRAM, thread and time of a device's FIRST failure in full and count
// the rest (summary at exit).  ds4_diag_launch_error returns true when it handled the report (the caller then skips its own line).
#if defined(STRATA_USE_HIP)
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
inline bool ds4_diag_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DS4_DIAG"); return v != nullptr && *v != 0 && *v != '0'; }();
    return on;
}
inline double ds4_diag_s() {   // seconds on the monotonic clock since the first call
    using C = std::chrono::steady_clock;
    static const C::time_point t0 = C::now();
    return std::chrono::duration<double>(C::now() - t0).count();
}
struct Ds4DiagErrs {
    static constexpr int kMax = 64;
    std::atomic<long> n[kMax];
    std::mutex mu;
    Ds4DiagErrs() { for (auto& a : n) a.store(0); }
    void summary() {
        for (int i = 0; i < kMax; ++i)
            if (n[i].load() > 1)
                std::fprintf(stderr, "DIAG: HIP device %d: %ld launch failures in all (the first is reported in full above)\n", i,
                             n[i].load());
    }
    ~Ds4DiagErrs() { summary(); }
};
inline Ds4DiagErrs& ds4_diag_errs() {
    static Ds4DiagErrs e;
    return e;
}
inline bool ds4_diag_launch_error(const char* where, const char* what, int code) {
    if (!ds4_diag_on()) return false;
    int dev = -1;
    (void) hipGetDevice(&dev);
    Ds4DiagErrs& E = ds4_diag_errs();
    const int slot = dev >= 0 && dev < Ds4DiagErrs::kMax ? dev : Ds4DiagErrs::kMax - 1;
    if (E.n[slot].fetch_add(1) != 0) return true;
    hipDeviceProp_t p{};
    (void) hipGetDeviceProperties(&p, dev);
    size_t fr = 0, tot = 0;
    const int me = (int) hipMemGetInfo(&fr, &tot);
    std::lock_guard<std::mutex> lk(E.mu);
    std::fprintf(stderr, "DIAG first launch failure on HIP device %d (%s, %s): %s %s: code %d %s (%s); free %.0f of %.0f MiB "
                         "(hipMemGetInfo %d); thread %zu; t=%.3f s\n",
                 dev, p.gcnArchName, p.name, where, what, code, code != 0 ? hipGetErrorName((hipError_t) code) : "(message only)",
                 code != 0 ? hipGetErrorString((hipError_t) code) : "no error code", (double) fr / 1048576.0, (double) tot / 1048576.0,
                 me, std::hash<std::thread::id>()(std::this_thread::get_id()), ds4_diag_s());
    return true;
}
#endif

// (the Metal debug dumps in ds4_model.cu name these; on CUDA there is no deferred queue and a device pointer is not a host view)
inline void ds4_metal_flush() {}
namespace strata::gpu {
inline void* host_view(const void* p) { return const_cast<void*>(p); }
}
#endif

// ds4_dbg("NAME"): the Mac engine's opt-in diagnostics (DS4_DBG_*, DS4_CPU_LOG, DS4_TLOG, ...).  On Metal it is an environment
// lookup; everywhere else it is the constant false, so the CUDA and HIP engines carry none of those checks on their hot paths.
#if defined(STRATA_METAL)
#include <cstdlib>
inline bool ds4_dbg(const char* name) { return std::getenv(name) != nullptr; }
#else
constexpr bool ds4_dbg(const char*) { return false; }
#endif
