#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/core/remote_expert_opt.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/core/remote_helper_fast.hpp"   // STRATA_HELPER_FAST (HIP builds only)
#endif
#include "strata/platform/aux_cpus.hpp"           // STRATA_HELPER_THREADS: the helper threads are placed, not swept

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // a window's rows x top-10

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}
} // namespace

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    if (device == primary_device()) {   // the primary may be any visible ordinal; the helper cannot be it
        err = "CUDA" + std::to_string(device) + " experts: the helper cannot be the primary device "
                                                 "(STRATA_PRIMARY_DEVICE)";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    stop_worker();
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        for (auto& kv : graphs_) if (kv.second.second) cudaGraphExecDestroy(kv.second.second);
        if (h_flag_) cudaFreeHost(h_flag_);
        if (d_seq_) cudaFree(d_seq_);
        if (read_stream_) cudaStreamSynchronize(read_stream_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_xr_) cudaFree(d_xr_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_meta_) cudaFree(d_meta_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (stream_) cudaStreamDestroy(stream_);
        if (read_stream_) cudaStreamDestroy(read_stream_);
        for (cudaEvent_t& e : ev_) if (e) cudaEventDestroy(e);
    }
    for (cudaEvent_t& e : ev_) e = nullptr;
    graphs_.clear();
    fmt_keys_.clear();
    graph_ = false;
    fast_ = 0;
    h_flag_ = z_flag_ = d_seq_ = nullptr;
    seq_ = 0;
    z_meta_ = nullptr;
    meta_copy_ = 0;
    timing_ = false;
    device_ = -1;
    pending_ = false;
    stream_ = nullptr;
    read_stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_xr_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size) {
    close();
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 0 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts ||
        device == primary_device()) {   // the helper cannot be the primary device (any visible ordinal)
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    skipped_held_ = 0;
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) >= 0 || claimed[index]) { ++skipped_held_; continue; }
        if (!picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + (512ull << 20) > free_bytes) break;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() : 0);
    // STRATA_HELPER_FAST (default 0 = the copy / zero-copy transport below, unchanged): see remote_experts.hpp.  HIP
    // builds only, not with --remote-expert-opt (its reduce returns through its own buffers).  The metadata staging is
    // then mapped (coherent, strata_hip.h) and a 16-byte multiple, the fetch kernel's word.
    fast_ = 0;
#if defined(STRATA_USE_HIP)
    fast_ = remote_opt_ == nullptr ? std::clamp(env_int("STRATA_HELPER_FAST", 0), 0, 2) : 0;
#endif
    if (fast_) meta_bytes = (meta_bytes + 15) / 16 * 16;
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable | (fast_ ? cudaHostAllocMapped : 0u)),
              "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device) &&
        (!strata::kernels::cpu::hadamard_spec().any() ||
         check(cudaMalloc((void**) &d_xr_, (size_t) CAP * H * sizeof(float)), "rotated input", err, device));
    if (!allocated) { close(); return false; }
    if (!check(cudaStreamCreateWithFlags(&read_stream_, cudaStreamNonBlocking), "cache read stream", err, device)) {
        close();
        return false;
    }
    if (const char* ht = std::getenv("STRATA_HELPER_TIMING"); ht != nullptr && std::atoi(ht) != 0) {
        timing_ = true;
        for (cudaEvent_t& e : ev_)
            if (cudaEventCreate(&e) != cudaSuccess) { timing_ = false; break; }
        cudaGetLastError();
    }
    ms_gpu_in_ = ms_gpu_kernels_ = ms_gpu_out_ = ms_round_trip_ = 0;
    groups_launched_ = 0;
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) FF), (int) FF);
    }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    // gfx906 helper: a wave64 helper (MI50) reading the RDNA build's non-coherent mapped host buffers across PCIe
    // was the bottleneck - measured on an XTX + MI50 32 GB, Flash-Next IQ3_S, 8K prompt, XTX cache capped at 2,500
    // slots: decode 33.5 tok/s with zero-copy against 45.0 with copies (36.7 without the helper).  So a wave64 helper
    // copies unless STRATA_REMOTE_ZEROCOPY=1 asks for zero-copy; every other card keeps the upstream default (on).
    bool zc_default = true;
    {
        cudaDeviceProp prop{};
#if defined(STRATA_USE_HIP) && defined(STRATA_HIP_HELPER_ARCHS)   // the helper build only; the gfx906 build's mapped memory is coherent
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.warpSize == 64) zc_default = false;
#else
        (void) prop;
#endif
        cudaGetLastError();
    }
    const bool zc_on = zc ? zc[0] != '0' : zc_default;
    zero_copy_ = zc_on &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    if (fast_) {   // STRATA_HELPER_FAST: the device views of the staging and the flag word; any failure: the copies
        zero_copy_ = false;   // (the fast path reads x through the fetch kernel, never across PCIe from the expert kernels)
        const bool ok_f =
            cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
            cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess &&
            cudaHostGetDevicePointer(&z_meta_, h_meta_, 0) == cudaSuccess &&
            cudaHostAlloc((void**) &h_flag_, 64, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
            cudaHostGetDevicePointer((void**) &z_flag_, h_flag_, 0) == cudaSuccess &&
            cudaMalloc((void**) &d_seq_, 64) == cudaSuccess && cudaMemset(d_seq_, 0, 64) == cudaSuccess &&
            cudaDeviceSynchronize() == cudaSuccess;
        cudaGetLastError();
        if (ok_f) {
            std::memset(h_flag_, 0, 64);
            seq_ = 0;
            meta_copy_ = meta_bytes;
            graph_ = env_int("STRATA_HELPER_GRAPH", 0) != 0 && !timing_;
        } else {
            std::fprintf(stderr, "strata: CUDA%d experts: STRATA_HELPER_FAST could not map its staging; this helper "
                                 "uses the copies\n", device);
            fast_ = 0;
            z_x_ = z_out_ = nullptr;
            z_meta_ = nullptr;
        }
    }
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    threaded_ = threads_requested() && remote_opt_ == nullptr;
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    if (!claim(layer, ids, n_tok, k, kind, primary_res, err)) return false;
    return group_id_.empty() || launch(layer, x, n_tok, err);
}

bool RemoteExperts::claim(int64_t layer, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
                          const int32_t* primary_res, std::string& err) {
    // cumulative host time in here and in launch() (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    rt0_ = timer.t0;
    if (threaded_)   // (a failed dispatch may have left this helper's thread a job: it reads what claim() rewrites)
        while (done_.load(std::memory_order_acquire) != posted_.load(std::memory_order_relaxed)) cpu_relax();
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::kWindowMaxT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    return true;
}

bool RemoteExperts::launch(int64_t layer, const float* x, int64_t n_tok, std::string& err) {
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    if (group_id_.empty()) return true;
    if (pending_) {   // the previous request was never finished (a failed pool call): nothing of it may still be in flight
        DeviceScope drain(device_);
        if (!drain.ok) { err = drain.error(device_); return false; }
        if (!check(cudaStreamSynchronize(stream_), "drain an unfinished request", err, device_)) return false;
        pending_ = false;
        if (fast_) seq_ = *(volatile const uint32_t*) h_flag_;   // the flags raised so far (the stream has drained)
    }
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    pending_ = true;   // from the first async call on; finish() clears it once the request has completed
    if (fast_) std::atomic_thread_fence(std::memory_order_seq_cst);   // the staging stores, before the launch that reads them
    bool queued = false;
    if (graph_ && !reduce) {   // STRATA_HELPER_GRAPH: direct the first time, captured the second, replayed after that
        const auto& lay = strata::kernels::cpu::expert_layout();
        uint64_t fk = 0;
        if (lay.native) {
            const auto& fmt = lay.fmt[(size_t) layer];
            fk = (uint64_t) fmt.gu_type | (uint64_t) fmt.d_type << 8 | (uint64_t) (fmt.had_gu ? 1 : 0) << 16 |
                 (uint64_t) (fmt.had_d ? 1 : 0) << 17 | (uint64_t) fmt.had_block << 18 | (uint64_t) fmt.n_ff << 34;
        }
        size_t fi = 0;
        while (fi < fmt_keys_.size() && fmt_keys_[fi] != fk) ++fi;
        if (fi == fmt_keys_.size()) fmt_keys_.push_back(fk);
        const uint64_t key = (uint64_t) fi | (uint64_t) n_tok << 8 | (uint64_t) groups_ << 16 | (uint64_t) dst_.size() << 32;
        auto it = graphs_.find(key);
        if (it == graphs_.end() && graphs_.size() < 4096) it = graphs_.emplace(key, std::make_pair(0, (cudaGraphExec_t) nullptr)).first;
        if (it != graphs_.end()) {
            auto& g = it->second;
            if (g.second == nullptr && g.first == 1) {   // the second sighting: capture (on failure: direct from then on)
                g.first = 2;
                cudaGraph_t graph = nullptr;
                std::string cerr;
                bool ok = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
                if (ok) {
                    const bool body = enqueue(layer, n_tok, false, true, cerr);
                    ok = cudaStreamEndCapture(stream_, &graph) == cudaSuccess && body && graph != nullptr;
                }
                cudaGraphExec_t exec = nullptr;
                if (ok) ok = cudaGraphInstantiate(&exec, graph, 0ull) == cudaSuccess;
                if (graph) cudaGraphDestroy(graph);
                cudaGetLastError();
                if (ok) { g.second = exec; ++graph_captures_; }
            } else if (g.first < 2) {
                ++g.first;
            }
            if (g.second != nullptr) {
                if (!check(cudaGraphLaunch(g.second, stream_), "replay the helper graph", err, device_)) return false;
                ++seq_;   // the graph's last node raises the flag
                ++graph_replays_;
                queued = true;
            }
        }
        if (!queued) ++graph_direct_;
    }
    if (!queued && !enqueue(layer, n_tok, reduce, false, err)) return false;
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    groups_launched_ += groups_;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) (n_tok * 10) * H * sizeof(float);
    return true;
}

bool RemoteExperts::enqueue(int64_t layer, int64_t n_tok, bool reduce, bool capturing, std::string& err) {
    const cudaStream_t s = stream_;
    const bool tm = timing_ && !capturing;
    if (tm) cudaEventRecord(ev_[0], s);
#if defined(STRATA_USE_HIP)
    if (fast_) {        // STRATA_HELPER_FAST: x and the metadata into VRAM by one kernel, from the coherent staging
        if (!check(helper_fast::fetch(z_x_, d_x_, (size_t) n_tok * H * sizeof(float), z_meta_, d_meta_, meta_copy_, s),
                   "fetch input", err, device_)) return false;
    } else
#endif
    {
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    }
    if (tm) cudaEventRecord(ev_[1], s);
    // where the expert kernels write: VRAM (then a copy or the push kernel), or the host rows directly (zero-copy, or
    // STRATA_HELPER_FAST=2)
    float* const rows_out = (zero_copy_ && !reduce) || fast_ == 2 ? z_out_ : d_out_;
    const float* const x_in = zero_copy_ ? z_x_ : d_x_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& fmt = lay.fmt[(size_t) layer];
        const float* xe = x_in;
        if (fmt.had_gu) {   // APR: Hadamard-folded gate/up take H (s * x)
            strata::kernels::hadamard_rows(xe, d_xr_, n_tok, H, fmt.had_block, d_sx_, s);
            xe = d_xr_;
        }
        strata::kernels::quantize_q8_1_rows(xe, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        if (fmt.had_d) { L.had_block = fmt.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, rows_out, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(x_in, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2_window(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, rows_out, s,
                                               (int) n_tok);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (tm) cudaEventRecord(ev_[2], s);
#if defined(STRATA_USE_HIP)
    if (fast_) {        // STRATA_HELPER_FAST: the rows to the coherent host rows by a kernel (1), then the flag
        if ((fast_ == 1 && !check(helper_fast::push(d_out_, z_out_, (size_t) compact_bytes, s), "push results", err, device_)) ||
            !check(helper_fast::flag_next(z_flag_, d_seq_, s), "raise the result flag", err, device_))
            return false;
        if (!capturing) ++seq_;   // (a captured flag counts when its graph is launched)
    } else
#endif
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    if (tm) cudaEventRecord(ev_[3], s);
    return true;
}

// STRATA_HELPER_FAST: the flag word holds the count of finished requests once the latest one's rows are in the host
// rows (the flag kernel runs after them on the stream and fences at system scope first).
bool RemoteExperts::flag_seen() const {
    const uint32_t v = *(volatile const uint32_t*) h_flag_;
    std::atomic_thread_fence(std::memory_order_acquire);   // the rows are read only after the word
    return v == seq_;
}

// Spin on the flag word: no HIP call per poll.  Every 4096 polls one stream query notices a failed stream (an error:
// report it) or a drained stream whose flag never arrived (a coherence fault: report it, never read the rows).
bool RemoteExperts::wait_flag(std::string& err) {
    for (uint32_t spins = 1;; ++spins) {
        if (flag_seen()) return true;
        if ((spins & 4095u) == 0) {
            DeviceScope scope(device_);
            const cudaError_t e = cudaStreamQuery(stream_);
            if (e == cudaErrorNotReady) continue;
            if (!check(e, "wait for the result flag", err, device_)) return false;
            if (flag_seen()) return true;
            err = "CUDA" + std::to_string(device_) + " experts: the stream finished but the result flag never arrived "
                  "(STRATA_HELPER_FAST: is the host staging coherent?)";
            return false;
        }
    }
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (group_id_.empty()) return true;
    if (fast_ && !timing_) {   // STRATA_HELPER_FAST: the flag word, no device switch and no stream query per poll
        const auto w0 = std::chrono::steady_clock::now();
        const bool ok = wait_flag(err);
        ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        if (!ok) return false;
        pending_ = false;
        for (size_t i = 0; i < original_row_.size(); ++i)
            std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
        return true;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    // Spin on the stream like PeerExperts::finish instead of the blocking sync: the layer waits for this helper
    // on the CPU pool's critical path, and a blocking sync's wake-up costs more than the helper's small batch
    // takes - on HIP especially, where the CUDA branch's cudaInitDevice(ScheduleSpin) does not exist and the
    // device keeps its default policy.  STRATA_REMOTE_SPIN=0 keeps the blocking sync (the same switch the CUDA
    // branch reads).
#if defined(STRATA_USE_HIP)   // HIP only: the CUDA branch's cudaInitDevice(ScheduleSpin) already makes the sync spin
    static const bool spin = [] {
        const char* v = std::getenv("STRATA_REMOTE_SPIN");
        return !(v != nullptr && v[0] == '0');
    }();
    if (spin) {
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!check(e, "finish", err, device_)) return false;
    } else if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) {
        return false;
    }
#else
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
#endif
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    pending_ = false;
    if (timing_) {   // the stream has drained: every event of this request is complete
        float a = 0, b = 0, c = 0;
        if (cudaEventElapsedTime(&a, ev_[0], ev_[1]) == cudaSuccess && cudaEventElapsedTime(&b, ev_[1], ev_[2]) == cudaSuccess &&
            cudaEventElapsedTime(&c, ev_[2], ev_[3]) == cudaSuccess) {
            ms_gpu_in_ += a; ms_gpu_kernels_ += b; ms_gpu_out_ += c;
        }
        cudaGetLastError();
        ms_round_trip_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt0_).count();
    }
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

std::string RemoteExperts::transport() const {
    std::string t = fast_ == 1 ? "kernel fetch + push + flag (STRATA_HELPER_FAST=1)"
                  : fast_ == 2 ? "kernel fetch + rows to host + flag (STRATA_HELPER_FAST=2)"
                  : zero_copy_ ? "zero-copy" : "copies";
    if (graph_) t += " + graph replay (STRATA_HELPER_GRAPH=1)";
    if (threaded_) t += "; own issue thread (STRATA_HELPER_THREADS=1)";
    return t;
}

// ---- STRATA_HELPER_THREADS (cycle 6): one thread per helper issues its request and collects its rows, so the four
// helpers' issue (staging + launches) and waits overlap instead of running one after another on the host thread.
bool RemoteExperts::threads_requested() {
    static const bool on = env_int("STRATA_HELPER_THREADS", 0) != 0;
    return on;
}

void RemoteExperts::post(int64_t layer, const float* x, int64_t n_tok, float* out) {
    if (!worker_.joinable()) {   // first request: the thread, on the next CPU of STRATA_HELPER_CPUS (if given)
        static std::atomic<int> next{0};
        const int idx = next.fetch_add(1);
        int cpu = -1;
        if (const char* list = std::getenv("STRATA_HELPER_CPUS"); list != nullptr && list[0] != '\0') {
            std::vector<int> cpus;
            if (strata::aux_cpus::parse_cpu_list(list, cpus) && !cpus.empty()) cpu = cpus[(size_t) idx % cpus.size()];
        }
        stop_.store(false);
        worker_ = std::thread([this, cpu] { worker_loop(cpu); });
    }
    // a job a failed dispatch never joined is still this thread's: wait for it before its fields are rewritten
    while (done_.load(std::memory_order_acquire) != posted_.load(std::memory_order_relaxed)) cpu_relax();
    job_layer_ = layer; job_x_ = x; job_ntok_ = n_tok; job_out_ = out;
    const uint64_t p = posted_.load(std::memory_order_relaxed) + 1;
    posted_.store(p, std::memory_order_seq_cst);   // the job's fields above, before the doorbell; and before the
    if (sleeping_.load(std::memory_order_seq_cst)) {  // sleeping_ read (the thread stores sleeping_, then reads posted_)
        std::lock_guard<std::mutex> lk(wake_mu_);
        wake_cv_.notify_one();
    }
}

bool RemoteExperts::join(std::string& err) {
    const uint64_t p = posted_.load(std::memory_order_relaxed);
    while (done_.load(std::memory_order_acquire) != p) cpu_relax();
    if (!job_ok_) { err = job_err_; return false; }
    return true;
}

void RemoteExperts::worker_loop(int cpu) {
#if defined(__linux__)
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    strata::aux_cpus::note_owned_thread();   // --aux-cpus' sweep leaves it where it is
    cudaSetDevice(device_);
    uint64_t seen = done_.load(std::memory_order_relaxed);
    auto idle_since = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    for (;;) {
        const uint64_t p = posted_.load(std::memory_order_acquire);
        if (p == seen) {
            if (stop_.load(std::memory_order_acquire)) return;
            cpu_relax();
            if ((++spins & 1023u) != 0) continue;
            // a parked thread spins ~20 ms (a decode's layers are microseconds apart), then sleeps until the next post
            if (std::chrono::steady_clock::now() - idle_since < std::chrono::milliseconds(20)) continue;
            std::unique_lock<std::mutex> lk(wake_mu_);
            sleeping_.store(true, std::memory_order_seq_cst);
            wake_cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                return posted_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire);
            });
            sleeping_.store(false, std::memory_order_relaxed);
            idle_since = std::chrono::steady_clock::now();
            continue;
        }
        std::string e;
        bool ok = launch(job_layer_, job_x_, job_ntok_, e);
        if (ok) ok = finish(job_out_, e);
        job_ok_ = ok;
        if (!ok) job_err_ = e;
        seen = p;
        done_.store(p, std::memory_order_release);   // the rows in out and the job's result, before the done word
        idle_since = std::chrono::steady_clock::now();
        spins = 0;
    }
}

void RemoteExperts::stop_worker() {
    if (!worker_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(wake_mu_);
        wake_cv_.notify_one();
    }
    worker_.join();
    stop_.store(false);
}

bool RemoteExperts::copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst,
                                      size_t n, uint8_t* hit, std::string& err) {
    std::fill(hit, hit + n, 0);
    if (device_ < 0 || read_stream_ == nullptr || n == 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    std::lock_guard<std::mutex> lock(read_mu_);
    bool queued = false;
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        const int32_t slot = cache_.slot_of(layers[i], experts[i]);
        if (slot < 0) continue;
        const size_t bytes = (size_t) lay.blob_bytes(layers[i]);
        if (!check(cudaMemcpyAsync(dst[i], cache_.device_slot(slot), bytes, cudaMemcpyDeviceToHost, read_stream_),
                   "cache read", err, device_)) {
            // Nothing may still be writing a caller's buffer when this returns.  If the drain succeeds, the
            // hits queued before the failure are complete and the caller may keep them; if it does not,
            // forget every hit and let the caller refill all of them from its fallback.
            if (cudaStreamSynchronize(read_stream_) != cudaSuccess) std::fill(hit, hit + n, 0);
            return false;
        }
        hit[i] = 1;
        queued = true;
    }
    if (queued && !check(cudaStreamSynchronize(read_stream_), "cache read wait", err, device_)) {
        std::fill(hit, hit + n, 0);   // a failed wait means no copied buffer may be trusted
        return false;
    }
    return true;
}

bool RemotePrefillSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (dst != nullptr && from_helper(layer, expert)) {
        const int32_t l = (int32_t) layer, e = (int32_t) expert;
        uint8_t* d = dst;
        std::string err;
        for (RemoteExperts* h : helpers_) {
            uint8_t got = 0;
            const bool ok = h->copy_cached_blobs(&l, &e, &d, 1, &got, err);
            if (got) {
                helper_blobs_.fetch_add(1, std::memory_order_relaxed);
                helper_bytes_.fetch_add(strata::kernels::cpu::expert_layout().blob_bytes(layer),
                                        std::memory_order_relaxed);
                return true;
            }
            if (!ok) break;
        }
        warn(err);
    }
    return fallback_.copy_blob(layer, expert, dst);
}

bool RemotePrefillSource::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) {
    if (helpers_.empty() || n == 0) return fallback_.copy_blobs(layers, experts, dst, n);
    thread_local std::vector<int32_t> hl, he;
    thread_local std::vector<uint8_t*> hd;
    thread_local std::vector<size_t> idx;
    thread_local std::vector<uint8_t> got;
    thread_local std::vector<uint8_t> served;
    thread_local std::vector<int32_t> ml, me;
    thread_local std::vector<uint8_t*> md;
    thread_local std::vector<int> which;
    which.assign(n, -1);
    served.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (dst[i] == nullptr) continue;
        for (size_t k = 0; k < helpers_.size(); ++k)
            if (helpers_[k]->holds(layers[i], experts[i])) { which[i] = (int) k; break; }
    }
    // One stream batch per helper, so several hits pay one wait; the misses go to the file source in one
    // call, which keeps its own batched-read merging for them.
    uint64_t blobs = 0, bytes = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    for (size_t k = 0; k < helpers_.size(); ++k) {
        hl.clear(); he.clear(); hd.clear(); idx.clear();
        for (size_t i = 0; i < n; ++i)
            if (which[i] == (int) k) {
                hl.push_back(layers[i]); he.push_back(experts[i]); hd.push_back(dst[i]); idx.push_back(i);
            }
        if (hl.empty()) continue;
        got.assign(hl.size(), 0);
        std::string err;
        if (!helpers_[k]->copy_cached_blobs(hl.data(), he.data(), hd.data(), hl.size(), got.data(), err)) warn(err);
        for (size_t j = 0; j < got.size(); ++j)
            if (got[j]) {
                served[idx[j]] = 1;
                ++blobs;
                bytes += lay.blob_bytes(hl[j]);
            }
    }
    if (blobs != 0) {
        helper_blobs_.fetch_add(blobs, std::memory_order_relaxed);
        helper_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    ml.clear(); me.clear(); md.clear();
    for (size_t i = 0; i < n; ++i)
        if (!served[i]) { ml.push_back(layers[i]); me.push_back(experts[i]); md.push_back(dst[i]); }
    if (ml.empty()) return true;
    return fallback_.copy_blobs(ml.data(), me.data(), md.data(), ml.size());
}

void RemotePrefillSource::warn(const std::string& err) {
    if (err.empty() || warned_.exchange(true)) return;
    std::fprintf(stderr, "prefill: a helper cache read failed (%s); the file source fills that blob instead\n",
                 err.c_str());
}
} // namespace strata::core
